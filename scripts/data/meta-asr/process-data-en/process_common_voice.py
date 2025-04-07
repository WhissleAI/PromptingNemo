import os
import json
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import pipeline
from dotenv import load_dotenv
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2PreTrainedModel, Wav2Vec2Model
load_dotenv()

class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender
model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
class AudioPredictor:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("audeering/wav2vec2-large-robust-6-ft-age-gender")
        self.model = AgeGenderModel.from_pretrained("audeering/wav2vec2-large-robust-6-ft-age-gender")
        self.model.eval()  

    def predict_age_and_gender(self, audio_path):
    
        try:

            audio, sr = librosa.load(audio_path, sr=16000)
          
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                hidden_states, logits_age, logits_gender = self.model(**inputs)
           
            age = logits_age.squeeze().item() * 100 
            gender_probs = logits_gender.squeeze()
            gender_idx = torch.argmax(gender_probs).item()
            
            gender_map = {0: "MALE", 1: "FEMALE", 2: "OTHER"}
            return round(age), gender_map.get(gender_idx, "OTHER")
            
        except Exception as e:
            print(f"Error processing audio: {audio_path}\nError: {str(e)}")
            return 25, "OTHER"  

def get_age_bucket(age_text: str) -> str:
    age_to_number = {
        'twenties': 25,
        'thirties': 35,
        'fourties': 45,
        'fifties': 55,
        'sixties': 65,
        'seventies': 75,
        'eighties': 85,
        'nineties': 95
    }
    
    
    age = age_to_number.get(age_text.lower(), 20)  
    
 
    age_brackets = [
        (18, "0_18"),
        (30, "18_30"),
        (45, "30_45"),
        (60, "45_60"),
        (float('inf'), "60PLUS")
    ]
    
    for threshold, bracket in age_brackets:
        if age < threshold:
            return bracket
    return "60PLUS"

def map_gender(gender: str) -> str:
    """Map gender to MALE/FEMALE/OTHER"""
    gender_mapping = {
        'male': 'MALE',
        'female': 'FEMALE',
        'other': 'OTHER'
    }
    return gender_mapping.get(gender.lower(), 'OTHER')

def predict_emotion(text: str, emotion_classifier) -> str:
   
    try:
        result = emotion_classifier(text)[0]
        emotion = result['label'].upper()
        return f"EMOTION_{emotion}"
    except Exception as e:
        print(f"Error predicting emotion for text: {text}\nError: {str(e)}")
        return "EMOTION_NEU"

def process_text_with_ner(text: str, ner_tagger) -> str:

    sentence = Sentence(text)
    ner_tagger.predict(sentence)
  
    entities = sorted(sentence.get_spans('ner'), key=lambda x: x.start_position)

    result_parts = []
    last_end = 0
    
    for entity in entities:
        result_parts.append(text[last_end:entity.start_position])
        # Add entity with tags, without underscore between tag and text
        result_parts.append(f"NER_{entity.tag} {entity.text} END")
        last_end = entity.end_position
    
    # Add remaining text
    result_parts.append(text[last_end:])
    
    return ''.join(result_parts).strip()

def process_data(tsv_path: str, output_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading NER tagger...")
    ner_tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
    
    print("Loading emotion classifier...")
    emotion_classifier = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base", 
        device=0 if torch.cuda.is_available() else -1
    )
    print("Loading age-gender model...")
    model = AgeGenderModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    print(f"Reading TSV file: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Found {len(df)} entries in TSV file")
    
    output_data = []
    processed_files = 0
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{len(df)}")
            
        try:

            audio_path = os.path.join("clips", row['path'])
            
            # Use TSV age and gender if available
            if pd.notna(row['age']) and pd.notna(row['gender']):
                age_bracket = get_age_bucket(row['age'])
                gender = map_gender(row['gender'])
            else:
                # Load and process audio for age-gender prediction
                audio, sr = librosa.load(audio_path, sr=16000)
                predictions = process_func(audio, sr)
                predicted_age = round(predictions[0][0][0],2)*100 
                predicted_gender = predictions[1][0] 

                age_bracket = get_age_bucket(predicted_age)
                gender = "MALE" if np.argmax(predicted_gender) == 1 else "FEMALE"
            
            sentence_text = row['sentence']
           
            processed_text = process_text_with_ner(sentence_text, ner_tagger)
            
        
            emotion = predict_emotion(sentence_text, emotion_classifier)
            
            text = f"{processed_text} AGE_{age_bracket} GENDER_{gender} {emotion}"
     
            entry = {
                "audio_filepath": audio_path,
                "text": text
            }
            
            output_data.append(entry)
            processed_files += 1
        
            if processed_files % 1000 == 0:
                print(f"Saving intermediate results... ({processed_files} files processed)")
                with open(f"{output_path}.temp", 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2)
                    
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
    
    print(f"\nProcessing complete!")
    print(f"Total entries processed: {processed_files}")
    
  
    print(f"Saving results to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)


tsv_path = "/external2/datasets/cv/cv-corpus-15.0-2023-09-08/en/merged.tsv"
output_path = "output_cv.json"

process_data(tsv_path, output_path)
