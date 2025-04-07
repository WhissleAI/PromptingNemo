import os
import json
import torch
import librosa
import numpy as np
from pathlib import Path
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel
import torch.nn as nn


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

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender

def get_age_bucket(age: float) -> str:
    actual_age = round(age * 100, 2)
    age_brackets = [
        (18, "0_18"),
        (30, "18_30"),
        (45, "30_45"),
        (60, "45_60"),
        (float('inf'), "60PLUS")
    ]
    
    for threshold, bracket in age_brackets:
        if actual_age < threshold:
            return bracket
    return "60PLUS"

def process_ner(text: str, ner_tagger) -> str:
    sentence = Sentence(text)
    ner_tagger.predict(sentence)
    
    formatted_text = text
    
    for entity in sentence.get_spans('ner'):
        entity_text = entity.text
        entity_type = entity.tag
        replacement = f"NER_{entity_type} {entity_text} END"
        formatted_text = formatted_text.replace(entity_text, replacement, 1)
    
    return formatted_text

def convert_json_format(input_dir: str, output_dir: str, audio_base_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    age_gender_model = AgeGenderModel.from_pretrained(model_name).to(device)
    ner_tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
   
                if isinstance(data_list, dict):
                    data_list = [data_list]
 
                for entry_data in data_list:
                    try:
                        # Get the audio filename from the original path
                        audio_filename = os.path.basename(entry_data["audio_path"])
                        # Construct the new audio path using the correct base directory
                        audio_path = os.path.join(audio_base_dir, audio_filename)
                        
                        video_id = entry_data["video_id"]
                        transcription = entry_data["transcription"]
                        emotion = entry_data["emotion_analysis"]
                        
                        if not os.path.exists(audio_path):
                            print(f"Audio file not found: {audio_path}")
                            continue
                            
                        signal, sr = librosa.load(audio_path, sr=16000)
                        
                        y = processor(signal, sampling_rate=sr)
                        y = y['input_values'][0]
                        y = y.reshape(1, -1)
                        y = torch.from_numpy(y).to(device)

                        with torch.no_grad():
                            model_output = age_gender_model(y)
                            age = float(model_output[1].detach().cpu().numpy()[0][0])
                            gender = np.argmax(model_output[2].detach().cpu().numpy())
              
                        ner_text = process_ner(transcription.lower(), ner_tagger)

                        age_bucket = get_age_bucket(age)
                        
                        gender_text = "MALE" if gender == 1 else "FEMALE"
                        emotion = emotion.upper()
                        formatted_text = f"{ner_text} AGE_{age_bucket} GENDER_{gender_text} EMOTION_{emotion}"

                        new_entry = {
                            "audio_filepath": audio_path,
                            "text": formatted_text
                        }
                        
                        all_results.append(new_entry)
                        print(f"Processed: {video_id}")
                        
                    except Exception as e:
                        print(f"Error processing entry with video_id {video_id}: {str(e)}")
                        continue
                        
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file {filename}: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue
    
    if all_results:
        output_path = os.path.join(output_dir, "avspeech-data_processed.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in all_results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"\nSaved {len(all_results)} entries to: {output_path}")


if __name__ == "__main__":
    input_dir = "/external1/datasets/asr-himanshu/avspeech-data/transcripts1"
    output_dir = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/ner_metadata"
    audio_base_dir = "/external1/datasets/asr-himanshu/avspeech-data/audio"
    
    print(f"Processing files from: {input_dir}")
    print(f"Audio files directory: {audio_base_dir}")
    print(f"Saving output to: {output_dir}")
    
    convert_json_format(input_dir, output_dir, audio_base_dir)