import os
import json
import torch
import librosa
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from transformers import (
    AutoModelForAudioClassification, 
    AutoFeatureExtractor, 
    Wav2Vec2Processor, 
    Wav2Vec2Model, 
    Wav2Vec2PreTrainedModel
)
import torch.nn as nn

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

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AgeGenderModel.from_pretrained(model_name).to(device)

def extract_emotion(audio_data: np.ndarray, sampling_rate: int = 16000) -> str:
    model_name = "superb/hubert-large-superb-er"
        
    if not hasattr(extract_emotion, 'model'):
        extract_emotion.model = AutoModelForAudioClassification.from_pretrained(model_name)
        extract_emotion.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
    if audio_data is None or len(audio_data) == 0:
        return "No Audio"
        
    if len(audio_data) < sampling_rate:
        return "Audio Too Short"

    inputs = extract_emotion.feature_extractor(
        audio_data, 
        sampling_rate=sampling_rate, 
        return_tensors="pt", 
        padding=True
    )

    outputs = extract_emotion.model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()
        
    return extract_emotion.model.config.id2label.get(predicted_class_idx, "Unknown")

@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    speaker: str
    age: float
    gender: int
    transcription: str
    emotion: str
    chunk_filename: str
    duration: float

@dataclass
class ChunkData:
    segments: List[AudioSegment] = field(default_factory=list)
    filepath: str = ""
    
    def get_formatted_text(self) -> str:
        texts = []
        
        for segment in self.segments:
            age_bucket = self.get_age_bucket(segment.age)
            gender_text = "MALE" if segment.gender == 1 else "FEMALE"
            
            transcription = segment.transcription.lower()
            metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{segment.emotion.upper()}"
            
            texts.append(f"{transcription} {metadata}")
        
        return " ".join(texts)

    @staticmethod
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

def get_file_pairs(audio_dir: str, text_dir: str) -> List[Tuple[str, str]]:
    """Get matching audio and text file pairs."""
    audio_files = {f: os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                  if f.endswith('.flac')}
    text_files = {os.path.splitext(f)[0]: os.path.join(text_dir, f) for f in os.listdir(text_dir) 
                 if f.endswith('.txt')}
    
    # Match files based on their base names (without extensions)
    pairs = []
    for audio_name, audio_path in audio_files.items():
        base_name = os.path.splitext(audio_name)[0]
        if base_name in text_files:
            pairs.append((audio_path, text_files[base_name]))
    
    return pairs

def get_transcription(text_file: str) -> str:
    """Read transcription from text file."""
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading transcription file {text_file}: {str(e)}")
        return ""

def process_audio_files(base_dir: str, output_dir: str = "output", batch_size: int = 50000) -> None:
    """Process audio files and their corresponding transcriptions."""
    os.makedirs(output_dir, exist_ok=True)
    
    audio_dir = os.path.join(base_dir, "audio")
    text_dir = os.path.join(base_dir, "text")
    
    if not os.path.exists(audio_dir) or not os.path.exists(text_dir):
        print(f"Audio or text directory not found in {base_dir}")
        return
    
    print(f"Processing files from:\nAudio: {audio_dir}\nText: {text_dir}")
    
    # Get matching audio-text pairs
    file_pairs = get_file_pairs(audio_dir, text_dir)
    print(f"Found {len(file_pairs)} matching audio-text pairs")
    
    all_results = []
    processed_count = 0
    batch_number = 1
    
    def save_batch(results, batch_num):
        if results:
            json_path = os.path.join(output_dir, f"audio_text_pairs_batch_{batch_num}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nSaved batch {batch_num} with {len(results)} entries to: {json_path}")
    
    for audio_path, text_path in file_pairs:
        try:
            transcription = get_transcription(text_path)
            if not transcription:
                continue
                
            print(f"Processing {os.path.basename(audio_path)}")
            
            signal, sr = librosa.load(audio_path, sr=16000)
            speaker = os.path.splitext(os.path.basename(audio_path))[0].split('_')[0]
            
            # Calculate duration in seconds
            duration = len(signal) / sr
            
            y = processor(signal, sampling_rate=sr)
            y = y['input_values'][0]
            y = y.reshape(1, -1)
            y = torch.from_numpy(y).to(device)

            with torch.no_grad():
                model_output = model(y)
                age = float(model_output[1].detach().cpu().numpy()[0][0])
                gender = np.argmax(model_output[2].detach().cpu().numpy())

            emotion = extract_emotion(signal)
            
            segment_data = AudioSegment(
                start_time=0,
                end_time=duration,
                speaker=speaker,
                age=age,
                gender=gender,
                transcription=transcription,
                emotion=emotion,
                chunk_filename=os.path.basename(audio_path),
                duration=duration
            )
            
            chunk = ChunkData()
            chunk.segments.append(segment_data)
            chunk.filepath = os.path.abspath(audio_path)
            
            all_results.append({
                "audio_filepath": chunk.filepath,
                "text": chunk.get_formatted_text(),
                "duration": duration
            })
            
            processed_count += 1
            
            if processed_count >= batch_size:
                save_batch(all_results, batch_number)
                batch_number += 1
                all_results = []
                processed_count = 0
                
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue
    
    # Save any remaining results
    if all_results:
        save_batch(all_results, batch_number)
        print(f"\nSaved final batch with {len(all_results)} entries")
    else:
        print("\nNo remaining results to save")

if __name__ == "__main__":
    base_dir = "/external2/datasets/pq"
    output_dir = "/external4/datasets/jsonl_data"
    
    print(f"Processing files from base directory: {base_dir}")
    print(f"Saving output to: {output_dir}")
    
    process_audio_files(base_dir, output_dir)