import os
import json
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import tempfile
from transformers import pipeline
from dotenv import load_dotenv
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from pydub import AudioSegment
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define device here
        self.model.to(self.device) # Move model to device here
        self.model.eval()

    def predict_age_and_gender(self, audio_path):

        try:

            audio, sr = librosa.load(audio_path, sr=16000)

            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device) # Move inputs to device here

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

def get_audio_duration(file_path):
    """
    Get the duration of an audio file in seconds.
    Supports multiple audio formats including OPUS.

    Args:
        file_path (str): Path to the audio file

    Returns:
        float: Duration in seconds
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return None

    # Get file extension
    extension = Path(file_path).suffix.lower()

    try:
        # Special handling for opus files
        if extension == '.opus':
            # Use ffprobe directly to get duration for opus files
            cmd = [
                'ffprobe',
                '-i', file_path,
                '-show_entries', 'format=duration',
                '-v', 'quiet',
                '-of', 'csv=p=0'
            ]
            output = subprocess.check_output(cmd).decode('utf-8').strip()
            try:
                duration = float(output)
                return round(duration, 2)
            except ValueError:
                print(f"Could not parse duration from ffprobe output: {output}")
                return None

        # For other standard formats, use pydub as before
        elif extension in ['.mp3', '.wav', '.ogg', '.flac']:
            if extension == '.mp3':
                audio = AudioSegment.from_mp3(file_path)
            elif extension == '.wav':
                audio = AudioSegment.from_wav(file_path)
            elif extension == '.ogg':
                audio = AudioSegment.from_ogg(file_path)
            elif extension == '.flac':
                audio = AudioSegment.from_file(file_path, format="flac")

        # Handle M4A, MP4, and WEBM files
        elif extension in ['.m4a', '.mp4', '.webm']:
            audio = AudioSegment.from_file(file_path)

        else:
            raise ValueError(f"Unsupported file format: {extension}")

        # For pydub processed files
        duration = len(audio) / 1000.0  # Convert milliseconds to seconds
        return round(duration, 2)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Alternative approach for opus: temporarily convert to wav
        if extension == '.opus':
            try:
                print(f"Trying alternative approach for {file_path}...")
                with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
                    temp_path = temp_file.name
                    # Convert opus to wav using ffmpeg
                    cmd = [
                        'ffmpeg',
                        '-i', file_path,
                        '-y',  # Overwrite output file if it exists
                        temp_path
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Get duration from the converted wav file
                    audio = AudioSegment.from_wav(temp_path)
                    duration = len(audio) / 1000.0
                    return round(duration, 2)
            except Exception as e2:
                print(f"Alternative approach also failed for {file_path}: {str(e2)}")

        # If all else fails, try librosa as a last resort
        try:
            print(f"Trying librosa to get duration for {file_path}...")
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            return round(duration, 2)
        except Exception as e3:
            print(f"Librosa approach also failed for {file_path}: {str(e3)}")
            return None

def process_data(tsv_path: str, output_path: str, clips_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open output file for writing JSONL format
    with open(output_path, 'w', encoding='utf-8') as out_file:
        processed_files = 0
        skipped_files = 0

        audio_predictor = AudioPredictor() # Initialize AudioPredictor outside the loop

        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processing row {idx}/{len(df)}")

            try:
                # Use the absolute path to the clips directory
                audio_path = os.path.join(clips_dir, row['path'])

                # Check if file exists
                if not os.path.exists(audio_path):
                    print(f"Skipping missing file: {audio_path}")
                    skipped_files += 1
                    continue

                # Get audio duration
                duration = get_audio_duration(audio_path)
                if duration is None:
                    print(f"Could not determine duration for {audio_path}, skipping...")
                    skipped_files += 1
                    continue

                # Use TSV age and gender if available
                if pd.notna(row.get('age')) and pd.notna(row.get('gender')):
                    age_bracket = get_age_bucket(row['age'])
                    gender = map_gender(row['gender'])
                else:
                    # Load and process audio for age-gender prediction
                    try:
                        predicted_age, predicted_gender = audio_predictor.predict_age_and_gender(audio_path)
                        age_bracket = get_age_bucket(str(round(predicted_age)))
                        gender = predicted_gender
                    except Exception as e:
                        print(f"Error predicting age/gender for {audio_path}: {str(e)}")
                        age_bracket = "30_45"  # Default
                        gender = "OTHER"  # Default

                sentence_text = row['sentence']

                # Predict emotion
                emotion = predict_emotion(sentence_text, emotion_classifier)

                # Create output text with age, gender, and emotion
                text = f"{sentence_text} AGE_{age_bracket} GENDER_{gender} {emotion}"

                entry = {
                    "audio_filepath": audio_path,
                    "text": text,
                    "duration": duration
                }

                # Write each entry as a JSON line (JSONL format)
                out_file.write(json.dumps(entry) + '\n')
                processed_files += 1

                if processed_files % 1000 == 0:
                    print(f"Processed {processed_files} files...")
                    # Save intermediate results
                    print(f"Current stats - Processed: {processed_files}, Skipped: {skipped_files}")

            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                skipped_files += 1

        print(f"\nProcessing complete!")
        print(f"Total entries processed: {processed_files}")
        print(f"Total entries skipped: {skipped_files}")



language_code = "ka"  # Change this to the language code you're processing (e.g., 'en', 'it', 'de')
base_dir = f"/external4/datasets/cv/cv-corpus-15.0-2023-09-08/{language_code}"
tsv_path = f"{base_dir}/validated.tsv"
clips_dir = f"{base_dir}/clips"
output_path = "/external4/datasets/jsonl_data/euro/gergian_cv.jsonl" 

process_data(tsv_path, output_path, clips_dir)