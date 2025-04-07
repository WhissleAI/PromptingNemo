import os
import json
import torch
import asyncio
import librosa
import numpy as np
import pandas as pd
import moviepy as mp
import torch.nn as nn
import soundfile as sf
from pathlib import Path
from dotenv import load_dotenv
# from whissle import WhissleClient # No longer needed
from flair.data import Sentence
from pyannote.audio import Pipeline
from collections import defaultdict
from flair.models import SequenceTagger
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
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
ner_tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=os.getenv("HF_TOKEN"))


def process_func(x: np.ndarray, sampling_rate: int, embeddings: bool = False) -> np.ndarray:
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0].detach().cpu().numpy()
        else:
            age_logits = y[1].detach().cpu().numpy()
            gender_logits = y[2].detach().cpu().numpy()
            y = [age_logits, gender_logits]

    return y


def get_speaker_changes(audio_path: str):
    diarization = pipeline({'audio': audio_path})
    speaker_changes = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_changes.append((turn.start, turn.end, speaker))
    return speaker_changes



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


def convert_mp4_to_wav(mp4_path: str) -> str:
    file_ext = os.path.splitext(mp4_path)[1].lower()

    if file_ext == '.mp4' or file_ext == '.webm':
        output_dir = "/external2/datasets/hf_data_output_audio"
        os.makedirs(output_dir, exist_ok=True)
        wav_path = os.path.join(output_dir, os.path.splitext(os.path.basename(mp4_path))[0] + ".wav")

        # Try using a direct FFmpeg command instead of moviepy
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-i", mp4_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
                "-ac", "1", wav_path, "-y"
            ]
            subprocess.run(cmd, check=True)
            return wav_path
        except Exception as e:
            print(f"FFmpeg direct command failed: {e}, trying moviepy as fallback")

            # Fallback to moviepy
            try:
                video = mp.VideoFileClip(mp4_path)
                video.audio.write_audiofile(wav_path, fps=16000)
                video.close()
                return wav_path
            except Exception as e:
                print(f"Error converting {mp4_path} to wav: {e}")
                # Return original path if conversion fails
                return mp4_path

    return mp4_path
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
    ner_tags: List[Dict[str, Any]] = field(default_factory=list)

def process_ner(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    sentence = Sentence(text)
    ner_tagger.predict(sentence)

    entities = []
    formatted_tokens = []
    current_index = 0

    for token in sentence:
        token_text = token.text

        ner_tag = token.get_labels('ner')[0].value if token.get_labels('ner') else 'O'

        start_position = text.find(token_text, current_index)
        end_position = start_position + len(token_text)
        current_index = end_position

        is_entity_start = False
        is_entity_end = False
        entity_type = None


        for entity in sentence.get_spans('ner'):
            if token in entity.tokens:
                if token == entity.tokens[0]:
                    is_entity_start = True
                    entity_type = entity.get_label('ner').value
                if token == entity.tokens[-1]:
                    is_entity_end = True

        if is_entity_start:
            formatted_tokens.append(f"NER_{entity_type}")
        formatted_tokens.append(token_text)
        if is_entity_end:
            formatted_tokens.append("END")

        if ner_tag != 'O':
            for entity in sentence.get_spans('ner'):
                if token in entity.tokens:
                    if entity.get_label('ner').value not in [ent['type'] for ent in entities if ent['text'] == entity.text]:
                        entities.append({
                            "text": entity.text,
                            "type": entity.get_label('ner').value,
                            "start_position": entity.start_pos,
                            "end_position": entity.end_pos
                        })
                    break

    formatted_text = " ".join(formatted_tokens)
    return formatted_text, entities


@dataclass
class ChunkData:
    segments: List[AudioSegment] = field(default_factory=list)
    filepath: str = ""

    def get_formatted_text(self) -> str:
        texts = []
        current_speaker = None

        for segment in self.segments:
            age_bucket = self.get_age_bucket(segment.age)
            gender_text = "MALE" if segment.gender == 1 else "FEMALE"


            ner_text, entities = process_ner(segment.transcription.lower())
            segment.ner_tags = entities
            emotion_text = segment.emotion.upper().replace(" ", "_")

            metadata = f"AGE_{age_bucket} GENDER_{gender_text} EMOTION_{emotion_text}"

            if current_speaker != segment.speaker:
                metadata += " SPEAKER_CHANGE"
                current_speaker = segment.speaker

            texts.append(f"{ner_text} {metadata}")

        return " ".join(texts)

    @staticmethod
    def get_age_bucket(age: float) -> str:
        actual_age: float = round(age*100, 2)


        age_brackets: List[tuple[float, str]] = [
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


def process_jsonl_data(
    jsonl_path: str,
    chunk_duration: float = 20.0,
    output_base_dir: str = "output"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    chunks_dir = os.path.abspath(os.path.join(output_base_dir, "audio_chunks"))
    results_dir = os.path.abspath(os.path.join(output_base_dir, "results"))

    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    all_data = []
    chunk_data: Dict[str, ChunkData] = defaultdict(ChunkData)

    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                audio_path = item['audio_filepath']
                provided_text = item['text']
                base_filename = os.path.splitext(os.path.basename(audio_path))[0]

                processed_audio_path = convert_mp4_to_wav(audio_path)
                try:
                    import soundfile as sf
                    signal, sr = sf.read(processed_audio_path)
                    if sr != 16000:
                        import resampy
                        signal = resampy.resample(signal, sr, 16000)
                        sr = 16000
                except Exception as e:
                    print(f"Error loading audio file {processed_audio_path}: {str(e)}")
                    continue


                chunk_filename = os.path.basename(audio_path)

                try:
                    diarization = pipeline({'audio': processed_audio_path})
                    speaker_changes = [
                        (turn.start, turn.end, speaker)
                        for turn, _, speaker in diarization.itertracks(yield_label=True)
                    ]

                    for speaker_idx, (start_time, end_time, speaker) in enumerate(speaker_changes):
                        start_sample = int(start_time * sr)
                        end_sample = int(end_time * sr)

                        speaker_segment = signal[start_sample:end_sample]

                        if len(speaker_segment) / sr < 1.5:
                            continue

                        y = processor(speaker_segment, sampling_rate=sr)
                        y = y['input_values'][0]
                        y = y.reshape(1, -1)
                        y = torch.from_numpy(y).to(device)

                        with torch.no_grad():
                            model_output = model(y)
                            age = float(model_output[1].detach().cpu().numpy()[0][0])
                            gender = np.argmax(model_output[2].detach().cpu().numpy())

                        temp_segment_path = os.path.join(
                            chunks_dir,
                            f"temp_segment_{base_filename}_{speaker_idx}.wav"
                        )
                        sf.write(temp_segment_path, speaker_segment, sr)

                        try:
                            speaker_segment_audio, _ = librosa.load(temp_segment_path, sr=16000)
                            emotion = extract_emotion(speaker_segment_audio)

                            transcription = provided_text

                            ner_text, ner_entities = process_ner(transcription.lower())

                            segment = AudioSegment(
                                start_time=start_time,
                                end_time=end_time,
                                speaker=speaker,
                                age=float(age),
                                gender=int(gender),
                                transcription=transcription,
                                emotion=emotion,
                                chunk_filename=chunk_filename,
                                ner_tags=ner_entities
                            )

                            segment_data = {
                                'start_time': start_time,
                                'end_time': end_time,
                                'speaker': speaker,
                                'age': float(age),
                                'gender': int(gender),
                                'transcription': transcription,
                                'ner_tagged_text': ner_text,
                                'ner_entities': ner_entities,
                                'emotion': emotion,
                                'audio_file_path': os.path.abspath(audio_path)
                            }

                            all_data.append(segment_data)

                            chunk_data[chunk_filename].segments.append(segment)
                            chunk_data[chunk_filename].filepath = os.path.abspath(audio_path)

                        except Exception as e:
                            print(f"Error processing segment in chunk {chunk_filename}: {str(e)}")
                            continue

                        finally:
                            if os.path.exists(temp_segment_path):
                                os.remove(temp_segment_path)


                except Exception as e:
                    print(f"Error processing chunk {chunk_filename}: {str(e)}")
                    continue


        if all_data:
            jsonl_path_output = os.path.join(results_dir, f"{os.path.splitext(os.path.basename(jsonl_path))[0]}_processed_data.jsonl")
            with open(jsonl_path_output, 'w', encoding='utf-8') as f:
                for item in all_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Saved JSONL to: {jsonl_path_output}")


            chunk_texts = [
                {
                    "audio_filepath": data.filepath,
                    "text": data.get_formatted_text()
                }
                for data in chunk_data.values()
            ]

            audio_text_jsonl_path = os.path.join(results_dir, f"{os.path.splitext(os.path.basename(jsonl_path))[0]}_audio_text_pairs.jsonl")
            with open(audio_text_jsonl_path, 'w', encoding='utf-8') as f:
                for item in chunk_texts:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Saved audio-text pairs JSONL to: {audio_text_jsonl_path}")

            return all_data, chunk_texts

        return [], []

    except Exception as e:
        print(f"Error processing JSONL file {jsonl_path}: {str(e)}")
        return [], []


if __name__ == "__main__":
    download_dir = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/whissle1_hf.jsonl"  # Specify the single JSONL file path here
    output_dir = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/whissle_processed_hf.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing JSONL file: {download_dir}")
    print(f"Saving output to: {output_dir}")


    if os.path.isfile(download_dir) and download_dir.lower().endswith('.jsonl'):
        print(f"\nProcessing jsonl file: {download_dir}")
        df, chunk_texts = process_jsonl_data(download_dir, output_base_dir=output_dir)
        print(f"Successfully processed {download_dir}")
    else:
        print(f"Error: {download_dir} is not a valid JSONL file path.")