#!/usr/bin/env python3

import argparse
import io
import wave
from pathlib import Path
from .piper import PiperVoice
#from nemo_text_processing.text_normalization.normalize import Normalizer
#normalizer = Normalizer(input_case='cased', lang='en')

import re

def clean_text_for_piper(text: str) -> str:
    
    #text = normalizer.normalize(text, verbose=True, punct_post_process=True)
    
    # Remove newlines and extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove bullet points and numbering
    text = re.sub(r'\d+\.\s*', '', text)
    text = re.sub(r'\s*-\s*', '', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s,.]', '', text)
    
    # Handle common abbreviations
    text = re.sub(r"\bSt\.\b", "Street", text)
    text = re.sub(r"\bDr\.\b", "Doctor", text)
    text = re.sub(r"\bMr\.\b", "Mister", text)
    text = re.sub(r"\bMrs\.\b", "Missus", text)
    text = re.sub(r"\bMs\.\b", "Miss", text)
    text = re.sub(r"\bAve\.\b", "Avenue", text)
    text = re.sub(r"\bRd\.\b", "Road", text)
    text = re.sub(r"\bBlvd\.\b", "Boulevard", text)
    
    # Ensure text is simple and clear
    sentences = re.split(r'[.!?]', text)
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            cleaned_sentences.append(sentence)
    
    return '. '.join(cleaned_sentences) + '.'

class PiperSynthesizer:
    def __init__(self, model_path: str, config_path: str, length_scale: float = None):
        self.model_path = model_path
        self.config_path = config_path

        # Load voice model
        self.voice = PiperVoice.load(model_path, config_path=config_path)

        # Override length scale if provided
        if length_scale is not None:
            self.voice.length_scale = length_scale

    def synthesize(self, text: str) -> bytes:
        # Synthesize text to audio and write to a BytesIO buffer
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                self.voice.synthesize(text, wav_file)
            wav_data = wav_io.getvalue()
        
        return wav_data

    def convert_to_wav(self, audio_data):
        sample_rate = self.voice.sample_rate
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                # Ensure data is in int16 format
                wav_data = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(wav_data.tobytes())
            return wav_io.getvalue()
