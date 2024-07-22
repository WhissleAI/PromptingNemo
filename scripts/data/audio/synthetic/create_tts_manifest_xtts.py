import os
import json
import sys
import io
import random
from pydub import AudioSegment
from pathlib import Path
import numpy as np
import glob

from TTS.api import TTS  # Correct import statement for TTS

import subprocess
import tempfile

class ProcessFiles:
    def __init__(self, config):
        self.config = config
        self.clone_voices = {
            "EN": glob.glob("/external2/datasets/LibriSpeech/converted-wav/*.wav"),
            "ES": glob.glob("/external2/datasets/spanish/wav/*.wav"),
            "FR": glob.glob("/external2/datasets/french/wav/*.wav"),
            "DE": glob.glob("/external2/datasets/german/wav/*.wav"),
            "IT": glob.glob("/external2/datasets/italian/wav/*.wav"),
            "HI": glob.glob("/external2/datasets/hindi/wav/*.wav"),
            "PA": glob.glob("/external2/datasets/punjabi/corpus/clips_wav/*.wav"),
            "BN": glob.glob("/external2/datasets/bengali/wav/*.wav"),
            "MR": glob.glob("/external2/datasets/marathi/wav/*.wav"),
            "GU": glob.glob("/external2/datasets/gujarati/wav/*.wav"),
            "TE": glob.glob("/external2/datasets/telugu/wav/*.wav")
        }
        self.tts_iso_codes = {
            "EN": "eng",
            "ES": "spa",
            "FR": "fra",
            "DE": "deu",
            "IT": "it-IT",
            "HI": "hin",
            "PA": "pan",
            "BN": "ben",
            "MR": "mar",
            "GU": "guj",
            "TE": "tel"
        }
        self.all_noise_files = glob.glob("/external2/datasets/noise/smart_speaker_sounds_wav/*.wav")
        self.tts = None

    def change_speed(self, audio, speed=1.0):
        return audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * speed)
        }).set_frame_rate(audio.frame_rate)

    def adjust_volume(self, audio, volume_adjustment=0.0):
        return audio + volume_adjustment

    def add_noise(self, speech, noise_files, snr_db):
        speech = speech.apply_gain(-speech.max_dBFS)
        mixed_audio = speech

        for noise_file in noise_files:
            noise = AudioSegment.from_file(noise_file, format="wav")
            noise = noise.apply_gain(-noise.max_dBFS)

            # Random noise volume adjustment
            noise_volume_adjustment = random.uniform(self.config['noise_volume_min'], self.config['noise_volume_max'])
            noise = noise + noise_volume_adjustment

            if len(noise) > len(speech):
                start = random.randint(0, len(noise) - len(speech))
                noise = noise[start:start + len(speech)]
            else:
                # Loop noise if it's shorter than speech
                noise = noise * (len(speech) // len(noise) + 1)
                noise = noise[:len(speech)]

            # Apply time-varying noise levels
            duration = len(speech)
            segments = []
            for i in range(0, duration, 1000):  # Process in 1-second segments
                segment_end = min(i + 1000, duration)
                segment = noise[i:segment_end]
                segment_volume_adjustment = random.uniform(-5.0, 5.0)
                segment = segment + segment_volume_adjustment
                segments.append(segment)

            varied_noise = segments[0]
            for segment in segments[1:]:
                varied_noise = varied_noise.append(segment, crossfade=0)

            speech_power = speech.dBFS
            noise_power = speech_power - snr_db
            varied_noise = varied_noise.apply_gain(noise_power - varied_noise.dBFS)
            mixed_audio = mixed_audio.overlay(varied_noise, loop=True)
        
        mixed_audio = mixed_audio.normalize()
        return mixed_audio

    def add_reverb(self, audio):
        reverberance = random.uniform(self.config['reverberance_min'], self.config['reverberance_max'])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input, tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            temp_input_path = temp_input.name
            temp_output_path = temp_output.name
            audio.export(temp_input_path, format="wav")
            
            process = subprocess.run(
                ['sox', temp_input_path, temp_output_path, 'reverb', str(reverberance)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if process.returncode != 0:
                raise RuntimeError(f"sox command failed with error: {process.stderr.decode()}")
            
            output_audio = AudioSegment.from_file(temp_output_path, format="wav")

        # Clean up the temporary files
        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return output_audio



    def generate_audio(self, text, mode="train", language="EN", mixer=True):
        language_voices = self.clone_voices[language]
        sample_voice = random.choice(language_voices)
        buffer = io.BytesIO()

        self.tts.tts_to_file(
            text=text,
            file_path=buffer,
            speaker_wav=sample_voice,
            split_sentences=False,
        )
        buffer.seek(0)
        speed = random.uniform(self.config['speed_min'], self.config['speed_max'])
        volume_adjustment = random.uniform(self.config['volume_min'], self.config['volume_max'])
        audio = AudioSegment.from_file(buffer, format="wav")
        audio = audio.set_frame_rate(16000)
        audio = self.change_speed(audio, speed)
        audio = self.adjust_volume(audio, volume_adjustment)

        if mixer:
            num_noises = random.randint(1, 4)  # Choose a random number of noise files to mix
            noise_files = random.sample(self.all_noise_files, num_noises)
            snr = random.uniform(self.config['snr_min'], self.config['snr_max'])
            audio = self.add_noise(audio, noise_files, snr)
        
        audio = self.add_reverb(audio)
        audio = audio.normalize()
        volume_adjustment = random.uniform(self.config['volume_min'], 0.0)
        audio = self.adjust_volume(audio, volume_adjustment)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        return buffer.read()

    def list_available_voices(self):
        client = texttospeech.TextToSpeechClient()
        response = client.list_voices()
        voice_dict = {
            'EN': [], 'ES': [], 'FR': [], 'DE': [], 'IT': [], 'PT': [], 'NL': [], 'SV': []
        }
        language_mapping = {
            'EN': 'en', 'ES': 'es', 'FR': 'fr', 'DE': 'de',
            'IT': 'it', 'PT': 'pt', 'NL': 'nl', 'SV': 'sv'
        }
        for voice in response.voices:
            for language_code in voice.language_codes:
                for key, value in language_mapping.items():
                    if language_code.startswith(value):
                        voice_dict[key].append(voice.name)
        return voice_dict

    def synthesize_speech(self, text, voice_name, language_code='en-US'):
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )
        response = client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        return response.audio_content

    def add_white_noise(self, audio, noise_level):
        noise = AudioSegment.silent(duration=len(audio)).overlay(
            AudioSegment.silent(duration=len(audio)).apply_gain(noise_level)
        )
        combined = audio.overlay(noise)
        return combined

    def get_audio_duration(self, audio_file_path):
        audio = AudioSegment.from_file(audio_file_path)
        duration = len(audio) / 1000.0
        return duration

    def save_audio_to_file(self, audio_content, filename, noise_level):
        audio = AudioSegment.from_file(io.BytesIO(audio_content), format="wav")
        audio_with_noise = self.add_white_noise(audio, noise_level)
        audio_with_noise.export(filename, format="wav")
        print(f'Audio content written to {filename} with noise level {noise_level} dB')

    def process_files(self):
        runs = int(self.config['runs'])
        os.makedirs(self.config['audio_path'], exist_ok=True)
        manifest_file_path = self.config['manifest_file']
        with open(manifest_file_path, 'w', encoding='utf-8') as manifest_file:
            first_entry = True

            with open(self.config['clean_text_file'], 'r') as file:
                lines = file.readlines()

            with open(self.config['tagged_text_file'], 'r') as file:
                tagged_lines = file.readlines()

            for n in range(runs):
                for i, (line, tagged_line) in enumerate(zip(lines, tagged_lines)):
                    line_len = len(line.strip().split())
                    if line_len <= self.config['max_len']:
                        line = " ".join(line.split()[1:])
                        audio_content = self.generate_audio(line.strip(), self.config['mode'], self.config['language'], self.config['mixer'])

                        audio_file = os.path.join(self.config['audio_path'], f"{Path(self.config['clean_text_file']).stem}_line_{i}_run_{n}.wav")
                        noise_level = random.uniform(self.config['noise_min'], self.config['noise_max'])
                        self.save_audio_to_file(audio_content, audio_file, noise_level)
                        duration = self.get_audio_duration(audio_file)

                        entry = {
                            "audio_filepath": audio_file,
                            "text": tagged_line.strip(),
                            "duration": duration
                        }

                        if not first_entry:
                            manifest_file.write('\n')
                        else:
                            first_entry = False

                        manifest_file.write(json.dumps(entry, ensure_ascii=False))

            manifest_file.write('\n')
            print(f"Manifest file written to {manifest_file_path}")

    def validate_json(self, json_file):
        try:
            with open(json_file, 'r') as file:
                data = file.read()
            parsed_json = json.loads(data)
            print("JSON is valid and parsed successfully")
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            start = max(e.pos - 50, 0)
            end = min(e.pos + 50, len(data))
            print(f"Error near: {data[start:end]}")
            return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <language_code> <runs>")
        sys.exit(1)
    
    language_code = sys.argv[1]
    runs = sys.argv[2]

    config = {
        'language': language_code,
        'runs': runs,
        'input_folder': "/external2/datasets/text/synthetic_non-command/processed/",
        'audio_path': "/external2/datasets/synthetic_audio_r4/",
        'clean_text_file': f"/external2/datasets/text/synthetic_non-command/processed/tagged_{language_code}_notag.txt",
        'tagged_text_file': f"/external2/datasets/text/synthetic_non-command/processed/tagged_{language_code}_clean.txt",
        'manifest_file': f"/external2/datasets/text/synthetic_non-command/processed/manifest_{language_code}.json",
        'max_len': 30,
        'mode': 'train',
        'mixer': True,
        'speed_min': 0.8,
        'speed_max': 1.2,
        'volume_min': -14.0,
        'volume_max': 8.0,
        'snr_min': 0,
        'snr_max': 20,
        'noise_min': -10,
        'noise_max': 0,
        'noise_volume_min': -10.0,  # Minimum noise volume adjustment in dB
        'noise_volume_max': 10.0,   # Maximum noise volume adjustment in dB
        'reverberance_min': 10,  # Minimum reverberance percentage
        'reverberance_max': 50   # Maximum reverberance percentage
    }

    processor = ProcessFiles(config)

    language_iso_code = processor.tts_iso_codes[language_code]
    if language_code == "IT":
        processor.tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to("cuda")
    else:
        processor.tts = TTS(f"tts_models/{language_iso_code}/fairseq/vits", gpu=True)

    processor.process_files()