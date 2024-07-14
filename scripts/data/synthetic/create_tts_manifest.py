import os
import json
import sys
import io
import random
from pydub import AudioSegment
from pydub.utils import mediainfo
from google.cloud import texttospeech

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ksingla/workspace/medical-ner/keys/google-tts-key.json"

def synthesize_speech(text, voice_name, language_code='en-US'):
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

def add_white_noise(audio, noise_level):
    noise = AudioSegment.silent(duration=len(audio)).overlay(
        AudioSegment.silent(duration=len(audio)).apply_gain(noise_level)
    )

    combined = audio.overlay(noise)
    return combined

def get_audio_duration(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    duration = len(audio) / 1000.0  # pydub returns duration in milliseconds
    return duration

def save_audio_to_file(audio_content, filename, noise_level):
    audio = AudioSegment.from_file(io.BytesIO(audio_content), format="wav")
    audio_with_noise = add_white_noise(audio, noise_level)
    audio_with_noise.export(filename, format="wav")
    print(f'Audio content written to {filename} with noise level {noise_level} dB')

def process_files(clean_text, tagged_text, audio_path, manifest_file, mode="train"):
    train_voices = [
        "en-US-Wavenet-A", "en-US-Wavenet-B", "en-US-Wavenet-C",
        "en-US-Wavenet-D", "en-US-Wavenet-E", "en-GB-Wavenet-A",
        "en-GB-Wavenet-B", "en-GB-Wavenet-C", "en-GB-Wavenet-D",
        "en-IN-Wavenet-A", "en-IN-Wavenet-B", "en-IN-Wavenet-C"
    ]
    
    dev_voices = [
        "en-IN-Wavenet-D", "en-AU-Wavenet-A", "en-AU-Wavenet-B",
        "en-AU-Wavenet-C", "en-AU-Wavenet-D"
    ]
    
    if mode == "train":
        voices = train_voices
    else:
        voices = dev_voices
    
    os.system(f"mkdir -p {audio_path}")
    
    with open(manifest_file, 'w', encoding='utf-8') as f:
    
        first_entry = True

        with open(clean_text, 'r') as file:
            lines = file.readlines()

        with open(tagged_text, 'r') as file:
            tagged_lines = file.readlines()

        for i, (line, tagged_line) in enumerate(zip(lines, tagged_lines)):
            for j, voice_name in enumerate(voices):
                
                try:
                    audio_content = synthesize_speech(line.strip(), voice_name)
                    
                    audio_file = os.path.join(audio_path, f"{os.path.basename(clean_text).replace('.txt', '')}_line_{i}_voice_{j}.wav")
                    if os.path.exists(audio_file):
                        print(f"Skipping {audio_file} as it already exists")
                        continue
                    else:
                        audio_content = synthesize_speech(line.strip(), voice_name)
                        noise_level = random.uniform(-30, -10)  # Random noise level between -30 dB and -10 dB
                        save_audio_to_file(audio_content, audio_file, noise_level)
                    
                    duration = get_audio_duration(audio_file)
                    
                    entry = {
                        "audio_filepath": audio_file,
                        "text": tagged_line.strip(),
                        "duration": duration
                    }
                    
                    if not first_entry:
                        f.write('\n')  # Add a comma before each entry except the first one
                    else:
                        first_entry = False
                    
                    f.write(json.dumps(entry, ensure_ascii=False))
                except:
                    continue
        
        f.write('\n')  # End of JSON array

    print(f"Manifest file written to {manifest_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <data_path>")
        sys.exit(1)

    clean_text = sys.argv[1]
    tagged_text = sys.argv[2]
    manifest_file = sys.argv[3]
    audio_path = sys.argv[4]
    mode = sys.argv[5]
    
    process_files(clean_text, tagged_text, audio_path, manifest_file, mode=mode)
