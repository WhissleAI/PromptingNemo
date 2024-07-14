import os
import json
import sys
import io
import random
from pydub import AudioSegment
from pydub.effects import speedup
from google.cloud import texttospeech
import glob

from TTS.api import TTS  # Correct import statement for TTS


tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

sample_train_voices = glob.glob("/external2/datasets/LibriSpeech/converted-wav/*.wav")
sample_valid_voices = glob.glob("/external2/datasets/LibriSpeech/test-converted-wav/*.wav")

def change_speed(audio, speed=1.0):
    if speed == 1.0:
        return audio
    new_frame_rate = int(audio.frame_rate * speed)
    return audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate}).set_frame_rate(audio.frame_rate)

def adjust_volume(audio, volume):
    return audio + volume

def generate_audio(text, mode="train"):
    # Select the appropriate sample voices based on the mode
    if mode == "train":
        sample_voices = sample_train_voices
    else:
        sample_voices = sample_valid_voices

    # Choose a random sample voice
    sample_voice = random.choice(sample_voices)

    # Create an in-memory bytes buffer to store the audio
    buffer = io.BytesIO()

    # Generate speech and save it to the buffer
    tts.tts_to_file(
        text=text,
        file_path=buffer,
        speaker_wav=sample_voice,
        language="en"
    )

    # Seek to the beginning of the buffer so it can be read
    buffer.seek(0)
    
    speed = random.uniform(0.8, 1.3)
    volume_adjustment = random.uniform(-8.0, 8.0)  # Adjust volume by -5dB to +5dB
    
    # Read the audio content and convert it to the desired sample rate
    audio = AudioSegment.from_file(buffer, format="wav")
    audio = audio.set_frame_rate(16000)

    # Adjust the speed of the audio
    audio = change_speed(audio, speed)
    audio = adjust_volume(audio, volume_adjustment)


    # Save the converted audio back to a bytes buffer
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    
    return buffer.read()

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

def process_files(clean_text, tagged_text, audio_path, manifest_file, mode="train", runs=30, max_len=30):
    
    runs = int(runs)
    
    os.system(f"mkdir -p {audio_path}")
    
    manifest_file = open(manifest_file, 'a', encoding='utf-8')

    first_entry = True

    with open(clean_text, 'r') as file:
        lines = file.readlines()

    with open(tagged_text, 'r') as file:
        tagged_lines = file.readlines()

    RUN = True
    
    for n in range(0,runs):
        
        for i, (line, tagged_line) in enumerate(zip(lines, tagged_lines)):
            #audio_content = synthesize_speech(line.strip(), voice_name)
            line_len = len(line.strip().split())
            
            if n == 0 and i < 4557:
                RUN = False
            else:
                RUN = True
            
            if RUN == True:    
                if line_len <= max_len:
                    audio_content = generate_audio(line.strip(), mode=mode)
                    audio_file = os.path.join(audio_path, f"{os.path.basename(clean_text).replace('.txt', '')}_line_{i}_run_{n}.wav")
                    
                    
                    noise_level = random.uniform(-30, -10)  # Random noise level between -30 dB and -10 dB
                    save_audio_to_file(audio_content, audio_file, noise_level)
                    duration = get_audio_duration(audio_file)
                    
                    entry = {
                        "audio_filepath": audio_file,
                        "text": tagged_line.strip(),
                        "duration": duration
                    }
                    
                    if not first_entry:
                        manifest_file.write('\n')  # Add a comma before each entry except the first one
                    else:
                        first_entry = False
                    
                    manifest_file.write(json.dumps(entry, ensure_ascii=False))

    manifest_file.write('\n')  # End of JSON array
    manifest_file.close()
    
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
    runs = sys.argv[6]
    
    process_files(clean_text, tagged_text, audio_path, manifest_file, mode=mode, runs=runs)
