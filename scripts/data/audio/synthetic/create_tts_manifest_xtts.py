import os
import json
import sys
import io
import random
from pydub import AudioSegment
from pydub.effects import speedup
from google.cloud import texttospeech
from pathlib import Path

import glob

from TTS.api import TTS  # Correct import statement for TTS



clone_train_voices = glob.glob("/external2/datasets/LibriSpeech/converted-wav/*.wav")
clone_valid_voices = glob.glob("/external2/datasets/LibriSpeech/test-converted-wav/*.wav")


english_voices = glob.glob("/external2/datasets/LibriSpeech/converted-wav/*.wav")
spanish_voices = glob.glob("/external2/datasets/spanish/wav/*.wav")
french_voices = glob.glob("/external2/datasets/french/wav/*.wav")
german_voices = glob.glob("/external2/datasets/german/wav/*.wav")
italian_voices = glob.glob("/external2/datasets/italian/wav/*.wav")
hindi_voices = glob.glob("/external2/datasets/hindi/wav/*.wav")
punjabi_voices = glob.glob("/external2/datasets/punjabi/corpus/clips_wav/*.wav")
bengali_voices = glob.glob("/external2/datasets/bengali/wav/*.wav")
marathi_voices = glob.glob("/external2/datasets/marathi/wav/*.wav")
gujarati_voices = glob.glob("/external2/datasets/gujarati/wav/*.wav")
telugu_voices = glob.glob("/external2/datasets/telugu/wav/*.wav")

clone_voices = {
                "EN": english_voices,
                "ES": spanish_voices,
                "FR": french_voices,
                "DE": german_voices,
                "IT": italian_voices,
                "HI": hindi_voices,
                "PA": punjabi_voices,
                "BN": bengali_voices,
                "MR": marathi_voices,
                "GU": gujarati_voices,
                "TE": telugu_voices,
}

tts_iso_codes = {
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
    "TE": "tel",
}


def change_speed(audio, speed=1.0):
    if speed == 1.0:
        return audio
    new_frame_rate = int(audio.frame_rate * speed)
    return audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate}).set_frame_rate(audio.frame_rate)

def adjust_volume(audio, volume):
    return audio + volume

def generate_audio(text, mode="train", language="EN"):
    # Select the appropriate sample voices based on the mode

    language_voices = clone_voices[language]

    # Choose a random sample voice
    sample_voice = random.choice(language_voices)

    # Create an in-memory bytes buffer to store the audio
    buffer = io.BytesIO()

    # Generate speech and save it to the buffer
    print("Speaker wav: ", sample_voice)
    tts.tts_to_file(
        text=text,
        file_path=buffer,
        speaker_wav=sample_voice,
    )

    # Seek to the beginning of the buffer so it can be read
    buffer.seek(0)
    
    speed = random.uniform(0.8, 1.2)
    volume_adjustment = random.uniform(-14.0, 8.0)
    
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

def list_available_voices():
    # Initialize the Text-to-Speech client
    client = texttospeech.TextToSpeechClient()

    # Performs the list voices request
    response = client.list_voices()

    # Initialize a dictionary to store voices by language code
    voice_dict = {
        'EN': [], 'ES': [], 'FR': [], 'DE': [], 'IT': [], 'PT': [], 'NL': [], 'SV': []
    }

    # Define the language code mapping
    language_mapping = {
        'EN': 'en', 'ES': 'es', 'FR': 'fr', 'DE': 'de',
        'IT': 'it', 'PT': 'pt', 'NL': 'nl', 'SV': 'sv'
    }

    # Populate the dictionary with available voices
    for voice in response.voices:
        for language_code in voice.language_codes:
            for key, value in language_mapping.items():
                if language_code.startswith(value):
                    voice_dict[key].append(voice.name)

    return voice_dict

google_voices = list_available_voices()

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

def process_files(clean_text_file, tagged_text_file, audio_path,
                  manifest_file, mode="train",
                  runs=30, max_len=30, language="EN"):
    
    runs = int(runs)
    
    os.system(f"mkdir -p {audio_path}")
    
    manifest_file = open(manifest_file, 'w', encoding='utf-8')
    
    first_entry = True

    with open(clean_text_file, 'r') as file:
        lines = file.readlines()

    with open(tagged_text_file, 'r') as file:
        tagged_lines = file.readlines()


    for n in range(0,runs):
        
        for i, (line, tagged_line) in enumerate(zip(lines, tagged_lines)):
            #audio_content = synthesize_speech(line.strip(), voice_name)
            line_len = len(line.strip().split())
            try:
                if line_len <= max_len:                    

                    line = " ".join(line.split()[1:])
                    audio_content = generate_audio(line.strip(), mode=mode, language=language)

                    audio_file = os.path.join(audio_path, f"{os.path.basename(clean_text).replace('.txt', '')}_line_{i}_run_{n}.wav")                    
                    
                    noise_level = random.uniform(-20, 0)  # Random noise level between -30 dB and -10 dB
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
            except:
                continue

                    

    manifest_file.write('\n')  # End of JSON array
    manifest_file.close()
    
    print(f"Manifest file written to {manifest_file}")

def validate_json(json_file):
    try:
        with open(json_file, 'r') as file:
            data = file.read()
        parsed_json = json.loads(data)
        print("JSON is valid and parsed successfully")
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        # Print offending part for debugging
        start = max(e.pos - 50, 0)
        end = min(e.pos + 50, len(data))
        print(f"Error near: {data[start:end]}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <data_path>")
        sys.exit(1)

    runs = sys.argv[1]
    
    language_codes = clone_voices.keys()
    
    language_codes = ["EN", "ES", "FR", "DE", "IT", "HI", "PA", "BN", "MR", "GU", "TE"]
    
    language_codes = ["EN"]

    for language in language_codes:
        
        language_iso_code = tts_iso_codes[language]
        
        input_folder = "/home/ksingla/workspace/PromptingNemo/data_v2/synthetic/processed"
        audio_path = "/external2/datasets/synthetic_audio/"
        clean_text = Path(input_folder + f"/tagged_" + language + "_notag.txt")
        tagged_text = Path(input_folder + f"/tagged_" + language + "_clean.txt")
        manifest_file = Path(input_folder + f"/manifest_" + language + ".json")
        mode = "train"
        
        if language == "IT":
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to("cuda")
        else:
            tts = TTS("tts_models/"+language_iso_code+"/fairseq/vits", gpu=True)

        process_files(clean_text, tagged_text, audio_path, manifest_file, mode=mode, runs=runs, language=language)

        #validate_json(manifest_file)