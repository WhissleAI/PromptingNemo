import json
import os
import datasets
import requests

# Define the European languages and desired total duration
languages = ['en', 'de', 'fr', 'es', 'it']
total_duration_hours = 500
total_duration_seconds = total_duration_hours * 3600

# Specify the folder to save downloaded audio files
audio_files_folder = '/external2/datasets/commonvoice'

# Ensure the audio files folder exists
os.makedirs(audio_files_folder, exist_ok=True)

# Initialize a list to hold all the examples and a counter for total duration
all_examples = []
total_duration = 0

# Function to create NeMo manifest entries
def create_nemo_manifest_entry(audio_path, duration, text):
    return {
        "audio_filepath": audio_path,
        "duration": duration,
        "text": text
    }

# Helper function to download and process a subset of the dataset
def download_subset(language, samples_per_language=10000):
    dataset = datasets.load_dataset('mozilla-foundation/common_voice_11_0', language, split='train[:{}]'.format(samples_per_language))
    return dataset

# Function to download an audio file and save it locally
def download_audio_file(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Download and process each language dataset
for lang in languages:
    dataset = download_subset(lang)
    for example in dataset:
        if total_duration >= total_duration_seconds:
            break
        if not example["audio"]["array"]:
            continue

        audio_url = example["audio"]["path"]
        audio_filename = os.path.basename(audio_url)
        audio_path = os.path.join(audio_files_folder, audio_filename)

        # Download the audio file
        download_audio_file(audio_url, audio_path)

        # Update total duration and add entry to manifest
        total_duration += example["audio"]["duration"]
        all_examples.append(create_nemo_manifest_entry(audio_path, example["audio"]["duration"], example["sentence"]))

    if total_duration >= total_duration_seconds:
        break

# Save the manifest to a file
with open('nemo_manifest.json', 'w') as f:
    for entry in all_examples:
        f.write(json.dumps(entry) + '\n')

print("NeMo manifest created successfully.")
