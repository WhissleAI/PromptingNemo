import os
import glob
import json
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
import subprocess

# Base path
BASE_PATH = '/home/compute/hkoduri/AI4Bharat/PromptingNemo/scripts/data/audio/1person/real/AI4Bharat'

# Dataset links for multiple languages
DATASET_LINKS = {
    "Malayalam": {
        "valid": "https://indicvoices.ai4bharat.org/backend/download_dataset/v3_Malayalam_valid.tgz?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MzcxMzM0OTYsImlhdCI6MTczNjk2MDY5NiwiZW1haWwiOiJoa29kdXJpQHdoaXNzbGUuYWkifQ.f6OS79DKvHst_gc4nqg888cjOe3qHNIiUcaWAXxxJtE",  # Replace with valid token
        "train": "https://indicvoices.ai4bharat.org/backend/download_dataset/v3_Malayalam_train.tgz?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MzcxMzM0OTYsImlhdCI6MTczNjk2MDY5NiwiZW1haWwiOiJoa29kdXJpQHdoaXNzbGUuYWkifQ.f6OS79DKvHst_gc4nqg888cjOe3qHNIiUcaWAXxxJtE"   # Replace with valid token
    }
}

# Ensure required libraries are installed
def ensure_dependencies():
    try:
        import pydub
        import tqdm
    except ImportError:
        print("Installing required libraries...")
        subprocess.run(["pip", "install", "pydub", "tqdm"], check=True)

# Helper function to ensure directories exist
def ensure_directory(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            print(f"Conflict detected: {path} exists as a file. Removing...")
            os.remove(path)
    os.makedirs(path, exist_ok=True)

# Step 1: Download datasets
def download_datasets(language, links, output_path):
    for subset, link in links.items():
        tar_file = os.path.join(output_path, f"v3_{language}_{subset}.tgz")
        print(f"Downloading {subset} dataset for {language}...")
        subprocess.run(["wget", "-O", tar_file, link], check=True)
        print(f"Downloaded {subset} dataset to {tar_file}")

# Step 2: Extract datasets
def extract_datasets(language, output_path):
    for subset in ["train", "valid"]:
        tar_file = os.path.join(output_path, f"v3_{language}_{subset}.tgz")
        extract_path = output_path  # Extract directly into the output_path
        print(f"Extracting {subset} dataset for {language}...")
        subprocess.run(["tar", "-xzvf", tar_file, "-C", extract_path], check=True)
        print(f"Extracted {subset} dataset to {extract_path}")

# Step 3: Process JSON files and create manifests
def process_subset(language, subset, output_path):
    subset_path = os.path.join(output_path, f"{language}/rv3/{subset}")
    json_list = glob.glob(os.path.join(subset_path, '*.json'))
    manifest_path = os.path.join(output_path, f"{language}_manifest_{subset}.jsonl")
    wavs_path = os.path.join(output_path, f"{language}_wavs_{subset}")
    ensure_directory(wavs_path)

    with open(manifest_path, 'w', encoding='utf-8') as out_f:
        for json_file in tqdm(json_list, desc=f"Processing {subset} JSON files for {language}"):
            path = Path(json_file)
            path_without_ext = path.with_suffix('')

            with open(json_file, 'r') as f:
                wavfile = AudioSegment.from_file(str(path_without_ext) + '.wav')
                data = json.load(f)

                for idx, chunk in enumerate(data['verbatim']):
                    chunk_segment = wavfile[chunk['start'] * 1000:chunk['end'] * 1000]
                    chunk_path = os.path.join(wavs_path, f"{path.stem}_{idx}.wav")
                    chunk_segment.export(chunk_path, format="wav")

                    manifest = {
                        'path': chunk_path,
                        'duration': chunk['end'] - chunk['start'],
                        'dialect': data['state'],
                        'gender': data['gender'],
                        'age_group': data['age_group'],
                        'intent': data['task_name'],
                        'text': chunk['text']
                    }
                    json.dump(manifest, out_f, ensure_ascii=False)
                    out_f.write("\n")

    print(f"Manifest for {subset} saved at {manifest_path}")

# Main processing function
def process_language(language, links):
    print(f"Processing language: {language}")
    output_path = os.path.join(BASE_PATH, f"{language}_processed")
    ensure_directory(output_path)

    # Change working directory to output_path
    os.chdir(output_path)

    # Step 1: Download the datasets
    download_datasets(language, links, output_path)

    # Step 2: Extract the datasets
    extract_datasets(language, output_path)

    # Step 3: Process train and valid subsets
    for subset in ["train", "valid"]:
        process_subset(language, subset, output_path)

    print(f"Processing completed for {language}!")

# Example usage
if __name__ == "__main__":
    ensure_dependencies()
    for lang, links in DATASET_LINKS.items():
        process_language(lang, links)
