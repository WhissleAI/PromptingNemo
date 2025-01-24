import os
import glob
import json
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
import subprocess

# Base path
BASE_PATH = '/external1/datasets/indicVoices_v3/hindi'

# Helper function to ensure directories exist
def ensure_directory(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            print(f"Conflict detected: {path} exists as a file. Removing...")
            os.remove(path)
    os.makedirs(path, exist_ok=True)

# Step 1: Extract datasets
def extract_datasets(language, tgz_directory, output_path):
    for subset in ["train", "valid"]:
        tar_file = os.path.join(tgz_directory, f"v3_{language}_{subset}.tgz")
        extract_path = output_path  # Extract directly into the output_path
        print(f"Extracting {subset} dataset for {language}...")
        subprocess.run(["tar", "-xzvf", tar_file, "-C", extract_path], check=True)
        print(f"Extracted {subset} dataset to {extract_path}")

# Step 2: Process JSON files and create manifests
def process_subset(language, subset, output_path):
    subset_path = os.path.join(output_path, f"{language}/v3/{subset}")
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
def process_language(language, tgz_directory, output_path):
    print(f"Processing language: {language}")
    ensure_directory(output_path)

    # Change working directory to output_path
    os.chdir(output_path)

    # Step 1: Extract the datasets
    extract_datasets(language, tgz_directory, output_path)

    # Step 2: Process train and valid subsets
    for subset in ["train", "valid"]:
        process_subset(language, subset, output_path)

    print(f"Processing completed for {language}!")

# Example usage
if __name__ == "__main__":
    

    tgz_directory = '/external1/datasets/indicVoices_v3/hindi'  # Specify the directory containing .tgz files
    languages = ["Hindi"]  # Add more languages as needed

    for lang in languages:
        output_path = os.path.join(BASE_PATH, f"{lang}_processed")
        process_language(lang, tgz_directory, output_path)
