import json
import sys
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
import shutil
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess
import time

# Load your NeMo manifest file
nemo_manifest_path = sys.argv[1]
repo_name = sys.argv[2]  # Repository name

# Set a custom cache directory with proper permissions
os.environ['HF_HOME'] = '/projects/whissle/custom_cache_directory'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

with open(nemo_manifest_path, 'r') as f:
    nemo_manifest = [json.loads(line) for line in f]

# Convert to Hugging Face Dataset and then to a DataFrame
data = {
    'audio': [item['audio_filepath'] for item in nemo_manifest],
    'text': [item['text'] for item in nemo_manifest],
    'duration': [item['duration'] for item in nemo_manifest],
    'tasks': [["transcription", "entities", "age", "gender", "dialect"] for item in nemo_manifest],
}
df = pd.DataFrame(data)

# Save DataFrame as Parquet file
parquet_path = "./dataset_repo"
os.makedirs(parquet_path, exist_ok=True)
parquet_file = os.path.join(parquet_path, "dataset.parquet")
table = pa.Table.from_pandas(df)
pq.write_table(table, parquet_file)

# Create temporary directory for audio files
audio_files_dir = os.path.join(parquet_path, "audio_files")
os.makedirs(audio_files_dir, exist_ok=True)

# Copy only the audio files specified in the manifest
for item in nemo_manifest:
    audio_file = item['audio_filepath']
    shutil.copy(audio_file, audio_files_dir)

# Log in to Hugging Face
token = "<add-hf-token>"  # Replace with your actual token
login(token=token, add_to_git_credential=True)

# Set up git credentials helper
subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=True)

# Initialize the repository using HfApi
api = HfApi()
full_repo_name = f"WhissleAI/{repo_name}"

# Check if the repository exists, if not, create it
try:
    repo_info = api.repo_info(repo_id=full_repo_name, repo_type="dataset")
    print(f"Repository {full_repo_name} already exists.")
except Exception as e:
    print(f"Repository {full_repo_name} not found. Creating it.")
    api.create_repo(repo_id=full_repo_name, repo_type="dataset")
    print(f"Repository {full_repo_name} created.")

# Create the dataset card
dataset_card_content = f"""
# Dataset Card for {repo_name}

## Dataset Description

This dataset contains audio files and their corresponding transcriptions. It is designed for training and evaluating speech recognition models.

### Structure

- `audio`: Paths to the audio files.
- `text`: Corresponding transcriptions.
- `duration`: Duration of each audio file.
- `tasks`: Tasks related to the audio files.

## Usage

This dataset can be used for training automatic speech recognition (ASR) models. 

### Train/Test Split

The dataset is split into training and test sets.

## License

Specify the license under which the dataset is distributed.

## Citation

Include any relevant citation information here.
"""

with open(os.path.join(parquet_path, "dataset_card.md"), "w") as f:
    f.write(dataset_card_content)

# Function to upload files in batches and skip errors
def upload_in_batches(files, folder_path, repo_id, token, batch_size=100):
    skipped_files = []  # Keep track of skipped files
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batch_folder = f"./batch_{i//batch_size}"
        os.makedirs(batch_folder, exist_ok=True)
        for file in batch_files:
            relative_path = os.path.relpath(file, folder_path)
            dest_path = os.path.join(batch_folder, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(file, dest_path)
        try:
            api.upload_folder(
                folder_path=batch_folder,
                path_in_repo=".",
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
            print(f"Batch {i//batch_size + 1} uploaded successfully.")
        except Exception as e:
            print(f"Error uploading batch {i//batch_size + 1}: {e}")
            # Skip this batch or file if the payload is too large
            skipped_files.extend(batch_files)
        finally:
            shutil.rmtree(batch_folder)

    if skipped_files:
        print("The following files were skipped due to upload errors:")
        for skipped_file in skipped_files:
            print(skipped_file)

# Get the list of audio files to be uploaded
audio_files = [os.path.join(audio_files_dir, os.path.basename(item['audio_filepath'])) for item in nemo_manifest]

# Push the dataset and files to the repository in batches
try:
    # Upload the Parquet file and dataset card first
    api.upload_folder(
        folder_path=parquet_path,
        path_in_repo=".",
        repo_id=full_repo_name,
        repo_type="dataset",
        token=token
    )
    print("Parquet file and dataset card uploaded successfully.")

    # Upload the audio files in batches and skip problematic files
    upload_in_batches(audio_files, audio_files_dir, full_repo_name, token, batch_size=20)
    print("Audio files successfully pushed to Hugging Face Hub (some files may have been skipped).")
except Exception as e:
    print(f"An error occurred during upload: {e}")
