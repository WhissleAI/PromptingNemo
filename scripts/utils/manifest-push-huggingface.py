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
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor
import backoff
from datetime import datetime, timedelta
import random

# Initialize HfApi
api = HfApi()

class RateLimitHandler:
    def __init__(self, max_retries=5, initial_delay=60):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.retry_counts = {}
        self.last_request_time = {}
        self.min_request_interval = 2  # Minimum seconds between requests

    def wait_for_rate_limit(self, file_path):
        current_time = time.time()
        if file_path in self.last_request_time:
            elapsed = current_time - self.last_request_time[file_path]
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time[file_path] = time.time()

    def handle_rate_limit(self, file_path):
        if file_path not in self.retry_counts:
            self.retry_counts[file_path] = 0
        
        self.retry_counts[file_path] += 1
        
        if self.retry_counts[file_path] > self.max_retries:
            return False
        
        delay = self.initial_delay * (2 ** (self.retry_counts[file_path] - 1))
        delay += random.uniform(0, 10)  # Add jitter
        print(f"Rate limited for {file_path}. Waiting {delay:.2f} seconds before retry.")
        time.sleep(delay)
        return True

rate_limit_handler = RateLimitHandler()

def upload_file_with_retry(file_path, repo_id, path_in_repo, token):
    try:
        rate_limit_handler.wait_for_rate_limit(file_path)
        
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        return True
    except Exception as e:
        if "429 Client Error: Too Many Requests" in str(e):
            if rate_limit_handler.handle_rate_limit(file_path):
                return upload_file_with_retry(file_path, repo_id, path_in_repo, token)
            else:
                print(f"Max retries exceeded for {file_path}")
                return False
        else:
            print(f"Error uploading {file_path}: {str(e)}")
            return False

def upload_files_parallel(files, repo_id, token, audio_files_dir, max_workers=2, batch_size=5):
    """Upload files in parallel with reduced concurrency and rate limiting"""
    failed_files = []
    successful_files = []
    
    def upload_batch(batch):
        results = []
        for file in batch:
            try:
                relative_path = os.path.relpath(file, audio_files_dir)
                success = upload_file_with_retry(
                    file,
                    repo_id=repo_id,
                    path_in_repo=f"audio_files/{relative_path}",
                    token=token
                )
                results.append((file, success))
                # Add delay between files in the same batch
                time.sleep(2)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                results.append((file, False))
        return results

    # Split files into smaller batches
    file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    
    with tqdm(total=len(files), desc="Uploading files") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch in file_batches:
                try:
                    results = executor.submit(upload_batch, batch).result()
                    for file, success in results:
                        if success:
                            successful_files.append(file)
                            # Save progress after each successful upload
                            save_progress(successful_files, failed_files)
                        else:
                            failed_files.append(file)
                        pbar.update(1)
                    # Add delay between batches
                    time.sleep(5)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    failed_files.extend(batch)
                    pbar.update(len(batch))
                    save_progress(successful_files, failed_files)

    return successful_files, failed_files

def save_progress(successful_files, failed_files):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"upload_progress_{timestamp}.json", "w") as f:
        json.dump({
            "successful_files": successful_files,
            "failed_files": failed_files,
            "timestamp": timestamp
        }, f)

def load_progress(progress_file):
    with open(progress_file, "r") as f:
        progress = json.load(f)
    return progress["successful_files"], progress["failed_files"]

def create_dataset_card(repo_name):
    return """---
annotations_creators:
- other
language:
- hi
language_creators:
- other
license:
- cc-by-4.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- automatic-speech-recognition
task_ids:
- speech-recognition
paperswithcode_id: null
pretty_name: {repo_name}
---

# Dataset Card for {repo_name}

## Dataset Description

This dataset contains audio files and their corresponding transcriptions in Hindi for automatic speech recognition (ASR) tasks.

### Languages
The dataset is primarily in Hindi.

### Data Collection
The dataset was collected through automated processes and manual transcription.

## Dataset Structure
The dataset contains:
- Audio files (.wav format)
- Transcriptions
- Duration information
- Additional task annotations

### Data Fields
- `audio`: Path to the audio file
- `text`: Transcription of the audio
- `duration`: Length of the audio in seconds
- `tasks`: List of associated tasks

## Additional Information
- **License**: CC-BY 4.0
- **Version**: 1.0.0
- **Publisher**: WhissleAI
""".format(repo_name=repo_name)

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <manifest_path> <repo_name> <hf_token>")
        sys.exit(1)

    # Load your NeMo manifest file
    nemo_manifest_path = sys.argv[1]
    repo_name = sys.argv[2]
    token = sys.argv[3]

    # Set custom cache directory
    os.environ['HF_HOME'] = '/projects/whissle/custom_cache_directory'
    os.makedirs(os.environ['HF_HOME'], exist_ok=True)

    # Load manifest
    print("Loading manifest file...")
    with open(nemo_manifest_path, 'r') as f:
        nemo_manifest = [json.loads(line) for line in f]

    # Create DataFrame
    print("Creating DataFrame...")
    data = {
        'audio': [item['audio_filepath'] for item in nemo_manifest],
        'text': [item['text'] for item in nemo_manifest],
        'duration': [item['duration'] for item in nemo_manifest],
        'tasks': [["transcription", "entities", "age", "gender", "dialect"] for item in nemo_manifest],
    }
    df = pd.DataFrame(data)

    # Create directories
    parquet_path = "/projects/whissle/hfdataset_repo"
    os.makedirs(parquet_path, exist_ok=True)
    audio_files_dir = os.path.join(parquet_path, "audio_files")
    os.makedirs(audio_files_dir, exist_ok=True)

    # Save Parquet file
    print("Saving Parquet file...")
    parquet_file = os.path.join(parquet_path, "dataset.parquet")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_file)

    # Copy audio files
    print("Copying audio files...")
    for item in tqdm(nemo_manifest, desc="Copying files"):
        audio_file = item['audio_filepath']
        dest_path = os.path.join(audio_files_dir, os.path.basename(audio_file))
        if not os.path.exists(dest_path):
            shutil.copy(audio_file, dest_path)

    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    login(token=token, add_to_git_credential=True)

    # Set up git credentials
    subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=True)

    # Create or verify repository
    full_repo_name = f"WhissleAI/{repo_name}"
    try:
        api.repo_info(repo_id=full_repo_name, repo_type="dataset")
        print(f"Repository {full_repo_name} exists.")
    except Exception:
        print(f"Creating repository {full_repo_name}")
        api.create_repo(repo_id=full_repo_name, repo_type="dataset", private=False)

    # Create and save dataset card
    dataset_card_content = create_dataset_card(repo_name)
    with open(os.path.join(parquet_path, "README.md"), "w") as f:
        f.write(dataset_card_content)

    # Upload parquet file and README first
    print("Uploading parquet file and README...")
    try:
        api.upload_file(
            path_or_fileobj=parquet_file,
            path_in_repo="dataset.parquet",
            repo_id=full_repo_name,
            repo_type="dataset",
            token=token
        )
        time.sleep(2)  # Add delay between uploads
        api.upload_file(
            path_or_fileobj=os.path.join(parquet_path, "README.md"),
            path_in_repo="README.md",
            repo_id=full_repo_name,
            repo_type="dataset",
            token=token
        )
    except Exception as e:
        print(f"Error uploading initial files: {str(e)}")
        return

    # Get list of audio files
    audio_files = [os.path.join(audio_files_dir, os.path.basename(item['audio_filepath'])) 
                   for item in nemo_manifest]

    # Check for previous progress
    progress_files = [f for f in os.listdir(".") if f.startswith("upload_progress_") and f.endswith(".json")]
    if progress_files:
        latest_progress = max(progress_files)
        print(f"Found previous progress file: {latest_progress}")
        response = input("Would you like to resume from previous progress? (y/n): ")
        if response.lower() == 'y':
            successful_files, failed_files = load_progress(latest_progress)
            # Remove already uploaded files from the list
            audio_files = [f for f in audio_files if f not in successful_files]
            print(f"Resuming upload with {len(audio_files)} remaining files...")

    print(f"Starting upload of {len(audio_files)} audio files...")
    successful_files, failed_files = upload_files_parallel(
        audio_files,
        full_repo_name,
        token,
        audio_files_dir,
        max_workers=2,  # Reduced number of workers
        batch_size=5    # Reduced batch size
    )

    # Save final progress
    save_progress(successful_files, failed_files)

    # Report results
    print(f"\nUpload complete!")
    print(f"Successfully uploaded: {len(successful_files)} files")
    if failed_files:
        print(f"Failed to upload: {len(failed_files)} files")
        with open("failed_uploads.txt", "w") as f:
            for file in failed_files:
                f.write(f"{file}\n")
        print("Failed files have been written to 'failed_uploads.txt'")
        
if __name__ == "__main__":
    main()    