import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import time

def upload_with_retry(api, path, repo_id, path_in_repo, token, max_retries=3, delay=5):
    """Attempt to upload file with retries on failure"""
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token
            )
            print(f"Successfully uploaded {path_in_repo}")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Upload attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                print(f"Error: {str(e)}")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Failed to upload {path_in_repo} after {max_retries} attempts: {str(e)}")
                return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <huggingface_token>")
        sys.exit(1)

    # Configuration
    repo_id = "WhissleAI/speech-tagger_be_ctc_meta"
    local_dir = Path("/projects/whissle/experiments/bengali-hf")
    nemo_model_path = Path("/projects/whissle/experiments/bengali-adapter-ai4bharat/2024-10-27_23-42-00/checkpoints/bengali-adapter-ai4bharat.nemo")
    hf_token = sys.argv[1]

    # Set up environment
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    local_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Hugging Face API
    api = HfApi()

    # Ensure repository exists
    try:
        api.repo_info(repo_id)
        print(f"Repository {repo_id} already exists")
    except RepositoryNotFoundError:
        print(f"Creating new repository: {repo_id}")
        create_repo(repo_id, token=hf_token, private=False)
    except Exception as e:
        print(f"Error checking/creating repository: {str(e)}")
        sys.exit(1)

    # Copy model file
    model_dest = local_dir / "bengali-adapter-ai4bharat.nemo"
    print(f"Copying model file to {model_dest}")
    shutil.copy2(nemo_model_path, model_dest)

    # Create README
    readme_path = local_dir / "README.md"
    readme_content = """# Bengali Speech Tagger - Conformer CTC Model

This speech tagger performs transcription for Bengali, annotates key entities, predicts speaker age, dialect and intent.

## Model Details

- **Model Type**: NeMo ASR
- **Architecture**: Conformer CTC
- **Language**: Bengali
- **Training Data**: AI4Bharat IndicVoices Bengali V1 and V2 dataset
- **Task**: Speech Recognition with Entity Tagging

## Usage

```python
import nemo.collections.asr as nemo_asr

# Load model
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained('WhissleAI/speech-tagger_be_ctc_meta')

# Transcribe audio
transcription = asr_model.transcribe(['path/to/audio.wav'])
print(transcription[0])
```

## Model Training

- Base model: Conformer CTC
- Fine-tuned on AI4Bharat IndicVoices Marathi dataset
- Optimized for real-time transcription

## License & Attribution

Please cite AI4Bharat when using this model:
https://indicvoices.ai4bharat.org/
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Upload files
    files_to_upload = [
        (model_dest, "bengali-adapter-ai4bharat.nemo"),
        (readme_path, "README.md")
    ]

    for file_path, repo_path in files_to_upload:
        print(f"\nUploading {repo_path}...")
        if not upload_with_retry(api, file_path, repo_id, repo_path, hf_token):
            print(f"Failed to upload {repo_path}")
            sys.exit(1)

    print("\nAll files uploaded successfully!")

if __name__ == "__main__":
    main()