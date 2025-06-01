import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import time
import tarfile
import tempfile

def extract_files_from_nemo(nemo_path, temp_dir):
    """Extract model config and weights from .nemo file"""
    with tarfile.open(nemo_path, 'r:') as tar:
        # List all members to find the config and weights files
        members = tar.getmembers()
        config_member = None
        weights_member = None
        
        for member in members:
            if member.name.endswith('config.yaml') or member.name.endswith('model_config.yaml'):
                config_member = member
            elif member.name.endswith('model_weights.ckpt'):
                weights_member = member
        
        if config_member is None:
            raise ValueError("Could not find config file in .nemo archive")
        if weights_member is None:
            raise ValueError("Could not find model weights file in .nemo archive")
            
        print(f"Found config file: {config_member.name}")
        print(f"Found weights file: {weights_member.name}")
        
        # Extract both files
        tar.extract(config_member, temp_dir)
        tar.extract(weights_member, temp_dir)
        
        return (
            os.path.join(temp_dir, config_member.name),
            os.path.join(temp_dir, weights_member.name)
        )

def upload_with_retry(api, path, repo_id, path_in_repo, token, max_retries=3, delay=5):
    """Attempt to upload file with retries on failure"""
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
                commit_message=f"Upload {path_in_repo}"  # Add commit message to force update
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
    repo_id = "WhissleAI/meta_stt_euro_v1"
    local_dir = Path("/home/ubuntu/workspace/temp/PromptingNemo/hf_upload")
    nemo_model_path = Path("/home/ubuntu/workspace/experiments/euro/parakeet-ctc-0.6b-finetune-euro/2025-05-22_08-43-59/checkpoints/parakeet-ctc-0.6b-finetune-euro.nemo")
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
    model_dest = local_dir / "parakeet-ctc-0.6b-finetune-euro.nemo"
    print(f"Copying model file to {model_dest}")
    shutil.copy2(nemo_model_path, model_dest)

    # Extract model config and weights
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path, weights_path = extract_files_from_nemo(nemo_model_path, temp_dir)
        config_dest = local_dir / "model_config.yaml"
        weights_dest = local_dir / "model_weights.ckpt"
        shutil.copy2(config_path, config_dest)
        shutil.copy2(weights_path, weights_dest)

    # Create README
    readme_path = local_dir / "README.md"
    readme_content = """---
language:
- en
- es
- fr
- it
- de
- pt
library_name: nemo
datasets:
- mozilla-foundation/common_voice_8_0
- MLCommons/peoples_speech
- librispeech_asr
thumbnail: null
tags:
- automatic-speech-recognition
- speech
- audio
- FastConformer
- Conformer
- pytorch
- NeMo
- hf-asr-leaderboard
- ctc
- entity-tagging
- speaker-attributes
license: cc-by-4.0
---

# Meta STT Euro V1

This model is a fine-tuned version of NVIDIA's Parakeet CTC 0.6B model, enhanced with entity tagging, speaker attributes, and multi-language support for European languages.

## Model Details

- **Base Model**: Parakeet CTC 0.6B (FastConformer)
- **Fine-tuned on**: Mix of CommonVoice (6 European languages), People's Speech, Indian accented English, and LibriSpeech
- **Languages**: English, Spanish, French, Italian, German, Portuguese
- **Additional Features**: Entity tagging, speaker attributes (age, gender, emotion), and intent detection

## Output Format

The model provides rich transcriptions including:
- Entity tags (PERSON_NAME, ORGANIZATION, etc.)
- Speaker attributes (AGE, GENDER, EMOTION)
- Intent classification
- Language-specific transcription

Example output:
```
ENTITY_PERSON_NAME Robert Hoke END was educated at the ENTITY_ORGANIZATION Pleasant Retreat Academy END. AGE_45_60 GER_MALE EMOTION_NEUTRAL INTENT_INFORM
```

## Usage

```python
import nemo.collections.asr as nemo_asr

# Load model
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained('WhissleAI/meta_stt_euro_v1')

# Transcribe audio
transcription = asr_model.transcribe(['path/to/audio.wav'])
print(transcription[0])
```

## Training Data

The model was fine-tuned on:
- CommonVoice dataset (6 European languages)
- People's Speech English corpus
- Indian accented English
- LibriSpeech corpus (en, es, fr, it, pt)

## Model Architecture

Based on FastConformer [1] architecture with 8x depthwise-separable convolutional downsampling, trained using CTC loss.

## License

This model is licensed under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

## References

[1] [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)
[2] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Upload files
    files_to_upload = [
        (model_dest, "parakeet-ctc-0.6b-finetune-euro.nemo"),
        (readme_path, "README.md"),
        (config_dest, "model_config.yaml"),
        (weights_dest, "model_weights.ckpt")
    ]

    for file_path, repo_path in files_to_upload:
        print(f"\nUploading {repo_path}...")
        if not upload_with_retry(api, file_path, repo_id, repo_path, hf_token):
            print(f"Failed to upload {repo_path}")
            sys.exit(1)

    print("\nAll files uploaded successfully!")

if __name__ == "__main__":
    main()