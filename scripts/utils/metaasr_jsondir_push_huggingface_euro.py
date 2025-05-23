#!/usr/bin/env python3
import os
import json
import argparse
from datasets import Dataset, DatasetDict
from huggingface_hub import login, HfApi, list_repo_files

def read_json_files(input_dir, split):
    """Read all JSON files in the split directory and return a list of samples."""
    split_dir = os.path.join(input_dir, split)
    samples = []
    if not os.path.isdir(split_dir):
        return samples
    for fname in os.listdir(split_dir):
        if fname.endswith('.json'):
            # Extract source and language from filename (e.g., train_commonvoice_de.json -> commonvoice_de)
            parts = fname.split('_', 2)
            if len(parts) >= 3:
                source = f"{parts[1]}_{parts[2].replace('.json', '')}"
            else:
                source = fname.split('_', 1)[-1].replace('.json', '')
            
            with open(os.path.join(split_dir, fname), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        if all(k in sample for k in ("audio_filepath", "text", "duration")):
                            sample['source'] = source
                            samples.append(sample)
                    except Exception as e:
                        print(f"Error parsing line in {fname}: {e}")
    return samples

def make_readme(meta_info, example_samples):
    readme = """# Meta Speech Recognition European Languages Dataset (v1)

This dataset contains only the metadata (JSON/Parquet) for European language speech recognition samples.  
**Audio files are NOT included.**

## Data Download Links

### CommonVoice
- [CommonVoice Dataset](https://commonvoice.mozilla.org/en/datasets)
  - German (de)
  - English (en)
  - Spanish (es)
  - French (fr)
  - Italian (it)
  - Portuguese (pt)

### Multilingual LibriSpeech (MLS)
- [Multilingual LibriSpeech Dataset](https://www.openslr.org/94/)
  - German: [mls_german.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_german.tar.gz)
  - English: [mls_english.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_english.tar.gz)
  - Spanish: [mls_spanish.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_spanish.tar.gz)
  - French: [mls_french.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_french.tar.gz)
  - Italian: [mls_italian.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_italian.tar.gz)
  - Portuguese: [mls_portuguese.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_portuguese.tar.gz)

### People's Speech
- [People's Speech Dataset](https://huggingface.co/datasets/MLCommons/peoples_speech)

## Setup Instructions

### 1. Download and Organize Audio Files
After downloading, organize your audio files as follows:
- `/cv` for CommonVoice audio (subdirectories by language)
- `/mls` for Multilingual LibriSpeech audio (subdirectories by language)
- `/peoplespeech_audio` for People's Speech audio

### 2. Convert Parquet Files to NeMo Manifests

Create a script `parquet_to_manifest.py`:
```python
from datasets import load_dataset
import json
import os

def convert_to_manifest(dataset, split, output_file):
    with open(output_file, 'w') as f:
        for item in dataset[split]:
            # Ensure paths match your mounted directories
            source, lang = item['source'].split('_')
            if source == 'commonvoice':
                item['audio_filepath'] = os.path.join('/cv', lang, item['audio_filepath'])
            elif source == 'librispeech':
                item['audio_filepath'] = os.path.join('/mls', lang, item['audio_filepath'])
            elif source == 'peoplespeech':
                item['audio_filepath'] = os.path.join('/peoplespeech_audio', item['audio_filepath'])
            
            manifest_entry = {
                'audio_filepath': item['audio_filepath'],
                'text': item['text'],
                'duration': item['duration']
            }
            f.write(json.dumps(manifest_entry) + '\\n')

# Load the dataset from Hugging Face
dataset = load_dataset("WhissleAI/Meta_STT_EURO_Set1")

# Convert each split to manifest
for split in dataset.keys():
    output_file = f"{split}_manifest.json"
    convert_to_manifest(dataset, split, output_file)
    print(f"Created manifest for {split}: {output_file}")
```

Run the conversion:
```bash
python parquet_to_manifest.py
```

This will create manifest files (`train_manifest.json`, `valid_manifest.json`, etc.) in NeMo format.

### 3. Pull and Run NeMo Docker
```bash
# Pull the NeMo Docker image
docker pull nvcr.io/nvidia/nemo:24.05

# Run the container with GPU support and mounted volumes
docker run --gpus all -it --rm \\
    -v /external1:/external1 \\
    -v /external2:/external2 \\
    -v /external3:/external3 \\
    -v /cv:/cv \\
    -v /mls:/mls \\
    -v /peoplespeech_audio:/peoplespeech_audio \\
    --shm-size=8g \\
    -p 8888:8888 -p 6006:6006 \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    --device=/dev/snd \\
    nvcr.io/nvidia/nemo:24.05
```

### 4. Fine-tuning Instructions

#### A. Create a config file (e.g., `config.yaml`):
```yaml
model:
  name: "ConformerCTC"
  pretrained_model: "nvidia/stt_en_conformer_ctc_large" # or your preferred model
  
  train_ds:
    manifest_filepath: "train_manifest.json"  # Path to the manifest created in step 2
    batch_size: 32
    
  validation_ds:
    manifest_filepath: "valid_manifest.json"  # Path to the manifest created in step 2
    batch_size: 32
    
  optim:
    name: adamw
    lr: 0.001
    
  trainer:
    devices: 1
    accelerator: "gpu"
    max_epochs: 100
```

#### B. Start Fine-tuning:
```bash
# Inside the NeMo container
python -m torch.distributed.launch --nproc_per_node=1 \\
    examples/asr/speech_to_text_finetune.py \\
    --config-path=. \\
    --config-name=config.yaml
```

## Dataset Statistics

### Splits and Sample Counts
"""
    for split, count in meta_info.items():
        readme += f"- **{split}**: {count} samples\n"
    readme += "\n"

    readme += "## Example Samples\n"
    for split, samples in example_samples.items():
        readme += f"### {split}\n"
        for sample in samples:
            readme += "```json\n"
            readme += json.dumps(sample, indent=2, ensure_ascii=False)
            readme += "\n```\n"
        readme += "\n"

    readme += """
## Usage Notes

1. The metadata in this repository contains paths to audio files that must match your local setup.
2. When fine-tuning, ensure your manifest files use the correct paths for your mounted directories.
3. For optimal performance:
   - Use a GPU with at least 16GB VRAM
   - Adjust batch size based on your GPU memory
   - Consider gradient accumulation for larger effective batch sizes
   - Monitor training with TensorBoard (accessible via port 6006)

## Common Issues and Solutions

1. **Path Mismatches**: Ensure audio file paths in manifests match the mounted directories in Docker
2. **Memory Issues**: Reduce batch size or use gradient accumulation
3. **Docker Permissions**: Ensure proper permissions for mounted volumes and audio devices
"""
    return readme

def delete_all_files_in_repo(repo_id, token):
    api = HfApi()
    files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    for file in files:
        if file == ".gitattributes":
            continue
        print(f"Deleting {file} from repo...")
        api.delete_file(path_in_repo=file, repo_id=repo_id, repo_type="dataset", token=token)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/external1/datasets/hf/euro")
    parser.add_argument("--repo_id", type=str, default="WhissleAI/Meta_STT_EURO_Set1")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token for authentication")
    parser.add_argument("--splits", type=str, default="train,valid,test", help="Comma-separated list of splits to upload (e.g. 'train,valid')")
    parser.add_argument("--examples-per-split", type=int, default=2, help="Number of example samples to show per split in README")
    parser.add_argument("--delete-existing", action="store_true", help="Delete all existing files in the repo before upload")
    args = parser.parse_args()

    # Login to Hugging Face with the provided token
    login(args.hf_token)

    if args.delete_existing:
        print(f"Deleting all files in repo {args.repo_id} ...")
        delete_all_files_in_repo(args.repo_id, args.hf_token)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    dataset_dict = {}
    meta_info = {}
    example_samples = {}

    for split in splits:
        print(f"Reading {split} split...")
        samples = read_json_files(args.input_dir, split)
        if samples:
            dataset_dict[split] = Dataset.from_list(samples)
            meta_info[split] = len(samples)
            example_samples[split] = samples[:args.examples_per_split]
        else:
            print(f"No samples found for split '{split}'.")

    if not dataset_dict:
        print("No data found for the selected splits. Exiting.")
        return

    ds_dict = DatasetDict(dataset_dict)

    print("Preparing README...")
    readme = make_readme(meta_info, example_samples)

    print(f"Pushing to HuggingFace repo: {args.repo_id} ...")
    ds_dict.push_to_hub(
        repo_id=args.repo_id,
        token=args.hf_token,
        private=False
    )

    # Upload README.md
    print("Uploading README.md ...")
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    api = HfApi()
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=args.hf_token
    )
    os.remove("README.md")
    print("Done!")

if __name__ == "__main__":
    main()
