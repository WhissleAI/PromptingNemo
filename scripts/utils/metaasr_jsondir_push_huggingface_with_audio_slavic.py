#!/usr/bin/env python3
import os
import json
import argparse
from datasets import Dataset, DatasetDict
from huggingface_hub import login, HfApi, list_repo_files

def read_json_file(json_file):
    """Read a single JSON file and return a list of samples."""
    samples = []
    # Extract language code from filename (e.g., train_be.json -> be)
    lang = os.path.basename(json_file).split('_')[1].split('.')[0]
    
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                if all(k in sample for k in ("audio_filepath", "text", "duration")):
                    # Update audio_filepath to point to Common Voice dataset
                    sample["audio_filepath"] = f"/cv/cv-corpus-15.0-2023-09-08/{lang}/clips/{os.path.basename(sample['audio_filepath'])}"
                    samples.append(sample)
            except Exception as e:
                print(f"Error parsing line in {json_file}: {e}")
    return samples

def make_readme(meta_info, example_samples):
    readme = """# Meta Speech Recognition Slavic Languages Dataset (Common Voice)

This dataset contains metadata for Slavic language speech recognition samples from Common Voice.

## Dataset Sources and Credits

This dataset contains samples from Mozilla Common Voice:
- Source: https://commonvoice.mozilla.org/en/datasets
- License: CC0-1.0
- Citation: Please acknowledge Mozilla Common Voice if you use this data

## Languages Included

The dataset includes the following Slavic languages:
- Belarusian (be)
- Bulgarian (bg)
- Czech (cs)
- Georgian (ka)
- Macedonian (mk)
- Polish (pl)
- Russian (ru)
- Slovak (sk)
- Slovenian (sl)
- Serbian (sr)
- Ukrainian (uk)

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
## Downloading Audio Files

To use this dataset, you need to download the Common Voice dataset for each language. The audio files are not included in this repository.

### Download Instructions

1. Download Common Voice dataset version 15.0 (2023-09-08) for each language:
```bash
# For each language (replace {lang} with language code: be, bg, cs, ka, mk, pl, ru, sk, sl, sr, uk)
wget https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-15.0-2023-09-08/{lang}.tar.gz

# Extract the files
tar -xzf {lang}.tar.gz
```

2. Place the extracted files in the following structure:
```
/cv/cv-corpus-15.0-2023-09-08/
├── be/
├── bg/
├── cs/
├── ka/
├── mk/
├── pl/
├── ru/
├── sk/
├── sl/
├── sr/
└── uk/
```

## Training NeMo Conformer ASR for Slavic Languages

### 1. Pull and Run NeMo Docker
```bash
# Pull the NeMo Docker image
docker pull nvcr.io/nvidia/nemo:24.05

# Run the container with GPU support
docker run --gpus all -it --rm \\
    -v /external1:/external1 \\
    -v /external2:/external2 \\
    -v /external3:/external3 \\
    -v /cv:/cv \\
    --shm-size=8g \\
    -p 8888:8888 -p 6006:6006 \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    nvcr.io/nvidia/nemo:24.05
```

### 2. Create Training Script
Create a script `train_nemo_asr_slavic.py`:
```python
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.data.audio_to_text import TarredAudioToTextDataset
import pytorch_lightning as pl
from omegaconf import OmegaConf
import os

# Load the dataset from Hugging Face
from datasets import load_dataset
dataset = load_dataset("WhissleAI/Meta_STT_SLAVIC_CommonVoice")

# Create config
config = OmegaConf.create({
    'model': {
        'name': 'EncDecCTCModel',
        'train_ds': {
            'manifest_filepath': None,  # Will be set dynamically
            'batch_size': 32,
            'shuffle': True,
            'num_workers': 4,
            'pin_memory': True,
            'use_start_end_token': False,
        },
        'validation_ds': {
            'manifest_filepath': None,  # Will be set dynamically
            'batch_size': 32,
            'shuffle': False,
            'num_workers': 4,
            'pin_memory': True,
            'use_start_end_token': False,
        },
        'optim': {
            'name': 'adamw',
            'lr': 0.001,
            'weight_decay': 0.01,
        },
        'trainer': {
            'devices': 1,
            'accelerator': 'gpu',
            'max_epochs': 100,
            'precision': 16,
        }
    }
})

# Initialize model
model = EncDecCTCModel(cfg=config.model)

# Create trainer
trainer = pl.Trainer(**config.model.trainer)

# Train
trainer.fit(model)
```

### 3. Create Config File
Create a config file `config_slavic.yaml`:
```yaml
model:
  name: "EncDecCTCModel"
  train_ds:
    manifest_filepath: "train.json"
    batch_size: 32
    shuffle: true
    num_workers: 4
    pin_memory: true
    use_start_end_token: false
    
  validation_ds:
    manifest_filepath: "valid.json"
    batch_size: 32
    shuffle: false
    num_workers: 4
    pin_memory: true
    use_start_end_token: false
    
  optim:
    name: adamw
    lr: 0.001
    weight_decay: 0.01
    
  trainer:
    devices: 1
    accelerator: "gpu"
    max_epochs: 100
    precision: 16
```

### 4. Start Training
```bash
# Inside the NeMo container
python -m torch.distributed.launch --nproc_per_node=1 \\
    train_nemo_asr_slavic.py \\
    --config-path=. \\
    --config-name=config_slavic.yaml
```

## Usage Notes

1. The dataset includes only metadata. Audio files must be downloaded separately from Common Voice.
2. Audio files should be placed in the `/cv/cv-corpus-15.0-2023-09-08/{lang}/` directory structure.
3. For optimal performance:
   - Use a GPU with at least 16GB VRAM
   - Adjust batch size based on your GPU memory
   - Consider gradient accumulation for larger effective batch sizes
   - Monitor training with TensorBoard (accessible via port 6006)

## Common Issues and Solutions

1. **Memory Issues**: 
   - Reduce batch size if you encounter OOM errors
   - Use gradient accumulation for larger effective batch sizes
   - Enable mixed precision training (fp16)

2. **Training Speed**:
   - Increase num_workers based on your CPU cores
   - Use pin_memory=True for faster data transfer to GPU
   - Consider using tarred datasets for faster I/O

3. **Model Performance**:
   - Adjust learning rate based on your batch size
   - Use learning rate warmup for better convergence
   - Consider using a pretrained model as initialization
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
    parser.add_argument("--input_dir", type=str, default="/external1/datasets/hf/slavic")
    parser.add_argument("--repo_id", type=str, default="WhissleAI/Meta_STT_SLAVIC_CommonVoice")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token (or use `huggingface-cli login`)")
    parser.add_argument("--examples-per-split", type=int, default=2, help="Number of example samples to show per split in README")
    parser.add_argument("--delete-existing", action="store_true", help="Delete all existing files in the repo before upload")
    args = parser.parse_args()

    if args.token:
        login(args.token)

    if args.delete_existing:
        print(f"Deleting all files in repo {args.repo_id} ...")
        delete_all_files_in_repo(args.repo_id, args.token)

    # Define Slavic language codes
    slavic_languages = ["be", "bg", "cs", "ka", "mk", "pl", "ru", "sk", "sl", "sr", "uk"]
    splits = ["train", "valid", "test"]
    
    dataset_dict = {}
    meta_info = {}
    example_samples = {}

    for split in splits:
        all_samples = []
        for lang in slavic_languages:
            json_file = os.path.join(args.input_dir, split, f"{split}_{lang}.json")
            if not os.path.exists(json_file):
                print(f"File not found: {json_file}")
                continue
                
            print(f"Reading {split} split for language {lang}...")
            samples = read_json_file(json_file)
            if samples:
                all_samples.extend(samples)
            else:
                print(f"No samples found for split '{split}' in language '{lang}'.")

        if all_samples:
            # Create dataset without audio feature
            dataset = Dataset.from_list(all_samples)
            dataset_dict[split] = dataset
            meta_info[split] = len(all_samples)
            example_samples[split] = all_samples[:args.examples_per_split]
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
        token=args.token
    )
    os.remove("README.md")
    print("Done!")

if __name__ == "__main__":
    main()
