#!/usr/bin/env python3
import os
import json
import argparse
from datasets import Dataset, DatasetDict, Audio
from huggingface_hub import login, HfApi, list_repo_files

def read_json_file(json_file):
    """Read a single JSON file and return a list of samples."""
    samples = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                if all(k in sample for k in ("audio_filepath", "text", "duration")):
                    samples.append(sample)
            except Exception as e:
                print(f"Error parsing line in {json_file}: {e}")
    return samples

def make_readme(meta_info, example_samples):
    readme = """# Meta Speech Recognition Mandarin Dataset (AISHELL3)

This dataset contains both metadata and audio files for Mandarin speech recognition samples from the AISHELL3 corpus.

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
## Dataset Information

- **Language**: Mandarin Chinese
- **Source**: AISHELL3
- **Format**: Each sample contains:
  - `audio_filepath`: Path to the audio file
  - `text`: Transcription text with speaker metadata
  - `duration`: Duration of the audio in seconds

## Speaker Metadata

The text field includes speaker metadata in the format:
- AGE_*: Speaker age range
- GENDER_*: Speaker gender
- DIALECT_*: Speaker dialect

## Usage Notes

1. The dataset includes both metadata and audio files.
2. Audio files are stored in the dataset repository.
3. The dataset is suitable for Mandarin speech recognition tasks.
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
    parser.add_argument("--input_dir", type=str, default="/external1/datasets/hf/mandarin")
    parser.add_argument("--repo_id", type=str, default="WhissleAI/Meta_STT_MN_AIShell3")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token (or use `huggingface-cli login`)")
    parser.add_argument("--examples-per-split", type=int, default=2, help="Number of example samples to show per split in README")
    parser.add_argument("--delete-existing", action="store_true", help="Delete all existing files in the repo before upload")
    args = parser.parse_args()

    if args.token:
        login(args.token)

    if args.delete_existing:
        print(f"Deleting all files in repo {args.repo_id} ...")
        delete_all_files_in_repo(args.repo_id, args.token)

    splits = ["train", "valid", "test"]
    dataset_dict = {}
    meta_info = {}
    example_samples = {}

    for split in splits:
        json_file = os.path.join(args.input_dir, f"{split}/{split}_aishell3.json")
        if not os.path.exists(json_file):
            print(f"File not found: {json_file}")
            continue
            
        print(f"Reading {split} split...")
        samples = read_json_file(json_file)
        if samples:
            # Create dataset with audio feature
            dataset = Dataset.from_list(samples)
            dataset = dataset.cast_column("audio_filepath", Audio())
            dataset_dict[split] = dataset
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
