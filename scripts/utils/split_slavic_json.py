#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import random

# Language code mapping
LANGUAGE_CODES = {
    'belarussian': 'be',
    'rem_bela': 'be',  # Also Belarusian
    'bulgarian': 'bg',
    'czech': 'cs',
    'georgian': 'ka',
    'macedonian': 'mk',
    'polish': 'pl',
    'russiun': 'ru',
    'serbian': 'sr',
    'slovak': 'sk',
    'slovenian': 'sl',
    'ukrainian': 'uk'
}

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

def replace_audio_path(sample):
    """Replace /external4/datasets/cv with /cv in audio_filepath."""
    if "audio_filepath" in sample:
        sample["audio_filepath"] = sample["audio_filepath"].replace("/external4/datasets/cv", "/cv")
    return sample

def write_json_file(samples, output_file):
    """Write samples to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            # Replace the audio path before writing
            sample = replace_audio_path(sample)
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def split_dataset(samples, train_ratio=0.9, valid_ratio=0.05):
    """Split dataset into train, validation, and test sets."""
    # Shuffle samples
    random.shuffle(samples)
    
    # Calculate split indices
    total_samples = len(samples)
    train_end = int(total_samples * train_ratio)
    valid_end = train_end + int(total_samples * valid_ratio)
    
    # Split the data
    train_samples = samples[:train_end]
    valid_samples = samples[train_end:valid_end]
    test_samples = samples[valid_end:]
    
    return train_samples, valid_samples, test_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save split JSON files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each JSON file in the input directory
    for json_file in os.listdir(args.input_dir):
        if not json_file.endswith('.jsonl'):
            continue

        # Extract language code from filename (e.g., 'belarussian.jsonl' -> 'belarussian')
        base_name = os.path.splitext(json_file)[0]
        
        # Get the correct language code
        if base_name not in LANGUAGE_CODES:
            print(f"Warning: No language code mapping found for {base_name}, skipping...")
            continue
            
        language_code = LANGUAGE_CODES[base_name]
        
        print(f"Processing {json_file}...")
        
        # Read samples
        input_path = os.path.join(args.input_dir, json_file)
        samples = read_json_file(input_path)
        
        if not samples:
            print(f"No valid samples found in {json_file}")
            continue
        
        # Split dataset
        train_samples, valid_samples, test_samples = split_dataset(samples)
        
        # Write split files
        write_json_file(train_samples, os.path.join(args.output_dir, f"train_{language_code}.json"))
        write_json_file(valid_samples, os.path.join(args.output_dir, f"valid_{language_code}.json"))
        write_json_file(test_samples, os.path.join(args.output_dir, f"test_{language_code}.json"))
        
        print(f"Split {json_file} into:")
        print(f"  - train_{language_code}.json: {len(train_samples)} samples")
        print(f"  - valid_{language_code}.json: {len(valid_samples)} samples")
        print(f"  - test_{language_code}.json: {len(test_samples)} samples")

if __name__ == "__main__":
    main() 