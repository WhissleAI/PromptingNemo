import argparse
import json
import os
from pathlib import Path
import shutil
import re
import sys

# Function to fix or validate paths in the manifest file and split them into components
def fix_and_split_paths(input_manifest, output_manifest):
    with open(input_manifest, 'r') as infile, open(output_manifest, 'w') as outfile:
        for line in infile:
            
            sample = json.loads(line)
            audio_path = sample.get("audio_filepath")
               
            
            filename = audio_path.split('/')[-1]
            filename_parts = filename.split('_')
            folder1 = filename_parts[0]
            folder2 = filename_parts[1]
            
            new_filename = f"{folder1}/{folder2}/{filename}"
            new_audio_path = audio_path.replace(filename, new_filename)
            new_audio_path = new_audio_path.replace('audio1', 'audio')
            
            sample["audio_filepath"] = new_audio_path
            outfile.write(json.dumps(sample) + '\n')
    
    outfile.close()
            

# Main function to handle argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix and split paths in a manifest file.")
    parser.add_argument("input_manifest", type=str, help="Path to the input manifest file.")
    parser.add_argument("output_manifest", type=str, help="Path to the output manifest file.")
    args = parser.parse_args()

    fix_and_split_paths(args.input_manifest, args.output_manifest)