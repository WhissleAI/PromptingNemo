import os
import json
import argparse
import subprocess
from pydub import AudioSegment
from tqdm import tqdm

def convert_opus_to_wav(audio_path, wav_path):
    """Convert an Opus file to a 16k PCM WAV file using FFmpeg."""
    try:
        if audio_path.startswith("/mls_spanish"):
            print(f"Starting conversion: {audio_path} -> {wav_path}")
        # Add '-y' to overwrite existing files without prompt
        command = f"ffmpeg -y -i {audio_path} -ar 16000 -ac 1 {wav_path}"
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if audio_path.startswith("/mls_spanish"):
            print(result.stdout.decode('utf-8'))  # Print FFmpeg output for debugging
            print(f"Successfully converted: {audio_path} -> {wav_path}")
        return True
    except subprocess.CalledProcessError as e:
        if audio_path.startswith("/mls_spanish"):
            print(f"Error converting {audio_path} to WAV: {e.stderr.decode('utf-8')}")
        return False

def verify_audio_manifest(input_manifest_path, valid_manifest_path, invalid_manifest_path):
    with open(input_manifest_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    valid_samples = []
    invalid_samples = []

    # Initialize progress bar
    with tqdm(total=len(lines), desc="Processing files", unit="file") as pbar:
        for line in lines:
            sample = json.loads(line)
            audio_path = sample.get("audio_filepath")

            if audio_path.endswith('.opus'):
                audio_path = os.path.splitext(audio_path)[0] + '.wav'
                sample["audio_filepath"] = audio_path
                # if audio_path.startswith("/mls_spanish"):
                #     print(f"Processing file: {audio_path}")
                #     print(f"Starting conversion: {audio_path} -> {wav_path}")
                
                # if convert_opus_to_wav(audio_path, wav_path):
                #     sample["audio_filepath"] = wav_path
                #     audio_path = wav_path
                # else:
                #     if audio_path.startswith("/mls_spanish"):
                #         print(f"Skipping file due to conversion error: {audio_path}")
                #     invalid_samples.append(sample)
                #     pbar.update(1)
                #     continue

            if os.path.exists(audio_path):
                valid_samples.append(sample)
            else:
                invalid_samples.append(sample)

            # Update progress bar
            pbar.update(1)

    with open(valid_manifest_path, 'w', encoding='utf-8') as valid_file:
        for sample in valid_samples:
            valid_file.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with open(invalid_manifest_path, 'w', encoding='utf-8') as invalid_file:
        for sample in invalid_samples:
            invalid_file.write(json.dumps(sample, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify audio manifest and separate valid and invalid samples.")
    parser.add_argument("input_manifest_folder", type=str, help="Path to a folder containing the input manifest files.")
    parser.add_argument("output_folder", type=str, help="Path to the folder where output manifests will be saved.")

    args = parser.parse_args()

    input_manifest_folder = args.input_manifest_folder
    input_manifest_files = [f for f in os.listdir(input_manifest_folder) if f.endswith('.json')]
    
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    for file in input_manifest_files:
        
        fileidx = file.split(".")[0]
        
        input_manifest = os.path.join(input_manifest_folder, file)
        print(f"Processing {input_manifest}...")

        # Create output folder for each manifest
        output_folder = os.path.join(args.output_folder, os.path.splitext(file)[0])
        os.makedirs(output_folder, exist_ok=True)

        valid_manifest = os.path.join(output_folder, fileidx + "_valid.json")
        invalid_manifest = os.path.join(output_folder, fileidx + "_invalid.json")
        # Verify the audio manifest
        verify_audio_manifest(input_manifest, valid_manifest, invalid_manifest)
