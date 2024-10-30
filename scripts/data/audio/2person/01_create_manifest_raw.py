import os
import json

# Define the directories containing the transcript and wav files
transcript_folder = '/external2/datasets/psychotherapy/Data/transcripts/'  # Update this to your actual transcript folder path
wav_folder = '/external2/datasets/psychotherapy/Data/audio_wav/'  # Update this to your actual wav folder path
manifest_path = '../../../datasets/manifests/manifest.json'  # Output path for the manifest

# List all transcript and wav files
transcript_files = [f for f in os.listdir(transcript_folder) if f.endswith('.txt')]
wav_files = [f for f in os.listdir(wav_folder) if f.endswith('.wav')]

# Create a dictionary mapping wav file names (without extensions) to their full paths
wav_file_paths = {os.path.splitext(f)[0]: os.path.join(wav_folder, f) for f in wav_files}

# Open the manifest file for writing
with open(manifest_path, 'w', encoding='utf-8') as outfile:
    # Iterate over the transcript files and match with corresponding wav files
    for transcript_file in transcript_files:
        transcript_name = os.path.splitext(transcript_file)[0]
        transcript_path = os.path.join(transcript_folder, transcript_file)
        
        # Try reading the text content with a robust approach to handle encoding issues
        text_content = ""
        try:
            with open(transcript_path, 'r', encoding='utf-8') as file:
                text_content = file.read().strip()
        except UnicodeDecodeError:
            try:
                with open(transcript_path, 'r', encoding='ISO-8859-1') as file:
                    text_content = file.read().strip()
            except UnicodeDecodeError:
                with open(transcript_path, 'r', encoding='utf-16') as file:
                    text_content = file.read().strip()

        # Skip empty or invalid text content
        if not text_content:
            print(f"Warning: Skipping {transcript_file} due to empty or invalid content.")
            continue

        # Check if there's a matching wav file
        if transcript_name in wav_file_paths:
            audio_filepath = wav_file_paths[transcript_name]
            # Create a JSON object for each entry
            entry = {
                "audio_filepath": audio_filepath,
                "text": text_content
            }
            # Write the JSON object as a separate line in the manifest file
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')
        else:
            print(f"Warning: No matching audio file for {transcript_file}")

print(f"Manifest successfully created at: {manifest_path}")
