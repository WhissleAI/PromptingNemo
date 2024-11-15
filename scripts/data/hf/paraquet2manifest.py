import glob
import pandas as pd
import json
import os
import numpy as np
import soundfile as sf

def save_audio(audio_data, save_path, sample_rate=16000):
    """Save audio data (in byte format) to a .wav file at the specified path."""
    # Print the type, keys, and length of audio data for debugging
    #print(f"Audio data type: {type(audio_data)}, keys: {audio_data.keys()}")
    audio_bytes = audio_data.get("bytes", b"")
    #print(f"Audio byte length: {len(audio_bytes)}")

    # Try converting audio bytes to an int16 numpy array directly
    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    except ValueError:
        # If int16 fails, use uint8 and then convert to int16
        audio_np = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.int16)

    # Reshape if it's a single channel
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(-1, 1)

    # Save audio data as .wav file with 16kHz PCM format
    sf.write(save_path, audio_np, sample_rate, format='WAV', subtype='PCM_16')

# Define your data folder paths
data_folder = "/external1/datasets/peoples_speech/data"
audio_folder = "/external1/datasets/peoples_speech/audio"
manifest_folder = "/external1/datasets/peoples_speech/manifests"

# Ensure folders exist
os.makedirs(audio_folder, exist_ok=True)
os.makedirs(manifest_folder, exist_ok=True)

# Find all .parquet files
parquet_files = glob.glob(f"{data_folder}/**/*.parquet", recursive=True)

total_files = len(parquet_files)
print(f"Found {total_files} parquet files")
# Process each parquet file
for pfile in parquet_files:
    print(f"Processing {pfile}")
    
    # Load the parquet data
    data = pd.read_parquet(pfile)
    
    # Define the manifest file path
    manifest_file = os.path.join(manifest_folder, os.path.basename(pfile).replace('.parquet', '.json'))
    
    # Open the manifest file for writing
    with open(manifest_file, 'w') as f:
        for _, row in data.iterrows():
            # Define the path to save the audio file as .wav
            try:
                audio_filename = f"{row['id']}.wav"
                local_audio_path = os.path.join(audio_folder, audio_filename)
                
                # Save audio data to the specified path as a .wav file
                save_audio(row["audio"], local_audio_path)
                    
                # Build the NeMo manifest entry
                entry = {
                    "audio_filepath": local_audio_path,
                    "duration": row["duration_ms"] / 1000,  # Convert milliseconds to seconds
                    "text": row["text"]
                }
                
                # Write entry as JSON line
                f.write(json.dumps(entry) + '\n')
            except Exception as e:
                print(f"Error processing row: {row['id']}, {e}")
                continue
            
    os.system("rm " + pfile)
    
    print(f"Manifest saved to {manifest_file}")
