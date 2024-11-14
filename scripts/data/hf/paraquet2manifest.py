import glob
import pandas as pd
import json
import os

def get_audio_path(audio_entry):
    """Extract audio path from audio entry which might be a dict or string"""
    if isinstance(audio_entry, dict):
        # You might need to adjust this depending on the actual structure of your audio dictionary
        # Common keys might be 'path', 'filename', or 'location'
        # Print a sample entry to check the structure if needed
        #print("Sample audio entry:", audio_entry)
        return audio_entry.get('path', '')  # adjust the key as needed
    return str(audio_entry)

# Define your data folder paths
data_folder = "/external1/datasets/peoples_speech/data"
audio_folder = "/external1/datasets/peoples_speech/audio"
manifest_folder = "/external1/datasets/peoples_speech/manifests"

# Ensure the manifest folder exists
os.makedirs(manifest_folder, exist_ok=True)

# Find all .parquet files
parquet_files = glob.glob(f"{data_folder}/**/*.parquet", recursive=True)

# Process each parquet file
for pfile in parquet_files[:2]:
    print(f"Processing {pfile}")
    
    # Load the parquet data
    data = pd.read_parquet(pfile)
    
    # # Print sample row to debug
    # if not data.empty:
    #     print("\nSample row structure:")
    #     print(data.iloc[0].to_dict())
    
    # Define the manifest file path
    manifest_file = os.path.join(manifest_folder, os.path.basename(pfile).replace('.parquet', '.json'))
    
    # Open the manifest file for writing
    with open(manifest_file, 'w') as f:
        for _, row in data.iterrows():

                # Get the audio path
                audio_path = get_audio_path(row["audio"])
                print(audio_path)
                # Build the NeMo manifest entry
                entry = {
                    "audio_filepath": os.path.join(audio_folder, audio_path),
                    "duration": row["duration_ms"] / 1000,  # Convert milliseconds to seconds
                    "text": row["text"]
                }
                
                # Write entry as JSON line
                f.write(json.dumps(entry) + '\n')
            
    
    print(f"Manifest saved to {manifest_file}")