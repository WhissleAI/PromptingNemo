import json
import os
from pydub import AudioSegment
from google.cloud import storage
import wave
import contextlib

def get_audio_duration(audio_path):
    """Get duration of a wav file"""
    with contextlib.closing(wave.open(audio_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def segment_and_upload_audio(json_file_path, local_audio_dir, gcs_bucket_name, gcs_output_folder):
    """
    Segment audio file based on transcription timestamps and upload to GCS.
    
    Args:
        json_file_path: Path to the JSON file containing transcription
        local_audio_dir: Directory containing the audio files
        gcs_bucket_name: Name of the GCS bucket
        gcs_output_folder: Folder in the bucket to upload segments
    """
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)
    
    # Read and parse JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Get audio file path from metadata
    audio_uri = data['metadata']['audio_uri']
    audio_filename = os.path.basename(audio_uri)
    local_audio_path = os.path.join(local_audio_dir, audio_filename)
    
    # Load audio file
    audio = AudioSegment.from_wav(local_audio_path)
    
    # Create temporary directory for segments
    temp_dir = "temp_segments"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each transcription segment
    for idx, segment in enumerate(data['transcription']):
        start_time = int(segment['start_time'] * 1000)  # Convert to milliseconds
        end_time = int(segment['end_time'] * 1000)
        
        # Extract segment
        audio_segment = audio[start_time:end_time]
        
        # Generate segment filename
        base_name = os.path.splitext(audio_filename)[0]
        segment_filename = f"{base_name}_segment_{idx:04d}_{start_time}_{end_time}.wav"
        segment_path = os.path.join(temp_dir, segment_filename)
        
        # Export segment
        audio_segment.export(segment_path, format="wav")
        
        # Upload to GCS
        blob_name = f"{gcs_output_folder}/{segment_filename}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(segment_path)
        
        print(f"Uploaded segment {idx + 1}/{len(data['transcription'])}: {blob_name}")
        
        # Clean up local segment file
        os.remove(segment_path)
    
    # Clean up temporary directory
    os.rmdir(temp_dir)

def process_directory(input_dir, local_audio_dir, gcs_bucket_name, gcs_output_folder):
    """Process all JSON files in a directory"""
    for filename in os.listdir(input_dir):
        if filename.endswith('_cleaned.json'):
            json_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            try:
                segment_and_upload_audio(json_path, local_audio_dir, gcs_bucket_name, gcs_output_folder)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Configuration
    input_dir = "/external1/datasets/youtube"
    local_audio_dir = "/path/to/local/audio/files"  # Update this
    gcs_bucket_name = "stream2action-audio"
    gcs_output_folder = "segmented_audio"
    
    process_directory(input_dir, local_audio_dir, gcs_bucket_name, gcs_output_folder)