import json
import os
from pydub import AudioSegment
from google.cloud import storage
import wave
import contextlib
import tempfile

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/external1/ksingla/workspace/google_service_account_files/stream2action-773bd306c0f0.json"

def download_from_gcs(gcs_path, local_path):
    """
    Download audio file from Google Cloud Storage
    
    Args:
        gcs_path: Full GCS path (e.g., 'gs://bucket-name/path/to/file.wav')
        local_path: Path where the file should be saved locally
    """
    try:
        # Remove 'gs://' prefix and split into bucket and blob path
        path_without_prefix = gcs_path.replace('gs://', '')
        bucket_name, blob_path = path_without_prefix.split('/', 1)
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Download the file
        blob.download_to_filename(local_path)
        print(f"Successfully downloaded: {blob_path}")
        
    except Exception as e:
        raise Exception(f"Error downloading from GCS: {str(e)}")

def get_audio_duration(audio_path):
    """Get duration of a wav file"""
    with contextlib.closing(wave.open(audio_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def segment_and_upload_audio(json_file_path, gcs_bucket_name, gcs_output_folder):
    """
    Download audio file from GCS, segment it based on transcription timestamps and upload back to GCS.
    
    Args:
        json_file_path: Path to the JSON file containing transcription
        gcs_bucket_name: Name of the GCS bucket for output
        gcs_output_folder: Folder in the bucket to upload segments
    """
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)
    
    # Read and parse JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Get audio URI from metadata
    gcs_audio_path = data['metadata']['audio_uri']
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_audio_dir:
        with tempfile.TemporaryDirectory() as temp_segments_dir:
            # Set up paths
            audio_filename = os.path.basename(gcs_audio_path)
            local_audio_path = os.path.join(temp_audio_dir, audio_filename)
            
            # Download audio file from GCS
            download_from_gcs(gcs_audio_path, local_audio_path)
            
            # Load audio file
            audio = AudioSegment.from_wav(local_audio_path)
            
            # Process each transcription segment
            for idx, segment in enumerate(data['transcription']):
                start_time = int(segment['start_time'] * 1000)  # Convert to milliseconds
                end_time = int(segment['end_time'] * 1000)
                
                # Extract segment
                audio_segment = audio[start_time:end_time]
                
                # Generate segment filename
                base_name = os.path.splitext(audio_filename)[0]
                segment_filename = f"{base_name}_segment_{idx:04d}_{start_time}_{end_time}.wav"
                segment_path = os.path.join(temp_segments_dir, segment_filename)
                
                # Export segment
                audio_segment.export(segment_path, format="wav")
                
                # Upload to GCS
                blob_name = f"{gcs_output_folder}/{segment_filename}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(segment_path)
                
                print(f"Uploaded segment {idx + 1}/{len(data['transcription'])}: {blob_name}")

def process_directory(input_dir, gcs_bucket_name, gcs_output_folder):
    """Process all JSON files in a directory"""
    for filename in os.listdir(input_dir):
        if filename.endswith('_cleaned.json'):
            json_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            try:
                segment_and_upload_audio(json_path, gcs_bucket_name, gcs_output_folder)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Configuration
    input_dir = "/external1/datasets/youtube/toy"
    gcs_bucket_name = "stream2action-audio"
    gcs_output_folder = "segmented_audio_toy"
    
    process_directory(input_dir, gcs_bucket_name, gcs_output_folder)