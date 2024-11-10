import os
import vertexai
from google.cloud import storage, speech_v1p1beta1 as speech
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part

# Set up the environment for Google Cloud credentials and project ID
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/external2/workspace/google_service_account_files/stream2action-773bd306c0f0.json"
PROJECT_ID = "stream2action"  # Replace with your actual Google Cloud project ID

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the specified Google Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}.")

# Set the path to your Google Cloud service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/external2/workspace/google_service_account_files/stream2action-773bd306c0f0.json"

# Initialize Vertex AI with the project ID and location
vertexai.init(project=PROJECT_ID, location="us-central1")

# Load the Gemini model
model = GenerativeModel("gemini-1.5-flash-002")

# Define the transcription prompt
prompt = """
Can you transcribe this soccer match audio, in the format of timecode, speaker, caption.
Use 'Commentator A', 'Commentator B', etc. to identify speakers.
"""
  
# # Define your local file path, bucket name, and file names
# local_file_path = "/external2/sadia/soccer_data_transcription/audio/LIVE_Manchester_United_vs_PAOK_UEFA_Europa_League_2024_Full_Match_PES_21_2024-11-07_18_28.wav"
# bucket_name = "stream2action-audio"
# destination_blob_name = "soccer_data/LIVE_Manchester_United_vs_PAOK.wav"

# Upload the file to Google Cloud Storage
#upload_to_gcs(bucket_name, local_file_path, destination_blob_name)

# # Specify the GCS URI of your audio file
# audio_file_uri = f"gs://{bucket_name}/{destination_blob_name}"

audio_file_uri = "gs://stream2action-audio/soccer_data/FULL_MATCH_Manchester_City_v_Manchester_United_2024.wav"
audio_file = Part.from_uri(audio_file_uri, mime_type="audio/wav")

# # Combine the prompt and audio file for input
contents = [audio_file, prompt]

# Configure and generate the transcription response with timecodes
response = model.generate_content(contents, generation_config=GenerationConfig(audio_timestamp=True))

# Print the transcription response
print(response)
