from google.cloud import storage, speech_v1p1beta1 as speech
import time
import os

# Set the path to your Google Cloud service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/external2/workspace/google_service_account_files/stream2action-773bd306c0f0.json"

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the specified Google Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}.")

def transcribe_long_audio_gcs(gcs_uri):
    """Asynchronously transcribes the audio file at the given GCS URI."""
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True
    )

    # Initiate the asynchronous transcription request
    operation = client.long_running_recognize(config=config, audio=audio)
    print("Waiting for operation to complete...")

    # Wait for the transcription operation to finish
    response = operation.result(timeout=3600)

    # Print transcription results
    for result in response.results:
        print("Transcript:", result.alternatives[0].transcript)
        print("Confidence:", result.alternatives[0].confidence)

# Define your local file path, bucket name, and file names
local_file_path = "/external2/sadia/soccer_data_transcription/audio/LIVE_Manchester_United_vs_PAOK_UEFA_Europa_League_2024_Full_Match_PES_21_2024-11-07_18_28.wav"
bucket_name = "stream2action-audio"
destination_blob_name = "soccer_data/LIVE_Manchester_United_vs_PAOK.wav"

# Upload the file to Google Cloud Storage
upload_to_gcs(bucket_name, local_file_path, destination_blob_name)

# Set the GCS URI and start the transcription process
gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
transcribe_long_audio_gcs(gcs_uri)
