import os
import glob
from time import sleep
import json
import vertexai
#from google.cloud import storage, speech_v1p1beta1 as speech
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part

PROJECT_ID = "stream2action"  # Replace with your actual Google Cloud project ID

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the specified Google Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}.")


# Initialize Vertex AI with the project ID and location
vertexai.init(project=PROJECT_ID, location="us-central1")

# Load the Gemini model
model = GenerativeModel("gemini-1.5-flash-002")

# Define the transcription prompt
prompt = """
Can you transcribe speech in this video, in the format of timecode, speaker, caption.
Use 'Speaker_A', 'Speaker_B', etc. to identify speakers.
"""


data_folder = "/external2/datasets/youtube/movie_data"
audio_uri_files = glob.glob(data_folder + "/*/uploaded_files_uris.txt")

print(audio_uri_files)

for audio_uri_file in audio_uri_files:
    
    audio_uris = open(audio_uri_file, "r").readlines()
    
    for audio_uri in audio_uris:
            
            audio_uri = audio_uri.strip()
            
            if audio_uri.endswith(".wav"):
            
                try:
                    print("Transcribing audio file: ", audio_uri)
                    audio_file_uri = audio_uri
                    audio_file = Part.from_uri(audio_file_uri, mime_type="audio/mpeg")
                    
                    # Combine the prompt and audio file for input
                    contents = [audio_file, prompt]
                    
                    # Configure and generate the transcription response with timecodes
                    response = model.generate_content(contents, generation_config=GenerationConfig(audio_timestamp=True))
                    
                    # Print the transcription response
                    
                    out_file = os.path.basename(audio_file_uri).split(".")[0] + ".txt"
                    out_file = data_folder + "/" + out_file
                    with open(out_file, "w") as f:
                        response = json.dumps(response.to_dict(), indent=2)
                        f.write(response)
                    print("Transcription saved to: ", out_file)
                    
                    
                    print(response)
                    
                    print("Sleeping for 10 seconds...")
                    sleep(10)
                
                except:
                    print("Error transcribing audio file: ", audio_uri)
                    continue