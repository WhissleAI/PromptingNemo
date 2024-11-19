import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob

# Initialize Vertex AI with the specified project and location
vertexai.init(project="youtube-audio-conversations", location="us-central1")

# Create an instance of the generative model
model = GenerativeModel("gemini-1.5-flash-002")

# Define your transcription prompt
prompt = """
Can you transcribe this interview, in the format of speaker, caption with tags, intent label, emotion label: Happy, Angry, Sad, Neutral, Surprise, Disgust, Trust, Anticipation.
Use speaker RoleA, speaker RoleB, etc., to identify speakers.

Caption text should be tagged with meaniniful entity tags, for example:

    ENTITY_ACTION Remind END me to ENTITY_ACTION take END my ENTITY_MEDICATION medication END at ENTITY_TIME 9 AM END.
    
Output a python list where each turn is a dict with keys: speaker, caption with tags, intent label, emotion label.
"""

# Specify the local folder containing audio files
local_audio_folder_path = "/projects/whissle/datasets/youtube/downloaded_audios2/"  # Replace with your folder path
bucket_name = "youtube-audio-conversations1"  # Replace with your Google Cloud Storage bucket name
output_folder_path = "/projects/whissle/datasets/youtube/gemini_outpu2/"  # Replace with your folder path

# Ensure the output directory exists
os.makedirs(output_folder_path, exist_ok=True)

# Function to upload a file to GCS and return the GCS URI
def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    print(f"File uploaded to {gcs_uri}")
    return gcs_uri

# Function to process a single audio file
def process_audio_file(file_name):
    local_audio_file_path = file_name
    print(f"Processing {file_name}...")
    # Upload the local file to GCS and get the URI
    audio_file_uri = upload_to_gcs(local_audio_file_path, bucket_name, os.path.basename(file_name))

    # Create a Part using the uploaded GCS URI
    audio_file_part = Part.from_uri(audio_file_uri, mime_type="audio/mpeg")

    # Combine the prompt and audio for the request
    contents = [audio_file_part, prompt]

    # Generate content from the model
    response = model.generate_content(contents)
    response_text = response.text
    response_text = response_text.replace("```python\n", "").replace("\n```", "")
    #print(f"Response for {file_name}:\n{response_text}")

    # Save the response to a txt file in the output folder
    file_name = file_name.replace(".mp3", "")
    output_file_name = os.path.join(output_folder_path, f"{os.path.basename(file_name)}.txt")
    with open(output_file_name, "w") as output_file:
        output_file.write(response_text)
    print(f"Response saved to {output_file_name}")

# Function to process all files concurrently
def process_audio_files_concurrently(folder_path):
    # List all files in the specified folder
    audio_files = glob.glob(os.path.join(folder_path, "*/*.mp3"))

    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers based on your needs
        # Submit tasks for each file
        futures = {executor.submit(process_audio_file, file_name): file_name for file_name in audio_files}

        # Process results as they complete
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                future.result()  # This will raise exceptions if any occurred during execution
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Run the process on all audio files in the folder concurrently
process_audio_files_concurrently(local_audio_folder_path)