# import os
# import glob
# from time import sleep
# import json
# import vertexai
# #from google.cloud import storage, speech_v1p1beta1 as speech
# from vertexai.generative_models import GenerativeModel, GenerationConfig, Part

# PROJECT_ID = "stream2action"  # Replace with your actual Google Cloud project ID

# # Initialize Vertex AI with the project ID and location
# vertexai.init(project=PROJECT_ID, location="us-central1")

# # Load the Gemini model
# model = GenerativeModel("gemini-1.5-flash-002")

# # Define the transcription prompt

# prompt = """
# Can you transcribe this interview, in the format of speaker, caption with tags, intent label, emotion label: Happy, Angry, Sad, Neutral, Surprise, Disgust, Trust, Anticipation.
# Use speaker RoleA, speaker RoleB, etc., to identify speakers.

# Caption text should be tagged with meaniniful entity tags, for example:

#     ENTITY_ACTION Remind END me to ENTITY_ACTION take END my ENTITY_MEDICATION medication END at ENTITY_TIME 9 AM END.
    
# Output a python list where each turn is a dict with keys: speaker, caption with tags, intent label, emotion label.
# """


# data_folder = "/external2/datasets/youtube/movie_data"
# audio_uri_files = glob.glob(data_folder + "/*/uploaded_files_uris.txt")

# print(audio_uri_files)

# for audio_uri_file in audio_uri_files:
    
#     audio_uris = open(audio_uri_file, "r").readlines()
    
#     for audio_uri in audio_uris:
            
#             audio_uri = audio_uri.strip()
            
#             if audio_uri.endswith(".wav"):
            
#                 try:
#                     print("Transcribing audio file: ", audio_uri)
#                     audio_file_uri = audio_uri
#                     audio_file = Part.from_uri(audio_file_uri, mime_type="audio/mpeg")
                    
#                     # Combine the prompt and audio file for input
#                     contents = [audio_file, prompt]
                    
#                     # Configure and generate the transcription response with timecodes
#                     response = model.generate_content(contents, generation_config=GenerationConfig(audio_timestamp=True))
                    
#                     # Print the transcription response
                    
#                     out_file = os.path.basename(audio_file_uri).split(".")[0] + ".txt"
#                     out_file = data_folder + "/" + out_file
#                     with open(out_file, "w") as f:
#                         response = json.dumps(response.to_dict(), indent=2)
#                         f.write(response)
#                     print("Transcription saved to: ", out_file)
                    
                    
#                     print(response)
                    
#                     print("Sleeping for 10 seconds...")
#                     sleep(10)
                
#                 except:
#                     print("Error transcribing audio file: ", audio_uri)
#                     continue

import os
import json
from time import sleep
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part

from typing import Dict, List, Any

def clean_transcription_output(response_dict: Dict[str, Any], audio_uri: str) -> Dict[str, Any]:
    """
    Clean and extract the transcription data from the Gemini model response.
    
    Args:
        response_dict: Raw response dictionary from the Gemini model
        audio_uri: URI of the audio file being transcribed
        
    Returns:
        Dictionary containing metadata and cleaned transcription entries
    """
    try:
        # Extract the text content from the response
        text_content = response_dict['candidates'][0]['content']['parts'][0]['text']
        
        # Remove markdown code block indicators if present
        text_content = text_content.replace('```python', '').replace('```', '').strip()
        
        # Parse the string as JSON
        transcription_data = json.loads(text_content)
        
        # Clean and validate each entry
        cleaned_data = []
        for entry in transcription_data:
            cleaned_entry = {
                "start_time": float(entry.get("start_time", 0)),
                "end_time": float(entry.get("end_time", 0)),
                "speaker": entry.get("speaker", "unknown").strip(),
                "caption": entry.get("caption", "").strip(),
                "intent_label": entry.get("intent_label", "").strip(),
                "emotion_label": entry.get("emotion_label", "").strip()
            }
            cleaned_data.append(cleaned_entry)
            
        # Create final output structure with metadata
        output = {
            "metadata": {
                "audio_uri": audio_uri,
                "transcription_model": "gemini-1.5-flash-002",
                "timestamp_format": "seconds"
            },
            "transcription": cleaned_data
        }
        
        return output
        
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error processing transcription: {e}")
        return {
            "metadata": {
                "audio_uri": audio_uri,
                "error": str(e)
            },
            "transcription": []
        }



def list_mp4_files(bucket_name, prefix):
    """List all MP4 files in the specified bucket and prefix."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs if blob.name.endswith('.wav')]

def process_audio_files(bucket_name: str, output_folder: str):
    """Process audio files from the bucket using Gemini model."""
    # Initialize Vertex AI
    vertexai.init(project="stream2action", location="us-central1")
    
    # Load the Gemini model
    model = GenerativeModel("gemini-1.5-flash-002")
    
    # Define the transcription prompt
    """Process audio files from the bucket using Gemini model."""
    vertexai.init(project="stream2action", location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-002")
    
    prompt = """
    Please transcribe this audio with extremely precise word-level timing and accurate speaker changes. Pay special attention to:

    1. Speaker Identification:
    - Label each unique speaker as Speaker A, Speaker B, etc.
    - Note speaker changes even during brief interjections
    - Mark overlapping speech if present
    - Be consistent with speaker labels throughout the transcript

    2. Timing Requirements:
    - Provide precise start_time and end_time for each utterance in seconds
    - Break long utterances into smaller segments at natural pauses (4-8 seconds each)
    - Ensure timestamps align with word boundaries
    - Account for pauses between speakers (>0.5 seconds)

    3. Content Formatting:
    Caption text should be tagged with detailed entity tags, for example:
    ENTITY_ACTION open END the ENTITY_OBJECT door END at ENTITY_TIME three o'clock END
    
    Common entity types include:
    - ENTITY_ACTION: verbs and activities
    - ENTITY_OBJECT: physical objects
    - ENTITY_PERSON: names and roles
    - ENTITY_LOCATION: places
    - ENTITY_TIME: temporal expressions
    - ENTITY_EMOTION: emotional expressions

    4. Additional Labels:
    - intent_label: Classify the communicative intent (e.g., Question, Statement, Request, Response, Agreement, Disagreement)
    - emotion_label: [Happy, Angry, Sad, Neutral, Surprise, Disgust, Trust, Anticipation]

    Output a Python list where each segment is a dictionary with these keys:
    {
        "start_time": float,
        "end_time": float,
        "speaker": str,
        "caption": str,  # with entity tags
        "intent_label": str,
        "emotion_label": str
    }
    """
    
    # Get list of audio files
    audio_files = list_mp4_files(bucket_name, "youtube-videos/meeting_recordings/")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each file
    for file_path in audio_files:
        try:
            audio_uri = f"gs://{bucket_name}/{file_path}"
            print(f"Transcribing audio file: {audio_uri}")
            
            audio_file = Part.from_uri(audio_uri, mime_type="audio/mpeg")
            contents = [audio_file, prompt]
            
            response = model.generate_content(
                contents,
                generation_config=GenerationConfig(audio_timestamp=True)
            )
            
            # Clean the response data and include audio URI
            cleaned_data = clean_transcription_output(response.to_dict(), audio_uri)
            
            # Save cleaned transcription
            out_file = os.path.join(
                output_folder,
                os.path.basename(file_path).replace('.wav', '_cleaned.json')
            )
            
            with open(out_file, "w", encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            
            print(f"Cleaned transcription saved to: {out_file}")
            
            # Rate limiting
            print("Sleeping for 10 seconds...")
            sleep(10)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

if __name__ == "__main__":
    BUCKET_NAME = "stream2action-audio"
    OUTPUT_FOLDER = "/external1/datasets/youtube/"
    
    process_audio_files(BUCKET_NAME, OUTPUT_FOLDER)
    
    
    