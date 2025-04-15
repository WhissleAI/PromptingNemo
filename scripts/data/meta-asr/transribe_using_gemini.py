import os
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai
import base64

load_dotenv()

def setup_gemini():
    """Initialize Gemini API with your credentials."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")

    genai.configure(api_key=api_key)

def transcribe_audio(audio_path, mime_type="audio/wav"):
    """
    Transcribe audio using Gemini, handling noise and empty audio.
    Returns an empty string if no speech is detected.
    """
    # Use Gemini 2.0 Flash model which is the latest and best for this task
    model_name = "models/gemini-2.0-flash"
    
    model = genai.GenerativeModel(model_name)

    try:
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    except FileNotFoundError:
        return "Error: File not found"
    except Exception as e:
        return f"Error reading audio file: {e}"

    parts = [
        {
            "text": (
                "Transcribe the audio, returning ONLY the spoken English words. "
                "If there is NO SPEECH, return an empty string. "
                "Do NOT include any other text, such as descriptions of sounds."
            )
         },
        {
            "inline_data": {
                "mime_type": mime_type,
                "data": audio_base64
            }
        }
    ]

    try:
        response = model.generate_content(
            parts,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            )
        )
        
        if response.candidates:
            # Fix for the FinishReason error
            # Instead of checking finish_reason, we just check if we have text
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            else:
                return ""
        else:
            # Handle case when no candidates are returned
            if hasattr(response, 'prompt_feedback'):
                return f"No candidates returned. Prompt Feedback: {response.prompt_feedback}"
            else:
                return "No candidates returned and no prompt feedback available."
    except Exception as e:
        return f"Error generating content: {type(e).__name__}: {e}"


def get_mime_type(audio_file_path):
    """Determines the MIME type of an audio file."""
    ext = audio_file_path.suffix.lower()
    if ext == ".wav":
        return "audio/wav"
    elif ext == ".mp3":
        return "audio/mpeg"
    elif ext == ".flac":
        return "audio/flac"
    elif ext == ".ogg":
        return "audio/ogg"
    elif ext == ".m4a":
        return "audio/mp4"
    else:
        return "application/octet-stream"


def main():
    audio_dir = Path("/external2/datasets/hf_data_output/audio_chunks")
    output_file = "transcriptions_using_gem.txt"

    if not audio_dir.exists():
        raise ValueError(f"Directory not found: {audio_dir}")

    setup_gemini()

    audio_files = list(audio_dir.glob("*.*"))
    audio_files = [
        f for f in audio_files
        if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    ]
    total_files = len(audio_files)

    print(f"Found {total_files} audio files to process")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Transcription Results\n")
        f.write("-" * 80 + "\n")
        f.write("Audio File Path | Transcription\n")
        f.write("-" * 80 + "\n")

    successful_transcriptions = 0
    empty_audio_count = 0

    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"Processing [{i}/{total_files}]: {audio_file}")

            file_size = audio_file.stat().st_size
            if file_size > 250 * 1024 * 1024:
                error_msg = (
                    f"Skipping: File too large ({file_size / 1024 / 1024:.2f} MB)"
                )
                print(error_msg)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"{audio_file} | {error_msg}\n")
                continue

            mime_type = get_mime_type(audio_file)
            transcription = transcribe_audio(str(audio_file), mime_type)

            if transcription.startswith("Error"):
                print(
                    f"Failed to transcribe [{i}/{total_files}]: "
                    f"{audio_file} - {transcription}"
                )
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"{audio_file} | {transcription}\n")

            elif transcription == "":
                empty_audio_count += 1
                print(f"No speech detected in [{i}/{total_files}]: {audio_file}")
                # Do NOT write to output file

            else:
                successful_transcriptions += 1
                print(f"Successfully transcribed [{i}/{total_files}]: {audio_file}")
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"{audio_file} | {transcription}\n")

        except Exception as e:
            error_msg = f"Error processing {audio_file}: {str(e)}"
            print(error_msg)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{audio_file} | ERROR: {str(e)}\n")

    print(
        f"Successfully transcribed {successful_transcriptions} out of "
        f"{total_files} audio files."
    )
    print(f"Detected {empty_audio_count} audio files with no speech.")


if __name__ == "__main__":
    main()