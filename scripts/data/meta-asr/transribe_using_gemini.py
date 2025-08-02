import os
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai
import base64
from datetime import datetime

load_dotenv()

def setup_gemini():
    """Initialize Gemini API with your credentials"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    genai.configure(api_key=api_key)

def transcribe_audio(audio_path):
    """
    Transcribe audio using Gemini Vision Pro model (which can handle audio files)
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Read and encode the audio file
    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    
    # Create proper parts structure for the API
    parts = [
        {"text": "Please transcribe this audio file accurately. Return only the transcription text."},
        {
            "inline_data": {
                "mime_type": "audio/wav",
                "data": audio_base64
            }
        }
    ]
    
    try:
        response = model.generate_content(
            parts,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                top_p=1,
                top_k=32,
                max_output_tokens=8192,
            )
        )
        
        if response.text:
            return response.text.strip()
        else:
            return "No transcription generated"
            
    except Exception as e:
        print(f"Error generating content: {str(e)}")
        return f"Error: {str(e)}"

def main():
    # Directory containing audio files
    audio_dir = Path("/external2/datasets/hf_data_output/audio_chunks")  # Updated path
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = f"transcriptions_{timestamp}.txt"
    
    if not audio_dir.exists():
        raise ValueError(f"Directory not found: {audio_dir}")
    
    # Setup Gemini
    setup_gemini()
    
    # Get total number of WAV files
    wav_files = list(audio_dir.glob("*.wav"))
    total_files = len(wav_files)
    
    print(f"Found {total_files} WAV files to process")
    
    # Create or clear the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Transcription Results\n")
        f.write(f"Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        f.write("-" * 80 + "\n")
        f.write("Audio File Path | Transcription\n")
        f.write("-" * 80 + "\n")
 
    successful_transcriptions = 0
    
    for i, audio_file in enumerate(wav_files, 1):
        try:
            print(f"Processing [{i}/{total_files}]: {audio_file}")
            
            file_size = audio_file.stat().st_size
            if file_size > 20 * 1024 * 1024:  # 20MB limit
                error_msg = f"Skipping: File too large ({file_size / 1024 / 1024:.2f} MB)"
                print(error_msg)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"{audio_file} | {error_msg}\n")
                continue
                
            transcription = transcribe_audio(str(audio_file))
            
            # Write to output file
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{audio_file} | {transcription}\n")
            
            if not transcription.startswith("Error:"):
                successful_transcriptions += 1
                print(f"Successfully transcribed [{i}/{total_files}]: {audio_file}")
            else:
                print(f"Failed to transcribe [{i}/{total_files}]: {audio_file}")
            
        except Exception as e:
            error_msg = f"Error processing {audio_file}: {str(e)}"
            print(error_msg)
            # Log the error to the output file
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{audio_file} | ERROR: {str(e)}\n")


if __name__ == "__main__":
    main()