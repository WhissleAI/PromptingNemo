from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import os

def convert_webm_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path, format="webm")
        audio.export(output_path, format="wav")
        return True
    except CouldntDecodeError as e:
        print(f"Error decoding {input_path}: {e}")
        return False

def convert_all_webm_to_wav(input_directory, output_directory=None):
    """Convert all .webm files in the given directory to .wav files."""
    if output_directory is None:
        output_directory = input_directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    converted_count = 0
    error_count = 0
    
    for filename in os.listdir(input_directory):
        if filename.lower().endswith('.webm'):
            webm_path = os.path.join(input_directory, filename)
            wav_filename = os.path.splitext(filename)[0] + '.wav'
            wav_path = os.path.join(output_directory, wav_filename)
            
            try:
                audio = AudioSegment.from_file(webm_path, format="webm")
                audio.export(wav_path, format="wav")
                print(f"Converted: {webm_path} -> {wav_path}")
                converted_count += 1
            except CouldntDecodeError as e:
                print(f"Error: Could not decode {webm_path} - file may be corrupted or invalid")
                error_count += 1
            except Exception as e:
                print(f"Unexpected error processing {webm_path}: {e}")
                error_count += 1
    
    print(f"\nConversion complete: {converted_count} files converted, {error_count} errors")

# Example usage:
input_dir = "E:/Meta_asr/datasets/raw_webm"
output_dir = "E:/Meta_asr/datasets/raw_wav"  # Specify output directory

convert_all_webm_to_wav(input_dir, output_dir)