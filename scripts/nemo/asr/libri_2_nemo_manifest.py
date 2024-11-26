import os
import json
import soundfile as sf

# Path to the test-clean folder
test_clean_path = "/external1/datasets/librespeech/LibriSpeech/test-clean"

# Output manifest file
manifest_path = "test_clean_manifest.jsonl"

# Function to get duration of audio file
def get_duration(file_path):
    with sf.SoundFile(file_path) as audio_file:
        return len(audio_file) / audio_file.samplerate

# Open the manifest file for writing
with open(manifest_path, "w") as manifest_file:
    for root, dirs, files in os.walk(test_clean_path):
        for file in files:
            if file.endswith(".flac"):
                # Full path to the .flac file
                flac_path = os.path.join(root, file)

                # Identify the transcription file based on the parent directory structure
                parent_dir = os.path.basename(root)  # e.g., 141084
                grandparent_dir = os.path.basename(os.path.dirname(root))  # e.g., 1580
                transcription_file = f"{grandparent_dir}-{parent_dir}.trans.txt"
                transcription_path = os.path.join(root, transcription_file)

                # Ensure the transcription file exists
                if not os.path.exists(transcription_path):
                    print(f"Transcription file not found: {transcription_path}")
                    continue

                # Parse the transcription file to find the text
                with open(transcription_path, "r") as transcriptions:
                    for line in transcriptions:
                        key, text = line.strip().split(" ", 1)
                        if key == os.path.splitext(file)[0]:  # Match file basename
                            # Calculate duration
                            duration = get_duration(flac_path)

                            # Create JSON entry
                            entry = {
                                "audio_filepath": flac_path,
                                "duration": duration,
                                "text": text.lower()
                            }

                            # Write to the manifest
                            manifest_file.write(json.dumps(entry) + "\n")
                            break
