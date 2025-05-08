import sys
import json
from pydub import AudioSegment
import os

# Function to get audio file duration
def get_audio_duration(audio_filepath):
    if os.path.exists(audio_filepath):
        audio = AudioSegment.from_wav(audio_filepath)
        duration_ms = len(audio)
        duration_seconds = duration_ms / 1000.0
        return duration_seconds
    else:
        return None

if len(sys.argv) != 3:
    print("Usage: python script.py input_file.json output_file.json")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Read JSON objects line by line
data = []

with open(input_file, 'r') as json_file:
    for line in json_file:
        try:
            item = json.loads(line)
            audio_filepath = item.get("audio_filepath")
            if audio_filepath:
                duration = get_audio_duration(audio_filepath)
                if duration is not None:
                    item["duration"] = duration
                    data.append(item)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line.strip()}")

# Save the updated data to a new JSON file
with open(output_file, 'w') as json_output_file:
    for item in data:
        json_output_file.write(json.dumps(item) + '\n')

print("JSON file with durations created successfully.")
