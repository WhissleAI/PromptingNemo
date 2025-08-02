import json
import nemo.collections.asr as nemo_asr
from tqdm import tqdm
import sys

def load_manifest(manifest_path):
    """Load the manifest file and return a list of audio file paths."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = [json.loads(line) for line in f]
    return manifest_data

def transcribe_audio_files(model, manifest_data):
    """Transcribe each audio file in the manifest using the ASR model."""
    transcriptions = []
    for entry in tqdm(manifest_data, desc="Transcribing audio files"):
        audio_path = entry["audio_filepath"]
        transcript = model.transcribe([audio_path])[0]
        transcriptions.append({
            "audio_filepath": audio_path,
            "text": entry["text"],
            "predicted_text": transcript
        })
    return transcriptions

def save_transcriptions(transcriptions, output_path):
    """Save transcriptions to a JSON file with UTF-8 encoding."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for transcription in transcriptions:
            f.write(json.dumps(transcription, ensure_ascii=False) + "\n")

# Paths to the manifest and model checkpoint, as provided by the user
manifest_path = sys.argv[1]
checkpoint_path = sys.argv[2]
output_path = sys.argv[3]

# Load ASR model from checkpoint
model = nemo_asr.models.EncDecCTCModel.restore_from(checkpoint_path)

# Load the manifest file
manifest_data = load_manifest(manifest_path)

# Transcribe audio files in the manifest
transcriptions = transcribe_audio_files(model, manifest_data)

# Save the transcriptions with UTF-8 encoding
save_transcriptions(transcriptions, output_path)

print(f"Transcriptions saved to {output_path}")

