import nemo.collections.asr as nemo_asr
import json
from nemo.collections.asr.metrics.wer import word_error_rate

# Load a pretrained ASR model
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_conformer_ctc_large")

# Path to the manifest file
manifest_path = "test_clean_manifest.jsonl"

# Output results
results = []

# Perform inference on each audio file
with open(manifest_path, "r") as manifest_file:
    for line in manifest_file:
        entry = json.loads(line)
        audio_path = entry["audio_filepath"]
        ground_truth = entry["text"]

        # Run inference
        predicted_text = asr_model.transcribe([audio_path])[0]

        # Append result
        results.append({
            "audio_filepath": audio_path,
            "ground_truth": ground_truth,
            "predicted_text": predicted_text
        })

# Save the results to a file
output_path = "asr_results.json"
with open(output_path, "w") as output_file:
    json.dump(results, output_file, indent=4)

print(f"Results saved to {output_path}")

# Calculate WER
references = [result["ground_truth"] for result in results]
hypotheses = [result["predicted_text"] for result in results]
wer = word_error_rate(hypotheses=hypotheses, references=references)

# Display results
print(f"Word Error Rate (WER): {wer:.2%}")

# # Optionally display detailed results
# for res in results:
#     print(f"\nAudio File: {res['audio_filepath']}")
#     print(f"Ground Truth: {res['ground_truth']}")
#     print(f"Predicted Text: {res['predicted_text']}")

