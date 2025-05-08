import os
import json
from datasets import load_dataset
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.metrics.wer import word_error_rate

# Step 1: Load the test set from Hugging Face
def load_and_prepare_dataset():
    print("Loading LibriSpeech test-clean dataset from Hugging Face...")
    librispeech = load_dataset(
        "librispeech_asr", "clean", split="test", download_mode="reuse_dataset_if_exists"
    )

    manifest_path = "test_manifest.json"
    print(f"Saving dataset as NeMo-compatible manifest at: {manifest_path}")
    with open(manifest_path, "w") as f:
        for example in librispeech:
            entry = {
                "audio_filepath": example["file"],
                "duration": None,  # Duration is optional; NeMo will infer it.
                "text": example["text"]
            }
            f.write(json.dumps(entry) + "\n")
    return manifest_path

# Step 2: Load a pre-trained NeMo ASR model
def load_asr_model():
    print("Loading pre-trained NeMo ASR model...")
    asr_model = ASRModel.from_pretrained(model_name="stt_en_conformer_ctc_large")
    return asr_model

# Step 3: Evaluate the ASR model on the test set
def evaluate_model(asr_model, manifest_path):
    print("Evaluating the ASR model...")
    predictions = []
    with open(manifest_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            audio_path = entry["audio_filepath"]
            reference_text = entry["text"]

            # Transcribe audio
            transcription = asr_model.transcribe([audio_path])[0]
            predictions.append({
                "reference": reference_text,
                "prediction": transcription
            })

    return predictions

# Step 4: Calculate Word Error Rate (WER)
def calculate_wer(predictions):
    print("Calculating Word Error Rate (WER)...")
    references = [entry["reference"] for entry in predictions]
    hypotheses = [entry["prediction"] for entry in predictions]

    wer_score = word_error_rate(hypotheses=hypotheses, references=references)
    return wer_score

# Step 5: Save detailed results
def save_results(predictions, output_path="asr_evaluation_results.json"):
    print(f"Saving results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)

# Main execution
def main():
    # Step 1: Prepare dataset
    manifest_path = load_and_prepare_dataset()

    # Step 2: Load the ASR model
    asr_model = load_asr_model()

    # Step 3: Evaluate the model
    predictions = evaluate_model(asr_model, manifest_path)

    # Step 4: Calculate WER
    wer_score = calculate_wer(predictions)
    print(f"Word Error Rate (WER): {wer_score * 100:.2f}%")

    # Step 5: Save results
    save_results(predictions)

if __name__ == "__main__":
    main()
