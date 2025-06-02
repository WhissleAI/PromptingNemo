import json
import argparse
import os
from nemo.collections.asr.models import EncDecCTCModel
from tqdm import tqdm
import torch

def load_jsonl(jsonl_path):
    """
    Load a JSONL file where each line is a JSON object.
    Returns a list of dicts.
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main(args):
    # Load the model from checkpoint
    print(f"Loading model from {args.checkpoint_path}...")
    map_location = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    model = EncDecCTCModel.restore_from(restore_path=args.checkpoint_path, map_location=map_location)
    model.eval()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    input_files = [f for f in os.listdir(args.input_folder) if os.path.isfile(os.path.join(args.input_folder, f)) and (f.endswith('.jsonl') or f.endswith('.json'))]

    if not input_files:
        print(f"No .jsonl or .json files found in {args.input_folder}")
        return

    for input_filename in input_files:
        input_filepath = os.path.join(args.input_folder, input_filename)
        output_filename = input_filename
        output_filepath = os.path.join(args.output_folder, output_filename)

        # Read manifest entries
        print(f"Loading manifest: {input_filepath}")
        try:
            entries = load_jsonl(input_filepath)
        except Exception as e:
            print(f"Error loading {input_filepath}: {e}. Skipping this file.")
            continue
        
        if not entries:
            print(f"No entries found in {input_filepath}. Skipping this file.")
            continue

        # Transcription loop
        print(f"Transcribing audio files from {input_filepath}...")
        transcriptions = []
        for i in tqdm(range(0, len(entries), args.batch_size), desc=f"Batches for {input_filename}"):
            batch = entries[i : i + args.batch_size]
            audio_files = [e['audio_filepath'] for e in batch]

            # Perform inference (pass audio paths positionally)
            try:
                preds = model.transcribe(audio_files, batch_size=len(audio_files), return_hypotheses=False)
            except Exception as e:
                print(f"Error during transcription for a batch in {input_filename}: {e}. Skipping batch.")
                # Optionally, add placeholder results or skip failed items
                for entry in batch:
                    transcriptions.append({
                        'audio_filepath': entry['audio_filepath'],
                        'predicted_text': "ERROR_TRANSCRIBING",
                        'error_message': str(e)
                    })
                continue


            # Collect results
            for entry, pred in zip(batch, preds):
                result = {
                    'audio_filepath': entry['audio_filepath'],
                    'predicted_text': pred
                }
                if 'text' in entry:
                    result['text'] = entry['text']
                transcriptions.append(result)

        # Write output JSONL
        # Ensure the immediate directory for the output file exists (though args.output_folder is already created)
        # os.makedirs(os.path.dirname(output_filepath), exist_ok=True) # Not strictly needed if output_folder is flat
        print(f"Writing {len(transcriptions)} transcriptions to {output_filepath}")
        with open(output_filepath, 'w', encoding='utf-8') as out_f:
            for rec in transcriptions:
                out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')

        print(f"Transcription for {input_filename} completed.")
    print("All transcriptions completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio in JSONL manifest files from a folder using a NeMo CTC model checkpoint"
    )
    parser.add_argument(
        '--checkpoint_path', '-c', required=True,
        help="Path to the .nemo checkpoint"
    )
    parser.add_argument(
        '--input_folder', '-i', required=True,
        help="Folder containing JSONL manifest files with 'audio_filepath' (and optional 'text') fields"
    )
    parser.add_argument(
        '--output_folder', '-o', required=True,
        help="Output folder to save transcriptions, maintaining original filenames"
    )
    parser.add_argument(
        '--batch_size', '-b', type=int, default=4,
        help="Number of files to process per batch"
    )
    parser.add_argument(
        '--use_gpu', action='store_true',
        help="Enable GPU inference if available"
    )

    args = parser.parse_args()
    main(args)

# Example usage:
# python infer_json_folder.py \\
#     --checkpoint_path /path/to/model.nemo \\
#     --input_folder /path/to/input_manifest_folder \\
#     --output_folder /path/to/output_folder \\
#     --batch_size 8 --use_gpu
