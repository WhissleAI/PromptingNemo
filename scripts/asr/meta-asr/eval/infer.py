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

    # Read manifest entries
    print(f"Loading manifest: {args.input_jsonl}")
    entries = load_jsonl(args.input_jsonl)

    # Transcription loop
    print("Transcribing audio files...")
    transcriptions = []
    for i in tqdm(range(0, len(entries), args.batch_size), desc="Batches"):  
        batch = entries[i : i + args.batch_size]
        audio_files = [e['audio_filepath'] for e in batch]

        # Perform inference (pass audio paths positionally)
        preds = model.transcribe(audio_files, batch_size=len(audio_files), return_hypotheses=False)

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
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    print(f"Writing {len(transcriptions)} transcriptions to {args.output_jsonl}")
    with open(args.output_jsonl, 'w', encoding='utf-8') as out_f:
        for rec in transcriptions:
            out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print("Transcription completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio in JSONL manifest using a NeMo CTC model checkpoint"
    )
    parser.add_argument(
        '--checkpoint_path', '-c', required=True,
        help="Path to the .nemo checkpoint"
    )
    parser.add_argument(
        '--input_jsonl', '-i', required=True,
        help="JSONL manifest with 'audio_filepath' (and optional 'text') fields"
    )
    parser.add_argument(
        '--output_jsonl', '-o', required=True,
        help="Output JSONL file to save transcriptions"
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
# python infer.py \
#     --checkpoint_path /path/to/model.nemo \
#     --input_jsonl /path/to/manifest.jsonl \
#     --output_jsonl /path/to/output.jsonl \
#     --batch_size 8 --use_gpu
