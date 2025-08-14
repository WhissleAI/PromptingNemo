import json
import argparse
import os
from nemo.collections.asr.models import EncDecCTCModel
from tqdm import tqdm
import torch
from types import SimpleNamespace

def load_jsonl(jsonl_path):
    """
    Load a JSONL file where each line is a JSON object.
    Returns a list of dicts.
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_text(pred):
    """Normalize a prediction object into a plain string.
    Handles strings, NeMo Hypothesis objects, dicts, or nested lists/tuples.
    """
    # direct string
    if isinstance(pred, str):
        return pred
    # empty list/tuple
    if isinstance(pred, (list, tuple)):
        return _to_text(pred[0]) if pred else ""
    # common Hypothesis or similar objects
    if hasattr(pred, 'text') and isinstance(getattr(pred, 'text'), str):
        return getattr(pred, 'text')
    if hasattr(pred, 'normalized_text') and isinstance(getattr(pred, 'normalized_text'), str):
        return getattr(pred, 'normalized_text')
    # dict-like
    if isinstance(pred, dict):
        for key in ('text', 'normalized_text', 'pred_text', 'transcription'):
            if key in pred and isinstance(pred[key], str):
                return pred[key]
    # fallback
    return str(pred)


def main(args):
    # Load the model from checkpoint
    print(f"Loading model from {args.checkpoint_path}...")
    map_location = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    model = EncDecCTCModel.restore_from(restore_path=args.checkpoint_path, map_location=map_location)
    # Ensure model is on the intended device
    if args.use_gpu and torch.cuda.is_available():
        model = model.to('cuda')
    else:
        model = model.to('cpu')
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
        'predicted_text': _to_text(pred)
            }
            if 'text' in entry:
                result['text'] = entry['text']
            transcriptions.append(result)

    # Write output JSONL
    out_dir = os.path.dirname(args.output_jsonl) or '.'
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing {len(transcriptions)} transcriptions to {args.output_jsonl}")
    with open(args.output_jsonl, 'w', encoding='utf-8') as out_f:
        for rec in transcriptions:
            out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print("Transcription completed.")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Transcribe audio in JSONL manifest using a NeMo CTC model checkpoint"
    # )
    # parser.add_argument(
    #     '--checkpoint_path', '-c', required=True,
    #     help="Path to the .nemo checkpoint"
    # )
    # parser.add_argument(
    #     '--input_jsonl', '-i', required=True,
    #     help="JSONL manifest with 'audio_filepath' (and optional 'text') fields"
    # )
    # parser.add_argument(
    #     '--output_jsonl', '-o', required=True,
    #     help="Output JSONL file to save transcriptions"
    # )
    # parser.add_argument(
    #     '--batch_size', '-b', type=int, default=4,
    #     help="Number of files to process per batch"
    # )
    # parser.add_argument(
    #     '--use_gpu', action='store_true',
    #     help="Enable GPU inference if available"
    # )

    # args = parser.parse_args()
    model_path = "/external3/databases/wellness-jsonl/experiment/wellness_adapter-bucket_data_himanshu/2025-08-13_12-34-29/checkpoints/wellness_adapter-bucket_data_himanshu.nemo"
    input_jsonl = "/external3/databases/wellness-jsonl/jsonl_files/data/valid.jsonl"
    output_jsonl = "/external3/databases/wellness-jsonl/jsonl_files/valid_test.jsonl"
    batch_size = 16
    use_gpu = True  # set to False to force CPU

    args = SimpleNamespace(
        checkpoint_path=model_path,
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        batch_size=batch_size,
        use_gpu=use_gpu,
    )
    main(args)

# Example usage:
# python infer.py \
#     --checkpoint_path /path/to/model.nemo \
#     --input_jsonl /path/to/manifest.jsonl \
#     --output_jsonl /path/to/output.jsonl \
#     --batch_size 8 --use_gpu
