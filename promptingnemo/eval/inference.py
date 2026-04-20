"""Inference utilities: transcribe audio files using a NeMo CTC checkpoint.

Extracted from scripts/asr/meta-asr/eval/infer.py.
"""

import json
import os

import torch
from nemo.collections.asr.models import EncDecCTCModel


def load_jsonl(jsonl_path: str):
    """Load a JSONL file where each line is a JSON object. Returns a list of dicts."""
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


def transcribe_manifest(
    checkpoint_path: str,
    input_jsonl: str,
    output_jsonl: str,
    batch_size: int = 4,
    use_gpu: bool = True,
    max_samples: int = None,
):
    """Transcribe audio files listed in a JSONL manifest using a NeMo CTC model.

    Args:
        checkpoint_path: Path to the .nemo checkpoint.
        input_jsonl: JSONL manifest with 'audio_filepath' (and optional 'text') fields.
        output_jsonl: Output JSONL file to save transcriptions.
        batch_size: Number of files to process per batch.
        use_gpu: Enable GPU inference if available.
        max_samples: If set, only process this many samples (useful for testing).
    """
    print(f"Loading model from {checkpoint_path}...")
    map_location = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    model = EncDecCTCModel.restore_from(restore_path=checkpoint_path, map_location=map_location)
    if use_gpu and torch.cuda.is_available():
        model = model.to('cuda')
    else:
        model = model.to('cpu')
    model.eval()

    print(f"Loading manifest: {input_jsonl}")
    entries = load_jsonl(input_jsonl)

    if max_samples is not None:
        entries = entries[:max_samples]
        print(f"Processing only first {len(entries)} samples")

    print("Transcribing audio files...")
    transcriptions = []
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        audio_files = [e['audio_filepath'] for e in batch]

        if not audio_files:
            continue

        preds = model.transcribe(audio_files, batch_size=len(audio_files), return_hypotheses=False)

        for entry, pred in zip(batch, preds):
            result = {
                'audio_filepath': entry['audio_filepath'],
                'predicted_text': _to_text(pred),
            }
            if 'text' in entry:
                result['text'] = entry['text']
            transcriptions.append(result)

    out_dir = os.path.dirname(output_jsonl) or '.'
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing {len(transcriptions)} transcriptions to {output_jsonl}")
    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        for rec in transcriptions:
            out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print("Transcription completed.")
    return transcriptions
