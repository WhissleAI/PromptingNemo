"""Select a diverse subset of training data using k-center greedy on teacher embeddings.

Usage:
    python scripts/asr/meta-asr/select_subset.py \
        --model /path/to/teacher.nemo \
        --manifest /path/to/train.json \
        --output /path/to/train_selected_500k.json \
        --k 500000 \
        --batch-size 32
"""

__all__ = ['extract_embeddings', 'k_center_greedy', 'write_selected_manifest']

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nemo.collections.asr.models import ASRModel

try:
    from promptingnemo.models.ctc_model import CustomEncDecCTCModelBPE
except ImportError:
    CustomEncDecCTCModelBPE = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s %(asctime)s] %(message)s')

CHECKPOINT_INTERVAL = 500  # save every N batches


def _load_and_pad_batch(audio_paths, device):
    """Load audio files and pad to a single batch tensor."""
    import soundfile as sf
    audios = []
    lengths = []
    valid_indices = []
    for i, path in enumerate(audio_paths):
        try:
            audio, sr = sf.read(path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audios.append(torch.tensor(audio, dtype=torch.float32))
            lengths.append(len(audio))
            valid_indices.append(i)
        except Exception:
            continue

    if not audios:
        return None, None, []

    max_len = max(lengths)
    batch = torch.zeros(len(audios), max_len, dtype=torch.float32)
    for i, audio in enumerate(audios):
        batch[i, :len(audio)] = audio
    lens = torch.tensor(lengths, dtype=torch.long)
    return batch.to(device), lens.to(device), valid_indices


def _save_checkpoint(checkpoint_path: str, embeddings_list: list, lang_ids: list,
                     line_indices: list, batch_cursor: int):
    """Save extraction progress to an incremental checkpoint on NFS."""
    if not embeddings_list:
        return
    emb_array = np.stack(embeddings_list, axis=0)
    tmp_base = checkpoint_path + '.tmp'
    np.savez(
        tmp_base,
        embeddings=emb_array,
        lang_ids=np.array(lang_ids, dtype=object),
        line_indices=np.array(line_indices),
        batch_cursor=np.array(batch_cursor),
    )
    # np.savez appends .npz if not already present
    tmp_actual = tmp_base if tmp_base.endswith('.npz') else tmp_base + '.npz'
    os.replace(tmp_actual, checkpoint_path)
    logging.info("Checkpoint saved: %d embeddings, batch_cursor=%d → %s",
                 len(embeddings_list), batch_cursor, checkpoint_path)


def _load_checkpoint(checkpoint_path: str):
    """Load extraction checkpoint. Returns (embeddings_list, lang_ids, line_indices, batch_cursor) or None."""
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None
    try:
        data = np.load(checkpoint_path, allow_pickle=True)
        embeddings_list = list(data['embeddings'])
        lang_ids = data['lang_ids'].tolist()
        line_indices = data['line_indices'].tolist()
        batch_cursor = int(data['batch_cursor'])
        logging.info("Resumed from checkpoint: %d embeddings, batch_cursor=%d",
                     len(embeddings_list), batch_cursor)
        return embeddings_list, lang_ids, line_indices, batch_cursor
    except Exception as e:
        logging.warning("Failed to load checkpoint %s: %s — starting fresh", checkpoint_path, e)
        return None


def extract_embeddings(model, manifest_path: str, batch_size: int = 32,
                       max_duration: float = 20.0, checkpoint_path: str = None) -> tuple:
    """Extract average-pooled encoder embeddings from the teacher model.

    Saves incremental checkpoints to checkpoint_path every CHECKPOINT_INTERVAL batches.
    On restart, resumes from the last checkpoint.

    Returns:
        embeddings: np.ndarray of shape [N, D]
        lang_ids: list of language IDs per sample
        line_indices: list of original manifest line indices
    """
    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            entry = json.loads(line.strip())
            dur = entry.get('duration', 0)
            if 0.5 <= dur <= max_duration:
                entries.append((i, entry))

    logging.info("Loaded %d samples (filtered by duration [0.5, %.0f]s)", len(entries), max_duration)

    # Try to resume from checkpoint
    start_batch = 0
    resumed = _load_checkpoint(checkpoint_path) if checkpoint_path else None
    if resumed:
        embeddings, lang_ids, line_indices, start_batch = resumed
    else:
        embeddings = []
        lang_ids = []
        line_indices = []

    model.eval()
    device = next(model.parameters()).device

    total_batches = (len(entries) + batch_size - 1) // batch_size
    batches_since_save = 0

    for batch_idx in tqdm(range(start_batch, total_batches), desc="Extracting embeddings",
                          initial=start_batch, total=total_batches):
        batch_start = batch_idx * batch_size
        batch_entries = entries[batch_start:batch_start + batch_size]
        audio_paths = [e['audio_filepath'] for _, e in batch_entries]
        batch_langs = [e.get('lang', e.get('lang_family', 'UNK')).upper() for _, e in batch_entries]
        batch_idxs = [idx for idx, _ in batch_entries]

        try:
            batch_audio, batch_lens, valid = _load_and_pad_batch(audio_paths, device)
            if batch_audio is None:
                batches_since_save += 1
                continue

            with torch.no_grad():
                processed, processed_len = model.preprocessor(
                    input_signal=batch_audio, length=batch_lens
                )
                encoded, encoded_len = model.encoder(
                    audio_signal=processed, length=processed_len
                )

            # encoded: [B, D, T] or [B, T, D]
            is_d_first = encoded.dim() == 3 and encoded.shape[1] > encoded.shape[2]

            for j, vi in enumerate(valid):
                t_len = encoded_len[j].item()
                if is_d_first:
                    emb = encoded[j, :, :t_len].mean(dim=1)
                else:
                    emb = encoded[j, :t_len, :].mean(dim=0)

                embeddings.append(emb.cpu().numpy())
                lang_ids.append(batch_langs[vi])
                line_indices.append(batch_idxs[vi])

        except Exception as e:
            logging.warning("Batch failed at %d, falling back to file-by-file: %s", batch_start, e)
            for idx, entry in batch_entries:
                try:
                    import soundfile as sf
                    audio_path = entry['audio_filepath']
                    lang = entry.get('lang', entry.get('lang_family', 'UNK')).upper()
                    audio, sr = sf.read(audio_path)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
                    audio_len = torch.tensor([len(audio)], dtype=torch.long, device=device)
                    with torch.no_grad():
                        processed, processed_len = model.preprocessor(
                            input_signal=audio_tensor, length=audio_len
                        )
                        encoded, encoded_len = model.encoder(
                            audio_signal=processed, length=processed_len
                        )
                    if encoded.dim() == 3:
                        if encoded.shape[1] > encoded.shape[2]:
                            emb = encoded[0, :, :encoded_len[0]].mean(dim=1)
                        else:
                            emb = encoded[0, :encoded_len[0], :].mean(dim=0)
                    else:
                        emb = encoded.mean(dim=0)
                    embeddings.append(emb.cpu().numpy())
                    lang_ids.append(lang)
                    line_indices.append(idx)
                except Exception:
                    continue

        batches_since_save += 1

        # Incremental checkpoint to NFS
        if checkpoint_path and batches_since_save >= CHECKPOINT_INTERVAL:
            _save_checkpoint(checkpoint_path, embeddings, lang_ids, line_indices, batch_idx + 1)
            batches_since_save = 0

    # Final checkpoint
    if checkpoint_path and batches_since_save > 0:
        _save_checkpoint(checkpoint_path, embeddings, lang_ids, line_indices, total_batches)

    embeddings = np.stack(embeddings, axis=0) if embeddings else np.zeros((0, 1))
    logging.info("Extracted embeddings: shape=%s", embeddings.shape)
    return embeddings, lang_ids, line_indices


def k_center_greedy(embeddings: np.ndarray, k: int, lang_ids: list = None,
                    min_per_lang: int = 1000) -> list:
    """K-center greedy selection for maximum diversity.

    Uses squared L2 via dot-product identity: ||a-b||² = ||a||² + ||b||² - 2a·b
    which reduces each iteration to a BLAS sgemv (~10x faster than naive norm).
    """
    n, d = embeddings.shape
    if k >= n:
        logging.info("k=%d >= n=%d, returning all samples", k, n)
        return list(range(n))

    selected = []

    # Phase 1: ensure minimum per-language coverage
    if lang_ids and min_per_lang > 0:
        lang_to_indices = defaultdict(list)
        for i, lang in enumerate(lang_ids):
            lang_to_indices[lang].append(i)

        for lang, indices in lang_to_indices.items():
            n_select = min(min_per_lang, len(indices), k // max(len(lang_to_indices), 1))
            if n_select > 0:
                rng = np.random.RandomState(42)
                chosen = rng.choice(indices, size=n_select, replace=False)
                selected.extend(chosen.tolist())

        selected = list(set(selected))
        logging.info("Phase 1: selected %d samples for language coverage (%d languages)",
                     len(selected), len(lang_to_indices))

    remaining_k = k - len(selected)
    if remaining_k <= 0:
        return selected[:k]

    # Phase 2: k-center greedy on remaining budget
    if not selected:
        first = np.random.RandomState(42).randint(0, n)
        selected.append(first)
        remaining_k -= 1

    # Precompute squared norms for dot-product distance trick
    sq_norms = np.einsum('ij,ij->i', embeddings, embeddings).astype(np.float32)

    min_dist_sq = np.full(n, np.inf, dtype=np.float32)
    for s in selected:
        dots = embeddings @ embeddings[s]
        dist_sq = sq_norms + sq_norms[s] - 2.0 * dots
        np.maximum(dist_sq, 0.0, out=dist_sq)
        np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)

    # Mark seeds as already selected
    for s in selected:
        min_dist_sq[s] = -1.0

    log_interval = max(remaining_k // 100, 1000)
    for i in tqdm(range(remaining_k), desc="K-center greedy"):
        new_idx = int(np.argmax(min_dist_sq))
        selected.append(new_idx)
        min_dist_sq[new_idx] = -1.0

        dots = embeddings @ embeddings[new_idx]
        dist_sq = sq_norms + sq_norms[new_idx] - 2.0 * dots
        np.maximum(dist_sq, 0.0, out=dist_sq)
        np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)

        if (i + 1) % log_interval == 0:
            logging.info("K-center progress: %d/%d (%.1f%%)", i + 1, remaining_k,
                         100 * (i + 1) / remaining_k)

    return selected


def write_selected_manifest(original_manifest: str, selected_line_indices: list, output_path: str):
    """Write a new manifest containing only the selected samples."""
    selected_set = set(selected_line_indices)

    with open(original_manifest, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    written = 0
    with open(output_path, 'w', encoding='utf-8') as out:
        for i, line in enumerate(lines):
            if i in selected_set:
                out.write(line)
                written += 1

    logging.info("Wrote %d samples to %s", written, output_path)


def main():
    parser = argparse.ArgumentParser(description="Select diverse training subset using teacher embeddings")
    parser.add_argument("--model", required=True, help="Path to teacher .nemo model")
    parser.add_argument("--manifest", required=True, help="Path to full training manifest")
    parser.add_argument("--output", required=True, help="Output path for selected manifest")
    parser.add_argument("--k", type=int, default=500000, help="Number of samples to select")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding extraction")
    parser.add_argument("--min-per-lang", type=int, default=20000, help="Minimum samples per language")
    parser.add_argument("--embeddings-cache", default=None, help="Path to cache/load embeddings (.npz)")
    args = parser.parse_args()

    # Use embeddings-cache as both final cache and incremental checkpoint path
    checkpoint_path = args.embeddings_cache

    if checkpoint_path and Path(checkpoint_path).exists():
        # Check if this is a completed cache (no batch_cursor) or an in-progress checkpoint
        try:
            data = np.load(checkpoint_path, allow_pickle=True)
            files_in_cache = list(data.files)
            if 'batch_cursor' not in files_in_cache:
                # Completed cache from a previous full run — use directly
                logging.info("Loading completed embeddings cache from %s", checkpoint_path)
                embeddings = data['embeddings']
                lang_ids = data['lang_ids'].tolist()
                line_indices = data['line_indices'].tolist()
            else:
                # In-progress checkpoint — will be resumed inside extract_embeddings
                logging.info("Found in-progress checkpoint, will resume extraction")
                embeddings = lang_ids = line_indices = None
        except Exception:
            embeddings = lang_ids = line_indices = None
    else:
        embeddings = lang_ids = line_indices = None

    if embeddings is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info("Loading teacher model from %s", args.model)
        if CustomEncDecCTCModelBPE is not None:
            model = CustomEncDecCTCModelBPE.restore_from(args.model, map_location=device, strict=False)
        else:
            model = ASRModel.restore_from(args.model, map_location=device)
        model = model.to(device)

        embeddings, lang_ids, line_indices = extract_embeddings(
            model, args.manifest, batch_size=args.batch_size,
            checkpoint_path=checkpoint_path,
        )

        # Save final cache without batch_cursor (marks it as complete)
        if checkpoint_path:
            tmp_base = checkpoint_path + '.tmp'
            np.savez(
                tmp_base,
                embeddings=embeddings,
                lang_ids=np.array(lang_ids, dtype=object),
                line_indices=np.array(line_indices),
            )
            tmp_actual = tmp_base if tmp_base.endswith('.npz') else tmp_base + '.npz'
            os.replace(tmp_actual, checkpoint_path)
            logging.info("Saved completed embeddings cache to %s", checkpoint_path)

        del model
        torch.cuda.empty_cache()

    # Normalize embeddings for better distance computation
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings = embeddings / norms

    selected_local_indices = k_center_greedy(
        embeddings, k=args.k, lang_ids=lang_ids, min_per_lang=args.min_per_lang
    )

    # Map back to original manifest line indices
    selected_line_indices = [line_indices[i] for i in selected_local_indices]

    # Log language distribution
    lang_counts = defaultdict(int)
    for i in selected_local_indices:
        lang_counts[lang_ids[i]] += 1
    logging.info("Selected %d samples. Language distribution:", len(selected_line_indices))
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        logging.info("  %s: %d (%.1f%%)", lang, count, 100 * count / len(selected_line_indices))

    write_selected_manifest(args.manifest, selected_line_indices, args.output)


if __name__ == "__main__":
    main()
