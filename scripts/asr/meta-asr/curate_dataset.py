"""Config-driven data curation pipeline for ASR training.

Merges multiple source manifests, applies quality filtering, extracts teacher
embeddings for diversity selection, and produces a balanced curated manifest.

Usage:
    python scripts/asr/meta-asr/curate_dataset.py \
        --config recipes/meta_asr/conf/curate_200k_distill.yaml
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from select_subset import extract_embeddings, k_center_greedy, write_selected_manifest
from clean_manifest_before_training import normalize_text, lang_family as LANG_CODE_TO_FAMILY

logging.basicConfig(level=logging.INFO, format='[%(levelname)s %(asctime)s] %(message)s')

KNOWN_FAMILIES = {
    'ENGLISH', 'EUROPEAN', 'SLAVIC', 'INDO_ARYAN',
    'DRAVIDIAN', 'MANDARIN', 'INDIAN_LOW_RESOURCE',
}


def _normalize_lang(raw_lang):
    """Map language code or family name to a canonical family string."""
    upper = raw_lang.upper().strip()
    if upper in KNOWN_FAMILIES:
        return upper
    if upper in LANG_CODE_TO_FAMILY:
        return LANG_CODE_TO_FAMILY[upper]
    return upper


def _apply_path_rewrites(audio_path, rewrites):
    """Apply prefix substitutions to audio_filepath."""
    if not rewrites:
        return audio_path
    for old_prefix, new_prefix in rewrites:
        if audio_path.startswith(old_prefix):
            return new_prefix + audio_path[len(old_prefix):]
    return audio_path


def merge_sources(sources_cfg, work_dir, path_rewrites=None):
    """Merge multiple source manifests into one with normalized lang field."""
    merged_path = os.path.join(work_dir, '_merged.json')
    total = 0
    lang_counts = Counter()

    rewrite_pairs = []
    if path_rewrites:
        for rule in path_rewrites:
            rewrite_pairs.append((rule['from'], rule['to']))

    with open(merged_path, 'w', encoding='utf-8') as out:
        for src in sources_cfg:
            manifest = src['manifest']
            lang_field = src.get('lang_field', 'lang')
            lang_override = src.get('lang_override', None)

            if not os.path.exists(manifest):
                logging.warning("Source manifest not found, skipping: %s", manifest)
                continue

            src_count = 0
            with open(manifest, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    raw_lang = lang_override or entry.get(lang_field, entry.get('lang_family', 'UNK'))
                    normalized = _normalize_lang(str(raw_lang))
                    entry['lang'] = normalized

                    if rewrite_pairs and 'audio_filepath' in entry:
                        entry['audio_filepath'] = _apply_path_rewrites(
                            entry['audio_filepath'], rewrite_pairs)

                    out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    lang_counts[normalized] += 1
                    total += 1
                    src_count += 1

            logging.info("Merged %d samples from %s", src_count, manifest)

    logging.info("Total merged: %d samples across %d languages", total, len(lang_counts))
    for lang, count in lang_counts.most_common():
        logging.info("  %s: %d", lang, count)

    return merged_path, total


def filter_manifest(manifest_path, work_dir, filter_cfg):
    """Apply duration and text quality filters."""
    filtered_path = os.path.join(work_dir, '_filtered.json')
    min_dur = filter_cfg.get('min_duration', 0.5)
    max_dur = filter_cfg.get('max_duration', 20.0)
    require_text = filter_cfg.get('require_text', True)
    do_normalize = filter_cfg.get('normalize_tags', False)

    kept = 0
    dropped = 0
    drop_reasons = Counter()

    with open(manifest_path, 'r', encoding='utf-8') as f, \
         open(filtered_path, 'w', encoding='utf-8') as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                drop_reasons['bad_json'] += 1
                continue

            dur = entry.get('duration', 0)
            if dur < min_dur:
                dropped += 1
                drop_reasons['too_short'] += 1
                continue
            if dur > max_dur:
                dropped += 1
                drop_reasons['too_long'] += 1
                continue

            text = entry.get('text', '')
            if require_text and not text.strip():
                dropped += 1
                drop_reasons['no_text'] += 1
                continue

            if do_normalize:
                entry['text'] = normalize_text(text)

            out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            kept += 1

    logging.info("Filtered: kept=%d, dropped=%d", kept, dropped)
    for reason, count in drop_reasons.most_common():
        logging.info("  dropped %s: %d", reason, count)

    return filtered_path, kept, dropped


def random_stratified_select(manifest_path, k, min_per_lang=1000, seed=42):
    """Fast stratified random selection as a baseline alternative to k-center."""
    entries_by_lang = defaultdict(list)
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            entry = json.loads(line.strip())
            lang = entry.get('lang', 'UNK')
            entries_by_lang[lang].append(i)

    rng = np.random.RandomState(seed)
    selected = []
    n_langs = len(entries_by_lang)

    for lang, indices in entries_by_lang.items():
        n_select = min(min_per_lang, len(indices))
        chosen = rng.choice(indices, size=n_select, replace=False)
        selected.extend(chosen.tolist())

    remaining_k = k - len(selected)
    if remaining_k > 0:
        all_indices = []
        for indices in entries_by_lang.values():
            all_indices.extend(indices)
        selected_set = set(selected)
        remaining = [i for i in all_indices if i not in selected_set]
        if remaining:
            extra = rng.choice(remaining, size=min(remaining_k, len(remaining)), replace=False)
            selected.extend(extra.tolist())

    selected = list(set(selected))[:k]
    logging.info("Random stratified: selected %d samples from %d languages", len(selected), n_langs)
    return selected


def compute_stats(manifest_path):
    """Compute and log language/duration statistics for a manifest."""
    lang_counts = Counter()
    lang_durations = defaultdict(float)
    total_dur = 0
    total = 0

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            lang = entry.get('lang', entry.get('lang_family', 'UNK'))
            dur = entry.get('duration', 0)
            lang_counts[lang] += 1
            lang_durations[lang] += dur
            total_dur += dur
            total += 1

    logging.info("=" * 60)
    logging.info("CURATED DATASET STATISTICS")
    logging.info("  Total samples: %d", total)
    logging.info("  Total duration: %.1f hours", total_dur / 3600)
    logging.info("  Languages: %d", len(lang_counts))
    logging.info("-" * 60)
    logging.info("  %-20s %8s %8s %8s", "Language", "Samples", "Hours", "Pct")
    logging.info("-" * 60)
    for lang, count in lang_counts.most_common():
        hours = lang_durations[lang] / 3600
        pct = 100 * count / total
        logging.info("  %-20s %8d %8.1f %7.1f%%", lang, count, hours, pct)
    logging.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Config-driven data curation for ASR training")
    parser.add_argument("--config", required=True, help="Path to curation YAML config")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding extraction (use cache)")
    parser.add_argument("--merge-only", action="store_true", help="Only run merge + filter steps")
    parser.add_argument("--device", default=None, help="Force device for embedding extraction (cpu/cuda)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    sources_cfg = cfg['sources']
    filter_cfg = cfg.get('filter', {})
    selection_cfg = cfg.get('selection', {})
    output_cfg = cfg['output']
    path_rewrites = cfg.get('path_rewrites', None)

    work_dir = output_cfg.get('work_dir', '/tmp/curate_work')
    os.makedirs(work_dir, exist_ok=True)

    # Step 1: Merge
    logging.info("=" * 60)
    logging.info("STEP 1: Merging %d source manifests", len(sources_cfg))
    logging.info("=" * 60)
    merged_path, total_merged = merge_sources(sources_cfg, work_dir, path_rewrites)

    # Step 2: Filter
    logging.info("=" * 60)
    logging.info("STEP 2: Filtering (duration=[%.1f, %.1f]s)",
                 filter_cfg.get('min_duration', 0.5), filter_cfg.get('max_duration', 20.0))
    logging.info("=" * 60)
    filtered_path, kept, dropped = filter_manifest(merged_path, work_dir, filter_cfg)

    if args.merge_only:
        logging.info("--merge-only: stopping after filter step")
        compute_stats(filtered_path)
        return

    k = selection_cfg.get('k', 200000)
    method = selection_cfg.get('method', 'k_center_greedy')
    min_per_lang = selection_cfg.get('min_per_lang', 20000)
    output_manifest = output_cfg['manifest']

    if method == 'random':
        # Step 3+4: Random stratified selection (no embeddings needed)
        logging.info("=" * 60)
        logging.info("STEP 3-4: Random stratified selection (k=%d)", k)
        logging.info("=" * 60)
        selected_indices = random_stratified_select(filtered_path, k, min_per_lang)
        write_selected_manifest(filtered_path, selected_indices, output_manifest)

    elif method == 'k_center_greedy':
        import torch
        from nemo.collections.asr.models import ASRModel
        try:
            from promptingnemo.models.ctc_model import CustomEncDecCTCModelBPE
        except ImportError:
            CustomEncDecCTCModelBPE = None

        embeddings_cache = selection_cfg.get('embeddings_cache', None)
        embeddings = None

        # Step 3: Embed
        if embeddings_cache and Path(embeddings_cache).exists() and not args.skip_embed:
            try:
                data = np.load(embeddings_cache, allow_pickle=True)
                if 'batch_cursor' not in data.files:
                    logging.info("Loading completed embeddings cache from %s", embeddings_cache)
                    embeddings = data['embeddings']
                    lang_ids = data['lang_ids'].tolist()
                    line_indices = data['line_indices'].tolist()
                else:
                    logging.info("Found in-progress checkpoint, will resume extraction")
            except Exception:
                pass

        if embeddings is None:
            logging.info("=" * 60)
            logging.info("STEP 3: Extracting teacher embeddings")
            logging.info("=" * 60)

            teacher_model_path = selection_cfg['teacher_model']
            batch_size = selection_cfg.get('batch_size', 32)
            device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

            logging.info("Loading teacher model from %s on %s", teacher_model_path, device)
            if CustomEncDecCTCModelBPE is not None:
                model = CustomEncDecCTCModelBPE.restore_from(
                    teacher_model_path, map_location=device, strict=False)
            else:
                model = ASRModel.restore_from(teacher_model_path, map_location=device)
            model = model.to(device)

            embeddings, lang_ids, line_indices = extract_embeddings(
                model, filtered_path, batch_size=batch_size,
                checkpoint_path=embeddings_cache,
            )

            if embeddings_cache:
                cache_dir = os.path.dirname(embeddings_cache)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                tmp_base = embeddings_cache + '.tmp'
                np.savez(
                    tmp_base,
                    embeddings=embeddings,
                    lang_ids=np.array(lang_ids, dtype=object),
                    line_indices=np.array(line_indices),
                )
                tmp_actual = tmp_base if tmp_base.endswith('.npz') else tmp_base + '.npz'
                os.replace(tmp_actual, embeddings_cache)
                logging.info("Saved embeddings cache to %s", embeddings_cache)

            del model
            torch.cuda.empty_cache()

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        # Step 4: K-center greedy selection
        logging.info("=" * 60)
        logging.info("STEP 4: K-center greedy selection (k=%d, min_per_lang=%d)", k, min_per_lang)
        logging.info("=" * 60)
        selected_local = k_center_greedy(
            embeddings, k=k, lang_ids=lang_ids, min_per_lang=min_per_lang
        )
        selected_line_indices = [line_indices[i] for i in selected_local]

        # Log language distribution of selection
        lang_counts = Counter()
        for i in selected_local:
            lang_counts[lang_ids[i]] += 1
        logging.info("Selected %d samples:", len(selected_line_indices))
        for lang, count in lang_counts.most_common():
            logging.info("  %s: %d (%.1f%%)", lang, count, 100 * count / len(selected_line_indices))

        write_selected_manifest(filtered_path, selected_line_indices, output_manifest)

    else:
        logging.error("Unknown selection method: %s (use 'k_center_greedy' or 'random')", method)
        sys.exit(1)

    # Step 5: Stats
    if output_cfg.get('stats', True):
        compute_stats(output_manifest)

    logging.info("Done. Curated manifest: %s", output_manifest)


if __name__ == "__main__":
    main()
