#!/usr/bin/env python3
"""Download WhissleAI HuggingFace datasets and create text-only manifests.

Creates JSONL manifests with normalized, cleaned tagged text for training
the Text CTC Tagger. Strips audio — only text is needed.

Usage:
    python download_and_prepare.py --output-dir /path/to/data --max-rows 0
    python download_and_prepare.py --output-dir /path/to/data --datasets EN_Set1 HI_Set1
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from promptingnemo.data.normalize import normalize_text, should_keep_line
from promptingnemo.data.tag_parser import (
    build_char_vocabulary,
    build_tag_vocabulary,
    decompose_tag,
    is_tag,
    parse_tagged_text,
    strip_tags,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DATASETS = {
    'EN_Set1': 'WhissleAI/Meta_STT_EN_Set1',
    'EURO_Set1': 'WhissleAI/Meta_STT_EURO_Set1',
    'EN_Set2': 'WhissleAI/Meta_STT_EN_Set2',
    'SLAVIC_CV': 'WhissleAI/Meta_STT_SLAVIC_CommonVoice',
    'Medical': 'WhissleAI/speech-simulated-medical-exams',
    'HI_Set1': 'WhissleAI/Meta_STT_HI_Set1',
    'EN_IN_Tech': 'WhissleAI/Meta_STT_EN-IN_Tech_Interviews',
}


def download_dataset(
    name: str,
    repo: str,
    output_dir: str,
    max_rows: int = 0,
):
    """Download one HF dataset and write text-only JSONL manifests."""
    out_path = Path(output_dir) / name
    out_path.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s (%s)...", name, repo)

    try:
        ds = load_dataset(repo, streaming=True, trust_remote_code=True)
    except Exception as e:
        log.error("Failed to load %s: %s", repo, e)
        return {}

    stats = {}
    for split_name in ds.keys():
        out_file = out_path / f'{split_name}.json'
        kept = 0
        dropped = 0
        total = 0

        with open(out_file, 'w', encoding='utf-8') as f:
            for row in ds[split_name]:
                total += 1
                if max_rows and total > max_rows:
                    break

                text = row.get('text', '')
                if not text or not isinstance(text, str):
                    dropped += 1
                    continue

                norm = normalize_text(text)
                clean = strip_tags(norm)
                if not clean.strip():
                    dropped += 1
                    continue

                entry = {
                    'text': norm,
                    'clean_text': clean,
                    'source': name,
                }
                lang = row.get('lang', row.get('language', ''))
                if lang:
                    entry['lang'] = lang

                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                kept += 1

        log.info("  %s/%s: %d kept, %d dropped, %d total", name, split_name, kept, dropped, total)
        stats[split_name] = {'kept': kept, 'dropped': dropped}

    return stats


def merge_manifests(data_dir: str, split: str = 'train'):
    """Merge all per-dataset manifests for a given split into one."""
    data_path = Path(data_dir)
    merged_path = data_path / f'merged_{split}.json'
    count = 0

    with open(merged_path, 'w', encoding='utf-8') as out:
        for ds_dir in sorted(data_path.iterdir()):
            if not ds_dir.is_dir():
                continue
            split_file = ds_dir / f'{split}.json'
            if not split_file.exists():
                # try 'validation' for datasets that use that name
                if split == 'valid':
                    for alt in ['validation', 'val']:
                        alt_file = ds_dir / f'{alt}.json'
                        if alt_file.exists():
                            split_file = alt_file
                            break
                if not split_file.exists():
                    continue

            with open(split_file, encoding='utf-8') as f:
                for line in f:
                    out.write(line)
                    count += 1

    log.info("Merged %s manifest: %d lines -> %s", split, count, merged_path)
    return str(merged_path)


def build_vocabularies(
    data_dir: str,
    max_tag_pieces: int = 10000,
    sp_vocab_size: int = 4096,
):
    """Build character vocabulary and tag vocabulary from merged manifests."""
    data_path = Path(data_dir)
    train_path = data_path / 'merged_train.json'

    if not train_path.exists():
        log.error("merged_train.json not found. Run merge first.")
        return

    # Build tag vocabulary
    log.info("Building tag vocabulary (max %d pieces)...", max_tag_pieces)
    tag_pieces, tag_counts = build_tag_vocabulary(
        [str(train_path)], max_tag_pieces=max_tag_pieces, min_count=5,
    )
    tag_vocab_path = data_path / 'tag_vocab.json'
    with open(tag_vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'tag_pieces': tag_pieces,
            'num_unique_tags': len(tag_counts),
            'num_tag_pieces': len(tag_pieces),
        }, f, indent=2, ensure_ascii=False)
    log.info("Tag vocabulary: %d unique tags -> %d compositional pieces", len(tag_counts), len(tag_pieces))

    # Show tag distribution
    prefix_counts = Counter()
    for tag, count in tag_counts.items():
        prefix = tag.split('_')[0] + '_'
        prefix_counts[prefix] += count
    log.info("Tag distribution by prefix:")
    for prefix, count in sorted(prefix_counts.items(), key=lambda x: -x[1]):
        log.info("  %s: %d", prefix, count)

    # Build character vocabulary
    log.info("Building character vocabulary...")
    chars = build_char_vocabulary([str(train_path)], min_count=10)
    char_vocab_path = data_path / 'char_vocab.json'
    with open(char_vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'special': ['<pad>', '<unk>'],
            'chars': chars,
        }, f, indent=2, ensure_ascii=False)
    log.info("Character vocabulary: %d characters (+ 2 special)", len(chars))

    # Train SentencePiece on clean text
    log.info("Training SentencePiece (vocab_size=%d)...", sp_vocab_size)
    _train_sentencepiece(str(train_path), str(data_path), sp_vocab_size)

    return str(tag_vocab_path), str(char_vocab_path)


def _train_sentencepiece(manifest_path: str, output_dir: str, vocab_size: int):
    """Train a SentencePiece BPE model on clean text from manifest."""
    import sentencepiece as spm
    import tempfile

    clean_text_file = os.path.join(output_dir, 'clean_text_for_sp.txt')
    with open(manifest_path, encoding='utf-8') as fin, \
         open(clean_text_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            entry = json.loads(line)
            clean = entry.get('clean_text', strip_tags(entry.get('text', '')))
            if clean.strip():
                fout.write(clean.strip() + '\n')

    model_prefix = os.path.join(output_dir, 'sp_text_tagger')
    spm.SentencePieceTrainer.Train(
        input=clean_text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        input_sentence_size=5000000,
        shuffle_input_sentence=True,
    )
    log.info("SentencePiece model saved: %s.model", model_prefix)
    os.remove(clean_text_file)


def main():
    parser = argparse.ArgumentParser(description='Download and prepare text tagger data')
    parser.add_argument('--output-dir', required=True, help='Output directory for data')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='Specific dataset keys to download (default: all)')
    parser.add_argument('--max-rows', type=int, default=0,
                        help='Max rows per split (0 = all)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, only build vocabularies')
    parser.add_argument('--max-tag-pieces', type=int, default=10000)
    parser.add_argument('--sp-vocab-size', type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_download:
        selected = args.datasets or list(DATASETS.keys())
        for key in selected:
            if key not in DATASETS:
                log.warning("Unknown dataset key: %s (available: %s)", key, list(DATASETS.keys()))
                continue
            download_dataset(key, DATASETS[key], args.output_dir, args.max_rows)

    # Merge
    for split in ['train', 'valid', 'test']:
        merge_manifests(args.output_dir, split)

    # Build vocabularies
    build_vocabularies(args.output_dir, args.max_tag_pieces, args.sp_vocab_size)

    log.info("Done! Data prepared in %s", args.output_dir)


if __name__ == '__main__':
    main()
