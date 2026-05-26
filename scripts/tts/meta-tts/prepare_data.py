#!/usr/bin/env python3
"""
Prepare F5-TTS training dataset from WhissleAI/Meta_STT_EURO_Set1.

Reads the HuggingFace parquet dataset, resolves audio file paths to NFS
locations, extracts speaker IDs, rearranges metadata tags for TTS
(tags → front of text), and creates an F5-TTS compatible Arrow dataset.

Output structure:
    <output_dir>/
    ├── raw/              # HF Arrow dataset with (audio_path, text) columns
    ├── duration.json     # {"duration": [float, ...]}
    └── vocab.txt         # Custom vocabulary with metadata + speaker tokens

Usage:
    python prepare_data.py \
        --audio-root /mnt/nfs/data/tts_audio \
        --output-dir /mnt/nfs/data/meta_tts_euro \
        --split train \
        --workers 16
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torchaudio

TAG_PATTERN = re.compile(
    r'\s+(AGE_\S+|GER_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+|DIALECT_\S+|DOMAIN_\S+)\s*'
)

TRAILING_TAGS_RE = re.compile(
    r'(\s+(?:AGE_\S+|GER_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+|DIALECT_\S+|DOMAIN_\S+))+\s*$'
)

AUDIO_PATH_MAP = {
    "/cv/": "cv/",
    "/librespeech-en/": "librispeech-en/",
    "/librispeech-en/": "librispeech-en/",
    "/mls/": "mls/",
    "/peoples_speech/": "peoples_speech/",
}

KNOWN_META_TOKENS = [
    "AGE_CHILD", "AGE_14_18", "AGE_18_30", "AGE_30_45", "AGE_45_60", "AGE_60PLUS",
    "AGE_<14", "AGE_>41", "AGE_20_30", "AGE_26_40",
    "GER_MALE", "GER_FEMALE",
    "GENDER_MALE", "GENDER_FEMALE",
    "EMOTION_NEUTRAL", "EMOTION_NEU", "EMOTION_HAP", "EMOTION_SAD",
    "EMOTION_ANG", "EMOTION_FEAR", "EMOTION_DISGUST", "EMOTION_SURPRISE",
    "EMOTION_FEA", "EMOTION_DIS", "EMOTION_SUR",
    "INTENT_INFORM", "INTENT_QUERY", "INTENT_COMMAND", "INTENT_REQUEST",
    "INTENT_GREETING", "INTENT_DESCRIBE", "INTENT_EXPLAIN", "INTENT_STATEMENT",
    "INTENT_QUESTION", "INTENT_ANNOUNCE",
    "DIALECT_NORTH", "DIALECT_SOUTH", "DIALECT_OTHER",
]


def resolve_audio_path(audio_filepath, audio_root):
    """Resolve dataset audio_filepath to absolute NFS path."""
    for prefix, mapped in AUDIO_PATH_MAP.items():
        if audio_filepath.startswith(prefix):
            rel = audio_filepath[len(prefix):]
            return os.path.join(audio_root, mapped, rel)

    return os.path.join(audio_root, audio_filepath.lstrip("/"))


def extract_speaker_id(audio_filepath, source):
    """Extract speaker ID from audio file path based on source dataset."""
    parts = Path(audio_filepath).parts

    if "commonvoice" in source:
        lang = source.split("_")[-1] if "_" in source else "unk"
        filename = Path(audio_filepath).stem
        spk = filename.split("_")[-1] if "_" in filename else filename[:8]
        return f"SPK_cv_{lang}_{spk}"

    if "librispeech" in source or "librespeech" in source:
        for i, part in enumerate(parts):
            if part in ("train-other-500", "train-clean-360", "train-clean-100",
                        "dev-clean", "dev-other", "test-clean", "test-other"):
                if i + 1 < len(parts):
                    return f"SPK_ls_{parts[i + 1]}"
        return f"SPK_ls_{Path(audio_filepath).stem[:4]}"

    if "mls" in source:
        for i, part in enumerate(parts):
            if part == "audio":
                if i + 1 < len(parts):
                    lang = source.split("_")[-1] if "_" in source else "unk"
                    return f"SPK_mls_{lang}_{parts[i + 1]}"
        return f"SPK_mls_{Path(audio_filepath).stem[:6]}"

    if "peoples" in source:
        return f"SPK_ps_{Path(audio_filepath).stem[:8]}"

    return f"SPK_unk_{Path(audio_filepath).stem[:8]}"


def parse_tags(text):
    """Extract metadata tags from text and return (clean_text, tags_list)."""
    tags = []
    m = TRAILING_TAGS_RE.search(text)
    if m:
        tag_str = m.group(0).strip()
        clean_text = text[:m.start()].strip()
        tags = tag_str.split()
    else:
        clean_text = text.strip()

    return clean_text, tags


def rearrange_for_tts(clean_text, tags, speaker_id):
    """Rearrange text with speaker ID and tags at front for TTS."""
    parts = [speaker_id] + tags + [clean_text]
    return " ".join(parts)


def verify_audio(args):
    """Verify audio file exists and get duration. For parallel processing."""
    audio_path, idx = args
    try:
        if not os.path.exists(audio_path):
            return idx, None, "not_found"
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate
        if duration < 0.3 or duration > 30.0:
            return idx, None, "duration_out_of_range"
        return idx, duration, "ok"
    except Exception as e:
        return idx, None, f"error: {e}"


def process_split(
    split,
    audio_root,
    output_dir,
    max_samples=None,
    languages=None,
    workers=8,
):
    """Process one split of Meta_STT_EURO_Set1."""
    from datasets import load_dataset as hf_load_dataset
    from datasets import Dataset as HFDataset

    print(f"\n{'=' * 60}")
    print(f"Processing split: {split}")
    print(f"{'=' * 60}")

    print("Loading Meta_STT_EURO_Set1 from HuggingFace...")
    ds = hf_load_dataset("WhissleAI/Meta_STT_EURO_Set1", split=split)
    print(f"  Loaded {len(ds)} samples")

    if languages:
        lang_set = set(languages)
        source_lang_map = {
            "commonvoice_en": "en", "commonvoice_de": "de", "commonvoice_es": "es",
            "commonvoice_fr": "fr", "commonvoice_it": "it", "commonvoice_pt": "pt",
            "librispeech_en": "en", "mls_en": "en", "mls_de": "de", "mls_es": "es",
            "mls_fr": "fr", "mls_it": "it", "mls_pt": "pt",
            "peoples_speech_en": "en",
        }

        def lang_filter(example):
            source = example.get("source", "")
            lang = source_lang_map.get(source, source.split("_")[-1])
            return lang in lang_set

        ds = ds.filter(lang_filter, desc="Filtering languages")
        print(f"  After language filter: {len(ds)} samples")

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))
        print(f"  Limited to {max_samples} samples")

    print("\nResolving audio paths...")
    audio_paths = []
    texts = []
    speaker_ids = []
    sources = []
    tag_counts = Counter()
    spk_counts = Counter()

    for i, sample in enumerate(ds):
        audio_filepath = sample["audio_filepath"]
        text = sample["text"]
        source = sample.get("source", "unknown")
        abs_path = resolve_audio_path(audio_filepath, audio_root)

        spk_id = extract_speaker_id(audio_filepath, source)

        clean_text, tags = parse_tags(text)
        for tag in tags:
            tag_counts[tag] += 1

        tts_text = rearrange_for_tts(clean_text, tags, spk_id)

        audio_paths.append(abs_path)
        texts.append(tts_text)
        speaker_ids.append(spk_id)
        sources.append(source)
        spk_counts[spk_id] += 1

        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1}/{len(ds)} paths")

    print(f"\nVerifying audio files exist ({workers} workers)...")
    verify_args = [(p, i) for i, p in enumerate(audio_paths)]
    valid_indices = []
    durations = []
    skip_reasons = Counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(verify_audio, arg): arg for arg in verify_args}
        done = 0
        for future in as_completed(futures):
            idx, duration, status = future.result()
            done += 1
            if status == "ok":
                valid_indices.append(idx)
                durations.append((idx, duration))
            else:
                skip_reasons[status] += 1

            if done % 50000 == 0:
                print(f"  Verified {done}/{len(verify_args)} "
                      f"({len(valid_indices)} valid, {done - len(valid_indices)} skipped)")

    valid_indices.sort()
    durations.sort(key=lambda x: x[0])

    print(f"\n  Valid samples: {len(valid_indices)}/{len(audio_paths)}")
    for reason, count in skip_reasons.most_common():
        print(f"    Skipped ({reason}): {count}")

    print("\nBuilding Arrow dataset...")
    valid_audio = [audio_paths[i] for i in valid_indices]
    valid_text = [texts[i] for i in valid_indices]
    valid_durations = [d for _, d in durations]

    arrow_ds = HFDataset.from_dict({
        "audio_path": valid_audio,
        "text": valid_text,
    })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrow_ds.save_to_disk(str(out / "raw"))
    print(f"  Saved Arrow dataset to {out / 'raw'}")

    with open(out / "duration.json", "w") as f:
        json.dump({"duration": valid_durations}, f)
    print(f"  Saved duration.json ({len(valid_durations)} entries)")

    print("\nBuilding vocabulary...")
    all_chars = set()
    all_meta_tokens = set()
    all_spk_tokens = set()

    for text in valid_text:
        parts = text.split()
        for part in parts:
            if part.startswith("SPK_"):
                all_spk_tokens.add(part)
            elif re.match(r'^(AGE_|GER_|GENDER_|EMOTION_|INTENT_|DIALECT_|DOMAIN_)', part):
                all_meta_tokens.add(part)
            else:
                for ch in part:
                    all_chars.add(ch)

    all_meta_tokens.update(KNOWN_META_TOKENS)

    vocab = OrderedDict()
    vocab[" "] = 0

    for ch in sorted(all_chars):
        if ch != " ":
            vocab[ch] = len(vocab)

    for token in sorted(all_meta_tokens):
        vocab[token] = len(vocab)

    for token in sorted(all_spk_tokens):
        vocab[token] = len(vocab)

    vocab_path = out / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

    print(f"  Vocabulary size: {len(vocab)}")
    print(f"    Characters: {len(all_chars)}")
    print(f"    Metadata tokens: {len(all_meta_tokens)}")
    print(f"    Speaker tokens: {len(all_spk_tokens)}")
    print(f"  Saved to {vocab_path}")

    print(f"\nTag distribution (top 20):")
    for tag, count in tag_counts.most_common(20):
        print(f"  {tag}: {count:,}")

    print(f"\nSource distribution:")
    source_counts = Counter(sources[i] for i in valid_indices)
    for source, count in source_counts.most_common():
        print(f"  {source}: {count:,}")

    print(f"\nSample outputs:")
    for i in range(min(5, len(valid_text))):
        print(f"  [{i}] audio: {valid_audio[i][:80]}...")
        print(f"       text:  {valid_text[i][:120]}...")
        print(f"       dur:   {valid_durations[i]:.2f}s")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare F5-TTS dataset from Meta_STT_EURO_Set1"
    )
    parser.add_argument(
        "--audio-root", required=True,
        help="Root directory where audio was downloaded"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for F5-TTS dataset"
    )
    parser.add_argument(
        "--split", default="train",
        choices=["train", "valid", "test"],
        help="Dataset split to process"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples (for testing)"
    )
    parser.add_argument(
        "--languages", default=None,
        help="Comma-separated language codes to include (default: all)"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel workers for audio verification"
    )
    args = parser.parse_args()

    languages = args.languages.split(",") if args.languages else None

    process_split(
        split=args.split,
        audio_root=args.audio_root,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        languages=languages,
        workers=args.workers,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
