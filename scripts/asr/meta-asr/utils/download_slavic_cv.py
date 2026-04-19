"""
Download Slavic CommonVoice audio and build NeMo manifests from WhissleAI/Meta_STT_SLAVIC_CommonVoice.

Strategy:
  1. Load WhissleAI manifest metadata (has annotated text with entity/intent/emotion tags)
  2. Download audio from fsicoli/common_voice_15_0 for each Slavic language
  3. Match by CommonVoice clip filename
  4. Write NeMo manifests with audio, annotated text, duration, and lang=SLAVIC

Usage:
  python download_slavic_cv.py --output-dir /mnt/training/data/slavic_cv --langs all
  python download_slavic_cv.py --output-dir /mnt/training/data/slavic_cv --langs be,ru,pl --max-per-lang 5000
"""
import argparse
import io
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_SR = 16000
ALL_SLAVIC_LANGS = ["be", "ru", "pl", "ka", "uk", "cs", "sk", "sl", "bg", "sr", "mk"]

WHISSLE_DATASET = "WhissleAI/Meta_STT_SLAVIC_CommonVoice"
CV_DATASET = "fsicoli/common_voice_15_0"


def extract_cv_filename(audio_filepath: str) -> str:
    return os.path.basename(audio_filepath)


def build_whissle_index(split: str):
    from datasets import load_dataset

    logging.info(f"Loading WhissleAI manifest split={split}...")
    ds = load_dataset(WHISSLE_DATASET, split=split)
    logging.info(f"  {len(ds):,} samples")

    index = {}
    for item in ds:
        filename = extract_cv_filename(item["audio_filepath"])
        lang_match = re.search(r"/cv/cv-corpus-[^/]+/(\w+)/", item["audio_filepath"])
        lang = lang_match.group(1) if lang_match else "unknown"
        index[filename] = {
            "text": item["text"],
            "duration": item["duration"],
            "lang_code": lang,
            "split": split,
        }
    return index


def download_and_match(lang, whissle_index, audio_dir, max_samples=None):
    from datasets import load_dataset

    logging.info(f"Downloading audio for lang={lang} from {CV_DATASET}...")
    try:
        ds = load_dataset(CV_DATASET, lang, split="train", trust_remote_code=True, streaming=True)
    except Exception as e:
        logging.error(f"Failed to load {lang} train split: {e}")
        return []

    entries = []
    matched = 0
    skipped_no_match = 0
    skipped_audio_err = 0

    for item in tqdm(ds, desc=f"{lang}"):
        if max_samples and matched >= max_samples:
            break

        audio_path = item.get("path", "")
        cv_filename = os.path.basename(audio_path)

        if cv_filename not in whissle_index:
            skipped_no_match += 1
            continue

        meta = whissle_index[cv_filename]
        audio_data = item.get("audio")
        if not audio_data or "array" not in audio_data:
            skipped_audio_err += 1
            continue

        waveform = np.array(audio_data["array"], dtype=np.float32)
        sample_rate = audio_data["sampling_rate"]

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if sample_rate != TARGET_SR:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)

        duration = len(waveform) / TARGET_SR
        if duration < 0.3 or duration > 30.0:
            continue

        out_filename = cv_filename.replace(".mp3", ".wav")
        out_path = os.path.join(audio_dir, out_filename)

        try:
            sf.write(out_path, waveform, TARGET_SR)
        except Exception as e:
            logging.error(f"Write error {out_filename}: {e}")
            skipped_audio_err += 1
            continue

        entries.append({
            "audio_filepath": os.path.abspath(out_path),
            "text": meta["text"],
            "duration": round(duration, 3),
            "lang": "SLAVIC",
        })
        matched += 1

    logging.info(f"  {lang}: matched={matched}, no_match={skipped_no_match}, audio_err={skipped_audio_err}")
    return entries


def download_cv_for_split(lang, split_name, whissle_index, audio_dir, max_samples=None):
    """Download from a specific CV split (train/test/validation)."""
    from datasets import load_dataset

    cv_split = "validation" if split_name == "valid" else split_name
    try:
        ds = load_dataset(CV_DATASET, lang, split=cv_split, trust_remote_code=True, streaming=True)
    except Exception:
        return []

    entries = []
    matched = 0

    for item in ds:
        if max_samples and matched >= max_samples:
            break

        cv_filename = os.path.basename(item.get("path", ""))
        if cv_filename not in whissle_index:
            continue

        meta = whissle_index[cv_filename]
        audio_data = item.get("audio")
        if not audio_data or "array" not in audio_data:
            continue

        waveform = np.array(audio_data["array"], dtype=np.float32)
        sample_rate = audio_data["sampling_rate"]
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != TARGET_SR:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)

        duration = len(waveform) / TARGET_SR
        if duration < 0.3 or duration > 30.0:
            continue

        out_filename = cv_filename.replace(".mp3", ".wav")
        out_path = os.path.join(audio_dir, out_filename)
        try:
            sf.write(out_path, waveform, TARGET_SR)
        except Exception:
            continue

        entries.append({
            "audio_filepath": os.path.abspath(out_path),
            "text": meta["text"],
            "duration": round(duration, 3),
            "lang": "SLAVIC",
        })
        matched += 1

    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--langs", default="all", help="Comma-separated lang codes or 'all'")
    parser.add_argument("--max-per-lang", type=int, default=None, help="Max samples per language (for testing)")
    args = parser.parse_args()

    langs = ALL_SLAVIC_LANGS if args.langs == "all" else [l.strip() for l in args.langs.split(",")]
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    start_time = time.time()

    # Build index from all WhissleAI splits
    whissle_index = {}
    for split in ["train", "valid", "test"]:
        idx = build_whissle_index(split)
        for k, v in idx.items():
            whissle_index[k] = v
    logging.info(f"WhissleAI index: {len(whissle_index):,} total entries")

    lang_counts = Counter()
    for v in whissle_index.values():
        lang_counts[v["lang_code"]] += 1
    logging.info(f"WhissleAI languages: {dict(lang_counts.most_common())}")

    # Download audio and match, grouped by WhissleAI split
    split_entries = defaultdict(list)
    for lang in langs:
        if lang not in [v["lang_code"] for v in whissle_index.values()]:
            logging.warning(f"Lang {lang} not found in WhissleAI index, skipping")
            continue

        # Filter index to this lang
        lang_index = {k: v for k, v in whissle_index.items() if v["lang_code"] == lang}
        logging.info(f"Processing {lang}: {len(lang_index):,} entries in WhissleAI index")

        # Download from all CV splits
        for cv_split in ["train", "test", "validation"]:
            whissle_split = "valid" if cv_split == "validation" else cv_split
            split_index = {k: v for k, v in lang_index.items() if v["split"] == whissle_split}
            if not split_index:
                # Try matching from train CV split for all WhissleAI splits
                continue

            entries = download_cv_for_split(lang, whissle_split, split_index, audio_dir, args.max_per_lang)
            split_entries[whissle_split].extend(entries)
            logging.info(f"  {lang}/{whissle_split}: {len(entries)} matched")

        # Also try matching unmatched entries from CV train split
        remaining = {k: v for k, v in lang_index.items()
                     if k not in {os.path.basename(e["audio_filepath"]).replace(".wav", ".mp3")
                                  for entries in split_entries.values() for e in entries}}
        if remaining:
            entries = download_and_match(lang, remaining, audio_dir, args.max_per_lang)
            for e in entries:
                orig_split = whissle_index.get(
                    os.path.basename(e["audio_filepath"]).replace(".wav", ".mp3"), {}
                ).get("split", "train")
                split_entries[orig_split].append(e)

    # Write manifests
    for split_name, entries in split_entries.items():
        if not entries:
            continue
        manifest_path = os.path.join(args.output_dir, f"{split_name}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logging.info(f"Wrote {len(entries):,} entries to {manifest_path}")

    elapsed = time.time() - start_time

    info = {
        "dataset": WHISSLE_DATASET,
        "cv_source": CV_DATASET,
        "langs": langs,
        "splits": {k: len(v) for k, v in split_entries.items()},
        "total_samples": sum(len(v) for v in split_entries.values()),
        "processing_time_seconds": round(elapsed, 1),
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logging.info(f"Done in {elapsed/60:.1f} minutes. Total: {info['total_samples']:,} samples")


if __name__ == "__main__":
    main()
