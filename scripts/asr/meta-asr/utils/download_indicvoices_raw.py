"""
Download raw IndicVoices / Kathbath / IndicVoices-R parquet data for a single
language, extract WAVs at 16 kHz, and produce NeMo-style JSONL manifests.

Handles datasets that store audio in parquet columns (decoded via HF datasets)
and preserves metadata columns (age_group, gender, state, task_name) when
available.

Usage:
  # IndicVoices Gujarati
  python download_indicvoices_raw.py \
      --dataset ai4bharat/IndicVoices --lang gujarati \
      --output-dir /mnt/nfs/data/gujarati_v1/raw/indicvoices

  # IndicVoices-R Gujarati (has train/test splits)
  python download_indicvoices_raw.py \
      --dataset ai4bharat/indicvoices_r --lang Gujarati \
      --output-dir /mnt/nfs/data/gujarati_v1/raw/indicvoices_r

  # Kathbath Gujarati
  python download_indicvoices_raw.py \
      --dataset ai4bharat/Kathbath --lang gujarati \
      --output-dir /mnt/nfs/data/gujarati_v1/raw/kathbath

  # FLEURS Gujarati
  python download_indicvoices_raw.py \
      --dataset google/fleurs --lang gu_in \
      --output-dir /mnt/nfs/data/gujarati_v1/raw/fleurs
"""
import argparse
import json
import logging
import os
import time

import numpy as np
import soundfile as sf
from datasets import load_dataset, get_dataset_split_names, Audio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 16000

METADATA_COLUMNS = ["age_group", "gender", "state", "task_name", "speaker_id"]

# Column name that holds the audio — varies by dataset
AUDIO_COL_CANDIDATES = ["audio", "audio_filepath", "path"]
TEXT_COL_CANDIDATES = ["text", "sentence", "transcription", "raw_text"]


def find_column(example, candidates):
    for c in candidates:
        if c in example:
            return c
    return None


def process_example(example, idx, audio_dir, split_name, prefix, audio_col, text_col):
    audio_data = example.get(audio_col)
    text = example.get(text_col, "")

    if audio_data is None or not text:
        return None

    if isinstance(audio_data, dict):
        if "array" in audio_data and audio_data["array"] is not None:
            waveform = np.array(audio_data["array"], dtype=np.float32)
            sr = audio_data.get("sampling_rate", TARGET_SR)
        elif "bytes" in audio_data and audio_data["bytes"]:
            import io
            try:
                waveform, sr = sf.read(io.BytesIO(audio_data["bytes"]))
                waveform = np.array(waveform, dtype=np.float32)
            except Exception as e:
                logger.debug("Cannot read audio bytes for %s/%d: %s", split_name, idx, e)
                return None
        elif "path" in audio_data and audio_data["path"]:
            try:
                waveform, sr = sf.read(audio_data["path"])
                waveform = np.array(waveform, dtype=np.float32)
            except Exception:
                return None
        else:
            return None
    elif isinstance(audio_data, str):
        try:
            waveform, sr = sf.read(audio_data)
            waveform = np.array(waveform, dtype=np.float32)
        except Exception:
            return None
    else:
        return None

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sr != TARGET_SR:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

    duration = len(waveform) / TARGET_SR
    if duration < 0.3 or duration > 30.0:
        return None

    filename = f"{prefix}_{split_name}_{idx:07d}.wav"
    filepath = os.path.join(audio_dir, filename)

    if not os.path.exists(filepath):
        try:
            sf.write(filepath, waveform, TARGET_SR)
        except Exception as e:
            logger.error("Failed to write %s: %s", filepath, e)
            return None

    record = {
        "audio_filepath": os.path.abspath(filepath),
        "text": text.strip(),
        "duration": round(duration, 3),
    }

    for col in METADATA_COLUMNS:
        val = example.get(col)
        if val is not None:
            record[col] = val

    return record


def process_split(ds_split, split_name, audio_dir, prefix, audio_col, text_col):
    entries = []
    skipped = 0
    for idx in tqdm(range(len(ds_split)), desc=f"Processing {split_name}"):
        try:
            example = ds_split[idx]
        except Exception as e:
            logger.debug("Error reading %s[%d]: %s", split_name, idx, e)
            skipped += 1
            continue
        entry = process_example(example, idx, audio_dir, split_name, prefix, audio_col, text_col)
        if entry:
            entries.append(entry)
        else:
            skipped += 1
    logger.info("%s: %d valid, %d skipped", split_name, len(entries), skipped)
    return entries


def write_manifest(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Wrote %d entries to %s", len(entries), path)


def main():
    parser = argparse.ArgumentParser(description="Download and extract Indic speech datasets to NeMo manifests")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset ID")
    parser.add_argument("--lang", required=True, help="Language/config name (e.g. gujarati, Gujarati, gu_in)")
    parser.add_argument("--output-dir", required=True, help="Output directory for audio + manifests")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio (single-split only)")
    parser.add_argument("--audio-col", default=None, help="Audio column name override")
    parser.add_argument("--text-col", default=None, help="Text column name override")
    args = parser.parse_args()

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    prefix = args.dataset.split("/")[-1].lower().replace("-", "_")
    start = time.time()

    logger.info("Loading dataset %s, config/lang=%s", args.dataset, args.lang)

    try:
        available_splits = get_dataset_split_names(args.dataset, args.lang, trust_remote_code=True)
    except Exception as e:
        logger.warning("get_dataset_split_names with lang=%s failed: %s", args.lang, e)
        try:
            available_splits = get_dataset_split_names(args.dataset, trust_remote_code=True)
        except Exception:
            available_splits = ["train"]

    logger.info("Available splits: %s", available_splits)

    has_train_test = "train" in available_splits and ("test" in available_splits or "validation" in available_splits)

    # Detect column names from first example
    probe_split = available_splits[0]
    try:
        probe_ds = load_dataset(args.dataset, args.lang, split=probe_split, streaming=True, trust_remote_code=True)
        probe_example = next(iter(probe_ds))
    except Exception as e:
        logger.warning("load_dataset with lang=%s failed: %s — retrying without lang", args.lang, e)
        probe_ds = load_dataset(args.dataset, split=probe_split, streaming=True, trust_remote_code=True)
        probe_example = next(iter(probe_ds))

    audio_col = args.audio_col or find_column(probe_example, AUDIO_COL_CANDIDATES)
    text_col = args.text_col or find_column(probe_example, TEXT_COL_CANDIDATES)

    if not audio_col or not text_col:
        logger.error("Cannot detect audio/text columns. Available: %s", list(probe_example.keys()))
        logger.error("Use --audio-col and --text-col to specify manually.")
        return

    logger.info("Using columns: audio=%s, text=%s", audio_col, text_col)
    logger.info("Available metadata columns: %s", [c for c in METADATA_COLUMNS if c in probe_example])

    split_map = {"train": "train.json", "test": "test.json", "valid": "valid.json", "validation": "valid.json"}

    if has_train_test:
        for split_name in available_splits:
            manifest_name = split_map.get(split_name)
            if not manifest_name:
                continue
            manifest_path = os.path.join(args.output_dir, manifest_name)
            if os.path.exists(manifest_path):
                count = sum(1 for _ in open(manifest_path))
                logger.info("%s already exists (%d entries), skipping.", manifest_name, count)
                continue

            logger.info("Loading split %s...", split_name)
            try:
                ds = load_dataset(args.dataset, args.lang, split=split_name, trust_remote_code=True)
            except Exception as e:
                logger.warning("load_dataset %s/%s with lang=%s failed: %s", args.dataset, split_name, args.lang, e)
                ds = load_dataset(args.dataset, split=split_name, trust_remote_code=True)

            if audio_col in ds.column_names:
                ds = ds.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))

            entries = process_split(ds, split_name, audio_dir, prefix, audio_col, text_col)
            write_manifest(entries, manifest_path)
    else:
        manifest_path = os.path.join(args.output_dir, "train.json")
        if os.path.exists(manifest_path):
            count = sum(1 for _ in open(manifest_path))
            logger.info("train.json already exists (%d entries), skipping.", count)
            return

        logger.info("Loading single split (will auto-split train/valid)...")
        try:
            ds = load_dataset(args.dataset, args.lang, split="train", trust_remote_code=True)
        except Exception as e:
            logger.warning("load_dataset single split with lang=%s failed: %s", args.lang, e)
            ds = load_dataset(args.dataset, split="train", trust_remote_code=True)

        if audio_col in ds.column_names:
            ds = ds.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))

        entries = process_split(ds, "train", audio_dir, prefix, audio_col, text_col)

        import random
        random.seed(42)
        random.shuffle(entries)
        val_count = max(1, int(len(entries) * args.val_ratio))
        val_entries = entries[:val_count]
        train_entries = entries[val_count:]

        write_manifest(train_entries, os.path.join(args.output_dir, "train.json"))
        write_manifest(val_entries, os.path.join(args.output_dir, "valid.json"))

    elapsed = time.time() - start
    info = {
        "dataset": args.dataset,
        "lang": args.lang,
        "processing_time_minutes": round(elapsed / 60, 1),
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logger.info("Done in %.1f minutes.", elapsed / 60)


if __name__ == "__main__":
    main()
