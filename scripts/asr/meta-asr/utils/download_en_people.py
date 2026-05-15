"""
Download PeopleSpeech audio and join with WhissleAI/Meta_STT_EN_Set1 annotations.

Meta_STT_EN_Set1 has text+tags but no audio data — just paths like
    /peoplespeech_audio/train-00769-of-00804_5.flac
which encode PeopleSpeech parquet shard index (769) and row index (5).

This script:
1. Loads Meta_STT_EN_Set1 annotations
2. Groups needed rows by shard index
3. Downloads only the needed PeopleSpeech parquet shards
4. Extracts matching audio rows and creates NeMo manifest

Usage:
    python download_en_people.py \
        --output-dir /mnt/nfs/data/multilingual_v1/raw/en_people \
        --lang ENGLISH
"""
import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TARGET_SR = 16000
FILENAME_RE = re.compile(r"train-(\d+)-of-(\d+)_(\d+)\.flac")


def parse_shard_row(audio_filepath):
    basename = os.path.basename(audio_filepath)
    m = FILENAME_RE.match(basename)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lang", default="ENGLISH")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.json")
    valid_path = os.path.join(args.output_dir, "valid.json")
    if os.path.exists(train_path):
        existing = sum(1 for _ in open(train_path))
        if existing > 0:
            logging.info(f"train.json already exists ({existing} entries), skipping.")
            return

    start_time = time.time()

    logging.info("Loading WhissleAI/Meta_STT_EN_Set1 annotations...")
    shard_row_to_ann = {}
    for split_name in ["train", "valid"]:
        ds = load_dataset("WhissleAI/Meta_STT_EN_Set1", split=split_name)
        logging.info(f"  {split_name}: {len(ds)} samples")
        for item in ds:
            afp = item.get("audio_filepath", "")
            text = item.get("text", "")
            shard_idx, total_shards, row_idx = parse_shard_row(afp)
            if shard_idx is not None and text:
                shard_row_to_ann[(shard_idx, row_idx)] = text.strip()

    logging.info(f"Total annotations with valid shard/row: {len(shard_row_to_ann)}")

    shard_to_rows = defaultdict(dict)
    for (shard_idx, row_idx), text in shard_row_to_ann.items():
        shard_to_rows[shard_idx][row_idx] = text
    needed_shards = sorted(shard_to_rows.keys())
    total_shards_count = 804
    logging.info(f"Need {len(needed_shards)} unique shards (of {total_shards_count})")

    entries = []
    matched = 0
    skipped_duration = 0
    errors = 0

    for shard_num, shard_idx in enumerate(tqdm(needed_shards, desc="Shards")):
        rows_needed = shard_to_rows[shard_idx]
        shard_file = f"clean/train-{shard_idx:05d}-of-{total_shards_count:05d}.parquet"

        try:
            shard_ds = load_dataset(
                "MLCommons/peoples_speech",
                data_files=shard_file,
                split="train",
            )
        except Exception as e:
            logging.error(f"Failed to load shard {shard_idx}: {e}")
            errors += len(rows_needed)
            continue

        for row_idx, text in rows_needed.items():
            if row_idx >= len(shard_ds):
                logging.warning(f"Shard {shard_idx} has {len(shard_ds)} rows, but need row {row_idx}")
                errors += 1
                continue

            item = shard_ds[row_idx]
            audio_data = item.get("audio", {})
            if "array" not in audio_data:
                errors += 1
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
                skipped_duration += 1
                continue

            out_filename = f"en_people_s{shard_idx:05d}_r{row_idx:05d}.wav"
            out_path = os.path.join(audio_dir, out_filename)
            sf.write(out_path, waveform, TARGET_SR)

            entries.append({
                "audio_filepath": os.path.abspath(out_path),
                "text": text,
                "duration": round(duration, 3),
                "lang": args.lang,
            })
            matched += 1

        del shard_ds

        if (shard_num + 1) % 50 == 0:
            logging.info(f"  Progress: {shard_num+1}/{len(needed_shards)} shards, {matched} matched, {errors} errors")

    logging.info(f"Matched: {matched}, skipped (duration): {skipped_duration}, errors: {errors}")

    if not entries:
        logging.warning("No entries matched.")
        for p in [train_path, valid_path]:
            open(p, "w").close()
    else:
        import random
        random.seed(42)
        random.shuffle(entries)
        val_count = max(1, int(len(entries) * args.val_ratio))
        val_entries = entries[:val_count]
        train_entries = entries[val_count:]

        for path, data in [(train_path, train_entries), (valid_path, val_entries)]:
            with open(path, "w", encoding="utf-8") as f:
                for e in data:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            logging.info(f"Wrote {len(data)} entries to {path}")

    elapsed = time.time() - start_time
    info = {
        "dataset": "WhissleAI/Meta_STT_EN_Set1",
        "audio_source": "MLCommons/peoples_speech (clean)",
        "lang": args.lang,
        "splits": {
            "train": len(train_entries) if entries else 0,
            "valid": len(val_entries) if entries else 0,
        },
        "total_valid_samples": len(entries),
        "total_annotations": len(shard_row_to_ann),
        "matched": matched,
        "skipped_duration": skipped_duration,
        "errors": errors,
        "processing_time_minutes": round(elapsed / 60, 1),
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    logging.info(f"Done in {elapsed/60:.1f} minutes. {matched} samples saved.")


if __name__ == "__main__":
    main()
