"""
Download Chinese subset of Voices-in-the-Wild-2M from HuggingFace → NeMo manifests.

Filters to Chinese-language samples (CJK characters in transcription),
saves audio as 16kHz WAV, and creates train/valid manifest splits.

Usage:
  python download_vitw_zh.py \
    --output-dir /mnt/nfs/data/vitw_zh \
    --val-ratio 0.05
"""
import argparse
import json
import logging
import os
import random
import re
import time

import numpy as np
import soundfile as sf
from datasets import load_dataset, get_dataset_split_names
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_SR = 16000
CJK_RE = re.compile(r'[一-鿿]')


def is_chinese(text: str) -> bool:
    return bool(CJK_RE.search(text))


def process_and_save(item, idx, audio_dir, split_name):
    text = item.get("answer") or item.get("text") or ""
    text = text.strip()
    if not text or not is_chinese(text):
        return None

    audio_data = item.get("audio")
    if audio_data is None:
        return None

    if isinstance(audio_data, dict):
        if "array" in audio_data:
            waveform = np.array(audio_data["array"], dtype=np.float32)
            sample_rate = audio_data["sampling_rate"]
        elif "bytes" in audio_data and audio_data["bytes"] is not None:
            import io
            try:
                waveform, sample_rate = sf.read(io.BytesIO(audio_data["bytes"]))
                waveform = np.array(waveform, dtype=np.float32)
            except Exception as e:
                logging.error(f"Could not read audio bytes for {split_name}/{idx}: {e}")
                return None
        elif "path" in audio_data and audio_data.get("path"):
            try:
                waveform, sample_rate = sf.read(audio_data["path"])
                waveform = np.array(waveform, dtype=np.float32)
            except Exception as e:
                logging.error(f"Could not read audio path for {split_name}/{idx}: {e}")
                return None
        else:
            return None
    else:
        return None

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sample_rate != TARGET_SR:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
        sample_rate = TARGET_SR

    duration = len(waveform) / sample_rate
    if duration < 0.3 or duration > 30.0:
        return None

    audio_filename = f"vitw_zh_{split_name}_{idx:07d}.wav"
    audio_filepath = os.path.join(audio_dir, audio_filename)

    try:
        sf.write(audio_filepath, waveform, sample_rate)
    except Exception as e:
        logging.error(f"Could not write audio for {split_name}/{idx}: {e}")
        return None

    return {
        "audio_filepath": os.path.abspath(audio_filepath),
        "text": text,
        "duration": round(duration, 3),
        "lang": "MANDARIN",
        "source": "voices_in_the_wild",
    }


def write_manifest(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logging.info(f"Wrote {len(entries)} entries to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Chinese subset of Voices-in-the-Wild-2M → NeMo manifests"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory on NFS")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    start_time = time.time()

    checkpoint_path = os.path.join(args.output_dir, "download_progress.json")
    processed_subsets = set()
    all_entries = []

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            progress = json.load(f)
            processed_subsets = set(progress.get("completed_subsets", []))
            partial_manifest = os.path.join(args.output_dir, "all_zh.json")
            if os.path.exists(partial_manifest):
                with open(partial_manifest, encoding="utf-8") as mf:
                    all_entries = [json.loads(line) for line in mf if line.strip()]
        logging.info(f"Resuming: {len(processed_subsets)} subsets done, {len(all_entries)} entries so far")

    logging.info("Loading dataset splits...")
    try:
        available_splits = get_dataset_split_names("zhifeixie/Voices-in-the-Wild-2M")
    except Exception:
        available_splits = ["train"]
    logging.info(f"Available splits: {len(available_splits)} — {available_splits[:5]}...")

    for split_name in available_splits:
        if split_name in processed_subsets:
            logging.info(f"  Skipping {split_name} (already done)")
            continue

        logging.info(f"Loading split '{split_name}'...")
        try:
            ds = load_dataset("zhifeixie/Voices-in-the-Wild-2M", split=split_name)
        except Exception as e:
            logging.error(f"  Failed to load split {split_name}: {e}")
            continue

        logging.info(f"  {split_name}: {len(ds)} total samples, filtering to Chinese...")

        split_entries = []
        skipped = 0
        for idx in tqdm(range(len(ds)), desc=f"Processing {split_name}"):
            entry = process_and_save(ds[idx], idx, audio_dir, split_name)
            if entry:
                split_entries.append(entry)
            else:
                skipped += 1

        all_entries.extend(split_entries)
        processed_subsets.add(split_name)

        logging.info(f"  {split_name}: {len(split_entries)} Chinese, {skipped} skipped")

        with open(os.path.join(args.output_dir, "all_zh.json"), "w", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with open(checkpoint_path, "w") as f:
            json.dump({"completed_subsets": list(processed_subsets)}, f)

    logging.info(f"Total Chinese entries: {len(all_entries)}")

    random.seed(42)
    random.shuffle(all_entries)
    val_count = max(1, int(len(all_entries) * args.val_ratio))
    val_entries = all_entries[:val_count]
    train_entries = all_entries[val_count:]

    write_manifest(train_entries, os.path.join(args.output_dir, "train.json"))
    write_manifest(val_entries, os.path.join(args.output_dir, "valid.json"))

    elapsed = time.time() - start_time

    info = {
        "dataset": "zhifeixie/Voices-in-the-Wild-2M",
        "filter": "chinese_only",
        "lang": "MANDARIN",
        "splits": {"train": len(train_entries), "valid": len(val_entries)},
        "total_valid_samples": len(all_entries),
        "processing_time_seconds": round(elapsed, 1),
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    temp_manifest = os.path.join(args.output_dir, "all_zh.json")
    if os.path.exists(temp_manifest):
        os.remove(temp_manifest)

    logging.info(f"Done in {elapsed/60:.1f} min — train: {len(train_entries)}, valid: {len(val_entries)}")


if __name__ == "__main__":
    main()
