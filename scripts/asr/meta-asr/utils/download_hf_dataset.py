"""
Generic downloader for any WhissleAI/Meta_STT_* HuggingFace dataset → NeMo manifests.

Handles both:
  - Single-split datasets (auto-splits into train/valid)
  - Multi-split datasets (train/test/valid already present)

Usage:
  python download_hf_dataset.py --dataset WhissleAI/Meta_STT_ZH_AIShell3 --output-dir /mnt/training/data/zh_aishell3 --lang MANDARIN
  python download_hf_dataset.py --dataset WhissleAI/Meta_STT_HI_Set1 --output-dir /mnt/training/data/hi_set1 --lang HINDI --family INDO_ARYAN
"""
import argparse
import json
import logging
import os
import time

import numpy as np
import soundfile as sf
from datasets import load_dataset, get_dataset_split_names
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_SR = 16000


def process_and_save(item, idx, audio_dir, target_sr, lang, dataset_prefix, split_name):
    audio_data = item.get("audio") or item.get("audio_filepath")
    text = item.get("text")

    if audio_data is None or text is None:
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

    if sample_rate != target_sr:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    duration = len(waveform) / sample_rate
    if duration < 0.3 or duration > 30.0:
        return None

    audio_filename = f"{dataset_prefix}_{split_name}_{idx:07d}.wav"
    audio_filepath = os.path.join(audio_dir, audio_filename)

    try:
        sf.write(audio_filepath, waveform, sample_rate)
    except Exception as e:
        logging.error(f"Could not write audio for {split_name}/{idx}: {e}")
        return None

    return {
        "audio_filepath": os.path.abspath(audio_filepath),
        "text": text.strip(),
        "duration": round(duration, 3),
        "lang": lang.upper(),
    }


def process_split(dataset_split, split_name, audio_dir, target_sr, lang, dataset_prefix):
    entries = []
    skipped = 0
    for idx in tqdm(range(len(dataset_split)), desc=f"Processing {split_name}"):
        item = dataset_split[idx]
        entry = process_and_save(item, idx, audio_dir, target_sr, lang, dataset_prefix, split_name)
        if entry:
            entries.append(entry)
        else:
            skipped += 1
    return entries, skipped


def write_manifest(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logging.info(f"Wrote {len(entries)} entries to {path}")


def count_metatags(entries):
    from collections import Counter
    tag_counter = Counter()
    for entry in entries:
        for word in entry["text"].split():
            upper = word.upper()
            if "_" in upper and any(upper.startswith(p) for p in
                                    ["ENTITY_", "INTENT_", "EMOTION_", "GENDER_",
                                     "AGE_", "KEYWORD_", "LANG_", "DIALECT_", "END"]):
                tag_counter[upper] += 1
    return tag_counter


def create_manifests(dataset_name, output_dir, lang, target_sr=16000, val_ratio=0.05):
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    dataset_prefix = dataset_name.split("/")[-1].lower().replace("-", "_")

    start_time = time.time()

    logging.info(f"Checking available splits for '{dataset_name}'...")
    try:
        available_splits = get_dataset_split_names(dataset_name)
    except Exception:
        available_splits = ["train"]
    logging.info(f"Available splits: {available_splits}")

    has_multi_splits = "train" in available_splits and ("test" in available_splits or "valid" in available_splits or "validation" in available_splits)

    all_entries = []
    split_counts = {}

    if has_multi_splits:
        logging.info("Dataset has multiple splits — using them as-is.")

        split_mapping = {
            "train": "train.json",
            "test": "test.json",
            "valid": "valid.json",
            "validation": "valid.json",
        }

        for split_name in available_splits:
            if split_name not in split_mapping:
                logging.info(f"Skipping unknown split: {split_name}")
                continue

            manifest_name = split_mapping[split_name]
            manifest_path = os.path.join(output_dir, manifest_name)

            if os.path.exists(manifest_path):
                existing_count = sum(1 for _ in open(manifest_path))
                logging.info(f"  {manifest_name} already exists ({existing_count} entries), skipping.")
                split_counts[split_name] = existing_count
                continue

            logging.info(f"Loading split '{split_name}'...")
            ds = load_dataset(dataset_name, split=split_name)
            logging.info(f"  {split_name}: {len(ds)} samples")

            entries, skipped = process_split(ds, split_name, audio_dir, target_sr, lang, dataset_prefix)
            write_manifest(entries, manifest_path)
            split_counts[split_name] = len(entries)
            all_entries.extend(entries)

            logging.info(f"  {split_name}: {len(entries)} valid, {skipped} skipped")

    else:
        logging.info("Dataset has single split — auto-splitting into train/valid.")

        logging.info(f"Loading dataset '{dataset_name}' (split=train)...")
        ds = load_dataset(dataset_name, split="train")
        logging.info(f"Dataset loaded: {len(ds)} samples")

        entries, skipped = process_split(ds, "train", audio_dir, target_sr, lang, dataset_prefix)

        import random
        random.seed(42)
        random.shuffle(entries)
        val_count = max(1, int(len(entries) * val_ratio))
        val_entries = entries[:val_count]
        train_entries = entries[val_count:]

        write_manifest(train_entries, os.path.join(output_dir, "train.json"))
        write_manifest(val_entries, os.path.join(output_dir, "valid.json"))

        split_counts = {"train": len(train_entries), "valid": len(val_entries)}
        all_entries = entries

        logging.info(f"Total: {len(entries)} valid, {skipped} skipped")

    elapsed = time.time() - start_time

    tag_counter = count_metatags(all_entries) if all_entries else {}
    if tag_counter:
        logging.info(f"Top metatags: {tag_counter.most_common(20)}")

    info = {
        "dataset": dataset_name,
        "lang": lang,
        "splits": split_counts,
        "total_valid_samples": sum(split_counts.values()),
        "processing_time_seconds": round(elapsed, 1),
    }
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logging.info(f"Dataset info saved to {info_path}")
    logging.info(f"Total processing time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WhissleAI HF dataset and create NeMo manifests")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name (e.g. WhissleAI/Meta_STT_ZH_AIShell3)")
    parser.add_argument("--output-dir", required=True, help="Output directory for audio and manifests")
    parser.add_argument("--lang", required=True, help="Language tag (e.g. MANDARIN, EN, HINDI)")
    parser.add_argument("--family", default=None, help="Language family for tokenizer lookup (e.g. INDO_ARYAN). If set, used as 'lang' in manifest instead of --lang")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Fraction for validation (single-split only)")
    args = parser.parse_args()

    manifest_lang = args.family.upper() if args.family else args.lang
    create_manifests(args.dataset, args.output_dir, manifest_lang, args.sample_rate, args.val_ratio)
