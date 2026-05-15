"""
Download English digit/number speech datasets for ASR training:
  1. Google Speech Commands v2 — 105k 1-second keyword utterances (includes 0-9)
  2. Free Spoken Digit Dataset (FSDD) — 3k digit recordings (0-9)

Extracts WAVs at 16 kHz and produces NeMo-style JSONL manifests.

Usage:
  python download_en_digits.py \
      --output-dir /mnt/nfs/data/english_v1/raw/digits
"""
import argparse
import json
import logging
import os
import random
import time

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 16000

DIGIT_WORDS = {
    "zero": "zero", "one": "one", "two": "two", "three": "three",
    "four": "four", "five": "five", "six": "six", "seven": "seven",
    "eight": "eight", "nine": "nine",
}

SPEECH_COMMANDS_KEYWORDS = {
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
    "backward", "forward", "follow", "learn", "visual",
}


def process_speech_commands(output_dir):
    audio_dir = os.path.join(output_dir, "speech_commands", "audio")
    os.makedirs(audio_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "speech_commands", "train.json")
    if os.path.exists(manifest_path):
        count = sum(1 for _ in open(manifest_path))
        logger.info("speech_commands/train.json already exists (%d entries), skipping.", count)
        return

    logger.info("Loading Google Speech Commands v2...")
    ds = load_dataset("google/speech_commands", "v0.02", split="train", trust_remote_code=True)
    logger.info("  %d samples in train split", len(ds))

    entries = []
    skipped = 0
    for idx in tqdm(range(len(ds)), desc="Speech Commands"):
        example = ds[idx]
        label = example.get("label")
        if isinstance(label, int):
            label = ds.features["label"].int2str(label)

        if label not in SPEECH_COMMANDS_KEYWORDS:
            skipped += 1
            continue

        audio = example.get("audio", {})
        if isinstance(audio, dict) and "array" in audio:
            waveform = np.array(audio["array"], dtype=np.float32)
            sr = audio.get("sampling_rate", TARGET_SR)
        else:
            skipped += 1
            continue

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if sr != TARGET_SR:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

        duration = len(waveform) / TARGET_SR
        if duration < 0.1 or duration > 5.0:
            skipped += 1
            continue

        filename = f"sc_{label}_{idx:06d}.wav"
        filepath = os.path.join(audio_dir, filename)
        if not os.path.exists(filepath):
            sf.write(filepath, waveform, TARGET_SR)

        entries.append({
            "audio_filepath": os.path.abspath(filepath),
            "text": label,
            "duration": round(duration, 3),
            "lang": "ENGLISH",
            "source": "speech_commands",
        })

    logger.info("Speech Commands: %d valid, %d skipped", len(entries), skipped)

    random.seed(42)
    random.shuffle(entries)
    val_count = max(1, int(len(entries) * 0.05))
    val_entries = entries[:val_count]
    train_entries = entries[val_count:]

    write_manifest(train_entries, manifest_path)
    write_manifest(val_entries, os.path.join(output_dir, "speech_commands", "valid.json"))


def process_fsdd(output_dir):
    audio_dir = os.path.join(output_dir, "fsdd", "audio")
    os.makedirs(audio_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "fsdd", "train.json")
    if os.path.exists(manifest_path):
        count = sum(1 for _ in open(manifest_path))
        logger.info("fsdd/train.json already exists (%d entries), skipping.", count)
        return

    logger.info("Loading Free Spoken Digit Dataset...")
    try:
        ds = load_dataset("jbischof/free_spoken_digit_dataset", split="train", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("Voxel51/Free-Spoken-Digit", split="train", trust_remote_code=True)
        except Exception as e:
            logger.warning("Could not load FSDD from HuggingFace: %s. Skipping.", e)
            return

    logger.info("  %d samples", len(ds))

    digit_names = ["zero", "one", "two", "three", "four", "five",
                   "six", "seven", "eight", "nine"]

    entries = []
    skipped = 0
    for idx in tqdm(range(len(ds)), desc="FSDD"):
        example = ds[idx]

        label = example.get("label", example.get("digit", None))
        if isinstance(label, int):
            if 0 <= label <= 9:
                text = digit_names[label]
            else:
                skipped += 1
                continue
        elif isinstance(label, str):
            text = label.lower()
        else:
            skipped += 1
            continue

        audio = example.get("audio", {})
        if isinstance(audio, dict) and "array" in audio:
            waveform = np.array(audio["array"], dtype=np.float32)
            sr = audio.get("sampling_rate", TARGET_SR)
        else:
            skipped += 1
            continue

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if sr != TARGET_SR:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

        duration = len(waveform) / TARGET_SR
        if duration < 0.1 or duration > 5.0:
            skipped += 1
            continue

        filename = f"fsdd_{text}_{idx:05d}.wav"
        filepath = os.path.join(audio_dir, filename)
        if not os.path.exists(filepath):
            sf.write(filepath, waveform, TARGET_SR)

        entries.append({
            "audio_filepath": os.path.abspath(filepath),
            "text": text,
            "duration": round(duration, 3),
            "lang": "ENGLISH",
            "source": "fsdd",
        })

    logger.info("FSDD: %d valid, %d skipped", len(entries), skipped)

    random.seed(42)
    random.shuffle(entries)
    val_count = max(1, int(len(entries) * 0.1))
    val_entries = entries[:val_count]
    train_entries = entries[val_count:]

    write_manifest(train_entries, manifest_path)
    write_manifest(val_entries, os.path.join(output_dir, "fsdd", "valid.json"))


def write_manifest(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Wrote %d entries to %s", len(entries), path)


def main():
    parser = argparse.ArgumentParser(description="Download English digit/keyword speech datasets")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    start = time.time()

    process_speech_commands(args.output_dir)
    process_fsdd(args.output_dir)

    elapsed = time.time() - start
    info = {
        "datasets": ["google/speech_commands_v2", "fsdd"],
        "processing_time_minutes": round(elapsed / 60, 1),
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logger.info("Done in %.1f minutes.", elapsed / 60)


if __name__ == "__main__":
    main()
