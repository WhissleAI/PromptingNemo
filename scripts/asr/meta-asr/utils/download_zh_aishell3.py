"""
Download WhissleAI/Meta_STT_ZH_AIShell3 from HuggingFace and create NeMo manifests.

This dataset contains Chinese speech with metatags (ENTITY_, EMOTION_, GENDER_, etc.)
embedded in the transcription text.

Dataset columns: audio_filepath (Audio), text (str), duration (float)
The audio is auto-decoded by HF datasets into {path, array, sampling_rate}.

Output: NeMo JSONL manifest with audio_filepath, text, duration, lang fields.
"""
import argparse
import json
import logging
import os

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_SR = 16000


def process_and_save(item, idx, audio_dir, target_sr):
    audio_data = item.get("audio_filepath") or item.get("audio")
    text = item.get("text")

    if audio_data is None or text is None:
        return None

    if isinstance(audio_data, dict):
        waveform = audio_data.get("array")
        sample_rate = audio_data.get("sampling_rate")
        if waveform is None or sample_rate is None:
            return None
        waveform = np.array(waveform, dtype=np.float32)
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

    audio_filename = f"aishell3_zh_{idx:07d}.wav"
    audio_filepath = os.path.join(audio_dir, audio_filename)

    try:
        sf.write(audio_filepath, waveform, sample_rate)
    except Exception as e:
        logging.error(f"Could not write audio for item {idx}: {e}")
        return None

    return {
        "audio_filepath": os.path.abspath(audio_filepath),
        "text": text.strip(),
        "duration": round(duration, 3),
        "lang": "MANDARIN",
    }


def create_manifests(dataset_name, output_dir, target_sr=16000, val_ratio=0.05):
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    logging.info(f"Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split="train")
    total = len(dataset)
    logging.info(f"Dataset loaded: {total} samples")

    all_entries = []
    skipped = 0
    for idx in tqdm(range(total), desc="Processing samples"):
        item = dataset[idx]
        entry = process_and_save(item, idx, audio_dir, target_sr)
        if entry:
            all_entries.append(entry)
        else:
            skipped += 1

    logging.info(f"Valid entries: {len(all_entries)} / {total} (skipped {skipped})")

    import random
    random.seed(42)
    random.shuffle(all_entries)
    val_count = max(1, int(len(all_entries) * val_ratio))
    val_entries = all_entries[:val_count]
    train_entries = all_entries[val_count:]

    for name, entries in [("train.json", train_entries), ("valid.json", val_entries)]:
        path = os.path.join(output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logging.info(f"Wrote {len(entries)} entries to {path}")

    from collections import Counter
    tag_counter = Counter()
    for entry in all_entries:
        for word in entry["text"].split():
            upper = word.upper()
            if "_" in upper and any(upper.startswith(p) for p in
                                    ["ENTITY_", "INTENT_", "EMOTION_", "GENDER_",
                                     "AGE_", "KEYWORD_", "LANG_", "DIALECT_", "END"]):
                tag_counter[upper] += 1
    if tag_counter:
        logging.info(f"Top metatags found: {tag_counter.most_common(20)}")
    else:
        logging.info("No metatags found in dataset text")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="WhissleAI/Meta_STT_ZH_AIShell3")
    parser.add_argument("--output-dir", default="/mnt/training/data/zh_aishell3")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    create_manifests(args.dataset, args.output_dir, args.sample_rate, args.val_ratio)
