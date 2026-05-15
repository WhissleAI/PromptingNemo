"""
Download WhissleAI/Meta_STT_EN-IN_Tech_Interviews → NeMo manifests.

This dataset has separate columns for age_group, gender, emotion, intent alongside text.
Some samples have inline META tags (with GER_/EMOTION_NEU bugs), some have NO tags in text.
This script:
  1. Extracts audio arrays → WAV files
  2. Assembles canonical inline META tags from columns when text lacks them
  3. Normalizes any existing inline tags (GER_→GENDER_, etc.)
  4. Writes NeMo-format JSONL manifests

Usage:
  python download_en_in_tech.py --output-dir /mnt/nfs/data/multilingual_v1/raw/en_in_tech
"""
import argparse
import json
import logging
import os
import re
import time

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_SR = 16000

GENDER_CANONICAL = {
    "MALE": "GENDER_MALE",
    "FEMALE": "GENDER_FEMALE",
    "OTHER": "GENDER_OTHER",
}

EMOTION_CANONICAL = {
    "NEU": "EMOTION_NEUTRAL",
    "NEUTRAL": "EMOTION_NEUTRAL",
    "ANG": "EMOTION_ANGRY",
    "ANGRY": "EMOTION_ANGRY",
    "ANGER": "EMOTION_ANGRY",
    "HAP": "EMOTION_HAPPY",
    "HAPPY": "EMOTION_HAPPY",
    "JOY": "EMOTION_HAPPY",
    "SAD": "EMOTION_SAD",
    "SADNESS": "EMOTION_SAD",
    "FEAR": "EMOTION_FEAR",
    "DISGUST": "EMOTION_DISGUST",
    "SURPRISE": "EMOTION_SURPRISE",
}

INTENT_CANONICAL = {
    "INFORM": "INTENT_INFORM",
    "QUESTION": "INTENT_QUESTION",
    "COMMAND": "INTENT_COMMAND",
    "INSTRUCT": "INTENT_COMMAND",
    "REQUEST": "INTENT_REQUEST",
    "EXCLAIM": "INTENT_EXCLAIM",
    "EXCLAMATION": "INTENT_EXCLAIM",
    "OPINION": "INTENT_OPINION",
    "EXPLAIN": "INTENT_EXPLAIN",
    "DESCRIBE": "INTENT_DESCRIBE",
    "STATEMENT": "INTENT_STATEMENT",
    "ASSERT": "INTENT_STATEMENT",
    "UNCLEAR": "INTENT_INFORM",
    "THANK": "INTENT_THANK",
    "INFORMATIONAL": "INTENT_INFORM",
}

INLINE_TAG_RE = re.compile(
    r'\b(AGE_\S+|GER_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+)\b'
)

INLINE_GENDER_FIX = re.compile(r'\bGER_(MALE|FEMALE|OTHER)\b')
INLINE_EMOTION_FIX = re.compile(r'\bEMOTION_(NEU|ANG|HAP|JOY|SADNESS|ANGER)\b')
INLINE_INTENT_FIX = re.compile(r'\bINTENT_(INFORMATIONAL|EXCLAMATION|INSTRUCT)\b')

EMOTION_NORM = {
    "NEU": "NEUTRAL", "ANG": "ANGRY", "HAP": "HAPPY",
    "JOY": "HAPPY", "SADNESS": "SAD", "ANGER": "ANGRY",
}
INTENT_NORM = {
    "INFORMATIONAL": "INFORM", "EXCLAMATION": "EXCLAIM", "INSTRUCT": "COMMAND",
}


def has_inline_tags(text: str) -> bool:
    return bool(INLINE_TAG_RE.search(text))


def normalize_inline_tags(text: str) -> str:
    text = INLINE_GENDER_FIX.sub(lambda m: f"GENDER_{m.group(1)}", text)
    text = INLINE_EMOTION_FIX.sub(lambda m: f"EMOTION_{EMOTION_NORM[m.group(1)]}", text)
    text = INLINE_INTENT_FIX.sub(lambda m: f"INTENT_{INTENT_NORM[m.group(1)]}", text)
    return text


def strip_inline_tags(text: str) -> str:
    cleaned = INLINE_TAG_RE.sub('', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


def assemble_tags_from_columns(item: dict) -> str:
    parts = []
    age = item.get("age_group", "")
    if age:
        parts.append(f"AGE_{age}")

    gender = item.get("gender", "")
    if gender and gender.upper() in GENDER_CANONICAL:
        parts.append(GENDER_CANONICAL[gender.upper()])

    emotion = item.get("emotion", "")
    if emotion and emotion.upper() in EMOTION_CANONICAL:
        parts.append(EMOTION_CANONICAL[emotion.upper()])

    intent = item.get("intent", "")
    if intent and intent.upper() in INTENT_CANONICAL:
        parts.append(INTENT_CANONICAL[intent.upper()])

    return " ".join(parts)


def process_item(item, idx, audio_dir, split_name):
    audio_data = item.get("audio")
    text = item.get("text", "")

    if audio_data is None or not text:
        return None

    if isinstance(audio_data, dict) and "array" in audio_data:
        waveform = np.array(audio_data["array"], dtype=np.float32)
        sample_rate = audio_data.get("sampling_rate", TARGET_SR)
    elif isinstance(audio_data, dict) and "bytes" in audio_data and audio_data["bytes"]:
        import io
        try:
            waveform, sample_rate = sf.read(io.BytesIO(audio_data["bytes"]))
            waveform = np.array(waveform, dtype=np.float32)
        except Exception as e:
            logging.error(f"Audio bytes error {split_name}/{idx}: {e}")
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
    if duration < 0.3 or duration > 60.0:
        return None

    audio_filename = f"en_in_tech_{split_name}_{idx:07d}.wav"
    audio_filepath = os.path.join(audio_dir, audio_filename)

    try:
        sf.write(audio_filepath, waveform, sample_rate)
    except Exception as e:
        logging.error(f"Write error {split_name}/{idx}: {e}")
        return None

    if has_inline_tags(text):
        clean_text = strip_inline_tags(text)
        tags = assemble_tags_from_columns(item)
        text = f"{clean_text} {tags}" if tags else clean_text
    else:
        tags = assemble_tags_from_columns(item)
        text = f"{text.strip()} {tags}" if tags else text.strip()

    return {
        "audio_filepath": os.path.abspath(audio_filepath),
        "text": text,
        "duration": round(duration, 3),
        "lang": "EN_IN",
        "lang_family": "ENGLISH",
    }


def main():
    parser = argparse.ArgumentParser(description="Download EN-IN Tech Interviews → NeMo manifests")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    dataset_name = "WhissleAI/Meta_STT_EN-IN_Tech_Interviews"
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    split_mapping = {"train": "train.json", "validation": "valid.json"}
    start_time = time.time()

    for split_name, manifest_name in split_mapping.items():
        manifest_path = os.path.join(args.output_dir, manifest_name)
        checkpoint_path = manifest_path + ".progress"

        start_idx = 0
        if args.resume and os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                start_idx = int(f.read().strip())
            logging.info(f"Resuming {split_name} from index {start_idx}")

        if not args.resume and os.path.exists(manifest_path):
            existing = sum(1 for _ in open(manifest_path))
            logging.info(f"{manifest_name} exists ({existing} entries), skipping. Use --resume to continue.")
            continue

        logging.info(f"Loading {split_name} split...")
        ds = load_dataset(dataset_name, split=split_name)
        total = len(ds)
        logging.info(f"  {split_name}: {total} samples")

        mode = "a" if args.resume and start_idx > 0 else "w"
        entries_count = 0
        skipped = 0

        with open(manifest_path, mode, encoding="utf-8") as f:
            for idx in tqdm(range(start_idx, total), desc=f"Processing {split_name}",
                            initial=start_idx, total=total):
                item = ds[idx]
                entry = process_item(item, idx, audio_dir, split_name)
                if entry:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    entries_count += 1
                else:
                    skipped += 1

                if (idx + 1) % 1000 == 0:
                    with open(checkpoint_path, "w") as cp:
                        cp.write(str(idx + 1))

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        logging.info(f"  {split_name}: {entries_count} valid, {skipped} skipped → {manifest_name}")

    elapsed = time.time() - start_time

    info = {
        "dataset": dataset_name,
        "lang": "EN_IN",
        "lang_family": "ENGLISH",
        "processing_time_seconds": round(elapsed, 1),
    }
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    logging.info(f"Done in {elapsed/60:.1f} minutes. Info saved to {info_path}")


if __name__ == "__main__":
    main()
