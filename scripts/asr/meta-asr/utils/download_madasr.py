#!/usr/bin/env python3
"""Download WhissleAI/Meta_STT_MADASR2.0_train_lg → NeMo manifests.

8 Indic language splits: bh (Bhojpuri), bn (Bengali), ch (Chhattisgarhi),
kn (Kannada), mg (Magahi), mt (Maithili), mr (Marathi), te (Telugu).

Inline tags have bugs (GER_M, GERDER_FEMALE, EMOTION_NEU, AGE_18-24).
This script fixes them during download.

Usage:
  python download_madasr.py --output-dir /mnt/nfs/data/multilingual_v1/raw/madasr
  python download_madasr.py --output-dir /mnt/nfs/data/multilingual_v1/raw/madasr --langs bn,mr,te
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
DATASET_NAME = "WhissleAI/Meta_STT_MADASR2.0_train_lg"

ALL_LANGS = ["bh", "bn", "ch", "kn", "mg", "mt", "mr", "te"]

LANG_NAMES = {
    "bh": "Bhojpuri", "bn": "Bengali", "ch": "Chhattisgarhi", "kn": "Kannada",
    "mg": "Magahi", "mt": "Maithili", "mr": "Marathi", "te": "Telugu",
}

GENDER_FIX_RE = re.compile(
    r'\b(GER_MALE|GER_FEMALE|GER_OTHER|GER_M|GER_F'
    r'|GERDER_MALE|GERDER_FEMALE|GERDER_OTHER|GERDER_M|GERDER_F)\b'
)
GENDER_FIX_MAP = {
    "GER_MALE": "GENDER_MALE", "GER_FEMALE": "GENDER_FEMALE", "GER_OTHER": "GENDER_OTHER",
    "GER_M": "GENDER_MALE", "GER_F": "GENDER_FEMALE",
    "GERDER_MALE": "GENDER_MALE", "GERDER_FEMALE": "GENDER_FEMALE",
    "GERDER_OTHER": "GENDER_OTHER", "GERDER_M": "GENDER_MALE", "GERDER_F": "GENDER_FEMALE",
}

EMOTION_FIX_RE = re.compile(r'\bEMOTION_(NEU|ANG|HAP|JOY|SADNESS|ANGER)\b')
EMOTION_FIX_MAP = {
    "NEU": "NEUTRAL", "ANG": "ANGRY", "HAP": "HAPPY",
    "JOY": "HAPPY", "SADNESS": "SAD", "ANGER": "ANGRY",
}

AGE_RANGE_RE = re.compile(r'\bAGE_(\d+[-_]\d+)\b')
AGE_RANGE_MAP = {
    "14-17": "0_18", "14_17": "0_18",
    "18-24": "18_30", "18_24": "18_30",
    "25-30": "18_30", "25_30": "18_30",
    "31-35": "30_45", "31_35": "30_45",
    "36-40": "30_45", "36_40": "30_45",
    "41-45": "45_60", "41_45": "45_60",
    "46-50": "45_60", "46_50": "45_60",
    "51-55": "45_60", "51_55": "45_60",
    "56-60": "45_60", "56_60": "45_60",
    "61-65": "60PLUS", "61_65": "60PLUS",
    "66-70": "60PLUS", "66_70": "60PLUS",
    "14_25": "18_30",
}

INTENT_FIX_RE = re.compile(r'\bINTENT_(INFORMATIONAL|EXCLAMATION|INSTRUCT)\b')
INTENT_FIX_MAP = {"INFORMATIONAL": "INFORM", "EXCLAMATION": "EXCLAIM", "INSTRUCT": "COMMAND"}

DOMAIN_RE = re.compile(r'\bDOMAIN_\S+\b')


def normalize_inline_tags(text: str) -> str:
    text = GENDER_FIX_RE.sub(lambda m: GENDER_FIX_MAP.get(m.group(1), m.group(0)), text)
    text = EMOTION_FIX_RE.sub(lambda m: f"EMOTION_{EMOTION_FIX_MAP[m.group(1)]}", text)
    text = AGE_RANGE_RE.sub(
        lambda m: f"AGE_{AGE_RANGE_MAP.get(m.group(1), m.group(1))}", text
    )
    text = INTENT_FIX_RE.sub(lambda m: f"INTENT_{INTENT_FIX_MAP[m.group(1)]}", text)
    text = DOMAIN_RE.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_item(item, idx, audio_dir, lang_code):
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
            logging.error(f"Audio bytes error {lang_code}/{idx}: {e}")
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

    audio_filename = f"madasr_{lang_code}_{idx:07d}.wav"
    audio_filepath = os.path.join(audio_dir, audio_filename)

    try:
        sf.write(audio_filepath, waveform, sample_rate)
    except Exception as e:
        logging.error(f"Write error {lang_code}/{idx}: {e}")
        return None

    text = normalize_inline_tags(text)

    return {
        "audio_filepath": os.path.abspath(audio_filepath),
        "text": text,
        "duration": round(duration, 3),
        "lang": lang_code.upper(),
        "lang_family": "INDO_ARYAN",
    }


def download_lang(lang_code: str, output_dir: str, resume: bool = False):
    lang_dir = os.path.join(output_dir, lang_code)
    audio_dir = os.path.join(lang_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    manifest_path = os.path.join(lang_dir, "train.json")
    checkpoint_path = manifest_path + ".progress"

    start_idx = 0
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            start_idx = int(f.read().strip())
        logging.info(f"  Resuming {lang_code} from index {start_idx}")

    if not resume and os.path.exists(manifest_path):
        existing = sum(1 for _ in open(manifest_path))
        if existing > 0:
            logging.info(f"  {lang_code}: {manifest_path} exists ({existing} entries), skipping.")
            return existing, 0

    logging.info(f"  Loading {lang_code} ({LANG_NAMES.get(lang_code, lang_code)})...")
    ds = load_dataset(DATASET_NAME, split=lang_code)
    total = len(ds)
    logging.info(f"  {lang_code}: {total} samples")

    mode = "a" if resume and start_idx > 0 else "w"
    valid_count = 0
    skipped = 0

    with open(manifest_path, mode, encoding="utf-8") as f:
        for idx in tqdm(range(start_idx, total), desc=f"{lang_code}",
                        initial=start_idx, total=total):
            item = ds[idx]
            entry = process_item(item, idx, audio_dir, lang_code)
            if entry:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                valid_count += 1
            else:
                skipped += 1

            if (idx + 1) % 2000 == 0:
                with open(checkpoint_path, "w") as cp:
                    cp.write(str(idx + 1))

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    logging.info(f"  {lang_code}: {valid_count} valid, {skipped} skipped")
    return valid_count, skipped


def main():
    parser = argparse.ArgumentParser(description="Download MADASR 2.0 → NeMo manifests")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--langs", help="Comma-separated language codes (default: all)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    langs = args.langs.split(",") if args.langs else ALL_LANGS
    start_time = time.time()
    total_valid = 0
    total_skipped = 0

    for lang in langs:
        if lang not in ALL_LANGS:
            logging.warning(f"Unknown language: {lang}, skipping")
            continue
        valid, skipped = download_lang(lang, args.output_dir, resume=args.resume)
        total_valid += valid
        total_skipped += skipped

    elapsed = time.time() - start_time

    info = {
        "dataset": DATASET_NAME,
        "languages": langs,
        "lang_family": "INDO_ARYAN",
        "total_valid": total_valid,
        "total_skipped": total_skipped,
        "processing_time_minutes": round(elapsed / 60, 1),
    }
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    logging.info(f"Done in {elapsed/60:.1f} min. {total_valid} valid, {total_skipped} skipped.")


if __name__ == "__main__":
    main()
