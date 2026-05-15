"""
Download CommonVoice audio for EURO + SLAVIC + EN languages and build NeMo manifests
from WhissleAI metadata datasets.

For each language:
  1. Load WhissleAI metadata (has annotated text with entity/intent/emotion tags)
  2. Download audio from mozilla-foundation/common_voice_17_0 (streaming)
  3. Match by CommonVoice clip filename
  4. Resample to 16kHz mono WAV
  5. Write NeMo manifests with audio path, annotated text, duration, lang, lang_family

Usage:
  # Download all EURO languages:
  python download_multilingual_cv.py --family euro --output-dir /mnt/nfs/data/multilingual_v1/raw/commonvoice_17/euro

  # Download all SLAVIC languages:
  python download_multilingual_cv.py --family slavic --output-dir /mnt/nfs/data/multilingual_v1/raw/commonvoice_17/slavic

  # Download specific languages:
  python download_multilingual_cv.py --langs de,fr,es --output-dir /mnt/nfs/data/multilingual_v1/raw/commonvoice_17/euro

  # Limit per language (for testing):
  python download_multilingual_cv.py --family euro --max-per-lang 1000 --output-dir /tmp/cv_test
"""
import argparse
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TARGET_SR = 16000

FAMILY_LANGS = {
    "euro": {
        "whissle_dataset": "WhissleAI/Meta_STT_EURO_Set1",
        "lang_family": "EUROPEAN",
        "cv_langs": ["de", "es", "fr", "it", "pt", "nl", "gl", "ro", "sv-SE", "da", "fi", "et"],
    },
    "slavic": {
        "whissle_dataset": "WhissleAI/Meta_STT_SLAVIC_CommonVoice",
        "lang_family": "SLAVIC",
        "cv_langs": ["be", "ru", "pl", "ka", "uk", "cs", "sk", "sl", "bg", "sr", "mk"],
    },
    "english": {
        "whissle_dataset": "WhissleAI/Meta_STT_EN_Set1",
        "lang_family": "ENGLISH",
        "cv_langs": ["en"],
    },
}

CV_DATASET = "fixie-ai/common_voice_17_0"

LANG_TO_FAMILY = {
    "de": "EUROPEAN", "es": "EUROPEAN", "fr": "EUROPEAN", "it": "EUROPEAN",
    "pt": "EUROPEAN", "nl": "EUROPEAN", "gl": "EUROPEAN", "ro": "EUROPEAN",
    "sv-SE": "EUROPEAN", "da": "EUROPEAN", "fi": "EUROPEAN", "et": "EUROPEAN",
    "be": "SLAVIC", "ru": "SLAVIC", "pl": "SLAVIC", "ka": "SLAVIC",
    "uk": "SLAVIC", "cs": "SLAVIC", "sk": "SLAVIC", "sl": "SLAVIC",
    "bg": "SLAVIC", "sr": "SLAVIC", "mk": "SLAVIC",
    "en": "ENGLISH",
}


def extract_lang_from_path(audio_filepath):
    """Extract language code from CommonVoice-style audio path."""
    m = re.search(r"/cv-corpus-[^/]+/([a-z]{2}(?:-[A-Z]{2})?)/", audio_filepath)
    if m:
        return m.group(1)
    m = re.search(r"/([a-z]{2}(?:-[A-Z]{2})?)/clips/", audio_filepath)
    if m:
        return m.group(1)
    return None


def build_whissle_index(whissle_dataset, splits=("train", "valid", "test")):
    """Load WhissleAI metadata and build filename→metadata index."""
    from datasets import load_dataset

    index = {}
    for split in splits:
        try:
            ds = load_dataset(whissle_dataset, split=split, trust_remote_code=True)
        except Exception as e:
            logging.warning(f"Could not load {whissle_dataset} split={split}: {e}")
            continue

        logging.info(f"  {whissle_dataset} split={split}: {len(ds):,} samples")
        for item in ds:
            filepath = item.get("audio_filepath", "")
            filename = os.path.basename(filepath)
            lang = extract_lang_from_path(filepath)
            index[filename] = {
                "text": item["text"],
                "duration": item.get("duration", 0),
                "lang_code": lang or "unknown",
                "split": split,
                "source": item.get("source", "commonvoice"),
            }
    return index


def download_lang(lang, whissle_index, audio_dir, lang_family, max_samples=None):
    """Download CommonVoice audio for one language, matching with WhissleAI metadata."""
    from datasets import load_dataset

    lang_index = {k: v for k, v in whissle_index.items() if v["lang_code"] == lang}
    if not lang_index:
        lang_short = lang.split("-")[0]
        lang_index = {k: v for k, v in whissle_index.items() if v["lang_code"] == lang_short}

    if not lang_index:
        logging.warning(f"  No WhissleAI entries for lang={lang}, trying direct download")

    os.makedirs(os.path.join(audio_dir, lang), exist_ok=True)
    matched_files = set()
    split_entries = defaultdict(list)

    for cv_split in ["train", "validation", "test"]:
        try:
            ds = load_dataset(CV_DATASET, lang, split=cv_split, streaming=True, token=os.environ.get("HF_TOKEN"))
        except Exception as e:
            logging.warning(f"  {lang}/{cv_split}: not available ({e})")
            continue

        matched = 0
        skipped = 0
        pbar = tqdm(ds, desc=f"  {lang}/{cv_split}", leave=False)

        for item in pbar:
            if max_samples and matched >= max_samples:
                break

            cv_filename = os.path.basename(item.get("path", ""))
            if not cv_filename:
                continue

            if lang_index and cv_filename not in lang_index:
                skipped += 1
                continue

            audio_data = item.get("audio")
            if not audio_data or "array" not in audio_data:
                continue

            waveform = np.array(audio_data["array"], dtype=np.float32)
            sr = audio_data["sampling_rate"]
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if sr != TARGET_SR:
                try:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
                except ImportError:
                    from scipy.signal import resample
                    new_len = int(len(waveform) * TARGET_SR / sr)
                    waveform = resample(waveform, new_len).astype(np.float32)

            duration = len(waveform) / TARGET_SR
            if duration < 0.3 or duration > 25.0:
                continue

            out_filename = cv_filename.rsplit(".", 1)[0] + ".wav"
            out_path = os.path.join(audio_dir, lang, out_filename)

            try:
                sf.write(out_path, waveform, TARGET_SR)
            except Exception:
                continue

            meta = lang_index.get(cv_filename, {})
            text = meta.get("text", item.get("sentence", ""))
            whissle_split = meta.get("split", "train") if meta else (
                "valid" if cv_split == "validation" else cv_split
            )

            entry = {
                "audio_filepath": os.path.abspath(out_path),
                "text": text,
                "duration": round(duration, 3),
                "lang": lang.split("-")[0].upper(),
                "lang_family": lang_family,
            }
            split_entries[whissle_split].append(entry)
            matched_files.add(cv_filename)
            matched += 1
            pbar.set_postfix(matched=matched)

        logging.info(f"  {lang}/{cv_split}: matched={matched}, skipped={skipped}")

    return split_entries


def main():
    parser = argparse.ArgumentParser(description="Download CommonVoice audio for multilingual training")
    parser.add_argument("--output-dir", required=True, help="Output directory for audio + manifests")
    parser.add_argument("--family", choices=list(FAMILY_LANGS.keys()), help="Language family to download")
    parser.add_argument("--langs", help="Comma-separated lang codes (overrides --family)")
    parser.add_argument("--max-per-lang", type=int, help="Max samples per language (for testing)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip languages that already have manifests")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        parser.error("HF_TOKEN env var required. CommonVoice is a gated dataset. "
                      "Get token from https://huggingface.co/settings/tokens")

    if args.langs:
        langs = [l.strip() for l in args.langs.split(",")]
        lang_family = LANG_TO_FAMILY.get(langs[0], "EUROPEAN")
        whissle_dataset = None
        for fam_info in FAMILY_LANGS.values():
            if langs[0] in fam_info["cv_langs"]:
                whissle_dataset = fam_info["whissle_dataset"]
                lang_family = fam_info["lang_family"]
                break
    elif args.family:
        fam_info = FAMILY_LANGS[args.family]
        langs = fam_info["cv_langs"]
        lang_family = fam_info["lang_family"]
        whissle_dataset = fam_info["whissle_dataset"]
    else:
        parser.error("Must specify --family or --langs")

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    start_time = time.time()

    whissle_index = {}
    if whissle_dataset:
        logging.info(f"Building WhissleAI index from {whissle_dataset}...")
        whissle_index = build_whissle_index(whissle_dataset)
        logging.info(f"WhissleAI index: {len(whissle_index):,} total entries")

        lang_counts = Counter(v["lang_code"] for v in whissle_index.values())
        logging.info(f"Languages in index: {dict(lang_counts.most_common())}")

    all_split_entries = defaultdict(list)

    for lang in langs:
        if args.skip_existing:
            manifest = os.path.join(args.output_dir, f"train_{lang}.json")
            if os.path.exists(manifest):
                logging.info(f"Skipping {lang} — manifest exists")
                continue

        logging.info(f"Processing {lang} (family={lang_family})...")
        family = LANG_TO_FAMILY.get(lang, lang_family)
        split_entries = download_lang(lang, whissle_index, audio_dir, family, args.max_per_lang)

        for split_name, entries in split_entries.items():
            all_split_entries[split_name].extend(entries)

            per_lang_manifest = os.path.join(args.output_dir, f"{split_name}_{lang}.json")
            with open(per_lang_manifest, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logging.info(f"  Wrote {len(entries):,} to {per_lang_manifest}")

    for split_name, entries in all_split_entries.items():
        manifest = os.path.join(args.output_dir, f"{split_name}.json")
        with open(manifest, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logging.info(f"Combined {split_name}: {len(entries):,} samples → {manifest}")

    elapsed = time.time() - start_time
    info = {
        "family": lang_family,
        "whissle_dataset": whissle_dataset,
        "cv_source": CV_DATASET,
        "langs": langs,
        "splits": {k: len(v) for k, v in all_split_entries.items()},
        "total_samples": sum(len(v) for v in all_split_entries.values()),
        "processing_time_minutes": round(elapsed / 60, 1),
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logging.info(f"Done in {elapsed/60:.1f} min. Total: {info['total_samples']:,} samples across {len(langs)} languages")


if __name__ == "__main__":
    main()
