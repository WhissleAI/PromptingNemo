"""
Merge English training manifests and normalize text for meta-ASR training.

Combines:
  1. WhissleAI/Meta_STT_EN_Set1 (pre-tagged with inline meta tags)
  2. CommonVoice English (tagged via annotation pipeline)
  3. Speech Commands / FSDD digits (tagged with emotion)

Applies:
  - Digit-to-word normalization ("8" → "eight", "12" → "twelve")
  - Meta tag canonicalization
  - Duration filtering (0.3s – 20s)
  - Deduplication by audio_filepath
  - Train/valid split (95/5)
  - Optional digit oversampling

Usage:
  python merge_english_manifests.py \
      --data-root /mnt/nfs/data \
      --output-dir /mnt/nfs/data/english_v1 \
      --digit-oversample 3
"""
import argparse
import json
import logging
import os
import random
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from num2words import num2words
    HAS_NUM2WORDS = True
except ImportError:
    HAS_NUM2WORDS = False
    logger.warning("num2words not installed — digit normalization will use basic mapping")

META_TAG_RE = re.compile(
    r'\b(AGE_\S+|GENDER_\S+|GER_\S+|EMOTION_\S+|INTENT_\S+|ENTITY_\S+|END)\b'
)
DIGIT_RE = re.compile(r'\b(\d+)\b')

BASIC_DIGIT_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty", "30": "thirty",
    "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy",
    "80": "eighty", "90": "ninety", "100": "one hundred",
}

TAG_RENAMES = {
    "GER_MALE": "GENDER_MALE",
    "GER_FEMALE": "GENDER_FEMALE",
    "EMOTION_NEU": "EMOTION_NEUTRAL",
}


def normalize_number(match):
    num_str = match.group(1)
    try:
        val = int(num_str)
    except ValueError:
        return num_str

    if HAS_NUM2WORDS:
        try:
            return num2words(val)
        except Exception:
            return BASIC_DIGIT_MAP.get(num_str, num_str)
    else:
        return BASIC_DIGIT_MAP.get(num_str, num_str)


def normalize_text(text):
    parts = []
    tags = []
    for word in text.split():
        if META_TAG_RE.match(word):
            renamed = TAG_RENAMES.get(word, word)
            tags.append(renamed)
        else:
            parts.append(word)

    clean_text = " ".join(parts)
    clean_text = DIGIT_RE.sub(normalize_number, clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    if tags:
        return clean_text + " " + " ".join(tags)
    return clean_text


def load_manifest(path):
    entries = []
    if not os.path.exists(path):
        logger.warning("Manifest not found: %s", path)
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                logger.debug("Bad JSON at %s:%d", path, line_num)
    return entries


def find_manifests(root, patterns=("train.json", "train_nemo.jsonl")):
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname in patterns:
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description="Merge English manifests for meta-ASR training")
    parser.add_argument("--data-root", required=True, help="Root data directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged manifests")
    parser.add_argument("--digit-oversample", type=int, default=3,
                        help="Oversample digit datasets by this factor (default: 3)")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Validation split ratio (default: 0.05)")
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=20.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    source_dirs = [
        ("en_people", os.path.join(args.data_root, "multilingual_v1/raw/en_people")),
        ("cv_english", os.path.join(args.data_root, "multilingual_v1/raw/commonvoice_17/english")),
        ("speech_commands", os.path.join(args.data_root, "english_v1/raw/digits/speech_commands")),
        ("fsdd", os.path.join(args.data_root, "english_v1/raw/digits/fsdd")),
    ]

    digit_sources = {"speech_commands", "fsdd"}

    all_entries = []
    stats = {}

    for source_name, source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            logger.warning("Source dir not found: %s (%s)", source_dir, source_name)
            continue

        manifests = find_manifests(source_dir)
        if not manifests:
            logger.warning("No manifests found in %s", source_dir)
            continue

        source_entries = []
        for mf in manifests:
            entries = load_manifest(mf)
            logger.info("  %s: %d entries from %s", source_name, len(entries), os.path.basename(mf))
            source_entries.extend(entries)

        for entry in source_entries:
            entry["source"] = source_name
            if "lang" not in entry:
                entry["lang"] = "ENGLISH"

        repeat = args.digit_oversample if source_name in digit_sources else 1
        for _ in range(repeat):
            all_entries.extend(source_entries)

        stats[source_name] = len(source_entries) * repeat

    logger.info("Total entries before filtering: %d", len(all_entries))
    for src, cnt in stats.items():
        logger.info("  %s: %d (after oversampling)", src, cnt)

    filtered = []
    seen_paths = set()
    dropped = {"duration": 0, "empty_text": 0, "duplicate": 0, "bad_audio": 0}

    for entry in all_entries:
        duration = entry.get("duration", 0)
        if duration < args.min_duration or duration > args.max_duration:
            dropped["duration"] += 1
            continue

        text = entry.get("text", "").strip()
        if not text:
            dropped["empty_text"] += 1
            continue

        afp = entry.get("audio_filepath", "")
        if not afp:
            dropped["bad_audio"] += 1
            continue

        key = afp
        if key in seen_paths and entry.get("source") not in digit_sources:
            dropped["duplicate"] += 1
            continue
        seen_paths.add(key)

        entry["text"] = normalize_text(text)
        filtered.append(entry)

    logger.info("After filtering: %d entries", len(filtered))
    for reason, count in dropped.items():
        if count:
            logger.info("  Dropped (%s): %d", reason, count)

    random.seed(42)
    random.shuffle(filtered)

    val_count = max(1, int(len(filtered) * args.val_ratio))
    valid_entries = filtered[:val_count]
    train_entries = filtered[val_count:]

    train_path = os.path.join(args.output_dir, "train.json")
    valid_path = os.path.join(args.output_dir, "valid.json")

    write_manifest(train_entries, train_path)
    write_manifest(valid_entries, valid_path)

    info = {
        "sources": {k: v for k, v in stats.items()},
        "total_train": len(train_entries),
        "total_valid": len(valid_entries),
        "digit_oversample": args.digit_oversample,
    }
    with open(os.path.join(args.output_dir, "merge_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logger.info("Done. Train: %d, Valid: %d", len(train_entries), len(valid_entries))


def write_manifest(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Wrote %d entries to %s", len(entries), path)


if __name__ == "__main__":
    main()
