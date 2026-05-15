"""
Merge annotated Gujarati manifests from multiple datasets into final
train.json and valid.json for NeMo training.

Uses IndicVoices-R test split as the primary validation set.
Deduplicates by audio_filepath.

Usage:
  python merge_gujarati_manifests.py \
      --data-root /mnt/nfs/data/gujarati_v1/raw \
      --output-dir /mnt/nfs/data/gujarati_v1
"""
import argparse
import json
import logging
import os
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def read_manifest(path):
    if not os.path.exists(path):
        logger.warning("Manifest not found: %s", path)
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def write_manifest(entries, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            out = {
                "audio_filepath": entry["audio_filepath"],
                "text": entry["text"],
                "duration": entry.get("duration", 0),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    logger.info("Wrote %d entries to %s", len(entries), path)


def deduplicate(entries):
    seen = set()
    deduped = []
    for e in entries:
        fp = e.get("audio_filepath", "")
        if fp not in seen:
            seen.add(fp)
            deduped.append(e)
    removed = len(entries) - len(deduped)
    if removed:
        logger.info("Removed %d duplicates", removed)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Merge Gujarati manifests into train/valid")
    parser.add_argument("--data-root", required=True, help="Root dir containing dataset subdirs")
    parser.add_argument("--output-dir", required=True, help="Output dir for final train.json / valid.json")
    parser.add_argument("--extra-val-ratio", type=float, default=0.02,
                        help="Extra validation samples from training data (on top of IndicVoices-R test)")
    args = parser.parse_args()

    train_entries = []
    valid_entries = []

    # Annotated manifests (prefer *_nemo.jsonl, fall back to train.json)
    datasets = {
        "indicvoices": {"train": ["train_nemo.jsonl", "train.json"], "valid": ["valid_nemo.jsonl", "valid.json"]},
        "indicvoices_r": {"train": ["train_nemo.jsonl", "train.json"], "valid": ["test_nemo.jsonl", "test.json"]},
        "kathbath": {"train": ["train_nemo.jsonl", "train.json"], "valid": ["valid_nemo.jsonl", "valid.json"]},
    }

    for dataset_name, splits in datasets.items():
        dataset_dir = os.path.join(args.data_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            logger.warning("Dataset dir not found: %s — skipping", dataset_dir)
            continue

        for split_type, candidates in splits.items():
            manifest = None
            for candidate in candidates:
                path = os.path.join(dataset_dir, candidate)
                if os.path.exists(path):
                    manifest = path
                    break

            if not manifest:
                logger.warning("No manifest found for %s/%s", dataset_name, split_type)
                continue

            entries = read_manifest(manifest)
            logger.info("%s/%s: %d entries from %s", dataset_name, split_type, len(entries), os.path.basename(manifest))

            if split_type == "valid" and dataset_name == "indicvoices_r":
                valid_entries.extend(entries)
            elif split_type == "valid":
                valid_entries.extend(entries)
            else:
                train_entries.extend(entries)

    # Deduplicate
    train_entries = deduplicate(train_entries)
    valid_entries = deduplicate(valid_entries)

    # If validation set is too small, supplement from training
    min_valid = 500
    if len(valid_entries) < min_valid and train_entries:
        extra = max(min_valid - len(valid_entries), int(len(train_entries) * args.extra_val_ratio))
        random.seed(42)
        random.shuffle(train_entries)
        valid_entries.extend(train_entries[:extra])
        train_entries = train_entries[extra:]
        logger.info("Supplemented validation with %d samples from training", extra)

    # Compute stats
    total_train_hours = sum(e.get("duration", 0) for e in train_entries) / 3600
    total_valid_hours = sum(e.get("duration", 0) for e in valid_entries) / 3600

    logger.info("=== Final Dataset Stats ===")
    logger.info("Training:   %d samples, %.1f hours", len(train_entries), total_train_hours)
    logger.info("Validation: %d samples, %.1f hours", len(valid_entries), total_valid_hours)

    write_manifest(train_entries, os.path.join(args.output_dir, "train.json"))
    write_manifest(valid_entries, os.path.join(args.output_dir, "valid.json"))

    info = {
        "train_samples": len(train_entries),
        "valid_samples": len(valid_entries),
        "train_hours": round(total_train_hours, 1),
        "valid_hours": round(total_valid_hours, 1),
        "sources": list(datasets.keys()),
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
