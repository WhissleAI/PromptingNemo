#!/usr/bin/env python3
"""Integrate Hindi Set1 data into the multilingual_v1 structure.

Fixes:
  1. audio_filepath: /mnt/training/data/ → /mnt/nfs/data/
  2. Adds lang_family: INDO_ARYAN
  3. Validates audio files exist
  4. Writes to multilingual_v1/raw/hindi_set1/

Usage:
  python integrate_hindi.py --source /mnt/nfs/data/meta_stt_hi_set1 \
    --output /mnt/nfs/data/multilingual_v1/raw/hindi_set1 --validate
"""
import argparse
import json
import logging
import os
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OLD_PATH_PREFIX = "/mnt/training/data/"
NEW_PATH_PREFIX = "/mnt/nfs/data/"


def fix_audio_path(path: str) -> str:
    if path.startswith(OLD_PATH_PREFIX):
        return path.replace(OLD_PATH_PREFIX, NEW_PATH_PREFIX, 1)
    return path


def process_manifest(source_path: str, output_path: str, validate: bool = False) -> dict:
    stats = {"total": 0, "valid": 0, "path_fixed": 0, "missing_audio": 0, "errors": 0}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(source_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                stats["errors"] += 1
                continue

            old_path = sample.get("audio_filepath", "")
            new_path = fix_audio_path(old_path)
            if new_path != old_path:
                stats["path_fixed"] += 1
            sample["audio_filepath"] = new_path

            if "lang_family" not in sample:
                sample["lang_family"] = "INDO_ARYAN"

            lang = sample.get("lang", "")
            if lang == "INDO_ARYAN":
                sample["lang"] = "HI"

            if validate and not os.path.exists(new_path):
                stats["missing_audio"] += 1
                continue

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stats["valid"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Integrate Hindi Set1 into multilingual_v1")
    parser.add_argument("--source", required=True, help="Source Hindi data dir")
    parser.add_argument("--output", required=True, help="Output dir under multilingual_v1/raw/")
    parser.add_argument("--validate", action="store_true", help="Skip samples with missing audio")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    manifests = [
        ("train.json", "train.json"),
        ("valid.json", "valid.json"),
    ]

    for src_name, out_name in manifests:
        src_path = os.path.join(args.source, src_name)
        if not os.path.exists(src_path):
            logging.warning(f"Source not found: {src_path}")
            continue

        out_path = os.path.join(args.output, out_name)
        logging.info(f"Processing {src_name}...")
        stats = process_manifest(src_path, out_path, validate=args.validate)
        logging.info(f"  {src_name}: {stats['total']} total, {stats['valid']} valid, "
                     f"{stats['path_fixed']} paths fixed, {stats['missing_audio']} missing audio")

    info = {
        "dataset": "WhissleAI/Meta_STT_HI_Set1",
        "lang": "HI",
        "lang_family": "INDO_ARYAN",
        "source": args.source,
    }
    with open(os.path.join(args.output, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logging.info("Done.")


if __name__ == "__main__":
    main()
