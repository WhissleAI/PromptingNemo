#!/usr/bin/env python3
"""Normalize aishell3 Chinese data for the multilingual pipeline.

Fixes:
  1. lang: MANDARIN → ZH
  2. Adds lang_family: EAST_ASIAN
  3. Adds EMOTION_NEUTRAL where no EMOTION_ tag exists
  4. Keeps DIALECT_ tags as-is

Usage:
  python normalize_aishell3.py --data-dir /mnt/nfs/data/multilingual_v1/raw/aishell3 --dry-run
  python normalize_aishell3.py --data-dir /mnt/nfs/data/multilingual_v1/raw/aishell3
"""
import argparse
import json
import logging
import os
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

HAS_EMOTION_RE = re.compile(r'\bEMOTION_\S+')
HAS_AGE_RE = re.compile(r'\bAGE_\S+')
HAS_GENDER_RE = re.compile(r'\bGENDER_\S+')


def normalize_sample(sample: dict) -> tuple[dict, bool]:
    changed = False

    if sample.get("lang") == "MANDARIN":
        sample["lang"] = "ZH"
        changed = True

    if sample.get("lang_family") != "EAST_ASIAN":
        sample["lang_family"] = "EAST_ASIAN"
        changed = True

    text = sample.get("text", "")
    if not HAS_EMOTION_RE.search(text):
        gender_match = HAS_GENDER_RE.search(text)
        age_match = HAS_AGE_RE.search(text)
        insert_pos = None
        if gender_match:
            insert_pos = gender_match.end()
        elif age_match:
            insert_pos = age_match.end()

        if insert_pos is not None:
            text = text[:insert_pos] + " EMOTION_NEUTRAL" + text[insert_pos:]
        else:
            text = text.rstrip() + " EMOTION_NEUTRAL"
        sample["text"] = text
        changed = True

    return sample, changed


def process_manifest(filepath: str, dry_run: bool) -> dict:
    stats = {"total": 0, "modified": 0}

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    stats["total"] = len(lines)
    output_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            output_lines.append(line)
            continue

        sample, changed = normalize_sample(sample)
        if changed:
            stats["modified"] += 1
        output_lines.append(json.dumps(sample, ensure_ascii=False))

    if not dry_run and stats["modified"] > 0:
        backup = filepath + ".pre_zh_norm.bak"
        if not os.path.exists(backup):
            os.rename(filepath, backup)
        else:
            os.remove(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines) + '\n')

    return stats


def main():
    parser = argparse.ArgumentParser(description="Normalize aishell3 Chinese data")
    parser.add_argument("--data-dir", required=True, help="aishell3 data directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_modified = 0
    for fname in sorted(os.listdir(args.data_dir)):
        if not fname.endswith('.json') or fname == 'dataset_info.json' or '.bak' in fname:
            continue
        filepath = os.path.join(args.data_dir, fname)
        stats = process_manifest(filepath, args.dry_run)
        logging.info(f"  {fname}: {stats['total']} samples, {stats['modified']} modified"
                     f"{' (dry-run)' if args.dry_run else ''}")
        total_modified += stats["modified"]

    logging.info(f"Total modified: {total_modified}")


if __name__ == "__main__":
    main()
