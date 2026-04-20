#!/usr/bin/env python3
"""Normalize text fields in a NeMo-format JSONL manifest.

Reads each line, applies promptingnemo.data.normalize.normalize_text() to the
``text`` field, and writes the cleaned manifest to the output path.

Usage:
    python normalize_manifest.py \
        --input-manifest data/train.json \
        --output-manifest data/train_normalized.json
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize text fields in a NeMo JSONL manifest."
    )
    parser.add_argument(
        "--input-manifest",
        required=True,
        help="Path to the input JSONL manifest.",
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="Path for the cleaned output JSONL manifest.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Import here so argparse --help works without the full dependency tree.
    from promptingnemo.data.normalize import normalize_text

    total = 0
    modified = 0
    errors = 0

    with open(args.input_manifest, "r", encoding="utf-8") as fin, \
         open(args.output_manifest, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: skipping line {line_no}: {exc}",
                    file=sys.stderr,
                )
                errors += 1
                continue

            total += 1
            original_text = entry.get("text", "")
            cleaned_text = normalize_text(original_text)

            if cleaned_text != original_text:
                modified += 1

            entry["text"] = cleaned_text
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Done. Processed {total} entries, modified {modified}, skipped {errors} errors.")
    print(f"Output written to: {args.output_manifest}")


if __name__ == "__main__":
    main()
