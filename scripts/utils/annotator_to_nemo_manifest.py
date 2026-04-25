#!/usr/bin/env python3
"""
Convert whissle-annotator output (WhissleMultimodalRecord JSONL) to NeMo
training manifests with tagged transcripts for meta-ASR training.

Reads annotated records and produces NeMo JSONL with:
  - audio_filepath: path to WAV file
  - text: tagged transcript (inline entities + trailing meta tags)
  - duration: float seconds
  - lang: language code

Tagged text format matches PromptingNemo meta-ASR expectations:
  "ENTITY_PERSON_NAME john END called about warranty AGE_30_45 GENDER_MALE EMOTION_NEUTRAL INTENT_INFORM DOMAIN_MEDICAL"

Usage:
    python annotator_to_nemo_manifest.py \
        --input /data/betrac/annotated.jsonl \
        --output /data/betrac/train_manifest.json \
        --audio-root /data/betrac/wav/segments \
        --tag-format legacy

    python annotator_to_nemo_manifest.py \
        --input /data/experiment_123/records.jsonl \
        --output-dir /data/experiment_123/manifests/ \
        --split-ratio 0.95 0.025 0.025 \
        --tag-format full
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRAILING_TAG_FIELDS_LEGACY = [
    ("audio_age", "AGE_"),
    ("audio_gender", "GENDER_"),
    ("audio_tonal_emotion", "EMOTION_"),
    ("nlp_intent", "INTENT_"),
    ("nlp_domain", "DOMAIN_"),
]

TRAILING_TAG_FIELDS_FULL = [
    ("audio_age", "AGE_"),
    ("audio_gender", "GENDER_"),
    ("audio_tonal_emotion", "EMOTION_"),
    ("audio_emotion_intensity", "EMOTION_INTENSITY_"),
    ("audio_speech_rate", "SPEECH_RATE_"),
    ("audio_volume", "VOLUME_"),
    ("audio_pitch", "PITCH_"),
    ("audio_prosody", "PROSODY_"),
    ("audio_voice_quality", "VOICE_QUALITY_"),
    ("audio_accent", "ACCENT_"),
    ("audio_disfluency", "DISFLUENCY_"),
    ("audio_background_noise_type", "NOISE_TYPE_"),
    ("audio_background_noise_level", "NOISE_LEVEL_"),
    ("audio_overlapping_speech", "OVERLAP_"),
    ("audio_speaker_count", "SPEAKER_COUNT_"),
    ("audio_audio_event", "AUDIO_EVENT_"),
    ("audio_music_presence", "MUSIC_"),
    ("audio_laughter", "LAUGHTER_"),
    ("audio_breathing", "BREATHING_"),
    ("audio_snr_quality", "SNR_"),
    ("nlp_intent", "INTENT_"),
    ("nlp_sentiment", "SENTIMENT_"),
    ("nlp_topic", "TOPIC_"),
    ("nlp_speech_act", "SPEECH_ACT_"),
    ("nlp_domain", "DOMAIN_"),
    ("nlp_formality", "FORMALITY_"),
    ("nlp_spam_label", "SPAM_"),
]


def read_annotator_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON parse error: {e}")
    logger.info(f"Read {len(records)} records from {path}")
    return records


def build_tagged_text(record: dict, tag_format: str = "legacy",
                      include_na: bool = False) -> str:
    """
    Build the tagged transcript from an annotator record.

    Inline entity tags are preserved from the `text` field.
    Trailing meta tags are appended from classification fields.
    """
    text = record.get("text", "").strip()
    if not text:
        text = record.get("text_raw", "").strip()
    if not text:
        return ""

    if tag_format == "full":
        tag_fields = TRAILING_TAG_FIELDS_FULL
    else:
        tag_fields = TRAILING_TAG_FIELDS_LEGACY

    trailing_tags = []
    for field_name, expected_prefix in tag_fields:
        value = record.get(field_name, "NA")
        if value and value != "NA":
            if not value.startswith(expected_prefix) and expected_prefix not in value:
                value = expected_prefix + value
            trailing_tags.append(value)
        elif include_na:
            trailing_tags.append("NA")

    if trailing_tags:
        return f"{text} {' '.join(trailing_tags)}"
    return text


def normalize_audio_path(
    filepath: str,
    audio_root: Optional[str] = None,
    remap_prefix: Optional[tuple[str, str]] = None,
) -> str:
    if remap_prefix:
        old, new = remap_prefix
        if filepath.startswith(old):
            filepath = new + filepath[len(old):]

    if audio_root and not os.path.isabs(filepath):
        filepath = os.path.join(audio_root, filepath)

    return filepath


def convert_record(
    record: dict,
    tag_format: str = "legacy",
    include_na: bool = False,
    audio_root: Optional[str] = None,
    remap_prefix: Optional[tuple[str, str]] = None,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> Optional[dict]:
    """Convert a single annotator record to NeMo manifest entry."""
    duration = record.get("duration", 0.0)
    if isinstance(duration, str):
        try:
            duration = float(duration)
        except ValueError:
            return None

    if duration < min_duration or duration > max_duration:
        return None

    audio_fp = record.get("audio_filepath", "NA")
    if audio_fp == "NA" or not audio_fp:
        return None

    audio_fp = normalize_audio_path(audio_fp, audio_root, remap_prefix)

    tagged_text = build_tagged_text(record, tag_format, include_na)
    if not tagged_text:
        return None

    lang = record.get("language", "en")

    return {
        "audio_filepath": audio_fp,
        "text": tagged_text,
        "duration": round(duration, 3),
        "lang": lang,
    }


def write_manifest(entries: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(entries)} entries to {output_path}")


def split_entries(
    entries: list[dict],
    ratios: tuple[float, float, float] = (0.95, 0.025, 0.025),
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    random.seed(seed)
    shuffled = list(entries)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test


def compute_stats(entries: list[dict]) -> dict:
    total_dur = sum(e["duration"] for e in entries)
    tag_counts = Counter()
    for e in entries:
        text = e.get("text", "")
        for word in text.split():
            if re.match(r'^[A-Z][A-Z0-9_]*_[A-Z0-9_+.]+$', word) or word == "END":
                tag_counts[word] += 1

    return {
        "num_samples": len(entries),
        "total_hours": round(total_dur / 3600, 2),
        "avg_duration_sec": round(total_dur / max(len(entries), 1), 2),
        "top_tags": tag_counts.most_common(20),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert whissle-annotator JSONL to NeMo training manifests"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input annotator JSONL file (or directory of JSONL files)")
    parser.add_argument("--output", "-o",
                        help="Output NeMo manifest file (single file mode)")
    parser.add_argument("--output-dir",
                        help="Output directory for train/val/test splits")
    parser.add_argument("--tag-format", choices=["legacy", "full"], default="legacy",
                        help="Tag format: legacy (AGE/GENDER/EMOTION/INTENT/DOMAIN) or full (all tags)")
    parser.add_argument("--include-na", action="store_true",
                        help="Include NA tokens in trailing tags (default: skip NA)")
    parser.add_argument("--audio-root",
                        help="Root directory to prepend to relative audio paths")
    parser.add_argument("--remap-from",
                        help="Audio path prefix to replace (used with --remap-to)")
    parser.add_argument("--remap-to",
                        help="Replacement prefix for audio paths")
    parser.add_argument("--split-ratio", nargs=3, type=float,
                        default=[0.95, 0.025, 0.025],
                        help="Train/val/test split ratios (default: 0.95 0.025 0.025)")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Min audio duration in seconds (default: 0.5)")
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="Max audio duration in seconds (default: 30.0)")
    parser.add_argument("--verify-audio", action="store_true",
                        help="Check that each audio file exists on disk")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split shuffling")
    parser.add_argument("--stats", action="store_true",
                        help="Print tag and duration statistics")

    args = parser.parse_args()

    if not args.output and not args.output_dir:
        parser.error("Must specify either --output or --output-dir")

    remap_prefix = None
    if args.remap_from and args.remap_to:
        remap_prefix = (args.remap_from, args.remap_to)

    input_path = args.input
    if os.path.isdir(input_path):
        jsonl_files = sorted(Path(input_path).glob("*.jsonl"))
        records = []
        for jf in jsonl_files:
            records.extend(read_annotator_jsonl(str(jf)))
    else:
        records = read_annotator_jsonl(input_path)

    if not records:
        logger.error("No records found")
        sys.exit(1)

    entries = []
    skipped = Counter()
    for rec in records:
        entry = convert_record(
            rec,
            tag_format=args.tag_format,
            include_na=args.include_na,
            audio_root=args.audio_root,
            remap_prefix=remap_prefix,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
        if entry is None:
            if rec.get("duration", 0) < args.min_duration:
                skipped["too_short"] += 1
            elif rec.get("duration", 0) > args.max_duration:
                skipped["too_long"] += 1
            elif rec.get("audio_filepath", "NA") == "NA":
                skipped["no_audio"] += 1
            else:
                skipped["no_text"] += 1
            continue

        if args.verify_audio and not os.path.exists(entry["audio_filepath"]):
            skipped["audio_missing"] += 1
            continue

        entries.append(entry)

    logger.info(f"Converted {len(entries)} entries, skipped {dict(skipped)}")

    if args.stats:
        stats = compute_stats(entries)
        logger.info(f"Stats: {json.dumps(stats, indent=2)}")

    if args.output:
        write_manifest(entries, args.output)
    elif args.output_dir:
        train, val, test = split_entries(
            entries,
            ratios=tuple(args.split_ratio),
            seed=args.seed,
        )
        write_manifest(train, os.path.join(args.output_dir, "train.json"))
        write_manifest(val, os.path.join(args.output_dir, "valid.json"))
        write_manifest(test, os.path.join(args.output_dir, "test.json"))

        logger.info(f"Splits: train={len(train)}, valid={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()
