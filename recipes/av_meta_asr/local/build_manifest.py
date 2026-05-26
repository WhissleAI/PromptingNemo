#!/usr/bin/env python3
"""Build NeMo JSONL manifests for AV-Meta-ASR training.

Combines parsed annotations with audio, video, and SigLIP feature files into
NeMo-compatible JSONL manifests. Each entry includes:
  - audio/video/feature file paths
  - transcript with inline event tokens and trailing metadata tags
  - per-category tag fields for the TrailingTagClassifier head
  - quality filtering and stratified train/val/test split

Usage:
    python build_manifest.py \
        --annotations /mnt/nfs/data/speakervid_5m/metadata/parsed_annotations.jsonl \
        --audio-dir /mnt/nfs/data/speakervid_5m/audio \
        --clips-dir /mnt/nfs/data/speakervid_5m/clips \
        --features-dir /mnt/nfs/data/speakervid_5m/siglip_features \
        --output-dir /mnt/nfs/data/speakervid_5m/manifests \
        --min-duration 0.5 --max-duration 20.0 \
        --min-asr-confidence 0.3
"""
import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# All trailing tag fields that go into the CTC text target
TRAILING_TAG_FIELDS = [
    "tag_VISUAL_TALKING",
    "tag_VISUAL_SPEAKERS",
    "tag_VISUAL_FACING",
    "tag_VISUAL_BODY",
    "tag_VISUAL_ACTIVITY",
    "tag_VISUAL_EXPRESSION",
    "tag_VISUAL_MOVEMENT",
    "tag_SCENE_TYPE",
    "tag_NOISE_LEVEL",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build NeMo manifests for AV-Meta-ASR")
    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to parsed_annotations.jsonl")
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--clips-dir", type=str, required=True)
    parser.add_argument("--features-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--min-asr-confidence", type=float, default=0.3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def build_tagged_text(entry: dict) -> str:
    """Build transcript text with inline events and trailing metadata tags.

    Output format:
        "hello <LAUGH> world <NOD> how are you VISUAL_TALKING VISUAL_FACING_FRONT SCENE_INDOOR NOISE_CLEAN"
    """
    transcript = entry.get("transcript", "").strip()
    if not transcript:
        return ""

    # Insert inline event tokens at the end of transcript (before trailing tags)
    # Since we don't have precise timestamps for inline events from MLLM captions,
    # we append them right after the transcript text
    inline_events = entry.get("inline_events", [])
    if inline_events:
        transcript = transcript + " " + " ".join(inline_events)

    # Append trailing metadata tags
    trailing_tags = []
    for field in TRAILING_TAG_FIELDS:
        val = entry.get(field, "")
        if val:
            trailing_tags.append(val)

    if trailing_tags:
        transcript = transcript + " " + " ".join(trailing_tags)

    return transcript


def build_manifest_entry(entry: dict, audio_dir: Path, clips_dir: Path,
                         features_dir: Path) -> dict | None:
    """Build a single NeMo manifest entry from parsed annotation."""
    clip_name = entry["clip_name"]

    audio_path = audio_dir / f"{clip_name}.wav"
    video_path = clips_dir / f"{clip_name}.mp4"
    feature_path = features_dir / f"{clip_name}.npz"

    if not audio_path.exists():
        return None
    if not video_path.exists():
        return None
    if not feature_path.exists():
        return None

    tagged_text = build_tagged_text(entry)
    if not tagged_text:
        return None

    manifest = {
        "audio_filepath": str(audio_path),
        "video_filepath": str(video_path),
        "feature_file": str(feature_path),
        "duration": entry.get("duration", 0),
        "text": tagged_text,
    }

    # Add individual tag fields for the TrailingTagClassifier
    for field in TRAILING_TAG_FIELDS:
        manifest[field] = entry.get(field, "")

    return manifest


def stratified_split(entries: list, train_ratio: float, val_ratio: float,
                     seed: int) -> tuple:
    """Split entries into train/val/test with stratification by scene_type × talking."""
    rng = random.Random(seed)

    # Group by stratification key
    groups = defaultdict(list)
    for entry in entries:
        scene = entry.get("tag_SCENE_TYPE", "SCENE_INDOOR")
        talking = entry.get("tag_VISUAL_TALKING", "VISUAL_NOT_TALKING")
        key = f"{scene}_{talking}"
        groups[key].append(entry)

    train, val, test = [], [], []
    test_ratio = 1.0 - train_ratio - val_ratio

    for key, items in groups.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def write_manifest(entries: list, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    clips_dir = Path(args.clips_dir)
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load parsed annotations
    logger.info("Loading annotations from %s", args.annotations)
    annotations = []
    with open(args.annotations, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))

    if args.limit > 0:
        annotations = annotations[:args.limit]

    logger.info("Loaded %d annotations", len(annotations))

    # Quality filtering
    filtered = []
    filter_stats = defaultdict(int)

    for entry in annotations:
        duration = entry.get("duration", 0)
        confidence = entry.get("asr_confidence", 0)
        transcript = entry.get("transcript", "")

        if not transcript.strip():
            filter_stats["empty_transcript"] += 1
            continue
        if duration < args.min_duration:
            filter_stats["too_short"] += 1
            continue
        if duration > args.max_duration:
            filter_stats["too_long"] += 1
            continue
        if confidence < args.min_asr_confidence:
            filter_stats["low_confidence"] += 1
            continue
        filtered.append(entry)

    logger.info("After quality filtering: %d / %d", len(filtered), len(annotations))
    for reason, count in sorted(filter_stats.items()):
        logger.info("  Filtered %s: %d", reason, count)

    # Build manifest entries (check file existence)
    logger.info("Building manifest entries...")
    t_start = time.time()
    manifest_entries = []
    missing_files = 0

    for entry in filtered:
        result = build_manifest_entry(entry, audio_dir, clips_dir, features_dir)
        if result is None:
            missing_files += 1
        else:
            manifest_entries.append(result)

    elapsed = time.time() - t_start
    logger.info("Built %d entries (%.1fs), %d missing files",
                len(manifest_entries), elapsed, missing_files)

    if not manifest_entries:
        logger.error("No valid manifest entries. Check file paths.")
        return

    # Stratified split
    train, val, test = stratified_split(
        manifest_entries, args.train_ratio, args.val_ratio, args.seed,
    )

    logger.info("Split: train=%d, val=%d, test=%d", len(train), len(val), len(test))

    # Write manifests
    train_path = output_dir / "train_manifest.json"
    val_path = output_dir / "val_manifest.json"
    test_path = output_dir / "test_manifest.json"

    write_manifest(train, train_path)
    write_manifest(val, val_path)
    write_manifest(test, test_path)

    logger.info("Written: %s (%d), %s (%d), %s (%d)",
                train_path, len(train), val_path, len(val), test_path, len(test))

    # Log tag distributions in training set
    logger.info("Training set tag distributions:")
    for field in TRAILING_TAG_FIELDS:
        dist = defaultdict(int)
        for entry in train:
            dist[entry.get(field, "UNKNOWN")] += 1
        logger.info("  %s: %s", field,
                     ", ".join(f"{k}={v}" for k, v in sorted(dist.items(), key=lambda x: -x[1])))

    # Write summary stats
    stats = {
        "total_annotations": len(annotations),
        "after_filtering": len(filtered),
        "with_files": len(manifest_entries),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "filter_stats": dict(filter_stats),
        "missing_files": missing_files,
    }
    stats_path = output_dir / "manifest_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats written to %s", stats_path)


if __name__ == "__main__":
    main()
