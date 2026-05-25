#!/usr/bin/env python3
"""Download SpeakerVid-5M annotations from HuggingFace to NFS.

The HF dataset stores annotations as tar.gz batches in:
  - merged_anno/tar_batch_*.tar.gz — merged per-clip JSONs (bbox, timing, talking, speakers)
  - raw_labels/asr/tar_batch_*.tar.gz — ASR transcripts
  - raw_labels/anno/tar_batch_*.tar.gz — MLLM captions (expression, movement, facing)
  - raw_labels/l_score/tar_batch_*.tar.gz — quality scores (51GB, optional)
  - raw_labels/scene_json/tar_batch_*.tar.gz — scene descriptions
  - raw_labels/speaker_json/tar_batch_*.tar.gz — speaker info
Plus top-level metadata: all_data_list.json, SFT_set.json, testset.json

Downloads tar.gz files and extracts them into per-clip JSON files.

Usage:
    python download_annotations.py \
        --output-dir /mnt/nfs/data/speakervid_5m/annotations

    # Include quality scores (adds ~52GB download + ~100GB+ extracted):
    python download_annotations.py \
        --output-dir /mnt/nfs/data/speakervid_5m/annotations \
        --include-l-score
"""
import argparse
import logging
import os
import subprocess
import sys
import tarfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ID = "dorni/SpeakerVid-5M-Dataset"

# Always download these
CORE_PATTERNS = [
    "all_data_list.json",
    "SFT_set.json",
    "testset.json",
    "merged_anno/**",
    "raw_labels/asr/**",
    "raw_labels/anno/**",
    "raw_labels/scene_json/**",
    "raw_labels/speaker_json/**",
]

# Optional (large)
LSCORE_PATTERN = "raw_labels/l_score/**"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download SpeakerVid-5M annotations from HuggingFace"
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--include-l-score", action="store_true",
        help="Also download l_score quality data (~52GB compressed)",
    )
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Download only, don't extract tar.gz files",
    )
    return parser.parse_args()


def extract_tarballs(base_dir: Path):
    """Find and extract all tar.gz files, then remove them to save space."""
    tar_files = sorted(base_dir.rglob("*.tar.gz"))
    logger.info("Found %d tar.gz files to extract", len(tar_files))

    for tar_path in tar_files:
        extract_dir = tar_path.parent
        logger.info("Extracting %s → %s", tar_path.name, extract_dir)
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            tar_path.unlink()
            logger.info("  Extracted and removed %s", tar_path.name)
        except Exception as e:
            logger.error("  Failed to extract %s: %s", tar_path.name, e)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)

    patterns = list(CORE_PATTERNS)
    if args.include_l_score:
        patterns.append(LSCORE_PATTERN)
        logger.info("Including l_score (~52GB compressed)")
    else:
        logger.info("Skipping l_score (use --include-l-score to include)")

    logger.info("Downloading SpeakerVid-5M annotations to %s", output_dir)
    logger.info("Repo: %s", REPO_ID)

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=patterns,
        token=args.token,
    )

    logger.info("Download complete: %s", local_dir)

    if not args.skip_extract:
        logger.info("Extracting tar.gz files...")
        extract_tarballs(output_dir)

    # Verify extracted files
    for subdir in ["merged_anno", "raw_labels/asr", "raw_labels/anno",
                   "raw_labels/scene_json", "raw_labels/speaker_json"]:
        dir_path = output_dir / subdir
        if dir_path.exists():
            json_count = len(list(dir_path.rglob("*.json")))
            logger.info("  %s: %d JSON files", subdir, json_count)
        else:
            logger.warning("  %s: NOT FOUND", subdir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
