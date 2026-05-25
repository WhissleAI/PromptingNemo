#!/usr/bin/env python3
"""Extract only the annotations needed for downloaded videos.

Re-downloads tar.gz files from HuggingFace and selectively extracts only
the files matching our downloaded videos. Much faster than full extraction.

Usage:
    python extract_targeted_annotations.py \
        --annotations-dir /mnt/nfs/data/speakervid_5m/annotations \
        --videos-dir /mnt/nfs/data/speakervid_5m/videos
"""
import argparse
import json
import logging
import os
import tarfile
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ID = "dorni/SpeakerVid-5M-Dataset"

ANNOTATION_SUBDIRS = {
    "merged_anno": "merged_anno",
    "raw_labels/asr": "raw_labels/asr",
    "raw_labels/anno": "raw_labels/anno",
    "raw_labels/scene_json": "raw_labels/scene_json",
    "raw_labels/speaker_json": "raw_labels/speaker_json",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Selectively extract annotations for downloaded videos only"
    )
    parser.add_argument("--annotations-dir", type=str, required=True)
    parser.add_argument("--videos-dir", type=str, required=True)
    parser.add_argument("--subdirs", type=str, nargs="+",
                        default=["merged_anno", "raw_labels/asr", "raw_labels/anno"],
                        help="Which annotation subdirs to process")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/anno_tars",
                        help="Temp dir for downloaded tars (local SSD, not NFS)")
    return parser.parse_args()


def get_needed_clip_names(annotations_dir: Path, videos_dir: Path) -> set[str]:
    data_list_path = annotations_dir / "all_data_list.json"
    logger.info("Loading all_data_list.json...")
    with open(data_list_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    available_videos = {p.stem for p in videos_dir.glob("*.mp4")}
    logger.info("Found %d downloaded videos", len(available_videos))

    needed = set()
    matched_videos = 0
    for key, clips in data_list.items():
        if not isinstance(clips, list) or not clips:
            continue
        youtube_id = clips[0][:11]
        if youtube_id in available_videos:
            matched_videos += 1
            for clip_file in clips:
                clip_name = clip_file.replace(".json", "")
                needed.add(clip_name)

    logger.info("Matched %d videos → %d clip names needed", matched_videos, len(needed))
    return needed


def get_asr_names(needed: set[str]) -> set[str]:
    """ASR files use a truncated name: first 3 underscore-separated parts."""
    asr_names = set()
    for clip_name in needed:
        parts = clip_name.split("_")
        if len(parts) >= 3:
            asr_name = "_".join(parts[:3])
            asr_names.add(asr_name)
    return asr_names


def download_and_extract_targeted(
    subdir: str,
    annotations_dir: Path,
    target_names: set[str],
    tmp_dir: Path,
):
    """Download tar.gz files for a subdir and extract only matching files."""
    from huggingface_hub import hf_hub_download, list_repo_tree

    output_dir = annotations_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check what's already on disk
    already = 0
    for name in target_names:
        if (output_dir / f"{name}.json").exists():
            already += 1

    logger.info("  %s: %d/%d already on disk", subdir, already, len(target_names))
    if already == len(target_names):
        logger.info("  All files present, skipping")
        return already

    remaining = target_names - {
        name for name in target_names
        if (output_dir / f"{name}.json").exists()
    }
    logger.info("  Need to extract %d more files", len(remaining))

    # List tar files in this subdir on HuggingFace
    tar_files = []
    try:
        for entry in list_repo_tree(REPO_ID, path_in_repo=subdir, repo_type="dataset"):
            if entry.path.endswith(".tar.gz"):
                tar_files.append(entry.path)
    except Exception as e:
        logger.error("  Failed to list repo tree for %s: %s", subdir, e)
        return already

    tar_files.sort()
    logger.info("  Found %d tar.gz files on HuggingFace", len(tar_files))

    extracted_total = 0
    for tar_hf_path in tar_files:
        if not remaining:
            break

        tar_name = os.path.basename(tar_hf_path)
        local_tar = tmp_dir / tar_name

        try:
            # Download to local SSD (fast) instead of NFS
            logger.info("  Downloading %s...", tar_name)
            hf_hub_download(
                repo_id=REPO_ID,
                filename=tar_hf_path,
                repo_type="dataset",
                local_dir=str(tmp_dir),
            )
            actual_path = tmp_dir / tar_hf_path
            if not actual_path.exists():
                actual_path = local_tar

            logger.info("  Scanning %s for matches...", tar_name)
            extracted = 0
            with tarfile.open(str(actual_path), "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    basename = os.path.basename(member.name)
                    name_no_ext = basename.replace(".json", "")
                    if name_no_ext in remaining:
                        member.name = basename
                        tar.extract(member, path=str(output_dir))
                        remaining.discard(name_no_ext)
                        extracted += 1

            extracted_total += extracted
            logger.info("  %s: extracted %d files (%d remaining)",
                        tar_name, extracted, len(remaining))

            # Clean up downloaded tar
            if actual_path.exists():
                actual_path.unlink()

        except Exception as e:
            logger.error("  Failed to process %s: %s", tar_name, e)

    final_count = sum(1 for n in target_names if (output_dir / f"{n}.json").exists())
    logger.info("  %s complete: %d/%d available (extracted %d new)",
                subdir, final_count, len(target_names), extracted_total)
    return final_count


def main():
    args = parse_args()
    annotations_dir = Path(args.annotations_dir)
    videos_dir = Path(args.videos_dir)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    needed = get_needed_clip_names(annotations_dir, videos_dir)
    if not needed:
        logger.error("No matching clips found")
        return

    for subdir in args.subdirs:
        logger.info("=== Processing %s ===", subdir)
        if "asr" in subdir:
            target = get_asr_names(needed)
            logger.info("  ASR mode: %d unique names from %d clips", len(target), len(needed))
        elif "scene_json" in subdir or "speaker_json" in subdir:
            # These are per-video, use YouTube IDs
            target = set()
            for name in needed:
                target.add(name[:11])
            logger.info("  Per-video mode: %d unique IDs", len(target))
        else:
            target = needed

        download_and_extract_targeted(subdir, annotations_dir, target, tmp_dir)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
