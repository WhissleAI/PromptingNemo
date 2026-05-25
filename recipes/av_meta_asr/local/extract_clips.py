#!/usr/bin/env python3
"""Extract speaker clips from downloaded videos using merge_anno annotations.

For each annotation in merge_anno/, extracts a temporally and spatially cropped
clip from the corresponding YouTube video using ffmpeg.

Usage:
    python extract_clips.py \
        --annotations-dir /mnt/nfs/data/speakervid_5m/annotations \
        --videos-dir /mnt/nfs/data/speakervid_5m/videos \
        --output-dir /mnt/nfs/data/speakervid_5m/clips \
        --workers 8
"""
import argparse
import json
import logging
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract speaker clips from videos using merge_anno"
    )
    parser.add_argument("--annotations-dir", type=str, required=True)
    parser.add_argument("--videos-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=20.0)
    return parser.parse_args()


def load_annotation(anno_path: Path) -> dict | None:
    try:
        with open(anno_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def extract_clip(
    clip_name: str,
    anno: dict,
    videos_dir: Path,
    output_dir: Path,
    min_duration: float,
    max_duration: float,
) -> dict:
    output_path = output_dir / f"{clip_name}.mp4"
    if output_path.exists() and output_path.stat().st_size > 0:
        return {"clip": clip_name, "status": "exists"}

    video_name = anno.get("video_name", "")
    if not video_name:
        return {"clip": clip_name, "status": "failed", "error": "no video_name"}

    # video_name is "{youtube_id}_{WxH}_full_video"; extract 11-char YouTube ID
    youtube_id = re.sub(r'_\d+x\d+_full_video$', '', video_name)
    if not youtube_id:
        youtube_id = video_name[:11]

    video_path = videos_dir / f"{youtube_id}.mp4"
    if not video_path.exists():
        return {"clip": clip_name, "status": "failed", "error": f"video not found: {youtube_id}"}

    start = anno.get("start_seconds", anno.get("start", 0))
    end = anno.get("end_seconds", start + anno.get("duration", 0))
    duration = end - start

    if duration < min_duration or duration > max_duration:
        return {"clip": clip_name, "status": "filtered", "error": f"duration {duration:.1f}s"}

    bbox = anno.get("bbox", [])

    try:
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        cmd += ["-ss", str(start), "-t", str(duration)]
        cmd += ["-i", str(video_path)]

        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            vw = anno.get("raw_video_width", anno.get("clip_video_width", 0))
            vh = anno.get("raw_video_height", anno.get("clip_video_height", 0))
            if vw > 0 and vh > 0:
                crop_x = max(0, int(x * vw))
                crop_y = max(0, int(y * vh))
                crop_w = min(int(w * vw), vw - crop_x)
                crop_h = min(int(h * vh), vh - crop_y)
                if crop_w > 0 and crop_h > 0:
                    cmd += ["-vf", f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"]

        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]
        cmd += ["-c:a", "aac", "-b:a", "128k"]
        cmd += [str(output_path)]

        subprocess.run(cmd, check=True, timeout=60, capture_output=True)

        if output_path.exists() and output_path.stat().st_size > 0:
            return {"clip": clip_name, "status": "success", "duration": duration}
        return {"clip": clip_name, "status": "failed", "error": "output file empty"}

    except subprocess.CalledProcessError as e:
        return {"clip": clip_name, "status": "failed", "error": e.stderr.decode()[:200] if e.stderr else str(e)}
    except subprocess.TimeoutExpired:
        return {"clip": clip_name, "status": "failed", "error": "ffmpeg timeout"}
    except Exception as e:
        return {"clip": clip_name, "status": "failed", "error": str(e)[:200]}


_ANNO_DIR = None
_VIDEOS_DIR = None
_OUTPUT_DIR = None
_MIN_DURATION = 0.5
_MAX_DURATION = 20.0


def _process_clip(clip_name: str) -> dict:
    """Module-level function for ProcessPoolExecutor (must be picklable)."""
    anno_path = _ANNO_DIR / f"{clip_name}.json"
    if not anno_path.exists():
        return {"clip": clip_name, "status": "failed", "error": "annotation not extracted yet"}
    anno = load_annotation(anno_path)
    if anno is None:
        return {"clip": clip_name, "status": "failed", "error": "invalid JSON"}
    return extract_clip(
        clip_name, anno, _VIDEOS_DIR, _OUTPUT_DIR,
        _MIN_DURATION, _MAX_DURATION,
    )


def get_clip_names_for_available_videos(annotations_dir: Path, videos_dir: Path,
                                       limit: int = 0) -> list[str]:
    """Use all_data_list.json to find clip names for downloaded videos only.

    Avoids globbing millions of annotation files on NFS.
    """
    data_list_path = annotations_dir / "all_data_list.json"
    if not data_list_path.exists():
        logger.error("all_data_list.json not found, falling back to glob")
        return None

    available_videos = {p.stem for p in videos_dir.glob("*.mp4")}
    logger.info("Found %d downloaded videos", len(available_videos))

    with open(data_list_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    clip_names = []
    matched_videos = 0
    for key, clips in data_list.items():
        if not isinstance(clips, list):
            continue
        # Extract YouTube ID from first clip filename
        if clips:
            youtube_id = clips[0][:11]
        elif len(key) == 11:
            youtube_id = key
        else:
            continue

        if youtube_id in available_videos:
            matched_videos += 1
            for clip_file in clips:
                clip_name = clip_file.replace(".json", "")
                clip_names.append(clip_name)

    logger.info("Matched %d videos → %d clip annotations", matched_videos, len(clip_names))
    if limit > 0:
        clip_names = clip_names[:limit]
    return sorted(clip_names)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = Path(args.videos_dir)
    annotations_dir = Path(args.annotations_dir)

    global _ANNO_DIR, _VIDEOS_DIR, _OUTPUT_DIR, _MIN_DURATION, _MAX_DURATION

    anno_dir = annotations_dir / "merged_anno"
    if not anno_dir.exists():
        anno_dir = annotations_dir / "merge_anno"
    if not anno_dir.exists():
        logger.error("Neither merged_anno/ nor merge_anno/ found: %s", args.annotations_dir)
        sys.exit(1)

    _ANNO_DIR = anno_dir
    _VIDEOS_DIR = videos_dir
    _OUTPUT_DIR = output_dir
    _MIN_DURATION = args.min_duration
    _MAX_DURATION = args.max_duration

    # Use all_data_list.json for targeted lookup instead of globbing millions of files
    clip_names = get_clip_names_for_available_videos(
        annotations_dir, videos_dir, args.limit,
    )

    if clip_names is None:
        # Fallback to glob
        anno_files = sorted(anno_dir.glob("*.json"))
        if args.limit > 0:
            anno_files = anno_files[:args.limit]
        clip_names = [f.stem for f in anno_files]

    logger.info("Processing %d clips, %d workers", len(clip_names), args.workers)

    completed = 0
    success = 0
    failed = 0
    filtered = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_clip, name): name for name in clip_names}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result["status"] == "success" or result["status"] == "exists":
                success += 1
            elif result["status"] == "failed":
                failed += 1
            elif result["status"] == "filtered":
                filtered += 1

            if completed % 5000 == 0 or completed == len(clip_names):
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(
                    "[%d/%d] success=%d, failed=%d, filtered=%d, %.1f/s",
                    completed, len(clip_names), success, failed, filtered, rate,
                )

    logger.info("Done: %d success, %d failed, %d filtered out of %d",
                success, failed, filtered, len(clip_names))


if __name__ == "__main__":
    main()
