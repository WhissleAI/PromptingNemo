#!/usr/bin/env python3
"""Download YouTube videos for SpeakerVid-5M dataset.

Reads all_data_list.json for unique video IDs and downloads each video
using yt-dlp. Supports parallel downloads, resume, and progress tracking.

Usage:
    python download_videos.py \
        --annotations-dir /mnt/nfs/data/speakervid_5m/annotations \
        --output-dir /mnt/nfs/data/speakervid_5m/videos \
        --workers 16 \
        --progress-file download_progress.jsonl
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download YouTube videos for SpeakerVid-5M"
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        required=True,
        help="Directory containing SpeakerVid-5M annotations (with all_data_list.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save downloaded videos",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel download workers (default: 16)",
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        default=None,
        help="JSONL file for tracking download progress (default: <output-dir>/download_progress.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of videos to download (0 = all, for testing)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        help="yt-dlp format string",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Download timeout per video in seconds (default: 300)",
    )
    parser.add_argument(
        "--min-video-id",
        type=str,
        default=None,
        help="Only download videos with YouTube ID >= this value (alphabetically). "
             "Use '23e9pbVQ_lo' to skip videos without merged_anno annotations.",
    )
    return parser.parse_args()


def extract_video_ids(annotations_dir: str) -> list[str]:
    data_list_path = Path(annotations_dir) / "all_data_list.json"
    if not data_list_path.exists():
        logger.error("all_data_list.json not found at %s", data_list_path)
        sys.exit(1)

    with open(data_list_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    video_ids = set()
    if isinstance(data_list, dict):
        for key, clips in data_list.items():
            if isinstance(clips, list) and clips:
                # Extract 11-char YouTube ID from first clip filename
                # Clip names: "{youtube_id}_{WxH}_full_video_{seg}_{spk}_{idx}.json"
                vid = clips[0][:11]
                if len(vid) == 11:
                    video_ids.add(vid)
            elif len(key) == 11:
                video_ids.add(key)
    elif isinstance(data_list, list):
        for entry in data_list:
            if isinstance(entry, dict):
                vid = entry.get("video_name") or entry.get("video_id", "")
                if vid:
                    video_ids.add(vid[:11])
            elif isinstance(entry, str) and len(entry) >= 11:
                video_ids.add(entry[:11])

    logger.info("Found %d unique video IDs", len(video_ids))
    return sorted(video_ids)


def load_progress(progress_path: Path) -> set:
    done = set()
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                if obj.get("status") in ("success", "exists"):
                    done.add(obj["video_id"])
    return done


def download_video(
    video_id: str,
    output_dir: Path,
    fmt: str,
    timeout: int,
) -> dict:
    video_path = output_dir / f"{video_id}.mp4"

    if video_path.exists() and video_path.stat().st_size > 0:
        return {"video_id": video_id, "status": "exists", "path": str(video_path)}

    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        cmd = [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            "-f", fmt,
            "--merge-output-format", "mp4",
            "--sleep-interval", "2",
            "--max-sleep-interval", "5",
            "-o", str(video_path),
            url,
        ]
        result = subprocess.run(
            cmd, check=True, timeout=timeout, capture_output=True, text=True,
        )
        if video_path.exists():
            return {"video_id": video_id, "status": "success", "path": str(video_path)}
        return {"video_id": video_id, "status": "failed", "error": "file not created"}
    except subprocess.CalledProcessError as e:
        return {"video_id": video_id, "status": "failed", "error": e.stderr[:200] if e.stderr else str(e)}
    except subprocess.TimeoutExpired:
        return {"video_id": video_id, "status": "failed", "error": f"timeout after {timeout}s"}
    except Exception as e:
        return {"video_id": video_id, "status": "failed", "error": str(e)[:200]}


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_path = Path(args.progress_file) if args.progress_file else output_dir / "download_progress.jsonl"

    video_ids = extract_video_ids(args.annotations_dir)

    if args.min_video_id:
        before = len(video_ids)
        video_ids = [v for v in video_ids if v >= args.min_video_id]
        logger.info("Filtered to IDs >= '%s': %d → %d videos",
                     args.min_video_id, before, len(video_ids))

    done = load_progress(progress_path)
    logger.info("Already downloaded: %d", len(done))

    pending = [vid for vid in video_ids if vid not in done]
    if args.limit > 0:
        pending = pending[:args.limit]

    logger.info("Pending: %d videos, %d workers", len(pending), args.workers)

    if not pending:
        logger.info("Nothing to download")
        return

    write_lock = Lock()
    completed = [0]
    failed = [0]
    t_start = time.time()

    def do_download(vid):
        result = download_video(vid, output_dir, args.format, args.timeout)

        with write_lock:
            with open(progress_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            completed[0] += 1
            if result["status"] == "failed":
                failed[0] += 1

            n = completed[0]
            if n % 100 == 0 or n == len(pending):
                elapsed = time.time() - t_start
                rate = n / elapsed if elapsed > 0 else 0
                eta = (len(pending) - n) / rate if rate > 0 else 0
                logger.info(
                    "[%d/%d] %d failed, %.1f/s, ETA %.0fs",
                    n, len(pending), failed[0], rate, eta,
                )

        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(do_download, vid): vid for vid in pending}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                vid = futures[future]
                logger.error("Fatal error for %s: %s", vid, e)

    elapsed = time.time() - t_start
    logger.info(
        "Download complete: %d processed in %.1fs (%d failed)",
        completed[0], elapsed, failed[0],
    )


if __name__ == "__main__":
    main()
