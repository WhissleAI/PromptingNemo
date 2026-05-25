#!/usr/bin/env python3
"""Extract 16kHz mono WAV audio from video clips.

Usage:
    python extract_audio.py \
        --clips-dir /mnt/nfs/data/speakervid_5m/clips \
        --output-dir /mnt/nfs/data/speakervid_5m/audio \
        --workers 16
"""
import argparse
import logging
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract audio from video clips")
    parser.add_argument("--clips-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def extract_audio(clip_path: Path, output_dir: Path, sample_rate: int) -> dict:
    output_path = output_dir / f"{clip_path.stem}.wav"
    if output_path.exists() and output_path.stat().st_size > 0:
        return {"clip": clip_path.stem, "status": "exists"}

    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(clip_path),
            "-ar", str(sample_rate),
            "-ac", "1",
            "-f", "wav",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, timeout=30, capture_output=True)

        if output_path.exists() and output_path.stat().st_size > 0:
            return {"clip": clip_path.stem, "status": "success"}
        return {"clip": clip_path.stem, "status": "failed", "error": "empty output"}
    except subprocess.CalledProcessError as e:
        return {"clip": clip_path.stem, "status": "failed", "error": str(e)[:200]}
    except Exception as e:
        return {"clip": clip_path.stem, "status": "failed", "error": str(e)[:200]}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = Path(args.clips_dir)

    clip_files = sorted(clips_dir.glob("*.mp4"))
    if args.limit > 0:
        clip_files = clip_files[:args.limit]

    logger.info("Extracting audio from %d clips, %d workers", len(clip_files), args.workers)

    completed = 0
    success = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_audio, f, output_dir, args.sample_rate): f for f in clip_files}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result["status"] in ("success", "exists"):
                success += 1

            if completed % 10000 == 0 or completed == len(clip_files):
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(
                    "[%d/%d] success=%d, %.1f/s",
                    completed, len(clip_files), success, rate,
                )

    logger.info("Done: %d success out of %d", success, len(clip_files))


if __name__ == "__main__":
    main()
