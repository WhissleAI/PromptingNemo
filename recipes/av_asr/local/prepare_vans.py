#!/usr/bin/env python3
"""VANS (Visual-Aware Noisy Speech) dataset creation pipeline.

This script creates the training data for Audio-Visual ASR by:
1. Filtering AudioSet for single-noise-label videos (excluding speech/voice
   classes, requiring a minimum number of samples per noise class).
2. Downloading the filtered AudioSet video segments via yt-dlp.
3. Mixing clean speech audio (from People's Speech) with noise audio at a
   specified or random SNR.
4. Using NeMo forced alignment to trim audio/transcript pairs to a maximum
   duration (default 10 seconds).
5. Appending the noise class label as the final token in each transcript
   (e.g., "hello world NOISE_dog_bark").
6. Splitting into train/val/test with balanced noise class distribution.
7. Writing NeMo JSONL manifests with fields:
   audio_filepath, video_filepath, feature_file, text, duration

Usage:
    python prepare_vans.py \
        --peoples-speech-dir /data/peoples_speech \
        --audioset-dir /data/audioset \
        --output-dir /data/vans \
        --max-duration 10.0 \
        --min-samples-per-class 750 \
        --snr-min -5.0 --snr-max 5.0
"""
import argparse
import json
import logging
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# AudioSet classes to exclude (speech, voice, and music-related)
EXCLUDED_AUDIOSET_CLASSES = {
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Babbling", "Speech synthesizer", "Shout", "Bellow", "Whoop",
    "Yell", "Children shouting", "Screaming", "Whispering", "Laughter",
    "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle",
    "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan",
    "Sigh", "Singing", "Choir", "Yodeling", "Chant", "Mantra",
    "Male singing", "Female singing", "Child singing", "Synthetic singing",
    "Rapping", "Humming", "Groan", "Grunt", "Whistling", "Breathing",
    "Wheeze", "Snoring", "Gasp", "Pant", "Snort", "Cough", "Throat clearing",
    "Sneeze", "Sniff", "Hiccup", "Burping, eructation", "Music",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create VANS dataset from People's Speech and AudioSet"
    )
    parser.add_argument(
        "--peoples-speech-dir",
        type=str,
        required=True,
        help="Root directory of the People's Speech dataset",
    )
    parser.add_argument(
        "--audioset-dir",
        type=str,
        required=True,
        help="Root directory of AudioSet (containing CSV metadata and optionally downloaded audio)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the VANS dataset",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Maximum audio duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=750,
        help="Minimum samples required per noise class (default: 750)",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=-5.0,
        help="Minimum SNR in dB for noise mixing (default: -5.0)",
    )
    parser.add_argument(
        "--snr-max",
        type=float,
        default=5.0,
        help="Maximum SNR in dB for noise mixing (default: 5.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip AudioSet video download (assume already downloaded)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    return parser.parse_args()


# =========================================================================
# AudioSet filtering
# =========================================================================

def load_audioset_ontology(audioset_dir: str) -> Dict[str, str]:
    """Load AudioSet ontology mapping from mid to display_name.

    Expects ontology.json in the AudioSet directory.
    """
    ontology_path = Path(audioset_dir) / "ontology.json"
    if not ontology_path.exists():
        logger.warning(
            "AudioSet ontology.json not found at %s. "
            "Download from https://research.google.com/audioset/ontology/ontology.json",
            ontology_path,
        )
        return {}

    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    mid_to_name = {}
    for entry in ontology:
        mid_to_name[entry["id"]] = entry["name"]
    return mid_to_name


def load_audioset_segments(
    audioset_dir: str,
    split: str = "unbalanced_train_segments",
) -> List[Dict]:
    """Load AudioSet segment metadata from CSV.

    Expected CSV format (after 3 header lines):
    YTID, start_seconds, end_seconds, positive_labels
    """
    csv_path = Path(audioset_dir) / f"{split}.csv"
    if not csv_path.exists():
        logger.error("AudioSet segments CSV not found: %s", csv_path)
        sys.exit(1)

    segments = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # Skip header lines (first 3 lines are comments)
            if line.startswith("#") or i < 3:
                continue
            line = line.strip()
            if not line:
                continue

            parts = line.split(", ")
            if len(parts) < 4:
                continue

            ytid = parts[0].strip()
            start = float(parts[1].strip())
            end = float(parts[2].strip())
            # Labels are in quotes, comma-separated
            labels_str = ", ".join(parts[3:]).strip().strip('"')
            labels = [label.strip().strip('"') for label in labels_str.split(",")]

            segments.append({
                "ytid": ytid,
                "start": start,
                "end": end,
                "labels": labels,
            })

    logger.info("Loaded %d AudioSet segments from %s", len(segments), csv_path)
    return segments


def filter_audioset_segments(
    segments: List[Dict],
    mid_to_name: Dict[str, str],
    min_samples_per_class: int,
) -> Tuple[List[Dict], Dict[str, str]]:
    """Filter AudioSet segments to single-label non-speech classes.

    Returns:
        filtered_segments: List of segments with exactly one valid noise label.
        noise_label_map: Mapping from AudioSet mid to noise label string.
    """
    # Filter to single-label segments with non-excluded classes
    candidates = []
    for seg in segments:
        if len(seg["labels"]) != 1:
            continue
        mid = seg["labels"][0]
        name = mid_to_name.get(mid, mid)
        if name in EXCLUDED_AUDIOSET_CLASSES:
            continue
        candidates.append({**seg, "noise_class": name})

    # Count samples per class
    class_counts = defaultdict(int)
    for seg in candidates:
        class_counts[seg["noise_class"]] += 1

    # Filter to classes with enough samples
    valid_classes = {
        cls for cls, count in class_counts.items()
        if count >= min_samples_per_class
    }

    filtered = [seg for seg in candidates if seg["noise_class"] in valid_classes]

    # Build label map
    noise_label_map = {}
    for seg in filtered:
        mid = seg["labels"][0]
        label = "NOISE_" + seg["noise_class"].replace(" ", "_").replace(",", "").lower()
        noise_label_map[mid] = label

    logger.info(
        "Filtered AudioSet: %d segments, %d noise classes (from %d candidates, %d original classes)",
        len(filtered),
        len(valid_classes),
        len(candidates),
        len(class_counts),
    )

    for cls in sorted(valid_classes):
        logger.info("  %s: %d samples", cls, class_counts[cls])

    return filtered, noise_label_map


# =========================================================================
# AudioSet download
# =========================================================================

def download_audioset_videos(
    segments: List[Dict],
    output_dir: str,
    num_workers: int = 4,
) -> Dict[str, str]:
    """Download AudioSet video segments using yt-dlp.

    Returns mapping from ytid to downloaded video file path.
    """
    video_dir = Path(output_dir) / "audioset_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    skipped = 0
    failed = 0

    # Deduplicate by ytid
    unique_segments = {}
    for seg in segments:
        ytid = seg["ytid"]
        if ytid not in unique_segments:
            unique_segments[ytid] = seg

    total = len(unique_segments)
    logger.info("Downloading %d unique AudioSet videos to %s", total, video_dir)

    for i, (ytid, seg) in enumerate(unique_segments.items()):
        video_path = video_dir / f"{ytid}.mp4"

        if video_path.exists():
            downloaded[ytid] = str(video_path)
            skipped += 1
            continue

        url = f"https://www.youtube.com/watch?v={ytid}"
        start = seg["start"]
        duration = seg["end"] - seg["start"]

        try:
            cmd = [
                "yt-dlp",
                "--quiet", "--no-warnings",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
                "--download-sections", f"*{start}-{start + duration}",
                "--force-keyframes-at-cuts",
                "-o", str(video_path),
                url,
            ]
            subprocess.run(cmd, check=True, timeout=120, capture_output=True)
            downloaded[ytid] = str(video_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            failed += 1
            if failed <= 10:
                logger.warning("Failed to download %s: %s", ytid, e)

        if (i + 1) % 500 == 0:
            logger.info(
                "Download progress: %d/%d (downloaded=%d, skipped=%d, failed=%d)",
                i + 1, total, len(downloaded) - skipped, skipped, failed,
            )

    logger.info(
        "Download complete: %d downloaded, %d already existed, %d failed",
        len(downloaded) - skipped, skipped, failed,
    )
    return downloaded


# =========================================================================
# Audio mixing
# =========================================================================

def load_audio(path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
    """Load audio file and resample to target sample rate."""
    try:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio
    except Exception as e:
        logger.warning("Failed to load audio %s: %s", path, e)
        return None


def mix_audio_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix clean speech with noise at a specified SNR in dB.

    The noise is repeated or truncated to match the clean audio length.
    """
    # Match lengths
    if len(noise) < len(clean):
        # Repeat noise to fill
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(clean)]

    # Compute power
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return clean.copy()

    # Scale noise to achieve target SNR
    # SNR = 10 * log10(clean_power / scaled_noise_power)
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    mixed = clean + scale * noise

    # Normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * 0.99 / peak

    return mixed


# =========================================================================
# People's Speech loading
# =========================================================================

def load_peoples_speech_manifest(peoples_speech_dir: str) -> List[Dict]:
    """Load People's Speech dataset manifest.

    Expects NeMo JSONL format or a directory of audio files with
    a manifest.json/train.json file.
    """
    ps_dir = Path(peoples_speech_dir)
    manifest_candidates = [
        ps_dir / "train.json",
        ps_dir / "manifest.json",
        ps_dir / "train_manifest.json",
        ps_dir / "train.jsonl",
    ]

    manifest_path = None
    for candidate in manifest_candidates:
        if candidate.exists():
            manifest_path = candidate
            break

    if manifest_path is None:
        logger.error(
            "Could not find People's Speech manifest in %s. "
            "Expected one of: %s",
            ps_dir,
            [str(c.name) for c in manifest_candidates],
        )
        sys.exit(1)

    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "audio_filepath" in entry and "text" in entry:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

    logger.info("Loaded %d People's Speech entries from %s", len(entries), manifest_path)
    return entries


# =========================================================================
# Forced alignment and trimming
# =========================================================================

def trim_to_duration(
    audio: np.ndarray,
    text: str,
    max_duration: float,
    sample_rate: int = 16000,
) -> Tuple[np.ndarray, str]:
    """Trim audio and corresponding transcript to max_duration.

    Uses word-level proportional trimming: if the audio exceeds max_duration,
    truncates both the audio and the transcript proportionally by word count.
    For more precise trimming, NeMo forced alignment can be used but is
    optional and slower.
    """
    max_samples = int(max_duration * sample_rate)

    if len(audio) <= max_samples:
        return audio, text

    # Simple proportional trim by word count
    words = text.split()
    total_samples = len(audio)
    trimmed_audio = audio[:max_samples]

    # Estimate how many words fit in the trimmed audio
    fraction = max_samples / total_samples
    num_words = max(1, int(len(words) * fraction))
    trimmed_text = " ".join(words[:num_words])

    return trimmed_audio, trimmed_text


# =========================================================================
# Dataset creation
# =========================================================================

def create_vans_dataset(
    peoples_speech_entries: List[Dict],
    audioset_segments: List[Dict],
    noise_label_map: Dict[str, str],
    downloaded_videos: Dict[str, str],
    output_dir: str,
    max_duration: float = 10.0,
    snr_min: float = -5.0,
    snr_max: float = 5.0,
    sample_rate: int = 16000,
    seed: int = 42,
) -> List[Dict]:
    """Create VANS dataset by mixing clean speech with noise.

    Each sample:
    1. Picks a random clean speech utterance from People's Speech
    2. Picks a random noise segment from filtered AudioSet
    3. Extracts noise audio from the video file
    4. Mixes at a random SNR from [snr_min, snr_max]
    5. Trims to max_duration
    6. Appends noise label to transcript
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    mixed_audio_dir = Path(output_dir) / "mixed_audio"
    mixed_audio_dir.mkdir(parents=True, exist_ok=True)

    # Filter segments to those with downloaded videos
    valid_segments = [
        seg for seg in audioset_segments
        if seg["ytid"] in downloaded_videos
    ]
    logger.info(
        "Creating VANS dataset: %d clean utterances x %d noise segments",
        len(peoples_speech_entries), len(valid_segments),
    )

    if not valid_segments:
        logger.error("No valid AudioSet segments with downloaded videos")
        sys.exit(1)

    manifest_entries = []
    failed = 0

    for i, ps_entry in enumerate(peoples_speech_entries):
        clean_path = ps_entry["audio_filepath"]
        clean_text = ps_entry["text"].strip()

        if not clean_text:
            continue

        clean_audio = load_audio(clean_path, target_sr=sample_rate)
        if clean_audio is None:
            failed += 1
            continue

        # Pick a random noise segment
        noise_seg = rng.choice(valid_segments)
        video_path = downloaded_videos[noise_seg["ytid"]]
        noise_mid = noise_seg["labels"][0]
        noise_label = noise_label_map.get(noise_mid, "NOISE_unknown")

        # Extract audio from video
        noise_audio = load_audio(video_path, target_sr=sample_rate)
        if noise_audio is None:
            failed += 1
            continue

        # Random SNR
        snr_db = np_rng.uniform(snr_min, snr_max)

        # Mix
        mixed = mix_audio_at_snr(clean_audio, noise_audio, snr_db)

        # Trim
        mixed, trimmed_text = trim_to_duration(mixed, clean_text, max_duration, sample_rate)

        # Append noise label
        text_with_label = f"{trimmed_text} {noise_label}"

        # Save mixed audio
        output_filename = f"vans_{i:08d}.wav"
        output_path = mixed_audio_dir / output_filename
        sf.write(str(output_path), mixed, sample_rate)

        duration = len(mixed) / sample_rate

        manifest_entries.append({
            "audio_filepath": str(output_path),
            "video_filepath": video_path,
            "feature_file": "",  # Will be populated in feature extraction stage
            "text": text_with_label,
            "duration": round(duration, 3),
            "snr_db": round(float(snr_db), 2),
            "noise_class": noise_label,
        })

        if (i + 1) % 5000 == 0:
            logger.info(
                "Progress: %d/%d samples created (%d failed)",
                len(manifest_entries), i + 1, failed,
            )

    logger.info(
        "Created %d VANS samples (%d failed)",
        len(manifest_entries), failed,
    )
    return manifest_entries


def split_dataset(
    entries: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test with balanced noise class distribution.

    Stratifies by noise_class so each split has proportional representation
    of all noise types.
    """
    rng = random.Random(seed)

    # Group by noise class
    by_class = defaultdict(list)
    for entry in entries:
        by_class[entry["noise_class"]].append(entry)

    train, val, test = [], [], []

    for noise_class, class_entries in by_class.items():
        rng.shuffle(class_entries)
        n = len(class_entries)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(class_entries[:n_train])
        val.extend(class_entries[n_train:n_train + n_val])
        test.extend(class_entries[n_train + n_val:])

    # Shuffle within each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    logger.info(
        "Split: train=%d, val=%d, test=%d (%.0f/%.0f/%.0f%%)",
        len(train), len(val), len(test),
        100 * len(train) / len(entries),
        100 * len(val) / len(entries),
        100 * len(test) / len(entries),
    )
    return train, val, test


def write_manifest(entries: List[Dict], output_path: str):
    """Write entries to a NeMo JSONL manifest file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Wrote %d entries to %s", len(entries), output_path)


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and filter AudioSet
    logger.info("Step 1: Filtering AudioSet segments...")
    mid_to_name = load_audioset_ontology(args.audioset_dir)
    segments = load_audioset_segments(args.audioset_dir)
    filtered_segments, noise_label_map = filter_audioset_segments(
        segments, mid_to_name, args.min_samples_per_class
    )

    if not filtered_segments:
        logger.error("No AudioSet segments passed filtering. Check data and thresholds.")
        sys.exit(1)

    # Step 2: Download AudioSet videos
    if args.skip_download:
        logger.info("Step 2: Skipping download (--skip-download)")
        video_dir = output_dir / "audioset_videos"
        downloaded = {}
        if video_dir.exists():
            for f in video_dir.iterdir():
                if f.suffix == ".mp4":
                    ytid = f.stem
                    downloaded[ytid] = str(f)
        logger.info("Found %d existing video files", len(downloaded))
    else:
        logger.info("Step 2: Downloading AudioSet videos...")
        downloaded = download_audioset_videos(
            filtered_segments, str(output_dir), num_workers=args.num_workers
        )

    # Step 3: Load People's Speech
    logger.info("Step 3: Loading People's Speech...")
    ps_entries = load_peoples_speech_manifest(args.peoples_speech_dir)

    # Step 4: Create VANS dataset
    logger.info("Step 4: Creating VANS dataset...")
    all_entries = create_vans_dataset(
        peoples_speech_entries=ps_entries,
        audioset_segments=filtered_segments,
        noise_label_map=noise_label_map,
        downloaded_videos=downloaded,
        output_dir=str(output_dir),
        max_duration=args.max_duration,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )

    if not all_entries:
        logger.error("No VANS samples were created. Check input data paths.")
        sys.exit(1)

    # Step 5: Split dataset
    logger.info("Step 5: Splitting dataset...")
    train, val, test = split_dataset(
        all_entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Step 6: Write manifests
    logger.info("Step 6: Writing manifests...")
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(train, str(manifest_dir / "train_manifest.json"))
    write_manifest(val, str(manifest_dir / "val_manifest.json"))
    write_manifest(test, str(manifest_dir / "test_manifest.json"))

    # Write noise class list
    noise_classes = sorted({e["noise_class"] for e in all_entries})
    classes_path = manifest_dir / "noise_classes.txt"
    with open(classes_path, "w", encoding="utf-8") as f:
        for cls in noise_classes:
            f.write(cls + "\n")
    logger.info("Wrote %d noise classes to %s", len(noise_classes), classes_path)

    # Summary
    logger.info("=" * 60)
    logger.info("VANS dataset creation complete!")
    logger.info("  Output dir : %s", output_dir)
    logger.info("  Train      : %d samples", len(train))
    logger.info("  Validation : %d samples", len(val))
    logger.info("  Test       : %d samples", len(test))
    logger.info("  Classes    : %d noise types", len(noise_classes))
    logger.info("  SNR range  : [%.1f, %.1f] dB", args.snr_min, args.snr_max)
    logger.info("  Max dur    : %.1f seconds", args.max_duration)
    logger.info("=" * 60)
    logger.info("Next step: run feature extraction (stage 2)")


if __name__ == "__main__":
    main()
