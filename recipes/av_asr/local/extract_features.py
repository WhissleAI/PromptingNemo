#!/usr/bin/env python3
"""CLIP visual feature extraction for Audio-Visual ASR.

Extracts CLIP ViT-L/14 features from video frames at a specified frame rate
(default 5 fps) and saves them as .npy files. Updates the NeMo JSONL manifest
with the path to each feature file.

The extracted features are used by the AV-CTC model during training and
inference. Pre-extracting features avoids loading the CLIP model during
training, significantly reducing GPU memory requirements and training time.

Usage:
    python extract_features.py \
        --data-dir /data/vans \
        --clip-model ViT-L/14 \
        --fps 5 \
        --batch-size 32

Input:
    Expects NeMo JSONL manifests in <data-dir>/manifests/ with a
    "video_filepath" field in each entry.

Output:
    - .npy feature files in <data-dir>/clip_features/
    - Updated manifests with "feature_file" field pointing to the .npy files
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLIP visual features from video frames"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root data directory containing manifests/ and video files",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-L/14",
        help="CLIP model variant (default: ViT-L/14)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frame rate for feature extraction (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for CLIP inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--manifests",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Specific manifest files to process. If not set, processes all "
            "*_manifest.json files in <data-dir>/manifests/"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract features even if .npy files already exist",
    )
    return parser.parse_args()


# =========================================================================
# Video frame extraction
# =========================================================================

def extract_frames_from_video(
    video_path: str,
    fps: int = 5,
) -> Optional[List[np.ndarray]]:
    """Extract frames from a video file at the specified fps.

    Uses PyAV (av) for efficient video decoding.

    Returns:
        List of frames as uint8 numpy arrays (H, W, 3) in RGB format,
        or None if extraction fails.
    """
    try:
        import av
    except ImportError:
        logger.error(
            "PyAV is required for video frame extraction. "
            "Install with: pip install av"
        )
        sys.exit(1)

    try:
        container = av.open(video_path)
    except Exception as e:
        logger.warning("Failed to open video %s: %s", video_path, e)
        return None

    try:
        stream = container.streams.video[0]
    except (IndexError, av.error.InvalidDataError):
        logger.warning("No video stream found in %s", video_path)
        container.close()
        return None

    # Calculate frame interval
    video_fps = float(stream.average_rate) if stream.average_rate else 30.0
    frame_interval = max(1, int(round(video_fps / fps)))

    frames = []
    try:
        for i, frame in enumerate(container.decode(video=0)):
            if i % frame_interval == 0:
                rgb_frame = frame.to_ndarray(format="rgb24")
                frames.append(rgb_frame)
    except Exception as e:
        logger.warning("Error decoding frames from %s: %s", video_path, e)
        if not frames:
            container.close()
            return None

    container.close()
    return frames if frames else None


# =========================================================================
# CLIP feature extraction
# =========================================================================

class CLIPFeatureExtractor:
    """Extracts visual features using a CLIP model."""

    def __init__(
        self,
        model_name: str = "ViT-L/14",
        device: Optional[str] = None,
    ):
        try:
            import open_clip
        except ImportError:
            logger.error(
                "open_clip_torch is required for CLIP feature extraction. "
                "Install with: pip install open_clip_torch"
            )
            sys.exit(1)

        import torch

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            "Loading CLIP model '%s' on device '%s'...",
            model_name, self.device,
        )

        # Map common CLIP model names to open_clip format
        model_map = {
            "ViT-L/14": ("ViT-L-14", "openai"),
            "ViT-B/32": ("ViT-B-32", "openai"),
            "ViT-B/16": ("ViT-B-16", "openai"),
            "ViT-H-14": ("ViT-H-14", "laion2b_s32b_b79k"),
        }

        if model_name in model_map:
            clip_name, pretrained = model_map[model_name]
        else:
            clip_name = model_name
            pretrained = "openai"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_name, pretrained=pretrained, device=self.device,
        )
        self.model.eval()

        # Get feature dimension from the model
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=self.device)
            dummy_out = self.model.encode_image(dummy)
            self.feat_dim = dummy_out.shape[-1]

        logger.info(
            "CLIP model loaded: %s (feat_dim=%d)", model_name, self.feat_dim
        )

    def extract_features(
        self,
        frames: List[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract CLIP features from a list of video frames.

        Args:
            frames: List of (H, W, 3) uint8 numpy arrays in RGB format.
            batch_size: Number of frames to process at once.

        Returns:
            Feature array of shape (num_frames, feat_dim) as float32.
        """
        import torch
        from PIL import Image

        all_features = []

        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start:start + batch_size]

            # Preprocess frames
            processed = []
            for frame in batch_frames:
                pil_image = Image.fromarray(frame)
                processed.append(self.preprocess(pil_image))

            batch_tensor = torch.stack(processed).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                # Normalize features
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0).astype(np.float32)


# =========================================================================
# Manifest processing
# =========================================================================

def process_manifest(
    manifest_path: str,
    feature_dir: str,
    extractor: CLIPFeatureExtractor,
    fps: int = 5,
    batch_size: int = 32,
    overwrite: bool = False,
) -> str:
    """Process a manifest file: extract features and update entries.

    Args:
        manifest_path: Path to input NeMo JSONL manifest.
        feature_dir: Directory to save .npy feature files.
        extractor: CLIPFeatureExtractor instance.
        fps: Frame rate for extraction.
        batch_size: Batch size for CLIP inference.
        overwrite: Whether to re-extract existing features.

    Returns:
        Path to the updated manifest file.
    """
    Path(feature_dir).mkdir(parents=True, exist_ok=True)

    # Read manifest
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    logger.info(
        "Processing manifest %s (%d entries)",
        manifest_path, len(entries),
    )

    updated = 0
    skipped = 0
    failed = 0

    for i, entry in enumerate(entries):
        video_path = entry.get("video_filepath", "")
        if not video_path or not Path(video_path).exists():
            failed += 1
            continue

        # Determine feature file path
        video_stem = Path(video_path).stem
        feature_filename = f"{video_stem}.npy"
        feature_path = str(Path(feature_dir) / feature_filename)

        # Skip if already extracted
        if not overwrite and Path(feature_path).exists():
            entry["feature_file"] = feature_path
            skipped += 1
            continue

        # Extract frames
        frames = extract_frames_from_video(video_path, fps=fps)
        if frames is None or len(frames) == 0:
            logger.warning("No frames extracted from %s", video_path)
            # Save zero features as fallback
            zero_features = np.zeros((1, extractor.feat_dim), dtype=np.float32)
            np.save(feature_path, zero_features)
            entry["feature_file"] = feature_path
            failed += 1
            continue

        # Extract CLIP features
        features = extractor.extract_features(frames, batch_size=batch_size)
        np.save(feature_path, features)

        entry["feature_file"] = feature_path
        updated += 1

        if (i + 1) % 1000 == 0:
            logger.info(
                "  Progress: %d/%d (extracted=%d, cached=%d, failed=%d)",
                i + 1, len(entries), updated, skipped, failed,
            )

    # Write updated manifest
    output_path = manifest_path  # Overwrite in place
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(
        "Updated manifest %s: extracted=%d, cached=%d, failed=%d",
        output_path, updated, skipped, failed,
    )
    return output_path


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    feature_dir = data_dir / "clip_features"
    manifest_dir = data_dir / "manifests"

    # Determine which manifests to process
    if args.manifests:
        manifest_paths = [Path(m) for m in args.manifests]
    else:
        if not manifest_dir.exists():
            logger.error("Manifest directory not found: %s", manifest_dir)
            sys.exit(1)
        manifest_paths = sorted(manifest_dir.glob("*_manifest.json"))

    if not manifest_paths:
        logger.error("No manifest files found to process")
        sys.exit(1)

    logger.info("Found %d manifest(s) to process", len(manifest_paths))

    # Initialize CLIP feature extractor
    extractor = CLIPFeatureExtractor(
        model_name=args.clip_model,
        device=args.device,
    )

    # Process each manifest
    for manifest_path in manifest_paths:
        logger.info("Processing: %s", manifest_path)
        process_manifest(
            manifest_path=str(manifest_path),
            feature_dir=str(feature_dir),
            extractor=extractor,
            fps=args.fps,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )

    logger.info("=" * 60)
    logger.info("Feature extraction complete!")
    logger.info("  Feature dir: %s", feature_dir)
    logger.info("  CLIP model : %s", args.clip_model)
    logger.info("  Frame rate : %d fps", args.fps)
    logger.info("  Feature dim: %d", extractor.feat_dim)
    logger.info("=" * 60)
    logger.info("Next step: run training (stage 3)")


if __name__ == "__main__":
    main()
