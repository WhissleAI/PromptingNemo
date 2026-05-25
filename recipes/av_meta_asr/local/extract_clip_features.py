#!/usr/bin/env python3
"""Extract SigLIP 2 So400m visual features from speaker video clips at 5fps.

Extracts both CLS token (global scene representation) and patch tokens
(spatial features for attention heat maps) from each frame.

Output: .npz files with keys:
  - cls: (num_frames, 1152) — global features
  - patches: (num_frames, 729, 1152) — spatial patch features (27×27 grid)

Usage:
    python extract_clip_features.py \
        --clips-dir /mnt/nfs/data/speakervid_5m/clips \
        --output-dir /mnt/nfs/data/speakervid_5m/siglip_features \
        --batch-size 16 \
        --fps 5
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract SigLIP 2 visual features from video clips"
    )
    parser.add_argument("--clips-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--model-name", type=str, default="google/siglip2-so400m-patch14-384",
    )
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cls-only", action="store_true",
        help="Save only CLS tokens (saves disk space, disables spatial heat maps)",
    )
    return parser.parse_args()


def extract_frames(video_path: str, fps: int = 5) -> Optional[List[np.ndarray]]:
    try:
        import av
    except ImportError:
        logger.error("PyAV is required. Install with: pip install av")
        sys.exit(1)

    try:
        container = av.open(video_path)
    except Exception as e:
        logger.warning("Failed to open %s: %s", video_path, e)
        return None

    try:
        stream = container.streams.video[0]
    except (IndexError, av.error.InvalidDataError):
        container.close()
        return None

    video_fps = float(stream.average_rate) if stream.average_rate else 30.0
    frame_interval = max(1, int(round(video_fps / fps)))

    frames = []
    try:
        for i, frame in enumerate(container.decode(video=0)):
            if i % frame_interval == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
    except Exception:
        if not frames:
            container.close()
            return None

    container.close()
    return frames if frames else None


class SigLIPFeatureExtractor:
    """Extract features from SigLIP 2 So400m vision encoder.

    Returns both CLS token (1152-dim global feature) and patch tokens
    (729 × 1152 spatial features from 27×27 patch grid at 384px resolution).
    """

    def __init__(self, model_name: str = "google/siglip2-so400m-patch14-384",
                 device: Optional[str] = None):
        import torch

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            logger.error("transformers required. Install with: pip install transformers")
            sys.exit(1)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Loading SigLIP 2 %s on %s...", model_name, self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Determine feature dimensions with a dummy forward pass
        with torch.no_grad():
            from PIL import Image
            dummy_img = Image.new("RGB", (384, 384))
            inputs = self.processor(images=dummy_img, return_tensors="pt").to(self.device)
            vision_outputs = self.model.vision_model(**inputs)
            hidden = vision_outputs.last_hidden_state  # (1, 730, 1152)
            self.feat_dim = hidden.shape[-1]
            self.num_patches = hidden.shape[1] - 1  # exclude CLS
            self.patch_grid = int(self.num_patches ** 0.5)

        logger.info(
            "SigLIP 2 loaded: feat_dim=%d, patches=%d (%d×%d grid)",
            self.feat_dim, self.num_patches, self.patch_grid, self.patch_grid,
        )

    def extract(self, frames: List[np.ndarray], batch_size: int = 16,
                cls_only: bool = False) -> dict:
        """Extract features from a list of RGB frames.

        Args:
            frames: List of (H, W, 3) numpy arrays
            batch_size: Processing batch size
            cls_only: If True, only return CLS tokens (saves memory)

        Returns:
            dict with keys:
                "cls": np.ndarray of shape (num_frames, feat_dim)
                "patches": np.ndarray of shape (num_frames, num_patches, feat_dim)
                           (omitted if cls_only=True)
        """
        import torch
        from PIL import Image

        all_cls = []
        all_patches = [] if not cls_only else None

        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size]
            images = [Image.fromarray(f) for f in batch]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                vision_outputs = self.model.vision_model(**inputs)
                hidden = vision_outputs.last_hidden_state  # (B, 1+P, D)

                cls_tokens = hidden[:, 0, :]  # (B, D)
                cls_tokens = cls_tokens / cls_tokens.norm(dim=-1, keepdim=True)
                all_cls.append(cls_tokens.cpu().numpy())

                if not cls_only:
                    patch_tokens = hidden[:, 1:, :]  # (B, P, D)
                    patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
                    all_patches.append(patch_tokens.cpu().numpy())

        result = {"cls": np.concatenate(all_cls, axis=0).astype(np.float32)}
        if not cls_only:
            result["patches"] = np.concatenate(all_patches, axis=0).astype(np.float32)

        return result


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = Path(args.clips_dir)

    clip_files = sorted(clips_dir.glob("*.mp4"))
    if args.limit > 0:
        clip_files = clip_files[:args.limit]

    if not args.overwrite:
        clip_files = [f for f in clip_files if not (output_dir / f"{f.stem}.npz").exists()]

    logger.info("Extracting SigLIP 2 features from %d clips", len(clip_files))

    if not clip_files:
        logger.info("Nothing to process")
        return

    extractor = SigLIPFeatureExtractor(args.model_name, args.device)

    completed = 0
    failed = 0
    t_start = time.time()

    for clip_path in clip_files:
        feature_path = output_dir / f"{clip_path.stem}.npz"

        frames = extract_frames(str(clip_path), fps=args.fps)
        if frames is None or len(frames) == 0:
            np.savez_compressed(
                str(feature_path),
                cls=np.zeros((1, extractor.feat_dim), dtype=np.float32),
                patches=np.zeros((1, extractor.num_patches, extractor.feat_dim), dtype=np.float32)
                if not args.cls_only else np.array([]),
            )
            failed += 1
        else:
            features = extractor.extract(frames, batch_size=args.batch_size,
                                         cls_only=args.cls_only)
            np.savez_compressed(str(feature_path), **features)

        completed += 1
        if completed % 500 == 0 or completed == len(clip_files):
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            logger.info(
                "[%d/%d] failed=%d, %.1f clips/s",
                completed, len(clip_files), failed, rate,
            )

    logger.info(
        "Done: %d extracted, %d failed (feat_dim=%d, patches=%d, fps=%d)",
        completed - failed, failed, extractor.feat_dim, extractor.num_patches, args.fps,
    )


if __name__ == "__main__":
    main()
