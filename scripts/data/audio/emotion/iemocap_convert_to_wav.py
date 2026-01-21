#!/usr/bin/env python3
"""
iemocap_convert_to_wav.py
========================
Convert processed IEMOCAP audio stored as *.pkl into real WAV files:
  - mono
  - 16 kHz
  - PCM 16-bit (pcm_s16le)

Input layout (as in this repo):
  IEMOCAP/<split>/audio/*.pkl

Output layout (created by this script):
  IEMOCAP/<split>/audio_wav/*.wav

Notes
-----
- These *.pkl files appear to contain Torch tensors (pickle references torch rebuild ops).
- This script therefore requires `torch`. If your env doesn’t have torch, install it in your conda/venv.
- If the pickle contains features (2D) rather than waveform (1D), conversion is not possible; the script will skip and report.

Usage:
  python iemocap_convert_to_wav.py \
    --iemocap-root /Users/karan/Desktop/work/whissle/pnemo/IEMOCAP \
    --splits train test \
    --target-sr 16000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _require_torch():
    try:
        import torch  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'torch' (required to load the IEMOCAP *.pkl tensors). "
            "Install torch in your active environment and rerun."
        ) from e
    return torch


def _load_pkl_audio(pkl_path: Path) -> Tuple[Optional[Any], Optional[int], str]:
    """
    Returns (waveform, sample_rate, note).
    waveform: torch.Tensor or numpy array like (T,) or (1,T)
    """
    torch = _require_torch()

    obj = None
    try:
        obj = torch.load(pkl_path, map_location="cpu")
    except Exception as e:
        return None, None, f"torch.load failed: {type(e).__name__}: {e}"

    # Helper: unwrap common dict structures
    if isinstance(obj, dict):
        sr = None
        for k in ("sr", "sample_rate", "sampling_rate", "rate"):
            if k in obj and isinstance(obj[k], int):
                sr = int(obj[k])
                break

        # Common waveform keys
        for k in ("waveform", "audio", "wav", "samples", "signal"):
            if k in obj:
                return obj[k], sr, f"dict[{k}]"

        # Repo hint: string 'audio_feature' appears in pickle stream
        if "audio_feature" in obj:
            return obj["audio_feature"], sr, "dict[audio_feature]"

        return None, sr, f"dict keys unsupported: {list(obj.keys())[:10]}"

    # Tensor / array directly
    return obj, None, f"{type(obj).__name__}"


def _to_mono_1d(x: Any) -> Tuple[Optional[Any], str]:
    torch = _require_torch()

    # Convert to tensor
    if isinstance(x, torch.Tensor):
        t = x
    else:
        try:
            t = torch.as_tensor(x)
        except Exception as e:
            return None, f"cannot convert to tensor: {type(e).__name__}: {e}"

    if t.numel() == 0:
        return None, "empty tensor"

    # Accept (T,), (1,T), (T,1), (C,T)
    if t.dim() == 1:
        return t, "1d"
    if t.dim() == 2:
        # If looks like features (T, F) where F is large, skip
        # Heuristic: if second dim > 8 and first dim is not small, likely features.
        if t.shape[0] > 8 and t.shape[1] > 8:
            return None, f"looks like features (shape={tuple(t.shape)})"

        # Treat as channels x time or time x channels
        if t.shape[0] == 1:
            return t[0], "2d (1,T)"
        if t.shape[1] == 1:
            return t[:, 0], "2d (T,1)"
        # Multi-channel: average channels to mono
        if t.shape[0] <= 8:
            return t.float().mean(dim=0), f"2d (C,T) mono-avg C={t.shape[0]}"
        if t.shape[1] <= 8:
            return t.float().mean(dim=1), f"2d (T,C) mono-avg C={t.shape[1]}"

        return None, f"unsupported 2d shape={tuple(t.shape)}"

    return None, f"unsupported tensor dim={t.dim()} shape={tuple(t.shape)}"


def _resample_if_needed(wav_1d: Any, orig_sr: int, target_sr: int) -> Any:
    torch = _require_torch()
    if orig_sr == target_sr:
        return wav_1d
    if orig_sr <= 0 or target_sr <= 0:
        return wav_1d

    # Prefer torchaudio if available; fall back to scipy
    try:
        import torchaudio  # type: ignore

        if not isinstance(wav_1d, torch.Tensor):
            wav_1d = torch.as_tensor(wav_1d)
        wav_1d = wav_1d.float().unsqueeze(0)  # (1,T)
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        y = resampler(wav_1d).squeeze(0)
        return y
    except Exception:
        pass

    try:
        import numpy as np
        from scipy.signal import resample_poly  # type: ignore

        x = wav_1d.detach().cpu().numpy() if isinstance(wav_1d, torch.Tensor) else np.asarray(wav_1d)
        g = math.gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        y = resample_poly(x.astype("float32"), up, down)
        return torch.from_numpy(y)
    except Exception:
        # As a last resort, return original
        return wav_1d


def _write_wav(path: Path, wav_1d: Any, sr: int) -> Tuple[bool, str]:
    torch = _require_torch()
    try:
        import numpy as np
        import soundfile as sf  # type: ignore

        x = wav_1d.detach().cpu().numpy() if isinstance(wav_1d, torch.Tensor) else np.asarray(wav_1d)
        x = x.astype("float32", copy=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), x, sr, subtype="PCM_16")
        return True, "ok"
    except Exception as e:
        return False, f"write failed: {type(e).__name__}: {e}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert IEMOCAP .pkl audio tensors to 16k mono PCM wav.")
    p.add_argument("--iemocap-root", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--in-subdir", default="audio", help="Input subdir under split (default: audio)")
    p.add_argument("--out-subdir", default="audio_wav", help="Output subdir under split (default: audio_wav)")
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument(
        "--assume-input-sr",
        type=int,
        default=16000,
        help="If sample rate is not stored in the pickle, assume this (default: 16000).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Convert only first N files per split (0 = no limit)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.iemocap_root
    if not root.exists():
        raise SystemExit(f"--iemocap-root does not exist: {root}")

    _require_torch()  # fail fast with clear message

    for split in args.splits:
        in_dir = root / split / args.in_subdir
        out_dir = root / split / args.out_subdir
        if not in_dir.exists():
            print(f"[{split}] missing input dir: {in_dir} (skipping)")
            continue

        pkl_files = sorted(in_dir.rglob("*.pkl"))
        if args.limit and args.limit > 0:
            pkl_files = pkl_files[: args.limit]

        n_ok = n_skip = n_fail = 0
        for pkl_path in pkl_files:
            out_path = out_dir / (pkl_path.stem + ".wav")
            if out_path.exists() and not args.overwrite:
                n_skip += 1
                continue

            wav_obj, sr, note = _load_pkl_audio(pkl_path)
            if wav_obj is None:
                n_fail += 1
                print(f"[{split}] FAIL load {pkl_path.name}: {note}")
                continue

            wav_1d, note2 = _to_mono_1d(wav_obj)
            if wav_1d is None:
                n_fail += 1
                print(f"[{split}] FAIL shape {pkl_path.name}: {note} / {note2}")
                continue

            sr_in = sr if sr is not None else int(args.assume_input_sr)
            wav_rs = _resample_if_needed(wav_1d, sr_in, int(args.target_sr))
            ok, msg = _write_wav(out_path, wav_rs, int(args.target_sr))
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                print(f"[{split}] FAIL write {pkl_path.name}: {msg}")

        print(f"[{split}] converted={n_ok} skipped={n_skip} failed={n_fail} -> {out_dir}")


if __name__ == "__main__":
    main()


