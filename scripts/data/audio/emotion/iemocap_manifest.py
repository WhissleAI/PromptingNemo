#!/usr/bin/env python3
"""
iemocap_manifest.py
==================
Create NeMo-style JSONL manifests from the processed `IEMOCAP/` folder in this repo.

Observed dataset layout:
  IEMOCAP/
    train.txt
    test.txt
    valid.txt
    train/audio/*.pkl
    test/audio/*.pkl
    ...

Each line in {split}.txt is expected to be:
  <utt_id> <TAB> <emotion_short> <TAB> <transcript>
Example:
  Ses05F_impro01_F000    neu    Hi, I need an ID.

Output JSONL record format (one per line):
  {
    "audio_filepath": "...",
    "text": "...",
    "emotion_label": "..."
  }

Usage:
  python iemocap_manifest.py \
    --iemocap-root /Users/karan/Desktop/work/whissle/pnemo/IEMOCAP \
    --out-dir /Users/karan/Desktop/work/whissle/pnemo/IEMOCAP/manifests
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LABEL_MAP = {
    # Common IEMOCAP short labels
    "neu": "Neutral",
    "hap": "Happy",
    "sad": "Sad",
    "ang": "Angry",
    "exc": "Excited",
    "fru": "Frustrated",
    "fea": "Fear",
    "sur": "Surprise",
    "dis": "Disgust",
    "oth": "Other",
}


@dataclass(frozen=True)
class Entry:
    utt_id: str
    emotion_short: str
    text: str


def _parse_split_txt(path: Path) -> List[Entry]:
    entries: List[Entry] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            # Prefer tab, fall back to whitespace maxsplit=2.
            if "\t" in line:
                parts = line.split("\t", 2)
            else:
                parts = line.split(None, 2)

            if len(parts) < 3:
                raise ValueError(f"{path.name}:{i}: expected 3 fields (utt_id, label, text), got: {line!r}")

            utt_id, emo, txt = parts[0].strip(), parts[1].strip(), parts[2].strip()
            entries.append(Entry(utt_id=utt_id, emotion_short=emo, text=txt))
    return entries


def _index_audio_files(audio_dir: Path) -> Dict[str, Path]:
    """
    Build a stem->path index for faster resolution.
    In this repo, audio samples live under {split}/audio as *.pkl.
    """
    idx: Dict[str, Path] = {}
    if not audio_dir.exists():
        return idx
    for p in audio_dir.rglob("*"):
        if p.is_file():
            idx.setdefault(p.stem, p)
    return idx


def _resolve_audio_path(utt_id: str, audio_index: Dict[str, Path], audio_dir: Path) -> Optional[Path]:
    p = audio_index.get(utt_id)
    if p is not None:
        return p

    # Fallback (if someone adds wavs later)
    for ext in (".pkl", ".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".webm"):
        cand = audio_dir / f"{utt_id}{ext}"
        if cand.exists():
            return cand
    return None


def _emotion_label(short: str, keep_short: bool) -> str:
    if keep_short:
        return short
    return LABEL_MAP.get(short.lower(), short)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create JSONL manifests from the processed IEMOCAP folder.")
    p.add_argument(
        "--iemocap-root",
        type=Path,
        required=True,
        help="Path to IEMOCAP folder (contains train.txt/test.txt and split subfolders).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for manifests (train.jsonl/test.jsonl/valid.jsonl).",
    )
    p.add_argument(
        "--audio-subdir",
        default="audio",
        help="Audio subdir under each split (default: audio). Use 'audio_wav' after conversion.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "valid"],
        help="Which splits to process (default: train test valid).",
    )
    p.add_argument(
        "--keep-short-labels",
        action="store_true",
        help="Keep short emotion labels (neu/fru/...) instead of mapping to friendly names.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any utt_id cannot be resolved to an audio file. Default: skip missing.",
    )
    p.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute audio_filepath paths. Default: write paths relative to iemocap-root.",
    )
    return p.parse_args()


def write_split(
    *,
    split: str,
    iemocap_root: Path,
    out_path: Path,
    keep_short_labels: bool,
    strict: bool,
    absolute_paths: bool,
    audio_subdir: str,
) -> Tuple[int, int]:
    split_txt = iemocap_root / f"{split}.txt"
    entries = _parse_split_txt(split_txt)

    audio_dir = iemocap_root / split / audio_subdir
    audio_index = _index_audio_files(audio_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_missing = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for e in entries:
            audio_path = _resolve_audio_path(e.utt_id, audio_index, audio_dir)
            if audio_path is None:
                n_missing += 1
                if strict:
                    raise FileNotFoundError(f"[{split}] Missing audio for utt_id={e.utt_id} under {audio_dir}")
                continue

            audio_fp = str(audio_path.resolve()) if absolute_paths else str(audio_path.relative_to(iemocap_root))

            rec = {
                "audio_filepath": audio_fp,
                "text": e.text,
                "emotion_label": _emotion_label(e.emotion_short, keep_short_labels),
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    return n_written, n_missing


def main() -> None:
    args = parse_args()
    root = args.iemocap_root
    out_dir = args.out_dir

    if not root.exists():
        raise SystemExit(f"--iemocap-root does not exist: {root}")

    for split in args.splits:
        out_path = out_dir / f"{split}.jsonl"
        try:
            n_written, n_missing = write_split(
                split=split,
                iemocap_root=root,
                out_path=out_path,
                keep_short_labels=args.keep_short_labels,
                strict=args.strict,
                absolute_paths=args.absolute_paths,
                audio_subdir=args.audio_subdir,
            )
        except Exception as e:
            raise SystemExit(f"Failed split={split}: {e}") from e

        print(f"[{split}] wrote={n_written} missing_audio={n_missing} -> {out_path}")


if __name__ == "__main__":
    main()

