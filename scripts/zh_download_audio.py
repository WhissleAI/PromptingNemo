#!/usr/bin/env python3
"""Download WhissleAI/Meta_STT_ZH_AIShell3 with audio saved as .wav files.

Usage:
    python zh_download_audio.py --output-dir /mnt/nfs/data/meta_stt_zh_aishell3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from promptingnemo.data.normalize import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = "WhissleAI/Meta_STT_ZH_AIShell3"


def process_split(ds_split, split_name: str, output_dir: str):
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, f"{split_name}.json")
    kept = 0
    errors = 0

    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds_split):
            try:
                audio = row.get("audio") or row.get("audio_filepath")
                text = row.get("text", "")
                if not text or not audio:
                    errors += 1
                    continue

                if isinstance(audio, dict):
                    array = audio.get("array")
                    sr = audio.get("sampling_rate", 16000)
                    orig_path = audio.get("path", f"sample_{i}.wav")
                elif isinstance(audio, str):
                    errors += 1
                    continue
                else:
                    errors += 1
                    continue

                if array is None:
                    errors += 1
                    continue

                arr = np.array(array, dtype=np.float32)

                basename = os.path.splitext(os.path.basename(str(orig_path)))[0]
                wav_filename = f"{basename}.wav"
                wav_path = os.path.join(audio_dir, wav_filename)

                sf.write(wav_path, arr, sr)

                duration = len(arr) / sr
                norm_text = normalize_text(text)

                entry = {
                    "audio_filepath": wav_path,
                    "text": norm_text,
                    "duration": round(duration, 4),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                kept += 1

                if kept % 5000 == 0:
                    log.info("  %s: %d saved so far...", split_name, kept)

            except Exception as e:
                errors += 1
                if errors <= 10:
                    log.warning("Error on row %d: %s", i, e)

    log.info("%s: %d kept, %d errors", split_name, kept, errors)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Loading %s...", REPO)
    ds = load_dataset(REPO, trust_remote_code=True)

    total = 0
    for split_name in ds.keys():
        log.info("Processing %s (%d rows)...", split_name, len(ds[split_name]))
        n = process_split(ds[split_name], split_name, args.output_dir)
        total += n

    log.info("Done! %d total samples saved to %s", total, args.output_dir)


if __name__ == "__main__":
    main()
