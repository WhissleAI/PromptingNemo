#!/usr/bin/env python3
"""Download Hinglish (Hindi-English code-switching) ASR datasets → NeMo manifests.

Datasets:
  mucs        — dianavdavidson/MUCS-Hinglish (52K train + 3K test, CC-BY-4.0)
  indicvoices — dianavdavidson/indicvoices-hinglish-spontaneous (272K train + 69K valid + 4K test)

Downloads audio, saves as 16kHz WAV, creates NeMo JSONL manifests.
No inline tags added here — annotation (emotion/age/gender via whissle-annotator,
intent/entity via Gemini) is done as a separate step.

Usage:
  python download_hinglish.py --dataset mucs --output-dir /mnt/nfs/data/multilingual_v1/raw/hinglish_mucs
  python download_hinglish.py --dataset indicvoices --output-dir /mnt/nfs/data/multilingual_v1/raw/hinglish_indicvoices
  python download_hinglish.py --dataset all --output-dir /mnt/nfs/data/multilingual_v1/raw
"""
import argparse
import io
import json
import logging
import os
import time

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_SR = 16000
MIN_DURATION = 0.3
MAX_DURATION = 60.0

DATASETS = ["mucs", "indicvoices"]

MUCS_HF = "dianavdavidson/MUCS-Hinglish"
INDICVOICES_HF = "dianavdavidson/indicvoices-hinglish-spontaneous"


def resample_if_needed(waveform, sr):
    if sr != TARGET_SR:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
    return waveform


def extract_audio(item):
    audio_data = item.get("audio")
    if audio_data is None:
        return None, None

    if isinstance(audio_data, dict) and "array" in audio_data and audio_data["array"] is not None:
        waveform = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data.get("sampling_rate", TARGET_SR)
    elif isinstance(audio_data, dict) and "bytes" in audio_data and audio_data["bytes"]:
        try:
            waveform, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            waveform = np.array(waveform, dtype=np.float32)
        except Exception:
            return None, None
    else:
        return None, None

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    waveform = resample_if_needed(waveform, sr)
    return waveform, TARGET_SR


def write_manifest(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    logging.info(f"  Wrote {len(entries):,} entries → {path}")


# ---------------------------------------------------------------------------
# MUCS-Hinglish
# ---------------------------------------------------------------------------

MUCS_SPLITS = {
    "train": "train.json",
    "test": "test.json",
}

def download_mucs(output_dir, resume=False):
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    total_valid = 0

    for hf_split, out_name in MUCS_SPLITS.items():
        out_path = os.path.join(output_dir, out_name)
        checkpoint_path = out_path + ".progress"

        start_idx = 0
        if resume and os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                start_idx = int(f.read().strip())
            logging.info(f"  Resuming mucs/{hf_split} from index {start_idx}")

        if not resume and os.path.exists(out_path):
            n = sum(1 for _ in open(out_path))
            if n > 0:
                logging.info(f"  mucs/{hf_split}: {out_path} exists ({n:,} entries), skipping.")
                total_valid += n
                continue

        logging.info(f"  Loading MUCS-Hinglish {hf_split}...")
        ds = load_dataset(MUCS_HF, split=hf_split, token=hf_token)
        total = len(ds)
        logging.info(f"  mucs/{hf_split}: {total:,} samples")

        mode = "a" if resume and start_idx > 0 else "w"
        valid = 0
        skipped = 0

        with open(out_path, mode, encoding="utf-8") as f:
            for idx in tqdm(range(start_idx, total), desc=f"mucs/{hf_split}",
                            initial=start_idx, total=total):
                item = ds[idx]

                transcript = item.get("transcript", "")
                if not transcript or not transcript.strip():
                    skipped += 1
                    continue

                waveform, sr = extract_audio(item)
                if waveform is None:
                    skipped += 1
                    continue

                duration = len(waveform) / sr
                if duration < MIN_DURATION or duration > MAX_DURATION:
                    skipped += 1
                    continue

                fname = f"mucs_{hf_split}_{idx:07d}.wav"
                fpath = os.path.join(audio_dir, fname)
                try:
                    sf.write(fpath, waveform, sr)
                except Exception as e:
                    logging.error(f"  Write error mucs/{hf_split}/{idx}: {e}")
                    skipped += 1
                    continue

                entry = {
                    "audio_filepath": os.path.abspath(fpath),
                    "text": transcript.strip(),
                    "duration": round(duration, 3),
                    "lang": "HI-EN",
                    "lang_family": "INDO_ARYAN",
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                valid += 1

                if (idx + 1) % 2000 == 0:
                    with open(checkpoint_path, "w") as cp:
                        cp.write(str(idx + 1))

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        total_valid += valid
        logging.info(f"  mucs/{hf_split}: {valid:,} valid, {skipped:,} skipped")

    return total_valid


# ---------------------------------------------------------------------------
# IndicVoices-Hinglish-Spontaneous
# ---------------------------------------------------------------------------

INDICVOICES_SPLITS = {
    "train": "train.json",
    "valid": "valid.json",
    "test": "test.json",
}

def download_indicvoices(output_dir, resume=False):
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    total_valid = 0

    for hf_split, out_name in INDICVOICES_SPLITS.items():
        out_path = os.path.join(output_dir, out_name)
        checkpoint_path = out_path + ".progress"

        start_idx = 0
        if resume and os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                start_idx = int(f.read().strip())
            logging.info(f"  Resuming indicvoices/{hf_split} from index {start_idx}")

        if not resume and os.path.exists(out_path):
            n = sum(1 for _ in open(out_path))
            if n > 0:
                logging.info(f"  indicvoices/{hf_split}: {out_path} exists ({n:,} entries), skipping.")
                total_valid += n
                continue

        logging.info(f"  Loading IndicVoices-Hinglish {hf_split}...")
        ds = load_dataset(INDICVOICES_HF, split=hf_split, token=hf_token)
        total = len(ds)
        logging.info(f"  indicvoices/{hf_split}: {total:,} samples")

        mode = "a" if resume and start_idx > 0 else "w"
        valid = 0
        skipped = 0

        with open(out_path, mode, encoding="utf-8") as f:
            for idx in tqdm(range(start_idx, total), desc=f"indicvoices/{hf_split}",
                            initial=start_idx, total=total):
                item = ds[idx]

                text = (item.get("hinglish_mixed_scripts") or
                        item.get("hinglish_mixed_script_lowercase") or
                        item.get("text") or "")
                if not text or not text.strip():
                    skipped += 1
                    continue

                waveform, sr = extract_audio(item)
                if waveform is None:
                    skipped += 1
                    continue

                duration = len(waveform) / sr
                if duration < MIN_DURATION or duration > MAX_DURATION:
                    skipped += 1
                    continue

                fname = f"indicvoices_{hf_split}_{idx:07d}.wav"
                fpath = os.path.join(audio_dir, fname)
                try:
                    sf.write(fpath, waveform, sr)
                except Exception as e:
                    logging.error(f"  Write error indicvoices/{hf_split}/{idx}: {e}")
                    skipped += 1
                    continue

                entry = {
                    "audio_filepath": os.path.abspath(fpath),
                    "text": text.strip(),
                    "duration": round(duration, 3),
                    "lang": "HI-EN",
                    "lang_family": "INDO_ARYAN",
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                valid += 1

                if (idx + 1) % 2000 == 0:
                    with open(checkpoint_path, "w") as cp:
                        cp.write(str(idx + 1))

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        total_valid += valid
        logging.info(f"  indicvoices/{hf_split}: {valid:,} valid, {skipped:,} skipped")

    return total_valid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DOWNLOAD_FNS = {
    "mucs": download_mucs,
    "indicvoices": download_indicvoices,
}


def main():
    parser = argparse.ArgumentParser(description="Download Hinglish ASR datasets → NeMo manifests")
    parser.add_argument("--dataset", required=True, choices=DATASETS + ["all"],
                        help="Which dataset to download")
    parser.add_argument("--output-dir", required=True,
                        help="Output dir. For 'all', subdirs created per dataset.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted download")
    args = parser.parse_args()

    start = time.time()
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    for ds_name in datasets:
        if args.dataset == "all":
            ds_dir = os.path.join(args.output_dir, f"hinglish_{ds_name}")
        else:
            ds_dir = args.output_dir

        logging.info(f"=== {ds_name} → {ds_dir} ===")
        fn = DOWNLOAD_FNS[ds_name]
        total = fn(ds_dir, resume=args.resume)
        logging.info(f"=== {ds_name}: {total or 0:,} valid samples ===\n")

    elapsed = time.time() - start
    logging.info(f"All done in {elapsed/60:.1f} min")

    info = {
        "datasets": datasets,
        "lang": "HI-EN",
        "lang_family": "INDO_ARYAN",
        "processing_time_minutes": round(elapsed / 60, 1),
    }
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
