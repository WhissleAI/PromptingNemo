#!/usr/bin/env python3
"""Download Chinese ASR datasets → NeMo manifests (no inline tags).

Datasets:
  cv-zh     — Common Voice 17 zh-CN (HuggingFace, streaming, ~50K samples)
  aishell1  — AISHELL-1 (OpenSLR 33, 178h, ~140K utterances)
  kespeech  — KeSpeech (HuggingFace TwinkStart/KeSpeech, test-only ~20K utterances)
  magicdata — MagicData (OpenSLR 68, 755h, ~600K utterances)

All output: NeMo JSONL manifests with audio_filepath, text, duration,
lang="ZH", lang_family="EAST_ASIAN". No tags — annotation done separately.

Usage:
  python download_zh_datasets.py --dataset cv-zh --output-dir /mnt/nfs/data/multilingual_v1/raw/cv_zh
  python download_zh_datasets.py --dataset aishell1 --output-dir /mnt/nfs/data/multilingual_v1/raw/aishell1
  python download_zh_datasets.py --dataset kespeech --output-dir /mnt/nfs/data/multilingual_v1/raw/kespeech
  python download_zh_datasets.py --dataset magicdata --output-dir /mnt/nfs/data/multilingual_v1/raw/magicdata
  python download_zh_datasets.py --dataset all --output-dir /mnt/nfs/data/multilingual_v1/raw
"""
import argparse
import glob
import json
import logging
import os
import subprocess
import tarfile
import time

import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_SR = 16000
MIN_DURATION = 0.3
MAX_DURATION = 30.0

DATASETS = ["cv-zh", "aishell1", "kespeech", "magicdata"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def resample_if_needed(waveform, sr):
    if sr != TARGET_SR:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
    return waveform


def write_manifest(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    logging.info(f"  Wrote {len(entries):,} entries → {path}")


def make_entry(audio_path, text, duration):
    return {
        "audio_filepath": os.path.abspath(audio_path),
        "text": text.strip(),
        "duration": round(duration, 3),
        "lang": "ZH",
        "lang_family": "EAST_ASIAN",
    }


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        logging.info(f"  Already downloaded: {dest_path}")
        return
    logging.info(f"  Downloading {url} → {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    subprocess.run(
        ["wget", "-c", "--progress=bar:force", "-O", dest_path, url],
        check=True,
    )


# ---------------------------------------------------------------------------
# 1. Common Voice ZH-CN (HuggingFace streaming)
# ---------------------------------------------------------------------------

def download_cv_zh(output_dir, max_samples=None, resume=False):
    """Download Common Voice 17 zh-CN via HuggingFace streaming."""
    from datasets import load_dataset

    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "train.json")
    valid_path = os.path.join(output_dir, "valid.json")
    test_path = os.path.join(output_dir, "test.json")

    if not resume and os.path.exists(manifest_path):
        n = sum(1 for _ in open(manifest_path))
        if n > 0:
            logging.info(f"  cv-zh: {manifest_path} exists ({n:,} entries), skipping. Use --resume to continue.")
            return

    hf_token = os.environ.get("HF_TOKEN")
    cv_dataset = "fixie-ai/common_voice_17_0"

    split_map = {
        "train": ("train", manifest_path),
        "validation": ("valid", valid_path),
        "test": ("test", test_path),
    }

    total_valid = 0
    for cv_split, (out_name, out_path) in split_map.items():
        entries = []
        idx = 0
        skipped = 0

        try:
            ds = load_dataset(cv_dataset, "zh-CN", split=cv_split, streaming=True, token=hf_token)
        except Exception as e:
            logging.warning(f"  cv-zh/{cv_split}: not available ({e})")
            continue

        logging.info(f"  Streaming cv-zh/{cv_split}...")
        pbar = tqdm(ds, desc=f"  cv-zh/{cv_split}", leave=False)

        for item in pbar:
            if max_samples and idx >= max_samples:
                break

            audio = item.get("audio")
            sentence = item.get("sentence", "")
            if not audio or not sentence:
                skipped += 1
                continue

            waveform = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = resample_if_needed(waveform, sr)

            duration = len(waveform) / TARGET_SR
            if duration < MIN_DURATION or duration > MAX_DURATION:
                skipped += 1
                continue

            fname = f"cv_zh_{out_name}_{idx:07d}.wav"
            fpath = os.path.join(audio_dir, fname)
            try:
                sf.write(fpath, waveform, TARGET_SR)
            except Exception as e:
                logging.error(f"  Write error {fname}: {e}")
                skipped += 1
                continue

            entries.append(make_entry(fpath, sentence, duration))
            idx += 1
            pbar.set_postfix(valid=idx, skipped=skipped)

        write_manifest(entries, out_path)
        total_valid += len(entries)
        logging.info(f"  cv-zh/{cv_split}: {len(entries):,} valid, {skipped:,} skipped")

    return total_valid


# ---------------------------------------------------------------------------
# 2. AISHELL-1 (OpenSLR 33)
# ---------------------------------------------------------------------------

AISHELL1_URL = "https://www.openslr.org/resources/33/data_aishell.tgz"

def download_aishell1(output_dir, max_samples=None, resume=False):
    """Download AISHELL-1 from OpenSLR 33, extract, build manifests."""
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "train.json")

    if not resume and os.path.exists(manifest_path):
        n = sum(1 for _ in open(manifest_path))
        if n > 0:
            logging.info(f"  aishell1: {manifest_path} exists ({n:,} entries), skipping.")
            return

    archive_path = os.path.join(output_dir, "data_aishell.tgz")
    extract_dir = os.path.join(output_dir, "_extracted")

    download_file(AISHELL1_URL, archive_path)

    transcript_path = os.path.join(extract_dir, "data_aishell", "transcript", "aishell_transcript_v0.8.txt")
    if not os.path.exists(transcript_path):
        logging.info(f"  Extracting {archive_path}...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)

        wav_tgz = os.path.join(extract_dir, "data_aishell", "wav", "*.tar.gz")
        for tgz in sorted(glob.glob(os.path.join(extract_dir, "data_aishell", "wav", "*.tar.gz"))):
            logging.info(f"    Extracting {os.path.basename(tgz)}...")
            with tarfile.open(tgz, "r:gz") as tar:
                tar.extractall(os.path.join(extract_dir, "data_aishell", "wav"))

    transcripts = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                uid, text = parts
                transcripts[uid] = text.replace(" ", "")

    logging.info(f"  Loaded {len(transcripts):,} transcripts")

    wav_base = os.path.join(extract_dir, "data_aishell", "wav")
    split_map = {"train": "train.json", "dev": "valid.json", "test": "test.json"}

    total_valid = 0
    for split_name, out_name in split_map.items():
        split_dir = os.path.join(wav_base, split_name)
        if not os.path.isdir(split_dir):
            logging.warning(f"  aishell1: split dir not found: {split_dir}")
            continue

        wav_files = sorted(glob.glob(os.path.join(split_dir, "**", "*.wav"), recursive=True))
        logging.info(f"  aishell1/{split_name}: {len(wav_files):,} wav files")

        entries = []
        skipped = 0
        for idx, wav_path in enumerate(tqdm(wav_files, desc=f"  aishell1/{split_name}")):
            if max_samples and idx >= max_samples:
                break

            uid = os.path.splitext(os.path.basename(wav_path))[0]
            text = transcripts.get(uid)
            if not text:
                skipped += 1
                continue

            try:
                waveform, sr = sf.read(wav_path)
                waveform = np.array(waveform, dtype=np.float32)
            except Exception:
                skipped += 1
                continue

            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = resample_if_needed(waveform, sr)
            duration = len(waveform) / TARGET_SR

            if duration < MIN_DURATION or duration > MAX_DURATION:
                skipped += 1
                continue

            out_fname = f"aishell1_{uid}.wav"
            out_path = os.path.join(audio_dir, out_fname)
            try:
                sf.write(out_path, waveform, TARGET_SR)
            except Exception:
                skipped += 1
                continue

            entries.append(make_entry(out_path, text, duration))

        out_path = os.path.join(output_dir, out_name)
        write_manifest(entries, out_path)
        total_valid += len(entries)
        logging.info(f"  aishell1/{split_name}: {len(entries):,} valid, {skipped:,} skipped")

    return total_valid


# ---------------------------------------------------------------------------
# 3. KeSpeech (HuggingFace: TwinkStart/KeSpeech)
# ---------------------------------------------------------------------------

KESPEECH_HF = "TwinkStart/KeSpeech"

def download_kespeech(output_dir, max_samples=None, resume=False):
    """Download KeSpeech from HuggingFace (1542 hours, ~1.3M utterances).

    The HF dataset has splits: train, validation, test.
    Each item has 'audio' (decoded waveform) and 'text' fields.
    """
    from datasets import load_dataset

    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "train.json")
    if not resume and os.path.exists(manifest_path):
        n = sum(1 for _ in open(manifest_path))
        if n > 0:
            logging.info(f"  kespeech: {manifest_path} exists ({n:,} entries), skipping.")
            return

    hf_token = os.environ.get("HF_TOKEN")
    split_map = {
        "train": ("train", manifest_path),
        "validation": ("valid", os.path.join(output_dir, "valid.json")),
        "test": ("test", os.path.join(output_dir, "test.json")),
    }

    total_valid = 0
    for hf_split, (out_name, out_path) in split_map.items():
        entries = []
        idx = 0
        skipped = 0

        try:
            ds = load_dataset(KESPEECH_HF, split=hf_split, streaming=True, token=hf_token)
        except Exception as e:
            logging.warning(f"  kespeech/{hf_split}: not available ({e})")
            continue

        logging.info(f"  Streaming kespeech/{hf_split}...")
        pbar = tqdm(ds, desc=f"  kespeech/{hf_split}", leave=False)

        for item in pbar:
            if max_samples and idx >= max_samples:
                break

            audio = item.get("audio")
            text = item.get("Text", "") or item.get("text", "") or item.get("sentence", "")
            if not audio or not text:
                skipped += 1
                continue

            waveform = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = resample_if_needed(waveform, sr)

            duration = len(waveform) / TARGET_SR
            if duration < MIN_DURATION or duration > MAX_DURATION:
                skipped += 1
                continue

            fname = f"kespeech_{out_name}_{idx:07d}.wav"
            fpath = os.path.join(audio_dir, fname)
            try:
                sf.write(fpath, waveform, TARGET_SR)
            except Exception:
                skipped += 1
                continue

            entries.append(make_entry(fpath, text, duration))
            idx += 1
            pbar.set_postfix(valid=idx, skipped=skipped)

        write_manifest(entries, out_path)
        total_valid += len(entries)
        logging.info(f"  kespeech/{hf_split}: {len(entries):,} valid, {skipped:,} skipped")

    return total_valid


# ---------------------------------------------------------------------------
# 4. MagicData (OpenSLR 68)
# ---------------------------------------------------------------------------

MAGICDATA_URLS = {
    "train": "https://www.openslr.org/resources/68/train_set.tar.gz",
    "dev": "https://www.openslr.org/resources/68/dev_set.tar.gz",
    "test": "https://www.openslr.org/resources/68/test_set.tar.gz",
}

def download_magicdata(output_dir, max_samples=None, resume=False):
    """Download MagicData from OpenSLR 68 (755 hours Mandarin read speech)."""
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    dl_dir = os.path.join(output_dir, "_downloads")
    os.makedirs(dl_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "train.json")
    if not resume and os.path.exists(manifest_path):
        n = sum(1 for _ in open(manifest_path))
        if n > 0:
            logging.info(f"  magicdata: {manifest_path} exists ({n:,} entries), skipping.")
            return

    extract_dir = os.path.join(output_dir, "_extracted")
    os.makedirs(extract_dir, exist_ok=True)

    split_out = {"train": "train.json", "dev": "valid.json", "test": "test.json"}
    total_valid = 0

    for split_name, url in MAGICDATA_URLS.items():
        tgz_name = os.path.basename(url)
        tgz_path = os.path.join(dl_dir, tgz_name)

        try:
            download_file(url, tgz_path)
        except subprocess.CalledProcessError:
            logging.warning(f"  Failed to download {tgz_name}, skipping")
            continue

        split_extract = os.path.join(extract_dir, split_name)
        if not os.path.exists(split_extract):
            logging.info(f"  Extracting {tgz_name}...")
            os.makedirs(split_extract, exist_ok=True)
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(split_extract)

        transcripts = {}
        for root, dirs, files in os.walk(split_extract):
            for fn in files:
                if fn.endswith(".txt") and fn != "README.txt":
                    fp = os.path.join(root, fn)
                    with open(fp, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            parts = line.split("\t")
                            if len(parts) >= 2:
                                uid = os.path.splitext(parts[0])[0]
                                transcripts[uid] = parts[1]
                            else:
                                parts = line.split(maxsplit=1)
                                if len(parts) == 2:
                                    uid = os.path.splitext(parts[0])[0]
                                    transcripts[uid] = parts[1]

        logging.info(f"  magicdata/{split_name}: {len(transcripts):,} transcripts")

        wav_files = sorted(glob.glob(os.path.join(split_extract, "**", "*.wav"), recursive=True))
        logging.info(f"  magicdata/{split_name}: {len(wav_files):,} wav files")

        entries = []
        skipped = 0
        for idx, wav_path in enumerate(tqdm(wav_files, desc=f"  magicdata/{split_name}")):
            if max_samples and idx >= max_samples:
                break

            uid = os.path.splitext(os.path.basename(wav_path))[0]
            text = transcripts.get(uid)
            if not text:
                skipped += 1
                continue

            try:
                waveform, sr = sf.read(wav_path)
                waveform = np.array(waveform, dtype=np.float32)
            except Exception:
                skipped += 1
                continue

            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = resample_if_needed(waveform, sr)
            duration = len(waveform) / TARGET_SR
            if duration < MIN_DURATION or duration > MAX_DURATION:
                skipped += 1
                continue

            out_fname = f"magicdata_{uid}.wav"
            out_path = os.path.join(audio_dir, out_fname)
            try:
                sf.write(out_path, waveform, TARGET_SR)
            except Exception:
                skipped += 1
                continue

            entries.append(make_entry(out_path, text, duration))

        out_path = os.path.join(output_dir, split_out[split_name])
        write_manifest(entries, out_path)
        total_valid += len(entries)
        logging.info(f"  magicdata/{split_name}: {len(entries):,} valid, {skipped:,} skipped")

    return total_valid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DOWNLOAD_FNS = {
    "cv-zh": download_cv_zh,
    "aishell1": download_aishell1,
    "kespeech": download_kespeech,
    "magicdata": download_magicdata,
}


def main():
    parser = argparse.ArgumentParser(description="Download Chinese ASR datasets → NeMo manifests")
    parser.add_argument("--dataset", required=True, choices=DATASETS + ["all"],
                        help="Which dataset to download")
    parser.add_argument("--output-dir", required=True,
                        help="Output dir. For 'all', subdirs created per dataset.")
    parser.add_argument("--max-samples", type=int, help="Limit samples per split (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted download")
    args = parser.parse_args()

    start = time.time()
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    for ds_name in datasets:
        if args.dataset == "all":
            ds_dir = os.path.join(args.output_dir, ds_name.replace("-", "_"))
        else:
            ds_dir = args.output_dir

        logging.info(f"=== {ds_name} → {ds_dir} ===")
        fn = DOWNLOAD_FNS[ds_name]
        total = fn(ds_dir, max_samples=args.max_samples, resume=args.resume)
        logging.info(f"=== {ds_name}: {total or 0:,} valid samples ===\n")

    elapsed = time.time() - start
    logging.info(f"All done in {elapsed/60:.1f} min")

    info = {
        "datasets": datasets,
        "lang": "ZH",
        "lang_family": "EAST_ASIAN",
        "processing_time_minutes": round(elapsed / 60, 1),
    }
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
