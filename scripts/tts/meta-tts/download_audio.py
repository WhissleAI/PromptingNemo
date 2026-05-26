#!/usr/bin/env python3
"""
Download public audio datasets required for META-TTS training.

Sources (matching Meta_STT_EURO_Set1 audio_filepath references):
  - CommonVoice 15.0: de, en, es, fr, it, pt
  - Multilingual LibriSpeech (MLS): de, en, es, fr, it, pt
  - LibriSpeech: train-other-500 (English)
  - People's Speech (English)

Usage:
    python download_audio.py --audio-root /mnt/nfs/data/tts_audio --languages en,de,es,fr,it,pt
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

MLS_URLS = {
    "en": "https://dl.fbaipublicfiles.com/mls/mls_english.tar.gz",
    "de": "https://dl.fbaipublicfiles.com/mls/mls_german.tar.gz",
    "es": "https://dl.fbaipublicfiles.com/mls/mls_spanish.tar.gz",
    "fr": "https://dl.fbaipublicfiles.com/mls/mls_french.tar.gz",
    "it": "https://dl.fbaipublicfiles.com/mls/mls_italian.tar.gz",
    "pt": "https://dl.fbaipublicfiles.com/mls/mls_portuguese.tar.gz",
}

MLS_LANG_NAMES = {
    "en": "english", "de": "german", "es": "spanish",
    "fr": "french", "it": "italian", "pt": "portuguese",
}

LIBRISPEECH_URLS = {
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
}

CV_LANG_CODES = {
    "en": "en", "de": "de", "es": "es",
    "fr": "fr", "it": "it", "pt": "pt",
}


def download_file(url, dest_path, desc=""):
    """Download a file with wget (resume support)."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  Already exists: {dest}")
        return True

    partial = dest.with_suffix(dest.suffix + ".partial")
    print(f"  Downloading {desc or url}")
    print(f"    → {dest}")

    try:
        cmd = [
            "wget", "-c", "--progress=dot:giga",
            "-O", str(partial), url
        ]
        subprocess.run(cmd, check=True)
        partial.rename(dest)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Download failed: {e}")
        return False
    except FileNotFoundError:
        try:
            cmd = [
                "curl", "-L", "-C", "-",
                "--progress-bar",
                "-o", str(partial), url
            ]
            subprocess.run(cmd, check=True)
            partial.rename(dest)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  Download failed (curl): {e}")
            return False


def extract_tar(tar_path, extract_to, desc=""):
    """Extract a tar.gz file."""
    marker = Path(extract_to) / f".extracted_{Path(tar_path).stem}"
    if marker.exists():
        print(f"  Already extracted: {desc or tar_path}")
        return True

    print(f"  Extracting {desc or tar_path}")
    print(f"    → {extract_to}")

    try:
        subprocess.run(
            ["tar", "-xzf", str(tar_path), "-C", str(extract_to)],
            check=True
        )
        marker.touch()
        return True
    except subprocess.CalledProcessError:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        marker.touch()
        return True


def download_mls(audio_root, languages):
    """Download Multilingual LibriSpeech datasets."""
    mls_dir = Path(audio_root) / "mls"
    mls_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        if lang not in MLS_URLS:
            continue

        lang_name = MLS_LANG_NAMES[lang]
        target_dir = mls_dir / f"mls_{lang_name}"

        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"  MLS {lang_name}: already present at {target_dir}")
            continue

        url = MLS_URLS[lang]
        tar_file = mls_dir / f"mls_{lang_name}.tar.gz"

        print(f"\n--- MLS {lang_name} ---")
        if download_file(url, tar_file, f"MLS {lang_name}"):
            extract_tar(tar_file, mls_dir, f"MLS {lang_name}")


def download_librispeech(audio_root):
    """Download LibriSpeech train-other-500."""
    ls_dir = Path(audio_root) / "librispeech-en"
    ls_dir.mkdir(parents=True, exist_ok=True)

    for split, url in LIBRISPEECH_URLS.items():
        target_dir = ls_dir / split
        if target_dir.exists() and any(target_dir.rglob("*.flac")):
            print(f"  LibriSpeech {split}: already present")
            continue

        tar_file = ls_dir / f"{split}.tar.gz"
        print(f"\n--- LibriSpeech {split} ---")
        if download_file(url, tar_file, f"LibriSpeech {split}"):
            extract_tar(tar_file, ls_dir, f"LibriSpeech {split}")
            extracted = ls_dir / "LibriSpeech" / split
            if extracted.exists() and not target_dir.exists():
                shutil.move(str(extracted), str(target_dir))


def download_commonvoice(audio_root, languages):
    """Download CommonVoice 15.0 datasets.

    CommonVoice requires agreement to terms. This function attempts
    download via the HuggingFace datasets library (which handles auth).
    If that fails, it prints manual download instructions.
    """
    cv_dir = Path(audio_root) / "cv" / "cv-corpus-15.0-2023-09-08"
    cv_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        if lang not in CV_LANG_CODES:
            continue

        clips_dir = cv_dir / lang / "clips"
        if clips_dir.exists() and any(clips_dir.glob("*.mp3")):
            print(f"  CommonVoice {lang}: already present at {clips_dir}")
            continue

        print(f"\n--- CommonVoice {lang} ---")
        print(f"  Attempting download via HuggingFace datasets library...")

        try:
            from datasets import load_dataset
            ds = load_dataset(
                "mozilla-foundation/common_voice_15_0",
                lang,
                split="train",
                trust_remote_code=True,
                cache_dir=str(Path(audio_root) / ".hf_cache"),
            )
            clips_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Downloaded {len(ds)} samples for {lang}")
            print(f"  Audio files cached by HuggingFace. Creating symlinks...")

            for i, sample in enumerate(ds):
                audio_path = sample.get("path", "")
                if audio_path and os.path.exists(audio_path):
                    dest = clips_dir / os.path.basename(audio_path)
                    if not dest.exists():
                        os.symlink(audio_path, dest)

                if (i + 1) % 10000 == 0:
                    print(f"    Processed {i + 1} samples")

        except Exception as e:
            print(f"  HuggingFace download failed: {e}")
            print(f"\n  === Manual Download Required ===")
            print(f"  1. Go to https://commonvoice.mozilla.org/en/datasets")
            print(f"  2. Download Common Voice Corpus 15.0 for: {lang}")
            print(f"  3. Extract to: {cv_dir / lang}/")
            print(f"  4. Ensure clips are at: {clips_dir}/")


def download_peoples_speech(audio_root):
    """Download People's Speech dataset."""
    ps_dir = Path(audio_root) / "peoples_speech"
    ps_dir.mkdir(parents=True, exist_ok=True)

    if any(ps_dir.rglob("*.flac")) or any(ps_dir.rglob("*.wav")):
        print(f"  People's Speech: already present at {ps_dir}")
        return

    print(f"\n--- People's Speech ---")
    print(f"  Attempting download via HuggingFace...")

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "MLCommons/peoples_speech",
            "clean",
            split="train",
            trust_remote_code=True,
            cache_dir=str(Path(audio_root) / ".hf_cache"),
        )
        print(f"  Downloaded {len(ds)} samples")
    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  Manual: https://huggingface.co/datasets/MLCommons/peoples_speech")


def print_summary(audio_root):
    """Print summary of downloaded data."""
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(audio_root):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    audio_root = Path(audio_root)

    sources = {
        "MLS": audio_root / "mls",
        "LibriSpeech": audio_root / "librispeech-en",
        "CommonVoice": audio_root / "cv",
        "People's Speech": audio_root / "peoples_speech",
    }

    for name, path in sources.items():
        if path.exists():
            count = sum(1 for _ in path.rglob("*") if _.is_file() and not _.name.startswith("."))
            print(f"  {name}: {count:,} files at {path}")
        else:
            print(f"  {name}: NOT DOWNLOADED")

    print(f"\n  Total disk usage: {total_size / (1024**3):.1f} GB")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download audio datasets for META-TTS training"
    )
    parser.add_argument(
        "--audio-root", required=True,
        help="Root directory for audio downloads (e.g., /mnt/nfs/data/tts_audio)"
    )
    parser.add_argument(
        "--languages", default="en,de,es,fr,it,pt",
        help="Comma-separated language codes to download"
    )
    parser.add_argument(
        "--sources", default="mls,librispeech,commonvoice,peoples_speech",
        help="Comma-separated data sources to download"
    )
    parser.add_argument(
        "--skip-large", action="store_true",
        help="Skip very large downloads (MLS English ~120GB)"
    )
    args = parser.parse_args()

    languages = [l.strip() for l in args.languages.split(",")]
    sources = [s.strip().lower() for s in args.sources.split(",")]
    audio_root = Path(args.audio_root)
    audio_root.mkdir(parents=True, exist_ok=True)

    print(f"Audio root: {audio_root}")
    print(f"Languages: {languages}")
    print(f"Sources: {sources}")
    print()

    if "mls" in sources:
        print("\n" + "=" * 40)
        print("Downloading Multilingual LibriSpeech")
        print("=" * 40)
        download_mls(audio_root, languages)

    if "librispeech" in sources and "en" in languages:
        print("\n" + "=" * 40)
        print("Downloading LibriSpeech")
        print("=" * 40)
        download_librispeech(audio_root)

    if "commonvoice" in sources:
        print("\n" + "=" * 40)
        print("Downloading CommonVoice 15.0")
        print("=" * 40)
        download_commonvoice(audio_root, languages)

    if "peoples_speech" in sources and "en" in languages:
        print("\n" + "=" * 40)
        print("Downloading People's Speech")
        print("=" * 40)
        download_peoples_speech(audio_root)

    print_summary(audio_root)
    print("\nDone!")


if __name__ == "__main__":
    main()
