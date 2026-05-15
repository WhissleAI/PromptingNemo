#!/usr/bin/env python3
"""Annotate NeMo manifests with AGE, GENDER, EMOTION inline tags using GPU models.

Uses wav2vec2-large-robust-6-ft-age-gender for age/gender classification and
superb/hubert-large-superb-er for emotion classification — same models as
whissle-annotator s04_audio_classify.

Reads each manifest line, loads the corresponding WAV file, runs inference,
and appends inline tags (AGE_*, GENDER_*, EMOTION_*) to the text field.
Writes annotated output to <manifest>.tagged or in-place with --in-place.

Usage:
  # Annotate a single manifest
  python annotate_audio_tags.py --file /mnt/nfs/data/.../train.json

  # Annotate all manifests under a directory
  python annotate_audio_tags.py --data-root /mnt/nfs/data/multilingual_v1/raw/hinglish_mucs

  # In-place update (no .tagged copy)
  python annotate_audio_tags.py --data-root /path/to/data --in-place

  # Resume from checkpoint
  python annotate_audio_tags.py --data-root /path/to/data --resume

  # Batch size for GPU (higher = more VRAM, faster)
  python annotate_audio_tags.py --data-root /path/to/data --batch-size 32
"""
import argparse
import json
import logging
import os
import re
import sys
import time
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

HAS_AGE_RE = re.compile(r'\bAGE_\S+')
HAS_GENDER_RE = re.compile(r'\bGENDER_\S+')
HAS_EMOTION_RE = re.compile(r'\bEMOTION_\S+')

# ---------------------------------------------------------------------------
# Model loading — same architecture as whissle-annotator s04_audio_classify
# ---------------------------------------------------------------------------

_age_gender_model = None
_age_gender_processor = None
_emotion_model = None
_emotion_extractor = None
_device = None
_model_lock = threading.Lock()


def _get_device():
    global _device
    if _device is None:
        import torch
        force_cpu = os.getenv("FORCE_CPU", "0") == "1"
        _device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    return _device


def _load_age_gender():
    global _age_gender_model, _age_gender_processor
    if _age_gender_model is not None:
        return
    with _model_lock:
        if _age_gender_model is not None:
            return

        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Model

        class ModelHead(nn.Module):
            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(getattr(config, "final_dropout", 0.1))
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, features, **kwargs):
                x = self.dropout(features)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                return self.out_proj(x)

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            all_tied_weights_keys = {}

            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.init_weights()

            def forward(self, input_values):
                outputs = self.wav2vec2(input_values)
                hidden = torch.mean(outputs[0], dim=1)
                return hidden, self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

        model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
        device = _get_device()
        logging.info(f"Loading age/gender model on {device}")
        _age_gender_processor = Wav2Vec2Processor.from_pretrained(model_name)
        _age_gender_model = AgeGenderModel.from_pretrained(model_name).to(device)
        _age_gender_model.eval()


def _load_emotion():
    global _emotion_model, _emotion_extractor
    if _emotion_model is not None:
        return
    with _model_lock:
        if _emotion_model is not None:
            return

        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

        model_name = "superb/hubert-large-superb-er"
        device = _get_device()
        logging.info(f"Loading emotion model on {device}")
        _emotion_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        _emotion_model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)
        _emotion_model.eval()


def predict_age_gender(audio: np.ndarray, sr: int = 16000):
    import torch
    _load_age_gender()
    device = _get_device()
    y = _age_gender_processor(audio, sampling_rate=sr)
    y = y["input_values"][0].reshape(1, -1)
    y = torch.from_numpy(y).to(device)
    with torch.no_grad():
        _, age_logits, gender_logits = _age_gender_model(y)
        age = float(age_logits.detach().cpu().numpy()[0][0])
        gender = int(np.argmax(gender_logits.detach().cpu().numpy()))
    return age, gender


def predict_emotion(audio: np.ndarray, sr: int = 16000):
    import torch
    _load_emotion()
    if len(audio) < sr:
        return "NEUTRAL", 0.5
    inputs = _emotion_extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(_get_device()) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _emotion_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence = probs.max(-1).values.item()
        idx = probs.argmax(-1).item()
    label = _emotion_model.config.id2label.get(idx, "NEUTRAL").upper()
    return label, confidence


def age_to_bucket(age_raw: float) -> str:
    actual_age = round(age_raw * 100, 2)
    if actual_age < 18:
        return "AGE_0_18"
    elif actual_age < 30:
        return "AGE_18_30"
    elif actual_age < 45:
        return "AGE_30_45"
    elif actual_age < 60:
        return "AGE_45_60"
    else:
        return "AGE_60PLUS"


def gender_to_token(gender_idx: int) -> str:
    return {0: "GENDER_MALE", 1: "GENDER_FEMALE"}.get(gender_idx, "GENDER_OTHER")


EMO_MAP = {
    "NEU": "EMOTION_NEUTRAL", "HAP": "EMOTION_HAPPY", "SAD": "EMOTION_SAD",
    "ANG": "EMOTION_ANGRY", "FEA": "EMOTION_FEAR", "DIS": "EMOTION_DISGUST",
    "SUR": "EMOTION_SURPRISE",
    "NEUTRAL": "EMOTION_NEUTRAL", "HAPPY": "EMOTION_HAPPY",
    "ANGRY": "EMOTION_ANGRY", "SAD": "EMOTION_SAD",
    "FEAR": "EMOTION_FEAR", "DISGUST": "EMOTION_DISGUST",
    "SURPRISE": "EMOTION_SURPRISE", "CONTEMPT": "EMOTION_CONTEMPT",
}


def emotion_to_token(emotion_label: str, confidence: float = 0.5) -> str:
    return EMO_MAP.get(emotion_label.upper(), "EMOTION_NEUTRAL")


# ---------------------------------------------------------------------------
# Manifest annotation
# ---------------------------------------------------------------------------

def already_tagged(text: str) -> bool:
    return bool(HAS_AGE_RE.search(text) and HAS_GENDER_RE.search(text) and HAS_EMOTION_RE.search(text))


def annotate_manifest(manifest_path: str, out_path: str, resume: bool = False):
    """Annotate a single manifest file with AGE/GENDER/EMOTION tags."""
    checkpoint_path = out_path + ".progress"

    lines = open(manifest_path, encoding="utf-8").readlines()
    total = len(lines)

    start_idx = 0
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            start_idx = int(f.read().strip())
        logging.info(f"  Resuming from line {start_idx}")

    if not resume and os.path.exists(out_path) and out_path != manifest_path:
        existing = sum(1 for _ in open(out_path))
        if existing >= total:
            logging.info(f"  {out_path} already complete ({existing} lines), skipping.")
            return existing, 0, 0

    mode = "a" if resume and start_idx > 0 else "w"
    annotated = 0
    skipped = 0
    errors = 0

    with open(out_path, mode, encoding="utf-8") as fout:
        for idx in range(start_idx, total):
            line = lines[idx].strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line + "\n")
                skipped += 1
                continue

            text = entry.get("text", "")

            if already_tagged(text):
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                skipped += 1
                continue

            audio_path = entry.get("audio_filepath", "")
            if not audio_path or not os.path.exists(audio_path):
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                skipped += 1
                continue

            try:
                audio, sr = sf.read(audio_path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.float32)
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    sr = 16000
            except Exception as e:
                logging.warning(f"  Audio read error {audio_path}: {e}")
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                errors += 1
                continue

            try:
                age_raw, gender_idx = predict_age_gender(audio, sr)
                age_tag = age_to_bucket(age_raw)
                gender_tag = gender_to_token(gender_idx)
            except Exception as e:
                logging.warning(f"  Age/gender error {audio_path}: {e}")
                age_tag = "AGE_18_30"
                gender_tag = "GENDER_OTHER"
                errors += 1

            try:
                emo_label, emo_conf = predict_emotion(audio, sr)
                emo_tag = emotion_to_token(emo_label, emo_conf)
            except Exception as e:
                logging.warning(f"  Emotion error {audio_path}: {e}")
                emo_tag = "EMOTION_NEUTRAL"
                errors += 1

            tags = f"{age_tag} {gender_tag} {emo_tag}"
            entry["text"] = f"{text} {tags}"
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            annotated += 1

            if (idx + 1) % 1000 == 0:
                with open(checkpoint_path, "w") as cp:
                    cp.write(str(idx + 1))
                logging.info(f"  [{idx+1}/{total}] annotated={annotated}, skipped={skipped}, errors={errors}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    logging.info(f"  Done: {annotated} annotated, {skipped} skipped, {errors} errors → {out_path}")
    return annotated, skipped, errors


def find_manifests(data_root: str) -> list[str]:
    manifests = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.endswith(".json") and not f.endswith(".tagged") and f != "dataset_info.json":
                manifests.append(os.path.join(root, f))
    return sorted(manifests)


def main():
    parser = argparse.ArgumentParser(description="Annotate manifests with AGE/GENDER/EMOTION from audio models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Single manifest file to annotate")
    group.add_argument("--data-root", help="Root directory to scan for manifests")
    parser.add_argument("--in-place", action="store_true", help="Overwrite original manifests")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Count samples needing annotation only")
    args = parser.parse_args()

    start = time.time()

    if args.file:
        manifests = [args.file]
    else:
        manifests = find_manifests(args.data_root)

    if not manifests:
        logging.error("No manifest files found.")
        sys.exit(1)

    logging.info(f"Found {len(manifests)} manifest(s)")

    if args.dry_run:
        for mf in manifests:
            lines = open(mf, encoding="utf-8").readlines()
            need = sum(1 for l in lines if l.strip() and not already_tagged(json.loads(l).get("text", "")))
            logging.info(f"  {mf}: {need}/{len(lines)} need annotation")
        return

    _load_age_gender()
    _load_emotion()
    logging.info(f"Models loaded on {_get_device()}")

    total_annotated = 0
    total_skipped = 0
    total_errors = 0

    for mf in manifests:
        logging.info(f"=== {mf} ===")
        out = mf if args.in_place else mf + ".tagged"
        a, s, e = annotate_manifest(mf, out, resume=args.resume)
        total_annotated += a
        total_skipped += s
        total_errors += e

    elapsed = time.time() - start
    logging.info(f"All done in {elapsed/60:.1f} min — {total_annotated} annotated, {total_skipped} skipped, {total_errors} errors")


if __name__ == "__main__":
    main()
