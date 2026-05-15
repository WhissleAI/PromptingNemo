#!/usr/bin/env python3
"""
benchmark_hindi.py — Evaluate Whissle v18 vs Deepgram Nova-2 vs Gemini-2.5-flash vs Sarvam Saaras v3
on Hindi in-house validation (5K samples) + FLEURS Hindi test set.

Usage:
    python benchmark_hindi.py \
        --model /mnt/nfs/experiments/hi-vakyansh-meta-v18-223k/checkpoints/hi-vakyansh-meta-v18-223k--val_wer=0.1773-epoch=0.ckpt \
        --manifest /mnt/nfs/data/hi-vakyansh-meta-v18-223k/valid.cleaned.json \
        --output-dir /mnt/nfs/experiments/hi-vakyansh-meta-v18-223k/benchmark_results \
        --deepgram-key <key> \
        --gemini-key <key> \
        --sarvam-key <key> \
        [--in-house-samples 5000] [--batch-size 32] [--cpu]
"""
import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np

# ---------------------------------------------------------------------------
# Reusable patterns from eval_full.py
# ---------------------------------------------------------------------------
TAG_PATTERNS = {
    "AGE":     re.compile(r"\b(AGE_\S+)\b"),
    "GENDER":  re.compile(r"\b(GENDER_\S+)\b"),
    "EMOTION": re.compile(r"\b(EMOTION_\S+)\b"),
    "INTENT":  re.compile(r"\b(INTENT_\S+)\b"),
}

ENTITY_RE = re.compile(r"\b(ENTITY_\S+)\b")
ALL_TAG_RE = re.compile(
    r"\b(?:AGE_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+|ENTITY_\S+|KEYWORD_\S+|LANG_\S+|DIALECT_\S+|END)\b"
)


def strip_tags(text: str) -> str:
    return " ".join(ALL_TAG_RE.sub("", text).split())


def _edit_distance(ref_tokens, hyp_tokens):
    n, m = len(ref_tokens), len(hyp_tokens)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m]


def compute_wer(refs, hyps):
    total_errors = total_words = 0
    for ref, hyp in zip(refs, hyps):
        ref_w = ref.split()
        total_words += len(ref_w)
        total_errors += _edit_distance(ref_w, hyp.split())
    return total_errors / max(total_words, 1)


def compute_cer(refs, hyps):
    total_errors = total_chars = 0
    for ref, hyp in zip(refs, hyps):
        ref_c = list(ref.replace(" ", ""))
        hyp_c = list(hyp.replace(" ", ""))
        total_chars += len(ref_c)
        total_errors += _edit_distance(ref_c, hyp_c)
    return total_errors / max(total_chars, 1)


def _classification_metrics(y_true, y_pred, labels):
    label_set = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(label_set)}
    n = len(label_set)
    conf = [[0] * n for _ in range(n)]

    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(t)
        pi = label_to_idx.get(p)
        if ti is not None and pi is not None:
            conf[ti][pi] += 1

    per_class = {}
    for i, label in enumerate(label_set):
        tp = conf[i][i]
        fp = sum(conf[j][i] for j in range(n)) - tp
        fn = sum(conf[i][j] for j in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(conf[i])
        per_class[label] = {"precision": prec, "recall": rec, "f1": f1, "support": support}

    total_correct = sum(conf[i][i] for i in range(n))
    total_samples = sum(sum(row) for row in conf)
    accuracy = total_correct / max(total_samples, 1)

    macro_f1 = sum(v["f1"] for v in per_class.values()) / max(len(per_class), 1)
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / max(total_samples, 1)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": conf,
        "labels": label_set,
        "total_samples": total_samples,
    }


def extract_entity_spans(text):
    spans = []
    tokens = text.split()
    i = 0
    while i < len(tokens):
        if ENTITY_RE.match(tokens[i]):
            etype = tokens[i]
            surface = []
            i += 1
            while i < len(tokens) and tokens[i] != "END" and not ENTITY_RE.match(tokens[i]):
                if not any(tokens[i].startswith(p) for p in ["AGE_", "GENDER_", "EMOTION_", "INTENT_"]):
                    surface.append(tokens[i])
                i += 1
            if i < len(tokens) and tokens[i] == "END":
                i += 1
            spans.append((etype, " ".join(surface)))
        else:
            i += 1
    return spans


# ---------------------------------------------------------------------------
# Whissle v18 inference
# ---------------------------------------------------------------------------
def run_whissle_inference(model_path, entries, device, batch_size):
    import torch
    import soundfile as sf

    project_root = os.environ.get('PROMPTINGNEMO_ROOT', '/mnt/nfs/code/PromptingNemo')
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from promptingnemo.models.ctc_model import CustomEncDecCTCModelBPE

    sys.path.insert(0, os.path.join(project_root, 'scripts', 'asr'))
    from meta_asr.tag_classifier import TrailingTagClassifier, build_trailing_tag_maps

    print(f"Loading Whissle v18 model from {model_path}...")
    is_ckpt = model_path.endswith('.ckpt')

    if is_ckpt:
        nemo_path = _build_nemo_from_ckpt(model_path, project_root)
    else:
        nemo_path = model_path

    model = CustomEncDecCTCModelBPE.restore_from(nemo_path, map_location='cpu', strict=False)
    if device == 'cuda':
        model = model.cuda()
    model.eval()

    tag_maps = _setup_tag_classifier(model, nemo_path, torch, TrailingTagClassifier, build_trailing_tag_maps)
    has_tag_cls = tag_maps is not None

    results = []
    total_latency = 0.0

    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        waveforms = []
        lengths = []

        for entry in batch:
            audio_data, sr = sf.read(entry['audio_filepath'], dtype='float32')
            waveform = torch.from_numpy(audio_data)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=-1)
            if sr != 16000:
                import librosa
                waveform = torch.from_numpy(
                    librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=16000)
                )
            waveforms.append(waveform)
            lengths.append(waveform.shape[0])

        max_len = max(lengths)
        padded = torch.zeros(len(batch), max_len)
        for j, w in enumerate(waveforms):
            padded[j, :w.shape[0]] = w

        input_signal = padded.to(device)
        input_length = torch.tensor(lengths, device=device)

        t0 = time.time()
        with torch.no_grad():
            log_probs, encoded_len, _ = model.forward(
                input_signal=input_signal,
                input_signal_length=input_length,
            )
            decode_result = model.decoding.ctc_decoder_predictions_tensor(
                log_probs, decoder_lengths=encoded_len, return_hypotheses=False,
            )
            best_hyps = decode_result[0] if isinstance(decode_result, tuple) else decode_result

            tag_preds_batch = {}
            if has_tag_cls and hasattr(model, '_last_encoder_output') and model._last_encoder_output is not None:
                encoder_out = model._last_encoder_output
                B_enc, D_enc, T_enc = encoder_out.shape
                time_mask = torch.arange(T_enc, device=encoder_out.device).unsqueeze(0) < encoded_len.unsqueeze(1)
                enc_btd = encoder_out.transpose(1, 2)
                mask_3d = time_mask.unsqueeze(-1).float()
                pooled = (enc_btd * mask_3d).sum(dim=1) / mask_3d.sum(dim=1).clamp(min=1)
                tag_logits = model.tag_classifier(pooled)
                for cat_name in tag_maps["category_names"]:
                    class_ids = tag_logits[cat_name].argmax(dim=-1).cpu().tolist()
                    tag_preds_batch[cat_name] = [
                        tag_maps["id_to_label"][cat_name].get(cid, "NONE") for cid in class_ids
                    ]

        batch_latency = time.time() - t0
        total_latency += batch_latency

        for j, entry in enumerate(batch):
            raw = best_hyps[j] if j < len(best_hyps) else ""
            pred_text = raw.text if hasattr(raw, 'text') else str(raw)
            result = {
                "audio_filepath": entry['audio_filepath'],
                "reference": entry.get('text', ''),
                "prediction": pred_text,
                "duration": entry.get('duration', 0),
                "latency": batch_latency / len(batch),
            }
            if tag_preds_batch:
                result["tag_preds"] = {cat: tag_preds_batch[cat][j] for cat in tag_preds_batch}
            results.append(result)

        done = min(i + batch_size, len(entries))
        if done % 500 < batch_size or done == len(entries):
            print(f"  Whissle [{done}/{len(entries)}]")

    avg_latency = total_latency / max(len(entries), 1)
    return results, avg_latency


def _build_nemo_from_ckpt(ckpt_path, project_root):
    """Build a .nemo archive from a .ckpt file for restore_from()."""
    import torch
    nemo_path = ckpt_path.replace('.ckpt', '.nemo')
    if os.path.exists(nemo_path):
        return nemo_path

    print(f"Building .nemo from checkpoint {ckpt_path}...")
    config_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_candidates = [
        os.path.join(config_dir, "hparams.yaml"),
        os.path.join(config_dir, "config.yaml"),
    ]
    config_path = None
    for c in config_candidates:
        if os.path.exists(c):
            config_path = c
            break

    if config_path is None:
        raise FileNotFoundError(f"Cannot find config YAML in {config_dir}")

    import tarfile
    import io

    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)

    with tarfile.open(nemo_path, 'w:') as tar:
        info = tarfile.TarInfo(name='model_config.yaml')
        with open(config_path, 'rb') as cf:
            config_bytes = cf.read()
        info.size = len(config_bytes)
        tar.addfile(info, io.BytesIO(config_bytes))

        info2 = tarfile.TarInfo(name='model_weights.ckpt')
        info2.size = buf.getbuffer().nbytes
        tar.addfile(info2, buf)

    print(f"Built .nemo at {nemo_path}")
    return nemo_path


def _setup_tag_classifier(model, nemo_path, torch, TrailingTagClassifier, build_trailing_tag_maps):
    import tarfile, io

    tc_state = {}
    with tarfile.open(nemo_path) as t:
        for m in t.getmembers():
            if 'model_weights' in m.name:
                f = t.extractfile(m)
                state = torch.load(io.BytesIO(f.read()), map_location='cpu')
                tc_state = {k: v for k, v in state.items() if 'tag_classifier' in k}
                break

    if not tc_state:
        print("  No tag classifier weights found in .nemo")
        return None

    category_sizes = {}
    for key, tensor in tc_state.items():
        if key.endswith('.weight'):
            cat_name = key.split('.')[2]
            category_sizes[cat_name] = tensor.shape[0]

    vocab = model.decoder.vocabulary
    encoder_dim = model.encoder.d_model

    classifier = TrailingTagClassifier(encoder_dim, category_sizes)
    classifier.load_state_dict({k.replace('tag_classifier.', ''): v for k, v in tc_state.items()})
    classifier.eval()

    device = next(model.parameters()).device
    classifier = classifier.to(device)
    model.tag_classifier = classifier

    def _capture_encoder_output(module, input, output):
        if isinstance(output, tuple):
            model._last_encoder_output = output[0]
        else:
            model._last_encoder_output = output
    model._encoder_hook = model.encoder.register_forward_hook(_capture_encoder_output)
    model._last_encoder_output = None

    _, category_to_id, _ = build_trailing_tag_maps(vocab, categories=sorted(category_sizes.keys()))

    id_to_label = {}
    for cat_name in sorted(category_sizes.keys()):
        cat_map = category_to_id.get(cat_name, {})
        reverse = {0: "NONE"}
        for vocab_id, class_idx in cat_map.items():
            token = vocab[vocab_id].lstrip('▁')
            reverse[class_idx] = token
        id_to_label[cat_name] = reverse

    category_names = sorted(category_sizes.keys())
    print(f"  Tag classifier set up: {category_names}")
    return {"category_names": category_names, "id_to_label": id_to_label}


# ---------------------------------------------------------------------------
# Deepgram inference
# ---------------------------------------------------------------------------
def _deepgram_single(entry, api_key):
    import requests
    audio_path = entry['audio_filepath']
    for attempt in range(3):
        try:
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            t0 = time.time()
            resp = requests.post(
                "https://api.deepgram.com/v1/listen",
                params={"model": "nova-2", "language": "hi", "punctuate": "true"},
                headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
                data=audio_bytes,
                timeout=30,
            )
            latency = time.time() - t0
            if resp.status_code == 200:
                data = resp.json()
                transcript = ""
                channels = data.get("results", {}).get("channels", [])
                if channels:
                    alts = channels[0].get("alternatives", [])
                    if alts:
                        transcript = alts[0].get("transcript", "")
                return {"audio_filepath": audio_path, "reference": entry.get('text', ''),
                        "prediction": transcript, "duration": entry.get('duration', 0), "latency": latency, "error": False}
            elif resp.status_code == 429:
                time.sleep((attempt + 1) * 5)
            else:
                return {"audio_filepath": audio_path, "reference": entry.get('text', ''),
                        "prediction": "", "duration": entry.get('duration', 0), "latency": 0, "error": True}
        except Exception:
            if attempt == 2:
                return {"audio_filepath": audio_path, "reference": entry.get('text', ''),
                        "prediction": "", "duration": entry.get('duration', 0), "latency": 0, "error": True}
            time.sleep((attempt + 1) * 2)
    return {"audio_filepath": audio_path, "reference": entry.get('text', ''),
            "prediction": "", "duration": entry.get('duration', 0), "latency": 0, "error": True}


def run_deepgram_inference(entries, api_key, workers=5):
    print(f"Running Deepgram Nova-2 inference on {len(entries)} samples ({workers} parallel)...")
    results = [None] * len(entries)
    errors = 0
    done_count = 0
    lock = Lock()

    def _worker(idx_entry):
        idx, entry = idx_entry
        return idx, _deepgram_single(entry, api_key)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, (i, e)): i for i, e in enumerate(entries)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            with lock:
                done_count_local = sum(1 for r in results if r is not None)
                err_local = sum(1 for r in results if r is not None and r.get("error"))
                if done_count_local % 200 == 0 or done_count_local == len(entries):
                    print(f"  Deepgram [{done_count_local}/{len(entries)}] (errors: {err_local})")

    errors = sum(1 for r in results if r.get("error"))
    for r in results:
        r.pop("error", None)
    total_latency = sum(r.get('latency', 0) for r in results)
    avg_latency = total_latency / max(len(results), 1)
    print(f"  Deepgram done: {len(results)} samples, {errors} errors, avg latency {avg_latency:.3f}s")
    return results, avg_latency


# ---------------------------------------------------------------------------
# Gemini inference
# ---------------------------------------------------------------------------
def _gemini_single(entry, api_key):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    audio_path = entry['audio_filepath']
    for attempt in range(3):
        try:
            audio_file = genai.upload_file(audio_path)

            t0 = time.time()
            response = model.generate_content(
                [
                    audio_file,
                    "Transcribe this Hindi audio accurately. Output only the Devanagari transcription, nothing else.",
                ],
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=500,
                ),
            )
            latency = time.time() - t0

            transcript = response.text.strip() if response.text else ""
            transcript = transcript.replace("\n", " ").strip()

            return {"audio_filepath": audio_path, "reference": entry.get('text', ''),
                    "prediction": transcript, "duration": entry.get('duration', 0),
                    "latency": latency, "error": False}

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                time.sleep((attempt + 1) * 10)
            elif attempt == 2:
                return {"audio_filepath": audio_path, "reference": entry.get('text', ''),
                        "prediction": "", "duration": entry.get('duration', 0),
                        "latency": 0, "error": True}
            else:
                time.sleep((attempt + 1) * 3)

    return {"audio_filepath": audio_path, "reference": entry.get('text', ''),
            "prediction": "", "duration": entry.get('duration', 0),
            "latency": 0, "error": True}


def run_gemini_inference(entries, api_key, workers=5):
    print(f"Running Gemini 2.5 Flash inference on {len(entries)} samples ({workers} parallel)...")
    results = [None] * len(entries)
    lock = Lock()

    def _worker(idx_entry):
        idx, entry = idx_entry
        return idx, _gemini_single(entry, api_key)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, (i, e)): i for i, e in enumerate(entries)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            with lock:
                done_count = sum(1 for r in results if r is not None)
                err_count = sum(1 for r in results if r is not None and r.get("error"))
                if done_count % 200 == 0 or done_count == len(entries):
                    print(f"  Gemini [{done_count}/{len(entries)}] (errors: {err_count})")

    errors = sum(1 for r in results if r.get("error"))
    for r in results:
        r.pop("error", None)
    total_latency = sum(r.get('latency', 0) for r in results)
    avg_latency = total_latency / max(len(results), 1)
    print(f"  Gemini done: {len(results)} samples, {errors} errors, avg latency {avg_latency:.3f}s")
    return results, avg_latency


# ---------------------------------------------------------------------------
# Sarvam inference
# ---------------------------------------------------------------------------
def _sarvam_single(idx, entry, api_key):
    from sarvamai import SarvamAI
    client = SarvamAI(api_subscription_key=api_key)
    audio_path = entry['audio_filepath']
    max_retries = 3
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            with open(audio_path, 'rb') as af:
                resp = client.speech_to_text.transcribe(
                    file=af,
                    model="saaras:v3",
                    mode="transcribe",
                    language_code="hi-IN",
                )
            latency = time.time() - t0
            transcript = (resp.transcript or "").replace("\n", " ").strip()
            return idx, {
                "audio_filepath": audio_path,
                "reference": entry.get('text', ''),
                "prediction": transcript,
                "duration": entry.get('duration', 0),
                "latency": latency,
            }, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return idx, None, str(e)


def run_sarvam_inference(entries, api_key, concurrency=5):
    print(f"Running Sarvam Saaras v3 inference on {len(entries)} samples ({concurrency} parallel)...")

    results = [None] * len(entries)
    total_errors = 0
    done_count = 0
    lock = Lock()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_sarvam_single, i, entry, api_key): i
            for i, entry in enumerate(entries)
        }
        for future in as_completed(futures):
            idx, result, error = future.result()
            with lock:
                done_count += 1
                if result:
                    results[idx] = result
                else:
                    results[idx] = {
                        "audio_filepath": entries[idx]['audio_filepath'],
                        "reference": entries[idx].get('text', ''),
                        "prediction": "",
                        "duration": entries[idx].get('duration', 0),
                        "latency": 0,
                    }
                    total_errors += 1
                if done_count % 500 == 0 or done_count == len(entries):
                    print(f"  Sarvam progress: {done_count}/{len(entries)} ({total_errors} errors)")

    # Fill any remaining None entries
    for i in range(len(results)):
        if results[i] is None:
            results[i] = {
                "audio_filepath": entries[i]['audio_filepath'],
                "reference": entries[i].get('text', ''),
                "prediction": "",
                "duration": entries[i].get('duration', 0),
                "latency": 0,
            }
            total_errors += 1

    total_latency = sum(r.get('latency', 0) for r in results)
    avg_latency = total_latency / max(len(results), 1)
    print(f"  Sarvam done: {len(results)} samples, {total_errors} errors, avg latency {avg_latency:.3f}s")
    return results, avg_latency


# ---------------------------------------------------------------------------
# FLEURS Hindi loader
# ---------------------------------------------------------------------------
def load_fleurs_hindi(manifest_path=None, output_dir=None):
    if manifest_path and os.path.exists(manifest_path):
        print(f"Loading FLEURS Hindi from existing manifest: {manifest_path}")
        entries = []
        with open(manifest_path) as f:
            for line in f:
                entries.append(json.loads(line))
        print(f"  FLEURS Hindi: {len(entries)} samples")
        return entries

    from datasets import load_dataset
    import soundfile as sf

    print("Loading FLEURS Hindi test set from HuggingFace...")
    ds = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)
    print(f"  FLEURS Hindi: {len(ds)} test samples")

    audio_dir = os.path.join(output_dir, "fleurs_audio")
    os.makedirs(audio_dir, exist_ok=True)

    entries = []
    for idx in range(len(ds)):
        item = ds[idx]
        audio = item["audio"]
        waveform = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        if sr != 16000:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000

        audio_path = os.path.join(audio_dir, f"fleurs_hi_{idx:05d}.wav")
        sf.write(audio_path, waveform, sr)

        entries.append({
            "audio_filepath": audio_path,
            "text": item["transcription"].strip(),
            "duration": len(waveform) / sr,
        })

    print(f"  Prepared {len(entries)} FLEURS Hindi samples")
    return entries


# ---------------------------------------------------------------------------
# Compute metrics for a single system on a single test set
# ---------------------------------------------------------------------------
def compute_system_metrics(results, system_name, has_tags=False):
    refs_clean = []
    preds_clean = []
    tag_true = defaultdict(list)
    tag_pred = defaultdict(list)
    entity_true_spans = []
    entity_pred_spans = []
    latencies = []

    for entry in results:
        ref = entry.get('reference', '')
        pred = entry.get('prediction', '')

        refs_clean.append(strip_tags(ref))
        preds_clean.append(strip_tags(pred))
        latencies.append(entry.get('latency', 0))

        if has_tags:
            for tag_name, pattern in TAG_PATTERNS.items():
                m_ref = pattern.search(ref)
                ref_label = m_ref.group(1) if m_ref else "NONE"

                if "tag_preds" in entry and tag_name in entry.get("tag_preds", {}):
                    pred_label = entry["tag_preds"][tag_name]
                else:
                    m_pred = pattern.search(pred)
                    pred_label = m_pred.group(1) if m_pred else "NONE"

                tag_true[tag_name].append(ref_label)
                tag_pred[tag_name].append(pred_label)

            entity_true_spans.append(set(extract_entity_spans(ref)))
            entity_pred_spans.append(set(extract_entity_spans(pred)))

    wer_val = compute_wer(refs_clean, preds_clean)
    cer_val = compute_cer(refs_clean, preds_clean)
    avg_latency = sum(latencies) / max(len(latencies), 1)

    metrics = {
        "system": system_name,
        "wer": round(wer_val, 4),
        "cer": round(cer_val, 4),
        "samples": len(results),
        "avg_latency_sec": round(avg_latency, 4),
        "tags": {},
        "entity": None,
    }

    if has_tags:
        tag_results = {}
        for tag_name in TAG_PATTERNS:
            y_true = tag_true[tag_name]
            y_pred = tag_pred[tag_name]
            pairs_with_label = [(t, p) for t, p in zip(y_true, y_pred) if t != "NONE" or p != "NONE"]
            if not pairs_with_label:
                continue
            yt, yp = zip(*pairs_with_label)
            all_labels = sorted(set(yt) | set(yp))
            cm = _classification_metrics(list(yt), list(yp), all_labels)
            tag_results[tag_name] = {
                "accuracy": round(cm["accuracy"], 4),
                "macro_f1": round(cm["macro_f1"], 4),
                "weighted_f1": round(cm["weighted_f1"], 4),
                "per_class": {
                    label: {k: round(v, 4) for k, v in vals.items()}
                    for label, vals in cm["per_class"].items()
                },
            }
        metrics["tags"] = tag_results

        entity_tp = entity_fp = entity_fn = 0
        for true_set, pred_set in zip(entity_true_spans, entity_pred_spans):
            true_types = {etype for etype, _ in true_set}
            pred_types = {etype for etype, _ in pred_set}
            entity_tp += len(true_types & pred_types)
            entity_fp += len(pred_types - true_types)
            entity_fn += len(true_types - pred_types)

        ent_prec = entity_tp / max(entity_tp + entity_fp, 1)
        ent_rec = entity_tp / max(entity_tp + entity_fn, 1)
        ent_f1 = 2 * ent_prec * ent_rec / max(ent_prec + ent_rec, 1e-9)
        metrics["entity"] = {
            "precision": round(ent_prec, 4),
            "recall": round(ent_rec, 4),
            "f1": round(ent_f1, 4),
        }

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hindi ASR benchmark: Whissle vs Deepgram vs Gemini vs Sarvam")
    parser.add_argument("--model", required=True, help="Path to .nemo or .ckpt model")
    parser.add_argument("--manifest", required=True, help="JSONL manifest for in-house validation")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--deepgram-key", required=True, help="Deepgram API key")
    parser.add_argument("--gemini-key", required=True, help="Gemini API key")
    parser.add_argument("--sarvam-key", default=None, help="Sarvam API key")
    parser.add_argument("--in-house-samples", type=int, default=5000, help="Number of in-house samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for Whissle inference")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for Whissle inference")
    parser.add_argument("--skip-whissle", action="store_true", help="Skip Whissle inference (load from cache)")
    parser.add_argument("--skip-deepgram", action="store_true", help="Skip Deepgram inference")
    parser.add_argument("--skip-gemini", action="store_true", help="Skip Gemini inference")
    parser.add_argument("--skip-sarvam", action="store_true", help="Skip Sarvam inference")
    parser.add_argument("--fleurs-manifest", default=None, help="Path to existing FLEURS Hindi manifest (JSONL)")
    parser.add_argument("--skip-fleurs", action="store_true", help="Skip FLEURS test set")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load in-house validation manifest ----
    print(f"\nLoading in-house manifest: {args.manifest}")
    all_entries = []
    with open(args.manifest) as f:
        for line in f:
            all_entries.append(json.loads(line))

    random.seed(42)
    if len(all_entries) > args.in_house_samples:
        in_house = random.sample(all_entries, args.in_house_samples)
    else:
        in_house = all_entries
    print(f"  In-house samples: {len(in_house)}")

    # ---- Load FLEURS Hindi ----
    fleurs = []
    if not args.skip_fleurs:
        fleurs = load_fleurs_hindi(manifest_path=args.fleurs_manifest, output_dir=str(output_dir))

    # ---- Run evaluations ----
    try:
        import torch
        device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    except ImportError:
        device = 'cpu'

    all_results = {}
    test_sets = {"in_house": in_house}
    if fleurs:
        test_sets["fleurs"] = fleurs

    for test_name, test_data in test_sets.items():
        print(f"\n{'=' * 60}")
        print(f"Test set: {test_name} ({len(test_data)} samples)")
        print(f"{'=' * 60}")

        # Determine which systems need to run (not cached/skipped)
        whissle_cache = output_dir / f"whissle_{test_name}_predictions.jsonl"
        dg_cache = output_dir / f"deepgram_{test_name}_predictions.jsonl"
        gemini_cache = output_dir / f"gemini_{test_name}_predictions.jsonl"
        sarvam_cache = output_dir / f"sarvam_{test_name}_predictions.jsonl"

        system_results = {}

        # Load cached results first
        if args.skip_whissle and whissle_cache.exists():
            print(f"Loading cached Whissle results from {whissle_cache}")
            whissle_results = [json.loads(l) for l in open(whissle_cache)]
            whissle_latency = sum(r.get('latency', 0) for r in whissle_results) / max(len(whissle_results), 1)
            system_results['whissle'] = (whissle_results, whissle_latency)

        if args.skip_deepgram and dg_cache.exists():
            print(f"Loading cached Deepgram results from {dg_cache}")
            dg_results = [json.loads(l) for l in open(dg_cache)]
            dg_latency = sum(r.get('latency', 0) for r in dg_results) / max(len(dg_results), 1)
            system_results['deepgram'] = (dg_results, dg_latency)

        if args.skip_gemini and gemini_cache.exists():
            print(f"Loading cached Gemini results from {gemini_cache}")
            gemini_results = [json.loads(l) for l in open(gemini_cache)]
            gemini_latency = sum(r.get('latency', 0) for r in gemini_results) / max(len(gemini_results), 1)
            system_results['gemini'] = (gemini_results, gemini_latency)

        if args.skip_sarvam and sarvam_cache.exists():
            print(f"Loading cached Sarvam results from {sarvam_cache}")
            sarvam_results = [json.loads(l) for l in open(sarvam_cache)]
            sarvam_latency = sum(r.get('latency', 0) for r in sarvam_results) / max(len(sarvam_results), 1)
            system_results['sarvam'] = (sarvam_results, sarvam_latency)

        # Run remaining systems in parallel
        cache_map = {
            'whissle': whissle_cache, 'deepgram': dg_cache,
            'gemini': gemini_cache, 'sarvam': sarvam_cache,
        }
        pending = {}
        with ThreadPoolExecutor(max_workers=4) as system_pool:
            if 'whissle' not in system_results:
                pending['whissle'] = system_pool.submit(
                    run_whissle_inference, args.model, test_data, device, args.batch_size
                )
            if 'deepgram' not in system_results:
                pending['deepgram'] = system_pool.submit(
                    run_deepgram_inference, test_data, args.deepgram_key
                )
            if 'gemini' not in system_results:
                pending['gemini'] = system_pool.submit(
                    run_gemini_inference, test_data, args.gemini_key
                )
            if 'sarvam' not in system_results and args.sarvam_key:
                pending['sarvam'] = system_pool.submit(
                    run_sarvam_inference, test_data, args.sarvam_key
                )

            for sys_name, future in pending.items():
                results_pair = future.result()
                system_results[sys_name] = results_pair
                with open(cache_map[sys_name], 'w', encoding='utf-8') as f:
                    for r in results_pair[0]:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')

        whissle_results, whissle_latency = system_results['whissle']
        dg_results, dg_latency = system_results['deepgram']
        gemini_results, gemini_latency = system_results['gemini']

        # Compute metrics
        has_in_house_tags = (test_name == "in_house")
        whissle_metrics = compute_system_metrics(whissle_results, "Whissle v18", has_tags=has_in_house_tags)
        dg_metrics = compute_system_metrics(dg_results, "Deepgram Nova-2", has_tags=False)
        gemini_metrics = compute_system_metrics(gemini_results, "Gemini 2.5 Flash", has_tags=False)

        all_results[test_name] = {
            "whissle": whissle_metrics,
            "deepgram": dg_metrics,
            "gemini": gemini_metrics,
        }

        if 'sarvam' in system_results:
            sarvam_results, sarvam_latency = system_results['sarvam']
            sarvam_metrics = compute_system_metrics(sarvam_results, "Sarvam Saaras v3", has_tags=False)
            all_results[test_name]["sarvam"] = sarvam_metrics

        # Print comparison table
        print(f"\n{'=' * 60}")
        print(f"Results: {test_name}")
        print(f"{'=' * 60}")
        print(f"{'System':<20} {'WER':>8} {'CER':>8} {'Latency':>10}")
        print(f"{'-' * 48}")
        sys_list = [("Whissle v18", whissle_metrics), ("Deepgram Nova-2", dg_metrics), ("Gemini 2.5 Flash", gemini_metrics)]
        if 'sarvam' in all_results[test_name]:
            sys_list.append(("Sarvam Saaras v3", all_results[test_name]["sarvam"]))
        for name, m in sys_list:
            print(f"{name:<20} {m['wer']:>7.2%} {m['cer']:>7.2%} {m['avg_latency_sec']:>9.3f}s")

        if has_in_house_tags and whissle_metrics["tags"]:
            print(f"\nWhissle v18 Tag Accuracy:")
            print(f"{'Category':<12} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
            print(f"{'-' * 46}")
            for cat, vals in sorted(whissle_metrics["tags"].items()):
                print(f"{cat:<12} {vals['accuracy']:>9.2%} {vals['macro_f1']:>9.2%} {vals['weighted_f1']:>11.2%}")

        if whissle_metrics.get("entity"):
            ent = whissle_metrics["entity"]
            print(f"\nWhissle v18 Entity Detection: P={ent['precision']:.2%}  R={ent['recall']:.2%}  F1={ent['f1']:.2%}")

    # ---- Write summary JSON ----
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nBenchmark summary saved to {summary_path}")

    # ---- Write Excel ----
    try:
        import xlsxwriter
        _export_benchmark_excel(output_dir / "benchmark_results.xlsx", all_results)
    except ImportError:
        print("xlsxwriter not available, skipping Excel export")

    # Print sample predictions
    for test_name in all_results:
        print(f"\n--- Sample predictions ({test_name}) ---")
        whissle_cache = output_dir / f"whissle_{test_name}_predictions.jsonl"
        if whissle_cache.exists():
            samples = [json.loads(l) for l in open(whissle_cache)][:3]
            for s in samples:
                ref_clean = strip_tags(s['reference'])
                pred_clean = strip_tags(s['prediction'])
                print(f"  REF:  {ref_clean[:120]}")
                print(f"  PRED: {pred_clean[:120]}")
                print()


def _export_benchmark_excel(path, all_results):
    import xlsxwriter

    wb = xlsxwriter.Workbook(str(path))
    bold = wb.add_format({"bold": True})
    pct = wb.add_format({"num_format": "0.00%"})
    num3 = wb.add_format({"num_format": "0.000"})

    for test_name, systems in all_results.items():
        ws = wb.add_worksheet(test_name[:31])
        ws.write(0, 0, f"Benchmark: {test_name}", bold)

        ws.write(2, 0, "System", bold)
        ws.write(2, 1, "WER", bold)
        ws.write(2, 2, "CER", bold)
        ws.write(2, 3, "Avg Latency (s)", bold)
        ws.write(2, 4, "Samples", bold)

        row = 3
        for sys_key in ["whissle", "deepgram", "gemini", "sarvam"]:
            if sys_key not in systems:
                continue
            m = systems[sys_key]
            ws.write(row, 0, m["system"])
            ws.write(row, 1, m["wer"], pct)
            ws.write(row, 2, m["cer"], pct)
            ws.write(row, 3, m["avg_latency_sec"], num3)
            ws.write(row, 4, m["samples"])
            row += 1

        if systems["whissle"]["tags"]:
            row += 2
            ws.write(row, 0, "Whissle Tag Classification", bold)
            row += 1
            ws.write(row, 0, "Category", bold)
            ws.write(row, 1, "Accuracy", bold)
            ws.write(row, 2, "Macro F1", bold)
            ws.write(row, 3, "Weighted F1", bold)
            row += 1
            for cat, vals in sorted(systems["whissle"]["tags"].items()):
                ws.write(row, 0, cat)
                ws.write(row, 1, vals["accuracy"], pct)
                ws.write(row, 2, vals["macro_f1"], pct)
                ws.write(row, 3, vals["weighted_f1"], pct)
                row += 1

        ent = systems["whissle"].get("entity")
        if ent:
            row += 1
            ws.write(row, 0, "Entity Detection", bold)
            row += 1
            ws.write(row, 0, "Precision")
            ws.write(row, 1, ent["precision"], pct)
            row += 1
            ws.write(row, 0, "Recall")
            ws.write(row, 1, ent["recall"], pct)
            row += 1
            ws.write(row, 0, "F1")
            ws.write(row, 1, ent["f1"], pct)

    wb.close()
    print(f"Excel saved to {path}")


if __name__ == "__main__":
    main()
