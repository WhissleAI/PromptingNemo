#!/usr/bin/env python3
"""Full evaluation of Gujlish wav2vec2 Meta-ASR on the complete validation set.

Produces per-source, per-language, and combined metrics:
  - WER (full output and clean transcript)
  - CER
  - Meta-tag accuracy (AGE, GENDER, EMOTION, INTENT)
  - Entity F1
  - Latency / RTF

Outputs a detailed JSON report to --output.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
TAG_PREFIXES = ("ENTITY_", "INTENT_", "EMOTION_", "GENDER_", "AGE_")


class MetaASRTokenizer:
    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.pad_id = vocab.get("<pad>", 0)
        self.unk_id = vocab.get("<unk>", 1)
        self.space_id = vocab.get("|", 2)

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            t = self.id_to_token.get(i, "")
            if t in ("<pad>", "<unk>", "<ctc>"):
                continue
            if t == "|":
                tokens.append(" ")
            else:
                tokens.append(t)
        return "".join(tokens).strip()


def edit_distance(ref: List[str], hyp: List[str]) -> int:
    d = list(range(len(ref) + 1))
    for j in range(1, len(hyp) + 1):
        prev = d[:]
        d[0] = j
        for i in range(1, len(ref) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i] = prev[i - 1]
            else:
                d[i] = 1 + min(prev[i - 1], prev[i], d[i - 1])
    return d[len(ref)]


def compute_wer(ref: str, hyp: str) -> Tuple[float, int, int]:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0, 0, len(hyp_words)
    errors = edit_distance(ref_words, hyp_words)
    return errors / len(ref_words), errors, len(ref_words)


def compute_cer(ref: str, hyp: str) -> Tuple[float, int, int]:
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0, 0, len(hyp_chars)
    errors = edit_distance(ref_chars, hyp_chars)
    return errors / len(ref_chars), errors, len(ref_chars)


def parse_tags(text: str) -> Tuple[str, Dict]:
    words = text.split()
    transcript = []
    tags = {"age": None, "gender": None, "emotion": None, "intent": None}
    entities = []
    entity_buf = []
    in_entity = False

    for w in words:
        if w.startswith("AGE_"):
            tags["age"] = w
        elif w.startswith("GENDER_"):
            tags["gender"] = w
        elif w.startswith("EMOTION_"):
            tags["emotion"] = w
        elif w.startswith("INTENT_"):
            tags["intent"] = w
        elif w.startswith("ENTITY_"):
            in_entity = True
            entity_buf = [w]
        elif w == "END" and in_entity:
            entity_buf.append(w)
            entities.append(" ".join(entity_buf))
            entity_buf = []
            in_entity = False
        elif in_entity:
            entity_buf.append(w)
        else:
            transcript.append(w)

    return " ".join(transcript), tags, entities


def entity_f1(ref_entities: List[str], hyp_entities: List[str]) -> Tuple[float, float, float]:
    if not ref_entities and not hyp_entities:
        return 1.0, 1.0, 1.0
    if not ref_entities:
        return 0.0, 0.0, 0.0
    if not hyp_entities:
        return 0.0, 0.0, 0.0
    ref_set = set(ref_entities)
    hyp_set = set(hyp_entities)
    tp = len(ref_set & hyp_set)
    precision = tp / len(hyp_set) if hyp_set else 0
    recall = tp / len(ref_set) if ref_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def load_audio(path: str) -> Optional[torch.Tensor]:
    try:
        waveform, sr = torchaudio.load(path)
    except Exception as e:
        return None
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Dir with model.safetensors, config.json, vocab.json")
    parser.add_argument("--manifest", required=True, help="Validation JSONL manifest")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import Wav2Vec2ForCTC

    # Load model
    log.info("Loading model from %s", args.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %.1fM params, device=%s", total_params / 1e6, device)

    # Load vocab
    with open(os.path.join(args.model_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer = MetaASRTokenizer(vocab)
    log.info("Vocab: %d tokens", len(vocab))

    # Load manifest
    entries = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if args.max_samples:
        entries = entries[:args.max_samples]
    log.info("Manifest: %d samples", len(entries))

    # Accumulators: per-source, per-lang, combined
    class Accum:
        def __init__(self):
            self.wer_errors = 0
            self.wer_words = 0
            self.clean_wer_errors = 0
            self.clean_wer_words = 0
            self.cer_errors = 0
            self.cer_chars = 0
            self.clean_cer_errors = 0
            self.clean_cer_chars = 0
            self.tag_correct = {"age": 0, "gender": 0, "emotion": 0, "intent": 0}
            self.tag_total = {"age": 0, "gender": 0, "emotion": 0, "intent": 0}
            self.tag_confusion = {t: defaultdict(Counter) for t in ["age", "gender", "emotion", "intent"]}
            self.entity_precision_sum = 0.0
            self.entity_recall_sum = 0.0
            self.entity_f1_sum = 0.0
            self.entity_count = 0
            self.latency_sum = 0.0
            self.count = 0
            self.duration_sum = 0.0
            self.failed = 0

        def report(self):
            n = max(self.count, 1)
            r = {
                "samples": self.count,
                "failed_audio": self.failed,
                "total_duration_hours": round(self.duration_sum / 3600, 2),
                "wer": round(self.wer_errors / max(self.wer_words, 1), 4),
                "clean_wer": round(self.clean_wer_errors / max(self.clean_wer_words, 1), 4),
                "cer": round(self.cer_errors / max(self.cer_chars, 1), 4),
                "clean_cer": round(self.clean_cer_errors / max(self.clean_cer_chars, 1), 4),
                "tags": {},
                "entity_f1": round(self.entity_f1_sum / max(self.entity_count, 1), 4),
                "entity_precision": round(self.entity_precision_sum / max(self.entity_count, 1), 4),
                "entity_recall": round(self.entity_recall_sum / max(self.entity_count, 1), 4),
                "avg_latency_ms": round(1000 * self.latency_sum / n, 2),
                "avg_rtf": round(self.latency_sum / max(self.duration_sum, 0.01), 4),
            }
            for tag in ["age", "gender", "emotion", "intent"]:
                total = max(self.tag_total[tag], 1)
                r["tags"][tag] = {
                    "accuracy": round(self.tag_correct[tag] / total, 4),
                    "correct": self.tag_correct[tag],
                    "total": self.tag_total[tag],
                    "confusion": {ref: dict(hyps.most_common(10)) for ref, hyps in self.tag_confusion[tag].items()},
                }
            return r

    accums = defaultdict(Accum)  # keyed by (source, lang, "combined")

    # Process in batches
    processed = 0
    for batch_start in range(0, len(entries), args.batch_size):
        batch_entries = entries[batch_start : batch_start + args.batch_size]
        waveforms = []
        valid_indices = []

        for idx, entry in enumerate(batch_entries):
            wav = load_audio(entry["audio_filepath"])
            if wav is None:
                source = entry.get("source", "unknown")
                lang_family = entry.get("lang_family", "UNK")
                for key in [f"source:{source}", f"lang:{lang_family}", "combined"]:
                    accums[key].failed += 1
                continue
            waveforms.append(wav)
            valid_indices.append(idx)

        if not waveforms:
            continue

        max_len = max(w.shape[0] for w in waveforms)
        padded = torch.zeros(len(waveforms), max_len)
        attn_mask = torch.zeros(len(waveforms), max_len, dtype=torch.long)
        for j, w in enumerate(waveforms):
            padded[j, :w.shape[0]] = w
            attn_mask[j, :w.shape[0]] = 1

        padded = padded.to(device)
        attn_mask = attn_mask.to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model(input_values=padded, attention_mask=attn_mask)
        batch_latency = time.time() - t0

        pred_ids = torch.argmax(outputs.logits, dim=-1)

        for j, orig_idx in enumerate(valid_indices):
            entry = batch_entries[orig_idx]
            source = entry.get("source", "unknown")
            lang_family = entry.get("lang_family", "UNK")
            duration = entry.get("duration", waveforms[j].shape[0] / SAMPLE_RATE)
            ref_text = entry.get("text", "")
            latency = batch_latency / len(valid_indices)

            # CTC decode
            seq = pred_ids[j].tolist()
            collapsed = []
            prev = -1
            for p in seq:
                if p != prev and p != tokenizer.pad_id:
                    collapsed.append(p)
                prev = p
            hyp_text = tokenizer.decode(collapsed)

            # Parse tags
            ref_clean, ref_tags, ref_entities = parse_tags(ref_text)
            hyp_clean, hyp_tags, hyp_entities = parse_tags(hyp_text)

            # WER/CER on full output
            _, we, ww = compute_wer(ref_text, hyp_text)
            _, ce, cc = compute_cer(ref_text, hyp_text)
            # WER/CER on clean transcript
            _, cwe, cww = compute_wer(ref_clean, hyp_clean)
            _, cce, ccc = compute_cer(ref_clean, hyp_clean)

            # Entity F1
            ep, er, ef = entity_f1(ref_entities, hyp_entities)

            # Update accumulators
            for key in [f"source:{source}", f"lang:{lang_family}", "combined"]:
                a = accums[key]
                a.wer_errors += we
                a.wer_words += ww
                a.clean_wer_errors += cwe
                a.clean_wer_words += cww
                a.cer_errors += ce
                a.cer_chars += cc
                a.clean_cer_errors += cce
                a.clean_cer_chars += ccc
                a.latency_sum += latency
                a.duration_sum += duration
                a.count += 1

                if ref_entities:
                    a.entity_precision_sum += ep
                    a.entity_recall_sum += er
                    a.entity_f1_sum += ef
                    a.entity_count += 1

                for tag in ["age", "gender", "emotion", "intent"]:
                    if ref_tags[tag]:
                        a.tag_total[tag] += 1
                        if hyp_tags[tag] == ref_tags[tag]:
                            a.tag_correct[tag] += 1
                        a.tag_confusion[tag][ref_tags[tag]][hyp_tags.get(tag, "NONE")] += 1

        processed += len(valid_indices)
        if processed % 2000 == 0 or batch_start + args.batch_size >= len(entries):
            combined = accums["combined"]
            cwer = combined.clean_wer_errors / max(combined.clean_wer_words, 1)
            log.info("Processed %d/%d | Clean WER: %.2f%% | Gender acc: %.1f%%",
                     processed, len(entries), cwer * 100,
                     100 * combined.tag_correct["gender"] / max(combined.tag_total["gender"], 1))

    # Build report
    report = {
        "model": args.model_dir,
        "manifest": args.manifest,
        "total_samples": len(entries),
        "combined": accums["combined"].report(),
        "by_language": {},
        "by_source": {},
    }

    for key, acc in sorted(accums.items()):
        if key == "combined":
            continue
        category, name = key.split(":", 1)
        bucket = "by_language" if category == "lang" else "by_source"
        report[bucket][name] = acc.report()

    # Write report
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Report written to %s", args.output)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    c = report["combined"]
    print(f"\nCombined ({c['samples']:,} samples, {c['total_duration_hours']}h)")
    print(f"  Clean WER: {c['clean_wer']*100:.2f}% | Full WER: {c['wer']*100:.2f}%")
    print(f"  Clean CER: {c['clean_cer']*100:.2f}% | Full CER: {c['cer']*100:.2f}%")
    print(f"  Latency: {c['avg_latency_ms']:.1f}ms | RTF: {c['avg_rtf']:.4f}")
    for tag in ["age", "gender", "emotion", "intent"]:
        t = c["tags"][tag]
        print(f"  {tag.upper():8s} accuracy: {t['accuracy']*100:.1f}% ({t['correct']}/{t['total']})")
    print(f"  Entity F1: {c['entity_f1']*100:.1f}%")

    print(f"\nBy Language:")
    for lang, m in sorted(report["by_language"].items()):
        print(f"  {lang} ({m['samples']:,} samples, {m['total_duration_hours']}h)")
        print(f"    Clean WER: {m['clean_wer']*100:.2f}% | Full WER: {m['wer']*100:.2f}%")
        for tag in ["age", "gender", "emotion", "intent"]:
            t = m["tags"][tag]
            print(f"    {tag.upper():8s}: {t['accuracy']*100:.1f}% ({t['correct']}/{t['total']})")
        print(f"    Entity F1: {m['entity_f1']*100:.1f}%")

    print(f"\nBy Source:")
    for src, m in sorted(report["by_source"].items()):
        print(f"  {src} ({m['samples']:,} samples, {m['total_duration_hours']}h)")
        print(f"    Clean WER: {m['clean_wer']*100:.2f}% | Full WER: {m['wer']*100:.2f}%")
        for tag in ["age", "gender", "emotion", "intent"]:
            t = m["tags"][tag]
            if t["total"] > 0:
                print(f"    {tag.upper():8s}: {t['accuracy']*100:.1f}% ({t['correct']}/{t['total']})")

    print(f"\n  Failed audio loads: {c['failed_audio']}")


if __name__ == "__main__":
    main()
