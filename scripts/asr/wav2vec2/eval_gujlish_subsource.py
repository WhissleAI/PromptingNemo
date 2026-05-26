#!/usr/bin/env python3
"""Eval Gujlish model with per-sub-source breakdown (IndicVoices, Kathbath, FLEURS, etc.)."""

import json, os, sys, time, argparse, logging
from collections import Counter, defaultdict
from typing import List, Optional, Tuple
import torch, torchaudio, numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)
SAMPLE_RATE = 16000

class MetaASRTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.pad_id = vocab.get("<pad>", 0)
    def decode(self, ids):
        tokens = []
        for i in ids:
            t = self.id_to_token.get(i, "")
            if t in ("<pad>", "<unk>", "<ctc>"): continue
            tokens.append(" " if t == "|" else t)
        return "".join(tokens).strip()

def edit_distance(ref, hyp):
    d = list(range(len(ref) + 1))
    for j in range(1, len(hyp) + 1):
        prev = d[:]
        d[0] = j
        for i in range(1, len(ref) + 1):
            d[i] = prev[i-1] if ref[i-1] == hyp[j-1] else 1 + min(prev[i-1], prev[i], d[i-1])
    return d[len(ref)]

def parse_tags(text):
    words = text.split()
    transcript, tags = [], {"age": None, "gender": None, "emotion": None, "intent": None}
    entities, entity_buf, in_entity = [], [], False
    for w in words:
        if w.startswith("AGE_"): tags["age"] = w
        elif w.startswith("GENDER_"): tags["gender"] = w
        elif w.startswith("EMOTION_"): tags["emotion"] = w
        elif w.startswith("INTENT_"): tags["intent"] = w
        elif w.startswith("ENTITY_"): in_entity = True; entity_buf = [w]
        elif w == "END" and in_entity: entity_buf.append(w); entities.append(" ".join(entity_buf)); entity_buf = []; in_entity = False
        elif in_entity: entity_buf.append(w)
        else: transcript.append(w)
    return " ".join(transcript), tags, entities

def detect_source(entry):
    src = entry.get("source", "unknown")
    if src == "gujarati":
        fp = entry["audio_filepath"].lower()
        if "indicvoices_r" in fp: return "gu_indicvoices_r"
        if "indicvoices" in fp: return "gu_indicvoices"
        if "kathbath" in fp: return "gu_kathbath"
        if "fleurs" in fp: return "gu_fleurs"
        if "commonvoice" in fp: return "gu_commonvoice"
        return "gu_other"
    return src

def load_audio(path):
    try:
        waveform, sr = torchaudio.load(path)
    except: return None
    if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != SAMPLE_RATE: waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform

class Accum:
    def __init__(self):
        self.clean_wer_errors = self.clean_wer_words = 0
        self.clean_cer_errors = self.clean_cer_chars = 0
        self.tag_correct = {t: 0 for t in ["age","gender","emotion","intent"]}
        self.tag_total = {t: 0 for t in ["age","gender","emotion","intent"]}
        self.entity_f1_sum = self.entity_count = 0
        self.latency_sum = self.count = 0
        self.duration_sum = 0.0
        self.failed = 0
    def report(self):
        n = max(self.count, 1)
        r = {"samples": self.count, "failed": self.failed,
             "duration_hours": round(self.duration_sum/3600, 2),
             "clean_wer": round(self.clean_wer_errors / max(self.clean_wer_words,1), 4),
             "clean_cer": round(self.clean_cer_errors / max(self.clean_cer_chars,1), 4),
             "tags": {}}
        for t in ["age","gender","emotion","intent"]:
            total = max(self.tag_total[t], 1)
            r["tags"][t] = {"accuracy": round(self.tag_correct[t]/total, 4),
                           "correct": self.tag_correct[t], "total": self.tag_total[t]}
        r["entity_f1"] = round(self.entity_f1_sum / max(self.entity_count,1), 4)
        r["avg_latency_ms"] = round(1000 * self.latency_sum / n, 2)
        return r

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import Wav2Vec2ForCTC
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    log.info("Model: %.1fM params on %s", sum(p.numel() for p in model.parameters())/1e6, device)

    with open(os.path.join(args.model_dir, "vocab.json")) as f:
        vocab = json.load(f)
    tokenizer = MetaASRTokenizer(vocab)

    entries = []
    with open(args.manifest) as f:
        for line in f:
            if line.strip(): entries.append(json.loads(line))
    log.info("Manifest: %d samples", len(entries))

    accums = defaultdict(Accum)
    processed = 0

    for batch_start in range(0, len(entries), args.batch_size):
        batch_entries = entries[batch_start:batch_start+args.batch_size]
        waveforms, valid_indices = [], []

        for idx, entry in enumerate(batch_entries):
            wav = load_audio(entry["audio_filepath"])
            if wav is None:
                src = detect_source(entry)
                accums[src].failed += 1
                accums["combined"].failed += 1
                continue
            waveforms.append(wav)
            valid_indices.append(idx)

        if not waveforms: continue

        max_len = max(w.shape[0] for w in waveforms)
        padded = torch.zeros(len(waveforms), max_len)
        attn = torch.zeros(len(waveforms), max_len, dtype=torch.long)
        for j, w in enumerate(waveforms):
            padded[j,:w.shape[0]] = w
            attn[j,:w.shape[0]] = 1
        padded, attn = padded.to(device), attn.to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model(input_values=padded, attention_mask=attn)
        batch_lat = time.time() - t0

        pred_ids = torch.argmax(outputs.logits, dim=-1)

        for j, oi in enumerate(valid_indices):
            entry = batch_entries[oi]
            src = detect_source(entry)
            lang = entry.get("lang_family", "UNK")
            dur = entry.get("duration", waveforms[j].shape[0]/SAMPLE_RATE)
            ref = entry.get("text", "")
            lat = batch_lat / len(valid_indices)

            seq = pred_ids[j].tolist()
            collapsed, prev = [], -1
            for p in seq:
                if p != prev and p != tokenizer.pad_id: collapsed.append(p)
                prev = p
            hyp = tokenizer.decode(collapsed)

            ref_clean, ref_tags, ref_ents = parse_tags(ref)
            hyp_clean, hyp_tags, hyp_ents = parse_tags(hyp)

            ref_w, hyp_w = ref_clean.split(), hyp_clean.split()
            cwe = edit_distance(ref_w, hyp_w) if ref_w else (len(hyp_w) if hyp_w else 0)
            ref_c = list(ref_clean.replace(" ",""))
            hyp_c = list(hyp_clean.replace(" ",""))
            cce = edit_distance(ref_c, hyp_c) if ref_c else (len(hyp_c) if hyp_c else 0)

            ref_es, hyp_es = set(ref_ents), set(hyp_ents)
            tp = len(ref_es & hyp_es)
            ef1 = 2*tp/(len(ref_es)+len(hyp_es)) if (ref_es or hyp_es) else 1.0

            for key in [src, f"lang:{lang}", "combined"]:
                a = accums[key]
                a.clean_wer_errors += cwe
                a.clean_wer_words += len(ref_w)
                a.clean_cer_errors += cce
                a.clean_cer_chars += len(ref_c)
                a.latency_sum += lat
                a.duration_sum += dur
                a.count += 1
                if ref_ents:
                    a.entity_f1_sum += ef1
                    a.entity_count += 1
                for t in ["age","gender","emotion","intent"]:
                    if ref_tags[t]:
                        a.tag_total[t] += 1
                        if hyp_tags[t] == ref_tags[t]:
                            a.tag_correct[t] += 1

            processed += 1

        if processed % 2000 == 0:
            c = accums["combined"]
            log.info("Processed %d/%d | WER: %.2f%%",
                     processed, len(entries),
                     100*c.clean_wer_errors/max(c.clean_wer_words,1))

    # Final log
    c = accums["combined"]
    log.info("Processed %d/%d | WER: %.2f%%",
             processed, len(entries),
             100*c.clean_wer_errors/max(c.clean_wer_words,1))

    report = {"combined": accums["combined"].report(), "by_source": {}, "by_language": {}}
    for key, acc in sorted(accums.items()):
        if key == "combined": continue
        if key.startswith("lang:"):
            report["by_language"][key[5:]] = acc.report()
        else:
            report["by_source"][key] = acc.report()

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report: %s", args.output)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION WITH SUB-SOURCE BREAKDOWN")
    print("="*80)

    c = report["combined"]
    print(f"\nCombined ({c['samples']:,} samples, {c['duration_hours']}h)")
    print(f"  Clean WER: {c['clean_wer']*100:.2f}%")
    for t in ["age","gender","emotion","intent"]:
        tt = c["tags"][t]
        print(f"  {t.upper():8s}: {tt['accuracy']*100:.1f}% ({tt['correct']}/{tt['total']})")
    print(f"  Entity F1: {c['entity_f1']*100:.1f}%")

    print("\nBy Source:")
    for src in ["en_people", "cv_english", "speech_commands",
                "gu_kathbath", "gu_indicvoices", "gu_indicvoices_r", "gu_fleurs", "gu_commonvoice", "gu_other"]:
        m = report["by_source"].get(src)
        if not m: continue
        print(f"\n  {src} ({m['samples']:,} samples, {m['duration_hours']}h)")
        print(f"    Clean WER: {m['clean_wer']*100:.2f}% | CER: {m['clean_cer']*100:.2f}%")
        for t in ["age","gender","emotion","intent"]:
            tt = m["tags"][t]
            if tt["total"] > 0:
                print(f"    {t.upper():8s}: {tt['accuracy']*100:.1f}% ({tt['correct']}/{tt['total']})")
        if m.get("entity_f1", 0) > 0:
            print(f"    Entity F1: {m['entity_f1']*100:.1f}%")

    print(f"\n  Failed audio loads: {c.get('failed', 0)}")


if __name__ == "__main__":
    main()
