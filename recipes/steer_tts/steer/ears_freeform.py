"""EARS freeform recordings -> transcribed, segmented utterances (manifest_earsff.jsonl), 2-15 s.

EARS ships 22 emotion freeform monologues + freeform speech per speaker (3,103 files, 42.6 h) WITHOUT transcripts,
so they were unused - while the scripted emotion sentences give only ~64 clips per emotion. This transcribes them
with Whisper-large-v3-turbo (timestamps), merges Whisper's segments into 2-15 s utterances, and writes them in the
corpus format (corpus "ears", the file's emotion as the style tag; plain freeform speech -> tag "none").

Excluded: every EARS file in ParaSpeechCaps test / holdout / dev (they are our benchmark's prompts).
Filters: 2-15 s, >= 3 words, 6-25 chars/s (catches hallucinated / truncated segments).

python -m steer.ears_freeform [--limit N]   -> data/wav24/ears/<id>.wav + data/meta/manifest_earsff.jsonl
Then: steer.extract earsff, steer.xvec earsff, steer.build_controls ... earsff.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .config import META, RAW, ROOT, SR, canonical_tag

WAV = ROOT / "data" / "wav24" / "ears"
OUT = META / "manifest_earsff.jsonl"
DONE = META / "earsff_done.txt"  # processed source files (resumable), kept out of the manifest
MODEL = "openai/whisper-large-v3-turbo"


def excluded():
    """EARS files used by ParaSpeechCaps evaluation splits: never train on them."""
    ex = set()
    for split in ("test", "holdout", "dev"):
        f = RAW / "paraspeechcaps" / "data" / f"{split}-00000-of-00001.parquet"
        if f.exists():
            t = pd.read_parquet(f, columns=["source", "relative_audio_path"])
            ex |= set(t[t.source == "ears"].relative_audio_path)
    return ex


def files():
    ex = excluded()
    out = []
    for f in sorted((RAW / "ears").glob("p[0-9][0-9][0-9]/*.wav")):
        rel = f"{f.parent.name}/{f.name}"
        if ("_freeform" in f.stem or f.stem.startswith("freeform_speech")) and rel not in ex:
            out.append(f)
    return out, len(ex)


def segments(model, proc, y16, dev):
    """Whisper long-form with timestamps -> list of dict(start, end, text, avg_logprob, no_speech)."""
    feats = proc(y16, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest",
                 return_attention_mask=True)
    if feats.input_features.shape[-1] < 3000:  # short audio: the model expects a 30 s window
        feats = proc(y16, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    out = model.generate(feats.input_features.to(dev, torch.float16), attention_mask=feats.attention_mask.to(dev),
                         language="en", task="transcribe", return_timestamps=True, return_segments=True,
                         condition_on_prev_tokens=False, temperature=0.0)  # greedy; quality filtered in merge/main
    segs = out["segments"][0] if isinstance(out, dict) else []
    res = []
    for s in segs:
        text = proc.decode(s["tokens"], skip_special_tokens=True).strip()
        lp = s.get("result", {}).get("avg_logprobs") if isinstance(s.get("result"), dict) else None
        res.append(dict(start=float(s["start"]), end=float(s["end"]), text=text,
                        avg_logprob=float(lp) if lp is not None else 0.0))
    return res


def merge(segs, lo=2.0, hi=15.0):
    """Greedy merge of consecutive Whisper segments into utterances of lo..hi seconds."""
    out, cur = [], None
    for s in segs:
        if not s["text"]:
            continue
        if cur and s["end"] - cur["start"] <= hi and s["start"] - cur["end"] < 1.0:
            cur["end"], cur["text"] = s["end"], cur["text"] + " " + s["text"]
            cur["lps"].append(s["avg_logprob"])
        else:
            if cur:
                out.append(cur)
            cur = dict(start=s["start"], end=s["end"], text=s["text"], lps=[s["avg_logprob"]])
    if cur:
        out.append(cur)
    return [u for u in out if lo <= u["end"] - u["start"] <= hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import soxr
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from .prep import trim_resample
    dev = "cuda"
    proc = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    fl, n_ex = files()
    done = set(DONE.read_text().split("\n")) if DONE.exists() else set()
    todo = [f for f in fl if str(f) not in done][: a.limit or None]
    print(f"{len(fl)} freeform files ({n_ex} ParaSpeechCaps eval files excluded), {len(todo)} to do", flush=True)
    WAV.mkdir(parents=True, exist_ok=True)
    kept = drop = 0
    with open(OUT, "a") as man, open(DONE, "a") as dn:
        for k, f in enumerate(todo):
            y, sr = sf.read(f, dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            y16 = soxr.resample(y, sr, 16000)
            with torch.no_grad():
                utts = merge(segments(model, proc, y16, dev))
            spk = f.parent.name
            if f.stem.startswith("freeform_speech"):
                raw, tag = "freeform", "none"
            else:
                raw = f.stem.split("_")[1]
                tag = canonical_tag("ears", raw)
            rows = []
            for i, u in enumerate(utts):
                words, dur = u["text"].split(), u["end"] - u["start"]
                if len(words) < 3 or not (6 <= len(u["text"]) / dur <= 25):
                    drop += 1; continue
                a0, a1 = max(0, int((u["start"] - 0.1) * sr)), min(len(y), int((u["end"] + 0.15) * sr))
                seg = trim_resample(y[a0:a1], sr)
                uid = f"ears_{spk}_{f.stem}_{i:03d}"
                sf.write(WAV / f"{uid}.wav", seg, SR)
                rows.append(dict(id=uid, corpus="ears", speaker=f"ears_{spk}", text=u["text"], raw_style=raw, tag=tag,
                                 group=f"{spk}_{f.stem}", wav=str(WAV / f"{uid}.wav"), dur=round(len(seg) / SR, 3),
                                 src=str(f)))
                kept += 1
            for r in rows:
                man.write(json.dumps(r) + "\n")
            man.flush(); dn.write(str(f) + "\n"); dn.flush()
            if k % 50 == 0:
                print(f"  {k + 1}/{len(todo)} files, kept {kept} utts, dropped {drop}", flush=True)
    print(f"done: kept {kept}, dropped {drop}", flush=True)


if __name__ == "__main__":
    main()
