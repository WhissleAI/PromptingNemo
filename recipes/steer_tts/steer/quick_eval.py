"""Quick checkpoint comparison (a few minutes each, runs beside training) on fixed subsets of the benchmark sets:
  tts    120 LibriTTS-R test-clean sentences x 4 fixed training voices (2 EARS, 2 LibriTTS-R)   WER, UTMOS
  clone  80 zero-shot cloning items (unseen LibriTTS-R speakers, one reference clip)           WER, UTMOS, SIM-o
  prompt 64 ParaSpeechCaps test prompts through the prompt adapter (expressive renders)        WER, UTMOS
WER: distil-whisper-large-v3 (fp16), Whisper normaliser. SIM-o: WavLM-base-plus-sv cosine to the real recording.
The same items, voices and order every time, so differences between checkpoints are the checkpoints.

python -m steer.quick_eval name=ckpt [name=ckpt ...]     -> runs/quick/<name>.json + table (runs/quick/summary.md)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .config import ROOT, RUNS, SR

OUT = RUNS / "quick"
DEV = "cuda" if torch.cuda.is_available() else "cpu"  # CUDA_VISIBLE_DEVICES= runs it on the CPU beside training
BENCH = ROOT / "data" / "bench"
VOICES = ["ears_p001", "ears_p050", "libritts_2836", "libritts_6385"]


def items():
    lt = pd.read_json(BENCH / "libritts.jsonl", lines=True).head(120)
    cl = pd.read_json(BENCH / "clone.jsonl", lines=True).head(80)
    psc = pd.read_json(BENCH / "psc.jsonl", lines=True)
    psc = psc.sample(64, random_state=0) if len(psc) > 64 else psc
    out = [dict(set="tts", text=t, voice=VOICES[i % 4]) for i, t in enumerate(lt.text)]
    out += [dict(set="clone", text=r.text, ref=r.ref_path, target=r.gt_path) for r in cl.itertuples()]
    pcol = "description"
    out += [dict(set="prompt", text=r["text"], prompt=str(r[pcol]), voice=VOICES[i % 2]) for i, (_, r) in enumerate(psc.iterrows())]
    return out


def render(name, ck, its):
    from .prompt import PromptMapper
    from .synth import Synth
    s, pm = Synth(ck, device=DEV), PromptMapper(device=DEV)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(its):
        if it["set"] == "clone":
            y = s(it["text"], None, ref_wav=it["ref"])
        elif it["set"] == "prompt":
            m = pm(it["prompt"])
            tag = next((t for t, _ in m["top_tags"] if t in s.tag2i), "none")
            y = s(it["text"], it["voice"], tag=tag, ctrl=m["ctrl"] or None)
        else:
            y = s(it["text"], it["voice"])
        it["wav"] = str(d / f"{i:04d}.wav")
        sf.write(it["wav"], np.asarray(y, np.float32), SR)
    del s
    torch.cuda.empty_cache()


@torch.no_grad()
def score(its, dev=DEV):
    import soxr
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer
    from .benchmark import word_errors
    to16 = lambda p: soxr.resample(sf.read(p, dtype="float32")[0], SR, 16000) if sf.info(p).samplerate == SR \
        else soxr.resample(sf.read(p, dtype="float32")[0], sf.info(p).samplerate, 16000)
    proc = WhisperProcessor.from_pretrained("distil-whisper/distil-large-v3")
    dt = torch.float16 if dev == "cuda" else torch.float32
    asr = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v3", dtype=dt).to(dev).eval()
    norm = WhisperTokenizer.from_pretrained("openai/whisper-large-v3").normalize
    for i in range(0, len(its), 8):
        ch = its[i:i + 8]
        f = proc([to16(x["wav"]) for x in ch], sampling_rate=16000, return_tensors="pt").input_features
        hyp = proc.batch_decode(asr.generate(f.to(dev, dt), language="en", task="transcribe",
                                             max_new_tokens=200), skip_special_tokens=True)
        for x, h in zip(ch, hyp):
            x["err"], x["nw"] = word_errors(norm, x["text"], h)
    del asr
    mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(dev).eval()
    for x in its:
        x["utmos"] = float(mos(torch.from_numpy(to16(x["wav"]))[None].to(dev), 16000)[0])
    del mos
    from transformers import AutoFeatureExtractor, WavLMForXVector
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
    emb = lambda p: torch.nn.functional.normalize(sv(**fe(to16(p), sampling_rate=16000, return_tensors="pt").to(dev)).embeddings[0], dim=-1)
    for x in its:
        if x["set"] == "clone":
            x["sim_o"] = float(emb(x["wav"]) @ emb(x["target"]))
    del sv
    torch.cuda.empty_cache()


def summarize(name, its):
    df = pd.DataFrame(its)
    row = dict(name=name)
    for s, g in df.groupby("set"):
        row[f"{s}_wer"] = round(100 * g.err.sum() / g.nw.sum(), 2)
        row[f"{s}_utmos"] = round(g.utmos.mean(), 3)
        if s == "clone":
            row["clone_sim"] = round(g.sim_o.mean(), 3)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("QE_THREADS", 12)))
    for arg in sys.argv[1:]:
        name, ck = arg.split("=", 1)
        its = items()
        render(name, ck, its)
        score(its)
        row = summarize(name, its)
        json.dump(dict(summary=row, items=its), open(OUT / f"{name}.json", "w"), default=float)
        print(row, flush=True)
    rows = [json.load(open(f))["summary"] for f in sorted(OUT.glob("*.json"))]
    md = pd.DataFrame(rows).to_string(index=False)
    (OUT / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
