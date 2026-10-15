"""Open TTS Leaderboard protocol (huggingface/open-tts-leaderboard), run locally.

Stage 1 (this script) writes the leaderboard's own layout, so its unmodified stage 2-4 scripts run on our output:
  results/<model_safe>/MODEL_<model_safe>_DATASET_<ds>_<config>_<split>[_voice_clone].jsonl
  results/<model_safe>/<ds>_<config>_<split>[_voice_clone]/output_<id>.wav
  manifest rows: audio_filepath (relative to the model dir), duration, time (generation seconds), text, pred_text
  (+ prompt_audio_filepath, sim in cloning mode)
Datasets (English):
  seed  Seed-TTS eval `tts` config, split en: en/meta.lst (1088 rows), text = text to synthesize
  cv3   CV3-Eval `zero_shot` config, split en (500 rows; public mirror yuekai/CV3-Eval), text = target_text
Modes: without voice cloning, the model's own default voice (--voice); with --voice_clone, the row's prompt clip
(our model uses only its speaker vector). Then:
  transformers/transcribe.py (Qwen3-ASR-1.7B) -> transformers/score_similarity.py (cloning) -> scripts/score_model.py
RTFx is timed on THIS GPU at batch size 1 (the leaderboard uses an H200), so it is indicative only.

python -m steer.leaderboard_eval --ckpt runs/spk_enc_v3/step305000.pt,flow=runs/flow_v3/step100000.pt \
    --model_id whissle/steer-tts-9m-flow --set seed [--voice_clone]
"""
import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .config import ROOT, SR

OTL = Path(os.environ.get("OTL_ROOT", Path.home() / "Desktop/whissle/workspace/open-tts-leaderboard"))
SEED_META = Path.home() / "Desktop/whissle/workspace/seedtts/data/seedtts_testset/en/meta.lst"
CV3_PARQUET = Path.home() / "Desktop/whissle/workspace/lb_data/cv3_zero_shot_en.parquet"
DEFAULT_VOICE = "voice_4223"  # lowest WER of 8 candidates (2026-10-10, full Seed+CV3: 1.69% vs voice_1361 1.81%)


def rows(which):
    """[(id, text, prompt_wav_path)] in dataset order, plus the leaderboard's (dataset_path, config, split)."""
    if which == "seed":
        out = []
        for i, l in enumerate(open(SEED_META)):
            if l.strip():
                f = l.rstrip("\n").split("|")
                out.append((i, f[3], str(SEED_META.parent / f[2])))
        return out, ("local/seed_tts_eval", "tts", "en")
    d = pd.read_parquet(CV3_PARQUET)
    pdir = CV3_PARQUET.parent / "cv3_en_prompts"
    pdir.mkdir(exist_ok=True)
    out = []
    for i, r in enumerate(d.itertuples()):
        p = pdir / f"{r.id}.wav"
        if not p.exists():
            y, sr = sf.read(io.BytesIO(r.prompt_audio["bytes"]), dtype="float32")
            sf.write(p, y, sr)
        out.append((i, r.target_text, str(p)))
    return out, ("local/cv3_eval", "zero_shot", "en")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint, or base.pt,flow=flow.pt")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--set", choices=["seed", "cv3"], required=True)
    ap.add_argument("--voice_clone", action="store_true")
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max", type=int, default=0, help="first N rows only (smoke test)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flow_steps", type=int, default=10); ap.add_argument("--flow_cfg", type=float, default=0.0)
    ap.add_argument("--flow_temp", type=float, default=0.4)
    ap.add_argument("--start", type=int, default=0, help="skip the first N rows (with --max: a slice)")
    a = ap.parse_args()
    from .synth import Synth
    import re
    ck = re.sub(r"(^|,flow=|,pflow=)([^,]+)", lambda m: m.group(1) + (m.group(2) if Path(m.group(2)).is_absolute() else str(ROOT / m.group(2))), a.ckpt)
    s = Synth(ck, device=a.device, flow_steps=a.flow_steps, flow_cfg=a.flow_cfg, flow_temp=a.flow_temp)
    sync = torch.cuda.synchronize if a.device.startswith("cuda") else (lambda: None)
    items, (ds_path, config, split) = rows(a.set)
    items = items[a.start:]
    items = items[: a.max] if a.max else items
    suffix = "_voice_clone" if a.voice_clone else ""
    model_safe, ds_safe = a.model_id.replace("/", "-"), ds_path.replace("/", "-")
    model_dir = OTL / "results" / model_safe
    wav_dir_name = f"{ds_safe}_{config}_{split}{suffix}"
    (model_dir / wav_dir_name).mkdir(parents=True, exist_ok=True)
    manifest = model_dir / f"MODEL_{model_safe}_DATASET_{ds_safe}_{config}_{split}{suffix}.jsonl"

    def synth(text, prompt):
        return s(text, None, ref_wav=prompt) if a.voice_clone else s(text, a.voice)

    for _, text, prompt in items[: a.warmup]:  # untimed warm-up, as the leaderboard does
        synth(text, prompt)
    with open(manifest, "w", encoding="utf-8") as mf:
        for i, text, prompt in items:
            sync()
            t0 = time.perf_counter()
            y = np.asarray(synth(text, prompt), np.float32)
            sync()
            dt = time.perf_counter() - t0
            rel = f"{wav_dir_name}/output_{i}.wav"
            sf.write(model_dir / rel, y, SR)
            e = dict(audio_filepath=rel, duration=len(y) / SR, time=dt, text=text, pred_text="")
            if a.voice_clone:
                e.update(prompt_audio_filepath=prompt, sim="")
            mf.write(json.dumps(e, ensure_ascii=False) + "\n")
    tot = [json.loads(l) for l in open(manifest)]
    print(f"{manifest.name}: {len(tot)} rows, RTFx {sum(x['duration'] for x in tot) / sum(x['time'] for x in tot):.1f}"
          " (this GPU, batch 1)", flush=True)


if __name__ == "__main__":
    main()
