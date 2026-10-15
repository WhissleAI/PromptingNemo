"""Render the benchmark test sets with other small open TTS systems, for apples-to-apples rows in steer.benchmark.

Runs in its OWN env (Parler-TTS pins an older transformers): ~/anaconda3/envs/ttsbase, created by logs/env_ttsbase.sh.
Standalone on purpose - imports nothing from steer, so it does not need the training env.

STEER_TTS_ROOT=... ~/anaconda3/envs/ttsbase/bin/python steer/baselines.py kokoro|piper|matcha|parler_mini [sets]
-> runs/bench/_ext/<system>/<set>/<i:04d>.wav (24 kHz), then score with
python -m steer.benchmark run --wavs kokoro_82m=runs/bench/_ext/kokoro ...

Voice policy (none of these is prompted by voice id the way SteerTTS is):
  kokoro      af_heart / am_michael by the item's gender (LibriTTS: test speaker's F0; PSC: description gender;
              LJSpeech: female)
  piper       en_US-lessac-medium (F) / en_US-ryan-medium (M), same gender rule
  matcha      matcha_ljspeech (single female voice) + its HiFi-GAN - the checkpoint behind the paper's LJSpeech table
  parler_mini parler-tts/parler-tts-mini-v1; PSC uses the item's own description, other sets a neutral
              description with the item's gender. The only baseline that can follow ParaSpeechCaps prompts.
"""
import json
import os
import sys
from pathlib import Path

# CPU threads capped: unbounded onnxruntime/torch pools took 42 of 48 cores and starved training + feature extraction
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch

torch.set_num_threads(4)

ROOT = Path(os.environ["STEER_TTS_ROOT"])
BENCH = ROOT / "data" / "bench"
OUT = ROOT / "runs" / "bench" / "_ext"
SR = 24000
GENDER_CUT_ST = json.load(open(BENCH / "calibration.json"))["gender_cut_st"]


def items(set_name):
    df = pd.read_json(BENCH / f"{set_name}.jsonl", lines=True)
    for i, r in df.iterrows():
        if set_name == "libritts":
            g = "female" if 12 * np.log2(r.spk_f0 / 100.0) > GENDER_CUT_ST else "male"
            desc = None
        elif set_name == "psc":
            g, desc = r.gender, r.description
        else:
            g, desc = "female", None
        yield i, r.text, g, desc


def save(path, y, sr):
    y = np.asarray(y, np.float32).squeeze()
    if sr != SR:
        y = soxr.resample(y, sr, SR)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, SR)


def kokoro():
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a")
    voice = dict(female="af_heart", male="am_michael")

    def tts(text, g, desc):
        return np.concatenate([a.cpu().numpy() if torch.is_tensor(a) else a for _, _, a in pipe(text, voice=voice[g])]), 24000
    return tts


def piper():
    from piper import PiperVoice
    from piper.download_voices import download_voice
    d = OUT / "_piper_models"; d.mkdir(parents=True, exist_ok=True)
    names = dict(female="en_US-lessac-medium", male="en_US-ryan-medium")
    v = {}
    for g, n in names.items():
        if not (d / f"{n}.onnx").exists():
            download_voice(n, d)
        v[g] = PiperVoice.load(str(d / f"{n}.onnx"))

    def tts(text, g, desc):
        chunks = list(v[g].synthesize(text))
        return np.concatenate([c.audio_float_array for c in chunks]), v[g].config.sample_rate
    return tts


def matcha():
    from matcha.cli import load_matcha, load_vocoder, process_text, to_waveform, get_device, assert_model_downloaded, \
        MATCHA_URLS, VOCODER_URLS, get_user_data_dir
    dev = torch.device("cuda")
    d = get_user_data_dir()
    ck, vk = d / "matcha_ljspeech.ckpt", d / "hifigan_T2_v1"
    assert_model_downloaded(ck, MATCHA_URLS["matcha_ljspeech"])
    assert_model_downloaded(vk, VOCODER_URLS["hifigan_T2_v1"])
    model = load_matcha("matcha_ljspeech", ck, dev)
    voc, den = load_vocoder("hifigan_T2_v1", vk, dev)

    @torch.inference_mode()
    def tts(text, g, desc):
        t = process_text(0, text, dev)
        out = model.synthesise(t["x"], t["x_lengths"], n_timesteps=10, temperature=0.667, spks=None, length_scale=1.0)
        return to_waveform(out["mel"], voc, den).cpu().numpy(), 22050
    return tts


def parler_mini():
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    name = "parler-tts/parler-tts-mini-v1"
    m = ParlerTTSForConditionalGeneration.from_pretrained(name).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(name)
    neutral = {g: f"A {g} speaker delivers clear, natural speech at a moderate speed and pitch in a very clean, "
                  f"close-sounding recording." for g in ("female", "male")}

    @torch.inference_mode()
    def tts(text, g, desc):
        d = tok(desc or neutral[g], return_tensors="pt").to("cuda")
        p = tok(text, return_tensors="pt").to("cuda")
        torch.manual_seed(0)
        a = m.generate(input_ids=d.input_ids, attention_mask=d.attention_mask, prompt_input_ids=p.input_ids,
                       prompt_attention_mask=p.attention_mask)
        return a.cpu().numpy().squeeze(), m.config.sampling_rate
    return tts


SYSTEMS = dict(kokoro=kokoro, piper=piper, matcha=matcha, parler_mini=parler_mini)


def main():
    system = sys.argv[1]
    sets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["libritts", "psc", "ljspeech"]
    tts = SYSTEMS[system]()
    for s in sets:
        n = 0
        for i, text, g, desc in items(s):
            f = OUT / system / s / f"{i:04d}.wav"
            if f.exists():
                continue
            try:
                y, sr = tts(text, g, desc)
                save(f, y, sr); n += 1
            except Exception as e:  # noqa: BLE001 - one bad item must not stop a 1,246-item render
                print(f"{system} {s} {i} FAILED: {type(e).__name__}: {e}", flush=True)
        print(system, s, "rendered", n, flush=True)


if __name__ == "__main__":
    main()
