"""Paired steerability harness (Inworld-style directions, measured in our own control units).

For every (voice, text) we render an uninstructed baseline (no tag, all controls unspecified) and one
render per direction, measure all 12 control dims on both with the same code used for training data,
convert to speaker-z with the voice's neutral reference, and report per direction:
  steer   median(delta on the target dim) / requested step     (1.0 = got exactly what was asked)
  hit     share of pairs moving the right way by > 0.5 z
  p       one-sided Wilcoxon signed-rank p-value (Holm-corrected across directions)
  leak    mean |delta| on the other dims (z)
and a direction x dim leakage matrix.

python -m steer.harness --ckpt runs/base/last.pt --out runs/base/harness_<step>.json
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from .config import CONTROL_DIMS
from .features import measure
from .synth import Synth
from .text import to_phonemes, SP, SYMBOLS

# lexically neutral evaluation texts (not in training corpora)
TEXTS = [
    "The meeting has been moved to Thursday afternoon at three.",
    "Please remember to bring the blue folder when you come in.",
    "The train leaves from platform four in about ten minutes.",
    "I put the keys on the table next to the kitchen window.",
    "We will need two more chairs for the people arriving later.",
    "The report is finished, and I sent a copy to everyone.",
    "Turn left at the second light and park behind the building.",
    "She said the package should arrive sometime early next week.",
]
STEP = 2.0  # requested change in z for continuous directions
DIRECTIONS = {  # name -> (controls dict, target dim, sign)
    "faster": ({"rate": STEP}, "rate", 1), "slower": ({"rate": -STEP}, "rate", -1),
    "louder": ({"loud": STEP}, "loud", 1), "quieter": ({"loud": -STEP}, "loud", -1),
    "higher": ({"f0_med": STEP}, "f0_med", 1), "lower": ({"f0_med": -STEP}, "f0_med", -1),
    "wider_range": ({"f0_iqr": STEP}, "f0_iqr", 1), "monotone": ({"f0_iqr": -STEP}, "f0_iqr", -1),
    "rising_end": ({"f0_end": STEP}, "f0_end", 1), "falling_end": ({"f0_end": -STEP}, "f0_end", -1),
    "more_pauses": ({"pause": STEP}, "pause", 1),
    "brighter": ({"tilt": STEP}, "tilt", 1), "breathier": ({"cpps": -STEP, "h1h2": STEP}, "cpps", -1),
    "whisper_ctrl": ({"voiced": -4.0, "cpps": -4.0}, "voiced", -1),
}
TAG_DIRECTIONS = {  # style-tag only (no continuous control): which dim should move, which way
    "tag_whisper": ("whisper", "voiced", -1), "tag_loud": ("loud", "loud", 1), "tag_fast": ("fast", "rate", 1),
    "tag_slow": ("slow", "rate", -1), "tag_high_pitch": ("high_pitch", "f0_med", 1),
    "tag_anger": ("anger", "loud", 1), "tag_sadness": ("sadness", "loud", -1),
}
_PUNCT = {i for i, s in enumerate(SYMBOLS) if len(s) == 1}


def zmeasure(wav, text, speaker, stats):
    ids, _, _ = to_phonemes(text)
    n_ph = sum(1 for i in ids if i != SP and i not in _PUNCT)
    raw = measure(wav, n_phones=n_ph, want_mel=False)["controls"]
    mu = np.array([stats["spk_mean"][speaker][k] for k in CONTROL_DIMS])
    sd = np.array([stats["pooled_sd"][k] for k in CONTROL_DIMS])
    return (raw - mu) / sd


def run(ckpt, voices, out=None, texts=TEXTS):
    """Returns (summary DataFrame, audio) where audio maps direction -> wav for the first voice/text."""
    s = Synth(ckpt)
    stats = s.stats
    rows, audio = [], {}
    for v in voices:
        for ti, text in enumerate(texts):
            wav = s(text, v)
            if not audio:
                audio["baseline"] = wav
            base = zmeasure(wav, text, v, stats)
            rows.append(dict(voice=v, text=ti, direction="baseline", **dict(zip(CONTROL_DIMS, base))))
            for name, (ctrl, _, _) in DIRECTIONS.items():
                wav = s(text, v, ctrl=ctrl)
                if v == voices[0] and ti == 0:
                    audio[name] = wav
                z = zmeasure(wav, text, v, stats)
                rows.append(dict(voice=v, text=ti, direction=name, **dict(zip(CONTROL_DIMS, z))))
            for name, (tag, _, _) in TAG_DIRECTIONS.items():
                if tag in s.tag2i:
                    wav = s(text, v, tag=tag)
                    if v == voices[0] and ti == 0:
                        audio[name] = wav
                    z = zmeasure(wav, text, v, stats)
                    rows.append(dict(voice=v, text=ti, direction=name, **dict(zip(CONTROL_DIMS, z))))
    df = pd.DataFrame(rows)
    base = df[df.direction == "baseline"].set_index(["voice", "text"])[CONTROL_DIMS]
    res, leak = [], {}
    specs = {**{k: (v[1], v[2], abs(list(v[0].values())[0])) for k, v in DIRECTIONS.items()},
             **{k: (v[1], v[2], None) for k, v in TAG_DIRECTIONS.items()}}
    for name, (dim, sign, step) in specs.items():
        d = df[df.direction == name].set_index(["voice", "text"])[CONTROL_DIMS]
        if d.empty:
            continue
        delta = (d - base.loc[d.index]).astype(float)
        x = sign * delta[dim].dropna()
        try:
            p = wilcoxon(x, alternative="greater").pvalue if len(x) >= 5 else np.nan
        except ValueError:
            p = np.nan
        others = [c for c in CONTROL_DIMS if c != dim]
        res.append(dict(direction=name, dim=dim, n=len(x), median_delta=float(np.median(x)) if len(x) else np.nan,
                        steer=float(np.median(x) / step) if step and len(x) else np.nan,
                        hit=float((x > 0.5).mean()) if len(x) else np.nan, p=p,
                        leak=float(delta[others].abs().mean().mean())))
        leak[name] = delta.mean().round(2).to_dict()
    R = pd.DataFrame(res)
    # Holm correction across directions
    order = R.p.rank(method="first")
    m = R.p.notna().sum()
    R["p_holm"] = np.minimum(1, R.p * (m - order + 1))
    print(R.round(3).to_string(index=False))
    print("\nleakage (mean delta z, rows=direction):")
    print(pd.DataFrame(leak).T[CONTROL_DIMS].round(1).to_string())
    if out:
        json.dump(dict(summary=R.to_dict(orient="records"), leakage=leak, voices=voices), open(out, "w"), indent=1)
    return R, audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--voices", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stats = json.load(open(__import__("steer.config", fromlist=["META"]).META / "control_stats.json"))
    voices = a.voices.split(",") if a.voices else [s for s in stats["speakers"] if s.startswith("ears")][:3]
    run(a.ckpt, voices, a.out)


if __name__ == "__main__":
    main()
