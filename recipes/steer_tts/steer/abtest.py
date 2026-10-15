"""Listening A/B: render the same voices x texts x directions from several checkpoints, side by side.

Writes <out>/<case>/<model>.wav, an index.html with the models' players in adjacent columns, and a
TensorBoard run (<out>/tb, audio tab) so the comparison can be heard from TB as well.

python -m steer.abtest --ckpt attn=runs/mix_s1/step40000.pt --ckpt m3hyb=runs/mix_m3hyb/step40000.pt \
    --out runs/_ab_40k
"""
import argparse
import html
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SR
from .harness import TEXTS

DIRS = {  # name -> (tag, ctrl)
    "baseline": ("none", None),
    "faster": ("none", {"rate": 2.0}), "slower": ("none", {"rate": -2.0}),
    "louder": ("none", {"loud": 2.0}), "quieter": ("none", {"loud": -2.0}),
    "higher": ("none", {"f0_med": 2.0}), "lower": ("none", {"f0_med": -2.0}),
    "wider_range": ("none", {"f0_iqr": 2.0}), "monotone": ("none", {"f0_iqr": -2.0}),
    "rising_end": ("none", {"f0_end": 2.0}), "more_pauses": ("none", {"pause": 2.0}),
    "whisper_ctrl": ("none", {"voiced": -4.0, "cpps": -4.0}),
    "tag_anger": ("anger", None), "tag_sadness": ("sadness", None), "tag_happiness": ("happiness", None),
    "tag_fear": ("fear", None), "tag_whisper": ("whisper", None),
}
# free-form direction -> a text that suits it (prompts are judged on matched text, as in the demo presets);
# the neutral harness texts below are rendered too, so text sentiment cannot carry the style on its own
PROMPTS = {
    "speak sadly, as if something bad just happened": "He was my best friend, and I never got to say goodbye.",
    "you are furious and shouting at your neighbour": "You promised me this would be done yesterday, and it still isn't!",
    "whisper it like a secret": "I hid the key under the third flowerpot by the door.",
    "a calm, slow meditation guide": "Breathe in slowly. Hold it. And now, gently let it go.",
    "excited, talking fast and loud": "Tonight only, every ticket is half price, so get here now!",
    "cheerfully, like you just got great news": "They called this morning, and I got the job!",
    "nervous, with a shaky voice": "Um, I think I might have broken the printer again.",
    "in a deep, low voice, very slowly": "Long ago, in a village by the sea, there lived an old clockmaker.",
    "bored and monotone, reading a list": "Item one, paper. Item two, staples. Item three, more paper.",
    "high pitched and questioning": "You're telling me the cat opened the fridge by itself?",
}
VARIED = [  # text-robustness cases rendered with no direction
    "Did you really leave the front door open all night?",
    "Your order number is 4 7 2 9, and the total comes to thirty two dollars and fifty cents.",
    "When the rain finally stopped, the children ran outside, splashing through every puddle they could find, "
    "while their parents watched from the porch.",
    "Doctor Patel will see Ms. Johnson at Saint Mary's Hospital on Monday.",
    "No.",
]


def slug(key):  # filesystem- and HTML-attribute-safe path
    return re.sub(r"[^A-Za-z0-9_/.-]+", "_", key.replace(": ", "_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voices", default="", help="comma list; default 3 EARS voices spread over the list")
    ap.add_argument("--texts", type=int, default=2)
    a = ap.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    from .synth import Synth
    from .prompt import PromptMapper
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models = dict(c.split("=", 1) for c in a.ckpt)
    pm = PromptMapper()
    cases = {k: (tag, ctrl, None) for k, (tag, ctrl) in DIRS.items()}  # None = the harness texts
    for p, txt in PROMPTS.items():
        r = pm(p)
        cases[f"prompt: {p}"] = (r["top_tags"][0][0], r["ctrl"] or None, [txt] + TEXTS[:1])
    for i, txt in enumerate(VARIED):
        cases[f"text {i}: {txt[:40]}"] = ("none", None, [txt])
    tb = SummaryWriter(out / "tb")
    voices = None
    for name, ck in models.items():
        s = Synth(ck)
        if voices is None:
            ears = [v for v in s.stats["speakers"] if v.startswith("ears")]
            voices = a.voices.split(",") if a.voices else [ears[0], ears[len(ears) // 2], ears[-1]]
        for v in voices:
            for case, (tag, ctrl, texts) in cases.items():
                for ti, text in enumerate(texts or TEXTS[: a.texts]):
                    key = f"{case}/{v}_t{ti}"
                    wav = np.asarray(s(text, v, tag=tag if tag in s.tag2i else "none", ctrl=ctrl), np.float32)
                    d = out / slug(key)
                    d.mkdir(parents=True, exist_ok=True)
                    sf.write(d / f"{name}.wav", wav, SR)
                    tb.add_audio(f"{key}/{name}", wav[None], 0, sample_rate=SR)
        print("rendered", name, flush=True)
    tb.close()
    # side-by-side page
    rows = []
    for case, (tag, ctrl, texts) in cases.items():
        texts = texts or TEXTS[: a.texts]
        rows.append(f"<tr><th colspan={len(models) + 1}>{html.escape(case)} <small>tag={tag} ctrl={ctrl}</small></th></tr>")
        for v in voices:
            for ti, text in enumerate(texts):
                d = slug(f"{case}/{v}_t{ti}")
                cells = "".join(f"<td><audio controls preload=none src='{d}/{m}.wav'></audio></td>" for m in models)
                rows.append(f"<tr><td>{v}<br><small>{html.escape(text[:60])}</small></td>{cells}</tr>")
    head = "".join(f"<th>{m}</th>" for m in models)
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>SteerTTS A/B</title><style>body{font:14px sans-serif}"
        "th{text-align:left;padding-top:12px}td{padding:2px 8px}</style>"
        f"<table><tr><th></th>{head}</tr>{''.join(rows)}</table>")
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
