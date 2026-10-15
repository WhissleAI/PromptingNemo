"""Listening page: one or more checkpoints rendered side by side - plain voices, prompt+text pairs, control
directions, and (for reference-conditioned checkpoints) zero-shot cloning of unseen LibriTTS-R test speakers.

python -m steer.samples --ckpt spk_enc=runs/spk_enc/last.pt --ckpt m3hyb=runs/mix_m3hyb/final_116000.pt \
    --out runs/_listen_spk_enc      -> <out>/index.html (open in a browser; audio files next to it)
"""
import argparse
import html
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .config import ROOT, SR

TEXTS = ["The meeting has been moved to Thursday afternoon at three.",
         "Did you really leave the front door open all night?",
         "When the rain finally stopped, the children ran outside, splashing through every puddle they could find."]
PROMPTS = [("whisper it like a secret", "I hid the key under the third flowerpot by the door."),
           ("you are furious and shouting at your neighbour", "You promised me this would be done yesterday, and it still isn't!"),
           ("speak sadly, as if something bad just happened", "He was my best friend, and I never got to say goodbye."),
           ("cheerfully, like you just got great news", "They called this morning, and I got the job!"),
           ("a calm, slow meditation guide", "Breathe in slowly. Hold it. And now, gently let it go."),
           ("excited, talking fast and loud", "Tonight only, every ticket is half price, so get here now!"),
           ("sarcastically", "Oh great, another meeting that could have been an email."),
           ("in a deep, low voice, very slowly", "Long ago, in a village by the sea, there lived an old clockmaker."),
           ("bored and monotone, reading a list", "Item one, paper. Item two, staples. Item three, more paper."),
           ("nervous, with a shaky voice", "Um, I think I might have broken the printer again.")]
CTRLS = {"faster": {"rate": 2.0}, "slower": {"rate": -2.0}, "louder": {"loud": 2.0}, "quieter": {"loud": -2.0},
         "higher": {"f0_med": 2.0}, "lower": {"f0_med": -2.0}, "wider pitch range": {"f0_iqr": 2.0},
         "monotone": {"f0_iqr": -2.0}, "more pauses": {"pause": 2.0}, "whisper (controls)": {"voiced": -4.0, "cpps": -4.0}}
MARKUP = ["I did *not* say she took the money.",
          "Wait. [pause 600ms] Did you hear that? <whisper>Someone is in the house.</whisper>",
          "<slow>Take a deep breath,</slow> and then <fast>run as fast as you can!</fast>",
          "<sadness>I waited all night.</sadness> <style prompt=\"furious, shouting\">And you never even called!</style>",
          "The answer is <high>yes</high>, not <low>no</low>, and I'm <loud>certain</loud> of it."]


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path; the first should be reference-conditioned")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from .synth import Synth
    from .prompt import PromptMapper
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models = {n: Synth(p) for n, p in (c.split("=", 1) for c in a.ckpt)}
    pm = PromptMapper()
    from .voices import assign, public
    first = next(iter(models.values()))
    spk = first.stats["speakers"]
    assign(spk)
    # two expressive EARS voices, a LibriTTS-R voice, two Cartesia voices - shown only by public name (steer.voices)
    voices = ["ears_p001", "ears_p050", next(s for s in spk if s.startswith("libritts")),
              *[s for s in spk if s.startswith("cartesia")][:2]]
    label = public
    sections = []

    def render(case, row_label, fn):
        cells = {}
        for name, s in models.items():
            try:
                y = fn(s)
            except Exception as e:  # noqa: BLE001 - e.g. a voice the older checkpoint never saw
                cells[name] = f"<td><small>n/a ({html.escape(type(e).__name__)})</small></td>"; continue
            if y is None:
                cells[name] = "<td><small>n/a</small></td>"; continue
            f = out / slug(case) / f"{slug(row_label)}_{name}.wav"
            f.parent.mkdir(exist_ok=True)
            sf.write(f, np.asarray(y, np.float32), SR)
            cells[name] = f"<td><audio controls preload=none src='{f.relative_to(out)}'></audio></td>"
        return cells

    # A. plain voices
    rows = []
    for v in voices:
        for t in TEXTS[:2]:
            rows.append((f"{label(v)}<br><small>{html.escape(t)}</small>",
                         render("A_voices", f"{label(v)}_{t[:20]}", lambda s, v=v, t=t: s(t, v) if v in s.spk2i else None)))
    sections.append(("A. Voices (no direction)", rows))
    # B. prompts
    rows = []
    for p, t in PROMPTS:
        m = pm(p)
        tag = m["top_tags"][0][0]
        for v in voices[:2]:
            rows.append((f"<b>{html.escape(p)}</b> - {label(v)}<br><small>{html.escape(t)}<br>tag {tag}, ctrl {m['ctrl']}</small>",
                         render("B_prompts", f"{p[:25]}_{label(v)}", lambda s, v=v, t=t, tag=tag, c=m["ctrl"]:
                                s(t, v, tag=tag if tag in s.tag2i else "none", ctrl=c or None))))
    sections.append(("B. Prompt + text (prompt adapter -> style tag + controls)", rows))
    # C. control directions
    rows = []
    for d, c in {"baseline": None, **CTRLS}.items():
        rows.append((f"{d} <small>{c or ''}</small>", render("C_controls", d, lambda s, c=c: s(TEXTS[0], voices[0], ctrl=c))))
    sections.append((f"C. Control directions ({label(voices[0])})", rows))
    # D. zero-shot cloning of unseen LibriTTS-R test-clean speakers
    clone = ROOT / "data" / "bench" / "clone.jsonl"
    if clone.exists() and any(s.spk_ref is not None for s in models.values()):
        cl = pd.read_json(clone, lines=True).groupby("speaker").head(1).head(5)
        rows = []
        for k, (_, r) in enumerate(cl.iterrows(), 1):
            ref = out / "D_clone" / f"ref_unseen{k}.wav"; ref.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(r.ref_path, ref)
            refcell = f"reference (real recording, unseen speaker {k}):<br><audio controls preload=none src='{ref.relative_to(out)}'></audio>"
            for t in (r.text, TEXTS[1]):
                rows.append((f"{refcell}<br><small>{html.escape(t)}</small>",
                             render("D_clone", f"unseen{k}_{t[:20]}", lambda s, t=t, rp=r.ref_path:
                                    s(t, None, ref_wav=rp) if s.spk_ref is not None else None)))
                refcell = ""
        sections.append(("D. Zero-shot voice cloning (speakers never seen in training; one reference clip)", rows))
    # E. inline markup (steer.markup: per-word prosody, pauses, style spans; sentence-streamed)
    from .markup import MarkupSynth
    rows = []
    for mk in MARKUP:
        rows.append((f"<code>{html.escape(mk)}</code>", render("E_markup", mk[:40], lambda s, mk=mk:
                     MarkupSynth(s, pm).render(mk, voice=voices[0]))))
    sections.append((f"E. Inline markup ({label(voices[0])})", rows))
    head = "".join(f"<th>{html.escape(n)}</th>" for n in models)
    body = []
    for title, rows in sections:
        body.append(f"<tr><th colspan={len(models) + 1} class=sec>{html.escape(title)}</th></tr>")
        for lab, cells in rows:
            body.append(f"<tr><td class=lab>{lab}</td>{''.join(cells[n] for n in models)}</tr>")
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>SteerTTS samples</title><style>"
        "body{font:14px system-ui,sans-serif;margin:16px;background:#fafaf9;color:#1c1917}"
        "@media(prefers-color-scheme:dark){body{background:#1c1917;color:#f5f5f4}}"
        "th.sec{text-align:left;padding-top:22px;font-size:16px}td{padding:4px 8px;vertical-align:top}"
        "td.lab{max-width:420px}audio{width:260px}</style>"
        f"<h2>SteerTTS listening samples</h2><table><tr><th></th>{head}</tr>{''.join(body)}</table>")
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
