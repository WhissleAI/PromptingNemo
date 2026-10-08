"""Load a SteerTTS checkpoint and synthesize with explicit controls.

python -m steer.synth --ckpt runs/base/last.pt --text "..." --speaker ears_p004 --tag anger \
    --ctrl "loud=2,f0_med=1.5" --out out.wav
"""
import argparse
import json

import numpy as np
import soundfile as sf
import torch

from .config import META, SR, CONTROL_DIMS
from .data import PITCH_SCALE
from .model import SteerTTS
from .text import to_phonemes


class Synth:
    def __init__(self, ckpt, device="cuda"):
        self.dev = device
        ck = torch.load(ckpt, map_location=device)
        self.stats = json.load(open(META / "control_stats.json"))
        self.model = SteerTTS(ck["n_speakers"], ck["n_tags"]).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.spk2i = {s: i for i, s in enumerate(self.stats["speakers"])}
        self.tag2i = {t: i for i, t in enumerate(self.stats["tags"])}
        from vocos import Vocos
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()

    def control_tensors(self, ctrl: dict):
        c = torch.zeros(1, len(CONTROL_DIMS), device=self.dev)
        m = torch.zeros(1, len(CONTROL_DIMS), device=self.dev)
        for k, v in (ctrl or {}).items():
            i = CONTROL_DIMS.index(k)
            c[0, i], m[0, i] = float(v), 1.0
        return c, m

    @torch.no_grad()
    def __call__(self, text, speaker, tag="none", ctrl=None, pace=1.0, word_shift=None, return_mel=False):
        ids, widx, words = to_phonemes(text)
        ids_t = torch.tensor([ids], device=self.dev)
        widx_t = torch.tensor([widx], device=self.dev)
        c, m = self.control_tensors(ctrl)
        spk = torch.tensor([self.spk2i[speaker]], device=self.dev)
        tg = torch.tensor([self.tag2i.get(tag, 0)], device=self.dev)
        if word_shift:  # user units: pitch in semitones, energy in dB, dur factor
            word_shift = {w: dict(pitch=s.get("pitch", 0) / PITCH_SCALE, energy=s.get("energy", 0) / 10.0,
                                  dur=s.get("dur", 1.0)) for w, s in word_shift.items()}
        mel, aux = self.model.infer(ids_t, spk, tg, c, m, pace=pace, word_idx=widx_t, word_shift=word_shift)
        wav = self.vocos.decode(mel.float()).squeeze().cpu().numpy()
        return (wav, mel) if return_mel else wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--tag", default="none")
    ap.add_argument("--ctrl", default="")
    ap.add_argument("--pace", type=float, default=1.0)
    ap.add_argument("--out", default="out.wav")
    a = ap.parse_args()
    ctrl = dict(kv.split("=") for kv in a.ctrl.split(",") if kv)
    s = Synth(a.ckpt)
    wav = s(a.text, a.speaker, a.tag, {k: float(v) for k, v in ctrl.items()}, a.pace)
    sf.write(a.out, wav, SR)
    print(a.out, len(wav) / SR, "s")


if __name__ == "__main__":
    main()
