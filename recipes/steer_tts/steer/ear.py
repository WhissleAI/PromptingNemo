"""Whissle ear: perceived-emotion probe on WavLM-base+ layer 6, trained on CREMA-D crowd soft labels.

Input is a *paired delta*: embedding(clip) - mean embedding(same speaker's neutral clips). At evaluation
the reference is the same voice's uninstructed render of the same text, matching the harness design.

python -m steer.ear train      -> data/meta/ear.pt (+ actor-disjoint CV report)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import META

STUDY = os.environ.get("EMOTION_STUDY_OUT", str(Path(__file__).resolve().parents[1] / "emotion_study" / "out"))
EMOS = ["NEU", "HAP", "SAD", "ANG", "FEA", "DIS"]
LAYER = 6


class Ear(nn.Module):
    def __init__(self, d=768, n=len(EMOS)):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lin = nn.Linear(d, n)

    def forward(self, delta):
        return self.lin(self.norm(delta))


def load_crema():
    clips = pd.read_parquet(f"{STUDY}/clips.parquet")
    E = np.load(f"{STUDY}/wavlm_layers.npy")[:, LAYER].astype(np.float32)
    neu_mean = {a: E[(clips.actor == a).to_numpy() & (clips.emo == "NEU").to_numpy()].mean(0) for a in clips.actor.unique()}
    D = E - np.stack([neu_mean[a] for a in clips.actor])
    P = clips[[f"p_{e}" for e in EMOS]].to_numpy(np.float32)
    ok = ~np.isnan(P).any(1)
    return D[ok], P[ok], clips.actor.to_numpy()[ok], clips.emo.to_numpy()[ok]


def fit(D, P, epochs=300, lr=3e-3, wd=1e-3):
    m = Ear()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    X, Y = torch.from_numpy(D), torch.from_numpy(P)
    for _ in range(epochs):
        opt.zero_grad()
        loss = -(Y * torch.log_softmax(m(X), -1)).sum(-1).mean()  # soft-label cross-entropy
        loss.backward()
        opt.step()
    return m.eval()


def main():
    D, P, actor, emo = load_crema()
    # actor-disjoint 5-fold CV: UAR vs acted label and vs crowd majority
    folds = np.array([hash(int(a)) % 5 for a in actor])
    pred = np.zeros(len(D), int)
    for f in range(5):
        m = fit(D[folds != f], P[folds != f])
        with torch.no_grad():
            pred[folds == f] = m(torch.from_numpy(D[folds == f])).argmax(-1).numpy()
    from sklearn.metrics import balanced_accuracy_score
    crowd = P.argmax(1)
    acted = np.array([EMOS.index(e) for e in emo])
    print(f"actor-disjoint UAR vs acted {balanced_accuracy_score(acted, pred):.3f} | vs crowd majority "
          f"{balanced_accuracy_score(crowd, pred):.3f}")
    m = fit(D, P)
    torch.save(dict(state=m.state_dict(), emos=EMOS, layer=LAYER), META / "ear.pt")
    print("saved", META / "ear.pt")


class EarScorer:
    """Scores synthesized audio pairs: probs(instructed - baseline) over EMOS."""

    def __init__(self, device="cuda"):
        from transformers import AutoFeatureExtractor, WavLMModel
        ck = torch.load(META / "ear.pt", map_location="cpu")
        self.ear = Ear(); self.ear.load_state_dict(ck["state"]); self.ear.eval().to(device)
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.wl = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
        self.dev = device

    @torch.no_grad()
    def embed(self, wav24):
        import soxr
        y = soxr.resample(wav24.astype(np.float32), 24000, 16000)
        x = torch.tensor(self.fe(y, sampling_rate=16000).input_values[0])[None].to(self.dev)
        return self.wl(x, output_hidden_states=True).hidden_states[LAYER][0].mean(0)

    @torch.no_grad()
    def probs(self, wav, ref_wav):
        return torch.softmax(self.ear(self.embed(wav) - self.embed(ref_wav)), -1).cpu().numpy()


if __name__ == "__main__" and sys.argv[1:] == ["train"]:
    main()
