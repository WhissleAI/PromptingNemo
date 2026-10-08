"""Control readback network: mel -> speaker-relative control z-scores (differentiable).

Trained on ground-truth mels of real speech. Speaker-relative targets need a speaker reference, so the
network takes the mel and the speaker's mean neutral mel-frame (a 100-d vector) as input. Used in stage 2
as a cycle-consistency loss: synthesize with a (possibly perturbed) control vector, read the controls back
from the generated mel, penalise the mismatch on the requested dims.

python -m steer.readback train
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FEATS, META, CONTROL_DIMS

RB_DIMS = [d for d in CONTROL_DIMS if d not in ("rate", "pause", "f0_end")]  # timing/end contour come from predicted durations & phone pitch
RB_IDX = [CONTROL_DIMS.index(d) for d in RB_DIMS]


class Readback(nn.Module):
    def __init__(self, n_mels=100, d=192, n_out=len(RB_DIMS)):
        super().__init__()
        self.inp = nn.Conv1d(2 * n_mels, d, 5, padding=2)
        self.blocks = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2 * r, dilation=r) for r in (1, 2, 4, 1)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, d) for _ in self.blocks])
        self.att = nn.Conv1d(d, 1, 1)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, n_out))

    def forward(self, mel, mel_lens, spk_ref):  # mel [B,100,T], spk_ref [B,100]
        T = mel.shape[2]
        m = (torch.arange(T, device=mel.device)[None] < mel_lens[:, None]).float()[:, None]
        x = torch.cat([mel - spk_ref[..., None], spk_ref[..., None].expand_as(mel)], 1)
        h = F.silu(self.inp(x))
        for c, n in zip(self.blocks, self.norms):
            h = h + F.silu(n(c(h))) * m
        w = self.att(h).masked_fill(m == 0, -1e4).softmax(-1)  # attention pooling
        mean = (h * w).sum(-1)
        mx = (h * m - (1 - m) * 1e4).amax(-1)
        return self.out(torch.cat([mean, mx], -1))


def speaker_ref_mels(df):
    """Mean log-mel frame over each speaker's neutral utterances (speech frames only)."""
    ref = {}
    neu = df[(df.tag == "neutral") | (df.corpus == "libritts")]
    for spk, g in neu.groupby("speaker"):
        acc = []
        for uid in g.id.sample(min(len(g), 20), random_state=0):
            mel = np.load(FEATS / f"{uid}.npz")["mel"].astype(np.float32)
            e = mel.mean(0)
            acc.append(mel[:, e > np.percentile(e, 30)].mean(1))
        ref[spk] = np.mean(acc, 0)
    for spk in set(df.speaker) - set(ref):  # no neutral data: use all
        g = df[df.speaker == spk]
        ref[spk] = np.mean([np.load(FEATS / f"{u}.npz")["mel"].astype(np.float32).mean(1) for u in g.id[:20]], 0)
    return ref


def main():
    from .data import load_meta, split
    df, stats = load_meta()
    ref = speaker_ref_mels(df)
    np.save(META / "spk_ref_mel.npy", dict(ref), allow_pickle=True)
    tr, va = split(df)
    Z = lambda d: d[[f"z_{k}" for k in RB_DIMS]].to_numpy(np.float32)
    dev = "cuda"
    net = Readback().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)

    def batches(d, bs=24, shuffle=True):
        idx = np.random.permutation(len(d)) if shuffle else np.arange(len(d))
        for i in range(0, len(idx), bs):
            sub = d.iloc[idx[i:i + bs]]
            mels = [np.load(FEATS / f"{u}.npz")["mel"].astype(np.float32) for u in sub.id]
            # random crops (>= 3 s) so the head does not rely on full-utterance context only
            T = min(max(m.shape[1] for m in mels), 1400)
            M = np.full((len(mels), 100, T), -11.5, np.float32)
            L = []
            for j, m in enumerate(mels):
                m = m[:, :T]; M[j, :, :m.shape[1]] = m; L.append(m.shape[1])
            yield (torch.from_numpy(M).to(dev), torch.tensor(L, device=dev),
                   torch.from_numpy(np.stack([ref[s] for s in sub.speaker])).to(dev), torch.from_numpy(Z(sub)).to(dev))

    for ep in range(int(sys.argv[2]) if len(sys.argv) > 2 else 15):
        net.train(); tl = []
        for M, L, R, Y in batches(tr):
            p = net(M, L, R)
            ok = ~torch.isnan(Y)
            loss = F.smooth_l1_loss(p[ok], Y[ok])
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(float(loss))
        net.eval(); P, T_ = [], []
        with torch.no_grad():
            for M, L, R, Y in batches(va, shuffle=False):
                P.append(net(M, L, R).cpu().numpy()); T_.append(Y.cpu().numpy())
        P, T_ = np.concatenate(P), np.concatenate(T_)
        r = {d: round(float(pd.Series(P[:, i]).corr(pd.Series(T_[:, i]))), 3) for i, d in enumerate(RB_DIMS)}
        print(f"epoch {ep} train {np.mean(tl):.3f} | val corr {r}", flush=True)
    torch.save(dict(state=net.state_dict(), dims=RB_DIMS), META / "readback.pt")
    print("saved", META / "readback.pt")


if __name__ == "__main__" and sys.argv[1:2] == ["train"]:
    main()
