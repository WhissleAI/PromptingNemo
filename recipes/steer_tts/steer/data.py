"""Dataset, length-bucketed batch sampler and collate for SteerTTS."""
import json
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from scipy.stats import betabinom
from torch.utils.data import Dataset, Sampler

from .config import FEATS, META, CONTROL_DIMS

ZCOLS = [f"z_{k}" for k in CONTROL_DIMS]
PITCH_SCALE = 3.0  # frame pitch targets are semitones re speaker median / 3 (~unit variance)


# per-worker cache: entries are [T, N] float32 (up to a few MB), so keep it small -- maxsize=20000 grew to
# ~6 GB per DataLoader worker and OOM-froze the machine at 12 workers
@lru_cache(maxsize=256)
def beta_binomial_prior(n_text, n_mel, scaling=1.0):
    x = np.arange(0, n_text)[None, :]
    i = np.arange(1, n_mel + 1)[:, None]
    return betabinom.pmf(x, n_text - 1, scaling * i, scaling * (n_mel + 1 - i)).astype(np.float32)  # [T, N]


def load_meta():
    df = pd.read_parquet(META / "controls.parquet")
    stats = json.load(open(META / "control_stats.json"))
    return df, stats


def split(df, val_frac=0.02, seed=0):
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(df)) < val_frac
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


class TTSData(Dataset):
    def __init__(self, df, stats, train=True, p_dim_drop=0.35, p_all_drop=0.15, p_tag_drop=0.3):
        self.df = df
        self.spk2i = {s: i for i, s in enumerate(stats["speakers"])}
        self.tag2i = {t: i for i, t in enumerate(stats["tags"])}
        self.train = train
        self.p_dim, self.p_all, self.p_tag = p_dim_drop, p_all_drop, p_tag_drop
        self.Z = df[ZCOLS].to_numpy(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        z = np.load(FEATS / f"{r['id']}.npz")
        mel = z["mel"].astype(np.float32)
        f0 = z["f0"][: mel.shape[1]]
        voiced = f0 > 0
        st = 12 * np.log2(np.maximum(f0, 1.0) / 100.0) - r.spk_f0_st
        # octave errors / whisper noise: drop implausible frames, and treat near-unvoiced utterances as unvoiced
        voiced &= (st > -12) & (st < 18)
        if voiced.mean() < 0.12:
            voiced[:] = False
        pitch = np.where(voiced, st / PITCH_SCALE, 0.0).astype(np.float32)
        energy = ((z["energy"][: mel.shape[1]] - r.spk_loud) / 10.0).astype(np.float32)
        ctrl = self.Z[i].copy()
        cmask = (~np.isnan(ctrl)).astype(np.float32)
        tag = self.tag2i.get(r.tag, 0)
        if self.train:
            cmask *= (np.random.rand(len(ctrl)) > self.p_dim)
            if np.random.rand() < self.p_all:
                cmask[:] = 0
            if np.random.rand() < self.p_tag:
                tag = 0
        ids = z["ids"].astype(np.int64)
        return dict(id=r["id"], ids=ids, widx=z["widx"].astype(np.int64), mel=mel, pitch=pitch, voiced=voiced,
                    energy=energy, ctrl=np.nan_to_num(ctrl), cmask=cmask, spk=self.spk2i[r.speaker], tag=tag)


def collate(items):
    B = len(items)
    N = max(len(x["ids"]) for x in items)
    T = max(x["mel"].shape[1] for x in items)
    ids = torch.zeros(B, N, dtype=torch.long)
    widx = torch.full((B, N), -1, dtype=torch.long)
    mel = torch.full((B, 100, T), -11.5)  # log(1e-5) ~ silence
    pitch = torch.zeros(B, T); energy = torch.zeros(B, T); voiced = torch.zeros(B, T, dtype=torch.bool)
    prior = torch.zeros(B, T, N)
    tl, ml = [], []
    for b, x in enumerate(items):
        n, t = len(x["ids"]), x["mel"].shape[1]
        ids[b, :n] = torch.from_numpy(x["ids"]); widx[b, :n] = torch.from_numpy(x["widx"])
        mel[b, :, :t] = torch.from_numpy(x["mel"])
        pitch[b, :t] = torch.from_numpy(x["pitch"]); energy[b, :t] = torch.from_numpy(x["energy"])
        voiced[b, :t] = torch.from_numpy(x["voiced"])
        prior[b, :t, :n] = torch.from_numpy(beta_binomial_prior(n, t))
        tl.append(n); ml.append(t)
    return dict(ids=ids, widx=widx, text_lens=torch.tensor(tl), mel=mel, mel_lens=torch.tensor(ml), pitch=pitch,
                energy=energy, voiced=voiced, prior=prior,
                ctrl=torch.from_numpy(np.stack([x["ctrl"] for x in items])),
                cmask=torch.from_numpy(np.stack([x["cmask"] for x in items])),
                spk=torch.tensor([x["spk"] for x in items]), tag=torch.tensor([x["tag"] for x in items]),
                id=[x["id"] for x in items])


class FrameBucketSampler(Sampler):
    """Batches of similar length capped by total mel frames (memory ~ frames)."""

    def __init__(self, n_frames, max_frames=12000, shuffle=True, seed=0, weights=None, max_items=64):
        self.n = np.asarray(n_frames)
        self.max_frames, self.shuffle, self.seed, self.epoch = max_frames, shuffle, seed, 0
        self.max_items = max_items  # caps padding cost on the text side for batches of very short clips
        self.weights = weights  # optional per-item sampling weights (oversampling)

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.weights is not None:
            p = np.array(self.weights, dtype=float); p /= p.sum()
            idx = rng.choice(len(self.n), size=len(self.n), replace=True, p=p)
        else:
            idx = rng.permutation(len(self.n)) if self.shuffle else np.arange(len(self.n))
        # sort within large chunks for bucketing, keep randomness across chunks
        chunks = [idx[i:i + 2000] for i in range(0, len(idx), 2000)]
        batches = []
        for ch in chunks:
            ch = ch[np.argsort(self.n[ch])]
            cur, mx = [], 0
            for j in ch:
                mx2 = max(mx, self.n[j])
                if cur and (mx2 * (len(cur) + 1) > self.max_frames or len(cur) >= self.max_items):
                    batches.append(cur); cur, mx2 = [], self.n[j]
                cur.append(int(j)); mx = mx2
            if cur:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        b = self._batches()
        self.epoch += 1
        return iter(b)

    def __len__(self):
        return len(self._batches())
