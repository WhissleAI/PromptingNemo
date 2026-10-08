"""Prompt pathway: free-form voice direction -> (soft style tag, which control dims, z values).

Frozen sentence encoder (all-MiniLM-L6-v2, 22M, outside the TTS budget; embeddings are cacheable for a
controlled vocabulary) + a ~0.5M-parameter adapter trained on acoustically anchored captions.

python -m steer.prompt train
python -m steer.prompt try "[speak sadly, as if something bad just happened]"
"""
import json
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import META, CONTROL_DIMS
from .captions import caption

ENC = "sentence-transformers/all-MiniLM-L6-v2"
N = len(CONTROL_DIMS)


class Adapter(nn.Module):
    def __init__(self, n_tags, d_in=384, h=512):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h), nn.GELU())
        self.tag = nn.Linear(h, n_tags)
        self.spec = nn.Linear(h, N)
        self.val = nn.Linear(h, N)

    def forward(self, e):
        h = self.body(e)
        return self.tag(h), self.spec(h), self.val(h)


_enc = None


def encoder(device="cuda"):
    global _enc
    if _enc is None:
        from sentence_transformers import SentenceTransformer
        _enc = SentenceTransformer(ENC, device=device)
    return _enc


def build_pairs(df, per_utt=6, seed=0):
    random.seed(seed)
    zc = [f"z_{k}" for k in CONTROL_DIMS]
    Z = df[zc].to_numpy(np.float32)
    rows = []
    for i, tag in enumerate(df.tag.to_numpy()):
        for _ in range(per_utt):
            txt, used_tag, dims = caption(tag, Z[i])
            rows.append((txt, used_tag, dims, Z[i]))
    return rows


def train():
    from .data import load_meta
    df, stats = load_meta()
    tags = stats["tags"]
    t2i = {t: i for i, t in enumerate(tags)}
    pairs = build_pairs(df)
    texts = [p[0] for p in pairs]
    uniq = sorted(set(texts))
    print(len(pairs), "pairs,", len(uniq), "unique captions", flush=True)
    E = encoder().encode(uniq, batch_size=512, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    emb = dict(zip(uniq, E))
    X = torch.tensor(np.stack([emb[t] for t in texts]))
    Yt = torch.tensor([t2i.get(p[1], 0) for p in pairs])
    S = torch.zeros(len(pairs), N)
    V = torch.zeros(len(pairs), N)
    for j, p in enumerate(pairs):
        for d in p[2]:
            k = CONTROL_DIMS.index(d); S[j, k] = 1; V[j, k] = float(p[3][k])
    dev = "cuda"
    m = Adapter(len(tags)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    perm = torch.randperm(len(pairs)); n_val = len(pairs) // 50
    va, tr = perm[:n_val], perm[n_val:]
    for ep in range(12):
        m.train()
        for b in tr[torch.randperm(len(tr))].split(1024):
            x, yt, s, v = X[b].to(dev), Yt[b].to(dev), S[b].to(dev), V[b].to(dev)
            lt, ls, lv = m(x)
            loss = F.cross_entropy(lt, yt) + F.binary_cross_entropy_with_logits(ls, s) + \
                (F.smooth_l1_loss(lv, v, reduction="none") * s).sum() / s.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            x = X[va].to(dev); lt, ls, lv = m(x)
            acc = (lt.argmax(-1).cpu() == Yt[va]).float().mean()
            sp = ((torch.sigmoid(ls).cpu() > 0.5) == (S[va] > 0)).float().mean()
            mae = ((lv.cpu() - V[va]).abs() * S[va]).sum() / S[va].sum()
        print(f"epoch {ep} tag acc {acc:.3f} | spec acc {sp:.3f} | value MAE {mae:.3f} z", flush=True)
    torch.save(dict(state=m.state_dict(), tags=tags), META / "prompt_adapter.pt")
    print("saved", META / "prompt_adapter.pt")


class PromptMapper:
    def __init__(self, device="cuda"):
        ck = torch.load(META / "prompt_adapter.pt", map_location=device)
        self.tags = ck["tags"]
        self.m = Adapter(len(self.tags)).to(device).eval()
        self.m.load_state_dict(ck["state"])
        self.dev = device
        self.cache = {}

    @torch.no_grad()
    def __call__(self, prompt, spec_threshold=0.5, top_tags=3):
        key = " ".join(prompt.lower().split())
        if key not in self.cache:
            e = encoder(self.dev).encode([prompt], normalize_embeddings=True, convert_to_tensor=True).to(self.dev)
            lt, ls, lv = self.m(e)
            self.cache[key] = (torch.softmax(lt, -1)[0], torch.sigmoid(ls)[0], lv[0])
        tp, sp, v = self.cache[key]
        ctrl = {CONTROL_DIMS[i]: round(float(v[i]), 2) for i in range(N) if sp[i] > spec_threshold}
        top = torch.topk(tp, top_tags)
        return dict(tag_probs=tp.cpu().numpy(), top_tags=[(self.tags[i], round(float(p), 3)) for p, i in
                                                          zip(top.values.tolist(), top.indices.tolist())], ctrl=ctrl)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "try":
        pm = PromptMapper()
        for p in sys.argv[2:]:
            r = pm(p)
            print(f"{p!r}\n   tags {r['top_tags']}\n   ctrl {r['ctrl']}")
