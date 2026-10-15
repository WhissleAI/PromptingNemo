"""Prompt pathway: free-form voice direction -> (soft style tag, which control dims, z values).

Frozen sentence encoder (all-MiniLM-L6-v2, 22M, outside the TTS budget; embeddings are cacheable for a
controlled vocabulary) + a ~0.5M-parameter adapter trained on acoustically anchored captions.

python -m steer.prompt train            # -> data/meta/prompt_adapter.pt
python -m steer.prompt train_env        # + acoustic-environment heads (scene, room, phone, SNR, start/end sound
                                        #   event; steer.env_captions) -> data/meta/prompt_adapter_env.pt
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
    def __init__(self, n_tags, d_in=384, h=512, env=False):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h), nn.GELU())
        self.tag = nn.Linear(h, n_tags)
        self.spec = nn.Linear(h, N)
        self.val = nn.Linear(h, N)
        self.env = None
        if env:  # index 0 of every categorical head = none
            from .acoustic import ROOMS, SCENES
            from .text import EVENT_TOKENS
            self.env = nn.ModuleDict(dict(scene=nn.Linear(h, len(SCENES) + 1), room=nn.Linear(h, len(ROOMS) + 1),
                                          phone=nn.Linear(h, 1), snr=nn.Linear(h, 2),  # (value, given logit)
                                          ev_start=nn.Linear(h, len(EVENT_TOKENS) + 1), ev_end=nn.Linear(h, len(EVENT_TOKENS) + 1)))

    def forward(self, e):
        h = self.body(e)
        if self.env is None:
            return self.tag(h), self.spec(h), self.val(h)
        return self.tag(h), self.spec(h), self.val(h), {k: m(h) for k, m in self.env.items()}


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


def real_pairs(df, seed=0):
    """Pairs whose text is a real instruction/description and whose targets are the clip's MEASURED z-controls:
      cartesia      the instruction the clip was rendered with (3 paraphrases each). Specified dims = the ones the
                    instruction names a setting for (speed -> rate, volume -> loud); tag = its emotion tag.
      paraspeechcaps  train_base descriptions of EARS/Expresso clips we have (never its dev/test/holdout splits).
                    Specified dims: pitch level -> f0_med, speaking rate -> rate; tag = the clip's own tag.
    """
    import pandas as pd
    from .config import RAW
    rng = random.Random(seed)
    zc = [f"z_{k}" for k in CONTROL_DIMS]
    zi = dict(zip(df.id, df[zc].to_numpy(np.float32)))
    tag = dict(zip(df.id, df.tag))
    rows = []
    man = META / "manifest_cartesia.jsonl"
    if man.exists():
        from .cartesia_gen import instruction
        n0 = len(rows)
        for l in open(man):
            r = json.loads(l)
            if r["id"] not in zi:
                continue
            dims = (["rate"] if r["speed"] != 1.0 else []) + (["loud"] if r["volume"] != 1.0 else [])
            for k in range(3):
                # irritation-tagged clips: always re-phrase (their stored instruction said "angrily", see cartesia_gen)
                fresh = k > 0 or tag[r["id"]] == "irritation"
                txt = instruction(r["emotion"], r["speed"], r["volume"], rng) if fresh else r["instruction"]
                rows.append((txt, tag[r["id"]], dims, zi[r["id"]]))
        print(f"real pairs: cartesia {len(rows) - n0}", flush=True)
    psc = RAW / "paraspeechcaps" / "data" / "train_base-00000-of-00001.parquet"
    if psc.exists():
        from pathlib import Path
        t = pd.read_parquet(psc, columns=["source", "relative_audio_path", "text_description"])
        t = t[t.source.isin(["ears", "expresso"])]
        n0 = len(rows)
        for src, path, desc in zip(t.source, t.relative_audio_path, t.text_description):
            p = Path(path)
            uid = f"expresso_{p.stem}" if src == "expresso" else f"ears_{p.parent.name}_{p.stem}"
            if uid in zi:
                rows.append((str(desc[0]).strip(), tag[uid], ["f0_med", "rate"], zi[uid]))
        print(f"real pairs: paraspeechcaps {len(rows) - n0}", flush=True)
    return rows


def env_targets(pairs, seed=0, p_env=0.5, p_env_only=0.04):
    """Rewrites captions with an environment phrase (steer.env_captions) for ~half the pairs, and adds a few
    environment-only prompts; returns (pairs, targets) with targets[j] = dict of head indices (0 = none)."""
    from .acoustic import ROOMS, SCENES
    from .env_captions import combine, env_caption
    from .text import EVENT_TOKENS
    rng = random.Random(seed)
    none = dict(scene=None, room=None, phone=False, snr=None, ev_start=None, ev_end=None)
    out_p, out_t = [], []
    for t, g, dims, z in pairs:
        tgt = none
        if rng.random() < p_env:
            ph, tgt = env_caption(rng)
            t = combine(t, ph, rng)
        out_p.append((t, g, dims, z)); out_t.append(tgt)
    for _ in range(int(len(pairs) * p_env_only)):  # "rain in the background" alone: style untouched
        ph, tgt = env_caption(rng)
        if ph:
            out_p.append((ph, "none", [], np.full(N, np.nan, np.float32))); out_t.append(tgt)
    enc = lambda tg: dict(scene=SCENES.index(tg["scene"]) + 1 if tg["scene"] else 0,
                          room=ROOMS.index(tg["room"]) + 1 if tg["room"] else 0, phone=float(tg["phone"]),
                          snr=(tg["snr"] - 15) / 10 if tg["snr"] is not None else 0.0, snr_given=float(tg["snr"] is not None),
                          ev_start=EVENT_TOKENS.index(tg["ev_start"]) + 1 if tg["ev_start"] else 0,
                          ev_end=EVENT_TOKENS.index(tg["ev_end"]) + 1 if tg["ev_end"] else 0)
    return out_p, [enc(x) for x in out_t]


def train(env=False):
    from .data import load_meta
    df, stats = load_meta()
    tags = stats["tags"]
    t2i = {t: i for i, t in enumerate(tags)}
    pairs = build_pairs(df) + real_pairs(df)
    # a named dim whose measured z is undefined (e.g. f0 of a whispered clip) is dropped from that pair: one NaN
    # target turns every adapter weight NaN
    pairs = [(t, g, [d for d in dims if np.isfinite(z[CONTROL_DIMS.index(d)])], z) for t, g, dims, z in pairs]
    if env:
        pairs, etg = env_targets(pairs)
        ET = {k: torch.tensor([e[k] for e in etg]) for k in etg[0]}
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
    m = Adapter(len(tags), env=env).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    perm = torch.randperm(len(pairs)); n_val = len(pairs) // 50
    va, tr = perm[:n_val], perm[n_val:]
    for ep in range(12):
        m.train()
        for b in tr[torch.randperm(len(tr))].split(1024):
            x, yt, s, v = X[b].to(dev), Yt[b].to(dev), S[b].to(dev), V[b].to(dev)
            out = m(x)
            lt, ls, lv = out[:3]
            loss = F.cross_entropy(lt, yt) + F.binary_cross_entropy_with_logits(ls, s) + \
                (F.smooth_l1_loss(lv, v, reduction="none") * s).sum() / s.sum().clamp_min(1)
            if env:
                o, e = out[3], {k: t[b].to(dev) for k, t in ET.items()}
                loss = loss + sum(F.cross_entropy(o[k], e[k]) for k in ("scene", "room", "ev_start", "ev_end")) \
                    + F.binary_cross_entropy_with_logits(o["phone"][:, 0], e["phone"]) \
                    + F.binary_cross_entropy_with_logits(o["snr"][:, 1], e["snr_given"]) \
                    + (F.smooth_l1_loss(o["snr"][:, 0], e["snr"], reduction="none") * e["snr_given"]).sum() / e["snr_given"].sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            x = X[va].to(dev); out = m(x); lt, ls, lv = out[:3]
            acc = (lt.argmax(-1).cpu() == Yt[va]).float().mean()
            sp = ((torch.sigmoid(ls).cpu() > 0.5) == (S[va] > 0)).float().mean()
            mae = ((lv.cpu() - V[va]).abs() * S[va]).sum() / S[va].sum()
            ex = ""
            if env:
                ex = " | env acc " + " ".join(f"{k} {(out[3][k].argmax(-1).cpu() == ET[k][va]).float().mean():.3f}"
                                              for k in ("scene", "room", "ev_start", "ev_end"))
        print(f"epoch {ep} tag acc {acc:.3f} | spec acc {sp:.3f} | value MAE {mae:.3f} z{ex}", flush=True)
    f = META / ("prompt_adapter_env.pt" if env else "prompt_adapter.pt")
    torch.save(dict(state=m.state_dict(), tags=tags, env=env), f)
    print("saved", f)


class PromptMapper:
    def __init__(self, device="cuda", env=False):
        """env=True: the adapter with acoustic-environment heads (prompt_adapter_env.pt); results then also carry
        env=dict(scene, room, phone, snr) for Synth and events=dict(start, end) sound-event tokens."""
        ck = torch.load(META / ("prompt_adapter_env.pt" if env else "prompt_adapter.pt"), map_location=device)
        self.tags = ck["tags"]
        self.m = Adapter(len(self.tags), env=ck.get("env", False)).to(device).eval()
        self.m.load_state_dict(ck["state"])
        self.dev = device
        self.cache = {}

    @torch.no_grad()
    def __call__(self, prompt, spec_threshold=0.5, top_tags=3):
        key = " ".join(prompt.lower().split())
        if key not in self.cache:
            e = encoder(self.dev).encode([prompt], normalize_embeddings=True, convert_to_tensor=True).to(self.dev)
            out = self.m(e)
            lt, ls, lv = out[:3]
            self.cache[key] = (torch.softmax(lt, -1)[0], torch.sigmoid(ls)[0], lv[0], out[3] if len(out) > 3 else None)
        tp, sp, v, eo = self.cache[key]
        ctrl = {CONTROL_DIMS[i]: round(float(v[i]), 2) for i in range(N) if sp[i] > spec_threshold}
        top = torch.topk(tp, top_tags)
        res = dict(tag_probs=tp.cpu().numpy(), top_tags=[(self.tags[i], round(float(p), 3)) for p, i in
                                                         zip(top.values.tolist(), top.indices.tolist())], ctrl=ctrl)
        if eo is not None:
            from .acoustic import ROOMS, SCENES
            from .text import EVENT_TOKENS
            pick = lambda k, names: (lambda i: names[i - 1] if i > 0 else None)(int(eo[k][0].argmax()))
            env = dict(scene=pick("scene", SCENES), room=pick("room", ROOMS), phone=bool(eo["phone"][0, 0] > 0))
            if env["scene"] and eo["snr"][0, 1] > 0:
                env["snr"] = round(float(eo["snr"][0, 0]) * 10 + 15, 1)
            res["env"] = {k: v for k, v in env.items() if v}
            res["events"] = {k: pick(f"ev_{k}", EVENT_TOKENS) for k in ("start", "end")}
        return res


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "train_env":
        train(env=True)
    elif sys.argv[1] in ("try", "try_env"):
        pm = PromptMapper(env=sys.argv[1] == "try_env")
        for p in sys.argv[2:]:
            r = pm(p)
            print(f"{p!r}\n   tags {r['top_tags']}\n   ctrl {r['ctrl']}\n   env {r.get('env')} events {r.get('events')}")
