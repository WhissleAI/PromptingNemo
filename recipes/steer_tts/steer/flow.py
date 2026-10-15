"""Flow-matching mel decoder on top of a frozen SteerTTS ("refiner").

The L1 decoder predicts the conditional MEDIAN mel: upper harmonics and noise regions come out smeared (hazy
spectrograms even teacher-forced; val mel flat at ~0.645 since 260k with train ~= val). That floor is the objective,
not the step count. Here a conditional flow (OT-CFM, Lipman et al. 2023; as the Matcha-TTS decoder, Mehta et al.
2024) learns p(real mel | base mel mu, g): it SAMPLES the detail the regression averages away. Everything that
steers (durations, pitch, energy, the coarse spectrum, g) still comes from the base model, so all controls / tags /
CFG keep working; the flow only adds texture.

  train   x1 = real mel, mu = the base model's teacher-forced mel for the same clip (same frames), x0 ~ N(0, I)
          x_t = (1 - (1 - s) t) x0 + t x1,  regress v(x_t, t | mu, g) on x1 - (1 - s) x0   (s = 1e-4)
          on random 256-frame crops (the network is convolutional; mu carries the long-range structure)
  infer   Euler / midpoint from x0 ~ N(0, temp^2 I) at t=0 to t=1, mu from the base model's FREE-RUNNING output

python -m steer.flow gta   --base runs/spk_enc_v3/step305000.pt --out data/gta/v3_305k      (teacher-forced mu dump)
python -m steer.flow train --gta data/gta/v3_305k --run runs/flow_v3
python -m steer.flow eval  --flow runs/flow_v3/last.pt                                        (UTMOS L1 vs flow)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FEATS, META, N_MELS, ROOT, RUNS, SR

SIGMA = 1e-4
CROP = 256
SILENCE = -11.5


# ---------------------------------------------------------------- network
class TimeEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))

    def forward(self, t):  # [B] in [0, 1]
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = 1000 * t[:, None] * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class Block(nn.Module):
    """ConvNeXt block (depthwise dilated conv -> LN -> MLP) with AdaLN-zero conditioning on (t, g)."""

    def __init__(self, d, c, k=7, dil=1, mult=3):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, padding=dil * (k // 2), dilation=dil, groups=d)
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.mod = nn.Linear(c, 3 * d)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
        self.pw1, self.pw2 = nn.Linear(d, mult * d), nn.Linear(mult * d, d)

    def forward(self, x, m, c):  # x [B,d,T], m [B,1,T], c [B,c]
        sc, sh, gate = self.mod(c)[:, None].chunk(3, -1)
        h = self.dw(x * m).transpose(1, 2)
        h = self.ln(h) * (1 + sc) + sh
        h = self.pw2(F.gelu(self.pw1(h))) * gate
        return x + h.transpose(1, 2)


class AttnBlock(nn.Module):
    """Full self-attention over the sequence with AdaLN-zero conditioning: lets the audio prompt (prompted flow)
    reach every target frame, beyond the conv receptive field (~170 frames each side)."""

    def __init__(self, d, c, heads=6):
        super().__init__()
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.mod = nn.Linear(c, 3 * d)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, x, m, c):
        sc, sh, gate = self.mod(c)[:, None].chunk(3, -1)
        h = self.ln(x.transpose(1, 2)) * (1 + sc) + sh
        h, _ = self.att(h, h, h, key_padding_mask=~m[:, 0].bool(), need_weights=False)
        return x + (h * gate).transpose(1, 2) * m


def rope(x):  # x [B,H,T,dh]: rotary position embedding (Su et al. 2021) over the frame axis
    T, dh = x.shape[-2], x.shape[-1]
    f = torch.exp(-math.log(10000) * torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh)
    a = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * f[None]
    cos, sin = a.cos().to(x.dtype), a.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class PromptPool(nn.Module):
    """Perceiver-resampler pooling of the prompt (XTTS; SupertonicTTS's learnable timbre queries): n learnable
    queries cross-attend to the encoded prompt frames -> n global voice tokens. A learned null key is always
    present, so an item without a prompt still gets defined (prompt-free) tokens."""

    def __init__(self, d, n, heads=6, layers=2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n, d) * 0.02)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.heads = heads
        self.layers = nn.ModuleList([nn.ModuleDict(dict(
            lq=nn.LayerNorm(d), lk=nn.LayerNorm(d), q=nn.Linear(d, d), kv=nn.Linear(d, 2 * d), o=nn.Linear(d, d),
            lf=nn.LayerNorm(d), f=nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))))
            for _ in range(layers)])

    def forward(self, e, cm):  # e [B,d,T] encoded prompt (zero off-prompt), cm [B,1,T]
        B, d, T = e.shape
        kv_in = torch.cat([self.null.expand(B, 1, d), e.transpose(1, 2)], 1)
        keep = torch.cat([torch.ones(B, 1, device=e.device, dtype=torch.bool), cm[:, 0].bool()], 1)[:, None, None]
        q = self.q[None].expand(B, -1, -1)
        H = self.heads
        for l in self.layers:
            k, v = l["kv"](l["lk"](kv_in)).chunk(2, -1)
            qq = l["q"](l["lq"](q))
            sp = lambda z: z.view(B, -1, H, d // H).transpose(1, 2)
            h = F.scaled_dot_product_attention(sp(qq), sp(k), sp(v), attn_mask=keep)
            q = q + l["o"](h.transpose(1, 2).reshape(B, -1, d))
            q = q + l["f"](l["lf"](q))
        return q  # [B,n,d]


class RAttnBlock(nn.Module):
    """Self-attention with RoPE positions (frames), plus optional cross-attention to the pooled prompt tokens; both
    zero-gated (AdaLN-zero gate / zero output projection), so adding these blocks to a trained net changes nothing
    at init."""

    def __init__(self, d, c, heads=6, xattn=False):
        super().__init__()
        self.heads = heads
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.mod = nn.Linear(c, 3 * d)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.xattn = xattn
        if xattn:
            self.lx = nn.LayerNorm(d)
            self.xq, self.xkv, self.xo = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
            nn.init.zeros_(self.xo.weight); nn.init.zeros_(self.xo.bias)

    def forward(self, x, m, c, tok=None):  # x [B,d,T], tok [B,n,d]
        B, d, T = x.shape
        H = self.heads
        sp = lambda z: z.view(B, -1, H, d // H).transpose(1, 2)
        sc, sh, gate = self.mod(c)[:, None].chunk(3, -1)
        h = self.ln(x.transpose(1, 2)) * (1 + sc) + sh
        q, k, v = (sp(z) for z in self.qkv(h).chunk(3, -1))
        h = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=m[:, 0].bool()[:, None, None])
        h = self.o(h.transpose(1, 2).reshape(B, T, d)) * gate
        if self.xattn and tok is not None:
            hx = self.lx(x.transpose(1, 2) + h)
            k2, v2 = self.xkv(tok).chunk(2, -1)
            h = h + self.xo(F.scaled_dot_product_attention(sp(self.xq(hx)), sp(k2), sp(v2)).transpose(1, 2).reshape(B, T, d))
        return x + h.transpose(1, 2) * m


class FlowNet(nn.Module):
    """prompt=True (prompted flow): also takes a context mel ctx with mask cm (1 = real audio given: the reference
    clip's frames), as Voicebox / F5-TTS infilling. The context enters the input stack, a masked mean of it is added
    to the conditioning (global voice), and attention layers at attn_at carry it across the whole sequence.

    v5 additions (all zero-initialised, so a v4 checkpoint loads into a v5 net unchanged):
      rope_at   layers that get an extra RoPE self-attention block (RAttnBlock)
      n_ptok    pooled prompt tokens (PromptPool) that the RoPE blocks cross-attend to
      skips     E2-TTS-style long skips, block i -> block n-1-i
      align_at  layers whose target-frame mean is mapped to a speaker-verification vector (TLA-SA-style feature
                alignment loss; training only)"""

    def __init__(self, n_mel=N_MELS, d=384, n_layers=12, g_dim=128, dils=(1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4),
                 prompt=False, attn_at=(3, 7, 11), rope_at=(), n_ptok=0, skips=False, align_at=(), align_dim=512):
        super().__init__()
        self.cfg = dict(n_mel=n_mel, d=d, n_layers=n_layers, g_dim=g_dim, dils=tuple(dils), prompt=prompt,
                        attn_at=tuple(attn_at), rope_at=tuple(rope_at), n_ptok=n_ptok, skips=skips,
                        align_at=tuple(align_at), align_dim=align_dim)
        self.prompt = prompt
        self.inp = nn.Conv1d((3 * n_mel + 1) if prompt else 2 * n_mel, d, 3, padding=1)
        self.temb = TimeEmb(d)
        self.gemb = nn.Sequential(nn.Linear(g_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, d, dil=dils[i % len(dils)]) for i in range(n_layers)])
        if prompt:
            self.attn = nn.ModuleDict({str(i): AttnBlock(d, d) for i in attn_at})
            self.ctx_enc = nn.Sequential(nn.Conv1d(n_mel, d, 3, padding=1), nn.SiLU(), nn.Conv1d(d, d, 3, padding=1))
            self.ctx_out = nn.Linear(d, d)
            nn.init.zeros_(self.ctx_out.weight); nn.init.zeros_(self.ctx_out.bias)
            self.rattn = nn.ModuleDict({str(i): RAttnBlock(d, d, xattn=n_ptok > 0) for i in rope_at})
            self.pool = PromptPool(d, n_ptok) if n_ptok else None
        self.skips = skips
        if skips:
            self.skip = nn.ModuleDict({str(i): nn.Linear(d, d) for i in range(n_layers - n_layers // 2, n_layers)})
            for l in self.skip.values():
                nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
        self.align_at = tuple(align_at)
        if align_at:
            self.align = nn.ModuleDict({str(i): nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, align_dim))
                                        for i in align_at})
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.out = nn.Conv1d(d, n_mel, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t, mu, g, m, ctx=None, cm=None, feats=False):  # normalised mels [B,100,T]; m, cm [B,1,T]
        """feats=True also returns {layer: speaker vector predicted from the target-frame mean} for align_at."""
        c = self.temb(t) + self.gemb(g)
        tok = None
        if self.prompt:
            if cm is None:
                ctx, cm = torch.zeros_like(x), torch.zeros_like(m)
            ctx = ctx * cm
            e = self.ctx_enc(ctx) * cm
            c = c + self.ctx_out(e.sum(-1) / cm.sum(-1).clamp_min(1))
            if getattr(self, "pool", None) is not None:
                tok = self.pool(e, cm)
            h = self.inp(torch.cat([x, mu, ctx, cm], 1) * m)
        else:
            h = self.inp(torch.cat([x, mu], 1) * m)
        rattn = getattr(self, "rattn", {})
        stack, out = [], {}
        n = len(self.blocks)
        for i, b in enumerate(self.blocks):
            if self.skips and str(i) in self.skip:
                h = h + self.skip[str(i)](stack.pop().transpose(1, 2)).transpose(1, 2) * m
            h = b(h, m, c)
            if self.prompt and str(i) in self.attn:
                h = self.attn[str(i)](h, m, c)
            if str(i) in rattn:
                h = rattn[str(i)](h, m, c, tok)
            if self.skips and i < n // 2:
                stack.append(h)
            if feats and i in self.align_at:
                w = m * (1 - cm) if cm is not None else m
                out[i] = self.align[str(i)](((h * w).sum(-1) / w.sum(-1).clamp_min(1)).float())
        v = self.out(self.ln(h.transpose(1, 2)).transpose(1, 2)) * m
        return (v, out) if feats else v


class Flow(nn.Module):
    """FlowNet + mel normalisation; sample() maps a base mel to a refined mel (log-mel units, same frames)."""

    def __init__(self, mean=-5.0, std=2.5, mu_cmn=False, **kw):
        """mu_cmn (prompted flow v2): remove the per-band time-mean of the base mel (its static spectral envelope:
        the long-term voice colouring the base model guessed from a 512-d speaker vector) so the flow must take the
        voice from the audio prompt (or g); timing / pitch / energy / phonetic detail of mu are kept. Without it, mu
        made from the TRUE speaker vector already carries the training speaker's colouring and the prompt adds
        little (v3p @20k: Seed-TTS SIM 31.2 -> 33.2 only)."""
        super().__init__()
        self.net = FlowNet(**kw)
        self.mu_cmn = mu_cmn
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def norm(self, m):
        return (m - self.mean) / self.std

    def _cmn(self, mu_n, w):
        """mu_n minus its per-band mean over the frames w marks (the target frames)."""
        if not self.mu_cmn:
            return mu_n
        return mu_n - (mu_n * w).sum(-1, keepdim=True) / w.sum(-1, keepdim=True).clamp_min(1)

    def loss(self, mel, mu, g, m, cm=None, xv=None):
        """cm (prompted flow): 1 on context frames - their real mel is given (and replaces mu there), and only the
        remaining frames are scored.
        xv [B,512] (v5): speaker-verification vector of each target clip; then returns (fm, align) where align is
        the mean 1 - cos between it and the vector each align_at layer predicts from its target frames."""
        x1, mu = self.norm(mel), self.norm(mu)
        mu = self._cmn(mu, m if cm is None else m * (1 - cm))
        ctx = None
        if cm is not None:
            ctx = x1 * cm
            mu = torch.where(cm.bool(), x1, mu)
            m = m * (1 - cm)  # loss on the infilled frames only (the net still sees all of m via its inputs)
        B = x1.shape[0]
        # t: half uniform, half logit-normal (more weight mid-path, Esser et al. 2024)
        t = torch.where(torch.rand(B, device=x1.device) < 0.5, torch.rand(B, device=x1.device),
                        torch.sigmoid(torch.randn(B, device=x1.device)))
        x0 = torch.randn_like(x1)
        tt = t[:, None, None]
        xt = (1 - (1 - SIGMA) * tt) * x0 + tt * x1
        m_in = m if cm is None else m + cm  # frames the network sees
        u = x1 - (1 - SIGMA) * x0
        if xv is not None:
            v, fe = self.net(xt, t, mu, g, m_in, ctx, cm, feats=True)
            fm = (((v - u) ** 2) * m).sum() / (m.sum().clamp_min(1) * x1.shape[1])
            al = torch.stack([1 - F.cosine_similarity(p, xv.float(), -1) for p in fe.values()]).mean()
            return fm, al
        v = self.net(xt, t, mu, g, m_in, ctx, cm)
        return (((v - u) ** 2) * m).sum() / (m.sum().clamp_min(1) * x1.shape[1])

    @torch.no_grad()
    def sample(self, mu, g, m=None, steps=10, temp=0.8, method="midpoint", seed=None, ctx=None, cm=None, cfg=0.0):
        """ctx / cm (prompted flow): real mel on the context frames (log-mel units), mask 1 there; mu is replaced by
        the context on those frames, as in training. Returns every frame (callers slice off the context).
        cfg (prompted flow): classifier-free guidance on the prompt, v + cfg * (v - v_no_prompt); the no-prompt
        velocity is the same net with the context removed (trained on p_none of the items)."""
        m = torch.ones_like(mu[:, :1]) if m is None else m
        gen = None if seed is None else torch.Generator(device=mu.device).manual_seed(seed)
        mu_n = self._cmn(self.norm(mu), m if cm is None else m * (1 - cm))
        if cm is not None:
            ctx = self.norm(ctx) * cm
            mu_n = torch.where(cm.bool(), ctx, mu_n)
        x = temp * torch.randn(mu_n.shape, device=mu.device, generator=gen) * m
        ts = torch.linspace(0, 1, steps + 1, device=mu.device)

        def vel(x, t):
            v = self.net(x, t.expand(x.shape[0]), mu_n, g, m, ctx, cm)
            if cfg and cm is not None:
                v = v + cfg * (v - self.net(x, t.expand(x.shape[0]), mu_n, g, m, None, None))
            return v

        for i in range(steps):
            t0, dt = ts[i], ts[i + 1] - ts[i]
            v = vel(x, t0)
            if method == "midpoint":
                v = vel(x + 0.5 * dt * v, t0 + 0.5 * dt)
            x = x + dt * v
        return x * self.std + self.mean

    @staticmethod
    def load(path, device="cuda"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        f = Flow(**ck["cfg"])
        f.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
        return f.to(device).eval()


# ---------------------------------------------------------------- teacher-forced dump
def load_base(path, dev):
    from .model import SteerTTS
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SteerTTS(ck["n_speakers"], ck["n_tags"], **ck.get("arch", {}))
    model.load_state_dict(ck["model"])
    stats = json.load(open(Path(path).parent / "control_stats.json"))
    return model.to(dev).eval(), stats, ck


def select(df, n_libritts=60000, max_frames=1600, seed=0):
    rng = np.random.default_rng(seed)
    d = df[df.n_frames.between(40, max_frames)]
    lt = d[d.corpus == "libritts"]
    lt = lt.iloc[rng.permutation(len(lt))[:n_libritts]]
    return pd.concat([d[d.corpus != "libritts"], lt]).reset_index(drop=True)


@torch.no_grad()
def gta(a):
    from torch.utils.data import DataLoader
    from .data import FrameBucketSampler, TTSData, collate, load_meta, split
    from .xvec import load_all
    dev = "cuda"
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    (out / "mu").mkdir(parents=True, exist_ok=True)
    model, stats, ck = load_base(a.base, dev)
    df, _ = load_meta()
    tr, va = split(df)
    xvec = load_all()
    parts = {"train": select(tr[tr.id.isin(xvec)], a.n_libritts), "val": select(va[va.id.isin(xvec)], 600).head(1500)}
    n_env = ck["arch"].get("n_env", 0)
    for name, sub in parts.items():
        sub = sub[sub.speaker.isin(stats["speakers"])].reset_index(drop=True)
        ds = TTSData(sub, stats, train=False, xvec=xvec, env_dim=n_env)
        if a.swap:
            # voice-swapped dump: each item's speaker vector is one clip of ANOTHER speaker - half a near neighbour
            # (rank 1-10 by centroid cosine), half anyone - so mu / g carry a wrong-but-plausible voice, as the base
            # gives for an unseen (cloned) speaker; the prompted flow trained on it must take the voice from the prompt
            rng = np.random.default_rng(1)
            spks = sorted(ds.ref_pool)
            s2k = {sp: k for k, sp in enumerate(spks)}
            rows_of = [np.array([ds.xv_row[i] for i in ds.ref_pool[sp]]) for sp in spks]
            cen = np.stack([ds.xv_mat[r].astype(np.float32).mean(0) for r in rows_of])
            cen /= np.linalg.norm(cen, axis=1, keepdims=True)
            near = np.argsort(-(cen @ cen.T), 1)[:, 1:11]
            id2spk = dict(zip(sub.id, sub.speaker))
        dl = DataLoader(ds, batch_sampler=FrameBucketSampler(sub.n_frames, a.max_frames, shuffle=False),
                        collate_fn=collate, num_workers=10)
        G, ids, t0 = [], [], time.time()
        for k, b in enumerate(dl):
            b = {kk: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for kk, v in b.items()}
            if a.swap:
                sv = []
                for i in b["id"]:
                    k = s2k[id2spk[i]]
                    oth = near[k, rng.integers(near.shape[1])] if rng.random() < 0.5 else (k + 1 + rng.integers(len(spks) - 1)) % len(spks)
                    sv.append(ds.xv_mat[rng.choice(rows_of[oth])].astype(np.float32))
                b["spk_vec"] = torch.from_numpy(np.stack(sv)).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, o = model(b)
                g = model.condition(b["spk"], b["tag"], b["ctrl"], b["cmask"], b.get("spk_vec"), b.get("env"))
            mel_p = o["mel_p"].float().cpu().numpy()
            for j, i in enumerate(b["id"]):
                np.save(out / "mu" / f"{i}.npy", mel_p[j, :, : int(b["mel_lens"][j])].astype(np.float16))
            G.append(g.float().cpu().numpy()); ids += b["id"]
            if k % 200 == 0:
                print(f"  {name} {len(ids)}/{len(sub)}  {time.time() - t0:.0f}s", flush=True)
        G = np.concatenate(G)
        meta = sub.set_index("id").loc[ids, ["corpus", "speaker", "spk_loud", "n_frames"]].reset_index()
        np.save(out / f"{name}_g.npy", G.astype(np.float32))
        meta.to_parquet(out / f"{name}.parquet")
        print(f"{name}: {len(ids)} items", flush=True)
    json.dump(dict(base=str(a.base), swap=bool(a.swap)), open(out / "info.json", "w"))


# ---------------------------------------------------------------- training data
class CropData(torch.utils.data.Dataset):
    def __init__(self, gta_dir, part="train", crop=CROP, train=True, alt=None, p_alt=0.0):
        """alt: a second (voice-swapped, gta --swap) dump of the same items; each item takes its mu / g from it with
        probability p_alt."""
        from .data import LEVEL_DB
        self.dir, self.crop, self.train = Path(gta_dir), crop, train
        self.meta = pd.read_parquet(self.dir / f"{part}.parquet")
        self.G = np.load(self.dir / f"{part}_g.npy")
        self.alt, self.p_alt = (Path(alt) if alt else None), p_alt
        if self.alt is not None:
            am = pd.read_parquet(self.alt / f"{part}.parquet")
            self.alt_row = dict(zip(am.id, range(len(am))))
            self.alt_G = np.load(self.alt / f"{part}_g.npy")
        self.shift = ((LEVEL_DB - self.meta.spk_loud.to_numpy()) * np.log(10.0) / 20.0).astype(np.float32)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        try:
            return self._item(i)
        except Exception as e:  # a corrupt feature file (one bad CRC in 130k, 2026-10-10: zipfile.BadZipFile)
            print("skipping unreadable item", self.meta.id.iloc[i], type(e).__name__, flush=True)
            return self._item(np.random.randint(len(self)))

    def _item(self, i):
        r = self.meta.iloc[i]
        g = self.G[i]
        if self.alt is not None and r["id"] in self.alt_row and np.random.rand() < self.p_alt:
            mu = np.load(self.alt / "mu" / f"{r['id']}.npy").astype(np.float32)
            g = self.alt_G[self.alt_row[r["id"]]]
        else:
            mu = np.load(self.dir / "mu" / f"{r['id']}.npy").astype(np.float32)
        mel = np.load(FEATS / f"{r['id']}.npz")["mel"].astype(np.float32) + self.shift[i]
        T = min(mu.shape[1], mel.shape[1])
        o = np.random.randint(0, T - self.crop + 1) if (self.train and T > self.crop) else 0
        n = min(self.crop, T)
        return mel[:, o:o + n], mu[:, o:o + n], g


def context_mask(m, p_none=0.1, lo=0.1, hi=0.5, p_prefix=0.7, min_len=24, gen=None):
    """Context spans for prompted-flow training, [B,1,T] (1 = real mel given): per item none (p_none: the model
    must still work without a prompt), else lo..hi of its valid frames - a prefix (as a reference clip placed before
    the target at inference) with p_prefix, otherwise at a random position."""
    B, _, T = m.shape
    cm = torch.zeros_like(m)
    n = m[:, 0].sum(-1).long().tolist()
    r = torch.rand(B, 3, generator=gen).tolist()
    for b in range(B):
        if r[b][0] < p_none or n[b] < 3 * min_len:
            continue
        L = max(min_len, int((lo + (hi - lo) * r[b][1]) * n[b]))
        s0 = 0 if r[b][2] < p_prefix else int(torch.randint(0, n[b] - L + 1, (1,), generator=gen))
        cm[b, :, s0:s0 + L] = 1
    return cm


def crop_collate(items, crop=CROP):
    B = len(items)
    mel = torch.full((B, N_MELS, crop), SILENCE); mu = torch.full((B, N_MELS, crop), SILENCE)
    m = torch.zeros(B, 1, crop)
    for b, (x, u, _) in enumerate(items):
        n = x.shape[1]
        mel[b, :, :n], mu[b, :, :n], m[b, :, :n] = torch.from_numpy(x), torch.from_numpy(u), 1
    return mel, mu, torch.from_numpy(np.stack([g for *_, g in items])), m


def oversample_weights(meta):
    w = meta.corpus.map({"ears": 2.0, "expresso": 2.0, "cartesia": 1.0, "nvtts": 1.5, "libritts": 1.0}).fillna(1.0)
    return torch.as_tensor(w.to_numpy(), dtype=torch.double)


# ---------------------------------------------------------------- training
def train(a):
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torch.utils.tensorboard import SummaryWriter
    dev = "cuda"
    run = ROOT / a.run if not Path(a.run).is_absolute() else Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    gdir = ROOT / a.gta if not Path(a.gta).is_absolute() else Path(a.gta)
    alt = (ROOT / a.gta_alt if not Path(a.gta_alt).is_absolute() else Path(a.gta_alt)) if a.gta_alt else None
    ds = CropData(gdir, "train", crop=a.crop, alt=alt, p_alt=a.p_alt)
    vs = CropData(gdir, "val", crop=a.crop, train=False)
    # mel normalisation from a sample of real mels
    smp = np.concatenate([ds[i][0].ravel() for i in np.random.default_rng(0).choice(len(ds), 300, replace=False)])
    cfg = dict(mean=float(smp.mean()), std=float(smp.std()), d=a.d, n_layers=a.layers, g_dim=ds.G.shape[1],
               prompt=a.prompt, mu_cmn=a.mu_cmn)
    if a.attn_at:  # prompted flow: which layers get a (zero-gated) attention block; default (3, 7, 11)
        cfg["attn_at"] = [int(x) for x in a.attn_at.split(",")]
    base = json.load(open(gdir / "info.json"))["base"]  # the frozen SteerTTS whose mu / g this flow was trained on
    flow = Flow(**cfg).to(dev)
    if a.init and not (run / "last.pt").exists():
        # start from a trained (unprompted) flow: every shape-matching tensor, and its input conv into the first
        # 2 x 100 input channels of the wider prompted one (context channels start at zero)
        src = torch.load(ROOT / a.init if not Path(a.init).is_absolute() else a.init, map_location="cpu",
                         weights_only=False)
        src = src.get("ema", src.get("model"))
        own = flow.state_dict()
        for k, v in src.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
            elif k == "net.inp.weight" and own[k].shape[1] > v.shape[1]:
                w = torch.zeros_like(own[k]); w[:, : v.shape[1]] = v; own[k] = w
        cfg.update(mean=float(src["mean"]), std=float(src["std"]))
        own["mean"], own["std"] = src["mean"], src["std"]
        flow.load_state_dict(own)
        print("initialised from", a.init, flush=True)
    ema = Flow(**cfg).to(dev).eval(); ema.load_state_dict(flow.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"flow params {sum(p.numel() for p in flow.parameters()) / 1e6:.2f}M | norm mean {cfg['mean']:.2f} std {cfg['std']:.2f} "
          f"| train {len(ds)} val {len(vs)}", flush=True)
    opt = torch.optim.AdamW(flow.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.01)
    step = 0
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        flow.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print("resumed at", step, flush=True)
    sched = lambda s: min(1.0, s / 1000) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    dl = DataLoader(ds, batch_size=a.batch, sampler=WeightedRandomSampler(oversample_weights(ds.meta), 10 ** 8),
                    collate_fn=lambda it: crop_collate(it, a.crop), num_workers=a.workers, pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=4)
    vdl = DataLoader(vs, batch_size=a.batch, collate_fn=lambda it: crop_collate(it, a.crop), num_workers=4)
    tb = SummaryWriter(run / "tb")
    json.dump(vars(a), open(run / "args.json", "w"), indent=1)
    t0, acc = time.time(), []
    for mel, mu, g, m in dl:
        if step >= a.steps:
            break
        mel, mu, g, m = (x.to(dev, non_blocking=True) for x in (mel, mu, g, m))
        for pg in opt.param_groups:
            pg["lr"] = a.lr * sched(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow.loss(mel, mu, g, m, context_mask(m).to(dev) if a.prompt else None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), flow.parameters()):
                pe.lerp_(p, 1 - a.ema)
        step += 1
        acc.append(loss.item())
        if step % 100 == 0:
            l = float(np.mean(acc)); acc = []
            tb.add_scalar("train/fm", l, step); tb.add_scalar("train/grad_norm", float(gn), step)
            tb.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
            print(f"step {step} | fm {l:.4f} gn {float(gn):.2f} | {100 / (time.time() - t0):.1f} it/s", flush=True)
            t0 = time.time()
        if step % a.eval_every == 0 or step == a.steps:
            validate(ema, vdl, tb, step, dev)
            torch.save(dict(model=flow.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, cfg=cfg,
                            gta=str(gdir), base=base), run / "last.pt")
            if step % (a.eval_every * 4) == 0:
                torch.save(dict(ema=ema.state_dict(), step=step, cfg=cfg, base=base), run / f"step{step}.pt")
            t0 = time.time()
    tb.close()


@torch.no_grad()
def validate(ema, vdl, tb, step, dev, n_batches=12):
    """Val FM loss, plus sample quality where it can be measured against the real clip: L1 to the real mel of
    the base mel vs the flow sample (the sample SHOULD be a bit worse in L1 - it is a sample, not a mean), and the
    per-band variance ratio sample/real over time (texture: the base mel is ~0.3-0.5 in the upper bands, real 1.0)."""
    from .train import as_image
    fm, l1_mu, l1_s, vr_mu, vr_s, l1_ctx = [], [], [], [], [], []
    prompted = getattr(ema.net, "prompt", False)
    for k, (mel, mu, g, m) in enumerate(vdl):
        if k >= n_batches:
            break
        mel, mu, g, m = (x.to(dev) for x in (mel, mu, g, m))
        fm.append(ema.loss(mel, mu, g, m).item())
        s = ema.sample(mu, g, m, steps=10, seed=0)
        if prompted:  # a fixed 30% prefix of real mel as the prompt; scored on the rest (the infilled frames)
            cm = torch.zeros_like(m); nv = m[:, 0].sum(-1).long()
            for b in range(m.shape[0]):
                cm[b, :, : int(0.3 * nv[b])] = 1
            sc = ema.sample(mu, g, m, steps=10, seed=0, ctx=mel, cm=cm)
            mt = m * (1 - cm)
            l1_ctx.append((((sc - mel).abs() * mt).sum() / (mt.sum() * N_MELS)).item())
            l1_s.append((((s - mel).abs() * mt).sum() / (mt.sum() * N_MELS)).item())  # same frames, no prompt
            l1_mu.append((((mu - mel).abs() * mt).sum() / (mt.sum() * N_MELS)).item())
        n = m.sum() * N_MELS
        if not prompted:
            l1_mu.append((((mu - mel).abs() * m).sum() / n).item()); l1_s.append((((s - mel).abs() * m).sum() / n).item())
        def tvar(x):  # temporal variance of the frame-to-frame difference per band: texture / sharpness
            d = (x[..., 1:] - x[..., :-1]) * m[..., 1:]
            return (d ** 2).sum(-1) / m[..., 1:].sum(-1).clamp_min(1)
        vr = tvar(mel)[:, 50:].mean()
        vr_mu.append((tvar(mu)[:, 50:].mean() / vr).item()); vr_s.append((tvar(s)[:, 50:].mean() / vr).item())
        if k == 0:
            n0 = int(m[0, 0].sum())
            img = torch.cat([mel[0, :, :n0], mu[0, :, :n0], s[0, :, :n0]], 0).cpu().numpy()
            tb.add_image("val/real_base_flow", as_image(img), step)
    r = dict(fm=np.mean(fm), l1_base=np.mean(l1_mu), l1_flow=np.mean(l1_s), texture_base=np.mean(vr_mu),
             texture_flow=np.mean(vr_s), **({"l1_flow_prompted": np.mean(l1_ctx)} if prompted else {}))
    for k, v in r.items():
        tb.add_scalar(f"val/{k}", v, step)
    print(f"[val {step}] " + " ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)


# ---------------------------------------------------------------- end-to-end check
def evaluate(a):
    """Renders the quick_eval tts/clone/prompt items through the base (L1) path and the flow path and scores both
    (WER, UTMOS, SIM-o) with steer.quick_eval's scorers."""
    import soundfile as sf
    from . import quick_eval as qe
    from .synth import Synth
    from .prompt import PromptMapper
    out = qe.OUT
    base = a.base or torch.load(a.flow, map_location="cpu", weights_only=False)["base"]
    name = a.name or Path(a.flow).parent.name
    s = Synth(base, flow=a.flow, flow_steps=a.steps, flow_temp=a.temp, vocoder=a.vocoder)
    pm = PromptMapper()
    its = qe.items()
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(its):
        kw = {}
        if it["set"] == "clone":
            y = s(it["text"], None, ref_wav=it["ref"])
        elif it["set"] == "prompt":
            mp = pm(it["prompt"])
            tag = next((t for t, _ in mp["top_tags"] if t in s.tag2i), "none")
            y = s(it["text"], it["voice"], tag=tag, ctrl=mp["ctrl"] or None)
        else:
            y = s(it["text"], it["voice"])
        it["wav"] = str(d / f"{i:04d}.wav")
        sf.write(it["wav"], np.asarray(y, np.float32), SR)
    del s
    torch.cuda.empty_cache()
    qe.score(its)
    row = qe.summarize(name, its)
    json.dump(dict(summary=row, items=its), open(out / f"{name}.json", "w"), default=float)
    print(row, flush=True)


AB_CASES = [  # (public voice, text, tag, ctrl)
    ("voice_2184", "The meeting has been moved to Thursday afternoon at three.", "none", None),
    ("voice_1361", "She said the package should arrive sometime early next week.", "none", None),
    ("voice_2336", "I cannot believe you did that without asking me first!", "anger", None),
    ("voice_2184", "Please keep your voice down, everyone is sleeping.", "whisper", None),
    ("voice_2485", "Turn left at the second light and park behind the building.", "none", {"rate": 1.5}),
    ("voice_2336", "I really miss the old days when we were all together.", "sadness", None),
    ("voice_1361", "Honestly, that was the best news I have heard all year!", "happiness", None),
    ("voice_2485", "We will need two more chairs for the people arriving later.", "none", {"f0_med": -1.5, "rate": -1.0}),
]


def ab(a):
    """Listening set for the demo's A/B tab: the same cases through the base (L1) decoder and the flow at several
    training steps -> <out>/<case>/<variant>.wav + manifest.json (with UTMOS per file)."""
    import soundfile as sf
    import soxr
    from .synth import Synth
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    s = Synth(a.base)
    flows = [(f"flow {int(Path(p).stem[4:]) // 1000}k (T={a.temp})", Flow.load(ROOT / p))
             for p in a.flows]
    mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).cuda().eval()
    M = lambda y: float(mos(torch.from_numpy(soxr.resample(np.asarray(y, np.float32), SR, 16000))[None].cuda(), 16000)[0])
    man = []
    for k, (v, text, tag, ctrl) in enumerate(AB_CASES):
        d = out / f"case{k}"
        d.mkdir(exist_ok=True)
        _, mel = s(text, v, tag=tag, ctrl=ctrl, return_mel=True)
        # the same base mel for every variant (identical durations / pitch): only the decoder differs
        from .text import to_phonemes
        ids, _, _ = to_phonemes(text)
        c, m = s.control_tensors(ctrl)
        mel, aux = s.model.infer(torch.tensor([ids], device=s.dev), torch.tensor([len(s.spk2i)], device=s.dev),
                                 torch.tensor([s.tag2i.get(tag, 0)], device=s.dev), c, m, spk_vec=s.speaker_vector(v))
        variants = [("base L1 decoder (before)", mel)]
        variants += [(n, f.sample(mel.float(), aux["g"].float(), steps=10, temp=a.temp, seed=0)) for n, f in flows]
        row = dict(voice=v, text=text, tag=tag, ctrl=ctrl or {}, audio=[])
        for j, (n, mm) in enumerate(variants):
            y = s.vocos.decode(mm.float()).squeeze().cpu().numpy()
            y = y / max(1.0, float(np.abs(y).max()))
            fn = f"case{k}/{j}.wav"
            sf.write(out / fn, y, SR)
            row["audio"].append(dict(name=n, file=fn, utmos=round(M(y), 2)))
        man.append(row)
        print(k, [(x["name"], x["utmos"]) for x in row["audio"]], flush=True)
    json.dump(man, open(out / "manifest.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gta")
    g.add_argument("--base", required=True); g.add_argument("--out", required=True)
    g.add_argument("--n_libritts", type=int, default=60000); g.add_argument("--max_frames", type=int, default=24000)
    g.add_argument("--swap", action="store_true", help="speaker vectors of other speakers (voice-swapped mu / g)")
    t = sub.add_parser("train")
    t.add_argument("--gta", required=True); t.add_argument("--run", required=True)
    t.add_argument("--steps", type=int, default=120000); t.add_argument("--batch", type=int, default=48)
    t.add_argument("--lr", type=float, default=2e-4); t.add_argument("--ema", type=float, default=0.9995)
    t.add_argument("--d", type=int, default=384); t.add_argument("--layers", type=int, default=12)
    t.add_argument("--workers", type=int, default=10); t.add_argument("--eval_every", type=int, default=2500)
    t.add_argument("--prompt", action="store_true", help="prompted flow: train with real-mel context spans")
    t.add_argument("--crop", type=int, default=CROP); t.add_argument("--init", default=None)
    t.add_argument("--mu_cmn", action="store_true", help="remove the base mel's static spectral envelope (Flow.mu_cmn)")
    t.add_argument("--attn_at", default=None, help="prompted flow: comma list of layers with attention (default 3,7,11)")
    t.add_argument("--gta_alt", default=None, help="voice-swapped dump of the same items (gta --swap)")
    t.add_argument("--p_alt", type=float, default=0.5)
    e = sub.add_parser("eval")
    e.add_argument("--flow", required=True); e.add_argument("--base", default=None); e.add_argument("--name", default=None)
    e.add_argument("--vocoder", default=None); e.add_argument("--steps", type=int, default=10); e.add_argument("--temp", type=float, default=0.4)
    b = sub.add_parser("ab")
    b.add_argument("--base", required=True); b.add_argument("--flows", nargs="+", required=True)
    b.add_argument("--out", default="runs/_flow_ab"); b.add_argument("--temp", type=float, default=0.4)
    a = ap.parse_args()
    {"gta": gta, "train": train, "eval": evaluate, "ab": ab}[a.cmd](a)


if __name__ == "__main__":
    main()
