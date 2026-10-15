"""Prompted flow v5: train the voice-cloning refiner the way it is used, on all the data.

What v1-v4 got wrong (2026-10-10 review, Seed SIM ~37 vs 72 for real speech):
  * data: a fixed teacher-forced mu dump of 131k clips / 220 h (LibriTTS capped at 60k clips). Here mu is computed
    ONLINE by the frozen base model on every batch, so the whole corpus (~410k clips, ~660 h, ~4.9k speakers)
    is used with no disk dump.
  * train/test mismatch: v1-v4 trained on 384-frame crops whose prompt was 10-50 % of the SAME crop (<= 2 s), while
    Synth gives up to 8 s of a DIFFERENT recording before the target (sequences up to ~1,700 frames). Here most items
    are [prompt crop of another clip of the same speaker | target crop], as at inference.
  * the mu shortcut: mu comes from the speaker vector, so the flow can "clean up mu" and ignore the prompt. Here a
    share of items (p_swap) render mu / g from a random OTHER speaker's vector; the voice must come from the prompt.
  * architecture (FlowNet v5 options): RoPE self-attention at the even layers (the odd ones already have attention),
    pooled prompt tokens every RoPE block cross-attends to, E2-style long skips, and a TLA-SA-style feature loss
    aligning target-frame features with the target clip's speaker vector (WavLM-base-plus-sv x-vectors, precomputed:
    no gradient through any judge, unlike flow_sim's DRaFT loss that overfit its judge).
All new modules start at zero, so --init from a v4 checkpoint starts exactly at v4.

python -m steer.pflow5 --init runs/flow_v4pa/step40000.pt --run runs/flow_v5
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import FEATS, N_MELS, ROOT
from .flow import SILENCE, Flow, context_mask, load_base

CORPUS_W = {"libritts": 1.0, "ears": 2.0, "expresso": 1.0, "cartesia": 1.5, "nvtts": 2.0}


class OnlineData(torch.utils.data.Dataset):
    """TTSData items (everything the base model needs for a teacher-forced pass) plus, with probability p_ref, a
    prompt: a random lo..hi-frame crop of another clip of the same speaker (level-normalised like the target), and
    the target clip's own speaker vector (for the alignment loss)."""

    def __init__(self, df, stats, xvec, env_dim, p_ref=0.7, lo=150, hi=600):
        from .data import TTSData
        self.tts = TTSData(df, stats, train=True, xvec=xvec, env_dim=env_dim)
        self.df = df
        self.p_ref, self.lo, self.hi = p_ref, lo, hi
        self.by_spk = {k: np.asarray(v) for k, v in df.groupby("speaker").indices.items()}
        self.spk_of = dict(zip(df.id, df.speaker))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from .data import LEVEL_DB
        it = self.tts[i]
        row = self.tts.xv_row.get(it["id"])
        it["xv_tgt"] = (self.tts.xv_mat[row] if row is not None else np.zeros(self.tts.xv_mat.shape[1])).astype(np.float32)
        it["pmel"] = None
        if np.random.rand() < self.p_ref:
            # (TTSData may have substituted another clip for an unreadable one: key on the id it returned)
            pool = self.by_spk[self.spk_of[it["id"]]]
            for _ in range(3):
                j = int(pool[np.random.randint(len(pool))])
                if self.df.id.iloc[j] == it["id"] and len(pool) > 1:
                    continue
                try:
                    pm = np.load(FEATS / f"{self.df.id.iloc[j]}.npz")["mel"].astype(np.float32)
                except Exception:  # noqa: BLE001  corrupt cached file (see data.TTSData)
                    continue
                pm = pm + np.float32((LEVEL_DB - self.df.spk_loud.iloc[j]) * np.log(10.0) / 20.0)
                L = int(np.random.randint(self.lo, self.hi + 1))
                if pm.shape[1] > L:
                    o = np.random.randint(0, pm.shape[1] - L + 1)
                    pm = pm[:, o:o + L]
                if pm.shape[1] >= 60:
                    it["pmel"] = pm
                break
        return it


def online_collate(items):
    from .data import collate
    b = collate(items)
    b["xv_tgt"] = torch.from_numpy(np.stack([x["xv_tgt"] for x in items]))
    b["pmel"] = [None if x["pmel"] is None else torch.from_numpy(x["pmel"]) for x in items]
    return b


def build_flow_batch(mel, mu, mel_lens, pmel, crop, gen=None):
    """[prompt | target crop] sequences (or a target crop with an in-clip context span from context_mask) ->
    mel, mu, m, cm [B,*,T] padded to the batch max."""
    seqs = []
    for k in range(mel.shape[0]):
        L = int(mel_lens[k])
        n = min(L, crop)
        o = int(torch.randint(0, L - n + 1, (1,))) if L > n else 0
        x, u = mel[k, :, o:o + n], mu[k, :, o:o + n]
        p = pmel[k]
        if p is not None:
            p = p.to(mel.device)
            seqs.append((torch.cat([p, x], 1), torch.cat([p, u], 1), p.shape[1]))
        else:
            seqs.append((x, u, -1))
    T = max(s[0].shape[1] for s in seqs)
    B = len(seqs)
    M = torch.full((B, N_MELS, T), SILENCE, device=mel.device); U = M.clone()
    m = torch.zeros(B, 1, T, device=mel.device); cm = torch.zeros_like(m)
    for k, (x, u, P) in enumerate(seqs):
        n = x.shape[1]
        M[k, :, :n], U[k, :, :n], m[k, :, :n] = x, u, 1
        if P > 0:
            cm[k, :, :P] = 1
    # items without a reference clip: an in-clip context span (or none, p_none) as v1-v4
    own = torch.tensor([s[2] < 0 for s in seqs], device=mel.device)
    if own.any():
        cm_own = context_mask(m.cpu(), gen=gen).to(mel.device)
        cm = torch.where(own[:, None, None], cm_own, cm)
    return M, U, m, cm


def train(a):
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from .data import FrameBucketSampler, load_meta, split
    from .xvec import load_all
    dev = "cuda"
    run = ROOT / a.run
    run.mkdir(parents=True, exist_ok=True)
    base, stats, bck = load_base(ROOT / a.base, dev)
    for p in base.parameters():
        p.requires_grad_(False)
    n_env = bck["arch"].get("n_env", 0)
    df, _ = load_meta()
    tr, _ = split(df)  # same split as the base model / gta dumps: never train on their val clips
    xvec = load_all()
    tr = tr[tr.id.isin(xvec) & tr.speaker.isin(stats["speakers"]) & tr.n_frames.between(80, a.max_utt)]
    tr = tr.reset_index(drop=True)
    ds = OnlineData(tr, stats, xvec, n_env, p_ref=a.p_ref, lo=a.ref_lo, hi=a.ref_hi)
    w = tr.corpus.map(CORPUS_W).fillna(1.0).to_numpy()
    print(f"online data: {len(tr)} clips, {tr.n_frames.sum() / 93.75 / 3600:.0f} h, {tr.speaker.nunique()} speakers",
          flush=True)
    sampler = FrameBucketSampler(tr.n_frames, a.max_frames, weights=w, max_items=a.max_items)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=online_collate, num_workers=a.workers, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)

    src = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)
    cfg = dict(src["cfg"])
    n_layers = cfg.get("n_layers", 12)
    have = set(cfg.get("attn_at", (3, 7, 11)))
    cfg.update(rope_at=[i for i in range(n_layers) if i not in have] if a.rope else [], n_ptok=a.n_ptok,
               skips=a.skips, align_at=[int(x) for x in a.align_at.split(",")] if a.align_at else [])
    flow = Flow(**cfg).to(dev)
    own = flow.state_dict()
    sd = src.get("ema", src.get("model"))
    miss = [k for k in own if k not in sd]
    for k, v in sd.items():
        assert k in own and own[k].shape == v.shape, k
        own[k] = v
    flow.load_state_dict(own)
    print(f"init {a.init}: {len(sd)} tensors loaded, {len(miss)} new | flow params "
          f"{sum(p.numel() for p in flow.parameters()) / 1e6:.2f}M | cfg {cfg}", flush=True)
    ema = Flow(**cfg).to(dev).eval(); ema.load_state_dict(flow.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    new = [p for n, p in flow.named_parameters() if n in miss]
    old = [p for n, p in flow.named_parameters() if n not in miss]
    opt = torch.optim.AdamW([dict(params=old, lr=a.lr), dict(params=new, lr=a.lr * a.new_lr_mult)],
                            betas=(0.9, 0.99), weight_decay=0.01)
    base_lrs = [a.lr, a.lr * a.new_lr_mult]
    step = 0
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        flow.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print("resumed at", step, flush=True)
    sched = lambda s: min(1.0, s / 1000) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    tb = SummaryWriter(run / "tb")
    json.dump(vars(a), open(run / "args.json", "w"), indent=1)
    xv_all = torch.from_numpy(ds.tts.xv_mat.astype(np.float32)).to(dev)
    t0, acc = time.time(), {"fm": [], "align": [], "T": [], "B": []}
    while step < a.steps:
        for b in dl:
            if step >= a.steps:
                break
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            B = b["mel"].shape[0]
            # voice-swapped share: mu / g from a random clip's speaker vector (mostly another speaker)
            sw = torch.rand(B, device=dev) < a.p_swap
            if sw.any():
                rnd = xv_all[torch.randint(len(xv_all), (B,), device=dev)]
                b["spk_vec"] = torch.where(sw[:, None], rnd, b["spk_vec"])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, o = base(b)
                g = base.condition(b["spk"], b["tag"], b["ctrl"], b["cmask"], b.get("spk_vec"), b.get("env"))
            mel, mu, m, cm = build_flow_batch(b["mel"], o["mel_p"].float(), b["mel_lens"], b["pmel"], a.crop)
            for pg, lr0 in zip(opt.param_groups, base_lrs):
                pg["lr"] = lr0 * sched(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if cfg["align_at"]:
                    fm, al = flow.loss(mel, mu, g.float(), m, cm, xv=b["xv_tgt"])
                    loss = fm + a.w_align * al
                else:
                    fm = loss = flow.loss(mel, mu, g.float(), m, cm)
                    al = torch.zeros(())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), flow.parameters()):
                    pe.lerp_(p, 1 - a.ema)
            step += 1
            acc["fm"].append(fm.item()); acc["align"].append(al.item()); acc["T"].append(mel.shape[-1]); acc["B"].append(B)
            if step % 100 == 0:
                r = {k: float(np.mean(v)) for k, v in acc.items()}
                acc = {k: [] for k in acc}
                for k, v in r.items():
                    tb.add_scalar(f"train/{k}", v, step)
                tb.add_scalar("train/grad_norm", float(gn), step)
                print(f"step {step} | fm {r['fm']:.4f} align {r['align']:.4f} gn {float(gn):.2f} | B {r['B']:.0f} "
                      f"T {r['T']:.0f} | {100 / (time.time() - t0):.2f} it/s | "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
                t0 = time.time()
            if step % a.save_every == 0 or step == a.steps:
                tmp = run / "last.pt.tmp"
                torch.save(dict(model=flow.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step,
                                cfg=cfg, base=str(a.base)), tmp)
                tmp.replace(run / "last.pt")
                torch.save(dict(ema=ema.state_dict(), step=step, cfg=cfg, base=str(a.base)), run / f"step{step}.pt")
                t0 = time.time()
    tb.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="runs/spk_enc_v3/step312500.pt")
    ap.add_argument("--steps", type=int, default=100000); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--new_lr_mult", type=float, default=2.0); ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--max_frames", type=int, default=16000, help="base-pass frame budget per batch (full clips)")
    ap.add_argument("--max_items", type=int, default=24); ap.add_argument("--max_utt", type=int, default=1400)
    ap.add_argument("--crop", type=int, default=512, help="target frames per item")
    ap.add_argument("--p_ref", type=float, default=0.7); ap.add_argument("--ref_lo", type=int, default=150)
    ap.add_argument("--ref_hi", type=int, default=600); ap.add_argument("--p_swap", type=float, default=0.3)
    ap.add_argument("--rope", type=int, default=1); ap.add_argument("--n_ptok", type=int, default=32)
    ap.add_argument("--skips", type=int, default=1)
    ap.add_argument("--align_at", default="5,8,11"); ap.add_argument("--w_align", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--save_every", type=int, default=5000)
    main_args = ap.parse_args()
    train(main_args)


if __name__ == "__main__":
    main()
