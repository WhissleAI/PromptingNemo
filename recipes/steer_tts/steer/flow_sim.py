"""Speaker-similarity fine-tune of a prompted flow (steer.flow --prompt).

The prompted flow is trained only with flow matching, so nothing asks its cloned renders to sound like the prompt
speaker. This fine-tune renders the way inference does - another clip of the same speaker as the prompt, the
item's (often voice-swapped, gta --swap) base mel as mu - and adds, through the last K sampler steps (DRaFT-K):
  sim   1 - cos(judge(render), x-vector of the prompt clip)                                  pull to the prompt voice
  neg   margin-gated push from the base render's voice on voice-swapped items (the wrong voice mu carries):
        relu(cos(render, base) - cos(render, prompt) + margin)                               off once the render is
                                                                                              closer to the prompt
  ctc   intelligibility guard: CTC of the transcript under frozen wav2vec2-base-960h on the render
  fm    the ordinary prompted flow-matching loss on a CropData batch (anchor against judge hacking)
The judge is WavLM-base-plus-sv (the x-vector model); the leaderboard scores SIM with a different model (WavLM-large
seed-tts-eval), so a gain there is not the judge being gamed.

python -m steer.flow_sim --init runs/flow_v3ps/step240000.pt --run runs/flow_sim_a --neg 0.25
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import FEATS, ROOT
from .flow import CropData, Flow, context_mask, crop_collate

GTA = "data/gta/v3_305k"
GTA_SWAP = "data/gta/v3_305k_swap"


class SimData(torch.utils.data.Dataset):
    """Whole short utterances (target) + a prompt crop from ANOTHER utterance of the same speaker."""

    def __init__(self, gta, swap, p_swap=0.7, lo=150, hi=450, prompt=(150, 300)):
        from .data import LEVEL_DB, load_meta
        from .xvec import load_all
        self.gta, self.swap, self.p_swap, self.prompt = Path(gta), Path(swap), p_swap, prompt
        meta = pd.read_parquet(self.gta / "train.parquet")
        meta["row"] = np.arange(len(meta))
        sm = pd.read_parquet(self.swap / "train.parquet")
        self.swap_row = dict(zip(sm.id, range(len(sm))))
        self.G, self.Gs = np.load(self.gta / "train_g.npy"), np.load(self.swap / "train_g.npy")
        df, _ = load_meta()
        text = dict(zip(df.id, df.text))
        xv = load_all()
        meta = meta[meta.id.isin(xv) & meta.id.map(lambda i: isinstance(text.get(i), str))]
        self.by_spk = meta.groupby("speaker").id.apply(list).to_dict()
        self.items = meta[meta.n_frames.between(lo, hi) & meta.speaker.map(lambda s: len(self.by_spk[s]) > 1)]
        self.items = self.items.reset_index(drop=True)
        self.text, self.xv = text, xv
        self.shift = dict(zip(meta.speaker, (LEVEL_DB - meta.spk_loud.to_numpy()) * np.log(10.0) / 20.0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        try:
            return self._item(i)
        except Exception:  # a corrupt feature file (zipfile.BadZipFile)
            return self._item(np.random.randint(len(self)))

    def _item(self, i):
        r = self.items.iloc[i]
        sh = np.float32(self.shift[r.speaker])
        swapped = r.id in self.swap_row and np.random.rand() < self.p_swap
        if swapped:
            mu, g = np.load(self.swap / "mu" / f"{r.id}.npy"), self.Gs[self.swap_row[r.id]]
        else:
            mu, g = np.load(self.gta / "mu" / f"{r.id}.npy"), self.G[r.row]
        refs = [k for k in self.by_spk[r.speaker] if k != r.id]
        ref = refs[np.random.randint(len(refs))]
        pm = np.load(FEATS / f"{ref}.npz")["mel"].astype(np.float32) + sh
        P = min(pm.shape[1], np.random.randint(*self.prompt))
        o = np.random.randint(0, pm.shape[1] - P + 1)
        return dict(mu=mu.astype(np.float32), g=g.astype(np.float32), pm=pm[:, o:o + P], pos=self.xv[ref],
                    text=self.text[r.id], swapped=swapped)


def sim_collate(items):
    from .config import N_MELS
    from .flow import SILENCE
    B = len(items)
    P = [x["pm"].shape[1] for x in items]; T = [x["mu"].shape[1] for x in items]
    L = max(p + t for p, t in zip(P, T))
    full = torch.full((B, N_MELS, L), SILENCE); m = torch.zeros(B, 1, L); cm = torch.zeros(B, 1, L)
    tgt = torch.zeros(B, 1, L)
    for b, x in enumerate(items):
        p, t = P[b], T[b]
        full[b, :, :p] = torch.from_numpy(x["pm"]); full[b, :, p:p + t] = torch.from_numpy(x["mu"])
        m[b, :, :p + t] = 1; cm[b, :, :p] = 1; tgt[b, :, p:p + t] = 1
    return dict(full=full, m=m, cm=cm, tgt=tgt, P=torch.tensor(P), T=torch.tensor(T),
                g=torch.from_numpy(np.stack([x["g"] for x in items])),
                pos=torch.from_numpy(np.stack([x["pos"] for x in items])), text=[x["text"] for x in items],
                swapped=torch.tensor([x["swapped"] for x in items], dtype=torch.float32))


def sample_grad(flow, mu, g, m, ctx, cm, steps=10, temp=0.4, k=2, seed=None):
    """Flow.sample (midpoint) with gradients through the last k steps only."""
    mu_n = flow._cmn(flow.norm(mu), m * (1 - cm))
    ctx_n = flow.norm(ctx) * cm
    mu_n = torch.where(cm.bool(), ctx_n, mu_n)
    gen = None if seed is None else torch.Generator(device=mu.device).manual_seed(seed)
    x = temp * torch.randn(mu_n.shape, device=mu.device, generator=gen) * m
    ts = torch.linspace(0, 1, steps + 1, device=mu.device)
    B = x.shape[0]
    for i in range(steps):
        t0, dt = ts[i], ts[i + 1] - ts[i]
        with torch.set_grad_enabled(i >= steps - k):
            v = flow.net(x, t0.expand(B), mu_n, g, m, ctx_n, cm)
            v = flow.net(x + 0.5 * dt * v, (t0 + 0.5 * dt).expand(B), mu_n, g, m, ctx_n, cm)
            x = x + dt * v
    return x * flow.std + flow.mean


class Judges:
    def __init__(self, dev):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WavLMForXVector
        from vocos import Vocos
        fz = lambda mm: mm.to(dev).eval().requires_grad_(False)
        self.voc = fz(Vocos.from_pretrained("charactr/vocos-mel-24khz"))
        self.sv = fz(WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv"))
        self.asr_proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.asr = fz(Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h"))
        self.dev = dev

    def wav16(self, mel, n):
        """mel [B,100,T] (target frames left-aligned), n frames -> 16 kHz audio [B,S] (zeroed past n), samples."""
        import torchaudio.functional as AF
        y = self.voc.head(self.voc.backbone(mel.float()))
        ns = (n * 256).clamp_max(y.shape[1])
        y = y * (torch.arange(y.shape[1], device=y.device)[None] < ns[:, None])
        return AF.resample(y, 24000, 16000), ns * 2 // 3

    def emb(self, y, n):
        msk = (torch.arange(y.shape[1], device=y.device)[None] < n[:, None]).float()
        return F.normalize(self.sv(input_values=y * msk, attention_mask=msk.long()).embeddings, dim=-1)

    def ctc(self, y, n, texts):
        msk = (torch.arange(y.shape[1], device=y.device)[None] < n[:, None]).float()
        mu = (y * msk).sum(1, keepdim=True) / msk.sum(1, keepdim=True)
        sd = (((y - mu) * msk) ** 2).sum(1, keepdim=True).div(msk.sum(1, keepdim=True)).sqrt().clamp_min(1e-5)
        logits = self.asr((y - mu) / sd * msk).logits
        in_len = (n.float() / y.shape[1] * logits.shape[1]).long().clamp(1, logits.shape[1])
        lab = [self.asr_proc.tokenizer(re.sub(r"[^A-Z' ]", "", t.upper()).strip()).input_ids for t in texts]
        tgt = torch.tensor([t for l in lab for t in l], dtype=torch.long)
        return F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), tgt, in_len.cpu(), torch.tensor([len(l) for l in lab]),
                          blank=self.asr_proc.tokenizer.pad_token_id, zero_infinity=True, reduction="mean")


def left_align(mel, P, T):
    """[B,100,L] full render -> target frames of each item moved to the start, [B,100,max T]."""
    out = torch.zeros(mel.shape[0], mel.shape[1], int(T.max()), device=mel.device, dtype=mel.dtype)
    for b in range(mel.shape[0]):
        out[b, :, : T[b]] = mel[b, :, P[b]:P[b] + T[b]]
    return out


def train(a):
    dev = "cuda"
    run = ROOT / a.run
    run.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(run / "args.json", "w"), indent=1)
    src = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)
    flow = Flow.load(ROOT / a.init, dev).train()
    ema = Flow.load(ROOT / a.init, dev).eval().requires_grad_(False)
    J = Judges(dev)
    ds = SimData(ROOT / GTA, ROOT / GTA_SWAP, p_swap=a.p_swap)
    dl = iter(torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=6, collate_fn=sim_collate,
                                          persistent_workers=True, drop_last=True))
    fm_ds = CropData(ROOT / GTA, "train", crop=384, alt=ROOT / GTA_SWAP, p_alt=0.5)
    fm_dl = iter(torch.utils.data.DataLoader(fm_ds, batch_size=24, shuffle=True, num_workers=4, drop_last=True,
                                             collate_fn=lambda it: crop_collate(it, 384), persistent_workers=True))
    print(f"sim items {len(ds)} | init {a.init} | neg {a.neg} ctc {a.ctc} fm {a.fm}", flush=True)
    opt = torch.optim.AdamW(flow.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    acc, t0 = {}, time.time()
    for step in range(1, a.steps + 1):
        b = next(dl)
        full, m, cm, g = (b[k].to(dev) for k in ("full", "m", "cm", "g"))
        P, T = b["P"].to(dev), b["T"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mel = sample_grad(flow, full, g, m, full, cm, steps=a.flow_steps, k=a.k)
        mel = left_align(mel.float(), P, T)
        y, n = J.wav16(mel, T)
        e = J.emb(y, n)
        pos = F.normalize(b["pos"].to(dev), dim=-1)
        cpos = (e * pos).sum(-1)
        loss_sim = (1 - cpos).mean()
        out = dict(sim=loss_sim.item(), cos_pos=cpos.mean().item())
        loss = a.w_sim * loss_sim
        if a.neg > 0:
            with torch.no_grad():
                yb, nb = J.wav16(left_align(full, P, T), T)  # the base render (mu alone, through the vocoder)
                eb = J.emb(yb, nb)
            cneg = (e * eb).sum(-1)
            sw = b["swapped"].to(dev)  # only a swapped base render carries a WRONG voice to push from
            loss_neg = (F.relu(cneg - cpos + a.margin) * sw).sum() / sw.sum().clamp_min(1)
            loss = loss + a.neg * loss_neg
            out.update(neg=loss_neg.item(), cos_base=cneg.mean().item())
        if a.ctc > 0:
            loss_ctc = J.ctc(y, n, b["text"])
            loss = loss + a.ctc * loss_ctc
            out["ctc"] = loss_ctc.item()
        if a.fm > 0:
            mel_r, mu_r, g_r, m_r = (x.to(dev) for x in next(fm_dl))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_fm = flow.loss(mel_r, mu_r, g_r, m_r, context_mask(m_r).to(dev))
            loss = loss + a.fm * loss_fm
            out["fm"] = loss_fm.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), flow.parameters()):
                pe.lerp_(p, 1 - a.ema)
        for k_, v in out.items():
            acc.setdefault(k_, []).append(v)
        if step % 25 == 0:
            print(f"step {step} | " + " ".join(f"{k_} {np.mean(v):.4f}" for k_, v in acc.items()) +
                  f" | gn {float(gn):.2f} | {25 / (time.time() - t0):.2f} it/s", flush=True)
            acc, t0 = {}, time.time()
        if step % a.save_every == 0 or step == a.steps:
            torch.save(dict(ema=ema.state_dict(), step=step, cfg=src["cfg"], base=src.get("base")), run / f"step{step}.pt")
            torch.save(dict(ema=ema.state_dict(), step=step, cfg=src["cfg"], base=src.get("base")), run / "last.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--flow_steps", type=int, default=10); ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--w_sim", type=float, default=1.0); ap.add_argument("--neg", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.05); ap.add_argument("--ctc", type=float, default=0.1)
    ap.add_argument("--fm", type=float, default=1.0); ap.add_argument("--p_swap", type=float, default=0.7)
    ap.add_argument("--save_every", type=int, default=500)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
