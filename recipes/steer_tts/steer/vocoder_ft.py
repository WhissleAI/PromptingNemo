"""Fine-tune the Vocos vocoder on the mels the TTS actually produces (GTA fine-tuning, as in HiFi-GAN / FastSpeech 2).

charactr/vocos-mel-24khz was trained on mels of real audio. Our renders are flow samples (steer.flow) of the base
model's mel: close to real but not identical, so the vocoder meets inputs slightly off its training distribution
(metallic / buzzy upper bands). Here each training segment pairs a generated mel with the REAL waveform of the same
clip, frame-aligned (the generated mel is teacher-forced: same durations as the recording):
  60%  flow sample of the teacher-forced base mel  (what synthesis feeds the vocoder)
  25%  real mel                                    (keeps copy-synthesis quality)
  15%  the base (L1) mel itself                    (the no-flow path stays usable)
Mels are level-normalised per speaker (steer.data.LEVEL_DB), so the target waveform is scaled by the same gain.

Losses as Vocos (Siuzdak 2023): hinge adversarial on a multi-period + multi-resolution discriminator, feature
matching, 45 x mel L1. The pretrained release has no discriminator weights, so they train alone first (--d_warmup).

python -m steer.vocoder_ft --gta data/gta/v3_305k --flow runs/flow_v3/last.pt --run runs/vocoder_v3
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .config import FEATS, HOP, N_MELS, ROOT, RUNS, SR

SEG = 64  # frames per training segment (16384 samples, as Vocos)
CTX = 48  # flow context frames on each side of the segment


class SegData(torch.utils.data.Dataset):
    def __init__(self, gta_dir, part="train", train=True):
        from .data import LEVEL_DB
        self.dir, self.train = Path(gta_dir), train
        meta = pd.read_parquet(self.dir / f"{part}.parquet")
        G = np.load(self.dir / f"{part}_g.npy")
        keep = (meta.n_frames >= SEG + 2 * CTX + 4).to_numpy()
        self.meta, self.G = meta[keep].reset_index(drop=True), G[keep]
        self.db = (LEVEL_DB - self.meta.spk_loud.to_numpy()).astype(np.float32)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        r = self.meta.iloc[i]
        rng = np.random if self.train else np.random.RandomState(i)
        mu = np.load(self.dir / "mu" / f"{r['id']}.npy").astype(np.float32)
        shift = np.float32(self.db[i] * np.log(10.0) / 20.0)
        mel = np.load(FEATS / f"{r['id']}.npz")["mel"].astype(np.float32) + shift
        T = min(mu.shape[1], mel.shape[1])
        o = rng.randint(CTX, T - SEG - CTX + 1)  # segment start frame
        y, _ = sf.read(ROOT / "data" / "wav24" / r.corpus / f"{r['id']}.wav", start=o * HOP, stop=(o + SEG) * HOP,
                       dtype="float32")
        y = (np.pad(y, (0, SEG * HOP - len(y))) * 10 ** (self.db[i] / 20)).astype(np.float32)
        w = slice(o - CTX, o + SEG + CTX)
        return mel[:, w], mu[:, w], self.G[i], y


def seg_collate(items):
    mel, mu, g, y = zip(*items)
    f = lambda xs: torch.from_numpy(np.stack(xs))
    return f(mel), f(mu), f(g), f(y)


def train(a):
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from vocos import Vocos
    from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
    from vocos.loss import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, MelSpecReconstructionLoss
    from .flow import Flow
    dev = "cuda"
    run = ROOT / a.run if not Path(a.run).is_absolute() else Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    gdir = ROOT / a.gta if not Path(a.gta).is_absolute() else Path(a.gta)
    flow = Flow.load(a.flow, dev) if a.flow else None
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev).train()
    mpd, mrd = MultiPeriodDiscriminator().to(dev), MultiResolutionDiscriminator().to(dev)
    opt_g = torch.optim.AdamW(voc.parameters(), lr=a.lr_g, betas=(0.8, 0.9))
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=a.lr_d, betas=(0.8, 0.9))
    dloss, gloss, fmloss = DiscriminatorLoss(), GeneratorLoss(), FeatureMatchingLoss()
    melloss = MelSpecReconstructionLoss(sample_rate=SR).to(dev)
    step = 0
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        voc.load_state_dict(ck["vocos"]); mpd.load_state_dict(ck["mpd"]); mrd.load_state_dict(ck["mrd"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"]); step = ck["step"]
        print("resumed at", step, flush=True)
    ds, vs = SegData(gdir, "train"), SegData(gdir, "val", train=False)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, collate_fn=seg_collate, num_workers=a.workers,
                    pin_memory=True, persistent_workers=True, drop_last=True)
    tb = SummaryWriter(run / "tb")
    json.dump(vars(a), open(run / "args.json", "w"), indent=1)
    print(f"vocoder ft: train {len(ds)} val {len(vs)} segments | flow {'yes' if flow else 'no'}", flush=True)
    seg = slice(CTX, CTX + SEG)
    if step == 0:
        validate(voc, flow, vs, tb, 0, dev, melloss)
    t0, acc = time.time(), {}
    while step < a.steps:
        for mel, mu, g, y in dl:
            if step >= a.steps:
                break
            mel, mu, g, y = (x.to(dev, non_blocking=True) for x in (mel, mu, g, y))
            with torch.no_grad():
                inp = make_input(mel, mu, g, flow, a)[..., seg]
            lr_scale = min(1.0, (step + 1) / 500) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
            for pg in opt_g.param_groups:
                pg["lr"] = a.lr_g * lr_scale
            y_hat = voc.head(voc.backbone(inp))  # (SEG - 1) * HOP samples, aligned at the segment start
            # (not voc.decode(): it runs under torch.inference_mode, so the generator would get no gradients)
            y = y[:, : y_hat.shape[1]]
            # discriminators
            r_mp, g_mp, _, _ = mpd(y, y_hat.detach())
            r_mr, g_mr, _, _ = mrd(y, y_hat.detach())
            ld = dloss(r_mp, g_mp)[0] + dloss(r_mr, g_mr)[0]
            opt_d.zero_grad(set_to_none=True); ld.backward(); opt_d.step()
            logs = dict(d=ld.item())
            if step >= a.d_warmup:  # generator (discriminators trained alone first: they start from scratch)
                r_mp, g_mp, fr_mp, fg_mp = mpd(y, y_hat)
                r_mr, g_mr, fr_mr, fg_mr = mrd(y, y_hat)
                lm = melloss(y_hat, y)
                la = gloss(g_mp)[0] + gloss(g_mr)[0]
                lf = fmloss(fr_mp, fg_mp) + fmloss(fr_mr, fg_mr)
                lg = la + lf + 45 * lm
                opt_g.zero_grad(set_to_none=True); lg.backward()
                torch.nn.utils.clip_grad_norm_(voc.parameters(), 10.0)
                opt_g.step()
                logs.update(g_adv=la.item(), g_fm=lf.item(), g_mel=lm.item())
            step += 1
            for k, v in logs.items():
                acc.setdefault(k, []).append(v)
            if step % 100 == 0:
                m = {k: float(np.mean(v)) for k, v in acc.items()}; acc = {}
                for k, v in m.items():
                    tb.add_scalar(f"train/{k}", v, step)
                print(f"step {step} | " + " ".join(f"{k} {v:.4f}" for k, v in m.items())
                      + f" | {100 / (time.time() - t0):.2f} it/s", flush=True)
                t0 = time.time()
            if step % a.eval_every == 0 or step == a.steps:
                validate(voc, flow, vs, tb, step, dev, melloss)
                torch.save(dict(vocos=voc.state_dict(), mpd=mpd.state_dict(), mrd=mrd.state_dict(),
                                opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict(), step=step), run / "last.pt")
                torch.save(dict(vocos=voc.state_dict(), step=step), run / f"vocos_{step}.pt")
                t0 = time.time()
    tb.close()


def make_input(mel, mu, g, flow, a):
    """Per item: flow sample / real mel / base mel (shares a.p_flow, a.p_real, rest)."""
    u = torch.rand(mel.shape[0], device=mel.device)
    out = torch.where((u < a.p_flow + a.p_real)[:, None, None], mel, mu)
    if flow is not None:
        sel = u < a.p_flow
        if sel.any():
            out[sel] = flow.sample(mu[sel], g[sel], steps=a.flow_steps, temp=a.flow_temp)
    return out


@torch.no_grad()
def validate(voc, flow, vs, tb, step, dev, melloss, n=48):
    """Mel L1 of the vocoded audio vs the real segment, per input type (real mel / base mel / flow sample)."""
    voc.eval()
    seg = slice(CTX, CTX + SEG)
    items = [vs[i] for i in range(min(n, len(vs)))]
    mel, mu, g, y = (x.to(dev) for x in seg_collate(items))
    srcs = {"real": mel, "base": mu}
    if flow is not None:
        srcs["flow"] = flow.sample(mu, g, steps=10, seed=0)
    res = {}
    for k, m in srcs.items():
        y_hat = voc.decode(m[..., seg])[:, : y.shape[1]]
        res[k] = melloss(y_hat, y[:, : y_hat.shape[1]]).item()
        tb.add_scalar(f"val/mel_l1_{k}", res[k], step)
        for j in range(3):
            tb.add_audio(f"val_{k}/{j}", (y_hat[j] / max(1.0, y_hat[j].abs().max().item())).cpu(), step, SR)
    if step == 0:
        for j in range(3):
            tb.add_audio(f"val_real_wav/{j}", (y[j] / max(1.0, y[j].abs().max().item())).cpu(), step, SR)
    print(f"[val {step}] " + " ".join(f"mel_l1_{k} {v:.4f}" for k, v in res.items()), flush=True)
    voc.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gta", required=True); ap.add_argument("--flow", default=None); ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr_g", type=float, default=5e-5); ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--d_warmup", type=int, default=2000); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--p_flow", type=float, default=0.6); ap.add_argument("--p_real", type=float, default=0.25)
    ap.add_argument("--flow_steps", type=int, default=8); ap.add_argument("--flow_temp", type=float, default=0.4)
    ap.add_argument("--eval_every", type=int, default=2000)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
