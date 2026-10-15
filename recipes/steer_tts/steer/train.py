"""Train SteerTTS.

python -m steer.train --run base --max_frames 12000 --steps 200000
tensorboard --logdir $STEER_TTS_ROOT/runs

TensorBoard layout (runs/<run>/tb):
  train/*          losses, total loss, lr, grad norm, alignment-binarisation weight   (every --log_every)
  perf/*           steps/s, mel frames/s, utterances per batch, peak GPU memory, host RAM (main + workers), skipped (non-finite) steps
  val/*            validation losses                                                 (every --eval_every)
  val_audio/*      free-running synthesis of fixed validation utterances (+ vocoded ground truth once)
  val_mel/*        mel images: ground truth | teacher-forced | free-running; val_align/* soft alignment
  steer_ratio/*    harness: median achieved / requested z per direction  (1.0 = exact; every --harness_every)
  steer_hit/*      harness: share of paired renders moving the right way by > 0.5 z
  steer_leak/*     harness: mean |delta z| on the non-target dims
  steer_summary/*  means of the above over continuous-control and style-tag directions
  steer_audio/*    one text / voice rendered under every direction
  config/*         args, model size, data composition (text)
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import PROSODY_DIMS, RUNS, SR
from .data import TTSData, FrameBucketSampler, PairSampler, collate, load_meta, pair_delta_loss, pair_pool, split
from .model import SteerTTS
from .text import SYMBOLS

W = dict(mel=1.0, dur=0.1, pitch=0.1, energy=0.1, align=1.0)
N_VAL_SAMPLES = 4


def bin_weight(step, start=4000, ramp=4000):
    return 0.0 if step < start else min(1.0, (step - start) / ramp)


def lr_at(step, base, warmup=4000, total=200000):
    if step < warmup:
        return base * step / warmup
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / (total - warmup)))) * 0.95 + base * 0.05


def to(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def as_image(x):
    """2-D array -> [1,H,W] in [0,1], first row at the bottom (low mel bins / first tokens)."""
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)[::-1].copy()[None]


def init_from(model, path, stats, dev):
    """Initialise from a checkpoint whose speaker/tag vocabularies may differ.

    Shape-matching tensors are copied. Speaker and tag embedding rows are copied by NAME using the
    control_stats.json saved next to that checkpoint; speakers new to this run start at the mean
    of the known speaker rows, new tags at the 'none' row.
    """
    from pathlib import Path
    ck = torch.load(path, map_location=dev)
    src, own = ck["model"], model.state_dict()
    old_stats_f = Path(path).parent / "control_stats.json"
    old = json.load(open(old_stats_f)) if old_stats_f.exists() else None
    copied, skipped = 0, []
    for k, v in src.items():
        if k not in own:
            skipped.append(k)
        elif k in ("spk.weight", "tag.weight") and old is not None:
            names_old, names_new = (old["speakers"], stats["speakers"]) if k == "spk.weight" else (old["tags"], stats["tags"])
            o_idx = {n: i for i, n in enumerate(names_old)}
            w = own[k].clone()
            fill = v[:len(names_old)].mean(0) if k == "spk.weight" else v[o_idx["none"]]
            hit = 0
            for j, n in enumerate(names_new):
                if n in o_idx:
                    w[j] = v[o_idx[n]]; hit += 1
                else:
                    w[j] = fill
            if k == "spk.weight":
                w[-1] = v[-1]  # 'unknown speaker' row
            own[k] = w
            print(f"init {k}: {hit}/{len(names_new)} rows copied by name", flush=True)
        elif k == "emb.weight" and own[k].shape[0] > v.shape[0] and own[k].shape[1:] == v.shape[1:]:
            w = own[k].clone(); w[: v.shape[0]] = v  # vocabulary grew (sound-event tokens appended): keep old rows
            w[v.shape[0]:] = v[1].clone() + 0.02 * torch.randn_like(w[v.shape[0]:])  # new rows start near <sp>
            own[k] = w; copied += 1
            print(f"init emb.weight: {v.shape[0]} rows copied, {w.shape[0] - v.shape[0]} new event tokens", flush=True)
        elif own[k].shape == v.shape:
            own[k] = v; copied += 1
        else:
            skipped.append(k)
    fresh = [k for k in own if k not in src]
    model.load_state_dict(own)
    print(f"init from {path} (step {ck.get('step', '?')}): {copied} tensors copied, skipped {skipped}, "
          f"{len(fresh)} new tensors left at their init", flush=True)
    return ck.get("step", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--max_frames", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--oversample", type=str, default="", help="json {corpus: weight}")
    ap.add_argument("--init", type=str, default="", help="checkpoint to initialise weights from")
    ap.add_argument("--cons", type=float, default=0.0, help="weight of stage-2 control-consistency loss")
    ap.add_argument("--cons_every", type=int, default=1)
    ap.add_argument("--enc_mixer", default="attn", choices=["attn", "mamba3", "hybrid"],
                    help="encoder token mixer; hybrid = attention + zero-init bidirectional Mamba-3 branch")
    ap.add_argument("--dec_mixer", default="attn", choices=["attn", "mamba3", "hybrid"])
    ap.add_argument("--ssm_state", type=int, default=32, help="Mamba-3 state size N")
    ap.add_argument("--d_model", type=int, default=192, help="block width (param-matched mixer comparisons)")
    ap.add_argument("--ffn", type=int, default=384, help="conv FFN width")
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--init_continue", action="store_true",
                    help="fine-tune: continue the init checkpoint's step count (LR / bin-loss schedules, TB x-axis)")
    ap.add_argument("--rewarm", type=int, default=1000,
                    help="with --init_continue: linear LR re-warm over this many steps (new params have no Adam state)")
    ap.add_argument("--ckpt_every", type=int, default=1000, help="overwrite last.pt this often (crash recovery)")
    ap.add_argument("--keep_every", type=int, default=0, help="keep a numbered step<N>.pt this often (0 = 4x eval_every)")
    ap.add_argument("--harness_every", type=int, default=20000, help="steerability harness interval (0 = off)")
    ap.add_argument("--harness_voices", type=int, default=2)
    ap.add_argument("--harness_texts", type=int, default=4)
    ap.add_argument("--spk_enc", action="store_true",
                    help="condition the voice on a reference clip's speaker vector (steer.xvec) instead of the id table")
    ap.add_argument("--reward", type=float, default=0.0, help="weight of the reward fine-tuning loss (steer.rewards); 0 = off")
    ap.add_argument("--reward_every", type=int, default=2, help="reward step every N steps (each is ~2-3x a normal step)")
    ap.add_argument("--reward_k", type=int, default=4, help="items per reward step")
    ap.add_argument("--reward_w", type=str, default='{"control":1.0,"emotion":0.5,"ctc":0.2,"utmos":0.1,"spk":0.5}',
                    help="json weights of the individual rewards")
    ap.add_argument("--compile", action="store_true", help="torch.compile the Mamba-3 mixers (step is launch-bound)")
    ap.add_argument("--pair_w", type=float, default=0.0,
                    help="weight of the paired-intervention delta loss (data.pair_delta_loss); 0 = off")
    ap.add_argument("--pair_k", type=int, default=8, help="pairs per step for the delta loss")
    ap.add_argument("--pcgrad", action="store_true",
                    help="reward steps: drop the part of the reward gradient that conflicts with the supervised one")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="share of train items re-rendered in a sampled acoustic environment (steer.acoustic); > 0 also "
                         "adds the sound-event tokens and the environment conditioning to the model")
    ap.add_argument("--noise_lib", type=str, default="", help="noise library jsonl (default data/meta/noise_lib.jsonl)")
    ap.add_argument("--noise_mode", default="all", choices=["all", "nv"],
                    help="nv: only in-voice nonverbal splices ([laugh] [cough] ... in the speaker's own voice); scenes, "
                         "rooms, phone and environmental events are rendered by mixing at inference instead (v2: the L1 "
                         "decoder cannot learn stochastic textures - AST found 0/24 events and 0/13 scenes)")
    # control-space ablations (NAACL claim C3); defaults reproduce the full system
    ap.add_argument("--ctrl_norm", default="spk", choices=["spk", "global"],
                    help="abs_z ablation: global = controls z-scored corpus-wide instead of vs each speaker's neutral speech")
    ap.add_argument("--no_mask", action="store_true",
                    help="no_mask ablation: availability mask not an input (all-ones); unspecified dims are z = 0")
    ap.add_argument("--ctrl_dims", default="all", choices=["all", "prosody", "none"],
                    help="prosody: only rate/loud/f0_med/f0_iqr/f0_end/pause (prosody_only); none: no control vector "
                         "(tag_only)")
    ap.add_argument("--no_level_norm", action="store_true",
                    help="no_level_norm ablation: spk_enc runs keep each speaker's raw mel level (no LEVEL_DB shift)")
    ap.add_argument("--all_symbols", action="store_true",
                    help="text embedding over the full symbol table incl. inline event tokens (META v2 has NonverbalTTS "
                         "items with event tokens; without --noise the table stops at N_BASE and those ids overflow it)")
    ap.add_argument("--seed", type=int, default=None,
                    help="extra seeds for repeat runs: seeds torch / numpy and the batch sampler (default: unseeded init, "
                         "sampler seed 0, as before)")
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"  # cpu: smoke tests only
    out = RUNS / args.run
    (out / "samples").mkdir(parents=True, exist_ok=True)
    df, stats = load_meta()
    tr, va = split(df)
    gstats = None
    if args.ctrl_norm == "global":
        from .data import global_control_stats
        gstats = global_control_stats(df, stats)  # also needed at synthesis (steer.synth)
        json.dump(gstats, open(out / "abs_stats.json", "w"), indent=1)
        print(f"abs_z: corpus-wide control stats {gstats}", flush=True)
    dkw = dict(gstats=gstats, **({"level_norm": False} if args.no_level_norm else {}))
    json.dump(dict(vars(args), n_train=len(tr), n_val=len(va)), open(out / "args.json", "w"), indent=1)
    json.dump(stats, open(out / "control_stats.json", "w"))  # speaker/tag order of this run's checkpoints

    weights = None
    if args.oversample:
        ow = json.loads(args.oversample)
        # EARS freeform (spontaneous) clips share corpus "ears"; key "earsff" weights them separately
        key = tr.corpus.where(~tr.id.str.contains("_freeform_"), "earsff")
        weights = key.map(lambda c: ow.get(c, ow.get("ears", 1.0) if c == "earsff" else 1.0)).to_numpy()
    xvec = None
    if args.spk_enc:
        from .xvec import load_all, DIM
        xvec = load_all()
        missing = (~tr.id.isin(xvec)).sum()
        print(f"speaker vectors: {len(xvec)} loaded, {missing} train utts without one (dropped)", flush=True)
        tr, va = tr[tr.id.isin(xvec)].reset_index(drop=True), va[va.id.isin(xvec)].reset_index(drop=True)
        save_spk_refs(tr, xvec, out)
    aug, env_dim = None, 0
    if args.noise > 0:
        from .acoustic import Augmenter, ENV_DIM
        kw = dict(p_bg=0.0, p_room=0.0, p_phone=0.0, p_event=0.0, p_nv=1.0) if args.noise_mode == "nv" else {}
        aug = Augmenter(*([args.noise_lib] if args.noise_lib else []), p=args.noise, **kw)
        env_dim = ENV_DIM
        print(f"noise augmentation p={args.noise}: scenes {sorted(aug.bg)}, events {sorted(aug.ev)}, "
              f"nonverbal speakers {len(aug.nv)}, rirs { {k: len(v) for k, v in aug.rir.items()} }", flush=True)
    tr_ds = TTSData(tr, stats, train=True, xvec=xvec, aug=aug, env_dim=env_dim, **dkw)
    va_ds = TTSData(va, stats, train=False, xvec=xvec, env_dim=env_dim, **dkw)
    tr_dl = DataLoader(tr_ds, batch_sampler=FrameBucketSampler(tr.n_frames, args.max_frames, weights=weights,
                                                                       seed=args.seed or 0),
                       collate_fn=collate, num_workers=args.workers, pin_memory=True, persistent_workers=True,
                       prefetch_factor=4)
    va_dl = DataLoader(va_ds, batch_sampler=FrameBucketSampler(va.n_frames, args.max_frames, shuffle=False),
                       collate_fn=collate, num_workers=4)
    pair_it = None
    if args.pair_w > 0:
        pairs = pair_pool(tr)
        # full conditioning (no control / tag dropout) and no augmentation: the delta is between the two real renders
        pair_ds = TTSData(tr, stats, train=True, p_dim_drop=0.0, p_all_drop=0.0, p_tag_drop=0.0, xvec=xvec, env_dim=env_dim,
                          **dkw)
        # EARS style pairs (~3% of the pool, the only same-sentence pitch / loudness / whisper / rate contrasts)
        # are drawn ~30% of the time
        is_ears = (tr.corpus.to_numpy()[[a for a, _ in pairs]] == "ears")
        pw = np.where(is_ears, 0.3 / max(is_ears.mean(), 1e-6), 0.7 / max(1 - is_ears.mean(), 1e-6))
        pair_it = iter(DataLoader(pair_ds, batch_sampler=PairSampler(pairs, args.pair_k, weights=pw), collate_fn=collate,
                                  num_workers=3, pin_memory=True, prefetch_factor=4, persistent_workers=True))
        print(f"paired-intervention pairs: {len(pairs)} ({tr.loc[[p[0] for p in pairs], 'corpus'].value_counts().to_dict()})",
              flush=True)

    model = SteerTTS(len(stats["speakers"]), len(stats["tags"]), d=args.d_model, ffn=args.ffn, heads=args.heads,
                     enc_mixer=args.enc_mixer,
                     dec_mixer=args.dec_mixer, ssm_state=args.ssm_state, spk_enc_dim=DIM if args.spk_enc else 0,
                     n_symbols=len(SYMBOLS) if (env_dim or args.all_symbols) else None, n_env=env_dim,
                     ctrl_dims={"all": None, "prosody": PROSODY_DIMS, "none": []}[args.ctrl_dims], no_mask=args.no_mask,
                     ctrl_norm=args.ctrl_norm, level_norm=not args.no_level_norm).to(dev)
    if args.compile:
        from .model import BiMamba3
        BiMamba3.enable_compile()
    n_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("aligner"))
    n_align = sum(p.numel() for p in model.aligner.parameters())
    print(f"params (inference path) {n_params / 1e6:.2f}M | aligner {n_align / 1e6:.2f}M"
          f" | train {len(tr)} val {len(va)}", flush=True)
    # fused: one kernel for the whole update instead of ~10 per parameter tensor
    # newly added modules (environment projection) learn at 10x: they start from zero late in the LR schedule
    fresh = [p for n, p in model.named_parameters() if n.startswith("env_proj")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("env_proj")]
    groups = [dict(params=rest, lr_mult=1.0)] + ([dict(params=fresh, lr_mult=10.0)] if fresh else [])
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-6, fused=True)
    step = 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        for g in opt.param_groups:  # checkpoints from before fused=True carry fused=None/foreach in their groups
            g["fused"], g["foreach"] = True, False
        for st in opt.state.values():
            st["step"] = st["step"].to(dev, torch.float32)
        print("resumed at", step, flush=True)
    start_step = 0
    if args.init and not (args.resume and step):
        init_step = init_from(model, args.init, stats, dev)
        # least-squares spk_proj only when the init has none (from a table-only model); an spk_enc init keeps its
        # trained projection (copied by init_from)
        if args.spk_enc and "spk_proj.weight" not in torch.load(args.init, map_location="cpu", weights_only=False)["model"]:
            fit_spk_proj(model, args.init, out)
        if args.init_continue:
            step = start_step = init_step
    elif args.init and args.init_continue:  # resumed: recover where the fine-tune started, for the re-warm
        start_step = torch.load(args.init, map_location="cpu", weights_only=False).get("step", 0)
    rew = None
    if args.reward > 0:
        from .rewards import RewardLoss
        rew = RewardLoss(stats, dev, json.loads(args.reward_w), level_norm=args.spk_enc and not args.no_level_norm,
                         k=args.reward_k)
        if json.loads(args.reward_w).get("anchor"):
            # trust-region reference: a frozen copy of the STARTING checkpoint (--init), also after a resume
            import copy
            rew.ref = copy.deepcopy(model)
            if args.init:
                init_from(rew.ref, args.init, stats, dev)
            rew.ref.eval().requires_grad_(False)
    cons = None
    if args.cons > 0:
        from .consistency import Consistency
        cons = Consistency(stats, dev)

    # purge_step drops events logged after the checkpoint we resumed from, so curves don't double back
    tb = SummaryWriter(out / "tb", purge_step=step or None)
    tb.add_text("config/args", "```\n" + json.dumps(vars(args), indent=1) + "\n```", step)
    tb.add_text("config/model", f"inference-path params {n_params / 1e6:.2f}M, aligner {n_align / 1e6:.2f}M, "
                f"{len(stats['speakers'])} speakers, {len(stats['tags'])} tags", step)
    comp = tr.groupby("corpus").agg(utts=("id", "size"), hours=("n_frames", lambda f: f.sum() / 93.75 / 3600),
                                    speakers=("speaker", "nunique")).round(2)
    tb.add_text("config/data", f"train {len(tr)} / val {len(va)} utterances\n\n```\n" + comp.to_string() + "\n```", step)

    vocos = None
    t0, agg, cnt = time.time(), {}, {}
    perf = dict(t=time.time(), frames=0, utts=0, skipped=0, gn=0.0)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Nothing in the step reads a GPU value back (no float()/isfinite on the host), so the CPU can queue the
    # next step while the GPU runs this one: losses / grad norm / skips accumulate on device and sync once
    # per --log_every. A non-finite grad norm skips the update on device via the fused optimizer's found_inf.
    while step < args.steps:
        for batch in tr_dl:
            perf["frames"] += int(batch["mel_lens"].sum())  # CPU tensors, before the copy
            perf["utts"] += len(batch["mel_lens"])
            batch = to(batch, dev)
            for gp in opt.param_groups:
                gp["lr"] = gp.get("lr_mult", 1.0) * lr_at(step, args.lr, total=args.steps) * (
                    min(1.0, (step - start_step + 1) / args.rewarm) if args.init_continue else 1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses, _ = model(batch, step)
            loss = sum(W[k] * losses[k] for k in W) + bin_weight(step) * losses["bin"]
            if cons is not None and step % args.cons_every == 0:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cl = cons(model, batch)
                loss = loss + args.cons * (cl["cons_ac"] + cl["cons_t"])
                losses.update(cl)
            if pair_it is not None:
                pb = to(next(pair_it), dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pl, po = model(pb, step)
                    pd = pair_delta_loss(po, pb)
                loss = loss + 0.5 * (sum(W[k] * pl[k] for k in W) + bin_weight(step) * pl["bin"])  # pairs are data too
                if pd is not None:
                    loss = loss + args.pair_w * pd["pair_total"]
                    losses.update(pd)
            rl = None
            if rew is not None and step % args.reward_every == 0:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    rl = rew(model, batch)
                if rl is not None:  # None: no short-enough item in this batch (rewards.__call__)
                    losses.update(rl)
            opt.zero_grad(set_to_none=True)
            if rl is not None and args.pcgrad:
                # gradient surgery (PCGrad, Yu et al. 2020): the reward gradient keeps only its component that does
                # not oppose the supervised gradient, so to first order a reward step cannot increase the data loss
                # (the mel / duration / pitch drift of reward attempts 1-2). Computed on device, no host sync.
                params = [p for p in model.parameters() if p.requires_grad]
                loss.backward()
                gs = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                opt.zero_grad(set_to_none=True)
                (args.reward * rl["reward_total"]).backward()
                gr = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                dot = sum((a * b).sum() for a, b in zip(gs, gr))
                nn2 = sum((a * a).sum() for a in gs)
                coef = torch.clamp(dot, max=0.0) / (nn2 + 1e-12)
                losses["pc_conflict"] = (dot < 0).float()
                for p, a, b in zip(params, gs, gr):
                    p.grad = a + b - coef * a
                loss = loss + args.reward * rl["reward_total"].detach()
            else:
                if rl is not None:
                    loss = loss + args.reward * rl["reward_total"]
                loss.backward()
            losses["total"] = loss.detach()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ok = torch.isfinite(gn)
            opt.found_inf = (~ok).float()
            opt.step()
            perf["gn"] += torch.where(ok, gn, 0.0)
            perf["skipped"] += (~ok).int()
            step += 1
            for k, v in losses.items():  # per-key counts: reward/consistency terms only occur every Nth step
                agg[k] = agg.get(k, 0.0) + v.detach().float()
                cnt[k] = cnt.get(k, 0) + 1
            agg["n"] = agg.get("n", 0) + 1
            if step % args.log_every == 0:
                n = agg.pop("n")
                agg = {k: float(v) / cnt[k] for k, v in agg.items()}
                perf["gn"], perf["skipped"] = float(perf["gn"]), int(perf["skipped"])
                msg = " ".join(f"{k} {v:.4f}" for k, v in agg.items())
                for k, v in agg.items():
                    tb.add_scalar(f"train/{k}", v, step)
                tb.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                tb.add_scalar("train/bin_weight", bin_weight(step), step)
                n_ok = max(n - perf["skipped"], 1)
                tb.add_scalar("train/grad_norm", perf["gn"] / n_ok, step)
                dt = time.time() - perf["t"]
                tb.add_scalar("perf/steps_per_s", n / dt, step)
                tb.add_scalar("perf/mel_frames_per_s", perf["frames"] / dt, step)
                tb.add_scalar("perf/utts_per_batch", perf["utts"] / n, step)
                tb.add_scalar("perf/skipped_steps", perf["skipped"], step)
                tb.add_scalar("perf/peak_gpu_mem_gb", (torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0), step)
                tb.add_scalar("perf/host_rss_gb", host_rss_gb(), step)
                print(f"step {step} | {msg} | gn {perf['gn'] / n_ok:.2f} | {n / dt:.1f} it/s | {time.time() - t0:.0f}s",
                      flush=True)
                agg, cnt = {}, {}
                perf = dict(t=time.time(), frames=0, utts=0, skipped=0, gn=0.0)
                torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
            if step % args.ckpt_every == 0 and step % args.eval_every != 0:
                save_last(model, opt, step, stats, out)
            if step % args.eval_every == 0 or step == args.steps:
                vocos = evaluate(model, va_dl, va_ds, stats, step, out, tb, dev, vocos)
                save_last(model, opt, step, stats, out)
                if step % (args.keep_every or args.eval_every * 4) == 0:
                    torch.save(dict(model=model.state_dict(), step=step, arch=model.arch, n_speakers=len(stats["speakers"]),
                                    n_tags=len(stats["tags"])), out / f"step{step}.pt")
                if args.harness_every and (step % args.harness_every == 0 or step == args.steps):
                    steer_eval(out / "last.pt", stats, step, out, tb, args)
                    tb.flush()  # results tables (TEXT tab) for this run + the cross-run leaderboard, off the training process
                    subprocess.Popen([sys.executable, "-m", "steer.tb_report", args.run], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                tb.flush()
                perf["t"] = time.time()  # don't count evaluation time against throughput
            if step >= args.steps:
                break
    tb.close()


def save_spk_refs(tr, xvec, out):
    """Per-speaker reference vector = mean of its neutral-utterance vectors (renormalised): the default voice
    prompt for known speakers at synthesis time (runs/<run>/spk_ref.npz; Synth loads it)."""
    neu = tr[(tr.tag == "neutral") | (tr.corpus == "libritts")]
    if "speed" in tr:  # cartesia: default-setting renders only
        neu = neu[neu.speed.fillna(1.0).eq(1.0) & neu.volume.fillna(1.0).eq(1.0)]
    spks, vecs = [], []
    for spk, g in tr.groupby("speaker"):
        ids = neu[neu.speaker == spk].id.tolist() or g.id.tolist()
        v = np.mean([xvec[i] for i in ids], 0)
        spks.append(spk); vecs.append(v / (np.linalg.norm(v) + 1e-8))
    np.savez(out / "spk_ref.npz", speakers=np.array(spks), vec=np.stack(vecs).astype(np.float32))
    print(f"spk_ref.npz: {len(spks)} speakers", flush=True)


def fit_spk_proj(model, init_path, out, ridge=0.1):
    """Least-squares init of spk_proj so that proj(mean neutral vector of s) ~= the init checkpoint's table row for
    every speaker the init knew: known voices start where they were, new ones start at a sensible point in the same
    space instead of from a random projection."""
    from pathlib import Path
    old = json.load(open(Path(init_path).parent / "control_stats.json"))["speakers"]
    z = np.load(out / "spk_ref.npz")
    ref = dict(zip(z["speakers"].tolist(), z["vec"]))
    tab = model.spk.weight.detach().float().cpu().numpy()  # already copied by name from the init in init_from
    own = {n: i for i, n in enumerate(json.load(open(out / "control_stats.json"))["speakers"])}
    names = [n for n in old if n in ref and n in own]
    X = np.stack([ref[n] for n in names]); X1 = np.hstack([X, np.ones((len(X), 1))])
    Y = np.stack([tab[own[n]] for n in names])
    W = np.linalg.solve(X1.T @ X1 + ridge * np.eye(X1.shape[1]), X1.T @ Y)
    with torch.no_grad():
        model.spk_proj.weight.copy_(torch.from_numpy(W[:-1].T)); model.spk_proj.bias.copy_(torch.from_numpy(W[-1]))
    r2 = 1 - ((X1 @ W - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    print(f"spk_proj least-squares init on {len(names)} known speakers: R^2 {r2:.3f}", flush=True)


def save_last(model, opt, step, stats, out):
    tmp = out / "last.pt.tmp"  # write-then-rename so a crash mid-save never corrupts last.pt
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, arch=model.arch,
                    n_speakers=len(stats["speakers"]), n_tags=len(stats["tags"])), tmp)
    os.replace(tmp, out / "last.pt")


def host_rss_gb():
    """Resident memory of this process plus its DataLoader workers (children), from /proc."""
    me = os.getpid()
    pids = [me] + [int(p) for p in open(f"/proc/{me}/task/{me}/children").read().split()]
    kb = 0
    for p in pids:
        try:
            kb += int(next(l for l in open(f"/proc/{p}/status") if l.startswith("VmRSS")).split()[1])
        except (OSError, StopIteration):
            pass
    return kb / 2**20


@torch.no_grad()
def evaluate(model, va_dl, va_ds, stats, step, out, tb, dev, vocos):
    model.eval()
    agg, n = {}, 0
    for i, batch in enumerate(va_dl):
        if i >= 20:
            break
        batch = to(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses, _ = model(batch, step)
        for k, v in losses.items():
            agg[k] = agg.get(k, 0.0) + float(v.detach())
        n += 1
    for k, v in agg.items():
        tb.add_scalar(f"val/{k}", v / n, step)
    print(f"[val {step}] " + " ".join(f"{k} {v / n:.4f}" for k, v in agg.items()), flush=True)
    if vocos is None:
        from vocos import Vocos
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev).eval()
        first = True
    else:
        first = False
    # fixed validation utterances: free-running synthesis (full control vector given) + teacher-forced view
    for j in range(min(N_VAL_SAMPLES, len(va_ds))):
        x = va_ds[j * max(1, len(va_ds) // N_VAL_SAMPLES)]
        ids = torch.from_numpy(x["ids"])[None].to(dev)
        mel, _ = model.infer(ids, torch.tensor([x["spk"]], device=dev), torch.tensor([x["tag"]], device=dev),
                             torch.from_numpy(x["ctrl"])[None].to(dev), torch.from_numpy(x["cmask"])[None].to(dev),
                             spk_vec=torch.from_numpy(x["spk_vec"])[None].to(dev) if "spk_vec" in x else None)
        wav = vocos.decode(mel.float()).squeeze().cpu().numpy()
        sf.write(out / "samples" / f"{step:07d}_{x['id']}.wav", wav, SR)
        tag = f"{j}_{x['id']}"
        tb.add_audio(f"val_audio/{tag}/synth", wav[None], step, sample_rate=SR)
        b = to(collate([x]), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, o = model(b, step)
        gt = b["mel"][0].float()
        if first:
            gt_wav = vocos.decode(gt[None]).squeeze().cpu().numpy()
            tb.add_audio(f"val_audio/{tag}/vocoded_ground_truth", gt_wav[None], step, sample_rate=SR)
        tb.add_image(f"val_mel/{tag}/ground_truth", as_image(gt.cpu().numpy()), step)
        tb.add_image(f"val_mel/{tag}/teacher_forced", as_image(o["mel_p"][0].float().cpu().numpy()), step)
        tb.add_image(f"val_mel/{tag}/free_running", as_image(mel[0].float().cpu().numpy()), step)
        tb.add_image(f"val_align/{tag}", as_image(o["soft"][0].float().T.cpu().numpy()), step)  # tokens x frames
    model.train()
    return vocos


def steer_eval(ckpt, stats, step, out, tb, args):
    """Paired steerability harness on a small voice/text subset; never kills training if it fails."""
    from .harness import run, TEXTS
    try:
        t = time.time()
        voices = [s for s in stats["speakers"] if s.startswith("ears")][:args.harness_voices]
        R, audio = run(str(ckpt), voices, out=out / f"harness_{step}.json", texts=TEXTS[:args.harness_texts])
        for r in R.itertuples():
            if not np.isnan(r.steer):
                tb.add_scalar(f"steer_ratio/{r.direction}", r.steer, step)
            tb.add_scalar(f"steer_hit/{r.direction}", r.hit, step)
            tb.add_scalar(f"steer_leak/{r.direction}", r.leak, step)
        cont, tag = R[~R.direction.str.startswith("tag_")], R[R.direction.str.startswith("tag_")]
        tb.add_scalar("steer_summary/control_ratio_median", cont.steer.median(), step)
        tb.add_scalar("steer_summary/control_hit_mean", cont.hit.mean(), step)
        tb.add_scalar("steer_summary/control_leak_mean", cont.leak.mean(), step)
        tb.add_scalar("steer_summary/control_significant_share", (cont.p_holm < 0.05).mean(), step)
        if len(tag):
            tb.add_scalar("steer_summary/tag_hit_mean", tag.hit.mean(), step)
        for name, wav in audio.items():
            tb.add_audio(f"steer_audio/{name}", np.asarray(wav, dtype=np.float32)[None], step, sample_rate=SR)
        tb.add_text("steer_table", "```\n" + R.round(3).to_string(index=False) + "\n```", step)
        print(f"[harness {step}] {time.time() - t:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[harness {step}] failed: {type(e).__name__}: {e}", flush=True)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
