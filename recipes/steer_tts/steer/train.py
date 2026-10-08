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
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import RUNS, SR
from .data import TTSData, FrameBucketSampler, collate, load_meta, split
from .model import SteerTTS

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
    ap.add_argument("--init_continue", action="store_true",
                    help="fine-tune: continue the init checkpoint's step count (LR / bin-loss schedules, TB x-axis)")
    ap.add_argument("--rewarm", type=int, default=1000,
                    help="with --init_continue: linear LR re-warm over this many steps (new params have no Adam state)")
    ap.add_argument("--ckpt_every", type=int, default=1000, help="overwrite last.pt this often (crash recovery)")
    ap.add_argument("--harness_every", type=int, default=20000, help="steerability harness interval (0 = off)")
    ap.add_argument("--harness_voices", type=int, default=2)
    ap.add_argument("--harness_texts", type=int, default=4)
    args = ap.parse_args()

    dev = "cuda"
    out = RUNS / args.run
    (out / "samples").mkdir(parents=True, exist_ok=True)
    df, stats = load_meta()
    tr, va = split(df)
    json.dump(dict(vars(args), n_train=len(tr), n_val=len(va)), open(out / "args.json", "w"), indent=1)
    json.dump(stats, open(out / "control_stats.json", "w"))  # speaker/tag order of this run's checkpoints

    weights = None
    if args.oversample:
        ow = json.loads(args.oversample)
        weights = tr.corpus.map(lambda c: ow.get(c, 1.0)).to_numpy()
    tr_ds, va_ds = TTSData(tr, stats, train=True), TTSData(va, stats, train=False)
    tr_dl = DataLoader(tr_ds, batch_sampler=FrameBucketSampler(tr.n_frames, args.max_frames, weights=weights),
                       collate_fn=collate, num_workers=args.workers, pin_memory=True, persistent_workers=True,
                       prefetch_factor=4)
    va_dl = DataLoader(va_ds, batch_sampler=FrameBucketSampler(va.n_frames, args.max_frames, shuffle=False),
                       collate_fn=collate, num_workers=4)

    model = SteerTTS(len(stats["speakers"]), len(stats["tags"]), enc_mixer=args.enc_mixer,
                     dec_mixer=args.dec_mixer, ssm_state=args.ssm_state).to(dev)
    n_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("aligner"))
    n_align = sum(p.numel() for p in model.aligner.parameters())
    print(f"params (inference path) {n_params / 1e6:.2f}M | aligner {n_align / 1e6:.2f}M"
          f" | train {len(tr)} val {len(va)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-6)
    step = 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print("resumed at", step, flush=True)
    start_step = 0
    if args.init and not (args.resume and step):
        init_step = init_from(model, args.init, stats, dev)
        if args.init_continue:
            step = start_step = init_step
    elif args.init and args.init_continue:  # resumed: recover where the fine-tune started, for the re-warm
        start_step = torch.load(args.init, map_location="cpu", weights_only=False).get("step", 0)
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
    t0, agg = time.time(), {}
    perf = dict(t=time.time(), frames=0, utts=0, skipped=0, gn=0.0)
    torch.cuda.reset_peak_memory_stats()
    while step < args.steps:
        for batch in tr_dl:
            batch = to(batch, dev)
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step, args.lr, total=args.steps) * (
                    min(1.0, (step - start_step + 1) / args.rewarm) if args.init_continue else 1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses, _ = model(batch, step)
            loss = sum(W[k] * losses[k] for k in W) + bin_weight(step) * losses["bin"]
            if cons is not None and step % args.cons_every == 0:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cl = cons(model, batch)
                loss = loss + args.cons * (cl["cons_ac"] + cl["cons_t"])
                losses.update(cl)
            losses["total"] = loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gn):
                opt.step()
                perf["gn"] += float(gn)
            else:
                perf["skipped"] += 1
            step += 1
            perf["frames"] += int(batch["mel_lens"].sum())
            perf["utts"] += len(batch["mel_lens"])
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            agg["n"] = agg.get("n", 0) + 1
            if step % args.log_every == 0:
                n = agg.pop("n")
                msg = " ".join(f"{k} {v / n:.4f}" for k, v in agg.items())
                for k, v in agg.items():
                    tb.add_scalar(f"train/{k}", v / n, step)
                tb.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                tb.add_scalar("train/bin_weight", bin_weight(step), step)
                n_ok = max(n - perf["skipped"], 1)
                tb.add_scalar("train/grad_norm", perf["gn"] / n_ok, step)
                dt = time.time() - perf["t"]
                tb.add_scalar("perf/steps_per_s", n / dt, step)
                tb.add_scalar("perf/mel_frames_per_s", perf["frames"] / dt, step)
                tb.add_scalar("perf/utts_per_batch", perf["utts"] / n, step)
                tb.add_scalar("perf/skipped_steps", perf["skipped"], step)
                tb.add_scalar("perf/peak_gpu_mem_gb", torch.cuda.max_memory_allocated() / 2**30, step)
                tb.add_scalar("perf/host_rss_gb", host_rss_gb(), step)
                print(f"step {step} | {msg} | gn {perf['gn'] / n_ok:.2f} | {n / dt:.1f} it/s | {time.time() - t0:.0f}s",
                      flush=True)
                agg = {}
                perf = dict(t=time.time(), frames=0, utts=0, skipped=0, gn=0.0)
                torch.cuda.reset_peak_memory_stats()
            if step % args.ckpt_every == 0 and step % args.eval_every != 0:
                save_last(model, opt, step, stats, out)
            if step % args.eval_every == 0 or step == args.steps:
                vocos = evaluate(model, va_dl, va_ds, stats, step, out, tb, dev, vocos)
                save_last(model, opt, step, stats, out)
                if step % (args.eval_every * 4) == 0:
                    torch.save(dict(model=model.state_dict(), step=step, arch=model.arch, n_speakers=len(stats["speakers"]),
                                    n_tags=len(stats["tags"])), out / f"step{step}.pt")
                if args.harness_every and (step % args.harness_every == 0 or step == args.steps):
                    steer_eval(out / "last.pt", stats, step, out, tb, args)
                tb.flush()
                perf["t"] = time.time()  # don't count evaluation time against throughput
            if step >= args.steps:
                break
    tb.close()


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
                             torch.from_numpy(x["ctrl"])[None].to(dev), torch.from_numpy(x["cmask"])[None].to(dev))
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
