"""Training-compute accounting for the paper: lineage of the final checkpoint + GPU-hours, it/s, VRAM per stage.

CPU only (reads TensorBoard event files, args.json, checkpoint `step` fields, logs/watchdog.log). Re-run with:
  CUDA_VISIBLE_DEVICES="" STEER_TTS_ROOT=<workspace> PYTHONPATH=<recipes/steer_tts> python -m steer.paper.train_compute

Method
- Lineage: start at runs/spk_enc_v3/step312500.pt and follow `init` in each run's args.json (base runs: absolute
  path; flow runs: relative to STEER_TTS_ROOT). The step a stage contributes ends at the `step` stored in the
  checkpoint its child was initialised from; base runs continue the step counter (init_continue), flow runs restart at 0.
- Per run, every TB events file in runs/<run>/tb (not tb/results) is read; the main scalar (train/mel or train/fm,
  logged every 100 steps) gives (step, wall_time). Files are ordered by start; when a later file resumes at step r,
  events of earlier files with step > r are "wasted" (redone after a crash/restart).
- Hours of a stage = sum of wall-time deltas between consecutive surviving events in (start, end] (the first event of
  each file is measured from the file's creation event, so start-up / torch.compile time is included).
  gap-excluded: deltas > 10 min are dropped (downtime, freezes); raw: elapsed wall clock from stage start to end
  (includes downtime between restarts).
- it/s: perf/steps_per_s (base runs; train.py: steps / wall seconds over each 100-step log window); flow runs have no
  perf scalars -> 100 / delta of consecutive train/fm events (deltas <= 10 min). ci = p10 / p90.
- Peak VRAM: perf/peak_gpu_mem_gb = torch.cuda.max_memory_allocated per log window (allocator view, not reserved).
  watchdog.log (nvidia-smi memory.used every 20 s, ALL processes incl. desktop + concurrent jobs) is reported as a
  cross-check / the only source for flow runs. Watchdog starts 2026-10-08 12:05.
- Shared GPU: active intervals (event deltas <= 10 min) of every other TB run are intersected with the stage's.
  Non-TB GPU jobs (benchmarks, leaderboard scoring, GTA dumps, demos) are not detected.
"""
import bisect
import datetime as dt
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from tensorboard.backend.event_processing import event_accumulator as ea  # noqa: E402

ROOT = Path(os.environ.get("STEER_TTS_ROOT", "/home/karan/Desktop/whissle/workspace/steer_tts"))
RUNS = ROOT / "runs"
OUT = Path("/home/karan/Desktop/whissle/marketing_post_agent/paper/steertts/results/speed_training.json")
FINAL = RUNS / "spk_enc_v3/step312500.pt"
PFLOW = RUNS / "flow_v4pa/step40000.pt"        # prompted flow named in the paper task
PFLOW_V2 = RUNS / "flow_v3pc/last.pt"          # prompted flow v2 used for the leaderboard cloning numbers
GAP = 600.0
DEVICE = "RTX 5070"
TZ = dt.timezone(dt.timedelta(hours=-7))       # machine local time (PDT), matches log timestamps
SPEEDUP_DAY = "2026-10-08"


def W(p):
    p = str(Path(p).resolve()) if Path(p).exists() else str(p)
    return p.replace(str(ROOT.resolve()), "W").replace(str(ROOT), "W")


def ts(w):
    return dt.datetime.fromtimestamp(w, TZ).strftime("%Y-%m-%d %H:%M")


def ckpt_step(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return int(ck["step"])


def resolve_init(init):
    if not init:
        return None
    p = Path(init)
    return p if p.is_absolute() else ROOT / p


def lineage(ckpt):
    """[(run, start_step, end_step, init_path, args)] from root ancestor to ckpt."""
    chain, ck, end = [], Path(ckpt), None
    while ck is not None:
        run = ck.parent
        args = json.load(open(run / "args.json"))
        end = ckpt_step(ck)
        init = resolve_init(args.get("init"))
        is_flow = args.get("cmd") == "train" and "gta" in args
        start = ckpt_step(init) if (init is not None and args.get("init_continue")) else 0
        chain.append(dict(run=run.name, start=start, end=end, ckpt=str(ck), init=str(init) if init else None,
                          args=args, flow=is_flow))
        ck = init
    return chain[::-1]


# ----------------------------------------------------------------------------------------------- TB reading
_cache = {}


def read_file(f):
    if f in _cache:
        return _cache[f]
    a = ea.EventAccumulator(f, size_guidance={ea.SCALARS: 0})
    a.Reload()
    tags = a.Tags()["scalars"]
    main = next((t for t in ("train/mel", "train/fm") if t in tags), None)
    if main is None:
        main = next((t for t in tags if t.startswith("train/")), tags[0] if tags else None)
    d = dict(file=f, main=main, t0=a.FirstEventTimestamp() if tags else None, ev=[], perf={})
    if main:
        d["ev"] = [(e.step, e.wall_time) for e in a.Scalars(main)]
        for t in tags:
            if t.startswith("perf/"):
                d["perf"][t] = {e.step: e.value for e in a.Scalars(t)}
    _cache[f] = d
    return d


def run_files(run):
    fs = sorted(glob.glob(str(RUNS / run / "tb" / "events.out.tfevents.*")))
    ds = [read_file(f) for f in fs]
    return sorted([d for d in ds if d["ev"]], key=lambda d: d["t0"])


def surviving(files):
    """List of events (step, wall, delta, file, perf-dict) that survive restarts; plus wasted ones."""
    kept, wasted = [], []
    for d in files:
        steps = [s for s, _ in d["ev"]]
        li = int(np.median(np.diff(steps))) if len(steps) > 1 else 100
        r = d["ev"][0][0] - li  # resume point
        keep = [e for e in kept if e[0] <= r]
        wasted += [e for e in kept if e[0] > r]
        kept = keep
        prev = d["t0"]
        for s, w in d["ev"]:
            kept.append((s, w, w - prev, d["file"]))
            prev = w
    return kept, wasted


def intervals(files):
    """Active wall intervals of a run (consecutive event deltas <= GAP)."""
    iv = []
    for d in files:
        prev = d["t0"]
        for _, w in d["ev"]:
            if w - prev <= GAP:
                iv.append((prev, w))
            prev = w
    return merge(iv)


def merge(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def overlap(a, b):
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        tot += max(0.0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def all_runs():
    seen, out = set(), {}
    for p in sorted(RUNS.glob("*/tb")):
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        fs = run_files(p.parent.name)
        if fs:
            out[p.parent.name] = fs
    return out


# ----------------------------------------------------------------------------------------------- watchdog
def watchdog():
    f = ROOT / "logs/watchdog.log"
    if not f.exists():
        return np.array([]), np.array([]), np.array([])
    day = dt.date(2026, 10, 8)
    last = None
    T, M, U = [], [], []
    cur = None
    for line in open(f, errors="ignore"):
        line = line.strip()
        if re.fullmatch(r"\d\d:\d\d:\d\d", line):
            t = dt.time.fromisoformat(line)
            if last is not None and t < last:
                day += dt.timedelta(days=1)
            last = t
            cur = dt.datetime.combine(day, t, TZ).timestamp()
        elif cur is not None and "MiB" in line:
            p = [x.strip() for x in line.split(",")]
            try:
                T.append(cur); M.append(float(p[2].split()[0]) / 1024); U.append(float(p[3].split()[0]))
            except (ValueError, IndexError):
                pass
            cur = None
    return np.array(T), np.array(M), np.array(U)


def wd_stats(wd, iv):
    T, M, U = wd
    if len(T) == 0:
        return None
    idx = []
    for a, b in iv:
        lo, hi = bisect.bisect_left(T, a), bisect.bisect_right(T, b)
        idx.extend(range(lo, hi))
    if not idx:
        return None
    idx = np.array(idx)
    return dict(n=len(idx), vram_max=float(M[idx].max()), vram_p90=float(np.percentile(M[idx], 90)),
                vram_med=float(np.median(M[idx])), util_med=float(np.median(U[idx])))


# ----------------------------------------------------------------------------------------------- per stage
def q(x, p):
    return float(np.percentile(x, p)) if len(x) else None


def stage_stats(run, start, end, others, wd):
    files = run_files(run)
    kept, wasted = surviving(files)
    seg = [e for e in kept if start < e[0] <= end]
    if not seg:
        return None
    deltas = np.array([e[2] for e in seg])
    active = float(deltas[deltas <= GAP].sum())
    gaps = [(e[0], e[2]) for e in seg if e[2] > GAP]
    first_file = next(d for d in files if d["file"] == seg[0][3])
    t_start = seg[0][1] - seg[0][2]
    raw = seg[-1][1] - t_start
    seg_iv = merge([(e[1] - e[2], e[1]) for e in seg if e[2] <= GAP])
    # it/s
    perf = {}
    for d in files:
        for k, v in d["perf"].items():
            perf.setdefault(k, {}).setdefault(d["file"], v)
    def pv(tag):
        return np.array([perf[tag][e[3]][e[0]] for e in seg if tag in perf and e[3] in perf[tag] and e[0] in perf[tag][e[3]]])
    sps = pv("perf/steps_per_s")
    lis = np.diff([e[0] for e in seg])
    ev_rate = np.array([(s1[0] - s0[0]) / s1[2] for s0, s1 in zip(seg[:-1], seg[1:])
                        if s1[3] == s0[3] and 0 < s1[2] <= GAP])
    its_src = "perf/steps_per_s" if len(sps) else "TB step/wall deltas"
    its = sps if len(sps) else ev_rate
    mem = pv("perf/peak_gpu_mem_gb")
    upb = pv("perf/utts_per_batch")
    w_seg = [e for e in wasted if start < e[0] <= end + 10_000]
    w_d = np.array([e[2] for e in w_seg]) if w_seg else np.array([])
    shared = {}
    for r, fs in others.items():
        if r == run:
            continue
        o = overlap(seg_iv, intervals(fs))
        if o > 60:
            shared[r] = round(o / 3600, 2)
    restarts = sorted({e[3] for e in seg})
    return dict(run=run, start=start, end=end, steps=end - start, active_h=active / 3600, raw_h=raw / 3600,
                t_start=ts(t_start), t_end=ts(seg[-1][1]), n_events=len(seg),
                gaps=[(s, round(g / 60, 1)) for s, g in gaps],
                its=its, its_src=its_src, ev_rate=ev_rate, mem=mem, upb=upb,
                wasted_steps=len(w_seg) * (int(np.median(lis)) if len(lis) else 100),
                wasted_h=float(w_d[w_d <= GAP].sum()) / 3600 if len(w_d) else 0.0,
                n_files=len(restarts), files=restarts, shared=shared, wd=wd_stats(wd, seg_iv),
                source=W(first_file["file"]))


def log_its(name):
    f = ROOT / "logs" / f"{name}.log"
    if not f.exists():
        return None
    v = [float(m.group(1)) for m in re.finditer(r"\| ([\d.]+) it/s", open(f, errors="ignore").read())]
    return np.array(v) if v else None


# ----------------------------------------------------------------------------------------------- speedup check
def speedup(others):
    """mix_m3hyb perf/steps_per_s + peak mem per events file on 2026-10-08, with concurrency to mix_s1."""
    out = []
    s1 = intervals(others.get("mix_s1", []))
    for d in run_files("mix_m3hyb"):
        sps = np.array([d["perf"]["perf/steps_per_s"][s] for s, _ in d["ev"] if s in d["perf"].get("perf/steps_per_s", {})])
        mem = np.array([d["perf"]["perf/peak_gpu_mem_gb"][s] for s, _ in d["ev"] if s in d["perf"].get("perf/peak_gpu_mem_gb", {})])
        iv = merge([(a, b) for a, b in intervals([d])])
        alone, prev = [], d["t0"]
        for st, w in d["ev"]:
            if st in d["perf"].get("perf/steps_per_s", {}) and overlap([(prev, w)], s1) == 0:
                alone.append(d["perf"]["perf/steps_per_s"][st])
            prev = w
        alone = np.array(alone)
        shared = overlap(iv, s1) / max(1.0, sum(b - a for a, b in iv))
        out.append(dict(file=W(d["file"]), t0=ts(d["ev"][0][1]), t1=ts(d["ev"][-1][1]), steps=(d["ev"][0][0], d["ev"][-1][0]),
                        its_med=q(sps, 50), its_p10=q(sps, 10), its_p90=q(sps, 90), mem_med=q(mem, 50), mem_max=float(mem.max()) if len(mem) else None,
                        frac_shared_with_mix_s1=round(shared, 2), n=len(sps),
                        its_med_alone=q(alone, 50), n_alone=len(alone)))
    return out


# ----------------------------------------------------------------------------------------------- main
def row(metric, value, *, system, source_file, notes="", ci_lo=None, ci_hi=None, n=None):
    return dict(metric=metric, value=None if value is None else round(float(value), 4), ci_lo=None if ci_lo is None else round(float(ci_lo), 4),
                ci_hi=None if ci_hi is None else round(float(ci_hi), 4), n=n, device=DEVICE, threads=None, system=system,
                length_s=None, source_file=source_file, notes=notes)


def main():
    others = all_runs()
    wd = watchdog()
    chains = {"base": lineage(FINAL), "pflow_v4pa": lineage(PFLOW), "pflow_v2": lineage(PFLOW_V2)}
    print("LINEAGE")
    for k, c in chains.items():
        print(f"  {k}: " + " -> ".join(f"{s['run']}[{s['start']}->{s['end']}]" for s in c))
    # flow refiner = root of the pflow chain (flow_v3, no init); prompted = the rest
    flow_ref = [s for s in chains["pflow_v4pa"] if s["run"] == "flow_v3"]
    groups = {"total_base_lineage": chains["base"], "total_flow_refiner": flow_ref,
              "total_prompted_flow_v4pa40k": [s for s in chains["pflow_v4pa"] if s["run"] != "flow_v3"],
              "total_prompted_flow_v2": [s for s in chains["pflow_v2"] if s["run"] != "flow_v3"]}
    stages = {}
    for c in chains.values():
        for s in c:
            key = (s["run"], s["start"], s["end"])
            if key not in stages:
                st = stage_stats(s["run"], s["start"], s["end"], others, wd)
                st["args"] = s["args"]; st["init"] = s["init"]; st["ckpt"] = s["ckpt"]; st["flow"] = s["flow"]
                stages[key] = st
    # full flow_v4pa run (to 140k) for reference
    v4full = stage_stats("flow_v4pa", 0, 140000, others, wd)
    v4full["args"] = json.load(open(RUNS / "flow_v4pa/args.json")); v4full["flow"] = True; v4full["init"] = None; v4full["ckpt"] = None

    rows = []
    print("\nSTAGES")
    for key, st in list(stages.items()) + [(("flow_v4pa_full",), v4full)]:
        a = st["args"]
        sysname = st["run"] if key[0] != "flow_v4pa_full" else "flow_v4pa_full_140k"
        batch = f"batch {a.get('batch')}" if st["flow"] else f"max_frames {a.get('max_frames')} (args.json, last launch)"
        comp = bool(a.get("compile", False))
        its = st["its"]
        lg = log_its(st["run"]) if st["flow"] else None
        shared = ", ".join(f"{r} {h} h" for r, h in sorted(st["shared"].items(), key=lambda x: -x[1])) or "none (TB)"
        wdn = st["wd"]
        print(f"  {sysname:22s} {st['start']:>6}->{st['end']:<6} active {st['active_h']:.2f} h raw {st['raw_h']:.2f} h | "
              f"{st['t_start']} .. {st['t_end']} | it/s med {q(its, 50):.2f} [{q(its, 10):.2f},{q(its, 90):.2f}] ({st['its_src']}) | "
              f"mem max {st['mem'].max() if len(st['mem']) else float('nan'):.2f} | upb {q(st['upb'], 50) if len(st['upb']) else float('nan'):.1f} | "
              f"{batch} compile={comp} | files {st['n_files']} gaps {st['gaps']} | wasted {st['wasted_steps']} steps {st['wasted_h']:.2f} h | shared: {shared}"
              + (f" | wd max {wdn['vram_max']:.1f} GB util {wdn['util_med']:.0f}%" if wdn else " | wd n/a")
              + (f" | log it/s med {np.median(lg):.1f}" if lg is not None else ""))
        base_note = (f"steps {st['start']}->{st['end']} ({st['t_start']} .. {st['t_end']} PDT); {batch}; compile={comp}; "
                     f"{st['n_files']} TB file(s); init={W(st['init']) if st['init'] else 'scratch'}")
        rows.append(row("steps", st["steps"], system=sysname, source_file=st["source"], notes=base_note))
        rows.append(row("gpu_hours", st["active_h"], system=sysname, source_file=st["source"], n=st["n_events"],
                        notes=f"gap-excluded (event deltas > 10 min dropped; startup/compile incl.). gaps>10min: {st['gaps']}; "
                              f"redone-after-restart (not counted): {st['wasted_steps']} steps / {st['wasted_h']:.2f} h"))
        rows.append(row("wall_hours_raw", st["raw_h"], system=sysname, source_file=st["source"], n=st["n_events"],
                        notes="elapsed wall clock stage start -> end incl. downtime between restarts"))
        rows.append(row("it_per_s_median", q(its, 50), ci_lo=q(its, 10), ci_hi=q(its, 90), n=len(its), system=sysname,
                        source_file=st["source"], notes=f"source {st['its_src']}; GPU shared with: {shared}"
                        + (f"; log median {np.median(lg):.1f} it/s (W/logs/{st['run']}.log)" if lg is not None else "")))
        if len(st["mem"]):
            rows.append(row("peak_vram_gb", st["mem"].max(), ci_lo=q(st["mem"], 10), ci_hi=q(st["mem"], 90), n=len(st["mem"]),
                            system=sysname, source_file=st["source"], notes="perf/peak_gpu_mem_gb = torch max_memory_allocated per 100-step window; ci = p10/p90 of windows"))
        if len(st["upb"]):
            rows.append(row("utts_per_batch_median", q(st["upb"], 50), ci_lo=q(st["upb"], 10), ci_hi=q(st["upb"], 90), n=len(st["upb"]),
                            system=sysname, source_file=st["source"], notes="perf/utts_per_batch (dynamic batches capped by max_frames)"))
        if wdn:
            rows.append(row("watchdog_gpu_mem_used_gb_max", wdn["vram_max"], ci_lo=None, ci_hi=wdn["vram_p90"], n=wdn["n"], system=sysname,
                            source_file="W/logs/watchdog.log",
                            notes=f"nvidia-smi memory.used, ALL processes (desktop + concurrent jobs); ci_hi = p90; median util {wdn['util_med']:.0f}%"))
        if st["shared"]:
            rows.append(row("shared_gpu_hours", sum(st["shared"].values()), system=sysname, source_file=st["source"],
                            notes=f"active-interval overlap with other TB training runs: {shared}"))

    print("\nTOTALS")
    tot_all = 0.0
    for g, ss in groups.items():
        sts = [stages[(s["run"], s["start"], s["end"])] for s in ss]
        h = sum(s["active_h"] for s in sts); r = sum(s["raw_h"] for s in sts); n = sum(s["steps"] for s in sts)
        w = sum(s["wasted_h"] for s in sts)
        desc = " + ".join(f"{s['run']}({s['start']}->{s['end']})" for s in sts)
        print(f"  {g}: {h:.2f} GPU-h active, {r:.2f} h raw, {n} steps, redone {w:.2f} h | {desc}")
        rows.append(row("gpu_hours", h, system=g, source_file="W/runs/*/tb", n=len(sts),
                        notes=f"sum of gap-excluded stage hours: {desc}; raw elapsed {r:.2f} h; redone-after-restart {w:.2f} h not counted"))
        rows.append(row("steps", n, system=g, source_file="W/runs/*/args.json", n=len(sts), notes=desc))
        if g in ("total_base_lineage", "total_flow_refiner", "total_prompted_flow_v4pa40k"):
            tot_all += h
    rows.append(row("gpu_hours", tot_all, system="total_overall", source_file="W/runs/*/tb",
                    notes="base lineage + flow refiner (flow_v3) + prompted flow chain to flow_v4pa@40k (flow_v3p, flow_v3pc, flow_v4pa)"))
    print(f"  overall: {tot_all:.2f} GPU-h")
    # union of all TB training on the machine (incl. dead ends, ablations, crashed runs)
    allv = merge([iv for fs in others.values() for iv in intervals(fs)])
    allh = sum(b - a for a, b in allv) / 3600
    rows.append(row("gpu_busy_hours_all_tb_runs", allh, system="all_runs_union", source_file="W/runs/*/tb", n=len(others),
                    notes="union of active intervals of every TB training run in W/runs (dead ends, ablations, smoke tests incl.; eval/bench jobs not incl.)"))
    print(f"  union of all TB runs: {allh:.2f} h over {len(others)} runs")

    print("\nSPEEDUP (mix_m3hyb per TB file)")
    for s in speedup(others):
        print("  ", s)
        rows.append(row("it_per_s_median", s["its_med"], ci_lo=s["its_p10"], ci_hi=s["its_p90"], n=s["n"],
                        system=f"mix_m3hyb_file_{s['steps'][0]}-{s['steps'][1]}", source_file=s["file"],
                        notes=f"{s['t0']} .. {s['t1']}; peak VRAM median {s['mem_med']:.2f} max {s['mem_max']:.2f} GB; "
                              f"fraction of time concurrent with mix_s1 {s['frac_shared_with_mix_s1']}; "
                              f"median it/s over windows with mix_s1 not running: {s['its_med_alone']} (n={s['n_alone']})"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"\nwrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
