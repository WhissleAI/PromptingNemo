"""Results tables in TensorBoard (TEXT tab), per run and across runs.

python -m steer.tb_report                 # every run in runs/ + the cross-run leaderboard
python -m steer.tb_report spk_enc_full    # one run (+ leaderboard)

Per run -> runs/<run>/tb/results  (shows in TensorBoard as "<run>/tb/results", TEXT tab, tags results/*):
  results/progress    val losses + steering harness per harness step (the table reported in chat)
  results/seedtts     Seed-TTS-eval test-en WER / SIM of this run's checkpoints (runs/seedtts/<run>_<step|final>)
  results/benchmark   rows of runs/bench/report.md for systems whose checkpoint came from this run
Across runs -> runs/_results/tb: results/leaderboard (best harness row per run), results/seedtts_all, results/benchmark_all
(the full report incl. baselines + literature).

Rewritten from scratch each call (idempotent). Called automatically by train.py after every harness, by
benchmark.py after report(), and at the end of logs/seedtts.sh - so new runs get their tables with no extra step.
"""
import json
import re
import shutil
import sys
from pathlib import Path

from .config import ROOT

RUNS = ROOT / "runs"
LIT_SEED = ("Literature (Seed-TTS-eval test-en, WER % / SIM): F5-TTS 1.83 / 0.67, CosyVoice 2 2.57 / 0.65, "
            "MaskGCT 2.62 / 0.72, Spark-TTS 1.98 / 0.58, Seed-TTS 2.25 / 0.76, ground truth 2.06 / 0.73.")
PROG_COLS = [("val/mel", "val mel"), ("val/pitch", "val pitch"), ("val/dur", "val dur"),
             ("steer_summary/control_hit_mean", "ctrl hit"), ("steer_summary/control_leak_mean", "ctrl leak"),
             ("steer_summary/control_ratio_median", "ctrl ratio"), ("steer_summary/tag_hit_mean", "tag hit"),
             ("steer_hit/tag_anger", "anger"), ("steer_hit/tag_sadness", "sadness"), ("steer_hit/tag_whisper", "whisper")]


def _scalars(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(RUNS / run / "tb"), size_guidance={"scalars": 0, "images": 1, "audio": 1, "tensors": 1})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    return {t: {e.step: e.value for e in ea.Scalars(t)} for t, _ in PROG_COLS if t in have}


def progress(run):
    """-> (markdown table, rows) with one row per harness step (val from the same step)."""
    s = _scalars(run)
    steps = sorted(s.get("steer_summary/control_hit_mean", {}) or s.get("val/mel", {}))
    if not steps:
        return "", []
    rows = [{"step": st, **{lab: s[t].get(st) for t, lab in PROG_COLS if t in s}} for st in steps]
    cols = [lab for t, lab in PROG_COLS if t in s]
    md = ["| step | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    best = max(rows, key=lambda r: (r.get("ctrl hit") or 0) + (r.get("tag hit") or 0) - (r.get("ctrl leak") or 0))
    for r in rows:
        cell = lambda v: "-" if v is None else f"{v:.3f}"
        md.append(f"| {'**' if r is best else ''}{r['step'] // 1000}k{'**' if r is best else ''} | "
                  + " | ".join(cell(r.get(c)) for c in cols) + " |")
    md.append(f"\nBold = best harness step (ctrl hit + tag hit - ctrl leak). Harness: steering hit/leak/ratio on EARS voices "
              f"(steer/harness.py); val: held-out teacher-forced losses.")
    return "\n".join(md), rows


def seedtts_rows(run=None):
    out = []
    for d in sorted((RUNS / "seedtts").glob("*")):
        m = re.fullmatch(r"(.+)_(\d+k?|final)", d.name)
        if not m or (run and m.group(1) != run):
            continue
        wer = sim = None
        if (d / "wer.score").exists():
            w = re.findall(r"WER:\s*([\d.]+)%", (d / "wer.score").read_text())
            wer = float(w[-1]) if w else None
        if (d / "sim.score").exists():
            v = re.findall(r"ASV:\s*([\d.]+)", (d / "sim.score").read_text())
            sim = float(v[-1]) if v else None
        if wer is not None or sim is not None:
            out.append((d.name, wer, sim))
    return out


def seedtts_md(rows):
    if not rows:
        return ""
    f = lambda v, p: "-" if v is None else f"{v:.{p}f}"
    return "\n".join(["| system | WER % (whisper-large-v3) | SIM (WavLM-large SV) |", "|---|---|---|"]
                     + [f"| {n} | {f(w, 2)} | {f(s, 3)} |" for n, w, s in rows] + ["", LIT_SEED])


def bench_systems():
    """benchmark system name -> run directory its checkpoint came from (None for external baselines)."""
    out = {}
    for f in (RUNS / "bench").glob("*.json"):
        try:
            ck = json.load(open(f)).get("ckpt", "")
        except Exception:  # noqa: BLE001 - a partially written json
            continue
        out[f.stem] = Path(ck).parent.name if ck and not str(ck).startswith("wavs:") else None
    return out


def bench_md(run=None):
    rep = RUNS / "bench" / "report.md"
    if not rep.exists():
        return ""
    lines = rep.read_text().splitlines()
    if run is None:
        return "\n".join(lines)
    mine = {n for n, r in bench_systems().items() if r == run}
    out, desc, table = [], "", []

    def flush():
        rows = [l for l in table[2:] if l.split("|")[1].strip() in mine]
        if rows:
            out.extend(([desc, ""] if desc else []) + table[:2] + rows + [""])

    for ln in lines + [""]:
        if ln.startswith("|"):
            table.append(ln)
        else:
            if table:
                flush(); table = []
            if ln.strip() and not ln.startswith(("#", "Literature")):
                desc = ln
    return "\n".join(out)


def _write(logdir, texts, step):
    from torch.utils.tensorboard import SummaryWriter
    shutil.rmtree(logdir, ignore_errors=True)
    w = SummaryWriter(logdir)
    for tag, md in texts.items():
        if md:
            w.add_text(tag, md, step)
    w.close()


def write_run(run):
    md, rows = progress(run)
    texts = {"results/progress": md, "results/seedtts": seedtts_md(seedtts_rows(run)), "results/benchmark": bench_md(run)}
    if any(texts.values()):
        _write(RUNS / run / "tb" / "results", texts, rows[-1]["step"] if rows else 0)
    return rows


def write_all(only=None):
    runs = [p.parent.name for p in sorted(RUNS.glob("*/tb")) if not p.parent.name.startswith(("_", "debug"))]
    lead = []
    for r in runs:
        try:
            rows = write_run(r) if (only is None or r == only) else progress(r)[1]
        except Exception as e:  # noqa: BLE001 - one broken event file must not block the rest
            print("skip", r, type(e).__name__, e); continue
        if rows and any("ctrl hit" in x for x in rows):
            b = max(rows, key=lambda x: (x.get("ctrl hit") or 0) + (x.get("tag hit") or 0) - (x.get("ctrl leak") or 0))
            lead.append((r, b, rows[-1]["step"]))
    cols = ["val mel", "ctrl hit", "ctrl leak", "tag hit", "anger", "sadness"]
    md = ["| run | trained to | best step | " + " | ".join(cols) + " |", "|---" * (len(cols) + 3) + "|"]
    for r, b, last in lead:
        md.append(f"| {r} | {last // 1000}k | {b['step'] // 1000}k | "
                  + " | ".join("-" if b.get(c) is None else f"{b[c]:.3f}" for c in cols) + " |")
    _write(RUNS / "_results" / "tb", {"results/leaderboard": "\n".join(md),
                                      "results/seedtts_all": seedtts_md(seedtts_rows()),
                                      "results/benchmark_all": bench_md()}, 0)
    print("tb_report:", ", ".join(r for r, _, _ in lead), "+ runs/_results/tb")


if __name__ == "__main__":
    write_all(sys.argv[1] if len(sys.argv) > 1 else None)
