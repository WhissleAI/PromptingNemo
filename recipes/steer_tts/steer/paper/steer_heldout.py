"""Reusable held-out steerability evaluation (the X1 protocol) for any SteerTTS checkpoint.
Produced: reusable CLI (smoke test: queue job 06c3_steer_heldout_cli_smoke.sh -> runs/paper/cli_smoke/). All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.

    cd $STEER_TTS_ROOT
    python -m steer.paper.steer_heldout --ckpt runs/<run>/step<N>.pt --out runs/<run>/steer_heldout.json [--quick]

Protocol (steer.paper.x1_steer_heldout): LibriTTS-R test-clean voices that are in no training manifest, conditioned
by the x-vector of one real clip; harness TEXTS; per control dim requested z in {-2,-1,+1,+2} (single control, others
unspecified) vs the uninstructed render; readback by steer.features.measure AND the independent Praat extractor
(steer.paper.common.praat); Whisper-large-v3 fp16 WER; 1000-resample cluster bootstrap over (voice,text).
Full: 16 voices (8F/8M) x 8 texts = 6,272 renders (~6 min render + ~4 min measure + ~5 min ASR on an idle RTX 5070).
--quick: 8 voices (4F/4M, a subset of the full 16) x 4 texts = 1,568 renders (~4 min total).
--no_asr skips Whisper. Output: <out> (same row schema as results/steer_heldout.json: metric, value, ci_lo, ci_hi,
n, ..., target_dim, extractor, req_z, measured_dim) and <out stem>_raw.csv. Work files: <out dir>/<out stem>_work/.
The checkpoint's run dir must hold control_stats.json (and spk_ref.npz; reference-conditioned checkpoints only).
Speaker reference statistics (speaker means, Praat pooled SD) are reused from runs/paper/x1 when present.
GPU stages run in this process one after another; do not run two of these at once on a 12 GB card.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from . import common as C
from . import x1_steer_heldout as X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no_asr", action="store_true")
    ap.add_argument("--guidance", type=float, default=1.0)
    a = ap.parse_args()
    ckpt = str(Path(a.ckpt).resolve())
    out = Path(a.out).resolve()
    work = out.parent / f"{out.stem}_work"
    (work / "wav").mkdir(parents=True, exist_ok=True)
    X.OUT = work
    C.BASE = ckpt  # analyze() reads control_stats.json from the checkpoint's run dir
    V = C.heldout_voices(16)
    if a.quick:
        V = [v for v in V if v["gender"] == "F"][:4] + [v for v in V if v["gender"] == "M"][:4]
        X.TEXTS = X.TEXTS[:4]
    items = X.plan(V)
    X.render(V, items, ckpt=ckpt, guidance=a.guidance)
    M = C.measure_many([(it["path"], it["path"], it["text"]) for it in items])
    M.to_parquet(work / "measures.parquet")
    ref = C.WORK / "x1"
    if (ref / "spk_mu.csv").exists() and (ref / "pooled_sd_gt.csv").exists():
        for f in ("spk_mu.csv", "pooled_sd_gt.csv"):
            shutil.copy(ref / f, work / f)
    else:
        mu, sd, _ = X.speaker_refs(V)
        mu.to_csv(work / "spk_mu.csv"); sd.to_csv(work / "pooled_sd_gt.csv")
    if a.no_asr:
        hyps = {it["path"]: it["text"] for it in items}  # WER rows become 0 (not measured)
    else:
        hyps = C.transcribe([it["path"] for it in items], work / "asr_cache.jsonl")
    json.dump(hyps, open(work / "hyps.json", "w"))

    def write(name, rows, raw, meta=None):
        meta = dict(meta or {}, ckpt=ckpt, quick=a.quick, asr=not a.no_asr, guidance=a.guidance)
        json.dump(dict(name=name, meta=meta, rows=rows), open(out, "w"), indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
        raw.to_csv(out.with_name(out.stem + "_raw.csv"), index=False)
        print("wrote", out, len(rows), "rows", flush=True)
    C.write = write
    X.analyze(V, items)
    # one-line headline
    R = pd.DataFrame(json.load(open(out))["rows"])
    h = R[R.metric.isin(["mean_spearman_rho_12dims", "offdiag_over_diag"])][["metric", "extractor", "value"]]
    print(h.to_string(index=False))
    w = R[(R.metric == "wer") & (R.condition == "baseline")]
    print("baseline WER", w.value.tolist())


if __name__ == "__main__":
    main()
