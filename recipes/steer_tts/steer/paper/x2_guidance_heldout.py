"""X2: steering-guidance scale on HELD-OUT voices: steer ratio / hit / leak / WER vs s in {0,1,2,3}.
Produced: results/guidance_heldout.json + _raw.csv: python -m steer.paper.x2_guidance_heldout --stage render|measure|asr|analyze (needs X1 work files; runs/paper/chain12.sh). All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.

Directions: rate, loud, f0_med, f0_iqr, tilt at requested z = +-2 (single control), plus style tags anger and
whisper. Same 16 test-clean voices x 8 texts as X1; baseline = X1's uninstructed render (guidance only acts on
requests, so the baseline is identical for every s). Readback: ours + Praat (common.praat).

python -m steer.paper.x2_guidance_heldout [--stage render|measure|asr|analyze|all]
"""
import argparse
import json

import numpy as np
import pandas as pd
import soundfile as sf

from ..config import CONTROL_DIMS, SR
from ..harness import TEXTS
from . import common as C

NAME = "guidance_heldout"
CLUSTER = "cell"
OUT = C.WORK / "x2"
X1 = C.WORK / "x1"
SCALES = [0.0, 1.0, 2.0, 3.0]
DIRS = {f"{d}{'+' if z > 0 else '-'}": (dict(ctrl={d: z}), d, int(np.sign(z)), abs(z))
        for d in ("rate", "loud", "f0_med", "f0_iqr", "tilt") for z in (2.0, -2.0)}
DIRS.update({"tag_anger": (dict(tag="anger"), "loud", 1, None), "tag_whisper": (dict(tag="whisper"), "voiced", -1, None)})


def plan(V):
    items = []
    for v in V:
        for ti, text in enumerate(TEXTS):
            items.append(dict(vid=v["vid"], text_id=ti, text=text, direction="baseline", s=np.nan,
                              path=str(X1 / "wav" / f"{v['vid']}_{ti}_baseline_+0.wav")))
            for name in DIRS:
                for s in SCALES:
                    items.append(dict(vid=v["vid"], text_id=ti, text=text, direction=name, s=s,
                                      path=str(OUT / "wav" / f"{v['vid']}_{ti}_{name}_s{s:.0f}.wav")))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    V = C.heldout_voices(16)
    items = plan(V)
    ref = {v["vid"]: v["ref"] for v in V}
    if a.stage in ("render", "all"):
        syn = C.make_synth(C.BASE)
        (OUT / "wav").mkdir(parents=True, exist_ok=True)
        for i, it in enumerate(items):
            if it["direction"] == "baseline":
                continue
            kw = DIRS[it["direction"]][0]
            w = syn(it["text"], f"heldout_{it['vid']}", ref_wav=ref[it["vid"]], guidance=it["s"], **kw)
            sf.write(it["path"], w, SR)
            if i % 500 == 0:
                print(f"render {i}/{len(items)}", flush=True)
    if a.stage in ("measure", "all"):
        todo = [(it["path"], it["path"], it["text"]) for it in items if it["direction"] != "baseline"]
        M = C.measure_many(todo)
        M1 = pd.read_parquet(X1 / "measures.parquet")
        M = pd.concat([M, M1[M1.key.str.contains("_baseline_")]], ignore_index=True)
        M.to_parquet(OUT / "measures.parquet")
    if a.stage in ("asr", "all"):
        hyps = C.transcribe([it["path"] for it in items], X1 / "asr_cache.jsonl")
        json.dump(hyps, open(OUT / "hyps.json", "w"))
    if a.stage in ("analyze", "all"):
        analyze(items)


def analyze(items):
    stats = json.load(open(C.BASE.rsplit("/", 1)[0] + "/control_stats.json"))
    sd = pd.Series(stats["pooled_sd"])[CONTROL_DIMS]
    sd_gt = pd.read_csv(X1 / "pooled_sd_gt.csv", index_col=0).iloc[:, 0]
    M = pd.read_parquet(OUT / "measures.parquet").drop_duplicates("key").set_index("key")
    hyps = json.load(open(OUT / "hyps.json"))
    df = pd.DataFrame(items).join(M, on="path")
    df["hyp"] = df.path.map(hyps)
    ew = [C.word_errors(t, h) for t, h in zip(df.text, df.hyp)]
    df["err"], df["nwords"] = [e for e, _ in ew], [n for _, n in ew]
    base = df[df.direction == "baseline"].set_index(["vid", "text_id"])
    idx = pd.MultiIndex.from_arrays([df.vid, df.text_id])
    for d in CONTROL_DIMS:
        df[f"dz_{d}"] = (df[d].values - base[d].reindex(idx).values) / sd[d]
        f = C.PRAAT_MAP[d]
        df[f"dp_{d}"] = (df[f].values - base[f].reindex(idx).values) / sd_gt[f]
    df["base_err"] = base["err"].reindex(idx).values
    df["base_nwords"] = base["nwords"].reindex(idx).values
    src = f"{C.RES}/{NAME}_raw.csv"
    split = "libritts_r_test_clean_heldout16"
    asr = "whisper-large-v3 fp16 + whisper EnglishTextNormalizer"
    R = []
    U = ["vid"] if CLUSTER == "voice" else ["vid", "text_id"]
    for name, (_, dim, sign, step) in DIRS.items():
        for s in SCALES:
            sub = df[(df.direction == name) & (df.s == s)]
            cond = dict(condition="guidance", direction=name, target_dim=dim, s=s)
            others = [d for d in CONTROL_DIMS if d != dim]
            for ext, p in (("ours", "dz_"), ("praat", "dp_")):
                x = lambda q: sign * q[f"{p}{dim}"]
                if step:
                    R.append(C.row("steer_ratio_median", C.boot_df(sub, U, lambda q: np.nanmedian(x(q)) / step, B=300), len(sub),
                                   src, ext, split, extractor=ext, **cond))
                R.append(C.row("median_signed_delta_z", C.boot_df(sub, U, lambda q: np.nanmedian(x(q)), B=300), len(sub), src,
                               ext, split, extractor=ext, **cond))
                R.append(C.row("hit_rate", C.boot_ratio(sub, U, (x(sub) > 0.5).astype(float).where(x(sub).notna())),
                               len(sub), src, ext, split, extractor=ext, **cond))
                R.append(C.row("leak_mean_abs_dz_other", C.boot_ratio(sub, U, sub[[p + o for o in others]].abs().mean(1)),
                               len(sub), src, ext, split, extractor=ext, **cond))
            R.append(C.row("wer", C.boot_ratio(sub, U, "err", "nwords"), len(sub), src, asr, split, **cond))
            R.append(C.row("wer_minus_baseline", C.boot_df(sub, U, lambda q: q.err.sum() / q.nwords.sum() - q.base_err.sum() / q.base_nwords.sum(), B=300),
                           len(sub), src, asr, split, **cond))
    for s in SCALES:  # pooled over the 10 continuous directions
        sub = df[(df.s == s) & df.direction.str.match(r"^(rate|loud|f0_med|f0_iqr|tilt)[+-]$")].copy()
        tgt = sub.direction.map(lambda n: DIRS[n][1]); sg = sub.direction.map(lambda n: DIRS[n][2])
        sub["sx"] = [sg_ * r[f"dz_{t}"] for sg_, t, (_, r) in zip(sg, tgt, sub.iterrows())]
        sub["sxp"] = [sg_ * r[f"dp_{t}"] for sg_, t, (_, r) in zip(sg, tgt, sub.iterrows())]
        sub["lk"] = [np.nanmean([abs(r[f"dz_{o}"]) for o in CONTROL_DIMS if o != t]) for t, (_, r) in zip(tgt, sub.iterrows())]
        cond = dict(condition="guidance_pooled_5dims", s=s)
        R.append(C.row("steer_ratio_median", C.boot_df(sub, U, lambda q: np.nanmedian(q.sx) / 2, B=300), len(sub), src, "ours", split, extractor="ours", **cond))
        R.append(C.row("steer_ratio_median", C.boot_df(sub, U, lambda q: np.nanmedian(q.sxp) / 2, B=300), len(sub), src, "praat", split, extractor="praat", **cond))
        R.append(C.row("hit_rate", C.boot_ratio(sub, U, (sub.sx > 0.5).astype(float)), len(sub), src, "ours", split, extractor="ours", **cond))
        R.append(C.row("leak_mean_abs_dz_other", C.boot_ratio(sub, U, "lk"), len(sub), src, "ours", split, extractor="ours", **cond))
        R.append(C.row("wer", C.boot_ratio(sub, U, "err", "nwords"), len(sub), src, asr, split, **cond))
    pub = df.drop(columns=["path"]).assign(wav=df.path.str.replace(str(C.ROOT) + "/", "", regex=False))
    meta = dict(ckpt="runs/spk_enc_v3/step312500.pt", scales=SCALES, directions={k: [str(v[0]), v[1], v[2], v[3]] for k, v in DIRS.items()},
                guidance="SteerTTS.infer: g = g_base + s (g_req - g_base), g_base = no-request condition of the same voice/text; s=0 => baseline, s=1 => plain",
                steer_ratio="median(sign * delta_z on target) / |requested z|", reference_training_voices="runs/cfg_sweep_320k.json")
    meta["bootstrap_cluster"] = CLUSTER
    C.write(NAME if CLUSTER == "cell" else NAME + "_voiceboot", R, pub, meta)


if __name__ == "__main__":
    main()
