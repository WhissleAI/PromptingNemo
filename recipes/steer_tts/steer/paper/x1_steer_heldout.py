"""X1: steerability on HELD-OUT voices, read back by two extractors, with the intelligibility cost.
Produced: results/steer_heldout.json + _raw.csv: python -m steer.paper.x1_steer_heldout --stage render, then --stage measure, --stage asr, --stage analyze (runs/paper/chain12.sh). All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.

16 LibriTTS-R test-clean voices (not in training; cloned from one real clip via its x-vector) x 8 harness texts x
{uninstructed baseline} + {12 dims x requested z in -2,-1,+1,+2, single control specified, others unspecified}.
Every render is measured with (A) steer.features.measure (the training-label extractor) and (B) an independent
Praat/numpy extractor (common.praat), and transcribed by Whisper-large-v3 fp16.

python -m steer.paper.x1_steer_heldout [--stage render|measure|asr|analyze|all]
"""
import argparse
import json

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import spearmanr

from ..config import CONTROL_DIMS, SR
from ..harness import TEXTS
from . import common as C

NAME = "steer_heldout"
CLUSTER = "cell"  # bootstrap unit: "cell" = (voice,text), "voice" = all texts of a resampled voice
ZS = [-2.0, -1.0, 1.0, 2.0]
OUT = C.WORK / "x1"


def plan(V):
    items = []
    for v in V:
        for ti, text in enumerate(TEXTS):
            items.append(dict(vid=v["vid"], gender=v["gender"], text_id=ti, dim="baseline", req_z=0.0))
            for d in CONTROL_DIMS:
                for z in ZS:
                    items.append(dict(vid=v["vid"], gender=v["gender"], text_id=ti, dim=d, req_z=z))
    for it in items:
        it["text"] = TEXTS[it["text_id"]]
        it["path"] = str(OUT / "wav" / f"{it['vid']}_{it['text_id']}_{it['dim']}_{it['req_z']:+.0f}.wav")
    return items


def render(V, items, ckpt=C.BASE, guidance=1.0):
    s = C.make_synth(ckpt, guidance=guidance)
    ref = {v["vid"]: v["ref"] for v in V}
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items):
        ctrl = None if it["dim"] == "baseline" else {it["dim"]: it["req_z"]}
        w = s(it["text"], f"heldout_{it['vid']}", ctrl=ctrl, ref_wav=ref[it["vid"]])
        sf.write(it["path"], w, SR)
        if i % 500 == 0:
            print(f"render {i}/{len(items)}", flush=True)


def speaker_refs(V, all_bench=True):
    """Per held-out voice: mean of both extractors over its real test-clean clips (speaker reference), and pooled
    within-speaker SD for the Praat features over all 36 test-clean speakers (independent normaliser)."""
    rows = [json.loads(l) for l in open(C.BENCH)]
    items = [(f"{r['speaker']}|{i}", r["gt_path"], r["text"]) for i, r in enumerate(rows)]
    M = C.measure_many(items)
    M["speaker"] = M.key.str.split("|").str[0]
    vmap = {v["speaker"]: v["vid"] for v in V}
    feats = CONTROL_DIMS + C.PRAAT_FEATS
    resid = M[feats] - M.groupby("speaker")[feats].transform("mean")
    sd = resid.std(ddof=1)
    mu = M[M.speaker.isin(vmap)].groupby("speaker")[feats].mean()
    mu.index = [vmap[s] for s in mu.index]
    M.to_csv(OUT / "gt_measures_private.csv", index=False)
    return mu, sd, len(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--n_voices", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    V = C.heldout_voices(a.n_voices)
    items = plan(V)
    st = a.stage
    if st in ("render", "all"):
        render(V, items)
    if st in ("measure", "all"):
        M = C.measure_many([(it["path"], it["path"], it["text"]) for it in items])
        M.to_parquet(OUT / "measures.parquet")
        mu, sd, n_gt = speaker_refs(V)
        mu.to_csv(OUT / "spk_mu.csv"); sd.to_csv(OUT / "pooled_sd_gt.csv")
        print("measured", len(M), flush=True)
    if st in ("asr", "all"):
        hyps = C.transcribe([it["path"] for it in items], OUT / "asr_cache.jsonl")
        json.dump(hyps, open(OUT / "hyps.json", "w"))
    if st in ("analyze", "all"):
        analyze(V, items)


def analyze(V, items):
    stats = json.load(open(C.BASE.rsplit("/", 1)[0] + "/control_stats.json"))
    sd_train = pd.Series(stats["pooled_sd"])[CONTROL_DIMS]
    present = [d for d in CONTROL_DIMS if any(it["dim"] == d for it in items)]  # target dims in this item set
    PI = [CONTROL_DIMS.index(d) for d in present]
    M = pd.read_parquet(OUT / "measures.parquet").set_index("key")
    mu = pd.read_csv(OUT / "spk_mu.csv", index_col=0)
    sd_gt = pd.read_csv(OUT / "pooled_sd_gt.csv", index_col=0).iloc[:, 0]
    hyps = json.load(open(OUT / "hyps.json"))
    df = pd.DataFrame(items)
    df = df.join(M, on="path")
    df["hyp"] = df.path.map(hyps)
    ew = [C.word_errors(t, h) for t, h in zip(df.text, df.hyp)]
    df["err"] = [e for e, _ in ew]
    df["nwords"] = [n for _, n in ew]
    # z units: ours in the TRAINING pooled SD (conditioning units); Praat in the pooled within-speaker SD of real
    # held-out test-clean speech measured by the Praat extractor itself
    P = [C.PRAAT_MAP[d] for d in CONTROL_DIMS]
    for d in CONTROL_DIMS:
        df[f"zabs_{d}"] = (df[d] - df.vid.map(mu[d])) / sd_train[d]
    base = df[df.dim == "baseline"].set_index(["vid", "text_id"])
    for d in CONTROL_DIMS:
        b = base[d].reindex(pd.MultiIndex.from_arrays([df.vid, df.text_id])).values
        df[f"dz_{d}"] = (df[d].values - b) / sd_train[d]
    for f in C.PRAAT_FEATS:
        b = base[f].reindex(pd.MultiIndex.from_arrays([df.vid, df.text_id])).values
        df[f"dp_{f}"] = (df[f].values - b) / sd_gt[f]
    pub = df.drop(columns=["path"]).copy()
    pub["wav"] = df.path.str.replace(str(C.ROOT) + "/", "", regex=False)

    # tensor D[u, target, level, measured]
    units = sorted({(v, t) for v, t in zip(df.vid, df.text_id)})
    U = {u: i for i, u in enumerate(units)}
    nd, nl = len(CONTROL_DIMS), len(ZS)
    cols = [f"dz_{d}" for d in CONTROL_DIMS] + [f"dp_{f}" for f in P]
    D = np.full((len(units), nd, nl, len(cols)), np.nan)
    Z = np.array(ZS)
    for r in df[df.dim != "baseline"].itertuples(index=False):
        D[U[(r.vid, r.text_id)], CONTROL_DIMS.index(r.dim), ZS.index(r.req_z)] = [getattr(r, c) for c in cols]
    Wb = df[df.dim == "baseline"].set_index(["vid", "text_id"])
    E = np.zeros((len(units), nd, nl)); N = np.zeros((len(units), nd, nl))
    for r in df[df.dim != "baseline"].itertuples(index=False):
        E[U[(r.vid, r.text_id)], CONTROL_DIMS.index(r.dim), ZS.index(r.req_z)] = r.err
        N[U[(r.vid, r.text_id)], CONTROL_DIMS.index(r.dim), ZS.index(r.req_z)] = r.nwords
    Eb = np.array([Wb.loc[u, "err"] for u in units]); Nb = np.array([Wb.loc[u, "nwords"] for u in units])

    rng = np.random.default_rng(0)
    B = 1000
    if CLUSTER == "voice":
        vids = sorted({u[0] for u in units})
        byv = [np.array([U[u] for u in units if u[0] == v]) for v in vids]
        boots = [np.arange(len(units))] + [np.concatenate([byv[k] for k in rng.integers(0, len(vids), len(vids))])
                                           for _ in range(B)]
    else:
        boots = [np.arange(len(units))] + [rng.integers(0, len(units), len(units)) for _ in range(B)]
    UC = ["vid"] if CLUSTER == "voice" else ["vid", "text_id"]

    def ci(fn):
        vals = np.array([fn(ix) for ix in boots], float)
        v, bs = vals[0], vals[1:]
        bs = bs[np.isfinite(bs)]
        return (v, np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if len(bs) > 10 else (v, np.nan, np.nan)

    def rho(ix, i, k):
        x = np.repeat(Z[None], len(ix), 0).ravel(); y = D[ix, i, :, k].ravel()
        m = np.isfinite(y)
        return spearmanr(x[m], y[m]).statistic if m.sum() > 10 else np.nan

    def slope(ix, i, k):
        y = D[ix, i, :, k]; z = np.repeat(Z[None], len(ix), 0)
        m = np.isfinite(y)
        return np.sum(z[m] * y[m]) / np.sum(z[m] ** 2) if m.any() else np.nan  # OLS through 0 (delta=0 at baseline)

    def agree(ix, i):
        x, y = D[ix, i, :, i].ravel(), D[ix, i, :, nd + i].ravel()
        m = np.isfinite(x) & np.isfinite(y)
        return np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 10 else np.nan

    src = f"{C.RES}/{NAME}_raw.csv"
    split = "libritts_r_test_clean_heldout16"
    R = []
    n_u = len(units)
    for i, d in enumerate(CONTROL_DIMS):
        if d not in present:
            continue
        for ext, k, scorer in (("ours", i, "steer.features.measure"), ("praat", nd + i, f"praat:{C.PRAAT_MAP[d]}")):
            cond = dict(condition="single_control", target_dim=d, extractor=ext)
            R.append(C.row("spearman_rho", ci(lambda ix: rho(ix, i, k)), n_u * nl, src, scorer, split, **cond))
            R.append(C.row("slope_dz_per_reqz", ci(lambda ix: slope(ix, i, k)), n_u * nl, src, scorer, split, **cond))
            for l, z in enumerate(ZS):
                c2 = dict(cond, req_z=z)
                R.append(C.row("mean_delta_z", ci(lambda ix: np.nanmean(D[ix, i, l, k])), n_u, src, scorer, split, **c2))
                R.append(C.row("hit_rate", ci(lambda ix: np.nanmean(np.where(np.isfinite(D[ix, i, l, k]),
                               (np.sign(z) * D[ix, i, l, k] > 0.5).astype(float), np.nan))), n_u, src, scorer, split, **c2))
                R.append(C.row("direction_acc", ci(lambda ix: np.nanmean(np.where(np.isfinite(D[ix, i, l, k]),
                               (np.sign(z) * D[ix, i, l, k] > 0).astype(float), np.nan))), n_u, src, scorer, split, **c2))
            others = [j for j in range(nd) if j != i] if ext == "ours" else [nd + j for j in range(nd) if j != i]
            for l in (0, 3):
                R.append(C.row("leak_mean_abs_dz_other", ci(lambda ix: np.nanmean(np.abs(D[ix, i, l][:, others]))),
                               n_u, src, scorer, split, **dict(cond, req_z=ZS[l])))
        # absolute targeting (ours): measured z vs the voice's real-speech mean, per requested level
        for l, z in enumerate(ZS):
            sub = df[(df.dim == d) & (df.req_z == z)]
            R.append(C.row("mean_abs_z_vs_speaker_ref", C.boot_ratio(sub, UC, f"zabs_{d}"),
                           len(sub), src, "steer.features.measure", split, condition="single_control", target_dim=d,
                           extractor="ours", req_z=z))
        sub = df[df.dim == "baseline"]
        R.append(C.row("mean_abs_z_vs_speaker_ref", C.boot_ratio(sub, UC, f"zabs_{d}"),
                       len(sub), src, "steer.features.measure", split, condition="baseline", target_dim=d, extractor="ours"))
        # extractor agreement on the target dim, over all renders that requested it (deltas)
        x, y = D[:, i, :, i].ravel(), D[:, i, :, nd + i].ravel()
        m = np.isfinite(x) & np.isfinite(y)
        R.append(C.row("extractor_agreement_pearson_r", ci(lambda ix: agree(ix, i)), int(m.sum()), src,
                       f"ours vs praat:{C.PRAAT_MAP[d]}", split, condition="single_control", target_dim=d))
        R.append(C.row("extractor_agreement_spearman", spearmanr(x[m], y[m]).statistic, int(m.sum()), src,
                       f"ours vs praat:{C.PRAAT_MAP[d]}", split, condition="single_control", target_dim=d))
        # intelligibility per level
        for l, z in enumerate(ZS):
            R.append(C.row("wer", ci(lambda ix: E[ix, i, l].sum() / N[ix, i, l].sum()), n_u, src,
                           "whisper-large-v3 fp16 + whisper EnglishTextNormalizer", split, condition="single_control",
                           target_dim=d, req_z=z))
            R.append(C.row("wer_minus_baseline", ci(lambda ix: E[ix, i, l].sum() / N[ix, i, l].sum() - Eb[ix].sum() / Nb[ix].sum()),
                           n_u, src, "whisper-large-v3 fp16 + whisper EnglishTextNormalizer", split,
                           condition="single_control", target_dim=d, req_z=z))
    # cross-talk matrices: slope of delta on measured dim j per unit of requested z on target i
    for ext, off in (("ours", 0), ("praat", nd)):
        for i, d in enumerate(CONTROL_DIMS):
            if d not in present:
                continue
            for j, d2 in enumerate(CONTROL_DIMS):
                R.append(C.row("crosstalk_slope", ci(lambda ix: slope(ix, i, off + j)), n_u * nl, src,
                               "steer.features.measure" if ext == "ours" else f"praat:{C.PRAAT_MAP[d2]}", split,
                               condition="single_control", target_dim=d, measured_dim=d2, extractor=ext))
    # global summaries
    R.append(C.row("wer", ci(lambda ix: Eb[ix].sum() / Nb[ix].sum()), n_u, src,
                   "whisper-large-v3 fp16 + whisper EnglishTextNormalizer", split, condition="baseline"))
    for l, z in enumerate(ZS):
        R.append(C.row("wer", ci(lambda ix: E[ix, :, l].sum() / N[ix, :, l].sum()), n_u * nd, src,
                       "whisper-large-v3 fp16 + whisper EnglishTextNormalizer", split, condition="all_dims", req_z=z))
    for ext, off in (("ours", 0), ("praat", nd)):
        R.append(C.row("mean_spearman_rho_12dims", np.nanmean([rho(boots[0], i, off + i) for i in PI]), n_u * nl * nd,
                       src, ext, split, condition="single_control", extractor=ext))
        diag = lambda ix: np.mean([abs(slope(ix, i, off + i)) for i in PI])
        offd = lambda ix: np.mean([abs(slope(ix, i, off + j)) for i in PI for j in range(nd) if i != j])
        R.append(C.row("mean_abs_diag_slope", ci(diag), n_u, src, ext, split, condition="single_control", extractor=ext))
        R.append(C.row("mean_abs_offdiag_slope", ci(offd), n_u, src, ext, split, condition="single_control", extractor=ext))
        R.append(C.row("offdiag_over_diag", ci(lambda ix: offd(ix) / diag(ix)), n_u, src, ext, split,
                       condition="single_control", extractor=ext))
    meta = dict(ckpt=C.BASE, guidance=1.0, voices=len(V),
                genders={g: sum(v["gender"] == g for v in V) for g in "FM"},
                gender_rule="median F0 of the speaker's test-clean clips >= 165 Hz -> F (no gender labels on disk)",
                texts=len(TEXTS), req_z=ZS, heldout_check=C.heldout_check(C.BASE),
                conditioning="WavLM-base-plus-sv x-vector of one real test-clean clip (4-10 s) of the voice",
                z_units=dict(ours="training pooled within-speaker SD (control_stats.json)",
                             praat="pooled within-speaker SD of the Praat features over the 500 real test-clean bench clips (36 speakers)"),
                delta="directed render minus the uninstructed render of the same voice/text",
                praat_map=C.PRAAT_MAP, bootstrap="1000 cluster resamples over (voice,text) units, percentile 95% CI",
                hit_rate="share of units with sign(req_z)*delta > 0.5 z")
    meta["bootstrap_cluster"] = CLUSTER
    C.write(NAME if CLUSTER == "cell" else NAME + "_voiceboot", R, pub, meta)


if __name__ == "__main__":
    main()
