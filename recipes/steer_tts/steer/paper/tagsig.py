"""Tag signatures: which instructed styles/emotions leave a trace in the 12 speaker-relative controls.

Data analysis only (no synthesis), on the training controls table (data/meta/controls.parquet).
  * per (tag, source) mean speaker-relative z on every control with a 95% CI from a speaker-cluster bootstrap
    (speakers resampled, 2000 draws; each speaker contributes its own mean so prolific speakers do not dominate);
    sources: acted = EARS + Expresso, api = Cartesia renders at default speed and volume (so the emotion setting is
    the only difference from the voice's neutral render), nvtts = NonverbalTTS.
  * 12x12 within-speaker correlation of the controls (each speaker's own mean removed), all corpora and per corpus.

python -m steer.paper.tagsig  -> paper/results/tagsig_signatures.json, tagsig_corr.json
"""
import json

import numpy as np
import pandas as pd

from ..config import CONTROL_DIMS, META
from .common import RES

ZC = [f"z_{k}" for k in CONTROL_DIMS]
SRC = {"ears": "acted", "expresso": "acted", "cartesia": "api", "nvtts": "nvtts"}


def boot_ci(per_spk, n=2000, seed=0):
    """per_spk: (S, D) speaker means (NaN allowed). Returns mean over speakers, lo, hi (2.5/97.5 pct)."""
    rng = np.random.default_rng(seed)
    S = len(per_spk)
    m = np.nanmean(per_spk, 0)
    if S < 2:
        return m, m * np.nan, m * np.nan
    idx = rng.integers(0, S, (n, S))
    bs = np.stack([np.nanmean(per_spk[i], 0) for i in idx])
    return m, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)


def main():
    df = pd.read_parquet(META / "controls.parquet")
    df = df[df.corpus != "libritts"].copy()
    df["src"] = df.corpus.map(SRC)
    api_default = (df.corpus == "cartesia") & df.speed.fillna(1).eq(1) & df.volume.fillna(1).eq(1)
    df = df[(df.corpus != "cartesia") | api_default]
    rows = []
    for (tag, src), g in df.groupby(["tag", "src"]):
        if tag == "none" or len(g) < 50:
            continue
        ps = g.groupby("speaker")[ZC].mean().to_numpy(np.float64)
        m, lo, hi = boot_ci(ps)
        sig = (lo > 0) | (hi < 0)
        rows.append(dict(tag=tag, src=src, n_utt=len(g), n_spk=len(ps), mean=np.round(m, 3).tolist(),
                         lo=np.round(lo, 3).tolist(), hi=np.round(hi, 3).tolist(), sig=sig.tolist(),
                         max_abs=round(float(np.nanmax(np.abs(m))), 3), l2=round(float(np.sqrt(np.nansum(m ** 2))), 3),
                         n_sig_03=int(np.sum(sig & (np.abs(m) >= 0.3)))))
    # summary: "silent" tags = no control with |mean z| >= 0.3 and a CI excluding 0
    summ = {}
    for src in ("acted", "api", "nvtts"):
        r = [x for x in rows if x["src"] == src and x["tag"] != "neutral"]
        summ[src] = dict(n_tags=len(r), n_silent=sum(x["n_sig_03"] == 0 for x in r),
                         silent=[x["tag"] for x in r if x["n_sig_03"] == 0],
                         median_max_abs=round(float(np.median([x["max_abs"] for x in r])), 3) if r else None,
                         median_l2=round(float(np.median([x["l2"] for x in r])), 3) if r else None)
    # same emotion, two sources
    both = {}
    by = {(x["tag"], x["src"]): x for x in rows}
    for t in sorted({x["tag"] for x in rows}):
        if (t, "acted") in by and (t, "api") in by and t != "neutral":
            a, b = np.array(by[t, "acted"]["mean"]), np.array(by[t, "api"]["mean"])
            ok = np.isfinite(a) & np.isfinite(b)
            both[t] = dict(l2_acted=by[t, "acted"]["l2"], l2_api=by[t, "api"]["l2"],
                           cos=round(float(a[ok] @ b[ok] / (np.linalg.norm(a[ok]) * np.linalg.norm(b[ok]) + 1e-9)), 3))
    out = dict(dims=CONTROL_DIMS, ci="95% speaker-cluster bootstrap (2000)", api_subset="cartesia speed=1, volume=1",
               min_utts=50, silent_rule="no dim with |mean|>=0.3 and CI excluding 0", summary=summ, acted_vs_api=both,
               rows=rows)
    json.dump(out, open(RES / "tagsig_signatures.json", "w"), indent=1)

    # within-speaker correlation
    full = pd.read_parquet(META / "controls.parquet")
    corr = {}
    for name, g in [("all", full)] + [(c, full[full.corpus == c]) for c in sorted(full.corpus.unique())] + \
            [("expressive", full[full.corpus != "libritts"])]:
        x = g[ZC] - g.groupby("speaker")[ZC].transform("mean")
        c = x.corr(min_periods=100).to_numpy()
        off = c[~np.eye(12, dtype=bool)]
        corr[name] = dict(n=len(g), corr=np.round(c, 3).tolist(), mean_abs_offdiag=round(float(np.nanmean(np.abs(off))), 3),
                          n_abs_gt_03=int(np.sum(np.abs(off) > 0.3) // 2))
        # strongest pairs
        pairs = [(CONTROL_DIMS[i], CONTROL_DIMS[j], round(float(c[i, j]), 3)) for i in range(12) for j in range(i + 1, 12)]
        corr[name]["top_pairs"] = sorted(pairs, key=lambda p: -abs(p[2]) if p[2] == p[2] else 0)[:8]
    json.dump(dict(dims=CONTROL_DIMS, method="Pearson on z minus each speaker's own mean", sets=corr),
              open(RES / "tagsig_corr.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print(json.dumps(both, indent=1))
    for k in ("all", "expressive"):
        print(k, corr[k]["mean_abs_offdiag"], corr[k]["top_pairs"])


if __name__ == "__main__":
    main()
