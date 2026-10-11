"""Reviewer follow-up (b2): natural within-speaker covariation as the reference for cross-talk ("leak").
Produced: results/crosstalk_natural_reference.json + _raw.csv: python -m steer.paper.crosstalk_natural. All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.

Within-speaker residuals (clip minus speaker mean) of the 12 controls on the 500 real LibriTTS-R test-clean bench
clips (36 held-out speakers), measured by steer.features.measure and by the Praat extractor. For each pair (i, j):
natural slope b_ij = cov(r_i, r_j) / var(r_i) in z units, i.e. how much dim j moves in real speech when dim i moves by
1 z. Compared with the model's cross-talk slopes from results/steer_heldout.json (same units, same extractor).
python -m steer.paper.crosstalk_natural -> results/crosstalk_natural_reference.json (+ _raw.csv)
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import CONTROL_DIMS
from . import common as C


def main():
    M = pd.read_csv(C.WORK / "x1/gt_measures_private.csv")
    M["speaker"] = M.key.str.split("|").str[0]
    stats = json.load(open(C.BASE.rsplit("/", 1)[0] + "/control_stats.json"))
    sd_tr = pd.Series(stats["pooled_sd"])
    sd_gt = pd.read_csv(C.WORK / "x1/pooled_sd_gt.csv", index_col=0).iloc[:, 0]
    X = pd.read_json(C.RES / "steer_heldout.json").rows if False else json.load(open(C.RES / "steer_heldout.json"))["rows"]
    X = pd.DataFrame(X)
    R, raw = [], []
    rng = np.random.default_rng(0)
    spk = M.speaker.unique()
    for ext, cols, sd in (("ours", CONTROL_DIMS, sd_tr), ("praat", [C.PRAAT_MAP[d] for d in CONTROL_DIMS], sd_gt)):
        Z = M[cols].div(sd[cols])
        res = (Z - Z.groupby(M.speaker).transform("mean")).values

        def slopes(r):
            r = r[np.isfinite(r).all(1)]
            Cv = np.cov(r.T)
            return Cv / np.diag(Cv)[:, None]  # [i, j] = cov_ij / var_i

        B0 = slopes(res)
        idx = {s: np.where(M.speaker.values == s)[0] for s in spk}
        boots = [slopes(res[np.concatenate([idx[s] for s in rng.choice(spk, len(spk))])]) for _ in range(500)]
        off = ~np.eye(12, dtype=bool)
        nat = np.abs(B0[off]).mean()
        nb = [np.abs(b[off]).mean() for b in boots]
        R.append(C.row("natural_mean_abs_offdiag_slope", (nat, np.percentile(nb, 2.5), np.percentile(nb, 97.5)), len(M),
                       "data/bench/libritts_gt (500 real clips)", ext, "libritts_r_test_clean_real36spk",
                       condition="real_speech_within_speaker", extractor=ext, bootstrap_cluster="speaker"))
        ct = X[(X.metric == "crosstalk_slope") & (X.extractor == ext)].pivot(index="target_dim", columns="measured_dim", values="value")
        ct = ct.loc[CONTROL_DIMS, CONTROL_DIMS].values
        diag = np.abs(np.diag(ct))
        model_norm = (np.abs(ct) / diag[:, None])[off]  # off-diagonal per unit of REALISED target change
        R.append(C.row("model_mean_abs_offdiag_per_realised_unit", float(model_norm.mean()), 144 - 12,
                       str(C.RES / "steer_heldout.json"), ext, "libritts_r_test_clean_heldout16", condition="model", extractor=ext))
        rho = spearmanr(np.abs(B0[off]), np.abs((ct / diag[:, None])[off])).statistic
        R.append(C.row("spearman_abs_model_vs_natural_offdiag", float(rho), 132, str(C.RES / "steer_heldout.json"), ext,
                       "libritts_r_test_clean", condition="model_vs_natural", extractor=ext))
        r2 = spearmanr(B0[off], (ct / diag[:, None])[off]).statistic
        R.append(C.row("spearman_signed_model_vs_natural_offdiag", float(r2), 132, str(C.RES / "steer_heldout.json"), ext,
                       "libritts_r_test_clean", condition="model_vs_natural", extractor=ext))
        for i, d in enumerate(CONTROL_DIMS):
            for j, d2 in enumerate(CONTROL_DIMS):
                lo, hi = np.percentile([b[i, j] for b in boots], [2.5, 97.5])
                raw.append(dict(extractor=ext, target_dim=d, measured_dim=d2, natural_slope=B0[i, j], nat_ci_lo=lo,
                                nat_ci_hi=hi, model_slope=ct[i, j], model_slope_per_realised=ct[i, j] / diag[i]))
    C.write("crosstalk_natural_reference", R, pd.DataFrame(raw),
            dict(note="natural slope b_ij = cov(i,j)/var(i) of within-speaker residuals of real speech; model slope "
                      "normalised by its realised diagonal gain so both are 'dim j change per 1 z realised on dim i'"))


if __name__ == "__main__":
    main()
