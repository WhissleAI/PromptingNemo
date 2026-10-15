"""Paper numbers for the CREMA-D appendix -> $EMOTION_STUDY_OUT/paper_stats.json (+ printed tables).

Inputs (all in $EMOTION_STUDY_OUT): clips.parquet (s01), deltas.parquet + word_deltas.parquet +
report_data.json (s05), steer_controls.parquet (s08).

Every row has the schema metric, value, ci_lo, ci_hi, n, source_file, scorer, split, emotion, control.
CIs are 95% percentile bootstraps over ACTORS (B resamples of the 91 actors with replacement), so they
reflect speaker variation, not just clip count. No actor ids are written.

Part A  re-checks the claims of the original study (crowd agreement, heard-as-neutral rates, parallel
        deltas in the study's own Praat feature set, perception drivers, final-word concentration,
        actor spread).
Part B  parallel deltas in SteerTTS units: for each emotional clip, the 12 controls minus the mean of
        the same actor's NEUTRAL take(s) of the same sentence, divided by sigma_d = pooled within-actor
        SD of neutral clips (the scale of the paper's speaker-relative z, Eq. z). So d is "how many
        within-speaker neutral SDs the emotion moves control d". d_z = mean/SD of the deltas is also
        given. Deltas are winsorised at the 2nd/98th percentile per emotion x control (as in s05) so
        pitch-tracker octave errors do not drive the mean.

    python s09_paper_stats.py [B=2000]
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from common import OUT, EMOS
from steer.config import CONTROL_DIMS

B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
EMO5 = ["ANG", "HAP", "FEA", "SAD", "DIS"]
rng = np.random.default_rng(0)
ROWS = []


def row(metric, value, ci=(None, None), n=None, src="", scorer="", split="CREMA-D all clips", emotion=None,
        control=None):
    f = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)
    ROWS.append(dict(metric=metric, value=f(value), ci_lo=f(ci[0]), ci_hi=f(ci[1]), n=None if n is None else int(n),
                     source_file=src, scorer=scorer, split=split, emotion=emotion, control=control))


def actor_boot(df, stat, b=B):
    """Percentile CI of stat(df) over actor resamples (df must have an 'actor' column)."""
    groups = {a: g for a, g in df.groupby("actor")}
    actors = np.array(list(groups))
    vals = []
    for _ in range(b):
        pick = rng.choice(actors, len(actors), replace=True)
        vals.append(stat(pd.concat([groups[a] for a in pick], ignore_index=True)))
    return np.nanpercentile(vals, [2.5, 97.5])


def wins(x, lo=0.02, hi=0.98):
    x = x.dropna()
    return x.clip(*x.quantile([lo, hi])) if len(x) > 20 else x


clips = pd.read_parquet(OUT / "clips.parquet")
REP = json.load(open(OUT / "report_data.json"))

# ================================================================ Part A: labels
src = f"{OUT}/clips.parquet (s01, CREMA-D finishedResponses.csv, voice-only queryType=1)"
sc = "crowd majority (voice only)"
f = lambda g: (g.perc_emo == g.emo).mean()
row("crowd_agrees_with_acted", f(clips), actor_boot(clips, f, 500), len(clips), src, sc)
for emo in ["SAD", "HAP", "FEA", "DIS", "ANG"]:
    g = clips[clips.emo == emo]
    f = lambda g: (g.perc_emo == "NEU").mean()
    row("heard_as_neutral", f(g), actor_boot(g, f, 500), len(g), src, sc, emotion=emo)
    f = lambda g: (g.perc_emo == g.emo).mean()
    row("crowd_agrees_with_acted", f(g), actor_boot(g, f, 500), len(g), src, sc, emotion=emo)
hum = clips[clips.perc_emo.isin(EMOS)]
row("human_uar_6class", balanced_accuracy_score(hum.emo, hum.perc_emo), n=len(hum), src=src, scorer=sc,
    split="clips with a unique crowd majority")
emo_clips = clips[clips.emo != "NEU"]
pa = emo_clips.groupby("actor").p_acted.mean()
row("actor_p_acted_max", pa.max(), n=len(pa), src=src, scorer="share of raters choosing the acted emotion",
    split="emotional clips, per-actor mean")
row("actor_p_acted_min", pa.min(), n=len(pa), src=src, scorer="share of raters choosing the acted emotion",
    split="emotional clips, per-actor mean")
row("actor_p_acted_max_over_min", pa.max() / pa.min(), n=len(pa), src=src,
    scorer="share of raters choosing the acted emotion", split="emotional clips, per-actor mean")
row("actor_p_acted_p90_over_p10", pa.quantile(0.9) / pa.quantile(0.1), n=len(pa), src=src,
    scorer="share of raters choosing the acted emotion", split="emotional clips, per-actor mean")

# ================================================================ Part A: study-feature parallel deltas
D = pd.read_parquet(OUT / "deltas.parquet")
src = f"{OUT}/deltas.parquet (s05; Praat features from s02)"
sc = "study Praat features, d_z = mean/SD of delta vs same-actor neutral, same sentence"
STUDY = ["dInt_dB", "dAlpha_dB", "dSlope0_500", "dHamm_dB", "dF1_pct", "dF0_st", "dF0range_st", "dCPPS_dB",
         "dH1H2_dB", "dDur_pct", "dRate_pct", "dIntRange_dB", "dHNR_dB", "dCent_pct"]


def dz(g, c):
    x = wins(g[c])
    return x.mean() / x.std()


for emo in EMO5:
    g = D[D.emo == emo]
    for c in STUDY:
        row("study_delta_dz", dz(g, c), actor_boot(g[["actor", c]], lambda h: dz(h, c), 500), g[c].notna().sum(),
            src, sc, emotion=emo, control=c)
for emo, dd in REP["drivers"]["by_emo"].items():
    row("perception_auc", dd["auc"], n=(D.emo == emo).sum(), src=f"{OUT}/report_data.json drivers (s05)",
        scorer="actor-grouped 5-fold logistic on 12 deltas -> crowd p_acted>=0.5", emotion=emo)
    top = max(dd["coef"].items(), key=lambda kv: abs(kv[1]))
    row("perception_top_coef", top[1], src=f"{OUT}/report_data.json drivers (s05)",
        scorer="standardised logistic coefficient (largest |coef|)", emotion=emo, control=top[0])

# ================================================================ Part A: word position
W = pd.read_parquet(OUT / "word_deltas.parquet")
W["pos"] = np.where(W.w_idx == 0, "first", np.where(W.w_idx == W.groupby("clip").w_idx.transform("max"), "last", "middle"))
src = f"{OUT}/word_deltas.parquet (s05; wav2vec2 CTC word alignment, CER<=0.25)"
for emo in EMO5:
    for pos in ["first", "middle", "last"]:
        g = W[(W.emo == emo) & (W.pos == pos)]
        for c, unit in [("dInt_dB", "peak intensity (dB)"), ("dF0_st", "mean F0 (st)"), ("dDur_pct", "duration (%)")]:
            row("word_delta_median", g[c].median(), actor_boot(g[["actor", c]], lambda h: h[c].median(), 300),
                g[c].notna().sum(), src, f"median word {unit} delta vs same-actor neutral word",
                split=f"{pos} word", emotion=emo, control=c)
    # within-clip last-minus-first, so the CI is on the difference itself
    p = W[W.emo == emo].pivot_table(index=["clip", "actor"], columns="pos", values=["dInt_dB", "dF0_st"]).dropna()
    for c in ["dInt_dB", "dF0_st"]:
        diff = (p[(c, "last")] - p[(c, "first")]).rename("x").reset_index()
        row("word_last_minus_first_median", diff.x.median(), actor_boot(diff, lambda h: h.x.median(), 300), len(diff),
            src, "median within-clip (last word - first word) delta", split="last - first word", emotion=emo, control=c)

# ================================================================ Part B: SteerTTS units
S = pd.read_parquet(OUT / "steer_controls.parquet").merge(clips[["clip", "actor", "sent", "emo", "level"]], on="clip")
neu = S[S.emo == "NEU"]
src = f"{OUT}/steer_controls.parquet (s08: steer.prep.trim_resample + steer.features.measure)"
sc = "SteerTTS extractor; d = mean delta / pooled within-actor neutral SD"


def steer_d(S):
    neu = S[S.emo == "NEU"]
    resid = neu[CONTROL_DIMS] - neu.groupby("actor")[CONTROL_DIMS].transform("mean")
    k = neu.groupby("actor").size()
    sigma = np.sqrt((resid ** 2).sum() / (len(neu) - (k > 0).sum()))
    # S may hold duplicated actors after resampling; reference = same (actor, sentence) neutral mean
    ref = neu.groupby(["actor", "sent"])[CONTROL_DIMS].mean()
    e = S[S.emo != "NEU"].join(ref, on=["actor", "sent"], rsuffix="_ref")
    out = {}
    for emo in EMO5:
        g = e[e.emo == emo]
        for c in CONTROL_DIMS:
            x = wins(g[c] - g[c + "_ref"])
            out[(emo, c)] = (x.mean() / sigma[c], x.mean() / x.std(), x.mean(), len(x))
    return out, sigma


full, sigma = steer_d(S)
# actor bootstrap: relabel resampled actors so duplicates stay distinct speakers
groups = {a: g for a, g in S.groupby("actor")}
actors = np.array(list(groups))
boot = []
for b in range(B):
    pick = rng.choice(actors, len(actors), replace=True)
    Sb = pd.concat([groups[a].assign(actor=i) for i, a in enumerate(pick)], ignore_index=True)
    boot.append(steer_d(Sb)[0])
for (emo, c), (d, d_z, mu, n) in full.items():
    bd = np.array([bb[(emo, c)][0] for bb in boot])
    bz = np.array([bb[(emo, c)][1] for bb in boot])
    row("steer_delta_d", d, np.nanpercentile(bd, [2.5, 97.5]), n, src, sc, emotion=emo, control=c)
    row("steer_delta_dz", d_z, np.nanpercentile(bz, [2.5, 97.5]), n, src,
        "SteerTTS extractor; d_z = mean/SD of delta vs same-actor neutral", emotion=emo, control=c)
    row("steer_delta_raw_mean", mu, n=n, src=src, scorer="SteerTTS extractor; raw units (see Table controls)",
        emotion=emo, control=c)
for c in CONTROL_DIMS:
    row("steer_neutral_sigma", sigma[c], n=len(neu), src=src, scorer="pooled within-actor SD of neutral clips",
        split="NEU clips", control=c)

# per-actor spread of the anger loudness delta (same instruction, different absolute change)
ref = neu.groupby(["actor", "sent"])[["loud"]].mean()
e = S[S.emo == "ANG"].join(ref, on=["actor", "sent"], rsuffix="_ref")
pa = (e.loud - e.loud_ref).groupby(e.actor).mean()
row("actor_ang_loud_delta_p90_over_p10", pa.quantile(0.9) / pa.quantile(0.1), n=len(pa), src=src,
    scorer="per-actor mean ANG loudness delta (dB), 90th/10th percentile ratio over actors", emotion="ANG",
    control="loud")
row("actor_ang_loud_delta_p10_dB", pa.quantile(0.1), n=len(pa), src=src, scorer="per-actor mean ANG loudness delta (dB)",
    emotion="ANG", control="loud")
row("actor_ang_loud_delta_p90_dB", pa.quantile(0.9), n=len(pa), src=src, scorer="per-actor mean ANG loudness delta (dB)",
    emotion="ANG", control="loud")

json.dump(ROWS, open(OUT / "paper_stats.json", "w"), indent=1)
R = pd.DataFrame(ROWS)
pd.set_option("display.width", 200)
print(R[R.metric.isin(["crowd_agrees_with_acted", "heard_as_neutral", "human_uar_6class"]) | R.metric.str.startswith("actor")]
      [["metric", "emotion", "control", "value", "ci_lo", "ci_hi", "n"]].to_string())
for m in ["study_delta_dz", "steer_delta_d"]:
    print(f"\n== {m}\n", R[R.metric == m].pivot(index="control", columns="emotion", values="value")[EMO5].round(2))
print(R[R.metric.str.startswith(("word_", "perception"))][["metric", "split", "emotion", "control", "value", "ci_lo", "ci_hi"]].to_string())
print("wrote", OUT / "paper_stats.json", len(ROWS), "rows")
