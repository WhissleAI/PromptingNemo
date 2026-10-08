"""Clip table: acted label, demographics, and voice-only crowd soft labels.

Per-rater voice-only responses (queryType == 1) give, per clip:
  p_<EMO>       share of raters choosing each emotion (soft label)
  n_raters, agreement (top share), entropy (bits)
  perc_emo      crowd majority (ties -> 'MIX')
  perc_int      mean rated intensity (0-100) of raters who picked the acted emotion
  perc_int_all  mean rated intensity over all raters
"""
import numpy as np
import pandas as pd
from common import CREMA, OUT, EMOS, EMO_LETTER, SENTENCES

clips = pd.read_csv(CREMA / "SentenceFilenames.csv")["Filename"].rename("clip").to_frame()
parts = clips["clip"].str.split("_", expand=True)
clips["actor"] = parts[0].astype(int)
clips["sent"] = parts[1]
clips["emo"] = parts[2]
clips["level"] = parts[3]
clips["text"] = clips.sent.map(SENTENCES)

demo = pd.read_csv(CREMA / "VideoDemographics.csv").rename(columns=str.lower)
clips = clips.merge(demo.rename(columns={"actorid": "actor"}), on="actor", how="left")

r = pd.read_csv(CREMA / "finishedResponses.csv", usecols=["clipName", "queryType", "respEmo", "respLevel"])
r = r[r.queryType == 1].copy()
r["resp"] = r.respEmo.map(EMO_LETTER)
r["lvl"] = pd.to_numeric(r.respLevel, errors="coerce")
r = r.dropna(subset=["resp"])

counts = pd.crosstab(r.clipName, r.resp).reindex(columns=EMOS, fill_value=0)
n = counts.sum(1)
p = counts.div(n, axis=0)
soft = p.add_prefix("p_")
soft["n_raters"] = n
soft["agreement"] = p.max(1)
soft["entropy"] = -(p * np.log2(p.where(p > 0, 1))).sum(1)
top = p.eq(p.max(1), axis=0)
soft["perc_emo"] = np.where(top.sum(1) == 1, p.idxmax(1), "MIX")
soft["perc_int_all"] = r.groupby("clipName").lvl.mean()

clips = clips.merge(soft, left_on="clip", right_index=True, how="left")
r = r.merge(clips[["clip", "emo"]], left_on="clipName", right_on="clip")
clips = clips.merge(
    r[r.resp == r.emo].groupby("clip").lvl.mean().rename("perc_int"),
    left_on="clip", right_index=True, how="left")
clips["p_acted"] = clips.apply(lambda x: x[f"p_{x.emo}"], axis=1)

clips.to_parquet(OUT / "clips.parquet", index=False)
print(clips.shape)
print(clips.groupby("emo")[["p_acted", "agreement", "entropy", "perc_int", "n_raters"]].mean().round(3))
print("perceived == acted:", (clips.perc_emo == clips.emo).mean().round(3))
print(pd.crosstab(clips.emo, clips.perc_emo, normalize="index").round(2))
print(clips[clips.level != "XX"].groupby(["emo", "level"]).perc_int.mean().unstack().round(1))
