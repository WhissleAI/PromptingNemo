"""Compact data + example audio for the visual report -> viz/ (viz_data.js, audio/*.wav).

Example sets: for one male and one female actor, the sentence whose six takes
(5 emotions + neutral) listeners recognised best (max of the minimum p_acted
across emotions; IEO excluded since its emotions only exist at LO/MD/HI).
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT, FRAMES, WAV, EMOS, SENTENCES

VIZ = Path(__file__).resolve().parent / "viz"
(VIZ / "audio").mkdir(parents=True, exist_ok=True)

R = json.load(open(OUT / "report_data.json"))
clips = pd.read_parquet(OUT / "clips.parquet")
kc = pd.read_parquet(OUT / "align_clip.parquet")
words = pd.read_parquet(OUT / "align_words.parquet")
clips = clips.merge(kc[["clip", "cer"]], on="clip")

xx = clips[(clips.level == "XX") & (clips.sent != "IEO")]
g = xx.groupby(["actor", "sex", "sent"]).agg(n=("emo", "size"), worst=("p_acted", "min"),
                                              mean=("p_acted", "mean"), cer=("cer", "max")).reset_index()
g = g[(g.n == 6) & (g.cer <= 0.1)].sort_values(["worst", "mean"], ascending=False)
picks = [g[g.sex == "Male"].iloc[0], g[g.sex == "Female"].iloc[0]]

# same per-actor neutral references as s05: median neutral F0, mean neutral intensity
pros = pd.read_parquet(OUT / "prosody.parquet").merge(clips[["clip", "actor", "emo"]])
neu_f0 = pros[pros.emo == "NEU"].groupby("actor").f0_med.median()
neu_int = pros[pros.emo == "NEU"].groupby("actor").int_mean.mean()

examples = []
for p in picks:
    takes = {}
    for emo in EMOS:
        row = xx[(xx.actor == p.actor) & (xx.sent == p.sent) & (xx.emo == emo)].iloc[0]
        fr = np.load(FRAMES / f"{row['clip']}.npz")
        f0 = 12 * np.log2(fr["f0"] / neu_f0[p.actor])
        I = fr["int"] - neu_int[p.actor]
        wd = words[words["clip"] == row["clip"]].sort_values("w_idx")
        shutil.copy(WAV / f"{row['clip']}.wav", VIZ / "audio" / f"{row['clip']}.wav")
        takes[emo] = dict(
            clip=row["clip"], t0=0.0, dt=0.01,
            f0=[None if not np.isfinite(v) else round(float(v), 2) for v in f0],
            int=[None if not np.isfinite(v) else round(float(v), 1) for v in I],
            words=[dict(w=w.word.lower(), s=round(w.start, 2), e=round(w.end, 2)) for w in wd.itertuples()],
            heard={e: round(float(row[f"p_{e}"]), 2) for e in EMOS}, n=int(row.n_raters),
            intensity=None if pd.isna(row.perc_int) else round(float(row.perc_int)),
        )
    examples.append(dict(actor=int(p.actor), sex=p.sex, age=int(clips[clips.actor == p.actor].age.iat[0]),
                         sent=p.sent, text=SENTENCES[p.sent], takes=takes))
    print(p.actor, p.sex, p.sent, round(p.worst, 2), round(p["mean"], 2))

R["examples"] = examples
(VIZ / "viz_data.js").write_text("window.VIZ = " + json.dumps(R, separators=(",", ":")) + ";\n")
print("wrote", VIZ / "viz_data.js", (VIZ / "viz_data.js").stat().st_size // 1024, "KB")
