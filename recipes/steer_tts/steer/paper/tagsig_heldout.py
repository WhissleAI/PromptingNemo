"""Per-tag acoustic signature on HELD-OUT voices from X3's renders/measures (steer.paper.x3_tags_heldout; read-only):
mean speaker-relative delta z of the 12 controls (ours extractor) of each tag render vs the uninstructed render of
the same voice + sentence, with a cluster bootstrap over voice x sentence. -> results/tagsig_heldout.json/_raw.csv
python -m steer.paper.tagsig_heldout"""
import json

import numpy as np
import pandas as pd

from ..config import CONTROL_DIMS
from . import common as C
from . import x3_tags_heldout as X


def main():
    stats = json.load(open(C.BASE.rsplit("/", 1)[0] + "/control_stats.json"))
    sd = pd.Series(stats["pooled_sd"])[CONTROL_DIMS]
    M = pd.read_parquet(X.OUT / "measures.parquet").drop_duplicates("key").set_index("key")
    items = X.plan(C.heldout_voices(16))
    df = pd.DataFrame(items).join(M, on="path")
    base = df[df.tag == "baseline"].set_index(["vid", "text_id"])
    idx = pd.MultiIndex.from_arrays([df.vid, df.text_id])
    for d in CONTROL_DIMS:
        df[f"dz_{d}"] = (df[d].values - base[d].reindex(idx).values) / sd[d]
    R = []
    rng = np.random.default_rng(0)
    src, split = f"{C.RES}/tagsig_heldout_raw.csv", "libritts_r_test_clean_heldout16"
    for t, g in df[df.tag != "baseline"].groupby("tag"):
        # cluster bootstrap over voice x sentence (one render per unit per tag): resample rows
        D = g[[f"dz_{d}" for d in CONTROL_DIMS]].to_numpy(float)
        idx = rng.integers(0, len(D), (1000, len(D)))
        B = np.nanmean(D[idx], axis=1)  # [1000, 12]
        mu = np.nanmean(D, 0)
        for k, d in enumerate(CONTROL_DIMS):
            R.append(C.row("mean_delta_z", (float(mu[k]), float(np.nanpercentile(B[:, k], 2.5)),
                                            float(np.nanpercentile(B[:, k], 97.5))), len(g), src,
                           "steer.features.measure", split, tag=t, measured_dim=d))
        R.append(C.row("l2_mean_delta_z", float(np.sqrt(np.nansum(mu ** 2))), len(g), src, "steer.features.measure",
                       split, tag=t))
    # relative to the 'neutral' tag: what a tag adds beyond the common "any tag vs none" shift
    mu = df[df.tag != "baseline"].groupby("tag")[[f"dz_{d}" for d in CONTROL_DIMS]].mean()
    rel = mu.sub(mu.loc["neutral"])
    for t in mu.index:
        R.append(C.row("l2_delta_z_vs_neutral_tag", float(np.sqrt((rel.loc[t] ** 2).sum())), len(df[df.tag == t]), src,
                       "steer.features.measure", split, tag=t))
    for d in CONTROL_DIMS:
        R.append(C.row("median_over_tags_mean_delta_z", float(mu.drop(index=["whisper"])[f"dz_{d}"].median()), len(mu) - 1,
                       src, "steer.features.measure", split, measured_dim=d, note="excl. whisper"))
    pub = df[["vid", "text_id", "tag"] + [f"dz_{d}" for d in CONTROL_DIMS]]
    C.write("tagsig_heldout", R, pub, dict(ckpt="runs/spk_enc_v3/step312500.pt", source="X3 renders (x3_tags_heldout), "
                                           "16 held-out voices x 8 texts, no guidance; measures by steer.features.measure"))


if __name__ == "__main__":
    main()
