"""Raw control measurements -> speaker-relative z-scores + speaker reference stats.

python -m steer.build_controls ears expresso libritts
For each speaker, the reference is its neutral speech (EARS 'regular', Expresso 'default',
LibriTTS: all of it). z = (x - speaker_neutral_mean) / pooled_within-speaker neutral std.
Writes data/meta/controls.parquet and data/meta/control_stats.json.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .config import FEATS, META, CONTROL_DIMS

MAX_SEC = 20.0
MIN_CPS, MAX_CPS = 3.0, 40.0  # transcript characters per second of audio
MAX_TOK_PER_FRAME = 0.6  # normal speech is ~0.16 phone tokens per mel frame


def main():
    corpora = sys.argv[1:] or ["ears", "expresso", "libritts"]
    frames = []
    for c in corpora:
        man = pd.read_json(META / f"manifest_{c}.jsonl", lines=True)
        raw = pd.read_parquet(META / f"raw_controls_{c}.parquet")
        frames.append(man.merge(raw, on="id"))
    df = pd.concat(frames, ignore_index=True)
    # ids must be unique: duplicates (Expresso reuses a few) share one wav/feature file, so their labels are ambiguous
    dup = df.id.duplicated(keep=False)
    print(f"dropping {int(dup.sum())} rows with duplicated ids:", df[dup].groupby("corpus").size().to_dict())
    df = df[~dup].copy()
    df = df[(df.dur <= MAX_SEC) & (df.dur >= 0.8)].copy()
    # transcript/audio mismatches (e.g. Expresso longform/singing rows carrying another clip's text):
    # real speech here spans ~4-35 chars/s; outliers are wrong labels and blow up text-side batch memory
    cps = df.text.str.len() / df.dur
    bad = (cps > MAX_CPS) | (cps < MIN_CPS)
    print(f"dropping {int(bad.sum())} utts with implausible chars/s:", df[bad].groupby("corpus").size().to_dict())
    df = df[~bad].copy()
    # the feature file must hold a plausible number of tokens for its audio (the aligner needs >= 1 frame per token)
    with ThreadPoolExecutor(32) as ex:
        n_tok = np.array(list(ex.map(lambda i: len(np.load(FEATS / f"{i}.npz")["ids"]), df.id)))
    bad = n_tok > MAX_TOK_PER_FRAME * df.n_frames.to_numpy()
    print(f"dropping {int(bad.sum())} utts with more tokens than frames allow:", df[bad].groupby("corpus").size().to_dict())
    df = df[~bad].copy()
    # pitch measurements are meaningless without voicing (whisper): mask them
    unv = df.voiced < 0.2
    df.loc[unv, ["f0_med", "f0_iqr", "f0_end", "h1h2"]] = np.nan

    neu = df[(df.tag == "neutral") | (df.corpus == "libritts")]
    spk_mean = neu.groupby("speaker")[CONTROL_DIMS].mean()
    # speakers with no neutral speech fall back to their overall mean
    missing = set(df.speaker) - set(spk_mean.index)
    if missing:
        spk_mean = pd.concat([spk_mean, df[df.speaker.isin(missing)].groupby("speaker")[CONTROL_DIMS].mean()])
    resid = neu[CONTROL_DIMS] - spk_mean.loc[neu.speaker].values
    pooled_sd = resid.std()
    # robust floor so near-constant dims don't explode
    pooled_sd = pooled_sd.clip(lower=pooled_sd.median() * 0.05)

    Z = (df[CONTROL_DIMS] - spk_mean.loc[df.speaker].values) / pooled_sd.values
    Z = Z.clip(-5, 5)
    out = df[["id", "corpus", "speaker", "tag", "raw_style", "text", "dur", "n_frames"]].copy()
    for k in CONTROL_DIMS:
        out[f"z_{k}"] = Z[k].astype("float32")
    # speaker reference for frame-level pitch/energy normalisation (f0_med is semitones re 100 Hz)
    out["spk_f0_st"] = spk_mean.loc[df.speaker, "f0_med"].values.astype("float32")
    out["spk_loud"] = spk_mean.loc[df.speaker, "loud"].values.astype("float32")
    out.to_parquet(META / "controls.parquet", index=False)

    speakers = sorted(out.speaker.unique())
    tags = ["none"] + sorted(t for t in out.tag.unique() if t != "none")
    stats = dict(pooled_sd=pooled_sd.round(5).to_dict(), speakers=speakers, tags=tags,
                 spk_mean={s: spk_mean.loc[s].round(4).to_dict() for s in speakers})
    json.dump(stats, open(META / "control_stats.json", "w"), indent=1)
    print(len(out), "utts,", len(speakers), "speakers,", len(tags), "tags,", round(out.dur.sum() / 3600, 1), "h")
    print("pooled within-speaker neutral sd:\n", pooled_sd.round(3).to_string())
    print(out.groupby("tag")[[f"z_{k}" for k in CONTROL_DIMS]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
