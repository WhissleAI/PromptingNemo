"""Manifest -> per-utterance feature npz + raw control table.

python -m steer.extract ears [nproc]
data/feats/<id>.npz: mel f16 [100,T], f0 f32 [T] (Hz, 0=unvoiced), energy f32 [T] (dB), ids i16, widx i16
data/meta/raw_controls_<corpus>.parquet: id + 12 raw control measurements + n_frames
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .config import FEATS, META, CONTROL_DIMS
from .features import measure
from .text import to_phonemes, SP, SYMBOLS

torch.set_num_threads(1)
FEAT_VERSION = 2  # bump when features.py changes so stale npz files are recomputed
_PUNCT_IDS = {i for i, s in enumerate(SYMBOLS) if len(s) == 1}


def work(row):
    out = FEATS / f"{row['id']}.npz"
    try:
        ids, widx, _ = to_phonemes(row["text"])
        n_ph = sum(1 for i in ids if i != SP and i not in _PUNCT_IDS)
        if out.exists() and int(np.load(out).get("ver", 0)) == FEAT_VERSION:
            z = np.load(out)
            return dict(id=row["id"], n_frames=int(z["mel"].shape[1]), **dict(zip(CONTROL_DIMS, z["controls"].tolist())))
        y, sr = sf.read(row["wav"], dtype="float32")
        m = measure(y, n_phones=n_ph)
        np.savez(out, ver=FEAT_VERSION, mel=m["mel"].astype(np.float16), f0=m["f0"], energy=m["energy"],
                 ids=np.array(ids, np.int16), widx=np.array(widx, np.int16), controls=m["controls"])
        return dict(id=row["id"], n_frames=int(m["mel"].shape[1]), **dict(zip(CONTROL_DIMS, m["controls"].tolist())))
    except Exception as e:  # keep going; report failures
        return dict(id=row["id"], error=repr(e)[:200])


def main():
    corpus = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rows = [json.loads(l) for l in open(META / f"manifest_{corpus}.jsonl")]
    with ProcessPoolExecutor(nproc) as ex:
        res = list(ex.map(work, rows, chunksize=4))
    df = pd.DataFrame(res)
    err = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    print(corpus, len(df), "done,", len(err), "errors")
    if len(err):
        print(err.head().to_string())
    df = df[df.get("error").isna()] if "error" in df else df
    df.drop(columns=["error"], errors="ignore").to_parquet(META / f"raw_controls_{corpus}.parquet", index=False)
    print(df[CONTROL_DIMS].describe().T.round(3))


if __name__ == "__main__":
    main()
