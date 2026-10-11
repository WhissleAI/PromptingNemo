"""The 12 SteerTTS controls (steer.features.measure) for every CREMA-D clip (CPU, parallel).

Audio goes through exactly the SteerTTS data path: steer.prep.trim_resample (16 kHz -> 24 kHz,
energy trim with 120 ms pad) and steer.features.measure with the phone count from
steer.text.to_phonemes, as in steer.extract. So the units are the ones the TTS is conditioned on.

    PYTHONPATH=<recipes/steer_tts> python s08_steer_controls.py [nproc]
    -> $EMOTION_STUDY_OUT/steer_controls.parquet  (clip + 12 raw controls in CONTROL_DIMS order)
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from common import WAV, OUT, SENTENCES

from steer.config import CONTROL_DIMS
from steer.features import measure
from steer.prep import trim_resample
from steer.text import to_phonemes, SP, SYMBOLS

torch.set_num_threads(1)
_PUNCT = {i for i, s in enumerate(SYMBOLS) if len(s) == 1}


def n_phones(text):
    ids, _, _ = to_phonemes(text)
    return sum(1 for i in ids if i != SP and i not in _PUNCT)


NPH = {k: n_phones(v) for k, v in SENTENCES.items()}


def work(clip):
    try:
        y, sr = sf.read(WAV / f"{clip}.wav", dtype="float32")
        y = trim_resample(y, sr)
        c = measure(y, n_phones=NPH[clip.split("_")[1]], want_mel=False)["controls"]
        return dict(clip=clip, **dict(zip(CONTROL_DIMS, c.tolist())))
    except Exception as e:  # keep going; report failures
        return dict(clip=clip, error=repr(e)[:200])


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    clips = pd.read_parquet(OUT / "clips.parquet")["clip"].tolist()
    with ProcessPoolExecutor(nproc) as ex:
        rows = list(ex.map(work, clips, chunksize=16))
    df = pd.DataFrame(rows)
    if "error" in df:
        err = df[df.error.notna()]
        print(len(err), "errors"); print(err.head().to_string())
        df = df[df.error.isna()].drop(columns="error")
    df.to_parquet(OUT / "steer_controls.parquet", index=False)
    print(len(df), "clips; phones per sentence", NPH)
    print(df[CONTROL_DIMS].describe().T.round(3))


if __name__ == "__main__":
    main()
