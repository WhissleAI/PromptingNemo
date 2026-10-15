"""Reviewer follow-up (c): intelligibility cost of steering on more text.
Produced: results/wer_cost_extended.json + _raw.csv: python -m steer.paper.wer_extended --stage render; --stage asr; --stage analyze (queue job 06d_x2_wer_ext.sh). All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.

64 LibriTTS-R test-clean sentences (6-25 words) spoken by test-clean speakers OTHER than the 16 held-out voices, x 16
held-out voices x {uninstructed, rate/loud/f0_med/f0_iqr/tilt/h1h2 at z = +2 and -2} = 13,312 renders (base 312.5k,
guidance 1). Whisper-large-v3 fp16 + Whisper EnglishTextNormalizer. CIs: 1000-resample cluster bootstrap over voices
and over (voice, sentence) cells.

python -m steer.paper.wer_extended --stage render|asr|analyze  -> results/wer_cost_extended.json (+ _raw.csv)
"""
import argparse
import json

import numpy as np
import pandas as pd
import soundfile as sf

from ..config import SR
from . import common as C

NAME = "wer_cost_extended"
OUT = C.WORK / "wer_ext"
DIMS = ["rate", "loud", "f0_med", "f0_iqr", "tilt", "h1h2"]


def sentences(V, n=64):
    used = {v["speaker"] for v in V}
    rows = [json.loads(l) for l in open(C.BENCH)]
    rows = [r for r in rows if r["speaker"] not in used and 6 <= len(r["text"].split()) <= 25]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), n, replace=False)
    return [rows[i]["text"] for i in sorted(pick)]


def plan(V):
    S = sentences(V)
    items = []
    for v in V:
        for si, t in enumerate(S):
            conds = [("baseline", 0.0)] + [(d, z) for d in DIMS for z in (2.0, -2.0)]
            for d, z in conds:
                items.append(dict(vid=v["vid"], sent_id=si, text=t, dim=d, req_z=z,
                                  path=str(OUT / "wav" / f"{v['vid']}_{si:02d}_{d}_{z:+.0f}.wav")))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="render")
    a = ap.parse_args()
    V = C.heldout_voices(16)
    items = plan(V)
    ref = {v["vid"]: v["ref"] for v in V}
    if a.stage == "render":
        C.gpu_wait()
        (OUT / "wav").mkdir(parents=True, exist_ok=True)
        syn = C.make_synth(C.BASE)
        for i, it in enumerate(items):
            if C.Path(it["path"]).exists():
                continue
            ctrl = None if it["dim"] == "baseline" else {it["dim"]: it["req_z"]}
            sf.write(it["path"], syn(it["text"], f"heldout_{it['vid']}", ctrl=ctrl, ref_wav=ref[it["vid"]]), SR)
            if i % 1000 == 0:
                print(f"render {i}/{len(items)}", flush=True)
    if a.stage == "asr":
        json.dump(C.transcribe([it["path"] for it in items], OUT / "asr_cache.jsonl"), open(OUT / "hyps.json", "w"))
    if a.stage == "analyze":
        hyps = json.load(open(OUT / "hyps.json"))
        df = pd.DataFrame(items)
        df["hyp"] = df.path.map(hyps)
        ew = [C.word_errors(t, h) for t, h in zip(df.text, df.hyp)]
        df["err"], df["nwords"] = [e for e, _ in ew], [n for _, n in ew]
        base = df[df.dim == "baseline"].set_index(["vid", "sent_id"])
        idx = pd.MultiIndex.from_arrays([df.vid, df.sent_id])
        df["base_err"] = base.err.reindex(idx).values
        df["d_err"] = df.err - df.base_err
        src = f"{C.RES}/{NAME}_raw.csv"
        split = "libritts_r_test_clean_heldout16_x64sent"
        asr = "whisper-large-v3 fp16 + whisper EnglishTextNormalizer"
        R = []
        for cl, U in (("voice", ["vid"]), ("cell", ["vid", "sent_id"])):
            for (d, z), s in df.groupby(["dim", "req_z"], sort=False):
                cond = dict(condition=d if d == "baseline" else "single_control", target_dim=None if d == "baseline" else d,
                            req_z=None if d == "baseline" else z, bootstrap_cluster=cl)
                R.append(C.row("wer", C.boot_ratio(s, U, "err", "nwords"), len(s), src, asr, split, **cond))
                if d != "baseline":
                    R.append(C.row("wer_minus_baseline", C.boot_ratio(s, U, "d_err", "nwords"), len(s), src, asr, split, **cond))
            s = df[df.dim != "baseline"]
            R.append(C.row("wer", C.boot_ratio(s, U, "err", "nwords"), len(s), src, asr, split, condition="all_steered_abs_z2",
                           bootstrap_cluster=cl))
            R.append(C.row("wer_minus_baseline", C.boot_ratio(s, U, "d_err", "nwords"), len(s), src, asr, split,
                           condition="all_steered_abs_z2", bootstrap_cluster=cl))
        empty = int((df.hyp.fillna("") == "").sum())
        pub = df.drop(columns=["path"]).assign(wav=df.path.str.replace(str(C.ROOT) + "/", "", regex=False))
        C.write(NAME, R, pub, dict(ckpt=C.BASE, n_sentences=64, voices=16, dims=DIMS, empty_transcripts=empty,
                                   sentences="LibriTTS-R test-clean transcripts of speakers not among the 16 voices"))


if __name__ == "__main__":
    main()
