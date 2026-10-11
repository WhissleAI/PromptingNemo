"""Reviewer follow-up (a): DSP post-processing baseline on the exact X1 protocol.
Produced: results/steer_dsp_baseline.json + _raw.csv: python -m steer.paper.dsp_baseline --stage make; --stage gpu; --stage analyze (queue job 06a_x1_dsp_gpu.sh runs gpu+analyze). All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.

Input = our UNINSTRUCTED X1 renders (16 held-out voices x 8 texts). For a request of z on a dim, a classic DSP edit is
applied with its physical size set from the control's training pooled sigma (the units the model is conditioned in):
  rate    Praat overlap-add (PSOLA) time-scale: duration x r0 / (r0 + z*sd_rate), r0 = the render's measured rate
  f0_med  Praat PSOLA pitch shift of z*sd_f0med semitones (duration preserved)
  loud    gain of z*sd_loud dB
  tilt    smooth high shelf (logistic around 1 kHz, STFT domain) of z*sd_tilt dB above 1 kHz
Then the X1 readback (both extractors), cross-talk and WER (Whisper-large-v3), plus UTMOS for DSP vs our renders.

python -m steer.paper.dsp_baseline --stage make     (CPU: DSP + measure)
python -m steer.paper.dsp_baseline --stage gpu      (Whisper + UTMOS; via the GPU queue)
python -m steer.paper.dsp_baseline --stage analyze  -> results/steer_dsp_baseline.json (+ _raw.csv)
"""
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import soundfile as sf

from ..config import CONTROL_DIMS, SR
from . import common as C
from . import x1_steer_heldout as X1

OUT = C.WORK / "dsp"
DIMS = ["rate", "f0_med", "loud", "tilt"]


def dsp(y, dim, z, sd, base_ctrl):
    import parselmouth
    from parselmouth.praat import call
    y = np.asarray(y, np.float64)
    if dim == "loud":
        return y * 10 ** (z * sd["loud"] / 20)
    if dim == "tilt":
        import torch
        x = torch.from_numpy(y)
        X = torch.stft(x, 1024, 256, 1024, torch.hann_window(1024, dtype=torch.float64), return_complex=True)
        f = torch.fft.rfftfreq(1024, 1 / SR, dtype=torch.float64)
        g = 10 ** (z * sd["tilt"] / 20 / (1 + torch.exp(-(f - 1000) / 150)))  # 0 dB below ~1 kHz, full gain above
        return torch.istft(X * g[:, None], 1024, 256, 1024, torch.hann_window(1024, dtype=torch.float64), length=len(y)).numpy()
    snd = parselmouth.Sound(y, SR)
    if dim == "rate":
        r0 = base_ctrl["rate"]
        factor = r0 / max(r0 + z * sd["rate"], 0.3 * r0)  # duration factor
        out = call(snd, "Lengthen (overlap-add)", 75, 600, factor)
        return out.values[0]
    if dim == "f0_med":
        man = call(snd, "To Manipulation", 0.01, 75, 600)
        pt = call(man, "Extract pitch tier")
        call(pt, "Shift frequencies", snd.xmin, snd.xmax, z * sd["f0_med"], "semitones")
        call([pt, man], "Replace pitch tier")
        return call(man, "Get resynthesis (overlap-add)").values[0]
    raise ValueError(dim)


def _job(a):
    path, out, dim, z, sd, bc = a
    y, _ = sf.read(path, dtype="float64")
    w = dsp(y, dim, z, sd, bc)
    peak = np.abs(w).max()
    if peak > 0.999:  # no clipping (loud +2 sigma on hot renders): note it, scale is part of the readback
        w = w / peak * 0.999
    sf.write(out, w.astype(np.float32), SR)
    return out, float(peak)


def plan(V):
    items = []
    for it in X1.plan(V):
        if it["dim"] == "baseline":
            items.append(dict(it))
        elif it["dim"] in DIMS:
            d = dict(it)
            d["src"] = str(X1.OUT / "wav" / f"{it['vid']}_{it['text_id']}_baseline_+0.wav")
            d["ours_path"] = it["path"]
            d["path"] = str(OUT / "wav" / f"{it['vid']}_{it['text_id']}_{it['dim']}_{it['req_z']:+.0f}.wav")
            items.append(d)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="make")
    a = ap.parse_args()
    V = C.heldout_voices(16)
    items = plan(V)
    OUT.mkdir(parents=True, exist_ok=True)
    stats = json.load(open(C.BASE.rsplit("/", 1)[0] + "/control_stats.json"))
    sd = {k: float(stats["pooled_sd"][k]) for k in DIMS}
    if a.stage == "make":
        (OUT / "wav").mkdir(parents=True, exist_ok=True)
        M1 = pd.read_parquet(X1.OUT / "measures.parquet").set_index("key")
        jobs = [(it["src"], it["path"], it["dim"], it["req_z"], sd, {"rate": float(M1.loc[it["src"], "rate"])})
                for it in items if it["dim"] != "baseline"]
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(32) as pool:
            res = pool.map(_job, jobs, chunksize=8)
        json.dump(dict(res), open(OUT / "peaks.json", "w"))
        M = C.measure_many([(it["path"], it["path"], it["text"]) for it in items if it["dim"] != "baseline"])
        M = pd.concat([M, M1.reset_index()[M1.index.str.contains("_baseline_")]], ignore_index=True)
        M.to_parquet(OUT / "measures.parquet")
        for f in ("spk_mu.csv", "pooled_sd_gt.csv"):
            shutil.copy(X1.OUT / f, OUT / f)
        print("made", len(jobs), flush=True)
    if a.stage == "gpu":
        paths = [it["path"] for it in items]
        json.dump(C.transcribe(paths, X1.OUT / "asr_cache.jsonl"), open(OUT / "hyps.json", "w"))
        from .x4_quality_rescore import utmos
        utmos(paths + [it["ours_path"] for it in items if "ours_path" in it], OUT / "utmos.json")
    if a.stage == "analyze":
        analyze(V, items)


def analyze(V, items):
    X1.OUT = OUT
    X1.NAME = "steer_dsp_baseline"
    rows_out = {}
    orig = C.write

    def capture(name, R, pub, meta=None):
        rows_out.update(name=name, R=R, pub=pub, meta=meta)
    C.write = capture
    X1.analyze(V, items)
    C.write = orig
    R, pub, meta = rows_out["R"], rows_out["pub"], rows_out["meta"]
    for r in R:
        r["system"] = "dsp_postprocess"
    ut = json.load(open(OUT / "utmos.json"))
    df = pd.DataFrame(items)
    df["utmos_dsp"] = df.path.map(ut)
    df["utmos_ours"] = df.get("ours_path", pd.Series(dtype=str)).map(ut)
    src = f"{C.RES}/steer_dsp_baseline_raw.csv"
    split = "libritts_r_test_clean_heldout16"
    b = df[df.dim == "baseline"]
    R.append(C.row("utmos", C.boot_ratio(b, ["vid", "text_id"], "utmos_dsp"), len(b), src, "SpeechMOS utmos22_strong",
                   split, condition="baseline", system="ours_uninstructed"))
    for d in DIMS:
        for z in X1.ZS:
            s = df[(df.dim == d) & (df.req_z == z)]
            for sysn, col in (("dsp_postprocess", "utmos_dsp"), ("ours", "utmos_ours")):
                R.append(C.row("utmos", C.boot_ratio(s, ["vid", "text_id"], col), len(s), src, "SpeechMOS utmos22_strong",
                               split, condition="single_control", system=sysn, target_dim=d, req_z=z))
    pub = pub.merge(df[["vid", "text_id", "dim", "req_z", "utmos_dsp", "utmos_ours"]], on=["vid", "text_id", "dim", "req_z"], how="left")
    meta = dict(meta, dsp_ops=__doc__.split("Then")[0], pooled_sd=json.load(open(C.BASE.rsplit("/", 1)[0] + "/control_stats.json"))["pooled_sd"],
                peaks_clipped=sum(v > 0.999 for v in json.load(open(OUT / "peaks.json")).values()),
                note="Compare with results/steer_heldout.json (ours, same voices/texts/requests/extractors).",
                dnsmos="not computed (no DNSMOS model on disk); UTMOS only")
    C.write("steer_dsp_baseline", R, pub, meta)


if __name__ == "__main__":
    main()
