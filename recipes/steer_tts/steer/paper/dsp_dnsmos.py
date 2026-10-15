"""DNSMOS (and UTMOSv2 when its cache exists) for model-steered vs DSP-post-processed renders (independent of UTMOS22,
which was a SteerTTS training reward).

Input: results/steer_dsp_baseline_raw.csv (X1 protocol: 16 held-out voices x 8 texts; per dim x z: `ours_path` = the
model's directed render, `wav` = the DSP edit of the uninstructed render; baseline rows: `wav` = uninstructed render).
    nice -n 19 python -m steer.paper.dsp_dnsmos [--stage score|analyze|all]     (emotts env; DNSMOS runs in tb_score, 8 CPU procs)
Output: results/steer_dsp_dnsmos.json (+ _raw.csv): per dim x z, mean DNSMOS OVRL/SIG/BAK/P808 for model and DSP and
the paired difference model - DSP, 95% CIs from 1000 cluster resamples over voices (16 clusters).
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C

SRC = C.RES / "steer_dsp_baseline_raw.csv"
WORK = C.WORK / "dsp_dnsmos"
TB = Path.home() / "anaconda3/envs/tb_score/bin/python"
KEYS = ("ovrl", "sig", "bak", "p808")


def absp(p):
    p = str(p)
    return p if p.startswith("/") else str(C.ROOT / p)


def frame():
    d = pd.read_csv(SRC)
    d["dsp_wav"] = [absp(w) if dm != "baseline" else None for w, dm in zip(d.wav, d.dim)]
    d["model_wav"] = [absp(o) if dm != "baseline" else absp(w) for o, w, dm in zip(d.ours_path, d.wav, d.dim)]
    return d


def score(d):
    WORK.mkdir(parents=True, exist_ok=True)
    paths = sorted(set(d.model_wav.dropna()) | set(d.dsp_wav.dropna()))
    missing = [p for p in paths if not Path(p).exists()]
    assert not missing, f"{len(missing)} missing, e.g. {missing[:3]}"
    (WORK / "list.txt").write_text("\n".join(paths) + "\n")
    subprocess.run([str(TB), "-s", str(Path(__file__).with_name("dnsmos_score.py")), str(WORK / "list.txt"),
                    str(WORK / "dnsmos.json"), "8"], check=True,
                   env={"PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "OMP_NUM_THREADS": "1"})


def cluster_ci(vals_by_voice, B=1000, seed=0):
    """vals_by_voice: list of 1-d arrays (one per voice). Mean over all units; resample voices."""
    rng = np.random.default_rng(seed)
    allv = np.concatenate(vals_by_voice)
    v = float(np.mean(allv))
    bs = []
    for _ in range(B):
        idx = rng.integers(0, len(vals_by_voice), len(vals_by_voice))
        bs.append(np.mean(np.concatenate([vals_by_voice[i] for i in idx])))
    return v, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyze(d):
    dn = json.load(open(WORK / "dnsmos.json"))
    ut2 = json.load(open(WORK / "utmosv2.json")) if (WORK / "utmosv2.json").exists() else {}
    for k in KEYS:
        d[f"model_{k}"] = d.model_wav.map(lambda p: dn.get(p, {}).get(k, np.nan))
        d[f"dsp_{k}"] = d.dsp_wav.map(lambda p: dn.get(p, {}).get(k, np.nan) if isinstance(p, str) else np.nan)
    metrics = [f"dnsmos_{k}" for k in KEYS]
    if ut2:
        d["model_utmosv2"] = d.model_wav.map(ut2)
        d["dsp_utmosv2"] = d.dsp_wav.map(lambda p: ut2.get(p, np.nan) if isinstance(p, str) else np.nan)
        metrics.append("utmosv2")
    src = f"{C.RES}/steer_dsp_dnsmos_raw.csv"
    split = "libritts_r_test_clean_heldout16"
    R = []
    base = d[d.dim == "baseline"]
    for m in metrics:
        col = m.replace("dnsmos_", "")
        g = [x[f"model_{col}"].dropna().values for _, x in base.groupby("vid")]
        R.append(C.row(m, cluster_ci(g), len(base), src, _scorer(m), split, system="steertts_uninstructed",
                       condition="baseline"))
        for (dim, z), x in d[d.dim != "baseline"].groupby(["dim", "req_z"]):
            for sysn, c in (("steertts_model", f"model_{col}"), ("dsp_postprocess", f"dsp_{col}")):
                g = [y[c].dropna().values for _, y in x.groupby("vid")]
                R.append(C.row(m, cluster_ci(g), int(x[c].notna().sum()), src, _scorer(m), split, system=sysn,
                               condition="single_control", target_dim=dim, req_z=float(z)))
            diff = x[f"model_{col}"] - x[f"dsp_{col}"]
            g = [diff[x.vid == v].dropna().values for v in sorted(x.vid.unique())]
            R.append(C.row(f"{m}_model_minus_dsp", cluster_ci(g), int(diff.notna().sum()), src, _scorer(m), split,
                           system="paired", condition="single_control", target_dim=dim, req_z=float(z)))
        # pooled over all dims/z
        x = d[d.dim != "baseline"]
        diff = x[f"model_{col}"] - x[f"dsp_{col}"]
        g = [diff[x.vid == v].dropna().values for v in sorted(x.vid.unique())]
        R.append(C.row(f"{m}_model_minus_dsp", cluster_ci(g), int(diff.notna().sum()), src, _scorer(m), split,
                       system="paired", condition="all_dims_all_z"))
    pub = d[["vid", "gender", "text_id", "dim", "req_z"] + [c for c in d.columns if c.startswith(("model_", "dsp_"))]].copy()
    for c in ("model_wav", "dsp_wav"):
        pub[c] = pub[c].astype(str).str.replace(str(C.ROOT) + "/", "", regex=False)
    C.write("steer_dsp_dnsmos", R, pub, dict(
        source=str(SRC), dsp_ops="see steer_dsp_baseline.json meta.dsp_ops",
        scorers=dict(dnsmos="DNSMOS P.835 (SIG/BAK/OVRL) + P.808, Microsoft DNS-Challenge ONNX via speechmos, 16 kHz, "
                            "single-threaded; not used in any SteerTTS training",
                     utmosv2="UTMOSv2 (sarulab-speech), if present; not used in training"),
        bootstrap="1000 cluster resamples over the 16 held-out voices, percentile 95% CI",
        note="model = SteerTTS directed render; dsp = DSP edit (PSOLA rate/pitch, gain, shelf) of the uninstructed render "
             "with the same requested z; paired difference per (voice,text) unit"))


def _scorer(m):
    return "UTMOSv2" if m == "utmosv2" else f"DNSMOS {'P.808' if m.endswith('p808') else 'P.835 ' + m.split('_')[1].upper()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    d = frame()
    if a.stage in ("score", "all"):
        score(d)
    if a.stage in ("analyze", "all"):
        analyze(d)


if __name__ == "__main__":
    main()
