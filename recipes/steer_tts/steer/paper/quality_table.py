"""Compact quality table (round-2 review, Tier 1 #4): MOS predictors we did NOT train on (DNSMOS P.808 / P.835 OVRL,
UTMOSv2), Whisper-large-v3 WER, and WavLM-large SIM for cloning, on EXISTING LibriTTS-R test-clean renders.

    cd $STEER_TTS_ROOT && CUDA_VISIBLE_DEVICES= PYTHONPATH=<recipes/steer_tts> python -m steer.paper.quality_table

Per-utterance sources (no new rendering here):
  ground truth / SteerTTS (X4 renders) / Kokoro / Piper : results/quality_rescore_raw.csv (WER err+nwords, UTMOS22, SIM)
  other baselines (runs/paper/baselines_ext/<sys>/libritts): Whisper-large-v3 hyps in baselines_ext/_scores/asr_v3.jsonl
  DNSMOS: baselines_ext/_scores/dnsmos.json (speechmos ONNX, CPU);  UTMOSv2: runs/paper/quality_table/utmosv2.json
Same Whisper-large-v3 fp16 + EnglishTextNormalizer WER for every row. -> results/quality_table.json (+ _raw.csv)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import baselines_ext_eval as E

ROOT = C.ROOT
QT = C.WORK / "quality_table"
ALIAS = {"runs/bench/kokoro_82m/": "runs/bench/_ext/kokoro/", "runs/bench/piper_vits/": "runs/bench/_ext/piper/"}
NAME = {"kokoro_82m": "kokoro", "piper_vits": "piper"}
SPLIT = {"libritts": "libritts_r_test_clean_bench500", "clone": "libritts_r_test_clean_clone499"}
EXT = ["matcha", "fastpitch", "paradee", "kitten", "parastyle", "parler_mini", "styletts2", "f5", "xtts", "cosyvoice2",
       "vocos_copysyn"]


def absp(p):
    p = str(p)
    return p if p.startswith("/") else str(ROOT / p)


def lookup(d, p):
    if p in d:
        return d[p]
    for a, b in ALIAS.items():
        q = p.replace(a, b)
        if q in d:
            return d[q]
    return None


def main():
    dn = json.load(open(E.SC / "dnsmos.json"))
    u2 = json.load(open(QT / "utmosv2.json")) if (QT / "utmosv2.json").exists() else {}
    ut22 = json.load(open(E.SC / "utmos.json")) if (E.SC / "utmos.json").exists() else {}
    hyps = E.asr_cache("asr_v3.jsonl")
    q = pd.read_csv(C.RES / "quality_rescore_raw.csv")
    q["path"] = q.path.map(absp)
    q["system"] = q.system.map(lambda s: NAME.get(s, s))
    q = q[["set", "i", "system", "text", "path", "err", "nwords", "utmos", "sim"]]
    parts = [q]
    for s in EXT:
        sets = ["libritts"] + (["clone"] if E.valid(s, "clone") else [])
        for st in sets:
            its = E.present(E.rendered(s, st))
            if not its:
                continue
            df = pd.DataFrame(its)[["path", "text"] + (["ref"] if st == "clone" else [])]
            df["set"], df["system"] = st, s
            df["i"] = df.path.map(lambda p: int(Path(p).stem))
            df["hyp"] = df.path.map(hyps)
            ew = [C.word_errors(t, h) if isinstance(h, str) else (np.nan, np.nan) for t, h in zip(df.text, df.hyp)]
            df["err"], df["nwords"] = [e for e, _ in ew], [n for _, n in ew]
            df["utmos"] = df.path.map(ut22)
            if st == "clone":
                sims = {}
                for f in E.SC.glob("sim_*.out"):
                    sims.update(E.read_sim(f))
                df["sim"] = [sims.get((p, r), np.nan) for p, r in zip(df.path, df.ref)]
            parts.append(df.drop(columns=[c for c in ("hyp", "ref") if c in df]))
    raw = pd.concat(parts, ignore_index=True)
    for k in ("p808", "ovrl", "sig", "bak"):
        raw[f"dnsmos_{k}"] = raw.path.map(lambda p: (lookup(dn, p) or {}).get(k, np.nan))
    raw["utmosv2"] = raw.path.map(lambda p: lookup(u2, p))
    raw["utmosv2"] = raw.utmosv2.astype(float)

    asr = "whisper-large-v3 fp16 + whisper EnglishTextNormalizer"
    src = f"{C.RES}/quality_table_raw.csv"
    R = []
    for (st, s), g in raw.groupby(["set", "system"], sort=False):
        expected = 500 if st == "libritts" else 499
        cond = dict(system=s, condition=st, n_expected=expected,
                    group="reference" if s in ("ground_truth", "vocos_copysyn") else ("ours" if s.startswith("ours") else "baseline"))
        w = g[g.err.notna()]
        if len(w):
            R.append(C.row("wer", C.boot_ci(list(zip(w.err, w.nwords)), lambda u: 100 * sum(x for x, _ in u) / sum(y for _, y in u)),
                           len(w), src, asr, SPLIT[st], **cond))
        for col, scorer in (("dnsmos_p808", "DNSMOS P.808 (speechmos ONNX, not used in training)"),
                            ("dnsmos_ovrl", "DNSMOS P.835 OVRL (speechmos ONNX, not used in training)"),
                            ("utmosv2", "UTMOSv2 fusion_stage3 fold0 (not used in training)"),
                            ("utmos", "UTMOS22-strong (SpeechMOS; a TRAINING REWARD for SteerTTS v3: circular for ours)"),
                            ("sim", "seed-tts-eval WavLM-large SV (gen vs reference clip)")):
            v = g[col].dropna()
            if len(v):
                R.append(C.row(col if col != "sim" else "sim_wavlm_large", C.boot_ci(list(v), lambda u: float(np.mean(u))),
                               len(v), src, scorer, SPLIT[st], **cond))
    # partial systems: matched-subset rows for ground truth and SteerTTS on exactly the items that system rendered
    for (st, s), g in list(raw.groupby(["set", "system"], sort=False)):
        expected = 500 if st == "libritts" else 499
        if len(g) >= expected or s in ("ground_truth",) or s.startswith("ours"):
            continue
        ids = set(g.i)
        for ref in ("ground_truth", "ours_base", "ours_flow", s):
            h = raw[(raw.set == st) & (raw.system == ref) & raw.i.isin(ids)]
            cond = dict(system=ref, condition=f"{st}_subset_{s}", n_expected=len(ids), subset_of=s,
                        group="reference" if ref == "ground_truth" else ("ours" if ref.startswith("ours") else "baseline"))
            w = h[h.err.notna()]
            if len(w):
                R.append(C.row("wer", C.boot_ci(list(zip(w.err, w.nwords)), lambda u: 100 * sum(x for x, _ in u) / sum(y for _, y in u)),
                               len(w), src, asr, SPLIT[st] + f"_subset_{s}", **cond))
            for col in ("dnsmos_p808", "dnsmos_ovrl", "utmosv2"):
                v = h[col].dropna()
                if len(v):
                    R.append(C.row(col, C.boot_ci(list(v), lambda u: float(np.mean(u))), len(v), src, col, SPLIT[st] + f"_subset_{s}", **cond))
    for r in R:
        r["partial"] = bool(r["n"] < r["n_expected"])
    raw_pub = raw.copy()
    raw_pub["path"] = raw_pub.path.str.replace(str(ROOT) + "/", "", regex=False)
    C.write("quality_table", R, raw_pub, dict(
        note="Compact quality table on EXISTING LibriTTS-R test-clean renders. libritts = bench500 (SteerTTS renders its "
             "training-voice pool, not cloned; default-voice baselines use their gender-matched default voice; ground truth "
             "= the real clips); clone = clone499 (zero-shot from a different real clip of the same held-out speaker; SIM "
             "vs that reference). MOS predictors independent of our training: DNSMOS P.808 / P.835 OVRL and UTMOSv2; "
             "UTMOS22 kept for reference only (training reward for SteerTTS v3). vocos_copysyn = ground truth -> mel -> "
             "Vocos (our vocoder's upper bound). Rows with n < n_expected are partial; for each partial system, rows with "
             "condition '<set>_subset_<system>' give ground truth / SteerTTS / that system on exactly the same items.",
        bootstrap="1000 utterance resamples, percentile 95% CI", code="steer/paper/quality_table.py"))
    t = pd.DataFrame(R).pivot_table(index=["condition", "system"], columns="metric", values="value", aggfunc="first")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
