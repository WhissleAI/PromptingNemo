"""X4: one scorer configuration for every system on LibriTTS-R test-clean (data/bench).
Produced: results/quality_rescore.json + _raw.csv: python -m steer.paper.x4_quality_rescore --stage render|asr|utmos|sim|analyze (queue job logs/gpu_queue/*/06c2_x4_quality_rescore.sh). All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.

Sets: `libritts` (500 test-clean sentences; our systems use the benchmark's 4+4 training-voice pool by gender, the
baselines their own default voice, GT = the real recording) and `clone` (499: text + a different real clip of the
same held-out test-clean speaker as reference; zero-shot cloning).
Scorers (identical for every system): Whisper-large-v3 fp16 + Whisper EnglishTextNormalizer (WER), UTMOS22-strong
(SpeechMOS), SIM = seed-tts-eval WavLM-large SV (their unmodified verification code, seedeval env) between the
render and the reference clip (clone set only; GT row = target recording vs reference = ceiling).
Systems: ours base (spk_enc_v3 312.5k), ours + flow refiner (flow_v3 100k), ours + flow + prompted flow (clone
only, flow_v3pc), ground truth, and existing baseline renders on disk (Kokoro-82M, Piper-VITS; Parler-mini has no
LibriTTS renders). Bootstrap: 1000 utterance resamples.

python -m steer.paper.x4_quality_rescore [--stage render|asr|utmos|sim|analyze|all]
"""
import argparse
import json
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from ..config import SR
from . import common as C

NAME = "quality_rescore"
OUT = C.WORK / "x4"
PFLOW = str(C.ROOT / "runs/flow_v3pc/last.pt")
SYSTEMS = {"ours_base": C.BASE, "ours_flow": f"{C.BASE},flow={C.FLOW}",
           "ours_flow_pflow": f"{C.BASE},flow={C.FLOW},pflow={PFLOW}"}
EXISTING = {"kokoro_82m": C.ROOT / "runs/bench/kokoro_82m/libritts", "piper_vits": C.ROOT / "runs/bench/piper_vits/libritts",
            "parler_mini": C.ROOT / "runs/bench/parler_mini/libritts"}
SEED = C.Path("/home/karan/Desktop/whissle/workspace/seedtts")


def sets():
    lib = pd.read_json(C.ROOT / "data/bench/libritts.jsonl", lines=True)
    clone = pd.read_json(C.ROOT / "data/bench/clone.jsonl", lines=True, dtype={"id": str})
    return lib, clone


def plan():
    lib, clone = sets()
    calib = json.load(open(C.ROOT / "data/bench/calibration.json"))
    items = []
    for i, r in lib.iterrows():
        items.append(dict(set="libritts", i=i, system="ground_truth", text=r.text, path=r.gt_path, spk=str(r.speaker)))
        for s in EXISTING:
            p = EXISTING[s] / f"{i:04d}.wav"
            if p.exists():
                items.append(dict(set="libritts", i=i, system=s, text=r.text, path=str(p), spk=str(r.speaker)))
        for s in ("ours_base", "ours_flow"):
            items.append(dict(set="libritts", i=i, system=s, text=r.text, path=str(OUT / s / "libritts" / f"{i:04d}.wav"),
                              spk=str(r.speaker), gender_hz=float(r.spk_f0)))
    for i, r in clone.iterrows():
        items.append(dict(set="clone", i=i, system="ground_truth", text=r.text, path=r.gt_path, ref=r.ref_path, spk=str(r.speaker)))
        for s in SYSTEMS:
            items.append(dict(set="clone", i=i, system=s, text=r.text, path=str(OUT / s / "clone" / f"{i:04d}.wav"),
                              ref=r.ref_path, spk=str(r.speaker)))
    return items, calib


def render(items, calib):
    from ..benchmark import voices, gender_of_hz
    stats = json.load(open(C.BASE.rsplit("/", 1)[0] + "/control_stats.json"))
    pools, _, _ = voices(stats)
    json.dump(pools, open(OUT / "voice_pools_private.json", "w"))
    for s, ck in SYSTEMS.items():
        need = [it for it in items if it["system"] == s and not C.Path(it["path"]).exists()]
        if not need:
            continue
        syn = C.make_synth(ck)
        for it in need:
            C.Path(it["path"]).parent.mkdir(parents=True, exist_ok=True)
            if it["set"] == "libritts":
                g = gender_of_hz(it["gender_hz"], calib)
                y = syn(it["text"], pools[g][it["i"] % 4])
            else:
                y = syn(it["text"], None, ref_wav=it["ref"])
            sf.write(it["path"], y, SR)
        print("rendered", s, len(need), flush=True)
        del syn
        torch.cuda.empty_cache()


@torch.no_grad()
def utmos(paths, cache):
    import soxr
    res = json.load(open(cache)) if cache.exists() else {}
    todo = [p for p in paths if p not in res]
    if todo:
        m = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to("cuda").eval()
        for p in todo:
            y, sr = sf.read(p, dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            y = soxr.resample(y, sr, 16000) if sr != 16000 else y
            res[p] = float(m(torch.from_numpy(y)[None].cuda(), 16000)[0])
        del m
        torch.cuda.empty_cache()
        json.dump(res, open(cache, "w"))
    return res


def sim(pairs, out):
    """pairs: list of (gen, ref). seed-tts-eval WavLM-large SV, their verification code (sv_runner copy)."""
    pf = OUT / "sim_pairs.txt"
    pf.write_text("\n".join(f"{a}|{b}" for a, b in pairs) + "\n")
    subprocess.run(["/home/karan/anaconda3/envs/seedeval/bin/python", "verification_pair_list_v2.py", str(pf),
                    "--model_name", "wavlm_large", "--checkpoint", str(SEED / "wavlm_large_finetune.pth"), "--scores", str(out),
                    "--wav1_start_sr", "0", "--wav2_start_sr", "0", "--wav1_end_sr", "-1", "--wav2_end_sr", "-1",
                    "--device", "cuda:0"], cwd=str(SEED / "sv_runner"), check=True)


def read_sim(out):
    res = {}
    for l in open(out):
        if l.startswith("avg score"):
            continue
        k, v = l.rstrip("\n").split("\t")
        a, b = k.split("|")
        res[(a.rsplit("_0_-1", 1)[0], b.rsplit("_0_-1", 1)[0])] = float(v)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items, calib = plan()
    st = a.stage
    if st != "analyze":
        C.gpu_wait()
    if st in ("render", "all"):
        render(items, calib)
    if st in ("asr", "all"):
        json.dump(C.transcribe(sorted({it["path"] for it in items}), OUT / "asr_cache.jsonl"), open(OUT / "hyps.json", "w"))
    if st in ("utmos", "all"):
        utmos(sorted({it["path"] for it in items}), OUT / "utmos.json")
    if st in ("sim", "all"):
        sim(sorted({(it["path"], it["ref"]) for it in items if it["set"] == "clone"}), OUT / "sim.out")
    if st in ("analyze", "all"):
        analyze(items)


def analyze(items):
    df = pd.DataFrame(items)
    hyps = json.load(open(OUT / "hyps.json"))
    ut = json.load(open(OUT / "utmos.json"))
    sm = read_sim(OUT / "sim.out")
    df["hyp"] = df.path.map(hyps)
    ew = [C.word_errors(t, h) for t, h in zip(df.text, df.hyp)]
    df["err"], df["nwords"] = [e for e, _ in ew], [n for _, n in ew]
    df["utmos"] = df.path.map(ut)
    df["sim"] = [sm.get((p, r), np.nan) if isinstance(r, str) else np.nan for p, r in zip(df.path, df.get("ref", [np.nan] * len(df)))]
    src = f"{C.RES}/{NAME}_raw.csv"
    R = []
    asr = "whisper-large-v3 fp16 + whisper EnglishTextNormalizer"
    for (st, sysn), g in df.groupby(["set", "system"]):
        units = [r for _, r in g.iterrows()]
        split = "libritts_r_test_clean_bench500" if st == "libritts" else "libritts_r_test_clean_clone499"
        cond = dict(condition=st, system=sysn)
        e, n = g.err.values, g.nwords.values
        R.append(C.row("wer", C.boot_ci(list(zip(e, n)), lambda u: sum(x for x, _ in u) / sum(y for _, y in u)), len(g), src, asr, split, **cond))
        R.append(C.row("utmos", C.boot_ci(list(g.utmos.dropna()), lambda u: float(np.mean(u))), int(g.utmos.notna().sum()), src,
                       "SpeechMOS utmos22_strong", split, **cond))
        if st == "clone":
            v = g.sim.dropna()
            R.append(C.row("sim_wavlm_large", C.boot_ci(list(v), lambda u: float(np.mean(u))), len(v), src,
                           "seed-tts-eval WavLM-large SV (gen vs reference clip)", split, **cond))
        R.append(C.row("empty_transcripts", float((g.hyp.fillna("") == "").sum()), len(g), src, asr, split, **cond))
    pub = df.copy()
    pub["path"] = pub.path.str.replace(str(C.ROOT) + "/", "", regex=False)
    if "ref" in pub:
        pub["ref"] = pub.ref.astype(str).str.replace(str(C.ROOT) + "/", "", regex=False)
    pub = pub.drop(columns=["spk"])  # public raw: no speaker ids (paths keep LibriTTS-R file names, a public corpus)
    meta = dict(systems=dict(SYSTEMS, ground_truth="real recording", **{k: str(v) for k, v in EXISTING.items()}),
                note="libritts set: ours = benchmark training-voice pool (not cloned); baselines = their default voice. "
                     "SIM only on the clone set. Parler-mini has no LibriTTS renders on disk.")
    C.write(NAME, R, pub, meta)


if __name__ == "__main__":
    main()
