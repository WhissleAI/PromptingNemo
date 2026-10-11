"""Eval sets, scoring and tables for the extended baselines (steer.paper.baselines_ext). Runs in the emotts env.

    cd $STEER_TTS_ROOT && PYTHONPATH=<recipes/steer_tts> python -m steer.paper.baselines_ext_eval <cmd> [...]

  plan    --system S --sets libritts,clone,psc,seedtts,steer   -> runs/paper/baselines_ext/<S>/<set>/_jobs.jsonl
          (the GPU job then runs `<env python> -m steer.paper.baselines_ext render --jobs ... --out ...`)
  score   --systems a,b --sets ...   GPU: Whisper-large-v3 fp16 (+ distil-large-v2 on psc), UTMOS22-strong,
          seed-tts-eval WavLM-large SV (clone set).  Caches under runs/paper/baselines_ext/_scores/.
  dnsmos  --systems a,b|all          CPU, tb_score env (speechmos DNSMOS P.835 ONNX); also scores our X4 renders
          and the ground truth so every row of T_main uses the same DNSMOS.
  seedtts --system S                 GPU: official seed-tts-eval scripts (unmodified run_wer.py fp32 + WavLM SV)
  measure --systems a,b              CPU: steer.features.measure + Praat on the native-knob steerability renders
  analyze                            -> paper results/baselines_{quality,psc,seedtts,steer}.json (+ _raw.csv)
  table                              -> paper tables/T_main.tex

Protocol (identical to X4 / the benchmark, see x4_quality_rescore.py and benchmark.py):
  libritts  500 LibriTTS-R test-clean sentences.  Default-voice systems: their own gender-matched voice (benchmark
            F0 rule).  Cloning systems (voice=clone): zero-shot from a DIFFERENT real test-clean clip of the same
            speaker (the clone set's reference for that speaker).  Prompt systems: neutral gendered description.
  clone     499 (text, different real clip of the same held-out speaker) -> SIM to the reference (cloning systems).
  psc       ParaSpeechCaps test 246 descriptions (prompt-capable systems); attribute accuracy with the benchmark's
            estimator and calibration (gender / pitch level / speaking rate), WER (v3 + distil-large-v2), UTMOS.
  seedtts   Seed-TTS-eval test-en 1,088 (cloning systems), official scorer.
  steer     native explicit-control probe: steer.harness TEXTS (8) x voices (16 held-out x-vector/ref voices for
            cloning systems, the 2 default voices otherwise) x knob level in {-2,-1,+1,+2} + the knob-neutral render;
            rate knob factors RATE_LEVELS, FastPitch pitch shift PITCH_LEVELS (its normalised-pitch SD units).
            Measured exactly like X1 (dz = render minus the neutral render of the same voice/text, in the SteerTTS
            training pooled SD), so rows line up with results/steer_heldout.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from .baselines_ext import SYSTEMS, ROOT

WORK = C.WORK / "baselines_ext"
SC = WORK / "_scores"
PAPER = C.RES.parent
SEED = Path("/home/karan/Desktop/whissle/workspace/seedtts")
SEED_META = sorted((SEED / "data").glob("*/en/meta.lst"))
# Knob values matched to SteerTTS's requested z: rate factor = 1 + z * SD_rate / mean_rate (training pooled SD 1.02,
# mean speaker rate 6.35 -> 16% per z); FastPitch pitch shift = Hz offset of z * 2.09 st (f0_med pooled SD) at its LJ
# mean F0 212.4 Hz, divided by its pitch SD 68.5 Hz (model cfg pitch_mean / pitch_std).
RATE_LEVELS = {-2.0: 0.68, -1.0: 0.84, 1.0: 1.16, 2.0: 1.32}
PITCH_LEVELS = {-2.0: -0.66, -1.0: -0.36, 1.0: 0.39, 2.0: 0.84}
EXT_DIRS = {"kokoro": ROOT / "runs/bench/_ext/kokoro", "piper": ROOT / "runs/bench/_ext/piper",
            "parler_mini": ROOT / "runs/bench/_ext/parler_mini"}  # renders made earlier by steer/baselines.py
TB_SCORE = Path.home() / "anaconda3/envs/tb_score/bin/python"


# reference rows that are not TTS systems (rendered here, scored like everything else)
PSEUDO = {"vocos_copysyn": dict(group="reference", voice="default", controls=[], params_M=13.5,
                                checkpoint="charactr/vocos-mel-24khz: ground-truth 24 kHz audio -> its own 100-bin mel "
                                           "-> Vocos (vocoder upper bound for SteerTTS, which predicts the same mel)")}


def cfg(s):
    return SYSTEMS[s] if s in SYSTEMS else PSEUDO[s]


def cmd_copysyn():
    """Vocos copy-synthesis of the 500 LibriTTS-R GT clips (CPU)."""
    import soundfile as sf
    import torch
    from vocos import Vocos
    torch.set_num_threads(4)
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
    lib, _, _, _ = bench()
    d = WORK / "vocos_copysyn" / "libritts"
    d.mkdir(parents=True, exist_ok=True)
    for i, r in lib.iterrows():
        f = d / f"{i:04d}.wav"
        if f.exists():
            continue
        y = torch.from_numpy(C.load24(r.gt_path))[None]
        with torch.no_grad():
            out = voc.decode(voc.feature_extractor(y))[0].numpy()
        sf.write(f, out, 24000)
    print("copysyn", len(list(d.glob("*.wav"))))


def gender_rule(f0, calib):
    return "female" if 12 * np.log2(f0 / 100.0) > calib["gender_cut_st"] else "male"


def bench():
    lib = pd.read_json(ROOT / "data/bench/libritts.jsonl", lines=True)
    clone = pd.read_json(ROOT / "data/bench/clone.jsonl", lines=True, dtype={"id": str})
    psc = pd.read_json(ROOT / "data/bench/psc.jsonl", lines=True)
    calib = json.load(open(ROOT / "data/bench/calibration.json"))
    return lib, clone, psc, calib


def wav_dir(s, st):
    if st in ("libritts", "psc") and s in EXT_DIRS and (EXT_DIRS[s] / st).exists():
        return EXT_DIRS[s] / st
    return WORK / s / st


# ----------------------------------------------------------------------------------------------------- planning
def plan(s, st):
    """Job items for system s on set st (list of dicts; extra keys are kept for scoring)."""
    lib, clone, psc, calib = bench()
    text_of = dict(zip(lib.gt_path, lib.text))
    voice = cfg(s)["voice"]
    out = []
    if st == "libritts":
        ref_of = dict(zip(clone.speaker, clone.ref_path))
        for i, r in lib.iterrows():
            it = dict(id=f"{i:04d}", text=r.text, gender=gender_rule(r.spk_f0, calib))
            if voice == "clone":
                # a different real clip of the same speaker (clone-set reference first); never the target itself.
                # Speakers with a single bench clip fall back to the first other same-gender bench speaker's clip.
                cands = [ref_of.get(r.speaker)] + list(lib[lib.speaker == r.speaker].gt_path)
                ref = next((p for p in cands if p and p != r.gt_path), None)
                same = ref is not None
                if not same:
                    alt = lib[(lib.speaker != r.speaker) & (lib.spk_f0.map(lambda f: gender_rule(f, calib)) == it["gender"])]
                    ref = alt.gt_path.iloc[0]
                it.update(ref=ref, ref_text=text_of[ref], ref_same_speaker=same)
            out.append(it)
    elif st == "clone":
        assert voice == "clone", s
        for i, r in clone.iterrows():
            out.append(dict(id=f"{i:04d}", text=r.text, ref=r.ref_path, ref_text=text_of[r.ref_path],
                            gender=gender_rule(lib[lib.speaker == r.speaker].spk_f0.iloc[0], calib)))
    elif st == "psc":
        for i, r in psc.iterrows():
            it = dict(id=f"{i:04d}", text=r.text, gender=r.gender, desc=r.description)
            out.append(it)
    elif st == "seedtts":
        assert voice == "clone", s
        meta = SEED_META[0]
        for l in open(meta):
            utt, ptext, pwav, text = l.rstrip("\n").split("|")
            out.append(dict(id=utt, text=text, ref=str(meta.parent / pwav), ref_text=ptext))
    elif st == "steer":
        from ..harness import TEXTS
        knobs = [("rate", RATE_LEVELS)] + ([("pitch", PITCH_LEVELS)] if "pitch_shift" in cfg(s)["controls"] else [])
        if voice == "clone":
            V = [dict(vid=v["vid"], gender="female" if v["gender"] == "F" else "male", ref=v["ref"],
                      ref_text=text_of[v["ref"]]) for v in C.heldout_voices(16)]
        else:
            V = [dict(vid="DF", gender="female"), dict(vid="DM", gender="male")]
        for v in V:
            for ti, text in enumerate(TEXTS):
                base = dict(text=text, gender=v["gender"], vid=v["vid"], text_id=ti,
                            **({"ref": v["ref"], "ref_text": v["ref_text"]} if "ref" in v else {}))
                out.append(dict(base, id=f"{v['vid']}_{ti}_baseline_+0", dim="baseline", req_z=0.0))
                for k, lv in knobs:
                    for z, val in lv.items():
                        out.append(dict(base, id=f"{v['vid']}_{ti}_{k}_{z:+.0f}", dim=k, req_z=z, **{k: val}))
    return out


def cmd_plan(s, sets):
    for st in sets:
        items = plan(s, st)
        d = WORK / s / st
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "_jobs.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(s, st, len(items), "->", d / "_jobs.jsonl")


def rendered(s, st):
    d = wav_dir(s, st)
    items = plan(s, st)
    for it in items:
        it["path"] = str(d / f"{it['id']}.wav")
    return items


def present(items):
    return [it for it in items if Path(it["path"]).exists()]


def valid(s, st):
    """Sets a system can be evaluated on: clone/seedtts need a cloning system."""
    return st not in ("clone", "seedtts") or cfg(s)["voice"] == "clone"


# ------------------------------------------------------------------------------------------------------ scoring
def sim_pairs(pairs, out, tag):
    """seed-tts-eval WavLM-large SV (their verification code, seedeval env), as in x4_quality_rescore.sim."""
    pf = SC / f"sim_pairs_{tag}.txt"
    pf.write_text("\n".join(f"{a}|{b}" for a, b in pairs) + "\n")
    subprocess.run(["/home/karan/anaconda3/envs/seedeval/bin/python", "verification_pair_list_v2.py", str(pf),
                    "--model_name", "wavlm_large", "--checkpoint", str(SEED / "wavlm_large_finetune.pth"), "--scores",
                    str(out), "--wav1_start_sr", "0", "--wav2_start_sr", "0", "--wav1_end_sr", "-1", "--wav2_end_sr",
                    "-1", "--device", "cuda:0"], cwd=str(SEED / "sv_runner"), check=True)


def read_sim(out):
    res = {}
    if not Path(out).exists():
        return res
    for l in open(out):
        if l.startswith("avg score") or "\t" not in l:
            continue
        k, v = l.rstrip("\n").split("\t")
        a, b = k.split("|")
        res[(a.rsplit("_0_-1", 1)[0], b.rsplit("_0_-1", 1)[0])] = float(v)
    return res


def cmd_score(systems, sets):
    from .x4_quality_rescore import utmos
    SC.mkdir(parents=True, exist_ok=True)
    paths, psc_paths, pairs = [], [], []
    for s in systems:
        for st in sets:
            if st == "seedtts" or not valid(s, st):
                continue
            its = present(rendered(s, st))
            paths += [it["path"] for it in its]
            if st == "psc":
                psc_paths += [it["path"] for it in its]
            if st == "clone":
                pairs += [(it["path"], it["ref"]) for it in its]
    print(f"score: {len(paths)} wavs, {len(psc_paths)} psc, {len(pairs)} sim pairs", flush=True)
    if paths:
        C.transcribe(sorted(set(paths)), SC / "asr_v3.jsonl", batch=8)  # <6 GB peak (eval lane)
    if psc_paths:
        C.transcribe(sorted(set(psc_paths)), SC / "asr_dl2.jsonl", model="distil-whisper/distil-large-v2", batch=8)
    q = [p for p in paths if "/steer/" not in p]
    if q:
        utmos(sorted(set(q)), SC / "utmos.json")
    if pairs:
        tag = "_".join(systems)
        sim_pairs(sorted(set(pairs)), SC / f"sim_{tag}.out", tag)


def asr_cache(name):
    f = SC / name
    res = {}
    if f.exists():
        for l in open(f):
            d = json.loads(l)
            res[d["path"]] = d["hyp"]
    return res


def cmd_seedtts(s):
    """Official seed-tts-eval scoring, same calls as logs/seedtts.sh (unmodified fp32 run_wer.py)."""
    out = WORK / s / "seedtts"
    meta = SEED_META[0]
    sp = "/home/karan/anaconda3/envs/seedeval/bin/python"
    ev = SEED / "seed-tts-eval"
    n = len(list(out.glob("*.wav")))
    assert n == sum(1 for _ in open(meta)), f"{s}: {n} seedtts wavs"
    subprocess.run([sp, "get_wav_res_ref_text.py", str(meta), str(out), str(out / "wav_res_ref_text")], cwd=ev, check=True)
    subprocess.run([sp, "run_wer.py", str(out / "wav_res_ref_text"), str(out / "wer.out"), "en"], cwd=ev, check=True)
    subprocess.run([sp, "average_wer.py", str(out / "wer.out"), str(out / "wer.score")], cwd=ev, check=True)
    subprocess.run([sp, "verification_pair_list_v2.py", str(out / "wav_res_ref_text"), "--model_name", "wavlm_large",
                    "--checkpoint", str(SEED / "wavlm_large_finetune.pth"), "--scores", str(out / "sim.out"),
                    "--wav1_start_sr", "0", "--wav2_start_sr", "0", "--wav1_end_sr", "-1", "--wav2_end_sr", "-1",
                    "--device", "cuda:0"], cwd=SEED / "sv_runner", check=True)
    lines = [l for l in open(out / "sim.out") if not l.startswith("avg score")]
    (out / "sim.merge").write_text("".join(lines))
    subprocess.run([sp, "average.py", str(out / "sim.merge"), str(out / "sim.score")], cwd=SEED / "sv_runner", check=True)
    print(s, open(out / "wer.score").read().splitlines()[-1], open(out / "sim.score").read().split("\n")[-3:])


def ours_paths():
    """Our X4 renders + ground truth (for DNSMOS on identical audio)."""
    lib, clone, _, _ = bench()
    x4 = C.WORK / "x4"
    P = {("ground_truth", "libritts"): list(lib.gt_path), ("ground_truth", "clone"): list(clone.gt_path)}
    for s in ("ours_base", "ours_flow", "ours_flow_pflow"):
        for st, n in (("libritts", len(lib)), ("clone", len(clone))):
            ps = [str(x4 / s / st / f"{i:04d}.wav") for i in range(n)]
            P[(s, st)] = [p for p in ps if Path(p).exists()]
    return P


def cmd_dnsmos(systems):
    SC.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in systems:
        for st in ("libritts", "clone", "psc"):
            paths += [it["path"] for it in present(rendered(s, st))] if valid(s, st) else []
    for v in ours_paths().values():
        paths += v
    lst = SC / "dnsmos_list.txt"
    lst.write_text("\n".join(sorted(set(paths))) + "\n")
    subprocess.run([str(TB_SCORE), "-s", str(Path(__file__).with_name("dnsmos_score.py")), str(lst), str(SC / "dnsmos.json"), "8"],
                   check=True, env={"PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
                                    "OMP_NUM_THREADS": "1"})


def cmd_measure(systems):
    for s in systems:
        its = present(rendered(s, "steer"))
        if not its:
            continue
        M = C.measure_many([(it["path"], it["path"], it["text"]) for it in its], procs=8)
        M.to_parquet(WORK / s / "steer_measures.parquet")
        print("measured", s, len(M), flush=True)


# ------------------------------------------------------------------------------------------------------ analysis
def _row(metric, v, n, src, scorer, split, s, **kw):
    return C.row(metric, v, n, src, scorer, split, system=s, group=kw.pop("group", cfg(s)["group"] if (s in SYSTEMS or s in PSEUDO) else "ours"), **kw)


def quality_rows(s, st, hyps, ut, dn, sims, src):
    its = present(rendered(s, st))
    if not its:
        return [], []
    df = pd.DataFrame(its)
    df["system"], df["set"] = s, st
    df["hyp"] = df.path.map(hyps)
    ew = [C.word_errors(t, h) if isinstance(h, str) else (np.nan, np.nan) for t, h in zip(df.text, df.hyp)]
    df["err"], df["nwords"] = [e for e, _ in ew], [n for _, n in ew]
    df["utmos"] = df.path.map(ut)
    for k in ("ovrl", "sig", "bak", "p808"):
        df[f"dnsmos_{k}"] = df.path.map(lambda p: dn.get(p, {}).get(k, np.nan))
    if st == "clone":
        df["sim"] = [sims.get((p, r), np.nan) for p, r in zip(df.path, df.ref)]
    split = {"libritts": "libritts_r_test_clean_bench500", "clone": "libritts_r_test_clean_clone499"}[st]
    asr = "whisper-large-v3 fp16 + whisper EnglishTextNormalizer"
    R = []
    g = df[df.hyp.notna()]
    if len(g):
        R.append(_row("wer", C.boot_ci(list(zip(g.err, g.nwords)), lambda u: 100 * sum(x for x, _ in u) / sum(y for _, y in u)),
                      len(g), src, asr, split, s, condition=st))
        R.append(_row("empty_transcripts", float((g.hyp == "").sum()), len(g), src, asr, split, s, condition=st))
    for col, name, scorer in (("utmos", "utmos", "SpeechMOS utmos22_strong"),
                              ("dnsmos_ovrl", "dnsmos_ovrl", "DNSMOS P.835 OVRL (speechmos, ONNX)"),
                              ("dnsmos_sig", "dnsmos_sig", "DNSMOS P.835 SIG"), ("dnsmos_bak", "dnsmos_bak", "DNSMOS P.835 BAK"),
                              ("sim", "sim_wavlm_large", "seed-tts-eval WavLM-large SV (gen vs reference clip)")):
        if col in df and df[col].notna().any():
            v = list(df[col].dropna())
            R.append(_row(name, C.boot_ci(v, lambda u: float(np.mean(u))), len(v), src, scorer, split, s, condition=st))
    R.append(_row("n_rendered", float(len(df)), len(plan(s, st)), src, "count", split, s, condition=st))
    return R, [df]


def psc_rows(s, hyps, hyps_dl2, ut, dn, src):
    from ..benchmark import PITCH, RATE, bins, f0_and_speech_dur, load24, n_phones
    its = present(rendered(s, "psc"))
    if not its:
        return [], []
    _, _, psc, calib = bench()
    df = pd.DataFrame(its)
    df["i"] = df.id.astype(int)
    df = df.join(psc[["source", "leak", "pitch", "speaking_rate", "phonemes"]], on="i")
    est = []
    for p, r in zip(df.path, df.itertuples()):
        hz, dur = f0_and_speech_dur(load24(p))
        st_ = 12 * np.log2(hz / 100.0) if hz == hz else float("nan")
        est.append(dict(gender_ok=(gender_rule(hz, calib) if hz == hz else "?") == r.gender,
                        pitch_ok=(PITCH[bins(st_, calib[f"pitch_cuts_{r.gender}"])] if hz == hz else "?") == r.pitch,
                        rate_ok=RATE[bins(n_phones(r.phonemes) / max(dur, 0.3), calib["rate_cuts"])] == r.speaking_rate))
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(est)], axis=1)
    df["system"], df["set"] = s, "psc"
    df["utmos"] = df.path.map(ut)
    df["dnsmos_ovrl"] = df.path.map(lambda p: dn.get(p, {}).get("ovrl", np.nan))
    R = []
    for sub_name, sub in (("all", df), ("voxceleb", df[df.source == "voxceleb"]), ("no_leak", df[~df.leak.astype(bool)])):
        split = f"paraspeechcaps_test246_{sub_name}"
        for key, H, scorer in (("wer_v3", hyps, "whisper-large-v3 fp16 + EnglishTextNormalizer"),
                               ("wer_dl2", hyps_dl2, "distil-whisper-large-v2 fp16 + EnglishTextNormalizer")):
            h = sub.path.map(H)
            ok = h.notna()
            if ok.any():
                ew = [C.word_errors(t, x) for t, x in zip(sub.text[ok], h[ok])]
                R.append(_row(key, C.boot_ci(ew, lambda u: 100 * sum(x for x, _ in u) / sum(y for _, y in u)), int(ok.sum()),
                              src, scorer, split, s, condition="psc"))
        for col in ("gender_ok", "pitch_ok", "rate_ok"):
            R.append(_row(col.replace("_ok", "_acc"), C.boot_ci(list(sub[col].astype(float)), lambda u: float(np.mean(u))),
                          len(sub), src, "benchmark objective estimator (calibration.json)", split, s, condition="psc",
                          **({"flag": "uninformative: equals the constant 'high-pitched' base rate of the labels"}
                             if col == "pitch_ok" else {})))
        for col, scorer in (("utmos", "SpeechMOS utmos22_strong"), ("dnsmos_ovrl", "DNSMOS P.835 OVRL")):
            v = list(sub[col].dropna())
            if v:
                R.append(_row(col, C.boot_ci(v, lambda u: float(np.mean(u))), len(v), src, scorer, split, s, condition="psc"))
    return R, [df.drop(columns=["phonemes"])]


def seedtts_rows(s):
    out = WORK / s / "seedtts"
    if not (out / "sim.score").exists():
        return []
    wl = open(out / "wer.score").read().splitlines()[1:]
    per = np.array([float(x.split("\t")[1]) for x in wl if x.count("\t") >= 6])
    sims = [float(l.rsplit("\t", 1)[1]) for l in open(out / "sim.score") if "\t" in l]
    split = "seed_tts_eval_test_en_1088"
    return [_row("wer", C.boot_ci(list(per), lambda u: 100 * float(np.mean(u))), len(per), out / "wer.score",
                 "seed-tts-eval run_wer.py (whisper-large-v3 fp32, unmodified); mean of per-utterance WER", split, s),
            _row("sim_wavlm_large", C.boot_ci(sims, lambda u: float(np.mean(u))), len(sims), out / "sim.score",
                 "seed-tts-eval UniSpeech WavLM-large SV", split, s)]


def steer_rows(s, hyps):
    from scipy.stats import spearmanr
    f = WORK / s / "steer_measures.parquet"
    if not f.exists():
        return [], []
    stats = json.load(open(Path(C.BASE).parent / "control_stats.json"))
    sd = pd.Series(stats["pooled_sd"])
    M = pd.read_parquet(f).set_index("key")
    df = pd.DataFrame(present(rendered(s, "steer")))
    df = df.rename(columns={"rate": "knob_rate", "pitch": "knob_pitch"}).join(M, on="path")
    df["hyp"] = df.path.map(hyps)
    ew = [C.word_errors(t, h) if isinstance(h, str) else (np.nan, np.nan) for t, h in zip(df.text, df.hyp)]
    df["err"], df["nwords"] = [e for e, _ in ew], [n for _, n in ew]
    from ..config import CONTROL_DIMS
    base = df[df.dim == "baseline"].set_index(["vid", "text_id"])
    idx = pd.MultiIndex.from_arrays([df.vid, df.text_id])
    for d in CONTROL_DIMS:
        df[f"dz_{d}"] = (df[d].values - base[d].reindex(idx).values) / sd[d]
    src = f"{C.RES}/baselines_steer_raw.csv"
    split = "harness_texts8_" + ("heldout16" if cfg(s)["voice"] == "clone" else "default_voices2")
    R = []
    knob_target = {"rate": "rate", "pitch": "f0_med"}
    # bootstrap on a slim numeric frame (the full frame with string columns made each resample very slow)
    slim = df[["vid", "text_id", "dim", "req_z", "err", "nwords"] + [f"dz_{d}" for d in CONTROL_DIMS]].copy()
    slim["vid"] = slim.vid.astype("category").cat.codes
    for k, tgt in knob_target.items():
        sub = slim[slim.dim == k].drop(columns="dim")
        if sub.empty:
            continue
        cond = dict(condition="native_knob", knob=k, target_dim=tgt, extractor="ours",
                    knob_values=json.dumps(RATE_LEVELS if k == "rate" else PITCH_LEVELS))
        R.append(_row("spearman_rho", C.boot_df(sub, ["vid", "text_id"], lambda x: spearmanr(x.req_z, x[f"dz_{tgt}"]).statistic),
                      len(sub), src, "steer.features.measure", split, s, **cond))
        R.append(_row("slope_dz_per_level", C.boot_df(sub, ["vid", "text_id"], lambda x: float(np.sum(x.req_z * x[f"dz_{tgt}"]) / np.sum(x.req_z ** 2))),
                      len(sub), src, "steer.features.measure", split, s, **cond))
        others = [f"dz_{d}" for d in CONTROL_DIMS if d != tgt]
        for z in sorted(sub.req_z.unique()):
            g = sub[sub.req_z == z]
            c2 = dict(cond, req_level=float(z), knob_value=(RATE_LEVELS if k == "rate" else PITCH_LEVELS)[z])
            R.append(_row("mean_delta_z", C.boot_df(g, ["vid", "text_id"], lambda x: float(x[f"dz_{tgt}"].mean())), len(g), src,
                          "steer.features.measure", split, s, **c2))
            R.append(_row("hit_rate", C.boot_df(g, ["vid", "text_id"], lambda x: float((np.sign(z) * x[f"dz_{tgt}"] > 0.5).mean())),
                          len(g), src, "steer.features.measure", split, s, **c2))
            R.append(_row("direction_acc", C.boot_df(g, ["vid", "text_id"], lambda x: float((np.sign(z) * x[f"dz_{tgt}"] > 0).mean())),
                          len(g), src, "steer.features.measure", split, s, **c2))
            R.append(_row("leak_mean_abs_dz_other", C.boot_df(g, ["vid", "text_id"], lambda x: float(np.nanmean(np.abs(x[others].values)))),
                          len(g), src, "steer.features.measure", split, s, **c2))
            gw = g[g.err.notna()]
            if len(gw):
                R.append(_row("wer", C.boot_df(gw, ["vid", "text_id"], lambda x: 100 * x.err.sum() / x.nwords.sum()), len(gw), src,
                              "whisper-large-v3 fp16 + whisper EnglishTextNormalizer", split, s, **c2))
    b = slim[(slim.dim == "baseline") & df.hyp.notna().values]
    if len(b):
        R.append(_row("wer", C.boot_df(b, ["vid", "text_id"], lambda x: 100 * x.err.sum() / x.nwords.sum()), len(b), src,
                      "whisper-large-v3 fp16 + whisper EnglishTextNormalizer", split, s, condition="native_knob_neutral"))
    pub = df.drop(columns=[c for c in ("ref", "ref_text") if c in df]).copy()
    pub["system"] = s
    pub["path"] = pub.path.str.replace(str(ROOT) + "/", "", regex=False)
    return R, [pub]


def write(name, rows, raws, meta):
    raw = pd.concat(raws, ignore_index=True) if raws else pd.DataFrame()
    if "path" in raw:
        raw["path"] = raw.path.astype(str).str.replace(str(ROOT) + "/", "", regex=False)
    for c in ("ref", "speaker"):
        if c in raw:
            raw[c] = raw[c].astype(str).str.replace(str(ROOT) + "/", "", regex=False)
    C.write(name, rows, raw, meta)
    print("wrote", name, len(rows), "rows")


def meta_common():
    return dict(systems={k: {x: y for x, y in v.items() if x != "load"} for k, v in SYSTEMS.items()},
                rate_levels=RATE_LEVELS, pitch_levels=PITCH_LEVELS,
                bootstrap="1000 resamples (utterances; cluster (voice,text) for the steer probe), percentile 95% CI",
                code="steer/paper/baselines_ext.py + baselines_ext_eval.py")


def cmd_analyze():
    hyps, dl2 = asr_cache("asr_v3.jsonl"), asr_cache("asr_dl2.jsonl")
    ut = json.load(open(SC / "utmos.json")) if (SC / "utmos.json").exists() else {}
    dn = json.load(open(SC / "dnsmos.json")) if (SC / "dnsmos.json").exists() else {}
    sims = {}
    for f in SC.glob("sim_*.out"):
        sims.update(read_sim(f))
    R, raws = [], []
    src = f"{C.RES}/baselines_quality_raw.csv"
    for s in list(SYSTEMS) + list(PSEUDO):
        for st in ("libritts",) + (("clone",) if cfg(s)["voice"] == "clone" else ()):
            r, d = quality_rows(s, st, hyps, ut, dn, sims, src)
            R += r; raws += d
    # DNSMOS of our X4 renders + ground truth on identical audio (WER/UTMOS/SIM for these come from quality_rescore.json)
    for (s, st), ps in ours_paths().items():
        v = [dn[p]["ovrl"] for p in ps if p in dn]
        if v:
            R.append(C.row("dnsmos_ovrl", C.boot_ci(v, lambda u: float(np.mean(u))), len(v), src, "DNSMOS P.835 OVRL (speechmos, ONNX)",
                           {"libritts": "libritts_r_test_clean_bench500", "clone": "libritts_r_test_clean_clone499"}[st],
                           system=s, group="ours" if s != "ground_truth" else "reference", condition=st))
    write("baselines_quality", R, raws, dict(meta_common(), note="WER/UTMOS/SIM scorers identical to quality_rescore (X4). "
                                             "Cloning systems on libritts: reference = a different real clip of the same test speaker."))
    R, raws = [], []
    for s in SYSTEMS:
        r, d = psc_rows(s, hyps, dl2, ut, dn, f"{C.RES}/baselines_psc_raw.csv")
        R += r; raws += d
    write("baselines_psc", R, raws, dict(meta_common(), note="ASR judge: Whisper-large-v3 fp16 + Whisper EnglishTextNormalizer "
          "(wer_v3; distil-large-v2 kept as wer_dl2 for the old table); same gender/pitch/rate estimator and calibration as "
          "steer.benchmark. Subsets: all 246, voxceleb (source==voxceleb), no_leak (excludes Expresso-sourced prompts that "
          "overlap SteerTTS training). Prompt-blind systems get the oracle gender only."))
    R = []
    for s in SYSTEMS:
        R += seedtts_rows(s)
    write("baselines_seedtts", R, [], meta_common())
    R, raws = [], []
    for s in SYSTEMS:
        r, d = steer_rows(s, hyps)
        R += r; raws += d
    write("baselines_steer", R, raws, dict(meta_common(), delta="knob render minus the knob-neutral render of the same "
                                           "voice/text, in SteerTTS training pooled SD (control_stats.json of runs/spk_enc_v3)",
                                           hit_rate="share of units with sign(level)*delta > 0.5 z (as X1)"))


def cmd_knobs():
    """results/baselines_native_knobs.json: native-knob dose-response for every system with probe measurements, plus
    the matching SteerTTS rows of results/steer_heldout.json (ours extractor) for side-by-side reading."""
    hyps = asr_cache("asr_v3.jsonl")
    R, raws, units = [], [], {}
    for s in SYSTEMS:
        r, d = steer_rows(s, hyps)
        R += r; raws += d
        if d:
            units[s] = dict(n_units=int(d[0][["vid", "text_id"]].drop_duplicates().shape[0]),
                            voices=int(d[0].vid.nunique()), texts=int(d[0].text_id.nunique()))
    ours = [dict(x, system="steertts_base", group="ours") for x in load_json(C.RES / "steer_heldout.json").get("rows", [])
            if x.get("extractor") == "ours" and x.get("target_dim") in ("rate", "f0_med") and x.get("condition") == "single_control"
            and x["metric"] in ("spearman_rho", "slope_dz_per_reqz", "mean_delta_z", "hit_rate", "direction_acc",
                                "leak_mean_abs_dz_other")]
    write("baselines_native_knobs", R + ours, raws, dict(
        meta_common(), units=units,
        units_note="native-knob baselines: default-voice systems = 2 default voices (F/M) x 8 harness texts = 16 "
                   "(voice,text) units per level; cloning systems = 16 held-out LibriTTS-R voices x 8 texts = 128. "
                   "SteerTTS rows (system=steertts_base, copied from steer_heldout.json): 16 held-out voices x 8 texts "
                   "= 128 units per level.",
        gain="slope_dz_per_level: OLS slope through 0 of measured delta z on the requested level in {-2,-1,1,2}, "
             "knob values z-matched (rate_levels/pitch_levels); ours: slope_dz_per_reqz",
        leak="mean |delta z| over the other 11 control dims (steer.features.measure)",
        delta="knob render minus the knob-neutral render of the same voice/text, in SteerTTS training pooled SD"))


def load_json(f):
    return json.load(open(f)) if Path(f).exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "score", "dnsmos", "seedtts", "measure", "analyze", "knobs", "copysyn", "table"])
    ap.add_argument("--system")
    ap.add_argument("--systems", default="")
    ap.add_argument("--sets", default="libritts")
    a = ap.parse_args()
    systems = list(SYSTEMS) + list(PSEUDO) if a.systems == "all" else [x for x in a.systems.split(",") if x]
    sets = a.sets.split(",")
    if a.cmd == "plan":
        cmd_plan(a.system, sets)
    elif a.cmd == "score":
        cmd_score(systems, sets)
    elif a.cmd == "dnsmos":
        cmd_dnsmos(systems)
    elif a.cmd == "seedtts":
        cmd_seedtts(a.system)
    elif a.cmd == "measure":
        cmd_measure(systems)
    elif a.cmd == "analyze":
        cmd_analyze()
    elif a.cmd == "knobs":
        cmd_knobs()
    elif a.cmd == "copysyn":
        cmd_copysyn()
    else:
        from .baselines_ext_table import main as table
        table()


if __name__ == "__main__":
    sys.exit(main())
