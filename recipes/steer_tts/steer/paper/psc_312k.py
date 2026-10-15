"""Reviewer follow-up (d2): ParaSpeechCaps 246-prompt benchmark (steer/benchmark.py protocol: prompt adapter ->
Produced: results/psc_312k.json + _raw.csv: python -m steer.paper.psc_312k --stage run (queue job 06c_x4_psc_312k.sh), then --stage analyze. All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.
tag + controls, benchmark voice pools, attribute classifiers calibrated on real speech) for the deployed 312.5k base,
with Whisper-large-v3 as the single ASR judge, overall and by source (VoxCeleb-only = excludes Expresso/EARS prompts).

python -m steer.paper.psc_312k --stage run      (GPU; via the queue)
python -m steer.paper.psc_312k --stage analyze  -> results/psc_312k.json (+ _raw.csv)
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import common as C

NAME = "psc_312k"
SYS = "paper_psc_312k"


DEDUP_IDS = C.Path("/tmp/claude-1000/-home-karan-Desktop-whissle/dbda8e1e-5067-4908-8376-8ea786d4d17e/scratchpad/dedup/ids.json")


def dedup_flags():
    """Per bench-psc row: contaminated (audio of the same utterance in training, per steer.paper.dedup = results/
    dedup.json: Expresso read rows by clip id, Expresso conversational rows whose speech is inside an NVTTS training
    segment, VoxCeleb rows whose video+segment is in NVTTS) and speaker_in_training (VoxCeleb speakers also in NVTTS;
    all Expresso/EARS speakers are training speakers)."""
    import hashlib
    from pathlib import Path
    anon = lambda x, k: f"{k}-{hashlib.sha1(('steertts-dedup:' + str(x)).encode()).hexdigest()[:8]}"
    t = pd.read_parquet(C.ROOT / "data/raw/paraspeechcaps/data/test-00000-of-00001.parquet")
    b = pd.read_json(C.ROOT / "data/bench/psc.jsonl", lines=True)
    # bench psc.jsonl is the PSC test parquet in the same row order (checked: source, speaker, transcript all equal);
    # text keys are not unique (5 duplicates), so align by position
    assert len(t) == len(b) and (t.source.values == b.source.values).all() and \
        (t.speakerid.astype(str).values == b.speakerid.astype(str).values).all()
    rel = t.relative_audio_path.tolist()
    P = json.load(open(DEDUP_IDS))["psc_test"]
    conv = {h["psc"] for h in P["expresso"]["conversational_examples"]}
    vox = {h["psc"] for h in P["voxceleb"]["same_segment_examples"]}
    meta = pd.read_parquet(C.ROOT / "data/meta/controls.parquet", columns=["id", "corpus"])
    nvspk = set(meta[(meta.corpus == "nvtts") & meta.id.str.startswith("nvtts_id")].id.str[6:13])
    flags = []
    for r, p in zip(b.itertuples(), rel):
        a = anon(p, "psc")
        read = r.source == "expresso" and "conversational" not in p
        cont = read or a in conv or a in vox
        spk_tr = True if r.source in ("expresso", "ears") else p.split("/")[-3] in nvspk
        flags.append(dict(contaminated=bool(cont), speaker_in_training=bool(spk_tr)))
    f = pd.DataFrame(flags)
    exp = json.load(open(C.RES / "dedup.json"))["sets"]["paraspeechcaps_test"]["contaminated_rows"]  # 32
    assert int(f.contaminated.sum()) == exp, (int(f.contaminated.sum()), exp)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="run")
    a = ap.parse_args()
    from .. import benchmark as B
    out = B.OUT / f"{SYS}.json"
    if a.stage == "run":
        calib = json.load(open(B.BENCH / "calibration.json"))
        sets = {"psc": pd.read_json(B.BENCH / "psc.jsonl", lines=True)}
        bk = B.META / "backup_287spk" / "control_stats.json"
        stats = json.load(open(bk if bk.exists() else B.META / "control_stats.json"))
        pools, psc_pools, refs = B.voices(stats)
        res = B.synthesize(SYS, C.BASE, sets, pools, psc_pools, calib)
        text = sets["psc"].text.tolist()
        items = [(row, row["wav"], text[row["i"]]) for row in res["psc"]]
        B.score_asr(items, "openai/whisper-large-v3", "v3")
        B.score_utmos(items)
        res["summary"] = B.summarize(res)
        json.dump(res, open(out, "w"), indent=1, default=float)
    if a.stage == "analyze":
        res = json.load(open(out))
        df = pd.DataFrame(res["psc"])
        fl = dedup_flags()
        df = df.join(fl, on="i")
        src = f"{C.RES}/{NAME}_raw.csv"
        R = []
        groups = {"all": df, "voxceleb_only": df[df.source == "voxceleb"], "expresso": df[df.source == "expresso"],
                  "ears": df[df.source == "ears"], "no_leak": df[~df.leak.astype(bool)],
                  "clean214": df[~df.contaminated], "voxceleb_unseen_speakers": df[(df.source == "voxceleb") & ~df.speaker_in_training]}
        for g, s in groups.items():
            if s.empty:
                continue
            cond = dict(condition="psc_prompted", subset=g, system="ours_base_312.5k",
                        speaker_overlap="none" if g == "voxceleb_unseen_speakers" else
                        ("partial (22/54 VoxCeleb speakers in NonverbalTTS)" if g == "voxceleb_only" else "yes/partial"))
            R.append(C.row("wer", C.boot_ci(list(zip(s.err_v3, s.nwords)), lambda u: sum(x for x, _ in u) / sum(y for _, y in u)),
                           len(s), src, "whisper-large-v3 fp16 + whisper EnglishTextNormalizer (steer.benchmark.score_asr)",
                           "paraspeechcaps_bench246", **cond))
            R.append(C.row("utmos", C.boot_ci(list(s.utmos), lambda u: float(np.mean(u))), len(s), src, "SpeechMOS utmos22_strong",
                           "paraspeechcaps_bench246", **cond))
            for k in ("gender_ok", "pitch_ok", "rate_ok"):
                R.append(C.row(k.replace("_ok", "_acc"), C.boot_ci(list(s[k].astype(float)), lambda u: float(np.mean(u))), len(s),
                               src, "steer.benchmark calibrated attribute bins (calibration.json)", "paraspeechcaps_bench246", **cond))
        pub = df.drop(columns=[c for c in ("voice", "wav") if c in df])
        C.write(NAME, R, pub, dict(ckpt=C.BASE, protocol="steer/benchmark.py psc set, voice pools from backup_287spk stats",
                                   note="ParaSpeechCaps published Parler WER uses distil-whisper; here Whisper-large-v3 only. "
                                        "clean214 = rows results/dedup.json marks clean (no same-utterance audio in training); "
                                        "voxceleb_only is NOT speaker-clean (22/54 speakers in NonverbalTTS); "
                                        "voxceleb_unseen_speakers = VoxCeleb rows whose speaker is in no training corpus",
                                   n_contaminated=int(df.contaminated.sum()), n_clean=int((~df.contaminated).sum())))


if __name__ == "__main__":
    main()
