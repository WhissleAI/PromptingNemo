"""X3: the style/emotion TAG channel on HELD-OUT voices.
Produced: results/tags_heldout.json, _raw.csv, tags_heldout_ears_real_raw.csv: python -m steer.paper.x3_tags_heldout --stage render|measure|score|analyze (runs/paper/chain34.sh, chain34b.sh). All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.

Every tag in the base checkpoint (49, excluding 'none') x 16 test-clean voices x 8 texts, no controls, guidance 1.
Scored with (a) emotion2vec+ large (9-class SER, funasr in the seedeval env), (b) audeering wav2vec2 MSP-dim
arousal/dominance/valence regressor, (c) acoustic deltas vs the uninstructed render (ours + Praat), (d) Whisper WER.
Scorer validity check: emotion2vec+ on REAL EARS recordings of the same emotion labels (training-speaker audio, used
only to calibrate the scorer, never as a system).

python -m steer.paper.x3_tags_heldout [--stage render|measure|score|analyze|all]
"""
import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from ..config import CONTROL_DIMS, SR
from ..harness import TEXTS
from . import common as C

NAME = "tags_heldout"
OUT = C.WORK / "x3"
X1 = C.WORK / "x1"
E2V = {"anger": "angry", "irritation": "angry", "frustration": "angry", "disgust": "disgusted", "contempt": "disgusted",
       "fear": "fearful", "panic": "fearful", "anxiety": "fearful", "distress": "fearful", "happiness": "happy",
       "excited": "happy", "ecstasy": "happy", "amusement": "happy", "sadness": "sad", "disappointment": "sad",
       "amazement": "surprised", "neutral": "neutral"}
E2V_CORE = ["anger", "disgust", "fear", "happiness", "sadness", "amazement"]
EARS_REAL = ["neutral", "anger", "disgust", "fear", "sadness", "amazement", "ecstasy", "amusement", "distress",
             "disappointment"]
# model-based scoring (emotion2vec+, A/V/D, Whisper) on the emotion tags with an emotion2vec+ class + whisper; acoustic
# deltas for all tags (shared-GPU time budget)
SCORED = set(E2V) | {"whisper"}
SEEDEVAL_PY = "/home/karan/anaconda3/envs/seedeval/bin/python"


def tags():
    st = json.load(open(C.BASE.rsplit("/", 1)[0] + "/control_stats.json"))
    return [t for t in st["tags"] if t != "none"]


def plan(V):
    items = []
    for v in V:
        for ti, text in enumerate(TEXTS):
            items.append(dict(vid=v["vid"], text_id=ti, text=text, tag="baseline",
                              path=str(X1 / "wav" / f"{v['vid']}_{ti}_baseline_+0.wav")))
            for t in tags():
                items.append(dict(vid=v["vid"], text_id=ti, text=text, tag=t,
                                  path=str(OUT / "wav" / f"{v['vid']}_{ti}_{t}.wav")))
    return items


def ears_real(n_per=40, seed=0):
    rows = [json.loads(l) for l in open(C.ROOT / "data/meta/manifest_ears.jsonl")]
    rng = np.random.default_rng(seed)
    out = []
    for t in EARS_REAL:
        rs = [r for r in rows if r["tag"] == t]
        for k in rng.choice(len(rs), min(n_per, len(rs)), replace=False):
            out.append(dict(tag=t, path=rs[k]["wav"]))
    return out


class AVD(torch.nn.Module):
    """audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim: mean-pooled wav2vec2 + regression head
    -> (arousal, dominance, valence) in ~[0,1]."""

    def __init__(self, dev="cuda"):
        super().__init__()
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        self.proc = Wav2Vec2Processor.from_pretrained(name)
        self.w2v = Wav2Vec2Model.from_pretrained(name)
        H = self.w2v.config.hidden_size
        self.dense, self.out = torch.nn.Linear(H, H), torch.nn.Linear(H, 3)
        try:
            sd = load_file(hf_hub_download(name, "model.safetensors"))
        except Exception:
            sd = torch.load(hf_hub_download(name, "pytorch_model.bin"), map_location="cpu")
        self.dense.load_state_dict({"weight": sd["classifier.dense.weight"], "bias": sd["classifier.dense.bias"]})
        self.out.load_state_dict({"weight": sd["classifier.out_proj.weight"], "bias": sd["classifier.out_proj.bias"]})
        self.to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, y16):
        x = self.proc(y16, sampling_rate=16000, return_tensors="pt").input_values.to(self.dev)
        h = self.w2v(x).last_hidden_state.mean(1)
        return self.out(torch.tanh(self.dense(h)))[0].float().cpu().numpy()  # arousal, dominance, valence


def score_avd(paths, cache):
    import soxr
    res = json.load(open(cache)) if cache.exists() else {}
    todo = [p for p in paths if p not in res]
    if todo:
        m = AVD()
        for i, p in enumerate(todo):
            y, sr = sf.read(p, dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            res[p] = m(soxr.resample(y, sr, 16000) if sr != 16000 else y).tolist()
            if i % 1000 == 0:
                print(f"avd {i}/{len(todo)}", flush=True)
        del m
        torch.cuda.empty_cache()
        json.dump(res, open(cache, "w"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    V = C.heldout_voices(16)
    items = plan(V)
    ref = {v["vid"]: v["ref"] for v in V}
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("render", "all"):
        syn = C.make_synth(C.BASE)
        (OUT / "wav").mkdir(parents=True, exist_ok=True)
        for i, it in enumerate(items):
            if it["tag"] != "baseline":
                sf.write(it["path"], syn(it["text"], f"heldout_{it['vid']}", tag=it["tag"], ref_wav=ref[it["vid"]]), SR)
            if i % 1000 == 0:
                print(f"render {i}/{len(items)}", flush=True)
    if a.stage in ("measure", "all"):
        M = C.measure_many([(it["path"], it["path"], it["text"]) for it in items if it["tag"] != "baseline"])
        M1 = pd.read_parquet(X1 / "measures.parquet")
        pd.concat([M, M1[M1.key.str.contains("_baseline_")]], ignore_index=True).to_parquet(OUT / "measures.parquet")
    if a.stage in ("score", "all"):
        paths = [it["path"] for it in items if it["tag"] in SCORED or it["tag"] == "baseline"]
        er = ears_real()
        json.dump(er, open(OUT / "ears_real.json", "w"))
        lst = OUT / "e2v_list.txt"
        lst.write_text("\n".join(paths + [r["path"] for r in er]) + "\n")
        C.gpu_wait()
        subprocess.run([SEEDEVAL_PY, str(C.Path(__file__).with_name("e2v_score.py")), str(lst), str(OUT / "e2v.jsonl"), os.environ.get("PAPER_E2V_BATCH", "32")],
                       check=True, cwd=str(OUT))
        C.gpu_wait()
        score_avd(paths + [r["path"] for r in er], OUT / "avd.json")
        hyps = C.transcribe(paths, X1 / "asr_cache.jsonl")  # last: slowest on a shared GPU
        json.dump(hyps, open(OUT / "hyps.json", "w"))
    if a.stage in ("analyze", "all"):
        analyze(items)


def analyze(items):
    stats = json.load(open(C.BASE.rsplit("/", 1)[0] + "/control_stats.json"))
    sd = pd.Series(stats["pooled_sd"])[CONTROL_DIMS]
    sd_gt = pd.read_csv(X1 / "pooled_sd_gt.csv", index_col=0).iloc[:, 0]
    M = pd.read_parquet(OUT / "measures.parquet").drop_duplicates("key").set_index("key")
    hyps = json.load(open(OUT / "hyps.json"))
    e2v = {}
    for l in open(OUT / "e2v.jsonl"):
        d = json.loads(l)
        e2v[d["path"]] = dict(zip(d["labels"], d["scores"]))
    avd = json.load(open(OUT / "avd.json"))
    labs = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    df = pd.DataFrame(items).join(M, on="path")
    df["scored"] = df.tag.isin(SCORED | {"baseline"})
    df["hyp"] = df.path.map(hyps)
    ew = [C.word_errors(t, h) if isinstance(h, str) else (np.nan, np.nan) for t, h in zip(df.text, df.hyp)]
    df["err"], df["nwords"] = [e for e, _ in ew], [n for _, n in ew]
    miss = [p for p, sc in zip(df.path, df.scored) if sc and (p not in e2v or p not in avd or p not in hyps)]
    assert not miss, f"{len(miss)} scored clips lack scores"
    for l in labs:
        df[f"e2v_{l}"] = [e2v[p].get(l, np.nan) if p in e2v else np.nan for p in df.path]
    df["e2v_top"] = [max(labs, key=lambda l: e2v[p].get(l, 0)) if p in e2v else None for p in df.path]
    for k, n in enumerate(["arousal", "dominance", "valence"]):
        df[n] = [avd[p][k] if p in avd else np.nan for p in df.path]
    base = df[df.tag == "baseline"].set_index(["vid", "text_id"])
    idx = pd.MultiIndex.from_arrays([df.vid, df.text_id])
    for d in CONTROL_DIMS:
        df[f"dz_{d}"] = (df[d].values - base[d].reindex(idx).values) / sd[d]
        f = C.PRAAT_MAP[d]
        df[f"dp_{d}"] = (df[f].values - base[f].reindex(idx).values) / sd_gt[f]
    for c in [f"e2v_{l}" for l in labs] + ["arousal", "dominance", "valence"]:
        df[f"d_{c}"] = df[c].values - base[c].reindex(idx).values
    src = f"{C.RES}/{NAME}_raw.csv"
    split = "libritts_r_test_clean_heldout16"
    U = ["vid", "text_id"]
    R = []
    bsub = df[df.tag == "baseline"]
    for t in ["baseline"] + tags():
        sub = df[df.tag == t]
        cond = dict(condition="tag", tag=t)
        scored = t in SCORED or t == "baseline"
        if scored:
            R.append(C.row("wer", C.boot_ratio(sub, U, "err", "nwords"), len(sub), src,
                           "whisper-large-v3 fp16 + whisper EnglishTextNormalizer", split, **cond))
            for l in labs:
                R.append(C.row(f"e2v_top1_share_{l}", C.boot_ratio(sub, U, (sub.e2v_top == l).astype(float)), len(sub), src,
                               "emotion2vec_plus_large", split, **cond))
            for n in ["arousal", "dominance", "valence"]:
                R.append(C.row(n, C.boot_ratio(sub, U, n), len(sub), src,
                               "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", split, **cond))
        if t == "baseline":
            continue
        if t in E2V:
            lab = E2V[t]
            R.append(C.row("e2v_target_prob_delta", C.boot_ratio(sub, U, f"d_e2v_{lab}"), len(sub), src,
                           "emotion2vec_plus_large", split, e2v_label=lab, **cond))
            R.append(C.row("e2v_target_top1", C.boot_ratio(sub, U, (sub.e2v_top == lab).astype(float)), len(sub), src,
                           "emotion2vec_plus_large", split, e2v_label=lab, **cond))
            R.append(C.row("e2v_target_top1_baseline", C.boot_ratio(bsub, U, (bsub.e2v_top == lab).astype(float)), len(bsub),
                           src, "emotion2vec_plus_large", split, e2v_label=lab, **cond))
        if scored:
            for n in ["arousal", "dominance", "valence"]:
                R.append(C.row(f"delta_{n}", C.boot_ratio(sub, U, f"d_{n}"), len(sub), src,
                               "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", split, **cond))
        for d in CONTROL_DIMS:
            R.append(C.row("mean_delta_z", C.boot_ratio(sub, U, f"dz_{d}"), len(sub), src,
                           "steer.features.measure", split, extractor="ours", measured_dim=d, **cond))
            R.append(C.row("mean_delta_z", C.boot_ratio(sub, U, f"dp_{d}"), len(sub), src,
                           f"praat:{C.PRAAT_MAP[d]}", split, extractor="praat", measured_dim=d, **cond))
        R.append(C.row("acoustic_norm_l2_dz", float(np.sqrt(np.nanmean(sub[[f"dz_{d}" for d in CONTROL_DIMS]].values ** 2, 0).sum())),
                       len(sub), src, "steer.features.measure", split, extractor="ours", **cond))
    # scorer validity on real EARS emotional speech
    er = json.load(open(OUT / "ears_real.json"))
    E = pd.DataFrame(er)
    E["e2v_top"] = [max(labs, key=lambda l: e2v[p].get(l, 0)) for p in E.path]
    for k, n in enumerate(["arousal", "dominance", "valence"]):
        E[n] = [avd[p][k] for p in E.path]
    for t, g in E.groupby("tag"):
        lab = E2V.get(t)
        cond = dict(condition="real_ears_scorer_check", tag=t)
        if lab:
            R.append(C.row("e2v_target_top1", C.boot_ci([x for x in (g.e2v_top == lab).astype(float)], lambda u: float(np.mean(u))),
                           len(g), src, "emotion2vec_plus_large", "ears_real_training_speakers", e2v_label=lab, **cond))
        for n in ["arousal", "dominance", "valence"]:
            R.append(C.row(n, C.boot_ci(list(g[n]), lambda u: float(np.mean(u))), len(g), src,
                           "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", "ears_real_training_speakers", **cond))
    E.assign(path=E.path.str.replace(str(C.ROOT) + "/", "", regex=False)).to_csv(C.RES / f"{NAME}_ears_real_raw.csv", index=False)
    pub = df.drop(columns=["path"]).assign(wav=df.path.str.replace(str(C.ROOT) + "/", "", regex=False))
    meta = dict(ckpt="runs/spk_enc_v3/step312500.pt", n_tags=len(tags()), e2v_map=E2V,
                note="EARS real clips are training-speaker recordings used ONLY to check the scorers recognise the "
                     "emotion on real speech; tag renders are on held-out voices.")
    C.write(NAME, R, pub, meta)


if __name__ == "__main__":
    main()
