"""End-to-end language -> acoustics: hand-written directions rendered by SteerTTS on held-out voices, read back.

40 prompts with gold dimensions (fixed before any run: the 24 single-dim items, 12 attribute-only compositions and 4
distractor items of adapter_handwritten_gold_v2.json) x 4 held-out LibriTTS-R test-clean voices (2 F, 2 M; the
H01/H02/H09/H10 of common.heldout_voices(16), cloned from one real clip) x 2 neutral harness texts, under
  adapter  production adapter (data/meta/prompt_adapter.pt) -> tag + specified controls
  llm      qwen2.5:3b few-shot JSON (steer.paper.adapter_llm cache) -> top tag + controls
  oracle   gold dims at +-1.5 z, tag none (what perfect language understanding would request)
  base     no tag, nothing specified (one per voice x text)
Readback: steer.features.measure (common.ours); dz = (render - base render) / pooled within-speaker SD of the
checkpoint's training data. Per gold dim: hit = sign * dz > 0.5; leak = mean |dz| over dims outside spec + ok.

python -m steer.paper.adapter_e2e [--stage render|analyze|all]  (CPU)  -> paper/results/adapter_e2e.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ..config import CONTROL_DIMS, SR
from ..harness import TEXTS
from . import common as C

OUT = C.WORK / "adapter_e2e"
VIDS = ["H01", "H02", "H09", "H10"]
TXT = [0, 2]
ORACLE = 1.5


def prompts():
    g = json.load(open(C.RES / "adapter_handwritten_gold_v2.json"))["items"]
    pick = [it for it in g if it["cat"] == "dim"] + \
           [it for it in g if it["cat"] == "combo" and it["ok"] != "*"] + \
           [it for it in g if it["cat"] == "distractor" and it["spec"]][:4]
    return pick


def requests(items):
    from ..prompt import PromptMapper
    pm = PromptMapper(device="cpu")
    llm_f = C.WORK / "llm_qwen2.5_3b.json"
    llm = json.load(open(llm_f)) if llm_f.exists() else {}
    R = {}
    for it in items:
        r = pm(it["prompt"])
        R[(it["id"], "adapter")] = (r["top_tags"][0][0], r["ctrl"])
        R[(it["id"], "oracle")] = ("none", {d: ORACLE * s for d, s in it["spec"].items()})
        if it["prompt"] in llm:
            l = llm[it["prompt"]]
            R[(it["id"], "llm")] = (l["tags"][0], {k: float(np.clip(v, -5, 5)) for k, v in l["controls"].items()})
    return R


def render(items, R):
    torch.set_num_threads(12)
    V = {v["vid"]: v for v in C.heldout_voices(16)}
    s = C.make_synth(C.BASE, device="cpu")
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    n = 0
    for ti in TXT:  # all voices on the first text before the second (partial runs stay balanced)
        for vid in VIDS:
            jobs = [("base", "base", "none", None)] + [(iid, cond, *R[(iid, cond)]) for (iid, cond) in R]
            for iid, cond, tag, ctrl in jobs:
                p = OUT / "wav" / f"{vid}_{ti}_{iid}_{cond}.wav"
                if p.exists():
                    continue
                w = s(TEXTS[ti], f"heldout_{vid}", tag=tag, ctrl=ctrl or None, ref_wav=V[vid]["ref"])
                sf.write(p, w, SR); n += 1
                if n % 100 == 0:
                    print("rendered", n, flush=True)
    print("rendered", n)


def analyze(items, R):
    sd = json.load(open(Path(C.BASE).parent / "control_stats.json"))["pooled_sd"]
    sdv = np.array([sd[d] for d in CONTROL_DIMS])
    meas_f = OUT / "measure.json"
    M = json.load(open(meas_f)) if meas_f.exists() else {}
    for p in sorted((OUT / "wav").glob("*.wav")):
        if p.name not in M:
            y, _ = sf.read(p, dtype="float32")
            ti = int(p.name.split("_")[1])
            M[p.name] = C.ours(y, TEXTS[ti])
    json.dump(M, open(meas_f, "w"))
    vec = lambda k: np.array([M[k][d] for d in CONTROL_DIMS], float)
    conds = sorted({c for (_, c) in R})
    res = dict(n_prompts=len(items), voices=VIDS, texts=TXT, oracle_z=ORACLE, conditions={}, per_prompt={})
    for cond in conds:
        hits, leaks, exact, spec_ok, per = [], [], [], [], {}
        h_req, h_unreq = [], []  # per gold-dim unit: requested with the right sign vs not
        for it in items:
            if (it["id"], cond) not in R:
                continue
            gold = {CONTROL_DIMS.index(d): s for d, s in it["spec"].items()}
            ok = set(gold) | ({CONTROL_DIMS.index(d) for d in it["ok"]} if it["ok"] != "*" else set(range(12)))
            other = [k for k in range(12) if k not in ok]
            req = R[(it["id"], cond)][1]
            ph, pl = [], []
            for vid in VIDS:
                for ti in TXT:
                    k = f"{vid}_{ti}_{it['id']}_{cond}.wav"; b = f"{vid}_{ti}_base_base.wav"
                    if k not in M or b not in M:
                        continue
                    dz = (vec(k) - vec(b)) / sdv
                    h = [float(s * dz[j] > 0.5) if np.isfinite(dz[j]) else np.nan for j, s in gold.items()]
                    ph += h
                    for (j, sg), hv in zip(gold.items(), h):
                        d = CONTROL_DIMS[j]
                        (h_req if (d in req and np.sign(req[d]) == sg) else h_unreq).append(hv)
                    pl.append(np.nanmean(np.abs(dz[other])) if other else np.nan)
                    exact.append(float(np.all(np.array(h) == 1)))
            hits += ph; leaks += pl
            spec_ok.append(float(all(d in req and np.sign(req[d]) == s for d, s in it["spec"].items())))
            per[it["id"]] = dict(prompt=it["prompt"], request=dict(tag=R[(it["id"], cond)][0], ctrl=req),
                                 hit=round(float(np.nanmean(ph)), 3) if ph else None)
        hits = np.array(hits, float); leaks = np.array(leaks, float)
        rng = np.random.default_rng(0)
        hb = [np.nanmean(rng.choice(hits, len(hits))) for _ in range(1000)]
        res["conditions"][cond] = dict(
            hit=round(float(np.nanmean(hits)), 4), hit_ci=[round(float(np.percentile(hb, q)), 4) for q in (2.5, 97.5)],
            n_dim_units=int(np.isfinite(hits).sum()), leak_mean_abs_dz=round(float(np.nanmean(leaks)), 4),
            all_gold_dims_hit=round(float(np.mean(exact)), 4), request_has_all_gold_dims=round(float(np.mean(spec_ok)), 4),
            hit_when_requested=round(float(np.nanmean(h_req)), 4) if h_req else None, n_requested=len(h_req),
            hit_when_not_requested=round(float(np.nanmean(h_unreq)), 4) if h_unreq else None, n_not_requested=len(h_unreq))
        res["per_prompt"][cond] = per
        print(cond, res["conditions"][cond], flush=True)
    json.dump(res, open(C.RES / "adapter_e2e.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    items = prompts()
    R = requests(items)
    print(len(items), "prompts;", len(R), "requests", flush=True)
    if a.stage in ("render", "all"):
        render(items, R)
    if a.stage in ("analyze", "all"):
        analyze(items, R)


if __name__ == "__main__":
    main()
