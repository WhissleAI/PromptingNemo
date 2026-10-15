"""Paired bootstrap over the 40 end-to-end prompts (10k resamples, seed 0) of per-prompt mean hit, adapter minus each
other mapper (same method as results/adapter_e2e_paired.json).

python -m steer.paper.adapter_e2e_paired [--src adapter_e2e_ext.json] -> paper/results/adapter_e2e_paired_ext.json
"""
import argparse
import json

import numpy as np

from .common import RES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="adapter_e2e_ext.json")
    ap.add_argument("--out", default="adapter_e2e_paired_ext.json")
    a = ap.parse_args()
    d = json.load(open(RES / a.src))
    pp = d["per_prompt"]
    ids = sorted(pp["adapter"])
    rows = []
    for c in pp:
        v = [pp[c][i]["hit"] for i in ids if i in pp[c] and pp[c][i]["hit"] is not None]
        rows.append(dict(metric="hit_prompt_mean", system=c, value=round(float(np.mean(v)), 5), n=len(v)))
    for c in pp:
        if c == "adapter":
            continue
        common = [i for i in ids if i in pp[c] and pp[c][i]["hit"] is not None and pp["adapter"][i]["hit"] is not None]
        diff = np.array([pp["adapter"][i]["hit"] - pp[c][i]["hit"] for i in common])
        rng = np.random.default_rng(0)
        b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
        rows.append(dict(metric=f"hit_diff_adapter_minus_{c}", value=round(float(diff.mean()), 5),
                         ci_lo=round(float(np.percentile(b, 2.5)), 5), ci_hi=round(float(np.percentile(b, 97.5)), 5),
                         p_le_0=round(float((b <= 0).mean()), 4), n=len(diff)))
        print(rows[-1], flush=True)
    json.dump(dict(name="adapter_e2e_paired_ext", meta=dict(source=f"results/{a.src} per_prompt",
              method="paired bootstrap over prompts (10k resamples, seed 0) of per-prompt mean hit, adapter minus mapper"),
              rows=rows), open(RES / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
