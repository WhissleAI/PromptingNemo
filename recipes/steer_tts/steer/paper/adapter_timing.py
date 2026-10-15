"""Latency per direction: production prompt adapter (MiniLM encoder + 0.5M head, steer.prompt.PromptMapper) vs the
local LLM mappers (Ollama, CPU only), on the 161 hand-written directions + intensity probes.

Adapter: measured here, per prompt, batch 1, after 5 warm-up calls, at torch threads in --threads.
LLM: read from the per-call timings stored in the caches by steer.paper.adapter_llm (wall time of the HTTP call and
Ollama's own prompt-eval / eval durations); the first call of a run (model load) is excluded.
Load average and CPU model are recorded because the machine was shared.

python -m steer.paper.adapter_timing  -> paper/results/adapter_timing.json
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

from .common import RES, WORK
from .adapter_llm import all_prompts

CACHES = {"qwen2.5-3b (4-shot)": "llm_qwen2.5_3b.json", "qwen2.5-3b calibrated": "llm_qwen2.5_3b_cal.json",
          "qwen2.5-7b-instruct (4-shot)": "llm_qwen2.5_7b-instruct.json",
          "qwen2.5-7b-instruct calibrated": "llm_qwen2.5_7b-instruct_cal.json"}


def cpu_name():
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def stats(x):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), median_s=round(float(np.median(x)), 4), mean_s=round(float(np.mean(x)), 4),
                p90_s=round(float(np.percentile(x, 90)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,4")
    a = ap.parse_args()
    from ..prompt import PromptMapper
    ps = all_prompts()
    res = dict(device="cpu", cpu=cpu_name(), n_cpu=os.cpu_count(), load_avg_start=os.getloadavg(), adapter={}, llm={})
    for th in [int(t) for t in a.threads.split(",")]:
        torch.set_num_threads(th)
        t0 = time.time(); pm = PromptMapper(device="cpu"); load_s = time.time() - t0
        for p in ps[:5]:
            pm(p)
        ts = []
        for p in ps:
            t0 = time.perf_counter(); pm(p); ts.append(time.perf_counter() - t0)
        res["adapter"][f"threads_{th}"] = dict(load_s=round(load_s, 2), **stats(ts), load_avg=os.getloadavg())
        print("adapter", th, res["adapter"][f"threads_{th}"], flush=True)
    for name, fn in CACHES.items():
        f = WORK / fn
        if not f.exists():
            continue
        c = json.load(open(f))
        tim = [v["timing"] for v in c.values() if v.get("timing")]
        if len(tim) < 2:
            res["llm"][name] = dict(note="no per-call timings stored (run before timing was recorded)")
            continue
        tim = tim[1:]  # drop the model-load call
        tim = [t for t in tim if t.get("num_thread", 8) == 4] or tim  # 4-thread calls when available
        wall = [t["wall_s"] for t in tim]
        ev = [t["eval_count"] / (t["eval_duration"] / 1e9) for t in tim if t.get("eval_duration")]
        pe = [t["prompt_eval_count"] / (t["prompt_eval_duration"] / 1e9) for t in tim
              if t.get("prompt_eval_duration") and t.get("prompt_eval_count")]
        res["llm"][name] = dict(**stats(wall), gen_tokens_median=float(np.median([t["eval_count"] for t in tim])),
                                gen_tok_per_s_median=round(float(np.median(ev)), 2) if ev else None,
                                prompt_tok_per_s_median=round(float(np.median(pe)), 2) if pe else None,
                                num_thread=sorted({t.get("num_thread", 8) for t in tim}),
                                setting="ollama, num_gpu=0 (CPU), greedy, JSON format, shared machine")
        print(name, res["llm"][name], flush=True)
    res["load_avg_end"] = os.getloadavg()
    res["note"] = ("Machine shared with concurrent CPU jobs (load average recorded); both latencies are upper bounds "
                   "for an idle machine. LLM numbers are per-call wall times during the mapping runs.")
    json.dump(res, open(RES / "adapter_timing.json", "w"), indent=1)


if __name__ == "__main__":
    main()
