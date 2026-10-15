"""Instruction steerability of large instruction-following TTS on the 12 SteerTTS controls (X1 protocol, text interface).
Produced: results/instr_steer.json + _raw.csv (paper dir). All commands run from $STEER_TTS_ROOT with
PYTHONPATH=PromptingNemo/recipes/steer_tts.

    python -m steer.paper.instr_steer plan                       # emotts: jobs -> runs/paper/instr_steer/<sys>/_jobs.jsonl
    <env python> -m steer.paper.instr_steer render --system S    # in the system's env (GPU queue job; resumable)
    python -m steer.paper.instr_steer measure|asr|analyze        # emotts (measure = CPU, asr = GPU queue job)

Same protocol and readback as steer.paper.x1_steer_heldout: 8 harness TEXTS x voices x {uninstructed baseline} +
{12 dims x requested z in -2,-1,+1,+2}; delta = directed render minus the uninstructed render of the same voice/text,
read back by steer.features.measure (z in SteerTTS training pooled SD) and the Praat extractor (z in the pooled
within-speaker SD of real test-clean speech); Whisper-large-v3 fp16 WER; 1000-resample cluster bootstrap over
(voice,text). The analysis itself is x1_steer_heldout.analyze, run per system on this probe's renders.
Instead of a numeric z the systems get graded natural-language instructions (instr_steer_instructions.json, fixed
before rendering): z -> {much less, slightly less, slightly more, much more}.

Systems (loaders/checkpoints from steer.paper.baselines_ext):
  parler_mini  Parler-TTS Mini v1, description-conditioned. Voices = 8 of its named speakers (4F/4M). Baseline =
               '<Name> speaks in a very clean, close-sounding recording ...'; directed = the same description with the
               instruction phrase inserted. Batched over the 8 texts of one description, seed 0 per batch.
  cosyvoice2   CosyVoice2-0.5B inference_instruct2 with the held-out voice's real reference clip (zero-shot timbre;
               the first 4F + 4M of the 16 X1 held-out voices = the steer_heldout --quick subset). Baseline = a neutral
               instruction ('Speak in a natural, neutral way.'); directed = 'Speak <phrase>.'. Secondary baseline
               (baseline_mode=zero_shot): plain zero-shot cloning (reference clip + its transcript, no instruction).
Stdlib-only at module level: `render` runs inside the systems' own envs.
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(os.environ.get("STEER_TTS_ROOT", "/home/karan/Desktop/whissle/workspace/steer_tts"))
WORK = ROOT / "runs" / "paper" / "instr_steer"
INSTR = Path(__file__).with_name("instr_steer_instructions.json")
SYS = {"parler_mini": dict(n_voices=8), "cosyvoice2": dict(n_voices=8)}
ZS = [-2.0, -1.0, 1.0, 2.0]
SR_OUT = 24000
DIMS = ["rate", "loud", "f0_med", "f0_iqr", "f0_end", "pause", "tilt", "voiced", "flat", "centroid", "cpps", "h1h2"]


def instructions():
    return json.load(open(INSTR))


def describe(system, dim, z, name=None):
    """Instruction / description text for (dim, z); dim='baseline' -> the uninstructed one."""
    I = instructions()
    t = I["templates"][system]
    if dim == "baseline":
        return t["baseline"].format(name=name)
    return t["directed"].format(name=name, phrase=I["phrases"][dim][str(int(z))])


# ------------------------------------------------------------------------------------------------------------ plan
def cmd_plan():
    from ..harness import TEXTS
    from . import common as C
    I = instructions()
    # cosyvoice2: held-out voices (quick subset of the X1 16, same vids), interleaved F/M so a partial run keeps balance
    V = C.heldout_voices(16)
    F = [v for v in V if v["gender"] == "F"][:4]
    M = [v for v in V if v["gender"] == "M"][:4]
    bench = {json.loads(l)["gt_path"]: json.loads(l)["text"] for l in open(C.BENCH)}
    cv = [x for pair in zip(F, M) for x in pair]
    ps = sorted(I["parler_speakers"].items())
    pv = [x for pair in zip([p for p in ps if p[1][1] == "F"], [p for p in ps if p[1][1] == "M"]) for x in pair]
    voices = {"cosyvoice2": [dict(vid=v["vid"], gender=v["gender"], ref=v["ref"], ref_text=bench[v["ref"]]) for v in cv],
              "parler_mini": [dict(vid=k, gender=g, name=n) for k, (n, g) in pv]}
    for s, vs in voices.items():
        d = WORK / s
        (d / "wav").mkdir(parents=True, exist_ok=True)
        items = []
        for v in vs:  # unit-major order: a time-boxed partial run leaves complete (voice,text) units
            for ti, text in enumerate(TEXTS):
                conds = [("baseline", 0.0)] + ([("baseline_zs", 0.0)] if s == "cosyvoice2" else []) + \
                        [(dd, z) for dd in DIMS for z in ZS]
                for dim, z in conds:
                    it = dict(id=f"{v['vid']}_{ti}_{dim}_{z:+.0f}", system=s, vid=v["vid"], gender=v["gender"],
                              text_id=ti, text=text, dim=dim, req_z=z)
                    if s == "parler_mini":
                        it["desc"] = describe(s, dim, z, v["name"])
                    else:
                        it["ref"], it["ref_text"] = v["ref"], v["ref_text"]
                        it["desc"] = None if dim == "baseline_zs" else describe(s, dim, z)
                    it["instruction"] = "" if dim.startswith("baseline") else I["phrases"][dim][str(int(z))]
                    items.append(it)
        with open(d / "_jobs.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(s, len(vs), "voices", len(items), "jobs ->", d / "_jobs.jsonl")


# ---------------------------------------------------------------------------------------------------------- render
def _write(path, y, sr):
    import numpy as np
    import soundfile as sf
    import soxr
    y = np.asarray(y, np.float32).squeeze()
    if sr != SR_OUT:
        y = soxr.resample(y, sr, SR_OUT)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError("empty or non-finite audio")
    tmp = path.with_suffix(".tmp.wav")
    sf.write(tmp, np.clip(y, -1, 1), SR_OUT, subtype="PCM_16")
    os.replace(tmp, path)


def cmd_render(system, device, max_sec, limit=None):
    t0 = time.time()
    d = WORK / system
    items = [json.loads(l) for l in open(d / "_jobs.jsonl")]
    todo = [it for it in items if not (d / "wav" / f"{it['id']}.wav").exists()][:limit]
    print(f"{system}: {len(todo)}/{len(items)} to render", flush=True)
    if not todo:
        return
    n = fails = 0
    if system == "parler_mini":
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        torch.set_num_threads(4)
        name = "parler-tts/parler-tts-mini-v1"  # same checkpoint as baselines_ext.load_parler
        m = ParlerTTSForConditionalGeneration.from_pretrained(name).to(device).eval()
        tok = AutoTokenizer.from_pretrained(name)
        groups = {}
        for it in todo:
            groups.setdefault(it["desc"], []).append(it)
        for desc, its in groups.items():
            if time.time() - t0 > max_sec:
                print("time box reached", flush=True); break
            try:
                with torch.inference_mode():
                    di = tok([desc] * len(its), return_tensors="pt", padding=True).to(device)
                    p = tok([it["text"] for it in its], return_tensors="pt", padding=True).to(device)
                    torch.manual_seed(0)
                    o = m.generate(input_ids=di.input_ids, attention_mask=di.attention_mask,
                                   prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
                                   return_dict_in_generate=True)
                for k, it in enumerate(its):
                    y = o.sequences[k, :o.audios_length[k]].float().cpu().numpy()
                    _write(d / "wav" / f"{it['id']}.wav", y, m.config.sampling_rate); n += 1
            except Exception as e:  # noqa: BLE001
                fails += len(its)
                print(f"FAILED batch {its[0]['id']}: {type(e).__name__}: {e}", flush=True)
            if n % 200 < len(its):
                print(f"{system} {n}/{len(todo)} {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    else:
        from .baselines_ext import SYSTEMS
        import torch
        fn = SYSTEMS[system]["load"](device)
        for it in todo:
            if time.time() - t0 > max_sec:
                print("time box reached", flush=True); break
            try:
                y, sr = fn(it["text"], gender=it["gender"], desc=it.get("desc"), ref=it.get("ref"), ref_text=it.get("ref_text"))
                _write(d / "wav" / f"{it['id']}.wav", y, sr); n += 1
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAILED {it['id']}: {type(e).__name__}: {e}", flush=True)
            if n % 200 == 0 and n:
                print(f"{system} {n}/{len(todo)} {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    print(f"RENDERED {n} FAILED {fails} in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------- measure / asr
def _items(system, complete=True):
    d = WORK / system
    items = [json.loads(l) for l in open(d / "_jobs.jsonl")]
    for it in items:
        it["path"] = str(d / "wav" / f"{it['id']}.wav")
        it["exists"] = Path(it["path"]).exists()
    if complete:  # keep only (voice,text) units with every render present
        bad = {(it["vid"], it["text_id"]) for it in items if not it["exists"]}
        items = [it for it in items if (it["vid"], it["text_id"]) not in bad]
    return items


def cmd_measure(systems):
    import pandas as pd
    from . import common as C
    for s in systems:
        f = WORK / s / "measures.parquet"
        old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["key"])
        todo = [it for it in _items(s, complete=False) if it["exists"] and it["path"] not in set(old.key)]
        print(s, "measure", len(todo), flush=True)
        if todo:
            M = C.measure_many([(it["path"], it["path"], it["text"]) for it in todo], procs=int(os.environ.get("MEASURE_PROCS", 32)))
            pd.concat([old, M], ignore_index=True).to_parquet(f)


def cmd_asr(systems):
    from . import common as C
    for s in systems:
        its = [it for it in _items(s, complete=False) if it["exists"]]
        h = C.transcribe([it["path"] for it in its], WORK / s / "asr_cache.jsonl", batch=8)
        json.dump(h, open(WORK / s / "hyps.json", "w"))


# --------------------------------------------------------------------------------------------------------- analyze
def cmd_analyze(systems):
    import shutil
    import numpy as np
    import pandas as pd
    from ..harness import TEXTS
    from . import common as C
    from . import x1_steer_heldout as X
    from .baselines_ext import SYSTEMS
    I = instructions()
    R, raws, units = [], [], {}
    src = f"{C.RES}/instr_steer_raw.csv"
    for s in systems:
        d = WORK / s
        items = _items(s)
        for f in ("spk_mu.csv", "pooled_sd_gt.csv"):  # same speaker refs / Praat normaliser as X1
            shutil.copy(C.WORK / "x1" / f, d / f)
        M = pd.read_parquet(d / "measures.parquet")
        have = set(M.key)
        hyps = json.load(open(d / "hyps.json"))
        items = [it for it in items if it["path"] in have and it["path"] in hyps]
        modes = [("instructed_neutral" if s == "cosyvoice2" else "description_without_direction", "baseline")]
        if s == "cosyvoice2":
            modes.append(("zero_shot", "baseline_zs"))
        for mode, bdim in modes:
            its = []
            for it in items:
                if it["dim"].startswith("baseline") and it["dim"] != bdim:
                    continue
                it = {k: v for k, v in it.items() if k not in ("ref", "ref_text", "exists", "id", "system")}
                if it["dim"] == bdim:
                    it["dim"] = "baseline"
                its.append(it)
            ok = {(it["vid"], it["text_id"]) for it in its if it["dim"] == "baseline"}
            its = [it for it in its if (it["vid"], it["text_id"]) in ok]
            got = {}

            def capture(name, rows, raw, meta=None):
                got["rows"], got["raw"] = rows, raw
            X.OUT, C.write = d, capture
            X.analyze([], its)
            nv = len({it["vid"] for it in its}); nt = len({it["text_id"] for it in its})
            units[f"{s}/{mode}"] = dict(voices=nv, texts=nt, units=len(ok), renders=len(its))
            split = f"{s}_{'heldout' if s == 'cosyvoice2' else 'named_speakers'}{nv}_x_texts{nt}"
            for r in got["rows"]:
                if s == "parler_mini" and r["metric"] == "mean_abs_z_vs_speaker_ref":
                    continue  # Parler voices are not real speakers: no speaker reference
                r.update(system=s, baseline_mode=mode, params_M=SYSTEMS[s]["params_M"], split=split, source_file=src)
                td, z = r.get("target_dim"), r.get("req_z")
                if td in DIMS and z is not None and z == z and z != 0:
                    r["instruction"] = describe(s, td, z, "<Name>")
                elif td in DIMS:
                    r["instruction"] = json.dumps({str(int(k)): describe(s, td, k, "<Name>") for k in ZS})
                R.append(r)
            raw = got["raw"]
            raw.insert(0, "baseline_mode", mode)
            raw.insert(0, "system", s)
            raws.append(raw)
            h = [r for r in got["rows"] if r["metric"] in ("mean_spearman_rho_12dims", "offdiag_over_diag")]
            print(s, mode, units[f"{s}/{mode}"], [(r["metric"], r.get("extractor"), round(r["value"], 3)) for r in h], flush=True)
    meta = dict(
        systems={s: {k: v for k, v in SYSTEMS[s].items() if k != "load"} for s in systems},
        units=units, texts=len(TEXTS), req_z=ZS, instructions=I,
        interface="natural-language instruction (Parler: description; CosyVoice2: inference_instruct2 instruct text)",
        level_map="req_z -2/-1/+1/+2 -> much less / slightly less / slightly more / much more (graded phrases fixed "
                  "before rendering, steer/paper/instr_steer_instructions.json)",
        voices=dict(cosyvoice2="first 4F + 4M of the 16 X1 held-out LibriTTS-R test-clean voices (same vids H..; the "
                               "steer_heldout --quick subset), timbre from the same single real reference clip as X1",
                    parler_mini="8 Parler-TTS named speakers (4F/4M, anonymised P01-P08; names in meta.instructions)"),
        baseline_mode=dict(instructed_neutral="CosyVoice2 instruct2 with the neutral instruction (primary)",
                           zero_shot="CosyVoice2 plain zero-shot clone with the reference transcript (secondary)",
                           description_without_direction="Parler: same description without the direction phrase"),
        sampling="both systems sample stochastically; seed 0 per render (CosyVoice2) / per 8-text batch (Parler)",
        z_units=dict(ours="training pooled within-speaker SD (control_stats.json)",
                     praat="pooled within-speaker SD of the Praat features over the 500 real test-clean bench clips"),
        delta="directed render minus the uninstructed render of the same voice/text", praat_map=C.PRAAT_MAP,
        bootstrap="1000 cluster resamples over (voice,text) units, percentile 95% CI",
        hit_rate="share of units with sign(req_z)*delta > 0.5 z",
        analysis="steer.paper.x1_steer_heldout.analyze (identical metric code to results/steer_heldout.json)",
        code="steer/paper/instr_steer.py")
    raw = pd.concat(raws, ignore_index=True)
    json.dump(dict(name="instr_steer", meta=meta, rows=R), open(C.RES / "instr_steer.json", "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else (None if o != o else str(o)))
    raw.to_csv(C.RES / "instr_steer_raw.csv", index=False)
    print("wrote", C.RES / "instr_steer.json", len(R), "rows;", len(raw), "raw rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "render", "measure", "asr", "analyze"])
    ap.add_argument("--system")
    ap.add_argument("--systems", default="parler_mini,cosyvoice2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_sec", type=float, default=3000)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    ss = a.systems.split(",")
    if a.cmd == "plan":
        cmd_plan()
    elif a.cmd == "render":
        cmd_render(a.system, a.device, a.max_sec, a.limit)
    elif a.cmd == "measure":
        cmd_measure(ss)
    elif a.cmd == "asr":
        cmd_asr(ss)
    else:
        cmd_analyze(ss)


if __name__ == "__main__":
    main()
