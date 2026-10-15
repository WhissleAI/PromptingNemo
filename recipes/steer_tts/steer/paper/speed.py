"""Speed & efficiency benchmark for the SteerTTS paper (NAACL 2027, section "Why is it fast?").

One protocol for every system (ours and baselines), importable from ANY env (no steer imports at module level):

    from steer.paper.speed import time_system, file_fn, TEXTS
    rows = time_system(fn, TEXTS, device="cuda", threads=None, n_runs=20, system="kokoro")

    fn(text) -> (wav np.ndarray, sr)          one synthesis, batch 1, model already loaded (in-process)
    file_fn(synth) adapts synth(text, out_wav) (writes to /dev/shm, reads back; the I/O is inside the timing)

Protocol (time_system):
  * fixed English text sets (TEXTS): short ~2-3 s, medium ~6-8 s, long ~15 s, xlong ~30 s, longform ~90 s of speech;
  * per set: `warmup` untimed calls on the first text, then timed calls cycling over the set's texts until n_runs
    calls or the time budget is used (at least min_runs);
  * CUDA: torch.cuda.synchronize() before starting and after each call (fn returns CPU audio anyway);
  * per call: wall seconds (time.perf_counter), audio seconds = len(wav)/sr, RTFx = audio_s / wall_s;
  * rows: median with p10/p90 of RTFx and latency_ms per set, plus first-call stats; load average recorded.
Threads: torch.set_num_threads(threads) (+ the caller pins cores with `taskset -c`, see logs/gpu_queue 01_speed.sh).

CLI (STEER_TTS_ROOT and PYTHONPATH=recipes/steer_tts set):
  python -m steer.paper.speed steer   --device cuda --threads 8 --out results/speed_steer_gpu.json   # e2e + components
  python -m steer.paper.speed scaling --device cuda --out results/speed_scaling.json   # attn vs mamba3 vs hybrid
  python -m steer.paper.speed flops   --out results/speed_flops.json
  python -m steer.paper.speed stream  --device cuda --out results/speed_stream_gpu.json
  <env python> -m steer.paper.speed ext --system kokoro|piper|parler_mini|<baselines_ext name> --device cuda ...
Output rows: metric, value, p10, p90, n, device, threads, system, length_s, set, source_file, notes (+extra keys).
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------ fixed text sets
TEXTS = {
    "short": [  # ~2-3 s
        "The meeting starts at noon.",
        "Please close the window behind you.",
        "We should leave before it rains.",
    ],
    "medium": [  # ~6-8 s
        "After the long drive through the mountains, we finally reached a small town with one hotel and a bakery.",
        "She checked the schedule twice, but the last train had already left the station an hour earlier.",
        "If you water the garden every morning, the tomatoes will be ready to pick by the end of summer.",
    ],
    "long": [  # ~15 s
        "The library on the corner was built more than a century ago, and although the roof has been replaced twice, "
        "the reading room still has its original oak tables, tall windows, and a quiet that seems to slow down "
        "everyone who walks through the door.",
        "When the storm finally passed, the neighbours came out to clear the branches from the road, and by the "
        "afternoon the children were already back on their bicycles, racing each other past the puddles toward the "
        "river.",
    ],
    "xlong": [  # ~30 s
        "Every spring, the town holds a market in the old square, where farmers from the surrounding valleys bring "
        "cheese, honey, bread, and early vegetables. The stalls open before sunrise, and by eight o'clock the square "
        "is crowded with families, musicians, and visitors who have travelled for hours. Most people come for the "
        "food, but many stay for the conversations, because the market is one of the few places where everyone in "
        "the region still meets face to face.",
    ],
    "longform": [  # ~90 s paragraph
        "Long before the first road reached the valley, the only way in was a narrow path along the river. "
        "Travellers carried everything on their backs, and a trip to the nearest town took two full days. "
        "The people who lived there learned to depend on each other. They shared tools, traded seeds, and gathered "
        "in the largest kitchen whenever the weather turned bad. When the road was finally built, many expected "
        "the valley to change overnight, but the old habits proved surprisingly durable. Neighbours still helped "
        "with the harvest, and the long table in the community hall was still set every Sunday. What did change "
        "was the pace of news. Letters that once took a week now arrived the next morning, and young people began "
        "to leave for the cities, returning only for holidays. Some of them came back for good, bringing new ideas "
        "about farming, schooling, and how the valley might welcome visitors without losing its character. Today "
        "the river path is a popular walking trail. Families follow it on summer afternoons, stopping at the old "
        "stone bridge to watch the water, and few of them realise that, not so long ago, it was the only "
        "connection between this quiet place and the rest of the world.",
    ],
}
SET_ORDER = ["short", "medium", "long", "xlong", "longform"]
PAPER = Path(os.environ.get("SPEED_PAPER_DIR", "/home/karan/Desktop/whissle/marketing_post_agent/paper/steertts"))


def _q(x, q):
    return float(np.percentile(np.asarray(x, float), q)) if len(x) else None


def env_info():
    try:
        la = os.getloadavg()
    except OSError:
        la = (None, None, None)
    aff = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    d = dict(loadavg_1m=la[0], loadavg_5m=la[1], cpus_pinned=len(aff) if aff else None,
             host=os.uname().nodename)
    if os.environ.get("SPEED_FLAG"):  # e.g. "load_deadline": run started without the quiet-machine condition
        d["flag"] = os.environ["SPEED_FLAG"]
    return d


def _sync(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def set_threads(threads):
    if threads:
        os.environ["SPEED_THREADS"] = str(threads)
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[v] = str(threads)
        try:
            import torch
            torch.set_num_threads(int(threads))
            torch.set_num_interop_threads(1)
        except Exception:  # noqa: BLE001 - interop threads can only be set once
            pass


def row(metric, vals, *, system, device, threads, set_name=None, length_s=None, source_file=None, notes="", **extra):
    vals = [v for v in vals if v is not None]
    r = dict(metric=metric, value=_q(vals, 50) if vals else None, p10=_q(vals, 10), p90=_q(vals, 90), n=len(vals),
             device=device, threads=threads, system=system, length_s=length_s, set=set_name,
             source_file=source_file or "steer/paper/speed.py", notes=notes)
    r.update(extra)
    return r


def time_system(fn, texts=None, device="cuda", threads=None, n_runs=20, warmup=3, system="system", sets=None,
                budget_s=120.0, min_runs=5, notes="", extra=None, log=print):
    """Time fn(text)->(wav, sr) with the paper protocol (module docstring). Returns result rows."""
    texts = texts or TEXTS
    sets = [s for s in (sets or SET_ORDER) if s in texts]
    extra = dict(extra or {})
    rows, all_rtf = [], []
    for s in sets:
        tx = texts[s]
        t_first = None
        try:
            for i in range(warmup):
                _sync(device)
                t0 = time.perf_counter()
                fn(tx[0])
                _sync(device)
                if i == 0:
                    t_first = time.perf_counter() - t0
            wall, dur = [], []
            t_start = time.perf_counter()
            k = 0
            while k < n_runs and (k < min_runs or time.perf_counter() - t_start < budget_s):
                text = tx[k % len(tx)]
                _sync(device)
                t0 = time.perf_counter()
                y, sr = fn(text)
                _sync(device)
                wall.append(time.perf_counter() - t0)
                dur.append(len(np.asarray(y).reshape(-1)) / float(sr))
                k += 1
        except Exception as e:  # noqa: BLE001 - one failing set must not lose the others
            log(f"[{system}/{device}/{threads}] set {s} FAILED: {type(e).__name__}: {str(e)[:300]}")
            rows.append(row("error", [], system=system, device=device, threads=threads, set_name=s,
                            notes=f"{type(e).__name__}: {str(e)[:200]}", **extra))
            continue
        rtf = [d / w for d, w in zip(dur, wall)]
        all_rtf += rtf
        L = float(np.median(dur))
        e = dict(extra, **env_info())
        rows.append(row("rtfx", rtf, system=system, device=device, threads=threads, set_name=s, length_s=L,
                        notes=notes, **e))
        rows.append(row("latency_ms", [w * 1000 for w in wall], system=system, device=device, threads=threads,
                        set_name=s, length_s=L, notes=notes, **e))
        rows.append(row("audio_s", dur, system=system, device=device, threads=threads, set_name=s, length_s=L, **e))
        if t_first is not None:
            rows.append(row("first_call_ms", [t_first * 1000], system=system, device=device, threads=threads,
                            set_name=s, length_s=L, notes="first warm-up call of the set (cold for the first set)",
                            **e))
        log(f"[{system}/{device}/thr={threads}] {s:8s} audio {L:6.1f}s  RTFx {np.median(rtf):8.1f} "
            f"(p10 {_q(rtf, 10):.1f}, p90 {_q(rtf, 90):.1f})  lat {np.median(wall) * 1000:8.1f} ms  n={len(rtf)}  "
            f"load {e['loadavg_1m']}")
    if all_rtf:
        rows.append(row("rtfx_pooled", all_rtf, system=system, device=device, threads=threads, set_name="all",
                        notes="median over all timed calls of all sets", **extra))
    return rows


def file_fn(synth, suffix=".wav"):
    """Adapt synth(text, out_wav) to the fn(text)->(wav, sr) protocol (file I/O counted inside the timing)."""
    import soundfile as sf
    d = Path("/dev/shm") if Path("/dev/shm").exists() else Path(tempfile.gettempdir())
    out = d / f"speed_{os.getpid()}{suffix}"

    def fn(text):
        synth(text, str(out))
        y, sr = sf.read(str(out), dtype="float32")
        return y, sr
    return fn


def save_rows(rows, out):
    """Append-merge: rows of the same (system, device, threads, metric, set, variant) are replaced."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.load(open(out)) if out.exists() else []
    key = lambda r: (r.get("system"), r.get("device"), r.get("threads"), r.get("metric"), r.get("set"),
                     r.get("variant"), r.get("component"), r.get("length_s") if r.get("metric", "").startswith(
                         ("scal", "mem")) else None)
    new_keys = {key(r) for r in rows}
    merged = [r for r in old if key(r) not in new_keys] + rows
    tmp = out.with_suffix(".tmp")
    json.dump(merged, open(tmp, "w"), indent=1, default=float)
    tmp.replace(out)


# ------------------------------------------------------------------------------------------------ SteerTTS
CKPT = "runs/spk_enc_v3/step312500.pt"
FLOW = "runs/flow_v3/step100000.pt"
PFLOW = "runs/flow_v4pa/step40000.pt"
VOICE = "voice_4223"
REF_WAV = "/home/karan/Desktop/whissle/workspace/seedtts/data/seedtts_testset/en/prompt-wavs/common_voice_en_1004470.wav"
STEER_REQ = {"f0_iqr": 2.0}  # a request for the steered / guided timing variants (duration-neutral control)
GUIDE_S = 2.0  # guidance scale for the guided variant (cost is the same for any s != 1)
PROMPTS = ["Speak slowly in a calm, low voice.", "An excited young woman talking fast and loud.",
           "Whisper this sadly.", "A deep, serious narrator voice with long pauses."]


def _root():
    return Path(os.environ.get("STEER_TTS_ROOT", "/home/karan/Desktop/whissle/workspace/steer_tts"))


def load_synth(device, flow=False, pflow=False):
    from steer.synth import Synth
    ck = str(_root() / CKPT)
    if flow:
        ck += ",flow=" + str(_root() / FLOW)
    if pflow:
        ck += ",pflow=" + str(_root() / PFLOW)
    return Synth(ck, device=device)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def steer_components(S, text, device, flow_steps=None, ref_wav=None, xvec=True):
    """One synthesis through Synth's path, split into stages with a sync after each. -> dict ms, audio_s.
    Mirrors Synth.__call__ / SteerTTS.infer (guidance off = the default)."""
    import torch
    from steer.model import seq_mask, length_regulate
    from steer.text import to_phonemes
    from steer.config import SR
    m = S.model
    T = {}

    def tick(name, t0):
        _sync(device)
        T[name] = T.get(name, 0.0) + (time.perf_counter() - t0) * 1000
        return time.perf_counter()

    with torch.no_grad():
        _sync(device)
        t = time.perf_counter()
        ids, widx, _ = to_phonemes(text)
        n_sym = m.emb.num_embeddings
        ids = [i for i in ids if i < n_sym]
        ids_t = torch.tensor([ids], device=device)
        t = tick("g2p", t)
        c, mk = S.control_tensors(None, VOICE, ref_wav)
        speaker = VOICE if VOICE in S.spk2i else __import__("steer.voices", fromlist=["resolve"]).resolve(VOICE)
        spk = torch.tensor([S.spk2i.get(speaker, len(S.spk2i))], device=device)
        if ref_wav is not None:
            spk_vec = S.speaker_vector(None, ref_wav)
            t = tick("speaker_xvec", t)
        else:
            spk_vec = S.speaker_vector(speaker, None)
        tg = torch.tensor([0], device=device)
        g = m.condition(spk, tg, c, mk, spk_vec)
        t = tick("conditioning", t)
        text_lens = torch.tensor([ids_t.shape[1]], device=device)
        _, h, tmask = m.encode(ids_t, text_lens, g)
        t = tick("encoder", t)
        logdur, pitch, energy = m.dur_pred(h, tmask, g), m.pitch_pred(h, tmask, g), m.energy_pred(h, tmask, g)
        dur = torch.round((torch.exp(logdur) - 1).clamp_min(0)).long().clamp_min(0)
        dur[:, 0] = dur[:, 0].clamp_min(1)
        pe = m.pitch_emb(pitch[:, None]).transpose(1, 2) + m.energy_emb(energy[:, None]).transpose(1, 2)
        t = tick("variance", t)
        x, lens = length_regulate(h + pe, dur)
        t = tick("length_reg", t)
        mel = m.to_mel(m.decoder(x, seq_mask(lens, x.shape[1]), g)).transpose(1, 2)
        t = tick("decoder", t)
        fl = S.pflow if (ref_wav is not None and S.pflow is not None) else S.flow
        if fl is not None:
            steps = flow_steps or S.flow_steps
            pm = S.prompt_mel(ref_wav, mel) if (ref_wav is not None and getattr(fl.net, "prompt", False)) else None
            if pm is None:
                t = tick("flow_prep", t)
                mel = fl.sample(mel.float(), g.float(), steps=steps, temp=S.flow_temp, seed=0)
                t = tick("flow", t)
            else:
                t = tick("prompt_mel", t)
                P = pm.shape[-1]
                full = torch.cat([pm, mel.float()], -1)
                cm = torch.zeros_like(full[:, :1]); cm[..., :P] = 1
                mel = fl.sample(full, g.float(), steps=steps, temp=S.flow_temp, seed=0, ctx=full, cm=cm)[..., P:]
                t = tick("pflow", t)
        wav = S.vocos.decode(mel.float())
        t = tick("vocos", t)
        wav = wav.squeeze().cpu().numpy()
        t = tick("to_host", t)
    return T, len(wav) / SR, mel.shape[-1]


def bench_steer(a):
    """End-to-end RTFx (Synth.__call__, no inner syncs) for base / +flow / cloning, and per-component breakdown."""
    import torch
    set_threads(a.threads)
    dev = a.device
    out = a.out
    sysname = lambda v: f"steertts-{v}"
    variants = a.variants.split(",")
    S = load_synth(dev, flow="flow" in variants, pflow="clone" in variants)
    base_params = nparams(S.model) - sum(p.numel() for p in S.model.aligner.parameters())
    pinfo = dict(params_M_acoustic=base_params / 1e6, params_M_vocos=nparams(S.vocos) / 1e6,
                 params_M_flow=nparams(S.flow) / 1e6 if S.flow else None,
                 params_M_pflow=nparams(S.pflow) / 1e6 if S.pflow else None)
    print("params", pinfo, flush=True)
    budget = a.budget
    for v in variants:
        flow_saved = S.flow
        if v == "base":
            S.flow = None
            fn = lambda t: (S(t, VOICE), 24000)
        elif v == "flow":
            fn = lambda t: (S(t, VOICE), 24000)
        elif v == "clone":
            S.flow = None
            fn = lambda t: (S(t, VOICE, ref_wav=REF_WAV), 24000)
        elif v == "steered":  # a request (one control) at s = 1: same single pass as base
            S.flow = None
            fn = lambda t: (S(t, VOICE, ctrl=STEER_REQ, guidance=1.0), 24000)
        elif v == "guided":  # the same request with guidance s = 2: adds the null-request pass (eq. guidance)
            S.flow = None
            fn = lambda t: (S(t, VOICE, ctrl=STEER_REQ, guidance=GUIDE_S), 24000)
        else:
            continue
        sets = a.sets.split(",") if a.sets else None
        if v in ("flow", "clone") and dev == "cpu":  # flow / cloning on CPU: short + medium only (time budget)
            sets = [x for x in (sets or SET_ORDER) if x in ("short", "medium")]
        note = "Synth.__call__ end to end (g2p..Vocos, audio on host); default voice; guidance off"
        if v in ("steered", "guided"):
            note = (f"Synth.__call__ end to end; default voice; request ctrl={STEER_REQ}; guidance s="
                    f"{1.0 if v == 'steered' else GUIDE_S}")
        rows = time_system(fn, device=dev, threads=a.threads, n_runs=a.n_runs, system=sysname(v), sets=sets,
                           budget_s=budget, min_runs=a.min_runs, extra=dict(group="ours", variant=v, **pinfo),
                           notes=note)
        save_rows(rows, out)
        if v in ("steered", "guided"):  # no component breakdown for these (same stages as base)
            S.flow = flow_saved
            continue
        # component breakdown (sync after each stage), on the medium + long sets
        comp_rows = []
        for s in (["medium", "long", "longform"] if dev.startswith("cuda") else ["medium", "long"]):
            acc, auds = {}, []
            n = 0
            t_start = time.perf_counter()
            try:
                for i in range(2):
                    steer_components(S, TEXTS[s][0], dev, ref_wav=REF_WAV if v == "clone" else None)
                while n < a.n_runs and (n < 5 or time.perf_counter() - t_start < budget / 2):
                    T, aud, _ = steer_components(S, TEXTS[s][n % len(TEXTS[s])], dev,
                                                 ref_wav=REF_WAV if v == "clone" else None)
                    for k, ms in T.items():
                        acc.setdefault(k, []).append(ms / aud)  # ms per audio second
                    acc.setdefault("total", []).append(sum(T.values()) / aud)
                    auds.append(aud); n += 1
            except Exception as e:  # noqa: BLE001
                print("components failed", v, s, e, flush=True)
                continue
            for k, vals in acc.items():
                comp_rows.append(row("component_ms_per_audio_s", vals, system=sysname(v), device=dev,
                                     threads=a.threads, set_name=s, length_s=float(np.median(auds)), component=k,
                                     variant=v, notes="sync after every stage; ms of wall time per second of audio",
                                     **env_info()))
            print(f"[components {v} {s}] " + "  ".join(f"{k} {np.median(x):.2f}" for k, x in acc.items()), flush=True)
        save_rows(comp_rows, out)
        S.flow = flow_saved
    # flow refiner cost per ODE step (midpoint = 2 network evaluations per step)
    if S.flow is not None and "flow" in variants:
        rows = []
        mel, aux = None, None
        with torch.no_grad():
            from steer.text import to_phonemes  # noqa: F401
            S.flow, fl = None, S.flow
            _, mel = S(TEXTS["medium"][0], VOICE, return_mel=True)
            S.flow = fl
            g = torch.zeros(1, 128, device=dev)
            aud = mel.shape[-1] * 256 / 24000
            for steps in (1, 2, 4, 10, 20):
                ts = []
                for i in range(2 + (a.n_runs if dev.startswith("cuda") else 5)):
                    _sync(dev); t0 = time.perf_counter()
                    fl.sample(mel.float(), g, steps=steps, temp=S.flow_temp, seed=0)
                    _sync(dev)
                    if i >= 2:
                        ts.append((time.perf_counter() - t0) * 1000)
                rows.append(row("flow_ms", ts, system="flow-refiner", device=dev, threads=a.threads, set_name="medium",
                                length_s=aud, variant=f"steps{steps}", steps=steps, nfe=2 * steps,
                                notes="Flow.sample midpoint ODE, 2 NFE/step", **env_info()))
                print(f"[flow steps={steps}] {np.median(ts):.1f} ms for {aud:.1f}s audio", flush=True)
        save_rows(rows, out)
    # prompt adapter (MiniLM sentence encoder + 0.5M head), cache bypassed
    if a.prompt:
        try:
            from steer.prompt import PromptMapper, encoder
            pm = PromptMapper(device=dev)
            enc_ms, head_ms, tot = [], [], []
            for i in range(3 + a.n_runs):
                p = PROMPTS[i % len(PROMPTS)]
                pm.cache.clear()
                _sync(dev); t0 = time.perf_counter()
                e = encoder(dev).encode([p], normalize_embeddings=True, convert_to_tensor=True).to(dev)
                _sync(dev); t1 = time.perf_counter()
                with torch.no_grad():
                    pm.m(e)
                _sync(dev); t2 = time.perf_counter()
                pm.cache.clear()
                pm(p)
                _sync(dev); t3 = time.perf_counter()
                if i >= 3:
                    enc_ms.append((t1 - t0) * 1000); head_ms.append((t2 - t1) * 1000); tot.append((t3 - t2) * 1000)
            rows = [row("prompt_ms", enc_ms, system="prompt-adapter", device=dev, threads=a.threads,
                        component="minilm_encoder", notes="frozen all-MiniLM-L6-v2 sentence embedding"),
                    row("prompt_ms", head_ms, system="prompt-adapter", device=dev, threads=a.threads,
                        component="adapter_head", params_M=nparams(pm.m) / 1e6),
                    row("prompt_ms", tot, system="prompt-adapter", device=dev, threads=a.threads,
                        component="PromptMapper_total", notes="PromptMapper.__call__ uncached (incl. decoding)")]
            save_rows(rows, out)
            print(f"[prompt] minilm {np.median(enc_ms):.2f} ms  head {np.median(head_ms):.2f} ms  "
                  f"total {np.median(tot):.2f} ms", flush=True)
        except Exception as e:  # noqa: BLE001
            print("prompt adapter timing failed:", e, flush=True)
    if dev.startswith("cuda"):
        save_rows([row("peak_mem_mb", [torch.cuda.max_memory_allocated() / 2 ** 20], system="steertts-all-variants",
                       device=dev, threads=a.threads, notes="max_memory_allocated over the whole steer benchmark")],
                  out)


# ------------------------------------------------------------------------------------------------ scaling
ARCHS = {  # param-matched token-mixer configurations (mixer study, ~9.55M inference params)
    "hybrid": dict(d=192, ffn=384, enc_mixer="hybrid", dec_mixer="hybrid"),
    "attn": dict(d=256, ffn=512, enc_mixer="attn", dec_mixer="attn"),
    "mamba3": dict(d=224, ffn=320, enc_mixer="mamba3", dec_mixer="mamba3"),
}
TOK_PER_S = 13.0  # phone tokens (incl. <sp>/punct) per second of speech: measured on the medium set (see notes)


def random_model(arch, device):
    import torch
    from steer.model import SteerTTS
    torch.manual_seed(0)
    m = SteerTTS(10, 51, ssm_state=32, spk_enc_dim=512, n_symbols=118, **ARCHS[arch]).to(device).eval()
    return m


def acoustic_fixed(m, N, T, device, dtype=None):
    """Acoustic model on N tokens with durations fixed so the output has exactly T frames (no duration randomness)."""
    import torch
    from steer.model import seq_mask, length_regulate
    ids = torch.randint(3, 78, (1, N), device=device)
    g = m.condition(torch.zeros(1, dtype=torch.long, device=device), torch.zeros(1, dtype=torch.long, device=device),
                    torch.zeros(1, 12, device=device), torch.zeros(1, 12, device=device),
                    torch.zeros(1, 512, device=device))
    dur = torch.full((1, N), T // N, device=device, dtype=torch.long)
    dur[0, : T - int(dur.sum())] += 1
    ctx = torch.autocast(device.split(":")[0], dtype=dtype) if dtype is not None else torch.autocast(
        device.split(":")[0], enabled=False)
    with torch.no_grad(), ctx:
        _, h, tmask = m.encode(ids, torch.tensor([N], device=device), g)
        lp, p, e = m.dur_pred(h, tmask, g), m.pitch_pred(h, tmask, g), m.energy_pred(h, tmask, g)
        pe = m.pitch_emb(p[:, None]).transpose(1, 2) + m.energy_emb(e[:, None]).transpose(1, 2)
        x, lens = length_regulate(h + pe, dur)
        mel = m.to_mel(m.decoder(x, seq_mask(lens, x.shape[1]), g)).transpose(1, 2)
    return mel


def bench_scaling(a):
    """Latency + peak memory of the acoustic model vs output length, per token mixer; eager vs compile; fp32 vs bf16."""
    import torch
    set_threads(a.threads)
    dev = a.device
    lengths = [float(x) for x in a.lengths.split(",")]
    archs = a.archs.split(",")
    modes = a.modes.split(",")
    for arch in archs:
        m = random_model(arch, dev)
        np_inf = (nparams(m) - nparams(m.aligner) - m.spk.weight.numel()) / 1e6
        for mode in modes:
            from steer.model import Stack
            Stack.compiled = None
            dtype = None
            if mode == "compile":
                Stack.enable_compile()
            if mode == "bf16":
                if not dev.startswith("cuda"):
                    continue
                dtype = torch.bfloat16
            rows = []
            oom = False
            for L in lengths:
                if oom:
                    rows.append(row("scal_latency_ms", [], system=arch, device=dev, threads=a.threads, length_s=L,
                                    variant=mode, notes="skipped: OOM at a shorter length", oom=True))
                    continue
                T = int(round(L * 93.75)); N = max(4, int(round(L * TOK_PER_S)))
                ts = []
                try:
                    if dev.startswith("cuda"):
                        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                        base_mem = torch.cuda.memory_allocated()
                    t_budget = time.perf_counter()
                    nw = 3 if mode != "compile" else 4
                    for i in range(nw + a.n_runs):
                        _sync(dev); t0 = time.perf_counter()
                        acoustic_fixed(m, N, T, dev, dtype)
                        _sync(dev)
                        if i >= nw:
                            ts.append((time.perf_counter() - t0) * 1000)
                        if i >= nw + 3 and time.perf_counter() - t_budget > a.budget:
                            break
                    peak = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20 if dev.startswith("cuda") else None
                except torch.OutOfMemoryError:
                    oom = True
                    torch.cuda.empty_cache()
                    rows.append(row("scal_latency_ms", [], system=arch, device=dev, threads=a.threads, length_s=L,
                                    variant=mode, notes="CUDA OOM", oom=True, frames=T, tokens=N))
                    print(f"[scaling {arch} {mode}] {L:7.1f}s OOM", flush=True)
                    continue
                except Exception as e:  # noqa: BLE001
                    rows.append(row("scal_latency_ms", [], system=arch, device=dev, threads=a.threads, length_s=L,
                                    variant=mode, notes=f"{type(e).__name__}: {str(e)[:200]}", frames=T, tokens=N))
                    print(f"[scaling {arch} {mode}] {L:7.1f}s FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                    if mode == "compile":
                        break
                    continue
                rows.append(row("scal_latency_ms", ts, system=arch, device=dev, threads=a.threads, length_s=L,
                                variant=mode, frames=T, tokens=N, params_M=np_inf, **env_info()))
                rows.append(row("scal_rtfx", [L / (t / 1000) for t in ts], system=arch, device=dev,
                                threads=a.threads, length_s=L, variant=mode, frames=T, tokens=N, params_M=np_inf))
                if peak is not None:
                    rows.append(row("mem_peak_mb", [peak], system=arch, device=dev, threads=a.threads, length_s=L,
                                    variant=mode, frames=T, tokens=N, notes="max_memory_allocated minus weights"))
                print(f"[scaling {arch:6s} {mode:7s}] {L:7.1f}s T={T:6d} N={N:5d} lat {np.median(ts):9.2f} ms "
                      f"(n={len(ts)}) peak {peak if peak is None else round(peak, 1)} MB", flush=True)
            save_rows(rows, a.out)
            Stack.compiled = None
        del m
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    # Vocos alone vs length (shared by all mixers)
    if a.vocos:
        from vocos import Vocos
        voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev).eval()
        rows = []
        for L in lengths:
            T = int(round(L * 93.75)); ts = []
            try:
                mel = torch.randn(1, 100, T, device=dev) - 5
                if dev.startswith("cuda"):
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    base_mem = torch.cuda.memory_allocated()
                t_budget = time.perf_counter()
                with torch.no_grad():
                    for i in range(3 + a.n_runs):
                        _sync(dev); t0 = time.perf_counter()
                        voc.decode(mel)
                        _sync(dev)
                        if i >= 3:
                            ts.append((time.perf_counter() - t0) * 1000)
                        if i >= 6 and time.perf_counter() - t_budget > a.budget:
                            break
                peak = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20 if dev.startswith("cuda") else None
            except Exception as e:  # noqa: BLE001
                print("vocos", L, e, flush=True)
                break
            rows.append(row("scal_latency_ms", ts, system="vocos", device=dev, threads=a.threads, length_s=L,
                            variant="eager", frames=T))
            if peak is not None:
                rows.append(row("mem_peak_mb", [peak], system="vocos", device=dev, threads=a.threads, length_s=L,
                                variant="eager", frames=T))
            print(f"[scaling vocos] {L:7.1f}s lat {np.median(ts):9.2f} ms", flush=True)
        save_rows(rows, a.out)


# ------------------------------------------------------------------------------------------------ FLOP counting
# Two counts for every component and system (reviewer R3: matmul-only counts flatter recurrences / scans):
#   matmul : torch.utils.flop_counter (matmul, bmm, conv, SDPA; 2 FLOPs per multiply-add)
#   full   : matmul + every other arithmetic aten op at 1 FLOP per output element (add, mul, exp, cumsum, norms'
#            reductions, activations ...), softmax / log_softmax at 5 per element (max, sub, exp, sum, div), FFTs at
#            5 n log2 n per transform; pure data movement (view/copy/cat/gather/pad/index/fill/creation) is free.
_MOVE = {"view", "_unsafe_view", "reshape", "expand", "transpose", "permute", "t", "slice", "select", "unsqueeze",
         "squeeze", "as_strided", "detach", "alias", "clone", "copy_", "_to_copy", "to", "contiguous", "cat", "stack",
         "gather", "scatter", "index", "index_select", "index_put_", "constant_pad_nd", "reflection_pad1d",
         "replication_pad1d", "fill_", "zero_", "zeros", "zeros_like", "ones", "ones_like", "empty", "empty_like",
         "empty_strided", "new_zeros", "new_empty", "new_ones", "full", "full_like", "arange", "lift_fresh",
         "_local_scalar_dense", "split", "split_with_sizes", "unbind", "chunk", "narrow", "flip", "roll", "repeat",
         "embedding", "masked_fill", "masked_fill_", "where", "_reshape_alias", "unfold", "set_", "resize_",
         "randn", "rand", "randint", "normal_", "uniform_", "bernoulli_", "lift", "view_as_real", "view_as_complex",
         "_conj", "conj", "resolve_conj", "resolve_neg", "item", "nonzero", "sort", "topk", "argmax", "max_pool1d",
         "eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or", "logical_not", "bitwise_not", "bitwise_and",
         "_unsafe_index", "unsqueeze_", "squeeze_", "transpose_", "t_", "split_copy", "slice_scatter",
         "select_scatter", "diagonal", "tril", "triu", "_assert_async", "sym_size", "sym_stride", "is_nonzero"}


def flop_count(f):
    """-> (matmul_flops, full_flops, result) for calling f() (see the convention above)."""
    import math
    import torch
    from torch.utils.flop_counter import FlopCounterMode, flop_registry
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.nn.attention import sdpa_kernel, SDPBackend

    class Elementwise(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            pk = func.overloadpacket
            name = pk.__name__.split(".")[-1]
            if pk in flop_registry or name.rstrip("_") in _MOVE or name in _MOVE:
                return out
            outs = [o for o in (out if isinstance(out, (tuple, list)) else [out]) if isinstance(o, torch.Tensor)]
            n = sum(o.numel() for o in outs)
            if "softmax" in name:
                n *= 5
            elif name.startswith("_fft") or name.startswith("fft"):
                x = args[0]
                dims = args[1] if len(args) > 1 and isinstance(args[1], (list, tuple)) else [-1]
                size = max(int(np.prod([x.shape[d] for d in dims])), 2)
                n = 5 * x.numel() * math.log2(size)
            self.n += n
            return out

    ew = Elementwise()
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc, ew:
        r = f()
    mm = fc.get_total_flops()
    return mm, mm + ew.n, r


# ------------------------------------------------------------------------------------------------ FLOPs
def bench_flops(a):
    """FLOPs per second of generated audio per component (torch.utils.flop_counter: matmul/conv/SDPA/bmm only -
    elementwise ops, the Mamba-3 scan's cumsum/exp and the iSTFT are not counted). Runs on CPU (deterministic)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    set_threads(a.threads or 8)
    # the fused MHA fast path (eval, no grad) is invisible to FlopCounterMode: use the composite path (same math)
    torch.backends.mha.set_fastpath_enabled(False)
    dev = "cpu"
    S = load_synth(dev, flow=True)
    rows = []

    from torch.nn.attention import sdpa_kernel, SDPBackend

    FULL = {}

    def count(f, key=None):  # MATH backend: the CPU flash-attention kernel has no FLOP formula (would count 0)
        mm, full, r = flop_count(f)
        if key:
            FULL[key] = full
        count.last_full = full
        return mm, r

    from steer.model import seq_mask, length_regulate
    from steer.text import to_phonemes
    m = S.model
    for s in ["medium", "long"]:
        text = TEXTS[s][0]
        ids, _, _ = to_phonemes(text)
        with torch.no_grad():
            ids_t = torch.tensor([[i for i in ids if i < m.emb.num_embeddings]])
            spk_vec = S.speaker_vector(VOICE if VOICE in S.spk2i else __import__(
                "steer.voices", fromlist=["resolve"]).resolve(VOICE), None)
            c, mk = S.control_tensors(None)
            g = m.condition(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), c, mk, spk_vec)
            N = ids_t.shape[1]
            f_enc, (_, h, tmask) = count(lambda: m.encode(ids_t, torch.tensor([N]), g), "encoder")
            def var():
                lp, p, e = m.dur_pred(h, tmask, g), m.pitch_pred(h, tmask, g), m.energy_pred(h, tmask, g)
                return lp, m.pitch_emb(p[:, None]).transpose(1, 2) + m.energy_emb(e[:, None]).transpose(1, 2)
            f_var, (lp, pe) = count(var, "variance")
            dur = torch.round((torch.exp(lp) - 1).clamp_min(0)).long(); dur[:, 0] = dur[:, 0].clamp_min(1)
            x, lens = length_regulate(h + pe, dur)
            f_dec, mel = count(lambda: m.to_mel(m.decoder(x, seq_mask(lens, x.shape[1]), g)).transpose(1, 2),
                               "decoder")
            f_flow1, _ = count(lambda: S.flow.sample(mel, g, steps=1, temp=0.4, seed=0), "flow_per_step")
            f_voc, wav = count(lambda: S.vocos.decode(mel), "vocos")
            aud = wav.shape[-1] / 24000
            # Mamba-3 share of the stacks: count with the SSM branches only
            ssm_f, ssm_full = 0, 0
            for blk in list(m.encoder.layers):
                hh = torch.randn(1, N, m.emb.embedding_dim)
                f, _ = count(lambda: blk.ssm(hh, torch.ones(1, N, dtype=torch.bool))) if hasattr(blk, "ssm") else (0, 0)
                ssm_f += f
                ssm_full += count.last_full if hasattr(blk, "ssm") else 0
            for blk in list(m.decoder.layers):
                hh = torch.randn(1, x.shape[1], m.emb.embedding_dim)
                f, _ = count(lambda: blk.ssm(hh, torch.ones(1, x.shape[1], dtype=torch.bool))) if hasattr(blk, "ssm") else (0, 0)
                ssm_f += f
                ssm_full += count.last_full if hasattr(blk, "ssm") else 0
        comps = dict(encoder=f_enc, variance=f_var, decoder=f_dec, flow_per_step=f_flow1, flow_10_steps=10 * f_flow1,
                     vocos=f_voc, mamba3_branches=ssm_f, acoustic_total=f_enc + f_var + f_dec,
                     base_plus_vocos=f_enc + f_var + f_dec + f_voc,
                     with_flow10=f_enc + f_var + f_dec + 10 * f_flow1 + f_voc)
        F = FULL
        full = dict(encoder=F["encoder"], variance=F["variance"], decoder=F["decoder"],
                    flow_per_step=F["flow_per_step"], flow_10_steps=10 * F["flow_per_step"], vocos=F["vocos"],
                    mamba3_branches=ssm_full, acoustic_total=F["encoder"] + F["variance"] + F["decoder"],
                    base_plus_vocos=F["encoder"] + F["variance"] + F["decoder"] + F["vocos"],
                    with_flow10=F["encoder"] + F["variance"] + F["decoder"] + 10 * F["flow_per_step"] + F["vocos"])
        for k, f in full.items():
            rows.append(dict(metric="gflops_full_per_audio_s", value=f / 1e9 / aud, p10=None, p90=None, n=1,
                             device="cpu", threads=None, system="steertts", component=k, set=s, length_s=aud,
                             tokens=N, frames=int(mel.shape[-1]), source_file="steer/paper/speed.py flops",
                             notes="matmul + elementwise/reduction/scan at 1 FLOP per output element, softmax 5, "
                                   "FFT 5 n log2 n"))
        print(f"[flops-full {s}] " + "  ".join(f"{k} {v / 1e9 / aud:.2f}" for k, v in full.items()), flush=True)
        for k, f in comps.items():
            rows.append(dict(metric="gflops_per_audio_s", value=f / 1e9 / aud, p10=None, p90=None, n=1, device="cpu",
                             threads=None, system="steertts", component=k, set=s, length_s=aud, tokens=N,
                             frames=int(mel.shape[-1]), source_file="steer/paper/speed.py flops",
                             notes="torch.utils.flop_counter (matmul/conv/attention; elementwise + FFT not counted)"))
        print(f"[flops {s}] audio {aud:.1f}s " + "  ".join(f"{k} {v / 1e9 / aud:.2f}" for k, v in comps.items()),
              "GFLOP/s-audio", flush=True)
    # random-init param-matched mixers at fixed lengths
    for arch in ARCHS:
        mm = random_model(arch, "cpu")
        for L in [5.0, 30.0, 120.0]:
            T = int(round(L * 93.75)); N = int(round(L * TOK_PER_S))
            f, _ = count(lambda: acoustic_fixed(mm, N, T, "cpu"))
            rows.append(dict(metric="gflops_full_per_audio_s", value=count.last_full / 1e9 / L, p10=None, p90=None,
                             n=1, device="cpu", threads=None, system=arch, component="acoustic_total", set=None,
                             length_s=L, tokens=N, frames=T, variant="random_init",
                             source_file="steer/paper/speed.py flops", notes="full count (see flop_count)"))
            print(f"[flops-full {arch}] {L}s {count.last_full / 1e9 / L:.2f} GFLOP per audio second", flush=True)
            rows.append(dict(metric="gflops_per_audio_s", value=f / 1e9 / L, p10=None, p90=None, n=1, device="cpu",
                             threads=None, system=arch, component="acoustic_total", set=None, length_s=L, tokens=N,
                             frames=T, variant="random_init", source_file="steer/paper/speed.py flops",
                             notes="param-matched mixer config, random weights, fixed durations"))
            print(f"[flops {arch}] {L}s {f / 1e9 / L:.2f} GFLOP per audio second", flush=True)
    save_rows(rows, a.out)


# ------------------------------------------------------------------------------------------------ streaming
STREAM_TEXT = ("Thanks for calling. Your order shipped this morning and should arrive on Thursday. "
               "If you need to change the delivery address, just say so now. Otherwise, have a great day!")


def bench_stream(a):
    """Time to first audio of MarkupSynth.stream (sentence-level streaming), warm; and total stream time."""
    from steer.markup import MarkupSynth
    set_threads(a.threads)
    dev = a.device
    S = load_synth(dev, flow=a.flow)
    if not a.flow:
        S.flow = None
    ms = MarkupSynth(S)
    texts = [STREAM_TEXT, TEXTS["longform"][0], "Hello! " + TEXTS["medium"][0]]
    first, total, auds, firstaud = [], [], [], []
    for i in range(3 + a.n_runs):
        text = texts[i % len(texts)]
        _sync(dev); t0 = time.perf_counter()
        it = ms.stream(text, voice=VOICE)
        c0 = next(it)
        t1 = time.perf_counter()
        rest = [c.audio for c in it]
        t2 = time.perf_counter()
        if i >= 3:
            first.append((t1 - t0) * 1000); total.append((t2 - t0) * 1000)
            firstaud.append(len(c0.audio) / 24000); auds.append((len(c0.audio) + sum(map(len, rest))) / 24000)
    v = "flow" if a.flow else "base"
    rows = [row("first_audio_ms", first, system=f"steertts-{v}", device=dev, threads=a.threads, variant=v,
                set_name="stream", notes="MarkupSynth.stream: call -> first sentence chunk on host (warm)",
                first_chunk_audio_s=float(np.median(firstaud)), **env_info()),
            row("stream_total_rtfx", [x / (t / 1000) for x, t in zip(auds, total)], system=f"steertts-{v}",
                device=dev, threads=a.threads, variant=v, set_name="stream", **env_info())]
    save_rows(rows, a.out)
    print(f"[stream {v} {dev} thr={a.threads}] first audio {np.median(first):.1f} ms (p90 {_q(first, 90):.1f}); "
          f"first chunk {np.median(firstaud):.2f}s audio; total RTFx {rows[1]['value']:.1f}",
          flush=True)


# ------------------------------------------------------------------------------------------------ baselines
def _kokoro(device):
    import torch
    from kokoro import KPipeline, KModel
    model = KModel().to(device).eval()
    pipe = KPipeline(lang_code="a", model=model, device=device)

    def fn(text):
        with torch.inference_mode():
            y = np.concatenate([(a.cpu().numpy() if torch.is_tensor(a) else a) for _, _, a in pipe(text, voice="af_heart")])
        return y, 24000
    return fn, dict(params_M=sum(p.numel() for p in model.parameters()) / 1e6, group="closest",
                    vocoder="iSTFTNet decoder (inside the 82M)")


def _piper(device):
    from piper import PiperVoice
    d = _root() / "runs/bench/_ext/_piper_models"
    kw = {}
    if device.startswith("cuda"):
        kw["use_cuda"] = True
    v = PiperVoice.load(str(d / "en_US-lessac-medium.onnx"), **kw)
    th = os.environ.get("SPEED_THREADS")
    if th and not device.startswith("cuda"):  # ORT defaults to one thread per physical core: oversubscribes a pin
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = int(th), 1
        v.session = ort.InferenceSession(str(d / "en_US-lessac-medium.onnx"), sess_options=so,
                                         providers=["CPUExecutionProvider"])

    def fn(text):
        chunks = list(v.synthesize(text))
        return np.concatenate([c.audio_float_array for c in chunks]), v.config.sample_rate
    params = None
    try:
        import onnx
        mo = onnx.load(str(d / "en_US-lessac-medium.onnx"))
        params = sum(int(np.prod(t.dims)) for t in mo.graph.initializer) / 1e6
    except Exception:  # noqa: BLE001 - onnx not in ttsbase: counted once with onnx 1.15 (env nemo), graph initializers
        params = 15.650556
    return fn, dict(params_M=params, group="closest", vocoder="VITS (end-to-end, includes waveform decoder)",
                    runtime="onnxruntime " + __import__("onnxruntime").__version__)


def _parler_mini(device):
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    name = "parler-tts/parler-tts-mini-v1"
    m = ParlerTTSForConditionalGeneration.from_pretrained(name).to(device).eval()
    tok = AutoTokenizer.from_pretrained(name)
    desc = ("A female speaker delivers clear, natural speech at a moderate speed and pitch in a very clean, "
            "close-sounding recording.")
    d = tok(desc, return_tensors="pt").to(device)

    def gen(text):
        p = tok(text, return_tensors="pt").to(device)
        torch.manual_seed(0)
        with torch.inference_mode():
            a = m.generate(input_ids=d.input_ids, attention_mask=d.attention_mask, prompt_input_ids=p.input_ids,
                           prompt_attention_mask=p.attention_mask)
        return a.cpu().numpy().squeeze()

    def fn(text):
        # Parler generates <= ~30 s per call: longer inputs are split into sentences and concatenated
        if len(text) > 400:
            parts = [x for x in re.split(r"(?<=[.!?])\s+", text) if x]
            return np.concatenate([gen(x) for x in parts]), m.config.sampling_rate
        return gen(text), m.config.sampling_rate
    return fn, dict(params_M=sum(p.numel() for p in m.parameters()) / 1e6, group="advanced",
                    vocoder="DAC codec decoder (inside)", notes_extra="autoregressive codec LM")


BUILTIN = dict(kokoro=_kokoro, piper=_piper, parler_mini=_parler_mini)


def load_ext(name, device):
    if name in BUILTIN:
        return BUILTIN[name](device)
    from steer.paper import baselines_ext as be
    ent = be.SYSTEMS[name]
    if "load" in ent:
        r = ent["load"](device)
        fn = r[0] if isinstance(r, tuple) else r
    else:
        fn = ent["synth"]
    if getattr(fn, "_speed_file_api", True) and _takes_two(fn):
        fn = file_fn(fn)
    info = {k: v for k, v in ent.items() if isinstance(v, (str, int, float, bool)) or v is None}
    return fn, info


def _takes_two(fn):
    import inspect
    try:
        return len([p for p in inspect.signature(fn).parameters.values() if p.default is p.empty]) >= 2
    except (TypeError, ValueError):
        return False


def bench_ext(a):
    set_threads(a.threads)
    dev = a.device
    try:
        fn, info = load_ext(a.system, dev)
    except Exception as e:  # noqa: BLE001
        print(f"[{a.system}] load failed on {dev}: {type(e).__name__}: {e}", flush=True)
        save_rows([row("error", [], system=a.system, device=dev, threads=a.threads,
                       notes=f"load failed: {type(e).__name__}: {str(e)[:200]}")], a.out)
        return
    sets = a.sets.split(",") if a.sets else None
    rows = time_system(fn, device=dev, threads=a.threads, n_runs=a.n_runs, system=a.system, sets=sets,
                       budget_s=a.budget, min_runs=a.min_runs, extra=info, notes=a.notes)
    if a.flops:
        try:
            import torch
            from torch.utils.flop_counter import FlopCounterMode
            torch.backends.mha.set_fastpath_enabled(False)  # fused MHA is invisible to the counter
            fr = []
            for s in ["medium"]:
                mmf, fullf, (y, sr) = flop_count(lambda: fn(TEXTS[s][0]))
                aud = len(np.asarray(y).reshape(-1)) / sr
                fr.append(dict(metric="gflops_full_per_audio_s", value=fullf / 1e9 / aud, p10=None, p90=None, n=1,
                               device=dev, threads=a.threads, system=a.system, set=s, length_s=aud,
                               component="end_to_end", source_file="steer/paper/speed.py ext --flops",
                               notes="full count (see flop_count); torch ops only", **info))
                fr.append(dict(metric="gflops_per_audio_s", value=mmf / 1e9 / aud, p10=None,
                               p90=None, n=1, device=dev, threads=a.threads, system=a.system, set=s, length_s=aud,
                               component="end_to_end", source_file="steer/paper/speed.py ext --flops",
                               notes="torch.utils.flop_counter over one call (matmul/conv/SDPA; LSTM/RNN, elementwise, FFT and "
                                     "non-torch ops such as ONNX are not counted: a lower bound)", **info))
                print(f"[{a.system}] {fr[-1]['value']:.2f} GFLOP per audio second", flush=True)
            rows += fr
        except Exception as e:  # noqa: BLE001
            print(f"[{a.system}] flops failed: {e}", flush=True)
    save_rows(rows, a.out)


def bench_hifigan(a):
    """Upsampling-vocoder reference: HiFi-GAN V1 (Matcha-TTS's copy of the generator, random weights - speed and FLOPs
    do not depend on the weights) vs length, same protocol as the Vocos rows in bench_scaling. Run in env ttsbase."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from matcha.hifigan.models import Generator
    from matcha.hifigan.config import v1
    from matcha.hifigan.env import AttrDict
    set_threads(a.threads)
    dev = a.device
    g = Generator(AttrDict(v1)).to(dev).eval()
    params = nparams(g) / 1e6
    rows = []
    with torch.no_grad():
        mmf, fullf, y = flop_count(lambda: g(torch.randn(1, 80, 86, device=dev)))
        rows.append(dict(metric="gflops_full_per_audio_s", value=fullf / 1e9 / (y.shape[-1] / 22050),
                         p10=None, p90=None, n=1, device=dev, threads=a.threads, system="hifigan-v1",
                         component="vocoder", set=None, length_s=y.shape[-1] / 22050, params_M=params,
                         source_file="steer/paper/speed.py hifigan", notes="full count; random weights"))
        rows.append(dict(metric="gflops_per_audio_s", value=mmf / 1e9 / (y.shape[-1] / 22050),
                         p10=None, p90=None, n=1, device=dev, threads=a.threads, system="hifigan-v1",
                         component="vocoder", set=None, length_s=y.shape[-1] / 22050, params_M=params,
                         source_file="steer/paper/speed.py hifigan", notes="random weights; 22.05 kHz, hop 256"))
        for L in [float(x) for x in a.lengths.split(",")]:
            T = int(round(L * 22050 / 256)); ts = []
            try:
                mel = torch.randn(1, 80, T, device=dev)
                if dev.startswith("cuda"):
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    base_mem = torch.cuda.memory_allocated()
                tb = time.perf_counter()
                for i in range(3 + a.n_runs):
                    _sync(dev); t0 = time.perf_counter()
                    g(mel)
                    _sync(dev)
                    if i >= 3:
                        ts.append((time.perf_counter() - t0) * 1000)
                    if i >= 6 and time.perf_counter() - tb > a.budget:
                        break
                peak = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20 if dev.startswith("cuda") else None
            except Exception as e:  # noqa: BLE001
                print("hifigan", L, type(e).__name__, str(e)[:100], flush=True)
                break
            rows.append(row("scal_latency_ms", ts, system="hifigan-v1", device=dev, threads=a.threads, length_s=L,
                            variant="eager", frames=T, params_M=params, **env_info()))
            if peak is not None:
                rows.append(row("mem_peak_mb", [peak], system="hifigan-v1", device=dev, threads=a.threads,
                                length_s=L, variant="eager", frames=T))
            print(f"[hifigan-v1 {dev}] {L:6.1f}s {np.median(ts):9.2f} ms", flush=True)
    save_rows(rows, a.out)


def bench_llm(a):
    """LLM front end (language section): time the few-shot Ollama call of steer.paper.adapter_llm exactly as used
    for the paper's predictions (CPU, num_thread=8, greedy, JSON-constrained). Ollama's own per-call durations are
    recorded (load / prompt eval / generation) next to the client wall time. The Ollama server is NOT pinned with
    taskset (it is a separate process); the load average is recorded per row."""
    from steer.paper.adapter_llm import SYSTEM, SHOTS, all_prompts
    import urllib.request
    model = a.system or "qwen2.5:3b"
    host = "http://localhost:11434"
    msgs0 = [{"role": "system", "content": SYSTEM}]
    for p, ans in SHOTS:
        msgs0 += [{"role": "user", "content": p}, {"role": "assistant", "content": json.dumps(ans)}]
    th = a.threads or 8

    def call(prompt):
        body = dict(model=model, messages=msgs0 + [{"role": "user", "content": prompt}], stream=False, format="json",
                    keep_alive="30m", options=dict(temperature=0, num_gpu=0, num_thread=th, num_ctx=4096,
                                                   num_predict=200, seed=0))
        req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        r = json.load(urllib.request.urlopen(req, timeout=900))
        return time.perf_counter() - t0, r
    prompts = all_prompts()
    cold_s, cold = call(prompts[0])  # loads the model (load_duration) and fills the prompt cache
    for i in range(2):
        call(prompts[1 + i])
    wall, tot, pe, ev, ntok, nprompt = [], [], [], [], [], []
    t_start = time.perf_counter()
    k = 0
    while k < a.n_runs and (k < a.min_runs or time.perf_counter() - t_start < a.budget):
        w, r = call(prompts[(3 + k) % len(prompts)])
        wall.append(w * 1000); tot.append(r.get("total_duration", 0) / 1e6)
        pe.append(r.get("prompt_eval_duration", 0) / 1e6); ev.append(r.get("eval_duration", 0) / 1e6)
        ntok.append(r.get("eval_count", 0)); nprompt.append(r.get("prompt_eval_count", 0))
        k += 1
    e = dict(env_info(), model=model, params_B=3.09, component=None)
    common = dict(system=f"llm-{model}", device="cpu", threads=th, set_name="prompts",
                  source_file="steer/paper/speed.py llm")
    rows = [row("llm_wall_ms", wall, notes="client wall time per prompt (warm model, few-shot chat, JSON format)",
                **common, **dict(e, component="end_to_end")),
            row("llm_ms", tot, notes="ollama total_duration", **common, **dict(e, component="total")),
            row("llm_ms", pe, notes="ollama prompt_eval_duration (system + 4 shots + prompt; prefix cache may apply)",
                **common, **dict(e, component="prompt_eval")),
            row("llm_ms", ev, notes="ollama eval_duration (JSON generation)", **common, **dict(e, component="generate")),
            row("llm_tokens", ntok, notes="generated tokens (eval_count)", **common, **dict(e, component="generate")),
            row("llm_tokens", nprompt, notes="prompt tokens evaluated (prompt_eval_count)", **common,
                **dict(e, component="prompt_eval")),
            row("llm_cold_ms", [cold_s * 1000], notes=f"first call incl. model load "
                f"(load_duration {cold.get('load_duration', 0) / 1e6:.0f} ms)", **common, **dict(e, component="cold"))]
    save_rows(rows, a.out)
    print(f"[llm {model} thr={th}] wall {np.median(wall):.0f} ms (p10 {_q(wall, 10):.0f}, p90 {_q(wall, 90):.0f}); "
          f"prompt_eval {np.median(pe):.0f} ms / {np.median(nprompt):.0f} tok; gen {np.median(ev):.0f} ms / "
          f"{np.median(ntok):.0f} tok; cold {cold_s * 1000:.0f} ms; n={len(wall)} load {e['loadavg_1m']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["steer", "scaling", "flops", "stream", "ext", "hifigan", "llm"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--n_runs", type=int, default=20)
    ap.add_argument("--min_runs", type=int, default=5)
    ap.add_argument("--budget", type=float, default=90.0, help="seconds per (system, set) after min runs")
    ap.add_argument("--sets", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="base,flow,clone")
    ap.add_argument("--prompt", action="store_true")
    ap.add_argument("--lengths", default="2,5,10,20,40,80,160,320,640")
    ap.add_argument("--archs", default="hybrid,attn,mamba3")
    ap.add_argument("--modes", default="eager")
    ap.add_argument("--vocos", action="store_true")
    ap.add_argument("--flow", action="store_true")
    ap.add_argument("--system", default="")
    ap.add_argument("--flops", action="store_true")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    if a.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    t0 = time.time()
    dict(steer=bench_steer, scaling=bench_scaling, flops=bench_flops, stream=bench_stream, ext=bench_ext, hifigan=bench_hifigan, llm=bench_llm)[a.cmd](a)
    print(f"done {a.cmd} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
