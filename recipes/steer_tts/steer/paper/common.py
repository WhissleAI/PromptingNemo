"""Shared pieces for the paper experiments: held-out voices, cached-reference Synth, the two readback extractors,
Produced: library module (no results of its own). All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.
batched Whisper WER, bootstrap CIs and the result writers.

Held-out voices = LibriTTS-R test-clean speakers (data/bench/libritts.jsonl). None of them is in the base run's
speaker table (runs/spk_enc_v3/control_stats.json) or in any training manifest (data/meta/manifest_*.jsonl); checked
by heldout_check(). The model is conditioned only through the WavLM-SV x-vector of one real test-clean clip (ref_wav).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from ..config import CONTROL_DIMS, SR

ROOT = Path(os.environ.get("STEER_TTS_ROOT", "/home/karan/Desktop/whissle/workspace/steer_tts"))
RES = Path("/home/karan/Desktop/whissle/marketing_post_agent/paper/steertts/results")
WORK = ROOT / "runs" / "paper"
BASE = str(ROOT / "runs/spk_enc_v3/step312500.pt")
FLOW = str(ROOT / "runs/flow_v3/step100000.pt")
BENCH = ROOT / "data/bench/libritts.jsonl"

# ------------------------------------------------------------------------------------------------ held-out voices


def heldout_check(ckpt=BASE):
    stats = json.load(open(Path(ckpt).parent / "control_stats.json"))
    train = set(stats["speakers"])
    rows = [json.loads(l) for l in open(BENCH)]
    spk = sorted({r["speaker"] for r in rows})
    in_table = [s for s in spk if f"libritts_{s}" in train or s in train]
    in_manifest = set()
    for f in sorted((ROOT / "data/meta").glob("manifest_*.jsonl")):
        for l in open(f):
            sp = json.loads(l)["speaker"]
            if sp.startswith("libritts_") and sp[9:] in spk:
                in_manifest.add(sp)
    return dict(n_bench_speakers=len(spk), in_speaker_table=in_table, in_training_manifests=sorted(in_manifest),
                n_train_speakers=len(train))


def heldout_voices(n=16, min_ref=4.0, max_ref=10.0):
    """n test-clean speakers, gender-balanced by the bench's median-F0 (>=165 Hz -> F; LibriTTS gender labels are
    not on disk), each with one reference clip (conditioning) and its other clips (speaker reference statistics).
    Returns list of dict(vid, speaker, gender, ref, others=[(path, text)]). vid is an anonymous H01.. id."""
    rows = [json.loads(l) for l in open(BENCH)]
    by = {}
    for r in rows:
        by.setdefault(r["speaker"], []).append(r)
    cand = []
    for s, rs in by.items():
        if len(rs) < 6:
            continue
        durs = [sf.info(r["gt_path"]).duration for r in rs]
        ok = [i for i, d in enumerate(durs) if min_ref <= d <= max_ref]
        if not ok:
            continue
        ref = rs[min(ok, key=lambda i: abs(durs[i] - 6.0))]
        cand.append(dict(speaker=s, f0=float(np.median([r["spk_f0"] for r in rs])), ref=ref["gt_path"],
                         others=[(r["gt_path"], r["text"]) for r in rs if r is not ref], n=len(rs)))
    F = sorted([c for c in cand if c["f0"] >= 165], key=lambda c: -c["n"])
    M = sorted([c for c in cand if c["f0"] < 165], key=lambda c: -c["n"])
    k = n // 2
    pick = sorted(F[:k], key=lambda c: c["speaker"]) + sorted(M[:n - k], key=lambda c: c["speaker"])
    for i, c in enumerate(pick):
        c["vid"] = f"H{i + 1:02d}"
        c["gender"] = "F" if c["f0"] >= 165 else "M"
    priv = WORK / "heldout_voice_map_private.json"  # real ids stay out of public results
    priv.parent.mkdir(parents=True, exist_ok=True)
    json.dump({c["vid"]: c["speaker"] for c in pick}, open(priv, "w"), indent=1)
    return pick


def load24(path):
    import soxr
    y, sr = sf.read(str(path), dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    return soxr.resample(y, sr, SR) if sr != SR else y


def make_synth(ckpt=BASE, **kw):
    """steer.synth.Synth whose reference x-vector is computed once per reference clip (not once per call)."""
    from ..synth import Synth

    class CachedSynth(Synth):
        _vec = {}

        def speaker_vector(self, speaker=None, ref_wav=None):
            if ref_wav is None or not isinstance(ref_wav, (str, Path)):
                return super().speaker_vector(speaker, ref_wav)
            k = str(ref_wav)
            if k not in self._vec:
                self._vec[k] = super().speaker_vector(speaker, ref_wav)
            return self._vec[k]

    return CachedSynth(ckpt, **kw)


# ------------------------------------------------------------------------------------------------ extractor A (ours)
def n_phones(text):
    from ..text import to_phonemes, SP, SYMBOLS
    punct = {i for i, s in enumerate(SYMBOLS) if len(s) == 1}
    ids, _, _ = to_phonemes(text)
    return sum(1 for i in ids if i != SP and i not in punct)


def ours(y, text):
    from ..features import measure
    return dict(zip(CONTROL_DIMS, measure(np.asarray(y, np.float32), n_phones=n_phones(text), want_mel=False)["controls"].tolist()))


# ------------------------------------------------------------------------------------------------ extractor B (Praat)
# Independent readback: different pitch algorithm (Praat cross-correlation, fixed 75-600 Hz), Praat intensity /
# BS.1770 loudness, Praat silence detection for pauses / speech span, de Jong & Wempe syllable nuclei for rate,
# Praat CPPS + HNR, Welch-spectrum tilt slope, Praat centre of gravity, H1-H2 at Praat F0 with a 2048-pt window.
# Maps onto our dims: PRAAT_MAP[our_dim] = praat feature used as its independent counterpart.
PRAAT_MAP = {"rate": "p_rate_nuclei", "loud": "p_lufs", "f0_med": "p_f0_med", "f0_iqr": "p_f0_iqr",
             "f0_end": "p_f0_end", "pause": "p_pause", "tilt": "p_tilt_slope", "voiced": "p_voiced",
             "flat": "p_flatness", "centroid": "p_cog", "cpps": "p_cpps", "h1h2": "p_h1h2"}
PRAAT_EXTRA = ["p_intensity", "p_hnr", "p_rate_text"]
PRAAT_FEATS = list(PRAAT_MAP.values()) + PRAAT_EXTRA


def _syllables(text):
    import re
    from g2p_en import G2p  # noqa: F401 (cmudict via nltk)
    from nltk.corpus import cmudict
    global _CMU
    try:
        _CMU
    except NameError:
        _CMU = cmudict.dict()
    n = 0
    for w in re.findall(r"[a-z']+", text.lower()):
        if w in _CMU:
            n += sum(ch[-1].isdigit() for ch in _CMU[w][0])
        else:
            n += max(1, len(re.findall(r"[aeiouy]+", w)))
    return n


def praat(y, text, sr=SR):
    import parselmouth
    from parselmouth.praat import call
    y = np.asarray(y, np.float64)
    snd = parselmouth.Sound(y, sr)
    out = dict.fromkeys(PRAAT_FEATS, np.nan)
    dur = snd.get_total_duration()
    # speech span + pauses: Praat silence detection (-25 dB re max, min silence 0.1 s, min sounding 0.05 s)
    tg = call(snd, "To TextGrid (silences)", 100, 0.0, -25.0, 0.1, 0.05, "silent", "sounding")
    n_int = call(tg, "Get number of intervals", 1)
    ivs = [(call(tg, "Get start time of interval", 1, i), call(tg, "Get end time of interval", 1, i),
            call(tg, "Get label of interval", 1, i)) for i in range(1, n_int + 1)]
    snd_iv = [(a, b) for a, b, l in ivs if l == "sounding"]
    if not snd_iv:
        return out
    t0, t1 = snd_iv[0][0], snd_iv[-1][1]
    span = t1 - t0
    sil_inside = sum(b - a for a, b, l in ivs if l == "silent" and a >= t0 - 1e-6 and b <= t1 + 1e-6)
    out["p_pause"] = sil_inside / max(span, 1e-3)
    phon = max(span - sil_inside, 0.2)
    sound_mask = lambda t: np.any([(t >= a) & (t < b) for a, b in snd_iv], axis=0)
    # loudness: ITU-R BS.1770 integrated loudness of the speech span; Praat mean intensity (energy-averaged)
    try:
        import pyloudnorm as pyln
        seg = y[int(t0 * sr):int(t1 * sr)]
        if len(seg) > 0.4 * sr:
            out["p_lufs"] = float(pyln.Meter(sr).integrated_loudness(seg))
    except Exception:
        pass
    inten = snd.to_intensity(minimum_pitch=100.0)
    iv = inten.values[0]
    it = inten.xs()
    m = sound_mask(it)
    if m.any():
        out["p_intensity"] = float(10 * np.log10(np.mean(10 ** (iv[m] / 10))))
    # pitch: Praat cross-correlation, fixed range
    p = snd.to_pitch_cc(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
    f = p.selected_array["frequency"]
    pt = p.xs()
    pm = sound_mask(pt) if len(pt) else np.zeros(0, bool)
    if pm.any():
        out["p_voiced"] = float((f[pm] > 0).mean())
    v = (f > 0) & pm
    if v.sum() >= 10:
        st = 12 * np.log2(f[v] / 100.0)
        out["p_f0_med"] = float(np.median(st))
        q25, q75 = np.percentile(st, [25, 75])
        out["p_f0_iqr"] = float(q75 - q25)
        tv = pt[v]
        last = tv > tv[-1] - 0.3
        if last.sum() >= 8:
            out["p_f0_end"] = float(np.polyfit(tv[last], st[last], 1)[0])
    # syllable nuclei (de Jong & Wempe 2009, simplified): intensity peaks >= threshold, >= 2 dB dip from previous
    # counted peak, voiced at the peak; rate = nuclei / phonation time.  Text-based: CMUdict syllables / phonation time.
    if len(iv) > 3:
        thr = max(np.median(iv), iv.max() - 25.0)
        pk = [i for i in range(1, len(iv) - 1) if iv[i] >= iv[i - 1] and iv[i] > iv[i + 1] and iv[i] > thr]
        kept = []
        for i in pk:
            if kept:
                dip = iv[kept[-1]:i + 1].min()
                if min(iv[kept[-1]], iv[i]) - dip < 2.0:
                    if iv[i] > iv[kept[-1]]:
                        kept[-1] = i
                    continue
            kept.append(i)
        nuc = 0
        for i in kept:
            fv = p.get_value_at_time(it[i])
            nuc += int(fv == fv and fv > 0)  # not NaN
        out["p_rate_nuclei"] = nuc / phon
    out["p_rate_text"] = _syllables(text) / phon
    # periodicity: Praat CPPS (Hillenbrand defaults) on the speech span, HNR (cc) mean over voiced frames
    try:
        part = snd.extract_part(from_time=t0, to_time=t1, preserve_times=False)
        pc = call(part, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        out["p_cpps"] = float(call(pc, "Get CPPS", "no", 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0.05,
                                   "Straight", "Robust"))
        h = part.to_harmonicity_cc(time_step=0.01, minimum_pitch=75.0)
        hv = h.values[0]
        hv = hv[hv > -200]
        if len(hv):
            out["p_hnr"] = float(hv.mean())
        spec = part.to_spectrum()
        out["p_cog"] = float(np.log(call(spec, "Get centre of gravity", 2.0) + 1))
    except Exception:
        pass
    # long-term spectrum of sounding samples (Welch, 2048): tilt = slope dB/octave over 100-5000 Hz; flatness 50-8 kHz
    from scipy.signal import welch
    keep = np.zeros(len(y), bool)
    for a, b in snd_iv:
        keep[int(a * sr):int(b * sr)] = True
    ys = y[keep]
    if len(ys) > 4096:
        fr, P = welch(ys, sr, nperseg=2048)
        b = (fr >= 100) & (fr <= 5000)
        out["p_tilt_slope"] = float(np.polyfit(np.log2(fr[b]), 10 * np.log10(P[b] + 1e-20), 1)[0])
        # per-segment flatness (Welch-averaged spectra over 2048-pt frames is too smooth): 2048-pt frames
        n = 2048
        fr_ = np.fft.rfftfreq(n, 1 / sr)
        bb = (fr_ >= 50) & (fr_ < 8000)
        frames = np.lib.stride_tricks.sliding_window_view(ys, n)[::1024]
        S = np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2 + 1e-12
        S = S[:, bb]
        out["p_flatness"] = float(np.log(np.mean(np.exp(np.log(S).mean(1)) / S.mean(1))))
    # H1-H2 at Praat F0, 40 ms Hann frames, 4096-pt FFT
    if v.sum() >= 10:
        vals = []
        for t, f0 in zip(pt[v][::2], f[v][::2]):
            a = int((t - 0.02) * sr)
            if a < 0 or a + int(0.04 * sr) > len(y):
                continue
            fr_ = y[a:a + int(0.04 * sr)] * np.hanning(int(0.04 * sr))
            S = 20 * np.log10(np.abs(np.fft.rfft(fr_, 4096)) + 1e-10)
            fq = np.fft.rfftfreq(4096, 1 / sr)
            m1 = (fq > 0.9 * f0) & (fq < 1.1 * f0)
            m2 = (fq > 1.9 * f0) & (fq < 2.1 * f0)
            if m1.any() and m2.any():
                vals.append(S[m1].max() - S[m2].max())
        if vals:
            out["p_h1h2"] = float(np.median(vals))
    return out


def measure_both(args):
    """worker: (key, wav path, text) -> dict(key, ours..., praat...)"""
    key, path, text = args
    y, sr = sf.read(str(path), dtype="float32")
    d = dict(key=key)
    try:
        d.update(ours(y, text))
    except Exception:
        pass
    try:
        d.update(praat(y, text, sr))
    except Exception:
        pass
    return d


def measure_many(items, procs=40):
    """items: list of (key, path, text). CPU pool (Praat is single-threaded)."""
    import multiprocessing as mp
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with mp.get_context("spawn").Pool(procs, initializer=_init_worker) as pool:
        out = pool.map(measure_both, items, chunksize=8)
    return pd.DataFrame(out)


def _init_worker():
    torch.set_num_threads(1)


# ------------------------------------------------------------------------------------------------ ASR (Whisper)
_NORM = None


def normalize(s):
    global _NORM
    if _NORM is None:
        from transformers import WhisperTokenizer
        _NORM = WhisperTokenizer.from_pretrained("openai/whisper-large-v3").normalize
    return _NORM(s)


def word_errors(ref, hyp):
    r, h = normalize(ref).split(), normalize(hyp).split()
    d = np.arange(len(h) + 1)
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return int(d[-1]), len(r)


def transcribe(paths, cache_file, model="openai/whisper-large-v3", batch=16, dev="cuda"):
    """Batched fp16 Whisper; cached by content md5 in cache_file (jsonl). Returns {path: hyp}."""
    import soxr
    cache = {}
    cache_file = Path(cache_file)
    if cache_file.exists():
        for l in open(cache_file):
            d = json.loads(l)
            cache[d["md5"]] = d["hyp"]
    md5 = {str(p): hashlib.md5(Path(p).read_bytes()).hexdigest() for p in paths}
    todo = sorted({p for p, m in md5.items() if m not in cache})
    if todo:
        gpu_wait()
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        proc = WhisperProcessor.from_pretrained(model)
        asr = WhisperForConditionalGeneration.from_pretrained(model, dtype=torch.float16).to(dev).eval()
        batch = int(os.environ.get("PAPER_ASR_BATCH", batch))
        print(f"asr: {model} fp16, {len(todo)} clips, batch {batch}", flush=True)
        with open(cache_file, "a") as cf:
            for i in range(0, len(todo), batch):
                ch = todo[i:i + batch]
                ys = []
                for p in ch:
                    y, sr = sf.read(p, dtype="float32")
                    y = y.mean(1) if y.ndim > 1 else y
                    ys.append(soxr.resample(y, sr, 16000) if sr != 16000 else y)
                f = proc(ys, sampling_rate=16000, return_tensors="pt").input_features
                with torch.no_grad():
                    out = asr.generate(f.to(dev, torch.float16), language="en", task="transcribe", max_new_tokens=200)
                for p, h in zip(ch, proc.batch_decode(out, skip_special_tokens=True)):
                    cache[md5[p]] = h.strip()
                    cf.write(json.dumps(dict(md5=md5[p], path=p, hyp=h.strip())) + "\n")
                if (i // batch) % 50 == 0:
                    print(f"  asr {min(i + batch, len(todo))}/{len(todo)}", flush=True)
        del asr
        import gc
        gc.collect(); torch.cuda.empty_cache()
    hyps = {p: cache[m] for p, m in md5.items()}
    empty = sum(1 for h in hyps.values() if not h)
    print(f"asr: {empty}/{len(hyps)} empty transcripts", flush=True)
    assert empty < 0.05 * max(1, len(hyps)), "too many empty transcripts (GPU contention?)"
    return hyps


# ------------------------------------------------------------------------------------------------ stats + output
def boot_ci(units, stat, B=1000, seed=0):
    """units: list of per-unit arrays/frames grouped by bootstrap unit; stat(list_of_units) -> float.
    Returns (value, lo, hi) of the percentile 95% CI."""
    rng = np.random.default_rng(seed)
    val = stat(units)
    n = len(units)
    bs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            bs.append(stat([units[i] for i in idx]))
        except Exception:
            pass
    bs = np.array([b for b in bs if b == b])
    if len(bs) < 10:
        return val, np.nan, np.nan
    return val, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_df(df, unit_cols, fn, B=1000, seed=0):
    """Cluster bootstrap over unique values of unit_cols: fn(sub_df) -> float."""
    df = df.reset_index(drop=True)
    groups = list(df.groupby(unit_cols, sort=False).indices.values())
    return boot_ci(groups, lambda gs: fn(df.iloc[np.concatenate(gs)]), B=B, seed=seed)


def row(metric, v, n, source_file, scorer, split, **cond):
    val, lo, hi = v if isinstance(v, tuple) else (v, np.nan, np.nan)
    f = lambda x: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)
    return dict(metric=metric, value=f(val), ci_lo=f(lo), ci_hi=f(hi), n=int(n), source_file=str(source_file),
                scorer=scorer, split=split, **cond)


def write(name, rows, raw_df, meta=None):
    RES.mkdir(parents=True, exist_ok=True)
    out = dict(name=name, meta=meta or {}, rows=rows)
    json.dump(out, open(RES / f"{name}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    raw_df.to_csv(RES / f"{name}_raw.csv", index=False)
    print("wrote", RES / f"{name}.json", len(rows), "rows;", RES / f"{name}_raw.csv", len(raw_df), flush=True)


def boot_ratio(df, unit_cols, num, den=None, B=1000, seed=0):
    """Fast cluster bootstrap of sum(num)/sum(den) (den=None: mean of num over non-NaN rows). num/den: column names
    or Series aligned with df. Resampling weights = multinomial unit counts, so identical to boot_df for these stats."""
    x = df[num] if isinstance(num, str) else num
    x = pd.Series(np.asarray(x, float), index=df.index)
    d = pd.Series(np.isfinite(x).astype(float), index=df.index) if den is None else pd.Series(np.asarray(df[den] if isinstance(den, str) else den, float), index=df.index)
    keys = df[unit_cols].astype(str).agg("|".join, axis=1)
    s = x.fillna(0).groupby(keys, sort=False).sum().values
    c = d.where(np.isfinite(x), 0).groupby(keys, sort=False).sum().values if den is None else d.groupby(keys, sort=False).sum().values
    n = len(s)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = s.sum() / c.sum()
        bs = (W @ s) / (W @ c)
    bs = bs[np.isfinite(bs)]
    return (float(val), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if len(bs) > 10 else (float(val), np.nan, np.nan)


def gpu_wait(poll=15):
    """Block while the shared GPU queue (logs/gpu_queue/running/) has a job, so our heavy scorers never share the GPU."""
    import time
    if os.environ.get("PAPER_NO_GPU_WAIT") == "1":
        return
    d = ROOT / "logs/gpu_queue/running"
    t0 = time.time()
    while d.exists() and any(d.iterdir()):
        if int(time.time() - t0) % 300 < poll:
            print("gpu_wait: queue job running:", [p.name for p in d.iterdir()], flush=True)
        time.sleep(poll)
