"""External benchmark: LibriTTS-R test-clean + ParaSpeechCaps test (see docs/benchmarks.md for the literature numbers).

python -m steer.benchmark prep                                   # test lists + pitch/rate/gender calibration
python -m steer.benchmark run --ckpt m3hyb_58k=runs/mix_m3hyb/step58000_demo.pt [--ckpt ...] [--gt]
python -m steer.benchmark report                                 # runs/bench/report.md from every results json

Test sets (built by `prep` into data/bench/):
  libritts  500 LibriTTS-R test-clean utterances, 4-10 s, seed 0 (unseen speakers and text). The model is not
            zero-shot, so each utterance is voiced by one of 8 fixed LibriTTS-R training voices (4 low-, 4 high-F0)
            matched to the test speaker's gender by its real F0. Metrics: WER (whisper-large-v3, Whisper English
            normaliser), UTMOS22-strong, and SIM (WavLM-base-plus-sv) to a real clip of the voice used, i.e. voice
            consistency, NOT zero-shot cloning similarity.
  psc       ParaSpeechCaps `test` split: the paper's 246 tag-balanced prompts (VoxCeleb 137 / Expresso 83 / EARS 26).
            Description -> prompt adapter -> tag + controls; the voice is chosen from the description's gender only
            (as a description-conditioned model like Parler would). Metrics: WER with distil-whisper-large-v2 (the
            paper's ASR, so directly comparable to its table) and whisper-large-v3, UTMOS22, and objective
            attribute accuracy for the measurable basic tags - gender, pitch level (low/medium/high), speaking rate
            (slow/measured/fast) - with thresholds fitted on real EARS/Expresso audio against ParaSpeechCaps' own
            labels (so the label definition is theirs, the estimator ours). `leak` marks test rows whose audio is in
            our training data (Expresso read speech); EARS freeform is not trained on.
Outputs: runs/bench/<name>/<set>/*.wav, runs/bench/<name>.json (per-utterance + summary), runs/bench/report.md.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .config import META, RAW, ROOT, RUNS, SR

BENCH = ROOT / "data" / "bench"
OUT = RUNS / "bench"
PITCH = ["low-pitched", "medium-pitched", "high-pitched"]
RATE = ["slow speed", "measured speed", "fast speed"]


# ---------------------------------------------------------------------------------------------------- measurement
def load24(path_or_bytes):
    import soxr
    y, sr = sf.read(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    return soxr.resample(y, sr, SR) if sr != SR else y


def f0_and_speech_dur(y):
    """Median F0 (Hz) of voiced frames and speech duration (s) with leading/trailing silence trimmed."""
    import librosa
    from .features import pitch_track
    from .config import HOP
    T = len(y) // HOP + 1
    f0, _ = pitch_track(y, T)
    v = f0[f0 > 0]
    yt, _ = librosa.effects.trim(y, top_db=35)
    return (float(np.median(v)) if len(v) > 10 else float("nan")), len(yt) / SR


def n_phones(phonemes):
    return sum(1 for ch in phonemes if ch.isalpha() or ch in "ɐɑɒæɔəɛɜɪʊʌθðʃʒŋɹɾʔ")


def bins(x, cuts):
    return int(np.searchsorted(cuts, x))


def fit_cuts(x, y, k=3):
    """Two cut points maximising accuracy of searchsorted(cuts, x) == y (y in 0..k-1), grid over quantiles."""
    x, y = np.asarray(x), np.asarray(y)
    q = np.unique(np.nanquantile(x, np.linspace(0.01, 0.99, 99)))
    best = (-1, None)
    for i, a in enumerate(q):
        for b in q[i:]:
            acc = np.mean(np.searchsorted([a, b], x) == y)
            if acc > best[0]:
                best = (acc, [float(a), float(b)])
    return best[1], float(best[0])


# ---------------------------------------------------------------------------------------------------- prep
def psc_audio_path(row):
    if row.source == "ears":
        return RAW / "ears" / row.relative_audio_path
    if row.source == "expresso":
        # Expresso ships as parquet; the clips we extracted (8 of 12 shards) are in wav24/
        return ROOT / "data" / "wav24" / "expresso" / f"expresso_{Path(row.relative_audio_path).stem}.wav"
    return None


def prep_lj():
    """LJSpeech test split (the Tacotron2 filelist Matcha-TTS uses, 500 sentences): text only - every published
    small-model WER (FastSpeech2 / VITS / Matcha, Whisper-medium) is on these sentences."""
    rows = [dict(id=l.split("|")[0].split("/")[-1][:-4], text=l.split("|", 1)[1].strip())
            for l in open(BENCH / "ljs_test_filelist.txt") if "|" in l]
    pd.DataFrame(rows).to_json(BENCH / "ljspeech.jsonl", orient="records", lines=True)
    print("ljspeech", len(rows), "sentences", flush=True)


def prep_clone():
    """Zero-shot cloning set from the 500 LibriTTS-R test-clean items (36 speakers unseen in training): each item's
    voice prompt is a DIFFERENT test utterance of the same speaker; SIM is scored against the item's own real
    recording (SIM-o style). Only reference-conditioned checkpoints (--spk_enc) are run on it."""
    lt = pd.read_json(BENCH / "libritts.jsonl", lines=True)
    rows = []
    for spk, g in lt.groupby("speaker"):
        idx = g.index.tolist()
        for k, i in enumerate(idx):
            if len(idx) < 2:
                continue
            rows.append(dict(id=lt.id[i], speaker=spk, text=lt.text[i], gt_path=lt.gt_path[i],
                             ref_path=lt.gt_path[idx[(k + 1) % len(idx)]]))
    pd.DataFrame(rows).to_json(BENCH / "clone.jsonl", orient="records", lines=True)
    print("clone", len(rows), "items,", lt.speaker.nunique(), "speakers", flush=True)


def prep():
    BENCH.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # --- LibriTTS-R test-clean
    lt = pd.concat([pd.read_parquet(f) for f in sorted((RAW / "libritts_r/data/test.clean").glob("*.parquet"))])
    rows = []
    for _, r in lt.iterrows():
        y = load24(r.audio["bytes"])
        if 4.0 <= len(y) / SR <= 10.0:
            rows.append((r.id, r.speaker_id, r.text_normalized, r.audio["bytes"]))
    sel = rng.choice(len(rows), 500, replace=False)
    gt = BENCH / "libritts_gt"; gt.mkdir(exist_ok=True)
    spk_f0 = {}
    out = []
    for i in sorted(sel):
        uid, spk, text, b = rows[i]
        y = load24(b); sf.write(gt / f"{uid}.wav", y, SR)
        if spk not in spk_f0:
            spk_f0[spk] = []
        spk_f0[spk].append(f0_and_speech_dur(y)[0])
        out.append(dict(id=uid, speaker=spk, text=text, gt_path=str(gt / f"{uid}.wav")))
    for o in out:
        o["spk_f0"] = float(np.nanmedian(spk_f0[o["speaker"]]))
    pd.DataFrame(out).to_json(BENCH / "libritts.jsonl", orient="records", lines=True)
    print("libritts", len(out), "utterances,", len(spk_f0), "speakers", flush=True)

    prep_lj()
    prep_clone()

    # --- ParaSpeechCaps test + leakage flag
    t = pd.read_parquet(RAW / "paraspeechcaps/data/test-00000-of-00001.parquet")
    ctl = pd.read_parquet(META / "controls.parquet")
    trained = set(ctl.id)

    def our_id(r):
        p = Path(r.relative_audio_path)
        if r.source == "expresso":
            return "expresso_" + p.stem
        if r.source == "ears":
            return f"ears_{p.parent.name}_{p.stem}"
        return None
    t["our_id"] = t.apply(our_id, axis=1)
    t["leak"] = t.our_id.isin(trained)
    t["description"] = t.text_description.map(lambda a: str(a[0]).strip())
    t["text"] = t.transcription.str.strip()
    t["gt_path"] = [str(p) if p is not None and Path(p).exists() else None for p in t.apply(psc_audio_path, axis=1)]
    keep = ["source", "speakerid", "description", "text", "gender", "pitch", "speaking_rate", "tag_of_interest",
            "phonemes", "leak", "gt_path"]
    t[keep].to_json(BENCH / "psc.jsonl", orient="records", lines=True)
    print("psc", len(t), "prompts; leak", int(t.leak.sum()), "; gt audio", int(t["gt_path"].notna().sum()), flush=True)

    # --- calibration: our estimator on real EARS/Expresso audio vs ParaSpeechCaps' labels
    tb = pd.read_parquet(RAW / "paraspeechcaps/data/train_base-00000-of-00001.parquet")
    tb = tb[tb.source.isin(["ears", "expresso"])].sample(1500, random_state=0)
    cal = []
    for _, r in tb.iterrows():
        p = psc_audio_path(r)
        if p is None or not p.exists():
            continue
        f0, d = f0_and_speech_dur(load24(str(p)))
        cal.append(dict(f0=f0, rate=n_phones(r.phonemes) / max(d, 0.3), gender=r.gender, pitch=r.pitch,
                        speaking_rate=r.speaking_rate))
    cal = pd.DataFrame(cal).dropna()
    st = 12 * np.log2(cal.f0 / 100.0)
    g_y = (cal.gender == "female").astype(int).to_numpy()
    gq = np.unique(np.quantile(st, np.linspace(0.02, 0.98, 97)))
    g_acc = [np.mean((st > c).astype(int) == g_y) for c in gq]
    calib = dict(gender_cut_st=float(gq[int(np.argmax(g_acc))]), gender_acc=float(max(g_acc)), n=len(cal))
    for g in ["female", "male"]:
        m = cal.gender == g
        calib[f"pitch_cuts_{g}"], calib[f"pitch_acc_{g}"] = fit_cuts(st[m], cal.pitch[m].map(PITCH.index))
    calib["rate_cuts"], calib["rate_acc"] = fit_cuts(cal.rate, cal.speaking_rate.map(RATE.index))
    json.dump(calib, open(BENCH / "calibration.json", "w"), indent=1)
    print("calibration (accuracy on real audio = ceiling for these attribute metrics):", calib, flush=True)


# ---------------------------------------------------------------------------------------------------- scoring
# Scoring models are loaded ONE AT A TIME (synthesize everything first, then ASR v3 -> distil -> UTMOS -> WavLM):
# training (~5 GB) and the demo share the 12 GB GPU, and all scorers together need ~5.5 GB.
def to16(y):
    import soxr
    return soxr.resample(y.astype(np.float32), SR, 16000)


def free():
    import gc
    gc.collect(); torch.cuda.empty_cache()


def word_errors(norm, ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    d = np.arange(len(h) + 1)
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return int(d[-1]), len(r)


def score_asr(items, model, key, dev="cuda", batch=6):  # whisper cross-attn cache ~0.25 GB/clip: 6 fits beside training
    """items: list of (row dict, wav path, reference text); adds err_<key> / nwords / hyp_<key> to each row.
    Batched (one clip at a time left the GPU ~75% idle: 1h+ for 2.7k clips) and cached in OUT/_asr_<key>.jsonl keyed
    by (path, content md5), so re-scoring the same audio is free (external audio is re-copied every run)."""
    cache_f = OUT / f"_asr_{key}.jsonl"
    cache = {}
    if cache_f.exists():
        for l in open(cache_f):
            d = json.loads(l); cache[(d["path"], d["mtime"])] = d["hyp"]
    import hashlib
    keyed = [(row, str(path), text, hashlib.md5(Path(path).read_bytes()).hexdigest()) for row, path, text in items]
    todo = sorted({(p, m) for _, p, _, m in keyed if (p, m) not in cache})
    if todo:
        # model + processor loaded directly in fp16: the transformers-5 ASR pipeline silently kept large-v3 in fp32
        # (6.2 GB), which both OOMed beside training and made one-clip-at-a-time scoring crawl
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        proc = WhisperProcessor.from_pretrained(model)
        asr = WhisperForConditionalGeneration.from_pretrained(model, dtype=torch.float16).to(dev).eval()
        print(f"  asr {key}: {model} {next(asr.parameters()).dtype}, {len(todo)} clips", flush=True)
        with open(cache_f, "a") as cf:
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                f = proc([to16(load24(p)) for p, _ in chunk], sampling_rate=16000, return_tensors="pt").input_features
                with torch.no_grad():
                    out = asr.generate(f.to(dev, torch.float16), language="en", task="transcribe", max_new_tokens=200)
                for (p, m), h in zip(chunk, proc.batch_decode(out, skip_special_tokens=True)):
                    cache[(p, m)] = h.strip()
                    cf.write(json.dumps(dict(path=p, mtime=m, hyp=cache[(p, m)])) + "\n")
                if (i // batch) % 40 == 0:
                    print(f"  asr {key}: {min(i + batch, len(todo))}/{len(todo)}", flush=True)
        del asr; free()
    from transformers import WhisperTokenizer
    norm = WhisperTokenizer.from_pretrained("openai/whisper-large-v3").normalize
    for row, p, text, m in keyed:
        hyp = cache[(p, m)]
        row[f"err_{key}"], row["nwords"] = word_errors(norm, text, hyp)
        row[f"hyp_{key}"] = hyp


@torch.no_grad()
def score_utmos(items, dev="cuda"):
    m = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(dev).eval()
    for row, path, _ in items:
        row["utmos"] = float(m(torch.from_numpy(to16(load24(str(path))))[None].to(dev), 16000)[0])
    del m; free()


@torch.no_grad()
def score_sim(items, refs, dev="cuda"):
    """row["sim_voice"] = cosine(WavLM-SV(synth), WavLM-SV(real clip of the voice used)), for rows with a 'voice_ref'."""
    from transformers import AutoFeatureExtractor, WavLMForXVector
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()

    def emb(y):
        x = fe(to16(y), sampling_rate=16000, return_tensors="pt").to(dev)
        return torch.nn.functional.normalize(m(**x).embeddings[0], dim=-1)
    ref_emb = {v: emb(load24(str(p))) for v, p in refs.items()}
    for row, path, _ in items:
        if row.get("sim_target"):  # cloning: similarity to the target speaker's real recording of this text
            row["sim_o"] = float(emb(load24(str(path))) @ emb(load24(str(row["sim_target"]))))
        elif row.get("voice") in ref_emb and row.get("synthetic"):
            row["sim_voice"] = float(emb(load24(str(path))) @ ref_emb[row["voice"]])
    del m; free()


def voices(stats):
    """8 LibriTTS-R training voices (4 lowest-, 4 highest-F0 speakers with enough data) + EARS pools per gender."""
    ctl = pd.read_parquet(META / "controls.parquet")
    spk = ctl.groupby("speaker").agg(f0=("spk_f0_st", "first"), n=("id", "size"), corpus=("corpus", "first"))
    spk = spk[(spk.n >= 40) & spk.index.isin(stats["speakers"])]
    lib = spk[spk.corpus == "libritts"].sort_values("f0")
    ears = spk[spk.corpus == "ears"].sort_values("f0")
    pools = dict(male=list(lib.index[:4]), female=list(lib.index[-4:]))
    psc = dict(male=list(lib.index[:3]) + list(ears.index[:3]), female=list(lib.index[-3:]) + list(ears.index[-3:]))
    refs = {}
    for v in set(sum(pools.values(), []) + sum(psc.values(), [])):
        c = ctl[ctl.speaker == v].sort_values("dur")
        i = c.iloc[len(c) // 2].id
        refs[v] = ROOT / "data" / "wav24" / i.split("_")[0] / f"{i}.wav"
    return pools, psc, refs


def gender_of_hz(hz, calib):
    return "female" if 12 * np.log2(hz / 100.0) > calib["gender_cut_st"] else "male"


def synthesize(name, ck, sets, pools, psc_pools, calib):
    """Render (or copy, for ground truth) every test item; returns the per-set row lists, wavs on disk.
    ck: SteerTTS checkpoint, None = ground truth, or "wavs:<dir>" = audio pre-rendered by another system
    (steer/baselines.py) at <dir>/<set>/<i:04d>.wav - it is read, never re-synthesized, and scored identically."""
    ext = Path(ck[5:]) if ck and ck.startswith("wavs:") else None
    if ext is None:
        from .synth import Synth
        from .prompt import PromptMapper
    s, pm = (Synth(ck), PromptMapper()) if ck and ext is None else (None, None)
    res = dict(name=name, ckpt=ck, **{k: [] for k in sets})
    for set_name, df in sets.items():
        d = OUT / name / set_name; d.mkdir(parents=True, exist_ok=True)
        for i, r in df.iterrows():
            row = dict(i=int(i), synthetic=s is not None)
            if ext is not None:
                f = ext / set_name / f"{i:04d}.wav"
                if not f.exists():
                    continue
                y = load24(str(f))
                row.update(external=True)
                if set_name == "psc":
                    row.update(source=r.source, leak=bool(r.leak))
                    hz, dur = f0_and_speech_dur(y)
                    st = 12 * np.log2(hz / 100.0) if hz == hz else float("nan")
                    row.update(gender_ok=(gender_of_hz(hz, calib) if hz == hz else "?") == r.gender,
                               pitch_ok=(PITCH[bins(st, calib[f"pitch_cuts_{r.gender}"])] if hz == hz else "?") == r.pitch,
                               rate_ok=RATE[bins(n_phones(r.phonemes) / max(dur, 0.3), calib["rate_cuts"])] == r.speaking_rate)
            elif set_name == "clone":
                if s is None or s.spk_ref is None:
                    continue  # only reference-conditioned checkpoints can clone
                y = np.asarray(s(r.text, None, ref_wav=r.ref_path), np.float32)
                row.update(id=r.id, voice=r.speaker, sim_target=r.gt_path)
            elif set_name == "ljspeech":
                if s is None:
                    continue  # no LJSpeech audio locally; published vocoded-GT WER is 1.97
                voice = pools["female"][i % 4]
                y = np.asarray(s(r.text, voice), np.float32)
                row.update(voice=voice)
            elif set_name == "libritts":
                g = gender_of_hz(r.spk_f0, calib)
                voice = pools[g][i % 4]
                y = load24(r.gt_path) if s is None else np.asarray(s(r.text, voice), np.float32)
                row.update(id=r.id, voice=voice if s else r.speaker, gender=g)
            else:
                voice = psc_pools[r.gender][i % len(psc_pools[r.gender])]
                if s is None:
                    if not isinstance(r.gt_path, str):  # no local audio (VoxCeleb, missing Expresso shards)
                        continue
                    y = load24(r.gt_path)
                else:
                    m = pm(r.description)
                    tag = m["top_tags"][0][0]
                    y = np.asarray(s(r.text, voice, tag=tag if tag in s.tag2i else "none", ctrl=m["ctrl"] or None),
                                   np.float32)
                    row.update(tag=tag, ctrl=m["ctrl"])
                row.update(source=r.source, leak=bool(r.leak), voice=voice if s else r.speakerid)
                hz, dur = f0_and_speech_dur(y)
                st = 12 * np.log2(hz / 100.0) if hz == hz else float("nan")
                g_hat = gender_of_hz(hz, calib) if hz == hz else "?"
                p_hat = PITCH[bins(st, calib[f"pitch_cuts_{r.gender}"])] if hz == hz else "?"
                r_hat = RATE[bins(n_phones(r.phonemes) / max(dur, 0.3), calib["rate_cuts"])]
                row.update(gender_ok=g_hat == r.gender, pitch_ok=p_hat == r.pitch, rate_ok=r_hat == r.speaking_rate)
            row["wav"] = str(d / f"{i:04d}.wav")
            sf.write(row["wav"], y, SR)
            res[set_name].append(row)
        print(name, set_name, "synthesized", len(res[set_name]), flush=True)
    del s; free()
    return res


SETS = ("libritts", "psc", "ljspeech", "clone")


def run(ckpts, with_gt):
    calib = json.load(open(BENCH / "calibration.json"))
    sets = {k: pd.read_json(BENCH / f"{k}.jsonl", lines=True) for k in SETS}
    # the 287-speaker stats the benchmarked checkpoints use (META is rebuilt when the corpus grows)
    bk = META / "backup_287spk" / "control_stats.json"
    stats = json.load(open(bk if bk.exists() else META / "control_stats.json"))
    pools, psc_pools, refs = voices(stats)
    systems = dict(ckpts)
    if with_gt:
        systems = {**systems, "ground_truth": None}
    results = [synthesize(n, ck, sets, pools, psc_pools, calib) for n, ck in systems.items()]
    text = {k: v.text.tolist() for k, v in sets.items()}
    items = lambda names: [(row, row["wav"], text[sn][row["i"]]) for r in results for sn in names for row in r[sn]]
    score_asr(items(SETS), "openai/whisper-large-v3", "v3"); print("asr v3 done", flush=True)
    score_asr(items(["psc"]), "distil-whisper/distil-large-v2", "dl2"); print("asr distil done", flush=True)
    score_asr(items(["ljspeech"]), "openai/whisper-medium", "med"); print("asr medium done", flush=True)
    score_utmos(items(SETS)); print("utmos done", flush=True)
    score_sim(items(["libritts", "clone"]), refs); print("sim done", flush=True)
    for res in results:
        res["summary"] = summarize(res)
        json.dump(res, open(OUT / f"{res['name']}.json", "w"), indent=1, default=float)
    report()


def summarize(res):
    out = {}
    for set_name in SETS:
        df = pd.DataFrame(res.get(set_name, []))
        if df.empty:
            continue
        groups = {"all": df}
        if set_name == "psc":
            groups.update({src: df[df.source == src] for src in ("voxceleb", "expresso", "ears")})
            groups["no_leak"] = df[~df.leak]
        for gname, g in groups.items():
            if g.empty:
                continue
            m = dict(n=len(g), utmos=g.utmos.mean())
            for key in ("v3", "dl2", "med"):
                if f"err_{key}" in g:
                    m[f"wer_{key}"] = 100 * g[f"err_{key}"].sum() / g.nwords.sum()
            for k in ("sim_voice", "sim_o", "gender_ok", "pitch_ok", "rate_ok"):
                if k in g:
                    m[k] = float(g[k].mean())
            out[f"{set_name}/{gname}"] = {k: round(float(v), 4) for k, v in m.items()}
    return out


def report():
    rows = {}
    for f in sorted(OUT.glob("*.json")):
        r = json.load(open(f))
        rows[r["name"]] = r["summary"]
    calib = json.load(open(BENCH / "calibration.json"))
    lines = ["# SteerTTS external benchmark", "",
             "LibriTTS-R test-clean (500 utts, 4-10 s): WER whisper-large-v3 (%), UTMOS22, SIM to the voice used "
             "(WavLM-base-plus-sv; voice consistency, not cloning).", "",
             "| system | WER v3 | UTMOS | SIM voice |", "|---|---|---|---|"]
    for n, s in rows.items():
        m = s.get("libritts/all", {})
        lines.append(f"| {n} | {m.get('wer_v3', float('nan')):.2f} | {m.get('utmos', float('nan')):.2f} | "
                     f"{m.get('sim_voice', float('nan')):.3f} |")
    lines += ["", "LJSpeech test (Tacotron2/Matcha split, 500 sentences): WER whisper-medium (the Matcha paper's ASR) and "
              "whisper-large-v3 (%), UTMOS22.", "", "| system | WER medium | WER v3 | UTMOS |", "|---|---|---|---|"]
    for n, s in rows.items():
        m = s.get("ljspeech/all")
        if m:
            lines.append(f"| {n} | {m.get('wer_med', float('nan')):.2f} | {m.get('wer_v3', float('nan')):.2f} | "
                         f"{m['utmos']:.2f} |")
    lines += ["", "Zero-shot cloning, LibriTTS-R test-clean (unseen speakers; prompt = another utterance of the same "
              "speaker): WER whisper-large-v3 (%), SIM-o (WavLM-base-plus-sv cosine to the real recording).", "",
              "| system | WER v3 | SIM-o | UTMOS |", "|---|---|---|---|"]
    for n, s in rows.items():
        m = s.get("clone/all")
        if m:
            lines.append(f"| {n} | {m.get('wer_v3', float('nan')):.2f} | {m.get('sim_o', float('nan')):.3f} | {m['utmos']:.2f} |")
    lines += ["", "Literature SIM-o uses WavLM-large-SV (seed-tts-eval), not base-plus-sv: F5-TTS 0.66, GT 0.69 on "
              "LibriSpeech-PC - compare trends, not absolutes, until the large-SV scorer is added.", ""]
    lines += ["", "Literature (LJSpeech test, Whisper-medium WER, single-speaker LJ models): vocoded GT 1.97, "
              "FastSpeech2 (41M) 4.18, VITS (36M) 2.52, Matcha-TTS (18M) 2.09.", ""]
    lines += ["", "Literature (LibriSpeech test-clean UTMOS): GT 4.08, CosyVoice2 4.23, Spark-TTS 4.35. "
              "LJSpeech WER (Whisper-medium): FastSpeech2 4.18, VITS 2.52, Matcha 2.09.", "",
              "ParaSpeechCaps test (246 prompts): WER distil-whisper-large-v2 (paper's ASR, %) and whisper-large-v3; "
              "attribute accuracy for gender / pitch level / speaking rate (thresholds fitted on real EARS+Expresso "
              f"audio; accuracy of the same estimator on real audio: gender {calib['gender_acc']:.2f}, pitch "
              f"F {calib['pitch_acc_female']:.2f} / M {calib['pitch_acc_male']:.2f}, rate {calib['rate_acc']:.2f}).", "",
              "| system | subset | n | WER dl2 | WER v3 | UTMOS | gender | pitch | rate |",
              "|---|---|---|---|---|---|---|---|---|"]
    for n, s in rows.items():
        for k, m in s.items():
            if k.startswith("psc/"):
                f = lambda key, p=2: f"{m[key]:.{p}f}" if key in m else "-"
                lines.append(f"| {n} | {k[4:]} | {m['n']} | {f('wer_dl2')} | {f('wer_v3')} | {f('utmos')} | "
                             f"{f('gender_ok')} | {f('pitch_ok')} | {f('rate_ok')} |")
    lines += ["", "Literature (ParaSpeechCaps test, distil-whisper-large-v2 WER): Parler-TTS 4.62, "
              "Parler FT on LibriTTS-P/Expresso/EARS 7.14, ParaSpeechCaps-Scaled 8.63, ground truth 7.93.", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    from .tb_report import write_all
    write_all()  # the same tables in TensorBoard (runs/<run>/tb/results + runs/_results/tb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "prep_lj", "prep_clone", "run", "report"])
    ap.add_argument("--ckpt", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--gt", action="store_true", help="also score the real recordings (reference row)")
    ap.add_argument("--wavs", action="append", default=[], help="name=dir of audio pre-rendered by steer.baselines")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "prep_lj":
        prep_lj()
    elif a.cmd == "prep_clone":
        prep_clone()
    elif a.cmd == "run":
        run([c.split("=", 1) for c in a.ckpt] + [(w.split("=", 1)[0], "wavs:" + w.split("=", 1)[1]) for w in a.wavs], a.gt)
    else:
        report()


if __name__ == "__main__":
    main()
