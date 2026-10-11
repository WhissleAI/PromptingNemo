"""Paper: the non-prosodic steering channels on HELD-OUT voices (LibriTTS-R test-clean, steer.paper.common).

  events   inline sound-event tokens ([laugh] ... [dog] ...) at 3 positions (start / mid / end) in 3 sentences.
           model path = the token is part of the TTS input (what the checkpoint itself renders);
           mixer path = steer.markup + steer.mixer.EnvMixer (real recordings spliced in; positive control for the
           detector). Detection: AudioSet AST (MIT/ast-finetuned-audioset-10-10-0.4593), the token's classes must
           score > 0.2 and rise > 0.1 over the same voice + sentence without the token (steer.tag_eval criterion).
           Also: frames the duration predictor gives the event token, and Whisper WER of the surrounding words.
           Training voices that recorded the nonverbals (EARS, anonymised T01..) are a within-training control.
  env      (i) the checkpoint's environment vector (scene at 10 dB SNR), (ii) EnvMixer post-mixing (achieved SNR,
           AST scene detection), (iii) cloning from NOISY reference clips (noise at 20/10/5/0 dB): speaker similarity
           to a different clean clip of the speaker, AudioSet noise-class leakage, WER.
  markup   *emph* and <slow|fast|high|low|loud|soft> on one word (2 positions x 6 sentences): per-word achieved
           pitch change (Praat F0, semitones), level change (RMS dB) and duration ratio vs the unmarked render,
           and the same on the neighbouring words (leak). Word spans come from the model's own token durations.

CPU only (CUDA_VISIBLE_DEVICES=""), except the optional Whisper stage which can also run via the GPU queue.
python -m steer.paper.events_eval --stage render|ast|measure|asr|analyze|all
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from ..config import SR, HOP
from . import common as C

OUT = C.WORK / "events"
WAV = OUT / "wav"
ROOT = C.ROOT
TEXTS = ["The meeting has been moved to Thursday afternoon at three.",
         "Please remember to bring the blue folder when you come in.",
         "She said the package should arrive sometime early next week."]
MK_TEXTS = TEXTS + ["We will need two more chairs for the people arriving later.",
                    "Turn left at the second light and park behind the building.",
                    "I put the keys on the table next to the kitchen window."]
MARKUPS = ["emph", "slow", "fast", "high", "low", "loud", "soft"]
POS = ["start", "mid", "end"]
N_HELD = 8
ASR_NAME = "whisper-large-v3 fp16 + whisper EnglishTextNormalizer"
NOISE_SNRS = [20, 10, 5, 0]
NOISE_KINDS = ["crowd", "traffic", "static"]


def held_voices():
    V = C.heldout_voices(16)
    return V[:4] + V[8:12]


def train_voices(n=4):
    """EARS training speakers that recorded nonverbals (in-distribution control), anonymised T01.."""
    stats = json.load(open(Path(C.BASE).parent / "control_stats.json"))
    have = set()
    for l in open(ROOT / "data/meta/noise_lib.jsonl"):
        r = json.loads(l)
        if r["kind"] == "nv":
            have.add(r["speaker"])
    ears = sorted(s for s in stats["speakers"] if s in have)
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(ears, n, replace=False).tolist())
    return [dict(vid=f"T{i + 1:02d}", speaker=s) for i, s in enumerate(pick)]


def with_token(text, tok, pos):
    w = text.split()
    if pos == "start":
        return f"[{tok}] {text}"
    if pos == "end":
        return f"{text} [{tok}]"
    k = len(w) // 2
    return " ".join(w[:k] + [f"[{tok}]"] + w[k:])


# ------------------------------------------------------------------------------------------------ plan
def plan():
    from ..text import EVENTS, NONVERBAL
    from ..acoustic import SCENES
    H, T = held_voices(), train_voices()
    items = []
    toks = NONVERBAL + EVENTS
    for v in H + T:
        for ti, t in enumerate(TEXTS):
            items.append(dict(kind="ev_base", vid=v["vid"], text_id=ti, text=t, spoken=t))
            for tok in toks:
                for p in POS:
                    items.append(dict(kind="ev_model", vid=v["vid"], text_id=ti, text=t, tok=tok, pos=p,
                                      spoken=with_token(t, tok, p)))
                if v["vid"].startswith("H"):
                    items.append(dict(kind="ev_mixer", vid=v["vid"], text_id=ti, text=t, tok=tok, pos="end",
                                      spoken=with_token(t, tok, "end")))
    for v in H:
        for ti, t in enumerate(TEXTS[:2]):
            for sc in SCENES:
                items.append(dict(kind="env_model", vid=v["vid"], text_id=ti, text=t, scene=sc, snr=10, spoken=t))
                items.append(dict(kind="env_mixer", vid=v["vid"], text_id=ti, text=t, scene=sc, snr=10, spoken=t))
        for ti, t in enumerate(TEXTS):
            for nk in NOISE_KINDS:
                for snr in NOISE_SNRS:
                    items.append(dict(kind="noisy_ref", vid=v["vid"], text_id=ti, text=t, noise=nk, snr=snr, spoken=t))
        for ti, t in enumerate(MK_TEXTS):
            items.append(dict(kind="mk_base", vid=v["vid"], text_id=ti, text=t, spoken=t))
            w = t.split()
            for wpos in (2, len(w) - 3):
                for m in MARKUPS:
                    ww = list(w)
                    core = ww[wpos].rstrip(".,")
                    ww[wpos] = (f"*{core}*" if m == "emph" else f"<{m}>{core}</{m}>") + ww[wpos][len(core):]
                    items.append(dict(kind="mk", vid=v["vid"], text_id=ti, text=t, markup=m, word=wpos,
                                      spoken=" ".join(ww)))
    for i, it in enumerate(items):
        tag = "_".join(str(it[k]) for k in ("kind", "vid", "text_id", "tok", "pos", "scene", "noise", "snr", "markup", "word")
                       if k in it)
        it["path"] = str(WAV / f"{tag}.wav")
    return items, H, T


# ------------------------------------------------------------------------------------------------ render
_W = {}


def _init(vecs):
    torch.set_num_threads(2)
    from ..mixer import EnvMixer
    from ..markup import MarkupSynth
    s = C.make_synth(device="cpu")
    _W.update(s=s, vecs=vecs, mx=EnvMixer())
    _W["ms"] = MarkupSynth(s, None, _W["mx"])
    _W["ms0"] = MarkupSynth(s, None, None)


@torch.no_grad()
def render_core(s, text, vec, spk_name=None, word_shift=None, env=None):
    """Synth.__call__ without flow/guidance, returning token durations (frames) and word indices too."""
    from ..data import PITCH_SCALE
    from ..text import to_phonemes
    ids, widx, _ = to_phonemes(text)
    ids_t = torch.tensor([ids]); widx_t = torch.tensor([widx])
    c, m = s.control_tensors(None)
    spk = torch.tensor([s.spk2i.get(spk_name, len(s.spk2i))])
    tg = torch.tensor([0])
    if word_shift:
        word_shift = {w: dict(pitch=x.get("pitch", 0) / PITCH_SCALE, energy=x.get("energy", 0) / 10.0,
                              dur=x.get("dur", 1.0)) for w, x in word_shift.items()}
    kw = {}
    if env:
        from ..acoustic import env_vector
        kw["env"] = torch.from_numpy(env_vector(**env))[None]
    mel, aux = s.model.infer(ids_t, spk, tg, c, m, word_idx=widx_t, word_shift=word_shift,
                             spk_vec=torch.from_numpy(vec)[None], **kw)
    wav = s.vocos.decode(mel.float()).squeeze().numpy()
    return wav, aux["dur"][0].numpy().tolist(), ids, widx


def _render(it):
    s, vecs = _W["s"], _W["vecs"]
    p = Path(it["path"])
    meta = dict(path=it["path"])
    if p.exists() and p.with_suffix(".json").exists():
        return json.load(open(p.with_suffix(".json")))
    vkey = it["vid"] if it["kind"] != "noisy_ref" else f"{it['vid']}_{it['noise']}_{it['snr']}"
    vec = vecs[vkey]
    spk_name = vecs.get(f"name_{it['vid']}")
    k = it["kind"]
    if k in ("ev_base", "ev_model", "noisy_ref", "mk_base"):
        y, dur, ids, widx = render_core(s, it["spoken"], vec, spk_name)
        meta.update(dur=dur, ids=ids, widx=widx)
    elif k == "env_model":
        y, dur, ids, widx = render_core(s, it["spoken"], vec, spk_name, env=dict(scene=it["scene"], snr=it["snr"]))
    elif k == "mk":
        segs = _W["ms0"].parse(it["spoken"])
        assert len(segs) == 1, (it["spoken"], segs)
        y, dur, ids, widx = render_core(s, segs[0].text, vec, spk_name, word_shift=segs[0].word_shift)
        meta.update(dur=dur, ids=ids, widx=widx, word_shift={str(a): b for a, b in segs[0].word_shift.items()},
                    seg_text=segs[0].text)
    elif k in ("ev_mixer", "env_mixer"):
        # EnvMixer path: speaker vector via the reference clip (CachedSynth), as the demo / markup API does
        ms, ref = _W["ms"], vecs[f"ref_{it['vid']}"]
        env = dict(scene=it["scene"], snr=it["snr"]) if k == "env_mixer" else None
        chunks = list(ms.stream(it["spoken"], voice=None, ref_wav=ref, env=env))
        y = np.concatenate([c.audio for c in chunks])
        lens = [len(c.audio) for c in chunks]
        meta.update(chunk_lens=lens, chunk_kind=["event" if c.segment.event else "speech" for c in chunks])
        if k == "env_mixer":  # the clean speech under the same mixer state, for the achieved SNR
            clean = np.concatenate([c.audio for c in _W["ms0"].stream(it["spoken"], voice=None, ref_wav=ref)])
            sf.write(str(p.with_name(p.stem + "_clean.wav")), clean, SR)
    else:
        raise ValueError(k)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), np.asarray(y, np.float32), SR)
    json.dump(meta, open(p.with_suffix(".json"), "w"))
    return meta


def make_noisy_refs(H):
    """Reference clips with background noise mixed in at fixed SNRs (active-speech RMS / noise RMS)."""
    from ..acoustic import active_rms, rms
    lib = {}
    for l in open(ROOT / "data/meta/noise_lib.jsonl"):
        r = json.loads(l)
        if r["kind"] == "bg" and r["label"] in NOISE_KINDS and r.get("dur", 0) >= 8:
            lib.setdefault(r["label"], []).append(r["path"])
    rng = np.random.default_rng(0)
    out = {}
    for v in H:
        y = C.load24(v["ref"])
        for nk in NOISE_KINDS:
            nz = sf.read(lib[nk][int(rng.integers(len(lib[nk])))], dtype="float32")[0]
            nz = np.tile(nz, int(np.ceil(len(y) / len(nz))))[: len(y)]
            for snr in NOISE_SNRS:
                yy = y + nz * (active_rms(y) * 10 ** (-snr / 20) / (rms(nz) + 1e-9))
                yy = yy * min(1.0, 0.99 / np.abs(yy).max())
                p = OUT / "noisy_ref" / f"{v['vid']}_{nk}_{snr}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(p), yy.astype(np.float32), SR)
                out[f"{v['vid']}_{nk}_{snr}"] = str(p)
    return out


def stage_render(items, H, T, procs=20):
    import multiprocessing as mp
    s = C.make_synth(device="cpu")
    vecs = {}
    for v in H:
        vecs[v["vid"]] = s.speaker_vector(ref_wav=v["ref"])[0].numpy()
        vecs[f"ref_{v['vid']}"] = v["ref"]
    for v in T:
        vecs[v["vid"]] = np.asarray(s.spk_ref[v["speaker"]], np.float32)
        vecs[f"name_{v['vid']}"] = v["speaker"]
    for k, p in make_noisy_refs(H).items():
        vecs[k] = s.speaker_vector(ref_wav=p)[0].numpy()
    del s
    todo = [it for it in items if not (Path(it["path"]).exists() and Path(it["path"]).with_suffix(".json").exists())]
    print(f"render: {len(todo)}/{len(items)} to do", flush=True)
    os.environ["OMP_NUM_THREADS"] = "2"
    with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=(vecs,)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_render, todo, chunksize=4)):
            if i % 200 == 0:
                print(f"  render {i}/{len(todo)}", flush=True)


# ------------------------------------------------------------------------------------------------ AST
class AST:
    def __init__(self, dev="cpu"):
        import re
        from transformers import ASTForAudioClassification
        from ..acoustic import AUDIOSET
        from ..rewards import AUDIOSET_NV
        self.m = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).eval()
        rules = [(re.compile(p), v) for p, v in AUDIOSET] + [(re.compile(p), ("event", t)) for p, t in AUDIOSET_NV]
        self.cls = {}
        for i, lab in self.m.config.id2label.items():
            for rx, v in rules:
                if rx.match(lab):
                    self.cls.setdefault(v, []).append(int(i))
        self.noise_ids = sorted({i for v in self.cls.values() for i in v})
        self.labels = self.m.config.id2label
        self.dev = dev

    @torch.no_grad()
    def __call__(self, ys):
        import soxr
        import torchaudio.compliance.kaldi as K
        fbs = []
        for y in ys:
            x = torch.from_numpy(soxr.resample(np.asarray(y, np.float32), SR, 16000))
            fb = K.fbank(x[None], htk_compat=True, sample_frequency=16000, use_energy=False, window_type="hanning",
                         num_mel_bins=128, dither=0.0, frame_shift=10)
            fb = torch.nn.functional.pad(fb, (0, 0, 0, max(0, 1024 - fb.shape[0])))[:1024]
            fbs.append((fb - (-4.2677393)) / (4.5689974 * 2))
        return torch.sigmoid(self.m(input_values=torch.stack(fbs).to(self.dev)).logits).cpu().numpy()


def stage_ast(items, extra=(), dev="cpu"):
    torch.set_num_threads(int(os.environ.get("AST_THREADS", "40")))
    cache_f = OUT / "ast.npz"
    cache = dict(np.load(cache_f, allow_pickle=True)["d"].item()) if cache_f.exists() else {}
    paths = [it["path"] for it in items if it["kind"] not in ("mk", "mk_base")] + list(extra)
    todo = [p for p in paths if p not in cache]
    print(f"ast: {len(todo)} clips", flush=True)
    A = AST(dev)
    for i in range(0, len(todo), 16):
        ch = todo[i:i + 16]
        P = A([sf.read(p, dtype="float32")[0] for p in ch])
        for p, q in zip(ch, P):
            cache[p] = q.astype(np.float16)
        if (i // 16) % 20 == 0:
            print(f"  ast {i + len(ch)}/{len(todo)}", flush=True)
            np.savez(cache_f, d=np.array(cache, dtype=object))
    np.savez(cache_f, d=np.array(cache, dtype=object))
    return cache, A


# ------------------------------------------------------------------------------------------------ per-word measures
def word_spans(dur, widx):
    """word index -> (first frame, end frame) from the model's token durations."""
    cum = np.concatenate([[0], np.cumsum(dur)])
    sp = {}
    for k, w in enumerate(widx):
        if w < 0:
            continue
        a, b = cum[k], cum[k + 1]
        lo, hi = sp.get(w, (a, b))
        sp[w] = (min(lo, a), max(hi, b))
    return sp


def _word_feats(path):
    import parselmouth
    meta = json.load(open(Path(path).with_suffix(".json")))
    y, _ = sf.read(path, dtype="float32")
    snd = parselmouth.Sound(y.astype(np.float64), SR)
    pt = snd.to_pitch_ac(time_step=0.005, pitch_floor=75, pitch_ceiling=600)
    f0 = pt.selected_array["frequency"]; tt = pt.xs()
    out = {}
    for w, (a, b) in word_spans(meta["dur"], meta["widx"]).items():
        t0, t1 = a * HOP / SR, b * HOP / SR
        seg = y[int(a * HOP): int(b * HOP)]
        sel = (tt >= t0) & (tt < t1) & (f0 > 0)
        out[w] = dict(dur_s=t1 - t0, rms_db=20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else np.nan,
                      f0_st=12 * np.log2(np.median(f0[sel]) / 100.0) if sel.sum() >= 3 else np.nan)
    return path, out


def stage_measure(items, procs=40):
    import multiprocessing as mp
    paths = [it["path"] for it in items if it["kind"] in ("mk", "mk_base")]
    with mp.get_context("spawn").Pool(procs) as pool:
        res = dict(pool.map(_word_feats, paths, chunksize=8))
    json.dump({p: {str(w): v for w, v in d.items()} for p, d in res.items()}, open(OUT / "word_feats.json", "w"), default=float)


# ------------------------------------------------------------------------------------------------ ASR + SIM
def stage_asr(items, dev):
    paths = [it["path"] for it in items if it["kind"] in ("ev_base", "ev_model", "noisy_ref", "mk", "mk_base",
                                                          "ev_mixer")]
    if dev == "cpu":
        hyps = transcribe_cpu(paths, OUT / "asr_cache.jsonl")
    else:
        hyps = C.transcribe(paths, OUT / "asr_cache.jsonl", dev=dev)
    json.dump(hyps, open(OUT / "hyps.json", "w"))


def transcribe_cpu(paths, cache_file, model="openai/whisper-large-v3-turbo", batch=8):
    import hashlib
    import soxr
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    torch.set_num_threads(int(os.environ.get("ASR_THREADS", "40")))
    cache = {}
    if Path(cache_file).exists():
        for l in open(cache_file):
            d = json.loads(l); cache[d["md5"]] = d["hyp"]
    md5 = {p: hashlib.md5(Path(p).read_bytes()).hexdigest() for p in paths}
    todo = sorted({p for p, m in md5.items() if m not in cache})
    print(f"asr cpu {model}: {len(todo)} clips", flush=True)
    if todo:
        proc = WhisperProcessor.from_pretrained(model)
        asr = WhisperForConditionalGeneration.from_pretrained(model, dtype=torch.float32).eval()
        with open(cache_file, "a") as cf:
            for i in range(0, len(todo), batch):
                ch = todo[i:i + batch]
                ys = [soxr.resample(sf.read(p, dtype="float32")[0], SR, 16000) for p in ch]
                f = proc(ys, sampling_rate=16000, return_tensors="pt").input_features
                with torch.no_grad():
                    out = asr.generate(f, language="en", task="transcribe", max_new_tokens=120)
                for p, h in zip(ch, proc.batch_decode(out, skip_special_tokens=True)):
                    cache[md5[p]] = h.strip()
                    cf.write(json.dumps(dict(md5=md5[p], path=p, hyp=h.strip(), model=model)) + "\n")
                if (i // batch) % 25 == 0:
                    print(f"  asr {i + len(ch)}/{len(todo)}", flush=True)
    return {p: cache[m] for p, m in md5.items()}


def stage_sim(items, H, dev="cpu"):
    """Speaker similarity (WavLM-base-plus-sv; NB the same encoder family as the conditioning x-vector) of each
    noisy-reference render and of the clean-reference event baselines to a DIFFERENT clean clip of the speaker."""
    from transformers import AutoFeatureExtractor, WavLMForXVector
    import soxr
    torch.set_num_threads(40)
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()

    @torch.no_grad()
    def emb(y):
        x = fe([soxr.resample(np.asarray(y, np.float32), SR, 16000)], sampling_rate=16000, return_tensors="pt")
        e = m(**{k: v.to(dev) for k, v in x.items()}).embeddings[0].cpu()
        return (e / e.norm()).numpy()

    tgt = {}
    for v in H:
        tgt[v["vid"]] = np.mean([emb(C.load24(p)) for p, _ in v["others"][:3]], 0)
    out = {}
    for it in items:
        if it["kind"] in ("noisy_ref",) or (it["kind"] == "ev_base" and it["vid"].startswith("H")):
            e = emb(sf.read(it["path"], dtype="float32")[0])
            t = tgt[it["vid"]]
            out[it["path"]] = float(e @ t / np.linalg.norm(t))
    json.dump(out, open(OUT / "sim.json", "w"))


# ------------------------------------------------------------------------------------------------ analysis
def analyze(items, H, T):
    from ..text import NONVERBAL, EVENTS, SYM2ID
    from ..acoustic import active_rms, rms
    cache = dict(np.load(OUT / "ast.npz", allow_pickle=True)["d"].item())
    from transformers import AutoConfig
    id2label = AutoConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").id2label
    import re
    from ..acoustic import AUDIOSET
    from ..rewards import AUDIOSET_NV
    rules = [(re.compile(p), v) for p, v in AUDIOSET] + [(re.compile(p), ("event", t)) for p, t in AUDIOSET_NV]
    cls = {}
    for i, lab in id2label.items():
        for rx, v in rules:
            if rx.match(lab):
                cls.setdefault(v, []).append(int(i))
    noise_ids = sorted({i for v in cls.values() for i in v})
    speech_ids = [i for i, l in id2label.items() if l in ("Speech", "Male speech, man speaking", "Female speech, woman speaking")]
    hyps = json.load(open(OUT / "hyps.json")) if (OUT / "hyps.json").exists() else {}
    sim = json.load(open(OUT / "sim.json")) if (OUT / "sim.json").exists() else {}
    df = pd.DataFrame(items)
    df["grp"] = np.where(df.vid.str.startswith("H"), "heldout", "train_ears")

    def tokp(p, tok):
        c = cls.get(("event", tok))
        return float(cache[p][c].astype(np.float32).max()) if c else np.nan

    R = []
    src_ev = f"{C.RES}/events_raw.csv"
    split = "libritts_r_test_clean_heldout8"
    # ---- events
    base = {(r.vid, r.text_id): r.path for r in df[df.kind == "ev_base"].itertuples()}
    ev = df[df.kind.isin(["ev_model", "ev_mixer"])].copy()
    rows = []
    for r in ev.itertuples():
        p0 = base[(r.vid, r.text_id)]
        pa, pb = tokp(r.path, r.tok), tokp(p0, r.tok)
        d = dict(kind=r.kind, grp=r.grp, vid=r.vid, text_id=r.text_id, tok=r.tok, pos=r.pos,
                 nonverbal=r.tok in NONVERBAL, has_cls=("event", r.tok) in cls, p_tok=pa, p_tok_base=pb,
                 hit=bool(pa > 0.2 and pa - pb > 0.1) if ("event", r.tok) in cls else np.nan,
                 noise_max=float(cache[r.path][noise_ids].astype(np.float32).max()))
        meta = json.load(open(Path(r.path).with_suffix(".json")))
        if r.kind == "ev_model":
            k = meta["ids"].index(SYM2ID[f"[{r.tok}]"])
            d["tok_frames"] = meta["dur"][k]
            d["tok_s"] = meta["dur"][k] * HOP / SR
            d["len_s"] = sum(meta["dur"]) * HOP / SR
            m0 = json.load(open(Path(p0).with_suffix(".json")))
            d["len_base_s"] = sum(m0["dur"]) * HOP / SR
        else:
            d["tok_s"] = sum(l for l, k in zip(meta["chunk_lens"], meta["chunk_kind"]) if k == "event") / SR
        if hyps:
            e, n = C.word_errors(r.text, hyps.get(r.path, ""))
            e0, _ = C.word_errors(r.text, hyps.get(p0, ""))
            d.update(err=e, nwords=n, err_base=e0)
        rows.append(d)
    E = pd.DataFrame(rows)
    U = ["vid", "text_id"]
    for kind in ["ev_model", "ev_mixer"]:
        for grp in ["heldout", "train_ears"]:
            sub = E[(E.kind == kind) & (E.grp == grp)]
            if not len(sub):
                continue
            cnd = dict(path=kind.replace("ev_", ""), voices=grp)
            dd = sub[sub.has_cls]
            R.append(C.row("event_detect_rate", C.boot_df(dd, U, lambda q: q.hit.astype(float).mean()), len(dd), src_ev,
                           "AST AudioSet: p>0.2 & rise>0.1", split, **cnd))
            for nv in [True, False]:
                q2 = dd[dd.nonverbal == nv]
                R.append(C.row("event_detect_rate", C.boot_df(q2, U, lambda q: q.hit.astype(float).mean()), len(q2),
                               src_ev, "AST AudioSet: p>0.2 & rise>0.1", split, subset="nonverbal" if nv else "environmental", **cnd))
            R.append(C.row("event_token_seconds", C.boot_df(sub, U, lambda q: q.tok_s.mean()), len(sub), src_ev,
                           "model duration predictor" if kind == "ev_model" else "spliced clip length", split, **cnd))
            if kind == "ev_model":
                R.append(C.row("event_len_increase_s", C.boot_df(sub, U, lambda q: (q.len_s - q.len_base_s).mean()), len(sub),
                               src_ev, "model duration predictor", split, **cnd))
                for p in POS:
                    q2 = dd[dd.pos == p]
                    R.append(C.row("event_detect_rate", C.boot_df(q2, U, lambda q: q.hit.astype(float).mean()), len(q2),
                                   src_ev, "AST AudioSet: p>0.2 & rise>0.1", split, position=p, **cnd))
            if hyps and "err" in sub:
                R.append(C.row("wer_surrounding", C.boot_df(sub, U, lambda q: q.err.sum() / q.nwords.sum()), len(sub), src_ev,
                               ASR_NAME, split, **cnd))
                R.append(C.row("wer_without_token", C.boot_df(sub, U, lambda q: q.err_base.sum() / q.nwords.sum()), len(sub),
                               src_ev, ASR_NAME, split, **cnd))
            for tok, g in sub.groupby("tok"):
                R.append(C.row("event_detect_rate", (float(g.hit.astype(float).mean()) if g.has_cls.all() else None, np.nan, np.nan),
                               len(g), src_ev, "AST AudioSet", split, tok=tok, **cnd))
                R.append(C.row("event_token_seconds", (float(g.tok_s.mean()), np.nan, np.nan), len(g), src_ev,
                               "duration", split, tok=tok, **cnd))
                R.append(C.row("p_tok_mean", (float(g.p_tok.mean()), np.nan, np.nan), len(g), src_ev, "AST", split, tok=tok, **cnd))
    clean_bases = [r.path for r in df[(df.kind == "ev_base")].itertuples()]
    R.append(C.row("clean_false_positive", (float(np.mean([cache[p][noise_ids].astype(np.float32).max() > 0.2 for p in clean_bases])),
                                             np.nan, np.nan), len(clean_bases), src_ev, "AST any noise/event class > 0.2", split))
    # ---- environment
    env_rows = []
    for r in df[df.kind.isin(["env_model", "env_mixer"])].itertuples():
        c = cls.get(("bg", r.scene))
        p0 = base[(r.vid, r.text_id)]
        d = dict(kind=r.kind, vid=r.vid, text_id=r.text_id, scene=r.scene, has_cls=c is not None)
        if c:
            pa, pb = float(cache[r.path][c].astype(np.float32).max()), float(cache[p0][c].astype(np.float32).max())
            d.update(p_scene=pa, p_scene_base=pb, hit=bool(pa > 0.2 and pa - pb > 0.1))
        y = sf.read(r.path, dtype="float32")[0]
        if r.kind == "env_mixer":
            cl = sf.read(r.path.replace(".wav", "_clean.wav"), dtype="float32")[0]
            n = min(len(y), len(cl))
            g = float(np.dot(y[:n], cl[:n]) / (np.dot(cl[:n], cl[:n]) + 1e-9))
            nz = y[:n] - g * cl[:n]
            d["snr_achieved"] = 20 * np.log10(active_rms(g * cl[:n]) / (rms(nz) + 1e-9))
        else:
            y0 = sf.read(p0, dtype="float32")[0]
            n = min(len(y), len(y0))
            d["diff_db_vs_clean"] = 20 * np.log10(rms(y[:n] - y0[:n]) / (rms(y0[:n]) + 1e-9))
            d["len_ratio"] = len(y) / len(y0)
        env_rows.append(d)
    EN = pd.DataFrame(env_rows)
    src_env = f"{C.RES}/events_env_raw.csv"
    for kind in ["env_model", "env_mixer"]:
        sub = EN[(EN.kind == kind) & EN.has_cls]
        cnd = dict(path=kind.replace("env_", ""))
        R.append(C.row("scene_detect_rate", C.boot_df(sub, U, lambda q: q.hit.astype(float).mean()), len(sub), src_env,
                       "AST AudioSet: p>0.2 & rise>0.1", split, snr_requested=10, **cnd))
        allk = EN[EN.kind == kind]
        if kind == "env_mixer":
            R.append(C.row("snr_achieved_db", C.boot_df(allk, U, lambda q: q.snr_achieved.mean()), len(allk), src_env,
                           "active-speech RMS / residual RMS", split, snr_requested=10, **cnd))
            R.append(C.row("snr_achieved_abs_err_db", C.boot_df(allk, U, lambda q: (q.snr_achieved - 10).abs().mean()), len(allk),
                           src_env, "active-speech RMS / residual RMS", split, snr_requested=10, **cnd))
        else:
            R.append(C.row("render_change_db_vs_clean", C.boot_df(allk, U, lambda q: q.diff_db_vs_clean.mean()), len(allk),
                           src_env, "RMS(render - clean render) / RMS(clean)", split, **cnd))
        for sc, g in sub.groupby("scene"):
            R.append(C.row("scene_detect_rate", (float(g.hit.mean()), np.nan, np.nan), len(g), src_env, "AST", split, scene=sc, **cnd))
    # noisy references
    nr = []
    for r in df[df.kind == "noisy_ref"].itertuples():
        p0 = base[(r.vid, r.text_id)]
        d = dict(vid=r.vid, text_id=r.text_id, noise=r.noise, snr=r.snr,
                 noise_max=float(cache[r.path][noise_ids].astype(np.float32).max()),
                 noise_max_clean=float(cache[p0][noise_ids].astype(np.float32).max()),
                 speech_p=float(cache[r.path][speech_ids].astype(np.float32).max()))
        if sim:
            d.update(sim=sim.get(r.path, np.nan), sim_clean=sim.get(p0, np.nan))
        if hyps:
            e, n = C.word_errors(r.text, hyps.get(r.path, "")); e0, _ = C.word_errors(r.text, hyps.get(p0, ""))
            d.update(err=e, nwords=n, err_clean=e0)
        nr.append(d)
    NR = pd.DataFrame(nr)
    src_nr = f"{C.RES}/events_noisyref_raw.csv"
    for snr in ["clean"] + NOISE_SNRS:
        sub = NR if snr == "clean" else NR[NR.snr == snr]
        cnd = dict(ref_snr=snr)
        if sim:
            col = "sim_clean" if snr == "clean" else "sim"
            R.append(C.row("sim_to_other_clip", C.boot_df(sub, U, lambda q: q[col].mean()), len(sub), src_nr,
                           "wavlm-base-plus-sv cosine", split, **cnd))
        if hyps:
            ec = "err_clean" if snr == "clean" else "err"
            R.append(C.row("wer", C.boot_df(sub, U, lambda q: q[ec].sum() / q.nwords.sum()), len(sub), src_nr,
                           ASR_NAME, split, **cnd))
        col = "noise_max_clean" if snr == "clean" else "noise_max"
        R.append(C.row("ast_noise_class_max", C.boot_df(sub, U, lambda q: q[col].mean()), len(sub), src_nr, "AST", split, **cnd))
        R.append(C.row("ast_noise_gt_0.2", C.boot_df(sub, U, lambda q: (q[col] > 0.2).mean()), len(sub), src_nr, "AST", split, **cnd))
    # ---- markup
    wf = json.load(open(OUT / "word_feats.json"))
    mb = {(r.vid, r.text_id): r.path for r in df[df.kind == "mk_base"].itertuples()}
    mrows = []
    for r in df[df.kind == "mk"].itertuples():
        a, b = wf[r.path], wf[mb[(r.vid, r.text_id)]]
        for w in a:
            off = int(w) - r.word
            if w not in b:
                continue
            role = "target" if off == 0 else ("neighbour" if abs(off) == 1 else "other")
            mrows.append(dict(vid=r.vid, text_id=r.text_id, markup=r.markup, word=r.word, off=off, role=role,
                              d_st=a[w]["f0_st"] - b[w]["f0_st"], d_db=a[w]["rms_db"] - b[w]["rms_db"],
                              dur_ratio=a[w]["dur_s"] / max(b[w]["dur_s"], 1e-3)))
    MK = pd.DataFrame(mrows)
    if hyps:
        er = {}
        for r in df[df.kind.isin(["mk", "mk_base"])].itertuples():
            er[r.path] = C.word_errors(r.text, hyps.get(r.path, ""))
    src_mk = f"{C.RES}/markup_raw.csv"
    from ..markup import PROSODY
    MR = []
    for m in MARKUPS:
        for role in ["target", "neighbour", "other"]:
            sub = MK[(MK.markup == m) & (MK.role == role)]
            cnd = dict(markup=m, role=role, requested=PROSODY[m])
            MR.append(C.row("delta_st", C.boot_df(sub, U, lambda q: q.d_st.mean()), len(sub), src_mk, "Praat F0 median per word", split, **cnd))
            MR.append(C.row("delta_db", C.boot_df(sub, U, lambda q: q.d_db.mean()), len(sub), src_mk, "RMS dB per word", split, **cnd))
            MR.append(C.row("dur_ratio", C.boot_df(sub, U, lambda q: np.exp(np.log(q.dur_ratio).mean())), len(sub), src_mk,
                            "model token durations (geo mean)", split, **cnd))
        if hyps:
            ps = [r.path for r in df[(df.kind == "mk") & (df.markup == m)].itertuples()]
            MR.append(C.row("wer", (sum(er[p][0] for p in ps) / sum(er[p][1] for p in ps), np.nan, np.nan), len(ps), src_mk,
                            ASR_NAME, split, markup=m))
    if hyps:
        ps = [r.path for r in df[df.kind == "mk_base"].itertuples()]
        MR.append(C.row("wer", (sum(er[p][0] for p in ps) / sum(er[p][1] for p in ps), np.nan, np.nan), len(ps), src_mk,
                        ASR_NAME, split, markup="none"))
    meta = dict(ckpt="runs/spk_enc_v3/step312500.pt", voices="8 LibriTTS-R test-clean speakers (4F/4M, H01-04, H09-12 of "
                "steer.paper.common.heldout_voices(16)) + 4 EARS training speakers with nonverbal recordings (T01-T04)",
                detector="MIT/ast-finetuned-audioset-10-10-0.4593", texts=TEXTS, mk_texts=MK_TEXTS,
                ast_class_map={f"{k[0]}:{k[1]}": [id2label[i] for i in v] for k, v in cls.items()})
    C.write("events", R, E, meta)
    EN.to_csv(C.RES / "events_env_raw.csv", index=False)
    NR.to_csv(C.RES / "events_noisyref_raw.csv", index=False)
    C.write("markup", MR, MK, dict(meta, prosody=PROSODY))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--asr_dev", default="cpu")
    ap.add_argument("--procs", type=int, default=20)
    a = ap.parse_args()
    items, H, T = plan()
    print(f"plan: {len(items)} renders", pd.Series([i['kind'] for i in items]).value_counts().to_dict(), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    st = a.stage
    if st in ("render", "all"):
        stage_render(items, H, T, a.procs)
    if st in ("measure", "all"):
        stage_measure(items)
    if st in ("ast", "all"):
        stage_ast(items, dev=a.asr_dev)
    if st in ("sim", "all"):
        stage_sim(items, H, a.asr_dev)
    if st in ("asr", "all"):
        stage_asr(items, a.asr_dev)
    if st in ("analyze", "all"):
        analyze(items, H, T)


if __name__ == "__main__":
    main()
