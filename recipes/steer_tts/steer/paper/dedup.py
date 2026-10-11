"""Train/test hygiene audit for the SteerTTS paper: speaker, AUDIO and TEXT overlap between every evaluation set and
the training data (data/meta/controls.parquet = META, 410,566 utterances) + the prompt adapter's training captions.

    CUDA_VISIBLE_DEVICES= python -m steer.paper.dedup ids        # (1) speaker / clip-id overlap
    CUDA_VISIBLE_DEVICES= python -m steer.paper.dedup text       # (3) exact / normalised / 8-gram text overlap
    CUDA_VISIBLE_DEVICES= python -m steer.paper.dedup fp_train   # landmark fingerprints of all META clips (wav24)
    CUDA_VISIBLE_DEVICES= python -m steer.paper.dedup fp_eval    # fingerprints of every evaluation clip / prompt
    CUDA_VISIBLE_DEVICES= python -m steer.paper.dedup validate   # known dups (resampled/trimmed/noisy/excerpt) vs non-dups
    CUDA_VISIBLE_DEVICES= python -m steer.paper.dedup audio      # (2) eval clips vs training index
    CUDA_VISIBLE_DEVICES= python -m steer.paper.dedup report     # -> results/dedup.json + draft/dedup_note.tex

Audio fingerprint (chromaprint is not installed): Shazam-style spectral-peak landmarks. Audio -> 8 kHz mono, STFT
512/128 (16 ms hop), log magnitude, 2-D local maxima (+-5 frames x +-7 bins) over 100 Hz-3.9 kHz, the 10 strongest
per second kept; each anchor paired with its 3 nearest peaks 1-31 frames later within +-63 bins -> 23-bit hash
(f1, f2, dt) at the anchor time. Matching is an inverted index over ALL training clips (a superset of the corpus /
speaker / +-10 % duration blocking); a candidate's score is the number of hashes that agree on one time offset
(+-1 frame), so it is robust to resampling, gain, noise, trimming and excerpting (containment), but a different take
of the same text by the same voice does not align. Thresholds are set by `validate`.
Only compact fingerprints (uint32 hash, uint32 clip, uint16 time) are stored, in --work (default: scratch dir).
Example ids in the outputs are anonymised (salted hash); real ids stay in the private work dir.
"""
import argparse
import ast
import glob
import hashlib
import io
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import maximum_filter
from scipy.signal import resample_poly

WS = Path("/home/karan/Desktop/whissle/workspace")
DATA = WS / "steer_tts" / "data"
META = DATA / "meta"
RAW = DATA / "raw"
BENCH = DATA / "bench"
STEER = Path(__file__).resolve().parents[1]
PAPER = Path("/home/karan/Desktop/whissle/marketing_post_agent/paper/steertts")
OUT_JSON = PAPER / "results" / "dedup.json"
OUT_TEX = PAPER / "draft" / "dedup_note.tex"
SEED_EN = WS / "seedtts" / "data" / "seedtts_testset" / "en"
CV3 = WS / "lb_data"
WORK = Path(os.environ.get("DEDUP_WORK", "/tmp/claude-1000/-home-karan-Desktop-whissle/"
                           "dbda8e1e-5067-4908-8376-8ea786d4d17e/scratchpad/dedup"))
CREMA_SENTENCES = [  # the 12 CREMA-D sentences (analysis only)
    "It's eleven o'clock.", "That is exactly what happened.", "I'm on my way to the meeting.",
    "I wonder what this is about.", "The airplane is almost full.", "Maybe tomorrow it will be cold.",
    "I would like a new alarm clock.", "I think I have a doctor's appointment.", "Don't forget a jacket.",
    "I think I've seen this before.", "The surface is slick.", "We'll stop in a couple of minutes."]


def anon(s, kind="x"):
    return f"{kind}-{hashlib.sha1(('steertts-dedup:' + str(s)).encode()).hexdigest()[:8]}"


# ---------------------------------------------------------------------------------------------------- fingerprint
FS, NFFT, HOP = 8000, 512, 128
LO, HI = 7, 250          # bins: ~110 Hz .. 3.9 kHz
PEAKS_PER_S, FAN, DTMAX, DFMAX = 10, 3, 31, 63
FPS = FS / HOP


def load(src, sr=None):
    """path or bytes -> float32 mono; drops the file from the page cache (do not evict other jobs' data)."""
    if isinstance(src, (str, Path)):
        with open(src, "rb") as f:
            b = f.read()
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass
    else:
        b = src
    y, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    return y.mean(1), sr


def to_fs(y, sr):
    if sr != FS:
        g = gcd(int(sr), FS)
        y = resample_poly(y, FS // g, int(sr) // g)
    return y.astype(np.float32)


def landmarks(y, sr):
    """-> (hash uint32 [n], anchor frame uint16 [n])"""
    y = to_fs(y, sr)
    if len(y) < NFFT * 2:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint16)
    nfr = 1 + (len(y) - NFFT) // HOP
    fr = np.lib.stride_tricks.as_strided(y, (nfr, NFFT), (y.strides[0] * HOP, y.strides[0]))
    S = np.abs(np.fft.rfft(fr * np.hanning(NFFT).astype(np.float32), axis=1))[:, LO:HI]
    L = np.log(S + 1e-6)
    L = L - L.mean(0, keepdims=True) * 0.5  # mild per-band whitening (channel / EQ robustness)
    pk = (L == maximum_filter(L, size=(11, 15), mode="constant", cval=-1e9)) & (L > np.median(L) + 1.0)
    t, f = np.nonzero(pk)
    if len(t) == 0:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint16)
    v = L[t, f]
    blk = (t / FPS).astype(np.int64)  # strongest PEAKS_PER_S per 1-s block
    o = np.lexsort((-v, blk))
    first = np.r_[0, np.nonzero(np.diff(blk[o]))[0] + 1]
    rank = np.arange(len(o)) - np.repeat(first, np.diff(np.r_[first, len(o)]))
    keep = np.sort(o[rank < PEAKS_PER_S])
    t, f = t[keep].astype(np.int64), f[keep].astype(np.int64)
    o = np.lexsort((f, t)); t, f = t[o], f[o]
    n, M = len(t), 24
    j = np.arange(n)[:, None] + np.arange(1, M + 1)[None, :]
    jv = np.minimum(j, n - 1)
    dt, df = t[jv] - t[:, None], f[jv] - f[:, None]
    ok = (j < n) & (dt >= 1) & (dt <= DTMAX) & (np.abs(df) <= DFMAX)
    ok &= np.cumsum(ok, 1) <= FAN
    a, b = np.nonzero(ok)
    H = (f[a] << 14) | (f[jv[a, b]] << 5) | dt[a, b]
    return H.astype(np.uint32), np.minimum(t[a], 65535).astype(np.uint16)


def _fp_job(a):
    i, src = a
    try:
        y, sr = load(src)
        h, t = landmarks(y, sr)
        return i, h, t, len(y) / sr
    except Exception as e:  # noqa
        return i, np.zeros(0, np.uint32), np.zeros(0, np.uint16), -1.0


def fingerprint_many(srcs, nproc=12, chunksize=64, log=""):
    out = [None] * len(srcs)
    with ProcessPoolExecutor(nproc) as ex:
        for k, (i, h, t, d) in enumerate(ex.map(_fp_job, enumerate(srcs), chunksize=chunksize)):
            out[i] = (h, t, d)
            if log and k % 20000 == 0:
                print(log, k, "/", len(srcs), flush=True)
    return out


class Index:
    """Inverted index over fingerprints: sorted hashes -> (clip, time)."""

    def __init__(self, h, c, t, nh, max_post=3000):
        o = np.argsort(h, kind="stable")
        self.h, self.c, self.t = h[o], c[o], t[o]
        self.nh = nh  # hashes per clip
        u, s, n = np.unique(self.h, return_index=True, return_counts=True)
        self.u, self.s, self.n = u, s, n
        self.max_post = max_post

    @classmethod
    def load(cls, files):
        hs, cs, ts, nhs, ids, off = [], [], [], [], [], 0
        for f in files:
            z = np.load(f, allow_pickle=False)
            hs.append(z["h"]); ts.append(z["t"]); cs.append(z["c"].astype(np.int64) + off)
            nhs.append(z["nh"]); ids.extend(json.load(open(str(f).replace(".npz", ".ids.json"))))
            off += len(z["nh"])
        ix = cls(np.concatenate(hs), np.concatenate(cs).astype(np.uint32), np.concatenate(ts),
                 np.concatenate(nhs))
        ix.ids = ids
        return ix

    def query(self, h, t, topk=5, min_count=4):
        """-> list of (clip, aligned count, n hashes of clip), best first."""
        if len(h) == 0:
            return []
        k = np.searchsorted(self.u, h)
        k = np.minimum(k, len(self.u) - 1)
        ok = (self.u[k] == h) & (self.n[k] <= self.max_post)
        if not ok.any():
            return []
        k, tq = k[ok], t[ok].astype(np.int64)
        st, nn = self.s[k], self.n[k]
        rep = np.repeat(np.arange(len(k)), nn)
        pos = np.repeat(st, nn) + (np.arange(nn.sum()) - np.repeat(np.cumsum(nn) - nn, nn))
        clip = self.c[pos].astype(np.int64)
        off = self.t[pos].astype(np.int64) - tq[rep] + 70000
        best = Counter()
        for q in (off // 2, (off + 1) // 2):  # +-1-frame tolerance
            key = clip * 100000 + q
            uk, cnt = np.unique(key, return_counts=True)
            cl = uk // 100000
            o = np.lexsort((-cnt, cl))
            cl, cnt = cl[o], cnt[o]
            first = np.r_[True, cl[1:] != cl[:-1]]
            for a, b in zip(cl[first], cnt[first]):
                if b > best[a]:
                    best[a] = int(b)
        res = sorted(((a, b, int(self.nh[a])) for a, b in best.items() if b >= min_count), key=lambda x: -x[1])
        return res[:topk]


def is_dup(count, nq, nc, thr):
    return count >= thr["min_count"] and count / max(1, min(nq, nc)) >= thr["min_frac"]


# ---------------------------------------------------------------------------------------------------- data
def meta():
    return pd.read_parquet(META / "controls.parquet", columns=["id", "corpus", "speaker", "text", "dur", "tag"])


def wav_path(uid):
    d = uid.split("_")[0]
    return DATA / "wav24" / ("cartesia" if d == "cartcov" else d) / f"{uid}.wav"  # cartcov = Cartesia coverage pass


def psc(split="test"):
    return pd.read_parquet(RAW / "paraspeechcaps" / "data" / f"{split}-00000-of-00001.parquet")


def testclean():
    return pd.concat([pd.read_parquet(f, columns=["id", "speaker_id", "chapter_id", "text_normalized"])
                      for f in sorted((RAW / "libritts_r/data/test.clean").glob("*.parquet"))], ignore_index=True)


def seed_rows():
    rows = []
    for l in open(SEED_EN / "meta.lst"):
        p = l.rstrip("\n").split("|")
        if len(p) >= 4:
            rows.append(dict(id=p[0], prompt_text=p[1], prompt_wav=str(SEED_EN / p[2]), target_text=p[3]))
    return pd.DataFrame(rows)


def cv3_rows():
    d = pd.read_parquet(CV3 / "cv3_zero_shot_en.parquet", columns=["id", "prompt_text", "target_text"])
    d["prompt_wav"] = [str(CV3 / "cv3_en_prompts" / f"{i}.wav") for i in d.id]
    return d


def code_text_lists():
    """String literals inside list / tuple / dict / set literals of the evaluation scripts (harness TEXTS, paper/*
    text and prompt lists): sentences of >= 3 words. Docstrings and f-strings are not containers -> excluded."""
    files = [STEER / "harness.py", STEER / "tag_eval.py", STEER / "samples.py"] + sorted((STEER / "paper").glob("*.py"))
    out = []
    for f in files:
        if f.name == "dedup.py":
            continue
        try:
            tree = ast.parse(open(f).read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                elts = node.elts if not isinstance(node, ast.Dict) else list(node.values) + list(node.keys)
                for e in elts:
                    if isinstance(e, ast.Constant) and isinstance(e.value, str):
                        s = e.value.strip()
                        if len(s.split()) >= 3 and re.search(r"[a-z]", s) and not re.search(r"[{}=/\\%<>]|\.py|_", s):
                            out.append((f.name, s))
    return sorted(set(out))


# ---------------------------------------------------------------------------------------------------- (1) ids
def stage_ids():
    c = meta()
    res = {}
    lt = testclean()
    trs = set(c[c.corpus == "libritts"].speaker.str[len("libritts_"):])
    trch = set(c[c.corpus == "libritts"].id.str.split("_").str[2])
    bench = pd.read_json(BENCH / "libritts.jsonl", lines=True, dtype={"speaker": str, "id": str})
    clone = pd.read_json(BENCH / "clone.jsonl", lines=True, dtype={"speaker": str})
    sys.path.insert(0, str(STEER.parent))
    hv = json.load(open(WS / "steer_tts" / "runs" / "paper" / "heldout_voice_map_private.json")) \
        if (WS / "steer_tts" / "runs" / "paper" / "heldout_voice_map_private.json").exists() else None
    if hv is None:
        hits = glob.glob(str(WS / "steer_tts" / "**" / "heldout_voice_map_private.json"), recursive=True)
        hv = json.load(open(hits[0])) if hits else {}
    res["libritts_testclean"] = dict(
        n_utts=len(lt), n_speakers=int(lt.speaker_id.nunique()),
        speaker_overlap=len(set(lt.speaker_id.astype(str)) & trs),
        chapter_overlap=len(set(lt.chapter_id.astype(str)) & trch),
        utt_id_overlap=int(("libritts_" + lt.id).isin(set(c.id)).sum()),
        bench500=dict(n=len(bench), n_speakers=int(bench.speaker.nunique()),
                      speaker_overlap=len(set(bench.speaker) & trs)),
        clone=dict(n=len(clone), ref_in_bench=True, speaker_overlap=len(set(clone.speaker.astype(str)) & trs)),
        heldout_steering_voices=dict(n=len(hv), speaker_overlap=len(set(map(str, hv.values())) & trs)),
        training_libritts_speakers=len(trs))
    # ParaSpeechCaps test
    t = psc("test")
    out = {}
    for src in ("ears", "expresso"):
        s = set(t[t.source == src].speakerid)
        trsp = set(c[c.corpus == src].speaker.str.split("_").str[1])
        nvsp = set(c[c.corpus == "nvtts"].speaker.str[len("nvtts_"):])
        out[src] = dict(n=int((t.source == src).sum()), n_speakers=len(s), speaker_overlap=len(s & trsp),
                        speaker_overlap_nvtts=len(s & nvsp))
    # clip-id overlap (Expresso / EARS ids are comparable)
    tr_ids = set(c.id)
    ex = t[t.source == "expresso"]
    stems = ex.relative_audio_path.map(lambda p: Path(p).stem)
    out["expresso"]["clip_id_in_expresso_train"] = int(("expresso_" + stems).isin(tr_ids).sum())
    out["expresso"]["clip_id_in_nvtts_train"] = int(("nvtts_" + stems).isin(tr_ids).sum())
    # Expresso conversational segments (no local audio): NonverbalTTS trains on segments of the same dialogues ->
    # same dialogue file + >= 50 % of the PSC transcript's word 5-grams inside one NVTTS segment transcript
    nve = c[(c.corpus == "nvtts") & ~c.id.str.startswith("nvtts_id")].copy()
    nve["dlg"] = nve.id.str[6:].str.extract(r"^(ex\d\d-ex\d\d_[a-z-]+_\d{3})")[0]
    conv = ex[ex.relative_audio_path.str.contains("conversational")]
    conv_hits, conv_dlg = [], 0
    for p_, tr_ in zip(conv.relative_audio_path, conv.transcription):
        dlg = re.match(r"(ex\d\d-ex\d\d_[a-z-]+_\d{3})", Path(p_).stem).group(1)
        cand = nve[nve.dlg == dlg]
        conv_dlg += len(cand) > 0
        g = grams(norm(tr_), 5)
        if not len(cand) or not g:
            continue
        best = max(((len(g & grams(norm(x), 5)) / len(g), i) for x, i in zip(cand.text, cand.id)))
        if best[0] >= 0.5:
            conv_hits.append(dict(psc=anon(p_, "psc"), train=anon(best[1], "nvtts"), ngram5_containment=round(best[0], 2),
                                  n_words=len(norm(tr_).split())))
    out["expresso"]["conversational_rows"] = len(conv)
    out["expresso"]["conversational_rows_dialogue_in_nvtts_train"] = int(conv_dlg)
    out["expresso"]["conversational_rows_same_speech_in_nvtts_train"] = len(conv_hits)
    out["expresso"]["conversational_examples"] = conv_hits
    out["expresso"]["read_rows"] = int(len(ex) - len(conv))
    ea = t[t.source == "ears"]
    e_ids = ea.relative_audio_path.map(lambda p: f"ears_{Path(p).parent.name}_{Path(p).stem}")
    ff = [json.loads(l) for l in open(META / "manifest_earsff.jsonl")]
    ff_src = Counter(Path(r["src"]).parent.name + "/" + Path(r["src"]).name for r in ff if r["id"] in tr_ids)
    out["ears"]["clip_id_in_ears_train"] = int(e_ids.isin(tr_ids).sum())
    out["ears"]["source_file_segmented_into_earsff_train"] = int(ea.relative_audio_path.map(lambda p: ff_src[p] > 0).sum())
    # VoxCeleb vs NonverbalTTS (VoxCeleb part): speaker / YouTube video / segment ids
    nv = c[(c.corpus == "nvtts") & c.id.str.startswith("nvtts_id")].copy()
    nv["spk"], nv["vid"], nv["seg"] = nv.id.str[6:13], nv.id.str[14:-6], nv.id.str[-5:]
    v = t[t.source == "voxceleb"].copy()
    parts = v.relative_audio_path.str.split("/")
    v["spk"], v["vid"], v["seg"] = parts.str[-3], parts.str[-2], parts.str[-1].str[:5]
    so, vo = set(v.spk) & set(nv.spk), set(v.vid) & set(nv.vid)
    m = v.merge(nv, on=["vid", "seg"], suffixes=("_p", "_n"))
    same = []
    for _, r in m.iterrows():
        same.append(dict(psc=anon(r.relative_audio_path, "psc"), train=anon(r.id_n if "id_n" in r else r.id, "nvtts"),
                         psc_dur=round(float(r.duration), 2), train_dur=round(float(r.dur), 2),
                         text_sim=round(text_sim(r.transcription, r.text_n if "text_n" in r else r.text), 3)))
    out["voxceleb"] = dict(n=len(v), n_speakers=int(v.spk.nunique()), speaker_overlap_nvtts=len(so),
                           rows_with_speaker_in_nvtts=int(v.spk.isin(so).sum()), video_overlap_nvtts=len(vo),
                           rows_with_video_in_nvtts=int(v.vid.isin(vo).sum()),
                           rows_same_video_and_segment_in_nvtts=len(m), same_segment_examples=same,
                           nvtts_train_clips_of_overlapping_speakers=int(nv.spk.isin(so).sum()),
                           audio_available_locally=False)
    res["psc_test"] = out
    # Seed-TTS / CV3: Common Voice prompts; training has no Common Voice
    res["seedtts_en"] = dict(n=len(seed_rows()), note="Common Voice speakers; no Common Voice audio in training")
    res["cv3_zero_shot_en"] = dict(n=len(cv3_rows()), note="no speaker ids comparable with training")
    res["training_corpora"] = c.groupby("corpus").agg(n=("id", "size"), speakers=("speaker", "nunique")).to_dict("index")
    WORK.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(WORK / "ids.json", "w"), indent=1, default=int)
    print(json.dumps(res, indent=1, default=int)[:4000])


# ---------------------------------------------------------------------------------------------------- (3) text
_EVT = re.compile(r"\[[a-z_ ]+\]")


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = _EVT.sub(" ", s)
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = s.replace("'", "")
    return re.sub(r"\s+", " ", s).strip()


def grams(s, n=8):
    w = s.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def text_sim(a, b):
    a, b = set(norm(a).split()), set(norm(b).split())
    return len(a & b) / max(1, len(a | b))


class TextIndex:
    def __init__(self, texts, ids, n=8):
        self.n = n
        self.exact = defaultdict(list)
        self.norm = defaultdict(list)
        self.g = {}
        for t, i in zip(texts, ids):
            self.exact[str(t).strip()].append(i)
            nt = norm(t)
            self.norm[nt].append(i)
            for g in grams(nt, n):
                self.g.setdefault(hash(g), i)

    def check(self, t):
        nt = norm(t)
        ex = self.exact.get(str(t).strip(), [])
        nm = self.norm.get(nt, []) if nt else []
        gs = grams(nt, self.n)
        hit = [g for g in gs if hash(g) in self.g]
        frac = len(hit) / len(gs) if gs else None
        return dict(exact=len(ex), norm=len(nm), ngram_frac=frac,
                    ids=(ex or nm or [self.g[hash(h)] for h in hit])[:3])


def adapter_captions(c):
    """The prompt adapter's training text: template captions (6 per META clip, steer.captions, seed 0) + real
    instructions (Cartesia render instructions + 2 paraphrases) + ParaSpeechCaps train_base descriptions of EARS /
    Expresso clips in META (steer.prompt.real_pairs), regenerated deterministically on CPU."""
    os.environ.setdefault("STEER_TTS_ROOT", str(WS / "steer_tts"))  # steer.config paths (RAW / META)
    sys.path.insert(0, str(STEER.parent))
    from steer import prompt as P
    assert Path(P.META) == META, P.META
    full = pd.read_parquet(META / "controls.parquet")
    caps = [p[0] for p in P.build_pairs(full)]
    real = [p[0] for p in P.real_pairs(full)]
    return caps, real


def stage_text():
    c = meta()
    tr = TextIndex(c.text.tolist(), c.id.tolist())
    sets = {}
    lt = testclean()
    bench = pd.read_json(BENCH / "libritts.jsonl", lines=True, dtype={"id": str})
    sets["libritts_testclean_all"] = lt.text_normalized.tolist()
    sets["libritts_bench500"] = bench.text.tolist()
    sd = seed_rows(); sets["seedtts_en_target"] = sorted(set(sd.target_text)); sets["seedtts_en_prompt"] = sorted(set(sd.prompt_text))
    cv = cv3_rows(); sets["cv3_en_target"] = sorted(set(cv.target_text)); sets["cv3_en_prompt"] = sorted(set(cv.prompt_text))
    t = psc("test")
    for src in ("voxceleb", "expresso", "ears"):
        sets[f"psc_test_{src}_transcript"] = t[t.source == src].transcription.str.strip().tolist()
    sets["ljspeech_test500"] = pd.read_json(BENCH / "ljspeech.jsonl", lines=True).text.tolist()
    sets["cremad_sentences"] = CREMA_SENTENCES
    code = code_text_lists()
    sets["harness_and_paper_lists"] = sorted({s for _, s in code})
    res = {}
    for k, texts in sets.items():
        rows = [tr.check(x) for x in texts]
        ex = [i for i, r in enumerate(rows) if r["exact"]]
        nm = [i for i, r in enumerate(rows) if r["norm"]]
        ng = [i for i, r in enumerate(rows) if r["ngram_frac"] is not None and r["ngram_frac"] >= 0.5]
        corp = Counter(rows[i]["ids"][0].split("_")[0] for i in sorted(set(ex) | set(nm) | set(ng)) if rows[i]["ids"])
        anyi = set(ex) | set(nm) | set(ng)
        res[k] = dict(n=len(texts), exact=len(ex), normalised=len(nm), ngram8_ge50=len(ng),
                      any=len(anyi), any_ge6_words=sum(len(norm(texts[i]).split()) >= 6 for i in anyi), matched_training_corpora=dict(corp),
                      examples=[dict(text=texts[i][:120], train=anon(rows[i]["ids"][0], "train"),
                                     ngram_frac=rows[i]["ngram_frac"]) for i in sorted(set(nm) | set(ng))[:5]])
        print(k, {a: b for a, b in res[k].items() if a != "examples"}, flush=True)
    # where a harness / paper-list sentence came from, for the record
    res["harness_and_paper_lists"]["files"] = sorted({f for f, _ in code})
    # prompts / descriptions vs prompt-adapter training captions
    caps, real = adapter_captions(c)
    ci = TextIndex(caps + real, ["template"] * len(caps) + ["real"] * len(real))
    tb = psc("train_base")
    psc_train_desc = set(norm(str(d[0])) for d in tb.text_description)
    desc = t.text_description.map(lambda a: str(a[0]).strip()).tolist()
    prompts = sets["harness_and_paper_lists"]
    for k, texts in (("psc_test_descriptions", desc), ("eval_prompt_lists", prompts)):
        rows = [ci.check(x) for x in texts]
        nm = [i for i, r in enumerate(rows) if r["norm"]]
        res[k + "_vs_adapter_captions"] = dict(
            n=len(texts), exact=sum(bool(r["exact"]) for r in rows), normalised=len(nm),
            by_source=dict(Counter(rows[i]["ids"][0] for i in nm)),
            in_psc_train_base_any_source=sum(norm(x) in psc_train_desc for x in texts),
            examples=[texts[i][:120] for i in nm[:5]])
        if k == "eval_prompt_lists":
            src = defaultdict(list)
            for f, x in code:
                src[x].append(f)
            res[k + "_vs_adapter_captions"]["matches"] = [
                dict(prompt=texts[i], files=sorted(set(src[texts[i]])), caption_source=rows[i]["ids"][0]) for i in nm]
        print(k, res[k + "_vs_adapter_captions"], flush=True)
        json.dump(res, open(WORK / "text.partial.json", "w"), indent=1, default=int)
    res["adapter_caption_counts"] = dict(template=len(caps), template_unique=len(set(caps)), real=len(real),
                                         real_unique=len(set(real)))
    json.dump(res, open(WORK / "text.json", "w"), indent=1, default=int)


# ---------------------------------------------------------------------------------------------------- fingerprints
def stage_fp_train(nproc):
    c = meta()
    for corpus, g in c.groupby("corpus"):
        f = WORK / f"fp_train_{corpus}.npz"
        if f.exists():
            continue
        ids = g.id.tolist()
        res = fingerprint_many([str(wav_path(i)) for i in ids], nproc=nproc, log=corpus)
        save_fp(f, ids, res)
        n_bad = sum(r[2] < 0 for r in res)
        print(corpus, len(ids), "clips; unreadable", n_bad, flush=True)
        assert n_bad == 0, f"{n_bad} {corpus} clips could not be read"


def save_fp(f, ids, res):
    h = np.concatenate([r[0] for r in res]); t = np.concatenate([r[1] for r in res])
    nh = np.array([len(r[0]) for r in res], np.int32)
    cc = np.repeat(np.arange(len(ids), dtype=np.uint32), nh)
    np.savez(f, h=h, t=t, c=cc, nh=nh, dur=np.array([r[2] for r in res], np.float32))
    json.dump(list(ids), open(str(f).replace(".npz", ".ids.json"), "w"))


def eval_sources():
    """name -> list of (eval_id, src, meta dict). src = path or bytes."""
    import pyarrow.parquet as pq
    S = {}
    rows = []
    bench = set(pd.read_json(BENCH / "libritts.jsonl", lines=True, dtype={"id": str}).id)
    for f in sorted((RAW / "libritts_r/data/test.clean").glob("*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(columns=["id", "speaker_id", "audio"], batch_size=256):
            d = b.to_pydict()
            for i, s, au in zip(d["id"], d["speaker_id"], d["audio"]):
                rows.append((i, au["bytes"], dict(speaker=str(s), bench=i in bench)))
    S["libritts_testclean"] = rows
    t = psc("test")
    S["psc_test_ears"] = [(p, str(RAW / "ears" / p), dict(speaker=s)) for p, s in
                          zip(t[t.source == "ears"].relative_audio_path, t[t.source == "ears"].speakerid)]
    ex = t[t.source == "expresso"]
    want = {Path(p).stem: s for p, s in zip(ex.relative_audio_path, ex.speakerid)}
    rows = []
    for f in sorted((RAW / "expresso" / "read").glob("*.parquet")):
        tb = pq.read_table(f, columns=["id"]).to_pydict()["id"]
        hit = [k for k, i in enumerate(tb) if i in want]
        if hit:
            d = pq.read_table(f, columns=["id", "audio"]).take(hit).to_pydict()
            rows += [(i, au["bytes"], dict(speaker=want[i], raw_shard=f.name)) for i, au in zip(d["id"], d["audio"])]
    S["psc_test_expresso"] = rows
    sd = seed_rows()
    S["seedtts_en_prompt"] = [(Path(p).stem, p, {}) for p in sorted(set(sd.prompt_wav))]
    S["seedtts_en_target_gt"] = [(Path(p).stem, str(p), {}) for p in sorted((SEED_EN / "wavs").glob("*.wav"))]
    cv = cv3_rows()
    S["cv3_en_prompt"] = [(i, p, {}) for i, p in zip(cv.id, cv.prompt_wav) if Path(p).exists()]
    return S


def stage_fp_eval(nproc):
    S = eval_sources()
    for name, rows in S.items():
        f = WORK / f"fp_eval_{name}.npz"
        res = fingerprint_many([r[1] for r in rows], nproc=nproc, chunksize=8)
        save_fp(f, [r[0] for r in rows], res)
        json.dump([r[2] for r in rows], open(WORK / f"fp_eval_{name}.meta.json", "w"))
        print(name, len(rows), "clips, mean hashes", np.mean([len(r[0]) for r in res]), flush=True)


def load_fp(f):
    z = np.load(f)
    ids = json.load(open(str(f).replace(".npz", ".ids.json")))
    st = np.r_[0, np.cumsum(z["nh"])]
    return ids, [(z["h"][st[i]:st[i + 1]], z["t"][st[i]:st[i + 1]]) for i in range(len(ids))], z["dur"]


# ---------------------------------------------------------------------------------------------------- validation
def _augment(a):
    """(kind, uid, seed) -> fingerprint of a transformed copy of a training clip."""
    kind, uid, seed = a
    rng = np.random.default_rng(seed)
    y, sr = load(str(wav_path(uid)))
    if kind == "resample16k_gain":
        y = resample_poly(y, 2, 3) * 0.5; sr = 16000
    elif kind == "trim":
        y = y[int(0.3 * sr): len(y) - int(0.2 * sr)]
    elif kind == "noise15db":
        p = np.mean(y ** 2) + 1e-12
        y = y + rng.normal(0, np.sqrt(p / 10 ** 1.5), len(y)).astype(np.float32)
    elif kind == "excerpt3s":
        L = int(3 * sr); s = rng.integers(0, max(1, len(y) - L)); y = y[s:s + L]
    elif kind == "combined":  # 16 kHz + trim + 20 dB noise + gain
        y = y[int(0.25 * sr): len(y) - int(0.25 * sr)]
        y = resample_poly(y, 2, 3) * 0.7; sr = 16000
        p = np.mean(y ** 2) + 1e-12
        y = y + rng.normal(0, np.sqrt(p / 100), len(y)).astype(np.float32)
    h, t = landmarks(y.astype(np.float32), sr)
    return kind, uid, h, t


def train_index():
    return Index.load(sorted(WORK.glob("fp_train_*.npz")))


def stage_validate(nproc):
    c = meta()
    rng = np.random.default_rng(0)
    pos_ids = []
    for corpus, g in c.groupby("corpus"):
        g = g[g.dur >= 3.5]
        pos_ids += g.id.sample(min(60, len(g)), random_state=0).tolist()
    kinds = ["resample16k_gain", "trim", "noise15db", "excerpt3s", "combined"]
    jobs = [(k, u, int(rng.integers(1 << 30))) for u in pos_ids for k in kinds]
    with ProcessPoolExecutor(nproc) as ex:
        aug = list(ex.map(_augment, jobs, chunksize=8))
    ix = train_index()
    pos_of = {u: i for i, u in enumerate(ix.ids)}
    P = []
    for kind, uid, h, t in aug:
        r = ix.query(h, t, topk=3, min_count=1)
        tgt = pos_of[uid]
        sc = next((b for a, b, n in r if a == tgt), 0)
        other = max([b for a, b, n in r if a != tgt] + [0])
        P.append(dict(kind=kind, corpus=uid.split("_")[0], nq=int(len(h)), nc=int(ix.nh[tgt]), score=sc,
                      top1=bool(r and r[0][0] == tgt), best_other=other))
    # hard negatives: same voice + same text, different render / take (Cartesia groups; EARS same sentence by
    # other speakers; Expresso same text other style) - query clip A, score of sibling B. Plus LibriTTS dev-clean
    # (speakers in no training split) vs the whole index.
    N = []
    car = c[c.corpus == "cartesia"].copy()
    car["grp"] = car.id.str.rsplit("_", n=3).str[0]
    for g, d in list(car.groupby("grp"))[:400]:
        if len(d) < 2:
            continue
        a, sib = d.id.iloc[0], set(d.id.iloc[1:])
        N.append(("cartesia_same_voice_text", a, sib))
    ears = c[(c.corpus == "ears") & ~c.id.str.contains("freeform")].copy()
    ears["key"] = ears.id.str.split("_", n=2).str[2]
    for k, d in list(ears.groupby("key"))[:60]:
        if len(d) < 2:
            continue
        N.append(("ears_same_text_other_speaker", d.id.iloc[0], set(d.id.iloc[1:])))
    expr = c[c.corpus == "expresso"]
    for k, d in list(expr.groupby(["speaker", "text"]))[:4000]:
        if len(d) >= 2 and d.id.str.split("_").str[2].nunique() >= 2:
            N.append(("expresso_same_speaker_text_other_style", d.id.iloc[0], set(d.id.iloc[1:])))
    N = N[:900]
    neg = []
    qs = fingerprint_many([str(wav_path(a)) for _, a, _ in N], nproc=nproc, chunksize=8)
    for (kind, a, sib), (h, t, d) in zip(N, qs):
        r = ix.query(h, t, topk=200, min_count=1)
        sibs = {pos_of[s] for s in sib}
        sc = max([b for x, b, n in r if x in sibs] + [0])
        nc = max([n for x, b, n in r if x in sibs] + [1])
        self_sc = next((b for x, b, n in r if x == pos_of[a]), 0)
        neg.append(dict(kind=kind, nq=int(len(h)), score=sc, nc=nc, self_score=self_sc))
    import pyarrow.parquet as pq
    dev = []
    for f in sorted((RAW / "libritts_r/data/dev.clean").glob("*.parquet"))[:1]:
        for b in pq.ParquetFile(f).iter_batches(columns=["id", "audio"], batch_size=300):
            d = b.to_pydict(); dev = [(i, au["bytes"]) for i, au in zip(d["id"], d["audio"])]; break
    qs = fingerprint_many([b for _, b in dev], nproc=nproc, chunksize=8)
    for (i, _), (h, t, d) in zip(dev, qs):
        r = ix.query(h, t, topk=1, min_count=1)
        neg.append(dict(kind="libritts_devclean_vs_all", nq=int(len(h)), score=r[0][1] if r else 0,
                        nc=r[0][2] if r else 1, self_score=None))
    P, Ng = pd.DataFrame(P), pd.DataFrame(neg)
    P["frac"] = P.score / np.minimum(P.nq, P.nc).clip(1)
    Ng["frac"] = Ng.score / np.minimum(Ng.nq, Ng.nc).clip(1)
    # threshold: smallest (count, frac) pair separating every negative, maximising recall
    best = None
    for mc in range(5, 41):
        for mf in (0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2):
            fp_ = ((Ng.score >= mc) & (Ng.frac >= mf)).sum()
            tp = ((P.score >= mc) & (P.frac >= mf)).mean()
            if fp_ == 0 and (best is None or tp > best[0] + 1e-9):
                best = (tp, mc, mf)
    tp, mc, mf = best
    # add a safety margin: double the count floor the data needs, keep frac
    thr = dict(min_count=int(max(mc, Ng.score.max() + 1)), min_frac=mf)
    thr["min_count"] = int(max(thr["min_count"], 8))
    recall = {k: float(((g.score >= thr["min_count"]) & (g.frac >= thr["min_frac"])).mean()) for k, g in P.groupby("kind")}
    fpr = {k: float(((g.score >= thr["min_count"]) & (g.frac >= thr["min_frac"])).mean()) for k, g in Ng.groupby("kind")}
    out = dict(threshold=thr, n_pos=len(P), n_neg=len(Ng), recall_by_transform=recall,
               top1_by_transform=P.groupby("kind").top1.mean().to_dict(),
               pos_score_median=P.groupby("kind").score.median().to_dict(),
               pos_frac_median=P.groupby("kind").frac.median().round(3).to_dict(),
               false_pos_rate_by_kind=fpr, n_neg_by_kind=Ng.kind.value_counts().to_dict(),
               neg_score_max=Ng.groupby("kind").score.max().to_dict(),
               neg_score_p99=Ng.groupby("kind").score.quantile(0.99).to_dict(),
               neg_self_score_median=Ng.dropna(subset=["self_score"]).groupby("kind").self_score.median().to_dict())
    json.dump(out, open(WORK / "validate.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------------------------------------------- (2) audio
def stage_audio():
    thr = json.load(open(WORK / "validate.json"))["threshold"]
    c = meta().set_index("id")
    ix = train_index()
    tr_corpus = np.array([i.split("_")[0].replace("cartcov", "cartesia") for i in ix.ids])
    tr_spk = c.speaker.reindex(ix.ids).to_numpy()
    tr_dur = c.dur.reindex(ix.ids).to_numpy()
    res = {}
    family = {"libritts_testclean": ("libritts",), "psc_test_ears": ("ears",),
              "psc_test_expresso": ("expresso", "nvtts"), "seedtts_en_prompt": (), "seedtts_en_target_gt": (),
              "cv3_en_prompt": ()}
    for name in family:
        f = WORK / f"fp_eval_{name}.npz"
        ids, fps, durs = load_fp(f)
        md = json.load(open(WORK / f"fp_eval_{name}.meta.json"))
        hits, blocked_max, glob_max, nblock = [], [], [], []
        for eid, (h, t), d, m in zip(ids, fps, durs, md):
            r = ix.query(h, t, topk=50, min_count=3)
            glob_max.append(r[0][1] if r else 0)
            # blocked view (corpus family, speaker if comparable, duration +-10 %): best score inside the block
            bl = [x for x in r if tr_corpus[x[0]] in family[name] and
                  (m.get("speaker") is None or str(tr_spk[x[0]]).split("_", 1)[-1] == m["speaker"]
                   or tr_corpus[x[0]] == "nvtts") and abs(tr_dur[x[0]] - d) <= 0.1 * d]
            blocked_max.append(bl[0][1] if bl else 0)
            for a, b, n in r:
                if is_dup(b, len(h), n, thr):
                    hits.append(dict(eval=eid, train=ix.ids[a], count=b, frac=round(b / max(1, min(len(h), n)), 3),
                                     eval_dur=round(float(d), 2), train_dur=round(float(tr_dur[a]), 2),
                                     same_speaker=(m.get("speaker") is not None and
                                                   str(tr_spk[a]).split("_", 1)[-1] == m["speaker"]),
                                     bench=m.get("bench")))
        ev_hit = sorted({x["eval"] for x in hits})
        res[name] = dict(n=len(ids), n_with_audio_dup=len(ev_hit), n_dup_pairs=len(hits),
                         dup_train_corpora=dict(Counter(x["train"].split("_")[0] for x in hits)),
                         global_best_score_median=float(np.median(glob_max)) if glob_max else None,
                         global_best_score_p99=float(np.percentile(glob_max, 99)) if glob_max else None,
                         blocked_best_score_max=int(max(blocked_max)) if blocked_max else 0,
                         hits_private=hits)
        if name == "libritts_testclean":
            res[name]["bench500_with_audio_dup"] = len({x["eval"] for x in hits if x.get("bench")})
        print(name, {k: v for k, v in res[name].items() if k != "hits_private"}, flush=True)
    json.dump(res, open(WORK / "audio.json", "w"), indent=1, default=int)


# ---------------------------------------------------------------------------------------------------- report
def eval_texts():
    """eval clip id (as fingerprinted) -> transcript, for the transcript cross-check of audio candidates."""
    T = {}
    lt = testclean(); T.update(zip(lt.id, lt.text_normalized))
    t = psc("test")
    T.update((Path(p).stem, x) for p, x in zip(t.relative_audio_path, t.transcription))
    T.update((p, x) for p, x in zip(t.relative_audio_path, t.transcription))
    sd = seed_rows()
    T.update((Path(p).stem, x) for p, x in zip(sd.prompt_wav, sd.prompt_text))
    T.update((i, x) for i, x in zip(sd.id, sd.target_text))
    cv = cv3_rows(); T.update(zip(cv.id, cv.prompt_text))
    return T


_CONFIRM = {}


def _hits(audio, name):
    """Audio candidates above the validated threshold, each cross-checked against the transcripts: confirmed when
    the eval transcript and the training transcript share >= 50 % of words (Jaccard) or one contains the other's
    word 5-grams >= 50 % (excerpts). Candidates at the threshold with unrelated text are fingerprint collisions."""
    if not _CONFIRM:
        _CONFIRM["T"] = eval_texts()
        _CONFIRM["C"] = meta().set_index("id").text
    T, C = _CONFIRM["T"], _CONFIRM["C"]
    out = []
    for h in audio.get(name, {}).get("hits_private", []):
        a, b = norm(T.get(h["eval"], "")), norm(C.get(h["train"], ""))
        ga, gb = grams(a, 5), grams(b, 5)
        cont = max(len(ga & gb) / len(ga) if ga else 0, len(ga & gb) / len(gb) if gb else 0)
        h = dict(h, text_jaccard=round(text_sim(a, b), 3), text_containment=round(cont, 3))
        h["confirmed"] = h["text_jaccard"] >= 0.5 or cont >= 0.5
        out.append(h)
    return out


def stage_report():
    ids = json.load(open(WORK / "ids.json"))
    text = json.load(open(WORK / "text.json"))
    val = json.load(open(WORK / "validate.json"))
    audio = json.load(open(WORK / "audio.json"))
    thr = val["threshold"]
    bench = pd.read_json(BENCH / "libritts.jsonl", lines=True, dtype={"id": str, "speaker": str})
    clone = pd.read_json(BENCH / "clone.jsonl", lines=True, dtype={"speaker": str})
    hv = json.load(open(WS / "steer_tts" / "runs" / "paper" / "heldout_voice_map_private.json"))
    clone_ids = {Path(p).stem for p in clone.gt_path} | {Path(p).stem for p in clone.ref_path}
    hv_spk = set(map(str, hv.values()))
    hv_ids = set(bench[bench.speaker.isin(hv_spk)].id)

    def ta(k):
        x = text[k]
        return dict(n=x["n"], exact=x["exact"], normalised=x["normalised"], ngram8_overlap_ge50pct=x["ngram8_ge50"],
                    any=x["any"], any_ge6_words=x.get("any_ge6_words"), matched_training_corpora=x["matched_training_corpora"], examples=x["examples"][:3])

    def aud(name, subset=None):
        a = audio[name]
        hs = [h for h in _hits(audio, name) if subset is None or h["eval"] in subset]
        cf = [h for h in hs if h["confirmed"]]
        return dict(n_clips_checked=a["n"] if subset is None else len(subset),
                    n_candidates_above_threshold=len({h["eval"] for h in hs}),
                    n_rejected_unrelated_transcript=len({h["eval"] for h in hs} - {h["eval"] for h in cf}),
                    rejected_max_count=max([h["count"] for h in hs if not h["confirmed"]] + [0]),
                    n_with_near_dup=len({h["eval"] for h in cf}),
                    near_dup_train_corpora=dict(Counter(h["train"].split("_")[0] for h in cf)),
                    best_score_median=a["global_best_score_median"], best_score_p99=a["global_best_score_p99"],
                    blocked_best_score_max=a["blocked_best_score_max"],
                    examples=[dict(eval=anon(h["eval"], "eval"), train=anon(h["train"], "train"), count=h["count"],
                                   frac=h["frac"], eval_dur=h["eval_dur"], train_dur=h["train_dur"],
                                   text_jaccard=h["text_jaccard"], confirmed=h["confirmed"])
                              for h in (cf + [h for h in hs if not h["confirmed"]])[:3]])
    L = ids["libritts_testclean"]
    sets = {}
    sets["libritts_r_test_clean_bench500"] = dict(
        n=500, speakers=L["bench500"]["n_speakers"], speaker_overlap=L["bench500"]["speaker_overlap"],
        utt_id_overlap=0, chapter_overlap=L["chapter_overlap"],
        audio=aud("libritts_testclean", set(bench.id)), text=ta("libritts_bench500"),
        note="text matches are the same book passage read by a different LibriVox reader (speaker disjoint)")
    sets["libritts_r_test_clean_clone"] = dict(
        n=len(clone), speaker_overlap=L["clone"]["speaker_overlap"],
        audio=aud("libritts_testclean", clone_ids), note="targets and prompts are bench500 clips")
    sets["libritts_r_test_clean_heldout_steering_voices"] = dict(
        n_voices=L["heldout_steering_voices"]["n"], speaker_overlap=L["heldout_steering_voices"]["speaker_overlap"],
        audio=aud("libritts_testclean", hv_ids), texts="harness TEXTS (see harness_eval_texts)")
    sets["libritts_r_test_clean_all"] = dict(
        n=L["n_utts"], speakers=L["n_speakers"], speaker_overlap=L["speaker_overlap"],
        training_libritts_speakers=L["training_libritts_speakers"], audio=aud("libritts_testclean"),
        text=ta("libritts_testclean_all"))
    P = ids["psc_test"]
    ex_audio = {h["eval"] for h in _hits(audio, "psc_test_expresso") if h["confirmed"]}
    vox_dup = P["voxceleb"]["rows_same_video_and_segment_in_nvtts"]
    sets["paraspeechcaps_test"] = dict(
        n=246,
        voxceleb=dict(n=P["voxceleb"]["n"], speaker_overlap_nvtts=P["voxceleb"]["speaker_overlap_nvtts"],
                      rows_with_speaker_in_training=P["voxceleb"]["rows_with_speaker_in_nvtts"],
                      rows_with_youtube_video_in_training=P["voxceleb"]["rows_with_video_in_nvtts"],
                      rows_same_segment_in_training=vox_dup,
                      same_segment_examples=P["voxceleb"]["same_segment_examples"],
                      audio="not available locally (VoxCeleb); same video+segment id+identical transcript used",
                      text=ta("psc_test_voxceleb_transcript")),
        expresso=dict(n=P["expresso"]["n"], speaker_overlap=P["expresso"]["speaker_overlap"],
                      read_rows=P["expresso"]["read_rows"], conversational_rows=P["expresso"]["conversational_rows"],
                      conversational_same_speech_in_nvtts=P["expresso"]["conversational_rows_same_speech_in_nvtts_train"],
                      conversational_examples=P["expresso"]["conversational_examples"][:3],
                      audio_note="raw 48 kHz audio on disk only for the 23 read-speech rows (conversational shards not downloaded)",
                      clip_id_in_training=dict(expresso=P["expresso"]["clip_id_in_expresso_train"],
                                               nvtts=P["expresso"]["clip_id_in_nvtts_train"]),
                      audio=aud("psc_test_expresso"), text=ta("psc_test_expresso_transcript")),
        ears=dict(n=P["ears"]["n"], speaker_overlap=P["ears"]["speaker_overlap"],
                  clip_id_in_training=P["ears"]["clip_id_in_ears_train"],
                  source_file_in_earsff_training=P["ears"]["source_file_segmented_into_earsff_train"],
                  audio=aud("psc_test_ears"), text=ta("psc_test_ears_transcript")),
        descriptions_vs_prompt_adapter_captions=text["psc_test_descriptions_vs_adapter_captions"],
        contaminated=dict(
            expresso_read=max(P["expresso"]["clip_id_in_expresso_train"], len(ex_audio)),
            expresso_conversational=P["expresso"]["conversational_rows_same_speech_in_nvtts_train"],
            voxceleb=vox_dup, ears=0),
        existing_benchmark_leak_flag=int(pd.read_json(BENCH / "psc.jsonl", lines=True).leak.sum()))
    k = sets["paraspeechcaps_test"]["contaminated"]
    sets["paraspeechcaps_test"]["contaminated_rows"] = sum(k.values())
    sets["paraspeechcaps_test"]["clean_rows"] = 246 - sum(k.values())
    sets["seedtts_eval_test_en"] = dict(
        n=ids["seedtts_en"]["n"], speaker_overlap="n/a (Common Voice; no Common Voice in training)",
        audio_prompts=aud("seedtts_en_prompt"), audio_targets=aud("seedtts_en_target_gt"),
        text_target=ta("seedtts_en_target"), text_prompt=ta("seedtts_en_prompt"))
    sets["cv3_eval_zero_shot_en"] = dict(
        n=ids["cv3_zero_shot_en"]["n"], speaker_overlap="n/a (no comparable ids)", audio_prompts=aud("cv3_en_prompt"),
        text_target=ta("cv3_en_target"), text_prompt=ta("cv3_en_prompt"))
    sets["harness_eval_texts"] = dict(text=ta("harness_and_paper_lists"), files=text["harness_and_paper_lists"]["files"],
                                      vs_prompt_adapter_captions=text["eval_prompt_lists_vs_adapter_captions"])
    sets["ljspeech_test500_texts"] = dict(text=ta("ljspeech_test500"))
    sets["cremad_analysis"] = dict(text=ta("cremad_sentences"), speaker_overlap="0 (CREMA-D actors in no training corpus)",
                                   audio="not on disk; not fingerprinted")
    out = dict(
        date="2026-10-11", training=dict(meta="data/meta/controls.parquet", n_utts=410566,
                                         corpora=ids["training_corpora"],
                                         adapter_captions=text["adapter_caption_counts"]),
        method=dict(
            speaker="speaker / chapter / clip / VoxCeleb video+segment ids where comparable",
            audio=("spectral-peak landmark fingerprint (8 kHz, 16 ms hop, 10 peaks/s, fan-out 3, 23-bit hashes); "
                   "inverted index over all 410,566 training clips (superset of corpus/speaker/+-10%% duration "
                   "blocking); near-dup = >= %d time-aligned hashes and >= %.0f%% of the shorter clip's hashes"
                   % (thr["min_count"], 100 * thr["min_frac"])),
            text="exact, normalised (NFKC, lower-case, no punctuation/event tags) and 8-gram containment >= 50%",
            ids_anonymised="salted SHA-1 (8 hex)"),
        validation=val, sets=sets)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, default=int)
    print(json.dumps({k: v for k, v in out["sets"].items()}, indent=1, default=int)[:6000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--nproc", type=int, default=12)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    dict(ids=stage_ids, text=stage_text, report=stage_report, audio=stage_audio,
         fp_train=lambda: stage_fp_train(a.nproc), fp_eval=lambda: stage_fp_eval(a.nproc),
         validate=lambda: stage_validate(a.nproc))[a.stage]()


if __name__ == "__main__":
    main()
