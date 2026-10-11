"""Clean held-out evaluation of the prompt adapter (language -> tag, which dims, z values).

The adapter's own validation split (steer.prompt.train) is a random 2% of caption PAIRS: the same clips, speakers and
caption templates appear on both sides, so its numbers are inflated. Here:

  split     held-out speakers (sha1(speaker) mod 100 < 15 within every corpus; Expresso: the one speaker with the
            smallest hash) and held-out PHRASES: the last paraphrase of every caption-generator phrase list with >= 2
            (dims) / >= 3 (tags) entries, one grade adverb per intensity grade, three sentence frames, and the last
            Cartesia emotion / speed / volume paraphrase, one neutral phrase and one lead word.
  retrain   steer.prompt's exact recipe (6 captions per clip + Cartesia instructions x3 + ParaSpeechCaps train_base,
            12 epochs, AdamW 2e-3, batch 1024) on training speakers and training phrases only
            -> runs/paper/prompt_adapter_heldout.pt (data/meta/prompt_adapter.pt is never touched).
  test sets T_tpl_seen   held-out speakers, captions from TRAINING phrases (speaker generalisation only)
            T_tpl_new    held-out speakers, captions from HELD-OUT phrases only
            T_cartesia   held-out Cartesia voices, instructions from held-out paraphrases; gold dims = the requested
                         speed / volume change (sign from the request), values = measured z
            T_psc        ParaSpeechCaps dev/test/holdout descriptions (never used in training) of EARS/Expresso clips
                         in META; gold dims rate & f0_med, values = measured z, plus PSC's categorical rate label
            T_hand       59 hand-written directions (steer.paper.adapter_gold, gold written before any model run)
            random_val   the recipe's own random 2% pair split (to show the inflation)
  systems   prod (data/meta/prompt_adapter.pt; saw every speaker and phrase -> only T_hand is clean for it),
            heldout (retrained), lexicon (keywords mined automatically from the training pairs), knn1 / knn10
            (nearest training captions in MiniLM space, copy their targets).

python -m steer.paper.adapter_eval [--skip_train]   (CPU)  -> paper/results/adapter_*.json
"""
import argparse
import contextlib
import copy
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .. import captions as C
from .. import cartesia_gen as CG
from ..config import CONTROL_DIMS, META, RAW
from ..prompt import Adapter, encoder
from .common import RES, ROOT

N = len(CONTROL_DIMS)
DI = {d: i for i, d in enumerate(CONTROL_DIMS)}
OUT = ROOT / "runs" / "paper"
CKPT = OUT / "prompt_adapter_heldout.pt"
torch.set_num_threads(20)


# ------------------------------------------------------------------------------------------------------------ split
def h100(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % 100


def heldout_speakers(df):
    spk = df[["corpus", "speaker"]].drop_duplicates()
    held = {s for c, s in zip(spk.corpus, spk.speaker) if c != "expresso" and h100(f"{c}/{s}") < 15}
    ex = sorted(spk[spk.corpus == "expresso"].speaker, key=lambda s: h100(f"expresso/{s}"))
    held.add(ex[0])
    return held


def split_lists(lst, min_len):
    """(train, held): the last entry is held out when the list is long enough."""
    if len(lst) >= min_len:
        return lst[:-1], lst[-1:]
    return lst, []


ORIG = dict(DIM=copy.deepcopy(C.DIM_PHRASES), TAG=copy.deepcopy(C.TAG_PHRASES), GRADE=copy.deepcopy(C.GRADE),
            FI=list(C.FRAMES_IMP), FD=list(C.FRAMES_DESC), EMO=copy.deepcopy(CG.EMO_WORDS),
            SPD=copy.deepcopy(CG.SPEED_WORDS), VOL=copy.deepcopy(CG.VOL_WORDS))
NEUTRAL = ["say it normally", "in a neutral voice", "plainly, no particular emotion", "just read it"]
LEADS = ["", "say it ", "speak ", "read this ", "deliver it "]
HELD_FRAMES_IMP = ["Can you say it {x}?", "Voice: {x}."]
HELD_FRAMES_DESC = ["Someone saying this {x}."]
HELD_GRADE = {0.8: "somewhat", 1.6: "clearly", 2.6: "far"}


def phrase_sets():
    tr, he = {}, {}
    tr["DIM"], he["DIM"] = {}, {}
    for d, (pos, neg) in ORIG["DIM"].items():
        p0, p1 = split_lists(pos, 2); n0, n1 = split_lists(neg, 2)
        tr["DIM"][d], he["DIM"][d] = (p0, n0), (p1, n1)
    tr["TAG"], he["TAG"] = {}, {}
    for t, l in ORIG["TAG"].items():
        a, b = split_lists(l, 3)
        tr["TAG"][t] = a
        if b:
            he["TAG"][t] = b
    tr["GRADE"] = [(th, [w for w in ws if w != HELD_GRADE[th]]) for th, ws in ORIG["GRADE"]]
    he["GRADE"] = [(th, [HELD_GRADE[th]]) for th, _ in ORIG["GRADE"]]
    tr["FI"] = [f for f in ORIG["FI"] if f not in HELD_FRAMES_IMP]; he["FI"] = HELD_FRAMES_IMP
    tr["FD"] = [f for f in ORIG["FD"] if f not in HELD_FRAMES_DESC]; he["FD"] = HELD_FRAMES_DESC
    for k in ("EMO", "SPD", "VOL"):
        tr[k], he[k] = {}, {}
        for e, l in ORIG[k].items():
            a, b = split_lists(l, 2)
            tr[k][e], he[k][e] = a, b
    tr["NEU"], he["NEU"] = NEUTRAL[:-1], NEUTRAL[-1:]
    tr["LEAD"], he["LEAD"] = LEADS[:-1], LEADS[-1:]
    return tr, he


@contextlib.contextmanager
def phrases(ps):
    old = (C.DIM_PHRASES, C.TAG_PHRASES, C.GRADE, C.FRAMES_IMP, C.FRAMES_DESC)
    C.DIM_PHRASES, C.TAG_PHRASES, C.GRADE, C.FRAMES_IMP, C.FRAMES_DESC = ps["DIM"], ps["TAG"], ps["GRADE"], ps["FI"], ps["FD"]
    try:
        yield
    finally:
        C.DIM_PHRASES, C.TAG_PHRASES, C.GRADE, C.FRAMES_IMP, C.FRAMES_DESC = old


def instruction(ps, emotion, speed, volume, rng):
    """cartesia_gen.instruction with the phrase lists of one side of the split."""
    parts = []
    if emotion != "neutral":
        parts.append(rng.choice(ps["EMO"][emotion]))
    if speed != 1.0:
        parts.append(rng.choice(ps["SPD"][speed]))
    if volume != 1.0:
        parts.append(rng.choice(ps["VOL"][volume]))
    if not parts:
        return rng.choice(ps["NEU"])
    return (rng.choice(ps["LEAD"]) + ", ".join(parts)).strip()


# ------------------------------------------------------------------------------------------------------------ pairs
ZC = [f"z_{k}" for k in CONTROL_DIMS]


def template_pairs(df, ps, per_utt, seed):
    random.seed(seed)
    Z = df[ZC].to_numpy(np.float32)
    rows = []
    with phrases(ps):
        for i, tag in enumerate(df.tag.to_numpy()):
            for _ in range(per_utt):
                txt, used_tag, dims = C.caption(tag, Z[i])
                rows.append(dict(text=txt, tag=used_tag, dims=dims, z=Z[i], sign={d: int(np.sign(Z[i][DI[d]])) for d in dims},
                                 corpus=df.corpus.iat[i]))
    return rows


def cartesia_pairs(df, ps, n_per, seed):
    rng = random.Random(seed)
    zi = dict(zip(df.id, df[ZC].to_numpy(np.float32)))
    tag = dict(zip(df.id, df.tag))
    rows = []
    for l in open(META / "manifest_cartesia.jsonl"):
        r = json.loads(l)
        if r["id"] not in zi:
            continue
        dims = (["rate"] if r["speed"] != 1.0 else []) + (["loud"] if r["volume"] != 1.0 else [])
        sign = {}
        if r["speed"] != 1.0:
            sign["rate"] = 1 if r["speed"] > 1 else -1
        if r["volume"] != 1.0:
            sign["loud"] = 1 if r["volume"] > 1 else -1
        for _ in range(n_per):
            rows.append(dict(text=instruction(ps, r["emotion"], r["speed"], r["volume"], rng), tag=tag[r["id"]], dims=dims,
                             z=zi[r["id"]], sign=sign, corpus="cartesia"))
    return rows


def psc_uid(src, path):
    from pathlib import Path
    p = Path(path)
    return f"expresso_{p.stem}" if src == "expresso" else f"ears_{p.parent.name}_{p.stem}"


def psc_pairs(df, splits, all_desc=False):
    zi = dict(zip(df.id, df[ZC].to_numpy(np.float32)))
    tag = dict(zip(df.id, df.tag))
    rows = []
    for s in splits:
        f = RAW / "paraspeechcaps" / "data" / f"{s}-00000-of-00001.parquet"
        t = pd.read_parquet(f, columns=["source", "relative_audio_path", "text_description", "speaking_rate", "pitch"])
        t = t[t.source.isin(["ears", "expresso"])]
        for src, path, desc, sr, pi in zip(t.source, t.relative_audio_path, t.text_description, t.speaking_rate, t.pitch):
            uid = psc_uid(src, path)
            if uid not in zi:
                continue
            for d in (desc if all_desc else desc[:1]):
                rows.append(dict(text=str(d).strip(), tag=tag[uid], dims=["f0_med", "rate"], z=zi[uid], sign={},
                                 corpus=src, psc_rate=sr, psc_pitch=pi, split=s, uid=uid))
    return rows


# ------------------------------------------------------------------------------------------------------------ train
def embed(texts, cache):
    new = sorted(set(t for t in texts if t not in cache))
    if new:
        t0 = time.time()
        E = encoder("cpu").encode(new, batch_size=512, convert_to_numpy=True, normalize_embeddings=True,
                                  show_progress_bar=False)
        cache.update(zip(new, E.astype(np.float32)))
        print(f"embedded {len(new)} texts in {time.time() - t0:.0f}s", flush=True)
    return np.stack([cache[t] for t in texts])


def targets(rows, t2i):
    S = torch.zeros(len(rows), N); V = torch.zeros(len(rows), N)
    for j, r in enumerate(rows):
        for d in r["dims"]:
            k = DI[d]
            if np.isfinite(r["z"][k]):
                S[j, k] = 1; V[j, k] = float(r["z"][k])
    Yt = torch.tensor([t2i.get(r["tag"], 0) for r in rows])
    return Yt, S, V


def train(rows, X, tags, seed=0):
    torch.manual_seed(seed)
    t2i = {t: i for i, t in enumerate(tags)}
    Yt, S, V = targets(rows, t2i)
    X = torch.from_numpy(X)
    m = Adapter(len(tags))
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    perm = torch.randperm(len(rows)); n_val = len(rows) // 50
    va, tr = perm[:n_val], perm[n_val:]
    for ep in range(12):
        m.train(); t0 = time.time()
        for b in tr[torch.randperm(len(tr))].split(1024):
            lt, ls, lv = m(X[b])
            s, v = S[b], V[b]
            loss = F.cross_entropy(lt, Yt[b]) + F.binary_cross_entropy_with_logits(ls, s) + \
                (F.smooth_l1_loss(lv, v, reduction="none") * s).sum() / s.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            lt, ls, lv = m(X[va])
            acc = (lt.argmax(-1) == Yt[va]).float().mean()
            sp = ((torch.sigmoid(ls) > 0.5) == (S[va] > 0)).float().mean()
            mae = ((lv - V[va]).abs() * S[va]).sum() / S[va].sum()
        print(f"epoch {ep} tag acc {acc:.3f} | spec acc {sp:.3f} | value MAE {mae:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return m, va.numpy(), dict(tag_acc=round(float(acc), 4), spec_acc=round(float(sp), 4), value_mae=round(float(mae), 4))


# ------------------------------------------------------------------------------------------------------------ systems
class AdapterSys:
    def __init__(self, m, tags):
        self.m, self.tags = m.eval(), tags

    @torch.no_grad()
    def __call__(self, E):
        lt, ls, lv = self.m(torch.from_numpy(E))
        return dict(spec=torch.sigmoid(ls).numpy(), val=lv.numpy(), tagp=torch.softmax(lt, -1).numpy())


STOP = set("a an the and or of to in on at for with as it this that is be your you speak say read talk voice like "
           "please can its just are was it's i me my we our".split())


def words(t):
    return [w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", t.lower()) if w not in STOP]


class Lexicon:
    """Keyword lexicon mined from the training pairs: word w -> (dim, sign) if >= 60% of the >= 30 training pairs
    containing w specify that dim with that sign (value = their mean z); w -> tag likewise (tag != none)."""

    def __init__(self, rows, tags, max_rows=600000, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), min(max_rows, len(rows)), replace=False)
        cnt = Counter(); dim = defaultdict(Counter); val = defaultdict(list); tg = defaultdict(Counter)
        for j in idx:
            r = rows[j]
            ws = set(words(r["text"]))
            for w in ws:
                cnt[w] += 1
                tg[w][r["tag"]] += 1
                for d in r["dims"]:
                    k = DI[d]
                    if np.isfinite(r["z"][k]):
                        sg = 1 if r["z"][k] > 0 else -1
                        dim[w][(d, sg)] += 1; val[(w, d, sg)].append(float(r["z"][k]))
        self.dkw, self.tkw, self.tags = {}, {}, tags
        for w, n in cnt.items():
            if n < 30:
                continue
            (d, sg), c = dim[w].most_common(1)[0] if dim[w] else ((None, 0), 0)
            if d and c / n >= 0.6:
                self.dkw[w] = (d, sg, float(np.mean(val[(w, d, sg)])))
            t, c = tg[w].most_common(1)[0]
            if t != "none" and c / n >= 0.6:
                self.tkw[w] = (t, c / n)
        self.ti = {t: i for i, t in enumerate(tags)}
        print(f"lexicon: {len(self.dkw)} dim keywords, {len(self.tkw)} tag keywords", flush=True)

    def predict(self, texts):
        n = len(texts)
        spec, val, tagp = np.zeros((n, N)), np.zeros((n, N)), np.zeros((n, len(self.tags)))
        for j, t in enumerate(texts):
            acc = defaultdict(list)
            for w in words(t):
                if w in self.dkw:
                    d, sg, v = self.dkw[w]; acc[d].append(v)
                if w in self.tkw:
                    tt, p = self.tkw[w]; tagp[j, self.ti[tt]] = max(tagp[j, self.ti[tt]], p)
            for d, vs in acc.items():
                spec[j, DI[d]] = 1; val[j, DI[d]] = np.mean(vs)
            if tagp[j].sum() == 0:
                tagp[j, 0] = 1  # none
        return dict(spec=spec, val=val, tagp=tagp)


class KNN:
    """Nearest training captions (unique texts, aggregated targets) in MiniLM space."""

    def __init__(self, rows, emb_cache, tags, k):
        t2i = {t: i for i, t in enumerate(tags)}
        agg = {}
        for r in rows:
            a = agg.setdefault(r["text"], [0, np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(len(tags))])
            a[0] += 1
            for d in r["dims"]:
                kk = DI[d]
                if np.isfinite(r["z"][kk]):
                    a[1][kk] += 1; a[2][kk] += r["z"][kk]
            a[4][t2i.get(r["tag"], 0)] += 1
        self.texts = list(agg)
        self.E = torch.from_numpy(np.stack([emb_cache[t] for t in self.texts]))
        n = np.array([agg[t][0] for t in self.texts], float)[:, None]
        cs = np.stack([agg[t][1] for t in self.texts])
        self.spec = cs / n
        self.vsum, self.vcnt = np.stack([agg[t][2] for t in self.texts]), cs
        self.tag = np.stack([agg[t][4] for t in self.texts]) / n
        self.k = k
        print(f"knn{k}: {len(self.texts)} unique training captions", flush=True)

    def predict(self, Q):
        out_s, out_v, out_t = [], [], []
        for b in torch.from_numpy(Q).split(256):
            nn_ = (b @ self.E.T).topk(self.k, dim=1).indices.numpy()
            out_s.append(self.spec[nn_].mean(1))
            out_v.append(self.vsum[nn_].sum(1) / np.maximum(self.vcnt[nn_].sum(1), 1))
            out_t.append(self.tag[nn_].mean(1))
        return dict(spec=np.concatenate(out_s), val=np.concatenate(out_v), tagp=np.concatenate(out_t))


# ------------------------------------------------------------------------------------------------------------ metrics
def boot(fn, n_items, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n_items, n_items)) for _ in range(B)]
    vals = [v for v in vals if v == v]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else None


def score(pred, gold, tags, thr=0.5):
    """gold: dict(G (n,N) bool gold-specified, SG (n,N) sign, GV (n,N) value or nan, OK (n,N) bool ignored dims,
    TAGS list of acceptable-tag lists ([] = not scored))."""
    G, SG, GV, OK = gold["G"], gold["SG"], gold["GV"], gold["OK"]
    P = pred["spec"] > thr
    V = pred["val"]
    n = len(G)
    scored = G | ~OK  # dims that count for F1
    res = {}

    def f1(idx, dim=None):
        g, p, s = G[idx], P[idx], scored[idx]
        if dim is not None:
            g, p, s = g[:, dim], p[:, dim], s[:, dim]
        tp = (g & p & s).sum(); fp = (~g & p & s).sum(); fn = (g & ~p).sum()
        return 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else float("nan")

    allidx = np.arange(n)
    res["n"] = int(n)
    res["n_gold_dims"] = int(G.sum())
    res["spec_f1_micro"] = round(float(f1(allidx)), 4)
    res["spec_f1_micro_ci"] = boot(f1, n)
    per = {}
    for k, d in enumerate(CONTROL_DIMS):
        if G[:, k].sum() >= 10:
            per[d] = dict(f1=round(float(f1(allidx, k)), 3), n_gold=int(G[:, k].sum()))
    res["spec_f1_per_dim"] = per
    res["spec_f1_macro"] = round(float(np.mean([v["f1"] for v in per.values()])), 4) if per else None
    res["fp_dims_per_item"] = round(float((P & ~G & scored).sum() / n), 4)
    hasSign = G & (SG != 0)
    both = hasSign & P
    res["dir_acc"] = round(float((np.sign(V[both]) == SG[both]).mean()), 4) if both.any() else None

    def hit(idx):
        h, b = hasSign[idx], (hasSign & P & (np.sign(V) == SG))[idx]
        return b.sum() / max(h.sum(), 1)

    res["hit"] = round(float(hit(allidx)), 4) if hasSign.any() else None  # specified AND right sign / gold dims
    res["hit_ci"] = boot(hit, n) if hasSign.any() else None
    vm = G & P & np.isfinite(GV)
    if vm.sum() >= 10:
        a, b = V[vm], GV[vm]
        res["value_mae"] = round(float(np.abs(a - b).mean()), 4)
        res["value_r"] = round(float(np.corrcoef(a, b)[0, 1]), 4)
        res["value_mae_zero_baseline"] = round(float(np.abs(b).mean()), 4)  # predicting z = 0
        res["value_n"] = int(vm.sum())
        pr = {}
        for k, d in enumerate(CONTROL_DIMS):
            mk = vm[:, k]
            if mk.sum() >= 20:
                pr[d] = dict(mae=round(float(np.abs(V[mk, k] - GV[mk, k]).mean()), 3),
                             r=round(float(np.corrcoef(V[mk, k], GV[mk, k])[0, 1]), 3), n=int(mk.sum()))
        res["value_per_dim"] = pr
    T = gold["TAGS"]
    ti = [j for j in range(n) if T[j]]
    if ti:
        rank = np.argsort(-pred["tagp"][ti], 1)[:, :3]
        top1 = np.array([tags[rank[q, 0]] in T[j] for q, j in enumerate(ti)])
        top3 = np.array([any(tags[x] in T[j] for x in rank[q]) for q, j in enumerate(ti)])
        res["tag_n"] = len(ti)
        res["tag_top1"] = round(float(top1.mean()), 4)
        res["tag_top1_ci"] = boot(lambda i: top1[i].mean(), len(ti))
        res["tag_top3"] = round(float(top3.mean()), 4)
    return res


def gold_from_rows(rows, tag_scored=True):
    n = len(rows)
    G = np.zeros((n, N), bool); SG = np.zeros((n, N)); GV = np.full((n, N), np.nan)
    for j, r in enumerate(rows):
        for d in r["dims"]:
            k = DI[d]
            if not np.isfinite(r["z"][k]):
                continue
            G[j, k] = True; GV[j, k] = r["z"][k]
            SG[j, k] = r["sign"].get(d, 0)
    TAGS = [[r["tag"]] if (tag_scored and r["tag"] not in ("none",)) else [] for r in rows]
    return dict(G=G, SG=SG, GV=GV, OK=np.zeros((n, N), bool), TAGS=TAGS)


def gold_hand(items):
    n = len(items)
    G = np.zeros((n, N), bool); SG = np.zeros((n, N)); OK = np.zeros((n, N), bool)
    for j, it in enumerate(items):
        for d, s in it["spec"].items():
            G[j, DI[d]] = True; SG[j, DI[d]] = s
        if it["ok"] == "*":
            OK[j] = True
        else:
            for d in it["ok"]:
                OK[j, DI[d]] = True
    return dict(G=G, SG=SG, GV=np.full((n, N), np.nan), OK=OK, TAGS=[it["tags"] for it in items])


# ------------------------------------------------------------------------------------------------------------ main
def env_eval(gold, emb):
    """Production env adapter on the hand-written environment / event prompts."""
    try:
        from ..acoustic import ROOMS, SCENES
        from ..text import EVENT_TOKENS
        ce = torch.load(META / "prompt_adapter_env.pt", map_location="cpu")
        me = Adapter(len(ce["tags"]), env=True); me.load_state_dict(ce["state"]); me.eval()
        env_items = gold["env_items"]
        with torch.no_grad():
            out = me(torch.from_numpy(emb([it["prompt"] for it in env_items])))[3]
        pick = lambda k, names, j: (lambda i: names[i - 1] if i > 0 else None)(int(out[k][j].argmax()))
        er = []
        for j, it in enumerate(env_items):
            pr = dict(scene=pick("scene", SCENES, j), room=pick("room", ROOMS, j), phone=bool(out["phone"][j, 0] > 0),
                      ev_start=pick("ev_start", EVENT_TOKENS, j), ev_end=pick("ev_end", EVENT_TOKENS, j))
            ok = all((pr[k] in v) if isinstance(v, list) else (pr[k] == v) for k, v in it["env"].items())
            er.append(dict(id=it["id"], prompt=it["prompt"], gold=it["env"], pred=pr, correct=ok))
        json.dump(dict(n=len(er), acc=round(float(np.mean([e["correct"] for e in er])), 4), items=er,
                       note="production env adapter (prompt_adapter_env.pt); its training phrases were not held out, "
                            "but these prompts were written independently"),
                  open(RES / "adapter_env.json", "w"), indent=1)
        print("env acc", np.mean([e["correct"] for e in er]), flush=True)
    except Exception as ex:  # noqa: BLE001
        print("env eval failed:", ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--per_utt", type=int, default=6)
    ap.add_argument("--env_only", action="store_true")
    a = ap.parse_args()
    if a.env_only:
        gold = json.load(open(RES / "adapter_handwritten_gold.json"))
        cache = {}
        embed([it["prompt"] for it in gold["env_items"]], cache)
        env_eval(gold, lambda ts: np.stack([cache[t] for t in ts]).astype(np.float32))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    from ..data import load_meta
    df, stats = load_meta()
    tags = stats["tags"]
    held = heldout_speakers(df)
    is_held = df.speaker.isin(held)
    dtr, dte = df[~is_held].reset_index(drop=True), df[is_held].reset_index(drop=True)
    ps_tr, ps_he = phrase_sets()
    split_info = dict(rule="sha1(corpus/speaker) mod 100 < 15; expresso: 1 of 4 speakers", per_corpus={
        c: dict(train_spk=int(dtr[dtr.corpus == c].speaker.nunique()), heldout_spk=int(dte[dte.corpus == c].speaker.nunique()),
                train_utt=int((dtr.corpus == c).sum()), heldout_utt=int((dte.corpus == c).sum())) for c in sorted(df.corpus.unique())},
        heldout_phrases=dict(dim=sum(len(p) + len(q) for p, q in ps_he["DIM"].values()),
                             dim_train=sum(len(p) + len(q) for p, q in ps_tr["DIM"].values()),
                             tag=sum(len(v) for v in ps_he["TAG"].values()), tag_train=sum(len(v) for v in ps_tr["TAG"].values()),
                             grade=[w for _, (w,) in ps_he["GRADE"]], frames=ps_he["FI"] + ps_he["FD"],
                             cartesia=sum(len(v) for k in ("EMO", "SPD", "VOL") for v in ps_he[k].values()),
                             cartesia_train=sum(len(v) for k in ("EMO", "SPD", "VOL") for v in ps_tr[k].values())))
    # private: speaker ids stay out of the paper dir
    json.dump(sorted(held), open(OUT / "adapter_heldout_speakers.json", "w"))
    print(json.dumps(split_info, indent=1), flush=True)

    cache_f = OUT / "adapter_emb_cache.npz"
    cache = {}
    if cache_f.exists():
        z = np.load(cache_f, allow_pickle=True)
        cache = dict(zip(z["t"].tolist(), z["e"]))
        print("emb cache", len(cache), flush=True)

    # ---- training pairs (clean recipe)
    t0 = time.time()
    rows = template_pairs(dtr, ps_tr, a.per_utt, seed=0) + cartesia_pairs(dtr, ps_tr, 3, seed=0) + \
        psc_pairs(dtr, ["train_base"])
    rows = [r for r in rows if r["text"]]
    print(f"{len(rows)} training pairs, {len(set(r['text'] for r in rows))} unique ({time.time() - t0:.0f}s)", flush=True)
    X = embed([r["text"] for r in rows], cache)
    np.savez(cache_f, t=np.array(list(cache), dtype=object), e=np.stack(list(cache.values())))
    if a.skip_train and CKPT.exists():
        ck = torch.load(CKPT, map_location="cpu")
        m = Adapter(len(tags)); m.load_state_dict(ck["state"]); va = np.array(ck["val_idx"]); rv = ck["recipe_val"]
    else:
        m, va, rv = train(rows, X, tags)
        torch.save(dict(state=m.state_dict(), tags=tags, env=False, val_idx=va.tolist(), recipe_val=rv,
                        note="held-out-speaker / held-out-phrase retrain for the paper; NOT the production adapter"), CKPT)
        print("saved", CKPT, flush=True)
    # the baselines and the held-out adapter all see exactly the pairs not in the recipe's random val split
    trmask = np.ones(len(rows), bool); trmask[va] = False
    rows_tr = [r for r, k in zip(rows, trmask) if k]

    # ---- test sets
    rng = np.random.default_rng(1)
    samp = []
    for c in sorted(dte.corpus.unique()):
        ix = np.where(dte.corpus == c)[0]
        samp.append(dte.iloc[rng.choice(ix, min(1500, len(ix)), replace=False)])
    dsamp = pd.concat(samp).reset_index(drop=True)
    sets = {}
    sets["random_val"] = [rows[i] for i in va[:8000]]
    sets["T_tpl_seen"] = [r for r in template_pairs(dsamp, ps_tr, 1, seed=11) if r["text"]]
    sets["T_tpl_new"] = [r for r in template_pairs(dsamp, ps_he, 1, seed=12) if r["text"]]
    car = cartesia_pairs(dte, ps_he, 1, seed=13)
    sets["T_cartesia"] = [car[i] for i in np.random.default_rng(2).permutation(len(car))[:4000]]
    psc_all = psc_pairs(df, ["dev", "test", "holdout"])
    sets["T_psc"] = psc_all
    id2spk = dict(zip(df.id, df.speaker))
    sets["T_psc_heldspk"] = [r for r in psc_all if id2spk[r["uid"]] in held]
    gold = json.load(open(RES / "adapter_handwritten_gold.json"))
    items = gold["items"]

    all_texts = sorted({r["text"] for s in sets.values() for r in s} | {it["prompt"] for it in items} |
                       {it["prompt"] for it in gold["env_items"]} |
                       {p for pr in gold["probes"] for p in pr["prompts"]})
    embed(all_texts, cache)
    np.savez(cache_f, t=np.array(list(cache), dtype=object), e=np.stack(list(cache.values())))
    emb = lambda ts: np.stack([cache[t] for t in ts]).astype(np.float32)

    ck = torch.load(META / "prompt_adapter.pt", map_location="cpu")
    prod = Adapter(len(ck["tags"])); prod.load_state_dict(ck["state"])
    assert ck["tags"] == tags
    systems = dict(prod=AdapterSys(prod, tags), heldout=AdapterSys(m, tags))
    lex = Lexicon(rows_tr, tags)
    knn1, knn10 = KNN(rows_tr, cache, tags, 1), KNN(rows_tr, cache, tags, 10)

    def run(sysname, texts):
        if sysname in systems:
            return systems[sysname](emb(texts))
        if sysname == "lexicon":
            return lex.predict(texts)
        return (knn1 if sysname == "knn1" else knn10).predict(emb(texts))

    SYS = ["prod", "heldout", "lexicon", "knn1", "knn10"]
    main_res = dict(split=split_info, recipe_val_heldout_adapter=rv, sets={})
    for name, rs in sets.items():
        if not rs:
            continue
        g = gold_from_rows(rs)
        if name == "T_cartesia":
            pass
        if name.startswith("T_psc"):
            g["SG"][:] = 0  # PSC descriptions are absolute-register: no per-clip sign gold
        texts = [r["text"] for r in rs]
        main_res["sets"][name] = dict(n=len(rs), n_unique_text=len(set(texts)),
                                      by_corpus=dict(Counter(r["corpus"] for r in rs)), systems={})
        for s in SYS:
            if name == "random_val" and s in ("knn1", "knn10", "lexicon", "prod"):
                continue
            p = run(s, texts)
            sc = score(p, g, tags)
            if name.startswith("T_psc"):
                # rate value vs PSC categorical rate label (Spearman over slow < measured < fast)
                from scipy.stats import spearmanr
                o = {"slow speed": 0, "measured speed": 1, "fast speed": 2}
                ok = [j for j, r in enumerate(rs) if r["psc_rate"] in o]
                sc["rate_vs_psc_label_spearman"] = round(float(spearmanr([p["val"][j, DI["rate"]] for j in ok],
                                                                         [o[rs[j]["psc_rate"]] for j in ok])[0]), 4)
                sc["rate_spec_rate"] = round(float((p["spec"][:, DI["rate"]] > 0.5).mean()), 4)
                sc["f0_spec_rate"] = round(float((p["spec"][:, DI["f0_med"]] > 0.5).mean()), 4)
                sc["measured_z_rate_vs_psc_label_spearman"] = round(float(spearmanr(
                    [rs[j]["z"][DI["rate"]] for j in ok], [o[rs[j]["psc_rate"]] for j in ok])[0]), 4)
            main_res["sets"][name]["systems"][s] = sc
            print(name, s, {k: v for k, v in sc.items() if not isinstance(v, dict)}, flush=True)

    # ---- hand-written set
    texts = [it["prompt"] for it in items]
    gh = gold_hand(items)
    hand = dict(n=len(items), systems={}, by_cat={}, negation={}, probes={})
    preds = {}
    for s in SYS:
        p = run(s, texts)
        preds[s] = p
        hand["systems"][s] = score(p, gh, tags)
        for cat in sorted({it["cat"] for it in items}):
            ix = [j for j, it in enumerate(items) if it["cat"] == cat]
            sub = {k: v[ix] for k, v in gh.items() if k != "TAGS"}
            sub["TAGS"] = [gh["TAGS"][j] for j in ix]
            hand["by_cat"].setdefault(cat, {})[s] = score({k: v[ix] for k, v in p.items()}, sub, tags)
        # negation
        npass = []
        for j, it in enumerate(items):
            if it["cat"] != "neg":
                continue
            ok = True
            for d, sg in it.get("neg", {}).items():
                k = DI[d]
                if p["spec"][j, k] > 0.5 and np.sign(p["val"][j, k]) == sg:
                    ok = False
            if it.get("neg_tags") and tags[int(np.argmax(p["tagp"][j]))] in it["neg_tags"]:
                ok = False
            npass.append(ok)
        hand["negation"][s] = dict(pass_rate=round(float(np.mean(npass)), 4), n=len(npass), passes=npass)
        # intensity probes
        mono, signok, triples = 0, 0, []
        for pr in gold["probes"]:
            q = run(s, pr["prompts"])
            k = DI[pr["dim"]]
            v = q["val"][:, k] * pr["sign"]
            sp = q["spec"][:, k]
            is_mono = bool(v[0] < v[1] < v[2])
            mono += is_mono; signok += bool((v > 0).all() and (sp > 0.5).all())
            triples.append(dict(dim=pr["dim"], sign=pr["sign"], prompts=pr["prompts"], signed_val=np.round(v, 3).tolist(),
                                spec=np.round(sp, 3).tolist(), monotone=is_mono))
        hand["probes"][s] = dict(n=len(gold["probes"]), monotone=mono, specified_right_sign_all3=signok, triples=triples)
        print("hand", s, {k: v for k, v in hand["systems"][s].items() if not isinstance(v, dict)},
              "neg", hand["negation"][s]["pass_rate"], "mono", mono, flush=True)
    # per-item predictions (failure analysis)
    per_item = []
    for j, it in enumerate(items):
        e = dict(id=it["id"], cat=it["cat"], prompt=it["prompt"], gold_spec=it["spec"], gold_tags=it["tags"])
        for s in ("prod", "heldout", "lexicon", "knn1"):
            p = preds[s]
            e[s] = dict(ctrl={CONTROL_DIMS[k]: round(float(p["val"][j, k]), 2) for k in range(N) if p["spec"][j, k] > 0.5},
                        top3=[(tags[i], round(float(p["tagp"][j, i]), 3)) for i in np.argsort(-p["tagp"][j])[:3]])
        per_item.append(e)
    json.dump(dict(main=None, items=per_item), open(RES / "adapter_handwritten_preds.json", "w"), indent=1)
    json.dump(hand, open(RES / "adapter_handwritten.json", "w"), indent=1)

    env_eval(gold, emb)

    json.dump(main_res, open(RES / "adapter_main.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
