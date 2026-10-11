"""Hand-written directions (adapter_handwritten_gold.json, 59 items + 24 probe triples; adapter_handwritten_gold_v2.json,
102 items incl. 8 intensity triples): every system's prediction, then the metrics.

  --stage predict  prod / heldout (runs/paper/prompt_adapter_heldout.pt) / notpl (same MiniLM adapter recipe trained
                   WITHOUT template captions: Cartesia instructions + ParaSpeechCaps only, training speakers and
                   phrasings) / knn1 / knn10 / lexicon  -> runs/paper/hand_preds.json
  --stage score    + the LLM caches (runs/paper/llm_*.json)  -> paper/results/adapter_hand.json

Metrics: mask precision / recall / F1 (micro over gold dims and non-"ok" dims), sign accuracy (gold & predicted),
hit (gold dims specified with the right sign), intensity ordering (weak < plain < strong in |value|, all specified
with the right sign), negation accuracy, composition exact match (all gold dims hit, no predicted dim outside
spec + ok), tag top-1 / top-3. 95% item-bootstrap CIs for the headline numbers.
"""
import argparse
import json

import numpy as np
import torch

from ..config import CONTROL_DIMS
from . import adapter_eval as AE
from .common import RES, WORK

N = len(CONTROL_DIMS)
DI = {d: i for i, d in enumerate(CONTROL_DIMS)}
PRED_F = WORK / "hand_preds.json"
LLMS = {"llm_qwen2.5_3b.json": "qwen2.5-3b"}


def load_gold():
    g1 = json.load(open(RES / "adapter_handwritten_gold.json"))
    g2 = json.load(open(RES / "adapter_handwritten_gold_v2.json"))
    items = [dict(it, set="v1") for it in g1["items"]] + [dict(it, set="v2") for it in g2["items"]]
    triples = [dict(dim=p["dim"], sign=p["sign"], prompts=p["prompts"], set="v1") for p in g1["probes"]]
    for grp in sorted({it["grp"] for it in g2["items"] if "grp" in it}):
        its = sorted([it for it in g2["items"] if it.get("grp") == grp], key=lambda x: x["lvl"])
        d, s = next(iter(its[0]["spec"].items()))
        triples.append(dict(dim=d, sign=s, prompts=[it["prompt"] for it in its], set="v2"))
    return items, triples


def predict():
    import pandas as pd  # noqa: F401
    from ..data import load_meta
    from ..prompt import Adapter
    from ..config import META
    items, triples = load_gold()
    texts = list(dict.fromkeys([it["prompt"] for it in items] + [p for t in triples for p in t["prompts"]]))
    df, stats = load_meta()
    tags = stats["tags"]
    held = AE.heldout_speakers(df)
    dtr = df[~df.speaker.isin(held)].reset_index(drop=True)
    ps_tr, _ = AE.phrase_sets()
    z = np.load(AE.OUT / "adapter_emb_cache.npz", allow_pickle=True)
    cache = dict(zip(z["t"].tolist(), z["e"]))
    print("cache", len(cache), flush=True)
    rows = AE.template_pairs(dtr, ps_tr, 6, seed=0) + AE.cartesia_pairs(dtr, ps_tr, 3, seed=0) + AE.psc_pairs(dtr, ["train_base"])
    rows = [r for r in rows if r["text"]]
    ck = torch.load(AE.CKPT, map_location="cpu")
    va = set(ck["val_idx"])
    rows_tr = [r for i, r in enumerate(rows) if i not in va]
    AE.embed(texts, cache)
    E = np.stack([cache[t] for t in texts]).astype(np.float32)
    out = {}

    def put(name, p):
        out[name] = {t: dict(spec=np.round(p["spec"][j], 4).tolist(), val=np.round(p["val"][j], 4).tolist(),
                             top3=[tags[i] for i in np.argsort(-p["tagp"][j])[:3]]) for j, t in enumerate(texts)}

    cp = torch.load(META / "prompt_adapter.pt", map_location="cpu")
    m = Adapter(len(tags)); m.load_state_dict(cp["state"]); put("prod", AE.AdapterSys(m, tags)(E))
    m = Adapter(len(tags)); m.load_state_dict(ck["state"]); put("heldout", AE.AdapterSys(m, tags)(E))
    # no-template ablation: same recipe on the real-instruction pairs only
    real = [r for r in rows if r["corpus"] in ("cartesia", "ears", "expresso") and not r["text"].startswith(tuple(ps_tr["FI"][:0]))]
    real = AE.cartesia_pairs(dtr, ps_tr, 3, seed=0) + AE.psc_pairs(dtr, ["train_base"])
    real = [r for r in real if r["text"]]
    Xr = AE.embed([r["text"] for r in real], cache)
    mn, _, rv = AE.train(real, Xr, tags)
    torch.save(dict(state=mn.state_dict(), tags=tags, recipe_val=rv, note="no-template ablation"), AE.OUT / "prompt_adapter_notpl.pt")
    put("notpl", AE.AdapterSys(mn, tags)(E))
    put("lexicon", AE.Lexicon(rows_tr, tags).predict(texts))
    put("knn1", AE.KNN(rows_tr, cache, tags, 1).predict(E))
    put("knn10", AE.KNN(rows_tr, cache, tags, 10).predict(E))
    json.dump(dict(n_train_pairs=len(rows_tr), n_notpl_pairs=len(real), preds=out), open(PRED_F, "w"))
    print("saved", PRED_F, flush=True)


# ------------------------------------------------------------------------------------------------------- scoring
def llm_preds(fname):
    c = json.load(open(WORK / fname))
    return {t: dict(spec=[1.0 if d in v["controls"] else 0.0 for d in CONTROL_DIMS],
                    val=[float(v["controls"].get(d, 0.0)) for d in CONTROL_DIMS], top3=v["tags"][:3],
                    parse_error=v.get("parse_error", False)) for t, v in c.items()}


def boot(vals, B=1000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    b = [np.nanmean(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(B)]
    return [round(float(np.nanpercentile(b, 2.5)), 4), round(float(np.nanpercentile(b, 97.5)), 4)]


def score_items(P, items, triples):
    """P: {text: dict(spec, val, top3)}; only items whose prompt is in P are scored."""
    items = [it for it in items if it["prompt"] in P]
    tp = fp = fn = 0
    signs, hits, per_dim = [], [], {d: [0, 0, 0] for d in CONTROL_DIMS}
    exact, neg, t1, t3 = [], [], [], []
    for it in items:
        p = P[it["prompt"]]
        spec = np.array(p["spec"]) > 0.5
        val = np.array(p["val"])
        gold = {DI[d]: s for d, s in it["spec"].items()}
        okd = set(range(N)) if it["ok"] == "*" else {DI[d] for d in it["ok"]}
        fp_dims = [k for k in range(N) if spec[k] and k not in gold and k not in okd]
        for k, s in gold.items():
            hit = bool(spec[k] and np.sign(val[k]) == s)
            hits.append(hit)
            if spec[k]:
                signs.append(np.sign(val[k]) == s); tp += 1; per_dim[CONTROL_DIMS[k]][0] += 1
            else:
                fn += 1; per_dim[CONTROL_DIMS[k]][2] += 1
        fp += len(fp_dims)
        for k in fp_dims:
            per_dim[CONTROL_DIMS[k]][1] += 1
        if it["cat"] in ("combo", "distractor") and gold:
            exact.append(all(spec[k] and np.sign(val[k]) == s for k, s in gold.items()) and not fp_dims)
        if it["cat"] == "neg":
            ok = all(not (spec[DI[d]] and np.sign(val[DI[d]]) == s) for d, s in it.get("neg", {}).items())
            if it.get("neg_tags") and p["top3"][0] in it["neg_tags"]:
                ok = False
            neg.append(ok)
        if it["tags"]:
            t1.append(p["top3"][0] in it["tags"]); t3.append(any(t in it["tags"] for t in p["top3"]))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    mono, order = [], []
    for t in triples:
        if not all(q in P for q in t["prompts"]):
            continue
        k = DI[t["dim"]]
        v = np.array([P[q]["val"][k] for q in t["prompts"]]) * t["sign"]
        sp = np.array([P[q]["spec"][k] for q in t["prompts"]]) > 0.5
        mono.append(bool(sp.all() and (v > 0).all() and v[0] < v[1] < v[2]))
        order.append(bool(sp.all() and (v > 0).all()))
    r = lambda x: round(float(x), 4)
    return dict(n_items=len(items), n_gold_dims=tp + fn, mask_precision=r(prec), mask_recall=r(rec),
                mask_f1=r(2 * prec * rec / max(prec + rec, 1e-9)), sign_acc=r(np.mean(signs)) if signs else None,
                hit=r(np.mean(hits)) if hits else None, hit_ci=boot(hits),
                composition_exact=r(np.mean(exact)) if exact else None, n_composition=len(exact), composition_ci=boot(exact),
                negation_acc=r(np.mean(neg)) if neg else None, n_negation=len(neg),
                intensity_monotone=r(np.mean(mono)) if mono else None, intensity_specified_right_sign=r(np.mean(order)) if order else None,
                n_triples=len(mono), tag_top1=r(np.mean(t1)) if t1 else None, tag_top1_ci=boot(t1),
                tag_top3=r(np.mean(t3)) if t3 else None, n_tag=len(t1),
                per_dim_f1={d: round(2 * a / max(2 * a + b + c, 1), 3) for d, (a, b, c) in per_dim.items() if a + c >= 4})


def score():
    items, triples = load_gold()
    pr = json.load(open(PRED_F))
    P = dict(pr["preds"])
    for f, name in LLMS.items():
        if (WORK / f).exists():
            P[name] = llm_preds(f)
    res = dict(n_items=dict(v1=sum(it["set"] == "v1" for it in items), v2=sum(it["set"] == "v2" for it in items)),
               n_triples=len(triples), systems={}, by_set={}, by_cat={},
               coverage={k: sum(it["prompt"] in v for it in items) for k, v in P.items()})
    for name, p in P.items():
        res["systems"][name] = score_items(p, items, triples)
        res["by_set"][name] = {s: score_items(p, [it for it in items if it["set"] == s], [t for t in triples if t["set"] == s])
                               for s in ("v1", "v2")}
        res["by_cat"][name] = {c: score_items(p, [it for it in items if it["cat"] == c], [])
                               for c in sorted({it["cat"] for it in items})}
        s = res["systems"][name]
        print(f"{name:12s} n={s['n_items']} P/R/F1 {s['mask_precision']}/{s['mask_recall']}/{s['mask_f1']} sign {s['sign_acc']} "
              f"hit {s['hit']} comp {s['composition_exact']} neg {s['negation_acc']} mono {s['intensity_monotone']} "
              f"tag {s['tag_top1']}/{s['tag_top3']}", flush=True)
    # per-item outputs for failure analysis (v2)
    res["items_v2"] = [dict(id=it["id"], cat=it["cat"], prompt=it["prompt"], gold=it["spec"], tags=it["tags"],
                            **{n: dict(ctrl={CONTROL_DIMS[k]: round(p[it["prompt"]]["val"][k], 2) for k in range(N)
                                             if p[it["prompt"]]["spec"][k] > 0.5}, top3=p[it["prompt"]]["top3"])
                               for n, p in P.items() if n in ("prod", "heldout", "knn10", "qwen2.5-3b") and it["prompt"] in p})
                       for it in items if it["set"] == "v2"]
    json.dump(res, open(RES / "adapter_hand.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    if a.stage in ("predict", "all"):
        predict()
    if a.stage in ("score", "all"):
        score()
