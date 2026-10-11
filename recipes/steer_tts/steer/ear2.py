"""In-domain emotion probe ("ear2"): trained on real EARS emotional speech, speaker-disjoint.

The CREMA-D probe (steer.ear) recognises real EARS emotions poorly (anger 0.37, sadness 0.11, disgust 0.0), and
off-the-shelf SER models (superb-er, xlsr-RAVDESS) collapse to one class on EARS. ear2 uses the same paired design
- WavLM-base+ mean hidden state (layers 4-9 averaged) of the clip minus the mean of the same speaker's neutral
clips - and a multinomial logistic probe over the EARS emotion tags, evaluated with speaker-disjoint 5-fold CV so
its per-tag recall is known before it is used as a judge.

python -m steer.ear2 embed     -> data/meta/ear2_emb.npz   (EARS read emotional + neutral clips)
python -m steer.ear2 train     -> data/meta/ear2.pt + CV report (per-tag recall)
"""
import sys

import numpy as np
import pandas as pd
import torch

from .config import META, ROOT

LAYERS = (4, 5, 6, 7, 8, 9)


def embedder(dev):
    from transformers import WavLMModel
    return WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(dev).eval()


@torch.no_grad()
def embed_wav(wl, y16, dev):
    h = wl(torch.as_tensor(y16, dtype=torch.float32)[None].to(dev), output_hidden_states=True).hidden_states
    return torch.stack([h[i][0].mean(0) for i in LAYERS]).mean(0).cpu().numpy()


def embed():
    import soundfile as sf
    import soxr
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(12)
    df = pd.read_parquet(META / "controls.parquet")
    e = df[(df.corpus == "ears") & ~df.id.str.contains("freeform")]
    e = e[e.id.str.contains("_emo_") | (e.tag == "neutral")]
    wl = embedder(dev)
    X = []
    for k, i in enumerate(e.id):
        y = sf.read(ROOT / "data" / "wav24" / "ears" / f"{i}.wav", dtype="float32")[0]
        X.append(embed_wav(wl, soxr.resample(y, 24000, 16000)[: 16000 * 10], dev))
        if k % 500 == 0:
            print(f"  {k}/{len(e)}", flush=True)
    np.savez(META / "ear2_emb.npz", X=np.stack(X), ids=e.id.to_numpy(), spk=e.speaker.to_numpy(), tag=e.tag.to_numpy())
    print("wrote", META / "ear2_emb.npz", len(e))


def train():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, recall_score
    z = np.load(META / "ear2_emb.npz", allow_pickle=True)
    X, spk, tag = z["X"], z["spk"], z["tag"]
    neu = {s: X[(spk == s) & (tag == "neutral")].mean(0) for s in np.unique(spk)}
    D = X - np.stack([neu[s] for s in spk])
    keep = tag != "neutral"
    D, spk, tag = D[keep], spk[keep], tag[keep]
    classes = sorted(set(tag))
    y = np.array([classes.index(t) for t in tag])
    us = np.unique(spk)
    fold = {s: i % 5 for i, s in enumerate(np.random.default_rng(0).permutation(us))}
    f = np.array([fold[s] for s in spk])
    mu, sd = D.mean(0), D.std(0) + 1e-6
    Dn = (D - mu) / sd
    pred = np.zeros_like(y)
    for k in range(5):
        clf = LogisticRegression(C=0.05, max_iter=3000)
        clf.fit(Dn[f != k], y[f != k])
        pred[f == k] = clf.predict(Dn[f == k])
    rec = recall_score(y, pred, average=None)
    print(f"speaker-disjoint 5-fold: UAR {balanced_accuracy_score(y, pred):.3f} over {len(classes)} tags (chance {1 / len(classes):.3f})")
    print({c: round(float(r), 2) for c, r in sorted(zip(classes, rec), key=lambda x: -x[1])})
    clf = LogisticRegression(C=0.05, max_iter=3000).fit(Dn, y)
    torch.save(dict(W=clf.coef_.astype(np.float32), b=clf.intercept_.astype(np.float32), mu=mu.astype(np.float32),
                    sd=sd.astype(np.float32), classes=classes, layers=LAYERS,
                    cv_recall={c: float(r) for c, r in zip(classes, rec)}), META / "ear2.pt")
    print("saved", META / "ear2.pt")


if __name__ == "__main__":
    {"embed": embed, "train": train}[sys.argv[1]]()
