"""Word/character forced alignment of every clip to its scripted sentence (GPU).

wav2vec2-large-960h-lv60-self character CTC (20 ms frames) + Viterbi through the
blank-interleaved transcript. Also stores the greedy transcript so misread lines
(actors sometimes deviate from the script) can be flagged via CER.

out/align_words.parquet  clip, w_idx, word, start, end, score (mean log-prob of word chars)
out/align_chars.parquet  clip, w_idx, c_idx, char, start, end
out/align_clip.parquet   clip, greedy, cer, align_score
"""
import re

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from common import WAV, OUT, SENTENCES

MODEL = "facebook/wav2vec2-large-960h-lv60-self"
FRAME = 0.02


def norm(text):
    return re.sub(r"[^A-Z' ]", "", text.upper()).split()


def viterbi(logp, tokens, blank):
    """CTC forced alignment. logp [T, V] log-probs, tokens list[int].
    Returns per-token (start_frame, end_frame_exclusive)."""
    T, S = logp.shape[0], 2 * len(tokens) + 1
    ext = np.full(S, blank)
    ext[1::2] = tokens
    NEG = -1e30
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), np.int8)
    dp[0, 0] = logp[0, blank]
    dp[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        prev = dp[t - 1]
        stay = prev
        step = np.concatenate([[NEG], prev[:-1]])
        skip = np.concatenate([[NEG, NEG], prev[:-2]])
        skip_ok = (ext != blank) & (np.concatenate([[blank, blank], ext[:-2]]) != ext)
        skip = np.where(skip_ok, skip, NEG)
        cand = np.stack([stay, step, skip])
        bp[t] = cand.argmax(0)
        dp[t] = cand.max(0) + logp[t, ext]
    s = S - 1 if dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    path = np.empty(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    spans = []
    for k in range(len(tokens)):
        fr = np.where(path == 2 * k + 1)[0]
        spans.append((fr[0], fr[-1] + 1) if len(fr) else (None, None))
    return spans


def cer(a, b):
    a, b = list(a), list(b)
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1] / max(1, len(b))


def main():
    dev = "cuda"
    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL).to(dev).eval().half()
    vocab = proc.tokenizer.get_vocab()
    blank = vocab[proc.tokenizer.pad_token]
    inv = {v: k for k, v in vocab.items()}

    clips = pd.read_parquet(OUT / "clips.parquet")
    W, C, K = [], [], []
    for clip, sent in tqdm(zip(clips["clip"], clips.sent), total=len(clips)):
        y, _ = sf.read(WAV / f"{clip}.wav", dtype="float32")
        x = proc(y, sampling_rate=16000, return_tensors="pt").input_values.to(dev).half()
        with torch.no_grad():
            logp = model(x).logits.float().log_softmax(-1)[0].cpu().numpy()
        g = logp.argmax(-1)
        g = [i for k, i in enumerate(g) if i != blank and (k == 0 or i != g[k - 1])]
        greedy = "".join(inv[i] for i in g).replace("|", " ").strip()

        words = norm(SENTENCES[sent])
        toks, owner = [], []
        for wi, w in enumerate(words):
            if wi:
                toks.append(vocab["|"]); owner.append((-1, -1))
            for ci, ch in enumerate(w):
                toks.append(vocab[ch]); owner.append((wi, ci))
        spans = viterbi(logp, toks, blank)
        word_frames = {}
        for (wi, ci), (s, e), tok in zip(owner, spans, toks):
            if wi < 0 or s is None:
                continue
            C.append(dict(clip=clip, w_idx=wi, c_idx=ci, char=inv[tok], start=s * FRAME, end=e * FRAME))
            word_frames.setdefault(wi, []).append((s, e, logp[s:e, tok].mean()))
        scores = []
        for wi, w in enumerate(words):
            fs = word_frames.get(wi, [])
            if not fs:
                continue
            sc = float(np.mean([f[2] for f in fs]))
            scores.append(sc)
            W.append(dict(clip=clip, w_idx=wi, word=w, start=fs[0][0] * FRAME,
                          end=fs[-1][1] * FRAME, score=sc))
        K.append(dict(clip=clip, greedy=greedy, cer=cer(greedy, " ".join(words)),
                      align_score=float(np.mean(scores)) if scores else np.nan))

    pd.DataFrame(W).to_parquet(OUT / "align_words.parquet", index=False)
    pd.DataFrame(C).to_parquet(OUT / "align_chars.parquet", index=False)
    k = pd.DataFrame(K)
    k.to_parquet(OUT / "align_clip.parquet", index=False)
    print(k.describe())
    print("clips with CER > 0.2:", (k.cer > 0.2).sum())


if __name__ == "__main__":
    main()
