"""Speaker vectors for reference-conditioned (zero-shot / cloning) SteerTTS.

python -m steer.xvec ears expresso libritts libritts360     -> data/meta/xvec_<corpus>.npz {ids, vec [n,512] f16}

Frozen WavLM-base-plus-sv x-vectors (L2-normalised) of every trimmed 24 kHz utterance in manifest_<corpus>.jsonl.
Re-runnable: ids already in the npz are skipped. Training conditions on the vector of a DIFFERENT, preferably
neutral, utterance of the same speaker (data.TTSData), so the vector has to carry voice identity, not the target
utterance's style.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

from .config import META, SR

MODEL = "microsoft/wavlm-base-plus-sv"
DIM = 512


class XVec:
    def __init__(self, dev="cuda"):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL)
        self.m = WavLMForXVector.from_pretrained(MODEL).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, wavs24, max_s=8.0):
        """list of float32 arrays at 24 kHz -> [n, 512] L2-normalised (first max_s seconds of each)."""
        import soxr
        ys = [soxr.resample(w[: int(max_s * SR)].astype(np.float32), SR, 16000) for w in wavs24]
        # tiny clips (a few LibriTTS-R utterances are < 0.1 s) are shorter than WavLM's conv receptive field: tile
        # them up to 0.5 s; padding a whole batch of them with zeros would not help, the batch max is still short
        ys = [np.tile(y, int(np.ceil(8000 / max(len(y), 1)))) if len(y) < 8000 else y for y in ys]
        x = self.fe(ys, sampling_rate=16000, return_tensors="pt", padding=True).to(self.dev)
        e = self.m(**x).embeddings
        return torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()


def load_all(corpora=("ears", "expresso", "libritts", "libritts360", "libritts500", "cartesia", "earsff", "nvtts")):
    """id -> vector over every xvec_<corpus>.npz that exists."""
    out = {}
    for c in corpora:
        f = META / f"xvec_{c}.npz"
        if f.exists():
            z = np.load(f)
            out.update(zip(z["ids"].tolist(), z["vec"].astype(np.float32)))
    return out


def main():
    xv = XVec()
    for corpus in sys.argv[1:]:
        f = META / f"xvec_{corpus}.npz"
        have = dict(zip(*(lambda z: (z["ids"].tolist(), list(z["vec"])))(np.load(f)))) if f.exists() else {}
        rows = [json.loads(l) for l in open(META / f"manifest_{corpus}.jsonl")]
        todo = [r for r in rows if r["id"] not in have]
        print(corpus, len(rows), "utts,", len(todo), "to embed", flush=True)
        # batch by similar length to keep padding (and VRAM) small
        todo.sort(key=lambda r: r["dur"])
        bs = int(os.environ.get("XVEC_BATCH", 32))  # smaller when sharing the GPU (32 OOMed beside training + benchmark)
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            vecs = xv([sf.read(r["wav"], dtype="float32")[0] for r in chunk])
            have.update({r["id"]: v.astype(np.float16) for r, v in zip(chunk, vecs)})
            if (i // bs) % 200 == 0:
                print(f"  {corpus} {i + len(chunk)}/{len(todo)}", flush=True)
        ids = sorted(have)
        np.savez(f, ids=np.array(ids), vec=np.stack([have[k] for k in ids]))
        print("wrote", f, len(ids), flush=True)


if __name__ == "__main__":
    main()
