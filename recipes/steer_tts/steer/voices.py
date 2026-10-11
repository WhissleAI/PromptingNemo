"""Public voice names: dataset speaker ids (EARS / LibriTTS-R / Expresso / Cartesia ...) never leave the machine.

Every training speaker gets a stable opaque name, voice_0001 ... , assigned in a salted-hash order (so the number
carries no corpus or id information) and extended, never renumbered, when the corpus grows. The mapping lives only
in data/meta/voice_map_private.json, which must not be published. Everything user-facing (demo voice list, samples
pages, released checkpoints) uses the public names; Synth accepts either.

python -m steer.voices export --ckpt runs/X/final.pt --out release/steertts   # checkpoint + anonymised stats/refs
"""
import argparse
import hashlib
import json
import secrets
import shutil
from pathlib import Path

import numpy as np

from .config import META

MAP = META / "voice_map_private.json"
_cache = None


def _load():
    global _cache
    if _cache is None:
        _cache = json.load(open(MAP)) if MAP.exists() else {"salt": secrets.token_hex(16), "real2pub": {}}
    return _cache


def assign(speakers):
    """Gives every new speaker the next public name (existing names never change); returns real -> public."""
    m = _load()
    r2p = m["real2pub"]
    new = sorted((s for s in speakers if s not in r2p), key=lambda s: hashlib.sha256((m["salt"] + s).encode()).hexdigest())
    n = len(r2p)
    for i, s in enumerate(new):
        r2p[s] = f"voice_{n + i + 1:04d}"
    if new:
        MAP.write_text(json.dumps(m))
    return r2p


def public(real):
    return _load()["real2pub"].get(real, real if str(real).startswith("voice_") else "voice_unknown")


def resolve(name):
    """public or real name -> real (training) name; unknown names pass through."""
    if not name or not str(name).startswith("voice_"):
        return name
    m = _load()
    if "pub2real" not in m:
        m["pub2real"] = {v: k for k, v in m["real2pub"].items()}
    return m["pub2real"].get(name, name)


def export(ckpt, out):
    """Release copy: model weights only, speaker list / reference vectors under public names."""
    import torch
    ckpt, out = Path(ckpt), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stats = json.load(open(ckpt.parent / "control_stats.json"))
    assign(stats["speakers"])
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    torch.save({k: ck[k] for k in ("model", "step", "arch", "n_speakers", "n_tags") if k in ck}, out / "model.pt")
    stats = dict(stats, speakers=[public(s) for s in stats["speakers"]])
    json.dump(stats, open(out / "control_stats.json", "w"))
    ref = ckpt.parent / "spk_ref.npz"
    if ref.exists():
        z = np.load(ref)
        np.savez(out / "spk_ref.npz", speakers=np.array([public(s) for s in z["speakers"].tolist()]), vec=z["vec"])
    leaked = [s for s in stats["speakers"] if not s.startswith("voice_")]
    assert not leaked, leaked[:5]
    print(f"exported {out}: {len(stats['speakers'])} voices, all public names")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "assign"])
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.ckpt, a.out)
    else:
        from .data import load_meta
        print(len(assign(load_meta()[1]["speakers"])), "voices mapped ->", MAP)
