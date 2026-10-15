"""Build the Hugging Face-ready SteerTTS release directory (weights as safetensors + the self-contained `steertts`
package from recipes/steer_tts/release/). Nothing is uploaded.

python -m steer.release --out $STEER_TTS_ROOT/hf_release/steertts-v1

Voices: every training speaker with a stored reference vector, under its public name (steer.voices), EXCEPT the
VoxCeleb-derived NonverbalTTS speakers (ids nvtts_id*): those are recordings of identifiable public figures
collected without consent for synthesis, so they are not offered as presets (--include_voxceleb overrides).
Descriptions are measured, not copied from corpus metadata: gender from a logistic-regression probe on the
speaker vector (fit on the speakers that do have self-reported gender labels, cross-validated accuracy printed and
stored), pitch register from the voice's neutral median F0, pace from its neutral articulation rate.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from .config import META, RAW, ROOT

SRC = Path(__file__).resolve().parents[1] / "release"  # package source (steertts/, tests/, pyproject.toml, ...)
LEAK = re.compile(r"(ears_|libritts_|expresso_|nvtts_|cartesia_|earsff_|\bid\d{5}\b|\bp\d{3}\b)")


def _save(sd, path):
    from safetensors.torch import save_file
    save_file({k: v.detach().float().contiguous().clone() if v.is_floating_point() else v.contiguous().clone()
               for k, v in sd.items()}, str(path))
    Path(path).chmod(0o644)


def gender_probe(vec, names):
    """Labels from corpus metadata where it exists -> logistic regression on the 512-d speaker vectors."""
    lab = {}
    ears = json.load(open(RAW / "ears" / "speaker_statistics.json"))
    for p, m in ears.items():
        if m.get("gender") in ("male", "female"):
            lab[f"ears_{p}"] = m["gender"]
    for f in ("voices.json", "voices_coverage.json"):
        for v in json.load(open(RAW / "cartesia" / f)):
            g = {"masculine": "male", "feminine": "female"}.get(v.get("gender"))
            if g:
                lab[f"cartesia_{v['id'][:8]}"] = g
    idx = [i for i, n in enumerate(names) if n in lab]
    X = torch.tensor(vec[idx], dtype=torch.float32)
    y = torch.tensor([1.0 if lab[names[i]] == "female" else 0.0 for i in idx])

    def fit(X, y):
        w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200)

        def closure():
            opt.zero_grad()
            l = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 1e-3 * (w ** 2).sum()
            l.backward()
            return l
        opt.step(closure)
        return w.detach(), b.detach()

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(idx), generator=g)
    folds, correct = 5, 0
    for k in range(folds):
        te = perm[k::folds]
        tr = torch.tensor([i for i in range(len(idx)) if i not in set(te.tolist())])
        w, b = fit(X[tr], y[tr])
        correct += int((((X[te] @ w + b) > 0).float() == y[te]).sum())
    acc = correct / len(idx)
    w, b = fit(X, y)
    p = torch.sigmoid(torch.tensor(vec, dtype=torch.float32) @ w + b).numpy()
    print(f"gender probe: {len(idx)} labelled speakers, 5-fold accuracy {acc:.3f}")
    return p, acc, len(idx)


def describe(names_pub, spk_mean_pub, p_female):
    f0 = np.array([100 * 2 ** (spk_mean_pub[n]["f0_med"] / 12) for n in names_pub])
    rate = np.array([spk_mean_pub[n]["rate"] for n in names_pub], float)
    gender = np.where(p_female > 0.5, "female", "male")
    out = []
    terc = {}
    for gname in ("female", "male"):
        sel = gender == gname
        terc[gname] = np.nanpercentile(f0[sel], [33.3, 66.7])
    rt = np.nanpercentile(rate, [33.3, 66.7])
    for i, n in enumerate(names_pub):
        g = gender[i]
        conf = float(p_female[i] if g == "female" else 1 - p_female[i])
        lo, hi = terc[g]
        reg = "low" if f0[i] < lo else ("high" if f0[i] > hi else "mid")
        pace = "unknown" if not np.isfinite(rate[i]) else ("slow" if rate[i] < rt[0] else ("fast" if rate[i] > rt[1] else "average"))
        g_txt = g if conf >= 0.8 else f"{g} (uncertain)"
        out.append(dict(name=n, gender=g, gender_confidence=round(conf, 3), f0_median_hz=round(float(f0[i]), 1),
                        pitch_register=reg, pace=pace,
                        description=f"{g_txt} voice, {reg} pitch for its gender (median F0 ~{f0[i]:.0f} Hz), {pace} pace"))
    return out


def build(a):
    from .voices import assign, public
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.base)
    ck = torch.load(base, map_location="cpu", weights_only=False)
    stats = json.load(open(base.parent / "control_stats.json"))
    assert len(stats["speakers"]) == ck["n_speakers"]
    assign(stats["speakers"])

    # ---- acoustic model (aligner = training only)
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("aligner.")}
    _save(sd, out / "model.safetensors")
    n_params = sum(v.numel() for v in sd.values())
    arch = dict(ck["arch"])

    # ---- flows
    flows = {}
    for key, path, fname in (("refiner", a.flow, "flow.safetensors"), ("prompted_flow", a.pflow, "pflow.safetensors")):
        fk = torch.load(path, map_location="cpu", weights_only=False)
        st = fk["ema"] if "ema" in fk else fk["model"]
        _save(st, out / fname)
        flows[key] = dict(file=fname, cfg=fk["cfg"], step=fk["step"], params=sum(v.numel() for v in st.values()),
                          steps=10, temp=0.4)

    # ---- prompt adapter
    pa = torch.load(a.adapter, map_location="cpu", weights_only=False)
    assert not pa.get("env"), "the environment adapter is not part of this release"
    _save(pa["state"], out / "prompt_adapter.safetensors")

    # ---- voices
    z = np.load(base.parent / "spk_ref.npz")
    ref = dict(zip(z["speakers"].tolist(), z["vec"].astype(np.float32)))
    keep = [s for s in stats["speakers"] if s in ref and (a.include_voxceleb or not s.startswith("nvtts_id"))]
    names = sorted(keep, key=public)
    pub = [public(s) for s in names]
    assert all(p.startswith("voice_") for p in pub)
    vec = np.stack([ref[s] for s in names])
    p_female, acc, n_lab = gender_probe(vec, names)
    spk_mean_pub = {public(s): stats["spk_mean"][s] for s in names}
    voices = describe(pub, spk_mean_pub, p_female)
    from safetensors.numpy import save_file as save_np
    save_np({"vec": vec}, str(out / "speaker_vectors.safetensors"))
    (out / "speaker_vectors.safetensors").chmod(0o644)
    default = a.default_voice
    assert default in pub, default
    json.dump(dict(default_voice=default,
                   method=dict(gender=f"logistic-regression probe on the 512-d speaker vector, fit on {n_lab} speakers "
                                      f"with self-reported labels; 5-fold accuracy {acc:.3f}",
                               pitch_register="tertiles of neutral median F0 within the probed gender",
                               pace="tertiles of neutral articulation rate over all voices"),
                   voices=voices), open(out / "voices.json", "w"), indent=0)
    json.dump(dict(pooled_sd=stats["pooled_sd"], tags=stats["tags"], spk_mean=spk_mean_pub,
                   note="spk_mean: each voice's mean RAW control measurement over its neutral training speech (the z=0 "
                        "anchor of steertts.measure.measure_z); pooled_sd: the pooled within-speaker SD (z unit)"),
              open(out / "control_stats.json", "w"))

    cfg = dict(model_type="steer-tts", release="steertts-v1", sample_rate=24000, language="en",
               default_voice=default, voices_order=pub, hf_repo=None,
               files=dict(acoustic="model.safetensors", speaker_vectors="speaker_vectors.safetensors",
                          voices="voices.json", control_stats="control_stats.json"),
               acoustic=dict(step=ck["step"], n_speakers=ck["n_speakers"], n_tags=ck["n_tags"], arch=arch,
                             params=n_params),
               refiner=flows["refiner"], prompted_flow=flows["prompted_flow"],
               prompt_adapter=dict(file="prompt_adapter.safetensors", tags=pa["tags"],
                                   encoder="sentence-transformers/all-MiniLM-L6-v2",
                                   params=sum(v.numel() for v in pa["state"].values())),
               speaker_encoder="microsoft/wavlm-base-plus-sv", vocoder="charactr/vocos-mel-24khz",
               text_frontend="g2p_en ARPAbet + rule-based normalisation (numbers, units, currency, acronyms)")
    json.dump(cfg, open(out / "config.json", "w"), indent=1)

    # ---- package source + docs
    for item in ("steertts", "tests", "pyproject.toml", "inference.py", "README.md", "LICENSE.md", "TEST_LOG.md", "leaderboard",
                 "MANIFEST.in"):
        s = SRC / item
        if not s.exists():
            continue
        d = out / item
        if d.is_dir():
            shutil.rmtree(d)
        (shutil.copytree(s, d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
         if s.is_dir() else shutil.copy2(s, d))
    shutil.copy2(SRC / "steertts" / "request.schema.json", out / "request.schema.json")

    # ---- leak check: no corpus speaker ids anywhere in the text files
    hits = []
    for f in out.rglob("*"):
        if f.is_file() and f.suffix in (".json", ".py", ".md", ".txt", ".toml", ".sh", ".cfg", ".in", ""):
            for i, line in enumerate(open(f, errors="ignore")):
                if LEAK.search(line) and "LEAK = re.compile" not in line:
                    hits.append(f"{f.relative_to(out)}:{i + 1}: {line.strip()[:120]}")
    assert not (out / "voice_map_private.json").exists()
    print(f"built {out}: {len(pub)} voices (default {default}), acoustic {n_params / 1e6:.2f}M")
    if hits:
        print("LEAK CHECK FAILED:\n" + "\n".join(hits[:40]))
        raise SystemExit(1)
    print("leak check: 0 hits")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "hf_release" / "steertts-v1"))
    ap.add_argument("--base", default=str(ROOT / "runs/spk_enc_v3/step312500.pt"))
    ap.add_argument("--flow", default=str(ROOT / "runs/flow_v3/step100000.pt"))
    ap.add_argument("--pflow", default=str(ROOT / "runs/flow_v4pa/step40000.pt"))
    ap.add_argument("--adapter", default=str(META / "prompt_adapter.pt"))
    ap.add_argument("--default_voice", default="voice_1361")  # pre-registered default (not the test-set sweep pick)
    ap.add_argument("--include_voxceleb", action="store_true")
    build(ap.parse_args())
