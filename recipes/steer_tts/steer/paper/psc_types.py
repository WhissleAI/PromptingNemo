"""ParaSpeechCaps test descriptions by attribute type, and the existing per-prompt attribute accuracies split by type.

PSC tag taxonomy (Diwan et al. 2025): basic tags (gender, pitch level, speaking rate, noise) are binned per speaker /
utterance on ABSOLUTE scales (pitch bins per gender); intrinsic tags are speaker-level voice traits (texture, accent);
situational tags are utterance-level styles/emotions (whispered, angry, ...), the only ones that describe a deviation
from the speaker's usual delivery, i.e. what a speaker-relative control space encodes.
Per-prompt *_ok flags come from runs/bench/<system>.json (steer.benchmark), rows aligned with data/bench/psc.jsonl.

python -m steer.paper.psc_types -> paper/results/adapter_psc_types.json
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

from ..config import RAW
from .common import RES, ROOT

SYSTEMS = {"parler_mini": "Parler-mini", "kokoro_82m": "Kokoro-82M", "piper_vits": "Piper", "m3hyb_58k": "SteerTTS (58k)",
           "spk_enc_full_290k": "SteerTTS (290k)"}


def main():
    t = pd.read_parquet(RAW / "paraspeechcaps/data/test-00000-of-00001.parquet")
    bench = [json.loads(l) for l in open(ROOT / "data/bench/psc.jsonl")]
    assert len(bench) == len(t) and all(b["description"] == str(d[0]).strip() for b, d in zip(bench, t.text_description))
    L = lambda x: [] if x is None else list(x)
    n_sit = np.array([len(L(x)) for x in t.situational_tags])
    n_int = np.array([len(L(x)) for x in t.intrinsic_tags])
    typ = np.where(n_sit > 0, "situational", "absolute_only")
    out = dict(n=len(t), source=dict(Counter(t.source)),
               prompts_with_situational=int((n_sit > 0).sum()), prompts_with_intrinsic=int((n_int > 0).sum()),
               prompts_absolute_only=int((n_sit == 0).sum()),
               frac_absolute_only=round(float((n_sit == 0).mean()), 4),
               tag_mentions=dict(basic=int(sum(len(L(x)) for x in t.basic_tags)), intrinsic=int(n_int.sum()),
                                 situational=int(n_sit.sum())),
               top_situational=Counter(s for x in t.situational_tags for s in L(x)).most_common(15),
               top_intrinsic=Counter(s for x in t.intrinsic_tags for s in L(x)).most_common(15),
               pitch_labels=dict(Counter(t.pitch)), rate_labels=dict(Counter(t.speaking_rate)),
               definition="situational = has >=1 PSC situational tag (utterance-level style/emotion); absolute_only = "
                          "only basic (gender, absolute pitch/rate bins, noise) and intrinsic (speaker-level) tags",
               type_x_source={ty: dict(Counter(t.source[typ == ty])) for ty in ("absolute_only", "situational")},
               type_x_pitch_label={ty: dict(Counter(t.pitch[typ == ty])) for ty in ("absolute_only", "situational")},
               situational_share_of_mentions=round(float(n_sit.sum() / sum(len(L(x)) for c in ("basic_tags", "intrinsic_tags", "situational_tags") for x in t[c])), 4),
               caveat="type is confounded with source (situational tags occur only on EARS/Expresso emotional speech) and "
                      "pitch label; accuracies by type are descriptive, not causal",
               by_type={}, by_source={})
    for key, name in SYSTEMS.items():
        f = ROOT / "runs/bench" / f"{key}.json"
        if not f.exists():
            continue
        rows = json.load(open(f))["psc"]
        res = {}
        for ty in ("absolute_only", "situational"):
            ix = [r for r in rows if typ[r["i"]] == ty]
            res[ty] = dict(n=len(ix), **{k: round(float(np.mean([r[k] for r in ix])), 4) for k in ("gender_ok", "pitch_ok", "rate_ok")})
        out["by_type"][name] = res
        out["by_source"][name] = {src: dict(n=int((t.source == src).sum()), **{k: round(float(np.mean([r[k] for r in rows if t.source.iat[r["i"]] == src])), 4)
                                            for k in ("pitch_ok", "rate_ok")}) for src in sorted(set(t.source))}
    # pitch accuracy of a constant "high-pitched" guess, overall and per type (base rate)
    out["const_high_pitch_acc"] = dict(all=round(float((t.pitch == "high-pitched").mean()), 4),
                                       **{ty: round(float((t.pitch[typ == ty] == "high-pitched").mean()), 4)
                                          for ty in ("absolute_only", "situational")})
    json.dump(out, open(RES / "adapter_psc_types.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("top_intrinsic",)}, indent=1))


if __name__ == "__main__":
    main()
