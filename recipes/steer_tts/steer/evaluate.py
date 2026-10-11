"""Full evaluation of a finished checkpoint: steering harness on more voices/texts, Whisper intelligibility per
direction, and the perceived-emotion ear on emotion tags. Used to pick between runs at a matched step.

python -m steer.evaluate --ckpt runs/mix_s1/last.pt --out runs/mix_s1/eval_full.json [--voices 12]
"""
import argparse
import json
import time

import numpy as np

from .harness import run, TEXTS, DIRECTIONS, TAG_DIRECTIONS

EMO_TAGS = {"anger": "ANG", "sadness": "SAD", "happiness": "HAP", "fear": "FEA", "disgust": "DIS"}


def pick_voices(stats, n):
    """Every k-th EARS voice, so the set spans the speaker list instead of the first few."""
    ears = [s for s in stats["speakers"] if s.startswith("ears")]
    return ears[:: max(1, len(ears) // n)][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voices", type=int, default=12)
    ap.add_argument("--asr_voices", type=int, default=4, help="voices used for WER / ear (x all texts)")
    ap.add_argument("--texts", type=int, default=len(TEXTS))
    a = ap.parse_args()
    texts = TEXTS[: a.texts]
    t0 = time.time()

    from .synth import Synth
    s = Synth(a.ckpt)
    voices = pick_voices(s.stats, a.voices)
    del s
    R, _ = run(a.ckpt, voices, out=a.out.replace(".json", "_harness.json"), texts=texts)
    cont, tag = R[~R.direction.str.startswith("tag_")], R[R.direction.str.startswith("tag_")]
    summary = dict(ckpt=a.ckpt, voices=voices, n_texts=len(texts),
                   control_ratio_median=float(cont.steer.median()), control_hit_mean=float(cont.hit.mean()),
                   control_leak_mean=float(cont.leak.mean()),
                   control_significant=f"{int((cont.p_holm < 0.05).sum())}/{len(cont)}",
                   tag_hit_mean=float(tag.hit.mean()) if len(tag) else None)
    print(f"[eval] harness done {time.time() - t0:.0f}s", flush=True)

    # intelligibility + perceived emotion, paired against the same voice/text baseline
    from .asr import ASR, wer
    from .ear import EarScorer, EMOS
    s, asr, ear = Synth(a.ckpt), ASR(), EarScorer()
    wers = {k: [] for k in ["baseline", *DIRECTIONS, *TAG_DIRECTIONS]}
    emo = {t: [] for t in EMO_TAGS if t in s.tag2i}
    for v in voices[: a.asr_voices]:
        for text in texts:
            base = s(text, v)
            wers["baseline"].append(wer(text, asr(base)))
            for name, (ctrl, _, _) in DIRECTIONS.items():
                wers[name].append(wer(text, asr(s(text, v, ctrl=ctrl))))
            for name, (tg, _, _) in TAG_DIRECTIONS.items():
                if tg in s.tag2i:
                    wers[name].append(wer(text, asr(s(text, v, tag=tg))))
            for tg in emo:
                p = ear.probs(s(text, v, tag=tg), base)
                emo[tg].append(p)
    summary["wer"] = {k: round(float(np.mean(x)), 4) for k, x in wers.items() if x}
    summary["wer_mean_directions"] = float(np.mean([v for k, v in summary["wer"].items() if k != "baseline"]))
    ear_res = {}
    for tg, ps in emo.items():
        P = np.stack(ps); j = EMOS.index(EMO_TAGS[tg])
        ear_res[tg] = dict(p_target=round(float(P[:, j].mean()), 3), hit=round(float((P.argmax(1) == j).mean()), 3),
                           mean_probs=dict(zip(EMOS, np.round(P.mean(0), 3).tolist())))
    summary["ear"] = ear_res
    summary["ear_hit_mean"] = float(np.mean([r["hit"] for r in ear_res.values()])) if ear_res else None
    summary["seconds"] = round(time.time() - t0)
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("wer", "ear", "voices")}, indent=1))
    print("WER", summary["wer"]); print("EAR", {k: (v["p_target"], v["hit"]) for k, v in ear_res.items()})


if __name__ == "__main__":
    main()
