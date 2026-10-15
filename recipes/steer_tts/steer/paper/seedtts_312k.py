"""Reviewer follow-up (d1): Seed-TTS-eval test-en for the deployed 312.5k base with the OFFICIAL scorers
Produced: results/seedtts_312k.json + _raw.csv: bash logs/seedtts.sh v3_312k_base|v3_312k_clone <ckpt> then python -m steer.paper.seedtts_312k (queue job 06b2_seedtts_312k_excl.sh). All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.
(logs/seedtts.sh: unmodified run_wer.py = Whisper-large-v3 fp32 + their normaliser, WavLM-large SV SIM vs the prompt).
Runs: runs/seedtts/v3_312k_base (base only) and runs/seedtts/v3_312k_clone (base + flow refiner + prompted flow,
the cloning configuration). This module only parses their per-utterance outputs and adds bootstrap CIs (utterance
resampling of their per-utterance WER mean / SIM mean).

python -m steer.paper.seedtts_312k  -> results/seedtts_312k.json (+ _raw.csv)
"""
import numpy as np
import pandas as pd

from . import common as C

RUNS = {"v3_312k_base": "spk_enc_v3/step312500.pt", "v3_312k_clone": "spk_enc_v3/step312500.pt,flow=flow_v3/step100000.pt,pflow=flow_v3pc/last.pt"}


def main():
    R, raws = [], []
    for name, ck in RUNS.items():
        d = C.ROOT / "runs/seedtts" / name
        w = pd.read_csv(d / "wer.out", sep="\t", header=None, names=["wav", "wer", "ref", "hyp", "ins", "del", "sub"],
                        quoting=3, dtype={"wer": float})
        w = w[w.wav.astype(str).str.endswith(".wav")]
        w["wer"] = w.wer.astype(float)
        s = pd.read_csv(d / "sim.out", sep="\t", header=None, names=["pair", "sim"], quoting=3)
        s = s[~s.pair.astype(str).str.startswith("avg score")]
        s["wav"] = s.pair.str.split("|").str[0].str.replace("_0_-1", "", regex=False)
        cond = dict(condition="seedtts_test_en", system=name, config=ck)
        R.append(C.row("wer_official", C.boot_ci(list(w.wer), lambda u: float(np.mean(u))), len(w), str(d / "wer.out"),
                       "seed-tts-eval run_wer.py (whisper-large-v3 fp32), unmodified", "seedtts_test_en", **cond))
        R.append(C.row("sim_official", C.boot_ci(list(s.sim.astype(float)), lambda u: float(np.mean(u))), len(s),
                       str(d / "sim.out"), "seed-tts-eval WavLM-large SV (wavlm_large_finetune.pth), unmodified",
                       "seedtts_test_en", **cond))
        m = w[["wav", "wer"]].merge(s[["wav", "sim"]], on="wav", how="outer")
        m["system"] = name
        m["wav"] = m.wav.str.replace(str(C.ROOT) + "/", "", regex=False)
        raws.append(m)
    C.write("seedtts_312k", R, pd.concat(raws), dict(literature="F5-TTS 1.83/0.67, CosyVoice2 2.57/0.65, Spark 1.98/0.58 (WER%/SIM)"))


if __name__ == "__main__":
    main()
