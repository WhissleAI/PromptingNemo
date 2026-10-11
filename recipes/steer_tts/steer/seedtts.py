"""Seed-TTS-eval test-en (github.com/BytedanceSpeech/seed-tts-eval): zero-shot cloning, the protocol behind most
published WER / SIM numbers for open TTS models (F5-TTS, CosyVoice 2, Spark-TTS, MaskGCT, ...).

Each line of en/meta.lst is  name|prompt_text|prompt_wav|text_to_synthesize ; we render text_to_synthesize in the
voice of prompt_wav (reference-conditioned checkpoints only: --spk_enc runs) and write <out>/<name>.wav. Scoring
then uses the repo's OWN scorers unchanged - run_wer.py (Whisper-large-v3, their punctuation/lowercase
normalisation, jiwer) and UniSpeech verification_pair_list_v2.py (WavLM-large fine-tuned for SV) - via
logs/seedtts_score.sh in the separate `seedeval` env, so the numbers are comparable with the leaderboard.

python -m steer.seedtts --ckpt runs/spk_enc_full/last.pt --meta <seedtts>/data/seedtts_testset/en/meta.lst \
    --out runs/seedtts/<name>
Note: our model is not a prompt-continuation model - it uses only the prompt's speaker vector (not its text or
prosody), at the model's own level-normalised loudness.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from .synth import Synth
    s = Synth(a.ckpt)
    assert s.spk_ref is not None, "seed-tts-eval is zero-shot cloning: needs a reference-conditioned (--spk_enc) checkpoint"
    meta = Path(a.meta)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [l.rstrip("\n").split("|") for l in open(meta) if l.strip()]
    n = 0
    for r in rows:
        name, _prompt_text, prompt_wav, text = r[0], r[1], r[2], r[3]
        f = out / f"{name}.wav"
        if f.exists():
            continue
        ref = meta.parent / prompt_wav
        y = np.asarray(s(text, None, ref_wav=str(ref)), np.float32)
        sf.write(f, y, SR)
        n += 1
    print(f"rendered {n} (of {len(rows)}) -> {out}")


if __name__ == "__main__":
    main()
