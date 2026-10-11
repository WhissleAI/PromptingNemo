"""Re-run the X1 / X2 analyses with a cluster bootstrap over the 16 VOICES (all texts of a resampled voice move
Produced: results/steer_heldout_voiceboot.json, guidance_heldout_voiceboot.json (+ _raw.csv): python -m steer.paper.voiceboot. All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.
together) -> results/steer_heldout_voiceboot.json, results/guidance_heldout_voiceboot.json. CPU only; needs the X1/X2
work files under runs/paper/x1, x2.  python -m steer.paper.voiceboot
"""
from . import common as C
from . import x1_steer_heldout as X1
from . import x2_guidance_heldout as X2


def main():
    V = C.heldout_voices(16)
    X1.CLUSTER = "voice"
    X1.analyze(V, X1.plan(V))
    X2.CLUSTER = "voice"
    X2.analyze(X2.plan(V))


if __name__ == "__main__":
    main()
