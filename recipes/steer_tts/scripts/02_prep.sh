#!/bin/bash
# Raw corpora -> training META (data/meta) for the current (v3) checkpoints. CPU-heavy, GPU-light.
#   bash scripts/02_prep.sh [NPROC]
# Order matters: x-vectors and build_controls read every manifest; the prompt adapter and the readback net are trained
# on the controls that build_controls writes. NEVER rebuild controls.parquet / control_stats.json while a run that
# reads them is training (the z units would change under it): back up data/meta first.
source "$(dirname "$0")/env.sh"
NPROC=${1:-12}
cd $RECIPE
# 8 corpora of the v3 META (410,566 utterances / 659 h / 4,919 speakers)
CORPORA="ears expresso libritts libritts360 libritts500 cartesia earsff nvtts"

# 1. trimmed 24 kHz wavs + manifest_<corpus>.jsonl, then features (mel/F0/energy npz) + raw_controls_<corpus>.parquet
for c in ears expresso libritts libritts360 libritts500 nvtts; do
  echo "=== prep + extract $c $(date +%T)"
  [ -f $W/data/meta/manifest_$c.jsonl ] || $PY -m steer.prep $c $NPROC
  [ -f $W/data/meta/raw_controls_$c.parquet ] || $PY -m steer.extract $c $NPROC
done
# EARS freeform monologues: Whisper-large-v3-turbo transcription + segmentation (GPU), excludes ParaSpeechCaps
# test/holdout/dev files
[ -f $W/data/meta/manifest_earsff.jsonl ] || $PY -m steer.ears_freeform
[ -f $W/data/meta/raw_controls_earsff.parquet ] || $PY -m steer.extract earsff $NPROC
# Cartesia synthetic renders (optional, paid API; resumable). Skip the corpus if you have no renders.
if [ -f $W/data/meta/manifest_cartesia.jsonl ]; then
  [ -f $W/data/meta/raw_controls_cartesia.parquet ] || $PY -m steer.extract cartesia $NPROC
else
  echo "no Cartesia manifest: run python -m steer.cartesia_gen voices/plan/plan_coverage/run first, or drop it"
  CORPORA=${CORPORA/cartesia /}
fi

# 2. frozen WavLM-base-plus-sv x-vectors, data/meta/xvec_<corpus>.npz (resumable; XVEC_BATCH=8 if GPU is shared)
XVEC_BATCH=${XVEC_BATCH:-8} $PY -m steer.xvec $CORPORA

# 3. speaker-relative control z-scores -> data/meta/controls.parquet + control_stats.json
$PY -m steer.build_controls $CORPORA
# public voice names (voice_0001 ...); the private id map stays in data/meta/voice_map_private.json
$PY -m steer.voices assign

# 4. auxiliary nets trained on that META
$PY -m steer.prompt train                              # prompt adapter -> data/meta/prompt_adapter.pt
$PY -m steer.prompt train_env                          # + environment heads -> prompt_adapter_env.pt
$PY -m steer.readback train 8 --level_norm --max 80000 # readback_ln.pt + spk_ref_mel_ln.npy (reward / consistency)
# emotion probe for the reward / tag_eval (needs the CREMA-D study outputs, see emotion_study/)
[ -f $W/data/meta/ear.pt ] || $PY -m steer.ear train || echo "ear probe skipped (CREMA-D study outputs missing)"
# noise library for --noise_mode all (v3 uses --noise_mode nv = in-voice EARS nonverbals only, no library needed)
[ -f $W/data/meta/noise_lib.jsonl ] || $PY -m steer.acoustic build || echo "noise library skipped"
echo "PREP DONE $(date)"
