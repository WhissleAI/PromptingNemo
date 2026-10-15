#!/bin/bash
# SteerTTS end-to-end recipe. Each stage is re-runnable (finished outputs are skipped / overwritten).
#
#   ./run.sh <stage> [extra args]
#   stages: download | prep | ear | readback | captions | prompt | train | stage2 | harness | tensorboard
#
# Data, features and runs go under $STEER_TTS_ROOT (default: this directory).
# CORPORA (default "ears expresso libritts") picks which corpora prep/build_controls use.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export STEER_TTS_ROOT="${STEER_TTS_ROOT:-$HERE}"
CORPORA="${CORPORA:-ears expresso libritts}"
RUN="${RUN:-base}"
PY="${PY:-python}"
cd "$HERE"
mkdir -p "$STEER_TTS_ROOT/logs"
stage="${1:-}"; shift || true

case "$stage" in
download)  # EARS from GitHub releases; HF corpora via parallel resumable curl (needs data/.hf_curlrc)
  bash data/download.sh ears
  bash data/hf_curl.sh data/hf_files.tsv ;;
prep)      # 24 kHz trimmed wavs -> mel/f0/energy features + raw controls -> speaker-relative control z
  for c in $CORPORA; do
    $PY -m steer.prep "$c"
    $PY -m steer.extract "$c"
  done
  $PY -m steer.build_controls $CORPORA ;;
ear)       # perceived-emotion probe (needs emotion_study s01 + s04 outputs, see README)
  $PY -m steer.ear train ;;
readback)  # mel -> control z network used by the stage-2 consistency loss
  $PY -m steer.readback train ;;
captions | prompt)  # acoustically anchored captions -> prompt adapter
  $PY -m steer.prompt train ;;
train)     # stage 1: reconstruction training with dropout over control dims / tags
  $PY -m steer.train --run "$RUN" "$@" ;;
stage2)    # stage 2: control-consistency fine-tune from a stage-1 checkpoint
  $PY -m steer.train --run "${RUN}_s2" --init "$STEER_TTS_ROOT/runs/$RUN/last.pt" --cons 0.3 "$@" ;;
harness)   # full paired steerability evaluation
  $PY -m steer.harness --ckpt "$STEER_TTS_ROOT/runs/$RUN/last.pt" --out "$STEER_TTS_ROOT/runs/$RUN/harness_full.json" "$@" ;;
tensorboard)
  tensorboard --logdir "$STEER_TTS_ROOT/runs" --port "${PORT:-6006}" --samples_per_plugin "images=200,audio=200" ;;
*)
  sed -n 2,9p "$0"; exit 1 ;;
esac
