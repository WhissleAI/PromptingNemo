#!/bin/bash
# In-house evaluation of one checkpoint (see docs/evaluation.md for what each tool measures and its caveats).
#   bash scripts/32_benchmark.sh <name> <ckpt>       e.g.  v3_312k runs/spk_enc_v3/step312500.pt
# Steps: steerability harness, tag/event accuracy, quick_eval (WER/UTMOS/SIM-o subsets), external benchmark
# (LibriTTS-R test-clean, LJSpeech, clone set, ParaSpeechCaps test). Baseline systems are rendered separately in the
# `ttsbase` env: STEER_TTS_ROOT=... <ttsbase>/python steer/baselines.py kokoro|piper|matcha|parler_mini, then
# python -m steer.benchmark run --wavs kokoro_82m=runs/bench/_ext/kokoro ...
source "$(dirname "$0")/env.sh"
NAME=$1; CK=$2
cd $W
CKA=$CK; case $CK in /*) ;; *) CKA=$W/$CK;; esac
$PY -m steer.harness --ckpt $CKA --out $W/runs/_results/harness_$NAME.json
$PY -m steer.tag_eval --ckpt $CKA --out $W/runs/_results/tag_eval_$NAME.json
$PY -m steer.quick_eval $NAME=$CKA                               # -> runs/quick/<name>.json + runs/quick/summary.md
[ -d $W/data/bench ] || $PY -m steer.benchmark prep            # test lists + attribute calibration (once)
$PY -m steer.benchmark run --ckpt $NAME=$CKA
$PY -m steer.benchmark report                                  # -> runs/bench/report.md
