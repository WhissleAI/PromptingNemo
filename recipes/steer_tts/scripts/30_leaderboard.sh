#!/bin/bash
# Open TTS Leaderboard protocol (English: Seed-TTS test-en 1088 + CV3-Eval zero_shot en 500), run locally.
#   OTL_REPO=/path/to/open-tts-leaderboard bash scripts/30_leaderboard.sh <model_id> <ckpt-spec> [--voice_clone] [--max N]
# ckpt-spec: base.pt | base.pt,flow=F | base.pt,flow=F,pflow=P (paths relative to $STEER_TTS_ROOT), e.g.
#   whissle/steer-tts-9m-flow     runs/spk_enc_v3/step312500.pt,flow=runs/flow_v3/step100000.pt
#   whissle/steer-tts-9m-pflow4   runs/spk_enc_v3/step312500.pt,flow=runs/flow_v3/step100000.pt,pflow=runs/flow_v4pa/step40000.pt  --voice_clone
# Stage 1 = steer.leaderboard_eval (writes the repo's results/ layout; default voice voice_4223 without cloning);
# stages 2-4 = the repo's own unmodified scorers: transcribe.py (Qwen3-ASR-1.7B), score_similarity.py (seed-tts-eval
# WavLM-large SV, cloning only), score_model.py --no_sync. CV3 data: the public mirror yuekai/CV3-Eval
# (leaderboard_eval reads it as described in its docstring). Run one scorer at a time: concurrent scorers OOM and
# silently leave pred_text empty (WER 100 %).
source "$(dirname "$0")/env.sh"
: "${OTL_REPO:?set OTL_REPO to the open-tts-leaderboard clone}"
export OTL_ROOT=$OTL_REPO   # where steer.leaderboard_eval writes results/
# NOTE: leaderboard_eval reads the Seed-TTS meta.lst and the CV3 parquet (+ cv3_en_prompts/) from fixed paths under
# ~/Desktop/whissle/workspace/{seedtts,lb_data} (SEED_META / CV3_PARQUET in the module); symlink them elsewhere.
M=$1; CK=$2; shift 2
for set in seed cv3; do
  (cd $W && $PY -m steer.leaderboard_eval --ckpt "$CK" --model_id $M --set $set "$@")
done
cd $OTL_REPO
for f in results/${M/\//-}/*.jsonl; do
  case $f in *_wavlm_seed_tts.jsonl) continue;; esac
  $PY transformers/transcribe.py --manifest_path $f --asr_batch_size 32
  case $f in *_voice_clone.jsonl) $PY transformers/score_similarity.py --manifest_path $f;; esac
done
$PY scripts/score_model.py --model_id $M --no_sync | grep -E "WER =|SIM|RTFx ="
