#!/bin/bash
# Base SteerTTS lineage, ears25_base -> ... -> spk_enc_v3 (the paper's base checkpoint = runs/spk_enc_v3/step312500.pt).
#   bash scripts/10_train_base.sh <stage>      stages: s0 s1 s2 s3 s4 s5 s6   (each initialises from the previous one)
#
# The step counter is cumulative across stages (--init_continue keeps the step and LR schedule position, --rewarm
# re-warms the LR). Every stage was stopped by hand at the step listed; here `until_step` stops it
# there automatically. --steps only sets the cosine schedule length, so it is kept as originally launched.
#
# REPRODUCIBILITY CAVEAT: data/meta was rebuilt between stages as corpora were added (s0: EARS 25 speakers; s1/s2:
# EARS + Expresso + LibriTTS-R clean.100; s3: + clean.360 + Cartesia; s4: + other.500; s5/s6: + EARS freeform +
# NonverbalTTS = the v3 META of scripts/02_prep.sh). Running every stage on the final META is the closest available
# re-run; it is not bit-identical. The train/val split (2 %) is drawn over the current controls.parquet row order.
source "$(dirname "$0")/env.sh"
cd $RECIPE
R=$W/runs
HYB="--enc_mixer hybrid --dec_mixer hybrid --ssm_state 32 --compile"

until_step() {  # run step cmd...: launch a resumable trainer, stop it once runs/<run>/step<step>.pt exists
  local run=$1 step=$2; shift 2
  # launched directly (not through a shell function) so $! is the trainer itself and `kill $pid` stops exactly it
  if [ -n "${MEMCAP:-}" ]; then systemd-run --user --scope -p MemoryMax="$MEMCAP" -p MemorySwapMax=0 "$@" &
  else "$@" & fi
  local pid=$!
  until [ -f $R/$run/step$step.pt ] || ! kill -0 $pid 2>/dev/null; do sleep 60; done
  kill -0 $pid 2>/dev/null && { kill $pid; wait $pid 2>/dev/null; }   # our own child only, by PID
  [ -f $R/$run/step$step.pt ] || { echo "$run did not reach step $step"; return 1; }
  echo "=== $run stopped at $step"
}

case "${1:-}" in
s0)  # attention-only base on EARS (25 speakers at the time), from scratch; stopped at 50k (val pitch overfitting)
  until_step ears25_base 50000 $PY -m steer.train --run ears25_base --resume --steps 150000 --max_frames 16000 \
    --keep_every 10000
  cp $R/ears25_base/step50000.pt $R/ears25_base/step50000_full.pt ;;
s1)  # attention model on EARS + Expresso + LibriTTS-R clean.100 (speaker/tag rows mapped by name); 17k = graft point
  until_step mix_s1 17000 $PY -m steer.train --run mix_s1 --resume --init $R/ears25_base/step50000_full.pt \
    --oversample '{"ears":1.5,"expresso":1.5,"libritts":1.0}' --steps 150000 --max_frames 16000 --workers 12 \
    --eval_every 5000 --harness_every 10000 --ckpt_every 1000 --keep_every 1000 --harness_voices 3 --harness_texts 4
  cp $R/mix_s1/step17000.pt $R/mix_s1/init_for_m3.pt ;;
s2)  # graft zero-initialised bidirectional Mamba-3 branches into every block (identity at init); 17k -> 116k
  until_step mix_m3hyb 116000 $PY -m steer.train --run mix_m3hyb --resume --init $R/mix_s1/init_for_m3.pt \
    --init_continue --rewarm 1000 $HYB --oversample '{"ears":1.5,"expresso":1.5,"libritts":1.0}' --steps 150000 \
    --max_frames 16000 --workers 8 --eval_every 5000 --harness_every 10000 --ckpt_every 1000 --keep_every 1000 \
    --harness_voices 3 --harness_texts 4
  cp $R/mix_m3hyb/step116000.pt $R/mix_m3hyb/final_116000.pt ;;
s3)  # x-vector conditioning (--spk_enc: spk_proj initialised by ridge regression onto the speaker table), per-speaker
     # level normalisation; 116k -> 178k. (The original run enabled level normalisation only from 134k.)
  until_step spk_enc 178000 $PY -m steer.train --run spk_enc --resume --init $R/mix_m3hyb/final_116000.pt \
    --init_continue --rewarm 2000 --spk_enc $HYB --oversample '{"ears":1.5,"expresso":1.5,"libritts":1.0,"cartesia":1.0}' \
    --steps 300000 --max_frames 16000 --workers 8 --eval_every 5000 --harness_every 10000 --ckpt_every 1000 \
    --keep_every 1000 --harness_voices 3 --harness_texts 4
  cp $R/spk_enc/step178000.pt $R/spk_enc/phase1_last.pt ;;
s4)  # + LibriTTS-R other.500, rebalanced sampling (LibriTTS had been 85 % of batches); 178k -> 290k
  until_step spk_enc_full 290000 $PY -m steer.train --run spk_enc_full --resume --init $R/spk_enc/phase1_last.pt \
    --init_continue --rewarm 2000 --spk_enc $HYB --oversample '{"ears":4,"expresso":3,"libritts":0.6,"cartesia":2}' \
    --steps 300000 --max_frames 16000 --workers 8 --eval_every 5000 --harness_every 10000 --ckpt_every 1000 \
    --keep_every 10000 --harness_voices 3 --harness_texts 4
  cp $R/spk_enc_full/step290000.pt $R/spk_enc_full/final.pt ;;
s5)  # v2: paired-intervention delta loss + reward fine-tuning (PCGrad, trust-region anchor) + noise augmentation;
     # 290k -> 297.5k (best by harness; later steps degraded)
  RW='{"control":0.5,"leak":4.0,"emotion":0.3,"ctc":0.3,"utmos":0.3,"spk":0.5,"env":0.5,"anchor":1.0}'
  until_step spk_enc_v2 297500 $PY -m steer.train --run spk_enc_v2 --resume --init $R/spk_enc_full/final.pt \
    --init_continue --rewarm 1000 --spk_enc $HYB \
    --oversample '{"ears":4,"earsff":1,"expresso":3,"libritts":0.6,"cartesia":2,"nvtts":2}' --steps 320000 \
    --max_frames 10000 --workers 12 --eval_every 2500 --harness_every 2500 --keep_every 2500 --ckpt_every 500 \
    --harness_voices 4 --harness_texts 8 --noise 0.25 --pair_w 1.0 --pair_k 6 \
    --reward 0.25 --reward_every 4 --reward_k 4 --pcgrad --reward_w "$RW" ;;
s6)  # v3: EARS style pairs restored, in-voice nonverbal splices only (--noise_mode nv), no env/voicing reward;
     # 297.5k -> 317.5k, paper checkpoint = step312500.pt (best harness/eval). REWARD_MAX_FRAMES=500 avoids the judge
     # OOM seen at 305.1k (the run used the default 650 before that and 500 after).
  RW='{"control":0.5,"leak":4.0,"emotion":0.3,"ctc":0.3,"utmos":0.3,"spk":0.5,"anchor":1.0}'
  export REWARD_MAX_FRAMES=500
  until_step spk_enc_v3 317500 $PY -m steer.train --run spk_enc_v3 --resume \
    --init $R/spk_enc_v2/step297500.pt --init_continue --rewarm 1000 --spk_enc $HYB \
    --oversample '{"ears":4,"earsff":1,"expresso":3,"libritts":0.6,"cartesia":2,"nvtts":2}' --steps 336000 \
    --max_frames 8000 --workers 12 --eval_every 2500 --harness_every 2500 --keep_every 2500 --ckpt_every 500 \
    --harness_voices 4 --harness_texts 8 --noise 0.4 --noise_mode nv --pair_w 1.0 --pair_k 5 \
    --reward 0.25 --reward_every 4 --reward_k 4 --pcgrad --reward_w "$RW" ;;
*) sed -n 2,13p "$0"; exit 1 ;;
esac
