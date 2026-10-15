#!/bin/bash
# The ONE idle-CPU speed rerun (+ GPU end-to-end retake). Called either by run_speed_idle.sh (train lane empty, quiet
# machine) or by the eval-lane job 00_speed_cpu_excl (Oct 12 12:00 fallback; the _excl mechanism pauses training).
# Runs at most once (marker file).
P=/home/karan/Desktop/whissle/marketing_post_agent/paper/steertts/results
Q=/home/karan/Desktop/whissle/workspace/steer_tts/logs/gpu_queue
M=$P/.speed_cpu_final.started
if [ -e $M ]; then echo "speed CPU final already started ($(cat $M)); nothing to do"; exit 0; fi
echo "$(date) via ${1:-unknown}" > $M
echo "IDLE_RUN_START $(date) via ${1:-unknown}; load $(cut -d' ' -f1-3 /proc/loadavg)"
[ "${1:-}" = job ] && export SPEED_FLAG=train_paused_excl_noon
cp -n $P/speed_steer.json $P/speed_steer_gpu07_snapshot.json 2>/dev/null
# GPU batch-1 latency is launch-bound: retake end-to-end GPU rows on the quiet machine if the GPU is ours
others=$(ls $Q/running/*.sh 2>/dev/null | grep -v speed_cpu | wc -l)
if [ "$others" = 0 ]; then
  echo "=== GPU steer retake $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  cd /home/karan/Desktop/whissle/workspace/steer_tts
  STEER_TTS_ROOT=$PWD PYTHONPATH=/home/karan/Desktop/whissle/PromptingNemo/recipes/steer_tts TOKENIZERS_PARALLELISM=false \
    timeout 900 ~/anaconda3/envs/emotts/bin/python -m steer.paper.speed steer --device cuda --n_runs 20 --budget 30 \
    --variants base,steered,guided,flow --prompt --out $P/speed_steer.json
else
  echo "eval lane busy: GPU retake skipped"
fi
STAGE=cpu CPU8=16-23 CPU1=23 bash /home/karan/Desktop/whissle/PromptingNemo/recipes/steer_tts/steer/paper/run_speed.sh
echo "CPU IDLE SPEED DONE $(date)"
