#!/bin/bash
# At Mon 2026-10-12 12:00 PDT, release the idle-CPU speed fallback job into the eval lane unless already run.
Q=/home/karan/Desktop/whissle/workspace/steer_tts/logs/gpu_queue
M=/home/karan/Desktop/whissle/marketing_post_agent/paper/steertts/results/.speed_cpu_final.started
T=$(date -d "2026-10-12 12:00 PDT" +%s)
while [ $(date +%s) -lt $T ]; do [ -e $M ] && { echo "already ran; not releasing"; exit 0; }; sleep 60; done
[ -e $M ] && { echo "already ran"; exit 0; }
mv $Q/hold/00_speed_cpu_excl.sh $Q/pending/ && echo "released 00_speed_cpu_excl $(date)"
