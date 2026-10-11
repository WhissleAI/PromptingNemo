#!/bin/bash
# NAACL SteerTTS speed & efficiency study: ALL GPU timing in one job on an otherwise idle GPU (gpu_queue 01_speed.sh
# calls this). Each sub-benchmark is guarded (set -e off) and writes its rows incrementally to results/speed_*.json.
# CPU timing also runs here, so no training data loaders compete for cores. Target < 60 min.
# CPU pinning: 1 thread -> core 23; 8 threads -> physical cores 16-23 (no SMT siblings; 2970WX, 24C/48T).
set +e
W=/home/karan/Desktop/whissle/workspace/steer_tts R=/home/karan/Desktop/whissle/PromptingNemo/recipes/steer_tts
P=/home/karan/Desktop/whissle/marketing_post_agent/paper/steertts/results
PY=~/anaconda3/envs/emotts/bin/python
BPY=~/anaconda3/envs/ttsbase/bin/python
export STEER_TTS_ROOT=$W PYTHONPATH=$R PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd $W
SP="-m steer.paper.speed"
STAGE=${STAGE:-all}
T0=$(date +%s)
log() { echo "=== $* [$(date +%T), +$(( $(date +%s) - T0 ))s, load $(cut -d' ' -f1-3 /proc/loadavg)]"; }
nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu,clocks.sm --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

if [[ $STAGE == all || $STAGE == gpu ]]; then
log "steer GPU (base, flow, clone; components; flow steps; prompt adapter)"
timeout 900 $PY $SP steer --device cuda --n_runs 20 --budget 40 --variants base,flow,clone --prompt --out $P/speed_steer.json
log "stream GPU"
timeout 300 $PY $SP stream --device cuda --n_runs 20 --out $P/speed_stream.json
timeout 300 $PY $SP stream --device cuda --n_runs 20 --flow --out $P/speed_stream.json
log "scaling GPU eager"
timeout 900 $PY $SP scaling --device cuda --n_runs 10 --budget 20 --modes eager,bf16 --vocos --out $P/speed_scaling.json
log "scaling GPU compile"
timeout 600 $PY $SP scaling --device cuda --n_runs 10 --budget 20 --modes compile --lengths 2,5,10,20,40,80,160 --out $P/speed_scaling.json
log "HiFi-GAN V1 vocoder reference GPU"
timeout 300 $BPY $SP hifigan --device cuda --n_runs 10 --budget 15 --lengths 2,5,10,20,40,80,160,320,640 --out $P/speed_scaling.json
log "baselines GPU"
for s in kokoro parler_mini; do
  timeout 900 $BPY $SP ext --system $s --device cuda --n_runs 20 --budget 40 --flops --out $P/speed_baselines.json
done
# systems registered by the baselines agent (steer/paper/baselines_ext.py), each in its own env
if [ -f $R/steer/paper/baselines_ext.py ]; then
  $PY $R/steer/paper/speed_ext_list.py > /tmp/speed_ext_list.txt
  cat /tmp/speed_ext_list.txt
  while read -r name py; do
    case $name in kokoro|piper|parler_mini|"") continue;; esac
    [ -x "$py" ] || { echo "skip $name: no python $py"; continue; }
    log "ext GPU $name"
    timeout 600 $py $SP ext --system $name --device cuda --n_runs 20 --budget 40 --flops --out $P/speed_baselines.json
  done < /tmp/speed_ext_list.txt
fi
log "training-step compile ablation (300 steps each, hybrid, spk_enc config, supervised only)"
for c in on off; do
  run=_speed_compile_$c; rm -rf runs/$run
  flag=""; [ $c = on ] && flag="--compile"
  timeout 600 $PY -m steer.train --run $run --steps 300 --max_frames 8000 --workers 10 --eval_every 100000 \
    --ckpt_every 100000 --keep_every 100000 --harness_every 0 --log_every 50 --enc_mixer hybrid --dec_mixer hybrid \
    --spk_enc $flag > logs/$run.log 2>&1
  echo "compile=$c exit $?"; grep -E "it/s|step +[0-9]+" logs/$run.log | tail -4
done
fi

if [[ $STAGE == all || $STAGE == cpu ]]; then
# CPU: other agents may load the CPU; every row records the 1-min load average (rerun with STAGE=cpu if noisy)
for th in 8 1; do
  cores=${CPU8:-16-23}; [ $th = 1 ] && cores=${CPU1:-23}
  log "steer CPU threads=$th"
  timeout 900 taskset -c $cores $PY $SP steer --device cpu --threads $th --n_runs 10 --budget 15 --sets short,medium,long \
    --variants base,flow,clone --prompt --out $P/speed_steer.json
  log "stream CPU threads=$th"
  timeout 240 taskset -c $cores $PY $SP stream --device cpu --threads $th --n_runs 10 --out $P/speed_stream.json
  for s in piper kokoro; do
    log "$s CPU threads=$th"
    timeout 420 taskset -c $cores $BPY $SP ext --system $s --device cpu --threads $th --n_runs 10 --budget 15 \
      --sets short,medium,long --out $P/speed_baselines.json
  done
done
log "parler_mini CPU threads=8"
timeout 300 taskset -c ${CPU8:-16-23} $BPY $SP ext --system parler_mini --device cpu --threads 8 --n_runs 3 --min_runs 3 \
  --budget 30 --sets short --out $P/speed_baselines.json
log "HiFi-GAN V1 CPU 8 threads"
timeout 300 taskset -c ${CPU8:-16-23} $BPY $SP hifigan --device cpu --threads 8 --n_runs 3 --budget 10 --lengths 2,5,10,20 --out $P/speed_scaling.json
log "scaling CPU 8 threads"
timeout 420 taskset -c ${CPU8:-16-23} $PY $SP scaling --device cpu --threads 8 --n_runs 3 --budget 10 --lengths 2,5,10,20,40,80 \
  --vocos --out $P/speed_scaling.json
fi
log "SPEED DONE"
