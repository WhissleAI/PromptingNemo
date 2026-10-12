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
# CPU stage: never overlap the GPU speed job (its batch-1 timings are CPU-launch-bound)
waitgpu() { [[ $STAGE == cpu ]] || return 0; while ls /home/karan/Desktop/whissle/workspace/steer_tts/logs/gpu_queue/running 2>/dev/null | grep -q speed; do sleep 30; done; }
log() { waitgpu; echo "=== $* [$(date +%T), +$(( $(date +%s) - T0 ))s, load $(cut -d' ' -f1-3 /proc/loadavg)]"; }
nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu,clocks.sm --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Scope cut 2026-10-11 17:45 (draft/DEADLINE_PLAN.md): only rows cited by efficiency.tex / W_eff / app_efficiency.tex.
# The previous full study (bf16/compile sweeps, HiFi-GAN timing, all GPU baselines, compile ablation) is run_speed_full.sh.
if [[ $STAGE == all || $STAGE == gpu ]]; then
log "steer GPU (base, steered, guided, flow, clone; components; flow steps; prompt adapter)"
timeout 900 $PY $SP steer --device cuda --n_runs 20 --budget 30 --variants base,steered,guided,flow,clone --prompt --out $P/speed_steer.json
log "stream GPU"
timeout 300 $PY $SP stream --device cuda --n_runs 20 --out $P/speed_stream.json
log "scaling GPU eager (attn / mamba3 / hybrid + Vocos)"
timeout 900 $PY $SP scaling --device cuda --n_runs 10 --budget 15 --modes eager --vocos --out $P/speed_scaling.json
log "kokoro GPU"
timeout 600 $BPY $SP ext --system kokoro --device cuda --n_runs 20 --budget 30 --out $P/speed_baselines.json < /dev/null
fi

if [[ $STAGE == all || $STAGE == cpu ]]; then
# CPU (idle window): every row records the 1-min load average. 8 thr -> physical cores 16-23; 1 thr -> core 23.
for th in 8 1; do
  cores=${CPU8:-16-23}; [ $th = 1 ] && cores=${CPU1:-23}
  V=base,steered,guided,flow; [ $th = 8 ] && V=$V,clone
  log "steer CPU threads=$th ($V)"
  timeout 1500 taskset -c $cores $PY $SP steer --device cpu --threads $th --n_runs 10 --budget 15 --sets short,medium,long \
    --variants $V --prompt --out $P/speed_steer.json
  for s in piper kokoro; do
    sets=short,medium; [ $s = piper ] && sets=short,medium,long
    log "$s CPU threads=$th"
    timeout 1200 taskset -c $cores $BPY $SP ext --system $s --device cpu --threads $th --n_runs 10 --budget 15 \
      --sets $sets --out $P/speed_baselines.json < /dev/null
  done
done
log "stream CPU threads=8"
timeout 600 taskset -c ${CPU8:-16-23} $PY $SP stream --device cpu --threads 8 --n_runs 10 --out $P/speed_stream.json
# other W_eff rows (registered in baselines_ext.py, own envs): short + medium
$PY $R/steer/paper/speed_ext_list.py > /tmp/speed_ext_list_cpu.txt 2>/dev/null
while read -r name py grp; do
  case $name in fastpitch|matcha|paradee|kitten) THS="8 1";; parastyle|styletts2) THS="8";; *) continue;; esac
  [ -x "$py" ] || continue
  for th in $THS; do
    cores=${CPU8:-16-23}; [ $th = 1 ] && cores=${CPU1:-23}
    log "ext CPU $name threads=$th"
    BASELINE_THREADS=$th PYTHONNOUSERSITE=1 timeout 900 taskset -c $cores $py $SP ext --system $name --device cpu \
      --threads $th --n_runs 10 --budget 15 --sets short,medium --out $P/speed_baselines.json < /dev/null
  done
done < /tmp/speed_ext_list_cpu.txt
fi

if [[ $STAGE == all || $STAGE == cpu || $STAGE == llm ]]; then
# LLM front end (language section): Ollama qwen2.5:3b on CPU, 8 threads, as used for the paper's predictions
log "LLM front end CPU (ollama)"
timeout 1800 $PY $SP llm --threads 8 --n_runs 20 --min_runs 10 --budget 300 --out $P/speed_llm.json
curl -s localhost:11434/api/generate -d '{"model":"qwen2.5:3b","keep_alive":0}' > /dev/null
fi
log "SPEED DONE"
