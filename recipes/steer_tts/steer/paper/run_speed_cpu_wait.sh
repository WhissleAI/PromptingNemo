#!/bin/bash
# CPU half of the speed study, launched outside the GPU queue (lead, 2026-10-11): wait until the 5-min load average
# stays < 24 for 10 consecutive minutes and the GPU speed job is not running; hard deadline Mon 2026-10-12 12:00 PDT,
# after which it runs anyway and every row carries flag=load_deadline. Pinned to physical cores 16-23 (1 thr: 23).
# Launch: setsid nohup bash run_speed_cpu_wait.sh > <paper>/results/speed_cpu_run.log 2>&1 < /dev/null &
Q=/home/karan/Desktop/whissle/workspace/steer_tts/logs/gpu_queue
DEADLINE=$(date -d "2026-10-12 12:00 PDT" +%s)
quiet=0
while true; do
  now=$(date +%s); l5=$(cut -d' ' -f2 /proc/loadavg)
  busy_gpu_job=$(ls $Q/running/ 2>/dev/null | grep -c "speed")
  if [ "$now" -ge "$DEADLINE" ]; then export SPEED_FLAG=load_deadline; echo "deadline reached, load5 $l5"; break; fi
  if awk "BEGIN{exit !($l5 < 24)}" && [ "$busy_gpu_job" = 0 ]; then quiet=$((quiet + 1)); else quiet=0; fi
  [ $((quiet * 30)) -ge 600 ] && { echo "quiet for 10 min (load5 $l5) $(date)"; break; }
  [ $((now % 1800)) -lt 30 ] && echo "$(date +%T) waiting: load5 $l5 quiet ${quiet}x30s"
  sleep 30
done
STAGE=cpu CPU8=16-23 CPU1=23 bash /home/karan/Desktop/whissle/PromptingNemo/recipes/steer_tts/steer/paper/run_speed.sh
echo "CPU SPEED DONE $(date)"
