#!/bin/bash
# Idle-machine CPU rerun of the speed study (efficiency agent, 2026-10-11 after the 16:19 reboot).
# Waits until (a) the train lane is empty (no running or pending training), (b) the GPU speed job is not running,
# (c) the 1-min load average < 3.5 and the 5-min < 4 for 5 consecutive minutes. Hard deadline Mon 2026-10-12 16:00 PDT,
# after which it runs anyway and every row carries flag=load_deadline. Announced in paper/draft/IDLE_REQUEST.md.
# Launch: setsid nohup bash run_speed_idle.sh > <paper>/results/speed_cpu_idle_run.log 2>&1 < /dev/null &
Q=/home/karan/Desktop/whissle/workspace/steer_tts/logs/gpu_queue
DEADLINE=$(date -d "${IDLE_DEADLINE:-2026-10-12 16:00 PDT}" +%s)
quiet=0; last=0
while true; do
  now=$(date +%s); read l1 l5 _ < /proc/loadavg
  ntrain=$(ls $Q/train/running/*.sh $Q/train/pending/*.sh 2>/dev/null | wc -l)
  nspeed=$(ls $Q/running/ 2>/dev/null | grep -c speed)
  if [ "$now" -ge "$DEADLINE" ]; then export SPEED_FLAG=load_deadline; echo "deadline reached, load $l1 $l5 $(date)"; break; fi
  if [ "$ntrain" = 0 ] && [ "$nspeed" = 0 ] && awk "BEGIN{exit !($l1 < ${IDLE_L1:-3.5} && $l5 < ${IDLE_L5:-4})}"; then
    quiet=$((quiet + 1)); else quiet=0; fi
  [ $((quiet * 30)) -ge 300 ] && { echo "idle for 5 min (load $l1 $l5) $(date)"; break; }
  [ $((now - last)) -ge 1800 ] && { echo "$(date +%T) waiting: load $l1 $l5 train_jobs $ntrain speed_gpu $nspeed quiet ${quiet}x30s"; last=$now; }
  sleep 30
done
bash /home/karan/Desktop/whissle/PromptingNemo/recipes/steer_tts/steer/paper/run_speed_cpu_final.sh watcher
exit 0
