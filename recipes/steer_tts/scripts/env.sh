# Common environment for the SteerTTS launch scripts. Sourced, not executed.
#
#   STEER_TTS_ROOT  (required) data + runs root: data/{raw,wav24,feats,meta,gta,bench,noise}, runs/, logs/
#   PY              python of the training env (default: python on PATH; we use ~/anaconda3/envs/emotts/bin/python)
#   MEMCAP          optional host-RAM cap for long jobs, e.g. MEMCAP=48G wraps them in
#                   `systemd-run --user --scope -p MemoryMax=$MEMCAP` so a leak kills the job, not the machine
#   OTL_REPO        clone of github.com/huggingface/open-tts-leaderboard (leaderboard scoring)
#   SEEDTTS_DIR     seed-tts-eval workspace (repo checkout + test set + wavlm_large_finetune.pth)
#   SEEDEVAL_PY     python of the separate seed-tts-eval scoring env
: "${STEER_TTS_ROOT:?set STEER_TTS_ROOT to the data/runs directory}"
RECIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
export STEER_TTS_ROOT RECIPE PY
export PYTHONPATH="$RECIPE:$(cd "$RECIPE/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"   # steer + promptingnemo (Mamba-3)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$STEER_TTS_ROOT/logs"
W="$STEER_TTS_ROOT"

capped() {  # run a command under the optional memory cap
  if [ -n "${MEMCAP:-}" ]; then systemd-run --user --scope -p MemoryMax="$MEMCAP" -p MemorySwapMax=0 "$@"; else "$@"; fi
}

retry() {  # retry N cmd...: rerun a resumable job (train.py --resume / flow last.pt) after a crash
  local n=$1; shift
  for attempt in $(seq 1 "$n"); do
    echo "=== attempt $attempt: $* ($(date +%T))"
    "$@" && return 0
    sleep 30
  done
  echo "GAVE UP after $n attempts"; return 1
}
