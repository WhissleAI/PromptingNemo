#!/usr/bin/env bash
# Check status of all running experiments, disk usage, and GPU utilization.
#
# Usage:
#   ./status.sh                           # Check on current instance
#   ./status.sh --remote nemo-adapter-spot  # Check on remote instance
set -euo pipefail

TRAINING_ROOT="${TRAINING_ROOT:-/mnt/training}"
REMOTE=""
PROJECT="${GCP_PROJECT:-deepvoice-468015}"
ZONE="${GCP_ZONE:-us-central1-c}"
GCP_USER="${GCP_USER:?Set GCP_USER to your Whissle username (e.g. export GCP_USER=yourname)}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --remote) REMOTE="$2"; shift 2 ;;
    *) shift ;;
  esac
done

run_cmd() {
  if [[ -n "${REMOTE}" ]]; then
    gcloud compute ssh "${GCP_USER}@${REMOTE}" --zone="${ZONE}" --project="${PROJECT}" --command="$1" 2>/dev/null
  else
    bash -c "$1"
  fi
}

HEADER="${REMOTE:-localhost}"
echo "══════════════════════════════════════════"
echo "  Training Status: ${HEADER}"
echo "══════════════════════════════════════════"

echo ""
echo "── GPU ──"
run_cmd "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" 2>/dev/null || echo "  (no GPU or nvidia-smi not available)"

echo ""
echo "── Disk ──"
run_cmd "df -h ${TRAINING_ROOT} 2>/dev/null" || echo "  (training disk not mounted)"

echo ""
echo "── Running Experiments ──"
CONTAINERS=$(run_cmd "docker ps --format '{{.Names}}\t{{.Status}}\t{{.RunningFor}}' 2>/dev/null" || true)
if [[ -n "${CONTAINERS}" ]]; then
  echo "  NAME                STATUS              RUNNING"
  echo "${CONTAINERS}" | sed 's/^/  /'
else
  echo "  (no containers running)"
fi

echo ""
echo "── Latest WER Results ──"
CONTAINERS_LIST=$(run_cmd "docker ps --format '{{.Names}}' 2>/dev/null" || true)
for container in ${CONTAINERS_LIST}; do
  [[ "${container}" == "tensorboard" ]] && continue
  echo ""
  echo "  ${container}:"
  run_cmd "docker logs ${container} 2>&1 | grep 'val_wer.*reached' | tail -5" 2>/dev/null | sed 's/^/    /' || echo "    (no WER data yet)"
  PROGRESS=$(run_cmd "docker logs ${container} 2>&1 | grep -oP 'Epoch \d+:\s+\d+%.*' | tail -1" 2>/dev/null || true)
  if [[ -n "${PROGRESS}" ]]; then
    echo "    Current: ${PROGRESS}"
  fi
done

echo ""
echo "── Experiments on Disk ──"
run_cmd "ls -1 ${TRAINING_ROOT}/experiments/ 2>/dev/null" | while read -r exp; do
  CKPT_COUNT=$(run_cmd "find ${TRAINING_ROOT}/experiments/${exp} -name '*.ckpt' 2>/dev/null | wc -l" || echo "0")
  NEMO_COUNT=$(run_cmd "find ${TRAINING_ROOT}/experiments/${exp} -name '*.nemo' 2>/dev/null | wc -l" || echo "0")
  BEST_WER=$(run_cmd "ls ${TRAINING_ROOT}/experiments/${exp}/version_*/checkpoints/*val_wer*.ckpt 2>/dev/null | sort | head -1" || true)
  if [[ -n "${BEST_WER}" ]]; then
    WER_VAL=$(echo "${BEST_WER}" | grep -oP 'val_wer=\K[0-9.]+' || echo "?")
    echo "  ${exp}: ${CKPT_COUNT} ckpts, ${NEMO_COUNT} .nemo files, best WER=${WER_VAL}"
  else
    echo "  ${exp}: ${CKPT_COUNT} ckpts, ${NEMO_COUNT} .nemo files"
  fi
done 2>/dev/null || echo "  (no experiments found)"

echo ""
echo "── Models ──"
run_cmd "ls -1 ${TRAINING_ROOT}/models/ 2>/dev/null" | while read -r model; do
  SIZE=$(run_cmd "du -sh ${TRAINING_ROOT}/models/${model} 2>/dev/null | cut -f1" || echo "?")
  echo "  ${model}: ${SIZE}"
done 2>/dev/null || echo "  (no models downloaded)"

echo ""
echo "── Datasets ──"
run_cmd "ls -1 ${TRAINING_ROOT}/data/ 2>/dev/null" | while read -r ds; do
  if run_cmd "test -f ${TRAINING_ROOT}/data/${ds}/dataset_info.json" 2>/dev/null; then
    INFO=$(run_cmd "cat ${TRAINING_ROOT}/data/${ds}/dataset_info.json" 2>/dev/null)
    SAMPLES=$(echo "${INFO}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'train={d.get(\"train_samples\",\"?\")}, val={d.get(\"val_samples\",\"?\")}')" 2>/dev/null || echo "?")
    echo "  ${ds}: ${SAMPLES}"
  else
    echo "  ${ds}"
  fi
done 2>/dev/null || echo "  (no datasets downloaded)"
