#!/usr/bin/env bash
# Unified TensorBoard: syncs logs from all training instances to one place,
# runs TensorBoard, and opens an SSH tunnel to localhost:6006.
#
# Usage: ./sync-tb-logs.sh
#   View at http://localhost:6006
#   Ctrl+C to stop everything.
set -euo pipefail

ZONE="${GCP_ZONE:-us-central1-c}"
GCP_USER="${GCP_USER:?Set GCP_USER to your Whissle username (e.g. export GCP_USER=yourname)}"
CENTRAL="${CENTRAL_INSTANCE:?Set CENTRAL_INSTANCE to the instance hosting TensorBoard (e.g. export CENTRAL_INSTANCE=nemo-hindi-a100)}"
TB_DIR="/mnt/training/tb_logs"
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"

# Remote instances to sync from — add more as needed
REMOTE_INSTANCES=("${SYNC_INSTANCES:-nemo-adapter-spot}")
REMOTE_DIRS=("/home/${GCP_USER}/workspace/experiments")

SYNC_PID=""
TUNNEL_PID=""

cleanup() {
  echo ""
  echo "Shutting down..."
  [[ -n "${SYNC_PID}" ]] && kill "$SYNC_PID" 2>/dev/null
  [[ -n "${TUNNEL_PID}" ]] && kill "$TUNNEL_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

sync_once() {
  local i
  for i in "${!REMOTE_INSTANCES[@]}"; do
    local instance="${REMOTE_INSTANCES[$i]}"
    local src_dir="${REMOTE_DIRS[$i]}"
    local dest="${TB_DIR}/${instance}"

    if ! gcloud compute ssh "${GCP_USER}@${instance}" --zone="${ZONE}" --command="true" 2>/dev/null; then
      echo "  [$(date '+%H:%M:%S')] ${instance} unreachable, skipping"
      continue
    fi

    gcloud compute ssh "${GCP_USER}@${instance}" --zone="${ZONE}" --command="
      cd ${src_dir} && sudo tar cf /tmp/_tb_sync.tar \$(find . -name 'events.out*' 2>/dev/null)
    " 2>/dev/null

    gcloud compute scp "${GCP_USER}@${instance}:/tmp/_tb_sync.tar" "/tmp/_tb_sync_${instance}.tar" --zone="${ZONE}" 2>/dev/null
    gcloud compute scp "/tmp/_tb_sync_${instance}.tar" "${GCP_USER}@${CENTRAL}:/tmp/_tb_sync.tar" --zone="${ZONE}" 2>/dev/null
    gcloud compute ssh "${GCP_USER}@${CENTRAL}" --zone="${ZONE}" --command="
      sudo mkdir -p ${dest} && cd ${dest} && sudo tar xf /tmp/_tb_sync.tar
    " 2>/dev/null

    echo "  [$(date '+%H:%M:%S')] Synced ${instance}"
  done
}

sync_loop() {
  while true; do
    sleep "${SYNC_INTERVAL}"
    sync_once 2>/dev/null || true
  done
}

# ── Build logdir_spec dynamically ──────────────────────────────
build_logdir_spec() {
  local specs=()

  # Local experiments on central instance (Hindi)
  local hindi_dirs
  hindi_dirs=$(gcloud compute ssh "${GCP_USER}@${CENTRAL}" --zone="${ZONE}" --command="
    find /mnt/training/experiments -maxdepth 2 -name 'events.out*' -exec dirname {} \; 2>/dev/null | sort -u
  " 2>/dev/null)

  while IFS= read -r dir; do
    [[ -z "$dir" ]] && continue
    local name
    name=$(echo "$dir" | sed 's|.*/experiments/||; s|/|_|g')
    specs+=("${name}:/tb_logs/hindi/${dir#/mnt/training/experiments/}")
  done <<< "$hindi_dirs"

  # Remote experiments
  local i
  for i in "${!REMOTE_INSTANCES[@]}"; do
    local instance="${REMOTE_INSTANCES[$i]}"
    local remote_dirs
    remote_dirs=$(gcloud compute ssh "${GCP_USER}@${CENTRAL}" --zone="${ZONE}" --command="
      find ${TB_DIR}/${instance} -maxdepth 3 -name 'events.out*' -exec dirname {} \; 2>/dev/null | sort -u
    " 2>/dev/null)

    while IFS= read -r dir; do
      [[ -z "$dir" ]] && continue
      local name
      name=$(echo "$dir" | sed "s|.*${instance}/||; s|/|_|g")
      specs+=("${name}:/tb_logs/remote/${instance}/${dir#${TB_DIR}/${instance}/}")
    done <<< "$remote_dirs"
  done

  local IFS=','
  echo "${specs[*]}"
}

echo "═══════════════════════════════════════════"
echo "  Unified TensorBoard"
echo "  Central: ${CENTRAL}"
echo "  Sync interval: ${SYNC_INTERVAL}s"
echo "═══════════════════════════════════════════"
echo ""

# Initial sync
echo "Syncing remote logs..."
sync_once

# Discover experiments and build logdir_spec
echo "Discovering experiments..."
LOGDIR_SPEC=$(build_logdir_spec)
echo "  Experiments: ${LOGDIR_SPEC}"
echo ""

# Start TensorBoard
echo "Starting TensorBoard..."
gcloud compute ssh "${GCP_USER}@${CENTRAL}" --zone="${ZONE}" --command="
sudo docker stop tensorboard 2>/dev/null; sudo docker rm tensorboard 2>/dev/null
sudo docker run -d \
  --name tensorboard \
  -p 6006:6006 \
  -v /mnt/training/experiments:/tb_logs/hindi \
  -v ${TB_DIR}:/tb_logs/remote \
  tensorflow/tensorflow:latest \
  bash -c 'pip install tensorboard -q && tensorboard \
    --logdir_spec \"${LOGDIR_SPEC}\" \
    --bind_all --port 6006 --reload_interval 15'
" 2>/dev/null

# Start background sync
sync_loop &
SYNC_PID=$!

# Open SSH tunnel
echo ""
echo "  ➜  http://localhost:6006"
echo ""
echo "Syncing every ${SYNC_INTERVAL}s. Ctrl+C to stop."
echo ""
gcloud compute ssh "${GCP_USER}@${CENTRAL}" --zone="${ZONE}" -- -L 6006:localhost:6006 -N &
TUNNEL_PID=$!

wait $TUNNEL_PID
