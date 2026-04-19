#!/usr/bin/env bash
# Launch TensorBoard to visualize training experiments.
# Reads from the shared experiments directory so all runs are visible.
#
# Usage:
#   ./tensorboard.sh                                    # All experiments, port 6006
#   ./tensorboard.sh --exp-dir /mnt/training/experiments --port 6007
#   ./tensorboard.sh --remote nemo-adapter-spot         # SSH tunnel from local machine
set -euo pipefail

TRAINING_ROOT="${TRAINING_ROOT:-/mnt/training}"
EXP_DIR="${TRAINING_ROOT}/experiments"
PORT=6006
REMOTE=""
PROJECT="${GCP_PROJECT:-deepvoice-468015}"
ZONE="${GCP_ZONE:-us-central1-c}"
GCP_USER="${GCP_USER:?Set GCP_USER to your Whissle username (e.g. export GCP_USER=yourname)}"

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --exp-dir DIR       Experiments directory (default: /mnt/training/experiments)
  --port PORT         TensorBoard port (default: 6006)
  --remote INSTANCE   Launch TB on remote instance and create SSH tunnel
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --remote) REMOTE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -n "${REMOTE}" ]]; then
  echo "==> Starting TensorBoard on ${REMOTE} and tunneling to localhost:${PORT}"
  echo ""

  # Start TB on remote in background
  gcloud compute ssh "${GCP_USER}@${REMOTE}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --command="
      docker run -d --rm \
        --name tensorboard \
        -p ${PORT}:${PORT} \
        -v ${EXP_DIR}:${EXP_DIR}:ro \
        nemo-training:latest \
        tensorboard --logdir ${EXP_DIR} --port ${PORT} --bind_all 2>/dev/null \
      || echo 'TensorBoard already running'
    "

  echo ""
  echo "==> Creating SSH tunnel..."
  echo "    Open http://localhost:${PORT} in your browser"
  echo "    Press Ctrl+C to stop"
  echo ""

  gcloud compute ssh "${GCP_USER}@${REMOTE}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    -- -N -L "${PORT}:localhost:${PORT}"
else
  # Local mode - run directly or in Docker
  if command -v tensorboard &>/dev/null; then
    echo "==> Starting TensorBoard at http://localhost:${PORT}"
    echo "    Log dir: ${EXP_DIR}"
    echo ""
    tensorboard --logdir "${EXP_DIR}" --port "${PORT}" --bind_all
  else
    echo "==> Starting TensorBoard in Docker at http://localhost:${PORT}"
    echo "    Log dir: ${EXP_DIR}"
    echo ""
    docker run --rm \
      --name tensorboard \
      -p "${PORT}:${PORT}" \
      -v "${EXP_DIR}:${EXP_DIR}:ro" \
      nemo-training:latest \
      tensorboard --logdir "${EXP_DIR}" --port "${PORT}" --bind_all
  fi
fi
