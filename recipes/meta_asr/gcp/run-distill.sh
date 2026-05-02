#!/usr/bin/env bash
# Run knowledge distillation inside Docker on a GCP spot instance.
# Creates a 35M student model from the 600M teacher.
#
# Usage:
#   ./run-distill.sh --name distill-35m-v1
#   ./run-distill.sh --name distill-35m-v1 --config /path/to/custom.yaml
#   ./run-distill.sh --name distill-35m-v1 --resume /path/to/checkpoint.ckpt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="${TRAINING_ROOT:-/mnt/training}"
CODE_DIR="${TRAINING_ROOT}/code/PromptingNemo"

# Defaults
EXP_NAME=""
CONFIG_PATH=""
DETACH="true"
RESUME_FROM=""
GPU_COUNT=1

usage() {
  cat <<EOF
Usage: $0 --name EXP_NAME [OPTIONS]

Required:
  --name NAME           Experiment name (Docker container + TB logs)

Options:
  --config PATH         Path to distillation YAML config
                        (default: recipes/meta_asr/conf/distill_35m.yaml)
  --resume PATH         Resume from checkpoint path
  --gpus N              Number of GPUs (default: 1)
  --no-detach           Run in foreground (default: detached)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --name) EXP_NAME="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --resume) RESUME_FROM="$2"; shift 2 ;;
    --gpus) GPU_COUNT="$2"; shift 2 ;;
    --no-detach) DETACH="false"; shift ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "${EXP_NAME}" ]] && { echo "ERROR: --name is required"; usage; }

# Default config
if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${CODE_DIR}/recipes/meta_asr/conf/distill_35m.yaml"
fi

# ── Verify prerequisites ─────────────────────────────────────

MODEL_DIR="${TRAINING_ROOT}/models/stt_meta_1b"
NEMO_FILE=$(find "${MODEL_DIR}" -name "*.nemo" -type f 2>/dev/null | head -1)
if [[ -z "${NEMO_FILE}" ]]; then
  echo "ERROR: No .nemo file found in ${MODEL_DIR}"
  echo "Run: ./download-model.sh --model WhissleAI/STT-meta-1B"
  exit 1
fi
echo "==> Teacher model: ${NEMO_FILE}"

DATA_DIR="${TRAINING_ROOT}/data/meta_stt_euro_set1"
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: Dataset not found at ${DATA_DIR}"
  echo "Run: ./download-data.sh --dataset WhissleAI/Meta_STT_EURO_Set1 --lang EUROPEAN"
  exit 1
fi
echo "==> Dataset: ${DATA_DIR}"

# ── Ensure Docker image exists ────────────────────────────────

REPO_ROOT="${SCRIPT_DIR}/../../.."
DOCKER_DIR="${REPO_ROOT}/docker"
if ! docker image inspect nemo-training:latest &>/dev/null; then
  echo "==> Building Docker image: nemo-training:latest"
  docker build -t nemo-training:latest -f "${DOCKER_DIR}/Dockerfile.training" "${REPO_ROOT}"
fi

# ── Ensure code is on disk ────────────────────────────────────

if [[ ! -d "${CODE_DIR}" ]]; then
  echo "==> Cloning PromptingNemo..."
  mkdir -p "${TRAINING_ROOT}/code"
  git clone https://github.com/WhissleAI/PromptingNemo.git "${CODE_DIR}"
fi

# ── Stop existing container ───────────────────────────────────

if docker ps -a --format '{{.Names}}' | grep -q "^${EXP_NAME}$"; then
  echo "==> Stopping existing container: ${EXP_NAME}"
  docker stop "${EXP_NAME}" 2>/dev/null || true
  docker rm "${EXP_NAME}" 2>/dev/null || true
fi

# ── Build Docker command ──────────────────────────────────────

DOCKER_ARGS=(
  --name "${EXP_NAME}"
  --gpus all
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "${TRAINING_ROOT}:${TRAINING_ROOT}"
  -v "${CODE_DIR}:/workspace/code"
  -e PYTHONPATH="/workspace/code:/workspace/code/NeMo-W"
  -e CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
)

TRAIN_CMD="cd /workspace/code && python scripts/asr/meta-asr/distill.py --config ${CONFIG_PATH}"

if [[ -n "${RESUME_FROM}" ]]; then
  TRAIN_CMD="${TRAIN_CMD} --resume_from ${RESUME_FROM}"
fi

echo ""
echo "══════════════════════════════════════════"
echo "  Experiment:  ${EXP_NAME}"
echo "  Mode:        distillation (35M student)"
echo "  Config:      ${CONFIG_PATH}"
echo "  Teacher:     ${NEMO_FILE}"
echo "  GPUs:        ${GPU_COUNT}"
echo "══════════════════════════════════════════"
echo ""

if [[ "${DETACH}" == "true" ]]; then
  docker run -d "${DOCKER_ARGS[@]}" nemo-training:latest bash -c "${TRAIN_CMD}"
  echo ""
  echo "==> Distillation started in background."
  echo ""
  echo "Monitor:"
  echo "  docker logs -f ${EXP_NAME}"
  echo "  docker logs ${EXP_NAME} 2>&1 | grep -E 'val_wer|ctc_loss|kd_loss'"
  echo ""
  echo "TensorBoard:"
  echo "  $(dirname "$0")/tensorboard.sh --exp-dir ${TRAINING_ROOT}/experiments/"
else
  docker run --rm "${DOCKER_ARGS[@]}" nemo-training:latest bash -c "${TRAIN_CMD}"
fi
