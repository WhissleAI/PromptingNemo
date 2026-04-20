#!/usr/bin/env bash
# Resume training after spot preemption.
# Re-attaches the persistent disk to a new (or restarted) instance and resumes
# from the latest checkpoint.
#
# Usage:
#   ./resume-training.sh --name zh-adapter-v1                          # Resume on current instance
#   ./resume-training.sh --name zh-adapter-v1 --instance nemo-spot-2   # Resume on different instance
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="${TRAINING_ROOT:-/mnt/training}"
PROJECT="${GCP_PROJECT:-deepvoice-468015}"
ZONE="${GCP_ZONE:-us-central1-c}"
TRAINING_DISK="${TRAINING_DISK:-nemo-training-disk}"
GCP_USER="${GCP_USER:?Set GCP_USER to your Whissle username (e.g. export GCP_USER=yourname)}"

EXP_NAME=""
INSTANCE=""
ATTACH_DISK="false"

usage() {
  cat <<EOF
Usage: $0 --name EXP_NAME [OPTIONS]

Options:
  --name NAME           Experiment name to resume
  --instance INSTANCE   Target instance (default: current host)
  --attach-disk         Attach the training disk to the instance first
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --name) EXP_NAME="$2"; shift 2 ;;
    --instance) INSTANCE="$2"; shift 2 ;;
    --attach-disk) ATTACH_DISK="true"; shift ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "${EXP_NAME}" ]] && { echo "ERROR: --name is required"; usage; }

# ── Optionally attach disk ───────────────────────────────────

if [[ "${ATTACH_DISK}" == "true" && -n "${INSTANCE}" ]]; then
  echo "==> Attaching ${TRAINING_DISK} to ${INSTANCE}..."

  gcloud compute instances attach-disk "${INSTANCE}" \
    --disk="${TRAINING_DISK}" \
    --device-name=nemo-training-disk \
    --zone="${ZONE}" \
    --project="${PROJECT}" 2>/dev/null \
    || echo "    Disk already attached or instance not ready"

  echo "==> Mounting disk..."
  gcloud compute ssh "${GCP_USER}@${INSTANCE}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --command="
      sudo mkdir -p /mnt/training
      DISK_DEV=/dev/disk/by-id/google-nemo-training-disk
      if [ -e \"\${DISK_DEV}\" ]; then
        sudo mount -o discard,defaults \"\${DISK_DEV}\" /mnt/training 2>/dev/null || echo 'Already mounted'
        sudo chmod 777 /mnt/training
      else
        echo 'ERROR: Disk device not found'
        exit 1
      fi
    "
fi

# ── Find latest checkpoint ──────────────────────────────────

run_cmd() {
  if [[ -n "${INSTANCE}" ]]; then
    gcloud compute ssh "${GCP_USER}@${INSTANCE}" \
      --zone="${ZONE}" --project="${PROJECT}" \
      --command="$1"
  else
    bash -c "$1"
  fi
}

EXP_DIR="${TRAINING_ROOT}/experiments/${EXP_NAME}"
CONFIG_PATH="${EXP_DIR}/config.yml"

echo "==> Looking for checkpoints in ${EXP_DIR}..."

LATEST_CKPT=$(run_cmd "find ${EXP_DIR} -name '*.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-" 2>/dev/null || true)

if [[ -z "${LATEST_CKPT}" ]]; then
  echo "    No checkpoints found. Starting fresh training."
  RESUME_ARG=""
else
  echo "    Found checkpoint: ${LATEST_CKPT}"
  RESUME_ARG="--resume ${LATEST_CKPT}"
fi

# Verify config exists
CONFIG_EXISTS=$(run_cmd "test -f ${CONFIG_PATH} && echo yes || echo no" 2>/dev/null)
if [[ "${CONFIG_EXISTS}" != "yes" ]]; then
  echo "ERROR: Config not found at ${CONFIG_PATH}"
  echo "Cannot resume without the original config. Check experiment directory."
  exit 1
fi

echo ""
echo "==> Resuming experiment: ${EXP_NAME}"

if [[ -n "${INSTANCE}" ]]; then
  gcloud compute ssh "${GCP_USER}@${INSTANCE}" \
    --zone="${ZONE}" --project="${PROJECT}" \
    --command="
      cd ${TRAINING_ROOT}/code/PromptingNemo/recipes/meta_asr/gcp && \
      TRAINING_ROOT=${TRAINING_ROOT} ./run-finetune.sh --name ${EXP_NAME} --config ${CONFIG_PATH} ${RESUME_ARG}
    "
else
  cd "${SCRIPT_DIR}"
  TRAINING_ROOT="${TRAINING_ROOT}" ./run-finetune.sh --name "${EXP_NAME}" --config "${CONFIG_PATH}" ${RESUME_ARG}
fi
