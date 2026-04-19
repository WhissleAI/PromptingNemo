#!/usr/bin/env bash
# Launch a spot instance with the training disk attached, ready for fine-tuning.
#
# Usage:
#   ./launch-experiment.sh --name my-exp --gpu t4
#   ./launch-experiment.sh --name my-exp --gpu t4 --machine n1-standard-16
#
# After launch, SSH in and run:
#   /mnt/training/code/PromptingNemo/scripts/asr/meta-asr/gcp/run-finetune.sh --help
set -euo pipefail

PROJECT="${GCP_PROJECT:-deepvoice-468015}"
ZONE="${GCP_ZONE:-us-central1-c}"
TRAINING_DISK="${TRAINING_DISK:-nemo-training-disk}"

# Defaults
INSTANCE_NAME=""
MACHINE_TYPE=""
DEFAULT_MACHINE="n1-standard-8"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="50GB"

usage() {
  echo "Usage: $0 --name INSTANCE_NAME [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --name NAME          Instance name (required)"
  echo "  --gpu TYPE           GPU type: t4, v100, a100 (default: t4)"
  echo "  --gpu-count N        Number of GPUs (default: 1)"
  echo "  --machine TYPE       Machine type (default: n1-standard-8)"
  echo "  --boot-size SIZE     Boot disk size (default: 50GB)"
  echo "  --disk DISK          Training disk name (default: nemo-training-disk)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --name) INSTANCE_NAME="$2"; shift 2 ;;
    --gpu)
      case "$2" in
        t4)   GPU_TYPE="nvidia-tesla-t4"; DEFAULT_MACHINE="n1-standard-8" ;;
        v100) GPU_TYPE="nvidia-tesla-v100"; DEFAULT_MACHINE="n1-standard-8" ;;
        l4)   GPU_TYPE="nvidia-l4"; DEFAULT_MACHINE="g2-standard-12" ;;
        a100) GPU_TYPE="nvidia-tesla-a100"; DEFAULT_MACHINE="a2-highgpu-1g" ;;
        a100-80) GPU_TYPE="nvidia-a100-80gb"; DEFAULT_MACHINE="a2-ultragpu-1g" ;;
        h100) GPU_TYPE="nvidia-h100-80gb"; DEFAULT_MACHINE="a3-highgpu-1g" ;;
        *)    GPU_TYPE="$2"; DEFAULT_MACHINE="" ;;
      esac
      shift 2 ;;
    --gpu-count) GPU_COUNT="$2"; shift 2 ;;
    --machine) MACHINE_TYPE="$2"; DEFAULT_MACHINE="$2"; shift 2 ;;
    --boot-size) BOOT_DISK_SIZE="$2"; shift 2 ;;
    --disk) TRAINING_DISK="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "${INSTANCE_NAME}" ]] && { echo "ERROR: --name is required"; usage; }

# Use default machine type if not explicitly set
[[ -z "${MACHINE_TYPE}" ]] && MACHINE_TYPE="${DEFAULT_MACHINE}"

echo "==> Creating spot instance: ${INSTANCE_NAME}"
echo "    GPU: ${GPU_TYPE} x${GPU_COUNT}"
echo "    Machine: ${MACHINE_TYPE}"
echo "    Training disk: ${TRAINING_DISK}"

STARTUP_SCRIPT='#!/bin/bash
DISK_DEV="/dev/disk/by-id/google-nemo-training-disk"
MOUNT_POINT="/mnt/training"

# Wait for disk device
for i in $(seq 1 30); do
  [ -e "${DISK_DEV}" ] && break
  sleep 2
done

mkdir -p "${MOUNT_POINT}"

# Format only if not already formatted
if ! blkid "${DISK_DEV}" | grep -q ext4; then
  mkfs.ext4 -m 0 -E lazy_itable_init=0 "${DISK_DEV}"
fi

mount -o discard,defaults "${DISK_DEV}" "${MOUNT_POINT}"
chmod 777 "${MOUNT_POINT}"

# Create directory structure
mkdir -p "${MOUNT_POINT}"/{models,data,experiments,code,docker}
chmod -R 777 "${MOUNT_POINT}"

# Add to fstab for remount after reboot
grep -q "${MOUNT_POINT}" /etc/fstab || \
  echo "${DISK_DEV} ${MOUNT_POINT} ext4 discard,defaults,nofail 0 2" >> /etc/fstab
'

gcloud compute instances create "${INSTANCE_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --boot-disk-size="${BOOT_DISK_SIZE}" \
  --boot-disk-type=pd-standard \
  --image-family=pytorch-2-4-cu124-debian-12 \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=True" \
  --metadata-from-file="startup-script=<(echo '${STARTUP_SCRIPT}')" \
  --disk="name=${TRAINING_DISK},device-name=nemo-training-disk,mode=rw,auto-delete=no" \
  --scopes=cloud-platform

echo ""
echo "==> Waiting for instance to be ready..."
sleep 45

echo "==> Instance ready!"
echo ""
echo "Connect:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT}"
echo ""
echo "Training disk mounted at /mnt/training/"
echo ""
echo "Start training:"
echo "  /mnt/training/code/PromptingNemo/scripts/asr/meta-asr/gcp/run-finetune.sh \\"
echo "    --model WhissleAI/STT-meta-1B --dataset WhissleAI/Meta_STT_ZH_AIShell3 --mode adapter"
