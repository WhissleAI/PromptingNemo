#!/usr/bin/env bash
# One-time: create a persistent SSD for training data, models, and experiments.
# This disk survives spot instance preemption and can be attached to any instance.
#
# Usage: ./create-training-disk.sh [--size 500] [--name nemo-training-disk]
set -euo pipefail

PROJECT="${GCP_PROJECT:-deepvoice-468015}"
ZONE="${GCP_ZONE:-us-central1-c}"
DISK_NAME="${1:-nemo-training-disk}"
DISK_SIZE="${2:-500}"

echo "==> Creating persistent SSD: ${DISK_NAME} (${DISK_SIZE}GB)"
gcloud compute disks create "${DISK_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --size="${DISK_SIZE}GB" \
  --type=pd-ssd \
  --description="Persistent training disk for NeMo ASR experiments"

echo ""
echo "Disk created. Attach to an instance with:"
echo "  gcloud compute instances attach-disk INSTANCE --disk=${DISK_NAME} --zone=${ZONE}"
echo ""
echo "Then format and mount (first time only):"
echo "  sudo mkfs.ext4 -m 0 -E lazy_itable_init=0 /dev/sdb"
echo "  sudo mkdir -p /mnt/training"
echo "  sudo mount /dev/sdb /mnt/training"
echo "  sudo chown \$(whoami) /mnt/training"
