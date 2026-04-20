#!/usr/bin/env bash
# One-time setup for a GCP instance: install Docker, NVIDIA drivers, mount disk.
# Run this after creating an instance with launch-experiment.sh, or on any
# existing instance that needs to be configured for training.
#
# Usage: ./setup-instance.sh [--disk nemo-training-disk]
set -euo pipefail

TRAINING_DISK="${1:-nemo-training-disk}"
MOUNT_POINT="/mnt/training"

echo "==> Installing Docker..."
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  echo "    Docker installed. You may need to re-login for group changes."
else
  echo "    Docker already installed."
fi

echo ""
echo "==> Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-container-cli &>/dev/null; then
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  echo "    NVIDIA Container Toolkit installed."
else
  echo "    NVIDIA Container Toolkit already installed."
fi

echo ""
echo "==> Mounting training disk..."
DISK_DEV="/dev/disk/by-id/google-${TRAINING_DISK}"

if [ ! -e "${DISK_DEV}" ]; then
  echo "    WARNING: Disk device ${DISK_DEV} not found."
  echo "    Make sure the disk is attached to this instance."
  exit 1
fi

sudo mkdir -p "${MOUNT_POINT}"

if ! blkid "${DISK_DEV}" 2>/dev/null | grep -q ext4; then
  echo "    Formatting disk (first time)..."
  sudo mkfs.ext4 -m 0 -E lazy_itable_init=0 "${DISK_DEV}"
fi

if ! mountpoint -q "${MOUNT_POINT}"; then
  sudo mount -o discard,defaults "${DISK_DEV}" "${MOUNT_POINT}"
fi
sudo chmod 777 "${MOUNT_POINT}"

mkdir -p "${MOUNT_POINT}"/{models,data,experiments,code,docker,.hf_cache}

grep -q "${MOUNT_POINT}" /etc/fstab 2>/dev/null || \
  echo "${DISK_DEV} ${MOUNT_POINT} ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab

echo "    Disk mounted at ${MOUNT_POINT}"
df -h "${MOUNT_POINT}"

echo ""
echo "==> Building Docker image..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../../.."
DOCKER_DIR="${REPO_ROOT}/docker"
if [[ -d "${DOCKER_DIR}" ]]; then
  docker build -t nemo-training:latest -f "${DOCKER_DIR}/Dockerfile.training" "${REPO_ROOT}"
  echo "    Image built: nemo-training:latest"
else
  echo "    WARNING: Docker directory not found at ${DOCKER_DIR}"
  echo "    Copy the training scripts to ${MOUNT_POINT}/code/ first."
fi

echo ""
echo "==> Verifying GPU..."
docker run --rm --gpus all nemo-training:latest nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || \
  echo "    WARNING: GPU not accessible in Docker. Check NVIDIA drivers."

echo ""
echo "══════════════════════════════════════════"
echo "  Instance ready for training!"
echo ""
echo "  Training disk: ${MOUNT_POINT}"
echo "  Directory structure:"
ls -1 "${MOUNT_POINT}/" | sed 's/^/    /'
echo ""
echo "  Next steps:"
echo "    1. Download model:  ./download-model.sh --model WhissleAI/STT-meta-1B"
echo "    2. Download data:   ./download-data.sh --dataset WhissleAI/Meta_STT_ZH_AIShell3 --lang MANDARIN"
echo "    3. Start training:  ./run-finetune.sh --model WhissleAI/STT-meta-1B \\"
echo "                          --dataset WhissleAI/Meta_STT_ZH_AIShell3 --lang MANDARIN \\"
echo "                          --mode adapter --name my-experiment"
echo "══════════════════════════════════════════"
