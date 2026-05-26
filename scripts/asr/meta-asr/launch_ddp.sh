#!/bin/bash
# Launch 2-node DDP training for Whisper fine-tuning
# Run this script on BOTH VMs simultaneously.
#
# Usage:
#   On nemo-whisper-a100 (master, node 0):
#     bash launch_ddp.sh 0
#
#   On nemo-vakyansh-a100 (node 1):
#     bash launch_ddp.sh 1
#
set -euo pipefail

NODE_RANK="${1:?Usage: launch_ddp.sh <0|1>}"

# Master node IP (nemo-whisper-a100 internal IP)
MASTER_ADDR="10.128.0.31"
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=1

SCRIPT=/mnt/nfs/code/PromptingNemo/scripts/asr/meta-asr/finetune_whisper.py
CONFIG=/mnt/nfs/experiments/finetune_whisper_multilingual.yaml
LOG_DIR=/mnt/nfs/experiments/whisper-turbo-multilingual-v1/logs
mkdir -p "$LOG_DIR"

echo "=== Node ${NODE_RANK} starting DDP training ==="
echo "  MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "  NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}"

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  ${SCRIPT} \
  --config ${CONFIG} \
  2>&1 | stdbuf -oL tee "${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
