#!/usr/bin/env bash
# End-to-end Hindi fine-tuning experiment:
#   1. Wait for data download to finish (if still running)
#   2. Run adapter fine-tuning
#   3. Benchmark the model
#   4. Upload to HuggingFace
#
# Usage: ./run-hindi-experiment.sh --hf-repo WhissleAI/MY_MODEL_NAME --hf-token hf_xxx
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="${TRAINING_ROOT:-/mnt/training}"

EXP_NAME="hi-set1-meta-asr-meta1b-adapter"
MODEL_REPO="WhissleAI/STT-meta-1B"
DATASET_REPO="WhissleAI/Meta_STT_HI_Set1"
LANG="HINDI"
MODE="adapter"
HF_REPO=""
HF_TOKEN=""
MODEL_DISPLAY_NAME=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --hf-repo) HF_REPO="$2"; shift 2 ;;
    --hf-token) HF_TOKEN="$2"; shift 2 ;;
    --model-name) MODEL_DISPLAY_NAME="$2"; shift 2 ;;
    --name) EXP_NAME="$2"; shift 2 ;;
    *) shift ;;
  esac
done

[[ -z "${HF_REPO}" ]] && { echo "ERROR: --hf-repo is required (e.g. WhissleAI/STT-meta-1B-hi)"; exit 1; }
[[ -z "${HF_TOKEN}" ]] && { echo "ERROR: --hf-token is required"; exit 1; }
[[ -z "${MODEL_DISPLAY_NAME}" ]] && MODEL_DISPLAY_NAME="${HF_REPO##*/}"

DATA_DIR="${TRAINING_ROOT}/data/meta_stt_hi_set1"

echo "══════════════════════════════════════════"
echo "  Hindi Fine-Tuning Experiment"
echo "  Experiment: ${EXP_NAME}"
echo "  Model: ${MODEL_REPO}"
echo "  Dataset: ${DATASET_REPO}"
echo "  Upload to: ${HF_REPO}"
echo "══════════════════════════════════════════"

# ── Step 1: Wait for data ────────────────────────────────────
echo ""
echo "==> Step 1: Checking data..."

while [[ ! -f "${DATA_DIR}/train.json" ]]; do
  if pgrep -f "download_hf_dataset" >/dev/null 2>&1 || pgrep -f "download-data.sh" >/dev/null 2>&1; then
    echo "    Data download in progress... waiting 60s"
    sleep 60
  else
    echo "    Data not ready and no download running. Starting download..."
    cd "${SCRIPT_DIR}"
    TRAINING_ROOT="${TRAINING_ROOT}" ./download-data.sh --dataset "${DATASET_REPO}" --lang "${LANG}"
    break
  fi
done

TRAIN_COUNT=$(wc -l < "${DATA_DIR}/train.json")
echo "    Data ready: ${TRAIN_COUNT} training samples"

if [[ -f "${DATA_DIR}/valid.json" ]]; then
  VAL_COUNT=$(wc -l < "${DATA_DIR}/valid.json")
  echo "    Validation: ${VAL_COUNT} samples"
fi

if [[ -f "${DATA_DIR}/test.json" ]]; then
  TEST_COUNT=$(wc -l < "${DATA_DIR}/test.json")
  echo "    Test: ${TEST_COUNT} samples"
fi

# ── Step 2: Fine-tune ───────────────────────────────────────
echo ""
echo "==> Step 2: Starting fine-tuning..."
cd "${SCRIPT_DIR}"
TRAINING_ROOT="${TRAINING_ROOT}" ./run-finetune.sh \
  --model "${MODEL_REPO}" \
  --dataset "${DATASET_REPO}" \
  --lang "${LANG}" \
  --mode "${MODE}" \
  --name "${EXP_NAME}" \
  --batch-size 16 \
  --no-detach

# ── Step 3: Benchmark ───────────────────────────────────────
echo ""
echo "==> Step 3: Running benchmark..."
cd "${SCRIPT_DIR}"
TRAINING_ROOT="${TRAINING_ROOT}" ./benchmark.sh --name "${EXP_NAME}" --batch-size 32

# ── Step 4: Upload ───────────────────────────────────────────
echo ""
echo "==> Step 4: Uploading to HuggingFace..."
cd "${SCRIPT_DIR}"
TRAINING_ROOT="${TRAINING_ROOT}" ./upload-model.sh \
  --name "${EXP_NAME}" \
  --hf-repo "${HF_REPO}" \
  --hf-token "${HF_TOKEN}" \
  --model-name "${MODEL_DISPLAY_NAME}"

echo ""
echo "══════════════════════════════════════════"
echo "  EXPERIMENT COMPLETE"
echo "  Model: https://huggingface.co/${HF_REPO}"
echo "  Logs: ${TRAINING_ROOT}/code/PromptingNemo/scripts/asr/meta-asr/logs/"
echo "══════════════════════════════════════════"
