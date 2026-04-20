#!/usr/bin/env bash
# Download a WhissleAI HuggingFace dataset and prepare NeMo manifests.
# Runs inside Docker on the training instance.
#
# Usage:
#   ./download-data.sh --dataset WhissleAI/Meta_STT_ZH_AIShell3 --lang MANDARIN
#   ./download-data.sh --dataset WhissleAI/Meta_STT_EN_Welness_podcast --lang EN
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="${TRAINING_ROOT:-/mnt/training}"

DATASET=""
LANG=""
FAMILY=""
VAL_RATIO="0.05"

usage() {
  echo "Usage: $0 --dataset HF_DATASET_NAME --lang LANGUAGE_TAG [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --dataset NAME     HuggingFace dataset (e.g. WhissleAI/Meta_STT_ZH_AIShell3)"
  echo "  --lang TAG         Language tag (e.g. MANDARIN, EN, HINDI)"
  echo "  --family TAG       Language family for tokenizer (e.g. INDO_ARYAN). Auto-resolved if omitted."
  echo "  --val-ratio FLOAT  Validation split ratio (default: 0.05)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset) DATASET="$2"; shift 2 ;;
    --lang) LANG="$2"; shift 2 ;;
    --family) FAMILY="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "${DATASET}" ]] && { echo "ERROR: --dataset is required"; usage; }
[[ -z "${LANG}" ]] && { echo "ERROR: --lang is required"; usage; }

# Derive output directory from dataset name
DATASET_SLUG=$(echo "${DATASET}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
OUTPUT_DIR="${TRAINING_ROOT}/data/${DATASET_SLUG}"

if [[ -f "${OUTPUT_DIR}/train.json" && -f "${OUTPUT_DIR}/valid.json" ]]; then
  TRAIN_COUNT=$(wc -l < "${OUTPUT_DIR}/train.json")
  VAL_COUNT=$(wc -l < "${OUTPUT_DIR}/valid.json")
  echo "Dataset already prepared at ${OUTPUT_DIR}"
  echo "  train: ${TRAIN_COUNT} samples, valid: ${VAL_COUNT} samples"
  echo "  To re-download, delete ${OUTPUT_DIR} and re-run."
  exit 0
fi

echo "==> Downloading dataset: ${DATASET}"
echo "    Language: ${LANG}"
echo "    Output: ${OUTPUT_DIR}"
echo ""

UTILS_DIR="${SCRIPT_DIR}/../../../scripts/asr/meta-asr/utils"

docker run --rm \
  -v "${TRAINING_ROOT}:${TRAINING_ROOT}" \
  -v "${UTILS_DIR}:/scripts" \
  -e HF_HOME="${TRAINING_ROOT}/.hf_cache" \
  nemo-training:latest \
  python /scripts/download_hf_dataset.py \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --lang "${LANG}" \
    ${FAMILY:+--family "${FAMILY}"} \
    --val-ratio "${VAL_RATIO}"

echo ""
echo "==> Dataset ready at ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"/*.json
