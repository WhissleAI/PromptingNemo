#!/usr/bin/env bash
# Upload a fine-tuned NeMo model to HuggingFace.
#
# Usage:
#   ./upload-model.sh --name zh-adapter-v1 --hf-repo WhissleAI/STT-meta-1B-zh --hf-token hf_xxx
#   ./upload-model.sh --name zh-adapter-v1 --hf-repo WhissleAI/STT-meta-1B-zh --hf-token hf_xxx --model-name "STT Meta 1B Chinese Adapter"
set -euo pipefail

TRAINING_ROOT="${TRAINING_ROOT:-/mnt/training}"

EXP_NAME=""
HF_REPO=""
HF_TOKEN=""
MODEL_NAME=""
INCLUDE_ADAPTER_ONLY="false"

usage() {
  cat <<EOF
Usage: $0 --name EXP_NAME --hf-repo REPO --hf-token TOKEN [OPTIONS]

Required:
  --name NAME           Experiment name (matches run-finetune.sh --name)
  --hf-repo REPO        HuggingFace repo to push to (e.g. WhissleAI/STT-meta-1B-zh)
  --hf-token TOKEN       HuggingFace write token

Options:
  --model-name NAME     Human-readable model name for the model card
  --adapter-only        Only upload adapter weights (not full .nemo)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --name) EXP_NAME="$2"; shift 2 ;;
    --hf-repo) HF_REPO="$2"; shift 2 ;;
    --hf-token) HF_TOKEN="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --adapter-only) INCLUDE_ADAPTER_ONLY="true"; shift ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "${EXP_NAME}" ]] && { echo "ERROR: --name is required"; usage; }
[[ -z "${HF_REPO}" ]] && { echo "ERROR: --hf-repo is required"; usage; }
[[ -z "${HF_TOKEN}" ]] && { echo "ERROR: --hf-token is required"; usage; }

EXP_DIR="${TRAINING_ROOT}/experiments/${EXP_NAME}"
[[ -z "${MODEL_NAME}" ]] && MODEL_NAME="${EXP_NAME}"

# Find best .nemo file
NEMO_FILE=$(find "${EXP_DIR}" -name "*.nemo" -type f -not -name "*-v[0-9]*" 2>/dev/null | head -1)
if [[ -z "${NEMO_FILE}" ]]; then
  NEMO_FILE=$(find "${EXP_DIR}" -name "*.nemo" -type f 2>/dev/null | sort -t= -k2 -n | head -1)
fi

if [[ -z "${NEMO_FILE}" ]]; then
  echo "ERROR: No .nemo file found in ${EXP_DIR}"
  exit 1
fi

echo "==> Found model: ${NEMO_FILE}"
NEMO_SIZE=$(du -h "${NEMO_FILE}" | cut -f1)
echo "    Size: ${NEMO_SIZE}"

# Find benchmark results if available
BENCHMARK_FILE="${EXP_DIR}/benchmark_results.json"
BENCH_INFO=""
if [[ -f "${BENCHMARK_FILE}" ]]; then
  BENCH_INFO=$(cat "${BENCHMARK_FILE}")
  echo "==> Found benchmark results"
fi

# Find config
CONFIG_FILE=$(find "${EXP_DIR}" -name "config.yml" -type f 2>/dev/null | head -1)

# Find best WER from checkpoints
BEST_WER=$(find "${EXP_DIR}" -name "*val_wer*" -type f 2>/dev/null | grep -oP 'val_wer=\K[0-9.]+' | sort -n | head -1 || echo "N/A")
echo "    Best val_wer: ${BEST_WER}"

# Prepare upload directory
UPLOAD_DIR=$(mktemp -d)
trap "rm -rf ${UPLOAD_DIR}" EXIT

cp "${NEMO_FILE}" "${UPLOAD_DIR}/"
[[ -f "${CONFIG_FILE}" ]] && cp "${CONFIG_FILE}" "${UPLOAD_DIR}/"
[[ -f "${BENCHMARK_FILE}" ]] && cp "${BENCHMARK_FILE}" "${UPLOAD_DIR}/"

# Copy training logs
LOGS_DIR="${EXP_DIR}/logs"
if [[ -d "${LOGS_DIR}" ]]; then
  mkdir -p "${UPLOAD_DIR}/logs"
  cp "${LOGS_DIR}"/*.json "${UPLOAD_DIR}/logs/" 2>/dev/null || true
fi

# Generate model card
NEMO_BASENAME=$(basename "${NEMO_FILE}")
cat > "${UPLOAD_DIR}/README.md" <<CARDEOF
---
language:
- multilingual
library_name: nemo
tags:
- automatic-speech-recognition
- NeMo
- WhissleAI
- meta-asr
---

# ${MODEL_NAME}

Fine-tuned from [WhissleAI/STT-meta-1B](https://huggingface.co/WhissleAI/STT-meta-1B) using adapter-based fine-tuning with PromptingNemo.

## Model Details

- **Base model:** WhissleAI/STT-meta-1B
- **Architecture:** FastConformer CTC BPE with adapter layers
- **Best val_wer:** ${BEST_WER}
- **Training framework:** NeMo 2.7 + PyTorch Lightning

## Usage

\`\`\`python
from nemo.collections.asr.models import ASRModel

model = ASRModel.restore_from("${NEMO_BASENAME}")
transcription = model.transcribe(["audio.wav"])
\`\`\`

## Training

Fine-tuned using [PromptingNemo](https://github.com/WhissleAI/PromptingNemo) training scripts.

$(if [[ -n "${BENCH_INFO}" ]]; then
echo "## Benchmark Results"
echo ""
echo '```json'
echo "${BENCH_INFO}"
echo '```'
fi)
CARDEOF

echo ""
echo "==> Uploading to ${HF_REPO}..."

UPLOAD_SCRIPT=$(cat <<'PYEOF'
import sys
import os
from huggingface_hub import HfApi, create_repo

hf_repo = sys.argv[1]
hf_token = sys.argv[2]
upload_dir = sys.argv[3]

api = HfApi(token=hf_token)

try:
    create_repo(repo_id=hf_repo, token=hf_token, repo_type="model", exist_ok=True)
    print(f"Repository {hf_repo} ready.")
except Exception as e:
    print(f"Note: {e}")

api.upload_folder(
    folder_path=upload_dir,
    repo_id=hf_repo,
    repo_type="model",
    commit_message=f"Upload fine-tuned model from experiment",
)

print(f"\nModel uploaded to: https://huggingface.co/{hf_repo}")
PYEOF
)

docker run --rm \
  -v "${UPLOAD_DIR}:/upload" \
  -v "${TRAINING_ROOT}:${TRAINING_ROOT}" \
  nemo-training:latest \
  python -c "${UPLOAD_SCRIPT}" "${HF_REPO}" "${HF_TOKEN}" "/upload"

echo ""
echo "══════════════════════════════════════════"
echo "  Model uploaded to: https://huggingface.co/${HF_REPO}"
echo "══════════════════════════════════════════"
