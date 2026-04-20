#!/usr/bin/env bash
# Download a pretrained .nemo model from HuggingFace.
#
# Usage:
#   ./download-model.sh --model WhissleAI/STT-meta-1B
#   ./download-model.sh --model WhissleAI/STT-meta-1B --filename stt_meta_1b.nemo
set -euo pipefail

TRAINING_ROOT="${TRAINING_ROOT:-/mnt/training}"
MODELS_DIR="${TRAINING_ROOT}/models"

MODEL_REPO=""
FILENAME=""

usage() {
  echo "Usage: $0 --model HF_REPO [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --model REPO       HuggingFace model repo (e.g. WhissleAI/STT-meta-1B)"
  echo "  --filename FILE    Specific .nemo file to download (auto-detected if omitted)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL_REPO="$2"; shift 2 ;;
    --filename) FILENAME="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "${MODEL_REPO}" ]] && { echo "ERROR: --model is required"; usage; }

MODEL_SLUG=$(echo "${MODEL_REPO}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
MODEL_DIR="${MODELS_DIR}/${MODEL_SLUG}"
mkdir -p "${MODEL_DIR}"

# Check if already downloaded
EXISTING=$(find "${MODEL_DIR}" -name "*.nemo" -type f 2>/dev/null | head -1)
if [[ -n "${EXISTING}" ]]; then
  echo "Model already downloaded: ${EXISTING}"
  echo "  To re-download, delete ${MODEL_DIR} and re-run."
  exit 0
fi

echo "==> Downloading model: ${MODEL_REPO}"
echo "    Destination: ${MODEL_DIR}"
echo ""

DOWNLOAD_SCRIPT=$(cat <<'PYEOF'
import sys
import os
from huggingface_hub import snapshot_download, list_repo_files

repo_id = sys.argv[1]
local_dir = sys.argv[2]
filename_filter = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None

files = list_repo_files(repo_id)
nemo_files = [f for f in files if f.endswith('.nemo')]
yaml_files = [f for f in files if f.endswith('.yaml') or f.endswith('.yml')]
json_files = [f for f in files if f.endswith('.json') and 'config' in f.lower()]

if filename_filter:
    allow = [filename_filter] + yaml_files + json_files
    snapshot_download(repo_id, local_dir=local_dir, allow_patterns=allow)
elif nemo_files:
    allow = nemo_files + yaml_files + json_files
    snapshot_download(repo_id, local_dir=local_dir, allow_patterns=allow)
else:
    snapshot_download(repo_id, local_dir=local_dir)

downloaded = []
for root, dirs, fnames in os.walk(local_dir):
    for f in fnames:
        if f.endswith('.nemo'):
            downloaded.append(os.path.join(root, f))

if downloaded:
    print(f"\nDownloaded {len(downloaded)} .nemo file(s):")
    for p in downloaded:
        size_gb = os.path.getsize(p) / (1024**3)
        print(f"  {p}  ({size_gb:.2f} GB)")
else:
    print(f"\nNo .nemo files found. Contents of {local_dir}:")
    for root, dirs, fnames in os.walk(local_dir):
        for f in fnames:
            print(f"  {os.path.join(root, f)}")
PYEOF
)

docker run --rm \
  -v "${TRAINING_ROOT}:${TRAINING_ROOT}" \
  -e HF_HOME="${TRAINING_ROOT}/.hf_cache" \
  nemo-training:latest \
  python -c "${DOWNLOAD_SCRIPT}" "${MODEL_REPO}" "${MODEL_DIR}" "${FILENAME}"

echo ""
echo "==> Model ready at ${MODEL_DIR}"
find "${MODEL_DIR}" -name "*.nemo" -exec ls -lh {} \;
