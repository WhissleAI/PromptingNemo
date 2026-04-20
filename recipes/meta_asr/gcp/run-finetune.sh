#!/usr/bin/env bash
# Run NeMo fine-tuning inside Docker. Supports full and adapter modes.
# Handles config generation, tokenizer training, and model training.
#
# Usage:
#   ./run-finetune.sh --model WhissleAI/STT-meta-1B --dataset WhissleAI/Meta_STT_ZH_AIShell3 --lang MANDARIN --mode adapter --name zh-adapter-v1
#   ./run-finetune.sh --model WhissleAI/STT-meta-1B --dataset WhissleAI/Meta_STT_ZH_AIShell3 --lang MANDARIN --mode full --name zh-full-v1
#   ./run-finetune.sh --config /path/to/existing/config.yml --name zh-custom
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="${TRAINING_ROOT:-/mnt/training}"
CODE_DIR="${TRAINING_ROOT}/code/PromptingNemo"

# Defaults
MODEL_REPO=""
DATASET_REPO=""
LANG=""
MODE="adapter"
EXP_NAME=""
CONFIG_PATH=""
BATCH_SIZE=""
LR=""
MAX_STEPS=""
GPU_COUNT=1
ADAPTER_DIM=128
STREAMING="false"
DETACH="true"
RESUME_FROM=""

usage() {
  cat <<EOF
Usage: $0 --name EXP_NAME [OPTIONS]

Required (either --config OR --model + --dataset + --lang):
  --name NAME           Experiment name (used for Docker container and TB logs)
  --config PATH         Path to existing YAML config (skips config generation)
  --model REPO          HuggingFace model repo (e.g. WhissleAI/STT-meta-1B)
  --dataset REPO        HuggingFace dataset repo (e.g. WhissleAI/Meta_STT_ZH_AIShell3)
  --lang TAG            Language tag (e.g. MANDARIN, EN, HINDI)

Options:
  --mode MODE           Training mode: adapter or full (default: adapter)
  --batch-size N        Override batch size
  --lr FLOAT            Override learning rate
  --max-steps N         Override max training steps
  --adapter-dim N       Adapter hidden dim (default: 128, only for adapter mode)
  --gpus N              Number of GPUs (default: 1)
  --resume PATH         Resume from checkpoint path
  --streaming           Enable hybrid streaming attention (causal + multi-lookahead)
  --no-detach           Run in foreground (default: detached)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --name) EXP_NAME="$2"; shift 2 ;;
    --model) MODEL_REPO="$2"; shift 2 ;;
    --dataset) DATASET_REPO="$2"; shift 2 ;;
    --lang) LANG="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --adapter-dim) ADAPTER_DIM="$2"; shift 2 ;;
    --gpus) GPU_COUNT="$2"; shift 2 ;;
    --resume) RESUME_FROM="$2"; shift 2 ;;
    --streaming) STREAMING="true"; shift ;;
    --no-detach) DETACH="false"; shift ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "${EXP_NAME}" ]] && { echo "ERROR: --name is required"; usage; }
[[ "${MODE}" != "adapter" && "${MODE}" != "full" ]] && { echo "ERROR: --mode must be 'adapter' or 'full'"; usage; }

# ── Resolve paths ──────────────────────────────────────────────

if [[ -z "${CONFIG_PATH}" ]]; then
  [[ -z "${MODEL_REPO}" ]] && { echo "ERROR: --model is required when --config is not provided"; usage; }
  [[ -z "${DATASET_REPO}" ]] && { echo "ERROR: --dataset is required when --config is not provided"; usage; }
  [[ -z "${LANG}" ]] && { echo "ERROR: --lang is required when --config is not provided"; usage; }

  MODEL_SLUG=$(echo "${MODEL_REPO}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  DATASET_SLUG=$(echo "${DATASET_REPO}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

  MODEL_DIR="${TRAINING_ROOT}/models/${MODEL_SLUG}"
  DATA_DIR="${TRAINING_ROOT}/data/${DATASET_SLUG}"

  # Verify model exists
  NEMO_FILE=$(find "${MODEL_DIR}" -name "*.nemo" -type f 2>/dev/null | head -1)
  if [[ -z "${NEMO_FILE}" ]]; then
    echo "ERROR: No .nemo file found in ${MODEL_DIR}"
    echo "Run download-model.sh first: ./download-model.sh --model ${MODEL_REPO}"
    exit 1
  fi
  NEMO_FILENAME=$(basename "${NEMO_FILE}")

  # Verify data exists
  if [[ ! -f "${DATA_DIR}/train.json" ]]; then
    echo "ERROR: No train.json found in ${DATA_DIR}"
    echo "Run download-data.sh first: ./download-data.sh --dataset ${DATASET_REPO} --lang ${LANG}"
    exit 1
  fi

  # ── Generate config ────────────────────────────────────────────

  LANG_UPPER=$(echo "${LANG}" | tr '[:lower:]' '[:upper:]')
  LANG_LOWER=$(echo "${LANG}" | tr '[:upper:]' '[:lower:]')

  # Map individual languages to the base model's tokenizer family.
  # The STT-meta-1B aggregate tokenizer uses family-level keys:
  #   INDO_ARYAN, MANDARIN, DRAVIDIAN, GERMANIC, ROMANCE, SLAVIC, JAPONIC
  # When fine-tuning for a specific language, map it to the right family
  # so the correct tokenizer is selected from the aggregate tokenizer.
  declare -A LANG_TO_FAMILY=(
    # Already family-level (identity mapping)
    [INDO_ARYAN]=INDO_ARYAN
    [MANDARIN]=MANDARIN
    [DRAVIDIAN]=DRAVIDIAN
    [GERMANIC]=GERMANIC
    [ROMANCE]=ROMANCE
    [SLAVIC]=SLAVIC
    [JAPONIC]=JAPONIC
    # Indo-Aryan languages
    [HINDI]=INDO_ARYAN
    [BENGALI]=INDO_ARYAN
    [MARATHI]=INDO_ARYAN
    [GUJARATI]=INDO_ARYAN
    [PUNJABI]=INDO_ARYAN
    [URDU]=INDO_ARYAN
    [NEPALI]=INDO_ARYAN
    [SINHALA]=INDO_ARYAN
    [ODIA]=INDO_ARYAN
    [ASSAMESE]=INDO_ARYAN
    # Dravidian languages
    [TAMIL]=DRAVIDIAN
    [TELUGU]=DRAVIDIAN
    [KANNADA]=DRAVIDIAN
    [MALAYALAM]=DRAVIDIAN
    # Germanic languages
    [ENGLISH]=GERMANIC
    [EN]=GERMANIC
    [GERMAN]=GERMANIC
    [DUTCH]=GERMANIC
    # Romance languages
    [SPANISH]=ROMANCE
    [FRENCH]=ROMANCE
    [PORTUGUESE]=ROMANCE
    [ITALIAN]=ROMANCE
    # Slavic languages
    [RUSSIAN]=SLAVIC
    [POLISH]=SLAVIC
    [CZECH]=SLAVIC
    [UKRAINIAN]=SLAVIC
    # East Asian
    [CHINESE]=MANDARIN
    [ZH]=MANDARIN
    [JAPANESE]=JAPONIC
    [JA]=JAPONIC
  )

  LANG_FAMILY="${LANG_TO_FAMILY[${LANG_UPPER}]:-}"
  if [[ -z "${LANG_FAMILY}" ]]; then
    echo "WARNING: No family mapping for ${LANG_UPPER}, using as-is (may fail if not a known family)"
    LANG_FAMILY="${LANG_UPPER}"
  fi
  echo "==> Language: ${LANG_UPPER} → Family: ${LANG_FAMILY}"

  # Set defaults based on mode
  if [[ "${MODE}" == "adapter" ]]; then
    DEFAULT_BATCH_SIZE=8
    DEFAULT_LR=0.001
    DEFAULT_MAX_STEPS=30000
    DEFAULT_ACCUM=4
    DEFAULT_WARMUP=500
  else
    DEFAULT_BATCH_SIZE=4
    DEFAULT_LR=0.0001
    DEFAULT_MAX_STEPS=50000
    DEFAULT_ACCUM=8
    DEFAULT_WARMUP=1000
  fi

  BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
  LR="${LR:-${DEFAULT_LR}}"
  MAX_STEPS="${MAX_STEPS:-${DEFAULT_MAX_STEPS}}"

  CONFIG_DIR="${TRAINING_ROOT}/experiments/${EXP_NAME}"
  mkdir -p "${CONFIG_DIR}"
  CONFIG_PATH="${CONFIG_DIR}/config.yml"

  # Find tokenizer metadata files next to the .nemo file
  TOKENIZER_LANGS_PATH=""
  SHARED_TOKENS_PATH=""
  AGGREGATE_VOCAB_PATH=""
  for f in "${MODEL_DIR}"/stt_meta_*_tokenizer_langs.yaml "${MODEL_DIR}"/*tokenizer_langs.yaml; do
    [[ -f "$f" ]] && TOKENIZER_LANGS_PATH=$(basename "$f") && break
  done
  for f in "${MODEL_DIR}"/stt_meta_*_shared_special_tokens.yaml "${MODEL_DIR}"/*shared_special_tokens.yaml; do
    [[ -f "$f" ]] && SHARED_TOKENS_PATH=$(basename "$f") && break
  done
  for f in "${MODEL_DIR}"/stt_meta_*_aggregate_vocab.txt "${MODEL_DIR}"/*aggregate_vocab.txt; do
    [[ -f "$f" ]] && AGGREGATE_VOCAB_PATH=$(basename "$f") && break
  done

  # Write adapter section only for adapter mode
  ADAPTER_SECTION=""
  if [[ "${MODE}" == "adapter" ]]; then
    ADAPTER_SECTION="
adapter:
  enabled: true
  name: ${LANG_LOWER}_adapter
  dim: ${ADAPTER_DIM}
  activation: swish
  norm_position: pre
  unfreeze_decoder: true"
  fi

  # Streaming attention config (added to encoder override section).
  # FastConformer supports hybrid attention: train with multiple context sizes
  # so the model works in both full-context and streaming modes.
  # att_context_size: [left, right] in frames. [-1,-1] = full, [70,0] = causal.
  # The model can be trained with a mix: some batches use full context,
  # others use limited context, producing a single model that handles both.
  STREAMING_SECTION=""
  if [[ "${STREAMING:-false}" == "true" ]]; then
    STREAMING_SECTION="
streaming:
  enabled: true
  att_context_size: [70, 0]
  att_context_style: regular_causal
  att_context_prelen: 0
  multi_lookahead:
    - [70, 0]
    - [70, 10]
    - [-1, -1]"
  fi

  cat > "${CONFIG_PATH}" <<YAMLEOF
model:
  model_root: ${MODEL_DIR}/
  model_name: ${NEMO_FILENAME}
  tokenizer_folder: tokenizer
  new_tokenizer_folder: ${LANG_LOWER}_meta_tokenizer
  use_aggregate_tokenizer: true
  dynamic_tokenizer_params:
    type: bpe
    dir_prefix: tokenizer_
    non_special_tokens_per_lang: 3000
    non_special_tokens_per_lang_overrides: {}
    character_coverage: 0.995
    character_coverage_overrides: {}
    tokenizer_options: null
  special_token_prefixes:
  - ENTITY_
  - INTENT_
  - EMOTION_
  - GENDER_
  - AGE_
  - KEYWORD_
  - LANG_
  tokenizer_langs: {}
  shared_special_tokens: []
  aggregate_vocabulary: []
  language_families:
  - ${LANG_FAMILY}
  tokenizer_langs_path: ${TOKENIZER_LANGS_PATH}
  shared_special_tokens_path: ${SHARED_TOKENS_PATH}
  aggregate_vocabulary_path: ${AGGREGATE_VOCAB_PATH}
  language_family_map:
    ${LANG_UPPER}: ${LANG_FAMILY}
${ADAPTER_SECTION}
${STREAMING_SECTION}

training:
  lang_field: lang
  data_dir: ${DATA_DIR}
  train_manifest: train.json
  test_manifest: valid.json
  batch_size: ${BATCH_SIZE}
  accumulate_grad_batches: ${DEFAULT_ACCUM}
  max_duration: 16.0
  max_steps: ${MAX_STEPS}
  use_mixed_precision: true
  num_workers: 4
  pin_memory: true
  keyword_loss_weight: 0.5
  use_keyword_loss: true
  keyword_loss_warmup_steps: 500
  use_family_loss_weights: false
  spec_augment:
    time_masks: 4
    time_width: 80
  optim:
    lr: ${LR}
    weight_decay: 0.0
    sched:
      warmup_steps: ${DEFAULT_WARMUP}
  validation_workers: null
  skip_audio_validation: false
  keyphrase_oversample_factor: 0.0
  devices: ${GPU_COUNT}

experiment:
  exp_dir: ${TRAINING_ROOT}/experiments/
  exp_name: ${EXP_NAME}
  monitor: val_wer
  mode: min
  always_save_nemo: true
  save_top_k: 3
  every_n_train_steps: 2000
YAMLEOF

  echo "==> Generated config: ${CONFIG_PATH}"
fi

# ── Ensure Docker image exists ───────────────────────────────

REPO_ROOT="${SCRIPT_DIR}/../../.."
DOCKER_DIR="${REPO_ROOT}/docker"
if ! docker image inspect nemo-training:latest &>/dev/null; then
  echo "==> Building Docker image: nemo-training:latest"
  docker build -t nemo-training:latest -f "${DOCKER_DIR}/Dockerfile.training" "${REPO_ROOT}"
fi

# ── Ensure PromptingNemo code is on disk ─────────────────────

if [[ ! -d "${CODE_DIR}" ]]; then
  echo "==> Cloning PromptingNemo..."
  mkdir -p "${TRAINING_ROOT}/code"
  git clone https://github.com/WhissleAI/PromptingNemo.git "${CODE_DIR}"
fi

# ── Stop any existing container with the same name ───────────

if docker ps -a --format '{{.Names}}' | grep -q "^${EXP_NAME}$"; then
  echo "==> Stopping existing container: ${EXP_NAME}"
  docker stop "${EXP_NAME}" 2>/dev/null || true
  docker rm "${EXP_NAME}" 2>/dev/null || true
fi

# ── Build Docker run command ─────────────────────────────────

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

TRAIN_CMD="cd /workspace/code && python scripts/asr/meta-asr/main.py --mode both --config ${CONFIG_PATH}"

if [[ -n "${RESUME_FROM}" ]]; then
  TRAIN_CMD="${TRAIN_CMD} --resume-from ${RESUME_FROM}"
fi

echo ""
echo "══════════════════════════════════════════"
echo "  Experiment: ${EXP_NAME}"
echo "  Mode:       ${MODE}"
echo "  Config:     ${CONFIG_PATH}"
[[ -n "${MODEL_REPO}" ]] && echo "  Model:      ${MODEL_REPO}"
[[ -n "${DATASET_REPO}" ]] && echo "  Dataset:    ${DATASET_REPO}"
echo "  GPUs:       ${GPU_COUNT}"
echo "══════════════════════════════════════════"
echo ""

if [[ "${DETACH}" == "true" ]]; then
  docker run -d "${DOCKER_ARGS[@]}" nemo-training:latest bash -c "${TRAIN_CMD}"
  echo ""
  echo "==> Training started in background."
  echo ""
  echo "Monitor:"
  echo "  docker logs -f ${EXP_NAME}"
  echo "  docker logs ${EXP_NAME} 2>&1 | grep 'val_wer.*reached'"
  echo ""
  echo "TensorBoard:"
  echo "  $(dirname "$0")/tensorboard.sh --exp-dir ${TRAINING_ROOT}/experiments/"
else
  docker run --rm "${DOCKER_ARGS[@]}" nemo-training:latest bash -c "${TRAIN_CMD}"
fi
