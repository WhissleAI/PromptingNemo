#!/usr/bin/env bash
# =============================================================================
# Audio-Visual ASR training recipe
# =============================================================================
# Usage:
#   ./run.sh --config av_conformer_ctc --snr rand --gpus 1
#   ./run.sh --config audio_only_baseline --snr 10.0 --gpus 4
#   ./run.sh --stage 1 --stop-stage 1 --output-dir /data/vans
#   ./run.sh --stage 2 --stop-stage 2 --output-dir /data/vans
#   ./run.sh --stage 3 --stop-stage 3 --config av_conformer_ctc
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- defaults ----------
config="av_conformer_ctc"
snr="rand"
gpus=1
stage=1
stop_stage=100
resume=""

# Data directories (for stages 1-2)
peoples_speech_dir=""
audioset_dir=""
output_dir=""

# Training overrides (for stage 3)
train_manifest=""
val_manifest=""
exp_dir=""
tokenizer_dir=""

# Feature extraction settings (for stage 2)
clip_model="ViT-L/14"
fps=5
clip_batch_size=32

# VANS dataset settings (for stage 1)
max_duration=10.0
min_samples_per_class=750
snr_min=-5.0
snr_max=5.0

# ---------- help ----------
usage() {
    cat <<EOF
Audio-Visual ASR training recipe (EMNLP 2025).

Usage:
  $(basename "$0") [OPTIONS]

Stages:
  1  Data preparation (VANS dataset creation)
  2  Feature extraction (CLIP visual features)
  3  Model training

Options:
  --config CONFIG        Config name in conf/ (default: av_conformer_ctc)
  --snr SNR              SNR ratio: 'rand' or float (default: rand)
  --gpus N               Number of GPUs (default: 1)
  --stage N              Start from stage N (default: 1)
  --stop-stage N         Stop after stage N (default: 100)
  --resume PATH          Checkpoint to resume training from

  Data preparation (stage 1):
  --peoples-speech-dir   Path to People's Speech dataset
  --audioset-dir         Path to AudioSet dataset
  --output-dir           Output directory for VANS dataset and features
  --max-duration SECS    Max audio duration in seconds (default: 10.0)
  --min-samples N        Min samples per noise class (default: 750)
  --snr-min DB           Min SNR for uniform mixing (default: -5.0)
  --snr-max DB           Max SNR for uniform mixing (default: 5.0)

  Feature extraction (stage 2):
  --clip-model MODEL     CLIP model name (default: ViT-L/14)
  --fps N                Video frame rate for features (default: 5)
  --clip-batch-size N    Batch size for CLIP inference (default: 32)

  Training overrides (stage 3):
  --train-manifest PATH  Override training manifest
  --val-manifest PATH    Override validation manifest
  --exp-dir PATH         Override experiment output directory
  --tokenizer-dir PATH   Override tokenizer directory

  -h, --help             Show this help message

Examples:
  # Run full pipeline
  ./run.sh --peoples-speech-dir /data/ps --audioset-dir /data/as --output-dir /data/vans

  # Only prepare data
  ./run.sh --stage 1 --stop-stage 1 --output-dir /data/vans

  # Only extract features
  ./run.sh --stage 2 --stop-stage 2 --output-dir /data/vans

  # Only train (AV-UNI-SNR, best model)
  ./run.sh --stage 3 --stop-stage 3 --config av_conformer_ctc --snr rand

  # Train audio-only baseline
  ./run.sh --stage 3 --stop-stage 3 --config audio_only_baseline --snr rand

  # Train with fixed 10dB SNR
  ./run.sh --stage 3 --stop-stage 3 --config av_conformer_ctc --snr 10.0 --gpus 4

  # Resume training from a checkpoint
  ./run.sh --stage 3 --stop-stage 3 --resume /path/to/checkpoint.ckpt
EOF
    exit 0
}

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)              config="$2";              shift 2 ;;
        --snr)                 snr="$2";                 shift 2 ;;
        --gpus)                gpus="$2";                shift 2 ;;
        --stage)               stage="$2";               shift 2 ;;
        --stop-stage)          stop_stage="$2";          shift 2 ;;
        --resume)              resume="$2";              shift 2 ;;
        --peoples-speech-dir)  peoples_speech_dir="$2";  shift 2 ;;
        --audioset-dir)        audioset_dir="$2";        shift 2 ;;
        --output-dir)          output_dir="$2";          shift 2 ;;
        --max-duration)        max_duration="$2";        shift 2 ;;
        --min-samples)         min_samples_per_class="$2"; shift 2 ;;
        --snr-min)             snr_min="$2";             shift 2 ;;
        --snr-max)             snr_max="$2";             shift 2 ;;
        --clip-model)          clip_model="$2";          shift 2 ;;
        --fps)                 fps="$2";                 shift 2 ;;
        --clip-batch-size)     clip_batch_size="$2";     shift 2 ;;
        --train-manifest)      train_manifest="$2";      shift 2 ;;
        --val-manifest)        val_manifest="$2";        shift 2 ;;
        --exp-dir)             exp_dir="$2";             shift 2 ;;
        --tokenizer-dir)       tokenizer_dir="$2";       shift 2 ;;
        -h|--help)             usage ;;
        *)                     echo "Unknown argument: $1"; usage ;;
    esac
done

# ---------- resolve config path ----------
CONFIG_PATH="${SCRIPT_DIR}/conf/${config}.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: config file not found: $CONFIG_PATH"
    echo "Available configs:"
    ls "${SCRIPT_DIR}/conf/"*.yaml 2>/dev/null | while read -r f; do
        echo "  $(basename "$f" .yaml)"
    done
    exit 1
fi

echo "============================================"
echo " Audio-Visual ASR Training Recipe"
echo " Config     : ${config}"
echo " SNR        : ${snr}"
echo " GPUs       : ${gpus}"
echo " Stages     : ${stage} -> ${stop_stage}"
echo "============================================"

# ==========================================================================
# Stage 1: Data preparation (VANS dataset)
# ==========================================================================
if [[ $stage -le 1 && $stop_stage -ge 1 ]]; then
    echo ""
    echo "============================================"
    echo " Stage 1: Data Preparation (VANS dataset)"
    echo "============================================"

    if [[ -z "$output_dir" ]]; then
        echo "Error: --output-dir is required for stage 1"
        exit 1
    fi

    prepare_args=(
        --output-dir "$output_dir"
        --max-duration "$max_duration"
        --min-samples-per-class "$min_samples_per_class"
        --snr-min "$snr_min"
        --snr-max "$snr_max"
    )

    if [[ -n "$peoples_speech_dir" ]]; then
        prepare_args+=(--peoples-speech-dir "$peoples_speech_dir")
    fi
    if [[ -n "$audioset_dir" ]]; then
        prepare_args+=(--audioset-dir "$audioset_dir")
    fi

    python "${SCRIPT_DIR}/local/prepare_vans.py" "${prepare_args[@]}"
    echo "Stage 1 complete."
fi

# ==========================================================================
# Stage 2: Feature extraction (CLIP)
# ==========================================================================
if [[ $stage -le 2 && $stop_stage -ge 2 ]]; then
    echo ""
    echo "============================================"
    echo " Stage 2: CLIP Feature Extraction"
    echo "============================================"

    if [[ -z "$output_dir" ]]; then
        echo "Error: --output-dir is required for stage 2"
        exit 1
    fi

    extract_args=(
        --data-dir "$output_dir"
        --clip-model "$clip_model"
        --fps "$fps"
        --batch-size "$clip_batch_size"
    )

    python "${SCRIPT_DIR}/local/extract_features.py" "${extract_args[@]}"
    echo "Stage 2 complete."
fi

# ==========================================================================
# Stage 3: Training
# ==========================================================================
if [[ $stage -le 3 && $stop_stage -ge 3 ]]; then
    echo ""
    echo "============================================"
    echo " Stage 3: Model Training"
    echo "============================================"

    train_args=(
        --config "$CONFIG_PATH"
        --snr "$snr"
        --gpus "$gpus"
    )

    if [[ -n "$resume" ]]; then
        train_args+=(--resume "$resume")
    fi
    if [[ -n "$train_manifest" ]]; then
        train_args+=(--train-manifest "$train_manifest")
    fi
    if [[ -n "$val_manifest" ]]; then
        train_args+=(--val-manifest "$val_manifest")
    fi
    if [[ -n "$exp_dir" ]]; then
        train_args+=(--exp-dir "$exp_dir")
    fi
    if [[ -n "$tokenizer_dir" ]]; then
        train_args+=(--tokenizer-dir "$tokenizer_dir")
    fi

    python "${SCRIPT_DIR}/train.py" "${train_args[@]}"
    echo "Stage 3 complete."
fi

echo ""
echo "============================================"
echo " All requested stages complete."
echo "============================================"
