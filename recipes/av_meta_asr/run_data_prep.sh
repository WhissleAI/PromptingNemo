#!/bin/bash
# =============================================================================
# End-to-end data preparation for AV-Meta-ASR (SpeakerVid-5M)
# =============================================================================
# Orchestrates all data prep stages in order:
#   Phase 1: Download annotations + videos (CPU, parallel)
#   Phase 2: Extract clips, audio, SigLIP features (GPU for features)
#   Phase 3: Parse annotations + build manifests (CPU)
#
# Usage:
#   bash run_data_prep.sh [--stage N] [--limit N]
#
# Options:
#   --stage N    Start from stage N (1-7, default 1)
#   --limit N    Limit items to process (0 = all, for testing)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${SCRIPT_DIR}/local"

# Default paths (override via environment)
BASE_DIR="${BASE_DIR:-/mnt/nfs/data/speakervid_5m}"
ANNOTATIONS_DIR="${BASE_DIR}/annotations"
VIDEOS_DIR="${BASE_DIR}/videos"
CLIPS_DIR="${BASE_DIR}/clips"
AUDIO_DIR="${BASE_DIR}/audio"
FEATURES_DIR="${BASE_DIR}/siglip_features"
MANIFESTS_DIR="${BASE_DIR}/manifests"
METADATA_DIR="${BASE_DIR}/metadata"

# Settings
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"
CLIP_WORKERS="${CLIP_WORKERS:-8}"
AUDIO_WORKERS="${AUDIO_WORKERS:-16}"
PARSE_WORKERS="${PARSE_WORKERS:-8}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-16}"
FEATURE_FPS="${FEATURE_FPS:-5}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip2-so400m-patch14-384}"

# Parse arguments
START_STAGE=1
LIMIT=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) START_STAGE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LIMIT_ARG=""
if [ "$LIMIT" -gt 0 ]; then
    LIMIT_ARG="--limit ${LIMIT}"
fi

echo "============================================"
echo "AV-Meta-ASR Data Preparation Pipeline"
echo "============================================"
echo "Base dir:    ${BASE_DIR}"
echo "Start stage: ${START_STAGE}"
echo "Limit:       ${LIMIT:-all}"
echo "============================================"

# ── Stage 1: Download annotations from HuggingFace ──
if [ "$START_STAGE" -le 1 ]; then
    echo ""
    echo "[Stage 1/7] Downloading annotations from HuggingFace..."
    python "${LOCAL_DIR}/download_annotations.py" \
        --output-dir "${ANNOTATIONS_DIR}"
fi

# ── Stage 2: Download YouTube videos ──
if [ "$START_STAGE" -le 2 ]; then
    echo ""
    echo "[Stage 2/7] Downloading YouTube videos..."
    python "${LOCAL_DIR}/download_videos.py" \
        --annotations-dir "${ANNOTATIONS_DIR}" \
        --output-dir "${VIDEOS_DIR}" \
        --workers "${DOWNLOAD_WORKERS}" \
        ${LIMIT_ARG}
fi

# ── Stage 3: Extract clips from videos ──
if [ "$START_STAGE" -le 3 ]; then
    echo ""
    echo "[Stage 3/7] Extracting speaker clips..."
    python "${LOCAL_DIR}/extract_clips.py" \
        --annotations-dir "${ANNOTATIONS_DIR}" \
        --videos-dir "${VIDEOS_DIR}" \
        --output-dir "${CLIPS_DIR}" \
        --workers "${CLIP_WORKERS}" \
        ${LIMIT_ARG}
fi

# ── Stage 4: Extract audio from clips ──
if [ "$START_STAGE" -le 4 ]; then
    echo ""
    echo "[Stage 4/7] Extracting audio..."
    python "${LOCAL_DIR}/extract_audio.py" \
        --clips-dir "${CLIPS_DIR}" \
        --output-dir "${AUDIO_DIR}" \
        --workers "${AUDIO_WORKERS}" \
        ${LIMIT_ARG}
fi

# ── Stage 5: Extract SigLIP 2 features (GPU) ──
if [ "$START_STAGE" -le 5 ]; then
    echo ""
    echo "[Stage 5/7] Extracting SigLIP 2 features (GPU required)..."
    python "${LOCAL_DIR}/extract_clip_features.py" \
        --clips-dir "${CLIPS_DIR}" \
        --output-dir "${FEATURES_DIR}" \
        --model-name "${SIGLIP_MODEL}" \
        --fps "${FEATURE_FPS}" \
        --batch-size "${FEATURE_BATCH_SIZE}" \
        ${LIMIT_ARG}
fi

# ── Stage 6: Parse annotations ──
if [ "$START_STAGE" -le 6 ]; then
    echo ""
    echo "[Stage 6/7] Parsing annotations..."
    python "${LOCAL_DIR}/parse_annotations.py" \
        --annotations-dir "${ANNOTATIONS_DIR}" \
        --output "${METADATA_DIR}/parsed_annotations.jsonl" \
        --workers "${PARSE_WORKERS}" \
        ${LIMIT_ARG}
fi

# ── Stage 7: Build manifests ──
if [ "$START_STAGE" -le 7 ]; then
    echo ""
    echo "[Stage 7/7] Building manifests..."
    python "${LOCAL_DIR}/build_manifest.py" \
        --annotations "${METADATA_DIR}/parsed_annotations.jsonl" \
        --audio-dir "${AUDIO_DIR}" \
        --clips-dir "${CLIPS_DIR}" \
        --features-dir "${FEATURES_DIR}" \
        --output-dir "${MANIFESTS_DIR}" \
        ${LIMIT_ARG}
fi

echo ""
echo "============================================"
echo "Data preparation complete!"
echo "============================================"
echo "Manifests: ${MANIFESTS_DIR}/"
echo ""
echo "Next step: Train the model with:"
echo "  python train.py --config conf/av_meta_nextgen.yaml"
echo "============================================"
