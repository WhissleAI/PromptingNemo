#!/usr/bin/env bash
# Meta-ASR training entry point.
# Usage:
#   ./run.sh --lang hindi --mode both
#   ./run.sh --lang mandarin --mode tokenizer
#   ./run.sh --lang english --mode train
#   ./run.sh --config conf/custom.yaml --mode train
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- defaults ----------
LANG=""
MODE="both"
CONFIG=""
EXTRA_ARGS=()

# ---------- help ----------
usage() {
    cat <<EOF
Meta-ASR training recipe.

Usage:
  $(basename "$0") --lang <language> [--mode <mode>] [-- extra-args...]
  $(basename "$0") --config <path>   [--mode <mode>] [-- extra-args...]

Options:
  --lang   LANG    Language name. Mapped to a config in conf/<lang>.yaml.
                   Available languages:
                     bengali, english, gujarati, hindi, indo_aryan,
                     kannada, malayalam, mandarin, mandarin_aishell3,
                     marathi, pretrain_meta, punjabi, slavic, vils, wellness
  --config PATH    Path to a YAML config file (overrides --lang).
  --mode   MODE    One of: tokenizer, train, both, validate_data (default: both).
  -h, --help       Show this help message.

Any arguments after '--' are forwarded to the training CLI.

Examples:
  # Train Hindi from scratch (tokenizer + model)
  ./run.sh --lang hindi --mode both

  # Only rebuild the Mandarin tokenizer
  ./run.sh --lang mandarin --mode tokenizer

  # Resume English training from a checkpoint
  ./run.sh --lang english --mode train -- --resume_from /path/to/ckpt

  # Use a custom config file
  ./run.sh --config conf/custom.yaml --mode train
EOF
    exit 0
}

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang)   LANG="$2";   shift 2 ;;
        --mode)   MODE="$2";   shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help) usage ;;
        --)       shift; EXTRA_ARGS=("$@"); break ;;
        *)        echo "Unknown argument: $1"; usage ;;
    esac
done

# ---------- resolve config ----------
if [[ -n "$CONFIG" ]]; then
    CONFIG_PATH="$CONFIG"
elif [[ -n "$LANG" ]]; then
    CONFIG_PATH="${SCRIPT_DIR}/conf/${LANG}.yaml"
else
    echo "Error: either --lang or --config is required."
    echo ""
    usage
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: config file not found: $CONFIG_PATH"
    echo "Available configs:"
    ls "${SCRIPT_DIR}/conf/"*.yaml 2>/dev/null | while read -r f; do
        echo "  $(basename "$f" .yaml)"
    done
    exit 1
fi

# ---------- validate mode ----------
case "$MODE" in
    tokenizer|train|both|validate_data) ;;
    *)
        echo "Error: --mode must be one of: tokenizer, train, both (got: $MODE)"
        exit 1
        ;;
esac

# ---------- run ----------
echo "============================================"
echo " Meta-ASR Training"
echo " Config : $CONFIG_PATH"
echo " Mode   : $MODE"
echo "============================================"

python -m promptingnemo.training.cli \
    --config "$CONFIG_PATH" \
    --mode "$MODE" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
