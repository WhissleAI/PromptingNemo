#!/usr/bin/env bash
# Benchmark a fine-tuned NeMo model on test data.
# Calculates WER, CER, logs results, and saves to experiment directory.
#
# Usage:
#   ./benchmark.sh --name zh-adapter-v1
#   ./benchmark.sh --name zh-adapter-v1 --test-manifest /mnt/training/data/zh_aishell3/test.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="${TRAINING_ROOT:-/mnt/training}"
CODE_DIR="${TRAINING_ROOT}/code/PromptingNemo"
LOGS_DIR="${CODE_DIR}/scripts/asr/meta-asr/logs"

EXP_NAME=""
TEST_MANIFEST=""
BATCH_SIZE=16
NUM_WORKERS=4

usage() {
  cat <<EOF
Usage: $0 --name EXP_NAME [OPTIONS]

Required:
  --name NAME              Experiment name

Options:
  --test-manifest PATH     Path to test manifest (auto-detected from config if omitted)
  --batch-size N           Batch size for inference (default: 16)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --name) EXP_NAME="$2"; shift 2 ;;
    --test-manifest) TEST_MANIFEST="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "${EXP_NAME}" ]] && { echo "ERROR: --name is required"; usage; }

EXP_DIR="${TRAINING_ROOT}/experiments/${EXP_NAME}"

# Find the best .nemo model
NEMO_FILE=$(find "${EXP_DIR}" -name "*.nemo" -type f -not -name "*-v[0-9]*" 2>/dev/null | head -1)
if [[ -z "${NEMO_FILE}" ]]; then
  echo "ERROR: No .nemo file found in ${EXP_DIR}"
  exit 1
fi
echo "==> Model: ${NEMO_FILE}"

# Find test manifest from config if not provided
if [[ -z "${TEST_MANIFEST}" ]]; then
  CONFIG_FILE=$(find "${EXP_DIR}" -name "config.yml" -type f 2>/dev/null | head -1)
  if [[ -n "${CONFIG_FILE}" ]]; then
    DATA_DIR=$(grep 'data_dir:' "${CONFIG_FILE}" | awk '{print $2}')
    TEST_FILE=$(grep 'test_manifest:' "${CONFIG_FILE}" | awk '{print $2}')
    if [[ -n "${DATA_DIR}" && -n "${TEST_FILE}" ]]; then
      TEST_MANIFEST="${DATA_DIR}/${TEST_FILE}"
    fi
    # Also try valid.json if test.json not found
    if [[ ! -f "${TEST_MANIFEST}" ]]; then
      TEST_MANIFEST="${DATA_DIR}/valid.json"
    fi
    # Try test.json
    if [[ ! -f "${TEST_MANIFEST}" && -f "${DATA_DIR}/test.json" ]]; then
      TEST_MANIFEST="${DATA_DIR}/test.json"
    fi
  fi
fi

if [[ -z "${TEST_MANIFEST}" || ! -f "${TEST_MANIFEST}" ]]; then
  echo "ERROR: Test manifest not found. Specify with --test-manifest"
  exit 1
fi

TEST_SAMPLES=$(wc -l < "${TEST_MANIFEST}")
echo "==> Test manifest: ${TEST_MANIFEST} (${TEST_SAMPLES} samples)"

mkdir -p "${LOGS_DIR}" "${EXP_DIR}"

BENCHMARK_SCRIPT=$(cat <<'PYEOF'
import sys
import os
import json
import time
import datetime

nemo_file = sys.argv[1]
test_manifest = sys.argv[2]
batch_size = int(sys.argv[3])
exp_name = sys.argv[4]
exp_dir = sys.argv[5]
logs_dir = sys.argv[6]

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

print(f"Loading model from {nemo_file}...")
load_start = time.time()

from nemo.collections.asr.models import ASRModel
import torch
import numpy as np

model = ASRModel.restore_from(nemo_file, map_location="cuda")
model.eval()
model.freeze()
load_time = time.time() - load_start
print(f"Model loaded in {load_time:.1f}s")

# Read test manifest
with open(test_manifest, "r") as f:
    test_entries = [json.loads(line) for line in f]

audio_files = [e["audio_filepath"] for e in test_entries]
references = [e["text"] for e in test_entries]
durations = [e.get("duration", 0) for e in test_entries]

total_audio_hours = sum(durations) / 3600
print(f"Test set: {len(audio_files)} files, {total_audio_hours:.1f} hours")

# Transcribe in batches
print(f"Transcribing (batch_size={batch_size})...")
infer_start = time.time()

hypotheses = []
for i in range(0, len(audio_files), batch_size):
    batch_files = audio_files[i:i+batch_size]
    batch_hyps = model.transcribe(batch_files, batch_size=batch_size)
    if isinstance(batch_hyps, tuple):
        batch_hyps = batch_hyps[0]
    hypotheses.extend(batch_hyps)
    done = min(i + batch_size, len(audio_files))
    if done % (batch_size * 10) == 0 or done == len(audio_files):
        print(f"  {done}/{len(audio_files)}")

infer_time = time.time() - infer_start
rtf = infer_time / (sum(durations) if sum(durations) > 0 else 1)

# Calculate WER and CER
from jiwer import wer, cer

overall_wer = wer(references, hypotheses)
overall_cer = cer(references, hypotheses)

# Per-sample results for analysis
per_sample = []
for ref, hyp, dur, path in zip(references, hypotheses, durations, audio_files):
    sample_wer = wer(ref, hyp) if ref.strip() else 0.0
    per_sample.append({
        "audio": os.path.basename(path),
        "reference": ref,
        "hypothesis": hyp,
        "wer": round(sample_wer, 4),
        "duration": dur,
    })

# Sort by WER descending to find worst cases
per_sample.sort(key=lambda x: x["wer"], reverse=True)

results = {
    "experiment": exp_name,
    "model": nemo_file,
    "test_manifest": test_manifest,
    "timestamp": datetime.datetime.now().isoformat(),
    "metrics": {
        "wer": round(overall_wer, 5),
        "cer": round(overall_cer, 5),
        "num_samples": len(audio_files),
        "total_audio_hours": round(total_audio_hours, 2),
    },
    "performance": {
        "model_load_time_s": round(load_time, 1),
        "inference_time_s": round(infer_time, 1),
        "real_time_factor": round(rtf, 4),
        "samples_per_second": round(len(audio_files) / infer_time, 1),
    },
    "worst_samples": per_sample[:20],
    "best_samples": per_sample[-10:],
}

# Save results
results_path = os.path.join(exp_dir, "benchmark_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {results_path}")

# Also save to shared logs directory
log_filename = f"benchmark_{exp_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
log_path = os.path.join(logs_dir, log_filename)
os.makedirs(logs_dir, exist_ok=True)
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Log saved to {log_path}")

# Save full transcriptions
trans_path = os.path.join(exp_dir, "transcriptions.json")
with open(trans_path, "w", encoding="utf-8") as f:
    json.dump(per_sample, f, indent=2, ensure_ascii=False)
print(f"Full transcriptions saved to {trans_path}")

# Print summary
print(f"\n{'='*50}")
print(f"  BENCHMARK RESULTS: {exp_name}")
print(f"{'='*50}")
print(f"  WER:  {overall_wer*100:.2f}%")
print(f"  CER:  {overall_cer*100:.2f}%")
print(f"  Samples: {len(audio_files)}")
print(f"  Audio: {total_audio_hours:.1f} hours")
print(f"  RTF: {rtf:.4f} (inference {infer_time:.0f}s for {sum(durations):.0f}s audio)")
print(f"{'='*50}")
print(f"\n  Worst 5 samples:")
for s in per_sample[:5]:
    print(f"    WER={s['wer']:.2f} | ref: {s['reference'][:60]}")
    print(f"             | hyp: {s['hypothesis'][:60]}")
PYEOF
)

echo ""
echo "==> Running benchmark for ${EXP_NAME}..."

docker run --rm \
  --gpus all \
  --ipc=host \
  -v "${TRAINING_ROOT}:${TRAINING_ROOT}" \
  -v "${CODE_DIR}:/workspace/code" \
  -e PYTHONPATH="/workspace/code" \
  nemo-training:latest \
  python -c "${BENCHMARK_SCRIPT}" \
    "${NEMO_FILE}" "${TEST_MANIFEST}" "${BATCH_SIZE}" "${EXP_NAME}" "${EXP_DIR}" "${LOGS_DIR}"

echo ""
echo "==> Benchmark complete. Results saved to:"
echo "    ${EXP_DIR}/benchmark_results.json"
echo "    ${LOGS_DIR}/"
