#!/usr/bin/env bash
set -euo pipefail

# Sequential extended GRU lookback scan for Delta Thv(t+15).
# Completed runs are skipped, so this file can safely resume an interrupted scan.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON_BIN="${TASK_PYTHON_BIN:-python3}"
TASK_OVERWRITE="${TASK_OVERWRITE:-0}"
PREDICT_STEPS=15
COMMON_ORIGIN_LOOKBACK=120
SEEDS=(42 62 82)
LOOKBACKS=(60 75 90 105 120)
OUTPUT_DIR="${SCRIPT_DIR}/gru_lookback_extended_val260501_test0715BACK"
LOG_DIR="${SCRIPT_DIR}/logs_extended_h15"

mkdir -p "${LOG_DIR}"

for seed in "${SEEDS[@]}"; do
  for lookback in "${LOOKBACKS[@]}"; do
    metrics="${OUTPUT_DIR}/horizon_15/gru/lookback_$(printf '%02d' "${lookback}")/seed_${seed}/metrics.json"

    if [[ -f "${metrics}" && "${TASK_OVERWRITE}" != "1" ]]; then
      echo "SKIP existing lookback=${lookback} seed=${seed}: ${metrics}"
      continue
    fi

    command=(
      "${TASK_PYTHON_BIN}"
      "${SCRIPT_DIR}/train_thv_delta_gru_lookback_extended.py"
      --lookback "${lookback}"
      --seed "${seed}"
      --predict-steps "${PREDICT_STEPS}"
      --output-dir "${OUTPUT_DIR}"
    )
    if [[ "${TASK_OVERWRITE}" == "1" ]]; then
      command+=(--overwrite)
    fi

    echo "START lookback=${lookback} seed=${seed} horizon=${PREDICT_STEPS}"
    "${command[@]}" 2>&1 | tee "${LOG_DIR}/lookback_$(printf '%02d' "${lookback}")_seed_${seed}.log"
    echo "COMPLETE lookback=${lookback} seed=${seed}"
  done
done

"${TASK_PYTHON_BIN}" "${SCRIPT_DIR}/summarize_gru_lookback.py" \
  --output-dir "${OUTPUT_DIR}" \
  --predict-steps "${PREDICT_STEPS}" \
  --expected-seeds 42,62,82 \
  --expected-lookbacks 60,75,90,105,120 \
  --common-origin-lookback "${COMMON_ORIGIN_LOOKBACK}" \
  2>&1 | tee "${LOG_DIR}/summary.log"

echo "EXTENDED_LOOKBACK_SCAN_COMPLETE=true"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
