#!/usr/bin/env bash
set -euo pipefail

PARALLEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${PARALLEL_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${MODEL_DIR}/.." && pwd)"

MODEL_PYTHON_BIN="${MODEL_PYTHON_BIN:-python3}"
MODEL_LOOKBACK=60
MODEL_COMMON_ORIGIN=60
MODEL_PREDICT_STEPS=15
MODEL_RESULTS_DIR="${MODEL_RESULTS_DIR:-${MODEL_DIR}/outputs}"
MODEL_DATA_DIR="${MODEL_DATA_DIR:-${PROJECT_DIR}/processed_data}"
MODEL_LOG_DIR="${MODEL_DIR}/logs_h15_lb60_parallel"
MODEL_OVERWRITE="${MODEL_OVERWRITE:-0}"
SHARED_CODE_DIR=""

for candidate in "${PROJECT_DIR}/残差lookback测试" "${PROJECT_DIR}"; do
  if [[ -f "${candidate}/train_thv_delta_lstm_lookback.py" && \
        -f "${candidate}/train_thv_delta_rnn_compare.py" ]]; then
    SHARED_CODE_DIR="${candidate}"
    break
  fi
done

if [[ -z "${SHARED_CODE_DIR}" ]]; then
  echo "Missing train_thv_delta_lstm_lookback.py and train_thv_delta_rnn_compare.py"
  echo "Checked: ${PROJECT_DIR}/残差lookback测试"
  echo "Checked: ${PROJECT_DIR}"
  exit 2
fi
if [[ ! -d "${MODEL_DATA_DIR}" ]]; then
  echo "Missing processed data directory: ${MODEL_DATA_DIR}"
  exit 2
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${SHARED_CODE_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${SHARED_CODE_DIR}"
fi

mkdir -p "${MODEL_LOG_DIR}"

