#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

command=(
  "${MODEL_PYTHON_BIN}"
  "${MODEL_DIR}/evaluate_selected_models.py"
  --lookback "${MODEL_LOOKBACK}"
  --predict-steps "${MODEL_PREDICT_STEPS}"
  --seeds 42,62,82
  --data-dir "${MODEL_DATA_DIR}"
  --results-dir "${MODEL_RESULTS_DIR}"
  --batch-size 256
  --num-workers 0
)
if [[ "${MODEL_OVERWRITE}" == "1" ]]; then
  command+=(--overwrite)
fi

"${command[@]}" 2>&1 | tee "${MODEL_LOG_DIR}/04_final_test.log"

