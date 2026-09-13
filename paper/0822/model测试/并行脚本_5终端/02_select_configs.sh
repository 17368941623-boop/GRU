#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

"${MODEL_PYTHON_BIN}" "${MODEL_DIR}/select_model_configs.py" \
  --lookback "${MODEL_LOOKBACK}" \
  --common-origin-lookback "${MODEL_COMMON_ORIGIN}" \
  --predict-steps "${MODEL_PREDICT_STEPS}" \
  --search-seeds 42,62 \
  --results-dir "${MODEL_RESULTS_DIR}" \
  2>&1 | tee "${MODEL_LOG_DIR}/02_select_configs.log"

