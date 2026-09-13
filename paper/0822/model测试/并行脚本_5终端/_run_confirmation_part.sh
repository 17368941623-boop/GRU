#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

if [[ $# -ne 1 || ! "$1" =~ ^[1-5]$ ]]; then
  echo "Usage: bash _run_confirmation_part.sh PART_NUMBER_1_TO_5"
  exit 2
fi

PART_NUMBER="$1"
SHARD_INDEX=$((PART_NUMBER - 1))
command=(
  "${MODEL_PYTHON_BIN}"
  "${MODEL_DIR}/run_selected_configs.py"
  --lookback "${MODEL_LOOKBACK}"
  --common-origin-lookback "${MODEL_COMMON_ORIGIN}"
  --predict-steps "${MODEL_PREDICT_STEPS}"
  --seeds 82
  --results-dir "${MODEL_RESULTS_DIR}"
  --data-dir "${MODEL_DATA_DIR}"
  --epochs 100
  --batch-size 256
  --patience 12
  --num-workers 0
  --shard-count 5
  --shard-index "${SHARD_INDEX}"
)
if [[ "${MODEL_OVERWRITE}" == "1" ]]; then
  command+=(--overwrite)
fi

echo "CONFIRMATION_PART=${PART_NUMBER}/5"
"${command[@]}" 2>&1 | tee "${MODEL_LOG_DIR}/03_confirmation_part_${PART_NUMBER}.log"

