#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_all_model_tests.sh BEST_LOOKBACK [quick|full]"
  echo "Example: bash run_all_model_tests.sh 20 full"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOOKBACK="$1"
COMMON_ORIGIN_LOOKBACK="${COMMON_ORIGIN_LOOKBACK:-${LOOKBACK}}"
PROFILE="${2:-full}"
PREDICT_STEPS="${PREDICT_STEPS:-15}"
SEARCH_SEEDS="${SEARCH_SEEDS:-42,62}"
CONFIRM_SEEDS="${CONFIRM_SEEDS:-82}"
FINAL_SEEDS="${FINAL_SEEDS:-42,62,82}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-12}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OVERWRITE="${OVERWRITE:-0}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/processed_data}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/outputs}"
LOG_DIR="${SCRIPT_DIR}/logs_h${PREDICT_STEPS}_lb${LOOKBACK}"
SHARED_CODE_DIR=""

for candidate in "${PROJECT_DIR}/残差lookback测试" "${PROJECT_DIR}"; do
  if [[ -f "${candidate}/train_thv_delta_lstm_lookback.py" && \
        -f "${candidate}/train_thv_delta_rnn_compare.py" ]]; then
    SHARED_CODE_DIR="${candidate}"
    break
  fi
done

if [[ -z "${SHARED_CODE_DIR}" ]]; then
  echo "Missing shared training files. Expected both files in either:"
  echo "  ${PROJECT_DIR}/残差lookback测试"
  echo "  ${PROJECT_DIR}"
  echo "Required: train_thv_delta_lstm_lookback.py"
  echo "Required: train_thv_delta_rnn_compare.py"
  exit 2
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${SHARED_CODE_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${SHARED_CODE_DIR}"
fi

if [[ "${PROFILE}" != "quick" && "${PROFILE}" != "full" ]]; then
  echo "PROFILE must be quick or full"
  exit 2
fi
if (( COMMON_ORIGIN_LOOKBACK < LOOKBACK )); then
  echo "COMMON_ORIGIN_LOOKBACK cannot be smaller than LOOKBACK"
  exit 2
fi

mkdir -p "${LOG_DIR}"
echo "SHARED_CODE_DIR=${SHARED_CODE_DIR}"
"${PYTHON_BIN}" -c \
  "import train_thv_delta_lstm_lookback; import train_thv_delta_rnn_compare; print('SHARED_IMPORTS_OK=true')"
common=(
  --lookback "${LOOKBACK}"
  --common-origin-lookback "${COMMON_ORIGIN_LOOKBACK}"
  --predict-steps "${PREDICT_STEPS}"
  --data-dir "${DATA_DIR}"
  --results-dir "${RESULTS_DIR}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --patience "${PATIENCE}"
  --num-workers "${NUM_WORKERS}"
)
echo "PHASE 0/4: synthetic forward/backward smoke test"
"${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_test_models.py" \
  2>&1 | tee "${LOG_DIR}/00_smoke_test.log"

echo "PHASE 1/4: validation-only parameter search"
search_command=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/run_model_search.py"
  "${common[@]}"
  --seeds "${SEARCH_SEEDS}"
  --profile "${PROFILE}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  search_command+=(--overwrite)
fi
"${search_command[@]}" 2>&1 | tee "${LOG_DIR}/01_parameter_search.log"

echo "PHASE 2/4: select one configuration per model on 260501"
"${PYTHON_BIN}" "${SCRIPT_DIR}/select_model_configs.py" \
  --lookback "${LOOKBACK}" \
  --common-origin-lookback "${COMMON_ORIGIN_LOOKBACK}" \
  --predict-steps "${PREDICT_STEPS}" \
  --search-seeds "${SEARCH_SEEDS}" \
  --results-dir "${RESULTS_DIR}" \
  2>&1 | tee "${LOG_DIR}/02_select_configs.log"

echo "PHASE 3/4: train selected configurations for confirmation seeds"
confirmation_command=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/run_selected_configs.py"
  "${common[@]}"
  --seeds "${CONFIRM_SEEDS}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  confirmation_command+=(--overwrite)
fi
"${confirmation_command[@]}" 2>&1 | tee "${LOG_DIR}/03_confirmation_seeds.log"

echo "PHASE 4/4: open 0715-BACK only for final selected-model evaluation"
evaluation_command=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/evaluate_selected_models.py"
  --lookback "${LOOKBACK}"
  --predict-steps "${PREDICT_STEPS}"
  --seeds "${FINAL_SEEDS}"
  --data-dir "${DATA_DIR}"
  --results-dir "${RESULTS_DIR}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  evaluation_command+=(--overwrite)
fi
"${evaluation_command[@]}" 2>&1 | tee "${LOG_DIR}/04_final_test.log"

echo "MODEL_ABLATION_COMPLETE=true"
echo "RESULTS_DIR=${RESULTS_DIR}"
