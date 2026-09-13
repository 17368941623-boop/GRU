#!/usr/bin/env bash
set -euo pipefail

# Sequential GRU lookback scan for Thv(t+15).  The next run starts only after
# the previous command exits successfully.  Existing completed runs are skipped
# unless OVERWRITE=1 is set.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PREDICT_STEPS="${PREDICT_STEPS:-15}"
SEEDS="${SEEDS:-42 62 82}"
OVERWRITE="${OVERWRITE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/gru_lookback_outputs_val260501_test0715BACK}"
LOG_DIR="${SCRIPT_DIR}/logs_h${PREDICT_STEPS}"
LOOKBACKS=(10 15 20 25 30 35 40 45 50 55 60)

mkdir -p "${LOG_DIR}"

for seed in ${SEEDS}; do
  for lookback in "${LOOKBACKS[@]}"; do
    metrics="${OUTPUT_DIR}/horizon_$(printf '%02d' "${PREDICT_STEPS}")/gru/lookback_$(printf '%02d' "${lookback}")/seed_${seed}/metrics.json"
    if [[ -f "${metrics}" && "${OVERWRITE}" != "1" ]]; then
      echo "SKIP existing lookback=${lookback} seed=${seed}: ${metrics}"
      continue
    fi

    command=(
      "${PYTHON_BIN}"
      "${SCRIPT_DIR}/train_thv_delta_gru_lookback.py"
      --lookback "${lookback}"
      --seed "${seed}"
      --predict-steps "${PREDICT_STEPS}"
      --output-dir "${OUTPUT_DIR}"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
      command+=(--overwrite)
    fi

    echo "START lookback=${lookback} seed=${seed} horizon=${PREDICT_STEPS}"
    "${command[@]}" 2>&1 | tee "${LOG_DIR}/lookback_$(printf '%02d' "${lookback}")_seed_${seed}.log"
    echo "COMPLETE lookback=${lookback} seed=${seed}"
  done
done

expected_seeds="$(echo "${SEEDS}" | tr ' ' ',' | tr -s ',')"
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_gru_lookback.py" \
  --output-dir "${OUTPUT_DIR}" \
  --predict-steps "${PREDICT_STEPS}" \
  --expected-seeds "${expected_seeds}" \
  2>&1 | tee "${LOG_DIR}/summary.log"

echo "LOOKBACK_SCAN_COMPLETE=true"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

