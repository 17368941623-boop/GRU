#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${CODE_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-$(cd "${EXPERIMENT_DIR}/.." && pwd)/processed_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/output}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pytorch}"
BOOTSTRAP="${BOOTSTRAP:-50}"
RUN_LEVEL_SENSITIVITY="${RUN_LEVEL_SENSITIVITY:-1}"
MAX_SAMPLES_PER_SEGMENT="${MAX_SAMPLES_PER_SEGMENT:-3000}"
MAX_BOOTSTRAP_CANDIDATES="${MAX_BOOTSTRAP_CANDIDATES:-0}"
MAX_EXPORT_EDGES="${MAX_EXPORT_EDGES:-200}"
RUN_HMIX_DIAGNOSTICS="${RUN_HMIX_DIAGNOSTICS:-1}"
RUN_HMIX_TARGETED_PCMCI="${RUN_HMIX_TARGETED_PCMCI:-1}"
HMIX_MAX_LAG="${HMIX_MAX_LAG:-60}"
HMIX_MAX_SAMPLES_PER_SEGMENT="${HMIX_MAX_SAMPLES_PER_SEGMENT:-0}"

mkdir -p "${OUTPUT_DIR}"

if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python)
else
  PYTHON_CMD=("${PYTHON_BIN:-python}")
fi

"${PYTHON_CMD[@]}" "${CODE_DIR}/preflight.py" \
  --data-dir "${DATA_DIR}" \
  --output "${OUTPUT_DIR}/preflight.json" \
  | tee "${OUTPUT_DIR}/preflight.log"

if [[ "${RUN_HMIX_DIAGNOSTICS}" == "1" ]]; then
  "${PYTHON_CMD[@]}" "${CODE_DIR}/hmix_lag_diagnostics.py" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}/hmix_diagnostics" \
    --max-lag "${HMIX_MAX_LAG}" \
    | tee "${OUTPUT_DIR}/hmix_diagnostics.log"
fi

"${PYTHON_CMD[@]}" "${CODE_DIR}/discover_pcmci.py" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}/difference" \
  --transform difference \
  --bootstrap "${BOOTSTRAP}" \
  --max-samples-per-segment "${MAX_SAMPLES_PER_SEGMENT}" \
  --max-bootstrap-candidates "${MAX_BOOTSTRAP_CANDIDATES}" \
  --max-export-edges "${MAX_EXPORT_EDGES}" \
  | tee "${OUTPUT_DIR}/difference_run.log"

if [[ "${RUN_LEVEL_SENSITIVITY}" == "1" ]]; then
  "${PYTHON_CMD[@]}" "${CODE_DIR}/discover_pcmci.py" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}/level" \
    --transform level \
    --bootstrap "${BOOTSTRAP}" \
    --max-samples-per-segment "${MAX_SAMPLES_PER_SEGMENT}" \
    --max-bootstrap-candidates "${MAX_BOOTSTRAP_CANDIDATES}" \
    --max-export-edges "${MAX_EXPORT_EDGES}" \
    | tee "${OUTPUT_DIR}/level_run.log"
  "${PYTHON_CMD[@]}" "${CODE_DIR}/compare_transforms.py" \
    --output-root "${OUTPUT_DIR}" \
    | tee "${OUTPUT_DIR}/transform_comparison.log"
fi

if [[ "${RUN_HMIX_TARGETED_PCMCI}" == "1" ]]; then
  HMIX_TRANSFORMS=(difference)
  if [[ "${RUN_LEVEL_SENSITIVITY}" == "1" ]]; then
    HMIX_TRANSFORMS+=(level)
  fi
  for TRANSFORM in "${HMIX_TRANSFORMS[@]}"; do
    "${PYTHON_CMD[@]}" "${CODE_DIR}/discover_hmix_pcmci.py" \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}/hmix_targeted/${TRANSFORM}" \
      --transform "${TRANSFORM}" \
      --max-lag "${HMIX_MAX_LAG}" \
      --bootstrap "${BOOTSTRAP}" \
      --max-samples-per-segment "${HMIX_MAX_SAMPLES_PER_SEGMENT}" \
      | tee "${OUTPUT_DIR}/hmix_targeted_${TRANSFORM}.log"
  done
fi

echo "PCMCI finished. Copy the complete output directory back for analysis: ${OUTPUT_DIR}"
