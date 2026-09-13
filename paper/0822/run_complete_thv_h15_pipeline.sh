#!/usr/bin/env bash
set -euo pipefail

# Optional unattended pipeline: finish the GRU lookback scan, read the best
# validation-selected lookback, then launch the complete model ablation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-full}"
PREDICT_STEPS="${PREDICT_STEPS:-15}"

if [[ "${PREDICT_STEPS}" != "15" ]]; then
  echo "This pipeline is frozen for predict-steps=15"
  exit 2
fi
if [[ "${PROFILE}" != "quick" && "${PROFILE}" != "full" ]]; then
  echo "Usage: bash run_complete_thv_h15_pipeline.sh [quick|full]"
  exit 2
fi

echo "STAGE A: complete extended baseline lookback scan"
bash "${ROOT_DIR}/残差lookback测试/run_gru_extended_lookback_h15.sh"

SUMMARY_LOG="${ROOT_DIR}/残差lookback测试/logs_extended_h15/summary.log"
if [[ ! -f "${SUMMARY_LOG}" ]]; then
  echo "Missing lookback summary log: ${SUMMARY_LOG}"
  exit 1
fi
BEST_LOOKBACK="$(awk -F= '$1 == "BEST_LOOKBACK_BY_RAPID_VALIDATION_RMSE" {value=$2} END {print value}' "${SUMMARY_LOG}")"
if [[ ! "${BEST_LOOKBACK}" =~ ^[0-9]+$ ]]; then
  echo "Could not parse the best lookback from ${SUMMARY_LOG}"
  exit 1
fi

echo "BEST_LOOKBACK_FROM_260501=${BEST_LOOKBACK}"
echo "STAGE B: complete model/hyperparameter ablation"
bash "${ROOT_DIR}/model测试/run_all_model_tests.sh" "${BEST_LOOKBACK}" "${PROFILE}"

echo "COMPLETE_THV_H15_PIPELINE=true"
