#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
PREDICT_STEPS="${PREDICT_STEPS:-15}"
DEVICE="${DEVICE:-auto}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$PYTHON_BIN" "$ROOT_DIR/model_code/evaluate_test.py" --predict-steps "$PREDICT_STEPS" --device "$DEVICE"
"$PYTHON_BIN" "$ROOT_DIR/model_code/summarize.py" --stage test --predict-steps "$PREDICT_STEPS"
