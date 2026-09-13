#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$ROOT_DIR/model测试"
DATA_DIR="$ROOT_DIR/processed_data"
RESULTS_DIR="$ROOT_DIR/outputs_tcn_5seeds"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$RESULTS_DIR/launcher_logs"
export PYTORCH_ENABLE_MPS_FALLBACK=1

"$PYTHON_BIN" "$MODEL_DIR/smoke_test_models.py" \
  > "$RESULTS_DIR/launcher_logs/smoke_test.log" 2>&1

"$PYTHON_BIN" "$MODEL_DIR/train_validation_batch.py" \
  --study tcn \
  --shard-index 0 \
  --num-shards 1 \
  --data-dir "$DATA_DIR" \
  --results-dir "$RESULTS_DIR" \
  > "$RESULTS_DIR/launcher_logs/tcn_training.log" 2>&1

"$PYTHON_BIN" "$MODEL_DIR/summarize_validation_study.py" \
  --study tcn \
  --results-dir "$RESULTS_DIR" \
  > "$RESULTS_DIR/launcher_logs/validation_summary.log" 2>&1

echo "TCN_STUDY_COMPLETE=$RESULTS_DIR"

