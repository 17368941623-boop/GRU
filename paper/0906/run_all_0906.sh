#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$ROOT_DIR/model_code"
DATA_DIR="$ROOT_DIR/processed_data"
RESULTS_DIR="$ROOT_DIR/outputs_0906"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GRU_WORKERS="${GRU_WORKERS:-3}"
LSTM_WORKERS="${LSTM_WORKERS:-2}"
RUN_FULL_TEST="${RUN_FULL_TEST:-1}"
SAVE_PREDICTION_SEEDS="${SAVE_PREDICTION_SEEDS:-42}"
OVERWRITE="${OVERWRITE:-0}"

for worker_value in "$GRU_WORKERS" "$LSTM_WORKERS"; do
  if ! [[ "$worker_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "GRU_WORKERS and LSTM_WORKERS must be positive integers" >&2
    exit 2
  fi
done
TOTAL_WORKERS=$((GRU_WORKERS + LSTM_WORKERS))
if (( TOTAL_WORKERS > 10 )); then
  echo "Combined workers cannot exceed 10" >&2
  exit 2
fi
if [[ "$RUN_FULL_TEST" != "0" && "$RUN_FULL_TEST" != "1" ]]; then
  echo "RUN_FULL_TEST must be 0 or 1" >&2
  exit 2
fi
if [[ "$OVERWRITE" != "0" && "$OVERWRITE" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR/launcher_logs"
LOCK_DIR="$RESULTS_DIR/.run_all_0906.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another complete launcher may already be running: $LOCK_DIR" >&2
  echo "If no process is active, remove only this stale lock directory and retry." >&2
  exit 3
fi

child_pids=()
cleanup() {
  for pid in "${child_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

echo "PHASE 0/4: package and environment preflight"
"$PYTHON_BIN" -c "import joblib, matplotlib, numpy, pandas, scipy, torch; print('PYTHON_DEPENDENCIES_OK=true'); print('TORCH_VERSION=' + torch.__version__); print('MPS_AVAILABLE=' + str(torch.backends.mps.is_available())); print('CUDA_AVAILABLE=' + str(torch.cuda.is_available()))" \
  | tee "$RESULTS_DIR/launcher_logs/environment.log"
"$PYTHON_BIN" "$MODEL_DIR/package_preflight.py" \
  | tee "$RESULTS_DIR/launcher_logs/package_preflight.log"
"$PYTHON_BIN" "$MODEL_DIR/smoke_test_models.py" \
  | tee "$RESULTS_DIR/launcher_logs/smoke_test.log"

echo "PHASE 1/4: 85 runs in two families with $TOTAL_WORKERS total workers"
DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_DIR" PYTHON_BIN="$PYTHON_BIN" \
  GRU_WORKERS="$GRU_WORKERS" OVERWRITE="$OVERWRITE" \
  bash "$ROOT_DIR/run_gru_family_0906.sh" \
  > "$RESULTS_DIR/launcher_logs/gru_family_launcher.log" 2>&1 &
gru_launcher_pid="$!"
child_pids+=("$gru_launcher_pid")
echo "GRU_FAMILY_LAUNCHER_PID=$gru_launcher_pid"

DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_DIR" PYTHON_BIN="$PYTHON_BIN" \
  LSTM_WORKERS="$LSTM_WORKERS" OVERWRITE="$OVERWRITE" \
  bash "$ROOT_DIR/run_lstm_family_0906.sh" \
  > "$RESULTS_DIR/launcher_logs/lstm_family_launcher.log" 2>&1 &
lstm_launcher_pid="$!"
child_pids+=("$lstm_launcher_pid")
echo "LSTM_FAMILY_LAUNCHER_PID=$lstm_launcher_pid"

failed=0
if ! wait "$gru_launcher_pid"; then
  failed=1
fi
if ! wait "$lstm_launcher_pid"; then
  failed=1
fi
child_pids=()
if (( failed != 0 )); then
  echo "At least one model family failed. Inspect outputs_0906/launcher_logs/." >&2
  exit 1
fi

echo "PHASE 2/4: complete-validation summary"
"$PYTHON_BIN" "$MODEL_DIR/summarize_validation.py" \
  --results-dir "$RESULTS_DIR" \
  | tee "$RESULTS_DIR/launcher_logs/validation_summary.log"

if [[ "$RUN_FULL_TEST" == "1" ]]; then
  echo "PHASE 3/4: frozen-model inference on complete 0715-BACK"
  "$PYTHON_BIN" "$MODEL_DIR/evaluate_full_test.py" \
    --data-dir "$DATA_DIR" \
    --results-dir "$RESULTS_DIR" \
    --save-prediction-seeds "$SAVE_PREDICTION_SEEDS" \
    | tee "$RESULTS_DIR/launcher_logs/full_test.log"
else
  echo "PHASE 3/4: skipped because RUN_FULL_TEST=0"
fi

echo "PHASE 4/4: complete"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "VALIDATION_SUMMARY=$RESULTS_DIR/validation_summary/validation_model_summary.csv"
if [[ "$RUN_FULL_TEST" == "1" ]]; then
  echo "FULL_TEST_SUMMARY=$RESULTS_DIR/full_test_summary/test_model_summary.csv"
fi
