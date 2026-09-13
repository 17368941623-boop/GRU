#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$ROOT_DIR/model_code"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/processed_data}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/outputs_0907}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${WORKERS:-5}"
RUN_FULL_TEST="${RUN_FULL_TEST:-1}"
SAVE_PREDICTION_SEEDS="${SAVE_PREDICTION_SEEDS:-42}"
OVERWRITE="${OVERWRITE:-0}"

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || (( WORKERS > 10 )); then
  echo "WORKERS must be an integer from 1 to 10" >&2
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
LOCK_DIR="$RESULTS_DIR/.run_feature_ablation_0907.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "This launcher may already be running: $LOCK_DIR" >&2
  echo "If no process is active, remove only this stale lock directory and retry." >&2
  exit 3
fi

pids=()
cleanup() {
  if (( ${#pids[@]} > 0 )); then
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

echo "PHASE 0/4: environment, data-contract and synthetic model checks"
"$PYTHON_BIN" -c "import joblib, matplotlib, numpy, pandas, scipy, torch; print('PYTHON_DEPENDENCIES_OK=true'); print('TORCH_VERSION=' + torch.__version__); print('MPS_AVAILABLE=' + str(torch.backends.mps.is_available())); print('CUDA_AVAILABLE=' + str(torch.cuda.is_available()))" \
  | tee "$RESULTS_DIR/launcher_logs/environment.log"
"$PYTHON_BIN" "$MODEL_DIR/package_preflight.py" \
  | tee "$RESULTS_DIR/launcher_logs/package_preflight.log"
"$PYTHON_BIN" "$MODEL_DIR/smoke_test_feature_sets.py" \
  | tee "$RESULTS_DIR/launcher_logs/smoke_test.log"

echo "PHASE 1/4: 100 fixed feature-ablation runs with $WORKERS workers"
for ((shard=0; shard<WORKERS; shard++)); do
  command=("$PYTHON_BIN" "$MODEL_DIR/run_feature_shard.py"
    --shard-index "$shard" --num-shards "$WORKERS"
    --data-dir "$DATA_DIR" --results-dir "$RESULTS_DIR")
  if [[ "$OVERWRITE" == "1" ]]; then command+=(--overwrite); fi
  "${command[@]}" > "$RESULTS_DIR/launcher_logs/worker_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "WORKER_${shard}_PID=$!"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
pids=()
if (( failed != 0 )); then
  echo "At least one worker failed; inspect outputs_0907/launcher_logs." >&2
  exit 1
fi

echo "PHASE 2/4: freeze complete-validation ablation summary"
"$PYTHON_BIN" "$MODEL_DIR/summarize_feature_validation.py" \
  --results-dir "$RESULTS_DIR" \
  | tee "$RESULTS_DIR/launcher_logs/validation_summary.log"

if [[ "$RUN_FULL_TEST" == "1" ]]; then
  echo "PHASE 3/4: frozen checkpoints on complete Original 0715-BACK"
  "$PYTHON_BIN" "$MODEL_DIR/evaluate_feature_test.py" \
    --data-dir "$DATA_DIR" --results-dir "$RESULTS_DIR" \
    --save-prediction-seeds "$SAVE_PREDICTION_SEEDS" \
    | tee "$RESULTS_DIR/launcher_logs/full_test.log"
else
  echo "PHASE 3/4: test skipped; validation-only development is complete"
fi

echo "PHASE 4/4: complete"
echo "VALIDATION_SUMMARY=$RESULTS_DIR/validation_summary/validation_feature_summary.csv"
if [[ "$RUN_FULL_TEST" == "1" ]]; then
  echo "FULL_TEST_SUMMARY=$RESULTS_DIR/full_test_summary/test_feature_summary.csv"
fi
