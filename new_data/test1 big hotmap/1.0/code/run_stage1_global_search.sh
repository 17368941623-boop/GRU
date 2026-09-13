#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$CODE_DIR/.." && pwd)"
if [[ -z "${DATA_DIR:-}" ]]; then
  SERVER_DATA_DIR="$(cd "$PROJECT_DIR/.." && pwd)/processed_data"
  LOCAL_DATA_DIR="$(cd "$PROJECT_DIR/../.." && pwd)/processed_data"
  if [[ -f "$SERVER_DATA_DIR/data_build_config.json" ]]; then
    DATA_DIR="$SERVER_DATA_DIR"
  elif [[ -f "$LOCAL_DATA_DIR/data_build_config.json" ]]; then
    DATA_DIR="$LOCAL_DATA_DIR"
  else
    echo "processed_data was not found beside 1.0 or in the local archive root" >&2
    exit 2
  fi
fi
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/output}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${WORKERS:-10}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-12}"
LOADER_WORKERS="${LOADER_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
OVERWRITE="${OVERWRITE:-0}"

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || (( WORKERS > 10 )); then
  echo "WORKERS must be an integer from 1 to 10" >&2
  exit 2
fi
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if ! [[ "$PATIENCE" =~ ^[1-9][0-9]*$ ]]; then
  echo "PATIENCE must be a positive integer" >&2
  exit 2
fi
if [[ "$OVERWRITE" != "0" && "$OVERWRITE" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR/launcher_logs"
LOCK_DIR="$RESULTS_DIR/.stage1_global_search.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "The launcher may already be running: $LOCK_DIR" >&2
  echo "If no process is active, remove only this stale lock directory and retry." >&2
  exit 3
fi

pids=()
cleanup() {
  if [[ -n "${pids[*]-}" ]]; then
    for pid in "${pids[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

echo "PHASE 0/3: environment and no-leakage preflight"
"$PYTHON_BIN" -c "import joblib,numpy,pandas,scipy,torch; print('DEPENDENCIES_OK=true'); print('TORCH_VERSION='+torch.__version__); print('MPS_AVAILABLE='+str(torch.backends.mps.is_available())); print('CUDA_AVAILABLE='+str(torch.cuda.is_available()))" \
  | tee "$RESULTS_DIR/launcher_logs/environment.log"
"$PYTHON_BIN" "$CODE_DIR/preflight.py" \
  | tee "$RESULTS_DIR/launcher_logs/preflight.log"
"$PYTHON_BIN" "$CODE_DIR/smoke_test.py" \
  | tee "$RESULTS_DIR/launcher_logs/smoke_test.log"

echo "PHASE 1/3: 120 runs = 6 lookbacks x 4 hidden sizes x 5 paired seeds"
echo "WORKERS=$WORKERS EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE DEVICE=$DEVICE"
for ((shard=0; shard<WORKERS; shard++)); do
  command=(
    "$PYTHON_BIN" "$CODE_DIR/run_shard.py"
    --shard-index "$shard"
    --num-shards "$WORKERS"
    --data-dir "$DATA_DIR"
    --results-dir "$RESULTS_DIR"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --patience "$PATIENCE"
    --num-workers "$LOADER_WORKERS"
    --device "$DEVICE"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    command+=(--overwrite)
  fi
  "${command[@]}" > "$RESULTS_DIR/launcher_logs/worker_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "WORKER_${shard}_PID=$!"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
pids=()
if (( failed != 0 )); then
  echo "At least one worker failed; inspect output/launcher_logs." >&2
  exit 1
fi

echo "PHASE 2/3: validation-only summary and one-SE coarse region"
"$PYTHON_BIN" "$CODE_DIR/summarize_results.py" --results-dir "$RESULTS_DIR" \
  | tee "$RESULTS_DIR/launcher_logs/summary.log"

echo "PHASE 3/3: complete"
echo "SEED_RUNS=$RESULTS_DIR/validation_summary/validation_seed_runs.csv"
echo "CELL_SUMMARY=$RESULTS_DIR/validation_summary/validation_cell_summary.csv"
echo "MEAN_MATRIX=$RESULTS_DIR/validation_summary/validation_rmse_mean_matrix.csv"
echo "FINE_REGION=$RESULTS_DIR/validation_summary/coarse_search_recommendation.json"
echo "TEST_DATA_OPENED=false"
