#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$ROOT_DIR/model_code"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/processed_data}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/outputs_0910}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${WORKERS:-10}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-12}"
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
LOCK_DIR="$RESULTS_DIR/.run_heatmap_training_0910.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "This launcher may already be running: $LOCK_DIR" >&2
  echo "If no process is active, remove only this stale lock directory and retry." >&2
  exit 3
fi

pids=()
cleanup() {
  if (( ${#pids[@]} > 0 )); then
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

echo "PHASE 0/4: environment, split contract and synthetic model checks"
"$PYTHON_BIN" -c "import joblib, matplotlib, numpy, pandas, scipy, torch; print('PYTHON_DEPENDENCIES_OK=true'); print('TORCH_VERSION=' + torch.__version__); print('MPS_AVAILABLE=' + str(torch.backends.mps.is_available())); print('CUDA_AVAILABLE=' + str(torch.cuda.is_available()))" \
  | tee "$RESULTS_DIR/launcher_logs/environment.log"
"$PYTHON_BIN" "$MODEL_DIR/package_preflight.py" \
  | tee "$RESULTS_DIR/launcher_logs/package_preflight.log"
"$PYTHON_BIN" "$MODEL_DIR/smoke_test_heatmap_models.py" \
  | tee "$RESULTS_DIR/launcher_logs/smoke_test.log"

echo "PHASE 1/4: 120 runs = 3 models x 8 feature stages x 5 paired seeds"
echo "Workers=$WORKERS | epochs=$EPOCHS | batch_size=$BATCH_SIZE | patience=$PATIENCE"
for ((shard=0; shard<WORKERS; shard++)); do
  command=(
    "$PYTHON_BIN" "$MODEL_DIR/run_heatmap_shard.py"
    --shard-index "$shard"
    --num-shards "$WORKERS"
    --data-dir "$DATA_DIR"
    --results-dir "$RESULTS_DIR"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --patience "$PATIENCE"
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
  echo "At least one worker failed; inspect outputs_0910/launcher_logs." >&2
  exit 1
fi

echo "PHASE 2/4: freeze complete-validation summaries"
"$PYTHON_BIN" "$MODEL_DIR/summarize_heatmap_validation.py" \
  --results-dir "$RESULTS_DIR" \
  | tee "$RESULTS_DIR/launcher_logs/validation_summary.log"

echo "PHASE 3/4: render validation heatmaps and source-data matrices"
"$PYTHON_BIN" "$MODEL_DIR/plot_validation_heatmaps.py" \
  --results-dir "$RESULTS_DIR" \
  | tee "$RESULTS_DIR/launcher_logs/heatmap.log"

echo "PHASE 4/4: complete"
echo "SEED_RUNS=$RESULTS_DIR/validation_summary/validation_seed_runs.csv"
echo "CELL_SUMMARY=$RESULTS_DIR/validation_summary/validation_cell_summary.csv"
echo "HEATMAPS=$RESULTS_DIR/validation_summary/figures"
echo "TEST_DATA_OPENED=false"
