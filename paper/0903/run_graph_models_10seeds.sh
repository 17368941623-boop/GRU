#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$ROOT_DIR/model测试"
DATA_DIR="$ROOT_DIR/processed_data"
RESULTS_DIR="$ROOT_DIR/outputs_graph_10seeds"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${WORKERS:-5}"

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer" >&2
  exit 2
fi
if (( WORKERS > 40 )); then
  echo "WORKERS cannot exceed the 40 expected graph runs" >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR/launcher_logs"
export PYTORCH_ENABLE_MPS_FALLBACK=1

"$PYTHON_BIN" "$MODEL_DIR/smoke_test_models.py" \
  > "$RESULTS_DIR/launcher_logs/smoke_test.log" 2>&1

pids=()
for ((shard=0; shard<WORKERS; shard++)); do
  "$PYTHON_BIN" "$MODEL_DIR/train_validation_batch.py" \
    --study graph \
    --shard-index "$shard" \
    --num-shards "$WORKERS" \
    --data-dir "$DATA_DIR" \
    --results-dir "$RESULTS_DIR" \
    > "$RESULTS_DIR/launcher_logs/worker_${shard}.log" 2>&1 &
  worker_pid="$!"
  pids+=("$worker_pid")
  echo "GRAPH_WORKER_${shard}_PID=$worker_pid"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "At least one graph worker failed. Check outputs_graph_10seeds/launcher_logs/." >&2
  exit 1
fi

"$PYTHON_BIN" "$MODEL_DIR/summarize_validation_study.py" \
  --study graph \
  --results-dir "$RESULTS_DIR" \
  > "$RESULTS_DIR/launcher_logs/validation_summary.log" 2>&1

echo "GRAPH_STUDY_COMPLETE=$RESULTS_DIR"
