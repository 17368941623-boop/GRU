#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$ROOT_DIR/model_code"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/processed_data}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/outputs_0906}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LSTM_WORKERS="${LSTM_WORKERS:-2}"
OVERWRITE="${OVERWRITE:-0}"

if ! [[ "$LSTM_WORKERS" =~ ^[1-9][0-9]*$ ]] || (( LSTM_WORKERS > 10 )); then
  echo "LSTM_WORKERS must be an integer from 1 to 10" >&2
  exit 2
fi
if [[ "$OVERWRITE" != "0" && "$OVERWRITE" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR/launcher_logs"
LOCK_DIR="$RESULTS_DIR/.lstm_family_0906.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "The LSTM-family launcher may already be running: $LOCK_DIR" >&2
  exit 3
fi

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

echo "LSTM_FAMILY_RUNS=35"
echo "LSTM_FAMILY_WORKERS=$LSTM_WORKERS"
for ((shard=0; shard<LSTM_WORKERS; shard++)); do
  command=(
    "$PYTHON_BIN" "$MODEL_DIR/run_training_shard.py"
    --group lstm_family
    --shard-index "$shard"
    --num-shards "$LSTM_WORKERS"
    --data-dir "$DATA_DIR"
    --results-dir "$RESULTS_DIR"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    command+=(--overwrite)
  fi
  "${command[@]}" \
    > "$RESULTS_DIR/launcher_logs/lstm_family_worker_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "LSTM_WORKER_${shard}_PID=$!"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
pids=()
if (( failed != 0 )); then
  echo "At least one LSTM-family worker failed; inspect launcher_logs." >&2
  exit 1
fi
echo "LSTM_FAMILY_COMPLETE=true"
