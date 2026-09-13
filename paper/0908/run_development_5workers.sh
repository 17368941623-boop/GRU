#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
PREDICT_STEPS="${PREDICT_STEPS:-15}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-auto}"
LOADER_WORKERS="${LOADER_WORKERS:-0}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT_DIR/logs"
GPU_COUNT="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
if [[ -n "${WORKER_COUNT:-}" ]]; then
  ACTIVE_WORKERS="$WORKER_COUNT"
elif [[ "$DEVICE" == "auto" || "$DEVICE" == "cuda" ]]; then
  ACTIVE_WORKERS="$GPU_COUNT"
  if (( ACTIVE_WORKERS < 1 )); then ACTIVE_WORKERS=1; fi
  if (( ACTIVE_WORKERS > 5 )); then ACTIVE_WORKERS=5; fi
else
  ACTIVE_WORKERS=1
fi
pids=()
for ((worker=0; worker<ACTIVE_WORKERS; worker++)); do
  worker_device="$DEVICE"
  if (( GPU_COUNT > 0 )) && [[ "$DEVICE" == "auto" || "$DEVICE" == "cuda" ]]; then
    worker_device="cuda:$((worker % GPU_COUNT))"
  fi
  "$PYTHON_BIN" "$ROOT_DIR/model_code/run_shard.py" \
    --worker-index "$worker" --worker-count "$ACTIVE_WORKERS" \
    --predict-steps "$PREDICT_STEPS" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --num-workers "$LOADER_WORKERS" \
    --device "$worker_device" \
    >"$ROOT_DIR/logs/worker_${worker}.out.log" \
    2>"$ROOT_DIR/logs/worker_${worker}.err.log" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "One or more workers failed. Check logs/." >&2
  exit 1
fi
"$PYTHON_BIN" "$ROOT_DIR/model_code/summarize.py" --stage validation --predict-steps "$PREDICT_STEPS"
