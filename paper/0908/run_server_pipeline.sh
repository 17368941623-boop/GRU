#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs"

CONDA_ENV="${CONDA_ENV:-pytorch}"
PREDICT_STEPS="${PREDICT_STEPS:-15}"
BOOTSTRAP="${BOOTSTRAP:-20}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-12}"
LOADER_WORKERS="${LOADER_WORKERS:-0}"
MAX_WORKERS="${MAX_WORKERS:-5}"
RAM_GB_PER_WORKER="${RAM_GB_PER_WORKER:-3}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
FORCE_CAUSAL="${FORCE_CAUSAL:-0}"
ALLOW_CPU="${ALLOW_CPU:-0}"
STATUS_FILE="$ROOT_DIR/logs/server_pipeline.status"
CURRENT_STAGE="initializing"

write_status() {
  local state="$1"
  local message="$2"
  local temporary="$STATUS_FILE.tmp"
  printf 'state=%s\nstage=%s\nmessage=%s\nupdated_at=%s\npid=%s\n' \
    "$state" "$CURRENT_STAGE" "$message" "$(date '+%Y-%m-%d %H:%M:%S')" "$$" \
    >"$temporary"
  mv "$temporary" "$STATUS_FILE"
}

on_error() {
  local code="$?"
  write_status "FAILED" "Pipeline exited with code $code; inspect logs/server_pipeline.console.log and worker logs."
  exit "$code"
}

on_stop() {
  write_status "STOPPED" "Pipeline received an interrupt or termination signal."
  exit 143
}

fail_pipeline() {
  local code="$1"
  local message="$2"
  echo "$message" >&2
  write_status "FAILED" "$message"
  exit "$code"
}

trap on_error ERR
trap on_stop INT TERM
write_status "RUNNING" "Preparing the conda environment."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    eval "$("$CONDA_EXE" shell.bash hook)"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    fail_pipeline 2 "conda is unavailable. Run after conda init, or set PYTHON_BIN explicitly."
  fi
  conda activate "$CONDA_ENV"
  PYTHON_BIN="python"
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements-server.txt"
fi

"$PYTHON_BIN" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()}")'
GPU_COUNT="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
if (( GPU_COUNT < 1 )) && [[ "$ALLOW_CPU" != "1" ]]; then
  fail_pipeline 3 "No CUDA GPU is visible. Refusing an accidental 80-run CPU job; set ALLOW_CPU=1 to override."
fi

if [[ -n "${SERVER_WORKERS:-}" ]]; then
  WORKER_COUNT="$SERVER_WORKERS"
elif (( GPU_COUNT > 0 )); then
  WORKER_COUNT="$GPU_COUNT"
  if (( WORKER_COUNT > MAX_WORKERS )); then WORKER_COUNT="$MAX_WORKERS"; fi
else
  WORKER_COUNT=1
fi
if (( WORKER_COUNT < 1 )); then
  fail_pipeline 4 "SERVER_WORKERS must be positive."
fi
if [[ -z "${SERVER_WORKERS:-}" ]]; then
  AVAILABLE_RAM_GB="$($PYTHON_BIN -c 'import os; print(max(1, int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3)))')"
  RAM_WORKERS="$((AVAILABLE_RAM_GB / RAM_GB_PER_WORKER))"
  if (( RAM_WORKERS < 1 )); then RAM_WORKERS=1; fi
  if (( WORKER_COUNT > RAM_WORKERS )); then WORKER_COUNT="$RAM_WORKERS"; fi
fi

CURRENT_STAGE="preflight"
write_status "RUNNING" "Checking data boundaries, graph schema, model shape, and causal control prefix."
"$PYTHON_BIN" "$ROOT_DIR/model_code/preflight.py"
"$PYTHON_BIN" "$ROOT_DIR/model_code/smoke_test.py"

GRAPH_DIR="$ROOT_DIR/graphs/generated"
CAUSAL_READY=1
for graph_name in \
  discovery_audit.json cd_select_lag.json dkcdv_select_lag.json \
  dkcdl_select_lag.json physical_all_lags.json dkcdv_shuffled_lag.json \
  random_same_size.json; do
  if [[ ! -s "$GRAPH_DIR/$graph_name" ]]; then CAUSAL_READY=0; fi
done
if [[ "$FORCE_CAUSAL" == "1" ]]; then CAUSAL_READY=0; fi

CURRENT_STAGE="causal_discovery"
if (( CAUSAL_READY == 1 )); then
  write_status "RUNNING" "Existing complete causal graph package found; skipping discovery."
  echo "Causal graph package already complete. Set FORCE_CAUSAL=1 to rebuild it."
else
  write_status "RUNNING" "Running screened PCMCI and fixed-parent segment bootstrap."
  "$PYTHON_BIN" "$ROOT_DIR/model_code/discover_causal_graph.py" --bootstrap "$BOOTSTRAP"
fi

CURRENT_STAGE="development_training"
write_status "RUNNING" "Training 8 model variants x 10 seeds with $WORKER_COUNT worker(s) on $GPU_COUNT GPU(s)."
export PYTHONUNBUFFERED=1
pids=()
for ((worker=0; worker<WORKER_COUNT; worker++)); do
  if (( GPU_COUNT > 0 )); then
    worker_device="cuda:$((worker % GPU_COUNT))"
  else
    worker_device="cpu"
  fi
  echo "worker=$worker device=$worker_device" | tee -a "$ROOT_DIR/logs/server_worker_map.log"
  "$PYTHON_BIN" "$ROOT_DIR/model_code/run_shard.py" \
    --worker-index "$worker" --worker-count "$WORKER_COUNT" \
    --predict-steps "$PREDICT_STEPS" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --patience "$PATIENCE" \
    --num-workers "$LOADER_WORKERS" --device "$worker_device" \
    >"$ROOT_DIR/logs/server_worker_${worker}.out.log" \
    2>"$ROOT_DIR/logs/server_worker_${worker}.err.log" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if (( failed != 0 )); then
  fail_pipeline 5 "At least one training worker failed. Inspect logs/server_worker_*.err.log."
fi

CURRENT_STAGE="validation_summary"
write_status "RUNNING" "Aggregating validation metrics and checking the 80-run matrix."
"$PYTHON_BIN" "$ROOT_DIR/model_code/summarize.py" \
  --stage validation --predict-steps "$PREDICT_STEPS"
SUMMARY_STATUS="$ROOT_DIR/outputs/validation_summary/horizon_$(printf '%02d' "$PREDICT_STEPS")/summary_status.json"
"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p["complete"], p' "$SUMMARY_STATUS"

CURRENT_STAGE="complete"
write_status "COMPLETE" "Causal graphs, all development runs, and validation summary are complete. Frozen test was not opened."
echo "Development training complete. Review validation summary before running run_frozen_test.sh."
