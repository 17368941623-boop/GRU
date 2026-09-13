#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT_DIR/logs"
PID_FILE="$ROOT_DIR/logs/server_pipeline.pid"
CONSOLE_LOG="$ROOT_DIR/logs/server_pipeline.console.log"

if [[ -s "$PID_FILE" ]]; then
  existing_pid="$(tr -d '[:space:]' <"$PID_FILE")"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Training is already running with PID $existing_pid."
    echo "Log: $CONSOLE_LOG"
    exit 0
  fi
fi

if command -v setsid >/dev/null 2>&1; then
  nohup setsid bash "$ROOT_DIR/run_server_pipeline.sh" \
    >"$CONSOLE_LOG" 2>&1 </dev/null &
else
  nohup bash "$ROOT_DIR/run_server_pipeline.sh" \
    >"$CONSOLE_LOG" 2>&1 </dev/null &
fi
pipeline_pid="$!"
printf '%s\n' "$pipeline_pid" >"$PID_FILE"
echo "Training started in the background. PID: $pipeline_pid"
echo "Progress: tail -f '$CONSOLE_LOG'"
echo "Status:   cat '$ROOT_DIR/logs/server_pipeline.status'"
echo "This pipeline stops after validation summary; it does not open the frozen test set."
