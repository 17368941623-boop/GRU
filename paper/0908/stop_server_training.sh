#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT_DIR/logs/server_pipeline.pid"
if [[ ! -s "$PID_FILE" ]]; then
  echo "No server pipeline PID file was found."
  exit 0
fi
pipeline_pid="$(tr -d '[:space:]' <"$PID_FILE")"
if [[ ! "$pipeline_pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pipeline_pid" 2>/dev/null; then
  echo "The recorded pipeline is no longer running."
  exit 0
fi

command_line="$(ps -p "$pipeline_pid" -o args= 2>/dev/null || true)"
if [[ "$command_line" != *"$ROOT_DIR/run_server_pipeline.sh"* ]]; then
  echo "PID $pipeline_pid does not match this training pipeline; nothing was stopped." >&2
  exit 2
fi

if [[ "$(ps -p "$pipeline_pid" -o pgid= | tr -d '[:space:]')" == "$pipeline_pid" ]]; then
  kill -TERM -- "-$pipeline_pid"
else
  terminate_tree() {
    local parent="$1"
    local child
    while IFS= read -r child; do
      if [[ -n "$child" ]]; then terminate_tree "$child"; fi
    done < <(pgrep -P "$parent" 2>/dev/null || true)
    kill -TERM "$parent" 2>/dev/null || true
  }
  terminate_tree "$pipeline_pid"
fi
echo "Stop signal sent to pipeline PID $pipeline_pid."
