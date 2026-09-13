#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
BOOTSTRAP="${BOOTSTRAP:-20}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$PYTHON_BIN" "$ROOT_DIR/model_code/discover_causal_graph.py" --bootstrap "$BOOTSTRAP"
