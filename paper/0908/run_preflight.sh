#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$PYTHON_BIN" "$ROOT_DIR/model_code/preflight.py"
"$PYTHON_BIN" "$ROOT_DIR/model_code/smoke_test.py"
