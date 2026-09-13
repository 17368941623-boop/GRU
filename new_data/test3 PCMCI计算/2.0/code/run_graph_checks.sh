#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-pytorch}"

conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/validate_graph_constraints.py"
conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/test_graph_constraints.py" -v
