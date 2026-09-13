#!/usr/bin/env python3
"""Final validation-only GRU lookback scan for Delta Thv(t+15).

Every candidate is evaluated from the same forecast origins requiring 120
historical samples. The test split is deliberately not loaded: lookback
selection uses Original 260501 validation only.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Required by deterministic CuBLAS execution on CUDA 10.2 and newer. This must
# be set before importing torch through the shared training implementation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from train_thv_delta_rnn_compare import run_experiment


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "processed_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "gru_lookback_final_unified_val260501"

FINAL_LOOKBACKS = (15, 30, 40, 45, 50, 55, 60, 65, 70, 75, 90, 120)
COMMON_ORIGIN_LOOKBACK = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lookback", type=int, required=True, choices=FINAL_LOOKBACKS
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--predict-steps", type=int, default=15, choices=(15,))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--control-hidden-size", type=int, default=32)
    parser.add_argument("--fusion-hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rapid-quantile", type=float, default=0.10)
    parser.add_argument("--rapid-weight", type=float, default=3.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.model = "gru"
    args.common_origin_lookback = COMMON_ORIGIN_LOOKBACK
    args.study_name = "Thv_delta_gru_final_unified_lookback_scan"
    args.comparison_policy = (
        "GRU architecture, processed-data version, common 120-step origins, "
        "future-control branch, weighted Huber loss, optimizer, early-stopping "
        "metric, and hyperparameters fixed; only lookback and random seed vary"
    )
    # The test set must remain unread until lookback selection is complete.
    args.evaluate_test_window = False
    return args


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
