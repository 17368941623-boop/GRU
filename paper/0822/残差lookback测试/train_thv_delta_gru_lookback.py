#!/usr/bin/env python3
"""Formal GRU lookback scan for leakage-safe Delta Thv(t+h) prediction.

The architecture screen selected GRU.  This entry point fixes that choice and
tests lookbacks 10--60 in steps of five.  Every lookback uses forecast origins
that have at least 60 historical samples, so the validation labels and sample
count are identical across the complete scan.  Checkpoints are selected only
on Original 260501 validation data; after selection, a fixed Original 0715-BACK
source-row window 0--7300 is evaluated as a secondary diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from train_thv_delta_rnn_compare import run_experiment


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "processed_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "gru_lookback_outputs_val260501_test0715BACK"

ALLOWED_GRU_LOOKBACKS = tuple(range(10, 61, 5))
COMMON_GRU_ORIGIN_LOOKBACK = max(ALLOWED_GRU_LOOKBACKS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Formal GRU lookback selection for Delta Thv(t+h), with known "
            "u(t)..u(t+h-1) controls and common 60-step history origins."
        )
    )
    parser.add_argument(
        "--lookback",
        type=int,
        required=True,
        choices=ALLOWED_GRU_LOOKBACKS,
        help="History length in 10-second samples: 10, 15, ..., 60.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--predict-steps",
        type=int,
        default=15,
        choices=(5, 10, 15, 20, 30),
        help=(
            "Forecast horizon h in 10-second samples. Only valve controls are "
            "read after t, through u(t+h-1)."
        ),
    )
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

    # Fixed study settings: they are deliberately not exposed as CLI choices.
    args.model = "gru"
    args.common_origin_lookback = COMMON_GRU_ORIGIN_LOOKBACK
    args.study_name = "Thv_delta_gru_formal_lookback_scan"
    args.comparison_policy = (
        "GRU architecture, data, common 60-step origins, future-control branch, "
        "loss, optimizer, and checkpoint rule fixed; only lookback and seed vary"
    )
    args.evaluate_test_window = True
    args.test_window_start = 0
    args.test_window_end = 7300
    return args


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
