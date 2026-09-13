#!/usr/bin/env python3
"""Frozen protocol for stage-1 global GRU lookback/hidden-size search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOOKBACKS = (15, 30, 45, 60, 75, 90)
HIDDEN_SIZES = (32, 64, 128, 256)
SEEDS = (42, 52, 62, 72, 82)
COMMON_ORIGIN_LOOKBACK = max(LOOKBACKS)
PREDICT_STEPS = 15
TARGET_COLUMN = "Thv"
LABEL_COLUMN = "Future_Thv_15step"
SELECTION_METRIC = "validation_rmse_k"


def tasks() -> list[tuple[int, int, int]]:
    # Hidden-size outer ordering makes modulo sharding load-balanced: with the
    # default ten workers, every worker receives three runs from each capacity.
    return [
        (lookback, hidden_size, seed)
        for hidden_size in HIDDEN_SIZES
        for lookback in LOOKBACKS
        for seed in SEEDS
    ]


def run_directory(
    results_dir: Path, lookback: int, hidden_size: int, seed: int
) -> Path:
    return (
        results_dir
        / "development"
        / f"horizon_{PREDICT_STEPS:02d}"
        / f"lookback_{lookback:03d}"
        / f"hidden_{hidden_size:03d}"
        / f"seed_{seed}"
    )


def completed_run_is_valid(
    results_dir: Path, lookback: int, hidden_size: int, seed: int
) -> bool:
    run_dir = run_directory(results_dir, lookback, hidden_size, seed)
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "study": "stage1_global_gru_lookback_hidden_search",
        "lookback": lookback,
        "hidden_size": hidden_size,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "seed": seed,
        "checkpoint_selection_metric": SELECTION_METRIC,
        "test_data_loaded": False,
        "engineered_features_used": False,
        "future_noncontrol_measurements_used": False,
        "future_label_used_as_input": False,
    }
    return all(metrics.get(key) == value for key, value in expected.items())


def protocol_payload() -> dict[str, Any]:
    return {
        "study": "stage1_global_gru_lookback_hidden_search",
        "study_type": "paired validation-only coarse two-dimensional search",
        "target": "Delta Thv(t+15)=Thv(t+15)-Thv(t), restored to absolute Thv for metrics",
        "lookbacks": list(LOOKBACKS),
        "hidden_sizes": list(HIDDEN_SIZES),
        "seeds": list(SEEDS),
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "history_input": "all enabled raw signals through forecast origin t",
        "known_future_control_input": "all enabled raw valve commands u(t)..u(t+14)",
        "engineered_features_used": False,
        "pcmci_used": False,
        "gru_layers": 1,
        "control_hidden_size": 32,
        "fusion_hidden_size": 64,
        "dropout": 0.1,
        "training_loss": "weighted Huber; train-defined lowest-delta 10% receives weight 3",
        "checkpoint_selection_metric": SELECTION_METRIC,
        "model_selection_split": "val_clean.pkl, Original validation trajectory only",
        "test_policy": (
            "Training, preflight and validation summary never deserialize test_clean.pkl "
            "or test_full_clean.pkl."
        ),
        "expected_runs": len(tasks()),
        "coarse_region_rule": (
            "cells with mean validation RMSE no greater than the global minimum mean "
            "plus the standard error of that minimum cell"
        ),
        "replication_note": (
            "five seeds quantify optimization variability and are not five independent "
            "process experiments"
        ),
    }
