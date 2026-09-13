#!/usr/bin/env python3
"""Frozen protocol for stage-2 local GRU lookback/hidden-size search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOOKBACKS = (50, 60, 70, 80, 90, 100)
HIDDEN_SIZES = (128, 192, 256, 320)
SEEDS = (42, 52, 62, 72, 82)
COMMON_ORIGIN_LOOKBACK = max(LOOKBACKS)
PREDICT_STEPS = 15
SAMPLE_INTERVAL_SECONDS = 10
TARGET_COLUMN = "Thv"
LABEL_COLUMN = "Future_Thv_15step"
CHECKPOINT_SELECTION_METRIC = "validation_rmse_k"
PRIMARY_RANKING_METRIC = "hard_cooling_pooled_rmse_k"
STUDY_NAME = "stage2_fine_gru_lookback_hidden_search"

# Target timestamps are inclusive. Each interval contains exactly 360 valid
# prediction targets in the frozen 260428 validation trajectory.
DIFFICULT_WINDOWS = (
    ("H1", "2026-04-29 04:37:26", "2026-04-29 05:37:16"),
    ("H2", "2026-04-29 08:40:16", "2026-04-29 09:40:06"),
    ("H3", "2026-04-29 10:24:26", "2026-04-29 11:24:16"),
    ("H4", "2026-04-29 14:18:46", "2026-04-29 15:18:36"),
)
EXPECTED_TARGETS_PER_DIFFICULT_WINDOW = 360
EXPECTED_DIFFICULT_TARGETS = (
    len(DIFFICULT_WINDOWS) * EXPECTED_TARGETS_PER_DIFFICULT_WINDOW
)


def tasks() -> list[tuple[int, int, int]]:
    # With 120 runs and the default 24 workers, each worker receives five runs.
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
    predictions_path = run_dir / "validation_predictions.csv"
    if not (
        metrics_path.is_file()
        and checkpoint_path.is_file()
        and predictions_path.is_file()
    ):
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "study": STUDY_NAME,
        "lookback": lookback,
        "hidden_size": hidden_size,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "seed": seed,
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "primary_ranking_metric": PRIMARY_RANKING_METRIC,
        "difficult_validation_targets": EXPECTED_DIFFICULT_TARGETS,
        "test_data_loaded": False,
        "engineered_features_used": False,
        "future_noncontrol_measurements_used": False,
        "future_label_used_as_input": False,
    }
    return all(metrics.get(key) == value for key, value in expected.items())


def protocol_payload() -> dict[str, Any]:
    return {
        "study": STUDY_NAME,
        "study_type": "paired validation-only local two-dimensional search",
        "target": "Delta Thv(t+15)=Thv(t+15)-Thv(t), restored to absolute Thv for metrics",
        "lookbacks": list(LOOKBACKS),
        "hidden_sizes": list(HIDDEN_SIZES),
        "seeds": list(SEEDS),
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "history_input": "all enabled raw signals through forecast origin t",
        "known_future_control_input": "all enabled raw valve commands u(t)..u(t+14)",
        "engineered_features_used": False,
        "pcmci_used": False,
        "gru_layers": 1,
        "control_hidden_size": 32,
        "fusion_hidden_size": 64,
        "dropout": 0.1,
        "training_loss": "weighted Huber; train-defined lowest-delta 10% receives weight 3",
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "primary_ranking_metric": PRIMARY_RANKING_METRIC,
        "difficult_window_time_semantics": "inclusive future target timestamp",
        "difficult_windows": [
            {"id": name, "start": start, "end": end}
            for name, start, end in DIFFICULT_WINDOWS
        ],
        "expected_targets_per_difficult_window": EXPECTED_TARGETS_PER_DIFFICULT_WINDOW,
        "expected_difficult_targets": EXPECTED_DIFFICULT_TARGETS,
        "model_selection_split": "val_clean.pkl, Original 260428 validation trajectory only",
        "test_policy": (
            "Training, preflight and validation summary never deserialize test_clean.pkl "
            "or test_full_clean.pkl."
        ),
        "expected_runs": len(tasks()),
        "one_se_rule": (
            "cells with mean hard-cooling pooled RMSE no greater than the global "
            "minimum mean plus the standard error of that minimum cell"
        ),
        "replication_note": (
            "five seeds quantify optimization variability and are not five independent "
            "process experiments"
        ),
    }
