#!/usr/bin/env python3
"""Fast integrity checks that do not open the held-out test pickle."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from study_protocol import (
    COMMON_ORIGIN_LOOKBACK,
    LOOKBACK,
    PREDICT_STEPS,
    SELECTION_METRIC,
    all_tasks,
    tasks_for_group,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "processed_data"


def unique_task_ids() -> set[tuple[str, str, int]]:
    return {
        (str(spec["model"]), str(spec["config_id"]), int(seed))
        for spec, seed in all_tasks()
    }


def main() -> None:
    required = (
        "data_build_config.json",
        "feature_catalog.json",
        "split_manifest.csv",
        "dataset_summary.csv",
        "train_clean.pkl",
        "val_clean.pkl",
        "test_full_clean.pkl",
    )
    missing = [name for name in required if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed-data files: {missing}")
    config = json.loads(
        (DATA_DIR / "data_build_config.json").read_text(encoding="utf-8")
    )
    expected = {
        "validation_parent": "260501",
        "validation_variant": "Original only",
        "test_parent": "0715-BACK",
        "test_variant": "Original only",
        "future_features_used": False,
        "random_row_split": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Data contract mismatch for {key}: {config.get(key)!r}")
    if "Future_Thv_15step" not in config.get("future_labels", []):
        raise ValueError("Future_Thv_15step is missing from the processed data contract")

    # Development data may be inspected before training; the held-out test pickle is
    # intentionally not loaded by this preflight.
    train = joblib.load(DATA_DIR / "train_clean.pkl")
    validation = joblib.load(DATA_DIR / "val_clean.pkl")
    if "Future_Thv_15step" not in train.columns or "Future_Thv_15step" not in validation.columns:
        raise KeyError("Required 15-step Thv label is absent")
    if set(validation["parent_id"].astype(str).unique()) != {"260501"}:
        raise ValueError("Validation is not exclusively parent 260501")
    if set(validation["source_variant"].astype(str).unique()) != {"Original"}:
        raise ValueError("Validation contains augmented rows")
    if "0715-BACK" in set(train["parent_id"].astype(str).unique()):
        raise ValueError("Held-out test parent appears in training data")
    if "260501" in set(train["parent_id"].astype(str).unique()):
        raise ValueError("Validation parent appears in training data")

    tasks = all_tasks()
    group_counts = {
        "gru_family": len(tasks_for_group("gru_family")),
        "lstm_family": len(tasks_for_group("lstm_family")),
    }
    if len(tasks) != 85 or len(unique_task_ids()) != 85:
        raise AssertionError("Expected 85 unique model/seed tasks")
    if group_counts != {"gru_family": 50, "lstm_family": 35}:
        raise AssertionError(f"Unexpected training-group counts: {group_counts}")
    print("PACKAGE_PREFLIGHT_OK=true")
    print(f"TRAIN_ROWS_RAW={len(train)}")
    print(f"VALIDATION_ROWS={len(validation)}")
    print("TEST_PICKLE_OPENED=false")
    print(f"LOOKBACK={LOOKBACK}")
    print(f"COMMON_ORIGIN_LOOKBACK={COMMON_ORIGIN_LOOKBACK}")
    print(f"PREDICT_STEPS={PREDICT_STEPS}")
    print(f"CHECKPOINT_SELECTION_METRIC={SELECTION_METRIC}")
    print(f"EXPECTED_RUNS={len(tasks)}")
    print(f"GRU_FAMILY_RUNS={group_counts['gru_family']}")
    print(f"LSTM_FAMILY_RUNS={group_counts['lstm_family']}")


if __name__ == "__main__":
    main()
