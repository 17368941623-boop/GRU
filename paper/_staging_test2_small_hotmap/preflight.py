#!/usr/bin/env python3
"""Audit the stage-2 package and development split without opening test data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import (
    DIFFICULT_WINDOWS,
    EXPECTED_DIFFICULT_TARGETS,
    EXPECTED_TARGETS_PER_DIFFICULT_WINDOW,
    HIDDEN_SIZES,
    LOOKBACKS,
    PREDICT_STEPS,
    SAMPLE_INTERVAL_SECONDS,
    SEEDS,
    tasks,
)
from train_fine_cell import (
    LABEL_COLUMN,
    PREDICT_STEPS,
    TARGET_COLUMN,
    audit_origins,
    build_common_origins,
    discover_default_data_dir,
    load_development_data,
    read_contract,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = discover_default_data_dir()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if len(tasks()) != 120:
        raise AssertionError("The frozen grid must contain exactly 120 runs")
    if len(LOOKBACKS) != 6 or len(HIDDEN_SIZES) != 4 or len(SEEDS) != 5:
        raise AssertionError("The grid dimensions changed")
    if len(DIFFICULT_WINDOWS) != 4 or EXPECTED_DIFFICULT_TARGETS != 1440:
        raise AssertionError("The frozen difficult-window protocol changed")
    build, history_columns, control_columns = read_contract(DATA_DIR)
    train, validation, _ = load_development_data(
        DATA_DIR, history_columns, control_columns
    )
    train_origins = build_common_origins(train)
    validation_origins = build_common_origins(validation)
    validation_target_times = (
        pd.to_datetime(
            validation.iloc[validation_origins]["source_timestamp"], errors="raise"
        )
        + pd.to_timedelta(PREDICT_STEPS * SAMPLE_INTERVAL_SECONDS, unit="s")
    )
    difficult_window_counts = {}
    difficult_union = np.zeros(len(validation_origins), dtype=bool)
    for window_id, start_text, end_text in DIFFICULT_WINDOWS:
        mask = (
            (validation_target_times >= pd.Timestamp(start_text))
            & (validation_target_times <= pd.Timestamp(end_text))
        ).to_numpy()
        count = int(mask.sum())
        if count != EXPECTED_TARGETS_PER_DIFFICULT_WINDOW:
            raise AssertionError(
                f"{window_id} contains {count} targets, expected "
                f"{EXPECTED_TARGETS_PER_DIFFICULT_WINDOW}"
            )
        if np.any(difficult_union & mask):
            raise AssertionError(f"Difficult windows overlap at {window_id}")
        difficult_union |= mask
        difficult_window_counts[window_id] = count
    if int(difficult_union.sum()) != EXPECTED_DIFFICULT_TARGETS:
        raise AssertionError("Difficult-window union count changed")
    label_audits = {
        f"lookback_{lookback}": {
            "training": audit_origins(
                train, train_origins, lookback, PREDICT_STEPS, "training"
            ),
            "validation": audit_origins(
                validation,
                validation_origins,
                lookback,
                PREDICT_STEPS,
                "validation",
            ),
        }
        for lookback in LOOKBACKS
    }
    if LABEL_COLUMN in history_columns or LABEL_COLUMN in control_columns:
        raise AssertionError("The future target entered an input list")
    if TARGET_COLUMN not in history_columns:
        raise AssertionError("Current Thv is missing from causal history")
    required_files = (
        "train_clean.pkl",
        "val_clean.pkl",
        "test_clean.pkl",
        "test_full_clean.pkl",
        "data_build_config.json",
        "feature_catalog.json",
    )
    file_manifest = {}
    for name in required_files:
        path = DATA_DIR / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        file_manifest[name] = {
            "bytes": path.stat().st_size,
            "sha256": (
                "not_read_before_model_selection"
                if name.startswith("test")
                else digest(path)
            ),
        }
    payload = {
        "status": "PASS",
        "expected_runs": len(tasks()),
        "lookbacks": list(LOOKBACKS),
        "hidden_sizes": list(HIDDEN_SIZES),
        "seeds": list(SEEDS),
        "history_raw_channels": len(history_columns),
        "future_control_channels": len(control_columns),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "training_windows_common_origin": len(train_origins),
        "validation_windows_common_origin": len(validation_origins),
        "difficult_window_target_counts": difficult_window_counts,
        "difficult_window_union_targets": int(difficult_union.sum()),
        "window_and_label_audits": label_audits,
        "training_parents": sorted(map(str, train["parent_id"].unique())),
        "validation_parent": str(build.get("validation_parent")),
        "held_out_test_parent": str(build.get("test_parent")),
        "validation_original_only": set(map(str, validation["source_variant"].unique())) == {"Original"},
        "all_model_values_finite": bool(
            np.isfinite(train[list(history_columns)].to_numpy(float)).all()
            and np.isfinite(validation[list(history_columns)].to_numpy(float)).all()
        ),
        "test_pickle_deserialized": False,
        "test_available_to_training": False,
        "data_files": file_manifest,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("PACKAGE_PREFLIGHT_OK=true")


if __name__ == "__main__":
    main()
