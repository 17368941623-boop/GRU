#!/usr/bin/env python3
"""Audit the stage-1 package and development split without opening test data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from protocol import HIDDEN_SIZES, LOOKBACKS, SEEDS, tasks
from train_global_cell import (
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
    build, history_columns, control_columns = read_contract(DATA_DIR)
    train, validation, _ = load_development_data(
        DATA_DIR, history_columns, control_columns
    )
    train_origins = build_common_origins(train)
    validation_origins = build_common_origins(validation)
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
