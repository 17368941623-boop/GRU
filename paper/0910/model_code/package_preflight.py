#!/usr/bin/env python3
"""Fail-fast audit for the portable 0910 heatmap training package."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SHARED_DIR = PROJECT_DIR / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from train_thv_delta_lstm_lookback import (  # noqa: E402
    EXCLUDED_PARENT,
    EXPECTED_TRAIN_PARENTS,
    EXPECTED_VALIDATION_PARENT,
    EXPECTED_TEST_PARENT_IN_BUILD_CONFIG,
    FUTURE_CONTROL_COLS,
)
from heatmap_protocol import (  # noqa: E402
    FEATURE_SPECS,
    MODEL_HISTORY_COLUMNS,
    MODEL_SPECS,
    PREDICT_STEPS,
    SEEDS,
    all_tasks,
    history_feature_mask,
    protocol_payload,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    data_dir = PROJECT_DIR / "processed_data"
    required = (
        "train_clean.pkl",
        "val_clean.pkl",
        "test_full_clean.pkl",
        "data_build_config.json",
        "feature_catalog.json",
    )
    for name in required:
        path = data_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    build = json.loads((data_dir / "data_build_config.json").read_text(encoding="utf-8"))
    if build.get("validation_parent") != EXPECTED_VALIDATION_PARENT:
        raise ValueError("Validation split changed")
    if build.get("test_parent") != EXPECTED_TEST_PARENT_IN_BUILD_CONFIG:
        raise ValueError("Test split changed")
    if build.get("future_features_used") is not False:
        raise ValueError("Causal-input contract missing")
    if PREDICT_STEPS not in set(map(int, build.get("horizons", []))):
        raise ValueError("Missing h=15 label")
    if len(MODEL_SPECS) != 3 or len(FEATURE_SPECS) != 8 or len(SEEDS) != 5:
        raise AssertionError("Frozen model, feature or seed count changed")
    if len(all_tasks()) != 120:
        raise AssertionError("Expected exactly 120 paired training runs")
    if any(name.startswith("Future_") for name in MODEL_HISTORY_COLUMNS):
        raise AssertionError("Future label in history")
    if any(name.startswith("Future_") for name in FUTURE_CONTROL_COLS):
        raise AssertionError("Future label in controls")
    if len(MODEL_HISTORY_COLUMNS) != 42 or len(set(MODEL_HISTORY_COLUMNS)) != 42:
        raise AssertionError("Expected 42 unique fixed history channels")
    if {len(history_feature_mask(spec)) for spec in FEATURE_SPECS} != {42}:
        raise AssertionError("Input dimensions differ across feature stages")

    label = f"Future_Thv_{PREDICT_STEPS}step"
    # Deliberately do not deserialize either test pickle before model selection.
    train = joblib.load(data_dir / "train_clean.pkl")
    train = train.loc[train.parent_id.astype(str) != EXCLUDED_PARENT]
    validation = joblib.load(data_dir / "val_clean.pkl")
    if set(map(str, train.parent_id.unique())) != EXPECTED_TRAIN_PARENTS:
        raise ValueError("Training parents changed")
    if set(map(str, validation.parent_id.unique())) != {EXPECTED_VALIDATION_PARENT}:
        raise ValueError("Validation parent changed")
    if set(map(str, validation.source_variant.unique())) != {"Original"}:
        raise ValueError("Validation contains augmentation")
    if set(map(str, train.source_group.unique())) & set(map(str, validation.source_group.unique())):
        raise ValueError("Train/validation source overlap")

    selected = list(dict.fromkeys((*MODEL_HISTORY_COLUMNS, *FUTURE_CONTROL_COLS, label)))
    for split_name, frame in (
        ("training after 0617 exclusion", train),
        ("validation", validation),
    ):
        missing = set(selected) - set(frame.columns)
        if missing:
            raise KeyError(f"{split_name} missing {sorted(missing)}")
        if not np.isfinite(frame[selected].to_numpy(float)).all():
            raise ValueError(f"{split_name} has NaN/inf")

    manifest: dict[str, dict[str, object]] = {}
    for name in required:
        path = data_dir / name
        item: dict[str, object] = {"bytes": path.stat().st_size}
        item["sha256"] = (
            "not_read_before_validation_freeze"
            if name.startswith("test")
            else digest(path)
        )
        manifest[name] = item
    payload = {
        "status": "PASS",
        "expected_runs": len(all_tasks()),
        "models": len(MODEL_SPECS),
        "feature_stages": len(FEATURE_SPECS),
        "seeds_per_cell": len(SEEDS),
        "model_history_channels": len(MODEL_HISTORY_COLUMNS),
        "future_control_channels": len(FUTURE_CONTROL_COLS),
        "data_files": manifest,
        "test_pickle_deserialized": False,
        "test_available_to_training": False,
        "protocol": protocol_payload(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("PACKAGE_PREFLIGHT_OK=true")


if __name__ == "__main__":
    main()
