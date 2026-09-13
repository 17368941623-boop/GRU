"""Create Raw54 train/validation caches without using the frozen test process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import discover_signal_columns, load_signal_file, parse_file_identity
from protocol import (
    FROZEN_TEST_PARENT,
    FUTURE_CONTROL_COLUMNS,
    TARGET,
    TRAIN_PARENTS,
    VALIDATION_PARENT,
    cache_dir,
    default_source_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=default_source_dir())
    parser.add_argument("--output-dir", type=Path, default=cache_dir())
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (args.output_dir / "train_raw54.pkl", args.output_dir / "validation_raw54.pkl")
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError("Raw54 caches already exist; pass --overwrite to rebuild")

    candidates = sorted(args.source_dir.glob("*.csv"))
    data_files = [
        path for path in candidates
        if path.name.startswith(("Original__", "Aug_01__", "Aug_02__"))
    ]
    if len(data_files) != 27:
        raise ValueError(f"Expected 27 filtered/augmented CSV files, found {len(data_files)}")
    original_reference = next(path for path in data_files if path.name.startswith("Original__"))
    signals, _ = discover_signal_columns(original_reference)
    missing_controls = [name for name in FUTURE_CONTROL_COLUMNS if name not in signals]
    if missing_controls or TARGET not in signals:
        raise ValueError(f"Missing target/controls: target={TARGET in signals}, controls={missing_controls}")

    exclusion_path = args.source_dir / "temperature_flypoint_endpoint_exclusions.csv"
    if not exclusion_path.exists():
        raise FileNotFoundError(f"Missing causal filter endpoint exclusions: {exclusion_path}")
    exclusions = pd.read_csv(exclusion_path)
    required_exclusion_columns = {"file", "unsafe_endpoint_start", "unsafe_endpoint_end"}
    if not required_exclusion_columns.issubset(exclusions.columns):
        raise ValueError("Filter endpoint exclusion manifest has an unexpected schema")
    unsafe_by_file: dict[str, set[int]] = {}
    for row in exclusions.itertuples(index=False):
        file_name = str(row.file)
        start = int(row.unsafe_endpoint_start)
        end = int(row.unsafe_endpoint_end)
        unsafe_by_file.setdefault(file_name, set()).update(range(start, end + 1))

    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    frozen_test_files: list[str] = []
    for path in data_files:
        variant, parent = parse_file_identity(path)
        if parent == FROZEN_TEST_PARENT:
            frozen_test_files.append(path.name)
            manifest.append({
                "file": path.name, "parent_id": parent, "variant": variant,
                "role": "frozen_test_not_read", "rows": "",
            })
            continue
        if parent == VALIDATION_PARENT and variant != "Original":
            manifest.append({
                "file": path.name, "parent_id": parent, "variant": variant,
                "role": "ignored_validation_augmentation_not_read", "rows": "",
            })
            continue
        if parent not in TRAIN_PARENTS and parent != VALIDATION_PARENT:
            raise ValueError(f"Unassigned process {parent}: {path.name}")
        frame = load_signal_file(path, signals, unsafe_by_file.get(path.name, set()))
        role = "validation" if parent == VALIDATION_PARENT else "train"
        (validation_parts if role == "validation" else train_parts).append(frame)
        values = frame[list(signals)].to_numpy(dtype=np.float64)
        manifest.append({
            "file": path.name,
            "parent_id": parent,
            "variant": variant,
            "role": role,
            "rows": len(frame),
            "rows_with_nonfinite_raw54": int((~np.isfinite(values)).any(axis=1).sum()),
            "nonfinite_raw54_cells": int((~np.isfinite(values)).sum()),
            "unsafe_filter_endpoints": int(frame["unsafe_filter_endpoint"].sum()),
            "timestamp_start": str(frame["source_timestamp"].iloc[0]),
            "timestamp_end": str(frame["source_timestamp"].iloc[-1]),
        })

    if len(train_parts) != len(TRAIN_PARENTS) * 3 or len(validation_parts) != 1:
        raise AssertionError(
            f"Unexpected split counts: train={len(train_parts)}, validation={len(validation_parts)}"
        )
    if len(frozen_test_files) != 3:
        raise AssertionError("Expected three frozen-test variants to remain unread")
    train = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(validation_parts, ignore_index=True)
    train.to_pickle(outputs[0])
    validation.to_pickle(outputs[1])

    categories = []
    for index, signal in enumerate(signals):
        if signal.startswith("TE") or signal in {"Thv", "Tef", "Tcd", "DTbr", "A管", "B管", "C管", "D管", "E管", "F管", "冷屏上", "冷屏中", "冷屏下"}:
            category = "temperature"
        elif signal.startswith("PT"):
            category = "pressure"
        elif signal.startswith("FT"):
            category = "flow"
        elif signal.startswith("CV") or signal in {"FC-V1", "EC-V2", "COOLDOWN"}:
            category = "valve_or_control"
        else:
            category = "other"
        categories.append({
            "node_index": index,
            "signal": signal,
            "category": category,
            "is_target": signal == TARGET,
            "is_future_control_in_0912": signal in FUTURE_CONTROL_COLUMNS,
        })
    pd.DataFrame(categories).to_csv(args.output_dir / "raw54_signal_catalog.csv", index=False)
    pd.DataFrame(manifest).to_csv(args.output_dir / "split_manifest.csv", index=False)
    report = {
        "schema_version": 1,
        "protocol": "Raw54-GRU baseline; no 0822 engineered features; no PCMCI/GNN/KAN",
        "source_dir": str(args.source_dir.resolve()),
        "signal_count": len(signals),
        "signals": list(signals),
        "future_controls": list(FUTURE_CONTROL_COLUMNS),
        "target": TARGET,
        "train_parents": list(TRAIN_PARENTS),
        "train_variants": ["Original", "Aug_01", "Aug_02"],
        "train_rows": len(train),
        "validation_parent": VALIDATION_PARENT,
        "validation_variant": "Original",
        "validation_rows": len(validation),
        "filter_endpoint_exclusion_manifest": str(exclusion_path.resolve()),
        "unsafe_filter_endpoints_train": int(train["unsafe_filter_endpoint"].sum()),
        "unsafe_filter_endpoints_validation": int(validation["unsafe_filter_endpoint"].sum()),
        "previously_excluded_260617_now_included": True,
        "frozen_test_parent_not_read": FROZEN_TEST_PARENT,
        "frozen_test_files_not_read": frozen_test_files,
    }
    (args.output_dir / "data_preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
