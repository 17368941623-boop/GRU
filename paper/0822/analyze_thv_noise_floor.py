#!/usr/bin/env python3
"""Estimate the high-frequency Thv measurement-noise scale without smoothing data."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "processed_data"
OUTPUT = ROOT / "thv_noise_floor_analysis.json"
MAD_NORMAL = 0.6744897501960817


def robust_sigma(values: np.ndarray, variance_factor: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values)
    mad = np.median(np.abs(values - center))
    return float(mad / MAD_NORMAL / math.sqrt(variance_factor))


def collect_differences(frame: pd.DataFrame, event_free: bool) -> tuple[np.ndarray, np.ndarray]:
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for _, part in frame.groupby("file_id", sort=False):
        y = part["Thv"].to_numpy(dtype=np.float64)
        rows = part["source_row_index"].to_numpy(dtype=np.int64)
        d1 = np.diff(y)
        valid1 = np.diff(rows) == 1
        d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
        valid2 = (rows[1:-1] - rows[:-2] == 1) & (rows[2:] - rows[1:-1] == 1)
        if event_free and "valve_event_mask" in part:
            events = part["valve_event_mask"].to_numpy(dtype=np.float64) != 0
            valid1 &= ~(events[:-1] | events[1:])
            valid2 &= ~(events[:-2] | events[1:-1] | events[2:])
        first.append(d1[valid1])
        second.append(d2[valid2])
    return np.concatenate(first), np.concatenate(second)


def summarize(path: Path) -> dict[str, object]:
    frame = joblib.load(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{path} does not contain a DataFrame")
    all_d1, all_d2 = collect_differences(frame, event_free=False)
    quiet_d1, quiet_d2 = collect_differences(frame, event_free=True)
    y = frame["Thv"].to_numpy(dtype=np.float64)
    unique = np.unique(y)
    positive_steps = np.diff(unique)
    positive_steps = positive_steps[positive_steps > 0]
    label = frame["Future_Thv_15step"].to_numpy(dtype=np.float64)
    finite = np.isfinite(label) & np.isfinite(y)
    delta15 = label[finite] - y[finite]
    return {
        "file": path.name,
        "rows": int(len(frame)),
        "segments": int(frame["file_id"].nunique()),
        "thv_range_k": [float(np.min(y)), float(np.max(y))],
        "smallest_positive_recorded_increment_k": float(np.min(positive_steps)),
        "all_contiguous_pairs": int(len(all_d1)),
        "event_free_contiguous_pairs": int(len(quiet_d1)),
        "robust_sigma_from_first_difference_k": robust_sigma(all_d1, 2.0),
        "robust_sigma_from_second_difference_k": robust_sigma(all_d2, 6.0),
        "event_free_sigma_from_first_difference_k": robust_sigma(quiet_d1, 2.0),
        "event_free_sigma_from_second_difference_k": robust_sigma(quiet_d2, 6.0),
        "absolute_second_difference_percentiles_k": {
            str(q): float(np.percentile(np.abs(all_d2), q)) for q in (50, 75, 90, 95, 99)
        },
        "delta15_rmse_persistence_k": float(np.sqrt(np.mean(delta15**2))),
        "delta15_absolute_percentiles_k": {
            str(q): float(np.percentile(np.abs(delta15), q)) for q in (50, 75, 90, 95, 99)
        },
    }


def main() -> None:
    report = {
        "method": (
            "Robust local-linear residual estimate: MAD of the second difference divided "
            "by 0.67449*sqrt(6). This preserves the source data and estimates only the "
            "high-frequency component; slow calibration drift is outside this estimate."
        ),
        "interpretation": (
            "For a perfect latent-state forecast, observed-target RMSE cannot normally be "
            "much below roughly one sensor-noise sigma. A residual forecast that cannot "
            "separate current measurement noise can have a floor approaching sqrt(2)*sigma."
        ),
        "splits": [
            summarize(DATA_DIR / "val_clean.pkl"),
            summarize(DATA_DIR / "test_full_clean.pkl"),
        ],
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
