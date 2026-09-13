#!/usr/bin/env python3
"""Descriptive lead/lag and event checks for the physical H_mix inputs.

This is deliberately separate from PCMCI: it helps distinguish a primary
response peak from a residual conditional association at the search boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_io import load_original_training_records, split_parent_runs
from protocol import SAMPLE_PERIOD_SECONDS, load_raw_columns


SOURCES = ("A管", "EC-V2", "COOLDOWN", "FC-V1")
VALVES = ("EC-V2", "COOLDOWN", "FC-V1")
TARGET = "Thv"


def lag_correlation(
    arrays: list[np.ndarray], source: int, target: int, lag: int
) -> tuple[float, int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for values in arrays:
        delta = np.diff(values, axis=0)
        if lag > 0 and len(delta) > lag:
            xs.append(delta[:-lag, source])
            ys.append(delta[lag:, target])
        elif lag < 0 and len(delta) > -lag:
            amount = -lag
            xs.append(delta[amount:, source])
            ys.append(delta[:-amount, target])
    if not xs:
        return np.nan, 0
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, len(x)
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def event_onsets(
    values: np.ndarray, source: int, threshold: float, quiet_steps: int
) -> list[tuple[int, str]]:
    change = np.diff(values[:, source])
    raw = np.flatnonzero(np.abs(change) >= threshold) + 1
    events: list[tuple[int, str]] = []
    last_raw = -100_000
    for index in raw:
        if int(index) - last_raw > quiet_steps:
            direction = "opening" if change[int(index) - 1] > 0 else "closing"
            events.append((int(index), direction))
        last_raw = int(index)
    return events


def matrix_frame(matrix: np.ndarray, names: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=names, columns=names)


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    experiment_dir = code_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=experiment_dir.parent / "processed_data")
    parser.add_argument("--output-dir", type=Path, default=experiment_dir / "output" / "hmix_diagnostics")
    parser.add_argument("--max-lag", type=int, default=60)
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--event-quiet-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.max_lag < 1 or args.event_threshold <= 0 or args.event_quiet_steps < 0:
        raise ValueError("Invalid diagnostic lag/event settings")

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_columns = load_raw_columns(data_dir)
    records, data_audit = load_original_training_records(
        data_dir / "train_clean.pkl", raw_columns, min_length=args.max_lag + 40
    )
    _, inference_records, _, inference_parents = split_parent_runs(records, args.seed)
    selected_names = SOURCES + (TARGET,)
    selected_indices = [raw_columns.index(name) for name in selected_names]

    def selected_arrays(items: list[dict[str, Any]]) -> list[np.ndarray]:
        return [np.asarray(item["values"])[:, selected_indices] for item in items]

    groups: dict[str, list[dict[str, Any]]] = {
        "all_parent_runs": records,
        "pcmci_inference_parent_runs": inference_records,
    }
    for parent_id in sorted({str(record["parent_id"]) for record in records}):
        groups[f"parent_{parent_id}"] = [
            record for record in records if str(record["parent_id"]) == parent_id
        ]

    correlation_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "interpretation": (
            "Positive lag means source change precedes Thv change. A peak at the upper "
            "boundary is a diagnostic warning, not proof of a physical delay beyond it."
        ),
        "sample_period_seconds": SAMPLE_PERIOD_SECONDS,
        "max_lag_steps": args.max_lag,
        "max_lag_seconds": args.max_lag * SAMPLE_PERIOD_SECONDS,
        "event_threshold_percentage_points": args.event_threshold,
        "event_quiet_steps": args.event_quiet_steps,
        "inference_parent_runs": inference_parents,
        "data_audit": data_audit,
        "groups": {},
    }
    for group_name, group_records in groups.items():
        arrays = selected_arrays(group_records)
        summary["groups"][group_name] = {}
        for source_index, source in enumerate(SOURCES):
            source_rows = []
            for lag in list(range(-args.max_lag, 0)) + list(range(1, args.max_lag + 1)):
                correlation, pairs = lag_correlation(arrays, source_index, 4, lag)
                row = {
                    "group": group_name,
                    "source": source,
                    "destination": TARGET,
                    "lag_steps": lag,
                    "lag_seconds": lag * SAMPLE_PERIOD_SECONDS,
                    "direction": "source_leads" if lag > 0 else "Thv_leads",
                    "correlation_delta": correlation,
                    "sample_pairs": pairs,
                }
                correlation_rows.append(row)
                source_rows.append(row)
            positive = [row for row in source_rows if row["lag_steps"] > 0 and np.isfinite(row["correlation_delta"])]
            negative = [row for row in source_rows if row["lag_steps"] < 0 and np.isfinite(row["correlation_delta"])]
            positive_peak = max(positive, key=lambda row: abs(row["correlation_delta"]))
            negative_peak = max(negative, key=lambda row: abs(row["correlation_delta"]))
            summary["groups"][group_name][source] = {
                "source_leads_peak_steps": int(positive_peak["lag_steps"]),
                "source_leads_peak_seconds": int(positive_peak["lag_seconds"]),
                "source_leads_peak_correlation": float(positive_peak["correlation_delta"]),
                "Thv_leads_peak_steps": int(negative_peak["lag_steps"]),
                "Thv_leads_peak_seconds": int(negative_peak["lag_seconds"]),
                "Thv_leads_peak_correlation": float(negative_peak["correlation_delta"]),
                "upper_boundary_warning": bool(
                    positive_peak["lag_steps"] >= args.max_lag - 2
                ),
            }
    pd.DataFrame(correlation_rows).to_csv(
        output_dir / "hmix_delta_lead_lag_correlations.csv", index=False
    )

    # Event responses are descriptive because valve changes are not randomized
    # and the three valves can move together under the operating schedule.
    event_rows: list[dict[str, Any]] = []
    all_arrays = selected_arrays(records)
    parent_ids = [str(record["parent_id"]) for record in records]
    offsets = np.arange(-args.max_lag, args.max_lag + 1)
    for source_index, source in enumerate(SOURCES[1:], start=1):
        curves: list[tuple[str, str, np.ndarray]] = []
        for parent_id, values in zip(parent_ids, all_arrays):
            for event_index, direction in event_onsets(
                values, source_index, args.event_threshold, args.event_quiet_steps
            ):
                if event_index + offsets[0] < 0 or event_index + offsets[-1] >= len(values):
                    continue
                curve = values[event_index + offsets, 4] - values[event_index, 4]
                curves.append((parent_id, direction, curve))
        for group_name, parent_filter in (
            [("all_parent_runs", None)]
            + [(f"parent_{parent}", parent) for parent in sorted(set(parent_ids))]
        ):
            for direction in ("opening", "closing"):
                selected = [
                    curve for parent, current_direction, curve in curves
                    if current_direction == direction
                    and (parent_filter is None or parent == parent_filter)
                ]
                if not selected:
                    continue
                matrix = np.asarray(selected)
                for column, offset in enumerate(offsets):
                    event_rows.append(
                        {
                            "group": group_name,
                            "source": source,
                            "direction": direction,
                            "events": len(matrix),
                            "offset_steps": int(offset),
                            "offset_seconds": int(offset * SAMPLE_PERIOD_SECONDS),
                            "mean_Thv_change": float(matrix[:, column].mean()),
                            "median_Thv_change": float(np.median(matrix[:, column])),
                            "std_Thv_change": float(matrix[:, column].std(ddof=1)) if len(matrix) > 1 else np.nan,
                        }
                    )
    pd.DataFrame(event_rows).to_csv(output_dir / "hmix_valve_event_response.csv", index=False)

    level_blocks = [values[:, 1:4] for values in all_arrays]
    difference_blocks = [np.diff(values[:, 1:4], axis=0) for values in all_arrays]
    matrix_frame(np.corrcoef(np.concatenate(level_blocks).T), VALVES).to_csv(
        output_dir / "hmix_valve_level_correlations.csv"
    )
    matrix_frame(np.corrcoef(np.concatenate(difference_blocks).T), VALVES).to_csv(
        output_dir / "hmix_valve_change_correlations.csv"
    )
    summary["warning"] = (
        "Do not convert a boundary MCI lag into a transport delay without agreement from "
        "the pooled lead/lag peak, event response, and parent-run stability."
    )
    (output_dir / "hmix_diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

