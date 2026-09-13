#!/usr/bin/env python3
"""Leakage-safe data loading and segment-preserving transformations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from protocol import ACTIVE_CONTROLS


def load_original_training_records(
    train_file: Path,
    raw_columns: tuple[str, ...],
    min_length: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load only Original rows from train_clean.pkl; never opens val/test files."""
    frame = joblib.load(train_file)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame in {train_file}, got {type(frame)!r}")
    required = set(raw_columns) | {
        "file_id", "source_variant", "parent_id", "source_row_index", "source_timestamp"
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"train_clean.pkl is missing required columns: {missing}")

    original_mask = frame["source_variant"].astype(str).str.casefold().eq("original")
    original = frame.loc[original_mask].copy()
    augmented_rows_ignored = int((~original_mask).sum())
    if original.empty:
        raise ValueError("No Original rows found in train_clean.pkl")

    records: list[dict[str, Any]] = []
    dropped_short = 0
    for file_id, part in original.groupby("file_id", sort=False):
        row_number = pd.to_numeric(part["source_row_index"], errors="coerce")
        if row_number.notna().all():
            part = part.assign(_row_number=row_number).sort_values("_row_number", kind="stable")
        values = part.loc[:, list(raw_columns)].to_numpy(dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        if not finite.all():
            # Split again rather than allowing a lag to jump across a missing block.
            changes = np.flatnonzero(np.diff(np.r_[False, finite, False]))
            spans = list(zip(changes[::2], changes[1::2]))
        else:
            spans = [(0, len(part))]
        for span_id, (start, stop) in enumerate(spans):
            if stop - start < min_length:
                dropped_short += int(stop - start)
                continue
            records.append(
                {
                    "parent_id": str(part["parent_id"].iloc[0]),
                    "file_id": f"{file_id}__finite_{span_id:03d}",
                    "values": values[start:stop],
                }
            )
    if not records:
        raise ValueError("No sufficiently long Original segments remain")
    audit = {
        "train_rows_total": int(len(frame)),
        "original_rows_total": int(len(original)),
        "augmented_rows_ignored": augmented_rows_ignored,
        "finite_segment_rows_used": int(sum(len(record["values"]) for record in records)),
        "short_or_invalid_rows_ignored": dropped_short,
        "original_parent_runs": len({record["parent_id"] for record in records}),
        "original_segments": len(records),
    }
    return records, audit


def split_parent_runs(
    records: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    parents = sorted({str(record["parent_id"]) for record in records})
    if len(parents) < 4:
        raise ValueError("At least four independent Original parent runs are required")
    rng = np.random.default_rng(seed)
    shuffled = [parents[index] for index in rng.permutation(len(parents))]
    screen_parent_set = set(shuffled[::2])
    inference_parent_set = set(parents) - screen_parent_set
    screen = [record for record in records if record["parent_id"] in screen_parent_set]
    inference = [record for record in records if record["parent_id"] in inference_parent_set]
    return screen, inference, sorted(screen_parent_set), sorted(inference_parent_set)


def transform_records(
    records: list[dict[str, Any]],
    raw_columns: tuple[str, ...],
    mode: str,
    max_samples_per_segment: int,
) -> tuple[dict[int, np.ndarray], dict[int, str], dict[str, list[float]]]:
    if mode not in {"difference", "level"}:
        raise ValueError(f"Unsupported transform: {mode}")
    # Targeted diagnostic runs may intentionally use only a Raw54 subset.
    control_indices = [
        raw_columns.index(name) for name in ACTIVE_CONTROLS if name in raw_columns
    ]
    arrays: list[np.ndarray] = []
    parent_by_segment: dict[int, str] = {}
    for record in records:
        raw = np.asarray(record["values"], dtype=np.float64)
        transformed = np.diff(raw, axis=0) if mode == "difference" else raw.copy()
        if max_samples_per_segment > 0 and len(transformed) > max_samples_per_segment:
            if mode == "difference" and control_indices:
                activity = np.abs(transformed[:, control_indices]).sum(axis=1)
            elif control_indices:
                activity = np.abs(
                    np.diff(
                        transformed[:, control_indices], axis=0,
                        prepend=transformed[:1, control_indices],
                    )
                ).sum(axis=1)
            else:
                activity = np.ones(len(transformed), dtype=np.float64)
            cumulative = np.r_[0.0, np.cumsum(activity)]
            scores = cumulative[max_samples_per_segment:] - cumulative[:-max_samples_per_segment]
            start = int(np.argmax(scores))
            transformed = transformed[start : start + max_samples_per_segment]
        if len(transformed) == 0:
            continue
        segment_id = len(arrays)
        arrays.append(transformed)
        parent_by_segment[segment_id] = str(record["parent_id"])

    stacked = np.concatenate(arrays, axis=0)
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    segments = {index: (array - mean) / scale for index, array in enumerate(arrays)}
    scaler = {"mean": mean.tolist(), "scale": scale.tolist()}
    return segments, parent_by_segment, scaler


def bootstrap_parent_runs(
    segments: dict[int, np.ndarray],
    parent_by_segment: dict[int, str],
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """Resample whole experimental parent runs, preserving their child segments."""
    parent_to_segments: dict[str, list[int]] = {}
    for segment_id, parent_id in parent_by_segment.items():
        parent_to_segments.setdefault(parent_id, []).append(segment_id)
    parents = sorted(parent_to_segments)
    sampled = rng.choice(parents, size=len(parents), replace=True)
    boot: dict[int, np.ndarray] = {}
    for parent_id in sampled:
        for segment_id in parent_to_segments[str(parent_id)]:
            boot[len(boot)] = segments[segment_id]
    return boot
