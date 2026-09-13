#!/usr/bin/env python3
"""Leakage-safe data preparation for direct multi-horizon Thv prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from protocol import (
    COMMON_ORIGIN_LOOKBACK,
    FUTURE_CONTROL_COLUMNS,
    HISTORY_COLUMNS,
    LOOKBACK,
    RAW_COLUMNS,
    TARGET,
)


def load_frame(path: Path) -> pd.DataFrame:
    """Load the compressed joblib DataFrame used by the existing project."""
    frame = joblib.load(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame in {path}, received {type(frame)!r}")
    return frame.reset_index(drop=True)


def exclude_0617(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude the run without reliable module-valve measurements."""
    source = frame["source_group"].astype(str)
    return frame.loc[~source.str.contains("0617", na=False)].reset_index(drop=True)


def assert_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def _segment_bounds(frame: pd.DataFrame, valid_row: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive-exclusive contiguous segments without crossing source gaps."""
    if len(frame) == 0:
        return []
    file_id = frame["file_id"].astype(str).to_numpy()
    row_id = frame["source_row_index"].to_numpy(dtype=np.int64)
    boundaries = np.ones(len(frame), dtype=bool)
    boundaries[1:] = (file_id[1:] != file_id[:-1]) | (row_id[1:] != row_id[:-1] + 1)
    boundaries |= ~valid_row
    boundaries[1:] |= ~valid_row[:-1]
    starts = np.flatnonzero(boundaries)
    ends = np.r_[starts[1:], len(frame)]
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if valid_row[start] and np.all(valid_row[start:end])
    ]


def valid_window_ends(
    frame: pd.DataFrame,
    horizon: int,
    history_columns: tuple[str, ...] = HISTORY_COLUMNS,
    common_origin_lookback: int = COMMON_ORIGIN_LOOKBACK,
) -> np.ndarray:
    """Find current-time indices t for [t-L+1:t] -> [t+1:t+H]."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    required = tuple(dict.fromkeys(history_columns + FUTURE_CONTROL_COLUMNS + (TARGET,)))
    assert_columns(frame, required + ("file_id", "source_row_index"), "frame")
    numeric = frame[list(required)].to_numpy(dtype=np.float64)
    valid_row = np.isfinite(numeric).all(axis=1)
    ends: list[np.ndarray] = []
    for start, stop in _segment_bounds(frame, valid_row):
        first_t = start + common_origin_lookback - 1
        last_t = stop - horizon - 1
        if first_t <= last_t:
            ends.append(np.arange(first_t, last_t + 1, dtype=np.int64))
    if not ends:
        return np.empty(0, dtype=np.int64)
    result = np.concatenate(ends)
    if np.any(result - LOOKBACK + 1 < 0) or np.any(result + horizon >= len(frame)):
        raise AssertionError("Window index escaped frame bounds")
    return result


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, axis: int = 0) -> "Standardizer":
        mean = np.nanmean(values, axis=axis)
        scale = np.nanstd(values, axis=axis)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        return cls(np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean

    def payload(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_payload(cls, payload: dict[str, list[float]]) -> "Standardizer":
        return cls(np.asarray(payload["mean"], dtype=np.float64), np.asarray(payload["scale"], dtype=np.float64))


def target_delta_matrix(frame: pd.DataFrame, ends: np.ndarray, horizon: int) -> np.ndarray:
    target = frame[TARGET].to_numpy(dtype=np.float64)
    future = np.stack([target[ends + step] for step in range(1, horizon + 1)], axis=1)
    return future - target[ends, None]


@dataclass
class PreparedSplit:
    frame: pd.DataFrame
    history: np.ndarray
    controls: np.ndarray
    target: np.ndarray
    ends: np.ndarray
    delta: np.ndarray
    delta_scaled: np.ndarray
    weights: np.ndarray


@dataclass
class ScalerBundle:
    history: Standardizer
    controls: Standardizer
    target_delta: Standardizer
    rapid_threshold: np.ndarray

    def payload(self) -> dict[str, object]:
        return {
            "history": self.history.payload(),
            "controls": self.controls.payload(),
            "target_delta": self.target_delta.payload(),
            "rapid_threshold": self.rapid_threshold.tolist(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "ScalerBundle":
        return cls(
            Standardizer.from_payload(payload["history"]),  # type: ignore[arg-type]
            Standardizer.from_payload(payload["controls"]),  # type: ignore[arg-type]
            Standardizer.from_payload(payload["target_delta"]),  # type: ignore[arg-type]
            np.asarray(payload["rapid_threshold"], dtype=np.float64),
        )


def fit_scalers(
    train_frame: pd.DataFrame,
    train_ends: np.ndarray,
    horizon: int,
    rapid_quantile: float,
) -> ScalerBundle:
    if not 0.0 < rapid_quantile < 1.0:
        raise ValueError("rapid_quantile must be in (0, 1)")
    history_values = train_frame[list(HISTORY_COLUMNS)].to_numpy(dtype=np.float64)
    control_values = train_frame[list(FUTURE_CONTROL_COLUMNS)].to_numpy(dtype=np.float64)
    history_valid = np.isfinite(history_values).all(axis=1)
    control_valid = np.isfinite(control_values).all(axis=1)
    deltas = target_delta_matrix(train_frame, train_ends, horizon)
    return ScalerBundle(
        history=Standardizer.fit(history_values[history_valid]),
        controls=Standardizer.fit(control_values[control_valid]),
        target_delta=Standardizer.fit(deltas),
        rapid_threshold=np.quantile(deltas, rapid_quantile, axis=0),
    )


def prepare_split(
    frame: pd.DataFrame,
    horizon: int,
    scalers: ScalerBundle,
    rapid_weight: float = 3.0,
) -> PreparedSplit:
    ends = valid_window_ends(frame, horizon)
    if len(ends) == 0:
        raise ValueError("No valid windows remain after boundary and finiteness checks")
    history = scalers.history.transform(
        frame[list(HISTORY_COLUMNS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    controls = scalers.controls.transform(
        frame[list(FUTURE_CONTROL_COLUMNS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    target = frame[TARGET].to_numpy(dtype=np.float64)
    delta = target_delta_matrix(frame, ends, horizon)
    delta_scaled = scalers.target_delta.transform(delta).astype(np.float32)
    weights = np.where(delta <= scalers.rapid_threshold[None, :], rapid_weight, 1.0).astype(np.float32)
    return PreparedSplit(frame, history, controls, target, ends, delta, delta_scaled, weights)


class TrajectoryWindowDataset(Dataset):
    """History through t, planned controls t..t+H-1, targets t+1..t+H."""

    def __init__(self, split: PreparedSplit, lookback: int = LOOKBACK) -> None:
        self.split = split
        self.lookback = int(lookback)

    def __len__(self) -> int:
        return len(self.split.ends)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        t = int(self.split.ends[item])
        horizon = self.split.delta_scaled.shape[1]
        history = self.split.history[t - self.lookback + 1 : t + 1]
        controls = self.split.controls[t : t + horizon]
        return (
            torch.from_numpy(history),
            torch.from_numpy(controls),
            torch.from_numpy(self.split.delta_scaled[item]),
            torch.from_numpy(self.split.weights[item]),
            torch.tensor(item, dtype=torch.long),
        )


def future_absolute(split: PreparedSplit) -> np.ndarray:
    return split.target[split.ends, None] + split.delta


def split_metadata(split: PreparedSplit) -> pd.DataFrame:
    selected = split.frame.iloc[split.ends]
    return selected[["file_id", "source_group", "source_row_index", "source_timestamp"]].reset_index(drop=True)


def original_training_segments(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = RAW_COLUMNS,
    min_length: int = 64,
) -> dict[int, np.ndarray]:
    """Original, non-0617, contiguous segments for causal discovery only."""
    records = original_training_segment_records(frame, columns, min_length)
    return {index: record["values"] for index, record in enumerate(records)}


def original_training_segment_records(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = RAW_COLUMNS,
    min_length: int = 64,
) -> list[dict[str, object]]:
    """Return causal-discovery segments together with their parent-run identity."""
    frame = exclude_0617(frame)
    if "source_variant" in frame.columns:
        original = frame["source_variant"].astype(str).str.lower().eq("original")
    else:
        original = frame["file_id"].astype(str).str.startswith("Original__")
    frame = frame.loc[original].reset_index(drop=True)
    assert_columns(frame, columns + ("file_id", "source_row_index"), "causal-discovery frame")
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    segments: list[dict[str, object]] = []
    for start, stop in _segment_bounds(frame, valid):
        if stop - start >= min_length:
            part = frame.iloc[start:stop]
            segments.append(
                {
                    "source_group": str(part["source_group"].iloc[0]),
                    "file_id": str(part["file_id"].iloc[0]),
                    "values": values[start:stop],
                }
            )
    if not segments:
        raise ValueError("No sufficiently long original training segments for causal discovery")
    return segments
