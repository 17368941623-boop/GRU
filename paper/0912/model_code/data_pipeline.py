"""Raw54-only, leakage-safe data loading and window construction.

This module intentionally does not import or reproduce 0822/data_fil.py.
It reads the already filtered/augmented signal files and uses only the 54
measured signals plus metadata needed to preserve process boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from protocol import (
    ALIASES,
    COMMON_ORIGIN_LOOKBACK,
    FUTURE_CONTROL_COLUMNS,
    LOOKBACK,
    SAMPLE_PERIOD_SECONDS,
    TARGET,
)


PARENT_PATTERN = re.compile(r"-(\d{6}(?:-BACK)?)-ALL\.csv$", re.IGNORECASE)


def canonical_signal(name: str) -> str:
    return ALIASES.get(name, name)


def parse_file_identity(path: Path) -> tuple[str, str]:
    variant, separator, source_name = path.name.partition("__")
    if not separator or variant not in {"Original", "Aug_01", "Aug_02"}:
        raise ValueError(f"Unexpected filtered-data filename: {path.name}")
    match = PARENT_PATTERN.search(source_name)
    if match is None:
        raise ValueError(f"Cannot parse parent process from {path.name}")
    return variant, match.group(1)


def discover_signal_columns(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    columns = tuple(pd.read_csv(path, nrows=0).columns)
    time_columns = tuple(column for column in columns if column.endswith(" Time"))
    value_columns = tuple(column for column in columns if column.endswith(" ValueY"))
    if len(time_columns) != 54 or len(value_columns) != 54:
        raise ValueError(
            f"{path.name}: expected 54 Time/ValueY pairs, found "
            f"{len(time_columns)} Time and {len(value_columns)} ValueY columns"
        )
    signals = tuple(canonical_signal(column[: -len(" ValueY")]) for column in value_columns)
    if len(set(signals)) != 54:
        raise ValueError(f"{path.name}: duplicate canonical signal names")
    return signals, value_columns


def load_signal_file(
    path: Path,
    expected_signals: tuple[str, ...],
    unsafe_endpoint_indices: set[int] | None = None,
) -> pd.DataFrame:
    signals, value_columns = discover_signal_columns(path)
    if signals != expected_signals:
        raise ValueError(f"Signal order differs in {path.name}")
    all_columns = tuple(pd.read_csv(path, nrows=0).columns)
    first_time = next(column for column in all_columns if column.endswith(" Time"))
    usecols = [first_time, *value_columns]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame = raw[list(value_columns)].copy()
    frame.columns = list(signals)
    for column in signals:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    timestamp = pd.to_datetime(
        raw[first_time], format="%Y/%m/%d %H:%M:%S", errors="coerce"
    )
    variant, parent = parse_file_identity(path)
    frame["source_timestamp"] = timestamp
    frame["source_row_index"] = np.arange(len(frame), dtype=np.int64)
    frame["file_id"] = path.stem
    frame["source_group"] = path.name.split("__", 1)[1]
    frame["parent_id"] = parent
    frame["source_variant"] = variant
    unsafe = np.zeros(len(frame), dtype=bool)
    if unsafe_endpoint_indices:
        indices = np.fromiter(sorted(unsafe_endpoint_indices), dtype=np.int64)
        if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= len(frame)):
            raise IndexError(f"Unsafe endpoint index outside {path.name}")
        unsafe[indices] = True
    frame["unsafe_filter_endpoint"] = unsafe
    return frame


def load_cached_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    train_path = data_dir / "train_raw54.pkl"
    validation_path = data_dir / "validation_raw54.pkl"
    catalog_path = data_dir / "raw54_signal_catalog.csv"
    for path in (train_path, validation_path, catalog_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run prepare_raw54_data.py first")
    train = pd.read_pickle(train_path).reset_index(drop=True)
    validation = pd.read_pickle(validation_path).reset_index(drop=True)
    catalog = pd.read_csv(catalog_path)
    signals = tuple(catalog.sort_values("node_index")["signal"].astype(str))
    if len(signals) != 54:
        raise ValueError("Raw54 catalog does not contain exactly 54 signals")
    return train, validation, signals


def assert_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def contiguous_segments(frame: pd.DataFrame, required_columns: tuple[str, ...]) -> list[tuple[int, int]]:
    assert_columns(
        frame,
        required_columns
        + ("file_id", "source_row_index", "source_timestamp", "unsafe_filter_endpoint"),
        "frame",
    )
    numeric = frame[list(required_columns)].to_numpy(dtype=np.float64)
    timestamp = pd.to_datetime(frame["source_timestamp"], errors="coerce")
    valid = (
        np.isfinite(numeric).all(axis=1)
        & timestamp.notna().to_numpy()
        & (~frame["unsafe_filter_endpoint"].astype(bool).to_numpy())
    )
    if len(frame) == 0:
        return []
    file_id = frame["file_id"].astype(str).to_numpy()
    row_id = frame["source_row_index"].to_numpy(dtype=np.int64)
    time_ns = timestamp.astype("int64").to_numpy()
    expected_ns = SAMPLE_PERIOD_SECONDS * 1_000_000_000
    boundary = np.ones(len(frame), dtype=bool)
    boundary[1:] = (
        (file_id[1:] != file_id[:-1])
        | (row_id[1:] != row_id[:-1] + 1)
        | ((time_ns[1:] - time_ns[:-1]) != expected_ns)
        | (~valid[1:])
        | (~valid[:-1])
    )
    starts = np.flatnonzero(boundary)
    stops = np.r_[starts[1:], len(frame)]
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if valid[start] and np.all(valid[start:stop])
    ]


def valid_window_ends(
    frame: pd.DataFrame,
    raw_columns: tuple[str, ...],
    horizon: int,
) -> np.ndarray:
    required = tuple(dict.fromkeys(raw_columns + FUTURE_CONTROL_COLUMNS + (TARGET,)))
    ends: list[np.ndarray] = []
    for start, stop in contiguous_segments(frame, required):
        first_t = start + COMMON_ORIGIN_LOOKBACK - 1
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
    valve_event: np.ndarray


def target_delta_matrix(frame: pd.DataFrame, ends: np.ndarray, horizon: int) -> np.ndarray:
    target = frame[TARGET].to_numpy(dtype=np.float64)
    future = np.stack([target[ends + step] for step in range(1, horizon + 1)], axis=1)
    return future - target[ends, None]


def fit_scalers(
    train_frame: pd.DataFrame,
    train_ends: np.ndarray,
    raw_columns: tuple[str, ...],
    horizon: int,
    rapid_quantile: float,
) -> ScalerBundle:
    history_values = train_frame[list(raw_columns)].to_numpy(dtype=np.float64)
    control_values = train_frame[list(FUTURE_CONTROL_COLUMNS)].to_numpy(dtype=np.float64)
    deltas = target_delta_matrix(train_frame, train_ends, horizon)
    return ScalerBundle(
        history=Standardizer.fit(history_values[np.isfinite(history_values).all(axis=1)]),
        controls=Standardizer.fit(control_values[np.isfinite(control_values).all(axis=1)]),
        target_delta=Standardizer.fit(deltas),
        rapid_threshold=np.quantile(deltas, rapid_quantile, axis=0),
    )


def prepare_split(
    frame: pd.DataFrame,
    raw_columns: tuple[str, ...],
    horizon: int,
    scalers: ScalerBundle,
    rapid_weight: float,
) -> PreparedSplit:
    ends = valid_window_ends(frame, raw_columns, horizon)
    if len(ends) == 0:
        raise ValueError("No valid windows remain")
    history = scalers.history.transform(
        frame[list(raw_columns)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    controls = scalers.controls.transform(
        frame[list(FUTURE_CONTROL_COLUMNS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    target = frame[TARGET].to_numpy(dtype=np.float64)
    delta = target_delta_matrix(frame, ends, horizon)
    delta_scaled = scalers.target_delta.transform(delta).astype(np.float32)
    weights = np.where(
        delta <= scalers.rapid_threshold[None, :], rapid_weight, 1.0
    ).astype(np.float32)
    control_physical = frame[list(FUTURE_CONTROL_COLUMNS)].to_numpy(dtype=np.float64)
    changes = np.zeros(len(frame), dtype=bool)
    changes[1:] = np.any(np.abs(control_physical[1:] - control_physical[:-1]) > 0.05, axis=1)
    same_file = frame["file_id"].astype(str).to_numpy()[1:] == frame["file_id"].astype(str).to_numpy()[:-1]
    changes[1:] &= same_file
    return PreparedSplit(
        frame, history, controls, target, ends, delta, delta_scaled, weights, changes[ends]
    )


class TrajectoryWindowDataset(Dataset):
    """History t-59..t and planned controls t..t+H-1 to Thv t+1..t+H."""

    def __init__(self, split: PreparedSplit) -> None:
        self.split = split

    def __len__(self) -> int:
        return len(self.split.ends)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        t = int(self.split.ends[item])
        horizon = self.split.delta_scaled.shape[1]
        return (
            torch.from_numpy(self.split.history[t - LOOKBACK + 1 : t + 1]),
            torch.from_numpy(self.split.controls[t : t + horizon]),
            torch.from_numpy(self.split.delta_scaled[item]),
            torch.from_numpy(self.split.weights[item]),
            torch.tensor(item, dtype=torch.long),
        )


def future_absolute(split: PreparedSplit) -> np.ndarray:
    return split.target[split.ends, None] + split.delta


def split_metadata(split: PreparedSplit) -> pd.DataFrame:
    selected = split.frame.iloc[split.ends]
    return selected[
        ["file_id", "source_group", "parent_id", "source_variant", "source_row_index", "source_timestamp"]
    ].reset_index(drop=True)
