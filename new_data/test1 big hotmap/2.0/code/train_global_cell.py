#!/usr/bin/env python3
"""Train one leakage-safe cell of the global GRU lookback/hidden grid.

Only train_clean.pkl and val_clean.pkl are opened.  Test pickles are neither
loaded nor imported by this module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from protocol import (
    COMMON_ORIGIN_LOOKBACK,
    HIDDEN_SIZES,
    LABEL_COLUMN,
    LOOKBACKS,
    PREDICT_STEPS,
    SELECTION_METRIC,
    TARGET_COLUMN,
    run_directory,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def discover_default_data_dir() -> Path:
    """Support both the local archive layout and the flat server bundle."""
    candidates = (
        PROJECT_DIR.parent / "processed_data",
        PROJECT_DIR.parent.parent / "processed_data",
    )
    for candidate in candidates:
        if (candidate / "data_build_config.json").is_file():
            return candidate
    return candidates[0]


DEFAULT_DATA_DIR = discover_default_data_dir()
DEFAULT_RESULTS_DIR = PROJECT_DIR / "output"


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if not np.isfinite(array).all():
            raise ValueError("Scaler fitting data contain NaN or infinity")
        mean = array.mean(axis=0)
        scale = array.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(mean.astype(np.float64), scale.astype(np.float64))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        one_dimensional = array.ndim == 1
        if one_dimensional:
            array = array.reshape(-1, 1)
        result = (array - self.mean) / self.scale
        return result.reshape(-1) if one_dimensional else result

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        one_dimensional = array.ndim == 1
        if one_dimensional:
            array = array.reshape(-1, 1)
        result = array * self.scale + self.mean
        return result.reshape(-1) if one_dimensional else result


class WindowDataset(Dataset):
    def __init__(
        self,
        history: np.ndarray,
        controls: np.ndarray,
        target_delta_scaled: np.ndarray,
        target_delta_k: np.ndarray,
        current_temperature_k: np.ndarray,
        weights: np.ndarray,
        ends: np.ndarray,
        lookback: int,
        predict_steps: int,
    ) -> None:
        self.history = history
        self.controls = controls
        self.target_delta_scaled = target_delta_scaled
        self.target_delta_k = target_delta_k
        self.current_temperature_k = current_temperature_k
        self.weights = weights
        self.ends = ends
        self.lookback = lookback
        self.predict_steps = predict_steps

    def __len__(self) -> int:
        return int(len(self.ends))

    def __getitem__(self, index: int):
        end = int(self.ends[index])
        start = end - self.lookback + 1
        return (
            torch.from_numpy(self.history[start : end + 1]),
            torch.from_numpy(self.controls[end : end + self.predict_steps]),
            torch.tensor(self.target_delta_scaled[end], dtype=torch.float32),
            torch.tensor(self.target_delta_k[end], dtype=torch.float32),
            torch.tensor(self.current_temperature_k[end], dtype=torch.float32),
            torch.tensor(self.weights[end], dtype=torch.float32),
            torch.tensor(end, dtype=torch.int64),
        )


class ControlledGRU(nn.Module):
    def __init__(
        self,
        history_features: int,
        control_features: int,
        hidden_size: int,
        control_hidden_size: int = 32,
        fusion_hidden_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history_gru = nn.GRU(
            input_size=history_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.control_gru = nn.GRU(
            input_size=control_features,
            hidden_size=control_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + control_hidden_size, fusion_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, 1),
        )

    def forward(self, history: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        _, history_state = self.history_gru(history)
        _, control_state = self.control_gru(controls)
        fused = torch.cat((history_state[-1], control_state[-1]), dim=-1)
        return self.head(fused).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback", type=int, required=True, choices=LOOKBACKS)
    parser.add_argument("--hidden-size", type=int, required=True, choices=HIDDEN_SIZES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--control-hidden-size", type=int, default=32)
    parser.add_argument("--fusion-hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rapid-quantile", type=float, default=0.10)
    parser.add_argument("--rapid-weight", type=float, default=3.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size and patience must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if args.control_hidden_size < 1 or args.fusion_hidden_size < 1:
        raise ValueError("hidden dimensions must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not 0 < args.rapid_quantile < 0.5 or args.rapid_weight <= 1:
        raise ValueError("rapid weighting arguments are invalid")
    if args.huber_delta <= 0 or args.num_workers < 0:
        raise ValueError("huber-delta must be positive and num-workers non-negative")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps":
            backend = getattr(torch.backends, "mps", None)
            if backend is None or not backend.is_available():
                raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    backend = getattr(torch.backends, "mps", None)
    if backend is not None and backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_contract(data_dir: Path) -> tuple[dict, tuple[str, ...], tuple[str, ...]]:
    build_path = data_dir / "data_build_config.json"
    catalog_path = data_dir / "feature_catalog.json"
    if not build_path.is_file() or not catalog_path.is_file():
        raise FileNotFoundError("processed_data is missing its JSON contracts")
    build = json.loads(build_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if PREDICT_STEPS not in set(map(int, build.get("horizons", []))):
        raise ValueError("The processed dataset has no 30-step target")
    if LABEL_COLUMN not in set(build.get("future_labels", [])):
        raise ValueError(f"The processed dataset has no {LABEL_COLUMN}")
    if build.get("future_features_used") is not False:
        raise ValueError("The data contract does not guarantee causal input construction")
    history_columns = tuple(catalog.get("raw_signal_columns", ()))
    control_columns = tuple(catalog.get("primary_valve_columns", ()))
    if len(history_columns) != int(catalog.get("enabled_raw_signal_count", -1)):
        raise ValueError("Raw feature count disagrees with feature_catalog.json")
    if len(history_columns) == 0 or len(control_columns) == 0:
        raise ValueError("Raw history or future-control column list is empty")
    if len(set(history_columns)) != len(history_columns):
        raise ValueError("Duplicate raw history columns")
    if not set(control_columns).issubset(history_columns):
        raise ValueError("A future control is not an enabled raw signal")
    if any(column.startswith("Future_") for column in (*history_columns, *control_columns)):
        raise AssertionError("A future label entered an input list")
    return build, history_columns, control_columns


def load_development_data(
    data_dir: Path,
    history_columns: tuple[str, ...],
    control_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    build, _, _ = read_contract(data_dir)
    train_path = data_dir / "train_clean.pkl"
    validation_path = data_dir / "val_clean.pkl"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("train_clean.pkl or val_clean.pkl is missing")
    train = joblib.load(train_path)
    validation = joblib.load(validation_path)
    if not isinstance(train, pd.DataFrame) or not isinstance(validation, pd.DataFrame):
        raise TypeError("Development pickles must contain pandas DataFrames")
    train = train.copy().reset_index(drop=True)
    validation = validation.copy().reset_index(drop=True)
    required = {
        *history_columns,
        *control_columns,
        TARGET_COLUMN,
        LABEL_COLUMN,
        "file_id",
        "source_group",
        "source_variant",
        "parent_id",
        "source_row_index",
        "source_timestamp",
    }
    for name, frame in (("training", train), ("validation", validation)):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{name} is missing columns: {sorted(missing)}")
        selected = list(dict.fromkeys((*history_columns, *control_columns, LABEL_COLUMN)))
        if not np.isfinite(frame[selected].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name} contains NaN or infinity in model columns")
    expected_validation = str(build.get("validation_parent"))
    if set(map(str, validation["parent_id"].unique())) != {expected_validation}:
        raise ValueError("Validation parent disagrees with data_build_config.json")
    if set(map(str, validation["source_variant"].unique())) != {"Original"}:
        raise ValueError("Validation must contain Original data only")
    if set(map(str, train["source_group"].unique())) & set(
        map(str, validation["source_group"].unique())
    ):
        raise ValueError("Training and validation share a source_group")
    return train, validation, build


def build_common_origins(
    frame: pd.DataFrame,
    common_lookback: int = COMMON_ORIGIN_LOOKBACK,
    predict_steps: int = PREDICT_STEPS,
) -> np.ndarray:
    ends: list[np.ndarray] = []
    for file_id, raw_indices in frame.groupby("file_id", sort=False).indices.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        if len(indices) < common_lookback + predict_steps:
            continue
        if len(indices) > 1 and not np.all(np.diff(indices) == 1):
            raise ValueError(f"file_id={file_id} is not contiguous in the DataFrame")
        source_rows = pd.to_numeric(
            frame.iloc[indices]["source_row_index"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(source_rows).all() or not np.all(np.diff(source_rows) == 1):
            raise ValueError(f"file_id={file_id} crosses a missing or removed source row")
        local = np.arange(common_lookback - 1, len(indices) - predict_steps)
        ends.append(indices[local])
    if not ends:
        raise ValueError("No valid common forecast origins were found")
    result = np.concatenate(ends).astype(np.int64)
    return result


def audit_origins(
    frame: pd.DataFrame,
    ends: np.ndarray,
    lookback: int,
    predict_steps: int,
    split_name: str,
) -> dict[str, float | int | str]:
    file_values = frame["file_id"].astype(str).to_numpy()
    source_rows = pd.to_numeric(frame["source_row_index"], errors="raise").to_numpy(int)
    target = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
    stored = frame[LABEL_COLUMN].to_numpy(dtype=np.float64)
    starts = ends - lookback + 1
    target_indices = ends + predict_steps
    control_ends = ends + predict_steps - 1
    if starts.min() < 0 or target_indices.max() >= len(frame):
        raise AssertionError("A window crosses a DataFrame boundary")
    origin_files = file_values[ends]
    if not (
        np.all(file_values[starts] == origin_files)
        and np.all(file_values[control_ends] == origin_files)
        and np.all(file_values[target_indices] == origin_files)
    ):
        raise AssertionError("A history/control/target window crosses file_id")
    expected_span = lookback - 1 + predict_steps
    if not np.all(source_rows[target_indices] - source_rows[starts] == expected_span):
        raise AssertionError("A window crosses a source-row gap")
    absolute_errors = np.abs(stored[ends] - target[target_indices])
    max_error = float(absolute_errors.max())
    if not np.allclose(stored[ends], target[target_indices], rtol=0.0, atol=1e-9):
        raise AssertionError(f"{LABEL_COLUMN} is not aligned with t+{predict_steps}")
    return {
        "split": split_name,
        "audited_origins": int(len(ends)),
        "lookback": lookback,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": predict_steps,
        "maximum_label_alignment_error_k": max_error,
    }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
        drop_last=False,
    )


def weighted_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    losses = F.huber_loss(prediction, target, reduction="none", delta=delta)
    return torch.sum(losses * weight) / torch.clamp(torch.sum(weight), min=1e-12)


def inverse_delta(tensor: torch.Tensor, scaler: Standardizer) -> torch.Tensor:
    mean = torch.as_tensor(float(scaler.mean.reshape(-1)[0]), device=tensor.device)
    scale = torch.as_tensor(float(scaler.scale.reshape(-1)[0]), device=tensor.device)
    return tensor * scale + mean


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scaler: Standardizer,
    huber_delta: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    predictions: list[np.ndarray] = []
    actual_deltas: list[np.ndarray] = []
    current_temperatures: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    loss_numerator = 0.0
    weight_total = 0.0
    for history, controls, target_scaled, target_k, current_k, weight, batch_ends in loader:
        history = history.to(device, non_blocking=True)
        controls = controls.to(device, non_blocking=True)
        target_scaled = target_scaled.to(device, non_blocking=True)
        weight_device = weight.to(device, non_blocking=True)
        output_scaled = model(history, controls)
        element_loss = F.huber_loss(
            output_scaled, target_scaled, reduction="none", delta=huber_delta
        )
        loss_numerator += float(torch.sum(element_loss * weight_device).item())
        weight_total += float(torch.sum(weight_device).item())
        predictions.append(inverse_delta(output_scaled, target_scaler).cpu().numpy())
        actual_deltas.append(target_k.numpy())
        current_temperatures.append(current_k.numpy())
        weights.append(weight.numpy())
        ends.append(batch_ends.numpy())
    predicted_delta = np.concatenate(predictions).astype(np.float64)
    actual_delta = np.concatenate(actual_deltas).astype(np.float64)
    current = np.concatenate(current_temperatures).astype(np.float64)
    weight_values = np.concatenate(weights).astype(np.float64)
    end_values = np.concatenate(ends).astype(np.int64)
    actual_temperature = current + actual_delta
    predicted_temperature = current + predicted_delta
    errors = predicted_temperature - actual_temperature
    rapid = weight_values > 1.0
    stats = {
        "weighted_huber_scaled": loss_numerator / max(weight_total, 1e-12),
        "rmse_k": float(np.sqrt(np.mean(errors**2))),
        "mae_k": float(np.mean(np.abs(errors))),
        "rapid_rmse_k": (
            float(np.sqrt(np.mean(errors[rapid] ** 2))) if rapid.any() else float("nan")
        ),
    }
    arrays = {
        "ends": end_values,
        "current_k": current,
        "actual_delta_k": actual_delta,
        "predicted_delta_k": predicted_delta,
        "actual_temperature_k": actual_temperature,
        "predicted_temperature_k": predicted_temperature,
        "weights": weight_values,
    }
    return stats, arrays


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "rmse_k": float(np.sqrt(np.mean(residual**2))),
        "mae_k": float(np.mean(np.abs(residual))),
        "p95_absolute_error_k": float(np.quantile(np.abs(residual), 0.95)),
        "max_absolute_error_k": float(np.max(np.abs(residual))),
        "bias_k": float(np.mean(residual)),
        "r2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan"),
    }


def direction_accuracy(actual_delta: np.ndarray, predicted_delta: np.ndarray) -> float:
    actual_direction = np.sign(np.asarray(actual_delta))
    predicted_direction = np.sign(np.asarray(predicted_delta))
    return float(np.mean(actual_direction == predicted_direction))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_datasets(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    history_columns: tuple[str, ...],
    control_columns: tuple[str, ...],
    train_ends: np.ndarray,
    validation_ends: np.ndarray,
    lookback: int,
    predict_steps: int,
    rapid_quantile: float,
    rapid_weight: float,
):
    train_current = train[TARGET_COLUMN].to_numpy(dtype=np.float64)
    validation_current = validation[TARGET_COLUMN].to_numpy(dtype=np.float64)
    train_future = train[LABEL_COLUMN].to_numpy(dtype=np.float64)
    validation_future = validation[LABEL_COLUMN].to_numpy(dtype=np.float64)
    train_delta = train_future - train_current
    validation_delta = validation_future - validation_current
    rapid_threshold = float(np.quantile(train_delta[train_ends], rapid_quantile))
    train_weights = np.ones(len(train), dtype=np.float32)
    validation_weights = np.ones(len(validation), dtype=np.float32)
    train_weights[train_delta <= rapid_threshold] = rapid_weight
    validation_weights[validation_delta <= rapid_threshold] = rapid_weight

    history_scaler = Standardizer.fit(train[list(history_columns)].to_numpy(float))
    control_scaler = Standardizer.fit(train[list(control_columns)].to_numpy(float))
    target_scaler = Standardizer.fit(train_delta[train_ends])
    train_history = history_scaler.transform(train[list(history_columns)].to_numpy(float)).astype(np.float32)
    val_history = history_scaler.transform(validation[list(history_columns)].to_numpy(float)).astype(np.float32)
    train_controls = control_scaler.transform(train[list(control_columns)].to_numpy(float)).astype(np.float32)
    val_controls = control_scaler.transform(validation[list(control_columns)].to_numpy(float)).astype(np.float32)
    train_scaled = target_scaler.transform(train_delta).astype(np.float32)
    val_scaled = target_scaler.transform(validation_delta).astype(np.float32)

    train_dataset = WindowDataset(
        train_history, train_controls, train_scaled, train_delta.astype(np.float32),
        train_current.astype(np.float32), train_weights, train_ends, lookback, predict_steps,
    )
    validation_dataset = WindowDataset(
        val_history, val_controls, val_scaled, validation_delta.astype(np.float32),
        validation_current.astype(np.float32), validation_weights, validation_ends,
        lookback, predict_steps,
    )
    return (
        train_dataset,
        validation_dataset,
        history_scaler,
        control_scaler,
        target_scaler,
        rapid_threshold,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.data_dir = args.data_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    run_dir = run_directory(args.results_dir, args.lookback, args.hidden_size, args.seed)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Run exists: {run_dir}; use --overwrite to replace it")
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    build, history_columns, control_columns = read_contract(args.data_dir)
    train, validation, build = load_development_data(
        args.data_dir, history_columns, control_columns
    )
    train_ends = build_common_origins(train)
    validation_ends = build_common_origins(validation)
    train_audit = audit_origins(
        train, train_ends, args.lookback, PREDICT_STEPS, "training"
    )
    validation_audit = audit_origins(
        validation, validation_ends, args.lookback, PREDICT_STEPS, "validation"
    )
    (
        train_dataset,
        validation_dataset,
        history_scaler,
        control_scaler,
        target_scaler,
        rapid_threshold,
    ) = build_datasets(
        train,
        validation,
        history_columns,
        control_columns,
        train_ends,
        validation_ends,
        args.lookback,
        PREDICT_STEPS,
        args.rapid_quantile,
        args.rapid_weight,
    )

    device = choose_device(args.device)
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.seed, args.num_workers, device
    )
    train_eval_loader = make_loader(
        train_dataset, args.batch_size, False, args.seed + 1, args.num_workers, device
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, False, args.seed + 2, args.num_workers, device
    )
    model = ControlledGRU(
        len(history_columns),
        len(control_columns),
        args.hidden_size,
        args.control_hidden_size,
        args.fusion_hidden_size,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, args.patience // 3),
        min_lr=1e-6,
    )
    print(
        f"LOOKBACK={args.lookback} HIDDEN={args.hidden_size} SEED={args.seed} "
        f"DEVICE={device} TRAIN_WINDOWS={len(train_dataset)} "
        f"VALIDATION_WINDOWS={len(validation_dataset)}",
        flush=True,
    )
    print(
        f"HISTORY_CHANNELS={len(history_columns)} FUTURE_CONTROL_CHANNELS={len(control_columns)} "
        f"VALIDATION_PARENT={build.get('validation_parent')} TEST_DATA_OPENED=false",
        flush=True,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_rmse = math.inf
    stale_epochs = 0
    history_rows: list[dict[str, float | int]] = []
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for history, controls, target_scaled, _, _, weights, _ in train_loader:
            history = history.to(device, non_blocking=True)
            controls = controls.to(device, non_blocking=True)
            target_scaled = target_scaled.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(history, controls)
            loss = weighted_huber(output, target_scaled, weights, args.huber_delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        train_stats, _ = evaluate(
            model, train_eval_loader, device, target_scaler, args.huber_delta
        )
        validation_stats, _ = evaluate(
            model, validation_loader, device, target_scaler, args.huber_delta
        )
        scheduler.step(validation_stats["rmse_k"])
        row = {
            "epoch": epoch,
            "train_weighted_huber_scaled": train_stats["weighted_huber_scaled"],
            "train_rmse_k": train_stats["rmse_k"],
            "train_rapid_rmse_k": train_stats["rapid_rmse_k"],
            "validation_weighted_huber_scaled": validation_stats["weighted_huber_scaled"],
            "validation_rmse_k": validation_stats["rmse_k"],
            "validation_rapid_rmse_k": validation_stats["rapid_rmse_k"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_rmse_k={train_stats['rmse_k']:.8f} "
            f"val_rmse_k={validation_stats['rmse_k']:.8f} "
            f"lr={row['learning_rate']:.3e}",
            flush=True,
        )
        current_rmse = validation_stats["rmse_k"]
        if current_rmse < best_validation_rmse - args.min_delta:
            best_validation_rmse = current_rmse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    synchronize(device)
    training_seconds = float(time.perf_counter() - started)
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    _, arrays = evaluate(model, validation_loader, device, target_scaler, args.huber_delta)
    actual = arrays["actual_temperature_k"]
    predicted = arrays["predicted_temperature_k"]
    current = arrays["current_k"]
    actual_delta = arrays["actual_delta_k"]
    predicted_delta = arrays["predicted_delta_k"]
    validation_metrics = regression_metrics(actual, predicted)
    delta_metrics = regression_metrics(actual_delta, predicted_delta)
    persistence_metrics = regression_metrics(actual, current)
    rapid_mask = arrays["weights"] > 1.0
    rapid_metrics = regression_metrics(actual[rapid_mask], predicted[rapid_mask])

    metadata_rows = validation.iloc[arrays["ends"]]
    predictions = pd.DataFrame(
        {
            "frame_row_index": arrays["ends"],
            "file_id": metadata_rows["file_id"].to_numpy(),
            "parent_id": metadata_rows["parent_id"].to_numpy(),
            "source_row_index": metadata_rows["source_row_index"].to_numpy(),
            "source_timestamp": metadata_rows["source_timestamp"].to_numpy(),
            "current_Thv_k": current,
            "actual_delta_Thv_30step_k": actual_delta,
            "predicted_delta_Thv_30step_k": predicted_delta,
            "actual_Future_Thv_30step_k": actual,
            "predicted_Future_Thv_30step_k": predicted,
            "residual_k": predicted - actual,
            "rapid_cooling_mask": rapid_mask.astype(np.int8),
        }
    )
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False)
    np.savez_compressed(
        run_dir / "scalers.npz",
        history_mean=history_scaler.mean,
        history_scale=history_scaler.scale,
        control_mean=control_scaler.mean,
        control_scale=control_scaler.scale,
        target_mean=target_scaler.mean,
        target_scale=target_scaler.scale,
        rapid_threshold_k=np.asarray([rapid_threshold]),
    )
    checkpoint = {
        "model_state_dict": best_state,
        "model_class": "ControlledGRU",
        "history_feature_columns": list(history_columns),
        "future_control_columns": list(control_columns),
        "lookback": args.lookback,
        "hidden_size": args.hidden_size,
        "control_hidden_size": args.control_hidden_size,
        "fusion_hidden_size": args.fusion_hidden_size,
        "predict_steps": PREDICT_STEPS,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "validation_metrics": validation_metrics,
        "checkpoint_selection_metric": SELECTION_METRIC,
    }
    torch.save(checkpoint, run_dir / "best_model.pt")
    payload = {
        "study": "stage1_global_gru_lookback_hidden_search",
        "model": "controlled_gru_delta",
        "lookback": args.lookback,
        "hidden_size": args.hidden_size,
        "gru_layers": 1,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "seed": args.seed,
        "target": "Delta_Thv_30step",
        "source_label": LABEL_COLUMN,
        "history_feature_columns": list(history_columns),
        "future_control_columns": list(control_columns),
        "future_control_alignment": "u(t)..u(t+29)",
        "engineered_features_used": False,
        "pcmci_used": False,
        "future_noncontrol_measurements_used": False,
        "future_label_used_as_input": False,
        "training_parents": sorted(map(str, train["parent_id"].unique())),
        "training_variants": sorted(map(str, train["source_variant"].unique())),
        "validation_parent": str(build.get("validation_parent")),
        "validation_variant": "Original",
        "held_out_test_parent": str(build.get("test_parent")),
        "test_data_loaded": False,
        "test_metrics_present": False,
        "scaler_fit_source": "train_clean.pkl only",
        "checkpoint_selection_metric": SELECTION_METRIC,
        "train_window_audit": train_audit,
        "validation_window_audit": validation_audit,
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "rapid_threshold_k_train_only": rapid_threshold,
        "rapid_quantile": args.rapid_quantile,
        "rapid_weight": args.rapid_weight,
        "trainable_parameters": count_parameters(model),
        "training_seconds_total": training_seconds,
        "completed_epochs": len(history_rows),
        "best_epoch": best_epoch,
        "best_epoch_validation_rmse_k": best_validation_rmse,
        "validation_metrics": validation_metrics,
        "rapid_validation_metrics": rapid_metrics,
        "delta_validation_metrics": delta_metrics,
        "persistence_validation_metrics": persistence_metrics,
        "validation_direction_accuracy": direction_accuracy(actual_delta, predicted_delta),
        "device": str(device),
        "torch_version": torch.__version__,
        "training_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    atomic_json(metrics_path, payload)
    print(f"BEST_EPOCH={best_epoch}", flush=True)
    print(f"FINAL_VALIDATION_RMSE_K={validation_metrics['rmse_k']:.10f}", flush=True)
    print(f"PERSISTENCE_VALIDATION_RMSE_K={persistence_metrics['rmse_k']:.10f}", flush=True)
    print("TEST_DATA_OPENED=false", flush=True)
    print(f"RESULTS_SAVED={run_dir}", flush=True)


if __name__ == "__main__":
    main()
