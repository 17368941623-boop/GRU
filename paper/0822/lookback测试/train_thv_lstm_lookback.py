#!/usr/bin/env python3
"""Leakage-safe LSTM lookback experiment for Thv(t+5).

This script is intentionally restricted to development data:

* train_clean.pkl is used for fitting the model and both scalers;
* 0617-ALL is removed before any fitted statistic is calculated;
* val_clean.pkl (Original 260118) is used for early stopping and metrics;
* no test PKL is opened anywhere in this program;
* windows are built independently inside each ``file_id`` and can never cross
  a parent file or a discontinuity segment created by data_fil.py;
* every lookback uses forecast origins with at least 60 historical samples, so
  all 12 lookback settings compare exactly the same labels and sample count.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "processed_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

ALLOWED_LOOKBACKS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
COMMON_ORIGIN_LOOKBACK = max(ALLOWED_LOOKBACKS)
TARGET_COL = "Thv"
LABEL_COL = "Future_Thv_5step"
PREDICT_STEPS = 5
SAMPLE_PERIOD_SECONDS = 10.0
EXCLUDED_PARENT = "0617-ALL"
EXPECTED_TRAIN_PARENTS = {
    "0623-BACK",
    "0715-ALL",
    "0715-BACK",
    "251130",
    "251226",
    "260403",
}
EXPECTED_VALIDATION_PARENT = "260118"
EXPECTED_TEST_PARENT_IN_BUILD_CONFIG = "260501"

# Raw exported process measurements only.  No engineered feature, availability
# mask, event mask, metadata, or future label enters this lookback comparison.
RAW_FEATURE_COLS = (
    "TE8310",
    "TE8351",
    "TE8352",
    "FT8351",
    "PT8310",
    "PT8351",
    "PT8352",
    "CV8312",
    "CV8311",
    "CV8310",
    "CV8313",
    "CV8300",
    "CV8351",
    "Thv",
    "TE8353",
    "Tef",
    "Tcd",
    "DTbr",
    "EC-V2",
    "COOLDOWN",
)


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
            raise ValueError("Cannot fit a scaler on NaN/inf values")
        mean = array.mean(axis=0)
        scale = array.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(mean=mean.astype(np.float64), scale=scale.astype(np.float64))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        original_ndim = array.ndim
        if original_ndim == 1:
            array = array.reshape(-1, 1)
        transformed = (array - self.mean) / self.scale
        if original_ndim == 1:
            transformed = transformed.reshape(-1)
        return transformed

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        original_ndim = array.ndim
        if original_ndim == 1:
            array = array.reshape(-1, 1)
        restored = array * self.scale + self.mean
        if original_ndim == 1:
            restored = restored.reshape(-1)
        return restored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raw-input LSTM lookback experiment for direct Thv(t+5) prediction."
    )
    parser.add_argument(
        "--lookback",
        type=int,
        required=True,
        choices=ALLOWED_LOOKBACKS,
        help="History length in samples.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--dynamic-history-steps",
        type=int,
        default=30,
        help="A validation origin is dynamic if a causal valve event occurred in this trailing window.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if args.hidden_size < 1 or args.num_layers < 1:
        raise ValueError("Hidden size and number of layers must be >= 1")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0,1)")
    if args.patience < 1 or args.min_delta < 0:
        raise ValueError("Patience must be >=1 and min-delta non-negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.dynamic_history_steps < 1:
        raise ValueError("--dynamic-history-steps must be >= 1")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {path}")
    frame = joblib.load(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{path} does not contain a pandas DataFrame")
    return frame.copy()


def assert_build_config(data_dir: Path) -> None:
    path = data_dir / "data_build_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing data build contract: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("validation_parent") != EXPECTED_VALIDATION_PARENT:
        raise ValueError("processed_data validation parent is not 260118")
    if config.get("test_parent") != EXPECTED_TEST_PARENT_IN_BUILD_CONFIG:
        raise ValueError("processed_data test parent is not 260501")
    labels = set(config.get("future_labels", []))
    if LABEL_COL not in labels:
        raise ValueError(f"processed_data does not contain the required label {LABEL_COL}")
    if config.get("future_features_used") is not False:
        raise ValueError("processed_data contract does not guarantee future-feature exclusion")


def prepare_development_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Intentionally load only train and validation.  There is no test filename
    # or test-loading branch in this lookback-selection program.
    train = load_dataframe(data_dir / "train_clean.pkl")
    val = load_dataframe(data_dir / "val_clean.pkl")

    required = {
        *RAW_FEATURE_COLS,
        LABEL_COL,
        "file_id",
        "source_group",
        "source_variant",
        "parent_id",
        "source_row_index",
        "source_timestamp",
        "valve_event_mask",
    }
    for name, frame in (("training", train), ("validation", val)):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{name} frame is missing columns: {sorted(missing)}")

    # The exclusion happens before feature/label scaling and before window
    # construction.  Resetting the index makes each retained file_id occupy a
    # contiguous range again without joining different file_ids together.
    train = train.loc[train["parent_id"].astype(str) != EXCLUDED_PARENT].copy()
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)

    train_parents = set(map(str, train["parent_id"].unique()))
    if train_parents != EXPECTED_TRAIN_PARENTS:
        raise ValueError(
            "Unexpected training parents after removing 0617-ALL: "
            f"expected={sorted(EXPECTED_TRAIN_PARENTS)}, actual={sorted(train_parents)}"
        )
    if EXCLUDED_PARENT in train_parents:
        raise AssertionError("0617-ALL survived the training exclusion")
    if set(map(str, val["parent_id"].unique())) != {EXPECTED_VALIDATION_PARENT}:
        raise ValueError("Validation is not exclusively parent 260118")
    if set(map(str, val["source_variant"].unique())) != {"Original"}:
        raise ValueError("Validation contains augmented samples")

    train_groups = set(map(str, train["source_group"].unique()))
    val_groups = set(map(str, val["source_group"].unique()))
    if train_groups & val_groups:
        raise ValueError(f"Train/validation source-group overlap: {train_groups & val_groups}")

    forbidden_features = [column for column in RAW_FEATURE_COLS if column.startswith("Future_")]
    if forbidden_features or LABEL_COL in RAW_FEATURE_COLS:
        raise AssertionError("A future label entered RAW_FEATURE_COLS")
    for name, frame in (("training", train), ("validation", val)):
        numeric = frame[[*RAW_FEATURE_COLS, LABEL_COL]].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            bad = frame[[*RAW_FEATURE_COLS, LABEL_COL]].columns[
                ~np.isfinite(numeric).all(axis=0)
            ].tolist()
            raise ValueError(f"{name} contains NaN/inf in selected columns: {bad}")
    return train, val


def _assert_source_rows_contiguous(part: pd.DataFrame, file_id: str) -> None:
    source_rows = pd.to_numeric(part["source_row_index"], errors="coerce").to_numpy()
    if not np.isfinite(source_rows).all():
        raise ValueError(f"file_id={file_id} has an invalid source_row_index")
    if len(source_rows) > 1 and not np.all(np.diff(source_rows) == 1):
        raise ValueError(
            f"file_id={file_id} crosses a removed row; data_fil.py should have split it"
        )


def build_common_window_ends(
    frame: pd.DataFrame,
    common_origin_lookback: int = COMMON_ORIGIN_LOOKBACK,
) -> np.ndarray:
    """Return common forecast origins without crossing file_id boundaries."""
    ends: list[np.ndarray] = []
    for file_id, indices in frame.groupby("file_id", sort=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) == 0:
            continue
        if not np.all(np.diff(idx) == 1):
            raise ValueError(f"Rows for file_id={file_id} are not contiguous in the DataFrame")
        part = frame.iloc[idx]
        if part["source_group"].nunique() != 1:
            raise ValueError(f"file_id={file_id} maps to multiple source files")
        _assert_source_rows_contiguous(part, str(file_id))
        if len(idx) >= common_origin_lookback:
            ends.append(
                np.arange(
                    idx[0] + common_origin_lookback - 1,
                    idx[-1] + 1,
                    dtype=np.int64,
                )
            )
    if not ends:
        raise ValueError("No segment has enough rows for the common 60-step origin rule")
    result = np.concatenate(ends)
    if len(np.unique(result)) != len(result):
        raise AssertionError("Duplicate forecast origins were created")
    return result


def assert_window_boundaries(frame: pd.DataFrame, ends: np.ndarray, lookback: int) -> None:
    starts = ends - lookback + 1
    if starts.min() < 0 or ends.max() >= len(frame):
        raise IndexError("Window origin is outside the DataFrame")
    file_ids = frame["file_id"].astype(str).to_numpy()
    source_groups = frame["source_group"].astype(str).to_numpy()
    if not np.array_equal(file_ids[starts], file_ids[ends]):
        raise AssertionError("At least one lookback window crosses a file_id boundary")
    if not np.array_equal(source_groups[starts], source_groups[ends]):
        raise AssertionError("At least one lookback window crosses a source file boundary")


class WindowDataset(Dataset):
    """Lazy windows indexed by prevalidated, common forecast origins."""

    def __init__(
        self,
        x_scaled: np.ndarray,
        y_scaled: np.ndarray,
        ends: np.ndarray,
        lookback: int,
    ):
        self.x = np.asarray(x_scaled, dtype=np.float32)
        self.y = np.asarray(y_scaled, dtype=np.float32).reshape(-1)
        self.ends = np.asarray(ends, dtype=np.int64)
        self.lookback = int(lookback)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, index: int):
        end = int(self.ends[index])
        start = end - self.lookback + 1
        return (
            torch.from_numpy(self.x[start:end + 1]),
            torch.tensor(self.y[end], dtype=torch.float32),
            torch.tensor(end, dtype=torch.int64),
        )


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(self.output_dropout(output[:, -1, :])).squeeze(-1)


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed + seed_offset)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )


def inverse_target(values: np.ndarray, scaler: Standardizer) -> np.ndarray:
    return scaler.inverse_transform(values).reshape(-1)


def evaluate_scaled_rmse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scaler: Standardizer,
) -> float:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            predictions.append(model(x.to(device)).cpu().numpy())
            targets.append(y.numpy())
    predicted = inverse_target(np.concatenate(predictions), target_scaler)
    actual = inverse_target(np.concatenate(targets), target_scaler)
    return float(np.sqrt(np.mean(np.square(predicted - actual))))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_scaler: Standardizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], list[dict[str, float]], int, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, args.patience // 3),
        min_lr=1e-6,
    )
    criterion = nn.HuberLoss(delta=1.0)
    best_rmse = float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = int(x.shape[0])
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size

        train_huber = loss_sum / max(sample_count, 1)
        val_rmse_k = evaluate_scaled_rmse(model, val_loader, device, target_scaler)
        scheduler.step(val_rmse_k)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_huber_scaled": train_huber,
                "val_rmse_k": val_rmse_k,
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_huber_scaled={train_huber:.8f} | "
            f"val_rmse_k={val_rmse_k:.8f} | lr={current_lr:.3e}"
        )

        if val_rmse_k < best_rmse - args.min_delta:
            best_rmse = val_rmse_k
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best_epoch={best_epoch}, best_val_rmse_k={best_rmse:.8f}"
                )
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    return best_state, history, best_epoch, best_rmse


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scaler: Standardizer,
) -> tuple[np.ndarray, np.ndarray]:
    scaled_predictions: list[np.ndarray] = []
    end_indices: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, _, ends in loader:
            scaled_predictions.append(model(x.to(device)).cpu().numpy())
            end_indices.append(ends.numpy())
    prediction = inverse_target(np.concatenate(scaled_predictions), target_scaler)
    return prediction, np.concatenate(end_indices).astype(np.int64)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    absolute_error = np.abs(error)
    denominator = float(np.sum(np.square(actual - actual.mean())))
    r2 = float("nan") if denominator <= 0 else 1.0 - float(np.sum(np.square(error))) / denominator
    return {
        "mae_k": float(absolute_error.mean()),
        "rmse_k": float(np.sqrt(np.mean(np.square(error)))),
        "p95_absolute_error_k": float(np.percentile(absolute_error, 95)),
        "max_absolute_error_k": float(absolute_error.max()),
        "bias_k": float(error.mean()),
        "r2": r2,
        "n_windows": int(len(actual)),
    }


def causal_dynamic_mask(frame: pd.DataFrame, history_steps: int) -> np.ndarray:
    result = np.zeros(len(frame), dtype=bool)
    for _, indices in frame.groupby("file_id", sort=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        events = frame.iloc[idx]["valve_event_mask"].to_numpy(dtype=float) > 0.5
        rolling = (
            pd.Series(events.astype(np.int8))
            .rolling(window=history_steps, min_periods=1)
            .max()
            .to_numpy(dtype=bool)
        )
        result[idx] = rolling
    return result


def save_training_plot(history: Sequence[dict[str, float]], path: Path) -> None:
    table = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(table["epoch"], table["train_huber_scaled"], color="#0072B2")
    axes[0].set_ylabel("Train Huber (scaled)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(table["epoch"], table["val_rmse_k"], color="#D55E00")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation RMSE (K)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_prediction_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    file_ids: np.ndarray,
    path: Path,
    title: str,
) -> None:
    x = np.arange(len(actual))
    residual = predicted - actual
    boundaries = np.flatnonzero(file_ids[1:] != file_ids[:-1]) + 1
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(x, actual, color="black", linewidth=0.9, label="Measured Thv(t+5)")
    axes[0].plot(x, predicted, color="#D55E00", linewidth=0.8, label="LSTM prediction")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(x, residual, color="#0072B2", linewidth=0.7)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Common validation forecast origin")
    axes[1].set_ylabel("Residual (K)")
    axes[1].grid(alpha=0.2)
    for boundary in boundaries:
        axes[0].axvline(boundary, color="gray", linewidth=0.5, alpha=0.35)
        axes[1].axvline(boundary, color="gray", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_scalers(
    path: Path,
    input_scaler: Standardizer,
    target_scaler: Standardizer,
) -> None:
    np.savez(
        path,
        feature_names=np.asarray(RAW_FEATURE_COLS, dtype=str),
        input_mean=input_scaler.mean,
        input_scale=input_scaler.scale,
        target_name=np.asarray([LABEL_COL], dtype=str),
        target_mean=target_scaler.mean,
        target_scale=target_scaler.scale,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    assert_build_config(args.data_dir)

    run_dir = args.output_dir / f"lookback_{args.lookback:02d}" / f"seed_{args.seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run already exists: {run_dir}. Pass --overwrite to replace its files."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_frame, val_frame = prepare_development_frames(args.data_dir)
    train_ends = build_common_window_ends(train_frame)
    val_ends = build_common_window_ends(val_frame)
    assert_window_boundaries(train_frame, train_ends, args.lookback)
    assert_window_boundaries(val_frame, val_ends, args.lookback)

    # Input statistics use only retained training rows.  Target statistics use
    # only the common training forecast origins, making them identical for all
    # lookback values and independent of validation/test data.
    input_scaler = Standardizer.fit(
        train_frame[list(RAW_FEATURE_COLS)].to_numpy(dtype=np.float64)
    )
    target_scaler = Standardizer.fit(
        train_frame[LABEL_COL].to_numpy(dtype=np.float64)[train_ends]
    )
    train_x = input_scaler.transform(
        train_frame[list(RAW_FEATURE_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    val_x = input_scaler.transform(
        val_frame[list(RAW_FEATURE_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    train_y = target_scaler.transform(
        train_frame[LABEL_COL].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    val_y = target_scaler.transform(
        val_frame[LABEL_COL].to_numpy(dtype=np.float64)
    ).astype(np.float32)

    train_dataset = WindowDataset(train_x, train_y, train_ends, args.lookback)
    val_dataset = WindowDataset(val_x, val_y, val_ends, args.lookback)
    train_loader = make_loader(train_dataset, args, shuffle=True, seed_offset=0)
    val_loader = make_loader(val_dataset, args, shuffle=False, seed_offset=1)

    device = choose_device()
    model = LSTMRegressor(
        input_size=len(RAW_FEATURE_COLS),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(
        f"Experiment=Thv(t+5) raw-input LSTM | lookback={args.lookback} | "
        f"seed={args.seed} | device={device}"
    )
    print(
        f"Training parents={sorted(EXPECTED_TRAIN_PARENTS)} | "
        f"excluded={EXCLUDED_PARENT} | train_windows={len(train_dataset)} | "
        f"val_windows={len(val_dataset)}"
    )
    print(
        "Boundary contract: each window stays inside one file_id/source_group; "
        "all lookbacks use common origins requiring 60 historical rows."
    )

    best_state, history, best_epoch, best_training_rmse = train_model(
        model,
        train_loader,
        val_loader,
        target_scaler,
        args,
        device,
    )
    model.load_state_dict(best_state)
    prediction, prediction_ends = predict(model, val_loader, device, target_scaler)
    if not np.array_equal(prediction_ends, val_ends):
        raise AssertionError("Validation prediction order changed unexpectedly")

    actual = val_frame[LABEL_COL].to_numpy(dtype=np.float64)[val_ends]
    persistence = val_frame[TARGET_COL].to_numpy(dtype=np.float64)[val_ends]
    validation_metrics = regression_metrics(actual, prediction)
    persistence_metrics = regression_metrics(actual, persistence)
    dynamic_rows = causal_dynamic_mask(val_frame, args.dynamic_history_steps)[val_ends]
    dynamic_metrics = (
        regression_metrics(actual[dynamic_rows], prediction[dynamic_rows])
        if dynamic_rows.any()
        else None
    )
    persistence_dynamic_metrics = (
        regression_metrics(actual[dynamic_rows], persistence[dynamic_rows])
        if dynamic_rows.any()
        else None
    )

    prediction_table = pd.DataFrame(
        {
            "frame_row_index": prediction_ends,
            "file_id": val_frame.iloc[prediction_ends]["file_id"].to_numpy(),
            "source_group": val_frame.iloc[prediction_ends]["source_group"].to_numpy(),
            "source_row_index": val_frame.iloc[prediction_ends]["source_row_index"].to_numpy(),
            "source_timestamp": val_frame.iloc[prediction_ends]["source_timestamp"].to_numpy(),
            "current_Thv_k": persistence,
            "actual_Future_Thv_5step_k": actual,
            "predicted_Future_Thv_5step_k": prediction,
            "residual_k": prediction - actual,
            "causal_dynamic_mask": dynamic_rows.astype(np.int8),
        }
    )
    prediction_table.to_csv(run_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    save_training_plot(history, run_dir / "training_curve.png")
    save_prediction_plot(
        actual,
        prediction,
        prediction_table["file_id"].astype(str).to_numpy(),
        run_dir / "validation_prediction.png",
        f"Thv(t+5) LSTM | lookback={args.lookback} | seed={args.seed}",
    )
    save_scalers(run_dir / "scalers.npz", input_scaler, target_scaler)

    metadata = {
        "experiment": "Thv_t_plus_5_raw_lstm_lookback",
        "lookback": args.lookback,
        "lookback_seconds": args.lookback * SAMPLE_PERIOD_SECONDS,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "seed": args.seed,
        "target": TARGET_COL,
        "label": LABEL_COL,
        "predict_steps": PREDICT_STEPS,
        "predict_seconds": PREDICT_STEPS * SAMPLE_PERIOD_SECONDS,
        "input_mode": "raw_measurements_only",
        "feature_columns": list(RAW_FEATURE_COLS),
        "feature_count": len(RAW_FEATURE_COLS),
        "excluded_parent": EXCLUDED_PARENT,
        "training_parents": sorted(EXPECTED_TRAIN_PARENTS),
        "validation_parent": EXPECTED_VALIDATION_PARENT,
        "test_loaded": False,
        "window_boundary_key": "file_id",
        "train_rows_after_exclusion": int(len(train_frame)),
        "validation_rows": int(len(val_frame)),
        "train_windows": int(len(train_dataset)),
        "validation_windows": int(len(val_dataset)),
        "best_epoch": best_epoch,
        "best_epoch_validation_rmse_k": best_training_rmse,
        "validation_metrics": validation_metrics,
        "persistence_validation_metrics": persistence_metrics,
        "dynamic_history_steps": args.dynamic_history_steps,
        "dynamic_validation_metrics": dynamic_metrics,
        "persistence_dynamic_validation_metrics": persistence_dynamic_metrics,
        "device": str(device),
        "training_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    metrics_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "model_class": "LSTMRegressor",
            "input_size": len(RAW_FEATURE_COLS),
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "feature_columns": list(RAW_FEATURE_COLS),
            "lookback": args.lookback,
            "seed": args.seed,
            "target": TARGET_COL,
            "label": LABEL_COL,
            "input_scaler": {
                "mean": input_scaler.mean,
                "scale": input_scaler.scale,
            },
            "target_scaler": {
                "mean": target_scaler.mean,
                "scale": target_scaler.scale,
            },
            "validation_metrics": validation_metrics,
        },
        run_dir / "best_model.pt",
    )

    print(f"BEST_EPOCH={best_epoch}")
    print(f"FINAL_VALIDATION_RMSE_K={validation_metrics['rmse_k']:.10f}")
    print(f"FINAL_VALIDATION_MAE_K={validation_metrics['mae_k']:.10f}")
    if dynamic_metrics is not None:
        print(f"FINAL_DYNAMIC_VALIDATION_RMSE_K={dynamic_metrics['rmse_k']:.10f}")
    print(f"PERSISTENCE_VALIDATION_RMSE_K={persistence_metrics['rmse_k']:.10f}")
    print(f"MODEL_SAVED={run_dir / 'best_model.pt'}")
    print(f"RESULTS_SAVED={run_dir}")


if __name__ == "__main__":
    main()
