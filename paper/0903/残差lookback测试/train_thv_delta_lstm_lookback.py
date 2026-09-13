#!/usr/bin/env python3
"""Leakage-safe residual LSTM lookback experiment for Thv(t+5).

The model predicts Delta_Thv_5 = Thv(t+5) - Thv(t).  A history LSTM sees
measurements through time t, while a separate control branch sees only the
known/planned valve sequence u(t), ..., u(t+4).  No future temperature,
pressure, flow, engineered feature, or future label enters the input.

Development-data contract:

* train_clean.pkl fits the model, all scalers, and the rapid-cooling threshold;
* 0617-ALL is removed before any fitted statistic is calculated;
* val_clean.pkl (Original 260501) is used for early stopping and lookback choice;
* test_clean.pkl / 0715-BACK is never opened by the development-data loader;
* every history/control window remains inside one contiguous file_id;
* all lookbacks use the same origins: 40 history rows and 5 control moves must
  be available, so sample counts and labels are identical across lookbacks;
* weighted Huber is fixed: rapid-cooling samples receive higher weight;
* best checkpoints are selected by rapid-cooling validation RMSE, while full
  validation RMSE remains a reported guardrail.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "processed_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

ALLOWED_LOOKBACKS = (10, 15, 20, 25, 30, 35, 40)
COMMON_ORIGIN_LOOKBACK = max(ALLOWED_LOOKBACKS)
TARGET_COL = "Thv"
LABEL_COL = "Future_Thv_5step"
PREDICT_STEPS = 5
CONTROL_HORIZON = 5
SAMPLE_PERIOD_SECONDS = 10.0

EXCLUDED_PARENT = "0617-ALL"
EXPECTED_TRAIN_PARENTS = {
    "0623-BACK",
    "0715-ALL",
    "251130",
    "251226",
    "260403",
    "260118",
}
EXPECTED_VALIDATION_PARENT = "260501"
EXPECTED_TEST_PARENT_IN_BUILD_CONFIG = "0715-BACK"

# History inputs are current/past raw measurements only.  The current target is
# observable at decision time and is therefore a valid history input.
HISTORY_FEATURE_COLS = (
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

# These are the only variables allowed beyond the forecast origin.  For a
# target at t+5, the five transition controls are u(t), ..., u(t+4).  This is
# the standard MPC/state-transition alignment; u(t+5) is simultaneous with the
# target point and is intentionally excluded.
FUTURE_CONTROL_COLS = (
    "CV8312",
    "CV8311",
    "CV8310",
    "CV8313",
    "CV8300",
    "CV8351",
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
        result = (array - self.mean) / self.scale
        return result.reshape(-1) if original_ndim == 1 else result

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        original_ndim = array.ndim
        if original_ndim == 1:
            array = array.reshape(-1, 1)
        result = array * self.scale + self.mean
        return result.reshape(-1) if original_ndim == 1 else result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Residual Thv(t+5) LSTM lookback test with a known future-control branch."
        )
    )
    parser.add_argument(
        "--lookback",
        type=int,
        required=True,
        choices=ALLOWED_LOOKBACKS,
        help="History length in 10-second samples.",
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
    parser.add_argument("--control-hidden-size", type=int, default=32)
    parser.add_argument("--fusion-hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--rapid-quantile",
        type=float,
        default=0.10,
        help="Training Delta quantile used to define rapid cooling.",
    )
    parser.add_argument(
        "--rapid-weight",
        type=float,
        default=3.0,
        help="Fixed Huber weight for rapid-cooling samples; normal weight is 1.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Huber transition point on the standardized Delta target.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be >= 1")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if min(args.hidden_size, args.num_layers, args.control_hidden_size, args.fusion_hidden_size) < 1:
        raise ValueError("All hidden sizes and --num-layers must be >= 1")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")
    if args.patience < 1 or args.min_delta < 0:
        raise ValueError("--patience must be >= 1 and --min-delta non-negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if not 0 < args.rapid_quantile < 0.5:
        raise ValueError("--rapid-quantile must be in (0, 0.5)")
    if args.rapid_weight <= 1:
        raise ValueError("--rapid-weight must be > 1")
    if args.huber_delta <= 0:
        raise ValueError("--huber-delta must be positive")


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
        raise ValueError("processed_data validation parent is not 260501")
    if config.get("validation_variant") != "Original only":
        raise ValueError("processed_data validation is not Original only")
    if config.get("test_parent") != EXPECTED_TEST_PARENT_IN_BUILD_CONFIG:
        raise ValueError("processed_data test parent is not 0715-BACK")
    if LABEL_COL not in set(config.get("future_labels", [])):
        raise ValueError(f"processed_data does not contain {LABEL_COL}")
    if config.get("future_features_used") is not False:
        raise ValueError("processed_data contract does not guarantee causal inputs")


def prepare_development_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Deliberately no test filename and no test-loading branch.
    train = load_dataframe(data_dir / "train_clean.pkl")
    val = load_dataframe(data_dir / "val_clean.pkl")

    required = {
        *HISTORY_FEATURE_COLS,
        *FUTURE_CONTROL_COLS,
        LABEL_COL,
        "file_id",
        "source_group",
        "source_variant",
        "parent_id",
        "source_row_index",
        "source_timestamp",
    }
    for name, frame in (("training", train), ("validation", val)):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{name} frame is missing columns: {sorted(missing)}")

    train = train.loc[train["parent_id"].astype(str) != EXCLUDED_PARENT].copy()
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)

    train_parents = set(map(str, train["parent_id"].unique()))
    if train_parents != EXPECTED_TRAIN_PARENTS:
        raise ValueError(
            "Unexpected training parents after removing 0617-ALL: "
            f"expected={sorted(EXPECTED_TRAIN_PARENTS)}, actual={sorted(train_parents)}"
        )
    if set(map(str, val["parent_id"].unique())) != {EXPECTED_VALIDATION_PARENT}:
        raise ValueError("Validation is not exclusively parent 260501")
    if set(map(str, val["source_variant"].unique())) != {"Original"}:
        raise ValueError("Validation contains augmented samples")
    if set(map(str, train["source_group"].unique())) & set(
        map(str, val["source_group"].unique())
    ):
        raise ValueError("Training and validation share a source_group")

    if LABEL_COL in HISTORY_FEATURE_COLS or LABEL_COL in FUTURE_CONTROL_COLS:
        raise AssertionError("Future label entered an input list")
    future_named_inputs = [
        column
        for column in (*HISTORY_FEATURE_COLS, *FUTURE_CONTROL_COLS)
        if column.startswith("Future_")
    ]
    if future_named_inputs:
        raise AssertionError(f"Forbidden future input columns: {future_named_inputs}")

    selected = list(dict.fromkeys((*HISTORY_FEATURE_COLS, *FUTURE_CONTROL_COLS, LABEL_COL)))
    for name, frame in (("training", train), ("validation", val)):
        numeric = frame[selected].to_numpy(dtype=np.float64)
        finite_by_column = np.isfinite(numeric).all(axis=0)
        if not finite_by_column.all():
            bad = np.asarray(selected, dtype=object)[~finite_by_column].tolist()
            raise ValueError(f"{name} contains NaN/inf in selected columns: {bad}")
        audit_stored_future_label_alignment(frame, name)
    return train, val


def audit_stored_future_label_alignment(
    frame: pd.DataFrame,
    split_name: str,
) -> dict[str, float | int | str]:
    """Prove that the stored label is Thv exactly five source rows ahead.

    Tail rows whose t+5 state is outside the retained processed segment cannot
    be reconstructed from the table and are skipped.  All interior rows are
    checked exactly; data_fil.py separately guarantees that stored tail labels
    were formed before segmentation from the same parent curve.
    """
    checked = 0
    max_absolute_error = 0.0
    for file_id, indices in frame.groupby("file_id", sort=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) <= PREDICT_STEPS:
            continue
        part = frame.iloc[idx]
        source_rows = part["source_row_index"].to_numpy(dtype=np.int64)
        aligned = (
            source_rows[PREDICT_STEPS:] - source_rows[:-PREDICT_STEPS]
            == PREDICT_STEPS
        )
        if not aligned.all():
            raise AssertionError(
                f"{split_name} file_id={file_id} is not source-row aligned at t+5"
            )
        stored = part[LABEL_COL].to_numpy(dtype=np.float64)[:-PREDICT_STEPS]
        expected = part[TARGET_COL].to_numpy(dtype=np.float64)[PREDICT_STEPS:]
        absolute_error = np.abs(stored - expected)
        if not np.allclose(stored, expected, rtol=0.0, atol=1e-10):
            raise AssertionError(
                f"{split_name} file_id={file_id}: {LABEL_COL} is not "
                f"{TARGET_COL}(t+{PREDICT_STEPS})"
            )
        checked += int(len(stored))
        if len(absolute_error):
            max_absolute_error = max(
                max_absolute_error, float(absolute_error.max())
            )
    if checked == 0:
        raise ValueError(f"No t+{PREDICT_STEPS} labels were auditable in {split_name}")
    return {
        "split": split_name,
        "target": TARGET_COL,
        "stored_label": LABEL_COL,
        "forecast_steps": PREDICT_STEPS,
        "checked_interior_rows": checked,
        "max_absolute_alignment_error_k": max_absolute_error,
    }


def _assert_source_rows_contiguous(part: pd.DataFrame, file_id: str) -> None:
    rows = pd.to_numeric(part["source_row_index"], errors="coerce").to_numpy()
    if not np.isfinite(rows).all():
        raise ValueError(f"file_id={file_id} has invalid source_row_index")
    if len(rows) > 1 and not np.all(np.diff(rows) == 1):
        raise ValueError(
            f"file_id={file_id} crosses a removed row; data_fil.py should split it"
        )


def build_common_window_ends(
    frame: pd.DataFrame,
    common_origin_lookback: int = COMMON_ORIGIN_LOOKBACK,
) -> np.ndarray:
    """Origins with common history and complete u(t:t+4), per file_id.

    ``common_origin_lookback`` is explicit so a later lookback study can use a
    longer common-history requirement without silently changing the forecast
    origins used by the original 10--40 comparison.
    """
    if common_origin_lookback < 1:
        raise ValueError("common_origin_lookback must be >= 1")
    all_ends: list[np.ndarray] = []
    future_tail = CONTROL_HORIZON - 1
    for file_id, indices in frame.groupby("file_id", sort=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) == 0:
            continue
        if not np.all(np.diff(idx) == 1):
            raise ValueError(f"Rows for file_id={file_id} are not contiguous")
        part = frame.iloc[idx]
        if part["source_group"].nunique() != 1:
            raise ValueError(f"file_id={file_id} maps to multiple source groups")
        _assert_source_rows_contiguous(part, str(file_id))
        first_end = idx[0] + common_origin_lookback - 1
        last_end = idx[-1] - future_tail
        if first_end <= last_end:
            all_ends.append(np.arange(first_end, last_end + 1, dtype=np.int64))
    if not all_ends:
        raise ValueError("No segment supports the common history/control origin rule")
    result = np.concatenate(all_ends)
    if len(result) != len(np.unique(result)):
        raise AssertionError("Duplicate forecast origins were created")
    return result


def assert_window_boundaries(frame: pd.DataFrame, ends: np.ndarray, lookback: int) -> None:
    starts = ends - lookback + 1
    control_last = ends + CONTROL_HORIZON - 1
    if starts.min() < 0 or control_last.max() >= len(frame):
        raise IndexError("History or control window is outside the DataFrame")
    file_ids = frame["file_id"].astype(str).to_numpy()
    groups = frame["source_group"].astype(str).to_numpy()
    for values, label in ((file_ids, "file_id"), (groups, "source_group")):
        if not np.array_equal(values[starts], values[ends]):
            raise AssertionError(f"A history window crosses a {label} boundary")
        if not np.array_equal(values[ends], values[control_last]):
            raise AssertionError(f"A future-control window crosses a {label} boundary")
    source_rows = frame["source_row_index"].to_numpy(dtype=np.int64)
    if not np.all(source_rows[control_last] - source_rows[ends] == CONTROL_HORIZON - 1):
        raise AssertionError("A future-control window crosses a removed source row")


class ResidualControlWindowDataset(Dataset):
    def __init__(
        self,
        history_scaled: np.ndarray,
        controls_scaled: np.ndarray,
        delta_scaled: np.ndarray,
        sample_weights: np.ndarray,
        ends: np.ndarray,
        lookback: int,
    ):
        self.history = np.asarray(history_scaled, dtype=np.float32)
        self.controls = np.asarray(controls_scaled, dtype=np.float32)
        self.delta = np.asarray(delta_scaled, dtype=np.float32).reshape(-1)
        self.weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        self.ends = np.asarray(ends, dtype=np.int64)
        self.lookback = int(lookback)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, index: int):
        end = int(self.ends[index])
        start = end - self.lookback + 1
        control_stop = end + CONTROL_HORIZON
        return (
            torch.from_numpy(self.history[start : end + 1]),
            torch.from_numpy(self.controls[end:control_stop]),
            torch.tensor(self.delta[end], dtype=torch.float32),
            torch.tensor(self.weights[end], dtype=torch.float32),
            torch.tensor(end, dtype=torch.int64),
        )


class ResidualControlLSTM(nn.Module):
    """History LSTM plus a small ordered future-control MLP branch."""

    def __init__(
        self,
        history_input_size: int,
        control_input_size: int,
        hidden_size: int,
        num_layers: int,
        control_hidden_size: int,
        fusion_hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.history_lstm = nn.LSTM(
            input_size=history_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.control_encoder = nn.Sequential(
            nn.Linear(CONTROL_HORIZON * control_input_size, control_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + control_hidden_size, fusion_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, 1),
        )

    def forward(self, history: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        history_output, _ = self.history_lstm(history)
        history_state = history_output[:, -1, :]
        control_state = self.control_encoder(controls.flatten(start_dim=1))
        return self.fusion(torch.cat((history_state, control_state), dim=1)).squeeze(-1)


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
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


def weighted_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    huber_delta: float,
) -> torch.Tensor:
    per_sample = F.huber_loss(
        prediction,
        target,
        reduction="none",
        delta=huber_delta,
    )
    return torch.sum(weights * per_sample) / torch.clamp(weights.sum(), min=1e-12)


def inverse_delta(values: np.ndarray, scaler: Standardizer) -> np.ndarray:
    return scaler.inverse_transform(values).reshape(-1)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(predicted) - np.asarray(actual)))))


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scaler: Standardizer,
    huber_delta: float,
) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    scaled_loss_numerator = 0.0
    weight_sum = 0.0
    model.eval()
    with torch.no_grad():
        for history, controls, target, weight, _ in loader:
            history = history.to(device)
            controls = controls.to(device)
            target_device = target.to(device)
            weight_device = weight.to(device)
            prediction = model(history, controls)
            per_sample = F.huber_loss(
                prediction,
                target_device,
                reduction="none",
                delta=huber_delta,
            )
            scaled_loss_numerator += float(torch.sum(weight_device * per_sample).item())
            weight_sum += float(weight_device.sum().item())
            predictions.append(prediction.cpu().numpy())
            targets.append(target.numpy())
            weights.append(weight.numpy())

    predicted_scaled = np.concatenate(predictions)
    actual_scaled = np.concatenate(targets)
    sample_weights = np.concatenate(weights)
    predicted_k = inverse_delta(predicted_scaled, target_scaler)
    actual_k = inverse_delta(actual_scaled, target_scaler)
    rapid_mask = sample_weights > 1.0
    if not rapid_mask.any():
        raise ValueError("Rapid-cooling subset is empty")
    return {
        "weighted_huber_scaled": scaled_loss_numerator / max(weight_sum, 1e-12),
        "rmse_k": rmse(actual_k, predicted_k),
        "rapid_rmse_k": rmse(actual_k[rapid_mask], predicted_k[rapid_mask]),
        "rapid_count": int(rapid_mask.sum()),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
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
    best_rapid_rmse = float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for history, controls, target, weight, _ in train_loader:
            history = history.to(device)
            controls = controls.to(device)
            target = target.to(device)
            weight = weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history, controls)
            loss = weighted_huber(
                prediction,
                target,
                weight,
                huber_delta=args.huber_delta,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_stats = evaluate_epoch(
            model,
            train_eval_loader,
            device,
            target_scaler,
            args.huber_delta,
        )
        val_stats = evaluate_epoch(
            model,
            val_loader,
            device,
            target_scaler,
            args.huber_delta,
        )
        scheduler.step(val_stats["rapid_rmse_k"])
        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_weighted_huber_scaled": train_stats["weighted_huber_scaled"],
            "train_rmse_k": train_stats["rmse_k"],
            "train_rapid_rmse_k": train_stats["rapid_rmse_k"],
            "validation_weighted_huber_scaled": val_stats["weighted_huber_scaled"],
            "validation_rmse_k": val_stats["rmse_k"],
            "validation_rapid_rmse_k": val_stats["rapid_rmse_k"],
            "learning_rate": learning_rate,
        }
        history_rows.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_rmse_k={train_stats['rmse_k']:.8f} | "
            f"train_rapid_rmse_k={train_stats['rapid_rmse_k']:.8f} | "
            f"val_rmse_k={val_stats['rmse_k']:.8f} | "
            f"val_rapid_rmse_k={val_stats['rapid_rmse_k']:.8f} | "
            f"lr={learning_rate:.3e}"
        )

        monitored = val_stats["rapid_rmse_k"]
        if monitored < best_rapid_rmse - args.min_delta:
            best_rapid_rmse = monitored
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
                    f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, "
                    f"best_val_rapid_rmse_k={best_rapid_rmse:.8f}"
                )
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    return best_state, history_rows, best_epoch, best_rapid_rmse


def predict_delta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scaler: Standardizer,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for history, controls, _, _, end in loader:
            prediction = model(history.to(device), controls.to(device))
            predictions.append(prediction.cpu().numpy())
            ends.append(end.numpy())
    return (
        inverse_delta(np.concatenate(predictions), target_scaler),
        np.concatenate(ends).astype(np.int64),
    )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    absolute_error = np.abs(error)
    denominator = float(np.sum(np.square(actual - actual.mean())))
    r2 = (
        float("nan")
        if denominator <= 0
        else 1.0 - float(np.sum(np.square(error))) / denominator
    )
    return {
        "mae_k": float(absolute_error.mean()),
        "rmse_k": float(np.sqrt(np.mean(np.square(error)))),
        "p95_absolute_error_k": float(np.percentile(absolute_error, 95)),
        "max_absolute_error_k": float(absolute_error.max()),
        "bias_k": float(error.mean()),
        "r2": r2,
        "n_windows": int(len(actual)),
    }


def direction_accuracy(actual_delta: np.ndarray, predicted_delta: np.ndarray) -> float:
    actual_sign = np.sign(np.asarray(actual_delta, dtype=np.float64))
    predicted_sign = np.sign(np.asarray(predicted_delta, dtype=np.float64))
    return float(np.mean(actual_sign == predicted_sign))


def save_training_plot(history: Sequence[dict[str, float]], path: Path) -> None:
    table = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(table["epoch"], table["train_rmse_k"], label="Train full RMSE")
    axes[0].plot(
        table["epoch"], table["validation_rmse_k"], label="Validation full RMSE"
    )
    axes[0].set_ylabel("Full RMSE (K)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        table["epoch"], table["train_rapid_rmse_k"], label="Train rapid RMSE"
    )
    axes[1].plot(
        table["epoch"],
        table["validation_rapid_rmse_k"],
        label="Validation rapid RMSE",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Rapid-cooling RMSE (K)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_validation_plots(
    actual_temperature: np.ndarray,
    predicted_temperature: np.ndarray,
    current_temperature: np.ndarray,
    actual_delta: np.ndarray,
    predicted_delta: np.ndarray,
    rapid_mask: np.ndarray,
    file_ids: np.ndarray,
    temperature_path: Path,
    delta_path: Path,
    title: str,
    predict_steps: int = PREDICT_STEPS,
) -> None:
    x = np.arange(len(actual_temperature))
    boundaries = np.flatnonzero(file_ids[1:] != file_ids[:-1]) + 1
    residual = predicted_temperature - actual_temperature

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(
        x,
        actual_temperature,
        color="black",
        linewidth=0.9,
        label=f"Measured Thv(t+{predict_steps})",
    )
    axes[0].plot(x, predicted_temperature, color="#D55E00", linewidth=0.8, label="Prediction")
    axes[0].plot(x, current_temperature, color="#999999", linewidth=0.6, alpha=0.7, label="Persistence")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(x, residual, color="#0072B2", linewidth=0.7)
    axes[1].scatter(x[rapid_mask], residual[rapid_mask], s=4, color="#CC79A7", label="Rapid cooling")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Common validation forecast origin")
    axes[1].set_ylabel("Residual (K)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    for boundary in boundaries:
        axes[0].axvline(boundary, color="gray", linewidth=0.5, alpha=0.35)
        axes[1].axvline(boundary, color="gray", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(temperature_path, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(x, actual_delta, color="black", linewidth=0.9, label="Measured Delta Thv")
    ax.plot(x, predicted_delta, color="#D55E00", linewidth=0.8, label="Predicted Delta Thv")
    ax.scatter(x[rapid_mask], actual_delta[rapid_mask], s=5, color="#CC79A7", label="Rapid-cooling target")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Common validation forecast origin")
    ax.set_ylabel(
        f"Delta Thv over {predict_steps * SAMPLE_PERIOD_SECONDS:g} s (K)"
    )
    ax.set_title(title + " | residual target")
    ax.legend()
    ax.grid(alpha=0.2)
    for boundary in boundaries:
        ax.axvline(boundary, color="gray", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(delta_path, dpi=220)
    plt.close(fig)


def save_scalers(
    path: Path,
    history_scaler: Standardizer,
    control_scaler: Standardizer,
    target_scaler: Standardizer,
    rapid_threshold_k: float,
    target_name: str = "Delta_Thv_5step",
) -> None:
    np.savez(
        path,
        history_feature_names=np.asarray(HISTORY_FEATURE_COLS, dtype=str),
        history_mean=history_scaler.mean,
        history_scale=history_scaler.scale,
        future_control_names=np.asarray(FUTURE_CONTROL_COLS, dtype=str),
        control_mean=control_scaler.mean,
        control_scale=control_scaler.scale,
        target_name=np.asarray([target_name], dtype=str),
        target_mean=target_scaler.mean,
        target_scale=target_scaler.scale,
        rapid_threshold_k=np.asarray([rapid_threshold_k], dtype=np.float64),
    )


def best_epoch_row(history: Sequence[dict[str, float]], best_epoch: int) -> dict[str, float]:
    for row in history:
        if int(row["epoch"]) == best_epoch:
            return dict(row)
    raise KeyError(f"Best epoch {best_epoch} is absent from training history")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    assert_build_config(args.data_dir)

    run_dir = args.output_dir / f"lookback_{args.lookback:02d}" / f"seed_{args.seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run already exists: {run_dir}. Pass --overwrite to replace it."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_frame, val_frame = prepare_development_frames(args.data_dir)
    train_ends = build_common_window_ends(train_frame)
    val_ends = build_common_window_ends(val_frame)
    assert_window_boundaries(train_frame, train_ends, args.lookback)
    assert_window_boundaries(val_frame, val_ends, args.lookback)

    train_current = train_frame[TARGET_COL].to_numpy(dtype=np.float64)
    val_current = val_frame[TARGET_COL].to_numpy(dtype=np.float64)
    train_future = train_frame[LABEL_COL].to_numpy(dtype=np.float64)
    val_future = val_frame[LABEL_COL].to_numpy(dtype=np.float64)
    train_delta = train_future - train_current
    val_delta = val_future - val_current

    rapid_threshold_k = float(np.quantile(train_delta[train_ends], args.rapid_quantile))
    train_weights = np.ones(len(train_frame), dtype=np.float32)
    val_weights = np.ones(len(val_frame), dtype=np.float32)
    train_weights[train_delta <= rapid_threshold_k] = args.rapid_weight
    val_weights[val_delta <= rapid_threshold_k] = args.rapid_weight

    history_scaler = Standardizer.fit(
        train_frame[list(HISTORY_FEATURE_COLS)].to_numpy(dtype=np.float64)
    )
    control_scaler = Standardizer.fit(
        train_frame[list(FUTURE_CONTROL_COLS)].to_numpy(dtype=np.float64)
    )
    target_scaler = Standardizer.fit(train_delta[train_ends])

    train_history = history_scaler.transform(
        train_frame[list(HISTORY_FEATURE_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    val_history = history_scaler.transform(
        val_frame[list(HISTORY_FEATURE_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    train_controls = control_scaler.transform(
        train_frame[list(FUTURE_CONTROL_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    val_controls = control_scaler.transform(
        val_frame[list(FUTURE_CONTROL_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    train_delta_scaled = target_scaler.transform(train_delta).astype(np.float32)
    val_delta_scaled = target_scaler.transform(val_delta).astype(np.float32)

    train_dataset = ResidualControlWindowDataset(
        train_history,
        train_controls,
        train_delta_scaled,
        train_weights,
        train_ends,
        args.lookback,
    )
    val_dataset = ResidualControlWindowDataset(
        val_history,
        val_controls,
        val_delta_scaled,
        val_weights,
        val_ends,
        args.lookback,
    )
    train_loader = make_loader(train_dataset, args, shuffle=True, seed_offset=0)
    train_eval_loader = make_loader(
        train_dataset, args, shuffle=False, seed_offset=1
    )
    val_loader = make_loader(val_dataset, args, shuffle=False, seed_offset=2)

    device = choose_device()
    model = ResidualControlLSTM(
        history_input_size=len(HISTORY_FEATURE_COLS),
        control_input_size=len(FUTURE_CONTROL_COLS),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        control_hidden_size=args.control_hidden_size,
        fusion_hidden_size=args.fusion_hidden_size,
        dropout=args.dropout,
    ).to(device)

    print(
        f"Experiment=Delta Thv(t+5) residual-control LSTM | "
        f"lookback={args.lookback} | seed={args.seed} | device={device}"
    )
    print(
        f"Known controls=u(t)..u(t+4), columns={list(FUTURE_CONTROL_COLS)}"
    )
    print(
        f"Rapid threshold (train-only q={args.rapid_quantile:.3f})="
        f"{rapid_threshold_k:.10f} K | rapid_weight={args.rapid_weight:.3f}"
    )
    print(
        f"Training parents={sorted(EXPECTED_TRAIN_PARENTS)} | "
        f"excluded={EXCLUDED_PARENT} | train_windows={len(train_dataset)} | "
        f"val_windows={len(val_dataset)}"
    )
    print(
        "Boundary contract: histories and future controls stay inside one "
        "contiguous file_id; all lookbacks use common 40-step origins."
    )

    best_state, history, best_epoch, best_rapid_rmse = train_model(
        model,
        train_loader,
        train_eval_loader,
        val_loader,
        target_scaler,
        args,
        device,
    )
    model.load_state_dict(best_state)
    predicted_delta, prediction_ends = predict_delta(
        model, val_loader, device, target_scaler
    )
    if not np.array_equal(prediction_ends, val_ends):
        raise AssertionError("Validation prediction order changed")

    actual_delta = val_delta[val_ends]
    current_temperature = val_current[val_ends]
    actual_temperature = val_future[val_ends]
    predicted_temperature = current_temperature + predicted_delta
    rapid_mask = actual_delta <= rapid_threshold_k
    if not rapid_mask.any():
        raise ValueError("Validation rapid-cooling subset is empty")

    validation_metrics = regression_metrics(actual_temperature, predicted_temperature)
    rapid_validation_metrics = regression_metrics(
        actual_temperature[rapid_mask], predicted_temperature[rapid_mask]
    )
    delta_validation_metrics = regression_metrics(actual_delta, predicted_delta)
    persistence_metrics = regression_metrics(actual_temperature, current_temperature)
    rapid_persistence_metrics = regression_metrics(
        actual_temperature[rapid_mask], current_temperature[rapid_mask]
    )
    best_row = best_epoch_row(history, best_epoch)

    prediction_data: dict[str, np.ndarray] = {
        "frame_row_index": prediction_ends,
        "file_id": val_frame.iloc[prediction_ends]["file_id"].to_numpy(),
        "source_group": val_frame.iloc[prediction_ends]["source_group"].to_numpy(),
        "source_row_index": val_frame.iloc[prediction_ends]["source_row_index"].to_numpy(),
        "source_timestamp": val_frame.iloc[prediction_ends]["source_timestamp"].to_numpy(),
        "current_Thv_k": current_temperature,
        "actual_delta_Thv_5step_k": actual_delta,
        "predicted_delta_Thv_5step_k": predicted_delta,
        "actual_Future_Thv_5step_k": actual_temperature,
        "predicted_Future_Thv_5step_k": predicted_temperature,
        "residual_k": predicted_temperature - actual_temperature,
        "rapid_cooling_mask": rapid_mask.astype(np.int8),
    }
    raw_val_controls = val_frame[list(FUTURE_CONTROL_COLS)].to_numpy(dtype=np.float64)
    for offset in range(CONTROL_HORIZON):
        rows = val_ends + offset
        for column_index, column in enumerate(FUTURE_CONTROL_COLS):
            prediction_data[f"control_{column}_t_plus_{offset}"] = raw_val_controls[
                rows, column_index
            ]
    pd.DataFrame(prediction_data).to_csv(
        run_dir / "validation_predictions.csv", index=False
    )
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    save_training_plot(history, run_dir / "training_curve.png")
    file_ids = val_frame.iloc[val_ends]["file_id"].astype(str).to_numpy()
    title = (
        f"Delta Thv(t+5) LSTM + known controls | "
        f"lookback={args.lookback} | seed={args.seed}"
    )
    save_validation_plots(
        actual_temperature,
        predicted_temperature,
        current_temperature,
        actual_delta,
        predicted_delta,
        rapid_mask,
        file_ids,
        run_dir / "validation_prediction.png",
        run_dir / "validation_delta.png",
        title,
    )
    save_scalers(
        run_dir / "scalers.npz",
        history_scaler,
        control_scaler,
        target_scaler,
        rapid_threshold_k,
    )

    metadata = {
        "experiment": "Thv_t_plus_5_delta_lstm_known_future_controls_lookback",
        "lookback": args.lookback,
        "lookback_seconds": args.lookback * SAMPLE_PERIOD_SECONDS,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "seed": args.seed,
        "target": TARGET_COL,
        "source_label": LABEL_COL,
        "trained_target": "Delta_Thv_5step = Future_Thv_5step - Thv",
        "prediction_reconstruction": "predicted_Thv_t_plus_5 = current_Thv + predicted_Delta",
        "predict_steps": PREDICT_STEPS,
        "predict_seconds": PREDICT_STEPS * SAMPLE_PERIOD_SECONDS,
        "history_feature_columns": list(HISTORY_FEATURE_COLS),
        "future_control_columns": list(FUTURE_CONTROL_COLS),
        "future_control_alignment": "u(t),u(t+1),u(t+2),u(t+3),u(t+4)",
        "future_noncontrol_measurements_used": False,
        "known_control_assumption": (
            "Validation is conditional on the recorded control trajectory; in MPC the "
            "same inputs are candidate/planned controls supplied by the optimizer."
        ),
        "loss": "fixed rapid-cooling weighted Huber on standardized Delta",
        "rapid_quantile": args.rapid_quantile,
        "rapid_weight": args.rapid_weight,
        "normal_weight": 1.0,
        "huber_delta_scaled": args.huber_delta,
        "rapid_threshold_k_train_only": rapid_threshold_k,
        "checkpoint_selection_metric": "validation_rapid_rmse_k",
        "excluded_parent": EXCLUDED_PARENT,
        "training_parents": sorted(EXPECTED_TRAIN_PARENTS),
        "validation_parent": EXPECTED_VALIDATION_PARENT,
        "validation_variant": "Original",
        "test_loaded": False,
        "window_boundary_key": "file_id",
        "train_rows_after_exclusion": int(len(train_frame)),
        "validation_rows": int(len(val_frame)),
        "train_windows": int(len(train_dataset)),
        "validation_windows": int(len(val_dataset)),
        "train_rapid_windows": int((train_delta[train_ends] <= rapid_threshold_k).sum()),
        "validation_rapid_windows": int(rapid_mask.sum()),
        "best_epoch": best_epoch,
        "best_epoch_validation_rapid_rmse_k": best_rapid_rmse,
        "best_epoch_history": best_row,
        "validation_metrics": validation_metrics,
        "rapid_validation_metrics": rapid_validation_metrics,
        "delta_validation_metrics": delta_validation_metrics,
        "persistence_validation_metrics": persistence_metrics,
        "rapid_persistence_validation_metrics": rapid_persistence_metrics,
        "validation_direction_accuracy": direction_accuracy(
            actual_delta, predicted_delta
        ),
        "rapid_validation_direction_accuracy": direction_accuracy(
            actual_delta[rapid_mask], predicted_delta[rapid_mask]
        ),
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
            "model_class": "ResidualControlLSTM",
            "history_input_size": len(HISTORY_FEATURE_COLS),
            "control_input_size": len(FUTURE_CONTROL_COLS),
            "control_horizon": CONTROL_HORIZON,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "control_hidden_size": args.control_hidden_size,
            "fusion_hidden_size": args.fusion_hidden_size,
            "dropout": args.dropout,
            "history_feature_columns": list(HISTORY_FEATURE_COLS),
            "future_control_columns": list(FUTURE_CONTROL_COLS),
            "future_control_alignment": "u(t)..u(t+4)",
            "lookback": args.lookback,
            "seed": args.seed,
            "target": TARGET_COL,
            "trained_target": "Delta_Thv_5step",
            "history_scaler": {
                "mean": history_scaler.mean,
                "scale": history_scaler.scale,
            },
            "control_scaler": {
                "mean": control_scaler.mean,
                "scale": control_scaler.scale,
            },
            "target_scaler": {
                "mean": target_scaler.mean,
                "scale": target_scaler.scale,
            },
            "rapid_threshold_k": rapid_threshold_k,
            "rapid_weight": args.rapid_weight,
            "validation_metrics": validation_metrics,
            "rapid_validation_metrics": rapid_validation_metrics,
        },
        run_dir / "best_model.pt",
    )

    print(f"BEST_EPOCH={best_epoch}")
    print(f"FINAL_TRAIN_RMSE_K={best_row['train_rmse_k']:.10f}")
    print(f"FINAL_TRAIN_RAPID_RMSE_K={best_row['train_rapid_rmse_k']:.10f}")
    print(f"FINAL_VALIDATION_RMSE_K={validation_metrics['rmse_k']:.10f}")
    print(
        f"FINAL_RAPID_VALIDATION_RMSE_K="
        f"{rapid_validation_metrics['rmse_k']:.10f}"
    )
    print(f"PERSISTENCE_VALIDATION_RMSE_K={persistence_metrics['rmse_k']:.10f}")
    print(
        f"PERSISTENCE_RAPID_VALIDATION_RMSE_K="
        f"{rapid_persistence_metrics['rmse_k']:.10f}"
    )
    print(f"MODEL_SAVED={run_dir / 'best_model.pt'}")
    print(f"RESULTS_SAVED={run_dir}")


if __name__ == "__main__":
    main()
