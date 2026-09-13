#!/usr/bin/env python3
"""Fair LSTM-versus-GRU comparison for residual Thv(t+h) prediction.

Only the recurrent cell used by the historical encoder changes.  Data,
forecast origins, Delta target, future-control branch, weighted Huber loss,
rapid-cooling threshold, early stopping, and all other hyperparameters remain
identical.  Architecture selection reads train_clean.pkl and val_clean.pkl.
The formal GRU wrapper may additionally load Original 0715-BACK from
test_full_clean.pkl only after training and checkpoint selection, for the fixed
source-row 0--7300 secondary diagnostic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from train_thv_delta_lstm_lookback import (
    ALLOWED_LOOKBACKS,
    COMMON_ORIGIN_LOOKBACK,
    EXCLUDED_PARENT,
    EXPECTED_TEST_PARENT_IN_BUILD_CONFIG,
    EXPECTED_TRAIN_PARENTS,
    EXPECTED_VALIDATION_PARENT,
    FUTURE_CONTROL_COLS,
    HISTORY_FEATURE_COLS,
    SAMPLE_PERIOD_SECONDS,
    TARGET_COL,
    Standardizer,
    best_epoch_row,
    choose_device,
    direction_accuracy,
    load_dataframe,
    make_loader,
    predict_delta,
    regression_metrics,
    save_scalers,
    save_training_plot,
    save_validation_plots,
    set_seed,
    train_model,
    validate_args as validate_shared_args,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "processed_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "rnn_compare_outputs"
RNN_TYPES = ("lstm", "gru")
ALLOWED_PREDICT_STEPS = (5, 10, 15, 20, 30)
DEFAULT_PREDICT_STEPS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LSTM and GRU encoders for Delta Thv(t+h)."
    )
    parser.add_argument("--model", required=True, choices=RNN_TYPES)
    parser.add_argument(
        "--lookback",
        type=int,
        required=True,
        choices=ALLOWED_LOOKBACKS,
        help="Use 20 and 40 for the initial short/long-history screen.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--predict-steps",
        type=int,
        default=DEFAULT_PREDICT_STEPS,
        choices=ALLOWED_PREDICT_STEPS,
        help=(
            "Forecast horizon h in 10-second samples. The model receives only "
            "measurements through t and controls u(t)..u(t+h-1)."
        ),
    )
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
    parser.add_argument("--rapid-quantile", type=float, default=0.10)
    parser.add_argument("--rapid-weight", type=float, default=3.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def label_column(predict_steps: int) -> str:
    return f"Future_Thv_{predict_steps}step"


def delta_name(predict_steps: int) -> str:
    return f"Delta_Thv_{predict_steps}step"


def assert_horizon_build_config(
    data_dir: Path,
    predict_steps: int,
    label_col: str,
) -> None:
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
    if predict_steps not in {int(value) for value in config.get("horizons", [])}:
        raise ValueError(
            f"processed_data does not contain horizon {predict_steps}; "
            "rerun data_fil.py --overwrite"
        )
    if label_col not in set(config.get("future_labels", [])):
        raise ValueError(f"processed_data does not contain {label_col}")
    if config.get("future_features_used") is not False:
        raise ValueError("processed_data contract does not guarantee causal inputs")


def audit_horizon_label_alignment(
    frame: pd.DataFrame,
    split_name: str,
    predict_steps: int,
    label_col: str,
) -> dict[str, float | int | str]:
    """Verify that the stored target is exactly Thv at t+h within each segment."""
    checked = 0
    max_absolute_error = 0.0
    for file_id, indices in frame.groupby("file_id", sort=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) <= predict_steps:
            continue
        part = frame.iloc[idx]
        source_rows = part["source_row_index"].to_numpy(dtype=np.int64)
        if not np.all(
            source_rows[predict_steps:] - source_rows[:-predict_steps]
            == predict_steps
        ):
            raise AssertionError(
                f"{split_name} file_id={file_id} is not source-row aligned at "
                f"t+{predict_steps}"
            )
        stored = part[label_col].to_numpy(dtype=np.float64)[:-predict_steps]
        expected = part[TARGET_COL].to_numpy(dtype=np.float64)[predict_steps:]
        absolute_error = np.abs(stored - expected)
        if not np.allclose(stored, expected, rtol=0.0, atol=1e-10):
            raise AssertionError(
                f"{split_name} file_id={file_id}: {label_col} is not "
                f"{TARGET_COL}(t+{predict_steps})"
            )
        checked += int(len(stored))
        if len(absolute_error):
            max_absolute_error = max(
                max_absolute_error, float(absolute_error.max())
            )
    if checked == 0:
        raise ValueError(
            f"No t+{predict_steps} labels were auditable in {split_name}"
        )
    return {
        "split": split_name,
        "target": TARGET_COL,
        "stored_label": label_col,
        "forecast_steps": predict_steps,
        "checked_interior_rows": checked,
        "max_absolute_alignment_error_k": max_absolute_error,
    }


def prepare_horizon_development_frames(
    data_dir: Path,
    predict_steps: int,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/validation only; the test set remains unopened here."""
    train = load_dataframe(data_dir / "train_clean.pkl")
    val = load_dataframe(data_dir / "val_clean.pkl")
    required = {
        *HISTORY_FEATURE_COLS,
        *FUTURE_CONTROL_COLS,
        label_col,
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
            f"expected={sorted(EXPECTED_TRAIN_PARENTS)}, "
            f"actual={sorted(train_parents)}"
        )
    if set(map(str, val["parent_id"].unique())) != {EXPECTED_VALIDATION_PARENT}:
        raise ValueError("Validation is not exclusively parent 260501")
    if set(map(str, val["source_variant"].unique())) != {"Original"}:
        raise ValueError("Validation contains augmented samples")
    if set(map(str, train["source_group"].unique())) & set(
        map(str, val["source_group"].unique())
    ):
        raise ValueError("Training and validation share a source_group")
    if label_col in HISTORY_FEATURE_COLS or label_col in FUTURE_CONTROL_COLS:
        raise AssertionError("Future label entered an input list")
    future_named_inputs = [
        column
        for column in (*HISTORY_FEATURE_COLS, *FUTURE_CONTROL_COLS)
        if column.startswith("Future_")
    ]
    if future_named_inputs:
        raise AssertionError(f"Forbidden future input columns: {future_named_inputs}")
    selected = list(
        dict.fromkeys((*HISTORY_FEATURE_COLS, *FUTURE_CONTROL_COLS, label_col))
    )
    for name, frame in (("training", train), ("validation", val)):
        numeric = frame[selected].to_numpy(dtype=np.float64)
        finite_by_column = np.isfinite(numeric).all(axis=0)
        if not finite_by_column.all():
            bad = np.asarray(selected, dtype=object)[~finite_by_column].tolist()
            raise ValueError(f"{name} contains NaN/inf in selected columns: {bad}")
        audit_horizon_label_alignment(
            frame, name, predict_steps, label_col
        )
    return train, val


def _assert_source_rows_contiguous(part: pd.DataFrame, file_id: str) -> None:
    rows = pd.to_numeric(part["source_row_index"], errors="coerce").to_numpy()
    if not np.isfinite(rows).all():
        raise ValueError(f"file_id={file_id} has invalid source_row_index")
    if len(rows) > 1 and not np.all(np.diff(rows) == 1):
        raise ValueError(f"file_id={file_id} crosses a removed source row")


def build_horizon_window_ends(
    frame: pd.DataFrame,
    common_origin_lookback: int,
    predict_steps: int,
) -> np.ndarray:
    """Origins with common history and complete u(t)..u(t+h-1)."""
    if common_origin_lookback < 1 or predict_steps < 1:
        raise ValueError("common_origin_lookback and predict_steps must be >= 1")
    all_ends: list[np.ndarray] = []
    future_tail = predict_steps - 1
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
        raise ValueError("No segment supports the history/control origin rule")
    result = np.concatenate(all_ends)
    if len(result) != len(np.unique(result)):
        raise AssertionError("Duplicate forecast origins were created")
    return result


def assert_horizon_window_boundaries(
    frame: pd.DataFrame,
    ends: np.ndarray,
    lookback: int,
    predict_steps: int,
) -> None:
    starts = ends - lookback + 1
    control_last = ends + predict_steps - 1
    if starts.min() < 0 or control_last.max() >= len(frame):
        raise IndexError("History or control window is outside the DataFrame")
    file_ids = frame["file_id"].astype(str).to_numpy()
    groups = frame["source_group"].astype(str).to_numpy()
    for values, name in ((file_ids, "file_id"), (groups, "source_group")):
        if not np.array_equal(values[starts], values[ends]):
            raise AssertionError(f"A history window crosses a {name} boundary")
        if not np.array_equal(values[ends], values[control_last]):
            raise AssertionError(f"A future-control window crosses a {name} boundary")
    source_rows = frame["source_row_index"].to_numpy(dtype=np.int64)
    if not np.all(
        source_rows[control_last] - source_rows[ends] == predict_steps - 1
    ):
        raise AssertionError("A future-control window crosses a removed source row")


class HorizonControlWindowDataset(Dataset):
    def __init__(
        self,
        history_scaled: np.ndarray,
        controls_scaled: np.ndarray,
        delta_scaled: np.ndarray,
        sample_weights: np.ndarray,
        ends: np.ndarray,
        lookback: int,
        predict_steps: int,
    ):
        self.history = np.asarray(history_scaled, dtype=np.float32)
        self.controls = np.asarray(controls_scaled, dtype=np.float32)
        self.delta = np.asarray(delta_scaled, dtype=np.float32).reshape(-1)
        self.weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        self.ends = np.asarray(ends, dtype=np.int64)
        self.lookback = int(lookback)
        self.predict_steps = int(predict_steps)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, index: int):
        end = int(self.ends[index])
        start = end - self.lookback + 1
        control_stop = end + self.predict_steps
        return (
            torch.from_numpy(self.history[start : end + 1]),
            torch.from_numpy(self.controls[end:control_stop]),
            torch.tensor(self.delta[end], dtype=torch.float32),
            torch.tensor(self.weights[end], dtype=torch.float32),
            torch.tensor(end, dtype=torch.int64),
        )


class ResidualControlRNN(nn.Module):
    """LSTM or GRU history encoder with an identical future-control branch."""

    def __init__(
        self,
        rnn_type: str,
        history_input_size: int,
        control_input_size: int,
        hidden_size: int,
        num_layers: int,
        control_horizon: int,
        control_hidden_size: int,
        fusion_hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        if rnn_type not in RNN_TYPES:
            raise ValueError(f"Unsupported recurrent type: {rnn_type}")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        recurrent_class = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn_type = rnn_type
        self.control_horizon = int(control_horizon)
        self.history_rnn = recurrent_class(
            input_size=history_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.control_encoder = nn.Sequential(
            nn.Linear(
                self.control_horizon * control_input_size,
                control_hidden_size,
            ),
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
        history_output, _ = self.history_rnn(history)
        history_state = history_output[:, -1, :]
        control_state = self.control_encoder(controls.flatten(start_dim=1))
        return self.fusion(torch.cat((history_state, control_state), dim=1)).squeeze(-1)


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def synchronize_device(device: torch.device) -> None:
    """Make wall-clock timing comparable on asynchronous CUDA/MPS devices."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _normalization_rows(
    branch: str,
    feature_names: tuple[str, ...],
    train_values: np.ndarray,
    validation_values: np.ndarray,
    scaler: Standardizer,
) -> list[dict[str, float | int | str]]:
    """Describe train-only per-variable standardization in a readable CSV."""
    train_array = np.asarray(train_values, dtype=np.float64)
    validation_array = np.asarray(validation_values, dtype=np.float64)
    if train_array.ndim == 1:
        train_array = train_array.reshape(-1, 1)
    if validation_array.ndim == 1:
        validation_array = validation_array.reshape(-1, 1)
    train_scaled = scaler.transform(train_array)
    validation_scaled = scaler.transform(validation_array)
    rows: list[dict[str, float | int | str]] = []
    for index, feature in enumerate(feature_names):
        train_column = train_array[:, index]
        validation_column = validation_array[:, index]
        rows.append(
            {
                "branch": branch,
                "feature": feature,
                "scaler": "per_feature_standardization",
                "fit_source": "train_clean_after_excluding_0617_only",
                "train_count": int(len(train_column)),
                "train_min": float(train_column.min()),
                "train_max": float(train_column.max()),
                "train_mean": float(train_column.mean()),
                "train_std_population": float(train_column.std(ddof=0)),
                "scaler_mean": float(scaler.mean[index]),
                "scaler_scale": float(scaler.scale[index]),
                "train_scaled_min": float(train_scaled[:, index].min()),
                "train_scaled_max": float(train_scaled[:, index].max()),
                "validation_count": int(len(validation_column)),
                "validation_min": float(validation_column.min()),
                "validation_max": float(validation_column.max()),
                "validation_mean": float(validation_column.mean()),
                "validation_std_population": float(validation_column.std(ddof=0)),
                "validation_scaled_min": float(validation_scaled[:, index].min()),
                "validation_scaled_max": float(validation_scaled[:, index].max()),
                "validation_below_train_min_count": int(
                    np.sum(validation_column < train_column.min())
                ),
                "validation_above_train_max_count": int(
                    np.sum(validation_column > train_column.max())
                ),
            }
        )
    return rows


def build_normalization_report(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    train_delta_at_origins: np.ndarray,
    validation_delta_at_origins: np.ndarray,
    history_scaler: Standardizer,
    control_scaler: Standardizer,
    target_scaler: Standardizer,
    predict_steps: int,
) -> pd.DataFrame:
    rows = _normalization_rows(
        "history",
        tuple(HISTORY_FEATURE_COLS),
        train_frame[list(HISTORY_FEATURE_COLS)].to_numpy(dtype=np.float64),
        validation_frame[list(HISTORY_FEATURE_COLS)].to_numpy(dtype=np.float64),
        history_scaler,
    )
    rows.extend(
        _normalization_rows(
            "future_control",
            tuple(FUTURE_CONTROL_COLS),
            train_frame[list(FUTURE_CONTROL_COLS)].to_numpy(dtype=np.float64),
            validation_frame[list(FUTURE_CONTROL_COLS)].to_numpy(dtype=np.float64),
            control_scaler,
        )
    )
    rows.extend(
        _normalization_rows(
            "target",
            (delta_name(predict_steps),),
            train_delta_at_origins,
            validation_delta_at_origins,
            target_scaler,
        )
    )
    return pd.DataFrame(rows)


def build_segment_metrics(
    predictions: pd.DataFrame,
    predict_steps: int,
) -> pd.DataFrame:
    """Retain file-level errors so one short transition cannot stay hidden."""
    rows: list[dict[str, float | int | str]] = []
    for file_id, part in predictions.groupby("file_id", sort=False):
        actual = part[
            f"actual_Future_Thv_{predict_steps}step_k"
        ].to_numpy(dtype=np.float64)
        predicted = part[
            f"predicted_Future_Thv_{predict_steps}step_k"
        ].to_numpy(dtype=np.float64)
        current = part["current_Thv_k"].to_numpy(dtype=np.float64)
        rapid_mask = part["rapid_cooling_mask"].to_numpy(dtype=np.int8).astype(bool)
        full = regression_metrics(actual, predicted)
        persistence = regression_metrics(actual, current)
        row: dict[str, float | int | str] = {
            "file_id": str(file_id),
            "n_windows": int(len(part)),
            "rapid_windows": int(rapid_mask.sum()),
            "validation_rmse_k": float(full["rmse_k"]),
            "validation_mae_k": float(full["mae_k"]),
            "validation_p95_absolute_error_k": float(full["p95_absolute_error_k"]),
            "validation_max_absolute_error_k": float(full["max_absolute_error_k"]),
            "persistence_rmse_k": float(persistence["rmse_k"]),
        }
        if rapid_mask.any():
            rapid = regression_metrics(actual[rapid_mask], predicted[rapid_mask])
            rapid_persistence = regression_metrics(
                actual[rapid_mask], current[rapid_mask]
            )
            row.update(
                {
                    "rapid_validation_rmse_k": float(rapid["rmse_k"]),
                    "rapid_validation_mae_k": float(rapid["mae_k"]),
                    "rapid_validation_p95_absolute_error_k": float(
                        rapid["p95_absolute_error_k"]
                    ),
                    "rapid_validation_max_absolute_error_k": float(
                        rapid["max_absolute_error_k"]
                    ),
                    "rapid_persistence_rmse_k": float(
                        rapid_persistence["rmse_k"]
                    ),
                }
            )
        else:
            row.update(
                {
                    "rapid_validation_rmse_k": float("nan"),
                    "rapid_validation_mae_k": float("nan"),
                    "rapid_validation_p95_absolute_error_k": float("nan"),
                    "rapid_validation_max_absolute_error_k": float("nan"),
                    "rapid_persistence_rmse_k": float("nan"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_full_test_frame(
    data_dir: Path,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    predict_steps: int,
    label_col: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Load Original 0715-BACK only after model fitting/early stopping is complete."""
    test = load_dataframe(data_dir / "test_full_clean.pkl")
    required = {
        *HISTORY_FEATURE_COLS,
        *FUTURE_CONTROL_COLS,
        label_col,
        "file_id",
        "source_group",
        "source_variant",
        "parent_id",
        "source_row_index",
        "source_timestamp",
    }
    missing = required - set(test.columns)
    if missing:
        raise KeyError(f"full test frame is missing columns: {sorted(missing)}")
    test = test.reset_index(drop=True)
    if set(map(str, test["parent_id"].unique())) != {
        EXPECTED_TEST_PARENT_IN_BUILD_CONFIG
    }:
        raise ValueError(
            "Full test is not exclusively parent "
            f"{EXPECTED_TEST_PARENT_IN_BUILD_CONFIG}"
        )
    if set(map(str, test["source_variant"].unique())) != {"Original"}:
        raise ValueError("Full test contains augmented samples")
    train_groups = set(map(str, train_frame["source_group"].unique()))
    validation_groups = set(map(str, validation_frame["source_group"].unique()))
    test_groups = set(map(str, test["source_group"].unique()))
    if train_groups & test_groups or validation_groups & test_groups:
        raise ValueError("Full test shares a source_group with train/validation")
    selected = list(
        dict.fromkeys((*HISTORY_FEATURE_COLS, *FUTURE_CONTROL_COLS, label_col))
    )
    numeric = test[selected].to_numpy(dtype=np.float64)
    finite_by_column = np.isfinite(numeric).all(axis=0)
    if not finite_by_column.all():
        bad = np.asarray(selected, dtype=object)[~finite_by_column].tolist()
        raise ValueError(f"Full test contains NaN/inf in selected columns: {bad}")
    label_audit = audit_horizon_label_alignment(
        test,
        "test_0715-BACK_full",
        predict_steps,
        label_col,
    )
    return test, label_audit


def evaluate_fixed_test_window(
    model: nn.Module,
    test_frame: pd.DataFrame,
    history_scaler: Standardizer,
    control_scaler: Standardizer,
    target_scaler: Standardizer,
    rapid_threshold_k: float,
    common_origin_lookback: int,
    test_window_start: int,
    test_window_end: int,
    predict_steps: int,
    label_col: str,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
) -> dict[str, object]:
    """Evaluate source-row 0..7300 without using it for optimization."""
    if test_window_start < 0 or test_window_end < test_window_start:
        raise ValueError("Invalid fixed test source-row window")
    all_ends = build_horizon_window_ends(
        test_frame, common_origin_lookback, predict_steps
    )
    assert_horizon_window_boundaries(
        test_frame, all_ends, args.lookback, predict_steps
    )
    source_rows = pd.to_numeric(
        test_frame.iloc[all_ends]["source_row_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    window_mask = (source_rows >= test_window_start) & (
        source_rows <= test_window_end
    )
    test_ends = all_ends[window_mask]
    if len(test_ends) == 0:
        raise ValueError(
            f"No valid test origins in source rows {test_window_start}.."
            f"{test_window_end}"
        )
    assert_horizon_window_boundaries(
        test_frame, test_ends, args.lookback, predict_steps
    )

    current = test_frame[TARGET_COL].to_numpy(dtype=np.float64)
    future = test_frame[label_col].to_numpy(dtype=np.float64)
    delta = future - current
    sample_weights = np.ones(len(test_frame), dtype=np.float32)
    sample_weights[delta <= rapid_threshold_k] = args.rapid_weight

    history_scaled = history_scaler.transform(
        test_frame[list(HISTORY_FEATURE_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    controls_scaled = control_scaler.transform(
        test_frame[list(FUTURE_CONTROL_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    delta_scaled = target_scaler.transform(delta).astype(np.float32)
    dataset = HorizonControlWindowDataset(
        history_scaled,
        controls_scaled,
        delta_scaled,
        sample_weights,
        test_ends,
        args.lookback,
        predict_steps,
    )
    loader = make_loader(dataset, args, shuffle=False, seed_offset=3)
    predicted_delta, prediction_ends = predict_delta(
        model, loader, device, target_scaler
    )
    if not np.array_equal(prediction_ends, test_ends):
        raise AssertionError("Fixed test prediction order changed")

    actual_delta = delta[test_ends]
    current_temperature = current[test_ends]
    actual_temperature = future[test_ends]
    predicted_temperature = current_temperature + predicted_delta
    rapid_mask = actual_delta <= rapid_threshold_k
    if not rapid_mask.any():
        raise ValueError("Fixed test window contains no train-threshold rapid samples")

    window_metrics = regression_metrics(actual_temperature, predicted_temperature)
    rapid_metrics = regression_metrics(
        actual_temperature[rapid_mask], predicted_temperature[rapid_mask]
    )
    persistence_metrics = regression_metrics(
        actual_temperature, current_temperature
    )
    rapid_persistence_metrics = regression_metrics(
        actual_temperature[rapid_mask], current_temperature[rapid_mask]
    )
    selected_rows = test_frame.iloc[test_ends]
    table = pd.DataFrame(
        {
            "test_window_index": np.arange(len(test_ends), dtype=np.int64),
            "frame_row_index": test_ends,
            "file_id": selected_rows["file_id"].to_numpy(),
            "source_row_index": selected_rows["source_row_index"].to_numpy(),
            "source_timestamp": selected_rows["source_timestamp"].to_numpy(),
            "current_Thv_k": current_temperature,
            f"actual_delta_Thv_{predict_steps}step_k": actual_delta,
            f"predicted_delta_Thv_{predict_steps}step_k": predicted_delta,
            f"actual_Future_Thv_{predict_steps}step_k": actual_temperature,
            f"predicted_Future_Thv_{predict_steps}step_k": predicted_temperature,
            "residual_k": predicted_temperature - actual_temperature,
            "rapid_cooling_mask": rapid_mask.astype(np.int8),
        }
    )
    test_prefix = (
        f"test_0715-BACK_h{predict_steps:02d}_window_"
        f"{test_window_start}_{test_window_end}"
    )
    table.to_csv(run_dir / f"{test_prefix}_predictions.csv", index=False)
    save_validation_plots(
        actual_temperature,
        predicted_temperature,
        current_temperature,
        actual_delta,
        predicted_delta,
        rapid_mask,
        selected_rows["file_id"].astype(str).to_numpy(),
        run_dir / f"{test_prefix}_prediction.png",
        run_dir / f"{test_prefix}_delta.png",
        (
            "Held-out 0715-BACK fixed source-row window | "
            f"h={predict_steps} | lookback={args.lookback} | seed={args.seed}"
        ),
        predict_steps=predict_steps,
    )
    payload: dict[str, object] = {
        "test_parent": EXPECTED_TEST_PARENT_IN_BUILD_CONFIG,
        "test_variant": "Original",
        "test_file": "test_full_clean.pkl",
        "predict_steps": predict_steps,
        "predict_seconds": predict_steps * SAMPLE_PERIOD_SECONDS,
        "stored_label": label_col,
        "future_control_alignment": f"u(t)..u(t+{predict_steps - 1})",
        "window_coordinate": "inclusive source_row_index at forecast origin t",
        "window_start": int(test_window_start),
        "window_end": int(test_window_end),
        "window_metrics": window_metrics,
        "rapid_subset_definition": (
            f"Delta_Thv_{predict_steps} <= frozen training threshold "
            f"{rapid_threshold_k:.12g} K"
        ),
        "rapid_window_metrics": rapid_metrics,
        "persistence_window_metrics": persistence_metrics,
        "rapid_persistence_window_metrics": rapid_persistence_metrics,
        "window_direction_accuracy": direction_accuracy(
            actual_delta, predicted_delta
        ),
        "rapid_window_direction_accuracy": direction_accuracy(
            actual_delta[rapid_mask], predicted_delta[rapid_mask]
        ),
        "used_for_gradient_updates": False,
        "used_for_early_stopping": False,
        "used_for_checkpoint_selection": False,
        "intended_role": "secondary fixed-window test diagnostic",
    }
    (run_dir / f"{test_prefix}_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def run_experiment(args: argparse.Namespace) -> None:
    validate_shared_args(args)
    set_seed(args.seed)
    predict_steps = int(getattr(args, "predict_steps", DEFAULT_PREDICT_STEPS))
    if predict_steps not in ALLOWED_PREDICT_STEPS:
        raise ValueError(
            f"predict_steps must be one of {ALLOWED_PREDICT_STEPS}, "
            f"got {predict_steps}"
        )
    label_col = label_column(predict_steps)
    target_delta_name = delta_name(predict_steps)
    assert_horizon_build_config(args.data_dir, predict_steps, label_col)

    common_origin_lookback = int(
        getattr(args, "common_origin_lookback", COMMON_ORIGIN_LOOKBACK)
    )
    if args.lookback > common_origin_lookback:
        raise ValueError(
            f"lookback={args.lookback} exceeds common_origin_lookback="
            f"{common_origin_lookback}"
        )
    study_name = str(
        getattr(args, "study_name", "Thv_delta_lstm_gru_architecture_comparison")
    )
    comparison_policy = str(
        getattr(
            args,
            "comparison_policy",
            "same hidden size, data, origins, future-control branch, loss, seed, "
            "optimizer, and early-stopping metric",
        )
    )

    run_dir = (
        args.output_dir
        / f"horizon_{predict_steps:02d}"
        / args.model
        / f"lookback_{args.lookback:02d}"
        / f"seed_{args.seed}"
    )
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run already exists: {run_dir}. Pass --overwrite to replace it."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_frame, val_frame = prepare_horizon_development_frames(
        args.data_dir, predict_steps, label_col
    )
    train_label_audit = audit_horizon_label_alignment(
        train_frame, "training", predict_steps, label_col
    )
    validation_label_audit = audit_horizon_label_alignment(
        val_frame, "validation", predict_steps, label_col
    )
    train_ends = build_horizon_window_ends(
        train_frame, common_origin_lookback, predict_steps
    )
    val_ends = build_horizon_window_ends(
        val_frame, common_origin_lookback, predict_steps
    )
    assert_horizon_window_boundaries(
        train_frame, train_ends, args.lookback, predict_steps
    )
    assert_horizon_window_boundaries(
        val_frame, val_ends, args.lookback, predict_steps
    )

    train_current = train_frame[TARGET_COL].to_numpy(dtype=np.float64)
    val_current = val_frame[TARGET_COL].to_numpy(dtype=np.float64)
    train_future = train_frame[label_col].to_numpy(dtype=np.float64)
    val_future = val_frame[label_col].to_numpy(dtype=np.float64)
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

    normalization_report = build_normalization_report(
        train_frame,
        val_frame,
        train_delta[train_ends],
        val_delta[val_ends],
        history_scaler,
        control_scaler,
        target_scaler,
        predict_steps,
    )
    normalization_report.to_csv(
        run_dir / "normalization_report.csv", index=False
    )

    train_dataset = HorizonControlWindowDataset(
        train_history,
        train_controls,
        train_delta_scaled,
        train_weights,
        train_ends,
        args.lookback,
        predict_steps,
    )
    val_dataset = HorizonControlWindowDataset(
        val_history,
        val_controls,
        val_delta_scaled,
        val_weights,
        val_ends,
        args.lookback,
        predict_steps,
    )
    train_loader = make_loader(train_dataset, args, shuffle=True, seed_offset=0)
    train_eval_loader = make_loader(train_dataset, args, shuffle=False, seed_offset=1)
    val_loader = make_loader(val_dataset, args, shuffle=False, seed_offset=2)

    device = choose_device()
    model = ResidualControlRNN(
        rnn_type=args.model,
        history_input_size=len(HISTORY_FEATURE_COLS),
        control_input_size=len(FUTURE_CONTROL_COLS),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        control_horizon=predict_steps,
        control_hidden_size=args.control_hidden_size,
        fusion_hidden_size=args.fusion_hidden_size,
        dropout=args.dropout,
    ).to(device)
    parameter_count = count_trainable_parameters(model)

    print(
        f"Experiment=LSTM-vs-GRU Delta Thv(t+{predict_steps}) | "
        f"model={args.model.upper()} | "
        f"lookback={args.lookback} | seed={args.seed} | device={device}"
    )
    print(
        f"Trainable parameters={parameter_count} | hidden_size={args.hidden_size} | "
        f"common_origin_lookback={common_origin_lookback}"
    )
    print(f"Comparison policy={comparison_policy}")
    print(
        "Normalization=independent train-only Standardizer for every history "
        "variable, every future-control variable, and the Delta target"
    )
    print(
        f"Rapid threshold (train-only q={args.rapid_quantile:.3f})="
        f"{rapid_threshold_k:.10f} K | rapid_weight={args.rapid_weight:.3f}"
    )
    print(
        f"train_windows={len(train_dataset)} | val_windows={len(val_dataset)} | "
        f"known controls=u(t)..u(t+{predict_steps - 1})"
    )

    synchronize_device(device)
    training_started = time.perf_counter()
    best_state, history, best_epoch, best_rapid_rmse = train_model(
        model,
        train_loader,
        train_eval_loader,
        val_loader,
        target_scaler,
        args,
        device,
    )
    synchronize_device(device)
    training_seconds = float(time.perf_counter() - training_started)
    completed_epochs = len(history)
    seconds_per_epoch = training_seconds / max(completed_epochs, 1)

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
    last_row = dict(history[-1])

    evaluate_test_window = bool(getattr(args, "evaluate_test_window", False))
    test_label_audit: dict[str, float | int | str] | None = None
    test_window_evaluation: dict[str, object] | None = None
    if evaluate_test_window:
        test_frame, test_label_audit = prepare_full_test_frame(
            args.data_dir,
            train_frame,
            val_frame,
            predict_steps,
            label_col,
        )
        test_window_evaluation = evaluate_fixed_test_window(
            model=model,
            test_frame=test_frame,
            history_scaler=history_scaler,
            control_scaler=control_scaler,
            target_scaler=target_scaler,
            rapid_threshold_k=rapid_threshold_k,
            common_origin_lookback=common_origin_lookback,
            test_window_start=int(getattr(args, "test_window_start", 0)),
            test_window_end=int(getattr(args, "test_window_end", 7300)),
            predict_steps=predict_steps,
            label_col=label_col,
            args=args,
            device=device,
            run_dir=run_dir,
        )

    prediction_data: dict[str, np.ndarray] = {
        "frame_row_index": prediction_ends,
        "file_id": val_frame.iloc[prediction_ends]["file_id"].to_numpy(),
        "source_group": val_frame.iloc[prediction_ends]["source_group"].to_numpy(),
        "source_row_index": val_frame.iloc[prediction_ends][
            "source_row_index"
        ].to_numpy(),
        "source_timestamp": val_frame.iloc[prediction_ends][
            "source_timestamp"
        ].to_numpy(),
        "current_Thv_k": current_temperature,
        f"actual_delta_Thv_{predict_steps}step_k": actual_delta,
        f"predicted_delta_Thv_{predict_steps}step_k": predicted_delta,
        f"actual_Future_Thv_{predict_steps}step_k": actual_temperature,
        f"predicted_Future_Thv_{predict_steps}step_k": predicted_temperature,
        "residual_k": predicted_temperature - actual_temperature,
        "rapid_cooling_mask": rapid_mask.astype(np.int8),
    }
    raw_validation_controls = val_frame[list(FUTURE_CONTROL_COLS)].to_numpy(
        dtype=np.float64
    )
    for offset in range(predict_steps):
        rows = val_ends + offset
        for column_index, column in enumerate(FUTURE_CONTROL_COLS):
            prediction_data[f"control_{column}_t_plus_{offset}"] = (
                raw_validation_controls[rows, column_index]
            )
    prediction_table = pd.DataFrame(prediction_data)
    prediction_table.to_csv(run_dir / "validation_predictions.csv", index=False)
    rapid_prediction_table = prediction_table.loc[
        prediction_table["rapid_cooling_mask"] == 1
    ].copy()
    rapid_prediction_table.insert(
        0, "rapid_subset_index", np.arange(len(rapid_prediction_table), dtype=np.int64)
    )
    rapid_prediction_table.to_csv(
        run_dir / "rapid_validation_predictions.csv", index=False
    )
    build_segment_metrics(prediction_table, predict_steps).to_csv(
        run_dir / "validation_segment_metrics.csv", index=False
    )
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    save_training_plot(history, run_dir / "training_curve.png")
    save_validation_plots(
        actual_temperature,
        predicted_temperature,
        current_temperature,
        actual_delta,
        predicted_delta,
        rapid_mask,
        val_frame.iloc[val_ends]["file_id"].astype(str).to_numpy(),
        run_dir / "validation_prediction.png",
        run_dir / "validation_delta.png",
        (
            f"{args.model.upper()} Delta Thv(t+{predict_steps}) + known controls | "
            f"lookback={args.lookback} | seed={args.seed}"
        ),
        predict_steps=predict_steps,
    )
    save_scalers(
        run_dir / "scalers.npz",
        history_scaler,
        control_scaler,
        target_scaler,
        rapid_threshold_k,
        target_name=target_delta_name,
    )

    leakage_audit = {
        "prediction_quantity": (
            f"Thv(t+{predict_steps}), reconstructed as Thv(t) + "
            f"predicted {target_delta_name}"
        ),
        "trained_target": f"{target_delta_name} = {label_col} - Thv(t)",
        "stored_label": label_col,
        "train_label_alignment": train_label_audit,
        "validation_label_alignment": validation_label_audit,
        "history_latest_time": "t",
        "history_feature_columns": list(HISTORY_FEATURE_COLS),
        "future_control_times": f"u(t)..u(t+{predict_steps - 1})",
        "future_control_columns": list(FUTURE_CONTROL_COLS),
        "future_temperature_pressure_flow_used": False,
        "future_label_used_as_input": False,
        "known_future_control_caveat": (
            f"Recorded u(t+1)..u(t+{predict_steps - 1}) are valid inputs only "
            "for conditional "
            "forecasting/MPC where the control trajectory is known or planned; "
            "they must not be described as unavailable-free-running forecasts."
        ),
        "current_Thv_used_as_input": True,
        "current_Thv_rationale": (
            f"The model predicts a {predict_steps * SAMPLE_PERIOD_SECONDS:g}-second "
            "increment; adding the observed current "
            "Thv back to the predicted increment is residual forecasting, not label leakage."
        ),
        "training_parent_ids": sorted(EXPECTED_TRAIN_PARENTS),
        "validation_parent_id": EXPECTED_VALIDATION_PARENT,
        "validation_variant": "Original",
        "held_out_test_parent_id": EXPECTED_TEST_PARENT_IN_BUILD_CONFIG,
        "held_out_test_loaded_after_training": evaluate_test_window,
        "test_label_alignment": test_label_audit,
        "test_window_used_for_gradient_or_checkpoint_selection": False,
        "scaler_fit_source": "training only after excluding 0617-ALL",
        "rapid_subset_definition": (
            f"actual Delta_Thv_{predict_steps} <= train-only "
            f"q={args.rapid_quantile:.6f} "
            f"threshold ({rapid_threshold_k:.12g} K)"
        ),
        "rapid_validation_windows": int(rapid_mask.sum()),
        "full_validation_windows": int(len(rapid_mask)),
    }
    (run_dir / "leakage_audit.json").write_text(
        json.dumps(leakage_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "experiment": study_name,
        "model_type": args.model,
        "comparison_policy": comparison_policy,
        "lookback": args.lookback,
        "lookback_seconds": args.lookback * SAMPLE_PERIOD_SECONDS,
        "common_origin_lookback": common_origin_lookback,
        "common_origin_seconds": common_origin_lookback * SAMPLE_PERIOD_SECONDS,
        "seed": args.seed,
        "target": target_delta_name,
        "source_label": label_col,
        "predict_steps": predict_steps,
        "predict_seconds": predict_steps * SAMPLE_PERIOD_SECONDS,
        "future_control_columns": list(FUTURE_CONTROL_COLS),
        "future_control_alignment": f"u(t)..u(t+{predict_steps - 1})",
        "future_noncontrol_measurements_used": False,
        "history_feature_columns": list(HISTORY_FEATURE_COLS),
        "rapid_threshold_k_train_only": rapid_threshold_k,
        "rapid_quantile": args.rapid_quantile,
        "rapid_weight": args.rapid_weight,
        "model_selection_split": "Original 260501 validation",
        "primary_evaluation_subset": "rapid-cooling validation windows",
        "primary_evaluation_metric": "rapid_validation_rmse_k",
        "primary_subset_prediction_file": "rapid_validation_predictions.csv",
        "checkpoint_selection_metric": "validation_rapid_rmse_k",
        "final_reported_model": "best checkpoint by validation_rapid_rmse_k",
        "normalization": {
            "strategy": "independent per-variable Standardizer",
            "fit_source": "training data only after excluding 0617-ALL",
            "history_scaler_fit_rows": "all retained training rows",
            "future_control_scaler_fit_rows": "all retained training rows",
            "target_scaler_fit_rows": "training Delta values at common forecast origins",
            "validation_used_to_fit_scalers": False,
            "report_file": "normalization_report.csv",
        },
        "leakage_audit_file": "leakage_audit.json",
        "excluded_parent": EXCLUDED_PARENT,
        "training_parents": sorted(EXPECTED_TRAIN_PARENTS),
        "validation_parent": EXPECTED_VALIDATION_PARENT,
        "test_parent": EXPECTED_TEST_PARENT_IN_BUILD_CONFIG,
        "test_loaded_after_training": evaluate_test_window,
        "fixed_test_window_evaluation": test_window_evaluation,
        "train_windows": int(len(train_dataset)),
        "validation_windows": int(len(val_dataset)),
        "validation_rapid_windows": int(rapid_mask.sum()),
        "trainable_parameters": parameter_count,
        "training_seconds_total": training_seconds,
        "completed_epochs": completed_epochs,
        "seconds_per_completed_epoch": seconds_per_epoch,
        "best_epoch": best_epoch,
        "best_epoch_validation_rapid_rmse_k": best_rapid_rmse,
        "best_epoch_history": best_row,
        "last_completed_epoch_history": last_row,
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
            "model_class": "ResidualControlRNN",
            "model_type": args.model,
            "history_input_size": len(HISTORY_FEATURE_COLS),
            "control_input_size": len(FUTURE_CONTROL_COLS),
            "control_horizon": predict_steps,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "control_hidden_size": args.control_hidden_size,
            "fusion_hidden_size": args.fusion_hidden_size,
            "dropout": args.dropout,
            "history_feature_columns": list(HISTORY_FEATURE_COLS),
            "future_control_columns": list(FUTURE_CONTROL_COLS),
            "future_control_alignment": f"u(t)..u(t+{predict_steps - 1})",
            "lookback": args.lookback,
            "common_origin_lookback": common_origin_lookback,
            "predict_steps": predict_steps,
            "predict_seconds": predict_steps * SAMPLE_PERIOD_SECONDS,
            "seed": args.seed,
            "experiment": study_name,
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
            "normalization_strategy": "independent train-only per-variable Standardizer",
            "validation_metrics": validation_metrics,
            "rapid_validation_metrics": rapid_validation_metrics,
        },
        run_dir / "best_model.pt",
    )

    print(f"BEST_EPOCH={best_epoch}")
    print(f"TRAINABLE_PARAMETERS={parameter_count}")
    print(f"TRAINING_SECONDS_TOTAL={training_seconds:.6f}")
    print(f"SECONDS_PER_COMPLETED_EPOCH={seconds_per_epoch:.6f}")
    print(
        f"PREDICTED_QUANTITY=Thv(t+{predict_steps})_via_"
        f"Delta_Thv_{predict_steps}"
    )
    print("MODEL_SELECTION_SPLIT=Original_260501_validation")
    print(
        "FIXED_TEST_SPLIT="
        + (
            "Original_0715-BACK_loaded_after_checkpoint_selection"
            if evaluate_test_window
            else "Original_0715-BACK_not_loaded"
        )
    )
    print(f"PRIMARY_RAPID_THRESHOLD_K={rapid_threshold_k:.10f}")
    print(f"PRIMARY_RAPID_VALIDATION_WINDOWS={int(rapid_mask.sum())}")
    print(f"BEST_CHECKPOINT_TRAIN_RMSE_K={best_row['train_rmse_k']:.10f}")
    print(
        f"BEST_CHECKPOINT_TRAIN_RAPID_RMSE_K="
        f"{best_row['train_rapid_rmse_k']:.10f}"
    )
    print(
        f"PRIMARY_FINAL_RAPID_VALIDATION_RMSE_K="
        f"{rapid_validation_metrics['rmse_k']:.10f}"
    )
    print(f"SECONDARY_FINAL_FULL_VALIDATION_RMSE_K={validation_metrics['rmse_k']:.10f}")
    print(
        f"LAST_COMPLETED_EPOCH_VALIDATION_RMSE_K="
        f"{last_row['validation_rmse_k']:.10f}"
    )
    print(
        f"LAST_COMPLETED_EPOCH_RAPID_VALIDATION_RMSE_K="
        f"{last_row['validation_rapid_rmse_k']:.10f}"
    )
    if test_window_evaluation is not None:
        fixed_metrics = test_window_evaluation["window_metrics"]
        fixed_rapid_metrics = test_window_evaluation["rapid_window_metrics"]
        assert isinstance(fixed_metrics, dict)
        assert isinstance(fixed_rapid_metrics, dict)
        print(
            "TEST_WINDOW_SOURCE_ROWS="
            f"{int(test_window_evaluation['window_start'])}.."
            f"{int(test_window_evaluation['window_end'])}_inclusive"
        )
        print(f"TEST_WINDOW_VALID_ORIGINS={int(fixed_metrics['n_windows'])}")
        print(f"TEST_WINDOW_RMSE_K={float(fixed_metrics['rmse_k']):.10f}")
        print(
            "TEST_WINDOW_TRAIN_THRESHOLD_RAPID_ORIGINS="
            f"{int(fixed_rapid_metrics['n_windows'])}"
        )
        print(
            "TEST_WINDOW_TRAIN_THRESHOLD_RAPID_RMSE_K="
            f"{float(fixed_rapid_metrics['rmse_k']):.10f}"
        )
    print(f"MODEL_SAVED={run_dir / 'best_model.pt'}")
    print(f"RESULTS_SAVED={run_dir}")


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
