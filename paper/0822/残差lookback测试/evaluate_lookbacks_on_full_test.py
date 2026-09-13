#!/usr/bin/env python3
"""Evaluate every frozen lookback checkpoint on the complete 0715-BACK test run.

This script performs inference only. It never trains a model, updates a weight,
or changes a checkpoint. All lookbacks use the same forecast origins requiring
120 historical samples, so the full-test RMSE values are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from train_thv_delta_lstm_lookback import Standardizer, choose_device, make_loader
from train_thv_delta_rnn_compare import (
    FUTURE_CONTROL_COLS,
    HISTORY_FEATURE_COLS,
    HorizonControlWindowDataset,
    ResidualControlRNN,
    assert_horizon_window_boundaries,
    build_horizon_window_ends,
    direction_accuracy,
    predict_delta,
    regression_metrics,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "gru_lookback_final_unified_val260501"
DEFAULT_DATA_DIR = PROJECT_DIR / "processed_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "gru_lookback_final_test0715BACK_full"
LOOKBACKS = (15, 30, 40, 45, 50, 55, 60, 65, 70, 75, 90, 120)
SEEDS = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)
PREDICT_STEPS = 15
COMMON_ORIGIN_LOOKBACK = 120
TARGET_COL = "Thv"
LABEL_COL = "Future_Thv_15step"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate_test_frame(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    required = {
        *HISTORY_FEATURE_COLS,
        *FUTURE_CONTROL_COLS,
        TARGET_COL,
        LABEL_COL,
        "file_id",
        "source_group",
        "source_variant",
        "parent_id",
        "source_row_index",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Test frame is missing columns: {sorted(missing)}")
    if config.get("test_parent") != "0715-BACK":
        raise ValueError("Processed-data contract does not name 0715-BACK as test")
    if config.get("test_variant") != "Original only":
        raise ValueError("Processed-data contract does not restrict test to Original")
    if config.get("future_features_used") is not False:
        raise ValueError("Processed-data contract does not guarantee causal features")
    if set(map(str, frame["parent_id"].unique())) != {"0715-BACK"}:
        raise ValueError("Full test frame is not exclusively 0715-BACK")
    if set(map(str, frame["source_variant"].unique())) != {"Original"}:
        raise ValueError("Full test frame contains augmented data")
    numeric = frame[list(HISTORY_FEATURE_COLS) + list(FUTURE_CONTROL_COLS) + [TARGET_COL, LABEL_COL]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Full test frame contains NaN or inf in model fields")
    return frame.reset_index(drop=True)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    required = {
        "model_state_dict", "model_type", "history_input_size", "control_input_size",
        "control_horizon", "hidden_size", "num_layers", "control_hidden_size",
        "fusion_hidden_size", "dropout", "history_feature_columns",
        "future_control_columns", "lookback", "common_origin_lookback",
        "predict_steps", "seed", "history_scaler", "control_scaler",
        "target_scaler", "rapid_threshold_k",
    }
    missing = required - set(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint {path} is missing: {sorted(missing)}")
    return checkpoint


def scaler(payload: dict) -> Standardizer:
    return Standardizer(
        mean=np.asarray(payload["mean"], dtype=np.float64),
        scale=np.asarray(payload["scale"], dtype=np.float64),
    )


def summarize(group: pd.DataFrame) -> pd.Series:
    values = group["test_full_rmse_k"].to_numpy(dtype=np.float64)
    return pd.Series(
        {
            "seed_count": int(len(values)),
            "test_full_rmse_mean_k": float(values.mean()),
            "test_full_rmse_std_k": float(values.std(ddof=1)),
            "test_full_rmse_median_k": float(np.median(values)),
            "test_full_rmse_q1_k": float(np.percentile(values, 25)),
            "test_full_rmse_q3_k": float(np.percentile(values, 75)),
            "test_full_rmse_min_k": float(values.min()),
            "test_full_rmse_max_k": float(values.max()),
            "test_full_mae_mean_k": float(group["test_full_mae_k"].mean()),
            "test_full_p95_absolute_error_mean_k": float(
                group["test_full_p95_absolute_error_k"].mean()
            ),
            "test_full_max_absolute_error_mean_k": float(
                group["test_full_max_absolute_error_k"].mean()
            ),
            "test_direction_accuracy_mean": float(group["test_direction_accuracy"].mean()),
            "test_windows": int(group["test_windows"].iloc[0]),
            "persistence_test_rmse_k": float(group["persistence_test_rmse_k"].iloc[0]),
        }
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid loader settings")
    config = json.loads((args.data_dir / "data_build_config.json").read_text(encoding="utf-8"))
    frame = validate_test_frame(joblib.load(args.data_dir / "test_full_clean.pkl"), config)
    test_ends = build_horizon_window_ends(
        frame, COMMON_ORIGIN_LOOKBACK, PREDICT_STEPS
    )
    current = frame[TARGET_COL].to_numpy(dtype=np.float64)
    future = frame[LABEL_COL].to_numpy(dtype=np.float64)
    actual_delta_all = future - current
    raw_history = frame[list(HISTORY_FEATURE_COLS)].to_numpy(dtype=np.float64)
    raw_controls = frame[list(FUTURE_CONTROL_COLS)].to_numpy(dtype=np.float64)
    device = choose_device()
    rows: list[dict[str, object]] = []

    print(f"DEVICE={device}")
    print(f"TEST_PARENT=0715-BACK")
    print(f"TEST_VARIANT=Original")
    print(f"COMMON_TEST_ORIGINS={len(test_ends)}")

    for lookback in LOOKBACKS:
        assert_horizon_window_boundaries(
            frame, test_ends, lookback, PREDICT_STEPS
        )
        for seed in SEEDS:
            checkpoint_path = (
                args.checkpoint_root
                / f"horizon_{PREDICT_STEPS:02d}"
                / "gru"
                / f"lookback_{lookback:02d}"
                / f"seed_{seed}"
                / "best_model.pt"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            checkpoint = load_checkpoint(checkpoint_path, device)
            if checkpoint["model_type"] != "gru":
                raise ValueError(f"Unexpected model type in {checkpoint_path}")
            if int(checkpoint["lookback"]) != lookback or int(checkpoint["seed"]) != seed:
                raise ValueError(f"Checkpoint identity mismatch: {checkpoint_path}")
            if int(checkpoint["predict_steps"]) != PREDICT_STEPS:
                raise ValueError(f"Forecast horizon mismatch: {checkpoint_path}")
            if int(checkpoint["common_origin_lookback"]) != COMMON_ORIGIN_LOOKBACK:
                raise ValueError(f"Common-origin mismatch: {checkpoint_path}")
            if tuple(checkpoint["history_feature_columns"]) != tuple(HISTORY_FEATURE_COLS):
                raise ValueError(f"History feature mismatch: {checkpoint_path}")
            if tuple(checkpoint["future_control_columns"]) != tuple(FUTURE_CONTROL_COLS):
                raise ValueError(f"Future-control feature mismatch: {checkpoint_path}")

            history_scaler = scaler(checkpoint["history_scaler"])
            control_scaler = scaler(checkpoint["control_scaler"])
            target_scaler = scaler(checkpoint["target_scaler"])
            history_scaled = history_scaler.transform(raw_history).astype(np.float32)
            controls_scaled = control_scaler.transform(raw_controls).astype(np.float32)
            delta_scaled = target_scaler.transform(actual_delta_all).astype(np.float32)
            dataset = HorizonControlWindowDataset(
                history_scaled=history_scaled,
                controls_scaled=controls_scaled,
                delta_scaled=delta_scaled,
                sample_weights=np.ones(len(frame), dtype=np.float32),
                ends=test_ends,
                lookback=lookback,
                predict_steps=PREDICT_STEPS,
            )
            loader_args = argparse.Namespace(
                seed=seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            loader = make_loader(dataset, loader_args, shuffle=False, seed_offset=20)
            model = ResidualControlRNN(
                rnn_type="gru",
                history_input_size=int(checkpoint["history_input_size"]),
                control_input_size=int(checkpoint["control_input_size"]),
                hidden_size=int(checkpoint["hidden_size"]),
                num_layers=int(checkpoint["num_layers"]),
                control_horizon=int(checkpoint["control_horizon"]),
                control_hidden_size=int(checkpoint["control_hidden_size"]),
                fusion_hidden_size=int(checkpoint["fusion_hidden_size"]),
                dropout=float(checkpoint["dropout"]),
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            predicted_delta, prediction_ends = predict_delta(
                model, loader, device, target_scaler
            )
            if not np.array_equal(prediction_ends, test_ends):
                raise AssertionError("Test prediction order changed")

            actual_delta = actual_delta_all[test_ends]
            actual_temperature = future[test_ends]
            predicted_temperature = current[test_ends] + predicted_delta
            metrics = regression_metrics(actual_temperature, predicted_temperature)
            persistence = regression_metrics(actual_temperature, current[test_ends])
            rows.append(
                {
                    "lookback": lookback,
                    "seed": seed,
                    "hidden_size": int(checkpoint["hidden_size"]),
                    "num_layers": int(checkpoint["num_layers"]),
                    "control_hidden_size": int(checkpoint["control_hidden_size"]),
                    "fusion_hidden_size": int(checkpoint["fusion_hidden_size"]),
                    "test_full_rmse_k": float(metrics["rmse_k"]),
                    "test_full_mae_k": float(metrics["mae_k"]),
                    "test_full_p95_absolute_error_k": float(metrics["p95_absolute_error_k"]),
                    "test_full_max_absolute_error_k": float(metrics["max_absolute_error_k"]),
                    "test_bias_k": float(metrics["bias_k"]),
                    "test_r2": float(metrics["r2"]),
                    "test_direction_accuracy": direction_accuracy(actual_delta, predicted_delta),
                    "test_windows": int(metrics["n_windows"]),
                    "persistence_test_rmse_k": float(persistence["rmse_k"]),
                    "checkpoint": str(checkpoint_path.relative_to(args.checkpoint_root)),
                }
            )
            print(
                f"DONE lookback={lookback:03d} seed={seed:03d} "
                f"test_full_rmse_k={float(metrics['rmse_k']):.6f}"
            )
            del model, loader, dataset, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

    all_runs = pd.DataFrame(rows).sort_values(["lookback", "seed"]).reset_index(drop=True)
    expected = len(LOOKBACKS) * len(SEEDS)
    if len(all_runs) != expected or all_runs.duplicated(["lookback", "seed"]).any():
        raise AssertionError("Incomplete or duplicate full-test evaluations")
    if set(all_runs["test_windows"]) != {len(test_ends)}:
        raise AssertionError("Test window counts differ between runs")
    summary = (
        all_runs.groupby("lookback", sort=True, as_index=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    if "level_0" in summary:
        summary.drop(columns="level_0", inplace=True)
    summary["lookback"] = list(LOOKBACKS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(args.output_dir / "lookback_test_all_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "lookback_test_summary.csv", index=False, encoding="utf-8-sig")
    best = summary.loc[summary["test_full_rmse_mean_k"].idxmin()]
    manifest = {
        "evaluation_type": "frozen-checkpoint inference on complete test sequence",
        "training_or_weight_updates": False,
        "checkpoint_selection_split": "Original 260501 validation",
        "evaluation_split": "Original 0715-BACK full test",
        "test_set_used_for_lookback_plot": True,
        "methodological_consequence": (
            "Because all lookbacks are compared on this test run, 0715-BACK is now a "
            "post-hoc model-selection/evaluation set rather than an untouched final holdout."
        ),
        "lookbacks": list(LOOKBACKS),
        "seeds": list(SEEDS),
        "predict_steps": PREDICT_STEPS,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "test_windows_per_run": int(len(test_ends)),
        "model_runs": expected,
        "numerical_minimum_lookback": int(best["lookback"]),
        "numerical_minimum_test_full_rmse_mean_k": float(best["test_full_rmse_mean_k"]),
        "summary_csv": "lookback_test_summary.csv",
        "all_runs_csv": "lookback_test_all_runs.csv",
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
