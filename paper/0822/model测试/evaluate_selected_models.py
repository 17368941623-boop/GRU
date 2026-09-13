#!/usr/bin/env python3
"""Evaluate only validation-selected configurations on Original 0715-BACK."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BASELINE_DIR = PROJECT_DIR / "残差lookback测试"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from train_thv_delta_lstm_lookback import (  # noqa: E402
    FUTURE_CONTROL_COLS,
    HISTORY_FEATURE_COLS,
    TARGET_COL,
    Standardizer,
    choose_device,
    direction_accuracy,
    make_loader,
    predict_delta,
    regression_metrics,
    save_validation_plots,
)
from train_thv_delta_rnn_compare import (  # noqa: E402
    HorizonControlWindowDataset,
    assert_horizon_window_boundaries,
    build_horizon_window_ends,
    label_column,
    prepare_full_test_frame,
    prepare_horizon_development_frames,
)

from model_components import ModelConfig, build_model, model_description  # noqa: E402


DEFAULT_DATA_DIR = PROJECT_DIR / "processed_data"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final test evaluation after validation-only configuration selection."
    )
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument("--predict-steps", type=int, default=15)
    parser.add_argument("--seeds", default="42,62,82")
    parser.add_argument("--test-window-start", type=int, default=0)
    parser.add_argument("--test-window-end", type=int, default=7300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(sorted({int(item.strip()) for item in text.split(",") if item.strip()}))
    if not seeds:
        raise ValueError("At least one final seed is required")
    return seeds


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def standardizer(payload: dict[str, object]) -> Standardizer:
    return Standardizer(
        mean=np.asarray(payload["mean"], dtype=np.float64),
        scale=np.asarray(payload["scale"], dtype=np.float64),
    )


def subset_metrics(
    actual_temperature: np.ndarray,
    predicted_temperature: np.ndarray,
    current_temperature: np.ndarray,
    actual_delta: np.ndarray,
    predicted_delta: np.ndarray,
    rapid_mask: np.ndarray,
) -> dict[str, object]:
    if not rapid_mask.any():
        raise ValueError("Rapid-cooling test subset is empty")
    return {
        "temperature": regression_metrics(actual_temperature, predicted_temperature),
        "rapid_temperature": regression_metrics(
            actual_temperature[rapid_mask], predicted_temperature[rapid_mask]
        ),
        "delta": regression_metrics(actual_delta, predicted_delta),
        "persistence": regression_metrics(actual_temperature, current_temperature),
        "rapid_persistence": regression_metrics(
            actual_temperature[rapid_mask], current_temperature[rapid_mask]
        ),
        "direction_accuracy": direction_accuracy(actual_delta, predicted_delta),
        "rapid_direction_accuracy": direction_accuracy(
            actual_delta[rapid_mask], predicted_delta[rapid_mask]
        ),
    }


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    study_dir = (
        args.results_dir
        / "development"
        / f"horizon_{args.predict_steps:02d}"
        / f"lookback_{args.lookback:02d}"
    )
    selected_path = study_dir / "selected_configs.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if bool(selected.get("test_data_read", True)):
        raise ValueError("Configuration selection was not test-blind")
    if int(selected["lookback"]) != args.lookback:
        raise ValueError("Selected lookback mismatch")
    if int(selected["predict_steps"]) != args.predict_steps:
        raise ValueError("Selected forecast horizon mismatch")

    label_col = label_column(args.predict_steps)
    # The held-out test is opened only after selected_configs.json exists.
    train_frame, validation_frame = prepare_horizon_development_frames(
        args.data_dir, args.predict_steps, label_col
    )
    test_frame, test_label_audit = prepare_full_test_frame(
        args.data_dir,
        train_frame,
        validation_frame,
        args.predict_steps,
        label_col,
    )
    common_origin_lookback = int(selected["common_origin_lookback"])
    if common_origin_lookback < args.lookback:
        raise ValueError("Selected common-origin lookback is smaller than lookback")
    all_ends = build_horizon_window_ends(
        test_frame, common_origin_lookback, args.predict_steps
    )
    assert_horizon_window_boundaries(
        test_frame, all_ends, args.lookback, args.predict_steps
    )
    source_rows = pd.to_numeric(
        test_frame.iloc[all_ends]["source_row_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    fixed_mask = (source_rows >= args.test_window_start) & (
        source_rows <= args.test_window_end
    )
    fixed_ends = all_ends[fixed_mask]
    if len(fixed_ends) == 0:
        raise ValueError("Fixed test source-row window contains no valid origins")

    test_current = test_frame[TARGET_COL].to_numpy(dtype=np.float64)
    test_future = test_frame[label_col].to_numpy(dtype=np.float64)
    test_delta = test_future - test_current
    device = choose_device()
    rows: list[dict[str, object]] = []

    for model_name, selection in selected["models"].items():
        config_id = str(selection["config_id"])
        for seed in seeds:
            run_dir = (
                study_dir / model_name / config_id / f"seed_{seed}"
            )
            checkpoint_path = run_dir / "best_model.pt"
            development_metrics_path = run_dir / "metrics.json"
            if not checkpoint_path.exists() or not development_metrics_path.exists():
                raise FileNotFoundError(
                    f"Missing selected checkpoint for {model_name}/{config_id}/seed_{seed}"
                )
            evaluation_dir = run_dir / "test_evaluation"
            test_metrics_path = evaluation_dir / "test_metrics.json"
            if test_metrics_path.exists() and not args.overwrite:
                payload = json.loads(test_metrics_path.read_text(encoding="utf-8"))
                rows.append(payload["summary_row"])
                print(
                    f"SKIP existing test evaluation: "
                    f"{model_name}/{config_id}/seed_{seed}",
                    flush=True,
                )
                continue
            evaluation_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = load_checkpoint(checkpoint_path)
            if bool(checkpoint.get("test_data_loaded_during_training", True)):
                raise ValueError(f"Checkpoint was trained after opening test data: {checkpoint_path}")
            if int(checkpoint["lookback"]) != args.lookback:
                raise ValueError("Checkpoint lookback mismatch")
            if int(checkpoint.get("common_origin_lookback", -1)) != common_origin_lookback:
                raise ValueError("Checkpoint common-origin lookback mismatch")
            if int(checkpoint["predict_steps"]) != args.predict_steps:
                raise ValueError("Checkpoint horizon mismatch")
            if str(checkpoint["model_type"]) != model_name:
                raise ValueError("Checkpoint model mismatch")
            if str(checkpoint["config_id"]) != config_id:
                raise ValueError("Checkpoint configuration mismatch")

            history_scaler = standardizer(checkpoint["history_scaler"])
            control_scaler = standardizer(checkpoint["control_scaler"])
            target_scaler = standardizer(checkpoint["target_scaler"])
            rapid_threshold_k = float(checkpoint["rapid_threshold_k"])
            model_config = ModelConfig(**checkpoint["model_config"])
            model = build_model(
                model_config,
                HISTORY_FEATURE_COLS,
                FUTURE_CONTROL_COLS,
                args.predict_steps,
                history_scaler.mean,
                history_scaler.scale,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)

            history_scaled = history_scaler.transform(
                test_frame[list(HISTORY_FEATURE_COLS)].to_numpy(dtype=np.float64)
            ).astype(np.float32)
            controls_scaled = control_scaler.transform(
                test_frame[list(FUTURE_CONTROL_COLS)].to_numpy(dtype=np.float64)
            ).astype(np.float32)
            delta_scaled = target_scaler.transform(test_delta).astype(np.float32)
            weights = np.ones(len(test_frame), dtype=np.float32)
            weights[test_delta <= rapid_threshold_k] = 3.0
            dataset = HorizonControlWindowDataset(
                history_scaled,
                controls_scaled,
                delta_scaled,
                weights,
                all_ends,
                args.lookback,
                args.predict_steps,
            )
            loader_args = SimpleNamespace(
                seed=seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            loader = make_loader(
                dataset, loader_args, shuffle=False, seed_offset=100
            )
            predicted_delta, predicted_ends = predict_delta(
                model, loader, device, target_scaler
            )
            if not np.array_equal(predicted_ends, all_ends):
                raise AssertionError("Test prediction order changed")

            current = test_current[all_ends]
            actual = test_future[all_ends]
            actual_delta = test_delta[all_ends]
            predicted = current + predicted_delta
            rapid = actual_delta <= rapid_threshold_k
            full_metrics = subset_metrics(
                actual, predicted, current, actual_delta, predicted_delta, rapid
            )

            fixed_indices = np.flatnonzero(fixed_mask)
            fixed_metrics = subset_metrics(
                actual[fixed_indices],
                predicted[fixed_indices],
                current[fixed_indices],
                actual_delta[fixed_indices],
                predicted_delta[fixed_indices],
                rapid[fixed_indices],
            )
            development = json.loads(
                development_metrics_path.read_text(encoding="utf-8")
            )
            summary_row = {
                "model": model_name,
                "model_description": model_description(model_name),
                "config_id": config_id,
                "seed": seed,
                "validation_rapid_rmse_k": float(
                    development["rapid_validation_metrics"]["rmse_k"]
                ),
                "validation_rmse_k": float(
                    development["validation_metrics"]["rmse_k"]
                ),
                "test_full_rmse_k": float(full_metrics["temperature"]["rmse_k"]),
                "test_full_mae_k": float(full_metrics["temperature"]["mae_k"]),
                "test_full_p95_absolute_error_k": float(
                    full_metrics["temperature"]["p95_absolute_error_k"]
                ),
                "test_full_max_absolute_error_k": float(
                    full_metrics["temperature"]["max_absolute_error_k"]
                ),
                "test_rapid_rmse_k": float(
                    full_metrics["rapid_temperature"]["rmse_k"]
                ),
                "test_rapid_mae_k": float(
                    full_metrics["rapid_temperature"]["mae_k"]
                ),
                "test_rapid_p95_absolute_error_k": float(
                    full_metrics["rapid_temperature"]["p95_absolute_error_k"]
                ),
                "test_rapid_max_absolute_error_k": float(
                    full_metrics["rapid_temperature"]["max_absolute_error_k"]
                ),
                "test_persistence_rmse_k": float(
                    full_metrics["persistence"]["rmse_k"]
                ),
                "test_rapid_persistence_rmse_k": float(
                    full_metrics["rapid_persistence"]["rmse_k"]
                ),
                "test_direction_accuracy": float(
                    full_metrics["direction_accuracy"]
                ),
                "test_rapid_direction_accuracy": float(
                    full_metrics["rapid_direction_accuracy"]
                ),
                "fixed_window_start": args.test_window_start,
                "fixed_window_end": args.test_window_end,
                "fixed_window_rmse_k": float(
                    fixed_metrics["temperature"]["rmse_k"]
                ),
                "fixed_window_rapid_rmse_k": float(
                    fixed_metrics["rapid_temperature"]["rmse_k"]
                ),
                "test_full_windows": int(full_metrics["temperature"]["n_windows"]),
                "test_rapid_windows": int(
                    full_metrics["rapid_temperature"]["n_windows"]
                ),
                "fixed_window_windows": int(
                    fixed_metrics["temperature"]["n_windows"]
                ),
                "trainable_parameters": int(development["trainable_parameters"]),
            }
            rows.append(summary_row)

            prediction_table = pd.DataFrame(
                {
                    "frame_row_index": all_ends,
                    "file_id": test_frame.iloc[all_ends]["file_id"].to_numpy(),
                    "source_row_index": source_rows,
                    "source_timestamp": test_frame.iloc[all_ends][
                        "source_timestamp"
                    ].to_numpy(),
                    "current_Thv_k": current,
                    f"actual_delta_Thv_{args.predict_steps}step_k": actual_delta,
                    f"predicted_delta_Thv_{args.predict_steps}step_k": predicted_delta,
                    f"actual_Future_Thv_{args.predict_steps}step_k": actual,
                    f"predicted_Future_Thv_{args.predict_steps}step_k": predicted,
                    "residual_k": predicted - actual,
                    "rapid_cooling_mask": rapid.astype(np.int8),
                    "fixed_window_mask": fixed_mask.astype(np.int8),
                }
            )
            prediction_table.to_csv(
                evaluation_dir / "test_predictions.csv", index=False
            )
            save_validation_plots(
                actual,
                predicted,
                current,
                actual_delta,
                predicted_delta,
                rapid,
                test_frame.iloc[all_ends]["file_id"].astype(str).to_numpy(),
                evaluation_dir / "test_prediction.png",
                evaluation_dir / "test_delta.png",
                f"TEST {model_name} | h={args.predict_steps} | seed={seed}",
                predict_steps=args.predict_steps,
            )
            payload = {
                "selection_file": str(selected_path),
                "selection_was_test_blind": True,
                "test_parent": "0715-BACK",
                "test_variant": "Original",
                "test_label_audit": test_label_audit,
                "model": model_name,
                "config_id": config_id,
                "seed": seed,
                "full_test_metrics": full_metrics,
                "fixed_window_metrics": fixed_metrics,
                "summary_row": summary_row,
            }
            test_metrics_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"TEST COMPLETE {model_name}/{config_id}/seed_{seed} | "
                f"full_rmse_k={summary_row['test_full_rmse_k']:.8f} | "
                f"rapid_rmse_k={summary_row['test_rapid_rmse_k']:.8f}",
                flush=True,
            )

    all_runs = pd.DataFrame(rows).sort_values(["model", "seed"]).reset_index(drop=True)
    final_dir = (
        args.results_dir
        / "final_test"
        / f"horizon_{args.predict_steps:02d}"
        / f"lookback_{args.lookback:02d}"
    )
    final_dir.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(final_dir / "final_all_seed_runs.csv", index=False)
    summary = (
        all_runs.groupby(["model", "model_description", "config_id"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            validation_rapid_rmse_mean_k=("validation_rapid_rmse_k", "mean"),
            validation_rapid_rmse_std_k=("validation_rapid_rmse_k", "std"),
            validation_rmse_mean_k=("validation_rmse_k", "mean"),
            test_full_rmse_mean_k=("test_full_rmse_k", "mean"),
            test_full_rmse_std_k=("test_full_rmse_k", "std"),
            test_full_mae_mean_k=("test_full_mae_k", "mean"),
            test_full_p95_absolute_error_mean_k=(
                "test_full_p95_absolute_error_k", "mean"
            ),
            test_full_max_absolute_error_mean_k=(
                "test_full_max_absolute_error_k", "mean"
            ),
            test_rapid_rmse_mean_k=("test_rapid_rmse_k", "mean"),
            test_rapid_rmse_std_k=("test_rapid_rmse_k", "std"),
            test_rapid_mae_mean_k=("test_rapid_mae_k", "mean"),
            test_rapid_p95_absolute_error_mean_k=(
                "test_rapid_p95_absolute_error_k", "mean"
            ),
            test_rapid_max_absolute_error_mean_k=(
                "test_rapid_max_absolute_error_k", "mean"
            ),
            fixed_window_rmse_mean_k=("fixed_window_rmse_k", "mean"),
            fixed_window_rapid_rmse_mean_k=("fixed_window_rapid_rmse_k", "mean"),
            test_rapid_direction_accuracy_mean=("test_rapid_direction_accuracy", "mean"),
            trainable_parameters=("trainable_parameters", "first"),
        )
        .sort_values("validation_rapid_rmse_mean_k")
        .reset_index(drop=True)
    )
    summary.insert(0, "validation_selected_rank", np.arange(1, len(summary) + 1))
    summary.to_csv(final_dir / "final_model_summary.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(max(15, len(summary) * 2.0), 5.5))
    labels = summary["model"].tolist()
    plot_specs = (
        ("validation_rapid_rmse_mean_k", "validation_rapid_rmse_std_k", "260501 rapid validation RMSE"),
        ("test_full_rmse_mean_k", "test_full_rmse_std_k", "0715-BACK full RMSE"),
        ("test_rapid_rmse_mean_k", "test_rapid_rmse_std_k", "0715-BACK rapid RMSE"),
    )
    for axis, (mean_col, std_col, title) in zip(axes, plot_specs):
        axis.bar(
            labels,
            summary[mean_col],
            yerr=summary[std_col].fillna(0.0),
            capsize=4,
            color="#457b9d",
            alpha=0.88,
        )
        axis.set_title(title)
        axis.set_ylabel("RMSE (K)")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Validation-selected Thv models; test excluded from ranking")
    figure.tight_layout()
    figure.savefig(final_dir / "final_model_comparison.png", dpi=220)
    plt.close(figure)

    audit = {
        "selection_file": str(selected_path),
        "selection_split": "Original 260501",
        "selection_primary_metric": "rapid validation RMSE",
        "lookback": args.lookback,
        "common_origin_lookback": common_origin_lookback,
        "test_split": "Original 0715-BACK",
        "test_opened_only_after_selected_configs_json": True,
        "test_used_to_change_configuration": False,
        "test_used_to_rank_summary": False,
        "final_seeds": list(seeds),
        "full_test_windows_are_reported": True,
        "fixed_test_source_window": [args.test_window_start, args.test_window_end],
    }
    (final_dir / "final_evaluation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("TEST_USED_FOR_MODEL_OR_HYPERPARAMETER_SELECTION=false")
    print(f"FINAL_SUMMARY={final_dir / 'final_model_summary.csv'}")


if __name__ == "__main__":
    main()
