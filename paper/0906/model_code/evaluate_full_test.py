#!/usr/bin/env python3
"""Evaluate every frozen 0906 checkpoint on the complete 0715-BACK sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SHARED_DIR = PROJECT_DIR / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

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
from study_protocol import (  # noqa: E402
    COMMON_ORIGIN_LOOKBACK,
    LOOKBACK,
    MODEL_SPECS,
    PREDICT_STEPS,
    SELECTION_METRIC,
    all_tasks,
    protocol_payload,
    run_directory,
    valid_completed_run,
)
from summarize_validation import (  # noqa: E402
    STRUCTURAL_CONTRASTS,
    holm_adjust,
    interval,
)


DEFAULT_DATA_DIR = PROJECT_DIR / "processed_data"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "outputs_0906"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--save-prediction-seeds",
        default="42",
        help="Comma-separated seeds whose full prediction curves are saved; empty saves none.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_seed_set(text: str) -> set[int]:
    return {int(item.strip()) for item in text.split(",") if item.strip()}


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


def validate_all_training_complete(results_dir: Path) -> None:
    missing = [
        f"{spec['model']}/{spec['config_id']}/seed_{seed}"
        for spec, seed in all_tasks()
        if not valid_completed_run(results_dir, spec, seed)
    ]
    if missing:
        raise RuntimeError(
            f"Refusing to open the test set because {len(missing)} training runs are incomplete"
        )
    validation_summary = results_dir / "validation_summary" / "validation_model_summary.csv"
    if not validation_summary.exists():
        raise FileNotFoundError(
            "Validation summary must be frozen before test evaluation: "
            f"{validation_summary}"
        )


def evaluate_one(
    args: argparse.Namespace,
    spec: dict[str, object],
    seed: int,
    test_frame: pd.DataFrame,
    test_ends: np.ndarray,
    test_current: np.ndarray,
    test_future: np.ndarray,
    test_delta: np.ndarray,
    device: torch.device,
    save_predictions: bool,
) -> dict[str, object]:
    run_dir = run_directory(args.results_dir, spec, seed)
    evaluation_dir = run_dir / "full_test_evaluation"
    metrics_path = evaluation_dir / "test_metrics.json"
    predictions_path = evaluation_dir / "test_predictions.csv"
    if metrics_path.exists() and not args.overwrite:
        if not save_predictions or predictions_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary_row = payload["summary_row"]
            if (
                int(summary_row["lookback"]) == LOOKBACK
                and int(summary_row["predict_steps"]) == PREDICT_STEPS
                and int(summary_row["seed"]) == seed
                and str(summary_row["model"]) == spec["model"]
            ):
                print(f"REUSE TEST {spec['model']}/seed_{seed}", flush=True)
                return summary_row

    checkpoint_path = run_dir / "best_model.pt"
    development_path = run_dir / "metrics.json"
    checkpoint = load_checkpoint(checkpoint_path)
    development = json.loads(development_path.read_text(encoding="utf-8"))
    identity = {
        "model_type": spec["model"],
        "config_id": spec["config_id"],
        "lookback": LOOKBACK,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "seed": seed,
        "checkpoint_selection_metric": SELECTION_METRIC,
    }
    for key, expected in identity.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Checkpoint {key} mismatch in {checkpoint_path}")
    if bool(checkpoint.get("test_data_loaded_during_training", True)):
        raise ValueError(f"Checkpoint reports test access during training: {checkpoint_path}")
    if tuple(checkpoint["history_feature_columns"]) != tuple(HISTORY_FEATURE_COLS):
        raise ValueError("History feature contract mismatch")
    if tuple(checkpoint["future_control_columns"]) != tuple(FUTURE_CONTROL_COLS):
        raise ValueError("Future-control feature contract mismatch")

    history_scaler = standardizer(checkpoint["history_scaler"])
    control_scaler = standardizer(checkpoint["control_scaler"])
    target_scaler = standardizer(checkpoint["target_scaler"])
    history_scaled = history_scaler.transform(
        test_frame[list(HISTORY_FEATURE_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    controls_scaled = control_scaler.transform(
        test_frame[list(FUTURE_CONTROL_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    delta_scaled = target_scaler.transform(test_delta).astype(np.float32)
    dataset = HorizonControlWindowDataset(
        history_scaled,
        controls_scaled,
        delta_scaled,
        np.ones(len(test_frame), dtype=np.float32),
        test_ends,
        LOOKBACK,
        PREDICT_STEPS,
    )
    loader_args = SimpleNamespace(
        seed=seed, batch_size=args.batch_size, num_workers=args.num_workers
    )
    loader = make_loader(dataset, loader_args, shuffle=False, seed_offset=100)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = build_model(
        model_config,
        HISTORY_FEATURE_COLS,
        FUTURE_CONTROL_COLS,
        PREDICT_STEPS,
        history_scaler.mean,
        history_scaler.scale,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    predicted_delta, predicted_ends = predict_delta(
        model, loader, device, target_scaler
    )
    if not np.array_equal(predicted_ends, test_ends):
        raise AssertionError("Test prediction order changed")

    current = test_current[test_ends]
    actual = test_future[test_ends]
    actual_delta = test_delta[test_ends]
    predicted = current + predicted_delta
    metrics = regression_metrics(actual, predicted)
    delta_metrics = regression_metrics(actual_delta, predicted_delta)
    persistence = regression_metrics(actual, current)
    summary_row: dict[str, object] = {
        "model": spec["model"],
        "model_label": spec["label"],
        "model_description": model_description(str(spec["model"])),
        "paper_role": spec["role"],
        "config_id": spec["config_id"],
        "seed": seed,
        "lookback": LOOKBACK,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "test_full_rmse_k": float(metrics["rmse_k"]),
        "test_full_mae_k": float(metrics["mae_k"]),
        "test_full_p95_absolute_error_k": float(metrics["p95_absolute_error_k"]),
        "test_full_max_absolute_error_k": float(metrics["max_absolute_error_k"]),
        "test_full_bias_k": float(metrics["bias_k"]),
        "test_full_r2": float(metrics["r2"]),
        "test_direction_accuracy": direction_accuracy(actual_delta, predicted_delta),
        "test_delta_rmse_k": float(delta_metrics["rmse_k"]),
        "test_windows": int(metrics["n_windows"]),
        "persistence_test_rmse_k": float(persistence["rmse_k"]),
        "validation_rmse_k": float(development["validation_metrics"]["rmse_k"]),
        "trainable_parameters": int(development["trainable_parameters"]),
        "checkpoint_selection_metric": development["checkpoint_selection_metric"],
    }
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_type": "frozen-checkpoint inference on complete test sequence",
        "training_or_weight_updates": False,
        "test_parent": "0715-BACK",
        "test_variant": "Original",
        "summary_row": summary_row,
        "full_test_metrics": metrics,
        "delta_test_metrics": delta_metrics,
        "persistence_test_metrics": persistence,
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if save_predictions:
        pd.DataFrame(
            {
                "frame_row_index": test_ends,
                "file_id": test_frame.iloc[test_ends]["file_id"].to_numpy(),
                "source_row_index": test_frame.iloc[test_ends]["source_row_index"].to_numpy(),
                "source_timestamp": test_frame.iloc[test_ends]["source_timestamp"].to_numpy(),
                "current_Thv_k": current,
                f"actual_delta_Thv_{PREDICT_STEPS}step_k": actual_delta,
                f"predicted_delta_Thv_{PREDICT_STEPS}step_k": predicted_delta,
                f"actual_Future_Thv_{PREDICT_STEPS}step_k": actual,
                f"predicted_Future_Thv_{PREDICT_STEPS}step_k": predicted,
                "residual_k": predicted - actual,
            }
        ).to_csv(predictions_path, index=False)
    print(
        f"TEST COMPLETE {spec['model']}/seed_{seed} | "
        f"full_rmse_k={metrics['rmse_k']:.8f}",
        flush=True,
    )
    return summary_row


def summarize_test(runs: pd.DataFrame) -> pd.DataFrame:
    metric_names = (
        "test_full_rmse_k",
        "test_full_mae_k",
        "test_full_p95_absolute_error_k",
        "test_full_max_absolute_error_k",
        "test_full_bias_k",
        "test_full_r2",
        "test_direction_accuracy",
    )
    rows: list[dict[str, object]] = []
    for model, group in runs.groupby("model", sort=False):
        row: dict[str, object] = {
            "model": model,
            "model_label": group["model_label"].iloc[0],
            "paper_role": group["paper_role"].iloc[0],
            "config_id": group["config_id"].iloc[0],
            "n_seeds": int(group["seed"].nunique()),
            "seed_set": ";".join(map(str, sorted(group["seed"].unique()))),
            "trainable_parameters": int(group["trainable_parameters"].iloc[0]),
            "test_windows": int(group["test_windows"].iloc[0]),
            "persistence_test_rmse_k": float(group["persistence_test_rmse_k"].iloc[0]),
        }
        for metric_name in metric_names:
            values = interval(group[metric_name].to_numpy(dtype=np.float64))
            for key in ("mean", "sd", "ci95_low", "ci95_high"):
                row[f"{metric_name}_{key}"] = values[key]
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["test_full_rmse_k_mean", "test_full_rmse_k_sd"]
    ).reset_index(drop=True)
    result.insert(0, "full_test_rank", np.arange(1, len(result) + 1))
    return result


def paired_test_row(
    runs: pd.DataFrame, left_model: str, right_model: str, comparison_id: str
) -> dict[str, object]:
    left = runs.loc[
        runs["model"].eq(left_model), ["seed", "test_full_rmse_k"]
    ]
    right = runs.loc[
        runs["model"].eq(right_model), ["seed", "test_full_rmse_k"]
    ]
    matched = left.merge(
        right, on="seed", suffixes=("_left", "_right"), validate="one_to_one"
    )
    differences = (
        matched["test_full_rmse_k_left"] - matched["test_full_rmse_k_right"]
    ).to_numpy(dtype=np.float64)
    effect = interval(differences)
    p_raw = (
        1.0
        if np.allclose(differences, 0.0)
        else float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
    )
    right_mean = float(matched["test_full_rmse_k_right"].mean())
    return {
        "comparison_id": comparison_id,
        "difference_definition": (
            f"{left_model} minus {right_model}; negative favors {left_model}"
        ),
        "left_model": left_model,
        "right_model": right_model,
        "n_paired_seeds": int(len(matched)),
        "paired_seeds": ";".join(map(str, matched["seed"].tolist())),
        "left_matched_mean_k": float(matched["test_full_rmse_k_left"].mean()),
        "right_matched_mean_k": right_mean,
        "mean_difference_k": effect["mean"],
        "relative_difference_vs_right_pct": (
            100.0 * float(effect["mean"]) / right_mean
        ),
        "sd_difference_k": effect["sd"],
        "ci95_low_k": effect["ci95_low"],
        "ci95_high_k": effect["ci95_high"],
        "left_wins": int(np.sum(differences < 0.0)),
        "wilcoxon_p_raw": p_raw,
    }


def paired_test_table(
    runs: pd.DataFrame, comparisons: list[tuple[str, str, str]]
) -> pd.DataFrame:
    result = pd.DataFrame(
        [
            paired_test_row(runs, left, right, comparison_id)
            for comparison_id, left, right in comparisons
        ]
    )
    result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p_raw"])
    return result


def paired_test_tables(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    versus_gru = paired_test_table(
        runs,
        [
            (f"{spec['model']}_vs_gru", str(spec["model"]), "gru_baseline")
            for spec in MODEL_SPECS
            if spec["model"] != "gru_baseline"
        ],
    )
    versus_lstm = paired_test_table(
        runs,
        [
            (f"{spec['model']}_vs_lstm", str(spec["model"]), "lstm_baseline")
            for spec in MODEL_SPECS
            if spec["model"] != "lstm_baseline"
        ],
    )
    structural = paired_test_table(runs, list(STRUCTURAL_CONTRASTS))
    return versus_gru, versus_lstm, structural


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    save_prediction_seeds = parse_seed_set(args.save_prediction_seeds)
    validate_all_training_complete(args.results_dir)

    label_col = label_column(PREDICT_STEPS)
    train_frame, validation_frame = prepare_horizon_development_frames(
        args.data_dir, PREDICT_STEPS, label_col
    )
    test_frame, test_label_audit = prepare_full_test_frame(
        args.data_dir,
        train_frame,
        validation_frame,
        PREDICT_STEPS,
        label_col,
    )
    test_ends = build_horizon_window_ends(
        test_frame, COMMON_ORIGIN_LOOKBACK, PREDICT_STEPS
    )
    assert_horizon_window_boundaries(
        test_frame, test_ends, LOOKBACK, PREDICT_STEPS
    )
    test_current = test_frame[TARGET_COL].to_numpy(dtype=np.float64)
    test_future = test_frame[label_col].to_numpy(dtype=np.float64)
    test_delta = test_future - test_current
    device = choose_device()
    print(f"DEVICE={device}")
    print("TEST_PARENT=0715-BACK")
    print("TEST_VARIANT=Original")
    print(f"COMMON_TEST_ORIGINS={len(test_ends)}")

    rows = [
        evaluate_one(
            args,
            spec,
            seed,
            test_frame,
            test_ends,
            test_current,
            test_future,
            test_delta,
            device,
            seed in save_prediction_seeds,
        )
        for spec, seed in all_tasks()
    ]
    runs = pd.DataFrame(rows).sort_values(["model", "seed"]).reset_index(drop=True)
    summary = summarize_test(runs)
    versus_gru, versus_lstm, structural = paired_test_tables(runs)
    output_dir = args.results_dir / "full_test_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "test_seed_runs.csv", index=False)
    summary.to_csv(output_dir / "test_model_summary.csv", index=False)
    versus_gru.to_csv(output_dir / "paired_test_vs_gru.csv", index=False)
    versus_lstm.to_csv(output_dir / "paired_test_vs_lstm.csv", index=False)
    structural.to_csv(
        output_dir / "paired_test_structural_effects.csv", index=False
    )
    audit = {
        **protocol_payload(),
        "evaluation_type": "frozen-checkpoint inference",
        "training_or_weight_updates_during_test": False,
        "test_parent": "0715-BACK",
        "test_variant": "Original",
        "test_label_audit": test_label_audit,
        "common_test_origins": int(len(test_ends)),
        "observed_model_runs": int(len(runs)),
        "complete": len(runs) == len(all_tasks()),
        "primary_test_metric": "complete-test RMSE",
        "rapid_subset_used_for_test_ranking": False,
        "saved_prediction_seeds": sorted(save_prediction_seeds),
        "methodological_note": (
            "0715-BACK was previously inspected for lookback comparison, so it is a "
            "post-hoc evaluation set rather than a pristine untouched holdout."
        ),
    }
    (output_dir / "test_evaluation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("FULL_TEST_SUMMARY_COMPLETE=true")
    print(f"FULL_TEST_SUMMARY={output_dir / 'test_model_summary.csv'}")


if __name__ == "__main__":
    main()
