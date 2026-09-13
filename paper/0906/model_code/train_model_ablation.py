#!/usr/bin/env python3
"""Train one Thv model-ablation configuration on train/validation only.

This program deliberately never opens test_clean.pkl or test_full_clean.pkl.
Hyperparameters and checkpoints are selected with Original 260501 validation;
the separate evaluate_selected_models.py program is the only test reader.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BASELINE_DIR = PROJECT_DIR / "shared"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from train_thv_delta_lstm_lookback import (  # noqa: E402
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
    make_loader,
    predict_delta,
    regression_metrics,
    save_scalers,
    save_training_plot,
    save_validation_plots,
    set_seed,
    train_model,
)
from train_thv_delta_rnn_compare import (  # noqa: E402
    assert_horizon_build_config,
    assert_horizon_window_boundaries,
    audit_horizon_label_alignment,
    build_horizon_window_ends,
    build_normalization_report,
    build_segment_metrics,
    count_trainable_parameters,
    delta_name,
    label_column,
    prepare_horizon_development_frames,
    synchronize_device,
)

from model_components import (  # noqa: E402
    MODEL_NAMES,
    ModelConfig,
    build_model,
    model_description,
)


DEFAULT_DATA_DIR = PROJECT_DIR / "processed_data"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "outputs"
ALLOWED_PREDICT_STEPS = (5, 10, 15, 20, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one leakage-safe Thv model-ablation configuration."
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--lookback", required=True, type=int)
    parser.add_argument(
        "--common-origin-lookback",
        type=int,
        default=None,
        help=(
            "Fixed origin history used to keep labels identical across models. "
            "Defaults to --lookback."
        ),
    )
    parser.add_argument(
        "--predict-steps", type=int, default=15, choices=ALLOWED_PREDICT_STEPS
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    parser.add_argument("--history-hidden", type=int, default=64)
    parser.add_argument("--graph-hidden", type=int, default=64)
    parser.add_argument("--graph-sweeps", type=int, default=1, choices=(1, 2))
    parser.add_argument("--edge-hidden", type=int, default=32)
    parser.add_argument("--kan-grid", type=int, default=8, choices=(5, 8, 12))
    parser.add_argument(
        "--tcn-levels", type=int, default=3, choices=(2, 3, 4, 5, 6, 7)
    )
    parser.add_argument("--tcn-kernel", type=int, default=3, choices=(2, 3, 5))
    parser.add_argument("--control-hidden", type=int, default=32)
    parser.add_argument("--fusion-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rapid-quantile", type=float, default=0.10)
    parser.add_argument("--rapid-weight", type=float, default=3.0)
    parser.add_argument(
        "--selection-metric",
        choices=("validation_rmse_k", "validation_rapid_rmse_k"),
        default="validation_rmse_k",
        help="Validation metric used for scheduling, early stopping and checkpoint selection.",
    )
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.lookback < 2:
        raise ValueError("lookback must be at least 2 for valve-action differences")
    if args.lookback > args.common_origin_lookback:
        raise ValueError("lookback cannot exceed common-origin-lookback")
    if not 0.0 < args.rapid_quantile < 1.0:
        raise ValueError("rapid-quantile must be between 0 and 1")
    if args.rapid_weight <= 1.0:
        raise ValueError("rapid-weight must exceed 1")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size and patience must be positive")
    if any(character in args.config_id for character in "/\\:"):
        raise ValueError("config-id cannot contain path separators or ':'")


def configuration_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        model_name=args.model,
        history_hidden=args.history_hidden,
        graph_hidden=args.graph_hidden,
        graph_sweeps=args.graph_sweeps,
        edge_hidden=args.edge_hidden,
        kan_grid=args.kan_grid,
        tcn_levels=args.tcn_levels,
        tcn_kernel=args.tcn_kernel,
        control_hidden=args.control_hidden,
        fusion_hidden=args.fusion_hidden,
        dropout=args.dropout,
    )


def main() -> None:
    args = parse_args()
    if args.common_origin_lookback is None:
        args.common_origin_lookback = args.lookback
    validate_args(args)
    set_seed(args.seed)
    label_col = label_column(args.predict_steps)
    target_delta_name = delta_name(args.predict_steps)
    assert_horizon_build_config(args.data_dir, args.predict_steps, label_col)

    run_dir = (
        args.results_dir
        / "development"
        / f"horizon_{args.predict_steps:02d}"
        / f"lookback_{args.lookback:02d}"
        / args.model
        / args.config_id
        / f"seed_{args.seed}"
    )
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run already exists: {run_dir}. Use --overwrite to replace it."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_frame, validation_frame = prepare_horizon_development_frames(
        args.data_dir, args.predict_steps, label_col
    )
    train_label_audit = audit_horizon_label_alignment(
        train_frame, "training", args.predict_steps, label_col
    )
    validation_label_audit = audit_horizon_label_alignment(
        validation_frame, "validation", args.predict_steps, label_col
    )
    train_ends = build_horizon_window_ends(
        train_frame, args.common_origin_lookback, args.predict_steps
    )
    validation_ends = build_horizon_window_ends(
        validation_frame, args.common_origin_lookback, args.predict_steps
    )
    assert_horizon_window_boundaries(
        train_frame, train_ends, args.lookback, args.predict_steps
    )
    assert_horizon_window_boundaries(
        validation_frame, validation_ends, args.lookback, args.predict_steps
    )

    train_current = train_frame[TARGET_COL].to_numpy(dtype=np.float64)
    validation_current = validation_frame[TARGET_COL].to_numpy(dtype=np.float64)
    train_future = train_frame[label_col].to_numpy(dtype=np.float64)
    validation_future = validation_frame[label_col].to_numpy(dtype=np.float64)
    train_delta = train_future - train_current
    validation_delta = validation_future - validation_current

    rapid_threshold_k = float(
        np.quantile(train_delta[train_ends], args.rapid_quantile)
    )
    train_weights = np.ones(len(train_frame), dtype=np.float32)
    validation_weights = np.ones(len(validation_frame), dtype=np.float32)
    train_weights[train_delta <= rapid_threshold_k] = args.rapid_weight
    validation_weights[validation_delta <= rapid_threshold_k] = args.rapid_weight

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
    validation_history = history_scaler.transform(
        validation_frame[list(HISTORY_FEATURE_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    train_controls = control_scaler.transform(
        train_frame[list(FUTURE_CONTROL_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    validation_controls = control_scaler.transform(
        validation_frame[list(FUTURE_CONTROL_COLS)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    train_delta_scaled = target_scaler.transform(train_delta).astype(np.float32)
    validation_delta_scaled = target_scaler.transform(validation_delta).astype(
        np.float32
    )

    from train_thv_delta_rnn_compare import HorizonControlWindowDataset

    train_dataset = HorizonControlWindowDataset(
        train_history,
        train_controls,
        train_delta_scaled,
        train_weights,
        train_ends,
        args.lookback,
        args.predict_steps,
    )
    validation_dataset = HorizonControlWindowDataset(
        validation_history,
        validation_controls,
        validation_delta_scaled,
        validation_weights,
        validation_ends,
        args.lookback,
        args.predict_steps,
    )
    train_loader = make_loader(train_dataset, args, shuffle=True, seed_offset=0)
    train_eval_loader = make_loader(
        train_dataset, args, shuffle=False, seed_offset=1
    )
    validation_loader = make_loader(
        validation_dataset, args, shuffle=False, seed_offset=2
    )

    normalization_report = build_normalization_report(
        train_frame,
        validation_frame,
        train_delta[train_ends],
        validation_delta[validation_ends],
        history_scaler,
        control_scaler,
        target_scaler,
        args.predict_steps,
    )
    normalization_report.to_csv(
        run_dir / "normalization_report.csv", index=False
    )

    model_config = configuration_from_args(args)
    model = build_model(
        model_config,
        HISTORY_FEATURE_COLS,
        FUTURE_CONTROL_COLS,
        args.predict_steps,
        history_scaler.mean,
        history_scaler.scale,
    )
    device = choose_device()
    model = model.to(device)
    parameter_count = count_trainable_parameters(model)

    print(
        f"MODEL={args.model} | CONFIG={args.config_id} | "
        f"LOOKBACK={args.lookback} | HORIZON={args.predict_steps} | "
        f"SEED={args.seed} | DEVICE={device}"
    )
    print(f"DESCRIPTION={model_description(args.model)}")
    print(f"TRAINABLE_PARAMETERS={parameter_count}")
    print(
        f"SELECTION=Original_260501_{args.selection_metric} | "
        "TEST_DATA_OPENED=false"
    )
    print(
        f"RAPID_THRESHOLD_TRAIN_ONLY_K={rapid_threshold_k:.10f} | "
        f"TRAIN_WINDOWS={len(train_dataset)} | "
        f"VALIDATION_WINDOWS={len(validation_dataset)}"
    )

    synchronize_device(device)
    started = time.perf_counter()
    best_state, history, best_epoch, best_selection_rmse = train_model(
        model,
        train_loader,
        train_eval_loader,
        validation_loader,
        target_scaler,
        args,
        device,
    )
    synchronize_device(device)
    training_seconds = float(time.perf_counter() - started)
    model.load_state_dict(best_state)

    predicted_delta, prediction_ends = predict_delta(
        model, validation_loader, device, target_scaler
    )
    if not np.array_equal(prediction_ends, validation_ends):
        raise AssertionError("Validation prediction order changed")
    actual_delta = validation_delta[validation_ends]
    current_temperature = validation_current[validation_ends]
    actual_temperature = validation_future[validation_ends]
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

    prediction_table = pd.DataFrame(
        {
            "frame_row_index": validation_ends,
            "file_id": validation_frame.iloc[validation_ends]["file_id"].to_numpy(),
            "source_group": validation_frame.iloc[validation_ends][
                "source_group"
            ].to_numpy(),
            "source_row_index": validation_frame.iloc[validation_ends][
                "source_row_index"
            ].to_numpy(),
            "source_timestamp": validation_frame.iloc[validation_ends][
                "source_timestamp"
            ].to_numpy(),
            "current_Thv_k": current_temperature,
            f"actual_delta_Thv_{args.predict_steps}step_k": actual_delta,
            f"predicted_delta_Thv_{args.predict_steps}step_k": predicted_delta,
            f"actual_Future_Thv_{args.predict_steps}step_k": actual_temperature,
            f"predicted_Future_Thv_{args.predict_steps}step_k": predicted_temperature,
            "residual_k": predicted_temperature - actual_temperature,
            "rapid_cooling_mask": rapid_mask.astype(np.int8),
        }
    )
    prediction_table.to_csv(run_dir / "validation_predictions.csv", index=False)
    prediction_table.loc[prediction_table["rapid_cooling_mask"] == 1].to_csv(
        run_dir / "rapid_validation_predictions.csv", index=False
    )
    build_segment_metrics(prediction_table, args.predict_steps).to_csv(
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
        validation_frame.iloc[validation_ends]["file_id"].astype(str).to_numpy(),
        run_dir / "validation_prediction.png",
        run_dir / "validation_delta.png",
        f"{args.model} | h={args.predict_steps} | lookback={args.lookback} | seed={args.seed}",
        predict_steps=args.predict_steps,
    )
    save_scalers(
        run_dir / "scalers.npz",
        history_scaler,
        control_scaler,
        target_scaler,
        rapid_threshold_k,
        target_name=target_delta_name,
    )

    best_row = best_epoch_row(history, best_epoch)
    metadata = {
        "experiment": "Thv_h15_lb60_final_architecture_study",
        "model_type": args.model,
        "model_description": model_description(args.model),
        "config_id": args.config_id,
        "model_config": model_config.as_dict(),
        "lookback": args.lookback,
        "common_origin_lookback": args.common_origin_lookback,
        "predict_steps": args.predict_steps,
        "predict_seconds": args.predict_steps * SAMPLE_PERIOD_SECONDS,
        "seed": args.seed,
        "target": target_delta_name,
        "source_label": label_col,
        "history_feature_columns": list(HISTORY_FEATURE_COLS),
        "future_control_columns": list(FUTURE_CONTROL_COLS),
        "future_control_alignment": f"u(t)..u(t+{args.predict_steps - 1})",
        "future_noncontrol_measurements_used": False,
        "future_label_used_as_input": False,
        "training_parents": sorted(EXPECTED_TRAIN_PARENTS),
        "validation_parent": EXPECTED_VALIDATION_PARENT,
        "validation_variant": "Original",
        "held_out_test_parent": EXPECTED_TEST_PARENT_IN_BUILD_CONFIG,
        "test_data_loaded": False,
        "test_metrics_present": False,
        "model_selection_split": "Original 260501 validation only",
        "checkpoint_selection_metric": args.selection_metric,
        "rapid_threshold_k_train_only": rapid_threshold_k,
        "rapid_quantile": args.rapid_quantile,
        "rapid_weight": args.rapid_weight,
        "train_label_alignment": train_label_audit,
        "validation_label_alignment": validation_label_audit,
        "scaler_fit_source": "training only after excluding 0617-ALL",
        "train_windows": int(len(train_dataset)),
        "validation_windows": int(len(validation_dataset)),
        "validation_rapid_windows": int(rapid_mask.sum()),
        "trainable_parameters": parameter_count,
        "training_seconds_total": training_seconds,
        "completed_epochs": len(history),
        "seconds_per_completed_epoch": training_seconds / max(len(history), 1),
        "best_epoch": best_epoch,
        "best_epoch_selection_metric": args.selection_metric,
        "best_epoch_selection_metric_value_k": best_selection_rmse,
        "best_epoch_history": best_row,
        "last_completed_epoch_history": dict(history[-1]),
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
            "model_type": args.model,
            "model_config": model_config.as_dict(),
            "config_id": args.config_id,
            "history_feature_columns": list(HISTORY_FEATURE_COLS),
            "future_control_columns": list(FUTURE_CONTROL_COLS),
            "lookback": args.lookback,
            "common_origin_lookback": args.common_origin_lookback,
            "predict_steps": args.predict_steps,
            "seed": args.seed,
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
            "checkpoint_selection_metric": args.selection_metric,
            "best_epoch": best_epoch,
            "best_epoch_selection_metric_value_k": best_selection_rmse,
            "validation_metrics": validation_metrics,
            "rapid_validation_metrics": rapid_validation_metrics,
            "test_data_loaded_during_training": False,
        },
        run_dir / "best_model.pt",
    )

    print(f"BEST_EPOCH={best_epoch}")
    print(
        f"PRIMARY_FINAL_FULL_VALIDATION_RMSE_K="
        f"{validation_metrics['rmse_k']:.10f}"
    )
    print(f"SECONDARY_RAPID_VALIDATION_RMSE_K={rapid_validation_metrics['rmse_k']:.10f}")
    print("TEST_DATA_OPENED=false")
    print(f"MODEL_SAVED={run_dir / 'best_model.pt'}")
    print(f"RESULTS_SAVED={run_dir}")


if __name__ == "__main__":
    main()
