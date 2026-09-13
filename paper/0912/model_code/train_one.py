"""Train one Raw54-GRU validation seed; never opens the frozen test process."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data_pipeline import (
    ScalerBundle,
    TrajectoryWindowDataset,
    fit_scalers,
    future_absolute,
    load_cached_frames,
    prepare_split,
    split_metadata,
    valid_window_ends,
)
from metrics import trajectory_metrics
from model import Raw54GRU, parameter_count
from protocol import (
    FUTURE_CONTROL_COLUMNS,
    LOOKBACK,
    PREDICT_STEPS,
    SEEDS,
    cache_dir,
    run_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=cache_dir())
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--rapid-quantile", type=float, default=0.10)
    parser.add_argument("--rapid-weight", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_huber(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    element = F.smooth_l1_loss(prediction, target, reduction="none", beta=1.0)
    return torch.sum(element * weight) / weight.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader, split, scalers: ScalerBundle, device: torch.device):
    model.eval()
    predictions: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for history, controls, _target, _weight, item in loader:
        output = model(
            history.to(device, non_blocking=True),
            controls.to(device, non_blocking=True),
        )
        predictions.append(output.cpu().numpy())
        ids.append(item.numpy())
    item_ids = np.concatenate(ids)
    prediction_scaled = np.concatenate(predictions)
    order = np.argsort(item_ids)
    item_ids = item_ids[order]
    prediction_delta = scalers.target_delta.inverse(prediction_scaled[order])
    current = split.target[split.ends[item_ids]]
    actual = current[:, None] + split.delta[item_ids]
    predicted = current[:, None] + prediction_delta
    metrics = trajectory_metrics(
        actual,
        predicted,
        current,
        scalers.rapid_threshold,
        split.valve_event[item_ids],
    )
    return metrics, actual, predicted, current, item_ids


def main() -> None:
    args = parse_args()
    if args.seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")
    destination = run_dir(args.seed)
    metrics_path = destination / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        print(f"SKIP complete seed {args.seed}: {metrics_path}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    train_frame, validation_frame, raw_columns = load_cached_frames(args.data_dir)
    train_ends = valid_window_ends(train_frame, raw_columns, PREDICT_STEPS)
    validation_ends = valid_window_ends(validation_frame, raw_columns, PREDICT_STEPS)
    scalers = fit_scalers(train_frame, train_ends, raw_columns, PREDICT_STEPS, args.rapid_quantile)
    train_split = prepare_split(
        train_frame, raw_columns, PREDICT_STEPS, scalers, args.rapid_weight
    )
    validation_split = prepare_split(
        validation_frame, raw_columns, PREDICT_STEPS, scalers, args.rapid_weight
    )
    if not np.array_equal(train_ends, train_split.ends):
        raise AssertionError("Train window ends changed during preparation")
    if not np.array_equal(validation_ends, validation_split.ends):
        raise AssertionError("Validation window ends changed during preparation")

    train_dataset = TrajectoryWindowDataset(train_split)
    validation_dataset = TrajectoryWindowDataset(validation_split)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = Raw54GRU(len(raw_columns), len(FUTURE_CONTROL_COLUMNS), PREDICT_STEPS).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_state = None
    best_epoch = 0
    best_rmse = float("inf")
    stale = 0
    history_rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_numerator = 0.0
        train_denominator = 0.0
        for history, controls, target, weight, _item in train_loader:
            history = history.to(device, non_blocking=True)
            controls = controls.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history, controls)
            loss = weighted_huber(prediction, target, weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_weight = float(weight.sum().item())
            train_numerator += float(loss.item()) * batch_weight
            train_denominator += batch_weight
        validation_metrics, *_ = evaluate(
            model, validation_loader, validation_split, scalers, device
        )
        validation_rmse = float(validation_metrics["trajectory_rmse_k"])
        scheduler.step(validation_rmse)
        row = {
            "epoch": epoch,
            "train_weighted_huber": train_numerator / max(train_denominator, 1.0),
            "validation_trajectory_rmse_k": validation_rmse,
            "validation_final_rmse_k": float(validation_metrics["final_horizon_rmse_k"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)
        print(json.dumps(row), flush=True)
        if validation_rmse < best_rmse - args.min_delta:
            best_rmse = validation_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No checkpoint was produced")
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    validation_metrics, actual, predicted, current, item_ids = evaluate(
        model, validation_loader, validation_split, scalers, device
    )
    metadata = split_metadata(validation_split).iloc[item_ids].reset_index(drop=True)
    np.savez_compressed(
        destination / "validation_predictions.npz",
        actual_k=actual,
        predicted_k=predicted,
        current_k=current,
        frame_row_index=validation_split.ends[item_ids],
    )
    metadata.to_csv(destination / "validation_prediction_index.csv", index=False)
    pd.DataFrame(history_rows).to_csv(destination / "training_history.csv", index=False)
    normalization = []
    for branch, names, scaler in (
        ("history_raw54", raw_columns, scalers.history),
        ("future_control", FUTURE_CONTROL_COLUMNS, scalers.controls),
    ):
        normalization.extend(
            {"branch": branch, "feature": name, "train_mean": mean, "train_scale": scale}
            for name, mean, scale in zip(names, scaler.mean, scaler.scale)
        )
    pd.DataFrame(normalization).to_csv(destination / "normalization_report.csv", index=False)
    checkpoint = {
        "state_dict": best_state,
        "experiment": "raw54_gru",
        "seed": args.seed,
        "lookback": LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "raw_columns": list(raw_columns),
        "future_control_columns": list(FUTURE_CONTROL_COLUMNS),
        "scalers": scalers.payload(),
        "best_epoch": best_epoch,
        "parameter_count": parameter_count(model),
    }
    torch.save(checkpoint, destination / "best_model.pt")
    metrics = {
        "experiment": "raw54_gru",
        "description": "54 raw histories + untransformed future-control GRU; no engineered features/PCMCI/GNN/KAN",
        "seed": args.seed,
        "lookback": LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "best_epoch": best_epoch,
        "selection_metric": "validation_trajectory_rmse_k",
        "training_seconds": training_seconds,
        "parameter_count": parameter_count(model),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "train_parent_count": 7,
        "train_sources_include_augmentation": True,
        "excluded_0617": False,
        "validation_source": "Original 260428",
        "frozen_test_data_loaded": False,
        "history_columns": list(raw_columns),
        "future_input_columns": list(FUTURE_CONTROL_COLUMNS),
        "future_noncontrol_measurements_used": False,
        "future_label_used_as_input": False,
        "output": "direct Thv(t+1..t+15)-Thv(t) trajectory",
        "validation": validation_metrics,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
