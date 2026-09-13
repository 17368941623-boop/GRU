#!/usr/bin/env python3
"""Train one development run without opening either test file."""

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
from torch.utils.data import DataLoader, Subset

from data_pipeline import (
    ScalerBundle,
    TrajectoryWindowDataset,
    exclude_0617,
    fit_scalers,
    future_absolute,
    load_frame,
    prepare_split,
    split_metadata,
    valid_window_ends,
)
from graph_tools import bootstrap_physical_graphs, load_edges
from metrics import trajectory_metrics
from models import build_model, parameter_count
from protocol import (
    DEFAULT_PREDICT_STEPS,
    EXPERIMENTS,
    FUTURE_CONTROL_COLUMNS,
    HISTORY_COLUMNS,
    LOOKBACK,
    SEEDS,
    generated_graph_dir,
    project_dir,
    run_dir,
)


def parse_args() -> argparse.Namespace:
    root = project_dir()
    parser = argparse.ArgumentParser(description="Train one parallel GRU-KAN-GNN trajectory model.")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--predict-steps", type=int, default=DEFAULT_PREDICT_STEPS)
    parser.add_argument("--data-dir", type=Path, default=root / "processed_data")
    parser.add_argument("--graph-dir", type=Path, default=generated_graph_dir())
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--rapid-quantile", type=float, default=0.10)
    parser.add_argument("--rapid-weight", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="auto",
        help="Torch device, for example auto, cpu, cuda, cuda:0, cuda:1, or mps.",
    )
    parser.add_argument("--max-train-windows", type=int, default=0, help="Smoke/debug only; 0 uses all windows.")
    parser.add_argument("--max-val-windows", type=int, default=0, help="Smoke/debug only; 0 uses all windows.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    normalized = name.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid torch device: {name!r}") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
        index = 0 if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {index} is unavailable; found {torch.cuda.device_count()} GPU(s)"
            )
        return torch.device(f"cuda:{index}")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("An MPS device was requested, but MPS is unavailable")
    if device.type not in {"cpu", "mps"}:
        raise ValueError(f"Unsupported device type: {device.type}")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, args: argparse.Namespace, shuffle: bool, offset: int) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed + offset)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0, generator=generator,
    )


def deterministic_subset(dataset: TrajectoryWindowDataset, limit: int) -> TrajectoryWindowDataset | Subset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    indices = np.linspace(0, len(dataset) - 1, limit).round().astype(int)
    return Subset(dataset, indices.tolist())


def weighted_huber(predicted: torch.Tensor, actual: torch.Tensor, weight: torch.Tensor, beta: float) -> torch.Tensor:
    element = F.smooth_l1_loss(predicted, actual, reduction="none", beta=beta)
    return torch.sum(element * weight) / weight.sum().clamp_min(1.0)


@torch.no_grad()
def predict_scaled(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    huber_beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    item_ids: list[np.ndarray] = []
    loss_numerator = 0.0
    loss_denominator = 0.0
    for history, controls, target, weight, item in loader:
        history = history.to(device, non_blocking=True)
        controls = controls.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        output = model(history, controls)
        element = F.smooth_l1_loss(output, target, reduction="none", beta=huber_beta)
        loss_numerator += float(torch.sum(element * weight).item())
        loss_denominator += float(weight.sum().item())
        predictions.append(output.cpu().numpy())
        item_ids.append(item.numpy())
    ids = np.concatenate(item_ids)
    prediction = np.concatenate(predictions)
    order = np.argsort(ids)
    return prediction[order], ids[order], float(loss_numerator / max(loss_denominator, 1.0))


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    split,
    scalers: ScalerBundle,
    device: torch.device,
    huber_beta: float = 1.0,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    predicted_scaled, item_ids, loss = predict_scaled(
        model, loader, device, huber_beta=huber_beta
    )
    predicted_delta = scalers.target_delta.inverse(predicted_scaled)
    actual_delta = split.delta[item_ids]
    current = split.target[split.ends[item_ids]]
    actual = current[:, None] + actual_delta
    predicted = current[:, None] + predicted_delta
    event = None
    if "valve_event_mask" in split.frame.columns:
        event = split.frame.iloc[split.ends[item_ids]]["valve_event_mask"].to_numpy(dtype=bool)
    result = trajectory_metrics(actual, predicted, current, scalers.rapid_threshold, event)
    result["weighted_huber_scaled"] = loss
    return result, predicted, item_ids


def graph_for_experiment(args: argparse.Namespace) -> list[dict[str, object]] | None:
    spec = EXPERIMENTS[args.experiment]
    graph_name = spec["graph"]
    if graph_name is None:
        return None
    graph_path = args.graph_dir / str(graph_name)
    if not graph_path.exists() and graph_name == "physical_all_lags.json":
        bootstrap_physical_graphs(
            project_dir() / "graphs" / "static_physical_graph.json", args.graph_dir
        )
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Missing {graph_path}. Run discover_causal_graph.py before this experiment."
        )
    return load_edges(graph_path)


def main() -> None:
    args = parse_args()
    if args.seed not in SEEDS:
        print(f"warning: seed {args.seed} is outside the frozen ten-seed set")
    if args.predict_steps < 1 or args.predict_steps > 30:
        raise ValueError("predict-steps must be in [1, 30] for this dataset/protocol")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size and patience must be positive")
    if args.num_workers < 0 or args.huber_beta <= 0.0:
        raise ValueError("num-workers must be non-negative and huber-beta must be positive")
    set_seed(args.seed)
    destination = run_dir(args.output_dir, args.experiment, args.predict_steps, args.seed)
    metrics_path = destination / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Run already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    # Deliberately name only development files here. This module never resolves a test path.
    train_frame = exclude_0617(load_frame(args.data_dir / "train_clean.pkl"))
    validation_frame = load_frame(args.data_dir / "val_clean.pkl")
    train_ends = valid_window_ends(train_frame, args.predict_steps)
    validation_ends = valid_window_ends(validation_frame, args.predict_steps)
    scalers = fit_scalers(train_frame, train_ends, args.predict_steps, args.rapid_quantile)
    train_split = prepare_split(train_frame, args.predict_steps, scalers, args.rapid_weight)
    validation_split = prepare_split(validation_frame, args.predict_steps, scalers, args.rapid_weight)
    if not np.array_equal(train_ends, train_split.ends) or not np.array_equal(validation_ends, validation_split.ends):
        raise AssertionError("Window construction changed between scaler fitting and dataset creation")

    train_dataset = deterministic_subset(TrajectoryWindowDataset(train_split), args.max_train_windows)
    validation_dataset = deterministic_subset(TrajectoryWindowDataset(validation_split), args.max_val_windows)
    train_loader = make_loader(train_dataset, args, True, 0)
    validation_loader = make_loader(validation_dataset, args, False, 1)
    edges = graph_for_experiment(args)
    spec = EXPERIMENTS[args.experiment]
    model = build_model(
        args.predict_steps, scalers.history.mean, scalers.history.scale,
        scalers.controls.mean, scalers.controls.scale,
        edges, str(spec["gate"]),
    )
    device = choose_device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_state = None
    best_epoch = 0
    best_rmse = float("inf")
    stale = 0
    history_rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight = 0.0
        for history, controls, target, weight, _ in train_loader:
            history = history.to(device, non_blocking=True)
            controls = controls.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(history, controls)
            loss = weighted_huber(output, target, weight, args.huber_beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_weight = float(weight.sum().item())
            train_loss_sum += float(loss.item()) * batch_weight
            train_weight += batch_weight
        validation_metrics, _, _ = evaluate(
            model, validation_loader, validation_split, scalers, device, args.huber_beta
        )
        validation_rmse = float(validation_metrics["trajectory_rmse_k"])
        scheduler.step(validation_rmse)
        row = {
            "epoch": epoch, "train_weighted_huber": train_loss_sum / train_weight,
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
        raise RuntimeError("Training never produced a checkpoint")
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    validation_metrics, prediction, item_ids = evaluate(
        model, validation_loader, validation_split, scalers, device, args.huber_beta
    )
    metadata = split_metadata(validation_split).iloc[item_ids].reset_index(drop=True)
    actual_all = future_absolute(validation_split)[item_ids]
    current = validation_split.target[validation_split.ends[item_ids]]
    np.savez_compressed(
        destination / "validation_predictions.npz", actual_k=actual_all,
        predicted_k=prediction, current_k=current,
        frame_row_index=validation_split.ends[item_ids],
    )
    metadata.to_csv(destination / "validation_prediction_index.csv", index=False)
    pd.DataFrame(history_rows).to_csv(destination / "training_history.csv", index=False)
    scaler_rows = []
    for branch, names, scaler in (
        ("history", HISTORY_COLUMNS, scalers.history),
        ("future_control", FUTURE_CONTROL_COLUMNS, scalers.controls),
    ):
        for name, mean, scale in zip(names, scaler.mean, scaler.scale):
            scaler_rows.append({"branch": branch, "feature": name, "train_mean": mean, "train_scale": scale})
    pd.DataFrame(scaler_rows).to_csv(destination / "normalization_report.csv", index=False)

    checkpoint = {
        "state_dict": best_state,
        "experiment": args.experiment,
        "experiment_spec": spec,
        "seed": args.seed,
        "predict_steps": args.predict_steps,
        "lookback": LOOKBACK,
        "history_columns": list(HISTORY_COLUMNS),
        "scalers": scalers.payload(),
        "edges": edges,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count(model),
    }
    torch.save(checkpoint, destination / "best_model.pt")
    metrics = {
        "experiment": args.experiment, "description": spec["description"],
        "seed": args.seed, "lookback": LOOKBACK, "predict_steps": args.predict_steps,
        "best_epoch": best_epoch, "selection_metric": "validation_trajectory_rmse_k",
        "training_seconds": training_seconds, "parameter_count": parameter_count(model),
        "train_windows": len(train_dataset), "validation_windows": len(validation_dataset),
        "train_sources_include_augmentation": True, "excluded_0617": True,
        "validation_source": "Original 260501", "test_data_loaded": False,
        "future_input_columns": list(FUTURE_CONTROL_COLUMNS),
        "future_noncontrol_measurements_used": False,
        "future_label_used_as_input": False,
        "output": "direct Thv(t+1..t+H)-Thv(t) trajectory",
        "validation": validation_metrics,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
