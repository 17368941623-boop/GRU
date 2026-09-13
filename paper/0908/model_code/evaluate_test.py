#!/usr/bin/env python3
"""Evaluate frozen checkpoints on Original 0715-BACK after development is locked."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline import (
    ScalerBundle,
    TrajectoryWindowDataset,
    future_absolute,
    load_frame,
    prepare_split,
    split_metadata,
)
from metrics import trajectory_metrics
from models import build_model
from protocol import DEFAULT_PREDICT_STEPS, EXPERIMENTS, LOOKBACK, SEEDS, project_dir, run_dir
from train_one import choose_device, predict_scaled


def parse_args() -> argparse.Namespace:
    root = project_dir()
    parser = argparse.ArgumentParser(description="Frozen-model full-test evaluation.")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--predict-steps", type=int, default=DEFAULT_PREDICT_STEPS)
    parser.add_argument("--data-dir", type=Path, default=root / "processed_data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="auto",
        help="Torch device, for example auto, cpu, cuda, cuda:0, cuda:1, or mps.",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def expected_tasks(args: argparse.Namespace) -> list[tuple[str, int]]:
    experiments = (args.experiment,) if args.experiment else tuple(EXPERIMENTS)
    seeds = (args.seed,) if args.seed is not None else SEEDS
    return [(experiment, seed) for experiment in experiments for seed in seeds]


def main() -> None:
    args = parse_args()
    development = args.output_dir / "development" / f"horizon_{args.predict_steps:02d}" / f"lookback_{LOOKBACK:02d}"
    manifest = development / "validation_manifest.json"
    if not manifest.exists() and not args.allow_incomplete:
        raise FileNotFoundError(
            f"{manifest} is absent. Complete all development runs and run summarize.py first; "
            "or pass --allow-incomplete for an explicitly exploratory test."
        )
    tasks = expected_tasks(args)
    checkpoints = []
    for experiment, seed in tasks:
        checkpoint = run_dir(args.output_dir, experiment, args.predict_steps, seed) / "best_model.pt"
        if checkpoint.exists():
            checkpoints.append((experiment, seed, checkpoint))
        elif not args.allow_incomplete:
            raise FileNotFoundError(checkpoint)
    if not checkpoints:
        raise FileNotFoundError("No frozen checkpoints found")

    test_frame = load_frame(args.data_dir / "test_full_clean.pkl")
    device = choose_device(args.device)
    for experiment, seed, checkpoint_path in checkpoints:
        destination = (
            args.output_dir / "final_test" / f"horizon_{args.predict_steps:02d}"
            / f"lookback_{LOOKBACK:02d}" / experiment / f"seed_{seed}"
        )
        metrics_path = destination / "metrics.json"
        if metrics_path.exists() and not args.overwrite:
            print(f"skip completed {experiment} seed={seed}")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint["predict_steps"]) != args.predict_steps or int(checkpoint["lookback"]) != LOOKBACK:
            raise ValueError(f"Protocol mismatch in {checkpoint_path}")
        scalers = ScalerBundle.from_payload(checkpoint["scalers"])
        split = prepare_split(test_frame, args.predict_steps, scalers)
        dataset = TrajectoryWindowDataset(split)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )
        spec = checkpoint["experiment_spec"]
        model = build_model(
            args.predict_steps, scalers.history.mean, scalers.history.scale,
            scalers.controls.mean, scalers.controls.scale,
            checkpoint["edges"], str(spec["gate"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        predicted_scaled, item_ids, _ = predict_scaled(model, loader, device)
        predicted_delta = scalers.target_delta.inverse(predicted_scaled)
        current = split.target[split.ends[item_ids]]
        actual = future_absolute(split)[item_ids]
        predicted = current[:, None] + predicted_delta
        event = None
        if "valve_event_mask" in split.frame.columns:
            event = split.frame.iloc[split.ends[item_ids]]["valve_event_mask"].to_numpy(dtype=bool)
        result = trajectory_metrics(actual, predicted, current, scalers.rapid_threshold, event)
        payload = {
            "experiment": experiment, "seed": seed, "lookback": LOOKBACK,
            "predict_steps": args.predict_steps, "test_source": "Original 0715-BACK",
            "checkpoint": str(checkpoint_path), "frozen_development_manifest": str(manifest),
            "test": result,
        }
        np.savez_compressed(
            destination / "test_predictions.npz", actual_k=actual, predicted_k=predicted,
            current_k=current, frame_row_index=split.ends[item_ids],
        )
        split_metadata(split).iloc[item_ids].reset_index(drop=True).to_csv(
            destination / "test_prediction_index.csv", index=False
        )
        metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
