#!/usr/bin/env python3
"""Run one deterministic shard of the 8 x 10 development grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from protocol import DEFAULT_PREDICT_STEPS, EXPERIMENTS, SEEDS, project_dir, run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=5)
    parser.add_argument("--predict-steps", type=int, default=DEFAULT_PREDICT_STEPS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=project_dir() / "processed_data")
    parser.add_argument("--output-dir", type=Path, default=project_dir() / "outputs")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def completed(path: Path, experiment: str, seed: int, horizon: int) -> bool:
    metrics = path / "metrics.json"
    checkpoint = path / "best_model.pt"
    if not metrics.exists() or not checkpoint.exists():
        return False
    try:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("experiment") == experiment and payload.get("seed") == seed
        and payload.get("predict_steps") == horizon and payload.get("test_data_loaded") is False
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be within [0, worker-count)")
    tasks = [(experiment, seed) for experiment in EXPERIMENTS for seed in SEEDS]
    selected = [task for index, task in enumerate(tasks) if index % args.worker_count == args.worker_index]
    script = Path(__file__).with_name("train_one.py")
    for experiment, seed in selected:
        destination = run_dir(args.output_dir, experiment, args.predict_steps, seed)
        if completed(destination, experiment, seed, args.predict_steps):
            print(f"skip completed {experiment} seed={seed}", flush=True)
            continue
        command = [
            sys.executable, str(script), "--experiment", experiment, "--seed", str(seed),
            "--predict-steps", str(args.predict_steps), "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--patience", str(args.patience),
            "--num-workers", str(args.num_workers),
            "--data-dir", str(args.data_dir), "--output-dir", str(args.output_dir),
            "--device", args.device,
        ]
        # A killed process can leave a directory or a partial metrics/checkpoint
        # pair. The shard runner has already established that it is incomplete,
        # so it is safe to rebuild that one run on restart.
        if destination.exists():
            command.append("--overwrite")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
