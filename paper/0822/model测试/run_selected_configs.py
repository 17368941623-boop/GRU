#!/usr/bin/env python3
"""Train validation-selected model configurations for confirmation seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PROGRAM = SCRIPT_DIR / "train_model_ablation.py"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected configurations for additional seeds."
    )
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument(
        "--common-origin-lookback",
        type=int,
        default=None,
        help="Defaults to the value recorded in selected_configs.json.",
    )
    parser.add_argument("--predict-steps", type=int, default=15)
    parser.add_argument("--seeds", default="82")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(sorted({int(item.strip()) for item in text.split(",") if item.strip()}))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


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
        raise ValueError("Selected configuration file is not test-blind")
    if int(selected["lookback"]) != args.lookback:
        raise ValueError("Selected configuration lookback mismatch")
    if int(selected["predict_steps"]) != args.predict_steps:
        raise ValueError("Selected configuration horizon mismatch")
    selected_common_origin = int(selected["common_origin_lookback"])
    if args.common_origin_lookback is None:
        args.common_origin_lookback = selected_common_origin
    if args.common_origin_lookback != selected_common_origin:
        raise ValueError("Selected configuration common-origin mismatch")
    if args.shard_count < 1:
        raise ValueError("shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")

    models: dict[str, dict[str, object]] = selected["models"]
    all_jobs = [
        (model, selection, seed)
        for model, selection in models.items()
        for seed in seeds
    ]
    shard_jobs = [
        (global_index, model, selection, seed)
        for global_index, (model, selection, seed) in enumerate(all_jobs)
        if global_index % args.shard_count == args.shard_index
    ]
    for local_index, (global_index, model, selection, seed) in enumerate(
        shard_jobs, start=1
    ):
        config_id = str(selection["config_id"])
        config: dict[str, object] = selection["model_config"]
        destination = (
            study_dir
            / model
            / config_id
            / f"seed_{seed}"
            / "metrics.json"
        )
        prefix = (
            f"[shard {args.shard_index + 1}/{args.shard_count} | "
            f"{local_index}/{len(shard_jobs)} | global "
            f"{global_index + 1}/{len(all_jobs)}]"
        )
        if destination.exists() and not args.overwrite:
            print(
                f"{prefix} SKIP existing {model}/{config_id}/seed_{seed}",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(TRAIN_PROGRAM),
            "--model", model,
            "--config-id", config_id,
            "--lookback", str(args.lookback),
            "--common-origin-lookback", str(args.common_origin_lookback),
            "--predict-steps", str(args.predict_steps),
            "--seed", str(seed),
            "--results-dir", str(args.results_dir),
            "--history-hidden", str(config["history_hidden"]),
            "--graph-hidden", str(config["graph_hidden"]),
            "--graph-sweeps", str(config["graph_sweeps"]),
            "--edge-hidden", str(config["edge_hidden"]),
            "--kan-grid", str(config["kan_grid"]),
            "--tcn-levels", str(config["tcn_levels"]),
            "--tcn-kernel", str(config["tcn_kernel"]),
            "--control-hidden", str(config["control_hidden"]),
            "--fusion-hidden", str(config["fusion_hidden"]),
            "--dropout", str(config["dropout"]),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--patience", str(args.patience),
            "--num-workers", str(args.num_workers),
        ]
        if args.data_dir is not None:
            command.extend(("--data-dir", str(args.data_dir)))
        if args.overwrite:
            command.append("--overwrite")
        print(
            f"{prefix} START selected {model}/{config_id}/seed_{seed}",
            flush=True,
        )
        subprocess.run(command, check=True)
        print(
            f"{prefix} COMPLETE selected {model}/{config_id}/seed_{seed}",
            flush=True,
        )
    print(f"CONFIRMATION_SHARD_INDEX={args.shard_index}")
    print(f"CONFIRMATION_SHARD_COUNT={args.shard_count}")
    print(f"CONFIRMATION_SHARD_RUNS_COMPLETE={len(shard_jobs)}")
    print(f"CONFIRMATION_GLOBAL_RUNS_EXPECTED={len(all_jobs)}")
    print("TEST_DATA_READ=false")


if __name__ == "__main__":
    main()
