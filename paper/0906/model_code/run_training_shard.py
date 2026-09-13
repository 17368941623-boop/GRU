#!/usr/bin/env python3
"""Run one non-overlapping shard of the frozen 0906 training protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from study_protocol import (
    COMMON_CONFIG,
    COMMON_ORIGIN_LOOKBACK,
    LOOKBACK,
    PREDICT_STEPS,
    SELECTION_METRIC,
    protocol_payload,
    run_directory,
    tasks_for_group,
    valid_completed_run,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "processed_data"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "outputs_0906"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument(
        "--group",
        choices=("all", "gru_family", "lstm_family"),
        default="all",
        help="Non-overlapping architecture family assigned to this launcher.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def command_for(
    args: argparse.Namespace, spec: dict[str, Any], seed: int
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "train_model_ablation.py"),
        "--model", str(spec["model"]),
        "--config-id", str(spec["config_id"]),
        "--lookback", str(LOOKBACK),
        "--common-origin-lookback", str(COMMON_ORIGIN_LOOKBACK),
        "--predict-steps", str(PREDICT_STEPS),
        "--seed", str(seed),
        "--data-dir", str(args.data_dir),
        "--results-dir", str(args.results_dir),
        "--selection-metric", SELECTION_METRIC,
        "--history-hidden", str(COMMON_CONFIG["history_hidden"]),
        "--graph-hidden", str(COMMON_CONFIG["graph_hidden"]),
        "--graph-sweeps", str(COMMON_CONFIG["graph_sweeps"]),
        "--edge-hidden", str(COMMON_CONFIG["edge_hidden"]),
        "--kan-grid", str(COMMON_CONFIG["kan_grid"]),
        "--tcn-levels", str(COMMON_CONFIG["tcn_levels"]),
        "--tcn-kernel", str(COMMON_CONFIG["tcn_kernel"]),
        "--control-hidden", str(COMMON_CONFIG["control_hidden"]),
        "--fusion-hidden", str(COMMON_CONFIG["fusion_hidden"]),
        "--dropout", str(COMMON_CONFIG["dropout"]),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--patience", str(args.patience),
        "--num-workers", str(args.num_workers),
        "--rapid-quantile", "0.10",
        "--rapid-weight", "3.0",
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> None:
    args = parse_args()
    tasks = tasks_for_group(args.group)
    if args.num_shards < 1 or args.num_shards > len(tasks):
        raise ValueError(f"num-shards must be between 1 and {len(tasks)}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= index < num-shards")
    args.data_dir = args.data_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)

    shard_tasks = [
        task for index, task in enumerate(tasks)
        if index % args.num_shards == args.shard_index
    ]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.results_dir / "study_protocol.json", protocol_payload())
    progress_path = (
        args.results_dir
        / f"shard_{args.group}_{args.shard_index:02d}_progress.json"
    )
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if args.dry_run:
        for spec, seed in shard_tasks:
            print(f"{spec['model']}/{spec['config_id']}/seed_{seed}")
        print(f"TRAINING_GROUP={args.group}")
        print(f"DRY_RUN_TASKS={len(shard_tasks)}")
        return

    started = time.time()
    completed = 0
    for local_index, (spec, seed) in enumerate(shard_tasks, start=1):
        identifier = f"{spec['model']}/{spec['config_id']}/seed_{seed}"
        atomic_json(
            progress_path,
            {
                "status": "training",
                "training_group": args.group,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "completed_in_shard": completed,
                "total_in_shard": len(shard_tasks),
                "current_run": identifier,
                "test_data_read": False,
            },
        )
        if not args.overwrite and valid_completed_run(args.results_dir, spec, seed):
            completed += 1
            print(f"[{local_index}/{len(shard_tasks)}] REUSE {identifier}", flush=True)
            continue

        print(f"[{local_index}/{len(shard_tasks)}] START {identifier}", flush=True)
        log_path = args.results_dir / "logs" / str(spec["model"]) / f"seed_{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
            log_file.write("\nCOMMAND=" + subprocess.list2cmdline(command_for(args, spec, seed)) + "\n")
            subprocess.run(
                command_for(args, spec, seed),
                cwd=SCRIPT_DIR,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not valid_completed_run(args.results_dir, spec, seed):
            raise RuntimeError(f"Post-training audit failed: {identifier}")
        completed += 1
        print(f"[{local_index}/{len(shard_tasks)}] DONE {identifier}", flush=True)

    atomic_json(
        progress_path,
        {
            "status": "complete",
            "training_group": args.group,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "completed_in_shard": completed,
            "total_in_shard": len(shard_tasks),
            "elapsed_seconds": time.time() - started,
            "current_run": None,
            "test_data_read": False,
        },
    )
    print(f"SHARD_COMPLETE={args.shard_index}", flush=True)


if __name__ == "__main__":
    main()
