#!/usr/bin/env python3
"""Run one resumable shard of the frozen 0910 heatmap experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from heatmap_protocol import all_tasks, protocol_payload, valid_completed_run

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "processed_data")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "outputs_0910")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command(
    args: argparse.Namespace,
    model: dict[str, str],
    feature: dict[str, object],
    seed: int,
) -> list[str]:
    result = [
        sys.executable,
        str(SCRIPT_DIR / "train_heatmap_cell.py"),
        "--model", str(model["model"]),
        "--feature-set", str(feature["feature_set"]),
        "--seed", str(seed),
        "--data-dir", str(args.data_dir),
        "--results-dir", str(args.results_dir),
        "--selection-metric", "validation_rmse_k",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--patience", str(args.patience),
        "--num-workers", str(args.num_workers),
        "--rapid-quantile", "0.10",
        "--rapid-weight", "3.0",
    ]
    if args.overwrite:
        result.append("--overwrite")
    return result


def main() -> None:
    args = parse_args()
    tasks = all_tasks()
    if not 1 <= args.num_shards <= len(tasks):
        raise ValueError(f"num-shards must be 1..{len(tasks)}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= index < num-shards")
    args.data_dir = args.data_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    selected = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_index]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.results_dir / "study_protocol.json", protocol_payload())
    if args.dry_run:
        for model, feature, seed in selected:
            print(f"{model['model']}/{feature['feature_set']}/seed_{seed}")
        print(f"DRY_RUN_TASKS={len(selected)}")
        return

    progress = args.results_dir / f"shard_{args.shard_index:02d}_progress.json"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    started = time.time()
    completed = 0
    for number, (model, feature, seed) in enumerate(selected, 1):
        identifier = f"{model['model']}/{feature['feature_set']}/seed_{seed}"
        atomic_json(progress, {
            "status": "training",
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "completed_in_shard": completed,
            "total_in_shard": len(selected),
            "current_run": identifier,
            "test_data_read": False,
        })
        if not args.overwrite and valid_completed_run(args.results_dir, model, feature, seed):
            completed += 1
            print(f"[{number}/{len(selected)}] REUSE {identifier}", flush=True)
            continue
        print(f"[{number}/{len(selected)}] START {identifier}", flush=True)
        log = (
            args.results_dir / "logs" / str(model["model"])
            / str(feature["feature_set"]) / f"seed_{seed}.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        run_command = command(args, model, feature, seed)
        with log.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write("\nCOMMAND=" + subprocess.list2cmdline(run_command) + "\n")
            subprocess.run(
                run_command,
                cwd=SCRIPT_DIR,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not valid_completed_run(args.results_dir, model, feature, seed):
            raise RuntimeError(f"Post-training audit failed: {identifier}")
        completed += 1
        print(f"[{number}/{len(selected)}] DONE {identifier}", flush=True)

    atomic_json(progress, {
        "status": "complete",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "completed_in_shard": completed,
        "total_in_shard": len(selected),
        "elapsed_seconds": time.time() - started,
        "current_run": None,
        "test_data_read": False,
    })
    print(f"SHARD_COMPLETE={args.shard_index}")


if __name__ == "__main__":
    main()
