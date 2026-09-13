#!/usr/bin/env python3
"""Run one resumable shard of the 120-cell stage-1 training matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from protocol import completed_run_is_valid, protocol_payload, tasks


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def discover_default_data_dir() -> Path:
    candidates = (
        PROJECT_DIR.parent / "processed_data",
        PROJECT_DIR.parent.parent / "processed_data",
    )
    for candidate in candidates:
        if (candidate / "data_build_config.json").is_file():
            return candidate
    return candidates[0]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=discover_default_data_dir())
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_tasks = tasks()
    if not 1 <= args.num_shards <= len(all_tasks):
        raise ValueError("num-shards is outside the valid range")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= index < num-shards")
    selected = [
        task for index, task in enumerate(all_tasks)
        if index % args.num_shards == args.shard_index
    ]
    if args.dry_run:
        for lookback, hidden_size, seed in selected:
            print(f"lookback_{lookback}/hidden_{hidden_size}/seed_{seed}")
        print(f"DRY_RUN_TASKS={len(selected)}")
        return
    args.data_dir = args.data_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.results_dir / "study_protocol.json", protocol_payload())
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.time()
    complete = 0
    progress_path = args.results_dir / f"shard_{args.shard_index:02d}_progress.json"
    for number, (lookback, hidden_size, seed) in enumerate(selected, 1):
        identifier = f"lookback_{lookback}/hidden_{hidden_size}/seed_{seed}"
        atomic_json(progress_path, {
            "status": "training",
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "completed_in_shard": complete,
            "total_in_shard": len(selected),
            "current_run": identifier,
            "test_data_read": False,
        })
        if not args.overwrite and completed_run_is_valid(
            args.results_dir, lookback, hidden_size, seed
        ):
            complete += 1
            print(f"[{number}/{len(selected)}] REUSE {identifier}", flush=True)
            continue
        command = [
            sys.executable,
            str(SCRIPT_DIR / "train_global_cell.py"),
            "--lookback", str(lookback),
            "--hidden-size", str(hidden_size),
            "--seed", str(seed),
            "--data-dir", str(args.data_dir),
            "--results-dir", str(args.results_dir),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--patience", str(args.patience),
            "--num-workers", str(args.num_workers),
            "--device", str(args.device),
        ]
        if args.overwrite:
            command.append("--overwrite")
        log_path = (
            args.results_dir / "logs" / f"lookback_{lookback:03d}"
            / f"hidden_{hidden_size:03d}" / f"seed_{seed}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{number}/{len(selected)}] START {identifier}", flush=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write("\nCOMMAND=" + subprocess.list2cmdline(command) + "\n")
            subprocess.run(
                command,
                cwd=SCRIPT_DIR,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not completed_run_is_valid(args.results_dir, lookback, hidden_size, seed):
            raise RuntimeError(f"Completed run failed its audit: {identifier}")
        complete += 1
        print(f"[{number}/{len(selected)}] DONE {identifier}", flush=True)
    atomic_json(progress_path, {
        "status": "complete",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "completed_in_shard": complete,
        "total_in_shard": len(selected),
        "elapsed_seconds": time.time() - started,
        "current_run": None,
        "test_data_read": False,
    })
    print(f"SHARD_COMPLETE={args.shard_index}")


if __name__ == "__main__":
    main()
