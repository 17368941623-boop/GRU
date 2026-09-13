#!/usr/bin/env python3
"""Run one non-overlapping shard of the lookback=65 validation study.

This launcher only calls train_model_ablation.py, whose data contract is
development-only: train_clean.pkl is used to fit every scaler and model, and
Original 260501 validation is used for checkpoint selection.  No test pickle
is opened by this program or by the called trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "processed_data"

LOOKBACK = 65
COMMON_ORIGIN_LOOKBACK = 120
PREDICT_STEPS = 15
GRAPH_SEEDS = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)
TCN_SEEDS = (42, 62, 82, 102, 122)

COMMON_CONFIG: dict[str, int | float] = {
    "history_hidden": 64,
    "graph_hidden": 64,
    "graph_sweeps": 2,
    "edge_hidden": 32,
    "kan_grid": 8,
    "tcn_levels": 6,
    "tcn_kernel": 3,
    "control_hidden": 32,
    "fusion_hidden": 64,
    "dropout": 0.1,
}

GRAPH_MODELS: tuple[dict[str, str], ...] = (
    {
        "model": "parallel_gru_mlp_gnn",
        "config_id": "controlled_hh64_gh64_gs2_mlp",
        "role": "parallel topology with unconstrained MLP edge gates",
    },
    {
        "model": "parallel_gru_kan_gnn",
        "config_id": "controlled_hh64_gh64_gs2_kan8",
        "role": "parallel topology with monotonic KAN valve gates",
    },
    {
        "model": "serial_mlp_gnn_gru",
        "config_id": "controlled_hh64_gh64_gs2_mlp",
        "role": "serial topology with unconstrained MLP edge gates",
    },
    {
        "model": "serial_kan_gnn_gru",
        "config_id": "controlled_hh64_gh64_gs2_kan8",
        "role": "serial topology with monotonic KAN valve gates",
    },
)

TCN_MODELS: tuple[dict[str, str], ...] = (
    {
        "model": "tcn_baseline",
        "config_id": "controlled_hh64_tl6",
        "role": "causal TCN temporal baseline with a 127-step receptive field",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=("graph", "tcn"))
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def study_definition(study: str) -> tuple[tuple[dict[str, str], ...], tuple[int, ...]]:
    if study == "graph":
        return GRAPH_MODELS, GRAPH_SEEDS
    if study == "tcn":
        return TCN_MODELS, TCN_SEEDS
    raise AssertionError(study)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run_directory(results_dir: Path, spec: dict[str, str], seed: int) -> Path:
    return (
        results_dir
        / "development"
        / f"horizon_{PREDICT_STEPS:02d}"
        / f"lookback_{LOOKBACK:02d}"
        / spec["model"]
        / spec["config_id"]
        / f"seed_{seed}"
    )


def valid_completed_run(path: Path, spec: dict[str, str], seed: int) -> bool:
    metrics_path = path / "metrics.json"
    checkpoint_path = path / "best_model.pt"
    if not metrics_path.exists() or not checkpoint_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "model_type": spec["model"],
        "config_id": spec["config_id"],
        "lookback": LOOKBACK,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "seed": seed,
        "test_data_loaded": False,
        "future_noncontrol_measurements_used": False,
        "future_label_used_as_input": False,
    }
    return all(metrics.get(key) == value for key, value in expected.items())


def command_for(args: argparse.Namespace, spec: dict[str, str], seed: int) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "train_model_ablation.py"),
        "--model", spec["model"],
        "--config-id", spec["config_id"],
        "--lookback", str(LOOKBACK),
        "--common-origin-lookback", str(COMMON_ORIGIN_LOOKBACK),
        "--predict-steps", str(PREDICT_STEPS),
        "--seed", str(seed),
        "--data-dir", str(args.data_dir),
        "--results-dir", str(args.results_dir),
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


def protocol(args: argparse.Namespace, models: tuple[dict[str, str], ...], seeds: tuple[int, ...]) -> dict[str, Any]:
    return {
        "study": args.study,
        "target": "Delta Thv at t+15; restored absolute Thv is used for metrics",
        "lookback": LOOKBACK,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "history_input": "measurements and controls through current time t",
        "known_future_control_input": "valve commands u(t) through u(t+14)",
        "future_measurements_used": False,
        "scaler_fit_split": "train_clean.pkl only",
        "checkpoint_selection_split": "Original 260501 validation only",
        "held_out_test_data_read": False,
        "checkpoint_selection_metric": "rapid-cooling validation RMSE",
        "rapid_loss": "training-only 10th-percentile threshold; weight 3 Huber loss",
        "seeds": list(seeds),
        "models": list(models),
        "common_model_config": COMMON_CONFIG,
        "expected_runs": len(models) * len(seeds),
    }


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= index < num-shards")
    args.data_dir = args.data_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    models, seeds = study_definition(args.study)
    tasks = [(spec, seed) for spec in models for seed in seeds]
    shard_tasks = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_index]
    if not shard_tasks:
        raise ValueError("this shard contains no runs; reduce num-shards")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.results_dir / "study_protocol.json", protocol(args, models, seeds))
    progress_path = args.results_dir / f"shard_{args.shard_index:02d}_progress.json"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("MPLBACKEND", "Agg")
    started = time.time()
    completed = 0

    for local_index, (spec, seed) in enumerate(shard_tasks, start=1):
        identifier = f"{spec['model']}/{spec['config_id']}/seed_{seed}"
        destination = run_directory(args.results_dir, spec, seed)
        atomic_json(
            progress_path,
            {
                "status": "training",
                "study": args.study,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "completed_in_shard": completed,
                "total_in_shard": len(shard_tasks),
                "current_run": identifier,
                "test_data_read": False,
            },
        )
        if not args.overwrite and valid_completed_run(destination, spec, seed):
            completed += 1
            print(f"[{local_index}/{len(shard_tasks)}] REUSE {identifier}", flush=True)
            continue

        print(f"[{local_index}/{len(shard_tasks)}] START {identifier}", flush=True)
        log_path = args.results_dir / "logs" / spec["model"] / f"seed_{seed}.log"
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
        if not valid_completed_run(destination, spec, seed):
            raise RuntimeError(f"post-training metadata audit failed: {identifier}")
        completed += 1
        print(f"[{local_index}/{len(shard_tasks)}] DONE {identifier}", flush=True)

    atomic_json(
        progress_path,
        {
            "status": "complete",
            "study": args.study,
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
