#!/usr/bin/env python3
"""Run the frozen lookback=65 Thv model comparison and ablation study.

The pipeline is deliberately test-blind during training.  Every configuration
is trained and checked on the Original 260501 validation split first.  Only
after all expected model/seed checkpoints exist is selected_configs.json
created and the held-out Original 0715-BACK test evaluator invoked.

The run is resumable: valid metrics.json files are reused, while incomplete
runs are trained again.  Existing completed runs are never overwritten unless
--overwrite is supplied explicitly.
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
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "modelout_lb65_paper"

LOOKBACK = 65
COMMON_ORIGIN_LOOKBACK = 120
PREDICT_STEPS = 15
DEFAULT_SEEDS = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)

# All controlled comparisons use the same latent widths and training recipe.
# Only the component named by the ablation is changed.  The TCN needs six
# dilation levels for a receptive field that covers all 65 historical steps.
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

MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "model": "gru_baseline",
        "config_id": "controlled_hh64",
        "paper_role": "temporal baseline and no-graph ablation",
    },
    {
        "model": "lstm_baseline",
        "config_id": "controlled_hh64",
        "paper_role": "recurrent temporal-encoder comparator",
    },
    {
        "model": "tcn_baseline",
        "config_id": "controlled_hh64_tl6",
        "paper_role": "convolutional temporal-encoder comparator",
    },
    {
        "model": "serial_mlp_gnn_gru",
        "config_id": "controlled_hh64_gh64_gs2_mlp",
        "paper_role": "serial topology with unconstrained MLP edge gates",
    },
    {
        "model": "serial_kan_gnn_gru",
        "config_id": "controlled_hh64_gh64_gs2_kan8",
        "paper_role": "serial topology with monotonic KAN valve gates",
    },
    {
        "model": "parallel_gru_mlp_gnn",
        "config_id": "controlled_hh64_gh64_gs2_mlp",
        "paper_role": "parallel topology with unconstrained MLP edge gates",
    },
    {
        "model": "parallel_gru_kan_gnn",
        "config_id": "controlled_hh64_gh64_gs2_kan8",
        "paper_role": "proposed parallel temporal/physical model",
    },
)


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in text.split(",") if item.strip()))
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=DEFAULT_SEEDS,
        help="Comma-separated random seeds.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Stop after all train/validation runs and preserve test blindness.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain completed runs and overwrite final evaluations.",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def study_dir(results_dir: Path) -> Path:
    return (
        results_dir
        / "development"
        / f"horizon_{PREDICT_STEPS:02d}"
        / f"lookback_{LOOKBACK:02d}"
    )


def run_dir(results_dir: Path, spec: dict[str, Any], seed: int) -> Path:
    return study_dir(results_dir) / spec["model"] / spec["config_id"] / f"seed_{seed}"


def selected_payload(seeds: tuple[int, ...]) -> dict[str, Any]:
    return {
        "selection_status": "frozen_controlled_ablation_without_opening_test_data",
        "selection_split": "Original 260501 validation only",
        "primary_metric": "rapid-cooling RMSE; test is opened only after all runs finish",
        "lookback": LOOKBACK,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "final_seeds": list(seeds),
        "test_data_read": False,
        "models": {
            spec["model"]: {
                "config_id": spec["config_id"],
                "paper_role": spec["paper_role"],
                "model_config": {"model_name": spec["model"], **COMMON_CONFIG},
            }
            for spec in MODEL_SPECS
        },
    }


def protocol_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "study": "Thv h15 lookback-65 controlled model comparison and ablation",
        "target": "Delta Thv at t+15, restored to absolute Thv for metrics",
        "history_window": LOOKBACK,
        "known_future_controls": "u(t) through u(t+14); no future measurements",
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "training_split": "train_clean.pkl excluding 0617-ALL by the existing loader contract",
        "validation_split": "Original 260501",
        "held_out_test_split": "Original 0715-BACK",
        "checkpoint_metric": "Original 260501 rapid-cooling RMSE",
        "rapid_threshold": "training-set 10th percentile of the 15-step Thv change",
        "loss": "rapid-weighted Huber loss; rapid samples receive weight 3",
        "scalers": "fit on training data only",
        "seeds": list(args.seeds),
        "model_count": len(MODEL_SPECS),
        "expected_training_runs": len(MODEL_SPECS) * len(args.seeds),
        "model_specs": list(MODEL_SPECS),
        "controlled_model_config": COMMON_CONFIG,
        "test_gate": (
            "selected_configs.json is created only after every expected development "
            "run passes its metadata audit"
        ),
        "test_data_read_at_protocol_creation": False,
    }


def load_valid_metrics(path: Path, spec: dict[str, Any], seed: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "model_type": spec["model"],
        "config_id": spec["config_id"],
        "lookback": LOOKBACK,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "seed": seed,
        "test_data_loaded": False,
    }
    if any(metrics.get(key) != value for key, value in expected.items()):
        return None
    if metrics.get("future_noncontrol_measurements_used") is not False:
        return None
    if metrics.get("future_label_used_as_input") is not False:
        return None
    if not (path.parent / "best_model.pt").exists():
        return None
    return metrics


def training_command(args: argparse.Namespace, spec: dict[str, Any], seed: int) -> list[str]:
    config = COMMON_CONFIG
    command = [
        sys.executable,
        str(SCRIPT_DIR / "train_model_ablation.py"),
        "--model", str(spec["model"]),
        "--config-id", str(spec["config_id"]),
        "--lookback", str(LOOKBACK),
        "--common-origin-lookback", str(COMMON_ORIGIN_LOOKBACK),
        "--predict-steps", str(PREDICT_STEPS),
        "--seed", str(seed),
        "--data-dir", str(args.data_dir.resolve()),
        "--results-dir", str(args.results_dir.resolve()),
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
        "--rapid-quantile", "0.10",
        "--rapid-weight", "3.0",
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def update_progress(
    path: Path,
    status: str,
    completed: int,
    total: int,
    current: str | None,
    started_at: float,
    error: str | None = None,
) -> None:
    elapsed = max(time.time() - started_at, 0.0)
    payload: dict[str, Any] = {
        "status": status,
        "completed_training_runs": completed,
        "total_training_runs": total,
        "fraction_complete": completed / total if total else 0.0,
        "current_run": current,
        "elapsed_seconds": elapsed,
        "mean_seconds_per_completed_run": elapsed / completed if completed else None,
        "updated_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_data_read": status in {"testing", "summarizing", "complete"},
    }
    if error is not None:
        payload["error"] = error
    atomic_json(path, payload)


def invoke(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write("\nCOMMAND=" + subprocess.list2cmdline(command) + "\n")
        subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_smoke(results_dir: Path, environment: dict[str, str]) -> None:
    log_path = results_dir / "logs" / "smoke_test.log"
    invoke([sys.executable, str(SCRIPT_DIR / "smoke_test_models.py")], log_path, environment)


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size and patience must be positive")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    progress_path = args.results_dir / "progress.json"
    protocol = protocol_payload(args)
    atomic_json(args.results_dir / "study_protocol.json", protocol)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("MPLBACKEND", "Agg")

    total = len(MODEL_SPECS) * len(args.seeds)
    completed = 0
    try:
        update_progress(progress_path, "smoke_test", completed, total, None, started_at)
        run_smoke(args.results_dir, environment)

        for spec in MODEL_SPECS:
            for seed in args.seeds:
                identifier = f"{spec['model']}/{spec['config_id']}/seed_{seed}"
                metrics_path = run_dir(args.results_dir, spec, seed) / "metrics.json"
                valid = None if args.overwrite else load_valid_metrics(metrics_path, spec, seed)
                if valid is not None:
                    completed += 1
                    update_progress(
                        progress_path, "training", completed, total, f"reused {identifier}", started_at
                    )
                    print(f"[{completed}/{total}] REUSE {identifier}", flush=True)
                    continue

                update_progress(progress_path, "training", completed, total, identifier, started_at)
                print(f"[{completed + 1}/{total}] START {identifier}", flush=True)
                log_path = args.results_dir / "logs" / spec["model"] / f"seed_{seed}.log"
                invoke(training_command(args, spec, seed), log_path, environment)
                if load_valid_metrics(metrics_path, spec, seed) is None:
                    raise RuntimeError(f"post-training audit failed for {identifier}")
                completed += 1
                update_progress(progress_path, "training", completed, total, identifier, started_at)
                print(f"[{completed}/{total}] DONE {identifier}", flush=True)

        selected_path = study_dir(args.results_dir) / "selected_configs.json"
        atomic_json(selected_path, selected_payload(args.seeds))
        print(f"TRAINING_COMPLETE={completed}/{total}", flush=True)
        print(f"TEST_GATE_CREATED={selected_path}", flush=True)

        if args.no_test:
            update_progress(progress_path, "development_complete_test_not_opened", completed, total, None, started_at)
            return

        update_progress(progress_path, "testing", completed, total, "held-out Original 0715-BACK", started_at)
        evaluation_command = [
            sys.executable,
            str(SCRIPT_DIR / "evaluate_selected_models.py"),
            "--lookback", str(LOOKBACK),
            "--predict-steps", str(PREDICT_STEPS),
            "--seeds", ",".join(str(seed) for seed in args.seeds),
            "--data-dir", str(args.data_dir),
            "--results-dir", str(args.results_dir),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
        ]
        if args.overwrite:
            evaluation_command.append("--overwrite")
        invoke(
            evaluation_command,
            args.results_dir / "logs" / "final_test_evaluation.log",
            environment,
        )

        update_progress(progress_path, "summarizing", completed, total, "paper tables and figures", started_at)
        invoke(
            [
                sys.executable,
                str(SCRIPT_DIR / "summarize_lb65_paper_study.py"),
                "--results-dir", str(args.results_dir),
            ],
            args.results_dir / "logs" / "paper_summary.log",
            environment,
        )
        update_progress(progress_path, "complete", completed, total, None, started_at)
        print(f"PIPELINE_COMPLETE={args.results_dir}", flush=True)
    except Exception as exc:
        update_progress(
            progress_path,
            "failed",
            completed,
            total,
            None,
            started_at,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
