#!/usr/bin/env python3
"""Run and resume the final unified lookback scan sequentially on one GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from train_thv_delta_gru_lookback_final import (
    COMMON_ORIGIN_LOOKBACK,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    FINAL_LOOKBACKS,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_thv_delta_gru_lookback_final.py"
DEFAULT_SEEDS = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in text.split(",")))
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lookbacks",
        type=parse_int_list,
        default=FINAL_LOOKBACKS,
        help="Comma-separated subset; default is the preregistered 12-point scan.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=DEFAULT_SEEDS,
        help="Comma-separated seeds; default contains ten seeds.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_path(output_dir: Path, lookback: int, seed: int) -> Path:
    return (
        output_dir
        / "horizon_15"
        / "gru"
        / f"lookback_{lookback:02d}"
        / f"seed_{seed}"
        / "metrics.json"
    )


def validate_completed(path: Path, lookback: int, seed: int) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "model_type": "gru",
        "predict_steps": 15,
        "lookback": lookback,
        "seed": seed,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "validation_parent": "260501",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Existing run has {key}={payload.get(key)!r}: {path}")
    if bool(payload.get("test_loaded_after_training")):
        raise ValueError(f"Existing run loaded the test set: {path}")
    if payload.get("fixed_test_window_evaluation") is not None:
        raise ValueError(f"Existing run contains test-window metrics: {path}")
    return True


def write_progress(
    output_dir: Path,
    lookbacks: tuple[int, ...],
    seeds: tuple[int, ...],
    started_at: str,
    current: dict[str, int] | None,
) -> None:
    completed = []
    pending = []
    for lookback in lookbacks:
        for seed in seeds:
            pair = {"lookback": lookback, "seed": seed}
            if metrics_path(output_dir, lookback, seed).exists():
                completed.append(pair)
            else:
                pending.append(pair)
    payload = {
        "updated_at": timestamp(),
        "started_at": started_at,
        "status": "complete" if not pending else "running",
        "lookbacks": list(lookbacks),
        "seeds": list(seeds),
        "total_runs": len(lookbacks) * len(seeds),
        "completed_runs": len(completed),
        "pending_runs": len(pending),
        "current_run": current,
        "completed_pairs": completed,
        "pending_pairs": pending,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "progress.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def collect_summary(
    output_dir: Path,
    lookbacks: tuple[int, ...],
    seeds: tuple[int, ...],
) -> None:
    rows: list[dict[str, object]] = []
    normalization_hashes: set[str] = set()
    for lookback in lookbacks:
        for seed in seeds:
            path = metrics_path(output_dir, lookback, seed)
            if not validate_completed(path, lookback, seed):
                raise FileNotFoundError(f"Missing completed run: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            validation = payload["validation_metrics"]
            rapid = payload["rapid_validation_metrics"]
            persistence = payload["persistence_validation_metrics"]
            rapid_persistence = payload["rapid_persistence_validation_metrics"]
            normalization_path = path.parent / "normalization_report.csv"
            if not normalization_path.exists():
                raise FileNotFoundError(normalization_path)
            normalization_hash = sha256(normalization_path)
            normalization_hashes.add(normalization_hash)
            rows.append(
                {
                    "lookback": lookback,
                    "seed": seed,
                    "best_epoch": int(payload["best_epoch"]),
                    "completed_epochs": int(payload["completed_epochs"]),
                    "validation_rmse_k": float(validation["rmse_k"]),
                    "rapid_validation_rmse_k": float(rapid["rmse_k"]),
                    "validation_mae_k": float(validation["mae_k"]),
                    "rapid_validation_mae_k": float(rapid["mae_k"]),
                    "validation_windows": int(validation["n_windows"]),
                    "rapid_validation_windows": int(rapid["n_windows"]),
                    "persistence_validation_rmse_k": float(persistence["rmse_k"]),
                    "rapid_persistence_validation_rmse_k": float(
                        rapid_persistence["rmse_k"]
                    ),
                    "rapid_threshold_k": float(
                        payload["rapid_threshold_k_train_only"]
                    ),
                    "training_seconds_total": float(payload["training_seconds_total"]),
                    "device": str(payload["device"]),
                    "normalization_report_sha256": normalization_hash,
                    "test_loaded": bool(payload["test_loaded_after_training"]),
                    "metrics_path": str(path.resolve()),
                }
            )
    if len(normalization_hashes) != 1:
        raise ValueError(
            "Normalization reports are not identical across the unified scan: "
            f"{sorted(normalization_hashes)}"
        )

    all_runs_path = output_dir / "lookback_all_runs_validation_only.csv"
    with all_runs_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    for lookback in lookbacks:
        part = [row for row in rows if row["lookback"] == lookback]
        rapid = np.asarray(
            [float(row["rapid_validation_rmse_k"]) for row in part], dtype=float
        )
        full = np.asarray(
            [float(row["validation_rmse_k"]) for row in part], dtype=float
        )
        summary_rows.append(
            {
                "lookback": lookback,
                "seed_count": len(part),
                "rapid_rmse_mean_k": float(rapid.mean()),
                "rapid_rmse_std_k": float(rapid.std(ddof=1)),
                "rapid_rmse_median_k": float(np.median(rapid)),
                "rapid_rmse_q1_k": float(np.quantile(rapid, 0.25)),
                "rapid_rmse_q3_k": float(np.quantile(rapid, 0.75)),
                "rapid_rmse_min_k": float(rapid.min()),
                "rapid_rmse_max_k": float(rapid.max()),
                "full_rmse_mean_k": float(full.mean()),
                "full_rmse_std_k": float(full.std(ddof=1)),
                "full_rmse_median_k": float(np.median(full)),
                "validation_windows": int(part[0]["validation_windows"]),
                "rapid_validation_windows": int(part[0]["rapid_validation_windows"]),
            }
        )
    best = min(summary_rows, key=lambda row: float(row["rapid_rmse_mean_k"]))
    summary_path = output_dir / "lookback_summary_validation_only.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    reference = int(best["lookback"])
    reference_by_seed = {
        int(row["seed"]): float(row["rapid_validation_rmse_k"])
        for row in rows
        if int(row["lookback"]) == reference
    }
    paired_rows = []
    for row in rows:
        paired_rows.append(
            {
                "lookback": int(row["lookback"]),
                "seed": int(row["seed"]),
                "reference_best_lookback": reference,
                "rapid_rmse_difference_vs_best_k": float(
                    row["rapid_validation_rmse_k"]
                )
                - reference_by_seed[int(row["seed"])],
            }
        )
    with (output_dir / "lookback_paired_vs_best_validation_only.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    manifest = {
        "completed_at": timestamp(),
        "selection_split": "Original 260501 validation",
        "test_set_loaded": False,
        "predict_steps": 15,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "lookbacks": list(lookbacks),
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "normalization_report_sha256": next(iter(normalization_hashes)),
        "best_lookback_by_mean_rapid_validation_rmse": reference,
        "best_mean_rapid_validation_rmse_k": float(best["rapid_rmse_mean_k"]),
        "all_runs_csv": all_runs_path.name,
        "summary_csv": summary_path.name,
    }
    (output_dir / "final_scan_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "BEST_LOOKBACK_BY_MEAN_RAPID_VALIDATION_RMSE="
        f"{reference}",
        flush=True,
    )
    print(
        f"BEST_MEAN_RAPID_VALIDATION_RMSE_K={float(best['rapid_rmse_mean_k']):.9f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if tuple(args.lookbacks) != tuple(
        value for value in FINAL_LOOKBACKS if value in set(args.lookbacks)
    ):
        raise ValueError(
            f"Lookbacks must be an ordered subset of {FINAL_LOOKBACKS}, "
            f"got {args.lookbacks}"
        )
    if any(value not in FINAL_LOOKBACKS for value in args.lookbacks):
        raise ValueError(f"Unsupported lookback in {args.lookbacks}")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")

    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = timestamp()
    total = len(args.lookbacks) * len(args.seeds)
    write_progress(args.output_dir, args.lookbacks, args.seeds, started_at, None)

    if not args.summary_only:
        completed_index = 0
        for lookback in args.lookbacks:
            for seed in args.seeds:
                completed_index += 1
                path = metrics_path(args.output_dir, lookback, seed)
                if validate_completed(path, lookback, seed) and not args.overwrite:
                    print(
                        f"[{completed_index}/{total}] SKIP lookback={lookback} "
                        f"seed={seed}",
                        flush=True,
                    )
                    continue
                command = [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--lookback",
                    str(lookback),
                    "--seed",
                    str(seed),
                    "--predict-steps",
                    "15",
                    "--data-dir",
                    str(args.data_dir),
                    "--output-dir",
                    str(args.output_dir),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--patience",
                    str(args.patience),
                    "--num-workers",
                    str(args.num_workers),
                ]
                if args.overwrite:
                    command.append("--overwrite")
                print(
                    f"[{completed_index}/{total}] START lookback={lookback} seed={seed} "
                    f"at {timestamp()}",
                    flush=True,
                )
                write_progress(
                    args.output_dir,
                    args.lookbacks,
                    args.seeds,
                    started_at,
                    {"lookback": lookback, "seed": seed},
                )
                if args.dry_run:
                    print("DRY_RUN " + subprocess.list2cmdline(command), flush=True)
                    continue
                run_log = log_dir / f"lookback_{lookback:03d}_seed_{seed}.log"
                run_started = time.perf_counter()
                with run_log.open("w", encoding="utf-8", buffering=1) as stream:
                    subprocess.run(
                        command,
                        cwd=SCRIPT_DIR,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                if not validate_completed(path, lookback, seed):
                    raise RuntimeError(f"Run ended without valid metrics: {path}")
                print(
                    f"[{completed_index}/{total}] COMPLETE lookback={lookback} "
                    f"seed={seed} minutes={(time.perf_counter()-run_started)/60:.2f}",
                    flush=True,
                )

    if args.dry_run:
        print("DRY_RUN_COMPLETE=true", flush=True)
        return
    collect_summary(args.output_dir, args.lookbacks, args.seeds)
    write_progress(args.output_dir, args.lookbacks, args.seeds, started_at, None)
    print("FINAL_UNIFIED_LOOKBACK_SCAN_COMPLETE=true", flush=True)


if __name__ == "__main__":
    main()
