#!/usr/bin/env python3
"""Audit completeness and summarize one validation-only model study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from train_validation_batch import (
    COMMON_ORIGIN_LOOKBACK,
    LOOKBACK,
    PREDICT_STEPS,
    run_directory,
    study_definition,
    valid_completed_run,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

GRAPH_CONTRASTS = (
    (
        "kan_to_mlp_parallel",
        "parallel_gru_mlp_gnn",
        "parallel_gru_kan_gnn",
        "parallel MLP minus parallel KAN",
    ),
    (
        "kan_to_mlp_serial",
        "serial_mlp_gnn_gru",
        "serial_kan_gnn_gru",
        "serial MLP minus serial KAN",
    ),
    (
        "parallel_to_serial_kan",
        "serial_kan_gnn_gru",
        "parallel_gru_kan_gnn",
        "serial KAN minus parallel KAN",
    ),
    (
        "parallel_to_serial_mlp",
        "serial_mlp_gnn_gru",
        "parallel_gru_mlp_gnn",
        "serial MLP minus parallel MLP",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=("graph", "tcn"))
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def mean_interval(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("summary sample must be finite and non-empty")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    sem = std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    half = float(stats.t.ppf(0.975, len(array) - 1) * sem) if len(array) > 1 else 0.0
    return {
        "n_seeds": int(len(array)),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def holm_adjust(values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(tuple(values), dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def collect_runs(study: str, results_dir: Path) -> pd.DataFrame:
    models, seeds = study_definition(study)
    missing: list[str] = []
    rows: list[dict[str, object]] = []
    for spec in models:
        for seed in seeds:
            run_dir = run_directory(results_dir, spec, seed)
            if not valid_completed_run(run_dir, spec, seed):
                missing.append(f"{spec['model']}/{spec['config_id']}/seed_{seed}")
                continue
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "study": study,
                    "model": spec["model"],
                    "config_id": spec["config_id"],
                    "paper_role": spec["role"],
                    "seed": seed,
                    "lookback": int(metrics["lookback"]),
                    "common_origin_lookback": int(metrics["common_origin_lookback"]),
                    "predict_steps": int(metrics["predict_steps"]),
                    "trainable_parameters": int(metrics["trainable_parameters"]),
                    "completed_epochs": int(metrics["completed_epochs"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "training_seconds_total": float(metrics["training_seconds_total"]),
                    "validation_rmse_k": float(metrics["validation_metrics"]["rmse_k"]),
                    "validation_mae_k": float(metrics["validation_metrics"]["mae_k"]),
                    "validation_p95_absolute_error_k": float(metrics["validation_metrics"]["p95_absolute_error_k"]),
                    "validation_max_absolute_error_k": float(metrics["validation_metrics"]["max_absolute_error_k"]),
                    "validation_rapid_rmse_k": float(metrics["rapid_validation_metrics"]["rmse_k"]),
                    "validation_rapid_mae_k": float(metrics["rapid_validation_metrics"]["mae_k"]),
                    "validation_rapid_p95_absolute_error_k": float(metrics["rapid_validation_metrics"]["p95_absolute_error_k"]),
                    "validation_rapid_max_absolute_error_k": float(metrics["rapid_validation_metrics"]["max_absolute_error_k"]),
                    "validation_direction_accuracy": float(metrics["validation_direction_accuracy"]),
                    "validation_rapid_direction_accuracy": float(metrics["rapid_validation_direction_accuracy"]),
                    "persistence_validation_rmse_k": float(metrics["persistence_validation_metrics"]["rmse_k"]),
                    "rapid_persistence_validation_rmse_k": float(metrics["rapid_persistence_validation_metrics"]["rmse_k"]),
                    "validation_windows": int(metrics["validation_windows"]),
                    "validation_rapid_windows": int(metrics["validation_rapid_windows"]),
                    "test_data_loaded": bool(metrics["test_data_loaded"]),
                }
            )
    if missing:
        report = results_dir / "missing_runs.txt"
        report.write_text("\n".join(missing) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(missing)} expected runs are incomplete; see {report}")
    frame = pd.DataFrame(rows)
    if frame["test_data_loaded"].any():
        raise ValueError("at least one development run reports test-data access")
    return frame


def model_summary(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "validation_rapid_rmse_k",
        "validation_rmse_k",
        "validation_rapid_mae_k",
        "validation_mae_k",
        "validation_rapid_p95_absolute_error_k",
        "validation_p95_absolute_error_k",
        "validation_rapid_direction_accuracy",
    )
    rows: list[dict[str, object]] = []
    for model, group in runs.groupby("model", sort=False):
        row: dict[str, object] = {
            "model": model,
            "config_id": group["config_id"].iloc[0],
            "paper_role": group["paper_role"].iloc[0],
            "n_seeds": int(group["seed"].nunique()),
            "trainable_parameters": int(group["trainable_parameters"].iloc[0]),
            "training_minutes_mean": float(group["training_seconds_total"].mean() / 60.0),
            "completed_epochs_mean": float(group["completed_epochs"].mean()),
        }
        for metric in metrics:
            summary = mean_interval(group[metric].to_numpy(dtype=np.float64))
            for key in ("mean", "std", "ci95_low", "ci95_high"):
                row[f"{metric}_{key}"] = summary[key]
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["validation_rapid_rmse_k_mean", "validation_rmse_k_mean", "validation_rapid_rmse_k_std"],
        ascending=True,
    ).reset_index(drop=True)
    result.insert(0, "validation_rank", np.arange(1, len(result) + 1))
    return result


def graph_effects(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for effect, left_model, right_model, definition in GRAPH_CONTRASTS:
        left = runs.loc[runs["model"] == left_model, ["seed", "validation_rapid_rmse_k"]].rename(
            columns={"validation_rapid_rmse_k": "left"}
        )
        right = runs.loc[runs["model"] == right_model, ["seed", "validation_rapid_rmse_k"]].rename(
            columns={"validation_rapid_rmse_k": "right"}
        )
        paired = left.merge(right, on="seed", validate="one_to_one").sort_values("seed")
        differences = paired["left"].to_numpy(dtype=np.float64) - paired["right"].to_numpy(dtype=np.float64)
        summary = mean_interval(differences)
        p_value = 1.0 if np.allclose(differences, 0.0) else float(stats.wilcoxon(differences).pvalue)
        rows.append(
            {
                "effect": effect,
                "difference_definition": definition,
                "left_model": left_model,
                "right_model": right_model,
                "n_paired_seeds": len(paired),
                "mean_difference_k": summary["mean"],
                "std_difference_k": summary["std"],
                "ci95_low_k": summary["ci95_low"],
                "ci95_high_k": summary["ci95_high"],
                "wilcoxon_p_raw": p_value,
            }
        )
    adjusted = holm_adjust(row["wilcoxon_p_raw"] for row in rows)
    for row, adjusted_p in zip(rows, adjusted):
        row["wilcoxon_p_holm"] = float(adjusted_p)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    runs = collect_runs(args.study, results_dir)
    summary_dir = results_dir / "validation_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(summary_dir / "validation_seed_runs.csv", index=False)
    summary = model_summary(runs)
    summary.to_csv(summary_dir / "validation_model_summary.csv", index=False)
    if args.study == "graph":
        graph_effects(runs).to_csv(summary_dir / "paired_graph_effects.csv", index=False)
    audit = {
        "study": args.study,
        "lookback": LOOKBACK,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "complete_runs": len(runs),
        "models": sorted(runs["model"].unique().tolist()),
        "seeds": sorted(int(seed) for seed in runs["seed"].unique()),
        "test_data_read": False,
        "selection_metric": "mean Original 260501 rapid-cooling validation RMSE",
        "uncertainty": "two-sided 95% Student-t CI across random seeds",
        "paired_effect_test": "two-sided Wilcoxon signed-rank with Holm correction" if args.study == "graph" else None,
        "important_scope": "seed variability on one fixed validation split; not independent physical-run generalization",
    }
    (summary_dir / "summary_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"VALIDATION_SUMMARY={summary_dir}", flush=True)


if __name__ == "__main__":
    main()
