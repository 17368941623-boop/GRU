#!/usr/bin/env python3
"""Summarize the complete validation-only stage-2 fine-search grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from protocol import (
    DIFFICULT_WINDOWS,
    HIDDEN_SIZES,
    LOOKBACKS,
    PRIMARY_RANKING_METRIC,
    SEEDS,
    completed_run_is_valid,
    protocol_payload,
    run_directory,
    tasks,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def collect(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    missing: list[str] = []
    for lookback, hidden_size, seed in tasks():
        identifier = f"lookback_{lookback}/hidden_{hidden_size}/seed_{seed}"
        if not completed_run_is_valid(results_dir, lookback, hidden_size, seed):
            missing.append(identifier)
            continue
        path = run_directory(results_dir, lookback, hidden_size, seed) / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        full = payload["validation_metrics"]
        hard = payload["difficult_validation_metrics"]
        row = {
            "lookback": lookback,
            "hidden_size": hidden_size,
            "seed": seed,
            "hard_cooling_pooled_rmse_k": float(hard["rmse_k"]),
            "hard_cooling_pooled_mae_k": float(hard["mae_k"]),
            "hard_cooling_pooled_p95_absolute_error_k": float(
                hard["p95_absolute_error_k"]
            ),
            "hard_cooling_pooled_bias_k": float(hard["bias_k"]),
            "hard_cooling_persistence_rmse_k": float(
                payload["difficult_persistence_metrics"]["rmse_k"]
            ),
            "validation_rmse_k": float(full["rmse_k"]),
            "validation_mae_k": float(full["mae_k"]),
            "validation_p95_absolute_error_k": float(full["p95_absolute_error_k"]),
            "validation_max_absolute_error_k": float(full["max_absolute_error_k"]),
            "validation_bias_k": float(full["bias_k"]),
            "validation_r2": float(full["r2"]),
            "validation_direction_accuracy": float(
                payload["validation_direction_accuracy"]
            ),
            "persistence_validation_rmse_k": float(
                payload["persistence_validation_metrics"]["rmse_k"]
            ),
            "trainable_parameters": int(payload["trainable_parameters"]),
            "best_epoch": int(payload["best_epoch"]),
            "completed_epochs": int(payload["completed_epochs"]),
            "training_seconds_total": float(payload["training_seconds_total"]),
            "test_data_loaded": bool(payload["test_data_loaded"]),
            "source_metrics": str(path.relative_to(results_dir)),
        }
        for window_id, _, _ in DIFFICULT_WINDOWS:
            metrics = payload["difficult_window_metrics"][window_id]
            row[f"{window_id.lower()}_rmse_k"] = float(metrics["rmse_k"])
            row[f"{window_id.lower()}_mae_k"] = float(metrics["mae_k"])
        rows.append(row)
    summary_dir = results_dir / "validation_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if missing:
        (summary_dir / "missing_runs.txt").write_text(
            "\n".join(missing) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"{len(missing)} of {len(tasks())} runs are incomplete")
    result = pd.DataFrame(rows)
    if len(result) != len(tasks()) or result["test_data_loaded"].any():
        raise ValueError("Completeness or no-test-access audit failed")
    counts = result.groupby(["lookback", "hidden_size"])["seed"].nunique()
    if not (counts == len(SEEDS)).all():
        raise ValueError("One or more cells do not contain all paired seeds")
    return result


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = (
        "hard_cooling_pooled_rmse_k",
        "hard_cooling_pooled_mae_k",
        "hard_cooling_pooled_p95_absolute_error_k",
        "hard_cooling_pooled_bias_k",
        "validation_rmse_k",
        "validation_mae_k",
        "validation_p95_absolute_error_k",
        "validation_max_absolute_error_k",
        "validation_bias_k",
        "validation_r2",
        "validation_direction_accuracy",
        "h1_rmse_k",
        "h2_rmse_k",
        "h3_rmse_k",
        "h4_rmse_k",
    )
    for (lookback, hidden_size), group in runs.groupby(
        ["lookback", "hidden_size"], sort=True
    ):
        row = {
            "lookback": int(lookback),
            "hidden_size": int(hidden_size),
            "n_seeds": int(group["seed"].nunique()),
            "seed_set": ";".join(map(str, sorted(group["seed"].unique()))),
            "trainable_parameters": int(group["trainable_parameters"].iloc[0]),
            "best_epoch_mean": float(group["best_epoch"].mean()),
            "completed_epochs_mean": float(group["completed_epochs"].mean()),
            "training_minutes_mean": float(group["training_seconds_total"].mean() / 60.0),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            se = sd / np.sqrt(len(values))
            critical = float(t.ppf(0.975, df=len(values) - 1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_se"] = se
            row[f"{metric}_ci95_low"] = mean - critical * se
            row[f"{metric}_ci95_high"] = mean + critical * se
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["lookback", "hidden_size"], kind="stable"
    ).reset_index(drop=True)
    metric = f"{PRIMARY_RANKING_METRIC}_mean"
    best_index = result[metric].idxmin()
    best = result.loc[best_index]
    threshold = float(
        best[metric] + best[f"{PRIMARY_RANKING_METRIC}_se"]
    )
    result["within_one_se_of_hard_cooling_minimum"] = result[metric] <= threshold
    result["hard_cooling_global_minimum_cell"] = False
    result.loc[best_index, "hard_cooling_global_minimum_cell"] = True
    return result


def metric_matrix(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    return cells.pivot(
        index="hidden_size", columns="lookback", values=metric
    ).loc[list(HIDDEN_SIZES), list(LOOKBACKS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "output")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    output = results_dir / "validation_summary"
    output.mkdir(parents=True, exist_ok=True)
    runs = collect(results_dir)
    cells = summarize(runs)
    runs.to_csv(output / "validation_seed_runs.csv", index=False)
    cells.to_csv(output / "validation_cell_summary.csv", index=False)

    matrices = {
        "hard_cooling_rmse_mean_matrix.csv": "hard_cooling_pooled_rmse_k_mean",
        "hard_cooling_rmse_sd_matrix.csv": "hard_cooling_pooled_rmse_k_sd",
        "hard_cooling_rmse_se_matrix.csv": "hard_cooling_pooled_rmse_k_se",
        "full_validation_rmse_mean_matrix.csv": "validation_rmse_k_mean",
        "full_validation_rmse_sd_matrix.csv": "validation_rmse_k_sd",
    }
    for filename, metric in matrices.items():
        metric_matrix(cells, metric).to_csv(output / filename)

    candidates = cells.loc[
        cells["within_one_se_of_hard_cooling_minimum"]
    ].copy()
    candidates.to_csv(output / "fine_candidate_cells.csv", index=False)
    best = cells.loc[cells["hard_cooling_global_minimum_cell"]].iloc[0]
    recommendation = {
        **protocol_payload(),
        "observed_runs": len(runs),
        "complete": True,
        "test_data_read_by_training": False,
        "hard_cooling_global_minimum": {
            "lookback": int(best["lookback"]),
            "hidden_size": int(best["hidden_size"]),
            "hard_cooling_pooled_rmse_k_mean": float(
                best["hard_cooling_pooled_rmse_k_mean"]
            ),
            "hard_cooling_pooled_rmse_k_sd": float(
                best["hard_cooling_pooled_rmse_k_sd"]
            ),
            "full_validation_rmse_k_mean": float(best["validation_rmse_k_mean"]),
        },
        "one_se_threshold_k": float(
            best["hard_cooling_pooled_rmse_k_mean"]
            + best["hard_cooling_pooled_rmse_k_se"]
        ),
        "candidate_cells": candidates[
            [
                "lookback",
                "hidden_size",
                "hard_cooling_pooled_rmse_k_mean",
                "hard_cooling_pooled_rmse_k_sd",
                "validation_rmse_k_mean",
                "trainable_parameters",
            ]
        ].to_dict(orient="records"),
        "boundary_warning": {
            "lookback_on_boundary": bool(
                candidates["lookback"].isin((min(LOOKBACKS), max(LOOKBACKS))).any()
            ),
            "hidden_size_on_boundary": bool(
                candidates["hidden_size"].isin(
                    (min(HIDDEN_SIZES), max(HIDDEN_SIZES))
                ).any()
            ),
        },
        "selection_note": (
            "Checkpoints are selected by full-validation RMSE. Hyperparameter cells "
            "are ranked by the frozen H1-H4 pooled RMSE; full-validation RMSE is a "
            "secondary safety metric."
        ),
        "statistical_caution": (
            "SD and SE are computed across optimization seeds. They describe training "
            "variability, not independent-process generalization uncertainty."
        ),
    }
    (output / "fine_search_recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metric_matrix(cells, "hard_cooling_pooled_rmse_k_mean").to_string())
    print(json.dumps(recommendation["hard_cooling_global_minimum"], ensure_ascii=False))
    print(f"ONE_SE_CANDIDATES={len(candidates)}")
    print("VALIDATION_SUMMARY_COMPLETE=true")
    print("TEST_DATA_OPENED=false")


if __name__ == "__main__":
    main()
