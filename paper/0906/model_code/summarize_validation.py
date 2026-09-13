#!/usr/bin/env python3
"""Summarize the complete-sequence validation results of all 0906 models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from study_protocol import (
    COMMON_ORIGIN_LOOKBACK,
    LOOKBACK,
    MODEL_SPECS,
    PREDICT_STEPS,
    SELECTION_METRIC,
    all_tasks,
    protocol_payload,
    run_directory,
    valid_completed_run,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "outputs_0906"

METRICS = (
    "validation_rmse_k",
    "validation_mae_k",
    "validation_p95_absolute_error_k",
    "validation_max_absolute_error_k",
    "validation_bias_k",
    "validation_r2",
    "validation_direction_accuracy",
)

STRUCTURAL_CONTRASTS = (
    ("serial_vs_parallel_mlp", "serial_mlp_gnn_gru", "parallel_gru_mlp_gnn"),
    ("serial_vs_parallel_kan", "serial_kan_gnn_gru", "parallel_gru_kan_gnn"),
    ("kan_vs_mlp_serial", "serial_kan_gnn_gru", "serial_mlp_gnn_gru"),
    ("kan_vs_mlp_parallel_gru", "parallel_gru_kan_gnn", "parallel_gru_mlp_gnn"),
    ("kan_vs_mlp_parallel_lstm", "parallel_lstm_kan_gnn", "parallel_lstm_mlp_gnn"),
    (
        "lstm_vs_gru_parallel_mlp",
        "parallel_lstm_mlp_gnn",
        "parallel_gru_mlp_gnn",
    ),
    (
        "lstm_vs_gru_parallel_kan",
        "parallel_lstm_kan_gnn",
        "parallel_gru_kan_gnn",
    ),
    (
        "lstm_mlp_gnn_vs_lstm_baseline",
        "parallel_lstm_mlp_gnn",
        "lstm_baseline",
    ),
    (
        "lstm_kan_gnn_vs_lstm_baseline",
        "parallel_lstm_kan_gnn",
        "lstm_baseline",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def interval(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Summary values must be finite and non-empty")
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half = (
        float(stats.t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array)))
        if len(array) > 1
        else 0.0
    )
    return {
        "n": int(len(array)),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(p_values), dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def collect_runs(results_dir: Path) -> pd.DataFrame:
    missing: list[str] = []
    rows: list[dict[str, object]] = []
    label_by_model = {str(spec["model"]): str(spec["label"]) for spec in MODEL_SPECS}
    role_by_model = {str(spec["model"]): str(spec["role"]) for spec in MODEL_SPECS}
    for spec, seed in all_tasks():
        if not valid_completed_run(results_dir, spec, seed):
            missing.append(f"{spec['model']}/{spec['config_id']}/seed_{seed}")
            continue
        path = run_directory(results_dir, spec, seed) / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload["validation_metrics"]
        rows.append(
            {
                "model": spec["model"],
                "model_label": label_by_model[str(spec["model"])],
                "paper_role": role_by_model[str(spec["model"])],
                "config_id": spec["config_id"],
                "seed": seed,
                "lookback": int(payload["lookback"]),
                "common_origin_lookback": int(payload["common_origin_lookback"]),
                "predict_steps": int(payload["predict_steps"]),
                "trainable_parameters": int(payload["trainable_parameters"]),
                "completed_epochs": int(payload["completed_epochs"]),
                "best_epoch": int(payload["best_epoch"]),
                "training_seconds_total": float(payload["training_seconds_total"]),
                "validation_rmse_k": float(validation["rmse_k"]),
                "validation_mae_k": float(validation["mae_k"]),
                "validation_p95_absolute_error_k": float(validation["p95_absolute_error_k"]),
                "validation_max_absolute_error_k": float(validation["max_absolute_error_k"]),
                "validation_bias_k": float(validation["bias_k"]),
                "validation_r2": float(validation["r2"]),
                "validation_direction_accuracy": float(payload["validation_direction_accuracy"]),
                "validation_windows": int(payload["validation_windows"]),
                "persistence_validation_rmse_k": float(payload["persistence_validation_metrics"]["rmse_k"]),
                "checkpoint_selection_metric": payload["checkpoint_selection_metric"],
                "test_data_loaded": bool(payload["test_data_loaded"]),
                "source_metrics": str(path.relative_to(results_dir)),
            }
        )
    if missing:
        missing_path = results_dir / "validation_summary" / "missing_runs.txt"
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(missing)} runs are incomplete; see {missing_path}")
    frame = pd.DataFrame(rows)
    if len(frame) != len(all_tasks()):
        raise AssertionError("Unexpected validation run count")
    if frame["test_data_loaded"].any():
        raise ValueError("At least one training run reports test-data access")
    if set(frame["checkpoint_selection_metric"]) != {SELECTION_METRIC}:
        raise ValueError("Checkpoint selection metric is inconsistent")
    return frame


def summarize_models(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in runs.groupby("model", sort=False):
        row: dict[str, object] = {
            "model": model,
            "model_label": group["model_label"].iloc[0],
            "paper_role": group["paper_role"].iloc[0],
            "config_id": group["config_id"].iloc[0],
            "n_seeds": int(group["seed"].nunique()),
            "seed_set": ";".join(map(str, sorted(group["seed"].unique()))),
            "trainable_parameters": int(group["trainable_parameters"].iloc[0]),
            "training_minutes_mean": float(group["training_seconds_total"].mean() / 60.0),
            "completed_epochs_mean": float(group["completed_epochs"].mean()),
        }
        for metric in METRICS:
            values = interval(group[metric].to_numpy(dtype=np.float64))
            for key in ("mean", "sd", "ci95_low", "ci95_high"):
                row[f"{metric}_{key}"] = values[key]
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["validation_rmse_k_mean", "validation_rmse_k_sd"]
    ).reset_index(drop=True)
    result.insert(0, "full_validation_rank", np.arange(1, len(result) + 1))
    return result


def paired_row(
    runs: pd.DataFrame, left_model: str, right_model: str, comparison_id: str
) -> dict[str, object]:
    left = runs.loc[runs["model"].eq(left_model), ["seed", "validation_rmse_k"]]
    right = runs.loc[runs["model"].eq(right_model), ["seed", "validation_rmse_k"]]
    matched = left.merge(right, on="seed", suffixes=("_left", "_right"), validate="one_to_one")
    differences = (
        matched["validation_rmse_k_left"] - matched["validation_rmse_k_right"]
    ).to_numpy(dtype=np.float64)
    stats_summary = interval(differences)
    if np.allclose(differences, 0.0):
        p_raw = 1.0
    else:
        p_raw = float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
    right_mean = float(matched["validation_rmse_k_right"].mean())
    return {
        "comparison_id": comparison_id,
        "difference_definition": f"{left_model} minus {right_model}; negative favors {left_model}",
        "left_model": left_model,
        "right_model": right_model,
        "n_paired_seeds": int(len(matched)),
        "paired_seeds": ";".join(map(str, matched["seed"].tolist())),
        "left_matched_mean_k": float(matched["validation_rmse_k_left"].mean()),
        "right_matched_mean_k": right_mean,
        "mean_difference_k": stats_summary["mean"],
        "relative_difference_vs_right_pct": 100.0 * float(stats_summary["mean"]) / right_mean,
        "sd_difference_k": stats_summary["sd"],
        "ci95_low_k": stats_summary["ci95_low"],
        "ci95_high_k": stats_summary["ci95_high"],
        "left_wins": int(np.sum(differences < 0.0)),
        "wilcoxon_p_raw": p_raw,
    }


def paired_tables(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gru_rows = [
        paired_row(runs, str(spec["model"]), "gru_baseline", f"{spec['model']}_vs_gru")
        for spec in MODEL_SPECS
        if spec["model"] != "gru_baseline"
    ]
    versus_gru = pd.DataFrame(gru_rows)
    versus_gru["wilcoxon_p_holm"] = holm_adjust(
        versus_gru["wilcoxon_p_raw"]
    )

    lstm_rows = [
        paired_row(
            runs,
            str(spec["model"]),
            "lstm_baseline",
            f"{spec['model']}_vs_lstm",
        )
        for spec in MODEL_SPECS
        if spec["model"] != "lstm_baseline"
    ]
    versus_lstm = pd.DataFrame(lstm_rows)
    versus_lstm["wilcoxon_p_holm"] = holm_adjust(
        versus_lstm["wilcoxon_p_raw"]
    )

    structural_rows = [
        paired_row(runs, left, right, comparison_id)
        for comparison_id, left, right in STRUCTURAL_CONTRASTS
    ]
    structural = pd.DataFrame(structural_rows)
    structural["wilcoxon_p_holm"] = holm_adjust(structural["wilcoxon_p_raw"])
    return versus_gru, versus_lstm, structural


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = results_dir / "validation_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(results_dir)
    summary = summarize_models(runs)
    versus_gru, versus_lstm, structural = paired_tables(runs)
    runs.to_csv(output_dir / "validation_seed_runs.csv", index=False)
    summary.to_csv(output_dir / "validation_model_summary.csv", index=False)
    versus_gru.to_csv(output_dir / "paired_vs_gru.csv", index=False)
    versus_lstm.to_csv(output_dir / "paired_vs_lstm.csv", index=False)
    structural.to_csv(output_dir / "paired_structural_effects.csv", index=False)
    audit = {
        **protocol_payload(),
        "observed_runs": int(len(runs)),
        "complete": len(runs) == len(all_tasks()),
        "test_data_read_by_training": bool(runs["test_data_loaded"].any()),
        "primary_ranking_metric": "complete Original 260501 validation RMSE",
        "uncertainty": "sample SD and two-sided 95% Student-t CI across seeds",
        "paired_test": "two-sided Wilcoxon signed-rank with within-table Holm correction",
    }
    (output_dir / "validation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("VALIDATION_SUMMARY_COMPLETE=true")
    print(f"VALIDATION_SUMMARY={output_dir / 'validation_model_summary.csv'}")


if __name__ == "__main__":
    main()
