#!/usr/bin/env python3
"""Freeze validation-only summaries for the complete 0910 heatmap study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from heatmap_protocol import (
    FEATURE_SPECS,
    MODEL_SPECS,
    SEEDS,
    SELECTION_METRIC,
    all_tasks,
    protocol_payload,
    run_directory,
    valid_completed_run,
)
from statistics_helpers import interval, paired_table

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "outputs_0910"
METRICS = (
    "validation_rmse_k",
    "validation_mae_k",
    "validation_p95_absolute_error_k",
    "validation_max_absolute_error_k",
    "validation_bias_k",
    "validation_r2",
    "validation_direction_accuracy",
)


def collect(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for model, feature, seed in all_tasks():
        identifier = f"{model['model']}/{feature['feature_set']}/seed_{seed}"
        if not valid_completed_run(results_dir, model, feature, seed):
            missing.append(identifier)
            continue
        path = run_directory(results_dir, model, feature, seed) / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload["validation_metrics"]
        rows.append({
            "model": model["model"],
            "model_label": model["label"],
            "model_short_label": model["short_label"],
            "model_role": model["role"],
            "feature_set": feature["feature_set"],
            "feature_set_label": feature["label"],
            "feature_stage_index": int(feature["stage_index"]),
            "introduced_feature_group": feature.get("introduced_group"),
            "seed": seed,
            "active_engineered_count": len(feature["engineered"]),
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
            "persistence_validation_rmse_k": float(
                payload["persistence_validation_metrics"]["rmse_k"]
            ),
            "checkpoint_selection_metric": payload["checkpoint_selection_metric"],
            "test_data_loaded": bool(payload["test_data_loaded"]),
            "source_metrics": str(path.relative_to(results_dir)),
        })
    output = results_dir / "validation_summary"
    output.mkdir(parents=True, exist_ok=True)
    if missing:
        (output / "missing_runs.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(missing)} of {len(all_tasks())} runs are incomplete")
    runs = pd.DataFrame(rows)
    if len(runs) != len(all_tasks()) or runs.test_data_loaded.any():
        raise ValueError("Training completeness or no-test-access contract failed")
    if set(runs.checkpoint_selection_metric) != {SELECTION_METRIC}:
        raise ValueError("Checkpoint selection metric changed")
    cell_sizes = runs.groupby(["model", "feature_set"]).seed.nunique()
    if not (cell_sizes == len(SEEDS)).all():
        raise ValueError("One or more heatmap cells do not contain all paired seeds")
    # Feature masking must not change parameter count within one architecture.
    if (runs.groupby("model").trainable_parameters.nunique() != 1).any():
        raise ValueError("Feature stages changed parameter count within an architecture")
    return runs


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, feature_set), group in runs.groupby(["model", "feature_set"], sort=False):
        row: dict[str, object] = {
            "model": model,
            "model_label": group.model_label.iloc[0],
            "model_short_label": group.model_short_label.iloc[0],
            "feature_set": feature_set,
            "feature_set_label": group.feature_set_label.iloc[0],
            "feature_stage_index": int(group.feature_stage_index.iloc[0]),
            "introduced_feature_group": group.introduced_feature_group.iloc[0],
            "n_seeds": group.seed.nunique(),
            "seed_set": ";".join(map(str, sorted(group.seed.unique()))),
            "active_engineered_count": int(group.active_engineered_count.iloc[0]),
            "trainable_parameters": int(group.trainable_parameters.iloc[0]),
            "training_minutes_mean": float(group.training_seconds_total.mean() / 60.0),
            "completed_epochs_mean": float(group.completed_epochs.mean()),
        }
        for metric in METRICS:
            values = interval(group[metric].to_numpy(float))
            for key in ("mean", "sd", "ci95_low", "ci95_high"):
                row[f"{metric}_{key}"] = values[key]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["feature_stage_index", "model"], kind="stable"
    ).reset_index(drop=True)


def paired_feature_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    tables = []
    feature_ids = [str(spec["feature_set"]) for spec in FEATURE_SPECS]
    for model in MODEL_SPECS:
        subset = runs.loc[runs.model.eq(model["model"])].copy()
        comparisons = []
        for index in range(1, len(feature_ids)):
            comparisons.append((
                f"{model['model']}__{feature_ids[index]}_vs_{feature_ids[index - 1]}",
                feature_ids[index],
                feature_ids[index - 1],
            ))
        # Stage 1 versus raw20 is already the first adjacent comparison.
        # Add only stages 2..7 here to avoid duplicating that row.
        comparisons.extend(
            (f"{model['model']}__{feature_id}_vs_raw20", feature_id, "raw20")
            for feature_id in feature_ids[2:]
        )
        table = paired_table(
            subset, comparisons, "validation_rmse_k", expected_seeds=len(SEEDS)
        )
        table.insert(0, "model", model["model"])
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def paired_architecture_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    architecture_ids = [str(spec["model"]) for spec in MODEL_SPECS]
    for feature in FEATURE_SPECS:
        subset = runs.loc[runs.feature_set.eq(feature["feature_set"])].copy()
        # Reuse paired_table by treating architectures as the comparison factor.
        renamed = subset.rename(columns={"feature_set": "original_feature_set"}).copy()
        renamed["feature_set"] = renamed["model"]
        comparisons = [
            (
                f"{feature['feature_set']}__{architecture}_vs_{architecture_ids[0]}",
                architecture,
                architecture_ids[0],
            )
            for architecture in architecture_ids[1:]
        ]
        table = paired_table(
            renamed, comparisons, "validation_rmse_k", expected_seeds=len(SEEDS)
        )
        table.insert(0, "feature_set", feature["feature_set"])
        table.insert(1, "feature_stage_index", int(feature["stage_index"]))
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def matrices(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    order_rows = [str(spec["feature_set"]) for spec in FEATURE_SPECS]
    order_columns = [str(spec["model"]) for spec in MODEL_SPECS]
    mean = summary.pivot(index="feature_set", columns="model", values="validation_rmse_k_mean")
    sd = summary.pivot(index="feature_set", columns="model", values="validation_rmse_k_sd")
    mean = mean.loc[order_rows, order_columns]
    sd = sd.loc[order_rows, order_columns]
    relative = 100.0 * mean.divide(mean.loc["raw20"], axis="columns") - 100.0
    return {"mean": mean, "sd": sd, "relative": relative}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    results = args.results_dir.resolve()
    output = results / "validation_summary"
    output.mkdir(parents=True, exist_ok=True)

    runs = collect(results)
    summary = summarize(runs)
    feature_pairs = paired_feature_comparisons(runs)
    architecture_pairs = paired_architecture_comparisons(runs)
    matrix = matrices(summary)

    runs.to_csv(output / "validation_seed_runs.csv", index=False)
    summary.to_csv(output / "validation_cell_summary.csv", index=False)
    feature_pairs.to_csv(output / "paired_cumulative_feature_effects.csv", index=False)
    architecture_pairs.to_csv(output / "paired_architecture_effects.csv", index=False)
    matrix["mean"].to_csv(output / "validation_rmse_mean_matrix.csv")
    matrix["sd"].to_csv(output / "validation_rmse_sd_matrix.csv")
    matrix["relative"].to_csv(output / "validation_relative_delta_pct_matrix.csv")

    audit = {
        **protocol_payload(),
        "observed_runs": len(runs),
        "complete": True,
        "test_data_read_by_training": False,
        "primary_ranking_metric": "complete Original 260501 validation RMSE",
        "uncertainty": "sample SD and two-sided 95% Student-t CI across five seeds",
        "statistical_caution": (
            "With five paired seeds, two-sided Wilcoxon tests are descriptive and "
            "cannot by themselves provide strong significance evidence."
        ),
    }
    (output / "validation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary[[
        "feature_stage_index", "feature_set", "model",
        "validation_rmse_k_mean", "validation_rmse_k_sd",
    ]].to_string(index=False))
    print("VALIDATION_SUMMARY_COMPLETE=true")


if __name__ == "__main__":
    main()
