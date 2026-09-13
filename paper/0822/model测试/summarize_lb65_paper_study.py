#!/usr/bin/env python3
"""Create paper-ready tables and figures for the lookback=65 model study.

All seed-level observations are retained.  Means are accompanied by two-sided
95% Student-t confidence intervals, and model comparisons use paired seeds.
Wilcoxon signed-rank p values are Holm-adjusted within each reported metric.
The held-out test is summarized, never used to select or rank configurations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "modelout_lb65_paper"
LOOKBACK = 65
PREDICT_STEPS = 15
PROPOSED = "parallel_gru_kan_gnn"

DISPLAY_NAMES = {
    "gru_baseline": "GRU baseline",
    "lstm_baseline": "LSTM baseline",
    "tcn_baseline": "TCN baseline",
    "serial_mlp_gnn_gru": "Serial MLP–GNN–GRU",
    "serial_kan_gnn_gru": "Serial KAN–GNN–GRU",
    "parallel_gru_mlp_gnn": "Parallel GRU–MLP–GNN",
    "parallel_gru_kan_gnn": "Parallel GRU–KAN–GNN (proposed)",
}

DISPLAY_ORDER = (
    "parallel_gru_kan_gnn",
    "parallel_gru_mlp_gnn",
    "serial_kan_gnn_gru",
    "serial_mlp_gnn_gru",
    "gru_baseline",
    "lstm_baseline",
    "tcn_baseline",
)

COLORS = {
    "parallel_gru_kan_gnn": "#0072B2",
    "parallel_gru_mlp_gnn": "#56B4E9",
    "serial_kan_gnn_gru": "#009E73",
    "serial_mlp_gnn_gru": "#E69F00",
    "gru_baseline": "#5F6368",
    "lstm_baseline": "#8A8D91",
    "tcn_baseline": "#B0B3B8",
}

ABLATION_CONTRASTS = (
    {
        "effect": "remove_physical_graph_branch",
        "label": "Remove physical graph\n(GRU only − proposed)",
        "ablation_model": "gru_baseline",
        "reference_model": PROPOSED,
    },
    {
        "effect": "kan_to_mlp_parallel",
        "label": "KAN → MLP, parallel\n(MLP − KAN)",
        "ablation_model": "parallel_gru_mlp_gnn",
        "reference_model": PROPOSED,
    },
    {
        "effect": "parallel_to_serial_kan",
        "label": "Parallel → serial, KAN\n(serial − parallel)",
        "ablation_model": "serial_kan_gnn_gru",
        "reference_model": PROPOSED,
    },
    {
        "effect": "kan_to_mlp_serial",
        "label": "KAN → MLP, serial\n(MLP − KAN)",
        "ablation_model": "serial_mlp_gnn_gru",
        "reference_model": "serial_kan_gnn_gru",
    },
    {
        "effect": "gru_to_lstm_baseline",
        "label": "GRU → LSTM baseline\n(LSTM − GRU)",
        "ablation_model": "lstm_baseline",
        "reference_model": "gru_baseline",
    },
    {
        "effect": "gru_to_tcn_baseline",
        "label": "GRU → TCN baseline\n(TCN − GRU)",
        "ablation_model": "tcn_baseline",
        "reference_model": "gru_baseline",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def development_dir(results_dir: Path) -> Path:
    return results_dir / "development" / f"horizon_{PREDICT_STEPS:02d}" / f"lookback_{LOOKBACK:02d}"


def final_test_path(results_dir: Path) -> Path:
    return (
        results_dir
        / "final_test"
        / f"horizon_{PREDICT_STEPS:02d}"
        / f"lookback_{LOOKBACK:02d}"
        / "final_all_seed_runs.csv"
    )


def read_protocol(results_dir: Path) -> dict[str, object]:
    return json.loads((results_dir / "study_protocol.json").read_text(encoding="utf-8"))


def mean_ci(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("mean_ci requires a finite, non-empty one-dimensional sample")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    sem = std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    half = float(stats.t.ppf(0.975, len(array) - 1) * sem) if len(array) > 1 else 0.0
    return {
        "n_seeds": int(len(array)),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(p_values), dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_statistics(
    table: pd.DataFrame,
    metric: str,
    ablation_model: str,
    reference_model: str,
) -> dict[str, float | int]:
    left = table.loc[table["model"] == ablation_model, ["seed", metric]].rename(columns={metric: "ablation"})
    right = table.loc[table["model"] == reference_model, ["seed", metric]].rename(columns={metric: "reference"})
    paired = left.merge(right, on="seed", how="inner", validate="one_to_one").sort_values("seed")
    if paired.empty:
        raise ValueError(f"no paired seeds for {ablation_model} versus {reference_model}")
    differences = paired["ablation"].to_numpy(dtype=np.float64) - paired["reference"].to_numpy(dtype=np.float64)
    summary = mean_ci(differences)
    if np.allclose(differences, 0.0):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
    return {
        "n_paired_seeds": int(len(differences)),
        "mean_difference_ablation_minus_reference_k": float(summary["mean"]),
        "std_difference_k": float(summary["std"]),
        "ci95_low_k": float(summary["ci95_low"]),
        "ci95_high_k": float(summary["ci95_high"]),
        "wilcoxon_p_raw": p_value,
    }


def collect_validation_runs(results_dir: Path, protocol: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = development_dir(results_dir)
    seeds = tuple(int(seed) for seed in protocol["seeds"])
    for spec in protocol["model_specs"]:
        model = str(spec["model"])
        config_id = str(spec["config_id"])
        for seed in seeds:
            metrics_path = base / model / config_id / f"seed_{seed}" / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            required = {
                "model_type": model,
                "config_id": config_id,
                "seed": seed,
                "lookback": LOOKBACK,
                "common_origin_lookback": int(protocol["common_origin_lookback"]),
                "predict_steps": PREDICT_STEPS,
                "test_data_loaded": False,
            }
            if any(metrics.get(key) != value for key, value in required.items()):
                raise ValueError(f"development metadata audit failed: {metrics_path}")
            rows.append(
                {
                    "model": model,
                    "display_name": DISPLAY_NAMES[model],
                    "config_id": config_id,
                    "paper_role": spec["paper_role"],
                    "seed": seed,
                    "trainable_parameters": int(metrics["trainable_parameters"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "completed_epochs": int(metrics["completed_epochs"]),
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
                    "validation_persistence_rmse_k": float(metrics["persistence_validation_metrics"]["rmse_k"]),
                    "validation_rapid_persistence_rmse_k": float(metrics["rapid_persistence_validation_metrics"]["rmse_k"]),
                    "validation_windows": int(metrics["validation_windows"]),
                    "validation_rapid_windows": int(metrics["validation_rapid_windows"]),
                }
            )
    frame = pd.DataFrame(rows)
    expected = len(protocol["model_specs"]) * len(seeds)
    if len(frame) != expected:
        raise ValueError(f"expected {expected} validation runs, found {len(frame)}")
    return frame


def aggregate_models(table: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in DISPLAY_ORDER:
        subset = table.loc[table["model"] == model]
        row: dict[str, object] = {
            "model": model,
            "display_name": DISPLAY_NAMES[model],
            "n_seeds": int(subset["seed"].nunique()),
            "trainable_parameters": int(subset["trainable_parameters"].iloc[0]),
            "training_seconds_mean": float(subset["training_seconds_total"].mean()),
        }
        for metric in metrics:
            summary = mean_ci(subset[metric].to_numpy(dtype=np.float64))
            for suffix in ("mean", "std", "sem", "ci95_low", "ci95_high"):
                row[f"{metric}_{suffix}"] = summary[suffix]
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(table: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        metric_rows: list[dict[str, object]] = []
        for competitor in DISPLAY_ORDER:
            if competitor == PROPOSED:
                continue
            row: dict[str, object] = {
                "metric": metric,
                "competitor_model": competitor,
                "competitor_display_name": DISPLAY_NAMES[competitor],
                "reference_model": PROPOSED,
                "reference_display_name": DISPLAY_NAMES[PROPOSED],
            }
            row.update(paired_statistics(table, metric, competitor, PROPOSED))
            metric_rows.append(row)
        adjusted = holm_adjust(row["wilcoxon_p_raw"] for row in metric_rows)
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["wilcoxon_p_holm"] = float(adjusted_p)
            rows.append(row)
    return pd.DataFrame(rows)


def ablation_table(table: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        metric_rows: list[dict[str, object]] = []
        for contrast in ABLATION_CONTRASTS:
            row: dict[str, object] = {
                "metric": metric,
                "effect": contrast["effect"],
                "display_label": contrast["label"].replace("\n", " "),
                "ablation_model": contrast["ablation_model"],
                "reference_model": contrast["reference_model"],
                "difference_definition": "ablation minus reference; positive means the replacement is worse",
            }
            row.update(
                paired_statistics(
                    table,
                    metric,
                    str(contrast["ablation_model"]),
                    str(contrast["reference_model"]),
                )
            )
            metric_rows.append(row)
        adjusted = holm_adjust(row["wilcoxon_p_raw"] for row in metric_rows)
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["wilcoxon_p_holm"] = float(adjusted_p)
            rows.append(row)
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_single_panel(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    axis = fig.axes[0]
    bounds = axis.get_position().bounds
    alignment = {
        "verdict": "NOT APPLICABLE",
        "reason": "single rendered plot area",
        "axes_bounds_figure_fraction": [float(value) for value in bounds],
    }
    output_stem.with_suffix(".alignment.json").write_text(
        json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_model_metric(
    runs: pd.DataFrame,
    metric: str,
    xlabel: str,
    title: str,
    output_stem: Path,
) -> None:
    source = runs.loc[:, ["model", "display_name", "seed", metric]].copy()
    source.to_csv(output_stem.with_name(output_stem.name + "_source_data.csv"), index=False)
    fig, axis = plt.subplots(figsize=(7.20, 3.75))
    positions = np.arange(len(DISPLAY_ORDER), dtype=np.float64)
    for y, model in zip(positions, DISPLAY_ORDER):
        values = source.loc[source["model"] == model, metric].to_numpy(dtype=np.float64)
        summary = mean_ci(values)
        color = COLORS[model]
        axis.scatter(values, np.full_like(values, y), s=12, color=color, alpha=0.35, linewidths=0, zorder=2)
        axis.errorbar(
            float(summary["mean"]),
            y,
            xerr=np.array(
                [[float(summary["mean"]) - float(summary["ci95_low"])],
                 [float(summary["ci95_high"]) - float(summary["mean"])]]
            ),
            fmt="D" if model == PROPOSED else "o",
            markersize=5.2 if model == PROPOSED else 4.2,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=3,
        )
    axis.set_yticks(positions, [DISPLAY_NAMES[model] for model in DISPLAY_ORDER])
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc="left", pad=8)
    axis.grid(axis="x", color="#D9DDE3", linewidth=0.6, alpha=0.75)
    axis.set_axisbelow(True)
    axis.margins(x=0.08, y=0.12)
    fig.text(
        0.995,
        0.012,
        "Points: individual seeds; marker and whisker: mean and 95% t-CI (n=10).",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#4C4F54",
    )
    fig.subplots_adjust(left=0.37, right=0.98, top=0.88, bottom=0.18)
    save_single_panel(fig, output_stem)


def plot_ablation_effects(
    runs: pd.DataFrame,
    metric: str,
    title: str,
    output_stem: Path,
) -> None:
    rows: list[dict[str, object]] = []
    fig, axis = plt.subplots(figsize=(7.20, 3.65))
    contrasts = ABLATION_CONTRASTS[:4]
    positions = np.arange(len(contrasts), dtype=np.float64)
    for y, contrast in zip(positions, contrasts):
        left = runs.loc[runs["model"] == contrast["ablation_model"], ["seed", metric]].rename(columns={metric: "ablation"})
        right = runs.loc[runs["model"] == contrast["reference_model"], ["seed", metric]].rename(columns={metric: "reference"})
        paired = left.merge(right, on="seed", validate="one_to_one").sort_values("seed")
        differences = paired["ablation"].to_numpy(dtype=np.float64) - paired["reference"].to_numpy(dtype=np.float64)
        summary = mean_ci(differences)
        for seed, difference in zip(paired["seed"], differences):
            rows.append(
                {
                    "effect": contrast["effect"],
                    "display_label": contrast["label"].replace("\n", " "),
                    "seed": int(seed),
                    "paired_rmse_difference_k": float(difference),
                }
            )
        axis.scatter(differences, np.full_like(differences, y), s=13, color="#7A7F87", alpha=0.42, linewidths=0, zorder=2)
        axis.errorbar(
            float(summary["mean"]),
            y,
            xerr=np.array(
                [[float(summary["mean"]) - float(summary["ci95_low"])],
                 [float(summary["ci95_high"]) - float(summary["mean"])]]
            ),
            fmt="D",
            markersize=5,
            color="#0072B2",
            ecolor="#0072B2",
            elinewidth=1.25,
            capsize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=3,
        )
    pd.DataFrame(rows).to_csv(output_stem.with_name(output_stem.name + "_source_data.csv"), index=False)
    axis.axvline(0.0, color="#34373C", linewidth=0.9, linestyle="--", zorder=1)
    axis.set_yticks(positions, [str(contrast["label"]) for contrast in contrasts])
    axis.invert_yaxis()
    axis.set_xlabel("Paired rapid-cooling RMSE difference (ablation − reference, K)")
    axis.set_title(title, loc="left", pad=8)
    axis.grid(axis="x", color="#D9DDE3", linewidth=0.6, alpha=0.75)
    axis.set_axisbelow(True)
    axis.margins(x=0.12, y=0.22)
    fig.text(
        0.995,
        0.012,
        "Positive values favor the reference model. Points are paired seeds; whiskers are 95% t-CIs.",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#4C4F54",
    )
    fig.subplots_adjust(left=0.34, right=0.98, top=0.87, bottom=0.20)
    save_single_panel(fig, output_stem)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    analysis_dir = results_dir / "paper_analysis"
    figures_dir = analysis_dir / "figures"
    tables_dir = analysis_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    protocol = read_protocol(results_dir)
    validation = collect_validation_runs(results_dir, protocol)
    validation.to_csv(tables_dir / "paper_validation_seed_runs.csv", index=False)

    validation_metrics = (
        "validation_rmse_k",
        "validation_mae_k",
        "validation_p95_absolute_error_k",
        "validation_rapid_rmse_k",
        "validation_rapid_mae_k",
        "validation_rapid_p95_absolute_error_k",
        "validation_direction_accuracy",
        "validation_rapid_direction_accuracy",
    )
    validation_summary = aggregate_models(validation, validation_metrics)
    validation_summary.to_csv(tables_dir / "paper_validation_model_summary.csv", index=False)

    test_csv = final_test_path(results_dir)
    if not test_csv.exists():
        raise FileNotFoundError(f"final held-out test summary is missing: {test_csv}")
    test = pd.read_csv(test_csv)
    expected_keys = validation[["model", "config_id", "seed"]].sort_values(["model", "config_id", "seed"]).reset_index(drop=True)
    actual_keys = test[["model", "config_id", "seed"]].sort_values(["model", "config_id", "seed"]).reset_index(drop=True)
    if not expected_keys.equals(actual_keys):
        raise ValueError("held-out test rows do not match the completed development runs")
    test["display_name"] = test["model"].map(DISPLAY_NAMES)
    test.to_csv(tables_dir / "paper_test_seed_runs.csv", index=False)

    combined = validation.merge(
        test.drop(columns=["model_description", "trainable_parameters"], errors="ignore"),
        on=["model", "config_id", "seed"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_test_file"),
    )
    test_metrics = (
        "test_full_rmse_k",
        "test_full_mae_k",
        "test_full_p95_absolute_error_k",
        "test_rapid_rmse_k",
        "test_rapid_mae_k",
        "test_rapid_p95_absolute_error_k",
        "fixed_window_rmse_k",
        "fixed_window_rapid_rmse_k",
        "test_direction_accuracy",
        "test_rapid_direction_accuracy",
    )
    all_metrics = validation_metrics + test_metrics
    model_summary = aggregate_models(combined, all_metrics)
    model_summary.to_csv(tables_dir / "paper_model_summary.csv", index=False)

    comparison_metrics = (
        "validation_rapid_rmse_k",
        "test_rapid_rmse_k",
        "test_full_rmse_k",
        "fixed_window_rapid_rmse_k",
    )
    comparisons = comparison_table(combined, comparison_metrics)
    comparisons.to_csv(tables_dir / "paper_paired_model_comparisons.csv", index=False)
    ablations = ablation_table(combined, comparison_metrics)
    ablations.to_csv(tables_dir / "paper_ablation_effects.csv", index=False)

    configure_plotting()
    plot_model_metric(
        combined,
        "validation_rapid_rmse_k",
        "Rapid-cooling RMSE on Original 260501 validation (K)",
        "Controlled model comparison at lookback = 65",
        figures_dir / "lb65_validation_rapid_rmse_comparison",
    )
    plot_model_metric(
        combined,
        "test_rapid_rmse_k",
        "Rapid-cooling RMSE on held-out Original 0715-BACK (K)",
        "Held-out rapid-cooling performance at lookback = 65",
        figures_dir / "lb65_test_rapid_rmse_comparison",
    )
    plot_model_metric(
        combined,
        "test_full_rmse_k",
        "Full held-out Original 0715-BACK RMSE (K)",
        "Held-out full-sequence performance at lookback = 65",
        figures_dir / "lb65_test_full_rmse_comparison",
    )
    plot_ablation_effects(
        combined,
        "test_rapid_rmse_k",
        "Paired component effects on held-out rapid-cooling RMSE",
        figures_dir / "lb65_test_rapid_rmse_ablation_effects",
    )

    best_validation = model_summary.sort_values("validation_rapid_rmse_k_mean").iloc[0]
    best_test = model_summary.sort_values("test_rapid_rmse_k_mean").iloc[0]
    manifest = {
        "lookback": LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "n_models": len(DISPLAY_ORDER),
        "n_seeds_per_model": int(combined["seed"].nunique()),
        "test_was_not_used_for_configuration_selection": True,
        "primary_metric": "rapid-cooling RMSE",
        "center": "arithmetic mean across seeds",
        "interval": "two-sided 95% Student-t confidence interval across seeds",
        "paired_test": "two-sided Wilcoxon signed-rank test on matched seeds",
        "multiple_comparison_correction": "Holm adjustment within each metric table",
        "best_validation_rapid_rmse_model": str(best_validation["model"]),
        "best_test_rapid_rmse_model_descriptive_only": str(best_test["model"]),
        "figure_archetype": "single-panel quantitative comparisons with paired-effect decomposition",
        "figure_claim": (
            "Assess whether the parallel GRU–KAN–GNN improves rapid-cooling "
            "prediction and identify which structural components account for the change."
        ),
        "figure_qa_status": "exports generated; final rendered PDF collision and glyph audits remain required before manuscript delivery",
    }
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(model_summary.to_string(index=False), flush=True)
    print(f"PAPER_ANALYSIS={analysis_dir}", flush=True)


if __name__ == "__main__":
    main()
