#!/usr/bin/env python3
"""Create source-traceable validation figures and seed-robustness tables."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


MODEL_ORDER = [
    "gru_full20",
    "physical_all_lags_kan",
    "cd_select_lag_kan",
    "dkcdv_select_lag_kan",
    "dkcdl_select_lag_kan",
    "dkcdv_select_lag_mlp",
    "dkcdv_shuffled_lag_kan",
    "random_same_size_kan",
]
DISPLAY = {
    "gru_full20": "GRU",
    "physical_all_lags_kan": "Physical",
    "cd_select_lag_kan": "CD",
    "dkcdv_select_lag_kan": "DK&CDV-KAN",
    "dkcdl_select_lag_kan": "DK&CDL-KAN",
    "dkcdv_select_lag_mlp": "DK&CDV-MLP",
    "dkcdv_shuffled_lag_kan": "Shuffled lag",
    "random_same_size_kan": "Random graph",
}
DISPLAY_SHORT = {
    "gru_full20": "GRU",
    "physical_all_lags_kan": "Phys.",
    "cd_select_lag_kan": "CD",
    "dkcdv_select_lag_kan": "DV-KAN",
    "dkcdl_select_lag_kan": "DL-KAN",
    "dkcdv_select_lag_mlp": "DV-MLP",
    "dkcdv_shuffled_lag_kan": "Lag-shuf.",
    "random_same_size_kan": "Random",
}
COLORS = {
    "gru_full20": "#767676",
    "physical_all rand": "#CFCECE",
    "physical_all_lags_kan": "#9A9A9A",
    "cd_select_lag_kan": "#7884B4",
    "dkcdv_select_lag_kan": "#0F4D92",
    "dkcdl_select_lag_kan": "#B4C0E4",
    "dkcdv_select_lag_mlp": "#42949E",
    "dkcdv_shuffled_lag_kan": "#B8A7C9",
    "random_same_size_kan": "#D8D8D8",
}
PRIMARY = "dkcdv_select_lag_kan"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "reports" / "validation_analysis_20260910",
    )
    parser.add_argument("--bootstrap", type=int, default=20_000)
    return parser.parse_args()


def panel_label(ax: plt.Axes, label: str) -> None:
    offset = ScaledTranslation(-8 / 72, 3 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0, 1, label, transform=ax.transAxes + offset,
        ha="left", va="bottom", fontsize=8, fontweight="bold",
    )


def run_alignment(
    fig: plt.Figure, base: Path, axes: list[plt.Axes], panel_ids: list[str]
) -> None:
    scripts = os.environ.get("NATURE_FIGURE_SCRIPTS")
    if not scripts:
        raise RuntimeError(
            "Set NATURE_FIGURE_SCRIPTS to the nature-figure scripts directory for QA."
        )
    sys.path.insert(0, scripts)
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        fig,
        axes=axes,
        panel_ids=panel_ids,
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )


def export_figure(fig: plt.Figure, base: Path, axes: list[plt.Axes], panel_ids: list[str]) -> None:
    fig.canvas.draw()
    run_alignment(fig, base, axes, panel_ids)
    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=300)
    fig.savefig(str(base) + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def load_metrics(root: Path) -> pd.DataFrame:
    path = root / "outputs" / "validation_summary" / "horizon_15" / "all_seed_metrics.csv"
    table = pd.read_csv(path)
    for column in ("rmse_by_horizon_k", "mae_by_horizon_k"):
        table[column] = table[column].map(ast.literal_eval)
    expected = {(model, seed) for model in MODEL_ORDER for seed in range(42, 133, 10)}
    received = set(zip(table["experiment"], table["seed"]))
    if expected != received:
        raise ValueError(f"Expected a complete 8 x 10 grid; missing={sorted(expected - received)}")
    return table


def training_metadata(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata: list[dict[str, float | int | str]] = []
    histories: list[pd.DataFrame] = []
    base = root / "outputs" / "development" / "horizon_15" / "lookback_60"
    for path in sorted(base.glob("*/seed_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = pd.read_csv(path.parent / "training_history.csv")
        history.insert(0, "seed", int(payload["seed"]))
        history.insert(0, "experiment", str(payload["experiment"]))
        histories.append(history)
        metadata.append(
            {
                "experiment": str(payload["experiment"]),
                "seed": int(payload["seed"]),
                "best_epoch": int(payload["best_epoch"]),
                "stopped_epoch": int(history["epoch"].iloc[-1]),
                "training_seconds": float(payload["training_seconds"]),
                "parameter_count": int(payload["parameter_count"]),
                "best_validation_trajectory_rmse_k": float(
                    payload["validation"]["trajectory_rmse_k"]
                ),
            }
        )
    return pd.DataFrame(metadata), pd.concat(histories, ignore_index=True)


def prepare_source_data(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    histories: pd.DataFrame,
    output_dir: Path,
    bootstrap_runs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    scalar_columns = [
        "experiment", "seed", "samples", "trajectory_rmse_k", "trajectory_mae_k",
        "final_horizon_rmse_k", "rapid_trajectory_rmse_k",
        "valve_event_trajectory_rmse_k", "peak_cooling_delta_mae_k",
        "peak_cooling_time_mae_seconds",
    ]
    table[scalar_columns].to_csv(source_dir / "model_seed_metrics.csv", index=False)
    metadata.to_csv(source_dir / "training_metadata.csv", index=False)
    histories.to_csv(source_dir / "all_training_histories.csv", index=False)

    horizon_rows: list[dict[str, float | int | str]] = []
    for row in table.itertuples(index=False):
        for step, value in enumerate(row.rmse_by_horizon_k, start=1):
            horizon_rows.append(
                {
                    "experiment": row.experiment,
                    "seed": int(row.seed),
                    "horizon_step": step,
                    "horizon_seconds": step * 10,
                    "rmse_k": float(value),
                }
            )
    horizon = pd.DataFrame(horizon_rows)
    horizon.to_csv(source_dir / "horizon_rmse_by_seed.csv", index=False)

    primary = table.loc[
        table["experiment"].eq(PRIMARY), ["seed", "trajectory_rmse_k"]
    ].rename(columns={"trajectory_rmse_k": "kan_rmse_k"})
    rng = np.random.default_rng(20260910)
    difference_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    paired_tests = pd.read_csv(
        output_dir.parents[1]
        / "outputs" / "validation_summary" / "horizon_15"
        / "paired_primary_comparisons.csv"
    )
    for model in MODEL_ORDER:
        if model == PRIMARY:
            continue
        other = table.loc[
            table["experiment"].eq(model), ["seed", "trajectory_rmse_k"]
        ].rename(columns={"trajectory_rmse_k": "comparison_rmse_k"})
        paired = primary.merge(other, on="seed", validate="one_to_one")
        paired["difference_k"] = paired["comparison_rmse_k"] - paired["kan_rmse_k"]
        for row in paired.itertuples(index=False):
            difference_rows.append(
                {
                    "comparison": model,
                    "seed": int(row.seed),
                    "kan_rmse_k": float(row.kan_rmse_k),
                    "comparison_rmse_k": float(row.comparison_rmse_k),
                    "comparison_minus_kan_k": float(row.difference_k),
                }
            )
        values = paired["difference_k"].to_numpy(dtype=float)
        resampled = values[
            rng.integers(0, len(values), size=(bootstrap_runs, len(values)))
        ].mean(axis=1)
        test = paired_tests.loc[paired_tests["comparison"].eq(model)].iloc[0]
        summary_rows.append(
            {
                "comparison": model,
                "mean_comparison_minus_kan_k": float(values.mean()),
                "median_comparison_minus_kan_k": float(np.median(values)),
                "seed_bootstrap_95_low_k": float(np.quantile(resampled, 0.025)),
                "seed_bootstrap_95_high_k": float(np.quantile(resampled, 0.975)),
                "kan_wins_of_10": int(np.sum(values > 0)),
                "wilcoxon_p_value": float(test["p_value"]),
                "holm_p_value": float(test["holm_p_value"]),
            }
        )
    differences = pd.DataFrame(difference_rows)
    paired_summary = pd.DataFrame(summary_rows)
    differences.to_csv(source_dir / "paired_seed_differences_vs_kan.csv", index=False)
    paired_summary.to_csv(source_dir / "paired_summary_vs_kan.csv", index=False)
    return horizon, differences, paired_summary


def plot_model_comparison(
    table: pd.DataFrame,
    horizon: pd.DataFrame,
    differences: pd.DataFrame,
    paired_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(7.20, 5.05))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.04, 1.0])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.155, top=0.955, hspace=0.53, wspace=0.40)

    curve_models = [PRIMARY, "dkcdv_select_lag_mlp", "gru_full20", "random_same_size_kan"]
    for model in curve_models:
        group = horizon.loc[horizon["experiment"].eq(model)]
        pivot = group.pivot(index="seed", columns="horizon_seconds", values="rmse_k")
        x = pivot.columns.to_numpy(dtype=float)
        mean = pivot.mean(axis=0).to_numpy(dtype=float)
        sd = pivot.std(axis=0, ddof=1).to_numpy(dtype=float)
        color = COLORS[model]
        width = 2.0 if model == PRIMARY else 1.25
        zorder = 5 if model == PRIMARY else 3
        ax_a.plot(x, mean, color=color, lw=width, label=DISPLAY[model], zorder=zorder)
        ax_a.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.14, linewidth=0)
    ax_a.set_xlim(10, 150)
    ax_a.set_xticks([10, 30, 60, 90, 120, 150])
    ax_a.set_xlabel("Forecast horizon (s)")
    ax_a.set_ylabel("RMSE (K)")
    ax_a.grid(axis="y", color="#E5E5E5", lw=0.6)
    ax_a.legend(ncol=4, loc="upper left", handlelength=2.5, columnspacing=1.1)
    panel_label(ax_a, "a")

    distributions = [
        table.loc[table["experiment"].eq(model), "trajectory_rmse_k"].to_numpy(dtype=float)
        for model in MODEL_ORDER
    ]
    positions = np.arange(1, len(MODEL_ORDER) + 1)
    box = ax_b.boxplot(
        distributions, positions=positions, widths=0.58, patch_artist=True,
        showfliers=False,
        medianprops={"color": "#272727", "linewidth": 1.0},
        whiskerprops={"color": "#606060", "linewidth": 0.8},
        capprops={"color": "#606060", "linewidth": 0.8},
        boxprops={"edgecolor": "#606060", "linewidth": 0.8},
    )
    for patch, model in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.78 if model == PRIMARY else 0.52)
    jitter_rng = np.random.default_rng(20260910)
    for position, model, values in zip(positions, MODEL_ORDER, distributions):
        jitter = jitter_rng.uniform(-0.12, 0.12, size=len(values))
        ax_b.scatter(
            position + jitter, values, s=9,
            facecolor="white" if model != PRIMARY else COLORS[model],
            edgecolor=COLORS[model] if model != PRIMARY else "white",
            linewidth=0.55, zorder=4,
        )
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(
        [DISPLAY_SHORT[model] for model in MODEL_ORDER], rotation=58, ha="right",
        rotation_mode="anchor", fontsize=6.2,
    )
    ax_b.set_ylabel("Trajectory RMSE (K)")
    ax_b.grid(axis="y", color="#E5E5E5", lw=0.6)
    panel_label(ax_b, "b")

    comparison_order = [model for model in MODEL_ORDER if model != PRIMARY]
    y = np.arange(len(comparison_order))[::-1]
    for yi, model in zip(y, comparison_order):
        points = differences.loc[
            differences["comparison"].eq(model), "comparison_minus_kan_k"
        ].to_numpy(dtype=float)
        summary = paired_summary.loc[paired_summary["comparison"].eq(model)].iloc[0]
        ax_c.scatter(points, np.full(len(points), yi), s=8, color="#A8A8A8", alpha=0.70, zorder=2)
        ax_c.plot(
            [summary["seed_bootstrap_95_low_k"], summary["seed_bootstrap_95_high_k"]],
            [yi, yi], color=COLORS[model], lw=1.4, zorder=3,
        )
        ax_c.scatter(
            summary["mean_comparison_minus_kan_k"], yi, marker="s", s=24,
            color=COLORS[model], edgecolor="#272727", linewidth=0.5, zorder=4,
        )
    ax_c.axvline(0, color="#606060", linestyle="--", lw=0.8)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([DISPLAY_SHORT[model] for model in comparison_order], fontsize=6.4)
    ax_c.set_xlabel("Comparator - KAN RMSE (K; positive favours KAN)")
    ax_c.set_title("Paired differences across matched seeds", fontsize=7, pad=5)
    ax_c.grid(axis="x", color="#E5E5E5", lw=0.6)
    panel_label(ax_c, "c")

    export_figure(
        fig, output_dir / "validation_model_comparison",
        [ax_a, ax_b, ax_c], ["a", "b", "c"],
    )


def plot_kan_convergence(
    metadata: pd.DataFrame, histories: pd.DataFrame, output_dir: Path
) -> None:
    selected_meta = metadata.loc[metadata["experiment"].eq(PRIMARY)].sort_values("seed")
    selected_history = histories.loc[histories["experiment"].eq(PRIMARY)]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.20, 2.75))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.175, top=0.925, wspace=0.34)

    for row in selected_meta.itertuples(index=False):
        part = selected_history.loc[selected_history["seed"].eq(row.seed)]
        ax_a.plot(
            part["epoch"], part["validation_trajectory_rmse_k"],
            color="#3775BA", alpha=0.37, lw=0.9,
        )
        best = part.loc[part["epoch"].eq(row.best_epoch)].iloc[0]
        ax_a.scatter(
            best["epoch"], best["validation_trajectory_rmse_k"],
            s=14, color="#0F4D92", edgecolor="white", linewidth=0.45, zorder=4,
        )
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Validation trajectory RMSE (K)")
    ax_a.grid(color="#E5E5E5", lw=0.6)
    panel_label(ax_a, "a")

    seed_positions = np.arange(len(selected_meta))
    ax_b.scatter(
        seed_positions, selected_meta["best_validation_trajectory_rmse_k"],
        s=28, color="#0F4D92", edgecolor="white", linewidth=0.6,
    )
    ax_b.set_xticks(seed_positions)
    ax_b.set_xticklabels(
        selected_meta["seed"].astype(str), rotation=0,
        rotation_mode="anchor", fontsize=6.2,
    )
    ax_b.set_xlabel("Random seed")
    ax_b.set_ylabel("Best validation trajectory RMSE (K)")
    ax_b.grid(color="#E5E5E5", lw=0.6)
    panel_label(ax_b, "b")

    export_figure(
        fig, output_dir / "kan_training_convergence",
        [ax_a, ax_b], ["a", "b"],
    )


def write_summary(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    paired_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    model_summary = (
        table.groupby("experiment", sort=False)[
            [
                "trajectory_rmse_k", "trajectory_mae_k", "final_horizon_rmse_k",
                "rapid_trajectory_rmse_k", "valve_event_trajectory_rmse_k",
                "peak_cooling_delta_mae_k",
            ]
        ]
        .agg(["mean", "std"])
    )
    kan = model_summary.loc[PRIMARY]
    kan_meta = metadata.loc[metadata["experiment"].eq(PRIMARY)]
    payload = {
        "analysis_scope": "validation only; frozen test not opened",
        "validation_parent_runs": 1,
        "validation_windows": int(
            table.loc[table["experiment"].eq(PRIMARY), "samples"].iloc[0]
        ),
        "random_seeds": 10,
        "seed_replication_note": "optimization repeats on the same split, not independent process runs",
        "kan": {
            "trajectory_rmse_mean_k": float(kan[("trajectory_rmse_k", "mean")]),
            "trajectory_rmse_seed_sd_k": float(kan[("trajectory_rmse_k", "std")]),
            "trajectory_mae_mean_k": float(kan[("trajectory_mae_k", "mean")]),
            "final_horizon_rmse_mean_k": float(kan[("final_horizon_rmse_k", "mean")]),
            "rapid_trajectory_rmse_mean_k": float(kan[("rapid_trajectory_rmse_k", "mean")]),
            "valve_event_trajectory_rmse_mean_k": float(
                kan[("valve_event_trajectory_rmse_k", "mean")]
            ),
            "peak_cooling_delta_mae_mean_k": float(
                kan[("peak_cooling_delta_mae_k", "mean")]
            ),
            "best_epoch_median": float(kan_meta["best_epoch"].median()),
            "best_epoch_range": [
                int(kan_meta["best_epoch"].min()), int(kan_meta["best_epoch"].max())
            ],
            "seed_rmse_range_k": [
                float(kan_meta["best_validation_trajectory_rmse_k"].min()),
                float(kan_meta["best_validation_trajectory_rmse_k"].max()),
            ],
        },
        "paired_summary": paired_summary.to_dict(orient="records"),
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1000:
        raise ValueError("Use at least 1000 paired-seed bootstrap resamples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = load_metrics(args.root)
    metadata, histories = training_metadata(args.root)
    horizon, differences, paired_summary = prepare_source_data(
        table, metadata, histories, args.output_dir, args.bootstrap
    )
    plot_model_comparison(table, horizon, differences, paired_summary, args.output_dir)
    plot_kan_convergence(metadata, histories, args.output_dir)
    write_summary(table, metadata, paired_summary, args.output_dir)
    print(json.dumps({"status": "PASS", "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
