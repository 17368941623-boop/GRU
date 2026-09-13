#!/usr/bin/env python3
"""Plot the continuous GRU lookback scan for the 15-step Thv forecast.

The requested range is lookback 15, 20, 25, ..., 60. The lookback-10 result
is deliberately omitted because the requested display begins at 15. Bars or
points show the mean across seeds 42, 62 and 82; uncertainty is one sample SD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


EXPECTED_LOOKBACKS = tuple(range(15, 61, 5))
EXPECTED_SEEDS = (42, 62, 82)
DPI = 600
ACCENT = "#D55E00"
BAR = "#8CB3D9"
MEAN = "#245A82"
SEED = "#9AAFC0"
EDGE = "#3F3F3F"
POINT = "#202020"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--all-runs-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qa-helper-dir", type=Path, required=True)
    parser.add_argument("--qa-output-dir", type=Path, required=True)
    return parser.parse_args()


def load_alignment_helper(helper_dir: Path):
    helper_dir = helper_dir.resolve()
    if not (helper_dir / "audit_panel_alignment.py").exists():
        raise FileNotFoundError(f"Missing alignment helper in {helper_dir}")
    sys.path.insert(0, str(helper_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    return require_matplotlib_panel_alignment


def validate_and_select(
    summary_all: pd.DataFrame,
    runs_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = set(summary_all["lookback"].astype(int))
    missing = sorted(set(EXPECTED_LOOKBACKS) - available)
    if missing:
        raise ValueError(f"Missing requested lookbacks: {missing}")

    # This is an explicit, documented display range requested by the user.
    excluded = sorted(available - set(EXPECTED_LOOKBACKS))
    print(f"EXCLUDED_LOOKBACKS={','.join(map(str, excluded)) or 'none'}")
    print("EXCLUSION_REASON=requested display begins at lookback 15")

    summary = summary_all.loc[
        summary_all["lookback"].astype(int).isin(EXPECTED_LOOKBACKS)
    ].copy()
    runs = runs_all.loc[
        runs_all["lookback"].astype(int).isin(EXPECTED_LOOKBACKS)
    ].copy()
    summary["lookback"] = summary["lookback"].astype(int)
    runs["lookback"] = runs["lookback"].astype(int)
    runs["seed"] = runs["seed"].astype(int)
    summary = summary.sort_values("lookback").reset_index(drop=True)
    runs = runs.sort_values(["lookback", "seed"]).reset_index(drop=True)

    if tuple(summary["lookback"]) != EXPECTED_LOOKBACKS:
        raise ValueError("Summary ordering or lookback values are inconsistent")
    if summary["seed_count"].astype(int).ne(3).any():
        raise ValueError("Every summary row must contain exactly three seeds")
    if runs["predict_steps"].astype(int).ne(15).any():
        raise ValueError("The forecast horizon must be 15 steps for every run")
    if summary["full_validation_windows_min"].nunique() != 1 or summary[
        "full_validation_windows_max"
    ].nunique() != 1:
        raise ValueError("Full-validation sample counts differ across lookbacks")
    if summary["rapid_validation_windows_min"].nunique() != 1 or summary[
        "rapid_validation_windows_max"
    ].nunique() != 1:
        raise ValueError("Rapid-cooling sample counts differ across lookbacks")

    for lookback, group in runs.groupby("lookback", sort=True):
        if tuple(group["seed"]) != EXPECTED_SEEDS:
            raise ValueError(f"Lookback {lookback} does not contain seeds 42, 62, 82")
        values = group["best_rapid_validation_rmse_k"].to_numpy(float)
        row = summary.loc[summary["lookback"].eq(lookback)].iloc[0]
        if not np.isclose(
            values.mean(), float(row["rapid_validation_rmse_mean_k"]), atol=1e-12
        ):
            raise ValueError(f"Mean RMSE mismatch at lookback {lookback}")
        if not np.isclose(
            values.std(ddof=1),
            float(row["rapid_validation_rmse_std_k"]),
            atol=1e-12,
        ):
            raise ValueError(f"RMSE SD mismatch at lookback {lookback}")

    minimum = int(
        summary.loc[summary["rapid_validation_rmse_mean_k"].idxmin(), "lookback"]
    )
    print(f"MINIMUM_MEAN_RMSE_LOOKBACK={minimum}")
    if minimum != 60:
        raise ValueError("The requested claim that lookback 60 is best is not supported")
    return summary, runs


def style_axis(axis: plt.Axes) -> None:
    axis.spines["left"].set_color(EDGE)
    axis.spines["bottom"].set_color(EDGE)
    axis.tick_params(colors=EDGE)
    axis.xaxis.label.set_color(EDGE)
    axis.yaxis.label.set_color(EDGE)
    axis.title.set_color(EDGE)
    axis.set_axisbelow(True)


def save_figure(
    figure: plt.Figure,
    base_name: str,
    output_dir: Path,
    qa_output_dir: Path,
    require_matplotlib_panel_alignment,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_output_dir.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(
        figure,
        axes=list(figure.axes),
        panel_ids=["a"],
        json_out=qa_output_dir / f"{base_name}.alignment.json",
        overlay_svg=qa_output_dir / f"{base_name}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    vector_kwargs = {"facecolor": "white", "transparent": False}
    raster_kwargs = {**vector_kwargs, "dpi": DPI}
    figure.savefig(output_dir / f"{base_name}.svg", **vector_kwargs)
    figure.savefig(output_dir / f"{base_name}.pdf", **vector_kwargs)
    figure.savefig(output_dir / f"{base_name}.png", **raster_kwargs)
    figure.savefig(output_dir / f"{base_name}.tiff", **raster_kwargs)
    plt.close(figure)


def plot_bar_chart(
    summary: pd.DataFrame,
    runs: pd.DataFrame,
    output_dir: Path,
    qa_output_dir: Path,
    require_matplotlib_panel_alignment,
) -> None:
    lookbacks = summary["lookback"].to_numpy(int)
    means = summary["rapid_validation_rmse_mean_k"].to_numpy(float)
    spread = summary["rapid_validation_rmse_std_k"].to_numpy(float)
    x = np.arange(len(lookbacks), dtype=float)
    colors = [ACCENT if value == 60 else BAR for value in lookbacks]

    figure, axis = plt.subplots(figsize=(7.20, 3.05))
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.20, top=0.86)
    axis.bar(
        x,
        means,
        width=0.68,
        color=colors,
        edgecolor=EDGE,
        linewidth=0.65,
        zorder=2,
    )
    axis.errorbar(
        x,
        means,
        yerr=spread,
        fmt="none",
        ecolor=EDGE,
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        zorder=4,
    )
    for index, lookback in enumerate(lookbacks):
        values = runs.loc[
            runs["lookback"].eq(lookback), "best_rapid_validation_rmse_k"
        ].to_numpy(float)
        axis.scatter(
            index + np.linspace(-0.10, 0.10, len(values)),
            values,
            s=10,
            c=POINT,
            linewidths=0,
            zorder=5,
        )
        axis.text(
            index,
            means[index] + spread[index] + 0.018,
            f"{means[index]:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=ACCENT if lookback == 60 else EDGE,
        )

    axis.set_xticks(x)
    axis.set_xticklabels([str(value) for value in lookbacks])
    axis.set_ylim(0, max(0.86, 1.14 * float(np.max(means + spread))))
    axis.set_xlabel("Lookback (samples)")
    axis.set_ylabel("Rapid-cooling validation RMSE (K)")
    axis.set_title("Thv forecast horizon = 15 steps; n = 3 seeds; mean ± SD")
    style_axis(axis)
    save_figure(
        figure,
        "lookback_15_60_rapid_rmse_bar",
        output_dir,
        qa_output_dir,
        require_matplotlib_panel_alignment,
    )


def plot_trend_chart(
    summary: pd.DataFrame,
    runs: pd.DataFrame,
    output_dir: Path,
    qa_output_dir: Path,
    require_matplotlib_panel_alignment,
) -> None:
    lookbacks = summary["lookback"].to_numpy(int)
    means = summary["rapid_validation_rmse_mean_k"].to_numpy(float)
    spread = summary["rapid_validation_rmse_std_k"].to_numpy(float)
    pivot = runs.pivot(
        index="lookback", columns="seed", values="best_rapid_validation_rmse_k"
    ).reindex(lookbacks)

    figure, axis = plt.subplots(figsize=(7.20, 3.05))
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.20, top=0.86)
    for seed in EXPECTED_SEEDS:
        axis.plot(
            lookbacks,
            pivot[seed].to_numpy(float),
            color=SEED,
            linewidth=0.8,
            marker="o",
            markersize=2.7,
            alpha=0.82,
            zorder=1,
        )
    axis.errorbar(
        lookbacks,
        means,
        yerr=spread,
        color=MEAN,
        linewidth=1.6,
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
        ecolor=MEAN,
        elinewidth=0.9,
        capsize=2.5,
        label="Mean ± SD",
        zorder=4,
    )
    selected_index = int(np.flatnonzero(lookbacks == 60)[0])
    axis.scatter(
        [60],
        [means[selected_index]],
        s=33,
        c=ACCENT,
        edgecolors=EDGE,
        linewidths=0.6,
        zorder=6,
        label=f"Lookback 60: minimum mean ({means[selected_index]:.3f} K)",
    )
    lower = float(np.min(pivot.to_numpy()))
    upper = float(np.max(pivot.to_numpy()))
    margin = 0.10 * (upper - lower)
    axis.set_ylim(lower - margin, upper + margin)
    axis.set_xticks(lookbacks)
    axis.set_xlabel("Lookback (samples)")
    axis.set_ylabel("Rapid-cooling validation RMSE (K)")
    axis.set_title("Lookback sensitivity for the 15-step Thv forecast")
    axis.legend(loc="upper right", ncols=2, fontsize=6.5)
    style_axis(axis)
    save_figure(
        figure,
        "lookback_15_60_rapid_rmse_trend",
        output_dir,
        qa_output_dir,
        require_matplotlib_panel_alignment,
    )


def main() -> None:
    args = parse_args()
    require_matplotlib_panel_alignment = load_alignment_helper(args.qa_helper_dir)
    summary, runs = validate_and_select(
        pd.read_csv(args.summary_csv), pd.read_csv(args.all_runs_csv)
    )
    plot_bar_chart(
        summary,
        runs,
        args.output_dir,
        args.qa_output_dir,
        require_matplotlib_panel_alignment,
    )
    plot_trend_chart(
        summary,
        runs,
        args.output_dir,
        args.qa_output_dir,
        require_matplotlib_panel_alignment,
    )
    print(f"FIGURE_OUTPUT_DIR={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
