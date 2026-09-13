#!/usr/bin/env python3
"""Plot complete-test RMSE versus lookback for the frozen Thv GRU models.

The legacy filename is retained at the user's request. The plotted metric is
the full-sequence 0715-BACK test RMSE, not a rapid-cooling subset metric.
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


BASE_NAME = Path(__file__).stem
EXPECTED_LOOKBACKS = (15, 30, 40, 45, 50, 55, 60, 65, 70, 75, 90, 120)
EXPECTED_SEEDS = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)
PLATEAU_START = 40
NUMERICAL_MINIMUM = 60
DPI = 600

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    PROJECT_DIR
    / "残差lookback测试"
    / "gru_lookback_final_test0715BACK_full"
)
DEFAULT_QA_HELPER_DIR = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"

COLORS = {
    "mean": "#245A82",
    "raw": "#8CAAC0",
    "plateau": "#EAF1F6",
    "minimum": "#D55E00",
    "edge": "#3F3F3F",
    "grid": "#D9DEE3",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "lookback_test_summary.csv",
    )
    parser.add_argument(
        "--all-runs-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "lookback_test_all_runs.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--qa-output-dir", type=Path, default=Path(__file__).parent / "qa")
    parser.add_argument("--qa-helper-dir", type=Path, default=DEFAULT_QA_HELPER_DIR)
    return parser.parse_args()


def validate_data(
    summary: pd.DataFrame, runs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_summary = {
        "lookback",
        "seed_count",
        "test_full_rmse_mean_k",
        "test_full_rmse_std_k",
        "test_windows",
        "persistence_test_rmse_k",
    }
    required_runs = {"lookback", "seed", "test_full_rmse_k", "test_windows"}
    if missing := required_summary - set(summary.columns):
        raise ValueError(f"Summary is missing columns: {sorted(missing)}")
    if missing := required_runs - set(runs.columns):
        raise ValueError(f"All-runs table is missing columns: {sorted(missing)}")

    summary = summary.copy()
    runs = runs.copy()
    summary["lookback"] = summary["lookback"].astype(int)
    runs["lookback"] = runs["lookback"].astype(int)
    runs["seed"] = runs["seed"].astype(int)
    summary = summary.sort_values("lookback").reset_index(drop=True)
    runs = runs.sort_values(["lookback", "seed"]).reset_index(drop=True)

    if tuple(summary["lookback"]) != EXPECTED_LOOKBACKS:
        raise ValueError(
            f"Expected lookbacks {EXPECTED_LOOKBACKS}, got {tuple(summary['lookback'])}"
        )
    if set(runs["lookback"]) != set(EXPECTED_LOOKBACKS):
        raise ValueError("All-runs lookbacks do not match the summary")
    if summary["seed_count"].astype(int).ne(len(EXPECTED_SEEDS)).any():
        raise ValueError("Every lookback must contain ten seeds")
    if runs.duplicated(["lookback", "seed"]).any():
        raise ValueError("Duplicate lookback/seed rows detected")
    if runs["test_windows"].nunique() != 1:
        raise ValueError("Full-test window counts differ across runs")
    if int(runs["test_windows"].iloc[0]) != 10659:
        raise ValueError("Unexpected number of common full-test forecast origins")

    for lookback, group in runs.groupby("lookback", sort=True):
        seeds = tuple(group["seed"].sort_values())
        if seeds != EXPECTED_SEEDS:
            raise ValueError(f"Lookback {lookback} has seeds {seeds}")
        values = group["test_full_rmse_k"].to_numpy(float)
        row = summary.loc[summary["lookback"].eq(lookback)].iloc[0]
        if not np.isclose(values.mean(), row["test_full_rmse_mean_k"], atol=1e-12):
            raise ValueError(f"Mean mismatch at lookback {lookback}")
        if not np.isclose(values.std(ddof=1), row["test_full_rmse_std_k"], atol=1e-12):
            raise ValueError(f"SD mismatch at lookback {lookback}")

    minimum = int(summary.loc[summary["test_full_rmse_mean_k"].idxmin(), "lookback"])
    if minimum != NUMERICAL_MINIMUM:
        raise ValueError(f"Expected numerical minimum {NUMERICAL_MINIMUM}, got {minimum}")
    plateau = summary.loc[summary["lookback"].ge(PLATEAU_START)]
    plateau_range = float(
        plateau["test_full_rmse_mean_k"].max()
        - plateau["test_full_rmse_mean_k"].min()
    )
    print(f"METRIC=complete_0715-BACK_test_RMSE")
    print(f"TEST_WINDOWS={int(runs['test_windows'].iloc[0])}")
    print(f"NUMERICAL_MINIMUM_LOOKBACK={minimum}")
    print(f"PLATEAU_MEAN_RANGE_K={plateau_range:.6f}")
    return summary, runs


def load_alignment_helper(helper_dir: Path):
    helper = helper_dir.resolve() / "audit_panel_alignment.py"
    if not helper.exists():
        raise FileNotFoundError(f"Missing alignment helper: {helper}")
    sys.path.insert(0, str(helper.parent))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    return require_matplotlib_panel_alignment


def style_axis(axis: plt.Axes) -> None:
    axis.spines["left"].set_color(COLORS["edge"])
    axis.spines["bottom"].set_color(COLORS["edge"])
    axis.tick_params(colors=COLORS["edge"])
    axis.xaxis.label.set_color(COLORS["edge"])
    axis.yaxis.label.set_color(COLORS["edge"])
    axis.title.set_color(COLORS["edge"])
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.85)
    axis.set_axisbelow(True)


def plot(summary: pd.DataFrame, runs: pd.DataFrame) -> plt.Figure:
    x = summary["lookback"].to_numpy(int)
    mean = summary["test_full_rmse_mean_k"].to_numpy(float)
    sd = summary["test_full_rmse_std_k"].to_numpy(float)

    figure, axis = plt.subplots(figsize=(7.20, 3.45))
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.28, top=0.84)

    axis.axvspan(
        PLATEAU_START - 2.5,
        max(EXPECTED_LOOKBACKS) + 2.5,
        color=COLORS["plateau"],
        linewidth=0,
        zorder=0,
    )

    for index, lookback in enumerate(x):
        values = runs.loc[
            runs["lookback"].eq(lookback), "test_full_rmse_k"
        ].to_numpy(float)
        axis.scatter(
            np.full(values.size, lookback, dtype=float),
            values,
            s=11,
            color=COLORS["raw"],
            alpha=0.55,
            edgecolors="none",
            zorder=2,
            label="Individual seeds" if index == 0 else None,
        )

    axis.errorbar(
        x,
        mean,
        yerr=sd,
        color=COLORS["mean"],
        linewidth=1.55,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.05,
        ecolor=COLORS["mean"],
        elinewidth=0.85,
        capsize=2.3,
        label="Mean ± SD",
        zorder=4,
    )

    minimum_row = summary.loc[summary["lookback"].eq(NUMERICAL_MINIMUM)].iloc[0]
    minimum_y = float(minimum_row["test_full_rmse_mean_k"])
    axis.scatter(
        [NUMERICAL_MINIMUM],
        [minimum_y],
        s=42,
        facecolor=COLORS["minimum"],
        edgecolor=COLORS["edge"],
        linewidth=0.6,
        zorder=6,
        label="Numerical minimum",
    )

    axis.annotate(
        f"Minimum: 60 samples, {minimum_y:.3f} K",
        xy=(NUMERICAL_MINIMUM, minimum_y),
        xytext=(47, 1.290),
        ha="right",
        va="center",
        fontsize=7.0,
        color=COLORS["edge"],
        arrowprops={"arrowstyle": "-", "color": COLORS["edge"], "lw": 0.7},
    )

    axis.set_xlim(11, 124)
    axis.set_ylim(1.085, 1.34)
    axis.set_xticks(x)
    axis.set_xlabel("Lookback")
    axis.set_ylabel("Complete-test RMSE (K)")
    axis.set_title("Lookback sensitivity on the complete 0715-BACK test sequence")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.30),
        ncols=3,
        fontsize=6.8,
        handlelength=1.8,
        columnspacing=1.5,
    )
    style_axis(axis)
    return figure


def save(
    figure: plt.Figure,
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
        json_out=qa_output_dir / f"{BASE_NAME}.alignment.json",
        overlay_svg=qa_output_dir / f"{BASE_NAME}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    vector = {"facecolor": "white", "transparent": False, "bbox_inches": "tight"}
    raster = {**vector, "dpi": DPI}
    figure.savefig(output_dir / f"{BASE_NAME}.svg", **vector)
    figure.savefig(output_dir / f"{BASE_NAME}.pdf", **vector)
    figure.savefig(output_dir / f"{BASE_NAME}.png", **raster)
    figure.savefig(output_dir / f"{BASE_NAME}.tiff", **raster)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    summary, runs = validate_data(
        pd.read_csv(args.summary_csv), pd.read_csv(args.all_runs_csv)
    )
    alignment_helper = load_alignment_helper(args.qa_helper_dir)
    save(plot(summary, runs), args.output_dir, args.qa_output_dir, alignment_helper)
    print(f"FIGURE_BASE_NAME={BASE_NAME}")
    print(f"FIGURE_OUTPUT_DIR={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
