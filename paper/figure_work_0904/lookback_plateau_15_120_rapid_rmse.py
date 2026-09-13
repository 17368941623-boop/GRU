#!/usr/bin/env python3
"""Plot the validated Thv lookback scan and the plateau beyond 65 samples.

The figure uses every completed lookback and all ten random seeds. Points show
individual runs; the dark line and error bars show mean +/- one sample SD.
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
PLATEAU_START = 65
DPI = 600

DEFAULT_DATA_DIR = Path(
    r"C:\Users\Administrator\Desktop\论文20260708\0822\残差lookback测试"
    r"\gru_lookback_final_unified_val260501"
)
DEFAULT_QA_HELPER_DIR = Path(
    r"C:\Users\Administrator\.codex\skills\nature-figure\scripts"
)

COLORS = {
    "mean": "#245A82",
    "raw": "#7F9DB5",
    "plateau": "#EAF1F6",
    "onset": "#D55E00",
    "minimum": "#3B3B3B",
    "edge": "#3F3F3F",
    "grid": "#D9DEE3",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
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
        default=DEFAULT_DATA_DIR / "lookback_summary_validation_only.csv",
    )
    parser.add_argument(
        "--all-runs-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "lookback_all_runs_validation_only.csv",
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
        "rapid_rmse_mean_k",
        "rapid_rmse_std_k",
        "validation_windows",
        "rapid_validation_windows",
    }
    required_runs = {
        "lookback",
        "seed",
        "rapid_validation_rmse_k",
        "validation_windows",
        "rapid_validation_windows",
    }
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
    if runs["validation_windows"].nunique() != 1:
        raise ValueError("Full-validation window counts differ across runs")
    if runs["rapid_validation_windows"].nunique() != 1:
        raise ValueError("Rapid-validation window counts differ across runs")

    for lookback, group in runs.groupby("lookback", sort=True):
        seeds = tuple(group["seed"].sort_values())
        if seeds != EXPECTED_SEEDS:
            raise ValueError(f"Lookback {lookback} has seeds {seeds}")
        values = group["rapid_validation_rmse_k"].to_numpy(float)
        row = summary.loc[summary["lookback"].eq(lookback)].iloc[0]
        if not np.isclose(values.mean(), row["rapid_rmse_mean_k"], atol=1e-12):
            raise ValueError(f"Mean mismatch at lookback {lookback}")
        if not np.isclose(values.std(ddof=1), row["rapid_rmse_std_k"], atol=1e-12):
            raise ValueError(f"SD mismatch at lookback {lookback}")

    plateau = summary.loc[summary["lookback"].ge(PLATEAU_START)]
    plateau_range = float(
        plateau["rapid_rmse_mean_k"].max() - plateau["rapid_rmse_mean_k"].min()
    )
    numerical_minimum = int(
        summary.loc[summary["rapid_rmse_mean_k"].idxmin(), "lookback"]
    )
    print(f"LOOKBACKS={','.join(map(str, EXPECTED_LOOKBACKS))}")
    print(f"SEEDS={','.join(map(str, EXPECTED_SEEDS))}")
    print(f"NUMERICAL_MINIMUM_LOOKBACK={numerical_minimum}")
    print(f"PLATEAU_MEAN_RANGE_K={plateau_range:.6f}")
    return summary, runs


def load_alignment_helper(helper_dir: Path):
    helper_dir = helper_dir.resolve()
    helper = helper_dir / "audit_panel_alignment.py"
    if not helper.exists():
        raise FileNotFoundError(f"Missing alignment helper: {helper}")
    sys.path.insert(0, str(helper_dir))
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
    mean = summary["rapid_rmse_mean_k"].to_numpy(float)
    sd = summary["rapid_rmse_std_k"].to_numpy(float)

    figure, axis = plt.subplots(figsize=(7.20, 3.45))
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.28, top=0.84)

    axis.axvspan(
        PLATEAU_START,
        max(EXPECTED_LOOKBACKS) + 2.5,
        color=COLORS["plateau"],
        linewidth=0,
        zorder=0,
    )

    for index, lookback in enumerate(x):
        values = runs.loc[
            runs["lookback"].eq(lookback), "rapid_validation_rmse_k"
        ].to_numpy(float)
        axis.scatter(
            np.full(values.size, lookback, dtype=float),
            values,
            s=11,
            color=COLORS["raw"],
            alpha=0.50,
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

    onset_row = summary.loc[summary["lookback"].eq(PLATEAU_START)].iloc[0]
    minimum_row = summary.loc[summary["rapid_rmse_mean_k"].idxmin()]
    axis.scatter(
        [PLATEAU_START],
        [onset_row["rapid_rmse_mean_k"]],
        s=38,
        facecolor=COLORS["onset"],
        edgecolor=COLORS["edge"],
        linewidth=0.6,
        zorder=6,
        label="Practical plateau onset",
    )
    axis.scatter(
        [minimum_row["lookback"]],
        [minimum_row["rapid_rmse_mean_k"]],
        s=42,
        marker="D",
        facecolor="white",
        edgecolor=COLORS["minimum"],
        linewidth=1.0,
        zorder=6,
        label="Numerical minimum",
    )

    plateau = summary.loc[summary["lookback"].ge(PLATEAU_START)]
    plateau_range = float(
        plateau["rapid_rmse_mean_k"].max() - plateau["rapid_rmse_mean_k"].min()
    )
    axis.text(
        92.0,
        0.810,
        f"Plateau region: lookback ≥ 65; mean range = {plateau_range:.3f} K",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["edge"],
    )
    axis.annotate(
        f"65: {float(onset_row['rapid_rmse_mean_k']):.3f} K",
        xy=(PLATEAU_START, float(onset_row["rapid_rmse_mean_k"])),
        xytext=(60, 0.810),
        ha="right",
        va="center",
        fontsize=7.0,
        color=COLORS["edge"],
        arrowprops={"arrowstyle": "-", "color": COLORS["edge"], "lw": 0.7},
    )

    axis.set_xlim(11, 124)
    axis.set_ylim(0.665, 0.825)
    axis.set_xticks(x)
    axis.set_xlabel("Lookback (samples)")
    axis.set_ylabel("Rapid-cooling validation RMSE (K)")
    axis.set_title("Lookback sensitivity for the 15-step Thv forecast (n = 10 seeds)")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.30),
        ncols=4,
        fontsize=6.8,
        handlelength=1.8,
        columnspacing=1.2,
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

    vector = {"facecolor": "white", "transparent": False}
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
