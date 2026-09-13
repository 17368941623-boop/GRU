#!/usr/bin/env python3
"""Create publication figures for the Delta-Thv lookback study.

Required inputs are the aggregate summary, per-seed run table, and the
lookback-60 validation prediction folders. Outputs are SVG, PDF, 600-dpi PNG,
and 600-dpi TIFF. Bars show seed means, error bars show one sample SD, and
individual seed values are overlaid.
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
from matplotlib.transforms import ScaledTranslation


# Editable vector text and publication-safe fonts.  Keep these settings in one
# literal update block so the static publication preflight can audit them.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
plt.rcParams["font.size"] = 7.0
plt.rcParams["axes.labelsize"] = 7.0
plt.rcParams["axes.titlesize"] = 7.5
plt.rcParams["xtick.labelsize"] = 6.5
plt.rcParams["ytick.labelsize"] = 6.5
plt.rcParams["axes.linewidth"] = 0.75
plt.rcParams["xtick.major.width"] = 0.65
plt.rcParams["ytick.major.width"] = 0.65
plt.rcParams["xtick.major.size"] = 3.0
plt.rcParams["ytick.major.size"] = 3.0
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


DPI = 600
FORMATS = ("svg", "pdf", "png", "tiff")
EXPECTED_LOOKBACKS = (60, 75, 90, 105, 120)
EXPECTED_SEEDS = (42, 62, 82)
HERO = "#D55E00"
COMPARISON = "#8CB3D9"
COMPARISON_DARK = "#3E78A8"
BASELINE = "#9A9A9A"
EDGE = "#3F3F3F"
POINT = "#202020"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--all-runs-csv", type=Path, required=True)
    parser.add_argument(
        "--prediction-root",
        type=Path,
        required=True,
        help="Directory containing seed_42, seed_62 and seed_82 for lookback 60.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--qa-helper-dir",
        type=Path,
        required=True,
        help="Directory containing audit_panel_alignment.py.",
    )
    parser.add_argument("--qa-output-dir", type=Path, required=True)
    return parser.parse_args()


def load_alignment_helper(helper_dir: Path):
    helper_dir = helper_dir.resolve()
    if not (helper_dir / "audit_panel_alignment.py").exists():
        raise FileNotFoundError(f"Missing alignment helper in {helper_dir}")
    sys.path.insert(0, str(helper_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    return require_matplotlib_panel_alignment


def validate_inputs(
    summary: pd.DataFrame,
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = summary.sort_values("lookback").reset_index(drop=True)
    runs = runs.sort_values(["lookback", "seed"]).reset_index(drop=True)
    if tuple(summary["lookback"].astype(int)) != EXPECTED_LOOKBACKS:
        raise ValueError("Summary does not contain the expected 60--120 scan")
    if set(runs["lookback"].astype(int)) != set(EXPECTED_LOOKBACKS):
        raise ValueError("Per-seed table has unexpected lookback values")
    for lookback, part in runs.groupby("lookback"):
        if tuple(sorted(part["seed"].astype(int))) != EXPECTED_SEEDS:
            raise ValueError(f"Incomplete seeds for lookback {lookback}")
        if part["predict_steps"].astype(int).nunique() != 1 or int(
            part["predict_steps"].iloc[0]
        ) != 15:
            raise ValueError("Forecast horizon is not consistently 15 steps")
    if summary["seed_count"].astype(int).ne(3).any():
        raise ValueError("Every lookback must contain three seeds")
    if summary["full_validation_windows_min"].nunique() != 1 or summary[
        "full_validation_windows_max"
    ].nunique() != 1:
        raise ValueError("Full validation window counts are not identical")
    if summary["rapid_validation_windows_min"].nunique() != 1 or summary[
        "rapid_validation_windows_max"
    ].nunique() != 1:
        raise ValueError("Rapid validation window counts are not identical")
    if runs["normalization_report_sha256"].nunique() != 1:
        raise ValueError("Normalization reports differ across runs")
    return summary, runs


def audit_current_value_shortcut(prediction_root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    usecols = [
        "actual_delta_Thv_15step_k",
        "predicted_delta_Thv_15step_k",
        "rapid_cooling_mask",
    ]
    for seed in EXPECTED_SEEDS:
        path = prediction_root / f"seed_{seed}" / "validation_predictions.csv"
        frame = pd.read_csv(path, usecols=usecols)
        actual = frame["actual_delta_Thv_15step_k"].to_numpy(dtype=float)
        predicted = frame["predicted_delta_Thv_15step_k"].to_numpy(dtype=float)
        rapid = frame["rapid_cooling_mask"].to_numpy(dtype=int) == 1
        residual = predicted - actual
        model_rmse = float(np.sqrt(np.mean(np.square(residual))))
        persistence_rmse = float(np.sqrt(np.mean(np.square(actual))))
        rapid_model_rmse = float(np.sqrt(np.mean(np.square(residual[rapid]))))
        rapid_persistence_rmse = float(
            np.sqrt(np.mean(np.square(actual[rapid])))
        )
        rows.append(
            {
                "seed": seed,
                "delta_correlation": float(np.corrcoef(actual, predicted)[0, 1]),
                "actual_delta_sd_k": float(np.std(actual, ddof=0)),
                "predicted_delta_sd_k": float(np.std(predicted, ddof=0)),
                "predicted_delta_near_zero_pct": float(
                    100.0 * np.mean(np.abs(predicted) < 0.01)
                ),
                "model_rmse_k": model_rmse,
                "persistence_rmse_k": persistence_rmse,
                "rmse_reduction_pct": 100.0 * (1.0 - model_rmse / persistence_rmse),
                "rapid_model_rmse_k": rapid_model_rmse,
                "rapid_persistence_rmse_k": rapid_persistence_rmse,
                "rapid_rmse_reduction_pct": 100.0
                * (1.0 - rapid_model_rmse / rapid_persistence_rmse),
            }
        )
    audit = pd.DataFrame(rows)
    copied_current = bool(
        (audit["rmse_reduction_pct"] < 10.0).any()
        or (audit["predicted_delta_sd_k"] < 0.10 * audit["actual_delta_sd_k"]).any()
        or (audit["predicted_delta_near_zero_pct"] > 90.0).any()
    )
    print(audit.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"CURRENT_VALUE_COPY_SUSPECTED={str(copied_current).lower()}")
    return audit


def add_panel_label(ax: plt.Axes, label: str) -> None:
    offset = ScaledTranslation(-6 / 72, 4 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0,
        1,
        label,
        transform=ax.transAxes + offset,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)
    ax.tick_params(colors=EDGE)
    ax.yaxis.label.set_color(EDGE)
    ax.xaxis.label.set_color(EDGE)
    ax.title.set_color(EDGE)
    ax.set_axisbelow(True)


def lookback_tick_labels(summary: pd.DataFrame) -> list[str]:
    return [
        f"{int(row.lookback)}\n({float(row.history_minutes):g})"
        for row in summary.itertuples(index=False)
    ]


def add_mean_sd_bars(
    ax: plt.Axes,
    categories: list[str],
    means: np.ndarray,
    standard_deviations: np.ndarray,
    seed_values: list[np.ndarray],
    colors: list[str],
    *,
    width: float = 0.66,
) -> None:
    x = np.arange(len(categories), dtype=float)
    ax.bar(
        x,
        means,
        width=width,
        color=colors,
        edgecolor=EDGE,
        linewidth=0.65,
        zorder=2,
    )
    ax.errorbar(
        x,
        means,
        yerr=standard_deviations,
        fmt="none",
        ecolor=EDGE,
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        zorder=4,
    )
    for index, values in enumerate(seed_values):
        values = np.asarray(values, dtype=float)
        offsets = np.linspace(-0.11, 0.11, len(values))
        ax.scatter(
            index + offsets,
            values,
            s=10,
            c=POINT,
            marker="o",
            linewidths=0,
            zorder=5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    style_axis(ax)


def save_figure(
    fig: plt.Figure,
    axes: list[plt.Axes],
    panel_ids: list[str],
    base_name: str,
    output_dir: Path,
    qa_output_dir: Path,
    require_matplotlib_panel_alignment,
) -> None:
    qa_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment_options: dict[str, object] = {
        "axes": axes,
        "panel_ids": panel_ids,
    }
    if len(axes) > 1:
        alignment_options["row_groups"] = [panel_ids]
    require_matplotlib_panel_alignment(
        fig,
        json_out=qa_output_dir / f"{base_name}.alignment.json",
        overlay_svg=qa_output_dir / f"{base_name}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=len(axes) > 1,
        strict=True,
        **alignment_options,
    )
    vector_kwargs: dict[str, object] = {
        "facecolor": "white",
        "transparent": False,
    }
    raster_kwargs = {**vector_kwargs, "dpi": DPI}
    fig.savefig(output_dir / f"{base_name}.svg", **vector_kwargs)
    fig.savefig(output_dir / f"{base_name}.pdf", **vector_kwargs)
    fig.savefig(output_dir / f"{base_name}.png", **raster_kwargs)
    fig.savefig(output_dir / f"{base_name}.tiff", **raster_kwargs)
    plt.close(fig)


def create_primary_rapid_figure(
    summary: pd.DataFrame,
    runs: pd.DataFrame,
    output_dir: Path,
    qa_output_dir: Path,
    require_alignment,
) -> None:
    fig, ax = plt.subplots(figsize=(3.50, 2.65))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.25, top=0.91)
    means = summary["rapid_validation_rmse_mean_k"].to_numpy(float)
    spread = summary["rapid_validation_rmse_std_k"].to_numpy(float)
    seed_values = [
        runs.loc[runs["lookback"].eq(lb), "best_rapid_validation_rmse_k"].to_numpy(float)
        for lb in EXPECTED_LOOKBACKS
    ]
    colors = [HERO if lb == 60 else COMPARISON for lb in EXPECTED_LOOKBACKS]
    add_mean_sd_bars(
        ax,
        lookback_tick_labels(summary),
        means,
        spread,
        seed_values,
        colors,
    )
    ax.set_ylim(0, 1.15 * float(np.max(means + spread)))
    ax.set_ylabel("Rapid-cooling RMSE (K)")
    ax.set_xlabel("Lookback (samples; minutes in parentheses)")
    ax.set_title("Original 260501 validation; forecast horizon = 15 steps", pad=5)
    ax.text(
        0.98,
        0.96,
        "n = 3 seeds; mean ± SD",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=EDGE,
    )
    save_figure(
        fig,
        [ax],
        ["a"],
        "lookback_primary_rapid_rmse",
        output_dir,
        qa_output_dir,
        require_alignment,
    )


def create_evidence_figure(
    summary: pd.DataFrame,
    runs: pd.DataFrame,
    output_dir: Path,
    qa_output_dir: Path,
    require_alignment,
) -> None:
    fig, axes_array = plt.subplots(1, 3, figsize=(7.20, 2.60))
    axes = list(axes_array)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.86, wspace=0.36)
    labels = lookback_tick_labels(summary)
    colors = [HERO if lb == 60 else COMPARISON for lb in EXPECTED_LOOKBACKS]

    rapid_means = summary["rapid_validation_rmse_mean_k"].to_numpy(float)
    rapid_sd = summary["rapid_validation_rmse_std_k"].to_numpy(float)
    rapid_seed_values = [
        runs.loc[runs["lookback"].eq(lb), "best_rapid_validation_rmse_k"].to_numpy(float)
        for lb in EXPECTED_LOOKBACKS
    ]
    add_mean_sd_bars(
        axes[0], labels, rapid_means, rapid_sd, rapid_seed_values, colors, width=0.64
    )
    axes[0].set_ylim(0, 1.17 * float(np.max(rapid_means + rapid_sd)))
    axes[0].set_ylabel("RMSE (K)")
    axes[0].set_xlabel("Lookback\n(samples; min in parentheses)")
    axes[0].set_title("Primary: rapid-cooling subset", pad=5)

    full_means = summary["validation_rmse_mean_k"].to_numpy(float)
    full_sd = summary["validation_rmse_std_k"].to_numpy(float)
    full_seed_values = [
        runs.loc[runs["lookback"].eq(lb), "best_validation_rmse_k"].to_numpy(float)
        for lb in EXPECTED_LOOKBACKS
    ]
    add_mean_sd_bars(
        axes[1], labels, full_means, full_sd, full_seed_values, colors, width=0.64
    )
    axes[1].set_ylim(0, 1.17 * float(np.max(full_means + full_sd)))
    axes[1].set_ylabel("RMSE (K)")
    axes[1].set_xlabel("Lookback\n(samples; min in parentheses)")
    axes[1].set_title("Guardrail: full validation", pad=5)

    pivot = runs.pivot(
        index="seed", columns="lookback", values="best_rapid_validation_rmse_k"
    ).sort_index()
    compared = np.asarray(EXPECTED_LOOKBACKS[1:], dtype=int)
    differences = [
        (pivot[lookback] - pivot[60]).to_numpy(float) for lookback in compared
    ]
    difference_means = np.asarray([values.mean() for values in differences])
    difference_sd = np.asarray([values.std(ddof=1) for values in differences])
    add_mean_sd_bars(
        axes[2],
        [str(value) for value in compared],
        difference_means,
        difference_sd,
        differences,
        [COMPARISON_DARK] * len(compared),
        width=0.62,
    )
    axes[2].axhline(0, color=HERO, linewidth=0.9, zorder=1)
    lower = min(-0.012, float(np.min([values.min() for values in differences])) - 0.008)
    upper = max(
        0.055,
        float(np.max(difference_means + difference_sd)) + 0.010,
    )
    axes[2].set_ylim(lower, upper)
    axes[2].set_ylabel("Paired ΔRMSE vs lookback 60 (K)")
    axes[2].set_xlabel("Compared lookback (samples)")
    axes[2].set_title("Robustness: matched seeds", pad=5)

    for axis, label in zip(axes, ("a", "b", "c")):
        add_panel_label(axis, label)
    save_figure(
        fig,
        axes,
        ["a", "b", "c"],
        "lookback_rmse_evidence",
        output_dir,
        qa_output_dir,
        require_alignment,
    )


def create_persistence_audit_figure(
    summary: pd.DataFrame,
    runs: pd.DataFrame,
    output_dir: Path,
    qa_output_dir: Path,
    require_alignment,
) -> None:
    selected = runs.loc[runs["lookback"].eq(60)].sort_values("seed")
    fig, axes_array = plt.subplots(1, 2, figsize=(7.20, 2.70))
    axes = list(axes_array)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.82, wspace=0.30)

    panel_specs = (
        (
            "Full validation",
            selected["best_validation_rmse_k"].to_numpy(float),
            selected["persistence_validation_rmse_k"].to_numpy(float),
            float(summary.loc[summary["lookback"].eq(60), "validation_rmse_reduction_vs_persistence_pct"].iloc[0]),
        ),
        (
            "Rapid-cooling subset",
            selected["best_rapid_validation_rmse_k"].to_numpy(float),
            selected["rapid_persistence_validation_rmse_k"].to_numpy(float),
            float(summary.loc[summary["lookback"].eq(60), "rapid_rmse_reduction_vs_persistence_pct"].iloc[0]),
        ),
    )
    for axis, (title, model_values, persistence_values, reduction) in zip(
        axes, panel_specs
    ):
        values = [model_values, persistence_values]
        means = np.asarray([array.mean() for array in values])
        spread = np.asarray([array.std(ddof=1) for array in values])
        add_mean_sd_bars(
            axis,
            ["GRU\nlookback 60", "Persistence\ncurrent Thv"],
            means,
            spread,
            values,
            [HERO, BASELINE],
            width=0.58,
        )
        axis.set_ylim(0, 1.14 * float(np.max(means + spread)))
        axis.set_ylabel("RMSE (K)")
        axis.set_title(f"{title}\nRMSE reduction = {reduction:.1f}%", pad=5)
    for axis, label in zip(axes, ("a", "b")):
        add_panel_label(axis, label)
    save_figure(
        fig,
        axes,
        ["a", "b"],
        "lookback_persistence_audit",
        output_dir,
        qa_output_dir,
        require_alignment,
    )


def main() -> None:
    args = parse_args()
    require_alignment = load_alignment_helper(args.qa_helper_dir)
    summary, runs = validate_inputs(
        pd.read_csv(args.summary_csv),
        pd.read_csv(args.all_runs_csv),
    )
    audit_current_value_shortcut(args.prediction_root)
    create_primary_rapid_figure(
        summary, runs, args.output_dir, args.qa_output_dir, require_alignment
    )
    create_evidence_figure(
        summary, runs, args.output_dir, args.qa_output_dir, require_alignment
    )
    create_persistence_audit_figure(
        summary, runs, args.output_dir, args.qa_output_dir, require_alignment
    )
    print(f"FIGURE_OUTPUT_DIR={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
