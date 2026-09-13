#!/usr/bin/env python3
"""Publication candidates for the model-by-cumulative-feature validation study."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_CODE_DIR = PROJECT_DIR / "model_code"
if str(MODEL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_CODE_DIR))

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
from heatmap_protocol import FEATURE_SPECS, MODEL_SPECS, SEEDS  # noqa: E402


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RESULTS_DIR = PROJECT_DIR / "outputs_0910"
SUMMARY_DIR = RESULTS_DIR / "validation_summary"
OUTPUT_DIR = SCRIPT_DIR / "generated"
QA_DIR = SCRIPT_DIR / "qa"


def truncated_cmap(name: str, lower: float, upper: float) -> mpl.colors.Colormap:
    base = mpl.colormaps[name]
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"{name}_{lower:.2f}_{upper:.2f}",
        base(np.linspace(lower, upper, 256)),
    )


def contrasting_text_color(
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    value: float,
) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.53 else "#202020"


def export_figure(fig: plt.Figure, stem: str, main_ax: plt.Axes, colorbar_ax: plt.Axes) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        axes=[main_ax],
        panel_ids=["a"],
        exclude_axes=[colorbar_ax],
        require_panel_labels=False,
        json_out=QA_DIR / f"{stem}.alignment.json",
        overlay_svg=QA_DIR / f"{stem}.alignment.svg",
        strict=True,
    )
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(
        OUTPUT_DIR / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def configure_axes(ax: plt.Axes, x_labels: list[str], y_labels: list[str], title: str) -> None:
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    ax.set_xlabel("Model architecture", labelpad=6)
    ax.set_ylabel("Cumulative feature configuration", labelpad=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    ax.tick_params(axis="both", which="major", length=0)
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_absolute(
    mean: np.ndarray,
    sd: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
) -> None:
    cmap = truncated_cmap("YlGnBu", 0.07, 0.78)
    spread = max(float(np.ptp(mean)), 1e-9)
    norm = mpl.colors.Normalize(
        vmin=float(mean.min() - 0.035 * spread),
        vmax=float(mean.max() + 0.035 * spread),
    )
    fig, ax = plt.subplots(figsize=(7.1, 5.15), layout="constrained")
    image = ax.imshow(mean, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.025)
    colorbar.set_label("Mean RMSE (K)")
    colorbar.ax.tick_params(length=2.5, width=0.7)
    configure_axes(
        ax,
        x_labels,
        y_labels,
        "Validation RMSE across architectures and cumulative features",
    )
    best = np.unravel_index(np.argmin(mean), mean.shape)
    for row, column in np.ndindex(mean.shape):
        value = float(mean[row, column])
        ax.text(
            column,
            row,
            f"{value:.3f} ± {sd[row, column]:.3f}",
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold" if (row, column) == best else "normal",
            color=contrasting_text_color(cmap, norm, value),
        )
    export_figure(fig, "model_feature_validation_rmse_heatmap", ax, colorbar.ax)


def plot_relative(
    relative: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
) -> None:
    cmap = truncated_cmap("RdBu_r", 0.12, 0.88)
    limit = max(float(np.abs(relative).max()), 1.0)
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, ax = plt.subplots(figsize=(7.1, 5.15), layout="constrained")
    image = ax.imshow(relative, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.025)
    colorbar.set_label("RMSE change (%)")
    colorbar.ax.tick_params(length=2.5, width=0.7)
    configure_axes(
        ax,
        x_labels,
        y_labels,
        "RMSE change relative to raw measurements",
    )
    for row, column in np.ndindex(relative.shape):
        value = float(relative[row, column])
        ax.text(
            column,
            row,
            f"{value:+.1f}%",
            ha="center",
            va="center",
            fontsize=7.2,
            color=contrasting_text_color(cmap, norm, value),
        )
    export_figure(fig, "model_feature_relative_rmse_heatmap", ax, colorbar.ax)


def main() -> None:
    summary_path = SUMMARY_DIR / "validation_cell_summary.csv"
    runs_path = SUMMARY_DIR / "validation_seed_runs.csv"
    if not summary_path.is_file() or not runs_path.is_file():
        raise FileNotFoundError("The complete validation summary is required before plotting")
    summary = pd.read_csv(summary_path)
    runs = pd.read_csv(runs_path)

    row_ids = [str(spec["feature_set"]) for spec in FEATURE_SPECS]
    model_ids = [str(spec["model"]) for spec in MODEL_SPECS]
    y_labels = [str(spec["label"]) for spec in FEATURE_SPECS]
    x_labels = [str(spec["short_label"]) for spec in MODEL_SPECS]

    cell_sizes = runs.groupby(["feature_set", "model"]).seed.nunique()
    if len(runs) != 120 or not (cell_sizes == len(SEEDS)).all():
        raise ValueError("Expected 24 complete cells with five paired seeds each")
    if runs.test_data_loaded.astype(bool).any():
        raise ValueError("Training metadata reports test access")

    mean_table = summary.pivot(
        index="feature_set", columns="model", values="validation_rmse_k_mean"
    ).loc[row_ids, model_ids]
    sd_table = summary.pivot(
        index="feature_set", columns="model", values="validation_rmse_k_sd"
    ).loc[row_ids, model_ids]
    relative_table = 100.0 * mean_table.divide(mean_table.loc["raw20"], axis="columns") - 100.0
    mean = mean_table.to_numpy(float)
    sd = sd_table.to_numpy(float)
    relative = relative_table.to_numpy(float)
    if not np.isfinite(mean).all() or not np.isfinite(sd).all() or not np.isfinite(relative).all():
        raise ValueError("A heatmap matrix contains a missing or non-finite value")

    source = summary[[
        "feature_stage_index",
        "feature_set",
        "feature_set_label",
        "model",
        "model_short_label",
        "n_seeds",
        "seed_set",
        "validation_rmse_k_mean",
        "validation_rmse_k_sd",
        "validation_rmse_k_ci95_low",
        "validation_rmse_k_ci95_high",
    ]].copy()
    relative_long = relative_table.stack().rename("rmse_change_vs_raw_pct").reset_index()
    source = source.merge(relative_long, on=["feature_set", "model"], validate="one_to_one")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source.sort_values(["feature_stage_index", "model"]).to_csv(
        OUTPUT_DIR / "model_feature_heatmap_source_data.csv", index=False
    )
    runs.sort_values(["feature_stage_index", "model", "seed"]).to_csv(
        OUTPUT_DIR / "model_feature_heatmap_seed_data.csv", index=False
    )

    plot_absolute(mean, sd, x_labels, y_labels)
    plot_relative(relative, x_labels, y_labels)

    notes = {
        "core_conclusion": (
            "Target-history dynamics provide the clearest cross-architecture gain; "
            "additional cumulative features do not yield monotonic improvement."
        ),
        "evidence_role": "architecture comparison and cumulative feature-integration robustness",
        "archetype": "single-panel quantitative grid",
        "split": "complete Original 260501 validation sequence",
        "metric": "absolute Thv(t+15) RMSE after restoring the predicted temperature increment",
        "independent_repeat_unit": "random initialization seed",
        "n_seeds_per_cell": len(SEEDS),
        "seed_set": list(SEEDS),
        "cell_annotation_absolute": "mean ± sample SD across paired seeds",
        "relative_reference": "the raw-measurement configuration within the same architecture",
        "missing_or_excluded_cells": 0,
        "interpretation_limit": (
            "The cumulative sequence was fixed before analysis. Later rows retain all earlier "
            "groups, so this figure does not identify order-independent effects of individual groups."
        ),
        "test_set_used": False,
    }
    (OUTPUT_DIR / "model_feature_heatmap_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"OUTPUT_DIR={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
