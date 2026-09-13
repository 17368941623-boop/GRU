#!/usr/bin/env python3
"""Create traceable publication candidates from the frozen validation summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heatmap_protocol import FEATURE_SPECS, MODEL_SPECS, SEEDS

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.8

PROJECT_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "outputs_0910")
    return parser.parse_args()


def text_color(cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize, value: float) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.52 else "#202020"


def export(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def heatmap(
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    colorbar_label: str,
    cmap_name: str,
    norm: mpl.colors.Normalize,
    annotation_format: str,
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.15), constrained_layout=True)
    cmap = mpl.colormaps[cmap_name]
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.025)
    colorbar.set_label(colorbar_label)
    colorbar.ax.tick_params(length=2.5, width=0.7)

    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    ax.set_xlabel("Model architecture")
    ax.set_ylabel("Cumulative feature configuration")
    ax.set_title(title, pad=8, fontweight="bold")
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            ax.text(
                column,
                row,
                annotation_format.format(value),
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color(cmap, norm, value),
            )
    fig.canvas.draw()
    export(fig, output_stem)


def main() -> None:
    args = parse_args()
    summary_dir = args.results_dir.resolve() / "validation_summary"
    mean_path = summary_dir / "validation_rmse_mean_matrix.csv"
    sd_path = summary_dir / "validation_rmse_sd_matrix.csv"
    relative_path = summary_dir / "validation_relative_delta_pct_matrix.csv"
    for path in (mean_path, sd_path, relative_path):
        if not path.is_file():
            raise FileNotFoundError(f"Run summarize_heatmap_validation.py first: {path}")

    row_ids = [str(spec["feature_set"]) for spec in FEATURE_SPECS]
    column_ids = [str(spec["model"]) for spec in MODEL_SPECS]
    row_labels = [str(spec["label"]) for spec in FEATURE_SPECS]
    column_labels = [str(spec["short_label"]) for spec in MODEL_SPECS]
    mean = pd.read_csv(mean_path, index_col=0).loc[row_ids, column_ids].to_numpy(float)
    sd = pd.read_csv(sd_path, index_col=0).loc[row_ids, column_ids].to_numpy(float)
    relative = pd.read_csv(relative_path, index_col=0).loc[row_ids, column_ids].to_numpy(float)
    if mean.shape != (len(FEATURE_SPECS), len(MODEL_SPECS)) or not np.isfinite(mean).all():
        raise ValueError("Mean RMSE matrix is incomplete or non-finite")
    if not np.isfinite(sd).all() or not np.isfinite(relative).all():
        raise ValueError("SD or relative matrix is incomplete or non-finite")

    spread = max(float(np.ptp(mean)), 1e-9)
    absolute_norm = mpl.colors.Normalize(
        vmin=float(mean.min() - 0.04 * spread),
        vmax=float(mean.max() + 0.04 * spread),
    )
    relative_limit = max(float(np.abs(relative).max()), 1.0)
    relative_norm = mpl.colors.TwoSlopeNorm(
        vmin=-relative_limit, vcenter=0.0, vmax=relative_limit
    )
    figure_dir = summary_dir / "figures"
    heatmap(
        mean,
        column_labels,
        row_labels,
        "Validation RMSE across models and cumulative features",
        "Mean RMSE (K)",
        "YlGnBu",
        absolute_norm,
        "{:.3f}",
        figure_dir / "validation_rmse_heatmap",
    )
    heatmap(
        relative,
        column_labels,
        row_labels,
        "RMSE change relative to raw measurements",
        "RMSE change (%)",
        "RdBu_r",
        relative_norm,
        "{:+.1f}%",
        figure_dir / "validation_relative_rmse_heatmap",
    )

    notes = {
        "core_conclusion": (
            "The figure tests whether progressively introduced physical information "
            "produces consistent validation changes across three model architectures."
        ),
        "evidence_role": "model-by-feature comparison and robustness across architectures",
        "archetype": "single-panel quantitative grid",
        "data_source": "validation_seed_runs.csv and derived mean/SD matrices",
        "split": "complete Original 260501 validation sequence",
        "metric": "absolute Thv(t+15) RMSE in kelvin after restoring the predicted delta",
        "center": "arithmetic mean across paired seeds",
        "spread": "sample SD is retained in validation_rmse_sd_matrix.csv",
        "n_seeds": len(SEEDS),
        "seed_set": list(SEEDS),
        "missing_or_excluded_cells": 0,
        "interpretation_warning": (
            "Rows are cumulative and order-dependent. The heatmap does not establish "
            "standalone causal importance for an individual feature group."
        ),
        "exports": ["SVG", "PDF", "600-dpi PNG", "600-dpi TIFF"],
    }
    (figure_dir / "figure_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"HEATMAPS_SAVED={figure_dir}")


if __name__ == "__main__":
    main()
