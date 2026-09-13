#!/usr/bin/env python3
"""Create a publication-ready seed-by-architecture RMSE heatmap.

The plotting source is deliberately self-contained once the companion source-data
CSV has been generated.  Pass the full test seed table on the first run; later
runs can use the compact CSV saved beside this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "gru_baseline",
    "lstm_baseline",
    "parallel_gru_mlp_gnn",
    "parallel_gru_kan_gnn",
    "parallel_lstm_mlp_gnn",
    "parallel_lstm_kan_gnn",
    "serial_mlp_gnn_gru",
    "serial_kan_gnn_gru",
]

DISPLAY = {
    "gru_baseline": "GRU",
    "lstm_baseline": "LSTM",
    "parallel_gru_mlp_gnn": "P-GRU–MLP",
    "parallel_gru_kan_gnn": "P-GRU–KAN",
    "parallel_lstm_mlp_gnn": "P-LSTM–MLP",
    "parallel_lstm_kan_gnn": "P-LSTM–KAN",
    "serial_mlp_gnn_gru": "S-MLP–GRU",
    "serial_kan_gnn_gru": "S-KAN–GRU",
}

SEED_ORDER = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
REQUIRED = {"model", "seed", "test_full_rmse_k"}
OPTIONAL_PROTOCOL = ["lookback", "predict_steps", "test_windows"]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=here / "model_architecture_seed_rmse_source_data.csv",
        help="Full test seed table or the compact source-data CSV.",
    )
    parser.add_argument("--output-dir", type=Path, default=here)
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=here / "qa",
        help="Directory for alignment and data-integrity records.",
    )
    return parser.parse_args()


def load_and_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["test_full_rmse_k"] = pd.to_numeric(
        df["test_full_rmse_k"], errors="raise"
    )
    df = df[df["seed"].isin(SEED_ORDER)].copy()

    duplicates = df.duplicated(["model", "seed"], keep=False)
    if duplicates.any():
        bad = df.loc[duplicates, ["model", "seed"]].to_dict("records")
        raise ValueError(f"Duplicate model/seed rows: {bad}")
    if not np.isfinite(df["test_full_rmse_k"]).all():
        raise ValueError("RMSE contains a non-finite value")
    if (df["test_full_rmse_k"] <= 0).any():
        raise ValueError("RMSE must be strictly positive")

    present = set(df["model"])
    if present != set(MODEL_ORDER):
        raise ValueError(
            f"Model mismatch; missing={sorted(set(MODEL_ORDER) - present)}, "
            f"unexpected={sorted(present - set(MODEL_ORDER))}"
        )

    expected_counts = {model: 10 for model in MODEL_ORDER}
    counts = df.groupby("model")["seed"].size().to_dict()
    if counts != expected_counts:
        raise ValueError(f"Unexpected seed counts: {counts}")

    for col, expected in (("lookback", 60), ("predict_steps", 15), ("test_windows", 10659)):
        if col in df.columns:
            values = set(pd.to_numeric(df[col], errors="raise").astype(int))
            if values != {expected}:
                raise ValueError(f"Protocol mismatch in {col}: {sorted(values)}")

    keep = ["model", "seed", "test_full_rmse_k"]
    keep += [c for c in OPTIONAL_PROTOCOL if c in df.columns]
    return df[keep].sort_values(["model", "seed"]).reset_index(drop=True)


def text_color(cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize, value: float) -> str:
    r, g, b, _ = cmap(norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if luminance > 0.56 else "#FFFFFF"


def run_alignment_audit(fig, colorbar_axis, qa_dir: Path) -> None:
    scripts_dir = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        return
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(qa_dir / "model_architecture_seed_rmse_heatmap.alignment.json"),
        overlay_svg=str(qa_dir / "model_architecture_seed_rmse_heatmap.alignment.svg"),
        exclude_axes=[colorbar_axis],
        require_panel_labels=False,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.qa_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_validate(args.source)
    source_out = args.output_dir / "model_architecture_seed_rmse_source_data.csv"
    df.to_csv(source_out, index=False, encoding="utf-8-sig")

    matrix = (
        df.pivot(index="seed", columns="model", values="test_full_rmse_k")
        .reindex(index=SEED_ORDER, columns=MODEL_ORDER)
        .to_numpy(dtype=float)
    )
    ns = np.sum(np.isfinite(matrix), axis=0)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(7.35, 4.65), constrained_layout=False)
    fig.subplots_adjust(left=0.092, right=0.885, bottom=0.145, top=0.865)

    base_cmap = mpl.colormaps["magma_r"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "muted_magma_r", base_cmap(np.linspace(0.08, 0.72, 256))
    )
    cmap.set_bad("#ECECEC")
    finite = matrix[np.isfinite(matrix)]
    norm = mpl.colors.Normalize(vmin=float(finite.min()), vmax=float(finite.max()))
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    # Thin white cell boundaries maintain readability without dominating the data.
    ax.set_xticks(np.arange(len(MODEL_ORDER) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(SEED_ORDER) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.72)
    ax.tick_params(which="minor", bottom=False, left=False)

    labels = [DISPLAY[m] for m in MODEL_ORDER]
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels(
        labels, rotation=0, ha="center", rotation_mode="anchor", fontsize=6.25
    )
    ax.set_yticks(np.arange(len(SEED_ORDER)))
    ax.set_yticklabels([str(seed) for seed in SEED_ORDER], fontsize=7)
    ax.tick_params(axis="both", length=0, pad=3)
    ax.set_xlabel("Model architecture", fontsize=7.5, labelpad=7)
    ax.set_ylabel("Random seed", fontsize=7.5, labelpad=6)

    # Annotate every observed RMSE; missing TCN seeds remain explicitly visible.
    for i in range(matrix.shape[0]):
        row = matrix[i]
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#8A8A8A")
                continue
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=6.25,
                color=text_color(cmap, norm, value),
                fontweight="normal",
            )

    # Quiet separators preserve the requested baseline-parallel-serial order.
    for boundary in (1.5, 5.5):
        ax.axvline(boundary, color="#505050", lw=1.0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.028, aspect=28)
    cbar.set_label("Test RMSE (K)", fontsize=7, labelpad=7)
    cbar.ax.tick_params(labelsize=6.5, length=2.5, width=0.6)
    cbar.outline.set_linewidth(0.55)

    fig.text(
        0.092,
        0.965,
        "Seed-wise RMSE across model architectures",
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#1A1A1A",
    )
    fig.canvas.draw()
    run_alignment_audit(fig, cbar.ax, args.qa_dir)

    prefix = args.output_dir / "model_architecture_seed_rmse_heatmap"
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        prefix.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )

    integrity = {
        "figure": "model_architecture_seed_rmse_heatmap",
        "metric": "test_full_rmse_k",
        "rows": int(len(df)),
        "models": MODEL_ORDER,
        "seed_order": SEED_ORDER,
        "model_seed_counts": {MODEL_ORDER[j]: int(ns[j]) for j in range(len(MODEL_ORDER))},
        "lookback": 60,
        "predict_steps": 15,
        "test_windows": 10659,
        "missing_cells": int(np.isnan(matrix).sum()),
        "missing_rule": "No missing cells; every retained architecture has ten seeds.",
        "color_norm": {
            "type": "Normalize",
            "vmin": float(finite.min()),
            "vmax": float(finite.max()),
            "cmap": "magma_r truncated to [0.08, 0.72] to avoid near-black saturation",
        },
    }
    (args.qa_dir / "data_integrity.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
