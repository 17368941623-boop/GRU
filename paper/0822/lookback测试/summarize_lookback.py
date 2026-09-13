#!/usr/bin/env python3
"""Aggregate completed Thv LSTM lookback runs without reading model datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
EXPECTED_LOOKBACKS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Thv lookback RMSE runs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_runs(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.glob("lookback_*/seed_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload["validation_metrics"]
        persistence = payload["persistence_validation_metrics"]
        dynamic = payload.get("dynamic_validation_metrics") or {}
        rows.append(
            {
                "lookback": int(payload["lookback"]),
                "seed": int(payload["seed"]),
                "best_epoch": int(payload["best_epoch"]),
                "validation_rmse_k": float(validation["rmse_k"]),
                "validation_mae_k": float(validation["mae_k"]),
                "dynamic_validation_rmse_k": float(dynamic.get("rmse_k", np.nan)),
                "dynamic_validation_mae_k": float(dynamic.get("mae_k", np.nan)),
                "persistence_validation_rmse_k": float(persistence["rmse_k"]),
                "run_dir": str(path.parent.resolve()),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No completed metrics.json files found under {output_dir}")
    frame = pd.DataFrame(rows).sort_values(["lookback", "seed"]).reset_index(drop=True)
    invalid = sorted(set(frame["lookback"]) - set(EXPECTED_LOOKBACKS))
    if invalid:
        raise ValueError(f"Unexpected lookback values in outputs: {invalid}")
    return frame


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lookback, part in runs.groupby("lookback", sort=True):
        def sample_std(column: str) -> float:
            values = part[column].dropna().to_numpy(dtype=float)
            return float(values.std(ddof=1)) if len(values) > 1 else 0.0

        rows.append(
            {
                "lookback": int(lookback),
                "history_seconds": int(lookback) * 10,
                "seed_count": int(part["seed"].nunique()),
                "seeds": ";".join(map(str, sorted(part["seed"].unique()))),
                "validation_rmse_mean_k": float(part["validation_rmse_k"].mean()),
                "validation_rmse_std_k": sample_std("validation_rmse_k"),
                "validation_mae_mean_k": float(part["validation_mae_k"].mean()),
                "validation_mae_std_k": sample_std("validation_mae_k"),
                "dynamic_rmse_mean_k": float(part["dynamic_validation_rmse_k"].mean()),
                "dynamic_rmse_std_k": sample_std("dynamic_validation_rmse_k"),
                "persistence_rmse_mean_k": float(part["persistence_validation_rmse_k"].mean()),
                "best_epoch_mean": float(part["best_epoch"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("lookback").reset_index(drop=True)


def save_plot(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(
        summary["lookback"],
        summary["validation_rmse_mean_k"],
        yerr=summary["validation_rmse_std_k"],
        marker="o",
        capsize=4,
        linewidth=1.5,
        color="#0072B2",
        label="LSTM validation RMSE (mean ± seed std)",
    )
    ax.plot(
        summary["lookback"],
        summary["persistence_rmse_mean_k"],
        linestyle="--",
        color="#D55E00",
        label="Persistence RMSE",
    )
    ax.set_xticks(EXPECTED_LOOKBACKS)
    ax.set_xlabel("Lookback (samples, 10 s/sample)")
    ax.set_ylabel("Validation RMSE (K)")
    ax.set_title("Thv(t+5) raw-input LSTM lookback comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = load_runs(args.output_dir)
    summary = aggregate(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "lookback_all_runs.csv", index=False)
    summary.to_csv(args.output_dir / "lookback_summary.csv", index=False)
    save_plot(summary, args.output_dir / "lookback_rmse_comparison.png")
    best = summary.loc[summary["validation_rmse_mean_k"].idxmin()]
    print(summary.to_string(index=False))
    print(f"BEST_LOOKBACK_BY_MEAN_VALIDATION_RMSE={int(best['lookback'])}")
    print(f"BEST_MEAN_VALIDATION_RMSE_K={best['validation_rmse_mean_k']:.10f}")
    print(f"SUMMARY_SAVED={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
