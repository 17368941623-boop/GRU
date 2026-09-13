#!/usr/bin/env python3
"""Aggregate completed residual Thv lookback runs without reading datasets."""

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
EXPECTED_LOOKBACKS = (10, 15, 20, 25, 30, 35, 40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Delta-Thv known-control lookback runs."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_runs(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.glob("lookback_*/seed_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload["validation_metrics"]
        rapid = payload["rapid_validation_metrics"]
        persistence = payload["persistence_validation_metrics"]
        rapid_persistence = payload["rapid_persistence_validation_metrics"]
        best_history = payload["best_epoch_history"]
        rows.append(
            {
                "lookback": int(payload["lookback"]),
                "seed": int(payload["seed"]),
                "best_epoch": int(payload["best_epoch"]),
                "train_rmse_k": float(best_history["train_rmse_k"]),
                "train_rapid_rmse_k": float(best_history["train_rapid_rmse_k"]),
                "validation_rmse_k": float(validation["rmse_k"]),
                "validation_mae_k": float(validation["mae_k"]),
                "rapid_validation_rmse_k": float(rapid["rmse_k"]),
                "rapid_validation_mae_k": float(rapid["mae_k"]),
                "persistence_validation_rmse_k": float(persistence["rmse_k"]),
                "rapid_persistence_validation_rmse_k": float(
                    rapid_persistence["rmse_k"]
                ),
                "validation_direction_accuracy": float(
                    payload["validation_direction_accuracy"]
                ),
                "rapid_validation_direction_accuracy": float(
                    payload["rapid_validation_direction_accuracy"]
                ),
                "rapid_threshold_k": float(payload["rapid_threshold_k_train_only"]),
                "run_dir": str(path.parent.resolve()),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No completed metrics.json files under {output_dir}")
    frame = pd.DataFrame(rows).sort_values(["lookback", "seed"]).reset_index(drop=True)
    invalid = sorted(set(frame["lookback"]) - set(EXPECTED_LOOKBACKS))
    if invalid:
        raise ValueError(f"Unexpected lookback values: {invalid}")
    return frame


def sample_std(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=float)
    return float(array.std(ddof=1)) if len(array) > 1 else 0.0


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lookback, part in runs.groupby("lookback", sort=True):
        rows.append(
            {
                "lookback": int(lookback),
                "history_seconds": int(lookback) * 10,
                "seed_count": int(part["seed"].nunique()),
                "seeds": ";".join(map(str, sorted(part["seed"].unique()))),
                "train_rmse_mean_k": float(part["train_rmse_k"].mean()),
                "train_rmse_std_k": sample_std(part["train_rmse_k"]),
                "train_rapid_rmse_mean_k": float(part["train_rapid_rmse_k"].mean()),
                "train_rapid_rmse_std_k": sample_std(part["train_rapid_rmse_k"]),
                "validation_rmse_mean_k": float(part["validation_rmse_k"].mean()),
                "validation_rmse_std_k": sample_std(part["validation_rmse_k"]),
                "rapid_validation_rmse_mean_k": float(
                    part["rapid_validation_rmse_k"].mean()
                ),
                "rapid_validation_rmse_std_k": sample_std(
                    part["rapid_validation_rmse_k"]
                ),
                "validation_mae_mean_k": float(part["validation_mae_k"].mean()),
                "rapid_validation_mae_mean_k": float(
                    part["rapid_validation_mae_k"].mean()
                ),
                "validation_direction_accuracy_mean": float(
                    part["validation_direction_accuracy"].mean()
                ),
                "rapid_direction_accuracy_mean": float(
                    part["rapid_validation_direction_accuracy"].mean()
                ),
                "persistence_rmse_mean_k": float(
                    part["persistence_validation_rmse_k"].mean()
                ),
                "rapid_persistence_rmse_mean_k": float(
                    part["rapid_persistence_validation_rmse_k"].mean()
                ),
                "best_epoch_mean": float(part["best_epoch"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("lookback").reset_index(drop=True)


def save_plot(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    axes[0].errorbar(
        summary["lookback"],
        summary["rapid_validation_rmse_mean_k"],
        yerr=summary["rapid_validation_rmse_std_k"],
        marker="o",
        capsize=4,
        linewidth=1.5,
        color="#CC79A7",
        label="Rapid validation RMSE (mean +/- seed std)",
    )
    axes[0].plot(
        summary["lookback"],
        summary["rapid_persistence_rmse_mean_k"],
        linestyle="--",
        color="#555555",
        label="Rapid persistence RMSE",
    )
    axes[0].set_ylabel("Rapid-cooling RMSE (K)")
    axes[0].set_title("Primary lookback criterion")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].errorbar(
        summary["lookback"],
        summary["validation_rmse_mean_k"],
        yerr=summary["validation_rmse_std_k"],
        marker="o",
        capsize=4,
        linewidth=1.5,
        color="#0072B2",
        label="Full validation RMSE (mean +/- seed std)",
    )
    axes[1].plot(
        summary["lookback"],
        summary["persistence_rmse_mean_k"],
        linestyle="--",
        color="#D55E00",
        label="Full persistence RMSE",
    )
    axes[1].set_xticks(EXPECTED_LOOKBACKS)
    axes[1].set_xlabel("Lookback (samples, 10 s/sample)")
    axes[1].set_ylabel("Full validation RMSE (K)")
    axes[1].set_title("Overall-accuracy guardrail")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Delta Thv(t+5) LSTM with known u(t)..u(t+4)")
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

    best = summary.loc[summary["rapid_validation_rmse_mean_k"].idxmin()]
    print(summary.to_string(index=False))
    print(f"BEST_LOOKBACK_BY_RAPID_VALIDATION_RMSE={int(best['lookback'])}")
    print(
        f"BEST_MEAN_RAPID_VALIDATION_RMSE_K="
        f"{best['rapid_validation_rmse_mean_k']:.10f}"
    )
    print(
        f"BEST_LOOKBACK_FULL_VALIDATION_RMSE_K="
        f"{best['validation_rmse_mean_k']:.10f}"
    )
    print(f"SUMMARY_SAVED={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
