#!/usr/bin/env python3
"""Summarize paired LSTM-versus-GRU architecture runs."""

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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "rnn_compare_outputs"
EXPECTED_MODELS = ("lstm", "gru")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LSTM/GRU paired runs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sample_std(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=float)
    return float(array.std(ddof=1)) if len(array) > 1 else 0.0


def load_runs(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.glob("*/lookback_*/seed_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload["model_type"]).lower()
        if model not in EXPECTED_MODELS:
            raise ValueError(f"Unexpected model type {model} in {path}")
        validation = payload["validation_metrics"]
        rapid = payload["rapid_validation_metrics"]
        best = payload["best_epoch_history"]
        rows.append(
            {
                "model": model,
                "lookback": int(payload["lookback"]),
                "seed": int(payload["seed"]),
                "best_epoch": int(payload["best_epoch"]),
                "train_rmse_k": float(best["train_rmse_k"]),
                "train_rapid_rmse_k": float(best["train_rapid_rmse_k"]),
                "validation_rmse_k": float(validation["rmse_k"]),
                "rapid_validation_rmse_k": float(rapid["rmse_k"]),
                "validation_mae_k": float(validation["mae_k"]),
                "rapid_validation_mae_k": float(rapid["mae_k"]),
                "validation_direction_accuracy": float(
                    payload["validation_direction_accuracy"]
                ),
                "rapid_direction_accuracy": float(
                    payload["rapid_validation_direction_accuracy"]
                ),
                "trainable_parameters": int(payload["trainable_parameters"]),
                "training_seconds_total": float(payload["training_seconds_total"]),
                "seconds_per_epoch": float(payload["seconds_per_completed_epoch"]),
                "run_dir": str(path.parent.resolve()),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No architecture metrics found under {output_dir}")
    frame = pd.DataFrame(rows).sort_values(["lookback", "seed", "model"])
    duplicates = frame.duplicated(["model", "lookback", "seed"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate model/lookback/seed runs:\n"
            + frame.loc[duplicates, ["model", "lookback", "seed", "run_dir"]].to_string(
                index=False
            )
        )
    return frame.reset_index(drop=True)


def aggregate_by_lookback(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, lookback), part in runs.groupby(["model", "lookback"], sort=True):
        rows.append(
            {
                "model": model,
                "lookback": int(lookback),
                "seed_count": int(part["seed"].nunique()),
                "seeds": ";".join(map(str, sorted(part["seed"].unique()))),
                "rapid_validation_rmse_mean_k": float(
                    part["rapid_validation_rmse_k"].mean()
                ),
                "rapid_validation_rmse_std_k": sample_std(
                    part["rapid_validation_rmse_k"]
                ),
                "validation_rmse_mean_k": float(part["validation_rmse_k"].mean()),
                "validation_rmse_std_k": sample_std(part["validation_rmse_k"]),
                "rapid_direction_accuracy_mean": float(
                    part["rapid_direction_accuracy"].mean()
                ),
                "trainable_parameters": int(part["trainable_parameters"].iloc[0]),
                "seconds_per_epoch_mean": float(part["seconds_per_epoch"].mean()),
                "best_epoch_mean": float(part["best_epoch"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["lookback", "model"]).reset_index(drop=True)


def architecture_score(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, part in runs.groupby("model", sort=True):
        rows.append(
            {
                "model": model,
                "run_count": int(len(part)),
                "lookbacks": ";".join(map(str, sorted(part["lookback"].unique()))),
                "seeds": ";".join(map(str, sorted(part["seed"].unique()))),
                "rapid_validation_rmse_mean_k": float(
                    part["rapid_validation_rmse_k"].mean()
                ),
                "rapid_validation_rmse_std_k": sample_std(
                    part["rapid_validation_rmse_k"]
                ),
                "validation_rmse_mean_k": float(part["validation_rmse_k"].mean()),
                "validation_rmse_std_k": sample_std(part["validation_rmse_k"]),
                "rapid_direction_accuracy_mean": float(
                    part["rapid_direction_accuracy"].mean()
                ),
                "trainable_parameters": int(part["trainable_parameters"].iloc[0]),
                "seconds_per_epoch_mean": float(part["seconds_per_epoch"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def paired_differences(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rapid_validation_rmse_k",
        "validation_rmse_k",
        "rapid_direction_accuracy",
        "seconds_per_epoch",
    ]
    rows = []
    for (lookback, seed), part in runs.groupby(["lookback", "seed"], sort=True):
        if set(part["model"]) != set(EXPECTED_MODELS):
            continue
        by_model = part.set_index("model")
        row: dict[str, float | int] = {"lookback": int(lookback), "seed": int(seed)}
        for metric in metrics:
            lstm_value = float(by_model.loc["lstm", metric])
            gru_value = float(by_model.loc["gru", metric])
            row[f"lstm_{metric}"] = lstm_value
            row[f"gru_{metric}"] = gru_value
            row[f"lstm_minus_gru_{metric}"] = lstm_value - gru_value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["lookback", "seed"]).reset_index(drop=True)


def long_history_gain(
    runs: pd.DataFrame,
    short_lookback: int = 20,
    long_lookback: int = 40,
) -> pd.DataFrame:
    """Measure whether extending history from 20 to 40 helps each cell type."""
    rows = []
    for (model, seed), part in runs.groupby(["model", "seed"], sort=True):
        by_lookback = part.set_index("lookback")
        if short_lookback not in by_lookback.index or long_lookback not in by_lookback.index:
            continue
        short_rapid = float(
            by_lookback.loc[short_lookback, "rapid_validation_rmse_k"]
        )
        long_rapid = float(
            by_lookback.loc[long_lookback, "rapid_validation_rmse_k"]
        )
        short_full = float(by_lookback.loc[short_lookback, "validation_rmse_k"])
        long_full = float(by_lookback.loc[long_lookback, "validation_rmse_k"])
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "short_lookback": short_lookback,
                "long_lookback": long_lookback,
                "short_rapid_validation_rmse_k": short_rapid,
                "long_rapid_validation_rmse_k": long_rapid,
                "long_minus_short_rapid_rmse_k": long_rapid - short_rapid,
                "short_validation_rmse_k": short_full,
                "long_validation_rmse_k": long_full,
                "long_minus_short_full_rmse_k": long_full - short_full,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["model", "seed"]).reset_index(drop=True)


def save_plot(summary: pd.DataFrame, path: Path) -> None:
    colors = {"lstm": "#0072B2", "gru": "#D55E00"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for model in EXPECTED_MODELS:
        part = summary.loc[summary["model"] == model].sort_values("lookback")
        if part.empty:
            continue
        axes[0].errorbar(
            part["lookback"],
            part["rapid_validation_rmse_mean_k"],
            yerr=part["rapid_validation_rmse_std_k"],
            marker="o",
            capsize=4,
            color=colors[model],
            label=model.upper(),
        )
        axes[1].errorbar(
            part["lookback"],
            part["validation_rmse_mean_k"],
            yerr=part["validation_rmse_std_k"],
            marker="o",
            capsize=4,
            color=colors[model],
            label=model.upper(),
        )
    axes[0].set_ylabel("Rapid-cooling validation RMSE (K)")
    axes[0].set_title("Primary architecture criterion")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("Lookback (10 s/sample)")
    axes[1].set_ylabel("Full validation RMSE (K)")
    axes[1].set_title("Overall-accuracy guardrail")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    completed_lookbacks = sorted(summary["lookback"].unique())
    axes[1].set_xticks(completed_lookbacks)
    fig.suptitle("LSTM versus GRU: identical Delta target and control branch")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = load_runs(args.output_dir)
    summary = aggregate_by_lookback(runs)
    scores = architecture_score(runs)
    paired = paired_differences(runs)
    history_gain = long_history_gain(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "architecture_all_runs.csv", index=False)
    summary.to_csv(args.output_dir / "architecture_by_lookback.csv", index=False)
    scores.to_csv(args.output_dir / "architecture_score.csv", index=False)
    paired.to_csv(args.output_dir / "architecture_paired_differences.csv", index=False)
    history_gain.to_csv(args.output_dir / "long_history_gain.csv", index=False)
    save_plot(summary, args.output_dir / "lstm_gru_comparison.png")

    print("BY_LOOKBACK")
    print(summary.to_string(index=False))
    print("\nARCHITECTURE_SCORE")
    print(scores.to_string(index=False))
    if paired.empty:
        print("\nPAIRED_RUNS=0 (complete matching LSTM/GRU runs first)")
    else:
        rapid_difference = float(
            paired["lstm_minus_gru_rapid_validation_rmse_k"].mean()
        )
        full_difference = float(paired["lstm_minus_gru_validation_rmse_k"].mean())
        preferred = "GRU" if rapid_difference > 0 else "LSTM"
        print(f"\nPAIRED_RUNS={len(paired)}")
        print(
            "MEAN_LSTM_MINUS_GRU_RAPID_VALIDATION_RMSE_K="
            f"{rapid_difference:.10f}"
        )
        print(
            "MEAN_LSTM_MINUS_GRU_FULL_VALIDATION_RMSE_K="
            f"{full_difference:.10f}"
        )
        print(f"PREFERRED_BY_PRIMARY_METRIC={preferred}")
        print(
            "INTERPRETATION=positive RMSE difference favors GRU; negative favors LSTM"
        )
    if history_gain.empty:
        print("LONG_HISTORY_PAIRS=0 (complete lookback 20 and 40 for each model/seed)")
    else:
        print(f"LONG_HISTORY_PAIRS={len(history_gain)}")
        for model, part in history_gain.groupby("model", sort=True):
            mean_gain = float(part["long_minus_short_rapid_rmse_k"].mean())
            print(
                f"{model.upper()}_MEAN_LOOKBACK40_MINUS20_RAPID_RMSE_K="
                f"{mean_gain:.10f}"
            )
        print("LONG_HISTORY_INTERPRETATION=negative values mean lookback 40 improved RMSE")
    print(f"SUMMARY_SAVED={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
