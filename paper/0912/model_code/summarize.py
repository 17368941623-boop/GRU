"""Aggregate the ten Raw54-GRU seeds and create validation figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from protocol import PREDICT_STEPS, SAMPLE_PERIOD_SECONDS, SEEDS, output_dir, project_dir, run_dir


SCALAR_METRICS = (
    "trajectory_rmse_k",
    "trajectory_mae_k",
    "final_horizon_rmse_k",
    "final_horizon_mae_k",
    "p95_absolute_error_k",
    "rapid_trajectory_rmse_k",
    "rapid_final_rmse_k",
    "valve_event_trajectory_rmse_k",
    "valve_event_final_rmse_k",
    "peak_cooling_delta_mae_k",
    "peak_cooling_time_mae_seconds",
    "mean_absolute_integrated_error_k_s",
    "direction_accuracy",
)


def main() -> None:
    summary_dir = output_dir() / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    records = []
    curves = []
    missing = []
    for seed in SEEDS:
        path = run_dir(seed) / "metrics.json"
        if not path.exists():
            missing.append(seed)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload["validation"]
        row = {
            "seed": seed,
            "best_epoch": payload["best_epoch"],
            "training_seconds": payload["training_seconds"],
            "parameter_count": payload["parameter_count"],
            "train_windows": payload["train_windows"],
            "validation_windows": payload["validation_windows"],
        }
        row.update({name: validation.get(name, np.nan) for name in SCALAR_METRICS})
        records.append(row)
        for step, (rmse, mae) in enumerate(
            zip(validation["rmse_by_horizon_k"], validation["mae_by_horizon_k"]), start=1
        ):
            curves.append({
                "seed": seed,
                "step": step,
                "seconds": step * SAMPLE_PERIOD_SECONDS,
                "rmse_k": rmse,
                "mae_k": mae,
            })
    if missing:
        raise RuntimeError(f"Cannot summarize; missing seeds: {missing}")
    seed_frame = pd.DataFrame(records).sort_values("seed")
    curve_frame = pd.DataFrame(curves)
    seed_frame.to_csv(summary_dir / "seed_metrics.csv", index=False)
    curve_frame.to_csv(summary_dir / "horizon_metrics_by_seed.csv", index=False)
    aggregate_rows = []
    for metric in SCALAR_METRICS:
        values = seed_frame[metric].to_numpy(dtype=float)
        aggregate_rows.append({
            "metric": metric,
            "n_seeds": int(np.isfinite(values).sum()),
            "mean": float(np.nanmean(values)),
            "std_across_seeds": float(np.nanstd(values, ddof=1)),
            "median": float(np.nanmedian(values)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        })
    pd.DataFrame(aggregate_rows).to_csv(summary_dir / "aggregate_metrics.csv", index=False)

    mean_curve = curve_frame.groupby("seconds").agg(
        rmse_mean=("rmse_k", "mean"),
        rmse_std=("rmse_k", "std"),
        mae_mean=("mae_k", "mean"),
        mae_std=("mae_k", "std"),
    ).reset_index()
    mean_curve.to_csv(summary_dir / "horizon_metrics_mean_std.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = mean_curve["seconds"].to_numpy()
    y = mean_curve["rmse_mean"].to_numpy()
    s = mean_curve["rmse_std"].to_numpy()
    ax.plot(x, y, color="#1f77b4", marker="o", ms=3, lw=1.8, label="Raw54-GRU")
    ax.fill_between(x, y - s, y + s, color="#1f77b4", alpha=0.18, label="±1 SD across seeds")
    ax.set_xlabel("Forecast horizon (s)")
    ax.set_ylabel("Validation RMSE (K)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(summary_dir / "validation_rmse_curve.png", dpi=300)
    fig.savefig(summary_dir / "validation_rmse_curve.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    values = seed_frame["trajectory_rmse_k"].to_numpy()
    ax.boxplot(values, widths=0.45, showmeans=True)
    rng = np.random.default_rng(20260912)
    ax.scatter(1 + rng.uniform(-0.07, 0.07, len(values)), values, s=28, color="#d62728", zorder=3)
    ax.set_xticks([1], ["Raw54-GRU"])
    ax.set_ylabel("Trajectory RMSE (K)")
    fig.tight_layout()
    fig.savefig(summary_dir / "seed_rmse_distribution.png", dpi=300)
    fig.savefig(summary_dir / "seed_rmse_distribution.pdf")
    plt.close(fig)

    report = {
        "experiment": "raw54_gru",
        "completed_seeds": list(SEEDS),
        "lookback": 60,
        "predict_steps": PREDICT_STEPS,
        "primary_metric_mean": float(seed_frame["trajectory_rmse_k"].mean()),
        "primary_metric_std_across_seeds": float(seed_frame["trajectory_rmse_k"].std(ddof=1)),
        "final_horizon_rmse_mean": float(seed_frame["final_horizon_rmse_k"].mean()),
        "total_training_seconds": float(seed_frame["training_seconds"].sum()),
        "test_data_loaded": False,
    }
    (summary_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
