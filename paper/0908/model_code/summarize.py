#!/usr/bin/env python3
"""Aggregate multi-seed validation or test metrics and freeze development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from protocol import DEFAULT_PREDICT_STEPS, EXPERIMENTS, LOOKBACK, SEEDS, project_dir, run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 0908 multi-seed experiments.")
    parser.add_argument("--stage", choices=("validation", "test"), default="validation")
    parser.add_argument("--predict-steps", type=int, default=DEFAULT_PREDICT_STEPS)
    parser.add_argument("--output-dir", type=Path, default=project_dir() / "outputs")
    return parser.parse_args()


def metric_path(root: Path, stage: str, horizon: int, experiment: str, seed: int) -> Path:
    if stage == "validation":
        return run_dir(root, experiment, horizon, seed) / "metrics.json"
    return (
        root / "final_test" / f"horizon_{horizon:02d}" / f"lookback_{LOOKBACK:02d}"
        / experiment / f"seed_{seed}" / "metrics.json"
    )


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    args = parse_args()
    rows = []
    hashes = {}
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            path = metric_path(args.output_dir, args.stage, args.predict_steps, experiment, seed)
            if not path.exists():
                continue
            raw = path.read_bytes()
            hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            payload = json.loads(raw.decode("utf-8"))
            metrics = payload["validation"] if args.stage == "validation" else payload["test"]
            rows.append({"experiment": experiment, "seed": seed, **metrics})
    table = pd.DataFrame(rows)
    summary_dir = args.output_dir / f"{args.stage}_summary" / f"horizon_{args.predict_steps:02d}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(summary_dir / "all_seed_metrics.csv", index=False)
    if table.empty:
        raise FileNotFoundError("No metrics were found")
    scalar_metrics = [
        column for column in table.columns
        if column not in {"experiment", "seed", "rmse_by_horizon_k", "mae_by_horizon_k"}
        and pd.api.types.is_numeric_dtype(table[column])
    ]
    grouped_rows = []
    for experiment, group in table.groupby("experiment", sort=False):
        row = {"experiment": experiment, "seeds": group["seed"].nunique()}
        for metric in scalar_metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        grouped_rows.append(row)
    summary = pd.DataFrame(grouped_rows).sort_values("trajectory_rmse_k_mean")
    summary.to_csv(summary_dir / "model_summary.csv", index=False)

    reference = "dkcdv_select_lag_kan"
    tests = []
    if reference in set(table["experiment"]):
        ref = table.loc[table["experiment"] == reference, ["seed", "trajectory_rmse_k"]]
        for experiment in EXPERIMENTS:
            if experiment == reference:
                continue
            other = table.loc[table["experiment"] == experiment, ["seed", "trajectory_rmse_k"]]
            paired = ref.merge(other, on="seed", suffixes=("_primary", "_other"))
            if len(paired) >= 5 and np.any(
                paired["trajectory_rmse_k_primary"] != paired["trajectory_rmse_k_other"]
            ):
                statistic, p_value = wilcoxon(
                    paired["trajectory_rmse_k_primary"], paired["trajectory_rmse_k_other"],
                    alternative="two-sided",
                )
                tests.append(
                    {
                        "primary": reference, "comparison": experiment, "pairs": len(paired),
                        "mean_primary": paired["trajectory_rmse_k_primary"].mean(),
                        "mean_comparison": paired["trajectory_rmse_k_other"].mean(),
                        "wilcoxon_statistic": statistic, "p_value": p_value,
                    }
                )
    if tests:
        adjusted = holm([float(row["p_value"]) for row in tests])
        for row, value in zip(tests, adjusted):
            row["holm_p_value"] = value
        pd.DataFrame(tests).to_csv(summary_dir / "paired_primary_comparisons.csv", index=False)

    expected = len(EXPERIMENTS) * len(SEEDS)
    complete = len(table) == expected and all(
        table.groupby("experiment")["seed"].nunique().reindex(EXPERIMENTS).fillna(0).eq(len(SEEDS))
    )
    status = {"stage": args.stage, "horizon": args.predict_steps, "runs_found": len(table), "runs_expected": expected, "complete": bool(complete)}
    (summary_dir / "summary_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if args.stage == "validation" and complete:
        development = args.output_dir / "development" / f"horizon_{args.predict_steps:02d}" / f"lookback_{LOOKBACK:02d}"
        manifest = {
            **status, "selection_metric": "validation_trajectory_rmse_k",
            "experiments": list(EXPERIMENTS), "seeds": list(SEEDS),
            "metric_file_sha256": hashes,
            "test_data_opened_by_development": False,
        }
        (development / "validation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(summary.to_string(index=False))
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
