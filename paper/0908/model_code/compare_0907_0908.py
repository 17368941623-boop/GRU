"""Common-window validation comparison between the 0907 and 0908 models.

The 0907 models predict only Thv(t+15). The 0908 model predicts a complete
15-step trajectory, so this audit compares only the 15th output on the exact
intersection of validation origins. It does not read either version's test set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT_0908 = Path(__file__).resolve().parents[1]
ROOT_0907 = ROOT_0908.parent / "0907" / "outputs_0907"
NEW_ROOT = (
    ROOT_0908
    / "outputs"
    / "development"
    / "horizon_15"
    / "lookback_60"
    / "dkcdv_select_lag_kan"
)
OLD_ROOT = (
    ROOT_0907
    / "development"
    / "horizon_15"
    / "lookback_60"
    / "parallel_gru_kan_gnn"
)
OUT = ROOT_0908 / "reports" / "cross_version_comparison_0907_0908"
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
OLD_CONFIGS = ["raw20", "full20", "full20_no_module_valve_dynamics"]


def metrics(actual: np.ndarray, predicted: np.ndarray, current: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    delta_actual = actual - current
    delta_predicted = predicted - current
    return {
        "rmse_k": float(np.sqrt(np.mean(np.square(error)))),
        "mae_k": float(np.mean(np.abs(error))),
        "p95_absolute_error_k": float(np.quantile(np.abs(error), 0.95)),
        "max_absolute_error_k": float(np.max(np.abs(error))),
        "bias_k": float(np.mean(error)),
        "direction_accuracy": float(np.mean(np.sign(delta_actual) == np.sign(delta_predicted))),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    for seed in SEEDS:
        new_npz = np.load(NEW_ROOT / f"seed_{seed}" / "validation_predictions.npz")
        new_index = pd.read_csv(NEW_ROOT / f"seed_{seed}" / "validation_prediction_index.csv")
        new = new_index[["file_id", "source_row_index"]].copy()
        new["current_k"] = new_npz["current_k"].astype(float)
        new["actual_k"] = new_npz["actual_k"][:, -1].astype(float)
        new["new_prediction_k"] = new_npz["predicted_k"][:, -1].astype(float)

        old_frames: dict[str, pd.DataFrame] = {}
        for config in OLD_CONFIGS:
            old = pd.read_csv(
                OLD_ROOT / config / f"seed_{seed}" / "validation_predictions.csv",
                usecols=[
                    "file_id",
                    "source_row_index",
                    "current_Thv_k",
                    "actual_Future_Thv_15step_k",
                    "predicted_Future_Thv_15step_k",
                    "rapid_cooling_mask",
                ],
            )
            old_frames[config] = old.rename(
                columns={
                    "current_Thv_k": f"{config}_current_k",
                    "actual_Future_Thv_15step_k": f"{config}_actual_k",
                    "predicted_Future_Thv_15step_k": f"{config}_prediction_k",
                    "rapid_cooling_mask": f"{config}_rapid_mask",
                }
            )

        merged = new
        for config in OLD_CONFIGS:
            merged = merged.merge(
                old_frames[config], on=["file_id", "source_row_index"], how="inner", validate="one_to_one"
            )
        if len(merged) != len(new):
            raise AssertionError(f"Expected all 0908 rows to be present in 0907, got {len(merged)}")

        for config in OLD_CONFIGS:
            actual_gap = np.max(np.abs(merged["actual_k"] - merged[f"{config}_actual_k"]))
            current_gap = np.max(np.abs(merged["current_k"] - merged[f"{config}_current_k"]))
            if actual_gap > 1e-8 or current_gap > 1e-8:
                raise AssertionError(
                    f"Target mismatch for {config}, seed {seed}: actual={actual_gap}, current={current_gap}"
                )

        actual = merged["actual_k"].to_numpy(float)
        current = merged["current_k"].to_numpy(float)
        rapid = merged["raw20_rapid_mask"].to_numpy(bool)

        new_metrics = metrics(actual, merged["new_prediction_k"].to_numpy(float), current)
        rows.append(
            {
                "seed": seed,
                "model": "0908_dkcdv_select_lag_kan",
                "common_validation_windows": len(merged),
                "rapid_windows_0907_definition": int(rapid.sum()),
                **new_metrics,
                "rapid_rmse_k": float(
                    np.sqrt(np.mean(np.square(merged.loc[rapid, "new_prediction_k"] - actual[rapid])))
                ),
            }
        )
        for config in OLD_CONFIGS:
            prediction = merged[f"{config}_prediction_k"].to_numpy(float)
            old_metrics = metrics(actual, prediction, current)
            rows.append(
                {
                    "seed": seed,
                    "model": f"0907_{config}",
                    "common_validation_windows": len(merged),
                    "rapid_windows_0907_definition": int(rapid.sum()),
                    **old_metrics,
                    "rapid_rmse_k": float(np.sqrt(np.mean(np.square(prediction[rapid] - actual[rapid])))),
                }
            )

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "common_window_metrics_by_seed.csv", index=False, encoding="utf-8-sig")

    metric_columns = [
        "rmse_k",
        "mae_k",
        "p95_absolute_error_k",
        "max_absolute_error_k",
        "bias_k",
        "direction_accuracy",
        "rapid_rmse_k",
    ]
    aggregate_rows = []
    for model, group in per_seed.groupby("model", sort=False):
        row: dict[str, float | int | str] = {"model": model, "n_seeds": len(group)}
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_seed_sd"] = float(group[column].std(ddof=1))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUT / "common_window_model_summary.csv", index=False, encoding="utf-8-sig")

    new_rmse = per_seed.loc[
        per_seed["model"].eq("0908_dkcdv_select_lag_kan"), ["seed", "rmse_k"]
    ].set_index("seed")["rmse_k"]
    paired_rows = []
    raw_p = []
    rng = np.random.default_rng(20260910)
    for config in OLD_CONFIGS:
        old_rmse = per_seed.loc[
            per_seed["model"].eq(f"0907_{config}"), ["seed", "rmse_k"]
        ].set_index("seed")["rmse_k"]
        diff = old_rmse.loc[SEEDS].to_numpy() - new_rmse.loc[SEEDS].to_numpy()
        boot = np.mean(rng.choice(diff, size=(50_000, len(diff)), replace=True), axis=1)
        p_value = float(wilcoxon(diff, alternative="two-sided", method="exact").pvalue)
        raw_p.append(p_value)
        paired_rows.append(
            {
                "comparison": f"0908 vs 0907 {config}",
                "difference_definition": "0907 RMSE minus 0908 RMSE; positive favours 0908",
                "n_paired_seeds": len(diff),
                "mean_improvement_k": float(np.mean(diff)),
                "relative_improvement_vs_0907_pct": float(100 * np.mean(diff) / old_rmse.mean()),
                "seed_bootstrap_95_low_k": float(np.quantile(boot, 0.025)),
                "seed_bootstrap_95_high_k": float(np.quantile(boot, 0.975)),
                "0908_wins_of_10": int(np.sum(diff > 0)),
                "wilcoxon_p_raw": p_value,
            }
        )
    adjusted = holm_adjust(raw_p)
    for row, p_value in zip(paired_rows, adjusted):
        row["wilcoxon_p_holm"] = p_value
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "paired_rmse_comparisons.csv", index=False, encoding="utf-8-sig")

    payload = {
        "scope": "common 260501 validation origins only; no test predictions opened",
        "comparison_endpoint": "Thv at t+15 (150 s)",
        "common_validation_windows": int(per_seed["common_validation_windows"].iloc[0]),
        "seeds": SEEDS,
        "model_summary": aggregate.to_dict(orient="records"),
        "paired_comparisons": paired.to_dict(orient="records"),
        "attribution_limit": (
            "0907 and 0908 differ in output task and architecture as well as graph construction; "
            "cross-version improvement cannot be attributed to causal feature analysis alone"
        ),
    }
    (OUT / "comparison_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(aggregate.to_string(index=False))
    print("\nPaired comparisons\n", paired.to_string(index=False))


if __name__ == "__main__":
    main()
