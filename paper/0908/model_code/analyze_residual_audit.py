"""Audit validation residual tails for the selected DK&CDV-KAN model.

This is a descriptive validation-only audit. It does not open the frozen test set
and must not be used to remove difficult events from the official metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = (
    ROOT
    / "outputs"
    / "development"
    / "horizon_15"
    / "lookback_60"
    / "dkcdv_select_lag_kan"
)
REPORT_ROOT = ROOT / "reports" / "validation_analysis_20260910"
SOURCE_ROOT = REPORT_ROOT / "source_data"
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]


def main() -> None:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    val = joblib.load(ROOT / "processed_data" / "val_clean.pkl")

    actual = None
    frame_rows = None
    residuals = []
    per_seed = []
    for seed in SEEDS:
        z = np.load(MODEL_ROOT / f"seed_{seed}" / "validation_predictions.npz")
        if actual is None:
            actual = z["actual_k"].astype(float)
            frame_rows = z["frame_row_index"].astype(int)
        err = z["predicted_k"].astype(float) - z["actual_k"].astype(float)
        residuals.append(err)
        ae = np.abs(err).ravel()
        per_seed.append(
            {
                "seed": seed,
                "abs_error_p50_k": float(np.quantile(ae, 0.50)),
                "abs_error_p90_k": float(np.quantile(ae, 0.90)),
                "abs_error_p95_k": float(np.quantile(ae, 0.95)),
                "abs_error_p99_k": float(np.quantile(ae, 0.99)),
                "abs_error_max_k": float(np.max(ae)),
                "fraction_abs_error_gt_0p5k": float(np.mean(ae > 0.5)),
                "fraction_abs_error_gt_1k": float(np.mean(ae > 1.0)),
                "fraction_abs_error_gt_2k": float(np.mean(ae > 2.0)),
                "fraction_abs_error_gt_5k": float(np.mean(ae > 5.0)),
            }
        )

    errors = np.stack(residuals, axis=0)
    abs_errors = np.abs(errors)
    mean_abs = abs_errors.mean(axis=0)
    mean_sq = np.square(errors).mean(axis=0)

    # Rank target points by their mean squared error across seeds. Mapping is
    # origin row + horizon step because step 1 predicts the next row.
    flat_order = np.argsort(mean_sq.ravel())[::-1][:100]
    event_rows = []
    for rank, flat_index in enumerate(flat_order, start=1):
        origin_i, horizon_i = np.unravel_index(flat_index, mean_sq.shape)
        origin_row = int(frame_rows[origin_i])
        target_row = origin_row + int(horizon_i) + 1
        row = val.iloc[target_row]
        event_rows.append(
            {
                "rank": rank,
                "origin_frame_row": origin_row,
                "horizon_step": int(horizon_i) + 1,
                "horizon_seconds": (int(horizon_i) + 1) * 10,
                "target_frame_row": target_row,
                "source_row_index": row["source_row_index"],
                "source_timestamp": row["source_timestamp"],
                "actual_thv_k": float(actual[origin_i, horizon_i]),
                "mean_abs_error_across_seeds_k": float(mean_abs[origin_i, horizon_i]),
                "root_mean_squared_error_across_seeds_k": float(np.sqrt(mean_sq[origin_i, horizon_i])),
                "min_prediction_across_seeds_k": float(
                    actual[origin_i, horizon_i] + errors[:, origin_i, horizon_i].min()
                ),
                "max_prediction_across_seeds_k": float(
                    actual[origin_i, horizon_i] + errors[:, origin_i, horizon_i].max()
                ),
                "file_id": row["file_id"],
                "source_group": row["source_group"],
            }
        )

    pd.DataFrame(per_seed).to_csv(
        SOURCE_ROOT / "kan_residual_tail_by_seed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(event_rows).to_csv(
        SOURCE_ROOT / "kan_top_residual_points.csv", index=False, encoding="utf-8-sig"
    )

    pooled = abs_errors.ravel()
    final_horizon = abs_errors[:, :, -1].ravel()
    summary = {
        "scope": "validation only; frozen test not opened",
        "interpretation": "descriptive residual-tail audit; official metrics remain unchanged",
        "absolute_error_all_seeds_all_horizons_k": {
            "p50": float(np.quantile(pooled, 0.50)),
            "p90": float(np.quantile(pooled, 0.90)),
            "p95": float(np.quantile(pooled, 0.95)),
            "p99": float(np.quantile(pooled, 0.99)),
            "max": float(np.max(pooled)),
            "fraction_gt_0p5k": float(np.mean(pooled > 0.5)),
            "fraction_gt_1k": float(np.mean(pooled > 1.0)),
            "fraction_gt_2k": float(np.mean(pooled > 2.0)),
            "fraction_gt_5k": float(np.mean(pooled > 5.0)),
        },
        "final_150s_absolute_error_k": {
            "p50": float(np.quantile(final_horizon, 0.50)),
            "p90": float(np.quantile(final_horizon, 0.90)),
            "p95": float(np.quantile(final_horizon, 0.95)),
            "p99": float(np.quantile(final_horizon, 0.99)),
            "max": float(np.max(final_horizon)),
        },
    }
    (REPORT_ROOT / "residual_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
