#!/usr/bin/env python3
"""Small statistical helpers shared by the 0907 summaries."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def interval(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Summary values must be finite and non-empty")
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half = (
        float(stats.t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array)))
        if len(array) > 1 else 0.0
    )
    return {"n": len(array), "mean": mean, "sd": sd,
            "ci95_low": mean - half, "ci95_high": mean + half}


def holm_adjust(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(tuple(values), dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def paired_row(
    runs: pd.DataFrame,
    left: str,
    right: str,
    metric: str,
    comparison_id: str,
) -> dict[str, object]:
    left_rows = runs.loc[runs.feature_set.eq(left), ["seed", metric]]
    right_rows = runs.loc[runs.feature_set.eq(right), ["seed", metric]]
    matched = left_rows.merge(
        right_rows, on="seed", suffixes=("_left", "_right"), validate="one_to_one"
    )
    if len(matched) != 10:
        raise ValueError(f"{comparison_id} does not have ten matched seeds")
    differences = (
        matched[f"{metric}_left"] - matched[f"{metric}_right"]
    ).to_numpy(dtype=np.float64)
    summary = interval(differences)
    p_raw = 1.0 if np.allclose(differences, 0.0) else float(
        stats.wilcoxon(differences, alternative="two-sided").pvalue
    )
    right_mean = float(matched[f"{metric}_right"].mean())
    return {
        "comparison_id": comparison_id,
        "difference_definition": f"{left} minus {right}; negative favors {left}",
        "left_feature_set": left,
        "right_feature_set": right,
        "n_paired_seeds": len(matched),
        "paired_seeds": ";".join(map(str, matched.seed.tolist())),
        "left_matched_mean_k": float(matched[f"{metric}_left"].mean()),
        "right_matched_mean_k": right_mean,
        "mean_difference_k": summary["mean"],
        "relative_difference_vs_right_pct": 100.0 * float(summary["mean"]) / right_mean,
        "sd_difference_k": summary["sd"],
        "ci95_low_k": summary["ci95_low"],
        "ci95_high_k": summary["ci95_high"],
        "left_wins": int(np.sum(differences < 0.0)),
        "wilcoxon_p_raw": p_raw,
    }


def paired_table(
    runs: pd.DataFrame,
    comparisons: list[tuple[str, str, str]],
    metric: str,
) -> pd.DataFrame:
    table = pd.DataFrame([
        paired_row(runs, left, right, metric, comparison_id)
        for comparison_id, left, right in comparisons
    ])
    table["wilcoxon_p_holm"] = holm_adjust(table.wilcoxon_p_raw)
    return table
