#!/usr/bin/env python3
"""Unify the lookback-65 Thv model runs and produce auditable ablation tables.

Only validation metrics are summarized. The held-out 0715-BACK test split is not
opened by this script.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_runs"
SUMMARY = ROOT / "summary"

MODEL_ORDER = [
    "gru_baseline",
    "lstm_baseline",
    "tcn_baseline",
    "serial_mlp_gnn_gru",
    "serial_kan_gnn_gru",
    "parallel_gru_mlp_gnn",
    "parallel_gru_kan_gnn",
]

MODEL_LABELS = {
    "gru_baseline": "GRU baseline",
    "lstm_baseline": "LSTM baseline",
    "tcn_baseline": "TCN baseline",
    "serial_mlp_gnn_gru": "Serial MLP-GNN-GRU",
    "serial_kan_gnn_gru": "Serial monotonic-KAN-GNN-GRU",
    "parallel_gru_mlp_gnn": "Parallel GRU-MLP-GNN",
    "parallel_gru_kan_gnn": "Parallel GRU-monotonic-KAN-GNN",
}

PAPER_ROLES = {
    "gru_baseline": "temporal baseline and no-graph ablation",
    "lstm_baseline": "recurrent temporal-encoder comparator",
    "tcn_baseline": "causal convolutional temporal-encoder comparator",
    "serial_mlp_gnn_gru": "serial topology with unconstrained MLP edge gates",
    "serial_kan_gnn_gru": "serial topology with monotonic KAN valve gates",
    "parallel_gru_mlp_gnn": "parallel topology with unconstrained MLP edge gates",
    "parallel_gru_kan_gnn": "parallel topology with monotonic KAN valve gates",
}

EXPECTED_SEEDS = {
    "gru_baseline": {42, 52, 62, 72, 82, 92, 102, 112, 122, 132},
    "lstm_baseline": {42, 52, 62, 72, 82, 92, 102, 112, 122, 132},
    "tcn_baseline": {42, 62, 82, 102, 122},
    "serial_mlp_gnn_gru": {42, 52, 62, 72, 82, 92, 102, 112, 122, 132},
    "serial_kan_gnn_gru": {42, 52, 62, 72, 82, 92, 102, 112, 122, 132},
    "parallel_gru_mlp_gnn": {42, 52, 62, 72, 82, 92, 102, 112, 122, 132},
    "parallel_gru_kan_gnn": {42, 52, 62, 72, 82, 92, 102, 112, 122, 132},
}

FLOAT_FIELDS = [
    "training_seconds_total",
    "validation_rmse_k",
    "validation_mae_k",
    "validation_p95_absolute_error_k",
    "validation_max_absolute_error_k",
    "validation_rapid_rmse_k",
    "validation_rapid_mae_k",
    "validation_rapid_p95_absolute_error_k",
    "validation_rapid_max_absolute_error_k",
    "validation_direction_accuracy",
    "validation_rapid_direction_accuracy",
    "persistence_validation_rmse_k",
    "rapid_persistence_validation_rmse_k",
]


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def t_critical_975(df: int) -> float:
    # Exact values needed by this batch; normal approximation is only a fallback.
    table = {4: 2.776445105, 9: 2.262157163}
    return table.get(df, 1.959963985)


def mean_sd_ci(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0, mean, mean
    sd = statistics.stdev(values)
    half = t_critical_975(len(values) - 1) * sd / math.sqrt(len(values))
    return mean, sd, mean - half, mean + half


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and math.isclose(indexed[j][1], indexed[i][1], rel_tol=0.0, abs_tol=1e-15):
            j += 1
        average = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = average
        i = j
    return ranks


def exact_wilcoxon_two_sided(differences: list[float]) -> float:
    nonzero = [d for d in differences if not math.isclose(d, 0.0, rel_tol=0.0, abs_tol=1e-15)]
    if not nonzero:
        return 1.0
    ranks = average_ranks([abs(d) for d in nonzero])
    observed = sum(rank for rank, diff in zip(ranks, nonzero) if diff > 0)
    sums = []
    for signs in itertools.product((0, 1), repeat=len(ranks)):
        sums.append(sum(rank for rank, sign in zip(ranks, signs) if sign))
    lower = sum(value <= observed + 1e-12 for value in sums) / len(sums)
    upper = sum(value >= observed - 1e-12 for value in sums) / len(sums)
    return min(1.0, 2.0 * min(lower, upper))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda idx: p_values[idx])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def load_recurrent_runs() -> list[dict]:
    rows = []
    recurrent_root = RAW / "recurrent_baselines_10seeds"
    for model in ("gru_baseline", "lstm_baseline"):
        for path in sorted((recurrent_root / model).glob("*/seed_*/metrics.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            validation = data["validation_metrics"]
            rapid = data["rapid_validation_metrics"]
            persistence = data["persistence_validation_metrics"]
            rapid_persistence = data["rapid_persistence_validation_metrics"]
            rows.append(
                {
                    "study": "recurrent",
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "config_id": data["config_id"],
                    "paper_role": PAPER_ROLES[model],
                    "seed": int(data["seed"]),
                    "lookback": int(data["lookback"]),
                    "common_origin_lookback": int(data["common_origin_lookback"]),
                    "predict_steps": int(data["predict_steps"]),
                    "trainable_parameters": int(data["trainable_parameters"]),
                    "completed_epochs": int(data["completed_epochs"]),
                    "best_epoch": int(data["best_epoch"]),
                    "training_seconds_total": float(data["training_seconds_total"]),
                    "validation_rmse_k": float(validation["rmse_k"]),
                    "validation_mae_k": float(validation["mae_k"]),
                    "validation_p95_absolute_error_k": float(validation["p95_absolute_error_k"]),
                    "validation_max_absolute_error_k": float(validation["max_absolute_error_k"]),
                    "validation_rapid_rmse_k": float(rapid["rmse_k"]),
                    "validation_rapid_mae_k": float(rapid["mae_k"]),
                    "validation_rapid_p95_absolute_error_k": float(rapid["p95_absolute_error_k"]),
                    "validation_rapid_max_absolute_error_k": float(rapid["max_absolute_error_k"]),
                    "validation_direction_accuracy": float(data["validation_direction_accuracy"]),
                    "validation_rapid_direction_accuracy": float(data["rapid_validation_direction_accuracy"]),
                    "persistence_validation_rmse_k": float(persistence["rmse_k"]),
                    "rapid_persistence_validation_rmse_k": float(rapid_persistence["rmse_k"]),
                    "validation_windows": int(data["validation_windows"]),
                    "validation_rapid_windows": int(data["validation_rapid_windows"]),
                    "rapid_threshold_k_train_only": float(data["rapid_threshold_k_train_only"]),
                    "test_data_loaded": as_bool(data["test_data_loaded"]),
                    "source_metrics": str(path.relative_to(ROOT)),
                }
            )
    return rows


def load_csv_runs(path: Path, study: str, rapid_threshold: float) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            model = source["model"]
            row = dict(source)
            row["study"] = study
            row["model_label"] = MODEL_LABELS[model]
            row["paper_role"] = PAPER_ROLES[model]
            for field in ("seed", "lookback", "common_origin_lookback", "predict_steps", "trainable_parameters", "completed_epochs", "best_epoch", "validation_windows", "validation_rapid_windows"):
                row[field] = int(float(row[field]))
            for field in FLOAT_FIELDS:
                row[field] = float(row[field])
            row["rapid_threshold_k_train_only"] = rapid_threshold
            row["test_data_loaded"] = as_bool(row["test_data_loaded"])
            row["source_metrics"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def validate_runs(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    assert set(grouped) == set(MODEL_ORDER), sorted(grouped)
    model_audit = {}
    for model in MODEL_ORDER:
        model_rows = grouped[model]
        seeds = {int(row["seed"]) for row in model_rows}
        assert seeds == EXPECTED_SEEDS[model], (model, seeds)
        assert len(model_rows) == len(seeds), f"Duplicate run found for {model}"
        assert {int(row["lookback"]) for row in model_rows} == {65}
        assert {int(row["common_origin_lookback"]) for row in model_rows} == {120}
        assert {int(row["predict_steps"]) for row in model_rows} == {15}
        assert {int(row["validation_windows"]) for row in model_rows} == {25600}
        assert {int(row["validation_rapid_windows"]) for row in model_rows} == {1596}
        assert not any(row["test_data_loaded"] for row in model_rows)
        model_audit[model] = {
            "n_runs": len(model_rows),
            "seeds": sorted(seeds),
            "test_data_loaded": False,
        }
    thresholds = {round(float(row["rapid_threshold_k_train_only"]), 12) for row in rows}
    assert len(thresholds) == 1, thresholds
    return {
        "audit_status": "PASS",
        "total_complete_runs": len(rows),
        "lookback": 65,
        "common_origin_lookback": 120,
        "predict_steps": 15,
        "validation_split": "Original 260501",
        "held_out_split_read": False,
        "validation_windows_per_run": 25600,
        "rapid_validation_windows_per_run": 1596,
        "rapid_fraction": 1596 / 25600,
        "rapid_threshold_k_train_only": next(iter(thresholds)),
        "rapid_definition": "actual Thv(t+15)-Thv(t) <= training-only 10th-percentile threshold",
        "models": model_audit,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_models(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    summaries = []
    metrics = [
        "validation_rapid_rmse_k",
        "validation_rmse_k",
        "validation_rapid_mae_k",
        "validation_mae_k",
        "validation_rapid_p95_absolute_error_k",
        "validation_p95_absolute_error_k",
        "validation_rapid_direction_accuracy",
    ]
    for model in MODEL_ORDER:
        model_rows = grouped[model]
        summary = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "paper_role": PAPER_ROLES[model],
            "n_seeds": len(model_rows),
            "seed_set": ";".join(str(v) for v in sorted(int(row["seed"]) for row in model_rows)),
            "trainable_parameters": int(model_rows[0]["trainable_parameters"]),
            "training_minutes_mean": statistics.fmean(float(row["training_seconds_total"]) for row in model_rows) / 60.0,
            "completed_epochs_mean": statistics.fmean(float(row["completed_epochs"]) for row in model_rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in model_rows]
            mean, sd, low, high = mean_sd_ci(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
            summary[f"{metric}_ci95_low"] = low
            summary[f"{metric}_ci95_high"] = high
        persistence = statistics.fmean(float(row["rapid_persistence_validation_rmse_k"]) for row in model_rows)
        summary["rapid_persistence_validation_rmse_k"] = persistence
        summary["rapid_rmse_reduction_vs_persistence_pct"] = 100.0 * (persistence - summary["validation_rapid_rmse_k_mean"]) / persistence
        summaries.append(summary)
    ranked = sorted(summaries, key=lambda row: row["validation_rapid_rmse_k_mean"])
    for rank, row in enumerate(ranked, start=1):
        row["rapid_validation_rank"] = rank
    return sorted(ranked, key=lambda row: row["rapid_validation_rank"])


def paired_comparisons(rows: list[dict], comparisons: list[tuple[str, str, str]]) -> list[dict]:
    by_model_seed = {(row["model"], int(row["seed"])): row for row in rows}
    output = []
    for metric in ("validation_rapid_rmse_k", "validation_rmse_k"):
        metric_rows = []
        for comparison_id, left, right in comparisons:
            common = sorted(EXPECTED_SEEDS[left] & EXPECTED_SEEDS[right])
            left_values = [float(by_model_seed[(left, seed)][metric]) for seed in common]
            right_values = [float(by_model_seed[(right, seed)][metric]) for seed in common]
            differences = [left_value - right_value for left_value, right_value in zip(left_values, right_values)]
            mean, sd, low, high = mean_sd_ci(differences)
            metric_rows.append(
                {
                    "metric": metric,
                    "comparison_id": comparison_id,
                    "difference_definition": f"{left} minus {right}; negative favors {left}",
                    "left_model": left,
                    "right_model": right,
                    "n_paired_seeds": len(common),
                    "paired_seeds": ";".join(str(seed) for seed in common),
                    "left_matched_mean_k": statistics.fmean(left_values),
                    "right_matched_mean_k": statistics.fmean(right_values),
                    "mean_difference_k": mean,
                    "relative_difference_vs_right_pct": 100.0 * mean / statistics.fmean(right_values),
                    "sd_difference_k": sd,
                    "ci95_low_k": low,
                    "ci95_high_k": high,
                    "left_wins": sum(left_value < right_value for left_value, right_value in zip(left_values, right_values)),
                    "wilcoxon_p_raw": exact_wilcoxon_two_sided(differences),
                }
            )
        adjusted = holm_adjust([row["wilcoxon_p_raw"] for row in metric_rows])
        for row, p_value in zip(metric_rows, adjusted):
            row["wilcoxon_p_holm"] = p_value
        output.extend(metric_rows)
    return output


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    recurrent = load_recurrent_runs()
    if not recurrent:
        raise RuntimeError("No recurrent baseline runs were found")
    threshold = float(recurrent[0]["rapid_threshold_k_train_only"])
    graph = load_csv_runs(
        RAW / "graph_models_10seeds" / "validation_summary" / "validation_seed_runs.csv",
        "graph",
        threshold,
    )
    tcn = load_csv_runs(
        RAW / "tcn_baseline_5seeds" / "validation_summary" / "validation_seed_runs.csv",
        "tcn",
        threshold,
    )
    rows = recurrent + tcn + graph
    rows.sort(key=lambda row: (MODEL_ORDER.index(row["model"]), int(row["seed"])))
    audit = validate_runs(rows)

    seed_fields = [
        "study", "model", "model_label", "config_id", "paper_role", "seed",
        "lookback", "common_origin_lookback", "predict_steps", "trainable_parameters",
        "completed_epochs", "best_epoch", "training_seconds_total",
        "validation_rmse_k", "validation_mae_k", "validation_p95_absolute_error_k",
        "validation_max_absolute_error_k", "validation_rapid_rmse_k",
        "validation_rapid_mae_k", "validation_rapid_p95_absolute_error_k",
        "validation_rapid_max_absolute_error_k", "validation_direction_accuracy",
        "validation_rapid_direction_accuracy", "persistence_validation_rmse_k",
        "rapid_persistence_validation_rmse_k", "validation_windows",
        "validation_rapid_windows", "rapid_threshold_k_train_only", "test_data_loaded",
        "source_metrics",
    ]
    write_csv(SUMMARY / "ablation_seed_runs.csv", rows, seed_fields)
    write_csv(SUMMARY / "ablation_model_summary.csv", summarize_models(rows))

    vs_gru = [(f"{model}_vs_gru", model, "gru_baseline") for model in MODEL_ORDER if model != "gru_baseline"]
    write_csv(SUMMARY / "paired_vs_gru.csv", paired_comparisons(rows, vs_gru))

    structural = [
        ("serial_vs_parallel_mlp", "serial_mlp_gnn_gru", "parallel_gru_mlp_gnn"),
        ("serial_vs_parallel_kan", "serial_kan_gnn_gru", "parallel_gru_kan_gnn"),
        ("kan_vs_mlp_serial", "serial_kan_gnn_gru", "serial_mlp_gnn_gru"),
        ("kan_vs_mlp_parallel", "parallel_gru_kan_gnn", "parallel_gru_mlp_gnn"),
    ]
    write_csv(SUMMARY / "paired_structural_effects.csv", paired_comparisons(rows, structural))

    (SUMMARY / "batch_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
