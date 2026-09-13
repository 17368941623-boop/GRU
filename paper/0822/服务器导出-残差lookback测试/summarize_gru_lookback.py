#!/usr/bin/env python3
"""Aggregate the formal GRU lookback scan without opening any dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "gru_lookback_outputs_val260501_test0715BACK"
EXPECTED_LOOKBACKS = tuple(range(10, 61, 5))
EXPECTED_COMMON_ORIGIN_LOOKBACK = max(EXPECTED_LOOKBACKS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the formal GRU Delta-Thv lookback scan."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--predict-steps",
        type=int,
        default=15,
        choices=(5, 10, 15, 20, 30),
        help="Only summarize runs for this forecast horizon.",
    )
    parser.add_argument(
        "--expected-seeds",
        default="42,62,82",
        help="Comma-separated seed set used for the formal mean/std comparison.",
    )
    return parser.parse_args()


def parse_seed_set(text: str) -> tuple[int, ...]:
    seeds = tuple(sorted({int(item.strip()) for item in text.split(",") if item.strip()}))
    if not seeds:
        raise ValueError("--expected-seeds must contain at least one integer")
    return seeds


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runs(output_dir: Path, predict_steps: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    paths = sorted(output_dir.glob("gru/lookback_*/seed_*/metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No completed GRU metrics.json files under {output_dir}")

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_type") != "gru":
            raise ValueError(f"Non-GRU result entered the scan: {path}")
        if int(payload.get("predict_steps", -1)) != predict_steps:
            raise ValueError(
                f"Result horizon is not {predict_steps} steps: {path}"
            )
        if str(payload.get("validation_parent")) != "260501":
            raise ValueError(f"Result does not use Original 260501 validation: {path}")
        if int(payload.get("common_origin_lookback", -1)) != EXPECTED_COMMON_ORIGIN_LOOKBACK:
            raise ValueError(
                f"Result does not use common 60-step origins: {path}"
            )
        validation = payload["validation_metrics"]
        rapid = payload["rapid_validation_metrics"]
        persistence = payload["persistence_validation_metrics"]
        rapid_persistence = payload["rapid_persistence_validation_metrics"]
        fixed_test = payload.get("fixed_test_window_evaluation")
        if not isinstance(fixed_test, dict):
            raise ValueError(f"Missing fixed 0715-BACK test-window metrics: {path}")
        if str(fixed_test.get("test_parent")) != "0715-BACK":
            raise ValueError(f"Fixed test diagnostic is not Original 0715-BACK: {path}")
        if int(fixed_test.get("predict_steps", -1)) != predict_steps:
            raise ValueError(f"Fixed test horizon mismatch: {path}")
        if any(
            bool(fixed_test.get(key, True))
            for key in (
                "used_for_gradient_updates",
                "used_for_early_stopping",
                "used_for_checkpoint_selection",
            )
        ):
            raise ValueError(f"0715-BACK test window entered model fitting: {path}")
        fixed_window = fixed_test["window_metrics"]
        fixed_rapid = fixed_test["rapid_window_metrics"]
        fixed_persistence = fixed_test["persistence_window_metrics"]
        fixed_rapid_persistence = fixed_test["rapid_persistence_window_metrics"]
        best = payload["best_epoch_history"]
        last = payload["last_completed_epoch_history"]
        normalization_path = path.parent / "normalization_report.csv"
        history_path = path.parent / "training_history.csv"
        if not normalization_path.exists() or not history_path.exists():
            raise FileNotFoundError(
                f"Missing normalization report or epoch history beside {path}"
            )
        rows.append(
            {
                "model": "gru",
                "predict_steps": predict_steps,
                "predict_seconds": float(payload["predict_seconds"]),
                "lookback": int(payload["lookback"]),
                "history_seconds": float(payload["lookback_seconds"]),
                "seed": int(payload["seed"]),
                "best_epoch": int(payload["best_epoch"]),
                "completed_epochs": int(payload["completed_epochs"]),
                "best_train_rmse_k": float(best["train_rmse_k"]),
                "best_train_rapid_rmse_k": float(best["train_rapid_rmse_k"]),
                "best_validation_rmse_k": float(validation["rmse_k"]),
                "full_validation_windows": int(validation["n_windows"]),
                "best_validation_mae_k": float(validation["mae_k"]),
                "best_validation_p95_absolute_error_k": float(
                    validation["p95_absolute_error_k"]
                ),
                "best_validation_max_absolute_error_k": float(
                    validation["max_absolute_error_k"]
                ),
                "best_validation_bias_k": float(validation["bias_k"]),
                "best_rapid_validation_rmse_k": float(rapid["rmse_k"]),
                "rapid_validation_windows": int(rapid["n_windows"]),
                "best_rapid_validation_mae_k": float(rapid["mae_k"]),
                "best_rapid_validation_p95_absolute_error_k": float(
                    rapid["p95_absolute_error_k"]
                ),
                "best_rapid_validation_max_absolute_error_k": float(
                    rapid["max_absolute_error_k"]
                ),
                "best_rapid_validation_bias_k": float(rapid["bias_k"]),
                "last_epoch_train_rmse_k": float(last["train_rmse_k"]),
                "last_epoch_train_rapid_rmse_k": float(last["train_rapid_rmse_k"]),
                "last_epoch_validation_rmse_k": float(last["validation_rmse_k"]),
                "last_epoch_rapid_validation_rmse_k": float(
                    last["validation_rapid_rmse_k"]
                ),
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
                "test_window_start_source_row": int(fixed_test["window_start"]),
                "test_window_end_source_row": int(fixed_test["window_end"]),
                "test_window_valid_origins": int(fixed_window["n_windows"]),
                "test_window_rmse_k": float(fixed_window["rmse_k"]),
                "test_window_mae_k": float(fixed_window["mae_k"]),
                "test_window_p95_absolute_error_k": float(
                    fixed_window["p95_absolute_error_k"]
                ),
                "test_window_max_absolute_error_k": float(
                    fixed_window["max_absolute_error_k"]
                ),
                "test_window_rapid_origins": int(fixed_rapid["n_windows"]),
                "test_window_rapid_rmse_k": float(fixed_rapid["rmse_k"]),
                "test_window_rapid_mae_k": float(fixed_rapid["mae_k"]),
                "test_window_persistence_rmse_k": float(
                    fixed_persistence["rmse_k"]
                ),
                "test_window_rapid_persistence_rmse_k": float(
                    fixed_rapid_persistence["rmse_k"]
                ),
                "rapid_threshold_k": float(payload["rapid_threshold_k_train_only"]),
                "trainable_parameters": int(payload["trainable_parameters"]),
                "training_seconds_total": float(payload["training_seconds_total"]),
                "seconds_per_completed_epoch": float(
                    payload["seconds_per_completed_epoch"]
                ),
                "device": str(payload["device"]),
                "normalization_report_sha256": file_sha256(normalization_path),
                "run_dir": str(path.parent.resolve()),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["lookback", "seed"]).reset_index(drop=True)
    invalid = sorted(set(frame["lookback"]) - set(EXPECTED_LOOKBACKS))
    if invalid:
        raise ValueError(f"Unexpected lookback values: {invalid}")
    if frame.duplicated(["lookback", "seed"]).any():
        raise ValueError("Duplicate (lookback, seed) results were found")
    return frame


def sample_std(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=np.float64)
    return float(array.std(ddof=1)) if len(array) > 1 else 0.0


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for lookback, part in runs.groupby("lookback", sort=True):
        rapid_mean = float(part["best_rapid_validation_rmse_k"].mean())
        full_mean = float(part["best_validation_rmse_k"].mean())
        rapid_persistence_mean = float(
            part["rapid_persistence_validation_rmse_k"].mean()
        )
        persistence_mean = float(part["persistence_validation_rmse_k"].mean())
        rows.append(
            {
                "lookback": int(lookback),
                "history_seconds": int(lookback) * 10,
                "history_minutes": int(lookback) / 6.0,
                "seed_count": int(part["seed"].nunique()),
                "seeds": ";".join(map(str, sorted(part["seed"].unique()))),
                "full_validation_windows_min": int(
                    part["full_validation_windows"].min()
                ),
                "full_validation_windows_max": int(
                    part["full_validation_windows"].max()
                ),
                "rapid_validation_windows_min": int(
                    part["rapid_validation_windows"].min()
                ),
                "rapid_validation_windows_max": int(
                    part["rapid_validation_windows"].max()
                ),
                "train_rmse_mean_k": float(part["best_train_rmse_k"].mean()),
                "train_rmse_std_k": sample_std(part["best_train_rmse_k"]),
                "train_rapid_rmse_mean_k": float(
                    part["best_train_rapid_rmse_k"].mean()
                ),
                "train_rapid_rmse_std_k": sample_std(
                    part["best_train_rapid_rmse_k"]
                ),
                "validation_rmse_mean_k": full_mean,
                "validation_rmse_std_k": sample_std(
                    part["best_validation_rmse_k"]
                ),
                "rapid_validation_rmse_mean_k": rapid_mean,
                "rapid_validation_rmse_std_k": sample_std(
                    part["best_rapid_validation_rmse_k"]
                ),
                "validation_mae_mean_k": float(
                    part["best_validation_mae_k"].mean()
                ),
                "rapid_validation_mae_mean_k": float(
                    part["best_rapid_validation_mae_k"].mean()
                ),
                "rapid_validation_p95_absolute_error_mean_k": float(
                    part["best_rapid_validation_p95_absolute_error_k"].mean()
                ),
                "rapid_validation_max_absolute_error_mean_k": float(
                    part["best_rapid_validation_max_absolute_error_k"].mean()
                ),
                "validation_direction_accuracy_mean": float(
                    part["validation_direction_accuracy"].mean()
                ),
                "rapid_direction_accuracy_mean": float(
                    part["rapid_validation_direction_accuracy"].mean()
                ),
                "test_window_start_source_row": int(
                    part["test_window_start_source_row"].iloc[0]
                ),
                "test_window_end_source_row": int(
                    part["test_window_end_source_row"].iloc[0]
                ),
                "test_window_valid_origins_min": int(
                    part["test_window_valid_origins"].min()
                ),
                "test_window_valid_origins_max": int(
                    part["test_window_valid_origins"].max()
                ),
                "test_window_rmse_mean_k": float(
                    part["test_window_rmse_k"].mean()
                ),
                "test_window_rmse_std_k": sample_std(
                    part["test_window_rmse_k"]
                ),
                "test_window_mae_mean_k": float(
                    part["test_window_mae_k"].mean()
                ),
                "test_window_p95_absolute_error_mean_k": float(
                    part["test_window_p95_absolute_error_k"].mean()
                ),
                "test_window_max_absolute_error_mean_k": float(
                    part["test_window_max_absolute_error_k"].mean()
                ),
                "test_window_rapid_origins_min": int(
                    part["test_window_rapid_origins"].min()
                ),
                "test_window_rapid_origins_max": int(
                    part["test_window_rapid_origins"].max()
                ),
                "test_window_rapid_rmse_mean_k": float(
                    part["test_window_rapid_rmse_k"].mean()
                ),
                "test_window_rapid_rmse_std_k": sample_std(
                    part["test_window_rapid_rmse_k"]
                ),
                "test_window_persistence_rmse_mean_k": float(
                    part["test_window_persistence_rmse_k"].mean()
                ),
                "persistence_rmse_mean_k": persistence_mean,
                "rapid_persistence_rmse_mean_k": rapid_persistence_mean,
                "validation_rmse_reduction_vs_persistence_pct": 100.0
                * (1.0 - full_mean / persistence_mean),
                "rapid_rmse_reduction_vs_persistence_pct": 100.0
                * (1.0 - rapid_mean / rapid_persistence_mean),
                "best_epoch_mean": float(part["best_epoch"].mean()),
                "best_epoch_std": sample_std(part["best_epoch"]),
                "seconds_per_epoch_mean": float(
                    part["seconds_per_completed_epoch"].mean()
                ),
                "trainable_parameters": int(part["trainable_parameters"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("lookback").reset_index(drop=True)


def collect_epoch_histories(runs: pd.DataFrame) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for run in runs.itertuples(index=False):
        path = Path(run.run_dir) / "training_history.csv"
        table = pd.read_csv(path)
        table.insert(0, "model", "gru")
        table.insert(1, "lookback", int(run.lookback))
        table.insert(2, "seed", int(run.seed))
        table["is_best_checkpoint_epoch"] = (
            table["epoch"].astype(int) == int(run.best_epoch)
        )
        tables.append(table)
    return pd.concat(tables, ignore_index=True).sort_values(
        ["lookback", "seed", "epoch"]
    )


def paired_vs_best(runs: pd.DataFrame, best_lookback: int) -> pd.DataFrame:
    best = runs.loc[runs["lookback"] == best_lookback, [
        "seed",
        "best_rapid_validation_rmse_k",
        "best_validation_rmse_k",
    ]].rename(
        columns={
            "best_rapid_validation_rmse_k": "best_lookback_rapid_rmse_k",
            "best_validation_rmse_k": "best_lookback_full_rmse_k",
        }
    )
    rows: list[dict[str, float | int | str]] = []
    for lookback, part in runs.groupby("lookback", sort=True):
        merged = part.merge(best, on="seed", how="inner")
        rapid_difference = (
            merged["best_rapid_validation_rmse_k"]
            - merged["best_lookback_rapid_rmse_k"]
        )
        full_difference = (
            merged["best_validation_rmse_k"] - merged["best_lookback_full_rmse_k"]
        )
        rows.append(
            {
                "candidate_lookback": int(lookback),
                "reference_best_lookback": int(best_lookback),
                "paired_seed_count": int(len(merged)),
                "paired_seeds": ";".join(map(str, sorted(merged["seed"].unique()))),
                "candidate_minus_best_rapid_rmse_mean_k": float(
                    rapid_difference.mean()
                ),
                "candidate_minus_best_rapid_rmse_std_k": sample_std(
                    rapid_difference
                ),
                "best_lookback_rapid_wins": int((rapid_difference > 0).sum()),
                "candidate_minus_best_full_rmse_mean_k": float(
                    full_difference.mean()
                ),
                "candidate_minus_best_full_rmse_std_k": sample_std(full_difference),
                "best_lookback_full_wins": int((full_difference > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def save_plot(
    summary: pd.DataFrame,
    path: Path,
    predict_steps: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].errorbar(
        summary["lookback"],
        summary["rapid_validation_rmse_mean_k"],
        yerr=summary["rapid_validation_rmse_std_k"],
        marker="o",
        capsize=4,
        linewidth=1.5,
        color="#D55E00",
        label="260501 rapid validation RMSE (mean +/- seed std)",
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
        label="260501 full validation RMSE (mean +/- seed std)",
    )
    axes[1].plot(
        summary["lookback"],
        summary["persistence_rmse_mean_k"],
        linestyle="--",
        color="#777777",
        label="Full persistence RMSE",
    )
    axes[1].set_ylabel("Full validation RMSE (K)")
    axes[1].set_title("Overall-accuracy guardrail")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].errorbar(
        summary["lookback"],
        summary["test_window_rmse_mean_k"],
        yerr=summary["test_window_rmse_std_k"],
        marker="o",
        capsize=4,
        linewidth=1.5,
        color="#009E73",
        label="0715-BACK source rows 0--7300 RMSE (diagnostic only)",
    )
    axes[2].plot(
        summary["lookback"],
        summary["test_window_persistence_rmse_mean_k"],
        linestyle="--",
        color="#777777",
        label="Fixed-window persistence RMSE",
    )
    axes[2].set_xticks(EXPECTED_LOOKBACKS)
    axes[2].set_xlabel("Lookback (samples, 10 s/sample)")
    axes[2].set_ylabel("Fixed test-window RMSE (K)")
    axes[2].set_title("Secondary test diagnostic; excluded from model selection")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    fig.suptitle(
        f"Delta Thv(t+{predict_steps}) GRU lookback scan: "
        "validation 260501, test 0715-BACK"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    expected_seeds = parse_seed_set(args.expected_seeds)
    horizon_dir = args.output_dir / f"horizon_{args.predict_steps:02d}"
    all_runs = load_runs(horizon_dir, args.predict_steps)
    formal_runs = all_runs.loc[all_runs["seed"].isin(expected_seeds)].copy()

    expected_pairs = {
        (lookback, seed)
        for lookback in EXPECTED_LOOKBACKS
        for seed in expected_seeds
    }
    actual_pairs = set(
        map(tuple, formal_runs[["lookback", "seed"]].to_numpy(dtype=int))
    )
    missing_pairs = sorted(expected_pairs - actual_pairs)
    formal_selection_ready = not missing_pairs
    selection_runs = formal_runs if formal_selection_ready else all_runs
    selection_status = "formal" if formal_selection_ready else "provisional_incomplete"

    summary = aggregate(selection_runs)
    best = summary.sort_values(
        [
            "rapid_validation_rmse_mean_k",
            "validation_rmse_mean_k",
            "rapid_validation_rmse_std_k",
        ]
    ).iloc[0]
    best_lookback = int(best["lookback"])
    paired = paired_vs_best(selection_runs, best_lookback)
    epochs = collect_epoch_histories(all_runs)

    horizon_dir.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(horizon_dir / "gru_lookback_all_runs.csv", index=False)
    summary.to_csv(horizon_dir / "gru_lookback_summary.csv", index=False)
    paired.to_csv(horizon_dir / "gru_lookback_paired_vs_best.csv", index=False)
    epochs.to_csv(horizon_dir / "gru_epoch_history_all_runs.csv", index=False)
    save_plot(
        summary,
        horizon_dir / "gru_lookback_rmse_comparison.png",
        args.predict_steps,
    )

    evidence_columns = [
        "lookback",
        "history_seconds",
        "history_minutes",
        "seed_count",
        "rapid_validation_windows_min",
        "rapid_validation_windows_max",
        "rapid_validation_rmse_mean_k",
        "rapid_validation_rmse_std_k",
        "validation_rmse_mean_k",
        "validation_rmse_std_k",
        "rapid_validation_mae_mean_k",
        "rapid_validation_p95_absolute_error_mean_k",
        "rapid_validation_max_absolute_error_mean_k",
        "rapid_direction_accuracy_mean",
        "rapid_rmse_reduction_vs_persistence_pct",
        "validation_rmse_reduction_vs_persistence_pct",
        "test_window_start_source_row",
        "test_window_end_source_row",
        "test_window_valid_origins_min",
        "test_window_valid_origins_max",
        "test_window_rmse_mean_k",
        "test_window_rmse_std_k",
        "test_window_mae_mean_k",
        "test_window_p95_absolute_error_mean_k",
        "test_window_max_absolute_error_mean_k",
        "test_window_rapid_origins_min",
        "test_window_rapid_origins_max",
        "test_window_rapid_rmse_mean_k",
        "test_window_rapid_rmse_std_k",
        "best_epoch_mean",
    ]
    summary[evidence_columns].to_csv(
        horizon_dir / "paper_evidence_lookback_table.csv", index=False
    )

    normalization_consistent = (
        all_runs["normalization_report_sha256"].nunique() == 1
    )
    print(summary.to_string(index=False))
    print(f"SELECTION_STATUS={selection_status}")
    print(f"FORMAL_SELECTION_READY={str(formal_selection_ready).lower()}")
    print(f"MISSING_RUN_COUNT={len(missing_pairs)}")
    if missing_pairs:
        print(
            "MISSING_LOOKBACK_SEED_PAIRS="
            + ";".join(f"{lookback}:{seed}" for lookback, seed in missing_pairs)
        )
    print(f"NORMALIZATION_REPORTS_IDENTICAL={str(normalization_consistent).lower()}")
    print("PRIMARY_METRIC=rapid_cooling_RMSE_on_Original_260501_validation")
    print(f"PREDICT_STEPS={args.predict_steps}")
    print(
        "PRIMARY_RAPID_WINDOW_COUNT_RANGE="
        f"{int(summary['rapid_validation_windows_min'].min())}.."
        f"{int(summary['rapid_validation_windows_max'].max())}"
    )
    print(
        "SECONDARY_TEST_DIAGNOSTIC="
        "Original_0715-BACK_source_rows_0..7300"
    )
    print("TEST_DIAGNOSTIC_USED_FOR_LOOKBACK_RANKING=false")
    print(f"BEST_LOOKBACK_BY_RAPID_VALIDATION_RMSE={best_lookback}")
    print(
        "BEST_MEAN_RAPID_VALIDATION_RMSE_K="
        f"{best['rapid_validation_rmse_mean_k']:.10f}"
    )
    print(
        "BEST_LOOKBACK_FULL_VALIDATION_RMSE_K="
        f"{best['validation_rmse_mean_k']:.10f}"
    )
    print(
        "BEST_LOOKBACK_TEST_WINDOW_RMSE_K="
        f"{best['test_window_rmse_mean_k']:.10f}"
    )
    print(
        "BEST_LOOKBACK_TEST_WINDOW_RAPID_RMSE_K="
        f"{best['test_window_rapid_rmse_mean_k']:.10f}"
    )
    print(f"SUMMARY_SAVED={horizon_dir.resolve()}")


if __name__ == "__main__":
    main()
