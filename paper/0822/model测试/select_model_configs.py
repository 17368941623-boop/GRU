#!/usr/bin/env python3
"""Select one hyperparameter configuration per model using validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only model configuration selection."
    )
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument(
        "--common-origin-lookback",
        type=int,
        default=None,
        help="Defaults to --lookback and must match every development run.",
    )
    parser.add_argument("--predict-steps", type=int, default=15)
    parser.add_argument("--search-seeds", default="42,62")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(sorted({int(item.strip()) for item in text.split(",") if item.strip()}))
    if not seeds:
        raise ValueError("At least one search seed is required")
    return seeds


def main() -> None:
    args = parse_args()
    if args.common_origin_lookback is None:
        args.common_origin_lookback = args.lookback
    if args.common_origin_lookback < args.lookback:
        raise ValueError("common-origin-lookback cannot be smaller than lookback")
    search_seeds = parse_seeds(args.search_seeds)
    study_dir = (
        args.results_dir
        / "development"
        / f"horizon_{args.predict_steps:02d}"
        / f"lookback_{args.lookback:02d}"
    )
    manifest_path = study_dir / "search_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing search manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    expected_configs = set(
        map(tuple, manifest[["model", "config_id"]].drop_duplicates().to_numpy())
    )
    manifest_seeds = set(manifest["seed"].astype(int))
    if manifest_seeds != set(search_seeds):
        raise ValueError(
            "Requested search seeds do not match search_manifest.csv: "
            f"requested={list(search_seeds)}, manifest={sorted(manifest_seeds)}"
        )
    paths = sorted(study_dir.glob("*/*/seed_*/metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No development metrics under {study_dir}")

    rows: list[dict[str, object]] = []
    payload_by_identity: dict[tuple[str, str, int], dict[str, object]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if bool(payload.get("test_data_loaded", True)):
            raise ValueError(f"Development run opened test data: {path}")
        if int(payload["predict_steps"]) != args.predict_steps:
            raise ValueError(f"Forecast horizon mismatch: {path}")
        if int(payload["lookback"]) != args.lookback:
            raise ValueError(f"Lookback mismatch: {path}")
        if int(payload.get("common_origin_lookback", -1)) != args.common_origin_lookback:
            raise ValueError(f"Common-origin lookback mismatch: {path}")
        seed = int(payload["seed"])
        if seed not in search_seeds:
            continue
        model = str(payload["model_type"])
        config_id = str(payload["config_id"])
        if (model, config_id) not in expected_configs:
            # Ignore stale configurations from a previous quick/full profile.
            continue
        identity = (model, config_id, seed)
        if identity in payload_by_identity:
            raise ValueError(f"Duplicate development result: {identity}")
        payload_by_identity[identity] = payload
        rows.append(
            {
                "model": model,
                "config_id": config_id,
                "seed": seed,
                "common_origin_lookback": int(payload["common_origin_lookback"]),
                "rapid_validation_rmse_k": float(
                    payload["rapid_validation_metrics"]["rmse_k"]
                ),
                "validation_rmse_k": float(payload["validation_metrics"]["rmse_k"]),
                "rapid_validation_mae_k": float(
                    payload["rapid_validation_metrics"]["mae_k"]
                ),
                "validation_mae_k": float(payload["validation_metrics"]["mae_k"]),
                "rapid_direction_accuracy": float(
                    payload["rapid_validation_direction_accuracy"]
                ),
                "trainable_parameters": int(payload["trainable_parameters"]),
                "training_seconds": float(payload["training_seconds_total"]),
                "best_epoch": int(payload["best_epoch"]),
                "metrics_path": str(path),
            }
        )
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise ValueError("No runs match the requested search seeds")
    actual_configs = set(
        map(tuple, runs[["model", "config_id"]].drop_duplicates().to_numpy())
    )
    if actual_configs != expected_configs:
        missing = sorted(expected_configs - actual_configs)
        raise ValueError(f"Search manifest has unfinished configurations: {missing}")

    expected_seed_set = set(search_seeds)
    for (model, config_id), part in runs.groupby(["model", "config_id"]):
        actual = set(part["seed"].astype(int))
        if actual != expected_seed_set:
            raise ValueError(
                f"Incomplete search seeds for {model}/{config_id}: "
                f"expected={sorted(expected_seed_set)}, actual={sorted(actual)}"
            )

    summary = (
        runs.groupby(["model", "config_id"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            rapid_validation_rmse_mean_k=("rapid_validation_rmse_k", "mean"),
            rapid_validation_rmse_std_k=("rapid_validation_rmse_k", "std"),
            validation_rmse_mean_k=("validation_rmse_k", "mean"),
            validation_rmse_std_k=("validation_rmse_k", "std"),
            rapid_validation_mae_mean_k=("rapid_validation_mae_k", "mean"),
            validation_mae_mean_k=("validation_mae_k", "mean"),
            rapid_direction_accuracy_mean=("rapid_direction_accuracy", "mean"),
            trainable_parameters=("trainable_parameters", "first"),
            training_seconds_mean=("training_seconds", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .sort_values(
            [
                "model",
                "rapid_validation_rmse_mean_k",
                "validation_rmse_mean_k",
                "rapid_validation_rmse_std_k",
                "trainable_parameters",
            ]
        )
        .reset_index(drop=True)
    )
    summary["rank_within_model"] = (
        summary.groupby("model").cumcount() + 1
    )
    summary.to_csv(study_dir / "development_config_summary.csv", index=False)
    runs.to_csv(study_dir / "development_all_runs.csv", index=False)

    selected_rows = summary.loc[summary["rank_within_model"] == 1].copy()
    selected: dict[str, object] = {
        "selection_status": "selected_without_opening_test_data",
        "selection_split": "Original 260501 validation only",
        "primary_metric": "mean rapid_validation_rmse_k across search seeds",
        "tie_breakers": [
            "mean validation_rmse_k",
            "rapid_validation_rmse_std_k",
            "trainable_parameters",
        ],
        "lookback": args.lookback,
        "common_origin_lookback": args.common_origin_lookback,
        "predict_steps": args.predict_steps,
        "search_seeds": list(search_seeds),
        "test_data_read": False,
        "models": {},
    }
    for row in selected_rows.itertuples(index=False):
        model = str(row.model)
        config_id = str(row.config_id)
        seed_payload = payload_by_identity[(model, config_id, search_seeds[0])]
        selected["models"][model] = {
            "config_id": config_id,
            "model_config": seed_payload["model_config"],
            "rapid_validation_rmse_mean_k": float(
                row.rapid_validation_rmse_mean_k
            ),
            "validation_rmse_mean_k": float(row.validation_rmse_mean_k),
            "rapid_validation_rmse_std_k": float(
                0.0
                if pd.isna(row.rapid_validation_rmse_std_k)
                else row.rapid_validation_rmse_std_k
            ),
            "trainable_parameters": int(row.trainable_parameters),
        }
    selected_path = study_dir / "selected_configs.json"
    selected_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(max(10, len(selected_rows) * 1.35), 5.5))
    labels = [
        f"{row.model}\n{row.config_id}" for row in selected_rows.itertuples(index=False)
    ]
    axis.bar(
        labels,
        selected_rows["rapid_validation_rmse_mean_k"],
        yerr=selected_rows["rapid_validation_rmse_std_k"].fillna(0.0),
        capsize=4,
        color="#2a9d8f",
        alpha=0.88,
    )
    axis.set_ylabel("Rapid-validation RMSE (K)")
    axis.set_title("Selected configuration per model — Original 260501 only")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(study_dir / "selected_config_validation_comparison.png", dpi=220)
    plt.close(figure)

    print(selected_rows.to_string(index=False))
    print("TEST_DATA_READ=false")
    print(f"SELECTED_CONFIGS={selected_path}")


if __name__ == "__main__":
    main()
