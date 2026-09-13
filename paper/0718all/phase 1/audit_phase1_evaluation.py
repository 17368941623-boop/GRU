"""Inference-only Phase 1 audit; never trains a model or reads a test split.

The audit evaluates saved validation-only checkpoints on:

1. the existing event-enriched validation dataframe; and
2. the complete, non-augmented Original 0501 cooling curve.

It reports persistence, aggregate errors, retrospective true-temperature-change
subsets, and strictly causal subsets defined only by six historical valve
signals in the 60-step input window.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_fil import TRAIN_DATA_FOLDER, clean_and_extract  # noqa: E402
from phase1_common import (  # noqa: E402
    DATA_DIR,
    DELTA_COL,
    LABEL_COL,
    LOOK_BACK,
    OUTPUT_DIR,
    PREDICT_STEPS,
    TARGET_COL,
    PlainGRU,
    WindowDataset,
    add_rebuilt_physics_features,
    add_target_delta,
    choose_device,
    find_raw_signal_column,
    load_frame,
    validate_feature_schema,
)


ORIGINAL_VALIDATION_CACHE = DATA_DIR / "val_original_0501_full_clean.pkl"
AUDIT_OUTPUT_DIR = OUTPUT_DIR / "phase1_evaluation_audit"
VALVE_KEYS = ("CV8300", "CV8313", "CV8310", "CV8311", "CV8312", "CV8351")
DELTA_BINS = (
    ("delta_[0,0.5)", 0.0, 0.5),
    ("delta_[0.5,1)", 0.5, 1.0),
    ("delta_[1,2)", 1.0, 2.0),
    ("delta_[2,5)", 2.0, 5.0),
    ("delta_[5,10)", 5.0, 10.0),
    ("delta_[10,inf)", 10.0, math.inf),
)
DYNAMIC_THRESHOLDS = (0.5, 1.0, 2.0, 5.0)
PLOT_SUBSETS = (
    "overall",
    "delta_>0.5",
    "delta_>1",
    "delta_>2",
    "delta_>5",
    "history_event_any",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit saved Phase 1 checkpoints")
    parser.add_argument("--experiment", default="selected_plus_rebuilt_physics")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 62, 82])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--valve-threshold",
        type=float,
        default=0.05,
        help="Absolute valve-opening change that defines a historical action event.",
    )
    parser.add_argument(
        "--rebuild-original-validation",
        action="store_true",
        help="Re-run causal preprocessing for Original 0501 and replace only its audit cache.",
    )
    parser.add_argument(
        "--skip-original-validation",
        action="store_true",
        help="Audit only the existing event-enriched validation dataframe.",
    )
    return parser.parse_args()


def load_or_build_original_validation(rebuild: bool) -> pd.DataFrame:
    if ORIGINAL_VALIDATION_CACHE.exists() and not rebuild:
        frame = joblib.load(ORIGINAL_VALIDATION_CACHE)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Audit cache is not a dataframe: {ORIGINAL_VALIDATION_CACHE}")
        return frame.copy()

    candidates = sorted(
        path
        for path in TRAIN_DATA_FOLDER.glob("*.csv")
        if path.name.lower().startswith("original_") and "0501" in path.name
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one Original 0501 CSV below {TRAIN_DATA_FOLDER}, found {candidates}"
        )
    frame = clean_and_extract([candidates[0].name], TRAIN_DATA_FOLDER, "Original0501Audit")
    if frame.empty:
        raise ValueError("Causal preprocessing produced an empty Original 0501 validation frame")
    joblib.dump(frame, ORIGINAL_VALIDATION_CACHE)
    print(f"Saved Original 0501 audit cache to {ORIGINAL_VALIDATION_CACHE}")
    return frame.copy()


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = add_target_delta(add_rebuilt_physics_features(frame.copy()))
    if prepared[LABEL_COL].isna().any():
        raise ValueError("Validation audit frame contains a missing future label")
    return prepared


def trusted_torch_load(path: Path, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_saved_model(run_dir: Path, device: torch.device):
    checkpoint_path = run_dir / "model.pt"
    scaler_x_path = run_dir / "scaler_x.pkl"
    scaler_y_path = run_dir / "scaler_y.pkl"
    for path in (checkpoint_path, scaler_x_path, scaler_y_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing saved Phase 1 artifact: {path}")

    checkpoint = trusted_torch_load(checkpoint_path, map_location=torch.device("cpu"))
    if checkpoint.get("test_evaluated"):
        raise ValueError(f"Audit expects a validation-only checkpoint: {checkpoint_path}")
    features = list(checkpoint["feature_columns"])
    state = checkpoint["model_state_dict"]
    hidden_size = int(state["gru1.weight_hh_l0"].shape[1])
    model = PlainGRU(len(features), hidden_size, dropout=0.0)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, features, joblib.load(scaler_x_path), joblib.load(scaler_y_path), checkpoint


def predict_frame(
    frame: pd.DataFrame,
    model: PlainGRU,
    features: list[str],
    scaler_x,
    scaler_y,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    validate_feature_schema(frame, features, "audit validation")
    x_scaled = scaler_x.transform(frame[features]).astype(np.float32)
    y_scaled = scaler_y.transform(frame[[DELTA_COL]]).astype(np.float32)
    dataset = WindowDataset(frame, x_scaled, y_scaled)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    predicted_scaled, current, actual = [], [], []
    with torch.no_grad():
        for x, _, current_temp, future_temp in loader:
            predicted_scaled.append(model(x.to(device)).cpu().numpy())
            current.append(current_temp.numpy())
            actual.append(future_temp.numpy())
    predicted_delta = scaler_y.inverse_transform(
        np.concatenate(predicted_scaled).reshape(-1, 1)
    ).reshape(-1)
    current_arr = np.concatenate(current).astype(float)
    actual_arr = np.concatenate(actual).astype(float)
    prediction_arr = current_arr + predicted_delta
    return prediction_arr, actual_arr, current_arr, predicted_delta, dataset.ends.copy()


def historical_valve_context(
    frame: pd.DataFrame,
    ends: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    valve_columns = [find_raw_signal_column(frame, key) for key in VALVE_KEYS]
    valves = frame[valve_columns].apply(pd.to_numeric, errors="coerce")
    valve_change = valves.groupby(frame["file_id"], sort=False).diff().abs()
    row_event = valve_change.gt(threshold).any(axis=1).to_numpy(bool)

    steps_since_event = np.full(len(frame), np.inf, dtype=float)
    for _, indices in frame.groupby("file_id", sort=False).indices.items():
        ordered = np.asarray(indices, dtype=int)
        last_event_position: int | None = None
        for position in ordered:
            if row_event[position]:
                last_event_position = position
            if last_event_position is not None:
                steps_since_event[position] = position - last_event_position

    origin_steps = steps_since_event[ends]
    has_historical_event = origin_steps <= LOOK_BACK - 1
    return has_historical_event, origin_steps


def subset_masks(
    true_delta: np.ndarray,
    has_historical_event: np.ndarray,
    steps_since_event: np.ndarray,
) -> dict[str, np.ndarray]:
    abs_delta = np.abs(true_delta)
    masks: dict[str, np.ndarray] = {"overall": np.ones(len(true_delta), dtype=bool)}
    for name, lower, upper in DELTA_BINS:
        masks[name] = (abs_delta >= lower) & (abs_delta < upper)
    for threshold in DYNAMIC_THRESHOLDS:
        masks[f"delta_>{threshold:g}"] = abs_delta > threshold
    masks["history_event_any"] = has_historical_event
    masks["history_event_none"] = ~has_historical_event
    masks["after_event_0_10"] = steps_since_event <= 10
    masks["after_event_11_30"] = (steps_since_event >= 11) & (steps_since_event <= 30)
    masks["after_event_31_59"] = (steps_since_event >= 31) & (steps_since_event <= 59)
    return masks


def metrics(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict:
    count = int(mask.sum())
    if count == 0:
        return {
            "n_windows": 0,
            "mae_k": None,
            "rmse_k": None,
            "r2": None,
            "bias_k": None,
            "p95_absolute_error_k": None,
            "max_absolute_error_k": None,
        }
    actual_subset = actual[mask]
    predicted_subset = predicted[mask]
    error = predicted_subset - actual_subset
    denominator = float(np.sum((actual_subset - np.mean(actual_subset)) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 0 else None
    return {
        "n_windows": count,
        "mae_k": float(np.mean(np.abs(error))),
        "rmse_k": float(np.sqrt(np.mean(error**2))),
        "r2": r2,
        "bias_k": float(np.mean(error)),
        "p95_absolute_error_k": float(np.percentile(np.abs(error), 95)),
        "max_absolute_error_k": float(np.max(np.abs(error))),
    }


def append_metric_rows(
    rows: list[dict],
    dataset_name: str,
    predictor: str,
    seed: int | None,
    actual: np.ndarray,
    predicted: np.ndarray,
    masks: dict[str, np.ndarray],
    persistence_metrics: dict[str, dict] | None = None,
) -> None:
    for subset, mask in masks.items():
        result = metrics(actual, predicted, mask)
        skill = None
        if predictor == "model" and result["mae_k"] is not None and persistence_metrics is not None:
            baseline_mae = persistence_metrics[subset]["mae_k"]
            if baseline_mae not in (None, 0.0):
                skill = 1.0 - result["mae_k"] / baseline_mae
        rows.append(
            {
                "dataset": dataset_name,
                "predictor": predictor,
                "seed": seed,
                "subset": subset,
                **result,
                "mae_skill_vs_persistence": skill,
            }
        )


def build_summary(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    summary_rows = []
    metric_names = (
        "mae_k",
        "rmse_k",
        "r2",
        "bias_k",
        "p95_absolute_error_k",
        "max_absolute_error_k",
        "mae_skill_vs_persistence",
    )
    for (dataset, subset), part in frame.groupby(["dataset", "subset"], sort=False):
        persistence = part[part["predictor"] == "persistence"].iloc[0]
        models = part[part["predictor"] == "model"]
        row = {
            "dataset": dataset,
            "subset": subset,
            "n_windows": int(persistence["n_windows"]),
            "model_seed_count": int(len(models)),
        }
        for name in metric_names[:-1]:
            row[f"persistence_{name}"] = persistence[name]
            values = pd.to_numeric(models[name], errors="coerce").dropna().to_numpy(float)
            row[f"model_{name}_mean"] = float(np.mean(values)) if len(values) else None
            row[f"model_{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        skill_values = pd.to_numeric(
            models["mae_skill_vs_persistence"], errors="coerce"
        ).dropna().to_numpy(float)
        row["model_mae_skill_mean"] = float(np.mean(skill_values)) if len(skill_values) else None
        row["model_mae_skill_std"] = (
            float(np.std(skill_values, ddof=1)) if len(skill_values) > 1 else 0.0
        )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def save_comparison_plot(summary: pd.DataFrame, dataset_name: str, path: Path) -> None:
    part = summary[summary["dataset"] == dataset_name].set_index("subset")
    subsets = [name for name in PLOT_SUBSETS if name in part.index]
    persistence = [part.loc[name, "persistence_mae_k"] for name in subsets]
    model_mean = [part.loc[name, "model_mae_k_mean"] for name in subsets]
    model_std = [part.loc[name, "model_mae_k_std"] for name in subsets]
    x = np.arange(len(subsets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, persistence, width, label="Persistence", color="#777777")
    ax.bar(
        x + width / 2,
        model_mean,
        width,
        yerr=model_std,
        capsize=4,
        label="Phase 1 model mean +/- SD",
        color="#0072B2",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, rotation=25, ha="right")
    ax.set_ylabel("MAE (K)")
    ax.set_title(f"Phase 1 audit | {dataset_name}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def prediction_frame(
    frame: pd.DataFrame,
    ends: np.ndarray,
    current: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    predicted_delta: np.ndarray,
    has_event: np.ndarray,
    steps_since_event: np.ndarray,
) -> pd.DataFrame:
    origins = frame.iloc[ends]
    return pd.DataFrame(
        {
            "origin_row": ends,
            "file_id": origins["file_id"].astype(str).to_numpy(),
            "source_group": origins["source_group"].astype(str).to_numpy(),
            "current_temperature_k": current,
            "actual_temperature_k": actual,
            "predicted_temperature_k": predicted,
            "predicted_delta_k": predicted_delta,
            "true_delta_k": actual - current,
            "historical_valve_event": has_event.astype(int),
            "steps_since_last_valve_event": steps_since_event,
        }
    )


def main() -> None:
    args = parse_args()
    device = choose_device()
    print(f"Phase 1 audit device={device}; experiment={args.experiment}; seeds={args.seeds}")
    datasets = {"event_enriched_0501": prepare_frame(load_frame("val_clean.pkl"))}
    if not args.skip_original_validation:
        original = load_or_build_original_validation(args.rebuild_original_validation)
        datasets["original_0501_full"] = prepare_frame(original)

    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    reference_features: list[str] | None = None

    for dataset_name, frame in datasets.items():
        print(f"\nAuditing {dataset_name}: rows={len(frame):,}")
        dataset_output = AUDIT_OUTPUT_DIR / args.experiment / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)
        reference_actual = reference_current = reference_ends = None
        masks = persistence_by_subset = None

        for seed in args.seeds:
            run_dir = OUTPUT_DIR / args.experiment / "validation_only" / f"seed_{seed}"
            model, features, scaler_x, scaler_y, checkpoint = load_saved_model(run_dir, device)
            if checkpoint.get("experiment") != args.experiment:
                raise ValueError(f"Checkpoint experiment mismatch: {run_dir}")
            if reference_features is None:
                reference_features = features
            elif features != reference_features:
                raise ValueError(f"Feature columns differ across seeds: {run_dir}")

            predicted, actual, current, predicted_delta, ends = predict_frame(
                frame,
                model,
                features,
                scaler_x,
                scaler_y,
                device,
                args.batch_size,
                args.num_workers,
            )
            has_event, steps_since_event = historical_valve_context(
                frame, ends, args.valve_threshold
            )
            if reference_actual is None:
                reference_actual = actual
                reference_current = current
                reference_ends = ends
                true_delta = actual - current
                masks = subset_masks(true_delta, has_event, steps_since_event)
                persistence_by_subset = {
                    name: metrics(actual, current, mask) for name, mask in masks.items()
                }
                append_metric_rows(
                    all_rows,
                    dataset_name,
                    "persistence",
                    None,
                    actual,
                    current,
                    masks,
                )
            else:
                if not np.array_equal(ends, reference_ends):
                    raise ValueError("Window origins differ across saved seeds")
                if not np.allclose(actual, reference_actual) or not np.allclose(current, reference_current):
                    raise ValueError("Validation targets differ across saved seeds")

            assert masks is not None and persistence_by_subset is not None
            append_metric_rows(
                all_rows,
                dataset_name,
                "model",
                seed,
                actual,
                predicted,
                masks,
                persistence_by_subset,
            )
            prediction_frame(
                frame,
                ends,
                current,
                actual,
                predicted,
                predicted_delta,
                has_event,
                steps_since_event,
            ).to_csv(dataset_output / f"predictions_seed_{seed}.csv", index=False)
            overall = metrics(actual, predicted, masks["overall"])
            print(
                f"  seed={seed}: MAE={overall['mae_k']:.6f}, "
                f"RMSE={overall['rmse_k']:.6f}, P95={overall['p95_absolute_error_k']:.6f}"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()

    long_frame = pd.DataFrame(all_rows)
    summary = build_summary(all_rows)
    audit_root = AUDIT_OUTPUT_DIR / args.experiment
    long_frame.to_csv(audit_root / "audit_metrics_long.csv", index=False)
    summary.to_csv(audit_root / "audit_summary.csv", index=False)
    payload = {
        "experiment": args.experiment,
        "seeds": args.seeds,
        "prediction_steps": PREDICT_STEPS,
        "look_back": LOOK_BACK,
        "valve_keys": list(VALVE_KEYS),
        "valve_threshold": args.valve_threshold,
        "test_set_read": False,
        "feature_columns": reference_features,
        "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
    }
    (audit_root / "audit_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for dataset_name in datasets:
        save_comparison_plot(
            summary,
            dataset_name,
            audit_root / f"mae_comparison_{dataset_name}.png",
        )
    print("\nKey summary rows:")
    columns = [
        "dataset",
        "subset",
        "n_windows",
        "persistence_mae_k",
        "model_mae_k_mean",
        "model_mae_k_std",
        "model_mae_skill_mean",
    ]
    print(summary[summary["subset"].isin(PLOT_SUBSETS)][columns].to_string(index=False))
    print(f"\nAudit outputs saved to {audit_root}")


if __name__ == "__main__":
    main()
