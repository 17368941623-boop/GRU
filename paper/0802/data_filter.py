#!/usr/bin/env python3
"""Causal sensor filtering and filter-quality review for the raw 0802 dataset.

This script deliberately does NOT:
1. split train/validation/test data;
2. create future labels;
3. build model features;
4. modify source CSV files.

The default ``review`` mode only compares several causal candidates on the ten
raw, unaugmented CSV files and writes metrics/plots for manual inspection.  The
``apply`` mode must be requested explicitly and requires an explicit filter
choice for temperature and pressure/flow channels.

Pipeline position:
    raw CSV -> data_filter.py -> filtered_original_data -> create.py augmentation

Leakage controls:
- ordinary filtering candidates use only current and past samples;
- the optional short-drop flypoint rule uses an explicitly logged, bounded
  look-ahead of at most two samples and writes unsafe endpoint ranges that the
  downstream window builder must exclude;
- no backward fill, centered rolling window, Savitzky-Golay, or filtfilt;
- time columns and all CV valve columns are kept unchanged;
- a valve opening of 0% is always treated as a valid measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "300-4.5k original data"
DEFAULT_REVIEW_DIR = SCRIPT_DIR / "filter_review"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "filtered_original_data"

RAW_DATA_FILE_RE = re.compile(r"^(?!Original__|Aug_\d+__).+\.csv$", flags=re.IGNORECASE)
AUGMENTED_FILE_RE = re.compile(r"^(Original|Aug_\d+)__.+\.csv$", flags=re.IGNORECASE)
REPORT_FILE_NAMES = {
    "filter_manifest.csv",
    "temperature_flypoint_events.csv",
    "temperature_flypoint_endpoint_exclusions.csv",
    "augmentation_manifest.csv",
    "augmentation_parameters.csv",
    "augmentation_validation.csv",
}

# These six are retained here only to make dynamic-review plots readable.
# Every column whose signal name starts with CV is protected from filtering.
MAIN_VALVES = ("CV8300", "CV8313", "CV8310", "CV8311", "CV8312", "CV8351")
EXCLUDED_TEMPERATURES = {"A管", "B管"}

TEMPERATURE_FILTERS = (
    "causal_fill_only",
    "causal_median_3",
    "causal_ewma_20s",
)
PRESSURE_FLOW_FILTERS = (
    "causal_fill_only",
    "causal_median_3",
    "causal_ewma_20s",
    "causal_ewma_40s",
    "causal_ewma_60s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review or apply leakage-safe causal filters to raw, unaugmented 0802 CSV data. "
            "The default mode is review and never writes filtered training data."
        )
    )
    parser.add_argument("--mode", choices=("review", "apply"), default="review")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--temperature-filter",
        choices=TEMPERATURE_FILTERS,
        default=None,
        help="Required in apply mode; choose after reviewing the comparison plots.",
    )
    parser.add_argument(
        "--pressure-flow-filter",
        choices=PRESSURE_FLOW_FILTERS,
        default=None,
        help="Required in apply mode; choose after reviewing the comparison plots.",
    )
    parser.add_argument(
        "--max-ffill-steps",
        type=int,
        default=6,
        help=(
            "Maximum causal forward-fill length. Longer gaps and leading gaps stay NaN. "
            "Use 0 to disable forward fill. Default: 6 samples."
        ),
    )
    parser.add_argument(
        "--flypoint-drop-threshold-k",
        type=float,
        default=5.0,
        help=(
            "Start the temperature flypoint check when a new sample falls by more "
            "than this many kelvin from the last accepted sample. Default: 5 K."
        ),
    )
    parser.add_argument(
        "--flypoint-recovery-steps",
        type=int,
        default=2,
        help=(
            "Maximum following samples allowed for a guarded drop to recover. "
            "Repaired events are logged for downstream endpoint exclusion. Default: 2."
        ),
    )
    parser.add_argument(
        "--flypoint-recovery-tolerance-k",
        type=float,
        default=2.0,
        help=(
            "A guarded temperature is considered recovered when it returns to within "
            "this distance below the pre-drop value. Default: 2 K."
        ),
    )
    parser.add_argument(
        "--disable-temperature-flypoint-guard",
        action="store_true",
        help="Disable the bounded >5 K short-drop repair for temperature channels.",
    )
    parser.add_argument(
        "--event-windows",
        type=int,
        default=2,
        help="Number of strongest separated valve-event windows plotted per original file.",
    )
    parser.add_argument(
        "--event-radius",
        type=int,
        default=180,
        help="Samples before/after each plotted valve event. Default: 180.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Create review CSVs without PNG plots.",
    )
    parser.add_argument(
        "--confirm-apply",
        action="store_true",
        help="Required safeguard for apply mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing same-named files already present in output-dir.",
    )
    return parser.parse_args()


def read_industrial_csv(path: Path) -> pd.DataFrame:
    attempts = (
        ("utf-8-sig", ","),
        ("utf-8", ","),
        ("gb18030", ","),
        ("utf-16", "\t"),
        ("gb18030", "\t"),
    )
    errors: List[str] = []
    for encoding, separator in attempts:
        try:
            frame = pd.read_csv(path, encoding=encoding, sep=separator, low_memory=False)
            if len(frame.columns) > 1:
                frame.columns = frame.columns.astype(str).str.replace('"', "", regex=False).str.strip()
                return frame
        except Exception as exc:  # keep concise diagnostics for mixed industrial exports
            errors.append(f"{encoding}/{repr(separator)}: {exc}")
    raise ValueError(f"Unable to read {path}. Attempts: {' | '.join(errors)}")


def discover_data_files(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    augmented = sorted(
        path.name
        for path in input_dir.iterdir()
        if path.is_file() and AUGMENTED_FILE_RE.match(path.name)
    )
    if augmented:
        preview = ", ".join(augmented[:3])
        raise ValueError(
            "data_filter.py must run before augmentation, but input-dir contains "
            f"augmented/prefixed files ({preview}). Point --input-dir to the raw CSV folder."
        )

    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and RAW_DATA_FILE_RE.match(path.name)
        and path.name.lower() not in REPORT_FILE_NAMES
    )
    if not files:
        raise FileNotFoundError(f"No raw, unaugmented CSV data files found in {input_dir}")
    return files


def signal_name(column: str) -> str:
    name = str(column).strip()
    return name[: -len(" ValueY")].strip() if name.endswith(" ValueY") else name


def signal_kind(column: str) -> str:
    name = signal_name(column)
    upper = name.upper().replace(" ", "")
    if str(column).strip().endswith(" Time") or "TIME" in upper:
        return "time"
    if upper.startswith("CV"):
        return "valve"
    if name in EXCLUDED_TEMPERATURES:
        return "excluded_temperature"
    if upper.startswith("PT") or "VAC" in upper:
        return "pressure"
    if upper.startswith("FT"):
        return "flow"
    if upper.startswith("TE") or upper in {"THV", "DTBR", "TEF", "TCD"}:
        return "temperature"
    return "other"


def is_absolute_temperature(column: str) -> bool:
    upper = signal_name(column).upper().replace(" ", "")
    # DTbr may be a temperature difference, where zero can be physically valid.
    return upper.startswith("TE") or upper in {"THV", "TEF", "TCD"}


def infer_sample_period_seconds(frame: pd.DataFrame) -> float:
    for column in frame.columns:
        if signal_kind(column) != "time":
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce")
        valid = parsed.dropna()
        if len(valid) < 2:
            continue
        elapsed = (valid.iloc[-1] - valid.iloc[0]).total_seconds()
        period = elapsed / max(len(valid) - 1, 1)
        if math.isfinite(period) and 0.05 <= period <= 600.0:
            return float(period)
        positive_diffs = valid.diff().dt.total_seconds()
        positive_diffs = positive_diffs[positive_diffs > 0]
        if not positive_diffs.empty:
            # Repeated minute-resolution timestamps are common in these exports.
            repeated_factor = max(len(valid) / max(valid.nunique(), 1), 1.0)
            period = float(positive_diffs.median() / repeated_factor)
            if math.isfinite(period) and period > 0:
                return period
    return 10.0


def repair_short_temperature_drop_flypoints(
    series: pd.Series,
    drop_threshold_k: float = 5.0,
    recovery_steps: int = 2,
    recovery_tolerance_k: float = 2.0,
) -> Tuple[pd.Series, Dict[str, int], List[Dict[str, object]]]:
    """Repair only downward flypoints that recover within one or two samples.

    A point is a candidate when it is more than ``drop_threshold_k`` below the
    preceding accepted value.  It is repaired only if one of the following
    ``recovery_steps`` samples returns to within ``recovery_tolerance_k`` of the
    pre-drop value.  The anomalous run is then linearly interpolated between the
    pre-drop value and the observed recovery value.  A drop that does not
    recover is preserved exactly.

    This exact pattern test intentionally uses a bounded look-ahead.  Every
    repaired event records its decision index so the downstream window builder
    can exclude endpoints before that decision and remain leakage-safe.
    """
    values = series.to_numpy(dtype=float, copy=True)
    output = values.copy()
    events: List[Dict[str, object]] = []
    stats = {
        "flypoint_candidate_episodes": 0,
        "flypoint_recovered_episodes": 0,
        "flypoint_repaired_samples": 0,
        "sustained_drop_episodes": 0,
        "sustained_drop_delayed_samples": 0,
        "unresolved_drop_episodes": 0,
    }
    if values.size == 0:
        return pd.Series(output, index=series.index, name=series.name), stats, events

    index = 1
    while index < len(values):
        reference = output[index - 1]
        raw_value = values[index]
        if not (math.isfinite(reference) and math.isfinite(raw_value)):
            index += 1
            continue
        initial_drop = reference - raw_value
        if initial_drop <= drop_threshold_k:
            index += 1
            continue

        stats["flypoint_candidate_episodes"] += 1
        recovery_index = -1
        for offset in range(1, recovery_steps + 1):
            candidate_index = index + offset
            if candidate_index >= len(values):
                break
            candidate_value = values[candidate_index]
            if math.isfinite(candidate_value) and abs(candidate_value - reference) <= recovery_tolerance_k:
                recovery_index = candidate_index
                break

        decision_index = min(index + recovery_steps, len(values) - 1)
        if recovery_index < 0:
            outcome = (
                "unresolved_end_of_file"
                if index + recovery_steps >= len(values)
                else "not_recovered_kept_raw"
            )
            if outcome == "unresolved_end_of_file":
                stats["unresolved_drop_episodes"] += 1
            else:
                stats["sustained_drop_episodes"] += 1
            events.append(
                {
                    "start_index": index,
                    "decision_index": decision_index,
                    "outcome": outcome,
                    "reference_k": reference,
                    "first_raw_k": raw_value,
                    "minimum_raw_k": float(
                        np.nanmin(values[index : decision_index + 1])
                    ),
                    "initial_drop_k": initial_drop,
                    "held_samples": 0,
                    "decision_value_k": values[decision_index],
                    "unsafe_endpoint_start": "",
                    "unsafe_endpoint_end": "",
                }
            )
            index += 1
            continue

        repaired_count = recovery_index - index
        recovery_value = values[recovery_index]
        interpolation = np.linspace(
            reference,
            recovery_value,
            repaired_count + 2,
            dtype=float,
        )[1:-1]
        output[index:recovery_index] = interpolation
        stats["flypoint_recovered_episodes"] += 1
        stats["flypoint_repaired_samples"] += repaired_count
        events.append(
            {
                "start_index": index,
                "decision_index": recovery_index,
                "outcome": "recovered_flypoint",
                "reference_k": reference,
                "first_raw_k": raw_value,
                "minimum_raw_k": float(np.min(values[index:recovery_index])),
                "initial_drop_k": initial_drop,
                "held_samples": repaired_count,
                "decision_value_k": recovery_value,
                "unsafe_endpoint_start": index,
                "unsafe_endpoint_end": recovery_index - 1,
            }
        )
        index = recovery_index + 1

    guarded = pd.Series(output, index=series.index, name=series.name)
    return guarded, stats, events


def prepare_signal(
    series: pd.Series,
    column: str,
    kind: str,
    max_ffill_steps: int,
    flypoint_guard_enabled: bool = True,
    flypoint_drop_threshold_k: float = 5.0,
    flypoint_recovery_steps: int = 2,
    flypoint_recovery_tolerance_k: float = 2.0,
) -> Tuple[pd.Series, Dict[str, int], List[Dict[str, object]]]:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    missing_raw = int(numeric.isna().sum())
    zero_invalid = 0

    # A valve opening of 0% is never removed. Pressure/flow zero is also retained.
    if kind == "temperature" and is_absolute_temperature(column):
        zero_mask = numeric.eq(0.0)
        zero_invalid = int(zero_mask.sum())
        numeric = numeric.mask(zero_mask)

    if max_ffill_steps > 0:
        prepared = numeric.ffill(limit=max_ffill_steps)
    else:
        prepared = numeric.copy()

    flypoint_stats = {
        "flypoint_candidate_episodes": 0,
        "flypoint_recovered_episodes": 0,
        "flypoint_repaired_samples": 0,
        "sustained_drop_episodes": 0,
        "sustained_drop_delayed_samples": 0,
        "unresolved_drop_episodes": 0,
    }
    flypoint_events: List[Dict[str, object]] = []
    if flypoint_guard_enabled and kind == "temperature" and is_absolute_temperature(column):
        prepared, flypoint_stats, flypoint_events = repair_short_temperature_drop_flypoints(
            prepared,
            drop_threshold_k=flypoint_drop_threshold_k,
            recovery_steps=flypoint_recovery_steps,
            recovery_tolerance_k=flypoint_recovery_tolerance_k,
        )

    stats = {
        "missing_raw": missing_raw,
        "zero_marked_invalid": zero_invalid,
        "missing_after_causal_fill": int(prepared.isna().sum()),
        **flypoint_stats,
    }
    return prepared, stats, flypoint_events


def causal_median(series: pd.Series, window: int = 3) -> pd.Series:
    result = series.rolling(window=window, min_periods=1, center=False).median()
    return result.mask(series.isna())


def causal_ewma(series: pd.Series, tau_seconds: float, sample_period_seconds: float) -> pd.Series:
    dt = max(float(sample_period_seconds), 1e-6)
    alpha = 1.0 - math.exp(-dt / float(tau_seconds))
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    result = series.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    return result.mask(series.isna())


def candidate_filters(
    prepared: pd.Series,
    kind: str,
    sample_period_seconds: float,
) -> Dict[str, pd.Series]:
    result = {"causal_fill_only": prepared.copy()}
    result["causal_median_3"] = causal_median(prepared, window=3)
    result["causal_ewma_20s"] = causal_ewma(prepared, 20.0, sample_period_seconds)
    if kind in {"pressure", "flow"}:
        result["causal_ewma_40s"] = causal_ewma(prepared, 40.0, sample_period_seconds)
        result["causal_ewma_60s"] = causal_ewma(prepared, 60.0, sample_period_seconds)
    return result


def choose_filter(
    prepared: pd.Series,
    filter_name: str,
    sample_period_seconds: float,
) -> pd.Series:
    if filter_name == "causal_fill_only":
        return prepared.copy()
    if filter_name == "causal_median_3":
        return causal_median(prepared, window=3)
    match = re.fullmatch(r"causal_ewma_(\d+)s", filter_name)
    if match:
        return causal_ewma(prepared, float(match.group(1)), sample_period_seconds)
    raise ValueError(f"Unknown filter: {filter_name}")


def finite_std_diff(series: pd.Series) -> float:
    values = series.diff().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size >= 2 else float("nan")


def quality_metrics(
    baseline: pd.Series,
    candidate: pd.Series,
    preparation_stats: Mapping[str, int],
) -> Dict[str, float | int]:
    base = baseline.to_numpy(dtype=float)
    cand = candidate.to_numpy(dtype=float)
    valid = np.isfinite(base) & np.isfinite(cand)
    difference = np.abs(cand[valid] - base[valid])
    scale = float(np.nanmedian(np.abs(base[valid]))) if valid.any() else 0.0
    tolerance = max(1e-12, scale * 1e-9)

    base_diff_std = finite_std_diff(baseline)
    candidate_diff_std = finite_std_diff(candidate)
    if math.isfinite(base_diff_std) and base_diff_std > 0 and math.isfinite(candidate_diff_std):
        roughness_reduction = 100.0 * (1.0 - candidate_diff_std / base_diff_std)
    else:
        roughness_reduction = float("nan")

    if valid.sum() >= 2 and np.std(base[valid]) > 0 and np.std(cand[valid]) > 0:
        correlation = float(np.corrcoef(base[valid], cand[valid])[0, 1])
    else:
        correlation = float("nan")

    return {
        **dict(preparation_stats),
        "valid_overlap": int(valid.sum()),
        "missing_candidate": int(np.isnan(cand).sum()),
        "changed_samples": int(np.sum(difference > tolerance)) if difference.size else 0,
        "mean_abs_change": float(np.mean(difference)) if difference.size else float("nan"),
        "p95_abs_change": float(np.percentile(difference, 95)) if difference.size else float("nan"),
        "max_abs_change": float(np.max(difference)) if difference.size else float("nan"),
        "baseline_diff_std": base_diff_std,
        "candidate_diff_std": candidate_diff_std,
        "roughness_reduction_pct": roughness_reduction,
        "correlation_with_baseline": correlation,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def downsample_indices(length: int, maximum: int = 20_000) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=int))


def create_overview_plot(
    file_name: str,
    kind_label: str,
    columns: Sequence[str],
    candidates: Mapping[str, Mapping[str, pd.Series]],
    sample_period_seconds: float,
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not columns:
        return
    figure, axes = plt.subplots(
        len(columns),
        1,
        figsize=(15, max(3.0 * len(columns), 4.0)),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, column in zip(axes_array, columns):
        series_map = candidates[column]
        n = len(next(iter(series_map.values())))
        index = downsample_indices(n)
        hours = index * sample_period_seconds / 3600.0
        for candidate_name, series in series_map.items():
            linewidth = 1.35 if candidate_name == "causal_fill_only" else 0.9
            alpha = 0.95 if candidate_name == "causal_fill_only" else 0.8
            axis.plot(hours, series.iloc[index], label=candidate_name, linewidth=linewidth, alpha=alpha)
        axis.set_ylabel(signal_name(column))
        axis.grid(alpha=0.2)
    axes_array[-1].set_xlabel("Elapsed time (h)")
    axes_array[0].set_title(
        f"{file_name} | {kind_label} causal-filter review | dt≈{sample_period_seconds:.2f}s"
    )
    handles, labels = axes_array[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", fontsize=8)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def select_valve_events(
    frame: pd.DataFrame,
    event_count: int,
    event_radius: int,
) -> List[int]:
    if event_count <= 0:
        return []
    valve_columns = [
        column
        for column in frame.columns
        if signal_kind(column) == "valve" and signal_name(column) in MAIN_VALVES
    ]
    if not valve_columns:
        valve_columns = [column for column in frame.columns if signal_kind(column) == "valve"]
    if not valve_columns:
        return []

    strength = np.zeros(len(frame), dtype=float)
    for column in valve_columns:
        values = pd.to_numeric(frame[column], errors="coerce").ffill().to_numpy(dtype=float)
        differences = np.abs(np.diff(values, prepend=values[0]))
        differences[~np.isfinite(differences)] = 0.0
        strength += differences

    selected: List[int] = []
    minimum_separation = max(2 * event_radius, 1)
    for index in np.argsort(strength)[::-1]:
        if strength[index] <= 0:
            break
        if all(abs(int(index) - previous) >= minimum_separation for previous in selected):
            selected.append(int(index))
            if len(selected) >= event_count:
                break
    return sorted(selected)


def find_value_column(frame: pd.DataFrame, requested_signal: str) -> str | None:
    for column in frame.columns:
        if signal_name(column).upper() == requested_signal.upper() and column.endswith(" ValueY"):
            return column
    return None


def create_event_plot(
    frame: pd.DataFrame,
    file_name: str,
    event_index: int,
    event_number: int,
    event_radius: int,
    candidates: Mapping[str, Mapping[str, pd.Series]],
    sample_period_seconds: float,
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = max(0, event_index - event_radius)
    stop = min(len(frame), event_index + event_radius + 1)
    positions = np.arange(start, stop)
    minutes = (positions - event_index) * sample_period_seconds / 60.0

    figure, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    valve_columns = [
        column
        for signal in MAIN_VALVES
        if (column := find_value_column(frame, signal)) is not None
    ]
    for column in valve_columns:
        axes[0].plot(minutes, pd.to_numeric(frame[column], errors="coerce").iloc[start:stop], label=signal_name(column))
    axes[0].set_ylabel("Valve opening")
    axes[0].legend(ncol=3, fontsize=8)

    requested = ("TE8353", "FT8351", "PT8351")
    labels = ("Temperature", "Flow", "Pressure")
    for axis, requested_signal, label in zip(axes[1:], requested, labels):
        column = find_value_column(frame, requested_signal)
        if column is None or column not in candidates:
            axis.text(0.5, 0.5, f"{requested_signal} unavailable", ha="center", va="center")
        else:
            for candidate_name, series in candidates[column].items():
                axis.plot(minutes, series.iloc[start:stop], label=candidate_name, linewidth=1.0)
            axis.legend(ncol=3, fontsize=7)
        axis.set_ylabel(label)

    for axis in axes:
        axis.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Minutes relative to valve event")
    axes[0].set_title(f"{file_name} | dynamic window {event_number} | sample {event_index}")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def add_flypoint_event_context(
    destination: List[Dict[str, object]],
    frame: pd.DataFrame,
    file_name: str,
    column: str,
    events: Iterable[Mapping[str, object]],
) -> None:
    paired_time_column = f"{signal_name(column)} Time"
    for event in events:
        row = {
            "file": file_name,
            "signal": signal_name(column),
            "column": column,
            **dict(event),
        }
        start_index = int(event["start_index"])
        decision_index = int(event["decision_index"])
        if paired_time_column in frame.columns:
            row["start_time"] = frame[paired_time_column].iloc[start_index]
            row["decision_time"] = frame[paired_time_column].iloc[decision_index]
        else:
            row["start_time"] = ""
            row["decision_time"] = ""
        destination.append(row)


def build_flypoint_endpoint_exclusions(
    event_rows: Iterable[Mapping[str, object]],
) -> List[Dict[str, object]]:
    exclusions: List[Dict[str, object]] = []
    for event in event_rows:
        if event.get("outcome") != "recovered_flypoint":
            continue
        exclusions.append(
            {
                "file": event["file"],
                "signal": event["signal"],
                "repaired_start_index": event["start_index"],
                "recovery_decision_index": event["decision_index"],
                "unsafe_endpoint_start": event["unsafe_endpoint_start"],
                "unsafe_endpoint_end": event["unsafe_endpoint_end"],
                "reason": "repaired value depends on recovery sample",
            }
        )
    return exclusions


def run_review(args: argparse.Namespace) -> None:
    files = discover_data_files(args.input_dir)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.review_dir / "plots"
    plots_enabled = not args.no_plots
    if plots_enabled and importlib.util.find_spec("matplotlib") is None:
        plots_enabled = False
        print(
            "WARNING: matplotlib is not installed; CSV review reports will still be created, "
            "but PNG plots are skipped. Install matplotlib and rerun review to create plots."
        )
    if plots_enabled:
        plot_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    flypoint_event_rows: List[Dict[str, object]] = []

    for file_index, path in enumerate(files, start=1):
        print(f"[review {file_index}/{len(files)}] {path.name}")
        frame = read_industrial_csv(path)
        dt_seconds = infer_sample_period_seconds(frame)
        candidate_by_column: Dict[str, Dict[str, pd.Series]] = {}
        preparation_totals = {
            "missing_raw": 0,
            "zero_marked_invalid": 0,
            "missing_after_causal_fill": 0,
            "flypoint_candidate_episodes": 0,
            "flypoint_recovered_episodes": 0,
            "flypoint_repaired_samples": 0,
            "sustained_drop_episodes": 0,
            "sustained_drop_delayed_samples": 0,
            "unresolved_drop_episodes": 0,
        }

        for column in frame.columns:
            kind = signal_kind(column)
            if kind not in {"temperature", "pressure", "flow"}:
                continue
            prepared, prep_stats, flypoint_events = prepare_signal(
                frame[column],
                column,
                kind,
                args.max_ffill_steps,
                flypoint_guard_enabled=not args.disable_temperature_flypoint_guard,
                flypoint_drop_threshold_k=args.flypoint_drop_threshold_k,
                flypoint_recovery_steps=args.flypoint_recovery_steps,
                flypoint_recovery_tolerance_k=args.flypoint_recovery_tolerance_k,
            )
            add_flypoint_event_context(
                flypoint_event_rows, frame, path.name, column, flypoint_events
            )
            candidate_map = candidate_filters(prepared, kind, dt_seconds)
            candidate_by_column[column] = candidate_map
            for key in preparation_totals:
                preparation_totals[key] += prep_stats[key]
            for candidate_name, candidate in candidate_map.items():
                metric_rows.append(
                    {
                        "file": path.name,
                        "signal": signal_name(column),
                        "column": column,
                        "kind": kind,
                        "candidate": candidate_name,
                        "rows": len(frame),
                        "sample_period_seconds": dt_seconds,
                        **quality_metrics(prepared, candidate, prep_stats),
                    }
                )

        summary_rows.append(
            {
                "file": path.name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "sample_period_seconds": dt_seconds,
                "reviewed_sensor_columns": len(candidate_by_column),
                "valve_columns_protected": sum(signal_kind(column) == "valve" for column in frame.columns),
                "time_columns_protected": sum(signal_kind(column) == "time" for column in frame.columns),
                "excluded_A_B_columns_protected": sum(
                    signal_kind(column) == "excluded_temperature" for column in frame.columns
                ),
                **preparation_totals,
                "source_sha256": sha256_file(path),
            }
        )

        if plots_enabled:
            temperature_columns = [
                column for column in candidate_by_column if signal_kind(column) == "temperature"
            ]
            pressure_flow_columns = [
                column for column in candidate_by_column if signal_kind(column) in {"pressure", "flow"}
            ]
            create_overview_plot(
                path.name,
                "temperature",
                temperature_columns,
                candidate_by_column,
                dt_seconds,
                plot_dir / f"{path.stem}__temperature.png",
            )
            create_overview_plot(
                path.name,
                "pressure_flow",
                pressure_flow_columns,
                candidate_by_column,
                dt_seconds,
                plot_dir / f"{path.stem}__pressure_flow.png",
            )
            events = select_valve_events(frame, args.event_windows, args.event_radius)
            for event_number, event_index in enumerate(events, start=1):
                create_event_plot(
                    frame,
                    path.name,
                    event_index,
                    event_number,
                    args.event_radius,
                    candidate_by_column,
                    dt_seconds,
                    plot_dir / f"{path.stem}__event_{event_number:02d}.png",
                )

    metrics_path = args.review_dir / "filter_candidate_metrics.csv"
    summary_path = args.review_dir / "filter_file_summary.csv"
    flypoint_events_path = args.review_dir / "temperature_flypoint_events.csv"
    endpoint_exclusions_path = (
        args.review_dir / "temperature_flypoint_endpoint_exclusions.csv"
    )
    config_path = args.review_dir / "filter_review_config.json"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(flypoint_event_rows).to_csv(
        flypoint_events_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(build_flypoint_endpoint_exclusions(flypoint_event_rows)).to_csv(
        endpoint_exclusions_path, index=False, encoding="utf-8-sig"
    )
    config = {
        "mode": "review",
        "input_dir": str(args.input_dir.resolve()),
        "reviewed_file_rule": "raw unaugmented *.csv only",
        "reviewed_files": len(files),
        "temperature_candidates": list(TEMPERATURE_FILTERS),
        "pressure_flow_candidates": list(PRESSURE_FLOW_FILTERS),
        "max_ffill_steps": args.max_ffill_steps,
        "temperature_flypoint_guard_enabled": not args.disable_temperature_flypoint_guard,
        "flypoint_drop_threshold_k": args.flypoint_drop_threshold_k,
        "flypoint_recovery_steps": args.flypoint_recovery_steps,
        "flypoint_recovery_tolerance_k": args.flypoint_recovery_tolerance_k,
        "flypoint_guard_implementation": "exact_recovery_test_with_bounded_lookahead",
        "bounded_temperature_lookahead_steps": (
            args.flypoint_recovery_steps
            if not args.disable_temperature_flypoint_guard
            else 0
        ),
        "downstream_endpoint_exclusion_required": not args.disable_temperature_flypoint_guard,
        "event_windows_per_file": args.event_windows,
        "event_radius_samples": args.event_radius,
        "plots_requested": not args.no_plots,
        "plots_generated": plots_enabled,
        "causal_only": args.disable_temperature_flypoint_guard,
        "backward_fill": False,
        "centered_filter": False,
        "valve_filtering": False,
        "valve_zero_is_valid": True,
        "data_split": False,
        "label_creation": False,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Review complete: {args.review_dir}")
    print(f"  metrics: {metrics_path.name}")
    print(f"  summary: {summary_path.name}")
    print(f"  flypoints: {flypoint_events_path.name}")
    print(f"  endpoint exclusions: {endpoint_exclusions_path.name}")
    if plots_enabled:
        print(f"  plots:   {plot_dir}")


def series_semantically_equal(left: pd.Series, right: pd.Series) -> bool:
    if len(left) != len(right):
        return False
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    numeric_mask = left_numeric.notna() | right_numeric.notna()
    if numeric_mask.any():
        return bool(
            np.array_equal(
                left_numeric.to_numpy(dtype=float),
                right_numeric.to_numpy(dtype=float),
                equal_nan=True,
            )
        )
    left_text = left.fillna("<NA>").astype(str)
    right_text = right.fillna("<NA>").astype(str)
    return left_text.equals(right_text)


def verify_protected_columns(before: pd.DataFrame, after: pd.DataFrame) -> Tuple[bool, List[str]]:
    changed: List[str] = []
    for column in before.columns:
        kind = signal_kind(column)
        if kind in {"time", "valve", "excluded_temperature", "other"}:
            if column not in after or not series_semantically_equal(before[column], after[column]):
                changed.append(column)
    return not changed, changed


def run_apply(args: argparse.Namespace) -> None:
    if not args.confirm_apply:
        raise ValueError("apply mode requires --confirm-apply")
    if args.temperature_filter is None or args.pressure_flow_filter is None:
        raise ValueError(
            "apply mode requires both --temperature-filter and --pressure-flow-filter"
        )
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("output-dir must differ from input-dir; source data is never overwritten")

    files = discover_data_files(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conflicts = [args.output_dir / path.name for path in files if (args.output_dir / path.name).exists()]
    if conflicts and not args.overwrite:
        preview = ", ".join(path.name for path in conflicts[:3])
        raise FileExistsError(
            f"{len(conflicts)} output files already exist ({preview}). Use --overwrite to replace them."
        )

    manifest_rows: List[Dict[str, object]] = []
    flypoint_event_rows: List[Dict[str, object]] = []
    for file_index, path in enumerate(files, start=1):
        print(f"[apply {file_index}/{len(files)}] {path.name}")
        source_hash_before = sha256_file(path)
        before = read_industrial_csv(path)
        after = before.copy(deep=True)
        dt_seconds = infer_sample_period_seconds(before)
        filtered_columns: List[str] = []
        file_flypoint_stats = {
            "flypoint_candidate_episodes": 0,
            "flypoint_recovered_episodes": 0,
            "flypoint_repaired_samples": 0,
            "sustained_drop_episodes": 0,
            "sustained_drop_delayed_samples": 0,
            "unresolved_drop_episodes": 0,
        }

        for column in before.columns:
            kind = signal_kind(column)
            if kind not in {"temperature", "pressure", "flow"}:
                continue
            prepared, prep_stats, flypoint_events = prepare_signal(
                before[column],
                column,
                kind,
                args.max_ffill_steps,
                flypoint_guard_enabled=not args.disable_temperature_flypoint_guard,
                flypoint_drop_threshold_k=args.flypoint_drop_threshold_k,
                flypoint_recovery_steps=args.flypoint_recovery_steps,
                flypoint_recovery_tolerance_k=args.flypoint_recovery_tolerance_k,
            )
            for key in file_flypoint_stats:
                file_flypoint_stats[key] += prep_stats[key]
            add_flypoint_event_context(
                flypoint_event_rows, before, path.name, column, flypoint_events
            )
            filter_name = (
                args.temperature_filter
                if kind == "temperature"
                else args.pressure_flow_filter
            )
            after[column] = choose_filter(prepared, filter_name, dt_seconds)
            filtered_columns.append(column)

        protected_ok_memory, changed_memory = verify_protected_columns(before, after)
        if not protected_ok_memory:
            raise AssertionError(
                f"Protected columns changed before writing {path.name}: {changed_memory}"
            )

        destination = args.output_dir / path.name
        after.to_csv(destination, index=False, encoding="utf-8-sig")
        reloaded = read_industrial_csv(destination)
        protected_ok_disk, changed_disk = verify_protected_columns(before, reloaded)
        if not protected_ok_disk:
            destination.unlink(missing_ok=True)
            raise AssertionError(
                f"Protected columns changed after CSV round-trip for {path.name}: {changed_disk}"
            )
        source_hash_after = sha256_file(path)
        if source_hash_before != source_hash_after:
            raise AssertionError(f"Source file changed unexpectedly: {path}")

        manifest_rows.append(
            {
                "source_file": path.name,
                "output_file": destination.name,
                "rows": len(after),
                "columns": len(after.columns),
                "sample_period_seconds": dt_seconds,
                "temperature_filter": args.temperature_filter,
                "pressure_flow_filter": args.pressure_flow_filter,
                "max_ffill_steps": args.max_ffill_steps,
                "filtered_column_count": len(filtered_columns),
                "valve_column_count": sum(signal_kind(column) == "valve" for column in before.columns),
                **file_flypoint_stats,
                "protected_columns_exact": protected_ok_disk,
                "source_unchanged": source_hash_before == source_hash_after,
                "source_sha256": source_hash_before,
                "output_sha256": sha256_file(destination),
            }
        )

    manifest_path = args.output_dir / "filter_manifest.csv"
    flypoint_events_path = args.output_dir / "temperature_flypoint_events.csv"
    endpoint_exclusions_path = (
        args.output_dir / "temperature_flypoint_endpoint_exclusions.csv"
    )
    config_path = args.output_dir / "filter_config.json"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(flypoint_event_rows).to_csv(
        flypoint_events_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(build_flypoint_endpoint_exclusions(flypoint_event_rows)).to_csv(
        endpoint_exclusions_path, index=False, encoding="utf-8-sig"
    )
    config = {
        "mode": "apply",
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "data_file_rule": "raw unaugmented *.csv only",
        "pipeline_stage": "filter_before_augmentation",
        "next_stage": "create.py reads this directory and writes filtered_data",
        "files_written": len(files),
        "temperature_filter": args.temperature_filter,
        "pressure_flow_filter": args.pressure_flow_filter,
        "max_ffill_steps": args.max_ffill_steps,
        "temperature_flypoint_guard_enabled": not args.disable_temperature_flypoint_guard,
        "flypoint_drop_threshold_k": args.flypoint_drop_threshold_k,
        "flypoint_recovery_steps": args.flypoint_recovery_steps,
        "flypoint_recovery_tolerance_k": args.flypoint_recovery_tolerance_k,
        "flypoint_guard_implementation": "exact_recovery_test_with_bounded_lookahead",
        "bounded_temperature_lookahead_steps": (
            args.flypoint_recovery_steps
            if not args.disable_temperature_flypoint_guard
            else 0
        ),
        "downstream_endpoint_exclusion_required": not args.disable_temperature_flypoint_guard,
        "causal_only": args.disable_temperature_flypoint_guard,
        "backward_fill": False,
        "centered_filter": False,
        "valve_filtering": False,
        "valve_zero_is_valid": True,
        "data_split": False,
        "label_creation": False,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Filtered files written and verified: {args.output_dir}")


def validate_args(args: argparse.Namespace) -> None:
    if args.max_ffill_steps < 0:
        raise ValueError("--max-ffill-steps must be >= 0")
    if args.flypoint_drop_threshold_k <= 0:
        raise ValueError("--flypoint-drop-threshold-k must be > 0")
    if args.flypoint_recovery_steps < 1:
        raise ValueError("--flypoint-recovery-steps must be >= 1")
    if args.flypoint_recovery_tolerance_k < 0:
        raise ValueError("--flypoint-recovery-tolerance-k must be >= 0")
    if args.event_windows < 0:
        raise ValueError("--event-windows must be >= 0")
    if args.event_radius < 1:
        raise ValueError("--event-radius must be >= 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.mode == "review":
        run_review(args)
    else:
        run_apply(args)


if __name__ == "__main__":
    main()
