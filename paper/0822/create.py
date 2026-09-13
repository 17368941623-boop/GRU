#!/usr/bin/env python3
"""Augment the already-filtered 0822 parent CSV files.

Required pipeline order::

    raw CSV -> data_filter.py -> filtered_original_data -> create.py -> filtered_data

For every filtered parent, this script writes one unchanged ``Original__`` copy
and two deterministic temperature-only augmented copies.  Pressure, flow,
valve, time, A-pipe and B-pipe columns are never augmented.  Missing values and
legitimate temperature zeros are preserved.  No data split, model feature, or
future label is created here.

The filtering provenance and bounded-lookahead flypoint exclusion manifests are
expanded to all three output variants so that the existing ``data_fil.py`` can
audit and consume ``filtered_data`` without changing its interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "filtered_original_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "filtered_data"

NUM_AUGMENTED_VARIANTS = 2
BASE_SEED = 20260802
COMMON_SHIFT_RANGE_K = (-0.1, 0.1)
SENSOR_SHIFT_RANGE_K = (-0.1, 0.1)
COMMON_DRIFT_AMPLITUDE_K = 0.025
SENSOR_DRIFT_AMPLITUDE_K = 0.025
DRIFT_WINDOW_STEPS = 60
NOISE_STD_BOUNDS_K = (0.005, 0.05)

EXCLUDED_TEMPERATURE_SIGNALS = {"A管", "B管"}
EXACT_TEMPERATURE_SIGNALS = {"DTBR", "EC-B-T", "THV", "TCD"}
OUTPUT_VARIANTS: Tuple[Tuple[str, int], ...] = (
    ("Original", 0),
    ("Aug_01", 1),
    ("Aug_02", 2),
)
PREFIXED_DATA_RE = re.compile(r"^(Original|Aug_\d+)__.+\.csv$", re.IGNORECASE)
REPORT_FILE_NAMES = {
    "filter_manifest.csv",
    "temperature_flypoint_events.csv",
    "temperature_flypoint_endpoint_exclusions.csv",
    "augmentation_manifest.csv",
    "augmentation_parameters.csv",
    "augmentation_validation.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Original + two deterministic temperature-augmented copies "
            "from data_filter.py output."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace expected same-named outputs; unrelated/stale data files are never removed.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the filtered input contract and planned outputs without writing files.",
    )
    return parser.parse_args()


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = (
        normalized.columns.astype(str).str.replace('"', "", regex=False).str.strip()
    )
    return normalized


def read_industrial_csv(path: Path) -> Tuple[pd.DataFrame, str, str]:
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
                return normalize_columns(frame), encoding, separator
        except Exception as exc:
            errors.append(f"{encoding}/{separator!r}: {exc}")
    raise ValueError(f"Could not read {path}. Attempts: {' | '.join(errors)}")


def signal_name(column: str) -> str:
    name = str(column).strip()
    if name.lower().endswith(" valuey"):
        return name[: -len(" ValueY")].strip()
    return name


def is_augmented_temperature_column(column: str) -> bool:
    if not str(column).strip().lower().endswith(" valuey"):
        return False
    signal = signal_name(column)
    if signal in EXCLUDED_TEMPERATURE_SIGNALS:
        return False
    upper = signal.upper()
    return upper.startswith("TE") or upper in EXACT_TEMPERATURE_SIGNALS


def estimate_noise_std(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & numeric.ne(0.0)
    differences = numeric.where(valid).diff().dropna().to_numpy(dtype=np.float64)
    if len(differences) < 20:
        return float(NOISE_STD_BOUNDS_K[0])
    median = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median)))
    estimate = 1.4826 * mad / np.sqrt(2.0)
    if not np.isfinite(estimate) or estimate <= 0:
        estimate = NOISE_STD_BOUNDS_K[0]
    return float(np.clip(estimate, *NOISE_STD_BOUNDS_K))


def smooth_drift(rng: np.random.Generator, length: int, amplitude: float) -> np.ndarray:
    if length <= 1:
        return np.zeros(length, dtype=np.float64)
    anchors = np.arange(0, length, DRIFT_WINDOW_STEPS, dtype=int)
    if anchors[-1] != length - 1:
        anchors = np.append(anchors, length - 1)
    values = rng.uniform(-amplitude, amplitude, size=len(anchors))
    return np.interp(np.arange(length), anchors, values)


def augment_temperature_columns(
    original: pd.DataFrame,
    temperature_columns: Sequence[str],
    seed: int,
    parent_file: str,
    variant_id: int,
) -> Tuple[pd.DataFrame, List[Dict[str, object]], float]:
    rng = np.random.default_rng(seed)
    augmented = original.copy()
    length = len(augmented)
    common_shift = float(rng.uniform(*COMMON_SHIFT_RANGE_K))
    common_drift = smooth_drift(rng, length, COMMON_DRIFT_AMPLITUDE_K)
    parameter_rows: List[Dict[str, object]] = []

    for column in temperature_columns:
        numeric = pd.to_numeric(original[column], errors="coerce")
        valid = numeric.notna() & numeric.ne(0.0)
        sensor_shift = float(rng.uniform(*SENSOR_SHIFT_RANGE_K))
        sensor_drift = smooth_drift(rng, length, SENSOR_DRIFT_AMPLITUDE_K)
        noise_std = estimate_noise_std(original[column])
        white_noise = rng.normal(0.0, noise_std, size=length)
        perturbation = (
            common_shift + common_drift + sensor_shift + sensor_drift + white_noise
        )
        output = numeric.copy()
        output.loc[valid] = (
            numeric.loc[valid] + perturbation[valid.to_numpy()]
        )
        augmented[column] = output
        parameter_rows.append(
            {
                "parent_file": parent_file,
                "variant_id": variant_id,
                "seed": seed,
                "temperature_column": column,
                "common_shift_k": common_shift,
                "sensor_shift_k": sensor_shift,
                "estimated_noise_std_k": noise_std,
                "common_drift_amplitude_k": COMMON_DRIFT_AMPLITUDE_K,
                "sensor_drift_amplitude_k": SENSOR_DRIFT_AMPLITUDE_K,
                "drift_window_steps": DRIFT_WINDOW_STEPS,
            }
        )
    return augmented, parameter_rows, common_shift


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def series_semantically_equal(left: pd.Series, right: pd.Series) -> bool:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    if left_numeric.notna().any() or right_numeric.notna().any():
        if not left_numeric.isna().equals(right_numeric.isna()):
            return False
        valid = left_numeric.notna()
        if valid.any() and not np.array_equal(
            left_numeric.loc[valid].to_numpy(dtype=float),
            right_numeric.loc[valid].to_numpy(dtype=float),
        ):
            return False
        nonnumeric = ~valid
        if nonnumeric.any():
            return (
                left.loc[nonnumeric].fillna("<NA>").astype(str).reset_index(drop=True)
                .equals(
                    right.loc[nonnumeric]
                    .fillna("<NA>")
                    .astype(str)
                    .reset_index(drop=True)
                )
            )
        return True
    return left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str))


def frames_semantically_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        return False
    return all(series_semantically_equal(left[column], right[column]) for column in left.columns)


def read_required_csv(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required provenance file: {path}")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame(columns=list(required_columns))
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise KeyError(f"{path.name} is missing columns: {sorted(missing)}")
    return frame


def discover_filtered_parents(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Filtered input directory does not exist: {input_dir}")
    prefixed = sorted(
        path.name
        for path in input_dir.iterdir()
        if path.is_file() and PREFIXED_DATA_RE.match(path.name)
    )
    if prefixed:
        preview = ", ".join(prefixed[:3])
        raise ValueError(
            "create.py expects filtered, unaugmented parent files, but input-dir "
            f"already contains prefixed augmentation outputs ({preview})."
        )
    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and path.name.lower() not in REPORT_FILE_NAMES
    )
    if not files:
        raise FileNotFoundError(f"No filtered parent CSV files found in {input_dir}")
    return files


def load_filter_contract(
    input_dir: Path, source_files: Sequence[Path]
) -> Tuple[Dict[str, object], pd.DataFrame]:
    config_path = input_dir / "filter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}; create.py only accepts audited data_filter.py output"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "apply":
        raise ValueError("filter_config.json is not an apply-mode filtering result")
    if config.get("pipeline_stage") != "filter_before_augmentation":
        raise ValueError(
            "filter_config.json does not declare filter_before_augmentation; rerun updated data_filter.py"
        )
    if config.get("data_split") is not False or config.get("label_creation") is not False:
        raise ValueError("Filtered input unexpectedly contains split/label operations")
    if (
        config.get("valve_filtering") is not False
        or config.get("valve_smoothing") is not False
        or config.get("valve_negative_clamp") is not True
        or config.get("valve_zero_is_valid") is not True
    ):
        raise ValueError("Filtered input violates the valve-preservation contract")
    if config.get("backward_fill") is not False or config.get("centered_filter") is not False:
        raise ValueError("Filtered input contains a non-approved backward/centered filter")

    manifest = read_required_csv(
        input_dir / "filter_manifest.csv",
        {
            "source_file",
            "output_file",
            "rows",
            "protected_columns_contract_ok",
            "source_unchanged",
            "output_sha256",
        },
    )
    expected = {path.name for path in source_files}
    listed = set(manifest["output_file"].astype(str))
    if expected != listed:
        raise AssertionError(
            "Filtered parent/manifest mismatch; "
            f"unlisted={sorted(expected - listed)}, missing_files={sorted(listed - expected)}"
        )
    if manifest["output_file"].duplicated().any():
        raise ValueError("filter_manifest.csv contains duplicate output_file rows")
    truthy = {"true", "1"}
    for column in ("protected_columns_contract_ok", "source_unchanged"):
        if not manifest[column].astype(str).str.lower().isin(truthy).all():
            raise ValueError(f"filter_manifest.csv contains a failed {column} check")
    hash_by_file = manifest.set_index("output_file")["output_sha256"].astype(str).to_dict()
    for path in source_files:
        actual = sha256_file(path)
        if actual != hash_by_file[path.name]:
            raise AssertionError(
                f"Filtered parent hash differs from filter_manifest.csv: {path.name}"
            )
    return config, manifest


def output_name(variant: str, parent_file: str) -> str:
    return f"{variant}__{parent_file}"


def expand_file_manifest(
    frame: pd.DataFrame,
    output_hashes: Mapping[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for record in frame.to_dict("records"):
        parent = str(record["output_file"])
        for variant, variant_id in OUTPUT_VARIANTS:
            final_name = output_name(variant, parent)
            expanded = dict(record)
            expanded.update(
                {
                    "filtered_parent_file": parent,
                    "output_file": final_name,
                    "augmentation_variant": variant,
                    "augmentation_variant_id": variant_id,
                    "post_filter_augmentation": variant_id != 0,
                    "filtered_parent_sha256": record["output_sha256"],
                    "output_sha256": output_hashes[final_name],
                }
            )
            rows.append(expanded)
    return pd.DataFrame(rows)


def expand_event_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rows: List[Dict[str, object]] = []
    for record in frame.to_dict("records"):
        parent = str(record["file"])
        for variant, variant_id in OUTPUT_VARIANTS:
            expanded = dict(record)
            expanded["filtered_parent_file"] = parent
            expanded["file"] = output_name(variant, parent)
            expanded["augmentation_variant"] = variant
            expanded["augmentation_variant_id"] = variant_id
            rows.append(expanded)
    return pd.DataFrame(rows)


def numeric_difference(left: pd.Series, right: pd.Series) -> Tuple[bool, float]:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    if not left_numeric.isna().equals(right_numeric.isna()):
        return False, float("inf")
    valid = left_numeric.notna()
    if not valid.any():
        equal = left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str))
        return equal, 0.0 if equal else float("inf")
    maximum = float(
        np.max(
            np.abs(
                left_numeric.loc[valid].to_numpy(dtype=float)
                - right_numeric.loc[valid].to_numpy(dtype=float)
            )
        )
    )
    return maximum <= 1e-12, maximum


def validate_augmented_frame(
    parent_file: str,
    variant_id: int,
    original: pd.DataFrame,
    augmented: pd.DataFrame,
    temperature_columns: Sequence[str],
) -> Dict[str, object]:
    schema_ok = original.shape == augmented.shape and list(original.columns) == list(
        augmented.columns
    )
    if not schema_ok:
        raise AssertionError(f"Augmented schema differs: {parent_file}/Aug_{variant_id:02d}")

    non_temperature_columns = [
        column for column in original.columns if column not in temperature_columns
    ]
    non_temperature_equal = True
    non_temperature_max_difference = 0.0
    for column in non_temperature_columns:
        equal, difference = numeric_difference(original[column], augmented[column])
        non_temperature_equal &= equal
        non_temperature_max_difference = max(non_temperature_max_difference, difference)

    masks_ok = True
    for column in temperature_columns:
        before = pd.to_numeric(original[column], errors="coerce")
        after = pd.to_numeric(augmented[column], errors="coerce")
        masks_ok &= before.isna().equals(after.isna())
        masks_ok &= before.eq(0.0).equals(after.eq(0.0))

    target_statistics: Dict[str, object] = {}
    target_columns = {
        "te8353": next(
            (column for column in temperature_columns if signal_name(column).upper() == "TE8353"),
            None,
        ),
        "thv": next(
            (column for column in temperature_columns if signal_name(column).upper() == "THV"),
            None,
        ),
    }
    if any(column is None for column in target_columns.values()):
        raise KeyError(f"TE8353 or Thv is missing from filtered parent {parent_file}")
    for label, column in target_columns.items():
        assert column is not None
        before = pd.to_numeric(original[column], errors="coerce")
        after = pd.to_numeric(augmented[column], errors="coerce")
        valid = before.notna() & before.ne(0.0)
        absolute = (after.loc[valid] - before.loc[valid]).abs()
        target_statistics[f"{label}_changed"] = bool(absolute.gt(1e-12).any())
        target_statistics[f"{label}_mean_abs_change_k"] = float(absolute.mean())
        target_statistics[f"{label}_max_abs_change_k"] = float(absolute.max())

    ab_columns = [
        column for column in original.columns if signal_name(column) in EXCLUDED_TEMPERATURE_SIGNALS
    ]
    ab_equal = all(
        numeric_difference(original[column], augmented[column])[0] for column in ab_columns
    )
    status = bool(
        schema_ok
        and non_temperature_equal
        and ab_equal
        and masks_ok
        and target_statistics["te8353_changed"]
        and target_statistics["thv_changed"]
    )
    if not status:
        raise AssertionError(f"Augmentation validation failed: {parent_file}/Aug_{variant_id:02d}")
    return {
        "parent_file": parent_file,
        "variant_id": variant_id,
        "rows": len(augmented),
        "columns": len(augmented.columns),
        "temperature_columns": len(temperature_columns),
        "schema_ok": schema_ok,
        "non_temperature_equal": non_temperature_equal,
        "non_temperature_max_difference": non_temperature_max_difference,
        "a_b_pipe_equal": ab_equal,
        "zero_and_missing_masks_preserved": masks_ok,
        **target_statistics,
        "status": "PASS",
    }


def validate_paths(args: argparse.Namespace) -> None:
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("output-dir must differ from input-dir; filtered parents are never overwritten")


def main() -> None:
    args = parse_args()
    validate_paths(args)
    source_files = discover_filtered_parents(args.input_dir)
    filter_config, filter_manifest = load_filter_contract(args.input_dir, source_files)
    expected_data_names = {
        output_name(variant, path.name)
        for path in source_files
        for variant, _ in OUTPUT_VARIANTS
    }

    existing_data_names: set[str] = set()
    if args.output_dir.exists():
        existing_data_names = {
            path.name
            for path in args.output_dir.iterdir()
            if path.is_file() and PREFIXED_DATA_RE.match(path.name)
        }
    stale = existing_data_names - expected_data_names
    if stale:
        raise FileExistsError(
            "Output directory contains stale/unrelated data CSVs; move them before rerunning: "
            + ", ".join(sorted(stale)[:5])
        )
    conflicts = existing_data_names & expected_data_names
    if conflicts and not args.overwrite:
        raise FileExistsError(
            f"{len(conflicts)} expected data outputs already exist. Use --overwrite to replace them."
        )

    event_required = {
        "file",
        "signal",
        "start_index",
        "decision_index",
        "unsafe_endpoint_start",
        "unsafe_endpoint_end",
    }
    exclusion_required = {
        "file",
        "signal",
        "repaired_start_index",
        "recovery_decision_index",
        "unsafe_endpoint_start",
        "unsafe_endpoint_end",
    }
    flypoint_events = read_required_csv(
        args.input_dir / "temperature_flypoint_events.csv", event_required
    )
    flypoint_exclusions = read_required_csv(
        args.input_dir / "temperature_flypoint_endpoint_exclusions.csv",
        exclusion_required,
    )
    parent_names = {path.name for path in source_files}
    for name, frame in (
        ("temperature_flypoint_events.csv", flypoint_events),
        ("temperature_flypoint_endpoint_exclusions.csv", flypoint_exclusions),
    ):
        unknown = set(frame["file"].astype(str)) - parent_names if not frame.empty else set()
        if unknown:
            raise AssertionError(f"{name} references unknown filtered parents: {sorted(unknown)}")

    if args.validate_only:
        print(
            f"Validation PASS: {len(source_files)} filtered parents -> "
            f"{len(expected_data_names)} planned data files; pipeline order is filter then augment."
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []
    parameter_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []
    output_hashes: Dict[str, str] = {}

    for file_index, source_path in enumerate(source_files, start=1):
        source_hash_before = sha256_file(source_path)
        original, source_encoding, source_separator = read_industrial_csv(source_path)
        temperature_columns = [
            column for column in original.columns if is_augmented_temperature_column(column)
        ]
        selected_signals = {signal_name(column).upper() for column in temperature_columns}
        if "TE8353" not in selected_signals or "THV" not in selected_signals:
            raise KeyError(f"TE8353 and Thv must both be selected in {source_path.name}")

        print(
            f"[{file_index:02d}/{len(source_files):02d}] {source_path.name}: "
            f"rows={len(original):,}, temperature_columns={len(temperature_columns)}"
        )
        original_name = output_name("Original", source_path.name)
        original_path = args.output_dir / original_name
        write_csv(original, original_path)
        original_reloaded, _, _ = read_industrial_csv(original_path)
        if not frames_semantically_equal(original, original_reloaded):
            raise AssertionError(f"Original output differs from filtered parent: {source_path.name}")
        output_hashes[original_name] = sha256_file(original_path)
        manifest_rows.append(
            {
                "parent_file": source_path.name,
                "output_file": original_name,
                "variant": "Original",
                "variant_id": 0,
                "seed": "",
                "is_back": "-BACK" in source_path.stem.upper(),
                "rows": len(original),
                "columns": len(original.columns),
                "temperature_columns": "|".join(temperature_columns),
                "source_encoding": source_encoding,
                "source_separator": "TAB" if source_separator == "\t" else source_separator,
                "filtered_parent_sha256": source_hash_before,
                "output_sha256": output_hashes[original_name],
            }
        )

        for variant_id in range(1, NUM_AUGMENTED_VARIANTS + 1):
            seed = BASE_SEED + file_index * 100 + variant_id
            augmented, variant_parameters, _ = augment_temperature_columns(
                original,
                temperature_columns,
                seed,
                source_path.name,
                variant_id,
            )
            name = output_name(f"Aug_{variant_id:02d}", source_path.name)
            path = args.output_dir / name
            write_csv(augmented, path)
            reloaded, _, _ = read_industrial_csv(path)
            validation_rows.append(
                validate_augmented_frame(
                    source_path.name,
                    variant_id,
                    original,
                    reloaded,
                    temperature_columns,
                )
            )
            output_hashes[name] = sha256_file(path)
            parameter_rows.extend(variant_parameters)
            manifest_rows.append(
                {
                    "parent_file": source_path.name,
                    "output_file": name,
                    "variant": f"Aug_{variant_id:02d}",
                    "variant_id": variant_id,
                    "seed": seed,
                    "is_back": "-BACK" in source_path.stem.upper(),
                    "rows": len(augmented),
                    "columns": len(augmented.columns),
                    "temperature_columns": "|".join(temperature_columns),
                    "source_encoding": source_encoding,
                    "source_separator": "TAB" if source_separator == "\t" else source_separator,
                    "filtered_parent_sha256": source_hash_before,
                    "output_sha256": output_hashes[name],
                }
            )
            print(f"    wrote {name} (seed={seed})")

        if sha256_file(source_path) != source_hash_before:
            raise AssertionError(f"Filtered parent changed unexpectedly: {source_path}")

    expected_validation_rows = len(source_files) * NUM_AUGMENTED_VARIANTS
    if len(validation_rows) != expected_validation_rows:
        raise AssertionError("Unexpected augmentation validation row count")

    pd.DataFrame(manifest_rows).to_csv(
        args.output_dir / "augmentation_manifest.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(parameter_rows).to_csv(
        args.output_dir / "augmentation_parameters.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(validation_rows).to_csv(
        args.output_dir / "augmentation_validation.csv", index=False, encoding="utf-8-sig"
    )

    expanded_filter_manifest = expand_file_manifest(filter_manifest, output_hashes)
    expanded_filter_manifest.to_csv(
        args.output_dir / "filter_manifest.csv", index=False, encoding="utf-8-sig"
    )
    expand_event_manifest(flypoint_events).to_csv(
        args.output_dir / "temperature_flypoint_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    expand_event_manifest(flypoint_exclusions).to_csv(
        args.output_dir / "temperature_flypoint_endpoint_exclusions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    augmentation_config = {
        "pipeline_order": ["raw", "filter", "augmentation", "data_split"],
        "input_is_filtered": True,
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "base_seed": BASE_SEED,
        "copies_per_parent": NUM_AUGMENTED_VARIANTS + 1,
        "augmented_variants_per_parent": NUM_AUGMENTED_VARIANTS,
        "common_shift_range_k": COMMON_SHIFT_RANGE_K,
        "sensor_shift_range_k": SENSOR_SHIFT_RANGE_K,
        "common_drift_amplitude_k": COMMON_DRIFT_AMPLITUDE_K,
        "sensor_drift_amplitude_k": SENSOR_DRIFT_AMPLITUDE_K,
        "drift_window_steps": DRIFT_WINDOW_STEPS,
        "noise_std_bounds_k": NOISE_STD_BOUNDS_K,
        "excluded_temperature_signals": sorted(EXCLUDED_TEMPERATURE_SIGNALS),
        "pressure_flow_valve_augmentation": False,
        "split_or_label_generation": False,
        "filter_manifest_expanded_to_all_variants": True,
        "flypoint_exclusions_expanded_to_all_variants": True,
    }
    (args.output_dir / "augmentation_config.json").write_text(
        json.dumps(augmentation_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    downstream_filter_config = dict(filter_config)
    downstream_filter_config.update(
        {
            "output_dir": str(args.output_dir.resolve()),
            "filtered_parent_dir": str(args.input_dir.resolve()),
            "data_file_rule": "Original__/Aug_01__/Aug_02__*.csv from filtered parents",
            "files_written": len(expected_data_names),
            "pipeline_stage": "post_filter_augmentation",
            "pipeline_order": ["raw", "filter", "augmentation", "data_split"],
            "post_filter_augmentation": True,
            "augmentation_config": "augmentation_config.json",
        }
    )
    (args.output_dir / "filter_config.json").write_text(
        json.dumps(downstream_filter_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    discovered = {
        path.name
        for path in args.output_dir.iterdir()
        if path.is_file() and PREFIXED_DATA_RE.match(path.name)
    }
    listed = set(expanded_filter_manifest["output_file"].astype(str))
    if discovered != expected_data_names or listed != expected_data_names:
        raise AssertionError("Final augmented data files and expanded filter manifest differ")
    if not pd.DataFrame(validation_rows)["status"].eq("PASS").all():
        raise AssertionError("At least one augmentation validation failed")

    print(
        f"Completed and verified: {len(source_files)} filtered parents -> "
        f"{len(expected_data_names)} data CSV files in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
