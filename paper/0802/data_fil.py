#!/usr/bin/env python3
"""Build leakage-safe TE8353 forecasting datasets from ``filtered_data``.

Responsibilities are intentionally separated:

* ``data_filter.py`` cleans the raw parent curves and writes audit manifests;
* ``create.py`` creates temperature-only copies from the filtered parents and
  expands the audit manifests to every output variant;
* this script performs split assignment, common-column extraction, causal
  candidate-feature construction, future-label construction, and PKL
  serialization;
* model-training scripts fit scalers on ``train_clean.pkl`` only.

Default split contract
----------------------
* validation: Original 260118 full curve;
* test: Original 260715, with both event-dense and full-curve outputs;
* training: every other parent, including every BACK parent, with Original,
  Aug_01, and Aug_02 kept in the same split;
* target/label: TE8353 and TE8353(t + 60), respectively;
* no Thv/THV future label is created.

Flypoint look-ahead control
---------------------------
``data_filter.py`` logs temperature samples repaired after a one- or two-step
recovery check.  A single tabular row cannot be both uncorrected at the event
time and corrected in later windows.  To keep downstream training strictly
causal without changing the existing window loader, this script conservatively
removes every repaired row and splits the surrounding data into independent
continuous segments.  Sustained real temperature drops were not repaired by
``data_filter.py`` and are retained here.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "filtered_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "processed_data"

TARGET_COL = "TE8353"
DEFAULT_PREDICT_STEPS = 60
DEFAULT_LOOK_BACK = 60
DEFAULT_VALIDATION_PARENT = "260118"
DEFAULT_TEST_PARENT = "260715"
SAMPLE_PERIOD_SECONDS = 10.0
A_NOMINAL_INLET_TEMPERATURE_K = 4.6
G1_NOMINAL_INLET_TEMPERATURE_K = 80.0
G2_NOMINAL_INLET_TEMPERATURE_K = 300.0
HELIUM_FLUID = "HEOS::Helium"
RECOMMENDED_COOLPROP_VERSION = "8.0.0"
BAR_TO_PA = 1.0e5
G_PER_S_TO_KG_PER_S = 1.0e-3

VALVE_COLS = ("CV8300", "CV8313", "CV8310", "CV8311", "CV8312", "CV8351")
RAW_SIGNAL_COLS = (
    "TE8310",
    "TE8351",
    "TE8352",
    "FT8351",
    "PT8310",
    "PT8351",
    "PT8352",
    "CV8312",
    "CV8311",
    "CV8310",
    "CV8313",
    "CV8300",
    "CV8351",
    "Thv",
    "TE8353",
    "Tef",
    "Tcd",
    "DTbr",
)
# A deliberately compact feature warehouse.  The 15 training candidates are
# grouped for ablation; valve_event_mask is retained only for selecting the
# event-dense diagnostic test set and should not enter the plain-GRU baseline.
FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "thermal_memory": (
        "TE8353_diff_1",
        "TE8353_history_slope_k_per_min_60",
    ),
    "g_main_flow_command": (
        "G1_main_flow_command_proxy",
        "G2_main_flow_command_proxy",
    ),
    # Kept separate because this opening ratio is only a routing hypothesis,
    # not a measured screen-flow fraction.  Its contribution must be tested
    # independently rather than bundled with the G-main features.
    "screen_diversion": (
        "G_screen_diversion_fraction",
    ),
    "g_source_thermal_drive": (
        "G1_main_thermal_drive_proxy",
        "G2_main_thermal_drive_proxy",
    ),
    "a_branch": (
        "A_control_thermal_drive_proxy",
        "CV8351_diff_1",
    ),
    "mainline_state_contrasts": (
        "Mainline_dT_8351_8352",
        "Mainline_dT_8352_8353",
        "Mainline_dP_8351_8352_bar",
    ),
    "coolprop_postmix_capacity": (
        "PostMix_available_cooling_8351_to_TE8353_W",
    ),
    "coolprop_apparent_heat_leak": (
        "Mainline_apparent_heat_leak_8351_8352_W",
        "Mainline_apparent_heat_leak_8352_8353_W",
    ),
    "evaluation_auxiliary": ("valve_event_mask",),
}
ENGINEERED_COLS = tuple(
    dict.fromkeys(column for columns in FEATURE_GROUPS.values() for column in columns)
)
TRAIN_CANDIDATE_COLS = tuple(
    column
    for group, columns in FEATURE_GROUPS.items()
    if group != "evaluation_auxiliary"
    for column in columns
)
FEATURE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "TE8353_diff_1": {
        "formula": "TE8353[t] - TE8353[t-1]",
        "meaning": "latest measured target-temperature increment",
    },
    "TE8353_history_slope_k_per_min_60": {
        "formula": "(TE8353[t] - TE8353[t-59]) / 9.8333 min",
        "meaning": "trailing 10-minute cooling/heating trend",
    },
    "G1_main_flow_command_proxy": {
        "formula": "series(open(CV8300), total_G_outlet_opening) * main_routing_fraction * sqrt(max(PT8310-PT8351,0 bar))",
        "meaning": "G1 main-line flow command under a unit full-open flow-coefficient assumption; not measured g/s",
    },
    "G2_main_flow_command_proxy": {
        "formula": "series(open(CV8313), total_G_outlet_opening) * main_routing_fraction * sqrt(max(PT8310-PT8351,0 bar))",
        "meaning": "G2 main-line flow command under a unit full-open flow-coefficient assumption; not measured g/s",
    },
    "G_screen_diversion_fraction": {
        "formula": "(open(CV8311)+open(CV8312)) / (open(CV8310)+open(CV8311)+open(CV8312))",
        "meaning": "nominal fraction of downstream G-valve conductance directed to screen branches",
    },
    "G1_main_thermal_drive_proxy": {
        "formula": "G1_main_flow_command_proxy * (TE8353 - 80 K)",
        "meaning": "positive when nominal G1 gas is colder than the target and can cool it; negative when it would heat it",
    },
    "G2_main_thermal_drive_proxy": {
        "formula": "G2_main_flow_command_proxy * (TE8353 - 300 K)",
        "meaning": "signed nominal G2 thermal effect; normally negative below 300 K, indicating a heating tendency",
    },
    "A_control_thermal_drive_proxy": {
        "formula": "open(CV8351) * (TE8353 - 4.6 K)",
        "meaning": "A-valve actuation gated by the nominal A-source temperature drive; not flow or power",
    },
    "CV8351_diff_1": {
        "formula": "CV8351[t] - CV8351[t-1]",
        "meaning": "latest A-branch valve action",
    },
    "Mainline_dT_8351_8352": {
        "formula": "TE8352 - TE8351",
        "meaning": "local measured temperature change on the main line",
    },
    "Mainline_dT_8352_8353": {
        "formula": "TE8353 - TE8352",
        "meaning": "downstream local measured temperature change on the main line",
    },
    "Mainline_dP_8351_8352_bar": {
        "formula": "PT8351 - PT8352",
        "meaning": "measured pressure difference along the instrumented main-line section",
    },
    "PostMix_available_cooling_8351_to_TE8353_W": {
        "formula": "FT8351[kg/s] * {h_He(TE8353,PT8351)-h_He(TE8351,PT8351)}",
        "meaning": "signed heat-absorption capacity of the measured post-mixing stream before it reaches the current TE8353 temperature, evaluated at common PT8351; positive means remaining cooling capacity",
    },
    "Mainline_apparent_heat_leak_8351_8352_W": {
        "formula": "FT8351[kg/s] * {h_He(TE8352,PT8352)-h_He(TE8351,PT8351)}",
        "meaning": "steady-flow enthalpy-gain estimate between the two fully instrumented stations; called apparent heat leak because simultaneous transient samples are not the same helium parcel",
    },
    "Mainline_apparent_heat_leak_8352_8353_W": {
        "formula": "FT8351[kg/s] * {h_He(TE8353,PT8352)-h_He(TE8352,PT8352)}",
        "meaning": "downstream apparent heat-leak proxy using PT8352 for TE8353 because no PT8353 measurement is available",
    },
    "valve_event_mask": {
        "formula": "1 if any of six valve openings changes by more than threshold, otherwise 0",
        "meaning": "evaluation-segment selector only; excluded from ordinary GRU inputs",
    },
}
METADATA_COLS = (
    "file_id",
    "source_group",
    "source_variant",
    "parent_id",
    "source_row_index",
    "source_timestamp",
)

DATA_FILE_RE = re.compile(r"^(Original|Aug_01|Aug_02)__(.+\.csv)$", re.IGNORECASE)
PARENT_RE = re.compile(r"-(\d{6})(-BACK)?\.csv$", re.IGNORECASE)
_COOLPROP_RUNTIME: Dict[str, Any] = {}


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    variant: str
    source_group: str
    parent_id: str
    role: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/validation/test PKLs for 60-step TE8353 forecasting."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-parent", default=DEFAULT_VALIDATION_PARENT)
    parser.add_argument("--test-parent", default=DEFAULT_TEST_PARENT)
    parser.add_argument("--predict-steps", type=int, default=DEFAULT_PREDICT_STEPS)
    parser.add_argument("--look-back", type=int, default=DEFAULT_LOOK_BACK)
    parser.add_argument("--valve-event-threshold", type=float, default=0.05)
    parser.add_argument("--dynamic-block-steps", type=int, default=300)
    parser.add_argument("--dynamic-block-fraction", type=float, default=0.35)
    parser.add_argument("--minimum-dynamic-blocks", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing processed_data outputs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build and verify all frames in memory without writing outputs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.predict_steps < 1:
        raise ValueError("--predict-steps must be >= 1")
    if args.look_back < 1:
        raise ValueError("--look-back must be >= 1")
    if args.valve_event_threshold < 0:
        raise ValueError("--valve-event-threshold must be >= 0")
    if args.dynamic_block_steps < 1:
        raise ValueError("--dynamic-block-steps must be >= 1")
    if not 0 < args.dynamic_block_fraction <= 1:
        raise ValueError("--dynamic-block-fraction must be in (0, 1]")
    if args.minimum_dynamic_blocks < 1:
        raise ValueError("--minimum-dynamic-blocks must be >= 1")
    if args.validation_parent.upper().endswith("-BACK"):
        raise ValueError("Validation parent must be a complete non-BACK run")
    if args.test_parent.upper().endswith("-BACK"):
        raise ValueError("Test parent must be a complete non-BACK run")
    if args.validation_parent == args.test_parent:
        raise ValueError("Validation and test parents must differ")


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
        except Exception as exc:
            errors.append(f"{encoding}/{repr(separator)}: {exc}")
    raise ValueError(f"Unable to read {path}. Attempts: {' | '.join(errors)}")


def discover_data_files(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Filtered-data directory does not exist: {input_dir}")
    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and DATA_FILE_RE.fullmatch(path.name)
    )
    if not files:
        raise FileNotFoundError(f"No filtered data CSV files found in {input_dir}")
    return files


def parse_identity(
    path: Path,
    validation_parent: str,
    test_parent: str,
) -> FileIdentity:
    match = DATA_FILE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected data filename: {path.name}")
    variant_raw, source_group = match.groups()
    variant = {
        "original": "Original",
        "aug_01": "Aug_01",
        "aug_02": "Aug_02",
    }[variant_raw.lower()]
    parent_match = PARENT_RE.search(source_group)
    if not parent_match:
        raise ValueError(f"Cannot parse parent date/BACK status from {source_group}")
    parent_id = parent_match.group(1) + ("-BACK" if parent_match.group(2) else "")

    if parent_id == validation_parent:
        role = "validation" if variant == "Original" else "ignored_validation_augmentation"
    elif parent_id == test_parent:
        role = "test" if variant == "Original" else "ignored_test_augmentation"
    else:
        role = "train"

    if parent_id.upper().endswith("-BACK") and role != "train":
        raise AssertionError(f"BACK file escaped the training split: {path.name}")
    return FileIdentity(path, variant, source_group, parent_id, role)


def load_filter_contract(input_dir: Path) -> Dict[str, object]:
    config_path = input_dir / "filter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path.name}; data_fil.py accepts only audited data_filter.py output"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("data_split") is not False or config.get("label_creation") is not False:
        raise ValueError("filtered_data unexpectedly contains split/label operations")
    if config.get("valve_filtering") is not False or config.get("valve_zero_is_valid") is not True:
        raise ValueError("filtered_data violates the valve-preservation contract")
    if config.get("backward_fill") is not False or config.get("centered_filter") is not False:
        raise ValueError("filtered_data contains a non-approved backward/centered filter")
    return config


def load_filter_manifest(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "filter_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing filter manifest: {path}")
    manifest = pd.read_csv(path, encoding="utf-8-sig")
    required = {"output_file", "rows", "protected_columns_exact", "source_unchanged"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"Filter manifest is missing columns: {sorted(missing)}")
    truthy = {"true", "1"}
    for column in ("protected_columns_exact", "source_unchanged"):
        if not manifest[column].astype(str).str.lower().isin(truthy).all():
            raise ValueError(f"Filter manifest contains failed checks in {column}")
    if manifest["output_file"].duplicated().any():
        raise ValueError("Filter manifest contains duplicate output_file rows")
    return manifest


def load_flypoint_exclusions(
    input_dir: Path,
    predict_steps: int,
    filter_config: Mapping[str, object],
) -> Tuple[Dict[str, set[int]], Dict[str, set[int]], pd.DataFrame]:
    path = input_dir / "temperature_flypoint_endpoint_exclusions.csv"
    required_by_contract = bool(filter_config.get("downstream_endpoint_exclusion_required", False))
    if not path.exists():
        if required_by_contract:
            raise FileNotFoundError(
                f"{path.name} is required because filtered temperatures use bounded look-ahead"
            )
        return {}, {}, pd.DataFrame()

    exclusions = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "file",
        "signal",
        "repaired_start_index",
        "recovery_decision_index",
        "unsafe_endpoint_start",
        "unsafe_endpoint_end",
    }
    missing = required - set(exclusions.columns)
    if missing:
        raise KeyError(f"Flypoint exclusion manifest is missing: {sorted(missing)}")

    input_rows: Dict[str, set[int]] = {}
    unsafe_label_origins: Dict[str, set[int]] = {}
    for row in exclusions.itertuples(index=False):
        filename = str(row.file)
        start = int(row.unsafe_endpoint_start)
        end = int(row.unsafe_endpoint_end)
        if end < start:
            raise ValueError(f"Invalid flypoint exclusion range in {filename}: {start}..{end}")
        input_rows.setdefault(filename, set()).update(range(start, end + 1))

        # If TE8353 itself is repaired, labels targeting those repaired rows also
        # depend on the recovery point and must be removed.
        if str(row.signal).upper() == TARGET_COL.upper():
            repaired_start = int(row.repaired_start_index)
            decision = int(row.recovery_decision_index)
            for target_index in range(repaired_start, decision):
                origin = target_index - predict_steps
                if origin >= 0:
                    unsafe_label_origins.setdefault(filename, set()).add(origin)

    return input_rows, unsafe_label_origins, exclusions


def resolve_value_column(frame: pd.DataFrame, canonical_signal: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(canonical_signal)}\s+ValueY$",
        flags=re.IGNORECASE,
    )
    matches = [str(column) for column in frame.columns if pattern.fullmatch(str(column).strip())]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one ValueY column for {canonical_signal}, found {matches}"
        )
    return matches[0]


def resolve_time_column(frame: pd.DataFrame, canonical_signal: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(canonical_signal)}\s+Time$",
        flags=re.IGNORECASE,
    )
    matches = [str(column) for column in frame.columns if pattern.fullmatch(str(column).strip())]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one Time column for {canonical_signal}, found {matches}"
        )
    return matches[0]


def contiguous_runs(mask: np.ndarray) -> List[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    return [chunk for chunk in np.split(indices, split_points) if chunk.size]


def load_coolprop() -> Tuple[Any, str]:
    """Load one frozen helium-property backend for the whole data build."""
    if _COOLPROP_RUNTIME:
        return _COOLPROP_RUNTIME["module"], str(_COOLPROP_RUNTIME["version"])
    try:
        import CoolProp  # type: ignore
        import CoolProp.CoolProp as CP  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CoolProp is required for the helium enthalpy-difference features. "
            f"Install the frozen version with: pip install CoolProp=={RECOMMENDED_COOLPROP_VERSION}"
        ) from exc
    version = str(getattr(CoolProp, "__version__", "unknown"))
    if version != RECOMMENDED_COOLPROP_VERSION:
        print(
            "WARNING: CoolProp version differs from the reproducibility target: "
            f"runtime={version}, recommended={RECOMMENDED_COOLPROP_VERSION}"
        )
    _COOLPROP_RUNTIME.update(module=CP, version=version)
    return CP, version


def helium_hmass_j_per_kg(
    temperature_k: pd.Series,
    pressure_bar: pd.Series,
    state_name: str,
) -> np.ndarray:
    """Return helium mass-specific enthalpy without guessing the fluid phase.

    CoolProp can reject a T-P state extremely close to the saturation line.
    Vector evaluation is used first for speed; recursive bisection isolates an
    invalid row without forward-filling, backward-filling, or imposing a phase
    that is not known from the measurements.
    """
    CP, _ = load_coolprop()
    temperature = temperature_k.to_numpy(dtype=float, copy=False)
    pressure_pa = pressure_bar.to_numpy(dtype=float, copy=False) * BAR_TO_PA
    if temperature.shape != pressure_pa.shape:
        raise ValueError(f"T/P shape mismatch for {state_name}")

    result = np.full(temperature.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(temperature)
        & np.isfinite(pressure_pa)
        & (temperature > 0.0)
        & (pressure_pa > 0.0)
    )
    valid_indices = np.flatnonzero(valid)

    def evaluate(indices: np.ndarray) -> None:
        if indices.size == 0:
            return
        try:
            values = np.asarray(
                CP.PropsSI(
                    "Hmass",
                    "T",
                    temperature[indices],
                    "P",
                    pressure_pa[indices],
                    HELIUM_FLUID,
                ),
                dtype=float,
            ).reshape(-1)
            if values.size != indices.size:
                raise ValueError(
                    f"CoolProp returned {values.size} values for {indices.size} states"
                )
            finite = np.isfinite(values)
            result[indices[finite]] = values[finite]
        except Exception:
            if indices.size == 1:
                return
            midpoint = indices.size // 2
            evaluate(indices[:midpoint])
            evaluate(indices[midpoint:])

    # Moderate batches keep memory bounded and make any recursive failure
    # isolation local.  The Python wrapper accepts equal-length vector inputs.
    for start in range(0, valid_indices.size, 100_000):
        evaluate(valid_indices[start : start + 100_000])
    failed = int(valid_indices.size - np.isfinite(result[valid_indices]).sum())
    if failed:
        print(
            f"WARNING: CoolProp rejected {failed}/{valid_indices.size} states for {state_name}; "
            "those rows will become segment boundaries instead of being imputed."
        )
    return result


def add_causal_features(
    frame: pd.DataFrame,
    valve_event_threshold: float,
) -> pd.DataFrame:
    """Build a compact set of strictly causal candidate features.

    Openings are monotone conductance surrogates, not measured flow.  No
    opening-weighted source temperature is constructed: a physically valid
    mixture temperature requires branch mass flows and helium enthalpies.
    """
    frame = frame.copy()
    valves = frame[list(VALVE_COLS)].astype(float)
    valve_diff = valves.diff().fillna(0.0)
    openings = valves.clip(lower=0.0, upper=100.0) / 100.0
    frame["valve_event_mask"] = (
        valve_diff.abs().gt(valve_event_threshold).any(axis=1).astype(float)
    )

    # Target dynamics and trailing 10-minute trend (60 samples at 10 s).  The
    # rolling mean is deliberately omitted after it produced prediction spikes
    # in the earlier ablation batch.
    frame["TE8353_diff_1"] = frame["TE8353"].diff().fillna(0.0)
    target_delta_60 = (frame["TE8353"] - frame["TE8353"].shift(59)).fillna(0.0)
    elapsed_minutes = 59.0 * SAMPLE_PERIOD_SECONDS / 60.0
    frame["TE8353_history_slope_k_per_min_60"] = target_delta_60 / elapsed_minutes

    # G1/G2 remain separate because their nominal source temperatures differ.
    # A unit full-open flow coefficient is assumed for each valve.  The shared
    # downstream valves are treated as parallel nominal conductances, while
    # each source valve is in series with that downstream network.
    main_outlet = openings["CV8310"]
    screen_outlet = openings["CV8311"] + openings["CV8312"]
    total_outlet = main_outlet + screen_outlet

    def series_proxy(upstream: pd.Series, downstream: pd.Series) -> pd.Series:
        denominator = np.sqrt(upstream.pow(2) + downstream.pow(2))
        result = pd.Series(0.0, index=frame.index, dtype=float)
        active = denominator > 1e-12
        result.loc[active] = (
            upstream.loc[active] * downstream.loc[active] / denominator.loc[active]
        )
        return result

    main_fraction = pd.Series(0.0, index=frame.index, dtype=float)
    screen_fraction = pd.Series(0.0, index=frame.index, dtype=float)
    outlet_active = total_outlet > 1e-12
    main_fraction.loc[outlet_active] = (
        main_outlet.loc[outlet_active] / total_outlet.loc[outlet_active]
    )
    screen_fraction.loc[outlet_active] = (
        screen_outlet.loc[outlet_active] / total_outlet.loc[outlet_active]
    )
    g1_network = series_proxy(openings["CV8300"], total_outlet)
    g2_network = series_proxy(openings["CV8313"], total_outlet)
    positive_g_dp = (frame["PT8310"] - frame["PT8351"]).clip(lower=0.0)
    pressure_drive = np.sqrt(positive_g_dp)
    g1_main_flow_command = g1_network * main_fraction * pressure_drive
    g2_main_flow_command = g2_network * main_fraction * pressure_drive
    frame["G1_main_flow_command_proxy"] = g1_main_flow_command
    frame["G2_main_flow_command_proxy"] = g2_main_flow_command
    frame["G_screen_diversion_fraction"] = screen_fraction

    # Signed source-specific thermal tendencies.  These are deliberately not
    # mixed together: opening G1 (80 K nominal) and G2 (300 K nominal) has a
    # different physical meaning at the same target temperature.
    frame["G1_main_thermal_drive_proxy"] = g1_main_flow_command * (
        frame["TE8353"] - G1_NOMINAL_INLET_TEMPERATURE_K
    )
    frame["G2_main_thermal_drive_proxy"] = g2_main_flow_command * (
        frame["TE8353"] - G2_NOMINAL_INLET_TEMPERATURE_K
    )

    # The A branch has no measured branch flow.  Its opening is therefore used
    # only as a gate on a known temperature driving difference, not as a mass
    # fraction in a mixture-temperature equation.
    frame["A_control_thermal_drive_proxy"] = openings["CV8351"] * (
        frame["TE8353"] - A_NOMINAL_INLET_TEMPERATURE_K
    )
    frame["CV8351_diff_1"] = valve_diff["CV8351"]

    # TE8351 -> TE8352 -> TE8353 is the measured post-mixing main line.  The
    # two local temperature rises and pressure drop remain useful direct state
    # contrasts for an ablation against the nonlinear helium-property terms.
    frame["Mainline_dT_8351_8352"] = frame["TE8352"] - frame["TE8351"]
    frame["Mainline_dT_8352_8353"] = frame["TE8353"] - frame["TE8352"]
    frame["Mainline_dP_8351_8352_bar"] = frame["PT8351"] - frame["PT8352"]

    # FT8351 is already a mass flow in g/s; multiplying it by density again
    # would be dimensionally wrong.  CoolProp Hmass is J/kg, so mdot*delta_h
    # is W after converting g/s to kg/s.  Only enthalpy differences are used;
    # absolute helium enthalpy depends on the selected reference state.
    h_8351_at_p8351 = helium_hmass_j_per_kg(
        frame["TE8351"], frame["PT8351"], "TE8351/PT8351"
    )
    h_8352_at_p8352 = helium_hmass_j_per_kg(
        frame["TE8352"], frame["PT8352"], "TE8352/PT8352"
    )
    h_8353_at_p8351 = helium_hmass_j_per_kg(
        frame["TE8353"], frame["PT8351"], "TE8353/PT8351-reference"
    )
    h_8353_at_p8352 = helium_hmass_j_per_kg(
        frame["TE8353"], frame["PT8352"], "TE8353/PT8352-proxy"
    )
    mass_flow_kg_per_s = frame["FT8351"].to_numpy(dtype=float) * G_PER_S_TO_KG_PER_S
    frame["PostMix_available_cooling_8351_to_TE8353_W"] = mass_flow_kg_per_s * (
        h_8353_at_p8351 - h_8351_at_p8351
    )
    frame["Mainline_apparent_heat_leak_8351_8352_W"] = mass_flow_kg_per_s * (
        h_8352_at_p8352 - h_8351_at_p8351
    )
    frame["Mainline_apparent_heat_leak_8352_8353_W"] = mass_flow_kg_per_s * (
        h_8353_at_p8352 - h_8352_at_p8352
    )

    missing = [column for column in ENGINEERED_COLS if column not in frame.columns]
    if missing:
        raise AssertionError(f"Causal feature builder did not create: {missing}")
    if len(TRAIN_CANDIDATE_COLS) > 15 or len(ENGINEERED_COLS) > 16:
        raise AssertionError("Candidate-feature budget exceeded")
    return frame


def build_file_segments(
    identity: FileIdentity,
    predict_steps: int,
    look_back: int,
    valve_event_threshold: float,
    unsafe_input_rows: Iterable[int],
    unsafe_label_origins: Iterable[int],
) -> Tuple[List[pd.DataFrame], Dict[str, object]]:
    raw = read_industrial_csv(identity.path)
    extracted = pd.DataFrame(index=np.arange(len(raw), dtype=np.int64))
    for signal in RAW_SIGNAL_COLS:
        source_column = resolve_value_column(raw, signal)
        extracted[signal] = pd.to_numeric(raw[source_column], errors="coerce").astype(float)
    timestamp_column = resolve_time_column(raw, TARGET_COL)
    source_timestamp = raw[timestamp_column].astype(str).where(raw[timestamp_column].notna(), "")

    label_col = f"Future_Target_{predict_steps}step"
    extracted[label_col] = extracted[TARGET_COL].shift(-predict_steps)
    feature_finite = np.isfinite(extracted[list(RAW_SIGNAL_COLS)].to_numpy(dtype=float)).all(axis=1)
    label_finite = np.isfinite(extracted[label_col].to_numpy(dtype=float))
    unsafe_input_mask = np.zeros(len(extracted), dtype=bool)
    unsafe_label_mask = np.zeros(len(extracted), dtype=bool)
    unsafe_input_index = np.asarray(sorted(set(unsafe_input_rows)), dtype=np.int64)
    unsafe_label_index = np.asarray(sorted(set(unsafe_label_origins)), dtype=np.int64)
    if unsafe_input_index.size:
        if unsafe_input_index.min() < 0 or unsafe_input_index.max() >= len(extracted):
            raise IndexError(f"Flypoint input exclusion is outside {identity.path.name}")
        unsafe_input_mask[unsafe_input_index] = True
    if unsafe_label_index.size:
        if unsafe_label_index.min() < 0 or unsafe_label_index.max() >= len(extracted):
            raise IndexError(f"Flypoint label exclusion is outside {identity.path.name}")
        unsafe_label_mask[unsafe_label_index] = True

    eligible = feature_finite & label_finite & ~unsafe_input_mask & ~unsafe_label_mask
    runs = contiguous_runs(eligible)
    output_segments: List[pd.DataFrame] = []
    short_segment_rows = 0
    engineered_invalid_rows = 0
    output_segment_number = 0

    for row_indices in runs:
        if len(row_indices) < look_back:
            short_segment_rows += len(row_indices)
            continue
        raw_part = extracted.iloc[row_indices].copy().reset_index(drop=True)
        provisional = add_causal_features(raw_part, valve_event_threshold)
        engineered_finite = np.isfinite(
            provisional[list(ENGINEERED_COLS)].to_numpy(dtype=float)
        ).all(axis=1)
        engineered_invalid_rows += int((~engineered_finite).sum())

        # A rejected helium T-P state becomes a hard segment boundary.  Causal
        # differences/trends are then recomputed inside each final segment so
        # no retained row depends on the removed state immediately before it.
        for local_indices in contiguous_runs(engineered_finite):
            if len(local_indices) < look_back:
                short_segment_rows += len(local_indices)
                continue
            source_indices = row_indices[local_indices]
            if engineered_finite.all():
                # Normal fast path: reuse the already computed CoolProp arrays.
                part = provisional
            else:
                part = extracted.iloc[source_indices].copy().reset_index(drop=True)
                part = add_causal_features(part, valve_event_threshold)
            if not np.isfinite(
                part[list(ENGINEERED_COLS)].to_numpy(dtype=float)
            ).all():
                raise AssertionError(
                    f"Non-finite engineered feature survived final split in {identity.path.name}"
                )
            part["file_id"] = (
                f"{identity.path.stem}__segment_{output_segment_number:03d}"
            )
            output_segment_number += 1
            part["source_group"] = identity.source_group
            part["source_variant"] = identity.variant
            part["parent_id"] = identity.parent_id
            part["source_row_index"] = [str(index) for index in source_indices]
            part["source_timestamp"] = source_timestamp.iloc[source_indices].to_numpy()
            part = part[
                [
                    *RAW_SIGNAL_COLS,
                    *ENGINEERED_COLS,
                    label_col,
                    *METADATA_COLS,
                ]
            ]
            output_segments.append(part)

    rows_output = sum(len(segment) for segment in output_segments)
    report = {
        "file": identity.path.name,
        "source_group": identity.source_group,
        "parent_id": identity.parent_id,
        "variant": identity.variant,
        "role": identity.role,
        "rows_input": len(raw),
        "rows_output": rows_output,
        "segments_output": len(output_segments),
        "rows_missing_raw_feature": int((~feature_finite).sum()),
        "rows_missing_engineered_feature": engineered_invalid_rows,
        "rows_missing_future_label": int((~label_finite).sum()),
        "rows_flypoint_input_excluded": int(unsafe_input_mask.sum()),
        "rows_flypoint_label_origin_excluded": int(unsafe_label_mask.sum()),
        "rows_in_short_segments_removed": short_segment_rows,
    }
    if not output_segments:
        raise ValueError(f"No usable segment remains in {identity.path.name}: {report}")
    return output_segments, report


def concatenate_segments(segments: Sequence[pd.DataFrame], split_name: str) -> pd.DataFrame:
    if not segments:
        raise ValueError(f"No segments were assigned to {split_name}")
    return pd.concat(segments, ignore_index=True)


def merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def select_dynamic_test_segments(
    test_full: pd.DataFrame,
    look_back: int,
    block_steps: int,
    block_fraction: float,
    minimum_blocks: int,
) -> pd.DataFrame:
    """Select valve-event-dense origin blocks while retaining causal history.

    For a selected origin block [start, end), the copied segment begins at
    start-(look_back-1).  The existing WindowDataset therefore evaluates
    exactly the selected block after consuming its historical context.
    """
    selected_segments: List[pd.DataFrame] = []
    for source_file_id, part in test_full.groupby("file_id", sort=False):
        part = part.reset_index(drop=True)
        candidates: List[Tuple[float, int, int]] = []
        for start in range(0, len(part), block_steps):
            end = min(start + block_steps, len(part))
            event_count = float(part.loc[start : end - 1, "valve_event_mask"].sum())
            candidates.append((event_count, start, end))
        if not candidates:
            continue
        select_count = min(
            len(candidates),
            max(minimum_blocks, int(math.ceil(len(candidates) * block_fraction))),
        )
        chosen = sorted(candidates, key=lambda item: (-item[0], item[1]))[:select_count]
        origin_intervals = merge_intervals([(start, end) for _, start, end in chosen])

        for dynamic_number, (origin_start, origin_end) in enumerate(origin_intervals):
            history_start = max(0, origin_start - (look_back - 1))
            segment = part.iloc[history_start:origin_end].copy().reset_index(drop=True)
            if len(segment) < look_back:
                continue
            segment["file_id"] = f"{source_file_id}__dynamic_{dynamic_number:03d}"
            selected_segments.append(segment)

    if not selected_segments:
        raise ValueError("No event-dense test segment contains a complete history window")
    return pd.concat(selected_segments, ignore_index=True)


def count_windows(frame: pd.DataFrame, look_back: int) -> int:
    return int(
        sum(max(0, len(part) - look_back + 1) for _, part in frame.groupby("file_id", sort=False))
    )


def assert_frame_schema(frame: pd.DataFrame, split_name: str, label_col: str) -> None:
    expected = [*RAW_SIGNAL_COLS, *ENGINEERED_COLS, label_col, *METADATA_COLS]
    if list(frame.columns) != expected:
        raise AssertionError(f"Unexpected {split_name} schema")
    numeric_columns = [*RAW_SIGNAL_COLS, *ENGINEERED_COLS, label_col]
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise AssertionError(f"{split_name} contains NaN/inf numeric values")
    if any(str(column).startswith("Future_Target_") and column != label_col for column in frame.columns):
        raise AssertionError(f"{split_name} contains an unexpected future label")
    if any("THV" in str(column).upper() and str(column).startswith("Future_") for column in frame.columns):
        raise AssertionError(f"{split_name} contains a forbidden Thv future label")
    for file_id, indices in frame.groupby("file_id", sort=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) and not np.all(np.diff(idx) == 1):
            raise AssertionError(f"Rows for file_id {file_id} are not contiguous in {split_name}")
        if frame.iloc[idx]["source_group"].nunique() != 1:
            raise AssertionError(f"file_id {file_id} maps to multiple source groups")


def assert_split_integrity(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    test_full: pd.DataFrame,
    validation_parent: str,
    test_parent: str,
    label_col: str,
) -> None:
    for name, frame in (
        ("train", train),
        ("validation", val),
        ("dynamic_test", test),
        ("full_test", test_full),
    ):
        assert_frame_schema(frame, name, label_col)

    train_groups = set(train["source_group"].unique())
    val_groups = set(val["source_group"].unique())
    test_groups = set(test_full["source_group"].unique())
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise AssertionError("Parent/source_group overlap across train/validation/test")
    if set(val["parent_id"].unique()) != {validation_parent}:
        raise AssertionError(f"Validation is not exclusively parent {validation_parent}")
    if set(test_full["parent_id"].unique()) != {test_parent}:
        raise AssertionError(f"Test is not exclusively parent {test_parent}")
    if set(test["source_group"].unique()) != test_groups:
        raise AssertionError("Dynamic and full test source groups differ")
    if set(val["source_variant"].unique()) != {"Original"}:
        raise AssertionError("Validation contains augmented data")
    if set(test_full["source_variant"].unique()) != {"Original"}:
        raise AssertionError("Test contains augmented data")
    back_rows = train[train["parent_id"].astype(str).str.upper().str.endswith("-BACK")]
    if back_rows.empty:
        raise AssertionError("No BACK data reached the training split")
    if not set(back_rows["source_variant"].unique()).issubset({"Original", "Aug_01", "Aug_02"}):
        raise AssertionError("Unexpected BACK source variant")


def dataset_summary(frame: pd.DataFrame, name: str, look_back: int) -> Dict[str, object]:
    return {
        "dataset": name,
        "rows": len(frame),
        "segments": int(frame["file_id"].nunique()),
        "windows": count_windows(frame, look_back),
        "source_groups": int(frame["source_group"].nunique()),
        "parent_ids": ";".join(sorted(map(str, frame["parent_id"].unique()))),
        "variants": ";".join(sorted(map(str, frame["source_variant"].unique()))),
        "valve_event_rows": int(frame["valve_event_mask"].sum()),
        "valve_event_ratio": float(frame["valve_event_mask"].mean()),
    }


def dump_pickle_compatible(obj: object, path: Path) -> str:
    """Use joblib when available; plain pickle remains readable by joblib.load."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        import joblib  # type: ignore

        joblib.dump(obj, temporary)
        writer = "joblib"
    except ImportError:
        with temporary.open("wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        writer = "pickle_fallback"
    temporary.replace(path)
    return writer


def ensure_output_targets(output_dir: Path, overwrite: bool) -> None:
    targets = (
        "train_clean.pkl",
        "val_clean.pkl",
        "test_clean.pkl",
        "test_full_clean.pkl",
        "split_manifest.csv",
        "dataset_summary.csv",
        "data_build_config.json",
        "feature_catalog.json",
    )
    conflicts = [output_dir / name for name in targets if (output_dir / name).exists()]
    if conflicts and not overwrite:
        preview = ", ".join(path.name for path in conflicts[:4])
        raise FileExistsError(
            f"Processed outputs already exist ({preview}). Use --overwrite to replace them."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    label_col = f"Future_Target_{args.predict_steps}step"

    filter_config = load_filter_contract(args.input_dir)
    filter_manifest = load_filter_manifest(args.input_dir)
    files = discover_data_files(args.input_dir)
    manifest_files = set(filter_manifest["output_file"].astype(str))
    discovered_names = {path.name for path in files}
    if manifest_files != discovered_names:
        missing = sorted(discovered_names - manifest_files)
        extra = sorted(manifest_files - discovered_names)
        raise AssertionError(f"Filter manifest/data mismatch; missing={missing}, extra={extra}")

    unsafe_input, unsafe_label, flypoint_exclusions = load_flypoint_exclusions(
        args.input_dir, args.predict_steps, filter_config
    )
    identities = [
        parse_identity(path, args.validation_parent, args.test_parent) for path in files
    ]
    if sum(item.role == "validation" for item in identities) != 1:
        raise AssertionError("Expected exactly one Original validation file")
    if sum(item.role == "test" for item in identities) != 1:
        raise AssertionError("Expected exactly one Original test file")

    train_segments: List[pd.DataFrame] = []
    val_segments: List[pd.DataFrame] = []
    test_full_segments: List[pd.DataFrame] = []
    file_reports: List[Dict[str, object]] = []
    manifest_rows = filter_manifest.set_index("output_file")["rows"].to_dict()

    used_identities = [item for item in identities if not item.role.startswith("ignored_")]
    for file_number, identity in enumerate(used_identities, start=1):
        print(f"[{file_number}/{len(used_identities)}] {identity.role}: {identity.path.name}")
        segments, report = build_file_segments(
            identity,
            predict_steps=args.predict_steps,
            look_back=args.look_back,
            valve_event_threshold=args.valve_event_threshold,
            unsafe_input_rows=unsafe_input.get(identity.path.name, set()),
            unsafe_label_origins=unsafe_label.get(identity.path.name, set()),
        )
        expected_rows = int(manifest_rows[identity.path.name])
        if int(report["rows_input"]) != expected_rows:
            raise AssertionError(
                f"Row-count mismatch for {identity.path.name}: "
                f"CSV={report['rows_input']}, filter_manifest={expected_rows}"
            )
        file_reports.append(report)
        if identity.role == "train":
            train_segments.extend(segments)
        elif identity.role == "validation":
            val_segments.extend(segments)
        elif identity.role == "test":
            test_full_segments.extend(segments)
        else:
            raise AssertionError(f"Unhandled role: {identity.role}")

    for identity in identities:
        if identity.role.startswith("ignored_"):
            file_reports.append(
                {
                    "file": identity.path.name,
                    "source_group": identity.source_group,
                    "parent_id": identity.parent_id,
                    "variant": identity.variant,
                    "role": identity.role,
                    "rows_input": int(manifest_rows[identity.path.name]),
                    "rows_output": 0,
                    "segments_output": 0,
                    "rows_missing_raw_feature": 0,
                    "rows_missing_engineered_feature": 0,
                    "rows_missing_future_label": 0,
                    "rows_flypoint_input_excluded": 0,
                    "rows_flypoint_label_origin_excluded": 0,
                    "rows_in_short_segments_removed": 0,
                }
            )

    train = concatenate_segments(train_segments, "train")
    val = concatenate_segments(val_segments, "validation")
    test_full = concatenate_segments(test_full_segments, "full_test")
    test_dynamic = select_dynamic_test_segments(
        test_full,
        look_back=args.look_back,
        block_steps=args.dynamic_block_steps,
        block_fraction=args.dynamic_block_fraction,
        minimum_blocks=args.minimum_dynamic_blocks,
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test_dynamic = test_dynamic.reset_index(drop=True)
    test_full = test_full.reset_index(drop=True)

    assert_split_integrity(
        train,
        val,
        test_dynamic,
        test_full,
        validation_parent=args.validation_parent,
        test_parent=args.test_parent,
        label_col=label_col,
    )

    summaries = [
        dataset_summary(train, "train_clean", args.look_back),
        dataset_summary(val, "val_clean_full_original", args.look_back),
        dataset_summary(test_dynamic, "test_clean_dynamic", args.look_back),
        dataset_summary(test_full, "test_full_clean", args.look_back),
    ]
    print(pd.DataFrame(summaries).to_string(index=False))
    used_file_names = {identity.path.name for identity in used_identities}
    used_flypoint_events = int(
        flypoint_exclusions["file"].astype(str).isin(used_file_names).sum()
    ) if not flypoint_exclusions.empty else 0
    used_flypoint_rows = sum(
        len(unsafe_input.get(filename, set())) for filename in used_file_names
    )
    print(
        "Flypoint exclusions consumed: "
        f"used_events={used_flypoint_events}, used_rows={used_flypoint_rows}, "
        f"manifest_events={len(flypoint_exclusions)}"
    )

    if args.validate_only:
        print("Validation-only run complete; no files were written.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_output_targets(args.output_dir, args.overwrite)
    writers = {
        "train_clean.pkl": dump_pickle_compatible(train, args.output_dir / "train_clean.pkl"),
        "val_clean.pkl": dump_pickle_compatible(val, args.output_dir / "val_clean.pkl"),
        "test_clean.pkl": dump_pickle_compatible(
            test_dynamic, args.output_dir / "test_clean.pkl"
        ),
        "test_full_clean.pkl": dump_pickle_compatible(
            test_full, args.output_dir / "test_full_clean.pkl"
        ),
    }
    pd.DataFrame(file_reports).sort_values(["role", "parent_id", "variant"]).to_csv(
        args.output_dir / "split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "dataset_summary.csv", index=False, encoding="utf-8-sig"
    )
    config = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "target": TARGET_COL,
        "future_label": label_col,
        "thv_future_label_created": False,
        "look_back": args.look_back,
        "predict_steps": args.predict_steps,
        "validation_parent": args.validation_parent,
        "validation_variant": "Original only",
        "test_parent": args.test_parent,
        "test_variant": "Original only",
        "back_policy": "all BACK parents and variants are training only",
        "training_augmentation_policy": "Original + Aug_01 + Aug_02 for training parents",
        "raw_signal_columns": list(RAW_SIGNAL_COLS),
        "engineered_columns": list(ENGINEERED_COLS),
        "engineered_feature_groups": {
            name: list(columns) for name, columns in FEATURE_GROUPS.items()
        },
        "metadata_columns": list(METADATA_COLS),
        "scaler_fitting": "not performed here; training script fits train_clean.pkl only",
        "additional_filtering": False,
        "thermophysical_properties": {
            "library": "CoolProp",
            "runtime_version": load_coolprop()[1],
            "recommended_version": RECOMMENDED_COOLPROP_VERSION,
            "fluid": HELIUM_FLUID,
            "temperature_input_unit": "K",
            "pressure_source_unit": "bar",
            "pressure_coolprop_unit": "Pa",
            "mass_flow_source_unit": "g/s",
            "mass_flow_power_unit": "kg/s",
            "enthalpy_output_unit": "J/kg",
            "power_output_unit": "W",
            "phase_policy": (
                "automatic phase detection; rejected T-P states are removed as segment boundaries; "
                "no phase is imposed and no property value is filled"
            ),
        },
        "backward_fill": False,
        "future_event_neighborhood": False,
        "flypoint_strategy": (
            "all repaired input rows are removed and segment boundaries are created; "
            "TE8353 repaired-label origins are also removed"
        ),
        "flypoint_exclusion_events_in_manifest": len(flypoint_exclusions),
        "flypoint_exclusion_events_in_used_files": used_flypoint_events,
        "flypoint_exclusion_rows_in_used_files": used_flypoint_rows,
        "dynamic_test": {
            "block_steps": args.dynamic_block_steps,
            "selected_fraction": args.dynamic_block_fraction,
            "minimum_blocks": args.minimum_dynamic_blocks,
            "selection_signal": "causal six-valve event mask only",
        },
        "pickle_writers": writers,
    }
    (args.output_dir / "data_build_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    feature_catalog = {
        "policy": (
            "data_fil.py materializes all strictly causal candidate columns; "
            "training scripts select ablation groups and fit scalers on train_clean.pkl only"
        ),
        "sample_period_seconds": SAMPLE_PERIOD_SECONDS,
        "raw_signal_columns": list(RAW_SIGNAL_COLS),
        "feature_groups": {name: list(columns) for name, columns in FEATURE_GROUPS.items()},
        "training_candidate_columns": list(TRAIN_CANDIDATE_COLS),
        "training_candidate_count": len(TRAIN_CANDIDATE_COLS),
        "evaluation_only_columns": list(FEATURE_GROUPS["evaluation_auxiliary"]),
        "feature_definitions": FEATURE_DEFINITIONS,
        "metadata_columns": list(METADATA_COLS),
        "thermophysical_properties": {
            "library": "CoolProp",
            "runtime_version": load_coolprop()[1],
            "recommended_version": RECOMMENDED_COOLPROP_VERSION,
            "fluid": HELIUM_FLUID,
            "calculation": "FT8351[kg/s] * helium mass-specific enthalpy difference[J/kg]",
            "phase_policy": (
                "automatic phase detection; invalid T-P states are excluded without imputation"
            ),
        },
        "not_materialized": {
            "true_pre_mixing_branch_power": (
                "G1/G2/A branch mass flows and calibrated valve Cv curves are unavailable; "
                "the upstream valve terms remain command proxies, not watts"
            ),
            "train_fitted_statistics": (
                "scalers, learned lags, thresholds, PCA and selection must be fitted on training only"
            ),
            "future_control_features": (
                "future valve trajectories belong to a later MPC dataset, not traditional forecasting"
            ),
        },
        "nominal_constants": {
            "A_inlet_temperature_k": A_NOMINAL_INLET_TEMPERATURE_K,
            "G1_inlet_temperature_k": G1_NOMINAL_INLET_TEMPERATURE_K,
            "G2_inlet_temperature_k": G2_NOMINAL_INLET_TEMPERATURE_K,
        },
        "caveats": {
            "units": "FT8351=g/s; PT*=bar; TE*=K",
            "valve_openings": (
                "dimensionless monotone conductance surrogates with unit full-open coefficients; "
                "not calibrated Cv and not measured flow"
            ),
            "mixture_temperature": (
                "not constructed because valve opening is not branch mass flow; a valid value "
                "requires mass-flow and helium-enthalpy weighting"
            ),
            "FT8351": (
                "measured mass flow in g/s and converted to kg/s before multiplication by "
                "CoolProp enthalpy differences"
            ),
            "postmix_capacity": (
                "a signed current-state cooling-capacity feature relative to TE8353 at PT8351; "
                "not the plant's total refrigeration duty"
            ),
            "apparent_heat_leak": (
                "steady-flow estimate; in transients, simultaneous spatial sensors observe "
                "different helium parcels and unknown transport delay/sensor lag remains"
            ),
            "PT8310_minus_PT8351": "downstream pressure surrogate, not a measured drop across CV8310",
            "G_screen_diversion_fraction": (
                "downstream valve-opening diversion fraction; not a measured branch-flow fraction; "
                "isolated as its own low-priority ablation group"
            ),
        },
    }
    (args.output_dir / "feature_catalog.json").write_text(
        json.dumps(feature_catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Processed datasets saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
