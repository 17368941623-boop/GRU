#!/usr/bin/env python3
"""Build leakage-safe forecasting tables for TE8353 and Thv.

Pipeline order
--------------
1. ``data_filter.py`` filters the raw parent curves.
2. ``create.py`` creates two temperature-only augmentations per parent.
3. This script assigns whole parent curves to train/validation/test, constructs
   strictly causal features, creates future labels, and writes PKL tables.
4. Training scripts choose feature groups and fit every scaler/statistic from
   ``train_clean.pkl`` only.

Default split (fixed by experiment identity, never by random rows)
-----------------------------------------------------------------
* validation: Original 260501 only;
* test: Original 0715-BACK only;
* training: every other parent, including 0715-ALL, 0623-BACK and 260118, with
  Original/Aug_01/Aug_02 kept together;
* augmented copies of validation and test are ignored.

The output contains both TE8353 and Thv labels at 5/10/15/20/30 steps.  Future label
columns are never used to construct an input feature.  EC-V2 and COOLDOWN are
optional because 0617-ALL has no valid measurements for them; their missing
values remain NaN and availability masks are written instead of replacing
missing values with the valid physical state 0 %.
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

DEFAULT_VALIDATION_PARENT = "260501"
DEFAULT_TEST_PARENT = "0715-BACK"
DEFAULT_HORIZONS = (5, 10, 15, 20, 30)
DEFAULT_LOOK_BACK = 60
SAMPLE_PERIOD_SECONDS = 10.0

A_NOMINAL_INLET_TEMPERATURE_K = 4.6
G1_NOMINAL_INLET_TEMPERATURE_K = 80.0
G2_NOMINAL_INLET_TEMPERATURE_K = 300.0
REFERENCE_TEMPERATURE_K = 300.0

HELIUM_FLUID = "HEOS::Helium"
RECOMMENDED_COOLPROP_VERSION = "8.0.0"
BAR_TO_PA = 1.0e5
G_PER_S_TO_KG_PER_S = 1.0e-3

TARGET_COLS = ("TE8353", "Thv")
REQUIRED_RAW_SIGNAL_COLS = (
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
OPTIONAL_MODULE_VALVE_COLS = ("EC-V2", "COOLDOWN")
RAW_SIGNAL_COLS = (*REQUIRED_RAW_SIGNAL_COLS, *OPTIONAL_MODULE_VALVE_COLS)
PRIMARY_VALVE_COLS = (
    "CV8300",
    "CV8313",
    "CV8310",
    "CV8311",
    "CV8312",
    "CV8351",
    "EC-V2",
    "COOLDOWN",
)
TECHNICAL_MASK_COLS = (
    "EC_V2_available_mask",
    "COOLDOWN_available_mask",
)

SIGNAL_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "Thv": ("Thv", "THV"),
    "COOLDOWN": ("COOLDOWN", "Cooldowm"),
}

FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "target_history_dynamics": (
        "TE8353_diff_1",
        "TE8353_history_slope_k_per_min_60points",
        "Thv_diff_1",
        "Thv_history_slope_k_per_min_60points",
    ),
    "g_main_flow_command": (
        "G1_main_flow_command_proxy",
        "G2_main_flow_command_proxy",
    ),
    "premix_postmix_thermal_drive": (
        "G1_to_postmix_thermal_drive_proxy",
        "G2_to_postmix_thermal_drive_proxy",
        "A_to_postmix_thermal_drive_proxy",
        "CV8351_diff_1",
    ),
    "postmix_cooling_bridge": (
        "PostMix_specific_cooling_300K_ref_J_kg",
        "PostMix_cooling_rate_300K_ref_W",
    ),
    "mainline_state_contrasts": (
        "Mainline_dT_8351_8352",
        "Mainline_dT_8352_8353",
        "Mainline_dP_8351_8352_bar",
    ),
    "thv_module_coupling": (
        "Module_dT_Thv_TE8353",
        "Module_flow_temperature_drive_proxy",
        "Module_dT_Thv_TE8310",
    ),
    "module_valve_dynamics": (
        "EC_V2_diff_1",
        "COOLDOWN_diff_1",
    ),
    "optional_coolprop_apparent_heat_leak": (
        "Mainline_apparent_heat_leak_8351_8352_W",
        "Mainline_apparent_heat_leak_8352_8353_W",
    ),
    "evaluation_auxiliary": ("valve_event_mask",),
}
ENGINEERED_COLS = tuple(
    dict.fromkeys(column for columns in FEATURE_GROUPS.values() for column in columns)
)
RECOMMENDED_ENGINEERED_COLS = tuple(
    column
    for group, columns in FEATURE_GROUPS.items()
    if group not in {"optional_coolprop_apparent_heat_leak", "evaluation_auxiliary"}
    for column in columns
)
OPTIONAL_NAN_COLS = (
    *OPTIONAL_MODULE_VALVE_COLS,
    *FEATURE_GROUPS["module_valve_dynamics"],
)
FINITE_ENGINEERED_COLS = tuple(
    column for column in ENGINEERED_COLS if column not in FEATURE_GROUPS["module_valve_dynamics"]
)

FEATURE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "TE8353_diff_1": {
        "formula": "TE8353[t] - TE8353[t-1]",
        "meaning": "latest observed TE8353 increment",
    },
    "TE8353_history_slope_k_per_min_60points": {
        "formula": "(TE8353[t] - TE8353[t-59]) / (590/60 min)",
        "meaning": "causal 60-point TE8353 trend; 60 points span 59 ten-second intervals",
    },
    "Thv_diff_1": {
        "formula": "Thv[t] - Thv[t-1]",
        "meaning": "latest observed module-average-temperature increment",
    },
    "Thv_history_slope_k_per_min_60points": {
        "formula": "(Thv[t] - Thv[t-59]) / (590/60 min)",
        "meaning": "causal 60-point Thv trend",
    },
    "G1_main_flow_command_proxy": {
        "formula": "series(open(CV8300),open(CV8310))*sqrt(max(PT8310-PT8351,0))",
        "meaning": "G1 main-line conductance/pressure-drive proxy, not measured mass flow",
    },
    "G2_main_flow_command_proxy": {
        "formula": "series(open(CV8313),open(CV8310))*sqrt(max(PT8310-PT8351,0))",
        "meaning": "G2 main-line conductance/pressure-drive proxy, not measured mass flow",
    },
    "G1_to_postmix_thermal_drive_proxy": {
        "formula": "G1_main_flow_command_proxy*(TE8351-80 K)",
        "meaning": "signed nominal G1-to-postmix thermal drive",
    },
    "G2_to_postmix_thermal_drive_proxy": {
        "formula": "G2_main_flow_command_proxy*(TE8351-300 K)",
        "meaning": "signed nominal G2-to-postmix thermal drive",
    },
    "A_to_postmix_thermal_drive_proxy": {
        "formula": "open(CV8351)*(TE8351-4.6 K)",
        "meaning": "A-branch command gated by its nominal source/postmix temperature drive",
    },
    "CV8351_diff_1": {
        "formula": "CV8351[t]-CV8351[t-1]",
        "meaning": "latest A-branch valve action",
    },
    "PostMix_specific_cooling_300K_ref_J_kg": {
        "formula": "h_He(300 K,PT8351)-h_He(TE8351,PT8351)",
        "meaning": "postmix helium specific cooling content relative to a 300 K reference",
    },
    "PostMix_cooling_rate_300K_ref_W": {
        "formula": "FT8351[kg/s]*PostMix_specific_cooling_300K_ref_J_kg",
        "meaning": "flow-weighted postmix cooling proxy, not rated refrigerator capacity",
    },
    "Mainline_dT_8351_8352": {
        "formula": "TE8351-TE8352",
        "meaning": "measured temperature contrast on the first main-line section",
    },
    "Mainline_dT_8352_8353": {
        "formula": "TE8352-TE8353",
        "meaning": "measured temperature contrast on the second main-line section",
    },
    "Mainline_dP_8351_8352_bar": {
        "formula": "PT8351-PT8352",
        "meaning": "measured main-line pressure contrast in bar",
    },
    "Module_dT_Thv_TE8353": {
        "formula": "Thv-TE8353",
        "meaning": "module-average versus module-inlet temperature contrast",
    },
    "Module_flow_temperature_drive_proxy": {
        "formula": "FT8351*(Thv-TE8353)",
        "meaning": "flow/temperature-drive proxy in g*K/s; not thermal power",
    },
    "Module_dT_Thv_TE8310": {
        "formula": "Thv-TE8310",
        "meaning": "temperature contrast used by the equipment state machine",
    },
    "EC_V2_diff_1": {
        "formula": "EC-V2[t]-EC-V2[t-1] within an available contiguous sequence",
        "meaning": "latest EC-V2 action; unavailable values remain missing",
    },
    "COOLDOWN_diff_1": {
        "formula": "COOLDOWN[t]-COOLDOWN[t-1] within an available contiguous sequence",
        "meaning": "latest COOLDOWN action; Cooldowm is accepted as a source alias",
    },
    "Mainline_apparent_heat_leak_8351_8352_W": {
        "formula": "FT8351[kg/s]*(h(TE8352,PT8352)-h(TE8351,PT8351))",
        "meaning": "optional transient apparent enthalpy-gain proxy",
    },
    "Mainline_apparent_heat_leak_8352_8353_W": {
        "formula": "FT8351[kg/s]*(h(TE8353,PT8352)-h(TE8352,PT8352))",
        "meaning": "optional downstream apparent enthalpy-gain proxy using PT8352",
    },
    "valve_event_mask": {
        "formula": "1 when any available primary valve changes more than the threshold",
        "meaning": "dynamic-test selector only; never a default model input",
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
        description=(
            "Build leakage-safe TE8353/Thv 5/10/15/20/30-step forecasting PKLs."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-parent", default=DEFAULT_VALIDATION_PARENT)
    parser.add_argument("--test-parent", default=DEFAULT_TEST_PARENT)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Future horizons in samples; default: 5 10 15 20 30.",
    )
    parser.add_argument("--look-back", type=int, default=DEFAULT_LOOK_BACK)
    parser.add_argument("--valve-event-threshold", type=float, default=0.05)
    parser.add_argument("--dynamic-block-steps", type=int, default=300)
    parser.add_argument("--dynamic-block-fraction", type=float, default=0.35)
    parser.add_argument("--minimum-dynamic-blocks", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    horizons = tuple(sorted(set(args.horizons)))
    if not horizons or horizons[0] < 1:
        raise ValueError("--horizons must contain positive integers")
    args.horizons = horizons
    if args.look_back < 1:
        raise ValueError("--look-back must be >= 1")
    if args.valve_event_threshold < 0:
        raise ValueError("--valve-event-threshold must be >= 0")
    if args.dynamic_block_steps < 1:
        raise ValueError("--dynamic-block-steps must be >= 1")
    if not 0 < args.dynamic_block_fraction <= 1:
        raise ValueError("--dynamic-block-fraction must be in (0,1]")
    if args.minimum_dynamic_blocks < 1:
        raise ValueError("--minimum-dynamic-blocks must be >= 1")
    if args.validation_parent == args.test_parent:
        raise ValueError("Validation and test parents must differ")
    if "BACK" in args.validation_parent.upper():
        raise ValueError("Validation must be a complete non-BACK parent")


def future_label_columns(horizons: Sequence[int]) -> Tuple[str, ...]:
    return tuple(
        f"Future_{target}_{step}step"
        for target in TARGET_COLS
        for step in horizons
    )


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
                frame.columns = (
                    frame.columns.astype(str)
                    .str.replace('"', "", regex=False)
                    .str.strip()
                )
                return frame
        except Exception as exc:
            errors.append(f"{encoding}/{separator!r}: {exc}")
    raise ValueError(f"Unable to read {path}. Attempts: {' | '.join(errors)}")


def discover_data_files(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Filtered-data directory does not exist: {input_dir}")
    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and DATA_FILE_RE.fullmatch(path.name)
    )
    if not files:
        raise FileNotFoundError(f"No filtered data CSV files found in {input_dir}")
    return files


def canonical_parent_id(source_group: str) -> str:
    name = Path(source_group).stem.lower().replace("_", "-")
    if "0715back" in name or ("0715" in name and "back" in name):
        return "0715-BACK"
    if "0623" in name and "back" in name:
        return "0623-BACK"
    if "0617" in name and "all" in name:
        return "0617-ALL"
    if "0715" in name and "all" in name:
        return "0715-ALL"
    six_digit = re.findall(r"(?<!\d)(\d{6})(?!\d)", name)
    if len(six_digit) == 1:
        return six_digit[0]
    raise ValueError(f"Cannot identify parent experiment from {source_group}")


def parse_identity(path: Path, validation_parent: str, test_parent: str) -> FileIdentity:
    match = DATA_FILE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected data filename: {path.name}")
    variant_raw, source_group = match.groups()
    variant = {
        "original": "Original",
        "aug_01": "Aug_01",
        "aug_02": "Aug_02",
    }[variant_raw.lower()]
    parent_id = canonical_parent_id(source_group)
    if parent_id == validation_parent:
        role = "validation" if variant == "Original" else "ignored_validation_augmentation"
    elif parent_id == test_parent:
        role = "test" if variant == "Original" else "ignored_test_augmentation"
    else:
        role = "train"
    if "BACK" in parent_id.upper() and role == "validation":
        raise AssertionError(f"BACK file must not be validation: {path.name}")
    return FileIdentity(path, variant, source_group, parent_id, role)


def load_filter_contract(input_dir: Path) -> Dict[str, object]:
    path = input_dir / "filter_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing audited filter contract: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("data_split") is not False or config.get("label_creation") is not False:
        raise ValueError("filtered_data unexpectedly contains split or label operations")
    if config.get("valve_smoothing") is not False:
        raise ValueError("filtered_data violates the no-valve-smoothing contract")
    if config.get("valve_negative_clamp") is not True:
        raise ValueError("filtered_data did not apply the approved valve <0 -> 0 rule")
    if float(config.get("valve_minimum", float("nan"))) != 0.0:
        raise ValueError("filtered_data valve minimum is not 0")
    if config.get("valve_zero_is_valid") is not True:
        raise ValueError("filtered_data does not preserve legal 0% valve states")
    if config.get("backward_fill") is not False or config.get("centered_filter") is not False:
        raise ValueError("filtered_data contains an unapproved backward/centered filter")
    return config


def load_filter_manifest(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "filter_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing filter manifest: {path}")
    manifest = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "output_file", "rows", "protected_columns_contract_ok", "source_unchanged"
    }
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"Filter manifest is missing columns: {sorted(missing)}")
    truthy = {"true", "1"}
    for column in ("protected_columns_contract_ok", "source_unchanged"):
        if not manifest[column].astype(str).str.lower().isin(truthy).all():
            raise ValueError(f"Filter manifest contains failed checks in {column}")
    if manifest["output_file"].duplicated().any():
        raise ValueError("Filter manifest contains duplicate output_file rows")
    return manifest


def load_flypoint_exclusions(
    input_dir: Path,
    horizons: Sequence[int],
    filter_config: Mapping[str, object],
) -> Tuple[Dict[str, set[int]], Dict[str, set[int]], pd.DataFrame]:
    path = input_dir / "temperature_flypoint_endpoint_exclusions.csv"
    required_by_contract = bool(
        filter_config.get("downstream_endpoint_exclusion_required", False)
    )
    if not path.exists():
        if required_by_contract:
            raise FileNotFoundError(f"Required flypoint exclusion manifest is missing: {path}")
        return {}, {}, pd.DataFrame()
    exclusions = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "file", "signal", "repaired_start_index", "recovery_decision_index",
        "unsafe_endpoint_start", "unsafe_endpoint_end",
    }
    missing = required - set(exclusions.columns)
    if missing:
        raise KeyError(f"Flypoint exclusion manifest is missing: {sorted(missing)}")

    unsafe_input_rows: Dict[str, set[int]] = {}
    unsafe_label_origins: Dict[str, set[int]] = {}
    target_names = {name.upper() for name in TARGET_COLS}
    for row in exclusions.itertuples(index=False):
        filename = str(row.file)
        start = int(row.unsafe_endpoint_start)
        end = int(row.unsafe_endpoint_end)
        if end < start:
            raise ValueError(f"Invalid exclusion range in {filename}: {start}..{end}")
        unsafe_input_rows.setdefault(filename, set()).update(range(start, end + 1))
        if str(row.signal).upper() in target_names:
            repaired_start = int(row.repaired_start_index)
            decision = int(row.recovery_decision_index)
            for target_index in range(repaired_start, decision):
                for horizon in horizons:
                    origin = target_index - horizon
                    if origin >= 0:
                        unsafe_label_origins.setdefault(filename, set()).add(origin)
    return unsafe_input_rows, unsafe_label_origins, exclusions


def signal_aliases(canonical_signal: str) -> Tuple[str, ...]:
    return SIGNAL_ALIASES.get(canonical_signal, (canonical_signal,))


def resolve_signal_column(
    frame: pd.DataFrame,
    canonical_signal: str,
    suffix: str,
    required: bool = True,
) -> str | None:
    matches: List[str] = []
    for alias in signal_aliases(canonical_signal):
        pattern = re.compile(rf"^{re.escape(alias)}\s+{re.escape(suffix)}$", re.IGNORECASE)
        matches.extend(
            str(column) for column in frame.columns
            if pattern.fullmatch(str(column).strip())
        )
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    raise KeyError(
        f"Expected one {suffix} column for {canonical_signal}, found {matches}"
    )


def contiguous_runs(mask: np.ndarray) -> List[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    return [chunk for chunk in np.split(indices, split_points) if chunk.size]


def load_coolprop() -> Tuple[Any, str]:
    if _COOLPROP_RUNTIME:
        return _COOLPROP_RUNTIME["module"], str(_COOLPROP_RUNTIME["version"])
    try:
        import CoolProp  # type: ignore
        import CoolProp.CoolProp as CP  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CoolProp is required. Install with: "
            f"pip install CoolProp=={RECOMMENDED_COOLPROP_VERSION}"
        ) from exc
    version = str(getattr(CoolProp, "__version__", "unknown"))
    if version != RECOMMENDED_COOLPROP_VERSION:
        print(
            "WARNING: CoolProp version differs from reproducibility target: "
            f"runtime={version}, recommended={RECOMMENDED_COOLPROP_VERSION}"
        )
    _COOLPROP_RUNTIME.update(module=CP, version=version)
    return CP, version


def helium_hmass_j_per_kg(
    temperature_k: pd.Series | np.ndarray,
    pressure_bar: pd.Series | np.ndarray,
    state_name: str,
) -> np.ndarray:
    CP, _ = load_coolprop()
    temperature = np.asarray(temperature_k, dtype=float)
    pressure_pa = np.asarray(pressure_bar, dtype=float) * BAR_TO_PA
    if temperature.shape != pressure_pa.shape:
        raise ValueError(f"T/P shape mismatch for {state_name}")
    result = np.full(temperature.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(temperature) & np.isfinite(pressure_pa)
        & (temperature > 0.0) & (pressure_pa > 0.0)
    )
    valid_indices = np.flatnonzero(valid)

    def evaluate(indices: np.ndarray) -> None:
        if indices.size == 0:
            return
        try:
            values = np.asarray(
                CP.PropsSI(
                    "Hmass", "T", temperature[indices], "P", pressure_pa[indices],
                    HELIUM_FLUID,
                ),
                dtype=float,
            ).reshape(-1)
            if values.size != indices.size:
                raise ValueError("CoolProp returned an unexpected result length")
            finite = np.isfinite(values)
            result[indices[finite]] = values[finite]
        except Exception:
            if indices.size == 1:
                return
            middle = indices.size // 2
            evaluate(indices[:middle])
            evaluate(indices[middle:])

    for start in range(0, valid_indices.size, 100_000):
        evaluate(valid_indices[start:start + 100_000])
    failed = int(valid_indices.size - np.isfinite(result[valid_indices]).sum())
    if failed:
        print(
            f"WARNING: CoolProp rejected {failed}/{valid_indices.size} states for "
            f"{state_name}; affected rows become segment boundaries."
        )
    return result


def causal_optional_diff(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    current_valid = np.isfinite(values.to_numpy(dtype=float))
    previous_valid = np.r_[False, current_valid[:-1]]
    result = pd.Series(np.nan, index=values.index, dtype=float)
    continuing = current_valid & previous_valid
    starting = current_valid & ~previous_valid
    result.loc[continuing] = values.diff().loc[continuing]
    result.loc[starting] = 0.0
    return result


def series_conductance_proxy(upstream: pd.Series, downstream: pd.Series) -> pd.Series:
    denominator = np.sqrt(upstream.pow(2) + downstream.pow(2))
    result = pd.Series(0.0, index=upstream.index, dtype=float)
    active = denominator > 1e-12
    result.loc[active] = (
        upstream.loc[active] * downstream.loc[active] / denominator.loc[active]
    )
    return result


def add_causal_features(frame: pd.DataFrame, valve_event_threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    elapsed_minutes = 59.0 * SAMPLE_PERIOD_SECONDS / 60.0

    frame["TE8353_diff_1"] = frame["TE8353"].diff().fillna(0.0)
    frame["TE8353_history_slope_k_per_min_60points"] = (
        (frame["TE8353"] - frame["TE8353"].shift(59)) / elapsed_minutes
    ).fillna(0.0)
    frame["Thv_diff_1"] = frame["Thv"].diff().fillna(0.0)
    frame["Thv_history_slope_k_per_min_60points"] = (
        (frame["Thv"] - frame["Thv"].shift(59)) / elapsed_minutes
    ).fillna(0.0)

    openings = frame[["CV8300", "CV8313", "CV8310", "CV8351"]].clip(lower=0.0) / 100.0
    pressure_drive = np.sqrt((frame["PT8310"] - frame["PT8351"]).clip(lower=0.0))
    g1_flow = series_conductance_proxy(openings["CV8300"], openings["CV8310"]) * pressure_drive
    g2_flow = series_conductance_proxy(openings["CV8313"], openings["CV8310"]) * pressure_drive
    frame["G1_main_flow_command_proxy"] = g1_flow
    frame["G2_main_flow_command_proxy"] = g2_flow
    frame["G1_to_postmix_thermal_drive_proxy"] = g1_flow * (
        frame["TE8351"] - G1_NOMINAL_INLET_TEMPERATURE_K
    )
    frame["G2_to_postmix_thermal_drive_proxy"] = g2_flow * (
        frame["TE8351"] - G2_NOMINAL_INLET_TEMPERATURE_K
    )
    frame["A_to_postmix_thermal_drive_proxy"] = openings["CV8351"] * (
        frame["TE8351"] - A_NOMINAL_INLET_TEMPERATURE_K
    )
    frame["CV8351_diff_1"] = frame["CV8351"].diff().fillna(0.0)

    h_8351_p8351 = helium_hmass_j_per_kg(
        frame["TE8351"], frame["PT8351"], "TE8351/PT8351"
    )
    reference_temperature = np.full(len(frame), REFERENCE_TEMPERATURE_K, dtype=float)
    h_300_p8351 = helium_hmass_j_per_kg(
        reference_temperature, frame["PT8351"], "300K/PT8351-reference"
    )
    specific_cooling = h_300_p8351 - h_8351_p8351
    mass_flow_kg_per_s = frame["FT8351"].to_numpy(dtype=float) * G_PER_S_TO_KG_PER_S
    frame["PostMix_specific_cooling_300K_ref_J_kg"] = specific_cooling
    frame["PostMix_cooling_rate_300K_ref_W"] = mass_flow_kg_per_s * specific_cooling

    frame["Mainline_dT_8351_8352"] = frame["TE8351"] - frame["TE8352"]
    frame["Mainline_dT_8352_8353"] = frame["TE8352"] - frame["TE8353"]
    frame["Mainline_dP_8351_8352_bar"] = frame["PT8351"] - frame["PT8352"]
    frame["Module_dT_Thv_TE8353"] = frame["Thv"] - frame["TE8353"]
    frame["Module_flow_temperature_drive_proxy"] = frame["FT8351"] * (
        frame["Thv"] - frame["TE8353"]
    )
    frame["Module_dT_Thv_TE8310"] = frame["Thv"] - frame["TE8310"]
    frame["EC_V2_diff_1"] = causal_optional_diff(frame["EC-V2"])
    frame["COOLDOWN_diff_1"] = causal_optional_diff(frame["COOLDOWN"])

    h_8352_p8352 = helium_hmass_j_per_kg(
        frame["TE8352"], frame["PT8352"], "TE8352/PT8352"
    )
    h_8353_p8352 = helium_hmass_j_per_kg(
        frame["TE8353"], frame["PT8352"], "TE8353/PT8352-proxy"
    )
    frame["Mainline_apparent_heat_leak_8351_8352_W"] = mass_flow_kg_per_s * (
        h_8352_p8352 - h_8351_p8351
    )
    frame["Mainline_apparent_heat_leak_8352_8353_W"] = mass_flow_kg_per_s * (
        h_8353_p8352 - h_8352_p8352
    )

    event = pd.Series(False, index=frame.index)
    for valve in PRIMARY_VALVE_COLS:
        values = pd.to_numeric(frame[valve], errors="coerce")
        difference = values.diff().abs()
        event |= difference.gt(valve_event_threshold).fillna(False)
    frame["valve_event_mask"] = event.astype(float)

    missing = [column for column in ENGINEERED_COLS if column not in frame.columns]
    if missing:
        raise AssertionError(f"Feature builder did not create: {missing}")
    return frame


def future_path_mask(base_valid: np.ndarray, max_horizon: int) -> np.ndarray:
    result = np.zeros(len(base_valid), dtype=bool)
    width = max_horizon + 1
    if len(base_valid) < width:
        return result
    counts = np.convolve(base_valid.astype(np.int16), np.ones(width, dtype=np.int16), mode="valid")
    result[:len(counts)] = counts == width
    return result


def build_file_segments(
    identity: FileIdentity,
    horizons: Sequence[int],
    look_back: int,
    valve_event_threshold: float,
    unsafe_input_rows: Iterable[int],
    unsafe_label_origins: Iterable[int],
) -> Tuple[List[pd.DataFrame], Dict[str, object]]:
    raw = read_industrial_csv(identity.path)
    extracted = pd.DataFrame(index=np.arange(len(raw), dtype=np.int64))
    for signal in REQUIRED_RAW_SIGNAL_COLS:
        source_column = resolve_signal_column(raw, signal, "ValueY", required=True)
        assert source_column is not None
        extracted[signal] = pd.to_numeric(raw[source_column], errors="coerce").astype(float)
    for signal in OPTIONAL_MODULE_VALVE_COLS:
        source_column = resolve_signal_column(raw, signal, "ValueY", required=False)
        if source_column is None:
            extracted[signal] = np.nan
        else:
            extracted[signal] = pd.to_numeric(raw[source_column], errors="coerce").astype(float)

    extracted["EC_V2_available_mask"] = np.isfinite(extracted["EC-V2"]).astype(float)
    extracted["COOLDOWN_available_mask"] = np.isfinite(extracted["COOLDOWN"]).astype(float)

    timestamp_column = resolve_signal_column(raw, "TE8353", "Time", required=True)
    assert timestamp_column is not None
    source_timestamp = raw[timestamp_column].astype(str).where(raw[timestamp_column].notna(), "")

    label_cols = future_label_columns(horizons)
    for target in TARGET_COLS:
        for horizon in horizons:
            extracted[f"Future_{target}_{horizon}step"] = extracted[target].shift(-horizon)

    required_raw_finite = np.isfinite(
        extracted[list(REQUIRED_RAW_SIGNAL_COLS)].to_numpy(dtype=float)
    ).all(axis=1)
    labels_finite = np.isfinite(extracted[list(label_cols)].to_numpy(dtype=float)).all(axis=1)

    unsafe_input_mask = np.zeros(len(extracted), dtype=bool)
    unsafe_label_mask = np.zeros(len(extracted), dtype=bool)
    unsafe_input_index = np.asarray(sorted(set(unsafe_input_rows)), dtype=np.int64)
    unsafe_label_index = np.asarray(sorted(set(unsafe_label_origins)), dtype=np.int64)
    if unsafe_input_index.size:
        if unsafe_input_index.min() < 0 or unsafe_input_index.max() >= len(extracted):
            raise IndexError(f"Flypoint input exclusion outside {identity.path.name}")
        unsafe_input_mask[unsafe_input_index] = True
    if unsafe_label_index.size:
        if unsafe_label_index.min() < 0 or unsafe_label_index.max() >= len(extracted):
            raise IndexError(f"Flypoint label exclusion outside {identity.path.name}")
        unsafe_label_mask[unsafe_label_index] = True

    current_state_safe = required_raw_finite & ~unsafe_input_mask
    path_safe = future_path_mask(current_state_safe, max(horizons))
    eligible = current_state_safe & path_safe & labels_finite & ~unsafe_label_mask
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
            provisional[list(FINITE_ENGINEERED_COLS)].to_numpy(dtype=float)
        ).all(axis=1)
        engineered_invalid_rows += int((~engineered_finite).sum())

        for local_indices in contiguous_runs(engineered_finite):
            if len(local_indices) < look_back:
                short_segment_rows += len(local_indices)
                continue
            source_indices = row_indices[local_indices]
            if engineered_finite.all():
                part = provisional.iloc[local_indices].copy().reset_index(drop=True)
            else:
                part = extracted.iloc[source_indices].copy().reset_index(drop=True)
                part = add_causal_features(part, valve_event_threshold)
            if not np.isfinite(
                part[[*REQUIRED_RAW_SIGNAL_COLS, *FINITE_ENGINEERED_COLS, *TECHNICAL_MASK_COLS, *label_cols]].to_numpy(dtype=float)
            ).all():
                raise AssertionError(f"Non-finite required value survived in {identity.path.name}")

            part["file_id"] = f"{identity.path.stem}__segment_{output_segment_number:03d}"
            output_segment_number += 1
            part["source_group"] = identity.source_group
            part["source_variant"] = identity.variant
            part["parent_id"] = identity.parent_id
            part["source_row_index"] = [str(index) for index in source_indices]
            part["source_timestamp"] = source_timestamp.iloc[source_indices].to_numpy()
            part = part[[
                *RAW_SIGNAL_COLS,
                *TECHNICAL_MASK_COLS,
                *ENGINEERED_COLS,
                *label_cols,
                *METADATA_COLS,
            ]]
            output_segments.append(part)

    report = {
        "file": identity.path.name,
        "source_group": identity.source_group,
        "parent_id": identity.parent_id,
        "variant": identity.variant,
        "role": identity.role,
        "rows_input": len(raw),
        "rows_output": sum(len(segment) for segment in output_segments),
        "segments_output": len(output_segments),
        "rows_missing_required_raw": int((~required_raw_finite).sum()),
        "rows_missing_EC_V2": int((~np.isfinite(extracted["EC-V2"])).sum()),
        "rows_missing_COOLDOWN": int((~np.isfinite(extracted["COOLDOWN"])).sum()),
        "rows_missing_any_future_label": int((~labels_finite).sum()),
        "rows_flypoint_input_excluded": int(unsafe_input_mask.sum()),
        "rows_flypoint_label_origin_excluded": int(unsafe_label_mask.sum()),
        "rows_future_path_excluded": int((current_state_safe & ~path_safe).sum()),
        "rows_missing_engineered_feature": engineered_invalid_rows,
        "rows_in_short_segments_removed": short_segment_rows,
    }
    if not output_segments:
        raise ValueError(f"No usable segment remains in {identity.path.name}: {report}")
    return output_segments, report


def concatenate_segments(segments: Sequence[pd.DataFrame], split_name: str) -> pd.DataFrame:
    if not segments:
        raise ValueError(f"No segments assigned to {split_name}")
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
    selected_segments: List[pd.DataFrame] = []
    for source_file_id, part in test_full.groupby("file_id", sort=False):
        part = part.reset_index(drop=True)
        candidates: List[Tuple[float, int, int]] = []
        for start in range(0, len(part), block_steps):
            end = min(start + block_steps, len(part))
            event_count = float(part.loc[start:end - 1, "valve_event_mask"].sum())
            candidates.append((event_count, start, end))
        if not candidates:
            continue
        select_count = min(
            len(candidates),
            max(minimum_blocks, int(math.ceil(len(candidates) * block_fraction))),
        )
        chosen = sorted(candidates, key=lambda item: (-item[0], item[1]))[:select_count]
        for number, (origin_start, origin_end) in enumerate(
            merge_intervals([(start, end) for _, start, end in chosen])
        ):
            history_start = max(0, origin_start - (look_back - 1))
            segment = part.iloc[history_start:origin_end].copy().reset_index(drop=True)
            if len(segment) < look_back:
                continue
            segment["file_id"] = f"{source_file_id}__dynamic_{number:03d}"
            selected_segments.append(segment)
    if not selected_segments:
        raise ValueError("No dynamic test segment contains a complete history window")
    return pd.concat(selected_segments, ignore_index=True)


def count_windows(frame: pd.DataFrame, look_back: int) -> int:
    return int(sum(
        max(0, len(part) - look_back + 1)
        for _, part in frame.groupby("file_id", sort=False)
    ))


def assert_frame_schema(
    frame: pd.DataFrame,
    split_name: str,
    label_cols: Sequence[str],
) -> None:
    expected = [
        *RAW_SIGNAL_COLS,
        *TECHNICAL_MASK_COLS,
        *ENGINEERED_COLS,
        *label_cols,
        *METADATA_COLS,
    ]
    if list(frame.columns) != expected:
        raise AssertionError(f"Unexpected {split_name} schema")
    required_numeric = [
        *REQUIRED_RAW_SIGNAL_COLS,
        *TECHNICAL_MASK_COLS,
        *FINITE_ENGINEERED_COLS,
        *label_cols,
    ]
    if not np.isfinite(frame[required_numeric].to_numpy(dtype=float)).all():
        raise AssertionError(f"{split_name} contains NaN/inf in required numeric columns")
    actual_future = [column for column in frame.columns if str(column).startswith("Future_")]
    if actual_future != list(label_cols):
        raise AssertionError(f"Unexpected future-label columns in {split_name}")

    for raw_col, mask_col, diff_col in (
        ("EC-V2", "EC_V2_available_mask", "EC_V2_diff_1"),
        ("COOLDOWN", "COOLDOWN_available_mask", "COOLDOWN_diff_1"),
    ):
        available = frame[mask_col].to_numpy(dtype=float) == 1.0
        raw_finite = np.isfinite(frame[raw_col].to_numpy(dtype=float))
        diff_finite = np.isfinite(frame[diff_col].to_numpy(dtype=float))
        if not np.array_equal(available, raw_finite):
            raise AssertionError(f"{split_name}: {raw_col} availability mask mismatch")
        if np.any(available & ~diff_finite) or np.any(~available & diff_finite):
            raise AssertionError(f"{split_name}: {diff_col} missingness mismatch")

    for file_id, indices in frame.groupby("file_id", sort=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) and not np.all(np.diff(idx) == 1):
            raise AssertionError(f"Rows for {file_id} are not contiguous in {split_name}")
        if frame.iloc[idx]["source_group"].nunique() != 1:
            raise AssertionError(f"{file_id} maps to multiple source groups")


def assert_split_integrity(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test_dynamic: pd.DataFrame,
    test_full: pd.DataFrame,
    validation_parent: str,
    test_parent: str,
    label_cols: Sequence[str],
) -> None:
    for name, frame in (
        ("train", train),
        ("validation", val),
        ("dynamic_test", test_dynamic),
        ("full_test", test_full),
    ):
        assert_frame_schema(frame, name, label_cols)

    train_groups = set(train["source_group"].unique())
    val_groups = set(val["source_group"].unique())
    test_groups = set(test_full["source_group"].unique())
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise AssertionError("Parent/source_group overlap across splits")
    if set(val["parent_id"].unique()) != {validation_parent}:
        raise AssertionError(f"Validation is not exclusively {validation_parent}")
    if set(test_full["parent_id"].unique()) != {test_parent}:
        raise AssertionError(f"Test is not exclusively {test_parent}")
    if set(val["source_variant"].unique()) != {"Original"}:
        raise AssertionError("Validation contains augmented data")
    if set(test_full["source_variant"].unique()) != {"Original"}:
        raise AssertionError("Test contains augmented data")
    if set(test_dynamic["source_group"].unique()) != test_groups:
        raise AssertionError("Dynamic/full test source-group mismatch")
    if "0715-ALL" not in set(train["parent_id"].unique()):
        raise AssertionError("0715-ALL did not enter training")
    variants_0715 = set(
        train.loc[train["parent_id"] == "0715-ALL", "source_variant"].unique()
    )
    if variants_0715 != {"Original", "Aug_01", "Aug_02"}:
        raise AssertionError("0715-ALL training variants are incomplete")
    back = train[train["parent_id"].astype(str).str.contains("BACK", case=False)]
    if back.empty:
        raise AssertionError("No BACK data reached training")


def dataset_summary(frame: pd.DataFrame, name: str, look_back: int) -> Dict[str, object]:
    both_module_valves = (
        (frame["EC_V2_available_mask"] == 1.0)
        & (frame["COOLDOWN_available_mask"] == 1.0)
    )
    return {
        "dataset": name,
        "rows": len(frame),
        "segments": int(frame["file_id"].nunique()),
        "windows": count_windows(frame, look_back),
        "source_groups": int(frame["source_group"].nunique()),
        "parent_ids": ";".join(sorted(map(str, frame["parent_id"].unique()))),
        "variants": ";".join(sorted(map(str, frame["source_variant"].unique()))),
        "module_valves_available_rows": int(both_module_valves.sum()),
        "valve_event_rows": int(frame["valve_event_mask"].sum()),
        "valve_event_ratio": float(frame["valve_event_mask"].mean()),
    }


def dump_pickle_compatible(obj: object, path: Path) -> str:
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
            f"Processed outputs already exist ({preview}); pass --overwrite to replace them"
        )


def validate_variant_contract(identities: Sequence[FileIdentity]) -> None:
    by_group: Dict[str, set[str]] = {}
    for item in identities:
        by_group.setdefault(item.source_group, set()).add(item.variant)
    expected = {"Original", "Aug_01", "Aug_02"}
    failures = {group: variants for group, variants in by_group.items() if variants != expected}
    if failures:
        raise AssertionError(f"Incomplete augmentation families: {failures}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    label_cols = future_label_columns(args.horizons)

    filter_config = load_filter_contract(args.input_dir)
    filter_manifest = load_filter_manifest(args.input_dir)
    files = discover_data_files(args.input_dir)
    manifest_files = set(filter_manifest["output_file"].astype(str))
    discovered_names = {path.name for path in files}
    if manifest_files != discovered_names:
        raise AssertionError(
            "Filter manifest/data mismatch; "
            f"missing={sorted(discovered_names-manifest_files)}, "
            f"extra={sorted(manifest_files-discovered_names)}"
        )

    identities = [
        parse_identity(path, args.validation_parent, args.test_parent) for path in files
    ]
    validate_variant_contract(identities)
    if sum(item.role == "validation" for item in identities) != 1:
        raise AssertionError("Expected exactly one Original validation file")
    if sum(item.role == "test" for item in identities) != 1:
        raise AssertionError("Expected exactly one Original test file")
    if sum(item.role == "ignored_validation_augmentation" for item in identities) != 2:
        raise AssertionError("Expected two ignored validation augmentations")
    if sum(item.role == "ignored_test_augmentation" for item in identities) != 2:
        raise AssertionError("Expected two ignored test augmentations")

    unsafe_input, unsafe_label, flypoint_exclusions = load_flypoint_exclusions(
        args.input_dir, args.horizons, filter_config
    )
    manifest_rows = filter_manifest.set_index("output_file")["rows"].to_dict()
    used_identities = [item for item in identities if not item.role.startswith("ignored_")]
    train_segments: List[pd.DataFrame] = []
    val_segments: List[pd.DataFrame] = []
    test_segments: List[pd.DataFrame] = []
    file_reports: List[Dict[str, object]] = []

    for number, identity in enumerate(used_identities, start=1):
        print(f"[{number}/{len(used_identities)}] {identity.role}: {identity.path.name}")
        segments, report = build_file_segments(
            identity,
            horizons=args.horizons,
            look_back=args.look_back,
            valve_event_threshold=args.valve_event_threshold,
            unsafe_input_rows=unsafe_input.get(identity.path.name, set()),
            unsafe_label_origins=unsafe_label.get(identity.path.name, set()),
        )
        if int(report["rows_input"]) != int(manifest_rows[identity.path.name]):
            raise AssertionError(f"Row-count mismatch for {identity.path.name}")
        file_reports.append(report)
        if identity.role == "train":
            train_segments.extend(segments)
        elif identity.role == "validation":
            val_segments.extend(segments)
        elif identity.role == "test":
            test_segments.extend(segments)
        else:
            raise AssertionError(f"Unhandled role: {identity.role}")

    for identity in identities:
        if identity.role.startswith("ignored_"):
            file_reports.append({
                "file": identity.path.name,
                "source_group": identity.source_group,
                "parent_id": identity.parent_id,
                "variant": identity.variant,
                "role": identity.role,
                "rows_input": int(manifest_rows[identity.path.name]),
                "rows_output": 0,
                "segments_output": 0,
            })

    train = concatenate_segments(train_segments, "train").reset_index(drop=True)
    val = concatenate_segments(val_segments, "validation").reset_index(drop=True)
    test_full = concatenate_segments(test_segments, "full_test").reset_index(drop=True)
    test_dynamic = select_dynamic_test_segments(
        test_full,
        look_back=args.look_back,
        block_steps=args.dynamic_block_steps,
        block_fraction=args.dynamic_block_fraction,
        minimum_blocks=args.minimum_dynamic_blocks,
    ).reset_index(drop=True)

    assert_split_integrity(
        train,
        val,
        test_dynamic,
        test_full,
        validation_parent=args.validation_parent,
        test_parent=args.test_parent,
        label_cols=label_cols,
    )
    summaries = [
        dataset_summary(train, "train_clean", args.look_back),
        dataset_summary(val, "val_clean_full_original", args.look_back),
        dataset_summary(test_dynamic, "test_clean_dynamic", args.look_back),
        dataset_summary(test_full, "test_full_clean", args.look_back),
    ]
    print(pd.DataFrame(summaries).to_string(index=False))
    print(
        "Split audit: "
        f"validation=Original {args.validation_parent}; "
        f"test=Original {args.test_parent}; "
        "0715-ALL=training Original+Aug_01+Aug_02."
    )

    if args.validate_only:
        print("Validation-only run complete; no processed_data files were written.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_output_targets(args.output_dir, args.overwrite)
    writers = {
        "train_clean.pkl": dump_pickle_compatible(train, args.output_dir / "train_clean.pkl"),
        "val_clean.pkl": dump_pickle_compatible(val, args.output_dir / "val_clean.pkl"),
        "test_clean.pkl": dump_pickle_compatible(test_dynamic, args.output_dir / "test_clean.pkl"),
        "test_full_clean.pkl": dump_pickle_compatible(test_full, args.output_dir / "test_full_clean.pkl"),
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
        "targets": list(TARGET_COLS),
        "horizons": list(args.horizons),
        "future_labels": list(label_cols),
        "look_back": args.look_back,
        "sample_period_seconds": SAMPLE_PERIOD_SECONDS,
        "validation_parent": args.validation_parent,
        "validation_variant": "Original only",
        "test_parent": args.test_parent,
        "test_variant": "Original only",
        "training_augmentation_policy": "Original+Aug_01+Aug_02 kept with each training parent",
        "back_policy": "0623-BACK is training only; 0715-BACK is the held-out test (Original only)",
        "raw_signal_columns": list(RAW_SIGNAL_COLS),
        "optional_module_valve_columns": list(OPTIONAL_MODULE_VALVE_COLS),
        "technical_availability_masks": list(TECHNICAL_MASK_COLS),
        "engineered_columns": list(ENGINEERED_COLS),
        "metadata_columns": list(METADATA_COLS),
        "scaler_fitting": "not performed here; training scripts fit train_clean.pkl only",
        "future_features_used": False,
        "random_row_split": False,
        "flypoint_policy": (
            "repaired rows are removed; preceding origins whose maximum-horizon path crosses "
            "an excluded/missing state are also removed; segments never cross those gaps"
        ),
        "thermophysical_properties": {
            "library": "CoolProp",
            "runtime_version": load_coolprop()[1],
            "recommended_version": RECOMMENDED_COOLPROP_VERSION,
            "fluid": HELIUM_FLUID,
            "temperature_unit": "K",
            "pressure_source_unit": "bar",
            "mass_flow_source_unit": "g/s",
            "enthalpy_unit": "J/kg",
            "power_unit": "W",
        },
        "dynamic_test": {
            "block_steps": args.dynamic_block_steps,
            "selected_fraction": args.dynamic_block_fraction,
            "minimum_blocks": args.minimum_dynamic_blocks,
            "selection_signal": "causal event mask from six process valves plus available EC-V2/COOLDOWN",
        },
        "pickle_writers": writers,
        "flypoint_manifest_events": len(flypoint_exclusions),
    }
    (args.output_dir / "data_build_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    feature_catalog = {
        "policy": (
            "all causal raw/engineered candidates are materialized in one table; separate "
            "TE8353 and Thv training runs choose feature groups and one future label"
        ),
        "raw_signal_columns": list(RAW_SIGNAL_COLS),
        "technical_availability_masks": list(TECHNICAL_MASK_COLS),
        "feature_groups": {name: list(columns) for name, columns in FEATURE_GROUPS.items()},
        "recommended_engineered_columns": list(RECOMMENDED_ENGINEERED_COLS),
        "optional_apparent_heat_leak_columns": list(
            FEATURE_GROUPS["optional_coolprop_apparent_heat_leak"]
        ),
        "evaluation_only_columns": list(FEATURE_GROUPS["evaluation_auxiliary"]),
        "feature_definitions": FEATURE_DEFINITIONS,
        "future_labels": list(label_cols),
        "future_labels_are_inputs": False,
        "module_valve_missing_policy": (
            "never fill missing EC-V2/COOLDOWN with zero; use availability masks and apply "
            "the same eligible-parent subset to every model in a module-valve ablation"
        ),
        "not_materialized": {
            "G_screen_diversion_fraction": "removed by the approved feature plan",
            "FC_V1": "not a module-side input valve and excluded",
            "target_rolling_mean": "removed after earlier prediction-spike observations",
            "future_valve_trajectory": "belongs to a later MPC dataset, not this forecasting dataset",
            "train_fitted_statistics": "scalers/selection/PCA/thresholds must be learned from training only",
        },
        "nominal_constants": {
            "A_inlet_temperature_k": A_NOMINAL_INLET_TEMPERATURE_K,
            "G1_inlet_temperature_k": G1_NOMINAL_INLET_TEMPERATURE_K,
            "G2_inlet_temperature_k": G2_NOMINAL_INLET_TEMPERATURE_K,
            "cooling_reference_temperature_k": REFERENCE_TEMPERATURE_K,
        },
    }
    (args.output_dir / "feature_catalog.json").write_text(
        json.dumps(feature_catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Processed datasets saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
