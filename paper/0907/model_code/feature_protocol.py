#!/usr/bin/env python3
"""Frozen protocol for the 0907 Thv physical-feature ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODEL_NAME = "parallel_gru_kan_gnn"
LOOKBACK = 60
COMMON_ORIGIN_LOOKBACK = 120
PREDICT_STEPS = 15
SELECTION_METRIC = "validation_rmse_k"
SEEDS = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)

MODEL_CONFIG: dict[str, int | float] = {
    "history_hidden": 64,
    "graph_hidden": 64,
    "graph_sweeps": 2,
    "edge_hidden": 32,
    "kan_grid": 8,
    "tcn_levels": 6,
    "tcn_kernel": 3,
    "control_hidden": 32,
    "fusion_hidden": 64,
    "dropout": 0.1,
}

RAW_HISTORY_COLUMNS = (
    "TE8310", "TE8351", "TE8352", "FT8351", "PT8310", "PT8351",
    "PT8352", "CV8312", "CV8311", "CV8310", "CV8313", "CV8300",
    "CV8351", "Thv", "TE8353", "Tef", "Tcd", "DTbr", "EC-V2",
    "COOLDOWN",
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "target_history_dynamics": (
        "TE8353_diff_1", "TE8353_history_slope_k_per_min_60points",
        "Thv_diff_1", "Thv_history_slope_k_per_min_60points",
    ),
    "g_main_flow_command": (
        "G1_main_flow_command_proxy", "G2_main_flow_command_proxy",
    ),
    "premix_postmix_thermal_drive": (
        "G1_to_postmix_thermal_drive_proxy",
        "G2_to_postmix_thermal_drive_proxy",
        "A_to_postmix_thermal_drive_proxy", "CV8351_diff_1",
    ),
    "postmix_cooling_bridge": (
        "PostMix_specific_cooling_300K_ref_J_kg",
        "PostMix_cooling_rate_300K_ref_W",
    ),
    "mainline_state_contrasts": (
        "Mainline_dT_8351_8352", "Mainline_dT_8352_8353",
        "Mainline_dP_8351_8352_bar",
    ),
    "thv_module_coupling": (
        "Module_dT_Thv_TE8353", "Module_flow_temperature_drive_proxy",
        "Module_dT_Thv_TE8310",
    ),
    "module_valve_dynamics": ("EC_V2_diff_1", "COOLDOWN_diff_1"),
}

OPTIONAL_HEAT_LEAK_FEATURES = (
    "Mainline_apparent_heat_leak_8351_8352_W",
    "Mainline_apparent_heat_leak_8352_8353_W",
)
ALL_ENGINEERED_COLUMNS = tuple(
    column for columns in FEATURE_GROUPS.values() for column in columns
) + OPTIONAL_HEAT_LEAK_FEATURES
MODEL_HISTORY_COLUMNS = RAW_HISTORY_COLUMNS + ALL_ENGINEERED_COLUMNS


def _recommended_without(group_to_remove: str | None = None) -> tuple[str, ...]:
    return tuple(
        column
        for group, columns in FEATURE_GROUPS.items()
        if group != group_to_remove
        for column in columns
    )


FEATURE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "feature_set": "raw20",
        "label": "Raw measurements only",
        "engineered": (),
        "role": "causal raw-input reference",
    },
    {
        "feature_set": "full20",
        "label": "All 20 recommended features",
        "engineered": _recommended_without(),
        "role": "complete recommended physical-feature model",
    },
    *(
        {
            "feature_set": f"full20_no_{group}",
            "label": f"All recommended except {group}",
            "engineered": _recommended_without(group),
            "role": f"leave-one-physical-group-out: {group}",
            "removed_group": group,
        }
        for group in FEATURE_GROUPS
    ),
    {
        "feature_set": "full22_with_apparent_heat_leak",
        "label": "All 20 recommended plus two apparent heat-leak features",
        "engineered": _recommended_without() + OPTIONAL_HEAT_LEAK_FEATURES,
        "role": "optional CoolProp apparent-enthalpy-gain check",
    },
)

_SPEC_BY_ID = {str(spec["feature_set"]): spec for spec in FEATURE_SPECS}
if len(_SPEC_BY_ID) != len(FEATURE_SPECS):
    raise AssertionError("Duplicate feature-set identifiers")


def feature_spec(feature_set: str) -> dict[str, Any]:
    try:
        return _SPEC_BY_ID[feature_set]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; expected {tuple(_SPEC_BY_ID)}"
        ) from exc


def history_columns(spec: dict[str, Any]) -> tuple[str, ...]:
    # Every ablation keeps the same 42-channel tensor and parameter count.
    # Inactive engineered channels are standardized and then masked to zero.
    columns = MODEL_HISTORY_COLUMNS
    if len(columns) != len(set(columns)):
        raise AssertionError(f"Duplicate history feature in {spec['feature_set']}")
    if any(name.startswith("Future_") for name in columns):
        raise AssertionError("A future label entered a history feature set")
    return columns


def active_history_columns(spec: dict[str, Any]) -> tuple[str, ...]:
    return RAW_HISTORY_COLUMNS + tuple(spec["engineered"])


def history_feature_mask(spec: dict[str, Any]) -> tuple[float, ...]:
    active = set(active_history_columns(spec))
    return tuple(1.0 if column in active else 0.0 for column in MODEL_HISTORY_COLUMNS)


def all_tasks() -> list[tuple[dict[str, Any], int]]:
    return [(spec, seed) for spec in FEATURE_SPECS for seed in SEEDS]


def run_directory(results_dir: Path, spec: dict[str, Any], seed: int) -> Path:
    return (
        results_dir / "development" / f"horizon_{PREDICT_STEPS:02d}"
        / f"lookback_{LOOKBACK:02d}" / MODEL_NAME
        / str(spec["feature_set"]) / f"seed_{seed}"
    )


def valid_completed_run(results_dir: Path, spec: dict[str, Any], seed: int) -> bool:
    run_dir = run_directory(results_dir, spec, seed)
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not metrics_path.exists() or not checkpoint_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "model_type": MODEL_NAME,
        "feature_set": spec["feature_set"],
        "lookback": LOOKBACK,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "seed": seed,
        "checkpoint_selection_metric": SELECTION_METRIC,
        "test_data_loaded": False,
        "future_noncontrol_measurements_used": False,
        "future_label_used_as_input": False,
    }
    return all(metrics.get(key) == value for key, value in expected.items())


def protocol_payload() -> dict[str, Any]:
    return {
        "study": "0907 Thv physical-feature ablation",
        "architecture": MODEL_NAME,
        "architecture_note": (
            "parallel GRU and directed physical graph with monotonic spline-KAN "
            "functions on each valve edge; this is local edge monotonicity, not "
            "a global output monotonicity guarantee"
        ),
        "target": "Delta Thv at t+15, restored to absolute Thv for metrics",
        "lookback": LOOKBACK,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "history_input": "raw and selected causal engineered features through t",
        "known_future_control_input": "eight raw valve commands u(t)..u(t+14)",
        "future_engineered_features_used": False,
        "training_loss": "weighted Huber; train-defined lowest-delta 10% weight=3",
        "checkpoint_selection_split": "Original 260501 validation only",
        "checkpoint_selection_metric": SELECTION_METRIC,
        "posthoc_test_split": "Original 0715-BACK complete forecastable sequence",
        "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
        "optional_heat_leak_features": list(OPTIONAL_HEAT_LEAK_FEATURES),
        "feature_sets": [
            {
                **{key: value for key, value in spec.items() if key != "engineered"},
                "engineered": list(spec["engineered"]),
                "history_columns": list(history_columns(spec)),
                "active_history_columns": list(active_history_columns(spec)),
                "masked_history_columns": [
                    column for column, keep in zip(
                        history_columns(spec), history_feature_mask(spec)
                    ) if not keep
                ],
                "seeds": list(SEEDS),
            }
            for spec in FEATURE_SPECS
        ],
        "model_config": MODEL_CONFIG,
        "expected_runs": len(all_tasks()),
        "test_policy": (
            "test is opened only after every configuration and seed is complete "
            "and the validation summary has been frozen"
        ),
    }
