#!/usr/bin/env python3
"""Frozen protocol for the 0910 architecture-by-cumulative-feature study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOOKBACK = 60
COMMON_ORIGIN_LOOKBACK = 120
PREDICT_STEPS = 15
SELECTION_METRIC = "validation_rmse_k"
SEEDS = (42, 52, 62, 72, 82)

MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "model": "gru_baseline",
        "label": "GRU baseline",
        "short_label": "GRU",
        "role": "temporal baseline",
    },
    {
        "model": "parallel_gru_mlp_gnn",
        "label": "Parallel GRU-MLP-GNN",
        "short_label": "Parallel GRU-MLP-GNN",
        "role": "parallel temporal and directed-graph model with MLP edge gates",
    },
    {
        "model": "parallel_gru_kan_gnn",
        "label": "Parallel GRU-KAN-GNN",
        "short_label": "Parallel GRU-KAN-GNN",
        "role": "parallel temporal and directed-graph model with monotonic KAN edge gates",
    },
)

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

# The cumulative order is fixed before training. It progresses from target-local
# dynamics toward increasingly upstream process information. Therefore the rows
# show an ordered information-integration path, not order-independent importance.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "target_history_dynamics": (
        "TE8353_diff_1", "TE8353_history_slope_k_per_min_60points",
        "Thv_diff_1", "Thv_history_slope_k_per_min_60points",
    ),
    "thv_module_coupling": (
        "Module_dT_Thv_TE8353", "Module_flow_temperature_drive_proxy",
        "Module_dT_Thv_TE8310",
    ),
    "module_valve_dynamics": ("EC_V2_diff_1", "COOLDOWN_diff_1"),
    "mainline_state_contrasts": (
        "Mainline_dT_8351_8352", "Mainline_dT_8352_8353",
        "Mainline_dP_8351_8352_bar",
    ),
    "postmix_cooling_bridge": (
        "PostMix_specific_cooling_300K_ref_J_kg",
        "PostMix_cooling_rate_300K_ref_W",
    ),
    "premix_postmix_thermal_drive": (
        "G1_to_postmix_thermal_drive_proxy",
        "G2_to_postmix_thermal_drive_proxy",
        "A_to_postmix_thermal_drive_proxy", "CV8351_diff_1",
    ),
    "g_main_flow_command": (
        "G1_main_flow_command_proxy", "G2_main_flow_command_proxy",
    ),
}

OPTIONAL_HEAT_LEAK_FEATURES = (
    "Mainline_apparent_heat_leak_8351_8352_W",
    "Mainline_apparent_heat_leak_8352_8353_W",
)
RECOMMENDED_ENGINEERED_COLUMNS = tuple(
    column for columns in FEATURE_GROUPS.values() for column in columns
)
ALL_ENGINEERED_COLUMNS = RECOMMENDED_ENGINEERED_COLUMNS + OPTIONAL_HEAT_LEAK_FEATURES
MODEL_HISTORY_COLUMNS = RAW_HISTORY_COLUMNS + ALL_ENGINEERED_COLUMNS


def _cumulative_features(count: int) -> tuple[str, ...]:
    names = tuple(FEATURE_GROUPS)
    return tuple(
        column for name in names[:count] for column in FEATURE_GROUPS[name]
    )


_STAGES: tuple[tuple[str, str], ...] = (
    ("raw20", "Raw measurements"),
    ("plus_target_dynamics", "+ Target dynamics"),
    ("plus_module_coupling", "+ Module coupling"),
    ("plus_module_valve_dynamics", "+ Module valve dynamics"),
    ("plus_mainline_contrasts", "+ Mainline contrasts"),
    ("plus_cooling_bridge", "+ Cooling bridge"),
    ("plus_thermal_drive", "+ Thermal-drive proxies"),
    ("full20", "+ G-main flow proxies"),
)

FEATURE_SPECS: tuple[dict[str, Any], ...] = tuple(
    {
        "feature_set": feature_set,
        "label": label,
        "stage_index": stage,
        "engineered": _cumulative_features(stage),
        "introduced_group": None if stage == 0 else tuple(FEATURE_GROUPS)[stage - 1],
        "role": "raw reference" if stage == 0 else "cumulative physical-feature introduction",
    }
    for stage, (feature_set, label) in enumerate(_STAGES)
)

_MODEL_BY_ID = {str(spec["model"]): spec for spec in MODEL_SPECS}
_FEATURE_BY_ID = {str(spec["feature_set"]): spec for spec in FEATURE_SPECS}
if len(_MODEL_BY_ID) != len(MODEL_SPECS) or len(_FEATURE_BY_ID) != len(FEATURE_SPECS):
    raise AssertionError("Duplicate model or feature identifiers")
if len(RECOMMENDED_ENGINEERED_COLUMNS) != 20:
    raise AssertionError("Expected exactly 20 recommended engineered features")
if len(MODEL_HISTORY_COLUMNS) != 42 or len(set(MODEL_HISTORY_COLUMNS)) != 42:
    raise AssertionError("Expected 42 unique fixed history channels")


def model_spec(model: str) -> dict[str, str]:
    try:
        return _MODEL_BY_ID[model]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model!r}; expected {tuple(_MODEL_BY_ID)}") from exc


def feature_spec(feature_set: str) -> dict[str, Any]:
    try:
        return _FEATURE_BY_ID[feature_set]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; expected {tuple(_FEATURE_BY_ID)}"
        ) from exc


def history_columns(spec: dict[str, Any]) -> tuple[str, ...]:
    # Fixed dimensionality prevents parameter-count changes from confounding
    # the cumulative feature comparison. Inactive standardized channels are 0.
    return MODEL_HISTORY_COLUMNS


def active_history_columns(spec: dict[str, Any]) -> tuple[str, ...]:
    return RAW_HISTORY_COLUMNS + tuple(spec["engineered"])


def history_feature_mask(spec: dict[str, Any]) -> tuple[float, ...]:
    active = set(active_history_columns(spec))
    return tuple(1.0 if column in active else 0.0 for column in MODEL_HISTORY_COLUMNS)


def all_tasks() -> list[tuple[dict[str, str], dict[str, Any], int]]:
    # Tasks are ordered deterministically and then assigned to workers by
    # modulo index. The default ten workers receive twelve runs each.
    return [
        (model, feature, seed)
        for model in MODEL_SPECS
        for feature in FEATURE_SPECS
        for seed in SEEDS
    ]


def run_directory(
    results_dir: Path,
    model: dict[str, str],
    feature: dict[str, Any],
    seed: int,
) -> Path:
    return (
        results_dir / "development" / f"horizon_{PREDICT_STEPS:02d}"
        / f"lookback_{LOOKBACK:02d}" / str(model["model"])
        / str(feature["feature_set"]) / f"seed_{seed}"
    )


def valid_completed_run(
    results_dir: Path,
    model: dict[str, str],
    feature: dict[str, Any],
    seed: int,
) -> bool:
    run_dir = run_directory(results_dir, model, feature, seed)
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not metrics_path.exists() or not checkpoint_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "model_type": model["model"],
        "feature_set": feature["feature_set"],
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
        "study": "0910 architecture-by-cumulative-feature matrix",
        "study_type": "paired validation-only development study",
        "target": "Delta Thv at t+15, restored to absolute Thv for metrics",
        "lookback": LOOKBACK,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "history_input": "raw and cumulative causal engineered features through t",
        "known_future_control_input": "eight raw valve commands u(t)..u(t+14)",
        "future_engineered_features_used": False,
        "training_loss": "weighted Huber; train-defined lowest-delta 10% weight=3",
        "checkpoint_selection_split": "Original 260501 validation only",
        "checkpoint_selection_metric": SELECTION_METRIC,
        "posthoc_test_split": "Original 0715-BACK; not read by this training study",
        "models": [dict(spec) for spec in MODEL_SPECS],
        "feature_group_order": [
            {"group": group, "columns": list(columns)}
            for group, columns in FEATURE_GROUPS.items()
        ],
        "feature_sets": [
            {
                **{key: value for key, value in spec.items() if key != "engineered"},
                "engineered": list(spec["engineered"]),
                "active_history_columns": list(active_history_columns(spec)),
                "masked_history_columns": [
                    column for column, keep in zip(
                        history_columns(spec), history_feature_mask(spec)
                    ) if not keep
                ],
            }
            for spec in FEATURE_SPECS
        ],
        "seeds": list(SEEDS),
        "model_config": MODEL_CONFIG,
        "expected_runs": len(all_tasks()),
        "interpretation_warning": (
            "Cumulative rows are order-dependent and demonstrate progressive "
            "information integration; they are not standalone feature-importance scores."
        ),
        "test_policy": (
            "Training and validation summary never deserialize a test pickle. "
            "Any post-hoc test evaluation must be launched separately after freezing conclusions."
        ),
    }
