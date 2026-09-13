#!/usr/bin/env python3
"""Frozen experiment protocol for the 0908 Thv trajectory study."""

from __future__ import annotations

from pathlib import Path


SAMPLE_PERIOD_SECONDS = 10
LOOKBACK = 60
COMMON_ORIGIN_LOOKBACK = 120
DEFAULT_PREDICT_STEPS = 15
MAX_CAUSAL_LAG = 30
TARGET = "Thv"

SEEDS = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)

RAW_COLUMNS = (
    "TE8310", "TE8351", "TE8352", "FT8351", "PT8310", "PT8351",
    "PT8352", "CV8312", "CV8311", "CV8310", "CV8313", "CV8300",
    "CV8351", "Thv", "TE8353", "Tef", "Tcd", "DTbr", "EC-V2",
    "COOLDOWN",
)

ENGINEERED_COLUMNS = (
    "TE8353_diff_1", "TE8353_history_slope_k_per_min_60points",
    "Thv_diff_1", "Thv_history_slope_k_per_min_60points",
    "G1_main_flow_command_proxy", "G2_main_flow_command_proxy",
    "G1_to_postmix_thermal_drive_proxy",
    "G2_to_postmix_thermal_drive_proxy",
    "A_to_postmix_thermal_drive_proxy", "CV8351_diff_1",
    "PostMix_specific_cooling_300K_ref_J_kg",
    "PostMix_cooling_rate_300K_ref_W",
    "Mainline_dT_8351_8352", "Mainline_dT_8352_8353",
    "Mainline_dP_8351_8352_bar", "Module_dT_Thv_TE8353",
    "Module_flow_temperature_drive_proxy", "Module_dT_Thv_TE8310",
    "EC_V2_diff_1", "COOLDOWN_diff_1",
)

HISTORY_COLUMNS = RAW_COLUMNS + ENGINEERED_COLUMNS
FUTURE_CONTROL_COLUMNS = (
    "CV8312", "CV8311", "CV8310", "CV8313", "CV8300", "CV8351",
    "EC-V2", "COOLDOWN",
)
VALVE_COLUMNS = frozenset(FUTURE_CONTROL_COLUMNS)

EXPERIMENTS = {
    "gru_full20": {
        "graph": None,
        "gate": "none",
        "description": "Parallel-branch ablation: GRU trajectory model without GNN.",
    },
    "physical_all_lags_kan": {
        "graph": "physical_all_lags.json",
        "gate": "kan",
        "description": "Static P&ID graph expanded over each documented lag band.",
    },
    "cd_select_lag_kan": {
        "graph": "cd_select_lag.json",
        "gate": "kan",
        "description": "PCMCI-only stable lag-selected graph.",
    },
    "dkcdv_select_lag_kan": {
        "graph": "dkcdv_select_lag.json",
        "gate": "kan",
        "description": "Primary: domain-validated PCMCI graph plus supported prior edges.",
    },
    "dkcdl_select_lag_kan": {
        "graph": "dkcdl_select_lag.json",
        "gate": "kan",
        "description": "PCMCI graph limited to physically admissible variable pairs.",
    },
    "dkcdv_select_lag_mlp": {
        "graph": "dkcdv_select_lag.json",
        "gate": "mlp",
        "description": "KAN-versus-unconstrained-MLP edge-gate ablation.",
    },
    "dkcdv_shuffled_lag_kan": {
        "graph": "dkcdv_shuffled_lag.json",
        "gate": "kan",
        "description": "Negative control: identical pairs with shuffled delays.",
    },
    "random_same_size_kan": {
        "graph": "random_same_size.json",
        "gate": "kan",
        "description": "Negative control: random graph with the primary graph's edge count.",
    },
}


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def generated_graph_dir() -> Path:
    return project_dir() / "graphs" / "generated"


def run_dir(output_root: Path, experiment: str, horizon: int, seed: int) -> Path:
    return (
        output_root / "development" / f"horizon_{horizon:02d}"
        / f"lookback_{LOOKBACK:02d}" / experiment / f"seed_{seed}"
    )


def assert_protocol() -> None:
    if len(RAW_COLUMNS) != 20 or len(ENGINEERED_COLUMNS) != 20:
        raise AssertionError("The frozen full20 protocol must contain 20 raw + 20 engineered columns")
    if len(set(HISTORY_COLUMNS)) != len(HISTORY_COLUMNS):
        raise AssertionError("Duplicate history column")
    if TARGET not in RAW_COLUMNS:
        raise AssertionError("Target is absent from raw graph nodes")
    if any(name.startswith("Future_") for name in HISTORY_COLUMNS):
        raise AssertionError("A future label leaked into the history input")


assert_protocol()
