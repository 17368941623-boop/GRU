#!/usr/bin/env python3
"""Frozen protocol for the 0906 Thv architecture comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOOKBACK = 60
COMMON_ORIGIN_LOOKBACK = 120
PREDICT_STEPS = 15
SELECTION_METRIC = "validation_rmse_k"

SEEDS_10 = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)
TCN_SEEDS = (42, 62, 82, 102, 122)
TRAINING_GROUPS = ("gru_family", "lstm_family")

COMMON_CONFIG: dict[str, int | float] = {
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

MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "model": "gru_baseline",
        "label": "GRU baseline",
        "config_id": "controlled_hh64",
        "role": "temporal baseline with known-future-control encoder",
        "training_group": "gru_family",
        "seeds": SEEDS_10,
    },
    {
        "model": "lstm_baseline",
        "label": "LSTM baseline",
        "config_id": "controlled_hh64",
        "role": "recurrent temporal-encoder comparator",
        "training_group": "lstm_family",
        "seeds": SEEDS_10,
    },
    {
        "model": "tcn_baseline",
        "label": "TCN baseline",
        "config_id": "controlled_hh64_tl6",
        "role": "causal TCN comparator with a 127-step receptive field",
        "training_group": "lstm_family",
        "seeds": TCN_SEEDS,
    },
    {
        "model": "parallel_gru_mlp_gnn",
        "label": "Parallel GRU-MLP-GNN",
        "config_id": "controlled_hh64_gh64_gs2_mlp",
        "role": "parallel directed topology with unconstrained MLP edge gates",
        "training_group": "gru_family",
        "seeds": SEEDS_10,
    },
    {
        "model": "parallel_gru_kan_gnn",
        "label": "Parallel GRU-monotonic-KAN-GNN",
        "config_id": "controlled_hh64_gh64_gs2_kan8",
        "role": "parallel directed topology with monotonic KAN valve gates",
        "training_group": "gru_family",
        "seeds": SEEDS_10,
    },
    {
        "model": "parallel_lstm_mlp_gnn",
        "label": "Parallel LSTM-MLP-GNN",
        "config_id": "controlled_hh64_gh64_gs2_mlp",
        "role": (
            "parallel LSTM history and directed topology with unconstrained "
            "MLP edge gates"
        ),
        "training_group": "lstm_family",
        "seeds": SEEDS_10,
    },
    {
        "model": "parallel_lstm_kan_gnn",
        "label": "Parallel LSTM-monotonic-KAN-GNN",
        "config_id": "controlled_hh64_gh64_gs2_kan8",
        "role": (
            "parallel LSTM history and directed topology with monotonic KAN "
            "valve gates"
        ),
        "training_group": "lstm_family",
        "seeds": SEEDS_10,
    },
    {
        "model": "serial_mlp_gnn_gru",
        "label": "Serial MLP-GNN-GRU",
        "config_id": "controlled_hh64_gh64_gs2_mlp",
        "role": "serial directed topology with unconstrained MLP edge gates",
        "training_group": "gru_family",
        "seeds": SEEDS_10,
    },
    {
        "model": "serial_kan_gnn_gru",
        "label": "Serial monotonic-KAN-GNN-GRU",
        "config_id": "controlled_hh64_gh64_gs2_kan8",
        "role": "serial directed topology with monotonic KAN valve gates",
        "training_group": "gru_family",
        "seeds": SEEDS_10,
    },
)


def all_tasks() -> list[tuple[dict[str, Any], int]]:
    return [(spec, int(seed)) for spec in MODEL_SPECS for seed in spec["seeds"]]


def tasks_for_group(group: str) -> list[tuple[dict[str, Any], int]]:
    if group == "all":
        return all_tasks()
    if group not in TRAINING_GROUPS:
        raise ValueError(
            f"Unknown training group {group!r}; expected one of {TRAINING_GROUPS}"
        )
    return [
        (spec, int(seed))
        for spec in MODEL_SPECS
        if spec["training_group"] == group
        for seed in spec["seeds"]
    ]


def run_directory(results_dir: Path, spec: dict[str, Any], seed: int) -> Path:
    return (
        results_dir
        / "development"
        / f"horizon_{PREDICT_STEPS:02d}"
        / f"lookback_{LOOKBACK:02d}"
        / str(spec["model"])
        / str(spec["config_id"])
        / f"seed_{seed}"
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
        "model_type": spec["model"],
        "config_id": spec["config_id"],
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
        "study": "0906 final Thv architecture comparison",
        "target": "Delta Thv at t+15, restored to absolute Thv for metrics",
        "lookback": LOOKBACK,
        "common_origin_lookback": COMMON_ORIGIN_LOOKBACK,
        "predict_steps": PREDICT_STEPS,
        "history_input": "measurements and controls through current time t",
        "known_future_control_input": "valve commands u(t) through u(t+14)",
        "future_measurements_used": False,
        "training_loss": "weighted Huber; training-defined rapid subset weight=3",
        "checkpoint_selection_split": "Original 260501 validation only",
        "checkpoint_selection_metric": SELECTION_METRIC,
        "test_split": "Original 0715-BACK complete forecastable sequence",
        "test_used_during_training": False,
        "models": [
            {
                **{key: value for key, value in spec.items() if key != "seeds"},
                "seeds": list(spec["seeds"]),
            }
            for spec in MODEL_SPECS
        ],
        "common_model_config": COMMON_CONFIG,
        "expected_runs": len(all_tasks()),
    }
