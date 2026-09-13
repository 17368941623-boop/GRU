#!/usr/bin/env python3
"""Data, graph, leakage and causal-prefix checks before server training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data_pipeline import exclude_0617, load_frame, valid_window_ends
from graph_tools import bootstrap_physical_graphs, physical_edges, validate_edges
from models import MonotonicEdgeSpline, build_model
from protocol import (
    FUTURE_CONTROL_COLUMNS,
    HISTORY_COLUMNS,
    RAW_COLUMNS,
    project_dir,
)


def main() -> None:
    root = project_dir()
    data_dir = root / "processed_data"
    static_path = root / "graphs" / "static_physical_graph.json"
    generated = root / "graphs" / "generated"
    bootstrap_physical_graphs(static_path, generated)
    report: dict[str, object] = {
        "history_features": len(HISTORY_COLUMNS),
        "raw_graph_nodes": len(RAW_COLUMNS),
        "future_controls": list(FUTURE_CONTROL_COLUMNS),
        "future_noncontrol_measurements_used": False,
    }
    split_rows = {}
    split_windows = {}
    for name, filename in (
        ("train", "train_clean.pkl"),
        ("validation", "val_clean.pkl"),
        ("test_full", "test_full_clean.pkl"),
    ):
        frame = load_frame(data_dir / filename)
        if name == "train":
            before = len(frame)
            frame = exclude_0617(frame)
            report["train_0617_rows_excluded"] = before - len(frame)
        split_rows[name] = len(frame)
        split_windows[name] = {
            "h15": len(valid_window_ends(frame, 15)),
            "h30": len(valid_window_ends(frame, 30)),
        }
        if any(column.startswith("Future_") for column in HISTORY_COLUMNS):
            raise AssertionError("Future label leakage in HISTORY_COLUMNS")
    report["rows"] = split_rows
    report["valid_windows"] = split_windows

    edges = physical_edges(static_path, all_lags=False)
    validate_edges(edges)
    report["static_relations"] = len(edges)
    mean = np.zeros(len(HISTORY_COLUMNS), dtype=np.float64)
    scale = np.ones(len(HISTORY_COLUMNS), dtype=np.float64)
    for horizon in (15, 30):
        model = build_model(
            horizon, mean, scale,
            np.zeros(len(FUTURE_CONTROL_COLUMNS)), np.ones(len(FUTURE_CONTROL_COLUMNS)),
            edges, "kan",
        )
        history = torch.randn(2, 60, len(HISTORY_COLUMNS))
        controls = torch.randn(2, horizon, len(FUTURE_CONTROL_COLUMNS), requires_grad=True)
        output = model(history, controls)
        if output.shape != (2, horizon):
            raise AssertionError("Trajectory output shape is wrong")
        gradient = torch.autograd.grad(output[:, 0].sum(), controls)[0]
        future_leak = float(gradient[:, 1:, :].abs().max()) if horizon > 1 else 0.0
        if future_leak > 1e-10:
            raise AssertionError(f"Control-prefix leakage detected: {future_leak}")
        report[f"h{horizon}_first_output_future_control_max_gradient"] = future_leak

    spline = MonotonicEdgeSpline(3, grid_size=8)
    grid = torch.linspace(0.0, 1.0, 101)[:, None].expand(-1, 3)
    spline_value = spline(grid).detach().numpy()
    minimum_increment = float(np.diff(spline_value, axis=0).min())
    if minimum_increment < -1e-7:
        raise AssertionError("KAN valve gate is not monotonic")
    report["kan_minimum_grid_increment"] = minimum_increment
    report["status"] = "PASS"
    destination = root / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "preflight_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
