#!/usr/bin/env python3
"""Export learned monotonic valve-conductance curves from a frozen KAN run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_pipeline import ScalerBundle
from models import MonotonicEdgeSpline, build_model
from protocol import FUTURE_CONTROL_COLUMNS, VALVE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--points", type=int, default=101)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["experiment_spec"]["gate"] != "kan":
        raise ValueError("Checkpoint does not use KAN gates")
    scalers = ScalerBundle.from_payload(checkpoint["scalers"])
    model = build_model(
        int(checkpoint["predict_steps"]), scalers.history.mean, scalers.history.scale,
        scalers.controls.mean, scalers.controls.scale,
        checkpoint["edges"], "kan",
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    destination = args.output_dir or args.checkpoint.parent / "kan_interpretability"
    destination.mkdir(parents=True, exist_ok=True)
    opening = torch.linspace(0.0, 1.0, args.points)

    control_gate = model.control_mapping
    if not isinstance(control_gate, MonotonicEdgeSpline):
        raise TypeError("Future-control KAN gate is missing")
    control_input = opening[:, None].expand(-1, len(FUTURE_CONTROL_COLUMNS))
    control_values = control_gate(control_input).detach().numpy()
    control_rows = []
    for index, valve in enumerate(FUTURE_CONTROL_COLUMNS):
        for percent, value in zip(opening.numpy() * 100.0, control_values[:, index]):
            control_rows.append({"valve": valve, "opening_percent": percent, "conductance_index": value})
    pd.DataFrame(control_rows).to_csv(destination / "future_control_kan_curves.csv", index=False)

    edge_rows = []
    if model.graph is not None and isinstance(model.graph.edge_gate, MonotonicEdgeSpline):
        edges = checkpoint["edges"]
        edge_count = len(edges)
        edge_input = opening[:, None].expand(-1, edge_count)
        values = model.graph.edge_gate(edge_input).detach().numpy()
        for index, edge in enumerate(edges):
            if str(edge["source"]) not in VALVE_COLUMNS:
                continue
            edge_id = f"{edge['source']}__to__{edge['destination']}__lag{edge['lag_steps']}"
            for percent, value in zip(opening.numpy() * 100.0, values[:, index]):
                edge_rows.append(
                    {
                        "edge_id": edge_id, "source": edge["source"],
                        "destination": edge["destination"], "lag_steps": edge["lag_steps"],
                        "opening_percent": percent, "conductance_gate": value,
                    }
                )
    pd.DataFrame(edge_rows).to_csv(destination / "graph_edge_kan_curves.csv", index=False)
    print(f"saved {len(control_rows)} control points and {len(edge_rows)} graph-edge points to {destination}")


if __name__ == "__main__":
    main()
