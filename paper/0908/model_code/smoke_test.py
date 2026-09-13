#!/usr/bin/env python3
"""Small synthetic forward/backward test for the primary model components."""

from __future__ import annotations

import numpy as np
import torch

from graph_tools import physical_edges
from models import build_model
from protocol import FUTURE_CONTROL_COLUMNS, HISTORY_COLUMNS, project_dir


def main() -> None:
    torch.manual_seed(20260908)
    edges = physical_edges(project_dir() / "graphs" / "static_physical_graph.json", all_lags=False)
    model = build_model(
        15, np.zeros(len(HISTORY_COLUMNS)), np.ones(len(HISTORY_COLUMNS)),
        np.zeros(len(FUTURE_CONTROL_COLUMNS)), np.ones(len(FUTURE_CONTROL_COLUMNS)),
        edges, "kan"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = torch.randn(8, 60, len(HISTORY_COLUMNS))
    controls = torch.randn(8, 15, len(FUTURE_CONTROL_COLUMNS))
    target = torch.randn(8, 15)
    before = None
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(history, controls)
        loss = torch.mean((output - target) ** 2)
        if not torch.isfinite(loss):
            raise AssertionError("Non-finite smoke loss")
        before = float(loss.item()) if before is None else before
        loss.backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise AssertionError("Non-finite gradient")
        optimizer.step()
    print({"status": "PASS", "shape": tuple(output.shape), "initial_loss": before, "final_loss": float(loss.item()), "edges": len(edges)})


if __name__ == "__main__":
    main()
