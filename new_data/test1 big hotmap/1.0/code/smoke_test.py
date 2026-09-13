#!/usr/bin/env python3
"""Synthetic forward/backward check for every hidden size."""

from __future__ import annotations

import torch

from protocol import HIDDEN_SIZES, PREDICT_STEPS
from train_global_cell import ControlledGRU


def main() -> None:
    torch.manual_seed(7)
    for hidden_size in HIDDEN_SIZES:
        model = ControlledGRU(54, 18, hidden_size)
        history = torch.randn(4, 90, 54)
        controls = torch.randn(4, PREDICT_STEPS, 18)
        target = torch.randn(4)
        output = model(history, controls)
        if output.shape != target.shape or not torch.isfinite(output).all():
            raise AssertionError(f"Invalid output for hidden_size={hidden_size}")
        loss = torch.mean((output - target) ** 2)
        loss.backward()
        if not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise AssertionError(f"Invalid gradients for hidden_size={hidden_size}")
        print(f"SMOKE_OK=hidden_{hidden_size}")
    print("SMOKE_TEST_COMPLETE=true")
    print("PROJECT_DATA_READ=false")


if __name__ == "__main__":
    main()

