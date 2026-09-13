#!/usr/bin/env python3
"""Synthetic forward/backward checks; project data are never opened."""

from __future__ import annotations

import numpy as np
import torch

from heatmap_protocol import (
    FEATURE_SPECS,
    MODEL_CONFIG,
    MODEL_HISTORY_COLUMNS,
    MODEL_SPECS,
    PREDICT_STEPS,
    history_feature_mask,
)
from model_components import ModelConfig, build_model

CONTROLS = ("CV8312", "CV8311", "CV8310", "CV8313", "CV8300", "CV8351", "EC-V2", "COOLDOWN")


def main() -> None:
    torch.manual_seed(7)
    parameter_counts: dict[str, set[int]] = {}
    for model_spec in MODEL_SPECS:
        model_name = str(model_spec["model"])
        parameter_counts[model_name] = set()
        for feature in FEATURE_SPECS:
            model = build_model(
                ModelConfig(model_name=model_name, **MODEL_CONFIG),
                MODEL_HISTORY_COLUMNS,
                CONTROLS,
                PREDICT_STEPS,
                np.zeros(len(MODEL_HISTORY_COLUMNS)),
                np.ones(len(MODEL_HISTORY_COLUMNS)),
            )
            mask = torch.tensor(history_feature_mask(feature), dtype=torch.float32)
            history = torch.randn(2, 60, len(MODEL_HISTORY_COLUMNS)) * mask
            controls = torch.randn(2, PREDICT_STEPS, len(CONTROLS))
            output = model(history, controls)
            if output.shape != (2,) or not torch.isfinite(output).all():
                raise AssertionError(f"{model_name}/{feature['feature_set']}")
            output.square().mean().backward()
            if not any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise AssertionError("No finite gradients")
            count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            parameter_counts[model_name].add(count)
        if len(parameter_counts[model_name]) != 1:
            raise AssertionError(f"Feature masking changed parameters for {model_name}")
        print(f"SMOKE_OK={model_name} | parameters={next(iter(parameter_counts[model_name]))}")
    print("SMOKE_TEST_COMPLETE=true")
    print("PROJECT_DATA_READ=false")


if __name__ == "__main__":
    main()
