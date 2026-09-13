#!/usr/bin/env python3
"""Synthetic forward/backward checks; does not read any project dataset."""

from __future__ import annotations

import numpy as np
import torch

from model_components import (
    MODEL_NAMES,
    ModelConfig,
    MonotonicSplineKAN,
    build_model,
)
from study_protocol import MODEL_SPECS


HISTORY_FEATURES = (
    "TE8310", "TE8351", "TE8352", "FT8351", "PT8310", "PT8351",
    "PT8352", "CV8312", "CV8311", "CV8310", "CV8313", "CV8300",
    "CV8351", "Thv", "TE8353", "Tef", "Tcd", "DTbr", "EC-V2",
    "COOLDOWN",
)
CONTROL_FEATURES = (
    "CV8312", "CV8311", "CV8310", "CV8313", "CV8300", "CV8351",
    "EC-V2", "COOLDOWN",
)


def main() -> None:
    torch.manual_seed(123)
    batch = 3
    lookback = 20
    horizon = 15
    history = torch.randn(batch, lookback, len(HISTORY_FEATURES))
    controls = torch.randn(batch, horizon, len(CONTROL_FEATURES))
    mean = np.zeros(len(HISTORY_FEATURES), dtype=np.float32)
    scale = np.ones(len(HISTORY_FEATURES), dtype=np.float32)

    study_model_names = tuple(str(spec["model"]) for spec in MODEL_SPECS)
    if len(study_model_names) != len(set(study_model_names)):
        raise AssertionError("Duplicate model names in the frozen study protocol")
    if not set(study_model_names).issubset(MODEL_NAMES):
        raise AssertionError("The study protocol names an unsupported model")

    for model_name in study_model_names:
        config = ModelConfig(
            model_name=model_name,
            history_hidden=16,
            graph_hidden=16,
            graph_sweeps=1,
            edge_hidden=8,
            kan_grid=5,
            tcn_levels=2,
            control_hidden=8,
            fusion_hidden=16,
            dropout=0.0,
        )
        model = build_model(
            config,
            HISTORY_FEATURES,
            CONTROL_FEATURES,
            horizon,
            mean,
            scale,
        )
        prediction = model(history, controls)
        if prediction.shape != (batch,):
            raise AssertionError(
                f"{model_name} returned shape {tuple(prediction.shape)}"
            )
        if not torch.isfinite(prediction).all():
            raise AssertionError(f"{model_name} returned NaN/inf")
        prediction.square().mean().backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable or not any(parameter.grad is not None for parameter in trainable):
            raise AssertionError(f"{model_name} did not backpropagate")
        print(f"SMOKE_OK={model_name}")

    kan = MonotonicSplineKAN(edge_count=2, grid_size=8)
    opening = torch.linspace(0.0, 1.0, 101).unsqueeze(1).repeat(1, 2)
    response = kan(opening)
    if torch.any(response[1:] - response[:-1] < -1e-7):
        raise AssertionError("MonotonicSplineKAN is not non-decreasing")
    print("SMOKE_OK=monotonic_valve_KAN")
    print(f"STUDY_MODELS_CHECKED={len(study_model_names)}")
    print("SMOKE_TEST_COMPLETE=true")
    print("PROJECT_DATA_READ=false")


if __name__ == "__main__":
    main()
