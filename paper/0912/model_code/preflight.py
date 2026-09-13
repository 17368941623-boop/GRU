"""Verify shapes, splits and one forward/backward pass before full training."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional as F

from data_pipeline import (
    TrajectoryWindowDataset,
    fit_scalers,
    load_cached_frames,
    prepare_split,
    valid_window_ends,
)
from model import Raw54GRU, parameter_count
from protocol import FUTURE_CONTROL_COLUMNS, PREDICT_STEPS, cache_dir, project_dir


def main() -> None:
    train, validation, raw_columns = load_cached_frames(cache_dir())
    train_ends = valid_window_ends(train, raw_columns, PREDICT_STEPS)
    validation_ends = valid_window_ends(validation, raw_columns, PREDICT_STEPS)
    scalers = fit_scalers(train, train_ends, raw_columns, PREDICT_STEPS, 0.10)
    train_split = prepare_split(train, raw_columns, PREDICT_STEPS, scalers, 3.0)
    validation_split = prepare_split(validation, raw_columns, PREDICT_STEPS, scalers, 3.0)
    dataset = TrajectoryWindowDataset(train_split)
    history, controls, target, weight, _ = dataset[0]
    model = Raw54GRU(len(raw_columns), len(FUTURE_CONTROL_COLUMNS), PREDICT_STEPS)
    prediction = model(history[None], controls[None])
    loss = torch.sum(F.smooth_l1_loss(prediction, target[None], reduction="none") * weight[None])
    loss.backward()
    report = {
        "status": "PASS",
        "raw_signal_count": len(raw_columns),
        "engineered_feature_count": 0,
        "pcmci_edges": 0,
        "gnn_enabled": False,
        "kan_enabled": False,
        "history_shape_one_sample": list(history.shape),
        "future_control_shape_one_sample": list(controls.shape),
        "target_shape_one_sample": list(target.shape),
        "prediction_shape_one_sample": list(prediction.shape),
        "train_rows": len(train),
        "train_windows": len(train_ends),
        "validation_rows": len(validation),
        "validation_windows": len(validation_ends),
        "parameter_count": parameter_count(model),
        "finite_train_windows": bool(np.isfinite(train_split.delta_scaled).all()),
        "finite_validation_windows": bool(np.isfinite(validation_split.delta_scaled).all()),
        "frozen_test_loaded": False,
    }
    path = project_dir() / "reports" / "preflight_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
