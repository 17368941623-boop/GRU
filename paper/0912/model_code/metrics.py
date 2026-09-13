"""Validation metrics for direct multi-step Thv trajectories."""

from __future__ import annotations

import numpy as np

from protocol import SAMPLE_PERIOD_SECONDS


def trajectory_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    current: np.ndarray,
    rapid_threshold: np.ndarray,
    valve_event: np.ndarray,
) -> dict[str, object]:
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("actual and predicted must have identical [sample, horizon] shapes")
    error = predicted - actual
    delta_actual = actual - current[:, None]
    delta_predicted = predicted - current[:, None]
    rmse_h = np.sqrt(np.mean(error ** 2, axis=0))
    mae_h = np.mean(np.abs(error), axis=0)
    persistence_error = current[:, None] - actual
    rapid = delta_actual[:, -1] <= float(rapid_threshold[-1])
    peak_actual_index = np.argmin(delta_actual, axis=1)
    peak_predicted_index = np.argmin(delta_predicted, axis=1)
    peak_actual = np.take_along_axis(delta_actual, peak_actual_index[:, None], axis=1)[:, 0]
    peak_predicted = np.take_along_axis(delta_predicted, peak_predicted_index[:, None], axis=1)[:, 0]
    integrated_error = np.trapz(
        np.concatenate((np.zeros((len(error), 1)), error), axis=1),
        dx=SAMPLE_PERIOD_SECONDS,
        axis=1,
    )
    result: dict[str, object] = {
        "samples": int(len(actual)),
        "trajectory_rmse_k": float(np.sqrt(np.mean(error ** 2))),
        "trajectory_mae_k": float(np.mean(np.abs(error))),
        "final_horizon_rmse_k": float(rmse_h[-1]),
        "final_horizon_mae_k": float(mae_h[-1]),
        "p95_absolute_error_k": float(np.quantile(np.abs(error), 0.95)),
        "persistence_trajectory_rmse_k": float(np.sqrt(np.mean(persistence_error ** 2))),
        "persistence_final_rmse_k": float(np.sqrt(np.mean(persistence_error[:, -1] ** 2))),
        "peak_cooling_delta_mae_k": float(np.mean(np.abs(peak_predicted - peak_actual))),
        "peak_cooling_time_mae_seconds": float(
            np.mean(np.abs(peak_predicted_index - peak_actual_index)) * SAMPLE_PERIOD_SECONDS
        ),
        "mean_absolute_integrated_error_k_s": float(np.mean(np.abs(integrated_error))),
        "direction_accuracy": float(np.mean(np.sign(delta_predicted[:, -1]) == np.sign(delta_actual[:, -1]))),
        "rmse_by_horizon_k": rmse_h.tolist(),
        "mae_by_horizon_k": mae_h.tolist(),
    }
    if rapid.any():
        result["rapid_samples"] = int(rapid.sum())
        result["rapid_trajectory_rmse_k"] = float(np.sqrt(np.mean(error[rapid] ** 2)))
        result["rapid_final_rmse_k"] = float(np.sqrt(np.mean(error[rapid, -1] ** 2)))
    if valve_event.any():
        result["valve_event_samples"] = int(valve_event.sum())
        result["valve_event_trajectory_rmse_k"] = float(
            np.sqrt(np.mean(error[valve_event] ** 2))
        )
        result["valve_event_final_rmse_k"] = float(
            np.sqrt(np.mean(error[valve_event, -1] ** 2))
        )
    return result
