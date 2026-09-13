#!/usr/bin/env python3
"""Trajectory metrics used consistently for validation and frozen test."""

from __future__ import annotations

import numpy as np

from protocol import SAMPLE_PERIOD_SECONDS


def trajectory_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    current: np.ndarray,
    rapid_threshold: np.ndarray,
    valve_event: np.ndarray | None = None,
) -> dict[str, object]:
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("actual and predicted must have identical [samples, horizon] shapes")
    error = predicted - actual
    horizon = actual.shape[1]
    rmse_by_horizon = np.sqrt(np.mean(error ** 2, axis=0))
    mae_by_horizon = np.mean(np.abs(error), axis=0)
    persistence_error = current[:, None] - actual
    actual_delta = actual - current[:, None]
    predicted_delta = predicted - current[:, None]

    rapid_sample = actual_delta[:, -1] <= float(rapid_threshold[-1])
    peak_actual_index = np.argmin(actual_delta, axis=1)
    peak_predicted_index = np.argmin(predicted_delta, axis=1)
    peak_actual_delta = np.take_along_axis(actual_delta, peak_actual_index[:, None], axis=1)[:, 0]
    peak_predicted_delta = np.take_along_axis(predicted_delta, peak_predicted_index[:, None], axis=1)[:, 0]
    # Include the known zero residual at the forecast origin t, yielding H full intervals.
    integrated_error = np.trapz(
        np.concatenate((np.zeros((len(error), 1)), error), axis=1),
        dx=SAMPLE_PERIOD_SECONDS,
        axis=1,
    )

    payload: dict[str, object] = {
        "samples": int(len(actual)),
        "horizon_steps": horizon,
        "horizon_seconds": horizon * SAMPLE_PERIOD_SECONDS,
        "trajectory_rmse_k": float(np.sqrt(np.mean(error ** 2))),
        "trajectory_mae_k": float(np.mean(np.abs(error))),
        "mean_horizon_rmse_k": float(np.mean(rmse_by_horizon)),
        "final_horizon_rmse_k": float(rmse_by_horizon[-1]),
        "final_horizon_mae_k": float(mae_by_horizon[-1]),
        "persistence_trajectory_rmse_k": float(np.sqrt(np.mean(persistence_error ** 2))),
        "persistence_final_rmse_k": float(np.sqrt(np.mean(persistence_error[:, -1] ** 2))),
        "peak_cooling_delta_mae_k": float(np.mean(np.abs(peak_predicted_delta - peak_actual_delta))),
        "peak_cooling_time_mae_seconds": float(
            np.mean(np.abs(peak_predicted_index - peak_actual_index)) * SAMPLE_PERIOD_SECONDS
        ),
        "mean_absolute_integrated_error_k_s": float(np.mean(np.abs(integrated_error))),
        "rmse_by_horizon_k": rmse_by_horizon.tolist(),
        "mae_by_horizon_k": mae_by_horizon.tolist(),
    }
    if rapid_sample.any():
        rapid_error = error[rapid_sample]
        payload.update(
            {
                "rapid_samples": int(rapid_sample.sum()),
                "rapid_trajectory_rmse_k": float(np.sqrt(np.mean(rapid_error ** 2))),
                "rapid_final_rmse_k": float(np.sqrt(np.mean(rapid_error[:, -1] ** 2))),
            }
        )
    if valve_event is not None:
        event = np.asarray(valve_event, dtype=bool)
        if event.shape != (len(actual),):
            raise ValueError("valve_event must contain one flag per sample")
        if event.any():
            event_error = error[event]
            payload.update(
                {
                    "valve_event_samples": int(event.sum()),
                    "valve_event_trajectory_rmse_k": float(np.sqrt(np.mean(event_error ** 2))),
                    "valve_event_final_rmse_k": float(np.sqrt(np.mean(event_error[:, -1] ** 2))),
                }
            )
    return payload
