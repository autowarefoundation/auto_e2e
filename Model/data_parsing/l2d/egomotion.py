"""Egomotion loading for the yaak-ai/L2D LeRobot dataset.

The dataset provides at 10 Hz:
    observation.state.vehicle float32 shape[8]:
        [speed, heading, heading_error, hp_loc_latitude, hp_loc_longitude,
         hp_loc_altitude, acceleration_x, acceleration_y]
    action.continuous float32 shape[3]:
        [gas, brake, steering]

Produces:
    egomotion_history  (256,) — 64 timesteps before the sample point × 4 signals
                                [speed, acceleration_x, yaw_rate, curvature]

    trajectory_target  (128,) — 64 timesteps after the sample point × 2 signals
                                [acceleration_x, curvature]
"""

from __future__ import annotations

import numpy as np
import torch

_HISTORY_TIMESTEPS = 64
_FUTURE_TIMESTEPS = 64
_NUM_HISTORY_SIGNALS = 4  # speed, acceleration_x, yaw_rate, curvature
_NUM_TARGET_SIGNALS = 2   # acceleration_x, curvature

EGOMOTION_DIM = _HISTORY_TIMESTEPS * _NUM_HISTORY_SIGNALS   # 256
TRAJECTORY_DIM = _FUTURE_TIMESTEPS * _NUM_TARGET_SIGNALS    # 128

MIN_FRAMES = _HISTORY_TIMESTEPS + _FUTURE_TIMESTEPS + 1  # 129

_DT = 0.1  # 10 Hz
_SPEED_EPS = 1e-6


def _derive_signals(vehicle_states: np.ndarray) -> np.ndarray:
    """Derive the 4 egomotion signals from vehicle state arrays.

    Args:
        vehicle_states: float32 array of shape (T, 8) where columns are
            [speed, heading, heading_error, lat, lon, alt, accel_x, accel_y].

    Returns:
        Float32 array of shape (T, 4): [speed, acceleration_x, yaw_rate, curvature].
    """
    speed = vehicle_states[:, 0]/3.6 # km/h to m/s
    # L2D stores heading in degrees (CW compass). Convert to standard CCW radians.
    heading = np.unwrap(np.deg2rad(-vehicle_states[:, 1]))
    accel_x = vehicle_states[:, 6]

    # yaw_rate = d(heading) / dt, with zero at boundaries
    yaw_rate = np.zeros_like(heading)
    yaw_rate[1:] = np.diff(heading) / _DT

    # curvature = yaw_rate / speed, guarding division by zero
    curvature = np.where(
        np.abs(speed) > _SPEED_EPS,
        yaw_rate / speed,
        0.0,
    )

    return np.stack([speed, accel_x, yaw_rate, curvature], axis=1).astype(np.float32)


def extract_egomotion(
    vehicle_states: np.ndarray,
    sample_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract egomotion history and trajectory target from vehicle states.

    Args:
        vehicle_states: float32 array of shape (T, 8) — the full episode or
            a sufficiently long window of observation.state.vehicle values.
        sample_idx: Index into the sequence to treat as the current moment.
            Must satisfy: _HISTORY_TIMESTEPS <= sample_idx <= T - _FUTURE_TIMESTEPS - 1.
            Defaults to the midpoint of the valid range.

    Returns:
        egomotion_history: Float tensor of shape (256,).
        trajectory_target: Float tensor of shape (128,).
    """
    T = len(vehicle_states)
    if T < MIN_FRAMES:
        raise ValueError(
            f"Need at least {MIN_FRAMES} frames, got {T}."
        )

    min_idx = _HISTORY_TIMESTEPS
    max_idx = T - _FUTURE_TIMESTEPS - 1

    if sample_idx is None:
        sample_idx = (min_idx + max_idx) // 2
    elif not (min_idx <= sample_idx <= max_idx):
        raise ValueError(
            f"sample_idx {sample_idx} out of valid range [{min_idx}, {max_idx}]."
        )

    signals = _derive_signals(vehicle_states)

    history = signals[sample_idx - _HISTORY_TIMESTEPS:sample_idx]  # (64, 4)
    future = signals[sample_idx + 1:sample_idx + 1 + _FUTURE_TIMESTEPS]  # (64, 4)

    # History: all 4 signals
    egomotion_history = torch.from_numpy(history.flatten())  # (256,)

    # Target: acceleration_x and curvature only (indices 1 and 3)
    trajectory_target = torch.from_numpy(
        future[:, [1, 3]].flatten()
    )  # (128,)

    return egomotion_history, trajectory_target
