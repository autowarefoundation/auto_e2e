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
# Physical speed floor (m/s) for the curvature = yaw_rate / speed division.
# A tiny epsilon (1e-6) is NOT enough: a creeping frame at ~1e-3 m/s with heading
# jitter passes it yet yields curvature ~100+ rad/m (radius < 1 cm), which enters
# trajectory_target directly and spikes the loss. Floor at a real crawl speed and
# clamp the magnitude to a physical bound.
_SPEED_FLOOR = 0.5          # m/s (below this, treat as stationary → curvature 0)
_MAX_CURVATURE = 0.5        # rad/m (radius >= 2 m; tighter than any road maneuver)


def _derive_signals(vehicle_states: np.ndarray) -> np.ndarray:
    """Derive the 4 egomotion signals from vehicle state arrays.

    Args:
        vehicle_states: float32 array of shape (T, 8) where columns are
            [speed, heading, heading_error, lat, lon, alt, accel_x, accel_y].

    Returns:
        Float32 array of shape (T, 4): [speed, acceleration_x, yaw_rate, curvature].

    Units (verified against yaak-ai/L2D meta/stats.json): the raw ``speed`` is in
    KM/H (max ~171.8 → 47.7 m/s) and ``heading`` is in DEGREES (range 0..360).
    Downstream physics (``integrate_trajectory``) and the curvature formula
    ``yaw_rate/speed`` both require SI units, so we convert speed → m/s and
    heading → radians here. Without this the derived yaw_rate is ~57× too large
    (deg vs rad) and speed ~3.6× too large, which makes the integrated ego
    trajectory (and hence ADE/FDE) explode.
    """
    speed = vehicle_states[:, 0] / 3.6                    # km/h → m/s
    heading = np.unwrap(np.radians(vehicle_states[:, 1]))  # degrees → radians
    accel_x = vehicle_states[:, 6]                         # already m/s²

    # yaw_rate = d(heading) / dt, with zero at boundaries
    yaw_rate = np.zeros_like(heading)
    yaw_rate[1:] = np.diff(heading) / _DT

    # curvature = yaw_rate / speed (rad/m). Below the crawl-speed floor treat the
    # vehicle as stationary (curvature 0) rather than dividing by a near-zero
    # speed; then clamp to a physical bound so no target sample can spike.
    curvature = np.where(
        speed > _SPEED_FLOOR,
        yaw_rate / np.maximum(speed, _SPEED_FLOOR),
        0.0,
    )
    curvature = np.clip(curvature, -_MAX_CURVATURE, _MAX_CURVATURE)

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
