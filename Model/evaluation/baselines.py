"""Open-loop evaluation baselines (#66 §5).

Simple, training-free baselines in the model's action space ``(accel, curv)``,
so they can be scored with the same metrics as AutoE2E. Their purpose is to
reveal whether the perception pipeline contributes **beyond ego-status
extrapolation** — if the model barely beats constant-velocity, perception isn't
helping yet (the ego-status critique of nuScenes planning, Zhai et al. 2023).

The trained "ego-status MLP" baseline from the proposal needs training and is a
follow-up; these are the zero-training references.
"""

from __future__ import annotations

import numpy as np


def constant_velocity_baseline(batch_size: int,
                               num_timesteps: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Maintain current speed, drive straight: ``accel = 0``, ``curv = 0``.

    Returns ``(accel, curv)`` each ``(batch_size, num_timesteps)``.
    """
    zeros = np.zeros((batch_size, num_timesteps), dtype=np.float64)
    return zeros, zeros.copy()


def hold_last_action_baseline(last_accel: np.ndarray, last_curv: np.ndarray,
                              num_timesteps: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Extrapolate the last observed action forward (ego-status, no perception).

    Holds the most recent ``(accel, curv)`` constant over the horizon — a
    stronger ego-only baseline than constant-velocity when the ego is mid-
    manoeuvre.

    Args:
        last_accel, last_curv: ``(B,)`` last observed action from egomotion.
    Returns:
        ``(accel, curv)`` each ``(B, num_timesteps)``.
    """
    accel = np.repeat(np.asarray(last_accel, dtype=np.float64)[:, None],
                      num_timesteps, axis=1)
    curv = np.repeat(np.asarray(last_curv, dtype=np.float64)[:, None],
                     num_timesteps, axis=1)
    return accel, curv
