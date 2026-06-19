"""Open-loop evaluation metrics for AutoE2E trajectory prediction.

The model predicts (acceleration_x, curvature) at 10Hz for 64 timesteps (6.4s).
To compute ADE/FDE, we integrate these signals into (x, y) positions and compare
against ground truth integrated from the same initial state.

Usage:
    from evaluation.metrics import compute_open_loop_metrics, gate_check

    metrics = compute_open_loop_metrics(pred_accel, pred_curv, gt_accel, gt_curv,
                                         initial_speed, initial_heading)
    passed = gate_check(metrics)
"""

from __future__ import annotations

import numpy as np


def integrate_trajectory(
    accel: np.ndarray,
    curvature: np.ndarray,
    v0: float,
    theta0: float = 0.0,
    dt: float = 0.1,
) -> np.ndarray:
    """Integrate acceleration + curvature into (x, y) positions.

    Args:
        accel: (T,) predicted longitudinal acceleration (m/s^2).
        curvature: (T,) predicted path curvature (1/m).
        v0: Initial speed (m/s) from egomotion history.
        theta0: Initial heading (rad). Default 0 = ego-centric frame.
        dt: Timestep (s). Default 0.1 = 10Hz.

    Returns:
        (T, 2) array of [x, y] positions relative to initial pose.
    """
    T = len(accel)
    positions = np.zeros((T, 2), dtype=np.float64)
    v = float(v0)
    theta = float(theta0)
    x, y = 0.0, 0.0

    for t in range(T):
        v = max(0.0, v + float(accel[t]) * dt)
        theta = theta + float(curvature[t]) * v * dt
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt
        positions[t] = [x, y]

    return positions


def compute_open_loop_metrics(
    pred_accel: np.ndarray,
    pred_curv: np.ndarray,
    gt_accel: np.ndarray,
    gt_curv: np.ndarray,
    initial_speed: np.ndarray,
    initial_heading: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute ADE/FDE and signal-level metrics over a batch.

    Args:
        pred_accel: (B, 64) predicted acceleration.
        pred_curv: (B, 64) predicted curvature.
        gt_accel: (B, 64) ground truth acceleration.
        gt_curv: (B, 64) ground truth curvature.
        initial_speed: (B,) speed at prediction start.
        initial_heading: (B,) heading at prediction start. None = all zeros.

    Returns:
        Dict of metric name → value.
    """
    B = pred_accel.shape[0]
    if initial_heading is None:
        initial_heading = np.zeros(B)

    ade_1s, ade_2s, ade_3s, ade_full, fde_full = [], [], [], [], []

    for i in range(B):
        pred_xy = integrate_trajectory(pred_accel[i], pred_curv[i],
                                       initial_speed[i], initial_heading[i])
        gt_xy = integrate_trajectory(gt_accel[i], gt_curv[i],
                                     initial_speed[i], initial_heading[i])
        errors = np.linalg.norm(pred_xy - gt_xy, axis=1)

        ade_1s.append(errors[:10].mean())
        ade_2s.append(errors[:20].mean())
        ade_3s.append(errors[:30].mean())
        ade_full.append(errors.mean())
        fde_full.append(errors[-1])

    return {
        "ADE@1s": float(np.mean(ade_1s)),
        "ADE@2s": float(np.mean(ade_2s)),
        "ADE@3s": float(np.mean(ade_3s)),
        "ADE@6.4s": float(np.mean(ade_full)),
        "FDE@6.4s": float(np.mean(fde_full)),
        "accel_mae": float(np.mean(np.abs(pred_accel - gt_accel))),
        "curvature_mae": float(np.mean(np.abs(pred_curv - gt_curv))),
    }


# Gate thresholds (initial baselines, tightened after first real training)
GATE_THRESHOLDS = {
    "ADE@3s": 2.0,
    "FDE@6.4s": 5.0,
}


def gate_check(
    metrics: dict[str, float],
    thresholds: dict[str, float] = GATE_THRESHOLDS,
) -> bool:
    """Returns True if all metrics pass gate thresholds."""
    for key, max_val in thresholds.items():
        if metrics.get(key, float("inf")) > max_val:
            return False
    return True
