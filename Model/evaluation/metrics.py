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


# ---------------------------------------------------------------------------
# Complementary metrics (#66 §2-3) — comfort and an off-road proxy.
# These extend the displacement metrics already provided by
# ``compute_open_loop_metrics`` above; they need no ground-truth trajectory
# (comfort) or no other-agent labels (off-road), which L2D lacks.
# ---------------------------------------------------------------------------

# nuPlan comfort thresholds (#66 §3).
COMFORT_THRESHOLDS = {
    "lon_jerk": 4.13,    # m/s^3
    "lat_accel": 4.89,   # m/s^2
    "lat_jerk": 8.37,    # m/s^3
    "yaw_rate": 0.95,    # rad/s
}


def compute_comfort_metrics(
    pred_accel: np.ndarray,
    pred_curv: np.ndarray,
    initial_speed: np.ndarray,
    dt: float = 0.1,
    thresholds: dict[str, float] = COMFORT_THRESHOLDS,
) -> dict[str, float]:
    """Comfort metrics straight from the ``(a, κ)`` outputs (#66 §3).

    These need no ground truth and are natural for an action-space model. With
    the per-step speed ``v[t] = v0 + Σ a·dt`` (clamped ≥ 0):
      * longitudinal jerk ``Δa/dt``
      * lateral acceleration ``v² κ``
      * lateral jerk ``Δa_lat/dt``
      * yaw rate ``v κ``
    Reports the batch-mean of each per-sample peak plus the **comfort violation
    rate** (fraction of samples exceeding ANY nuPlan threshold).

    Args:
        pred_accel, pred_curv: ``(B, T)`` predicted action signals.
        initial_speed: ``(B,)`` speed at the prediction start.
    """
    accel = np.asarray(pred_accel, dtype=np.float64)
    curv = np.asarray(pred_curv, dtype=np.float64)
    v0 = np.asarray(initial_speed, dtype=np.float64)[:, None]

    v = np.clip(v0 + np.cumsum(accel, axis=1) * dt, 0.0, None)   # (B, T)
    lon_jerk = np.abs(np.diff(accel, axis=1)) / dt               # (B, T-1)
    lat_accel = np.abs(v ** 2 * curv)                            # (B, T)
    lat_jerk = np.abs(np.diff(v ** 2 * curv, axis=1)) / dt       # (B, T-1)
    yaw_rate = np.abs(v * curv)                                  # (B, T)

    peaks = {
        "lon_jerk": lon_jerk.max(axis=1),
        "lat_accel": lat_accel.max(axis=1),
        "lat_jerk": lat_jerk.max(axis=1),
        "yaw_rate": yaw_rate.max(axis=1),
    }
    violated = np.zeros(accel.shape[0], dtype=bool)
    out: dict[str, float] = {}
    for name, peak in peaks.items():
        out[f"max_{name}"] = float(peak.mean())
        exceed = peak > thresholds[name]
        out[f"{name}_violation_rate"] = float(exceed.mean())
        violated |= exceed
    out["comfort_violation_rate"] = float(violated.mean())
    return out


def offroad_rate(
    positions: np.ndarray,
    drivable_mask: np.ndarray,
    meters_per_pixel: float,
    center_px: tuple[int, int] | None = None,
) -> float:
    """Off-road proxy for collision rate when agents are unlabelled (#66 §2).

    L2D has no other-agent annotations, so we use the BEV drivable mask: a
    trajectory is off-road if any predicted pose falls on a non-drivable cell.

    Args:
        positions: ``(B, T, 2)`` integrated ``(x_forward, y_left)`` in metres.
        drivable_mask: ``(H, W)`` boolean BEV; True = drivable.
        meters_per_pixel: BEV resolution.
        center_px: ego pixel ``(row, col)``; defaults to the grid centre.
            Convention (matches the repo's BEV rendering): forward +x → up
            (decreasing row), left +y → left (decreasing col).

    Returns:
        Fraction of trajectories that leave the drivable area.
    """
    H, W = drivable_mask.shape
    cr, cc = center_px if center_px is not None else (H // 2, W // 2)
    B = positions.shape[0]
    offroad = 0
    for i in range(B):
        rows = np.round(cr - positions[i, :, 0] / meters_per_pixel).astype(int)
        cols = np.round(cc - positions[i, :, 1] / meters_per_pixel).astype(int)
        inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        # Outside the grid counts as off-road; inside, check the mask.
        on_road = inside.copy()
        on_road[inside] = drivable_mask[rows[inside], cols[inside]]
        if not on_road.all():
            offroad += 1
    return offroad / max(B, 1)
