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

# nuPlan comfort bounds (the full set from nuplan-devkit `ego_is_comfortable`).
COMFORT_THRESHOLDS = {
    "lon_accel_max": 2.40,    # m/s^2   upper bound on longitudinal accel
    "lon_accel_min": -4.05,   # m/s^2   lower bound (braking)
    "lat_accel": 4.89,        # m/s^2   |lateral accel|
    "yaw_rate": 0.95,         # rad/s   |yaw rate|
    "yaw_accel": 1.93,        # rad/s^2 |yaw acceleration|
    "lon_jerk": 4.13,         # m/s^3   |longitudinal jerk|
    "mag_jerk": 8.37,         # m/s^3   |jerk magnitude| = sqrt(lon_jerk^2 + lat_jerk^2)
}


def compute_comfort_metrics(
    pred_accel: np.ndarray,
    pred_curv: np.ndarray,
    initial_speed: np.ndarray,
    dt: float = 0.1,
    thresholds: dict[str, float] = COMFORT_THRESHOLDS,
) -> dict[str, float]:
    """Comfort metrics from the ``(a, κ)`` outputs vs the nuPlan bounds (#66 §3).

    Mirrors nuplan-devkit's ``ego_is_comfortable`` set — no ground truth needed.
    With the per-step speed ``v[t] = v0 + Σ a·dt`` (clamped ≥ 0):
      * longitudinal acceleration ``a``         — two-sided bound ``[min, max]``
      * lateral acceleration      ``v² κ``       — ``|·|`` bound
      * yaw rate                  ``v κ``        — ``|·|`` bound
      * yaw acceleration          ``Δ(v κ)/dt``  — ``|·|`` bound
      * longitudinal jerk         ``Δa/dt``      — ``|·|`` bound
      * jerk magnitude            ``√(lon_jerk² + lat_jerk²)`` — bound (this is the
        8.37 m/s³ threshold; *not* lateral jerk)

    Reports the batch-mean of each per-sample peak, a per-metric violation rate,
    and the overall ``comfort_violation_rate`` (fraction of samples exceeding ANY
    bound).

    Args:
        pred_accel, pred_curv: ``(B, T)`` predicted action signals.
        initial_speed: ``(B,)`` speed at the prediction start.
    """
    accel = np.asarray(pred_accel, dtype=np.float64)                 # (B, T)
    curv = np.asarray(pred_curv, dtype=np.float64)                   # (B, T)
    v0 = np.asarray(initial_speed, dtype=np.float64)[:, None]

    v = np.clip(v0 + np.cumsum(accel, axis=1) * dt, 0.0, None)       # (B, T)
    lat_accel = v ** 2 * curv                                        # (B, T)
    yaw_rate = v * curv                                              # (B, T)
    lon_jerk = np.diff(accel, axis=1) / dt                           # (B, T-1)
    lat_jerk = np.diff(lat_accel, axis=1) / dt                       # (B, T-1)
    yaw_accel = np.diff(yaw_rate, axis=1) / dt                       # (B, T-1)
    mag_jerk = np.hypot(lon_jerk, lat_jerk)                          # (B, T-1)

    out: dict[str, float] = {}
    violated = np.zeros(accel.shape[0], dtype=bool)

    # Longitudinal acceleration: asymmetric two-sided bound.
    lon_max, lon_min = accel.max(axis=1), accel.min(axis=1)
    out["max_lon_accel"] = float(lon_max.mean())
    out["min_lon_accel"] = float(lon_min.mean())
    lon_exceed = (lon_max > thresholds["lon_accel_max"]) | (lon_min < thresholds["lon_accel_min"])
    out["lon_accel_violation_rate"] = float(lon_exceed.mean())
    violated |= lon_exceed

    # Magnitude-bounded quantities.
    abs_peaks = {
        "lat_accel": np.abs(lat_accel).max(axis=1),
        "yaw_rate": np.abs(yaw_rate).max(axis=1),
        "yaw_accel": np.abs(yaw_accel).max(axis=1),
        "lon_jerk": np.abs(lon_jerk).max(axis=1),
        "mag_jerk": mag_jerk.max(axis=1),
    }
    for name, peak in abs_peaks.items():
        out[f"max_{name}"] = float(peak.mean())
        exceed = peak > thresholds[name]
        out[f"{name}_violation_rate"] = float(exceed.mean())
        violated |= exceed

    out["comfort_violation_rate"] = float(violated.mean())
    return out


def _erode_drivable(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Shrink the drivable area by ``iterations`` pixels (4-neighbour erosion).

    A cell stays drivable only if it and its 4 neighbours are drivable (cells
    outside the grid count as non-drivable), so after ``k`` iterations any cell
    within Manhattan distance ``k`` of the boundary is removed. Pure-numpy, no
    scipy. Used to require a safety margin from the road edge.
    """
    eroded = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        nb = eroded.copy()
        nb[1:, :] &= eroded[:-1, :]      # up neighbour
        nb[:-1, :] &= eroded[1:, :]      # down neighbour
        nb[:, 1:] &= eroded[:, :-1]      # left neighbour
        nb[:, :-1] &= eroded[:, 1:]      # right neighbour
        nb[0, :] = nb[-1, :] = nb[:, 0] = nb[:, -1] = False   # border = off-road
        eroded = nb
    return eroded


def offroad_rate(
    positions: np.ndarray,
    drivable_mask: np.ndarray,
    meters_per_pixel: float,
    center_px: tuple[int, int] | None = None,
    headings: np.ndarray | None = None,
    ego_size: tuple[float, float] | None = None,
    dilation_px: int = 0,
) -> float:
    """Off-road proxy for collision rate when agents are unlabelled (#66 §2).

    L2D has no other-agent annotations, so we use the BEV drivable mask: a
    trajectory is off-road if it leaves the drivable area. By default this checks
    the trajectory **centre** point (lightweight). For drivable-area *compliance*
    the ego footprint matters — a corner can leave the road while the centre stays
    inside — so pass ``ego_size`` to check the four footprint corners, and/or
    ``dilation_px`` to require a safety margin from the boundary.

    Args:
        positions: ``(B, T, 2)`` integrated ``(x_forward, y_left)`` in metres.
        drivable_mask: ``(H, W)`` boolean BEV; True = drivable.
        meters_per_pixel: BEV resolution.
        center_px: ego pixel ``(row, col)``; defaults to the grid centre.
            Convention (matches the repo's BEV rendering): forward +x → up
            (decreasing row), left +y → left (decreasing col).
        headings: optional ``(B, T)`` heading per pose (rad) to orient the
            footprint. If ``None`` and ``ego_size`` is given, heading is taken
            from the finite-difference travel direction.
        ego_size: optional ``(length, width)`` in metres. When given, the four
            footprint corners are checked instead of only the centre.
        dilation_px: erode the drivable mask by this many pixels first (require a
            margin from the boundary). 0 = off.

    Returns:
        Fraction of trajectories that leave the drivable area.
    """
    mask = _erode_drivable(drivable_mask, dilation_px) if dilation_px > 0 \
        else np.asarray(drivable_mask, dtype=bool)
    H, W = mask.shape
    cr, cc = center_px if center_px is not None else (H // 2, W // 2)
    pos = np.asarray(positions, dtype=np.float64)
    B, T, _ = pos.shape

    if ego_size is None:
        query = pos[:, :, None, :]                                   # (B, T, 1, 2)
    else:
        length, width = ego_size
        corners = np.array([                                         # ego frame
            [length / 2, width / 2], [length / 2, -width / 2],
            [-length / 2, width / 2], [-length / 2, -width / 2],
        ])                                                           # (4, 2)
        if headings is not None:
            theta = np.asarray(headings, dtype=np.float64)
        elif T >= 2:
            d = np.diff(pos, axis=1)
            d = np.concatenate([d[:, :1, :], d], axis=1)             # (B, T, 2)
            theta = np.arctan2(d[..., 1], d[..., 0])                 # (B, T)
        else:
            theta = np.zeros((B, T))
        cos, sin = np.cos(theta), np.sin(theta)                      # (B, T)
        cx, cy = corners[:, 0], corners[:, 1]                        # (4,)
        qx = pos[..., 0:1] + cos[..., None] * cx - sin[..., None] * cy   # (B, T, 4)
        qy = pos[..., 1:2] + sin[..., None] * cx + cos[..., None] * cy   # (B, T, 4)
        query = np.stack([qx, qy], axis=-1)                          # (B, T, 4, 2)

    offroad = 0
    for i in range(B):
        rows = np.round(cr - query[i, ..., 0] / meters_per_pixel).astype(int)
        cols = np.round(cc - query[i, ..., 1] / meters_per_pixel).astype(int)
        inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        on_road = inside.copy()                                      # OOB = off-road
        on_road[inside] = mask[rows[inside], cols[inside]]
        if not on_road.all():
            offroad += 1
    return offroad / max(B, 1)
