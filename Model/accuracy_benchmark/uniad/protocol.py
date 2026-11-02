# SPDX-License-Identifier: Apache-2.0
"""Numerical adapter used by the original-image, wrapped-heading evaluation."""

import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = (
    "camera_base_front_center",
    "camera_ring_front_right",
    "camera_ring_front_left",
    "camera_ring_rear",
    "camera_ring_rear_left",
    "camera_ring_rear_right",
)
UNIAD_TO_KIT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
DENSE_TIMES = np.arange(1, 31, dtype=np.float64) / 10
NATIVE_TIMES = np.arange(1, 7, dtype=np.float64) / 2


def normalize_and_pad(images_rgb, valid_hw=None):
    """Apply stock BGR normalization, then bottom/right divisor-32 padding."""
    images_rgb = np.asarray(images_rgb)
    if (
        images_rgb.ndim != 4
        or images_rgb.shape[0] != 6
        or images_rgb.shape[-1] != 3
        or images_rgb.dtype != np.uint8
    ):
        raise ValueError("Expected six uint8 RGB images")
    _, height, width, _ = images_rgb.shape
    shapes = np.asarray(valid_hw if valid_hw is not None else [[height, width]] * 6)
    if shapes.shape != (6, 2) or not np.issubdtype(shapes.dtype, np.integer):
        raise ValueError("Expected six integer valid image shapes")
    if np.any(shapes <= 0) or np.any(shapes > [height, width]):
        raise ValueError("Invalid image extent")
    padded = np.zeros(
        (6, (height + 31) // 32 * 32, (width + 31) // 32 * 32, 3), dtype=np.float32
    )
    mean = np.array([103.530, 116.280, 123.675], dtype=np.float32)
    for i, (h, w) in enumerate(shapes):
        padded[i, :h, :w] = images_rgb[i, :h, :w, ::-1].astype(np.float32) - mean
    return padded, [tuple(map(int, hw)) + (3,) for hw in shapes]


def validate_poses(poses, reference):
    if poses.ndim != 2 or poses.shape[1] != 8 or len(poses) < 2:
        raise ValueError("Expected timestamp, xyz, xyzw pose rows")
    if not np.isfinite(poses).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite pose or reference timestamp")
    if reference.shape != (len(poses),) or not np.allclose(
        poses[:, 0], reference, atol=1e-4, rtol=0
    ):
        raise ValueError("Pose and camera reference timestamps do not match")
    dt = np.diff(poses[:, 0])
    if not np.all((dt >= 0.08) & (dt <= 0.12)):
        raise ValueError("Expected contiguous approximately 10 Hz reference poses")
    if not np.allclose(np.linalg.norm(poses[:, 4:8], axis=1), 1, atol=0.01, rtol=0):
        raise ValueError("Invalid pose quaternion")


def past_can_bus(poses, index, origin):
    """CAN-bus vector at ``index``; position is relative to frame ``origin``."""
    rotation = Rotation.from_quat(poses[index, 4:8])
    can = np.zeros(18, dtype=np.float64)
    can[:3] = poses[index, 1:4] - poses[origin, 1:4]
    can[3:7] = rotation.as_quat()[[3, 0, 1, 2]]
    yaw = rotation.as_euler("ZYX")[0] % (2 * np.pi)
    can[-2:] = [yaw, np.degrees(yaw)]
    if index >= 2:
        dt = poses[index, 0] - poses[index - 1, 0]
        previous_dt = poses[index - 1, 0] - poses[index - 2, 0]
        velocity = (poses[index, 1:4] - poses[index - 1, 1:4]) / dt
        previous_velocity = (
            poses[index - 1, 1:4] - poses[index - 2, 1:4]
        ) / previous_dt
        can[7:10] = rotation.inv().apply(
            (velocity - previous_velocity) / ((dt + previous_dt) / 2)
        )
        previous_rotation = Rotation.from_quat(poses[index - 1, 4:8])
        can[10:13] = (previous_rotation.inv() * rotation).as_rotvec() / dt
        can[13:16] = rotation.inv().apply(velocity)
    return can


def to_kit_xy(virtual_xy):
    virtual_xy = np.asarray(virtual_xy, dtype=float)
    if virtual_xy.shape != (6, 2) or not np.isfinite(virtual_xy).all():
        raise ValueError("Prediction must contain six finite native XY waypoints")
    return virtual_xy @ UNIAD_TO_KIT[:2, :2].T


def dense_prediction(native_xy):
    native_xy = np.asarray(native_xy, dtype=float)
    if native_xy.shape != (6, 2) or not np.isfinite(native_xy).all():
        raise ValueError("Expected six finite native KIT XY waypoints")
    with_origin = np.vstack([np.zeros((1, 2)), native_xy])
    times = np.r_[0.0, NATIVE_TIMES]
    return np.stack(
        [np.interp(DENSE_TIMES, times, with_origin[:, i]) for i in range(2)], axis=-1
    )


def oracle_turn(target_dense):
    # Upstream UniAD command heuristic, applied to the 3-second future endpoint.
    virtual_x = -float(target_dense[-1, 1])
    return 0 if virtual_x >= 2 else 1 if virtual_x <= -2 else 2


def phase_inputs(scored):
    if not scored or len(scored) != len(set(scored)) or min(scored) < 35:
        raise ValueError("Invalid packed anchor indices")
    result = []
    for phase in range(5):
        anchors = sorted(i for i in scored if i % 5 == phase)
        if anchors:
            result.append(
                (phase, list(range(anchors[0] - 35, anchors[-1] + 1, 5)), set(anchors))
            )
    return result
