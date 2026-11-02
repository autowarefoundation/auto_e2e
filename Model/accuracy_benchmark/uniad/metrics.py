# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy scoring and explicit target-frame conversions."""

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import body_to_published, world_delta_to_published
from .protocol import DENSE_TIMES, dense_prediction


def canonical_reference(poses, index):
    if index < 2 or index + 30 >= len(poses):
        raise ValueError("Missing causal history or 3-second future")
    rotation = Rotation.from_quat(poses[index, 4:8]).as_matrix()
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    world_to_yaw = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    target = (poses[index + 1 : index + 31, 1:3] - poses[index, 1:3]) @ world_to_yaw.T
    velocity = (poses[index, 1:3] - poses[index - 1, 1:3]) / (
        poses[index, 0] - poses[index - 1, 0]
    )
    return target, world_to_yaw @ rotation[:2, :2], velocity @ world_to_yaw.T, velocity


def independent_metrics(predictions, targets, hz=10):
    p, t = (
        np.asarray(predictions, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )
    if (
        hz not in (2, 10)
        or p.shape != t.shape
        or p.ndim != 3
        or p.shape[1:] != (3 * hz, 2)
        or not len(p)
    ):
        raise ValueError("Expected nonempty matching 3-second trajectories")
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Nonfinite trajectories cannot be dropped")
    errors = np.sqrt(np.sum((p - t) ** 2, axis=2))
    return {
        str(h): dict(
            ade_m=float(errors[:, : h * hz].mean(axis=1).mean()),
            fde_m=float(errors[:, h * hz - 1].mean()),
            count=len(p),
        )
        for h in (1, 2, 3)
    }


def row_trajectories(row, frame, poses, track):
    target, transform, velocity, world_velocity = canonical_reference(
        poses, frame["frame"]
    )
    targets = {"canonical": target}
    predictions = {"canonical": {"constant_velocity": DENSE_TIMES[:, None] * velocity}}
    if track == "dense":
        targets["legacy_wgs84"] = np.asarray(frame["target_xy_10hz"])
        predictions["legacy_wgs84"] = {
            "constant_velocity": world_delta_to_published(
                DENSE_TIMES[:, None] * world_velocity, frame["evaluation_pose"]
            )
        }
    variants = row["predictions_body_xy_2hz"]
    if "fixed_forward" not in variants or set(variants) - {
        "fixed_forward",
        "oracle_turn",
    }:
        raise ValueError("Unexpected prediction variants")
    for variant, native in variants.items():
        dense = dense_prediction(native)
        predictions["canonical"][variant] = dense @ transform.T
        if track == "dense":
            predictions["legacy_wgs84"][variant] = body_to_published(dense, frame)
    return predictions, targets
