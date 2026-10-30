# SPDX-License-Identifier: Apache-2.0
"""Calibration and canonical/legacy coordinate transforms."""

from functools import lru_cache
from types import SimpleNamespace
import struct
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from .protocol import UNIAD_TO_KIT


def raw_projection(calib, source_wh, target_wh):
    intrinsic = np.array(calib.intrinsic, dtype=float, copy=True)
    intrinsic[0] *= target_wh[0] / source_wh[0]
    intrinsic[1] *= target_wh[1] / source_wh[1]
    transform = np.eye(4)
    transform[:3, :3] = UNIAD_TO_KIT
    result = np.eye(4)
    result[:3] = intrinsic @ (np.linalg.inv(calib.extrinsic) @ transform)[:3]
    return result


def camera_calibration(payload, name):
    # The installed SDK expects root/calib.json; this release uses calibration/calib.json.
    entry = payload.get(name, payload.get(name + "_pinhole"))
    if entry is None or entry["camera_model"] != "pinhole":
        raise ValueError(f"Missing pinhole calibration: {name}")
    intrinsic = entry["intrinsics"]
    f, u, v = (
        float(intrinsic[k])
        for k in ("focal_length", "principal_point_u", "principal_point_v")
    )
    k = np.array([[f, 0.0, u], [0.0, f, v], [0.0, 0.0, 1.0]])
    t = np.asarray(entry["T_to_reference"], dtype=float)
    if (
        t.shape != (4, 4)
        or not np.isfinite(t).all()
        or not np.isfinite(k).all()
        or f <= 0
    ):
        raise ValueError("Invalid calibration matrix")
    if not np.allclose(t[3], [0, 0, 0, 1]) or not np.allclose(
        t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5
    ):
        raise ValueError("Invalid camera rigid transform")
    size = tuple(int(entry["resolution"][key]) for key in ("width", "height"))
    if min(size) <= 0:
        raise ValueError("Invalid native resolution")
    return SimpleNamespace(intrinsic=k, extrinsic=t, image_size=size)


def match_cameras(packed, candidates):
    packed, candidates = np.asarray(packed), np.asarray(candidates)
    if packed.shape != (6, 4, 4) or candidates.shape != packed.shape:
        raise ValueError("Expected six camera matrices")
    costs = np.linalg.norm(packed[:, None] - candidates[None], axis=(2, 3))
    rows, cols = linear_sum_assignment(costs)
    if not np.array_equal(rows, np.arange(6)) or len(set(cols)) != 6:
        raise ValueError("Camera assignment is not bijective")
    matched = candidates[cols]
    if not np.allclose(packed, matched, atol=0.03, rtol=1e-5):
        raise ValueError(
            f"Packed/raw camera geometry differs: max_abs={np.max(np.abs(packed - matched))}"
        )
    return cols, float(np.max(np.abs(packed - matched)))


def canonical_target(poses, index):
    if index + 30 >= len(poses):
        raise ValueError("Missing canonical future")
    rotation = Rotation.from_quat(poses[index, 4:8]).as_matrix()
    yaw = Rotation.from_quat(poses[index, 4:8]).as_euler("ZYX")[0]
    world_to_yaw = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    delta = poses[index + 1 : index + 31, 1:3] - poses[index, 1:3]
    return delta @ world_to_yaw.T, world_to_yaw @ rotation[:2, :2]


def decode_pose(blob):
    lat, lon, heading, timestamp_ns, accuracy = struct.unpack("<dddqf", blob)
    if not np.isfinite([lat, lon, heading]).all() or not (
        -90 <= lat <= 90 and -180 <= lon <= 180
    ):
        raise ValueError("Invalid packed current pose")
    return dict(
        latitude=lat, longitude=lon, heading_deg=heading, timestamp_ns=timestamp_ns
    )


def decode_gps(blob):
    return np.frombuffer(blob, dtype="<f8").reshape(65, 2).copy()


def wgs84_xy(gps, pose):
    # AutoE2E 40a75cf8, reactive_training_artifacts.py::wgs84_future_to_ego_xy.
    gps = np.asarray(gps, dtype=np.float64)
    scale = 6378137.0 * np.pi / 180
    east = (
        (gps[:, 1] - pose["longitude"]) * np.cos(np.radians(pose["latitude"])) * scale
    )
    north = (gps[:, 0] - pose["latitude"]) * scale
    heading = np.radians(pose["heading_deg"])
    return np.column_stack(
        (
            east * np.sin(heading) + north * np.cos(heading),
            -east * np.cos(heading) + north * np.sin(heading),
        )
    ).astype(np.float32)


@lru_cache(maxsize=1)
def transformers():
    from pyproj import Transformer

    # KITScenes publisher dataset.py::_utm32_to_wgs84 uses this CRS.
    return (
        Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True),
        Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True),
    )


def world_delta_to_published(world_xy, pose):
    delta = np.asarray(world_xy, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 2 or not np.isfinite(delta).all():
        raise ValueError("Expected finite world XY displacements")
    to_utm, to_wgs = transformers()
    origin = np.array(to_utm.transform(pose["longitude"], pose["latitude"]))
    xy = origin + delta
    lon, lat = to_wgs.transform(xy[:, 0], xy[:, 1])
    return wgs84_xy(np.column_stack((lat, lon)), pose)


def body_to_published(body_xy, frame):
    rotation = np.asarray(frame["body_to_utm_xy"], dtype=np.float64)
    if rotation.shape != (2, 2) or not np.isfinite(rotation).all():
        raise ValueError("Invalid body-to-UTM rotation")
    return world_delta_to_published(
        np.asarray(body_xy) @ rotation.T, frame["evaluation_pose"]
    )
