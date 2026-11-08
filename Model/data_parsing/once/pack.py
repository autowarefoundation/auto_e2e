"""Pack the ONCE train sequences with ONCE-3DLanes lanes as AutoBEV-only samples.

ONCE has seven cameras, a roof LiDAR, per-frame poses and 3D boxes on its
labelled 2 Hz frames, but no map: the ONCE paper withholds all map data, so
drivable_area, intersection, crosswalk and stop_line are never labelled.
ONCE annotates only cars, buses, trucks, pedestrians and cyclists, so cones
and barriers are unlabelled and other_obstacle is never labelled either.

Ego frame: x is the front camera's (cam01) optical axis projected on the
LiDAR xy plane, z is LiDAR up, and the origin is moved from the roof LiDAR
down to the road (``LIDAR_ABOVE_ROAD_M``) here, at pack time, so every
member (projections, depth, boxes, lanes) is in the road-level frame.

Cameras: ``CAMERAS`` takes the six of the seven ONCE cameras nearest to the
AutoBEV view yaws (front 0, front-left 55, front-right -55, rear-left 125,
rear 180, rear-right -125 degrees). In the ONCE calibration cam01 looks at
0, cam06 at 58, cam07 at -58, cam05 at 133, cam09 at -179 and cam08 at -128
degrees; the telephoto cam03 (also at 0 degrees, twice cam01's focal length)
is dropped, so the front view keeps a wide field that overlaps its
neighbours.

lane_boundary comes from ONCE-3DLanes (Yan et al., CVPR 2022): painted lane
markings as 3D polylines in the cam01 frame. It is valid only on frames with
a label that has at least one lane, where cam01 sees the ground up to the
farthest labelled point (at most 50 m), not under vehicles, and within the
corridor between the outermost painted lines (unpainted edges outside them
are not labelled). A sample is taken at most every second, at labelled
frames with seven frames 0.5 s apart before them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

import numpy as np

from data_parsing.nuplan.targets import _rasterize_polygons, _rasterize_polylines
from data_processing.autobev_members import BOX_CLASSES
from data_processing.autobev_pack import (
    EGO,
    LANE_WIDTH_M,
    bev_segmentation,
    camera_frame,
    decode_rgb,
    footprint,
    object_labels,
    object_valid,
    sample_members,
    undistort,
)
from model_components.autobev.geometry import AUTOBEV_GEOMETRY, HISTORY_FRAMES, IMAGE_H, IMAGE_W

DATASET: Final = "once"
# Bump when a rule that decides the packed bytes changes; part of dataset_id.
PACK_CONTRACT: Final = "once-autobev-v2"
CAMERAS: Final = ("cam01", "cam06", "cam07", "cam05", "cam09", "cam08")
LIDAR: Final = "lidar_roof"
LANES_ROOT: Final = "ONCE_3DLanes/train"
LANE_CAMERA: Final = "cam01"
# ONCE-3DLanes labels end 50 m ahead.
LANE_RANGE_M: Final = 50.0
# The roof LiDAR sits this far above the road (median ONCE-3DLanes point height within 20 m).
LIDAR_ABOVE_ROAD_M: Final = 1.78
NEVER_LABELLED: Final = frozenset({"drivable_area", "intersection", "crosswalk", "stop_line", "other_obstacle"})
# One partition per train sequence (six in ONCE train).
PARTITIONS: Final = 6
FRAME_MS: Final = 500
SAMPLE_STRIDE_MS: Final = 1000
# Neighbouring frames further apart than 0.5 s +- this break the T8 spacing.
FRAME_TOLERANCE_MS: Final = 100
OBJECTS: Final = {
    "Car": ("vehicle", "VEHICLE"), "Bus": ("vehicle", "VEHICLE"), "Truck": ("vehicle", "VEHICLE"),
    "Pedestrian": ("vulnerable_road_user", "PEDESTRIAN"), "Cyclist": ("vulnerable_road_user", "BICYCLE"),
}


def pose_matrix(pose: Sequence[float]) -> np.ndarray:
    """world_from_lidar from ONCE [qx, qy, qz, qw, tx, ty, tz]."""
    qx, qy, qz, qw, tx, ty, tz = pose
    out = np.eye(4)
    out[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    out[:3, 3] = [tx, ty, tz]
    return out


def ego_from_lidar(calib: Mapping) -> np.ndarray:
    """Turn the LiDAR axes about z so +x follows cam01, and put the origin on the road."""
    axis = np.asarray(calib[LANE_CAMERA]["cam_to_velo"])[:3, 2]
    yaw = math.atan2(axis[1], axis[0])
    out = np.eye(4)
    out[:2, :2] = [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
    out[2, 3] = LIDAR_ABOVE_ROAD_M
    return out


def lane_labels(lanes: Sequence, ego_from_camera: np.ndarray, front_projection: np.ndarray,
                vehicles: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """ONCE-3DLanes lane_boundary raster and its valid mask, or None without a usable label."""
    lines = [
        (ego_from_camera @ np.c_[points, np.ones(len(points))].T)[:3].T
        for points in (np.asarray(lane, np.float64).reshape(-1, 3) for lane in lanes)
        if len(points) >= 2
    ]
    if not lines:
        return None
    mask = _rasterize_polylines([line[:, :2] for line in lines], EGO, AUTOBEV_GEOMETRY, width_m=LANE_WIDTH_M)
    ground_z = float(np.median(np.concatenate(lines)[:, 2]))
    x, y = AUTOBEV_GEOMETRY.pixel_center_grids()
    uvw = front_projection @ np.stack([x.ravel(), y.ravel(), np.full(x.size, ground_z), np.ones(x.size)])
    # The road itself must be in view; row 2 of the cropped projection is camera depth.
    depth = uvw[2].reshape(x.shape)
    u = (uvw[0] / np.maximum(uvw[2], 1e-6)).reshape(x.shape)
    v = (uvw[1] / np.maximum(uvw[2], 1e-6)).reshape(x.shape)
    visible = (u >= 0) & (u < IMAGE_W) & (v >= 0) & (v < IMAGE_H)
    # ONCE-3DLanes has no lane points under vehicles; that ground is not negative.
    visible &= _rasterize_polygons(list(vehicles), EGO, AUTOBEV_GEOMETRY) < 0.5
    # Labels end where the LiDAR ground points run out, often short of 50 m.
    reach = min(LANE_RANGE_M, max(float(np.asarray(lane, np.float64).reshape(-1, 3)[:, 2].max())
                                  for lane in lanes if len(lane)))
    # Only paint is labelled; outside the corridor between the outermost lines
    # unpainted lane and road edges are likely, so only the corridor (one cell
    # wider) is negative.
    corridor = np.zeros_like(visible)
    for row in np.nonzero((mask > 0.5).any(axis=1))[0]:
        cols = np.nonzero(mask[row] > 0.5)[0]
        corridor[row, max(cols.min() - 1, 0): cols.max() + 2] = True
    return mask, visible & corridor & (depth > 0.0) & (depth <= reach)


class OnceSequence:
    """One ONCE sequence: calibration, poses and frames from its info JSON."""

    def __init__(self, files: Mapping[str, bytes], sequence: str):
        info = json.loads(files[f"data/{sequence}/{sequence}.json"])
        self.files, self.sequence, self.calib = files, sequence, info["calib"]
        self.frames = info["frames"]
        self.ego_from_lidar = ego_from_lidar(self.calib)
        self.poses = {frame["frame_id"]: pose_matrix(frame["pose"]) for frame in self.frames}

    def samples(self) -> list[tuple[list[str], dict]]:
        """``(history frame ids oldest first, labelled frame)`` per sample."""
        out, next_time = [], -1
        for index, frame in enumerate(self.frames):
            if "annos" not in frame or index < HISTORY_FRAMES or int(frame["frame_id"]) < next_time:
                continue
            history = [self.frames[index - k]["frame_id"] for k in range(HISTORY_FRAMES, 0, -1)]
            steps = np.diff([int(f) for f in (*history, frame["frame_id"])])
            if np.any(np.abs(steps - FRAME_MS) > FRAME_TOLERANCE_MS):
                continue
            out.append((history, frame))
            next_time = int(frame["frame_id"]) + SAMPLE_STRIDE_MS
        return out

    def camera_frame(self, frame_id: str, reference: str):
        lidar_from_ego = np.linalg.inv(self.ego_from_lidar)
        images, intrinsics, camera_from_reference = [], [], []
        for camera in CAMERAS:
            item = self.calib[camera]
            rgb = decode_rgb(self.files[f"data/{self.sequence}/{camera}/{frame_id}.jpg"])
            rectified, rectified_k = undistort(rgb, item["cam_intrinsic"], item["distortion"])
            world_from_camera = self.poses[frame_id] @ np.asarray(item["cam_to_velo"], np.float64)
            images.append(rectified)
            intrinsics.append(rectified_k)
            camera_from_reference.append(np.linalg.inv(world_from_camera) @ self.poses[reference] @ lidar_from_ego)
        return camera_frame(images, intrinsics, camera_from_reference)


def build_sample(seq: OnceSequence, history: list[str], frame: Mapping) -> tuple[str, dict[str, bytes]]:
    reference = frame["frame_id"]
    past = [seq.camera_frame(frame_id, reference) for frame_id in history]
    current = seq.camera_frame(reference, reference)
    raw = np.frombuffer(seq.files[f"data/{seq.sequence}/{LIDAR}/{reference}.bin"], np.float32).reshape(-1, 4)
    points_ego = (seq.ego_from_lidar @ np.c_[raw[:, :3], np.ones(len(raw))].T)[:3].T

    yaw_fix = math.atan2(seq.ego_from_lidar[1, 0], seq.ego_from_lidar[0, 0])
    footprints: dict[str, list[np.ndarray]] = {"vehicle": [], "vulnerable_road_user": []}
    boxes, labels = [], []
    for name, box in zip(frame["annos"]["names"], frame["annos"]["boxes_3d"], strict=True):
        bev_class, box_class = OBJECTS.get(name, (None, "OTHER"))
        cx, cy, cz, length, width, height, heading = box
        center = seq.ego_from_lidar @ np.array([cx, cy, cz, 1.0])
        heading = math.remainder(heading + yaw_fix, math.tau)
        if bev_class is not None:
            footprints[bev_class].append(footprint(center[0], center[1], heading, length, width))
        boxes.append([center[0], center[1], center[2], length, width, height, heading, 0.0, 0.0])
        labels.append(BOX_CLASSES.index(box_class))

    labelled = object_labels(footprints, [], object_valid(current, points_ego))
    lane_path = f"{LANES_ROOT}/{seq.sequence}/{LANE_CAMERA}/{reference}.json"
    ego_from_camera = seq.ego_from_lidar @ np.asarray(seq.calib[LANE_CAMERA]["cam_to_velo"], np.float64)
    lanes = lane_labels(
        json.loads(seq.files[lane_path])["lanes"], ego_from_camera,
        current.projections[CAMERAS.index(LANE_CAMERA)], footprints["vehicle"],
    ) if lane_path in seq.files else None
    if lanes is not None:
        labelled["lane_boundary"] = lanes
    members = sample_members(
        current, past, points_ego=points_ego,
        segmentation=bev_segmentation(labelled, never_labelled=NEVER_LABELLED),
        boxes=boxes, box_labels=labels,
        split_group_uid=f"{DATASET}-{seq.sequence}", meta={"dataset": DATASET, "sequence": seq.sequence, "frame_id": reference,
              "cameras": list(CAMERAS), "lanes": lanes is not None},
    )
    return f"once-{seq.sequence}-{reference}", members


def sequences(files: Mapping[str, bytes]) -> list[str]:
    """Sequence ids with an info JSON (``data/<seq>/<seq>.json``), sorted."""
    return sorted(
        parts[1] for parts in (name.split("/") for name in files)
        if len(parts) == 3 and parts[0] == "data" and parts[2] == f"{parts[1]}.json"
    )


def iter_samples(files: Mapping[str, bytes], part: int, parts: int) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Samples of every ``parts``-th sequence in id order, starting at ``part``."""
    for sequence in sequences(files)[part::parts]:
        seq = OnceSequence(files, sequence)
        for history, frame in seq.samples():
            yield build_sample(seq, history, frame)
