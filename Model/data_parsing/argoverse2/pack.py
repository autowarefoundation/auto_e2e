"""Pack Argoverse 2 Sensor logs as AutoBEV-only samples.

A sample is taken every 0.5 s at annotation timestamps from 3.5 s into the
log, with seven history frames 0.5 s apart. Pittsburgh logs are never
acquired: Pittsburgh is the held-out city of the nuPlan evaluation. The
snapshot holds only the files the samples need (``needed_files``), and the
packer finds the same samples again from the files present, since every
chosen camera frame is the nearest one in the full listing.

Cameras: AV2 has seven ring cameras and AutoBEV six views. ``CAMERAS`` maps
front-center, front-left, front-right, side-left, one rear camera and
side-right onto the views; the rear camera alternates between rear-left and
rear-right by a hash of the log id, so neither side is always blind. Cells
the six cameras cannot see become invalid through the camera visibility
mask. The portrait front-center image is cropped like any other.

Labels in the nuPlan taxonomy (AV2's ego origin is the rear axle on the road):

- drivable_area from drivable area polygons
- lane_boundary from both boundaries of every lane segment outside
  intersections, painted or not, except bike lanes, as nuPlan leaves out
  lane connectors and has no bike lanes
- crosswalk from pedestrian crossings (two edges)
- intersection and stop_line are never labelled: AV2 has no stop lines and
  marks intersections only per lane segment, unlike nuPlan's polygons
- vehicle / vulnerable_road_user / other_obstacle from annotation
  footprints; footprints of boxes with no interior LiDAR point are unknown
Box centres keep the annotated z.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

import numpy as np

from data_parsing.nuplan.packing import _quaternion_transform
from data_parsing.nuplan.targets import _rasterize_polygons, _rasterize_polylines
from data_processing.autobev_members import BOX_CLASSES
from data_processing.autobev_pack import (
    LANE_WIDTH_M,
    bev_segmentation,
    camera_frame,
    camera_valid,
    decode_rgb,
    footprint,
    object_labels,
    object_valid,
    quaternion_yaw,
    sample_members,
    undistort,
)
from model_components.autobev.geometry import AUTOBEV_GEOMETRY, FRAME_INTERVAL_US, HISTORY_FRAMES

DATASET: Final = "argoverse2"
# Bump when a rule that decides the packed bytes changes; part of dataset_id.
PACK_CONTRACT: Final = "argoverse2-autobev-v2"
UPSTREAM_BUCKET: Final = "argoverse"
UPSTREAM_PREFIX: Final = "datasets/av2/sensor"
# Training uses train and val; test has no annotations.
SPLITS: Final = ("train", "val")
# The geographic hold-out of the nuPlan evaluation.
HELD_OUT_CITY: Final = "PIT"
# View 4 takes one of the two rear cameras, picked per log by a hash of its id.
REAR_CAMERAS: Final = ("ring_rear_right", "ring_rear_left")
CAMERAS: Final = (
    "ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left", REAR_CAMERAS, "ring_side_right",
)
NEVER_LABELLED: Final = frozenset({"intersection", "stop_line"})
FRAME_NS: Final = FRAME_INTERVAL_US * 1000
# A sample at every T8 frame time; consecutive samples share history frames.
SAMPLE_STRIDE_NS: Final = FRAME_NS
# A camera frame further than this from its T8 time leaves the sample out.
CAMERA_TOLERANCE_NS: Final = 60_000_000
METADATA_FILES: Final = (
    "annotations.feather", "city_SE3_egovehicle.feather",
    "calibration/intrinsics.feather", "calibration/egovehicle_SE3_sensor.feather",
)
VEHICLES: Final = frozenset({
    "REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK", "TRUCK_CAB", "VEHICULAR_TRAILER",
    "SCHOOL_BUS", "ARTICULATED_BUS", "MOTORCYCLE", "RAILED_VEHICLE", "MESSAGE_BOARD_TRAILER",
})
VULNERABLE: Final = frozenset({
    "PEDESTRIAN", "BICYCLIST", "MOTORCYCLIST", "WHEELCHAIR", "STROLLER", "WHEELED_RIDER", "OFFICIAL_SIGNALER",
})
RIDERS: Final = frozenset({"BICYCLIST", "MOTORCYCLIST"})


def log_city(map_name: str) -> str:
    """City code of a ``log_map_archive_<log>____<CITY>_city_<id>.json`` file name."""
    return map_name.rsplit("/", 1)[-1].split("____")[-1].split("_")[0]


def is_map_archive(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base.startswith("log_map_archive_") and base.endswith(".json")


def training_logs(map_names: Mapping[str, str]) -> list[str]:
    """Logs (keyed to their map archive name) outside the held-out city, sorted."""
    return sorted(log for log, name in map_names.items() if log_city(name) != HELD_OUT_CITY)


def cameras(log: str) -> tuple[str, ...]:
    """The six cameras of a log in view order."""
    rear = REAR_CAMERAS[int(hashlib.sha256(log.encode()).hexdigest(), 16) % 2]
    return (*CAMERAS[:4], rear, CAMERAS[5])


def plan_samples(log: str, annotated: Sequence[int], camera_times: Mapping[str, Sequence[int]]) -> list[dict]:
    """``{"timestamp", "frames"}`` per sample; ``frames`` maps each camera to its frame, oldest first."""
    times = {camera: np.asarray(sorted(camera_times.get(camera, ())), np.int64) for camera in cameras(log)}
    if any(not len(values) for values in times.values()):
        return []
    annotated = sorted(annotated)
    samples, next_time = [], annotated[0] if annotated else 0
    for timestamp in annotated:
        if timestamp - annotated[0] < HISTORY_FRAMES * FRAME_NS or timestamp < next_time:
            continue
        frames = []
        for offset in range(-HISTORY_FRAMES, 1):
            target = timestamp + offset * FRAME_NS
            nearest = {camera: int(values[np.abs(values - target).argmin()]) for camera, values in times.items()}
            if any(abs(stamp - target) > CAMERA_TOLERANCE_NS for stamp in nearest.values()):
                break
            frames.append(nearest)
        if len(frames) == HISTORY_FRAMES + 1:
            samples.append({"timestamp": int(timestamp), "frames": frames})
            next_time = timestamp + SAMPLE_STRIDE_NS
    return samples


def needed_files(samples: Sequence[Mapping]) -> list[str]:
    """Log-relative files the samples read besides the metadata and the map archive."""
    names = {f"sensors/lidar/{sample['timestamp']}.feather" for sample in samples}
    names.update(
        f"sensors/cameras/{camera}/{stamp}.jpg"
        for sample in samples for frame in sample["frames"] for camera, stamp in frame.items()
    )
    return sorted(names)


def _object_class(category: str) -> tuple[str, str]:
    if category in VEHICLES:
        return "vehicle", "VEHICLE"
    if category in VULNERABLE:
        return "vulnerable_road_user", "BICYCLE" if category in RIDERS else "PEDESTRIAN"
    return "other_obstacle", "OTHER"


def _xy(points) -> np.ndarray:
    return np.asarray([[p["x"], p["y"]] for p in points], np.float64)


class Log:
    """One log's calibration, poses, annotations and map, read from the snapshot."""

    def __init__(self, files: Mapping[str, bytes], root: str):
        import pandas as pd

        def table(name):
            return pd.read_feather(io.BytesIO(files[f"{root}/{name}"]))

        self.files, self.root, self.log = files, root, root.rsplit("/", 1)[-1]
        self.intrinsics = table("calibration/intrinsics.feather").set_index("sensor_name")
        self.extrinsics = table("calibration/egovehicle_SE3_sensor.feather").set_index("sensor_name")
        self.poses = table("city_SE3_egovehicle.feather").set_index("timestamp_ns")
        self.annotations = table("annotations.feather")
        (map_name,) = [name for name in files if name.startswith(f"{root}/map/") and is_map_archive(name)]
        self.map = json.loads(files[map_name])

    def camera_times(self) -> dict[str, list[int]]:
        prefix = f"{self.root}/sensors/cameras/"
        times: dict[str, list[int]] = {}
        for name in self.files:
            if name.startswith(prefix) and name.endswith(".jpg"):
                camera, stamp = name[len(prefix):-4].split("/")
                times.setdefault(camera, []).append(int(stamp))
        return times


def _pose(table, key) -> np.ndarray:
    row = table.loc[key]
    return _quaternion_transform([row.tx_m, row.ty_m, row.tz_m], [row.qw, row.qx, row.qy, row.qz])


def _frame(log: Log, frame: Mapping[str, int], city_from_reference: np.ndarray):
    images, intrinsics, camera_from_reference = [], [], []
    for camera, stamp in frame.items():
        intr = log.intrinsics.loc[camera]
        rgb = decode_rgb(log.files[f"{log.root}/sensors/cameras/{camera}/{stamp}.jpg"])
        if rgb.shape[:2] != (int(intr.height_px), int(intr.width_px)):
            raise ValueError(f"{log.log} {camera}: image size differs from its calibration")
        k = np.array([[intr.fx_px, 0, intr.cx_px], [0, intr.fy_px, intr.cy_px], [0, 0, 1]], np.float64)
        rectified, rectified_k = undistort(rgb, k, [intr.k1, intr.k2, 0.0, 0.0, intr.k3])
        city_from_camera = _pose(log.poses, stamp) @ _pose(log.extrinsics, camera)
        images.append(rectified)
        intrinsics.append(rectified_k)
        camera_from_reference.append(np.linalg.inv(city_from_camera) @ city_from_reference)
    return camera_frame(images, intrinsics, camera_from_reference)


def _map_targets(city_map: Mapping, pose2d, valid: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    drivable = [_xy(area["area_boundary"]) for area in city_map["drivable_areas"].values()]
    crossings = [
        np.vstack([_xy(crossing["edge1"]), _xy(crossing["edge2"])[::-1]])
        for crossing in city_map["pedestrian_crossings"].values()
    ]
    lines = [
        _xy(lane[f"{side}_lane_boundary"])
        for lane in city_map["lane_segments"].values()
        if not lane["is_intersection"] and lane["lane_type"] != "BIKE"
        for side in ("left", "right")
    ]
    return {
        "drivable_area": (_rasterize_polygons(drivable, pose2d, AUTOBEV_GEOMETRY), valid),
        "crosswalk": (_rasterize_polygons(crossings, pose2d, AUTOBEV_GEOMETRY), valid),
        "lane_boundary": (_rasterize_polylines(lines, pose2d, AUTOBEV_GEOMETRY, width_m=LANE_WIDTH_M), valid),
    }


def build_sample(log: Log, sample: Mapping) -> tuple[str, dict[str, bytes]]:
    import pandas as pd

    timestamp = sample["timestamp"]
    city_from_reference = _pose(log.poses, timestamp)
    yaw = math.atan2(city_from_reference[1, 0], city_from_reference[0, 0])
    pose2d = (float(city_from_reference[0, 3]), float(city_from_reference[1, 3]), yaw)
    past = [_frame(log, frame, city_from_reference) for frame in sample["frames"][:-1]]
    current = _frame(log, sample["frames"][-1], city_from_reference)
    sweep = pd.read_feather(io.BytesIO(log.files[f"{log.root}/sensors/lidar/{timestamp}.feather"]))
    points_ego = sweep[["x", "y", "z"]].to_numpy(dtype=np.float64)

    footprints: dict[str, list[np.ndarray]] = {"vehicle": [], "vulnerable_road_user": [], "other_obstacle": []}
    unseen, boxes, labels = [], [], []
    for item in log.annotations[log.annotations.timestamp_ns == timestamp].itertuples():
        heading = quaternion_yaw(item.qw, item.qx, item.qy, item.qz)
        corners = footprint(item.tx_m, item.ty_m, heading, item.length_m, item.width_m)
        if item.num_interior_pts == 0:
            unseen.append(corners)
            continue
        bev_class, box_class = _object_class(item.category)
        footprints[bev_class].append(corners)
        boxes.append([item.tx_m, item.ty_m, item.tz_m, item.length_m, item.width_m, item.height_m,
                      heading, 0.0, 0.0])
        labels.append(BOX_CLASSES.index(box_class))

    labelled = {
        **_map_targets(log.map, pose2d, camera_valid(current)),
        **object_labels(footprints, unseen, object_valid(current, points_ego)),
    }
    members = sample_members(
        current, past, points_ego=points_ego,
        segmentation=bev_segmentation(labelled, never_labelled=NEVER_LABELLED),
        boxes=boxes, box_labels=labels,
        split_group_uid=f"{DATASET}-{log.log}", meta={"dataset": DATASET, "log": log.log, "timestamp_ns": int(timestamp),
              "cameras": list(sample["frames"][-1])},
    )
    return f"av2-{log.log}-{timestamp}", members


def iter_samples(files: Mapping[str, bytes], part: int, parts: int) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Samples of every ``parts``-th log (``<split>/<log>`` roots in the files), starting at ``part``."""
    roots = sorted({name.rsplit("/", 1)[0] for name in files if name.endswith("/annotations.feather")})
    for root in roots[part::parts]:
        log = Log(files, root)
        annotated = log.annotations.timestamp_ns.unique().tolist()
        for sample in plan_samples(log.log, annotated, log.camera_times()):
            yield build_sample(log, sample)

