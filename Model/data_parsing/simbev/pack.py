"""Pack the SimBEV 1.0 dataset (CARLA, synthetic) as AutoBEV-only samples.

SimBEV 1.0 (Mehr and Eskandarian, arXiv:2502.01894) has 320 scenes of 16 s at
20 Hz from 11 CARLA towns (train, val and test together; all are used for
training), recorded with six 1600x900 pinhole cameras in the nuScenes layout,
a 128-channel LiDAR and per-frame ego poses whose origin is on the road. The
snapshot holds only the members a sample reads (``needed_member``): the
camera frames of its eight T8 frames (every 0.5 s) and the LiDAR sweep, BEV
segmentation and boxes of its current frame. A sample is taken every second
from 3.5 s on, 13 per scene.

Cameras: the six SimBEV cameras map one to one onto the AutoBEV views in
``CAMERAS`` order (front, front-left, front-right, back-left, back,
back-right).

Labels on the AutoBEV grid:

- drivable_area from the ground truth ``road`` channel (360x360 cells of
  0.4 m centred on the ego), resampled to 0.5 m cells; cells past the 72 m
  edge of that grid are invalid
- vehicle / vulnerable_road_user from the 3D boxes by CARLA semantic tag;
  boxes without ``valid_flag`` (no LiDAR or radar point) are unknown for the
  object classes, and the ego vehicle (role ``hero``) is skipped
- lane_boundary, intersection, crosswalk, stop_line and other_obstacle are
  never labelled: SimBEV 1.0 has no map or static obstacle labels
"""

from __future__ import annotations

import io
import json
import math
import pickle
import re
from collections.abc import Iterator, Mapping
from typing import Final

import numpy as np

from data_parsing.nuplan.packing import _quaternion_transform
from data_processing.autobev_members import BOX_CLASSES
from data_processing.autobev_pack import (
    bev_segmentation,
    camera_frame,
    camera_valid,
    decode_rgb,
    object_labels,
    object_valid,
    sample_members,
)
from model_components.autobev.geometry import AUTOBEV_GEOMETRY, HISTORY_FRAMES

DATASET: Final = "simbev"
# Bump when a rule that decides the packed bytes changes; part of dataset_id.
PACK_CONTRACT: Final = "simbev-autobev-v1"
CAMERAS: Final = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
SPLITS: Final = ("train", "val", "test")
NEVER_LABELLED: Final = frozenset({"lane_boundary", "intersection", "crosswalk", "stop_line", "other_obstacle"})
# 320 scenes in 16 partitions of 260 samples; each reads the 590 MB of infos once.
PARTITIONS: Final = 16
# 20 Hz: 10 frames are 0.5 s and 20 frames 1 s; 320 frames per scene.
FRAMES_PER_SCENE: Final = 320
FRAME_STEP: Final = 10
SAMPLE_STEP: Final = 20
FIRST_SAMPLE: Final = HISTORY_FRAMES * FRAME_STEP
SAMPLE_FRAMES: Final = frozenset(range(FIRST_SAMPLE, FRAMES_PER_SCENE, SAMPLE_STEP))
CAMERA_FRAMES: Final = frozenset(f - k * FRAME_STEP for f in SAMPLE_FRAMES for k in range(HISTORY_FRAMES + 1))
SIMBEV_BEV_CELLS: Final = 360
SIMBEV_BEV_RES_M: Final = 0.4
# CARLA 0.9.15 CityObjectLabel tags of the boxes.
TAGS: Final = {
    12: ("vulnerable_road_user", "PEDESTRIAN"), 13: ("vulnerable_road_user", "BICYCLE"),
    14: ("vehicle", "VEHICLE"), 15: ("vehicle", "VEHICLE"), 16: ("vehicle", "VEHICLE"),
    17: ("vehicle", "VEHICLE"), 18: ("vehicle", "VEHICLE"), 19: ("vulnerable_road_user", "BICYCLE"),
}
INFOS: Final = frozenset(f"simbev_infos_{split}.json" for split in SPLITS)
_MEMBER_RE: Final = re.compile(r"SimBEV-scene-\d{4}-frame-(\d{4})-(.+)\.[a-z]+$")


def needed_member(name: str) -> bool:
    """Whether a member of a SimBEV archive is read: the infos, or a frame some sample uses."""
    base = name.rsplit("/", 1)[-1]
    if base in INFOS:
        return True
    match = _MEMBER_RE.search(base)
    if match is None:
        return False
    frame, sensor = int(match.group(1)), match.group(2)
    return frame in (CAMERA_FRAMES if sensor.startswith("RGB-") else SAMPLE_FRAMES)


class _ArrayUnpickler(pickle.Unpickler):
    """Load GT_DET object arrays without running arbitrary pickled code."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing {module}.{name}")


def load_detections(payload: bytes) -> list[dict]:
    from numpy.lib import format as npy_format

    stream = io.BytesIO(payload)
    major, _ = npy_format.read_magic(stream)
    # GT_DET files are .npy object arrays; the header is skipped and the pickle loaded safely.
    (npy_format.read_array_header_1_0 if major == 1 else npy_format.read_array_header_2_0)(stream)
    return list(_ArrayUnpickler(stream).load())


def road_labels(road: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample the ego-centred 0.4 m road channel to the AutoBEV grid and mark the cells it covers."""
    x, y = AUTOBEV_GEOMETRY.pixel_center_grids()
    mpp = AUTOBEV_GEOMETRY.meters_per_pixel
    half = SIMBEV_BEV_CELLS * SIMBEV_BEV_RES_M / 2.0
    # Row 0 of the SimBEV grid is the front edge (+x) and column 0 the left edge (+y).
    total = np.zeros_like(x)
    offsets = (np.arange(4) + 0.5) / 4 - 0.5
    for dx in offsets:
        for dy in offsets:
            rows = np.floor((half - (x + dx * mpp)) / SIMBEV_BEV_RES_M).astype(int)
            cols = np.floor((half - (y + dy * mpp)) / SIMBEV_BEV_RES_M).astype(int)
            inside = (rows >= 0) & (rows < SIMBEV_BEV_CELLS) & (cols >= 0) & (cols < SIMBEV_BEV_CELLS)
            last = SIMBEV_BEV_CELLS - 1
            total += np.where(inside, road[rows.clip(0, last), cols.clip(0, last)], 0.0)
    covered = (np.abs(x) + mpp / 2 <= half) & (np.abs(y) + mpp / 2 <= half)
    return (total / 16.0).astype(np.float32), covered


def detection_labels(detections: list[dict], ego_from_global: np.ndarray):
    """Ego-frame footprints by class, unseen footprints, box rows and box labels."""
    footprints: dict[str, list[np.ndarray]] = {"vehicle": [], "vulnerable_road_user": []}
    unseen, boxes, labels = [], [], []
    for item in detections:
        tags = item.get("semantic_tags") or []
        if (item.get("attributes") or {}).get("role_name") == "hero" or not tags or tags[0] not in TAGS:
            continue
        bev_class, box_class = TAGS[tags[0]]
        corners = np.asarray(item["bounding_box"], np.float64).reshape(8, 3)
        corners = (ego_from_global @ np.c_[corners, np.ones(8)].T)[:3].T
        # CARLA vertex order: 0 -> 4 runs along the length and 0 -> 2 along the width.
        ground = corners[[0, 2, 6, 4], :2]
        if not item.get("valid_flag"):
            unseen.append(ground)
            continue
        length_axis, width_axis = corners[4] - corners[0], corners[2] - corners[0]
        center = corners.mean(axis=0)
        boxes.append([
            center[0], center[1], center[2],
            float(np.linalg.norm(length_axis[:2])), float(np.linalg.norm(width_axis[:2])),
            float(corners[:, 2].max() - corners[:, 2].min()),
            math.atan2(length_axis[1], length_axis[0]), 0.0, 0.0,
        ])
        footprints[bev_class].append(ground)
        labels.append(BOX_CLASSES.index(box_class))
    return footprints, unseen, boxes, labels


def _base(path: str) -> str:
    return path.rsplit("/", 1)[-1]


class Scenes:
    """Scene records of every split and the snapshot path of every member by file name."""

    def __init__(self, files: Mapping[str, bytes]):
        self.files = files
        self.path = {_base(name): name for name in files}
        self.metadata: dict[str, dict] = {}
        self.scenes: dict[str, dict] = {}
        for split in SPLITS:
            infos = json.loads(files[self.path[f"simbev_infos_{split}.json"]])
            for scene, record in infos["data"].items():
                self.scenes[scene] = record
                self.metadata[scene] = infos["metadata"]

    def read(self, upstream_path: str) -> bytes:
        return self.files[self.path[_base(upstream_path)]]


def _global_from_ego(frame: Mapping) -> np.ndarray:
    return _quaternion_transform(frame["ego2global_translation"], frame["ego2global_rotation"])


def _frame(scenes: Scenes, metadata: Mapping, frame: Mapping, global_from_reference: np.ndarray):
    k = np.asarray(metadata["camera_intrinsics"], np.float64)
    images, cameras = [], []
    for camera in CAMERAS:
        rgb = decode_rgb(scenes.read(frame[f"RGB-{camera}"]))
        height, width = rgb.shape[:2]
        if abs(k[0, 2] * 2 - width) > 1 or abs(k[1, 2] * 2 - height) > 1:
            raise ValueError(f"{camera} image size {width}x{height} does not match the intrinsics")
        calib = metadata[camera]
        ego_from_camera = _quaternion_transform(calib["sensor2ego_translation"], calib["sensor2ego_rotation"])
        images.append(rgb)
        cameras.append(np.linalg.inv(_global_from_ego(frame) @ ego_from_camera) @ global_from_reference)
    return camera_frame(images, [k] * len(CAMERAS), cameras)


def build_sample(scenes: Scenes, scene: str, index: int) -> tuple[str, dict[str, bytes]]:
    metadata, frames = scenes.metadata[scene], scenes.scenes[scene]["scene_data"]
    current_frame = frames[index]
    global_from_reference = _global_from_ego(current_frame)
    past = [_frame(scenes, metadata, frames[index - k * FRAME_STEP], global_from_reference)
            for k in range(HISTORY_FRAMES, 0, -1)]
    current = _frame(scenes, metadata, current_frame, global_from_reference)

    lidar = metadata["LIDAR"]
    ego_from_lidar = _quaternion_transform(lidar["sensor2ego_translation"], lidar["sensor2ego_rotation"])
    with np.load(io.BytesIO(scenes.read(current_frame["LIDAR"]))) as archive:
        points = np.asarray(archive["data"], np.float64)[:, :3]
    points_ego = (ego_from_lidar @ np.c_[points, np.ones(len(points))].T)[:3].T
    with np.load(io.BytesIO(scenes.read(current_frame["GT_SEG"]))) as archive:
        road, covered = road_labels(np.asarray(archive["data"][0], np.float32))
    footprints, unseen, boxes, labels = detection_labels(
        load_detections(scenes.read(current_frame["GT_DET"])), np.linalg.inv(global_from_reference)
    )
    labelled = {
        "drivable_area": (road, camera_valid(current) & covered),
        **object_labels(footprints, unseen, object_valid(current, points_ego)),
    }
    members = sample_members(
        current, past, points_ego=points_ego,
        segmentation=bev_segmentation(labelled, never_labelled=NEVER_LABELLED),
        boxes=boxes, box_labels=labels,
        meta={"dataset": DATASET, "scene": scene, "frame": index,
              "map": scenes.scenes[scene]["scene_info"]["map"]},
    )
    return f"simbev-{scene}-{index:04d}", members


def iter_samples(files: Mapping[str, bytes], part: int, parts: int) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Samples of every ``parts``-th scene in name order, starting at ``part``."""
    scenes = Scenes(files)
    for scene in sorted(scenes.scenes)[part::parts]:
        for index in sorted(SAMPLE_FRAMES):
            yield build_sample(scenes, scene, index)
