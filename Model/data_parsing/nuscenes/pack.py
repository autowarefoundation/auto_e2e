"""Pack nuScenes v1.0 trainval keyframes as AutoBEV-only samples.

Every keyframe of every trainval scene with seven earlier keyframes is a
sample; train and val are both used for training, since the evaluation sets
are nuPlan Pittsburgh and KITScenes. Keyframes run at 2 Hz, so the seven
previous keyframes are the T8 history at 0.5 s. The reference frame is the
ego pose of the LIDAR_TOP keyframe; nuScenes puts the ego origin at the rear
axle on the road. Images are already rectified.

Cameras: the six nuScenes cameras map one to one onto the AutoBEV views in
``CAMERAS`` order (front, front-left, front-right, back-left, back,
back-right).

Labels on the AutoBEV grid, in the nuPlan taxonomy; every class is provided:

- drivable_area: map expansion drivable_area polygons minus their holes
- lane_boundary: left and right boundaries of every lane (not lane
  connectors), as nuPlan draws both boundaries of physical lanes
- intersection: road_segment polygons with is_intersection
- crosswalk: ped_crossing; stop_line: stop_line polygons
- vehicle / vulnerable_road_user / other_obstacle: footprints of annotations
  with at least one LiDAR or radar point; the footprints of the others are
  unknown for the object classes
Map classes are valid where a camera sees the ground, object classes where a
camera and the LiDAR both observe the cell. Sample keys carry the scene name,
so training can hold out whole scenes.

The table and map JSON are read directly (no nuscenes-devkit).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
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
)
from model_components.autobev.geometry import AUTOBEV_GEOMETRY, HISTORY_FRAMES

DATASET: Final = "nuscenes"
# Bump when a rule that decides the packed bytes changes; part of dataset_id.
PACK_CONTRACT: Final = "nuscenes-autobev-v2"
VERSION: Final = "v1.0-trainval"
CAMERAS: Final = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
LIDAR: Final = "LIDAR_TOP"
NEVER_LABELLED: Final = frozenset[str]()
# 850 scenes in 32 partitions of about 880 samples; each reads the whole metadata once.
PARTITIONS: Final = 32
# Map records within this square around the ego cover the grid's circumscribed radius (about 89 m).
PATCH_M: Final = 180.0
_TABLES: Final = (
    "scene", "sample", "sample_data", "calibrated_sensor", "ego_pose",
    "sample_annotation", "instance", "category", "sensor", "log",
)


def object_class(category: str) -> tuple[str | None, str]:
    """(BEV class or None, box class) for a nuScenes category name."""
    if category.startswith("human.pedestrian"):
        return "vulnerable_road_user", "PEDESTRIAN"
    if category == "vehicle.bicycle":
        return "vulnerable_road_user", "BICYCLE"
    if category.startswith("vehicle."):
        return "vehicle", "VEHICLE"
    if category.startswith(("movable_object.", "static_object.", "animal")):
        return "other_obstacle", "OTHER"
    return None, "OTHER"


def _pose(record: Mapping) -> np.ndarray:
    return _quaternion_transform(record["translation"], record["rotation"])


class Tables:
    """The keyframe view of the nuScenes tables that the packer needs."""

    def __init__(self, files: Mapping[str, bytes]):
        tables = {name: json.loads(files[f"{VERSION}/{name}.json"]) for name in _TABLES}
        by_token = {name: {row["token"]: row for row in rows} for name, rows in tables.items()}
        channel_of = {
            token: by_token["sensor"][row["sensor_token"]]["channel"]
            for token, row in by_token["calibrated_sensor"].items()
        }
        self.keyframes: dict[str, dict[str, dict]] = {}
        for row in tables["sample_data"]:
            if row["is_key_frame"]:
                self.keyframes.setdefault(row["sample_token"], {})[channel_of[row["calibrated_sensor_token"]]] = row
        used_poses = {row["ego_pose_token"] for rows in self.keyframes.values() for row in rows.values()}
        self.ego_pose = {token: by_token["ego_pose"][token] for token in used_poses}
        self.calibrated_sensor = by_token["calibrated_sensor"]
        self.sample = by_token["sample"]
        self.scenes = sorted(tables["scene"], key=lambda row: row["name"])
        self.location = {row["token"]: by_token["log"][row["log_token"]]["location"] for row in tables["scene"]}
        category_of = {
            token: by_token["category"][row["category_token"]]["name"]
            for token, row in by_token["instance"].items()
        }
        self.annotations: dict[str, list[dict]] = {}
        for row in tables["sample_annotation"]:
            self.annotations.setdefault(row["sample_token"], []).append(
                {**row, "category_name": category_of[row["instance_token"]]}
            )


def scene_samples(tables: Tables, scene: Mapping) -> list[tuple[list[str], str]]:
    """``(history tokens oldest first, sample token)`` for every keyframe with seven before it."""
    tokens = []
    token = scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = tables.sample[token]["next"]
    return [(tokens[i - HISTORY_FRAMES: i], tokens[i]) for i in range(HISTORY_FRAMES, len(tokens))]


class MapLayers:
    """Map expansion layers of one location, with bounding boxes for the patch query."""

    def __init__(self, payload: bytes):
        layers = json.loads(payload)
        nodes = {node["token"]: (node["x"], node["y"]) for node in layers["node"]}

        def ring(tokens):
            return np.asarray([nodes[token] for token in tokens], np.float64)

        polygons = {
            polygon["token"]: (ring(polygon["exterior_node_tokens"]),
                               [ring(hole["node_tokens"]) for hole in polygon["holes"]])
            for polygon in layers["polygon"]
        }

        def of(tokens):
            return [polygons[token] for token in tokens]

        self.polygons = {
            "drivable_area": [p for row in layers["drivable_area"] for p in of(row["polygon_tokens"])],
            "crosswalk": of(row["polygon_token"] for row in layers["ped_crossing"]),
            "stop_line": of(row["polygon_token"] for row in layers["stop_line"]),
            "intersection": of(row["polygon_token"] for row in layers["road_segment"] if row["is_intersection"]),
        }
        self.lines = [
            np.asarray([[node["x"], node["y"]] for node in lane[side]], np.float64)
            for lane in layers["lane"]
            for side in ("left_lane_divider_segment_nodes", "right_lane_divider_segment_nodes")
            if len(lane.get(side) or []) >= 2
        ]
        self._polygon_boxes = {
            name: _boxes([exterior for exterior, _ in items]) for name, items in self.polygons.items()
        }
        self._line_boxes = _boxes(self.lines)

    def targets(self, pose2d: tuple[float, float, float], valid: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        x, y, _ = pose2d
        patch = (x - PATCH_M / 2, y - PATCH_M / 2, x + PATCH_M / 2, y + PATCH_M / 2)
        out = {}
        for name, items in self.polygons.items():
            near = [items[i] for i in np.nonzero(_overlaps(self._polygon_boxes[name], patch))[0]]
            filled = _rasterize_polygons([exterior for exterior, _ in near], pose2d, AUTOBEV_GEOMETRY)
            holes = [hole for _, rings in near for hole in rings]
            filled = np.clip(filled - _rasterize_polygons(holes, pose2d, AUTOBEV_GEOMETRY), 0.0, 1.0)
            out[name] = (filled, valid)
        near_lines = [self.lines[i] for i in np.nonzero(_overlaps(self._line_boxes, patch))[0]]
        out["lane_boundary"] = (
            _rasterize_polylines(near_lines, pose2d, AUTOBEV_GEOMETRY, width_m=LANE_WIDTH_M), valid
        )
        return out


def _boxes(shapes: list[np.ndarray]) -> np.ndarray:
    if not shapes:
        return np.zeros((0, 4))
    return np.asarray([[*shape.min(axis=0), *shape.max(axis=0)] for shape in shapes])


def _overlaps(boxes: np.ndarray, patch: tuple[float, float, float, float]) -> np.ndarray:
    return (boxes[:, 0] <= patch[2]) & (boxes[:, 2] >= patch[0]) & (boxes[:, 1] <= patch[3]) & (boxes[:, 3] >= patch[1])


def _frame(files: Mapping[str, bytes], tables: Tables, token: str, global_from_reference: np.ndarray):
    images, intrinsics, cameras = [], [], []
    for channel in CAMERAS:
        data = tables.keyframes[token][channel]
        calib = tables.calibrated_sensor[data["calibrated_sensor_token"]]
        rgb = decode_rgb(files[data["filename"]])
        if rgb.shape[:2] != (data["height"], data["width"]):
            raise ValueError(f"{data['filename']}: image size differs from its record")
        global_from_camera = _pose(tables.ego_pose[data["ego_pose_token"]]) @ _pose(calib)
        images.append(rgb)
        intrinsics.append(np.asarray(calib["camera_intrinsic"], np.float64))
        cameras.append(np.linalg.inv(global_from_camera) @ global_from_reference)
    return camera_frame(images, intrinsics, cameras)


def build_sample(files: Mapping[str, bytes], tables: Tables, maps: MapLayers, scene: Mapping,
                 history: list[str], token: str) -> tuple[str, dict[str, bytes]]:
    lidar = tables.keyframes[token][LIDAR]
    global_from_reference = _pose(tables.ego_pose[lidar["ego_pose_token"]])
    reference_from_global = np.linalg.inv(global_from_reference)
    yaw = math.atan2(global_from_reference[1, 0], global_from_reference[0, 0])
    pose2d = (float(global_from_reference[0, 3]), float(global_from_reference[1, 3]), yaw)
    current = _frame(files, tables, token, global_from_reference)
    past = [_frame(files, tables, previous, global_from_reference) for previous in history]

    cloud = np.frombuffer(files[lidar["filename"]], dtype=np.float32).reshape(-1, 5)[:, :3]
    ego_from_lidar = _pose(tables.calibrated_sensor[lidar["calibrated_sensor_token"]])
    points_ego = (ego_from_lidar @ np.c_[cloud, np.ones(len(cloud))].T)[:3].T

    footprints: dict[str, list[np.ndarray]] = {"vehicle": [], "vulnerable_road_user": [], "other_obstacle": []}
    unseen, boxes, labels = [], [], []
    for annotation in tables.annotations.get(token, []):
        center = reference_from_global @ np.append(annotation["translation"], 1.0)
        heading = math.remainder(quaternion_yaw(*annotation["rotation"]) - yaw, math.tau)
        width, length, height = annotation["size"]
        corners = footprint(center[0], center[1], heading, length, width)
        if annotation["num_lidar_pts"] + annotation["num_radar_pts"] == 0:
            unseen.append(corners)
            continue
        bev_class, box_class = object_class(annotation["category_name"])
        if bev_class is not None:
            footprints[bev_class].append(corners)
        # nuScenes boxes stand on the road: the centre is half the height above it.
        boxes.append([center[0], center[1], height / 2.0, length, width, height, heading, 0.0, 0.0])
        labels.append(BOX_CLASSES.index(box_class))

    labelled = {
        **maps.targets(pose2d, camera_valid(current)),
        **object_labels(footprints, unseen, object_valid(current, points_ego)),
    }
    members = sample_members(
        current, past, points_ego=points_ego,
        segmentation=bev_segmentation(labelled, never_labelled=NEVER_LABELLED),
        boxes=boxes, box_labels=labels,
        split_group_uid=f"{DATASET}-{scene['name']}", meta={"dataset": DATASET, "scene": scene["name"], "location": tables.location[scene["token"]],
              "sample_token": token, "timestamp_us": tables.sample[token]["timestamp"]},
    )
    return f"nusc-{scene['name']}-{token}", members


def iter_samples(files: Mapping[str, bytes], part: int, parts: int) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Samples of every ``parts``-th scene in name order, starting at ``part``."""
    tables = Tables(files)
    maps: dict[str, MapLayers] = {}
    for scene in tables.scenes[part::parts]:
        location = tables.location[scene["token"]]
        if location not in maps:
            maps[location] = MapLayers(files[f"maps/expansion/{location}.json"])
        for history, token in scene_samples(tables, scene):
            yield build_sample(files, tables, maps[location], scene, history, token)
