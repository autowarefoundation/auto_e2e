"""What every AutoBEV-only packer shares (nuScenes, Argoverse 2, ONCE, SimBEV).

A packer reads upstream files from a ``Mapping[str, bytes]`` keyed by snapshot
member path, builds each sample in its current ego frame with the origin on
the road, and returns ``(key, members)`` pairs with exactly the members of
``autobev_members``. ``write_partition`` stores them as WebDataset shards and
writes the partition manifest the publication and the loader read.

These datasets have no trajectory, route or map raster members; the loader
fills those with zeros. Classes a dataset cannot label are invalid in every
cell, never negative.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from data_parsing.nuplan.packing import (
    camera_visibility_from_projection_matrices,
    lidar_observability_from_points,
)
from data_parsing.nuplan.targets import _rasterize_polygons
from data_processing.autobev_members import (
    BOXES_MEMBER,
    DEPTH_MEMBER,
    HISTORY_PROJECTION_MEMBER,
    PROJECTION_MEMBER,
    camera_member,
    check_projections,
    crop_projection,
    encode_boxes,
    encode_camera,
    encode_depth,
    encode_npy,
    history_camera_member,
    manifest_contract,
)
from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_CLASSES,
    BEV_SEGMENTATION_MEMBER,
    encode_bev_segmentation,
)
from model_components.autobev.geometry import AUTOBEV_GEOMETRY, IMAGE_H, IMAGE_W

SAMPLE_KIND: Final = "autobev_only"
META_MEMBER: Final = "meta.json"
# About 0.5 GB per shard at 8 MB per sample, as the experiment packs.
SAMPLES_PER_SHARD: Final = 64
# Two cells wide, as nuPlan's lane_boundary targets on the AutoBEV grid.
LANE_WIDTH_M: Final = 2.0 * AUTOBEV_GEOMETRY.meters_per_pixel
OBJECT_CLASSES: Final = ("vehicle", "vulnerable_road_user", "other_obstacle")
EGO: Final = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CameraFrame:
    """Six cropped views of one instant, projected from the current ego frame."""

    jpegs: tuple[bytes, ...]
    projections: np.ndarray  # [6, 3, 4] current ego to cropped pixels
    camera_from_ego: np.ndarray  # [6, 4, 4]


def camera_frame(
    images: Sequence[np.ndarray],
    intrinsics: Sequence[np.ndarray],
    camera_from_ego: Sequence[np.ndarray],
) -> CameraFrame:
    """Crop six undistorted images and carry their pinhole projections through the crop."""
    jpegs, projections = [], []
    for rgb, k, camera in zip(images, intrinsics, camera_from_ego, strict=True):
        height, width = rgb.shape[:2]
        jpegs.append(encode_camera(rgb))
        projections.append(crop_projection(np.asarray(k, np.float64), width, height) @ camera[:3])
    return CameraFrame(tuple(jpegs), np.stack(projections), np.stack(camera_from_ego).astype(np.float64))


def decode_rgb(payload: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def undistort(rgb: np.ndarray, k: np.ndarray, distortion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rectify an image with OpenCV distortion terms; returns the image and its pinhole matrix."""
    import cv2

    size = (rgb.shape[1], rgb.shape[0])
    k = np.asarray(k, np.float64)
    distortion = np.asarray(distortion, np.float64).reshape(-1)
    rectified_k, _ = cv2.getOptimalNewCameraMatrix(k, distortion, size, 0.0, size)
    return cv2.undistort(rgb, k, distortion, None, rectified_k), rectified_k


def quaternion_yaw(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def footprint(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    """The four ground corners [4, 2] of a box centred at (x, y) with heading ``yaw``."""
    cos, sin = math.cos(yaw), math.sin(yaw)
    half = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length / 2.0, width / 2.0]
    return half @ np.array([[cos, sin], [-sin, cos]]) + [x, y]


def object_valid(current: CameraFrame, points_ego: np.ndarray) -> np.ndarray:
    """Cells a current camera and the LiDAR both observe, where object classes are labelled."""
    return camera_valid(current) & lidar_observability_from_points(points_ego, geometry=AUTOBEV_GEOMETRY)


def camera_valid(current: CameraFrame) -> np.ndarray:
    """Cells whose ground some current camera sees, where map classes are labelled."""
    return camera_visibility_from_projection_matrices(
        current.projections, image_width=IMAGE_W, image_height=IMAGE_H, geometry=AUTOBEV_GEOMETRY
    )


def bev_segmentation(
    labelled: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    never_labelled: frozenset[str],
) -> bytes:
    """Encode the eight classes from ``{class: (target, valid)}``; every other class is invalid.

    ``never_labelled`` is the dataset's constant set of classes it cannot
    provide; labelling one of them is refused, so the set stays true.
    """
    unknown = set(labelled) - set(BEV_SEGMENTATION_CLASSES)
    if unknown or set(labelled) & never_labelled:
        raise ValueError(f"labelled classes {sorted(labelled)} break the class rules")
    shape = (len(BEV_SEGMENTATION_CLASSES), AUTOBEV_GEOMETRY.height_px, AUTOBEV_GEOMETRY.width_px)
    target = np.zeros(shape, np.float32)
    valid = np.zeros(shape, np.bool_)
    for index, name in enumerate(BEV_SEGMENTATION_CLASSES):
        if name in labelled:
            target[index], valid[index] = labelled[name]
    return encode_bev_segmentation(target, valid)


def object_labels(
    footprints: Mapping[str, Sequence[np.ndarray]],
    unseen: Sequence[np.ndarray],
    valid: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rasterize ego-frame object footprints; boxes no sensor saw are unknown, not background."""
    unknown = _rasterize_polygons(list(unseen), EGO, AUTOBEV_GEOMETRY) > 0
    return {
        name: (_rasterize_polygons(list(polygons), EGO, AUTOBEV_GEOMETRY), valid & ~unknown)
        for name, polygons in footprints.items()
    }


def sample_members(
    current: CameraFrame,
    history: Sequence[CameraFrame],
    *,
    points_ego: np.ndarray,
    segmentation: bytes,
    boxes: Sequence[Sequence[float]],
    box_labels: Sequence[int],
    meta: Mapping[str, object],
) -> dict[str, bytes]:
    """The AutoBEV members of one sample; ``history`` is oldest first."""
    history_projection = np.stack([frame.projections for frame in history])
    check_projections(current.projections, history_projection)
    members = {camera_member(view): jpeg for view, jpeg in enumerate(current.jpegs)}
    for index, frame in enumerate(history):
        members.update({history_camera_member(index, view): jpeg for view, jpeg in enumerate(frame.jpegs)})
    members[PROJECTION_MEMBER] = encode_npy(current.projections)
    members[HISTORY_PROJECTION_MEMBER] = encode_npy(history_projection)
    members[BEV_SEGMENTATION_MEMBER] = segmentation
    members[DEPTH_MEMBER] = encode_depth(points_ego, current.projections, current.camera_from_ego)
    members[BOXES_MEMBER] = encode_boxes(np.asarray(boxes, np.float32).reshape(-1, 9), box_labels)
    members[META_MEMBER] = json.dumps(dict(meta), sort_keys=True).encode()
    return members


def _add(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(payload), 0, 0o444
    archive.addfile(info, io.BytesIO(payload))


def write_partition(
    out: Path,
    *,
    partition_id: str,
    dataset: str,
    pack_contract: str,
    cameras: Sequence[str],
    never_labelled: frozenset[str],
    samples: Iterable[tuple[str, Mapping[str, bytes]]],
) -> dict[str, object]:
    """Write ``samples`` as deterministic WebDataset shards and the partition manifest."""
    out.mkdir(parents=True, exist_ok=True)
    names, counts = [], []
    archive = None
    for key, members in samples:
        if "." in key or "/" in key:
            raise ValueError(f"sample key {key!r} must not contain '.' or '/'")
        if archive is None:
            names.append(f"{partition_id}-{len(names):05d}.tar")
            counts.append(0)
            archive = tarfile.open(out / names[-1], "w")
        for member, payload in sorted(members.items()):
            _add(archive, f"{key}.{member}", payload)
        counts[-1] += 1
        if counts[-1] == SAMPLES_PER_SHARD:
            archive.close()
            archive = None
    if archive is not None:
        archive.close()
    manifest = {
        **manifest_contract(),
        "sample_kind": SAMPLE_KIND,
        "dataset": dataset,
        "pack_contract": pack_contract,
        "partition_id": partition_id,
        "total_samples": sum(counts),
        "shards": len(names),
        "shard_names": names,
        "shard_sample_counts": dict(zip(names, counts)),
        "shard_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in names},
        "camera_order": list(cameras),
        "bev_classes": list(BEV_SEGMENTATION_CLASSES),
        "never_labelled_classes": sorted(never_labelled),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest
