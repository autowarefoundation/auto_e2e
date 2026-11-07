"""Members of an AutoBEV sample, shared by every packer and the loader.

An AutoBEV sample carries what the AutoBEV encoder and its losses read, on
``model_components.autobev.geometry``:

- ``cam_{v}.jpg``: the six current cameras, undistorted, scaled to cover
  1344x768 with the aspect ratio kept and centre-cropped (``encode_camera``);
- ``hist_{h}_cam_{v}.jpg``: the same for the seven history frames, oldest
  first, every 0.5 s before the current frame (packers with a frame pool store
  pool references instead and the loader resolves them);
- ``projection.npy`` [6, 3, 4] and ``history_projection.npy`` [7, 6, 3, 4]:
  current-ego points (road-level origin) to the cropped pixels of the current
  and past images, float32;
- ``bev_segmentation.npz``: ``encode_bev_segmentation`` of the eight classes on
  the AutoBEV grid; classes a dataset cannot label are invalid, not negative;
- ``depth.npz``: the current LiDAR sweep as sparse per-camera pixel depth
  (``encode_depth``), read as the nearest return per stride-8 cell;
- ``boxes.npz``: 3D boxes in the current ego frame (``encode_boxes``).

LiDAR enters only these teachers (depth, label visibility); nothing the model
reads at inference comes from it.
"""

from __future__ import annotations

import io
from typing import Final, Mapping

import numpy as np

from model_components.autobev.geometry import (
    AUTOBEV_CONTRACT,
    FEATURE_STRIDE,
    HISTORY_FRAMES,
    IMAGE_H,
    IMAGE_W,
    NUM_VIEWS,
    check_autobev_contract,
)

AUTOBEV_MEMBERS_VERSION: Final = "autobev_members_v1"
PROJECTION_MEMBER: Final = "projection.npy"
HISTORY_PROJECTION_MEMBER: Final = "history_projection.npy"
DEPTH_MEMBER: Final = "depth.npz"
BOXES_MEMBER: Final = "boxes.npz"
BOX_CLASSES: Final = ("VEHICLE", "PEDESTRIAN", "BICYCLE", "OTHER")
BOX_COLUMNS: Final = ("x", "y", "z", "length", "width", "height", "yaw", "vx", "vy")
MAX_DEPTH_M: Final = 120.0
FEATURE_H: Final = IMAGE_H // FEATURE_STRIDE
FEATURE_W: Final = IMAGE_W // FEATURE_STRIDE
JPEG_QUALITY: Final = 90
# Manifest ``sample_kind``: "planner" packs (nuPlan) also carry navigation,
# ego and trajectory members; "autobev_only" packs carry only the members
# above and train the BEV losses alone.
SAMPLE_KINDS: Final = ("planner", "autobev_only")


def camera_member(view: int) -> str:
    return f"cam_{view}.jpg"


def history_camera_member(frame: int, view: int) -> str:
    return f"hist_{frame}_cam_{view}.jpg"


def manifest_contract() -> dict[str, object]:
    """Fields every pack manifest with AutoBEV members records and the loader checks."""
    return {
        "autobev_members_version": AUTOBEV_MEMBERS_VERSION,
        "autobev_geometry": dict(AUTOBEV_CONTRACT),
        "box_classes": list(BOX_CLASSES),
    }


def check_manifest_contract(manifest: Mapping[str, object], *, source: str) -> None:
    """Raise unless ``manifest`` records ``manifest_contract()`` and a known ``sample_kind``."""
    geometry = manifest.get("autobev_geometry")
    check_autobev_contract(geometry if isinstance(geometry, Mapping) else None, source=source)
    differing = sorted(key for key, value in manifest_contract().items() if manifest.get(key) != value)
    if differing:
        raise ValueError(f"{source} has a different AutoBEV member contract: {differing}")
    if manifest.get("sample_kind") not in SAMPLE_KINDS:
        raise ValueError(f"{source} has an unknown AutoBEV sample_kind {manifest.get('sample_kind')!r}")


def crop_params(width: int, height: int) -> tuple[float, float, int, int, int, int]:
    """Scale factors, scaled size and crop offsets that cover 1344x768 with the aspect kept."""
    scale = max(IMAGE_W / width, IMAGE_H / height)
    scaled_w, scaled_h = round(width * scale), round(height * scale)
    return (
        scaled_w / width,
        scaled_h / height,
        scaled_w,
        scaled_h,
        (scaled_w - IMAGE_W) // 2,
        (scaled_h - IMAGE_H) // 2,
    )


def encode_camera(rgb: np.ndarray) -> bytes:
    """Scale an undistorted [H, W, 3] uint8 image to cover 1344x768, centre-crop it, and JPEG it."""
    from PIL import Image

    height, width = rgb.shape[:2]
    _, _, scaled_w, scaled_h, left, top = crop_params(width, height)
    image = Image.fromarray(rgb).resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
    output = io.BytesIO()
    image.crop((left, top, left + IMAGE_W, top + IMAGE_H)).save(
        output, format="JPEG", quality=JPEG_QUALITY
    )
    return output.getvalue()


def crop_projection(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    """Apply the ``encode_camera`` pixel transform to a [..., 3, 3|4] intrinsic or projection."""
    sx, sy, _, _, left, top = crop_params(width, height)
    out = np.array(matrix, dtype=np.float64)
    out[..., 0, :] = out[..., 0, :] * sx - left * out[..., 2, :]
    out[..., 1, :] = out[..., 1, :] * sy - top * out[..., 2, :]
    return out


def encode_npy(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, np.ascontiguousarray(array, dtype=np.float32))
    return output.getvalue()


def decode_npy(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


def encode_depth(
    points_ego: np.ndarray,
    projections: np.ndarray,
    camera_from_ego: np.ndarray,
) -> bytes:
    """Project ego-frame LiDAR points into each current camera as sparse pixel depth.

    ``projections`` [6, 3, 4] map ego points to cropped pixels and
    ``camera_from_ego`` [6, 4, 4] gives each point's camera depth. Only the
    nearest return per pixel is kept, so occluded points do not leak.
    """
    homogeneous = np.concatenate([points_ego, np.ones((len(points_ego), 1))], axis=1)
    arrays = {}
    for view, (matrix, camera) in enumerate(zip(projections, camera_from_ego)):
        depth = (camera[:3] @ homogeneous.T)[2]
        uvw = matrix.astype(np.float64) @ homogeneous.T
        keep = (depth > 0.5) & (depth < MAX_DEPTH_M)
        u = uvw[0, keep] / uvw[2, keep]
        v = uvw[1, keep] / uvw[2, keep]
        d = depth[keep]
        inside = (u >= 0) & (u < IMAGE_W) & (v >= 0) & (v < IMAGE_H)
        u, v, d = u[inside].astype(np.int32), v[inside].astype(np.int32), d[inside]
        order = np.lexsort((d, v * IMAGE_W + u))
        flat = (v * IMAGE_W + u)[order]
        first = np.concatenate([[True], flat[1:] != flat[:-1]])
        arrays[f"uv_{view}"] = np.stack([u[order][first], v[order][first]], 1).astype(np.uint16)
        arrays[f"depth_{view}"] = d[order][first].astype(np.float16)
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def decode_stride_depth(payload: bytes) -> np.ndarray:
    """Nearest LiDAR depth per stride-8 cell, [6, 96, 168] metres, 0 where no point falls."""
    arrays = np.load(io.BytesIO(payload), allow_pickle=False)
    out = np.full((NUM_VIEWS, FEATURE_H, FEATURE_W), np.inf, dtype=np.float32)
    for view in range(NUM_VIEWS):
        uv = arrays[f"uv_{view}"].astype(np.int64)
        depth = arrays[f"depth_{view}"].astype(np.float32)
        cell = (uv[:, 1] // FEATURE_STRIDE) * FEATURE_W + uv[:, 0] // FEATURE_STRIDE
        np.minimum.at(out[view].reshape(-1), cell, depth)
    out[~np.isfinite(out)] = 0.0
    return out


def encode_boxes(boxes: np.ndarray, labels: np.ndarray) -> bytes:
    """Store [N, 9] boxes (``BOX_COLUMNS``, centre z above the road) and [N] ``BOX_CLASSES`` indices."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, len(BOX_COLUMNS))
    labels = np.asarray(labels, np.int64).reshape(-1)
    if len(boxes) != len(labels) or (labels.size and (labels.min() < 0 or labels.max() >= len(BOX_CLASSES))):
        raise ValueError("box labels must index BOX_CLASSES, one per box")
    output = io.BytesIO()
    np.savez_compressed(output, boxes=boxes, labels=labels)
    return output.getvalue()


def decode_boxes(payload: bytes | None) -> tuple[np.ndarray, np.ndarray]:
    if payload is None:
        return np.zeros((0, len(BOX_COLUMNS)), np.float32), np.zeros(0, np.int64)
    arrays = np.load(io.BytesIO(payload), allow_pickle=False)
    return arrays["boxes"].astype(np.float32), arrays["labels"].astype(np.int64)


def check_projections(projection: np.ndarray, history_projection: np.ndarray) -> None:
    if projection.shape != (NUM_VIEWS, 3, 4):
        raise ValueError(f"projection must be [{NUM_VIEWS},3,4], got {projection.shape}")
    if history_projection.shape != (HISTORY_FRAMES, NUM_VIEWS, 3, 4):
        raise ValueError(
            f"history projection must be [{HISTORY_FRAMES},{NUM_VIEWS},3,4], got {history_projection.shape}"
        )
    if not (np.isfinite(projection).all() and np.isfinite(history_projection).all()):
        raise ValueError("projections must be finite")
