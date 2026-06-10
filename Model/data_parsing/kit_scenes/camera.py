"""Camera frame loading for the KIT Scenes Multimodal dataset.

KIT Scenes stores per-frame JPEGs on disk (not videos), already at the 10 Hz
reference timeline, so a single ``frame_idx`` indexes every camera and the ego
poses alike. The ``kitscenes`` SDK's ``SensorDataLoader`` decodes a frame to an
RGB ``np.ndarray``; this module resizes/normalises it for the AutoE2E backbone
and stacks the views into the tensor the model expects.

Map tile (slot 7)
-----------------
KIT Scenes ships Lanelet2 HD maps, which are rasterised into a semantic RGB
image by ``map.generate_bev_map_tile``. The resulting ``(H, W, 3)`` uint8 array
is passed through the same backbone transform as the camera frames so that all
8 views have identical shape and normalisation. If the map is unavailable for
a scene (missing ``maps/map.osm`` or lanelet2 not installed), slot 7 falls
back to a zero tensor.
"""

from __future__ import annotations

import numpy as np
import torch
from kitscenes.sensors import SensorDataLoader
from PIL import Image
from torchvision.transforms import Compose

from .map import generate_bev_map_tile

# Camera directories used as visual tiles for the KIT Scenes dataset.
# Order: hi-res front, then the 6 surround ring cameras. The 2-camera stereo
# pair (camera_base_front_left_rect/_right_rect) is intentionally dropped; it
# duplicates forward coverage already given by the ring front camera.
CAMERA_NAMES: list[str] = [
    "camera_base_front_center",
    "camera_ring_front",
    "camera_ring_front_left",
    "camera_ring_front_right",
    "camera_ring_rear",
    "camera_ring_rear_left",
    "camera_ring_rear_right",
]

# Total views fed to the model = 7 cameras + 1 map tile.
NUM_VIEWS = 8

_BACKBONE_IMAGE_SIZE = 256

def scale_intrinsic(
    K: np.ndarray,
    original_wh: tuple[int, int],
    target_size: int = _BACKBONE_IMAGE_SIZE,
) -> np.ndarray:
    """Return K rescaled for a square ``target_size`` output.
 
    Args:
        K: (3, 3) pinhole camera matrix at the camera's native resolution.
        original_wh: (width, height) of the native camera image.
        target_size: Square side length after the backbone preprocessing
            transform. Defaults to ``_BACKBONE_IMAGE_SIZE`` (256).
 
    Returns:
        (3, 3) float64 array with ``fx``, ``fy``, ``cx``, ``cy`` scaled.
    """
    orig_w, orig_h = original_wh
    sx = target_size / orig_w
    sy = target_size / orig_h
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx   # fx
    K_scaled[1, 1] *= sy   # fy
    K_scaled[0, 2] *= sx   # cx
    K_scaled[1, 2] *= sy   # cy
    return K_scaled
 
 
def compute_camera_projection_matrices(
    loader: SensorDataLoader,
    camera_names: list[str] | None = None,
) -> torch.Tensor:
    """Compute ``(3, 4)`` projection matrices for each camera view.
 
    ``P = K_scaled @ T_ref_to_cam`` maps 3-D reference-frame points to
    pixel coordinates in the backbone-resized image.
 
    KIT Scenes ``calib.json`` always provides a ``resolution`` field, so
    ``CameraCalibration.image_size`` is always populated and no image I/O
    is required here.
 
    Args:
        loader: ``SensorDataLoader`` for the scene.
        camera_names: Cameras to compute matrices for, in slot order.
            Defaults to ``CAMERA_NAMES``.
 
    Returns:
        Float32 tensor of shape ``(len(camera_names), 3, 4)``.
        Does not include a slot for the map tile.
    """
    if camera_names is None:
        camera_names = CAMERA_NAMES
 
    matrices = []
    for cam_name in camera_names:
        calib = loader.get_camera_calibration(cam_name)
 
        if calib.image_size is None:
            raise ValueError(
                f"Camera {cam_name!r} has no resolution in calib.json. "
                "KIT Scenes calibration files are expected to always include "
                "a resolution field."
            )
 
        K_scaled = scale_intrinsic(calib.intrinsic, calib.image_size)
 
        # invert calib.extrinsic to get T_ref_to_cam.
        T_ref_to_cam = np.linalg.inv(calib.extrinsic)   # (4, 4)
        P = K_scaled @ T_ref_to_cam[:3, :]              # (3, 4)
        matrices.append(P)
 
    return torch.tensor(np.stack(matrices, axis=0), dtype=torch.float32)  # (V, 3, 4)


def load_camera_frame(
    loader: SensorDataLoader,
    frame_idx: int,
    transform: Compose,
    ego_xy: np.ndarray,
    ego_yaw: float = 0.0,
    camera_names: list[str] | None = None,
) -> torch.Tensor:
    """Load and preprocess the camera views at a single reference frame.

    KIT Scenes cameras and ego poses share the 10 Hz reference timeline, so
    ``frame_idx`` indexes both directly.

    The 8th tile (slot 7) is a semantic BEV map rasterised from the scene's
    Lanelet2 HD map, centred on the ego vehicle and rotated so the ego heading
    always points straight up. It is passed through the same backbone transform
    as the camera frames so all 8 views share the same shape and normalisation.

    Args:
        loader: ``SensorDataLoader`` for the scene, supplied by the dataset so
            its per-scene caches are reused across __getitem__ calls.
        frame_idx: Index into the scene's reference timeline.
        transform: Backbone preprocessing transform (resize + normalise).
        ego_xy: (2,) map-local position [x, y] in metres at this frame.
        ego_yaw: Ego heading in map frame (radians, Z-up). Rotates the BEV tile
            so the ego's heading always points straight up in the image.
        camera_names: Ordered list of camera directory names to load.
            Defaults to ``CAMERA_NAMES``.

    Returns:
        Float tensor of shape (8, 3, H, W):
        7 camera views followed by 1 semantic BEV map tile.
    """
    if camera_names is None:
        camera_names = CAMERA_NAMES

    camera_tensors = []
    for cam_name in camera_names:
        rgb_frame = loader.get_camera_image(cam_name, frame_idx)  # (H, W, 3) RGB
        camera_tensors.append(transform(Image.fromarray(rgb_frame)))  # (3, H, W)

    # Slot 7: semantic BEV map. generate_bev_map_tile returns (H, W, 3).
    # Passing through transform (PIL path) gives identical (3, H, W) float
    # normalisation as the camera tiles. Falls back to zeros on failure.
    bev_rgb = generate_bev_map_tile(
        scene_path=loader.scene_path,
        ego_x=float(ego_xy[0]),
        ego_y=float(ego_xy[1]),
        ego_yaw=float(ego_yaw),
    )
    if bev_rgb is None:
        map_tile = torch.zeros_like(camera_tensors[0])  # (3, H, W)
    else:
        map_tile = transform(Image.fromarray(bev_rgb))  # (3, H, W)
    camera_tensors.append(map_tile)

    return torch.stack(camera_tensors, dim=0)  # (8, 3, H, W)