"""Camera frame loading for the KIT Scenes Multimodal dataset.

KIT Scenes stores per-frame JPEGs on disk (not videos), already at the 10 Hz
reference timeline, so a single ``frame_idx`` indexes every camera and the ego
poses alike. The ``kitscenes`` SDK's ``SensorDataLoader`` decodes a frame to an
RGB ``np.ndarray``; this module resizes/normalises it for the AutoE2E backbone
and stacks the 7 camera views into the tensor the model expects.

Camera projection matrices are computed from KITScenes calibration files, with
intrinsics scaled to match the backbone's actual resize/crop transform.
"""

from __future__ import annotations

import numpy as np
import torch
from kitscenes.sensors import SensorDataLoader
from PIL import Image
from torchvision.transforms import Compose

# Shared, dataset-agnostic intrinsic scaling (re-exported for backward compat).
from ..calibration import scale_intrinsic

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

# Total views fed to the model = 7 cameras.
NUM_VIEWS = 7

def compute_camera_projection_matrices(
    loader: SensorDataLoader,
    transform: Compose,
    camera_names: list[str] | None = None,
) -> torch.Tensor:
    """Compute ``(3, 4)`` projection matrices for each camera view.
 
    ``P = K_scaled @ T_ref_to_cam`` maps 3-D reference-frame points to
    pixel coordinates in the backbone-resized image.
 
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
        K_scaled = scale_intrinsic(calib.intrinsic, calib.image_size, transform)
 
        # invert calib.extrinsic to get T_ref_to_cam.
        T_ref_to_cam = np.linalg.inv(calib.extrinsic)   # (4, 4)
        P = K_scaled @ T_ref_to_cam[:3, :]              # (3, 4)
        matrices.append(P)
 
    return torch.tensor(np.stack(matrices, axis=0), dtype=torch.float32)  # (V, 3, 4)


def load_camera_frame(
    loader: SensorDataLoader,
    frame_idx: int,
    transform: Compose,
    camera_names: list[str] | None = None,
) -> torch.Tensor:
    """Load and preprocess the camera views at a single reference frame.

    Args:
        loader: ``SensorDataLoader`` for the scene, supplied by the dataset so
            its per-scene caches are reused across __getitem__ calls.
        frame_idx: Index into the scene's reference timeline.
        transform: Backbone preprocessing transform (resize + normalise).
        camera_names: Ordered list of camera directory names to load.
            Defaults to ``CAMERA_NAMES``.

    Returns:
        Float tensor of shape (7, 3, H, W):
        7 camera views.
    """
    if camera_names is None:
        camera_names = CAMERA_NAMES

    camera_tensors = []
    for cam_name in camera_names:
        rgb_frame = loader.get_camera_image(cam_name, frame_idx)  # (H, W, 3) RGB
        camera_tensors.append(transform(Image.fromarray(rgb_frame)))  # (3, H, W)

    return torch.stack(camera_tensors, dim=0)  # (7, 3, H, W)