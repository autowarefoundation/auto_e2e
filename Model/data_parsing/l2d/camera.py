"""Camera frame loading for the yaak-ai/L2D LeRobot dataset.

L2D provides 7 image views stored as MP4 videos in the LeRobot format:
    6 surround cameras + 1 BEV nav-map (640x360).

The BEV map is NOT a camera: it is a raster prior already aligned to the ego/map
frame, so it has no camera projection and must not be fused through the camera
BEV spatial cross-attention. It is routed to the model's separate map branch
(``map_input``) instead. Hence ``CAMERA_NAMES`` lists the 6 real cameras and the
map is exposed separately via ``MAP_VIEW_NAME`` / ``NUM_VIEWS = 6``.

Camera extrinsics are provided in extrinsic_RDF.yaml. Intrinsics are NOT
published for L2D, so a real ego-to-pixel projection cannot be formed from
published data today; L2D therefore uses the calibration-free pseudo geometry
path (tracked separately). See Issue #77.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import Compose

# The 6 real surround cameras (BEV projection applies only to these).
CAMERA_NAMES: list[str] = [
    "observation.images.front_left",
    "observation.images.left_forward",
    "observation.images.right_forward",
    "observation.images.left_backward",
    "observation.images.rear",
    "observation.images.right_backward",
]

# The BEV nav-map view, routed to the map branch (not a camera).
MAP_VIEW_NAME = "observation.images.map"

NUM_VIEWS = 6


def make_camera_params_placeholder() -> torch.Tensor:
    """Return a placeholder camera_params tensor of shape (NUM_VIEWS, 3, 4).

    Uses identity-like projection matrices for the 6 real cameras. This is NOT a
    real projection: L2D has no published intrinsics, so meaningful BEV
    projection is not possible and the pseudo geometry path is used instead. Kept
    as a shape reference for calibration-free code paths.
    """
    params = torch.zeros(NUM_VIEWS, 3, 4, dtype=torch.float32)
    for i in range(NUM_VIEWS):
        params[i, :3, :3] = torch.eye(3)
    return params


def load_camera_frames(
    frames: dict[str, np.ndarray],
    transform: Compose,
) -> torch.Tensor:
    """Transform pre-loaded camera frames into model input tensor.

    Args:
        frames: Dict mapping camera key to HWC uint8 numpy array.
        transform: torchvision transform (from timm backbone config).

    Returns:
        Float tensor of shape (NUM_VIEWS, 3, H, W).
    """
    from PIL import Image

    tensors = []
    for cam_name in CAMERA_NAMES:
        if cam_name in frames:
            pil_img = Image.fromarray(frames[cam_name])
            tensors.append(transform(pil_img))
        else:
            raise KeyError(f"Missing camera frame: {cam_name}")

    return torch.stack(tensors, dim=0)


def load_map_frame(
    frames: dict[str, np.ndarray],
    transform: Compose,
) -> torch.Tensor:
    """Transform the BEV nav-map view into the model's map_input tensor.

    The map is a raster prior (not a camera); it is fused via the separate map
    branch, not the camera BEV projection.

    Returns:
        Float tensor of shape (3, H, W).
    """
    from PIL import Image

    if MAP_VIEW_NAME not in frames:
        raise KeyError(f"Missing map frame: {MAP_VIEW_NAME}")
    return transform(Image.fromarray(frames[MAP_VIEW_NAME]))
