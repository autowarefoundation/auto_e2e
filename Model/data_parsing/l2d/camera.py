"""Camera frame loading for the yaak-ai/L2D LeRobot dataset.

L2D provides 7 camera views stored as MP4 videos in the LeRobot format:
    6 surround cameras + 1 BEV map (640x360).

Camera extrinsics are provided in extrinsic_RDF.yaml. Intrinsics can be
combined with extrinsics to form a [3,4] projection matrix per view.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import Compose

CAMERA_NAMES: list[str] = [
    "observation.images.front_left",
    "observation.images.left_forward",
    "observation.images.right_forward",
    "observation.images.left_backward",
    "observation.images.rear",
    "observation.images.right_backward",
    "observation.images.map",
]

NUM_VIEWS = 7


def make_camera_params_placeholder() -> torch.Tensor:
    """Return a placeholder camera_params tensor of shape (NUM_VIEWS, 3, 4).

    Uses identity-like projection matrices. Replace with real intrinsic @
    extrinsic once calibration YAML is parsed.

    Note: This is a placeholder only suitable for concat and cross_attn fusion
    modes. BEV fusion mode requires real extrinsics parsed from the
    extrinsic_RDF.yaml calibration file to produce meaningful projections.
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
