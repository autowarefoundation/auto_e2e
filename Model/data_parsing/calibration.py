"""Dataset-agnostic camera-calibration helpers shared across parsers.

The projection matrices fed to BEV fusion must be expressed in the *model-input*
image coordinate frame, i.e. after the backbone's resize/crop transform. This
module holds the geometry-only utilities (no dataset SDK dependency) that turn a
raw intrinsic ``K`` into one scaled to the model input, so every dataset parser
(KITScenes, NVIDIA, future L2D) builds ``P = K_scaled @ T_ego_to_cam[:3, :]``
consistently.
"""

from __future__ import annotations

import numpy as np
from torchvision.transforms import CenterCrop, Compose, Resize


def scale_intrinsic(
    K: np.ndarray,
    original_wh: tuple[int, int],
    transform: Compose,
) -> np.ndarray:
    """Return ``K`` adjusted for the resize/crop steps in ``transform``.

    Walks the torchvision Compose pipeline and applies the geometric effect of
    each Resize and CenterCrop step to ``K``. Steps that do not touch pixel
    coordinates (Normalize, ToTensor, ColorJitter, ...) are ignored.

    Args:
        K: Camera intrinsic matrix, shape (3, 3).
        original_wh: Original image dimensions as (width, height).
        transform: Backbone preprocessing transform.

    Returns:
        Scaled intrinsic matrix K, shape (3, 3), as float64.
    """
    if K.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {K.shape}")

    cur_w, cur_h = original_wh
    if cur_w <= 0 or cur_h <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({cur_w}, {cur_h})")

    K_out = K.copy().astype(np.float64)

    for t in transform.transforms:
        if isinstance(t, Resize):
            size = t.size
            if isinstance(size, (list, tuple)):
                if len(size) == 1:
                    size = size[0]
                else:
                    # explicit (h, w)
                    scale_x = size[1] / cur_w
                    scale_y = size[0] / cur_h
                    cur_w, cur_h = size[1], size[0]
                    K_out[0, 0] *= scale_x
                    K_out[1, 1] *= scale_y
                    K_out[0, 2] *= scale_x
                    K_out[1, 2] *= scale_y
                    continue
            # resize with shortest-edge mode (see timm.data.transforms)
            scale = size / min(cur_h, cur_w)
            cur_w = int(cur_w * scale + 0.5)
            cur_h = int(cur_h * scale + 0.5)
            K_out[0, 0] *= scale
            K_out[1, 1] *= scale
            K_out[0, 2] *= scale
            K_out[1, 2] *= scale

        elif isinstance(t, CenterCrop):
            size = t.size
            crop_h, crop_w = (size, size) if isinstance(size, int) else size
            offset_x = (cur_w - crop_w) / 2.0
            offset_y = (cur_h - crop_h) / 2.0
            K_out[0, 2] -= offset_x
            K_out[1, 2] -= offset_y
            cur_w, cur_h = crop_w, crop_h

    return K_out.astype(np.float64)
