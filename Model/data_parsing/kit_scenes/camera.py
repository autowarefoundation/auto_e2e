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