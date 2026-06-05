"""Camera frame loading for the NVIDIA PhysicalAI-Autonomous-Vehicles dataset.

TODO: The NVIDIA dataset does not include rendered map tiles. The 8th view is
currently a zero tensor of shape (3, H, W). Replace ``_make_map_tile``
with a real renderer once one is available.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from physical_ai_av.video import SeekVideoReader
from torchvision.transforms import Compose

# Camera directories present in the NVIDIA PhysicalAI-Autonomous-Vehicles dataset.
CAMERA_NAMES: list[str] = [
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
]

# Total views fed to the model = 7 cameras + 1 map tile.
NUM_VIEWS = 8


def _make_map_tile(transform: Compose) -> torch.Tensor:
    """Return a placeholder map tile matching the transform output shape.

    TODO: Replace with a real renderer once a map source is integrated.
          The tile should pass through the same transform as camera frames.
    """
    dummy = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    return transform(dummy)


def _egomotion_ts_to_frame_idx(timestamps_path: Path, egomotion_timestamp_us: int) -> int:
    """Find the camera frame index closest to an egomotion timestamp.

    Both egomotion and camera timestamps are in microseconds relative to the
    same clip anchor (t=0). This finds the camera frame whose timestamp is
    nearest to the egomotion timestamp at the sample point.

    Args:
        timestamps_path: Path to the sidecar timestamps parquet for this camera.
        egomotion_timestamp_us: Egomotion timestamp in microseconds at the
            desired sample point, read directly from the egomotion parquet.

    Returns:
        0-based frame index into the video.
    """
    df = pd.read_parquet(timestamps_path)
    camera_timestamps_us = df["timestamp"].to_numpy()
    return int(np.argmin(np.abs(camera_timestamps_us - egomotion_timestamp_us)))


def load_camera_frame(
    data_root: Path | str,
    clip_uuid: str,
    egomotion_timestamp_us: int,
    transform: Compose,
    camera_names: list[str] | None = None,
) -> torch.Tensor:
    """Load and preprocess the camera frame aligned to an egomotion timestamp.

    Args:
        data_root: Root directory of the dataset subset.
        clip_uuid: UUID of the clip to load.
        egomotion_timestamp_us: Egomotion timestamp in microseconds at the
            desired sample point, read directly from the egomotion parquet.
        camera_names: Ordered list of camera directory names to load.
            Defaults to ``CAMERA_NAMES``.

    Returns:
        Float tensor of shape (8, 3, 224, 224):
        7 camera views followed by 1 map tile (currently zeros).
    """
    data_root = Path(data_root)
    camera_root = data_root / "camera"

    if not camera_root.exists():
        raise FileNotFoundError(f"Camera directory not found: {camera_root}")

    if camera_names is None:
        camera_names = CAMERA_NAMES

    camera_tensors = []

    for cam_name in camera_names:
        cam_dir = camera_root / cam_name
        video_path = cam_dir / f"{clip_uuid}.{cam_name}.mp4"
        timestamps_path = cam_dir / f"{clip_uuid}.{cam_name}.timestamps.parquet"

        if not video_path.exists():
            raise FileNotFoundError(f"Camera video not found: {video_path}")

        if not timestamps_path.exists():
            raise FileNotFoundError(
                f"Camera timestamps parquet not found: {timestamps_path}. "
                "Cannot align camera frame to egomotion timestamp without it."
            )

        frame_idx = _egomotion_ts_to_frame_idx(timestamps_path, egomotion_timestamp_us)

        video_data = io.BytesIO(video_path.read_bytes())
        reader = SeekVideoReader(video_data=video_data)
        try:
            indices = np.array([frame_idx], dtype=np.int64)
            rgb_frames = reader.decode_images_from_frame_indices(indices)
        finally:
            reader.close()

        pil_frame = Image.fromarray(rgb_frames[0])
        camera_tensors.append(transform(pil_frame))  # (3, 224, 224)

    camera_tensors.append(_make_map_tile(transform))

    return torch.stack(camera_tensors, dim=0)  # (8, 3, 224, 224)