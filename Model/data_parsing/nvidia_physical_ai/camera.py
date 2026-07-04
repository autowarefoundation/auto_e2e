"""Camera frame loading for the NVIDIA PhysicalAI-Autonomous-Vehicles dataset.

The dataset provides 7 real cameras. It has NO rendered map tile, so the map
branch receives a zero tensor (``make_map_tile``) until a renderer is
integrated. The map is NOT a camera view: it is kept out of ``visual_tiles`` (so
it never enters the camera BEV projection) and routed to the model's separate
map branch. Hence ``NUM_VIEWS = 7``.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from physical_ai_av.video import SeekVideoReader

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

# Real camera views fed to the BEV projection (the map is separate, not a view).
NUM_VIEWS = 7


def make_map_tile(reference: torch.Tensor) -> torch.Tensor:
    """Return a zero map tile matching the shape/dtype of one raw camera frame.

    NVIDIA has no rendered nav-map, so the map branch receives zeros for now.
    Shaped like a single (3, H, W) frame for the model's map_input.

    TODO: Replace with a real renderer once a map source is integrated.
    """
    return torch.zeros_like(reference)


def _egomotion_ts_to_frame_idx(
    egomotion_timestamp_us: int,
    camera_timestamps_us: np.ndarray,
) -> int:
    """Find the camera frame index closest to an egomotion timestamp.

    Both egomotion and camera timestamps are in microseconds relative to the
    same clip anchor (t=0). This finds the camera frame whose timestamp is
    nearest to the egomotion timestamp at the sample point.

    Args:
        egomotion_timestamp_us: Egomotion timestamp in microseconds at the
            desired sample point, read directly from the egomotion parquet.
        camera_timestamps_us: Pre-loaded array of camera timestamps in
            microseconds for this clip+camera. Pass this from NvidiaAVDataset
            to avoid re-reading the timestamps parquet on every __getitem__.

    Returns:
        0-based frame index into the video.
    """
    return int(np.argmin(np.abs(camera_timestamps_us - egomotion_timestamp_us)))

def load_camera_frame(
    data_root: Path | str,
    clip_uuid: str,
    egomotion_timestamp_us: int,
    camera_names: list[str] | None = None,
    camera_timestamps: dict[str, np.ndarray] | None = None,
) -> torch.Tensor:
    """Load the RAW camera frame aligned to an egomotion timestamp.

    Returns the decoded frame as an unmodified uint8 CHW tensor — no resize,
    crop or normalize. The dataset is a pre-extraction source: the shard packer
    owns the single, explicit, geometry-aware resize and the loader owns the
    single normalize, so the projection ABI targets a known frame (#77).

    Args:
        data_root: Root directory of the dataset subset.
        clip_uuid: UUID of the clip to load.
        egomotion_timestamp_us: Egomotion timestamp in microseconds at the
            desired sample point, read directly from the egomotion parquet.
        camera_names: Ordered list of camera directory names to load.
            Defaults to ``CAMERA_NAMES``.

    Returns:
        uint8 tensor of shape (7, 3, H, W): the 7 real camera views. The nav-map
        is not included here; see ``make_map_tile``.
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

        if not video_path.exists():
            raise FileNotFoundError(f"Camera video not found: {video_path}")
        
        if camera_timestamps is not None:
            timestamps_us = camera_timestamps[cam_name]
        else:
            timestamps_path = cam_dir / f"{clip_uuid}.{cam_name}.timestamps.parquet"
            if not timestamps_path.exists():
                raise FileNotFoundError(
                    f"Camera timestamps parquet not found: {timestamps_path}. "
                    "Cannot align camera frame to egomotion timestamp without it."
                )
            timestamps_us = pd.read_parquet(timestamps_path)["timestamp"].to_numpy()

        frame_idx = _egomotion_ts_to_frame_idx(egomotion_timestamp_us, timestamps_us)

        video_data = io.BytesIO(video_path.read_bytes()) #TODO: major bottleneck for training - consider sampling images in a seperate data processing step.
        reader = SeekVideoReader(video_data=video_data)
        try:
            indices = np.array([frame_idx], dtype=np.int64)
            rgb_frames = reader.decode_images_from_frame_indices(indices)
        finally:
            reader.close()

        # RAW uint8 CHW, no preprocessing (see module/function docstring).
        frame = torch.from_numpy(rgb_frames[0]).permute(2, 0, 1).contiguous()
        camera_tensors.append(frame)

    return torch.stack(camera_tensors, dim=0)  # (7, 3, H, W) uint8
