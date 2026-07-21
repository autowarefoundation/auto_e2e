"""Camera frame loading for the NVIDIA PhysicalAI-Autonomous-Vehicles dataset.

The dataset provides 7 real cameras. It has NO rendered map tile, so the map
branch receives a zero tensor (``make_map_tile``) until a renderer is
integrated. The map is NOT a camera view: it is kept out of ``visual_tiles`` (so
it never enters the camera BEV projection) and routed to the model's separate
map branch. Hence ``NUM_VIEWS = 7``.
"""

from __future__ import annotations

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

def load_front_clip(
    data_root: Path | str,
    clip_uuid: str,
    egomotion_timestamps_us: list[int],
    front_cam: str | None = None,
    camera_timestamps_us: np.ndarray | None = None,
) -> list[torch.Tensor]:
    """Decode ONLY the front camera at a list of egomotion timestamps (a clip).

    For offline reasoning labeling the teacher needs a temporal FRONT-camera
    clip (one frame per horizon), not all 7 cameras. This opens the front video
    ONCE and decodes exactly the requested frames (one per timestamp), so a
    5-horizon clip is a single-camera, 5-frame decode instead of 7-cam per frame.

    Args:
        data_root / clip_uuid: locate the video.
        egomotion_timestamps_us: one egomotion timestamp per horizon (0/1/2/…s).
        front_cam: front camera dir name (defaults to CAMERA_NAMES[0]).
        camera_timestamps_us: pre-loaded front-cam timestamp array (optional).

    Returns:
        list of RAW uint8 ``[3, H, W]`` front frames, one per input timestamp.
    """
    data_root = Path(data_root)
    front_cam = front_cam or CAMERA_NAMES[0]
    cam_dir = data_root / "camera" / front_cam
    video_path = cam_dir / f"{clip_uuid}.{front_cam}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Front camera video not found: {video_path}")

    if camera_timestamps_us is None:
        timestamps_path = cam_dir / f"{clip_uuid}.{front_cam}.timestamps.parquet"
        if not timestamps_path.exists():
            raise FileNotFoundError(
                f"Front camera timestamps parquet not found: {timestamps_path}.")
        camera_timestamps_us = pd.read_parquet(timestamps_path)["timestamp"].to_numpy()

    frame_indices = np.array(
        [_egomotion_ts_to_frame_idx(ts, camera_timestamps_us)
         for ts in egomotion_timestamps_us],
        dtype=np.int64,
    )
    # Same fix as load_camera_frame (#116): a plain buffered file handle keeps
    # PyAV's decoding lazy and seek-based instead of eagerly materializing the
    # full encoded clip in memory before decoding.
    with open(video_path, "rb") as video_data:
        reader = SeekVideoReader(video_data=video_data)
        try:
            rgb_frames = reader.decode_images_from_frame_indices(frame_indices)
        finally:
            reader.close()
    # RAW uint8 CHW per frame, no preprocessing.
    return [torch.from_numpy(rgb_frames[i]).permute(2, 0, 1).contiguous()
            for i in range(len(frame_indices))]


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

        # Fix for #116: SeekVideoReader wraps PyAV's av.open(), which accepts
        # any read+seekable file-like object and performs lazy, seek-based
        # reads internally (container.seek() + demux only the packets needed
        # around the target frame's keyframe — see SeekVideoReader's own
        # docstrings: keyframe indexing "is not costly as we are not decoding
        # anything from the container"). The previous
        # `io.BytesIO(video_path.read_bytes())` defeated that entirely by
        # eagerly materializing the FULL encoded clip in memory before a
        # single frame was requested, once per camera per __getitem__ call
        # (7 cameras x full-clip read, every sample). A plain buffered file
        # handle satisfies the same `.seek()` + read interface PyAV needs,
        # so decoding stays lazy: only the bytes needed to reach the target
        # frame's keyframe and decode forward are actually read from disk.
        with open(video_path, "rb") as video_data:
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
