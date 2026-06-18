"""NVIDIA PhysicalAI-AV IngestAdapter: ffmpeg-based frame extraction."""

from __future__ import annotations

import io
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .protocol import EpisodeRef, IngestAdapter, SamplePoint

logger = logging.getLogger(__name__)

_HISTORY_TIMESTEPS = 64
_FUTURE_TIMESTEPS = 64
_MIN_ROWS = _HISTORY_TIMESTEPS + _FUTURE_TIMESTEPS + 1
_SOURCE_HZ = 100.0
_TARGET_HZ = 10.0
_DOWNSAMPLE_STEP = int(_SOURCE_HZ / _TARGET_HZ)

_CAMERA_NAMES = [
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
]


class NvidiaAVAdapter(IngestAdapter):
    """Adapter for NVIDIA PhysicalAI-Autonomous-Vehicles dataset."""

    def __init__(self, data_root: str | None = None):
        self.data_root = Path(data_root) if data_root else None

    @property
    def camera_names(self) -> list[str]:
        return _CAMERA_NAMES

    def list_episodes(self, limit: int = 0) -> list[EpisodeRef]:
        """List clip UUIDs from egomotion parquet files."""
        ego_dir = self.data_root / "labels" / "egomotion"
        clips = sorted(p.stem.replace(".egomotion", "") for p in ego_dir.glob("*.egomotion.parquet"))
        if limit > 0:
            clips = clips[:limit]
        return [EpisodeRef(episode_id=uuid) for uuid in clips]

    def download_episode(self, ref: EpisodeRef, work_dir: Path) -> Path:
        """Download one clip via physical_ai_av SDK if not already on disk."""
        clip_dir = self.data_root or work_dir
        ego_path = clip_dir / "labels" / "egomotion" / f"{ref.episode_id}.egomotion.parquet"
        if ego_path.exists():
            return clip_dir

        # Use SDK to download single clip
        from physical_ai_av import PhysicalAIAV
        sdk = PhysicalAIAV()
        sdk.download(clip_uuids=[ref.episode_id], output_dir=str(clip_dir))
        return clip_dir

    def compute_valid_samples(self, episode_path: Path) -> list[SamplePoint]:
        """Find valid sample points from downsampled egomotion."""
        clip_uuid = episode_path.name if episode_path.is_dir() else episode_path.stem
        # Find the egomotion parquet
        ego_path = episode_path / "labels" / "egomotion" / f"{clip_uuid}.egomotion.parquet"
        if not ego_path.exists():
            # Try parent structure
            for p in episode_path.rglob("*.egomotion.parquet"):
                ego_path = p
                clip_uuid = p.stem.replace(".egomotion", "")
                break

        df = pd.read_parquet(ego_path)
        # Downsample 100Hz → 10Hz
        df_10hz = df.iloc[::_DOWNSAMPLE_STEP].reset_index(drop=True)

        if len(df_10hz) < _MIN_ROWS:
            return []

        # Derive signals
        signals = self._derive_signals(df_10hz)

        samples = []
        for i in range(_HISTORY_TIMESTEPS, len(df_10hz) - _FUTURE_TIMESTEPS):
            history = signals[i - _HISTORY_TIMESTEPS:i]
            future = signals[i + 1:i + 1 + _FUTURE_TIMESTEPS]
            timestamp_s = float(df_10hz.iloc[i]["timestamp"]) / 1e6  # us → s
            samples.append(SamplePoint(
                frame_idx=i * _DOWNSAMPLE_STEP,  # original 100Hz row index
                timestamp_s=timestamp_s,
                ego_history=history,
                ego_future=future[:, [1, 3]],  # acceleration, curvature
            ))
        return samples

    def extract_frame(
        self, episode_path: Path, sample: SamplePoint, camera_idx: int
    ) -> bytes:
        """Extract frame via ffmpeg seek to exact timestamp → JPEG 256x256."""
        cam_name = _CAMERA_NAMES[camera_idx]
        # Find video file
        clip_uuid = episode_path.name
        video_path = episode_path / "camera" / cam_name / f"{clip_uuid}.{cam_name}.mp4"
        if not video_path.exists():
            for p in episode_path.rglob(f"*.{cam_name}.mp4"):
                video_path = p
                break

        # ffmpeg: seek to timestamp, extract 1 frame, scale to 256x256, output JPEG
        cmd = [
            "ffmpeg", "-ss", f"{sample.timestamp_s:.4f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", "scale=256:256",
            "-f", "image2", "-c:v", "mjpeg", "-q:v", "2",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {video_path} @{sample.timestamp_s}s")
        return result.stdout

    @staticmethod
    def _derive_signals(df: pd.DataFrame) -> np.ndarray:
        """Derive [speed, acceleration, yaw_angle, curvature] from egomotion df."""
        from scipy.spatial.transform import Rotation
        quats = df[["qx", "qy", "qz", "qw"]].values
        yaw = Rotation.from_quat(quats).as_euler("ZYX")[:, 0]
        speed = np.sqrt(df["vx"].values**2 + df["vy"].values**2)
        accel = df["ax"].values
        curv = df["curvature"].values if "curvature" in df.columns else np.zeros(len(df))
        return np.stack([speed, accel, yaw, curv], axis=1).astype(np.float32)
