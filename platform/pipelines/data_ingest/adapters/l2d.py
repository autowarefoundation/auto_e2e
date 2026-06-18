"""L2D IngestAdapter: wraps existing l2d parser for offline pre-extraction."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .protocol import EpisodeRef, IngestAdapter, SamplePoint

logger = logging.getLogger(__name__)

# Reuse constants from existing parser
_HISTORY_TIMESTEPS = 64
_FUTURE_TIMESTEPS = 64
_MIN_FRAMES = _HISTORY_TIMESTEPS + _FUTURE_TIMESTEPS + 1

_CAMERA_NAMES = [
    "observation.images.front_left",
    "observation.images.left_forward",
    "observation.images.right_forward",
    "observation.images.left_backward",
    "observation.images.rear",
    "observation.images.right_backward",
    "observation.images.map",
]


class L2DAdapter(IngestAdapter):
    """Adapter for yaak-ai/L2D via lerobot."""

    def __init__(self, repo_id: str = "yaak-ai/L2D"):
        self.repo_id = repo_id
        self._dataset = None

    def _ensure_dataset(self):
        if self._dataset is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self._dataset = LeRobotDataset(repo_id=self.repo_id)

    @property
    def camera_names(self) -> list[str]:
        return _CAMERA_NAMES

    def list_episodes(self, limit: int = 0) -> list[EpisodeRef]:
        self._ensure_dataset()
        ep_col = np.asarray(self._dataset.hf_dataset["episode_index"])
        episode_ids = sorted(set(ep_col.tolist()))
        if limit > 0:
            episode_ids = episode_ids[:limit]
        return [EpisodeRef(episode_id=str(e)) for e in episode_ids]

    def download_episode(self, ref: EpisodeRef, work_dir: Path) -> Path:
        """L2D episodes are accessed via lerobot (HF cache). Return episode idx."""
        # lerobot handles caching; work_dir unused for L2D
        return work_dir / ref.episode_id

    def compute_valid_samples(self, episode_path: Path) -> list[SamplePoint]:
        """Compute valid sample points for one episode."""
        self._ensure_dataset()
        ep_idx = int(episode_path.name)
        hf = self._dataset.hf_dataset
        ep_col = np.asarray(hf["episode_index"])
        rows = np.nonzero(ep_col == ep_idx)[0]

        if len(rows) < _MIN_FRAMES:
            return []

        ep_start, ep_end = int(rows[0]), int(rows[-1]) + 1
        ep_len = ep_end - ep_start

        # Load vehicle states for egomotion derivation
        col = hf.select_columns(["observation.state.vehicle"])
        states = np.asarray(
            col[ep_start:ep_end]["observation.state.vehicle"], dtype=np.float32
        )

        # Derive signals (reuse logic from l2d/egomotion.py)
        signals = self._derive_signals(states)

        samples = []
        for i in range(_HISTORY_TIMESTEPS, ep_len - _FUTURE_TIMESTEPS):
            history = signals[i - _HISTORY_TIMESTEPS:i]
            future = signals[i + 1:i + 1 + _FUTURE_TIMESTEPS]
            samples.append(SamplePoint(
                frame_idx=ep_start + i,
                timestamp_s=i * 0.1,
                ego_history=history,
                ego_future=future[:, [1, 3]],  # accel_x, curvature only for target
            ))
        return samples

    def extract_frame(
        self, episode_path: Path, sample: SamplePoint, camera_idx: int
    ) -> bytes:
        """Extract one frame via lerobot video decode, return JPEG bytes."""
        self._ensure_dataset()
        item = self._dataset[sample.frame_idx]
        cam_name = _CAMERA_NAMES[camera_idx]
        frame_tensor = item[cam_name]  # CHW float [0,1]

        # Convert to PIL, resize to 256x256, encode as JPEG
        img = Image.fromarray(
            (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        img = img.resize((256, 256), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    @staticmethod
    def _derive_signals(vehicle_states: np.ndarray) -> np.ndarray:
        """Derive [speed, accel_x, yaw_rate, curvature] from vehicle states."""
        speed = vehicle_states[:, 0]
        heading = np.unwrap(vehicle_states[:, 1])
        accel_x = vehicle_states[:, 6]
        yaw_rate = np.zeros_like(heading)
        yaw_rate[1:] = np.diff(heading) / 0.1
        curvature = np.where(np.abs(speed) > 1e-6, yaw_rate / speed, 0.0)
        return np.stack([speed, accel_x, yaw_rate, curvature], axis=1).astype(np.float32)
