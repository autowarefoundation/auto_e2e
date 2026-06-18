"""IngestAdapter protocol: dataset-specific logic for raw → training-ready conversion.

Each adapter implements 5 methods that handle the dataset's unique format.
The common pipeline (shard packing, S3 upload, manifest) is shared across all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class EpisodeRef:
    """Opaque reference to one episode/clip in a raw dataset."""
    episode_id: str
    metadata: dict | None = None


@dataclass
class SamplePoint:
    """A valid training sample: frame index with sufficient history+future."""
    frame_idx: int
    timestamp_s: float
    ego_history: np.ndarray   # (history_steps, 7) float32
    ego_future: np.ndarray    # (future_steps, 7) float32


@runtime_checkable
class IngestAdapter(Protocol):
    """Protocol for dataset-specific ingest logic."""

    @property
    def camera_names(self) -> list[str]:
        """Ordered list of camera names for this dataset."""
        ...

    def list_episodes(self, limit: int = 0) -> list[EpisodeRef]:
        """Enumerate available episodes. limit=0 means all."""
        ...

    def download_episode(self, ref: EpisodeRef, work_dir: Path) -> Path:
        """Download/prepare one episode into work_dir. Returns episode root."""
        ...

    def compute_valid_samples(self, episode_path: Path) -> list[SamplePoint]:
        """Find frames with sufficient history+future context."""
        ...

    def extract_frame(
        self, episode_path: Path, sample: SamplePoint, camera_idx: int
    ) -> bytes:
        """Extract one camera frame as JPEG bytes (256x256)."""
        ...
