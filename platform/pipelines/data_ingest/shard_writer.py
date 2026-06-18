"""Shard packing: bundles extracted samples into WebDataset .tar files."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np

# Target shard size ~100MB; actual varies based on JPEG sizes
SHARD_MAX_BYTES = 100 * 1024 * 1024


class ShardWriter:
    """Writes samples into WebDataset tar shards, splitting at size threshold."""

    def __init__(self, output_dir: Path, prefix: str = "train"):
        self.output_dir = output_dir
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._shard_idx = 0
        self._current_tar: tarfile.TarFile | None = None
        self._current_size = 0
        self._sample_count = 0
        self._shard_paths: list[Path] = []
        self._open_shard()

    def _shard_path(self) -> Path:
        return self.output_dir / f"{self.prefix}-{self._shard_idx:06d}.tar"

    def _open_shard(self):
        if self._current_tar:
            self._current_tar.close()
        path = self._shard_path()
        self._current_tar = tarfile.open(path, "w")
        self._current_size = 0
        self._shard_paths.append(path)

    def _maybe_rotate(self):
        if self._current_size >= SHARD_MAX_BYTES:
            self._current_tar.close()
            self._shard_idx += 1
            self._open_shard()

    def add_sample(
        self,
        sample_id: str,
        camera_jpegs: list[bytes],
        ego_history: np.ndarray,
        ego_future: np.ndarray,
        metadata: dict,
    ):
        """Add one training sample (all cameras + ego) to current shard."""
        # Camera frames
        for i, jpeg_bytes in enumerate(camera_jpegs):
            self._add_file(f"{sample_id}.cam_{i}.jpg", jpeg_bytes)

        # Ego as concatenated history+future numpy array
        ego_combined = np.concatenate([ego_history, ego_future], axis=0)
        ego_bytes = ego_combined.astype(np.float32).tobytes()
        self._add_file(f"{sample_id}.ego.npy", ego_bytes)

        # Metadata
        meta_bytes = json.dumps(metadata).encode()
        self._add_file(f"{sample_id}.meta.json", meta_bytes)

        self._sample_count += 1
        self._maybe_rotate()

    def _add_file(self, name: str, data: bytes):
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        self._current_tar.addfile(info, io.BytesIO(data))
        self._current_size += len(data)

    def close(self) -> list[Path]:
        """Finalize all shards. Returns list of shard paths."""
        if self._current_tar:
            self._current_tar.close()
            self._current_tar = None
        return self._shard_paths

    @property
    def total_samples(self) -> int:
        return self._sample_count
