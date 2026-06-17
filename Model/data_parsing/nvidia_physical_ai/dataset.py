"""PyTorch Dataset for the NVIDIA PhysicalAI-Autonomous-Vehicles dataset.

Usage
-----
    from data_parsing.nvidia_physical_ai import NvidiaAVDataset

    # All valid samples across all clips (for training)
    dataset = NvidiaAVDataset(data_root="/path/to/nvidia_av_camera_subset")

    # Single clip (for smoke tests / forward pass validation)
    dataset = NvidiaAVDataset(
        data_root="/path/to/nvidia_av_camera_subset",
        clip_uuids=["fd1d1b6b-59bf-4292-8295-5028aa6aa5e3"],
    )

    sample = dataset[0]
    # sample["visual_tiles"]       (8, 3, 256, 256)
    # sample["visual_history"]     (896,)
    # sample["egomotion_history"]  (256,)
    # sample["trajectory_target"]  (128,)
    # sample["clip_uuid"]          str
    # sample["sample_idx"]         int
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import timm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .camera import CAMERA_NAMES, load_camera_frame
from .egomotion import (
    _EGOMOTION_COLUMNS,
    MIN_ROWS,
    _DOWNSAMPLE_STEP,
    _FUTURE_TIMESTEPS,
    _HISTORY_TIMESTEPS,
    load_egomotion,
)

logger = logging.getLogger(__name__)

_DISCOVERY_CAMERA = "camera_front_wide_120fov"

# 64 frames × 14-dim compressed scene memory; matches the trajectory
# planner's visual_history_proj input dim. Currently a zero placeholder
# until a rolling-buffer encoder writes real summaries each step.
_VISUAL_HISTORY_DIM = 896


class ClipSample(TypedDict):
    visual_tiles: torch.Tensor        # (8, 3, 256, 256)
    visual_history: torch.Tensor      # (896,)
    egomotion_history: torch.Tensor   # (256,)
    trajectory_target: torch.Tensor   # (128,)
    clip_uuid: str
    sample_idx: int


class NvidiaAVDataset(Dataset):
    """Dataset where each item is one valid (clip_uuid, sample_idx) pair.

    All valid sample indices across all clips are enumerated at construction
    time. __getitem__ does only I/O — no index arithmetic at call time.

    Args:
        data_root: Path to the subset directory.
        camera_names: Camera views to load. Defaults to ``CAMERA_NAMES``.
        clip_uuids: Optional explicit list of clip UUIDs. If ``None``, all
            valid clips are discovered automatically. Pass a single-element
            list for smoke tests or forward pass validation.
    """

    def __init__(
        self,
        data_root: Path | str,
        backbone_name: str = "swinv2_tiny_window8_256",
        camera_names: list[str] | None = None,
        clip_uuids: list[str] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.camera_names = camera_names or CAMERA_NAMES

        # Build the image transform from the backbone's own config so that
        # preprocessing always matches what the backbone expects.
        # create_model loads config only — no pretrained weights downloaded here.
        _backbone = timm.create_model(backbone_name, pretrained=False)
        data_config = timm.data.resolve_model_data_config(_backbone)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        del _backbone

        clips = clip_uuids if clip_uuids is not None else self._discover_clip_uuids()
        if not clips:
            raise ValueError(
                f"No valid clips found under: {self.data_root / 'camera' / _DISCOVERY_CAMERA}"
            )
        
        self._egomotion_dfs: dict[str, pd.DataFrame] = {}
        self._camera_timestamps: dict[tuple[str, str], np.ndarray] = {}

        # Build the flat sample index: list of (clip_uuid, sample_idx, egomotion_timestamp_us).
        # Precomputing this means __getitem__ never touches pandas.
        self._samples: list[tuple[str, int, int]] = []
        for clip_uuid in clips:
            samples = self._valid_samples_for_clip(clip_uuid)
            if samples:
                self._samples.extend(samples)
                self._load_camera_timestamps(clip_uuid)

        if not self._samples:
            raise ValueError("No valid samples found across all clips.")

        logger.info(
            "NvidiaAVDataset: %d samples from %d clips", len(self._samples), len(clips)
        )

    def _load_camera_timestamps(self, clip_uuid: str) -> None:
        """Load and cache camera timestamp arrays for all cameras in a clip."""
        for cam_name in self.camera_names:
            timestamps_path = (
                self.data_root / "camera" / cam_name
                / f"{clip_uuid}.{cam_name}.timestamps.parquet"
            )
 
            self._camera_timestamps[(clip_uuid, cam_name)] = (
                pd.read_parquet(timestamps_path)["timestamp"].to_numpy()
            )

    def _discover_clip_uuids(self) -> list[str]:
        """Scan the reference camera directory for clip UUIDs."""
        discovery_dir = self.data_root / "camera" / _DISCOVERY_CAMERA
        if not discovery_dir.exists():
            raise FileNotFoundError(
                f"Reference camera directory not found: {discovery_dir}"
            )
        return sorted(p.name.split(".")[0] for p in discovery_dir.glob("*.mp4"))

    def _validate_clip(self, clip_uuid: str) -> bool:
        """Validate a clip before adding it to the sample index.

        Checks:
        - All expected camera video files and timestamp parquets exist.

        Returns True if the clip passes all checks, False otherwise.
        """

        # Check camera file completeness
        for cam_name in self.camera_names:
            cam_dir = self.data_root / "camera" / cam_name
            video_path = cam_dir / f"{clip_uuid}.{cam_name}.mp4"
            timestamps_path = cam_dir / f"{clip_uuid}.{cam_name}.timestamps.parquet"

            if not video_path.exists():
                logger.warning(
                    "Clip %s: missing video file for camera %s. Skipping.",
                    clip_uuid, cam_name,
                )
                return False

            if not timestamps_path.exists():
                logger.warning(
                    "Clip %s: missing timestamps parquet for camera %s. Skipping.",
                    clip_uuid, cam_name,
                )
                return False

        return True


    def _validate_egomotion_timestamps(self, clip_uuid: str, df_ds: pd.DataFrame) -> bool:
        """Check that downsampled egomotion timestamps are strictly monotonically increasing."""
        timestamps = df_ds["timestamp"].to_numpy()
        if not np.all(np.diff(timestamps) > 0):
            logger.warning(
                "Clip %s: egomotion timestamps are not strictly monotonically increasing. Skipping.",
                clip_uuid,
            )
            return False
        return True

    def _valid_samples_for_clip(
        self, clip_uuid: str
    ) -> list[tuple[str, int, int]]:
        """Return all valid (clip_uuid, sample_idx, egomotion_timestamp_us) for one clip.
        Also checks whether all required columns are present in the parquet.

        A sample_idx is valid when there are _HISTORY_TIMESTEPS rows behind it
        and _FUTURE_TIMESTEPS rows ahead of it in the downsampled sequence.
        """

        if not self._validate_clip(clip_uuid):
            return []
        
        parquet_path = (
            self.data_root / "labels" / "egomotion" / f"{clip_uuid}.egomotion.parquet"
        )
        if not parquet_path.exists():
            logger.warning("Egomotion parquet missing for clip %s, skipping.", clip_uuid)
            return []

        # Checking for required columns
        try:
            df = pd.read_parquet(parquet_path, columns=_EGOMOTION_COLUMNS)
        except KeyError as e:
            logger.warning(
                "Clip %s: egomotion parquet missing columns: %s. Skipping.", clip_uuid, e
            )
            return []
        except Exception as e:
            logger.warning(
                "Clip %s: failed to read egomotion parquet: %s. Skipping.", clip_uuid, e
            )
            return []
        
        df_ds = df.iloc[::_DOWNSAMPLE_STEP].reset_index(drop=True)

        if len(df_ds) < MIN_ROWS:
            logger.warning(
                "Clip %s has only %d rows after downsampling (need %d), skipping.",
                clip_uuid, len(df_ds), MIN_ROWS,
            )
            return []
        
        if not self._validate_egomotion_timestamps(clip_uuid, df_ds):
            return []
        
        self._egomotion_dfs[clip_uuid] = df_ds  # cache the downsampled df for later use in __getitem__

        min_idx = _HISTORY_TIMESTEPS           # first valid sample_idx
        max_idx = len(df_ds) - _FUTURE_TIMESTEPS  - 1   # last valid sample_idx (inclusive) corrected by -1

        return [
            (clip_uuid, sample_idx, int(df_ds.iloc[sample_idx]["timestamp"]))
            for sample_idx in range(min_idx, max_idx + 1)
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> ClipSample:
        clip_uuid, sample_idx, egomotion_timestamp_us = self._samples[idx]

        camera_timestamps = {
            cam_name: self._camera_timestamps[(clip_uuid, cam_name)]
            for cam_name in self.camera_names
        }

        visual_tiles = load_camera_frame(
            self.data_root,
            clip_uuid,
            egomotion_timestamp_us=egomotion_timestamp_us,
            transform=self.transform,
            camera_names=self.camera_names,
            camera_timestamps=camera_timestamps,
        )

        egomotion_history, trajectory_target = load_egomotion(
            self.data_root,
            clip_uuid,
            sample_idx=sample_idx,
            df=self._egomotion_dfs[clip_uuid],
        )

        return ClipSample(
            visual_tiles=visual_tiles,
            visual_history=torch.zeros(_VISUAL_HISTORY_DIM),
            egomotion_history=egomotion_history,
            trajectory_target=trajectory_target,
            clip_uuid=clip_uuid,
            sample_idx=sample_idx,
        )
