"""PyTorch Dataset for the yaak-ai/L2D LeRobot dataset.

Usage
-----
    from data_parsing.l2d import L2DDataset

    dataset = L2DDataset(repo_id="yaak-ai/L2D")
    sample = dataset[0]
    # sample["visual_tiles"]       (6, 3, 256, 256)  6 real cameras
    # sample["map_tile"]           (3, 256, 256)     BEV nav-map (separate branch)
    # sample["egomotion_history"]  (256,)
    # sample["visual_history"]     (896,)
    # sample["trajectory_target"]  (128,)
    # sample["episode_index"]      int
    # sample["frame_index"]        int
"""

from __future__ import annotations

import logging
from typing import TypedDict

import timm
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import numpy as np

from .camera import CAMERA_NAMES, MAP_VIEW_NAME
from .egomotion import (
    MIN_FRAMES,
    _FUTURE_TIMESTEPS,
    _HISTORY_TIMESTEPS,
    extract_egomotion,
)

logger = logging.getLogger(__name__)

_VISUAL_HISTORY_DIM = 896


class L2DSample(TypedDict):
    visual_tiles: torch.Tensor       # (6, 3, H, W) — 6 real cameras
    map_tile: torch.Tensor           # (3, H, W) — BEV nav-map (map branch)
    egomotion_history: torch.Tensor  # (256,)
    visual_history: torch.Tensor     # (896,)
    trajectory_target: torch.Tensor  # (128,)
    episode_index: int
    frame_index: int


class L2DDataset(Dataset):
    """Dataset wrapping the yaak-ai/L2D LeRobotDataset.

    Each item is one valid frame from an episode, where sufficient past and
    future context exists for egomotion extraction.

    Args:
        repo_id: HuggingFace repo ID for the dataset.
        episodes: Optional list of episode indices to load. If None, all
            episodes are used.
        backbone_name: timm backbone for deriving image transforms.
        local_files_only: Accepted for backward compatibility; lerobot 0.5.x
            removed this option (it now reads from cache by default), so the
            flag is currently a no-op.
    """

    def __init__(
        self,
        repo_id: str = "yaak-ai/L2D",
        episodes: list[int] | None = None,
        backbone_name: str = "swinv2_tiny_window8_256",
        local_files_only: bool = False,
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError:
            # lerobot-dataset (relaxed-deps standalone) exposes the same class
            # under the `ledataset` namespace.
            from ledataset.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self._episodes = episodes

        # lerobot 0.5.x removed `local_files_only`; it now syncs from cache by
        # default and only re-fetches when `force_cache_sync=True`. We map the
        # legacy flag onto that: local_files_only=True means "don't force a
        # remote sync", which is already the default, so it is simply not passed.
        self.lerobot_dataset = LeRobotDataset(
            repo_id=repo_id,
            episodes=episodes,
        )

        _backbone = timm.create_model(backbone_name, pretrained=False)
        data_config = timm.data.resolve_model_data_config(_backbone)
        self._input_size = data_config["input_size"][1:]  # (H, W)
        self._mean = torch.tensor(data_config["mean"]).view(3, 1, 1)
        self._std = torch.tensor(data_config["std"]).view(3, 1, 1)
        del _backbone

        self._episode_ranges = self._episode_local_ranges()
        self._samples = self._build_sample_index()

        if not self._samples:
            raise ValueError("No valid samples found in the dataset.")

        logger.info("L2DDataset: %d samples", len(self._samples))

    def _episode_local_ranges(self) -> dict[int, tuple[int, int]]:
        """Map each episode to its [start, end) row range in ``hf_dataset``.

        Everything downstream indexes ``hf_dataset`` / ``lerobot_dataset``,
        which are local (0-based) to the loaded subset — when ``episodes`` is a
        subset, row 0 is the first frame of the first requested episode, not a
        global frame. We derive ranges from the ``episode_index`` column rather
        than ``meta.episodes`` (whose ``dataset_from_index`` stays global, so it
        would be off by the subset offset). Local rows are what every accessor
        below actually uses.
        """
        hf = self.lerobot_dataset.hf_dataset
        ep_col = np.asarray(hf["episode_index"])

        ranges: dict[int, tuple[int, int]] = {}
        for ep_idx in np.unique(ep_col):
            rows = np.nonzero(ep_col == ep_idx)[0]
            ranges[int(ep_idx)] = (int(rows[0]), int(rows[-1]) + 1)
        return ranges

    def _build_sample_index(self) -> list[tuple[int, int]]:
        """Enumerate all valid (episode_index, local_frame_idx) pairs.

        A frame is valid when there are _HISTORY_TIMESTEPS frames before it
        and _FUTURE_TIMESTEPS frames after it within the same episode. Indices
        are local rows into ``hf_dataset`` / ``lerobot_dataset``.
        """
        samples = []

        for ep_idx, (ep_start, ep_end) in sorted(self._episode_ranges.items()):
            ep_len = ep_end - ep_start

            if ep_len < MIN_FRAMES:
                continue

            min_frame = _HISTORY_TIMESTEPS
            max_frame = ep_len - _FUTURE_TIMESTEPS - 1

            for frame_idx in range(min_frame, max_frame + 1):
                samples.append((ep_idx, ep_start + frame_idx))

        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def _get_vehicle_states_window(self, ep_start: int, ep_end: int) -> np.ndarray:
        """Load vehicle state vectors for one episode (local row range).

        Reads directly from the underlying ``hf_dataset`` numeric table instead
        of indexing ``lerobot_dataset[i]``. The latter decodes all 7 camera
        videos per frame, which made this ~35s per sample; the vehicle state we
        need here is just an 8-dim vector, so we skip video decoding entirely.
        """
        hf = self.lerobot_dataset.hf_dataset
        col = hf.select_columns(["observation.state.vehicle"])
        states = np.asarray(
            col[ep_start:ep_end]["observation.state.vehicle"], dtype=np.float32
        )
        return states

    def __getitem__(self, idx: int) -> L2DSample:
        # row is the local index into hf_dataset / lerobot_dataset.
        ep_idx, row = self._samples[idx]
        ep_start, ep_end = self._episode_ranges[ep_idx]

        # Offset of the current frame within its own episode.
        sample_idx_in_episode = row - ep_start

        # Load vehicle states for egomotion (episode window, no video decode)
        vehicle_states = self._get_vehicle_states_window(ep_start, ep_end)
        egomotion_history, trajectory_target = extract_egomotion(
            vehicle_states, sample_idx=sample_idx_in_episode
        )

        # Load camera frames for the current timestep (decodes video)
        item = self.lerobot_dataset[row]

        def _prep(frame: torch.Tensor) -> torch.Tensor:
            frame = TF.resize(frame, list(self._input_size), antialias=True)
            return TF.normalize(frame, self._mean.squeeze(), self._std.squeeze())

        # 6 real cameras -> visual_tiles (BEV projection applies to these).
        tensors = [_prep(item[cam_name]) for cam_name in CAMERA_NAMES]
        visual_tiles = torch.stack(tensors, dim=0)

        # BEV nav-map view -> map_tile (routed to the separate map branch).
        map_tile = _prep(item[MAP_VIEW_NAME])

        visual_history = torch.zeros(_VISUAL_HISTORY_DIM, dtype=torch.float32)

        return L2DSample(
            visual_tiles=visual_tiles,
            map_tile=map_tile,
            egomotion_history=egomotion_history,
            visual_history=visual_history,
            trajectory_target=trajectory_target,
            episode_index=ep_idx,
            frame_index=sample_idx_in_episode,
        )
