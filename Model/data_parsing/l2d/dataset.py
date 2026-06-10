"""PyTorch Dataset for the yaak-ai/L2D LeRobot dataset.

Usage
-----
    from data_parsing.l2d import L2DDataset

    dataset = L2DDataset(repo_id="yaak-ai/L2D")
    sample = dataset[0]
    # sample["visual_tiles"]       (7, 3, 256, 256)
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

from .camera import CAMERA_NAMES, NUM_VIEWS
from .egomotion import (
    MIN_FRAMES,
    _FUTURE_TIMESTEPS,
    _HISTORY_TIMESTEPS,
    extract_egomotion,
)

logger = logging.getLogger(__name__)

_VISUAL_HISTORY_DIM = 896


class L2DSample(TypedDict):
    visual_tiles: torch.Tensor       # (7, 3, H, W)
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
        local_files_only: If True, only use local cache (no downloads).
    """

    def __init__(
        self,
        repo_id: str = "yaak-ai/L2D",
        episodes: list[int] | None = None,
        backbone_name: str = "swinv2_tiny_window8_256",
        local_files_only: bool = False,
    ) -> None:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id

        self.lerobot_dataset = LeRobotDataset(
            repo_id=repo_id,
            episodes=episodes,
            local_files_only=local_files_only,
        )

        _backbone = timm.create_model(backbone_name, pretrained=False)
        data_config = timm.data.resolve_model_data_config(_backbone)
        self._input_size = data_config["input_size"][1:]  # (H, W)
        self._mean = torch.tensor(data_config["mean"]).view(3, 1, 1)
        self._std = torch.tensor(data_config["std"]).view(3, 1, 1)
        del _backbone

        self._samples = self._build_sample_index()

        if not self._samples:
            raise ValueError("No valid samples found in the dataset.")

        logger.info("L2DDataset: %d samples", len(self._samples))

    def _build_sample_index(self) -> list[tuple[int, int]]:
        """Enumerate all valid (episode_index, frame_index) pairs.

        A frame is valid when there are _HISTORY_TIMESTEPS frames before it
        and _FUTURE_TIMESTEPS frames after it within the same episode.
        """
        samples = []
        episode_data_index = self.lerobot_dataset.episode_data_index

        num_episodes = len(episode_data_index["from"])
        for ep_idx in range(num_episodes):
            ep_start = episode_data_index["from"][ep_idx].item()
            ep_end = episode_data_index["to"][ep_idx].item()
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

    def _get_vehicle_states_window(
        self, global_idx: int, ep_start: int, ep_end: int
    ) -> np.ndarray:
        """Load vehicle state vectors for the full episode."""
        states = []
        for i in range(ep_start, ep_end):
            item = self.lerobot_dataset[i]
            states.append(item["observation.state.vehicle"].numpy())
        return np.stack(states, axis=0)

    def __getitem__(self, idx: int) -> L2DSample:
        ep_idx, global_frame_idx = self._samples[idx]

        episode_data_index = self.lerobot_dataset.episode_data_index
        ep_start = episode_data_index["from"][ep_idx].item()
        ep_end = episode_data_index["to"][ep_idx].item()

        local_frame_idx = global_frame_idx - ep_start

        # Load vehicle states for egomotion
        vehicle_states = self._get_vehicle_states_window(
            global_frame_idx, ep_start, ep_end
        )
        egomotion_history, trajectory_target = extract_egomotion(
            vehicle_states, sample_idx=local_frame_idx
        )

        # Load camera frames for the current timestep
        item = self.lerobot_dataset[global_frame_idx]
        tensors = []
        for cam_name in CAMERA_NAMES:
            frame = item[cam_name]  # CHW float [0,1]
            frame = TF.resize(frame, list(self._input_size), antialias=True)
            frame = TF.normalize(frame, self._mean.squeeze(), self._std.squeeze())
            tensors.append(frame)

        visual_tiles = torch.stack(tensors, dim=0)

        visual_history = torch.zeros(_VISUAL_HISTORY_DIM, dtype=torch.float32)

        return L2DSample(
            visual_tiles=visual_tiles,
            egomotion_history=egomotion_history,
            visual_history=visual_history,
            trajectory_target=trajectory_target,
            episode_index=ep_idx,
            frame_index=local_frame_idx,
        )
