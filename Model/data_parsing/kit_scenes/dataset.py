"""PyTorch Dataset for the KIT Scenes Multimodal dataset.

Usage
-----
    from data_parsing.kit_scenes import KitScenesDataset

    # All valid samples in a split (for training)
    dataset = KitScenesDataset(data_root="/path/to/kitscenes", split="train")

    # Single scene (for smoke tests / forward pass validation)
    dataset = KitScenesDataset(
        data_root="/path/to/kitscenes",
        scene_ids=["<scene-uuid>"],
    )

    sample = dataset[0]
    # sample["visual_tiles"]       (8, 3, H, W)  — 7 cameras + BEV map tile
    # sample["egomotion_history"]  (256,)
    # sample["visual_history"]     (896,)
    # sample["trajectory_target"]  (128,)
    # sample["scene_id"]           str
    # sample["frame_idx"]          int
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import timm
import torch
# Aliased to avoid confusion with our wrapper class below.
from kitscenes.dataset import KITScenesDataset as _KITScenesSDK
from kitscenes.poses import load_ego_poses
from torch.utils.data import Dataset

from .camera import CAMERA_NAMES, load_camera_frame, compute_camera_projection_matrices
from .egomotion import (
    MIN_ROWS,
    _FUTURE_TIMESTEPS,
    _HISTORY_TIMESTEPS,
    load_egomotion,
    poses_to_arrays,
)

logger = logging.getLogger(__name__)

_VISUAL_HISTORY_DIM = 896


class ClipSample(TypedDict):
    visual_tiles: torch.Tensor        # (8, 3, H, W) — 7 cameras + BEV map tile
    egomotion_history: torch.Tensor   # (256,)
    visual_history: torch.Tensor      # (896,)
    trajectory_target: torch.Tensor   # (128,)
    scene_id: str
    frame_idx: int


class KitScenesDataset(Dataset):
    """Dataset where each item is one valid (scene_id, frame_idx) pair.

    All valid frame indices across all scenes are enumerated at construction
    time. __getitem__ does only I/O — derived egomotion quantities and UTM
    translations are cached per scene during construction, and sensor loaders
    are fetched on demand from the SDK's own (lru_cached) ``get_sensor_loader``.

    Args:
        data_root: Path to the dataset root (HuggingFace layout:
            ``data/<split>/`` under this path). If ``None``, the SDK falls
            back to ``$KITSCENES_ROOT``.
        backbone_name: timm backbone whose preprocessing config drives the
            image transform.
        split: Restrict to one SDK split ('train', 'val', 'test', 'test_e2e',
            'overlap_train_val'). If ``None``, all scenes are discovered.
        camera_names: Camera views to load. Defaults to ``CAMERA_NAMES``.
        scene_ids: Optional explicit list of scene IDs. If ``None``, all valid
            scenes in the split are used. Pass a single-element list for smoke
            tests or forward pass validation.
    """

    def __init__(
        self,
        data_root: Path | str | None = None,
        backbone_name: str = "swinv2_tiny_window8_256",
        split: str | None = None,
        camera_names: list[str] | None = None,
        scene_ids: list[str] | None = None,
    ) -> None:
        self.camera_names = camera_names or CAMERA_NAMES

        # Build the image transform from the backbone's own config so that
        # preprocessing always matches what the backbone expects.
        # create_model loads config only — no pretrained weights downloaded here.
        _backbone = timm.create_model(backbone_name, pretrained=False)
        data_config = timm.data.resolve_model_data_config(_backbone)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        del _backbone

        self._sdk = _KITScenesSDK(root=data_root, split=split)

        scenes = scene_ids if scene_ids is not None else self._sdk.scene_ids
        if not scenes:
            raise ValueError(f"No scenes found under: {self._sdk.root}")

        # Per-scene caches populated during construction.
        self._scene_egomotion: dict[str, np.ndarray] = {}      # (T, 4) float32
        self._scene_positions: dict[str, np.ndarray] = {}      # (T, 2) float64
        self._scene_camera_params: dict[str, torch.Tensor] = {}  # (7, 3, 4) float32

        # Build the flat sample index: list of (scene_id, frame_idx).
        # Precomputing this means __getitem__ never touches the SDK metadata.
        self._samples: list[tuple[str, int]] = []
        for scene_id in scenes:
            self._samples.extend(self._valid_samples_for_scene(scene_id))

        if not self._samples:
            raise ValueError("No valid samples found across all scenes.")

        logger.info(
            "KitScenesDataset: %d samples from %d scenes",
            len(self._samples), len(self._scene_egomotion),
        )

    def _valid_samples_for_scene(self, scene_id: str) -> list[tuple[str, int]]:
        """Return all valid (scene_id, frame_idx) for one scene.

        Validates camera presence and pose-stream length, then caches the
        derived egomotion array and UTM translation array. A frame_idx is valid
        when there are _HISTORY_TIMESTEPS frames behind it and _FUTURE_TIMESTEPS
        ahead of it, within the span covered by both ego poses and camera frames.
        """
        # Work off the sensor loader rather than get_scene. get_scene is
        # lru_cache(maxsize=None) and would pin every scene's raw ego poses in
        # the SDK cache for the dataset's lifetime. 
        loader = self._sdk.get_sensor_loader(scene_id)

        present = set(loader.get_camera_names())
        missing = [c for c in self.camera_names if c not in present]
        if missing:
            logger.warning(
                "Scene %s: missing cameras %s. Skipping.", scene_id, missing
            )
            return []

        poses = load_ego_poses(loader.scene_path)
        if len(poses) < MIN_ROWS:
            logger.warning(
                "Scene %s has only %d ego poses (need %d). Skipping.",
                scene_id, len(poses), MIN_ROWS,
            )
            return []

        egomotion, translations_local = poses_to_arrays(poses)

        # Cameras and poses share the reference timeline but may differ in
        # count at the tail; cap the valid range to the span both cover.
        num_frames = len(loader.get_reference_timestamps())
        usable = min(len(egomotion), num_frames)

        min_idx = _HISTORY_TIMESTEPS
        max_idx = usable - _FUTURE_TIMESTEPS - 1
        if max_idx < min_idx:
            logger.warning(
                "Scene %s: usable span %d too short for a sample. Skipping.",
                scene_id, usable,
            )
            return []

        self._scene_egomotion[scene_id] = egomotion
        self._scene_positions[scene_id] = translations_local

        # Projection matrices are frame-invariant; compute once per scene.
        # min_idx is any valid frame; used only as fallback if calib.image_size
        # is absent from calib.json (triggers a single image-header read).
        self._scene_camera_params[scene_id] = compute_camera_projection_matrices(
            loader,
            camera_names=self.camera_names,
        )

        return [(scene_id, frame_idx) for frame_idx in range(min_idx, max_idx + 1)]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> ClipSample:
        scene_id, frame_idx = self._samples[idx]

        # get_sensor_loader is lru_cached on the SDK; safe to call in hot path.
        loader = self._sdk.get_sensor_loader(scene_id)

        # Map-local position and heading at this frame for BEV map tile.
        ego_xy = self._scene_positions[scene_id][frame_idx]       # (2,) float64
        ego_yaw = float(self._scene_egomotion[scene_id][frame_idx, 2])  # yaw, radians

        visual_tiles = load_camera_frame(
            loader,
            frame_idx,
            transform=self.transform,
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            camera_names=self.camera_names,
        )

        egomotion_history, trajectory_target = load_egomotion(
            self._scene_egomotion[scene_id],
            frame_idx=frame_idx,
        )

        visual_history = torch.zeros(_VISUAL_HISTORY_DIM, dtype=torch.float32)

        return ClipSample(
            visual_tiles=visual_tiles,
            egomotion_history=egomotion_history,
            visual_history=visual_history,
            trajectory_target=trajectory_target,
            scene_id=scene_id,
            frame_idx=frame_idx,
            camera_params=self._scene_camera_params[scene_id],
        )