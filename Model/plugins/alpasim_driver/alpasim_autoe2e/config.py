"""Configuration dataclasses for the AutoE2E AlpaSim driver plugin.

Defines model checkpoints, camera topology settings, and trajectory planning horizon settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class AutoE2EAlpaSimConfig:
    """Configuration options for ``AutoE2EAlpaSimModel`` driver plugin.

    Registered with AlpaSim under entry point ``alpasim.configs``.
    """

    checkpoint_path: str
    """Path to trained AutoE2E model checkpoint file."""

    allow_mock: bool = False
    """Whether to allow mock fallback mode when running without AlpaSim."""

    allow_untrained_model: bool = False
    """Whether to initialize an untrained AutoE2E model if model checkpoint is missing."""

    image_size: Tuple[int, int] = (256, 256)
    """Target camera resolution ``(H, W)`` expected by perception backbone."""

    planning_horizon_s: float = 3.0
    """Total future trajectory planning horizon in seconds."""

    planning_steps: int = 64
    """Number of output waypoint steps along the planning horizon."""

    camera_names: List[str] = field(
        default_factory=lambda: [
            "camera_base_front_center",
            "camera_ring_front",
            "camera_ring_front_left",
            "camera_ring_front_right",
            "camera_ring_rear",
            "camera_ring_rear_left",
            "camera_ring_rear_right",
        ]
    )
    """List of 7 logical camera names matching KitScenes topology."""
