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

    checkpoint_path: str = "checkpoints/autoe2e_kitscenes_v1.ckpt"
    """Path to trained AutoE2E model checkpoint file."""

    image_size: Tuple[int, int] = (256, 256)
    """Target camera resolution ``(H, W)`` expected by perception backbone."""

    planning_horizon_s: float = 3.0
    """Total future trajectory planning horizon in seconds."""

    planning_steps: int = 30
    """Number of output waypoint steps along the planning horizon."""

    camera_names: List[str] = field(
        default_factory=lambda: [
            "cam_front",
            "cam_front_left",
            "cam_front_right",
            "cam_side_left",
            "cam_side_right",
            "cam_rear_left",
            "cam_rear_right",
        ]
    )
    """List of 7 logical camera names matching KitScenes topology."""
