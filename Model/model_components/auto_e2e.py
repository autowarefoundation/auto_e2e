"""AutoE2E: the Reactive AutoBEV model with egomotion normalization.

The checkpoint keys keep the ``Reactive_E2E.`` prefix of the model they wrap.
"""

from typing import Final

import torch
import torch.nn as nn

from .reactive_e2e import ReactiveE2E

# Egomotion history is 64 timesteps x [speed, acceleration, yaw_rate, curvature]
# (the layout is identical for nuPlan, KITScenes and L2D).
EGOMOTION_SIGNALS: Final = 4
# Speed in m/s divided by 33 (74 mph) and acceleration in m/s^2 divided by 8
# (harsh emergency braking); the angular signals stay as they are.
EGOMOTION_SCALE: Final = (1.0 / 33.0, 1.0 / 8.0, 1.0, 1.0)


def normalize_egomotion(egomotion_history: torch.Tensor) -> torch.Tensor:
    """Scale the speed and acceleration of a ``[..., T * 4]`` egomotion history."""
    scale = torch.tensor(EGOMOTION_SCALE, dtype=egomotion_history.dtype, device=egomotion_history.device)
    shape = egomotion_history.shape
    return (egomotion_history.reshape(*shape[:-1], -1, EGOMOTION_SIGNALS) * scale).reshape(shape)


class AutoE2E(nn.Module):
    def __init__(self, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.Reactive_E2E = ReactiveE2E(pretrained_backbone=pretrained_backbone)

    def forward(self, egomotion_history: torch.Tensor, **inputs: torch.Tensor):
        """Normalize the egomotion and run ``ReactiveE2E.forward`` on the same named inputs."""
        return self.Reactive_E2E(egomotion_history=normalize_egomotion(egomotion_history), **inputs)
