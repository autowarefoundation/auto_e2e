"""Simple masked XY trajectory imitation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.losses.control_rollout import integrate_controls_torch


class TrajectoryXYImitationLoss(nn.Module):
    """Integrate controls and apply uniform masked Smooth L1 in ego XY."""

    def __init__(self, *, dt: float = 0.1, beta_m: float = 1.0) -> None:
        super().__init__()
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if beta_m <= 0.0:
            raise ValueError("beta_m must be positive")
        self.dt = float(dt)
        self.beta_m = float(beta_m)

    def predicted_xy(
        self,
        predicted_controls: torch.Tensor,
        initial_speed_mps: torch.Tensor,
    ) -> torch.Tensor:
        positions, _, _ = integrate_controls_torch(
            predicted_controls,
            initial_speed_mps,
            dt=self.dt,
        )
        return positions

    def forward(
        self,
        predicted_controls: torch.Tensor,
        target_xy_m: torch.Tensor,
        trajectory_valid: torch.Tensor,
        initial_speed_mps: torch.Tensor,
    ) -> torch.Tensor:
        predicted_xy = self.predicted_xy(
            predicted_controls,
            initial_speed_mps,
        )
        if target_xy_m.shape != predicted_xy.shape:
            raise ValueError(
                "target_xy_m must match integrated trajectory shape"
            )
        if trajectory_valid.shape != predicted_xy.shape[:2]:
            raise ValueError("trajectory_valid must have shape [B,T]")
        target = target_xy_m.to(
            device=predicted_xy.device,
            dtype=predicted_xy.dtype,
        )
        valid = trajectory_valid.to(
            device=predicted_xy.device,
            dtype=torch.bool,
        )
        if not bool(valid.any()):
            return predicted_xy.sum() * 0.0
        per_coordinate = F.smooth_l1_loss(
            predicted_xy,
            target,
            reduction="none",
            beta=self.beta_m,
        )
        mask = valid.unsqueeze(-1).to(per_coordinate.dtype)
        return (per_coordinate * mask).sum() / (
            2.0 * mask.sum()
        ).clamp_min(1.0)
