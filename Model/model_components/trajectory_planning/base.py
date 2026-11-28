from abc import ABC, abstractmethod

import torch.nn as nn
import torch.nn.functional as F


class BasePlanner(nn.Module, ABC):
    """Abstract trajectory planner.

    ``forward()`` always performs inference and returns ``(trajectory)``
    regardless of the underlying decoder. It must not return mode-dependent
    intermediate quantities (e.g. the flow-matching velocity field). A caller
    can rely on the first return being a fully-formed
    ``[B, num_timesteps * num_signals]`` trajectory.

    ``compute_planner_loss()`` computes the planner's training objective from
    conditioning inputs and ``trajectory_target`` and returns a dict containing
    at least ``"loss"``.
    """

    @abstractmethod
    def forward(self, bev_features, visual_history, egomotion_history,
                **kwargs):
        """Inference: return ``(trajectory)``."""
        raise NotImplementedError

    def _compute_imitation_loss_and_trajectory(
        self,
        bev_features,
        visual_history,
        egomotion_history,
        trajectory_target,
        *,
        loss_fn=None,
        reasoning_latent=None,
        reasoning_horizon_tokens=None,
        **kwargs,
    ):
        """Shared training-objective helper for deterministic rollout planners."""
        trajectory = self(
            bev_features,
            visual_history,
            egomotion_history,
            reasoning_latent=reasoning_latent,
            reasoning_horizon_tokens=reasoning_horizon_tokens,
            **kwargs,
        )
        criterion = loss_fn if loss_fn is not None else F.smooth_l1_loss
        loss = criterion(trajectory, trajectory_target)
        return {"loss": loss}, trajectory

    @abstractmethod
    def compute_planner_loss(
        self,
        bev_features,
        visual_history,
        egomotion_history,
        trajectory_target,
        *,
        loss_fn=None,
        reasoning_latent=None,
        reasoning_horizon_tokens=None,
        **kwargs,
    ):
        """Training: return a dict of scalar losses with at least ``"loss"``."""
        raise NotImplementedError
