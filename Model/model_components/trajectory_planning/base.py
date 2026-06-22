from abc import ABC, abstractmethod

import torch.nn as nn


class BasePlanner(nn.Module, ABC):
    """Abstract trajectory planner.

    The planner exposes two named entry points so that train and inference
    have stable, distinct contracts:

    * ``forward()`` always performs inference and returns
      ``(trajectory, ego_hidden)`` regardless of the underlying decoder.
      It must NOT return mode-dependent intermediate quantities (e.g. the
      flow-matching velocity field). A caller can rely on the first return
      being a fully-formed ``[B, num_timesteps * num_signals]`` trajectory.

    * ``compute_planner_loss()`` runs the training objective and returns
      ``(loss, ego_hidden)``. It owns any decoder-specific scratch tensors
      (noise samples, target velocities, ...) so they never escape into
      the caller's scope where they could be paired with the wrong target.

    This split mirrors Diffusion Policy / Alpamayo / torchcfm: a polymorphic
    ``forward()`` whose output meaning flips by mode is a footgun (e.g. an
    MSE-against-trajectory loop silently regresses a velocity in train mode);
    splitting the contract makes that mistake structurally impossible.
    """

    @abstractmethod
    def forward(self, bev_features, visual_history, egomotion_history,
                **kwargs):
        """Inference: return ``(trajectory, ego_hidden)``."""
        raise NotImplementedError

    @abstractmethod
    def compute_planner_loss(self, bev_features, visual_history,
                             egomotion_history, trajectory_target):
        """Return ``(loss, ego_hidden)``.

        Args:
            bev_features: [B, embed_dim, H, W].
            visual_history: [B, visual_history_dim].
            egomotion_history: [B, egomotion_dim].
            trajectory_target: [B, num_timesteps * num_signals] ground-truth
                trajectory.

        Returns:
            loss: scalar tensor — the planner's training objective given
                ``trajectory_target``. For the GRU planner this is an
                imitation (regression) loss on the trajectory; for Flow
                Matching it is the MSE between the predicted and target
                conditional velocities. The intermediate velocity is
                computed and consumed INSIDE this method and never
                returned, so a caller cannot accidentally MSE a velocity
                against a trajectory target.
            ego_hidden: [B, embed_dim] — same context vector ``forward()``
                produces, consumed downstream by FutureState.
        """
        raise NotImplementedError
