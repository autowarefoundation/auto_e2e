from abc import ABC, abstractmethod

import torch.nn as nn


class BasePlanner(nn.Module, ABC):
    """Abstract trajectory planner.

    The planner exposes two named entry points so that train and inference
    have stable, distinct contracts:

    * ``forward()`` always performs inference and returns
      ``(trajectory)`` regardless of the underlying decoder.
      It must NOT return mode-dependent intermediate quantities (e.g. the
      flow-matching velocity field). A caller can rely on the first return
      being a fully-formed ``[B, num_timesteps * num_signals]`` trajectory.

    * ``compute_planner_loss()`` runs the training objective and returns
      ``(loss)``. It owns any decoder-specific scratch tensors
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
        """Inference: return ``(trajectory)``."""
        raise NotImplementedError
