from abc import ABC, abstractmethod

import torch.nn as nn


class BasePlanner(nn.Module, ABC):
    """Abstract trajectory planner.

    Subclasses must produce a per-sample trajectory tensor and a final
    ego_hidden context vector consumed downstream by FutureState. The
    forward signature accepts an optional ``mode`` argument so swappable
    decoders (autoregressive GRU, Flow Matching ODE, diffusion, etc.) can
    branch on training vs inference, plus ``**kwargs`` so extra inputs
    needed by some planners (e.g. ``trajectory_target``, ``noisy_trajectory``,
    ``flow_timestep``) can flow through ``AutoE2E.forward`` without
    every planner having to declare them.
    """

    @abstractmethod
    def forward(self, bev_features, visual_history, egomotion_history,
                mode="train", **kwargs):
        raise NotImplementedError
