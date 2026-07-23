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

    * ``compute_planner_loss()`` runs the training objective and returns a
      ``dict[str, Tensor]`` with at least a ``"loss"`` key — the scalar
      actually used for backprop. Additional keys are diagnostic /
      loggable sub-terms specific to the decoder (e.g. ``"velocity_mse"``
      for FlowMatchingPlanner, ``"imitation_loss"`` for BezierPlanner).

      Returning a dict rather than a bare scalar is deliberate (see #123):
      it shapes ``compute_planner_loss`` as *the planner's training
      objective* in general, not specifically "the flow-matching loss" or
      "the imitation loss". ``train_il`` (or any future training loop)
      only ever reads ``result["loss"]`` and stays agnostic to which
      planner/stage produced it. A future stage-3 RL objective can swap in
      behind this same entry point — returning e.g.
      ``{"loss": total, "imitation_loss": ..., "reward": ...}`` — blending
      an imitation anchor with RL terms without forcing a signature change
      on every caller.

      Each planner owns any decoder-specific scratch tensors (noise
      samples, target velocities, ...) so they never escape into the
      caller's scope where they could be paired with the wrong target.

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

    @abstractmethod
    def compute_planner_loss(self, bev_features, visual_history,
                             egomotion_history, trajectory_target,
                             training_policy=None, **kwargs):
        """Training objective. Returns ``dict[str, Tensor]`` with a
        ``"loss"`` key (see class docstring). A missing implementation now
        fails loudly at planner-build time instead of silently mis-training
        (the #115 failure mode).

        Args:
            training_policy: optional ``DatasetTrainingPolicy``
                (``Model/training/dataset_policy.py``). Pass the object
                itself, not pre-extracted scalars — a caller that derives
                ``signal_scales``/``temporal_decay`` separately and passes
                them alongside the policy risks the two silently drifting
                apart (see #124 review: ``signal_scales=(1.0, 1.0)`` vs.
                production ``(0.79, 0.12)`` gave a 71% loss difference with
                no error). Implementations that don't use per-signal
                weighting (e.g. a velocity-MSE objective, where scaling
                (accel, curvature) channels isn't obviously the same
                operation as it is on a direct trajectory regression) may
                accept and ignore it — see FlowMatchingPlanner's docstring
                for why that's left unresolved rather than guessed at.
        """
        raise NotImplementedError

    def _validate_trajectory_target(self, trajectory_target, batch_size, device):
        """Shared shape/device guard for compute_planner_loss implementations.

        Lifted here (rather than left on FlowMatchingPlanner alone) because
        every subclass's compute_planner_loss needs the same check, and a
        missing batch dimension is a silent-wrong-answer bug, not a crash:
        smooth_l1_loss / mse_loss both broadcast a [T] target across a [B, T]
        prediction without error, training against the wrong sample for the
        whole batch. Requires ``self.trajectory_dim`` to be set by the
        subclass __init__ (num_timesteps * num_signals).
        """
        expected = (batch_size, self.trajectory_dim)
        if tuple(trajectory_target.shape) != expected:
            raise ValueError(
                f"trajectory_target must have shape {expected} "
                f"(batch_size, num_timesteps * num_signals), got "
                f"{tuple(trajectory_target.shape)}."
            )
        if trajectory_target.device != device:
            raise ValueError(
                f"trajectory_target must be on the same device as bev_features, "
                f"got {trajectory_target.device} and {device}."
            )
