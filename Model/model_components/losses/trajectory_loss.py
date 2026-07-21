from typing import Optional

import torch
import torch.nn as nn


class TrajectoryImitationLoss(nn.Module):
    """Dataset-neutral imitation loss over predicted control trajectories."""

    # Class-level annotations so mypy resolves these to their real types
    # instead of nn.Module's ``__getattr__ -> Tensor | Module`` fallback
    # (otherwise ``self.loss_fn(...)`` is flagged "Tensor not callable").
    loss_fn: nn.Module
    temporal_weights: torch.Tensor
    signal_scales: torch.Tensor

    # Neutral defaults keep this reusable. Production training must pass the
    # explicit dataset policy rather than inheriting values measured on L2D.
    _DEFAULT_SIGNAL_SCALES = (1.0, 1.0)

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        temporal_decay: float = 0.95,
        num_timesteps: int = 64,
        num_signals: int = 2,
        signal_scales: Optional[tuple] = None,
    ):
        super().__init__()
        if loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction="none")
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        self.num_timesteps = num_timesteps
        self.num_signals = num_signals
        if not 0.0 < temporal_decay <= 1.0:
            raise ValueError("temporal_decay must be in (0, 1]")

        if temporal_decay == 1.0:
            weights = torch.ones(num_timesteps)
        else:
            t = torch.arange(num_timesteps, dtype=torch.float32)
            weights = temporal_decay ** t
        self.register_buffer("temporal_weights", weights)

        scales = signal_scales if signal_scales is not None else self._DEFAULT_SIGNAL_SCALES
        if len(scales) != num_signals:
            raise ValueError(
                f"signal_scales must have {num_signals} entries, got {len(scales)}."
            )
        self.register_buffer(
            "signal_scales", torch.tensor(scales, dtype=torch.float32))

    def forward(self, trajectory_pred: torch.Tensor, trajectory_target: torch.Tensor) -> torch.Tensor:
        B = trajectory_pred.shape[0]
        pred = trajectory_pred.view(B, self.num_timesteps, self.num_signals)
        target = trajectory_target.view(B, self.num_timesteps, self.num_signals)

        # Dataset-specific scales make acceleration and curvature contribute
        # comparable gradients despite their different physical units.
        scales = self.signal_scales.view(1, 1, self.num_signals)
        pred = pred / scales
        target = target / scales

        per_element_loss = self.loss_fn(pred, target)
        per_timestep_loss = per_element_loss.mean(dim=2)

        weighted_loss = per_timestep_loss * self.temporal_weights.unsqueeze(0)

        return weighted_loss.mean()
