import torch
import torch.nn as nn


class TrajectoryImitationLoss(nn.Module):
    """Primary task loss: imitation loss over predicted trajectory."""

    # Class-level annotations so mypy resolves these to their real types
    # instead of nn.Module's ``__getattr__ -> Tensor | Module`` fallback
    # (otherwise ``self.loss_fn(...)`` is flagged "Tensor not callable").
    loss_fn: nn.Module
    temporal_weights: torch.Tensor
    signal_scales: torch.Tensor

    # Per-signal std of the trajectory target (accel_x m/s², curvature rad/m),
    # measured on real L2D shards: accel std≈0.79, curvature std≈0.12. Without
    # normalizing by these, SmoothL1(β=1) puts accel errors in the linear regime
    # (grad≈1) but curvature errors deep in the quadratic regime (grad≈error≈0),
    # so the planner learns longitudinal accel and under-learns curvature →
    # heading integrates wrong → large ADE/FDE. Dividing both signals by their std
    # makes them ~unit-variance so curvature gets comparable gradient. NOTE: the
    # previous curvature scale (0.014) was ~9× too small — it came from a truncated
    # sample; the real per-target std measured across full L2D shards is ~0.12
    # (accel ~0.79). Override via ``signal_scales`` if the target definition
    # changes.
    _DEFAULT_SIGNAL_SCALES = (0.79, 0.12)

    def __init__(self, loss_type: str = "smooth_l1", temporal_decay: float = 0.95,
                 num_timesteps: int = 64, num_signals: int = 2,
                 signal_scales: tuple = None):
        # temporal_decay defaults to 0.95 so near-future predictions are
        # weighted more heavily than far-future ones; near-future accuracy
        # is more safety-critical for planning.
        super().__init__()
        if loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction="none")
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        self.num_timesteps = num_timesteps
        self.num_signals = num_signals

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

        # Normalize each signal to ~unit variance so SmoothL1 gives accel and
        # curvature comparable gradient (see _DEFAULT_SIGNAL_SCALES rationale).
        scales = self.signal_scales.view(1, 1, self.num_signals)
        pred = pred / scales
        target = target / scales

        per_element_loss = self.loss_fn(pred, target)
        per_timestep_loss = per_element_loss.mean(dim=2)

        weighted_loss = per_timestep_loss * self.temporal_weights.unsqueeze(0)

        return weighted_loss.mean()
