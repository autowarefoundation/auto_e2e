import torch
import torch.nn as nn


class TrajectoryImitationLoss(nn.Module):
    """Primary task loss: imitation loss over predicted trajectory."""

    def __init__(self, loss_type: str = "smooth_l1", temporal_decay: float = 0.95,
                 num_timesteps: int = 64, num_signals: int = 2):
        # temporal_decay defaults to 0.95 so near-future predictions are
        # weighted more heavily than far-future ones; near-future accuracy
        # is more safety-critical for planning.
        super().__init__()
        self.loss_fn: nn.Module
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

    def forward(self, trajectory_pred: torch.Tensor, trajectory_target: torch.Tensor) -> torch.Tensor:
        B = trajectory_pred.shape[0]
        pred = trajectory_pred.view(B, self.num_timesteps, self.num_signals)
        target = trajectory_target.view(B, self.num_timesteps, self.num_signals)

        per_element_loss = self.loss_fn(pred, target)
        per_timestep_loss = per_element_loss.mean(dim=2)

        weighted_loss = per_timestep_loss * self.temporal_weights.unsqueeze(0)

        return weighted_loss.mean()
