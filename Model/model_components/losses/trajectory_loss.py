import torch
import torch.nn as nn


class TrajectoryImitationLoss(nn.Module):
    """Primary task loss: imitation loss over predicted trajectory."""

    def __init__(self, loss_type: str = "smooth_l1", temporal_decay: float = 1.0):
        super().__init__()
        if loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction="none")
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        self.temporal_decay = temporal_decay

    def _build_temporal_weights(self, num_timesteps: int, device: torch.device) -> torch.Tensor:
        if self.temporal_decay == 1.0:
            return torch.ones(num_timesteps, device=device)
        t = torch.arange(num_timesteps, device=device, dtype=torch.float32)
        return self.temporal_decay ** t

    def forward(self, trajectory_pred: torch.Tensor, trajectory_target: torch.Tensor) -> torch.Tensor:
        # trajectory shape: (B, 128) -> reshape to (B, 64, 2)
        B = trajectory_pred.shape[0]
        pred = trajectory_pred.view(B, 64, 2)
        target = trajectory_target.view(B, 64, 2)

        per_element_loss = self.loss_fn(pred, target)
        # Average over the 2 signals (acceleration, curvature)
        per_timestep_loss = per_element_loss.mean(dim=2)

        weights = self._build_temporal_weights(64, trajectory_pred.device)
        weighted_loss = per_timestep_loss * weights.unsqueeze(0)

        return weighted_loss.mean()
