import torch
import torch.nn as nn


class TrajectoryImitationLoss(nn.Module):
    """Primary task loss: imitation loss over predicted trajectory."""

    def __init__(self, loss_type: str = "smooth_l1", temporal_decay: float = 1.0,
                 num_timesteps: int = 64, num_signals: int = 2):
        super().__init__()
        if loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction="none")
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        self.temporal_decay = temporal_decay
        self.num_timesteps = num_timesteps
        self.num_signals = num_signals

    def _build_temporal_weights(self, num_timesteps: int, device: torch.device) -> torch.Tensor:
        if self.temporal_decay == 1.0:
            return torch.ones(num_timesteps, device=device)
        t = torch.arange(num_timesteps, device=device, dtype=torch.float32)
        return self.temporal_decay ** t

    def forward(self, trajectory_pred: torch.Tensor, trajectory_target: torch.Tensor) -> torch.Tensor:
        B = trajectory_pred.shape[0]
        pred = trajectory_pred.view(B, self.num_timesteps, self.num_signals)
        target = trajectory_target.view(B, self.num_timesteps, self.num_signals)

        per_element_loss = self.loss_fn(pred, target)
        per_timestep_loss = per_element_loss.mean(dim=2)

        weights = self._build_temporal_weights(self.num_timesteps, trajectory_pred.device)
        weighted_loss = per_timestep_loss * weights.unsqueeze(0)

        return weighted_loss.mean()
