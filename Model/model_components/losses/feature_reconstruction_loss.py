import torch
import torch.nn as nn


class FeatureReconstructionLoss(nn.Module):
    """JEPA-style feature reconstruction loss for the FutureState world model.

    Compares the future BEV feature maps predicted by ``FutureState`` (a
    tuple/list of ``num_future_steps`` tensors, each ``[B, C, H, W]``) against
    target feature maps of identical shapes. Following the JEPA recipe, the
    loss lives in *feature space* rather than pixel space: in production the
    targets are extracted by a frozen copy of the image backbone (no gradient)
    applied to the future frames at +1.6s, +3.2s, +4.8s and +6.4s, so the
    world model learns to predict abstract scene dynamics instead of
    reconstructing pixels.

    Supports mean reduction (optionally weighted per future timestep) and a
    ``"none"`` reduction that returns the per-timestep losses for logging.
    """

    # Declares the registered buffer's type so mypy resolves it to Tensor
    # (not Tensor | Module) when it is used in arithmetic below.
    temporal_weights: torch.Tensor

    def __init__(self, num_future_steps: int = 4, temporal_weights=None,
                 reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.num_future_steps = num_future_steps
        self.reduction = reduction

        if temporal_weights is None:
            weights = torch.ones(num_future_steps)
        else:
            weights = torch.as_tensor(temporal_weights, dtype=torch.float32)
            if weights.numel() != num_future_steps:
                raise ValueError(
                    f"temporal_weights must have {num_future_steps} elements, "
                    f"got {weights.numel()}"
                )
        # Normalise so the weighted mean stays on the same scale as plain MSE.
        # Guard against a zero (or numerically negligible) sum: dividing by it
        # would turn the loss into NaN. In that degenerate case we skip the
        # normalisation and keep the weights as provided.
        total = weights.sum()
        if torch.abs(total) > 1e-8:
            weights = weights / total * num_future_steps
        self.register_buffer("temporal_weights", weights)

    def forward(self, predicted_features, target_features) -> torch.Tensor:
        """Compute the loss.

        Args:
            predicted_features: tuple/list of ``num_future_steps`` tensors
                ``[B, C, H, W]`` as returned by ``FutureState.forward``.
            target_features: tuple/list of tensors with the same shapes,
                produced by a frozen backbone on the future frames (targets
                should be detached / require no grad).

        Returns:
            Scalar loss (``reduction="mean"``) or per-timestep losses of
            shape ``[num_future_steps]`` (``reduction="none"``).
        """
        if len(predicted_features) != self.num_future_steps:
            raise ValueError(
                f"Expected {self.num_future_steps} predicted feature maps, "
                f"got {len(predicted_features)}"
            )
        if len(target_features) != self.num_future_steps:
            raise ValueError(
                f"Expected {self.num_future_steps} target feature maps, "
                f"got {len(target_features)}"
            )

        per_step_losses: list[torch.Tensor] = []
        for pred, target in zip(predicted_features, target_features):
            if pred.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch: predicted {tuple(pred.shape)} vs "
                    f"target {tuple(target.shape)}"
                )
            per_step_losses.append(torch.mean((pred - target) ** 2))

        per_step = torch.stack(per_step_losses)  # [num_future_steps]
        weighted = per_step * self.temporal_weights

        if self.reduction == "none":
            return weighted
        return weighted.mean()
