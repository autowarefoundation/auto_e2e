"""Masked multi-label BEV segmentation auxiliary loss."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVSegmentationAuxiliaryLoss(nn.Module):
    """Equal mixture of class-balanced BCE and Soft Dice."""

    pos_weight: torch.Tensor

    def __init__(
        self,
        pos_weight: Sequence[float] | torch.Tensor,
        *,
        dice_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(pos_weight, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() == 0:
            raise ValueError("pos_weight must be a non-empty 1D sequence")
        if not torch.isfinite(weights).all() or bool((weights < 1.0).any()):
            raise ValueError("pos_weight entries must be finite and >= 1")
        if dice_epsilon <= 0.0:
            raise ValueError("dice_epsilon must be positive")
        self.register_buffer("pos_weight", weights)
        self.dice_epsilon = float(dice_epsilon)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("logits must have shape [B,C,H,W]")
        if target.shape != logits.shape or valid_mask.shape != logits.shape:
            raise ValueError("target and valid_mask must match logits")
        if logits.shape[1] != self.pos_weight.numel():
            raise ValueError("logit channels differ from pos_weight")
        target = target.to(device=logits.device, dtype=logits.dtype)
        valid = valid_mask.to(device=logits.device, dtype=torch.bool)
        active = valid.any(dim=(0, 2, 3))
        if not bool(active.any()):
            return logits.sum() * 0.0

        mask = valid.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight.view(1, -1, 1, 1),
            reduction="none",
        )
        valid_counts = mask.sum(dim=(0, 2, 3)).clamp_min(1.0)
        class_bce = (bce * mask).sum(dim=(0, 2, 3)) / valid_counts

        probabilities = logits.sigmoid()
        intersection = (probabilities * target * mask).sum(
            dim=(0, 2, 3)
        )
        denominator = ((probabilities + target) * mask).sum(
            dim=(0, 2, 3)
        )
        class_dice = 1.0 - (
            2.0 * intersection + self.dice_epsilon
        ) / (denominator + self.dice_epsilon)
        class_loss = 0.5 * class_bce + 0.5 * class_dice
        return class_loss[active].mean()
