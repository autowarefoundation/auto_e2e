"""Masked multi-label BEV segmentation auxiliary loss."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


BEV_SEGMENTATION_AUXILIARY_LOSS_VERSION = (
    "rank_corrected_sample_normalized_fixed_taxonomy_weighted_bce_"
    "quantized_positive_pair_dice_v7"
)


class BEVSegmentationAuxiliaryLoss(nn.Module):
    """FP32 mixture of class-balanced BCE and per-sample Soft Dice."""

    pos_weight: torch.Tensor
    class_weight: torch.Tensor
    positive_pair_frequency: torch.Tensor

    def __init__(
        self,
        pos_weight: Sequence[float] | torch.Tensor,
        *,
        class_weight: Sequence[float] | torch.Tensor | None = None,
        positive_pair_frequency: (
            Sequence[float] | torch.Tensor | None
        ) = None,
        dice_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(pos_weight, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() == 0:
            raise ValueError("pos_weight must be a non-empty 1D sequence")
        if not torch.isfinite(weights).all() or bool((weights < 1.0).any()):
            raise ValueError("pos_weight entries must be finite and >= 1")
        class_weights = torch.as_tensor(
            (
                class_weight
                if class_weight is not None
                else torch.ones_like(weights)
            ),
            dtype=torch.float32,
        )
        if (
            class_weights.shape != weights.shape
            or not torch.isfinite(class_weights).all()
            or bool((class_weights <= 0.0).any())
        ):
            raise ValueError(
                "class_weight entries must be finite, positive, and match "
                "pos_weight"
            )
        if dice_epsilon <= 0.0:
            raise ValueError("dice_epsilon must be positive")
        positive_pair_frequencies = torch.as_tensor(
            (
                positive_pair_frequency
                if positive_pair_frequency is not None
                else torch.ones_like(weights)
            ),
            dtype=torch.float32,
        )
        if (
            positive_pair_frequencies.shape != weights.shape
            or not torch.isfinite(positive_pair_frequencies).all()
            or bool((positive_pair_frequencies <= 0.0).any())
            or bool((positive_pair_frequencies > 1.0).any())
        ):
            raise ValueError(
                "positive_pair_frequency entries must be finite, in "
                "(0, 1], and match pos_weight"
            )
        self.register_buffer("pos_weight", weights)
        self.register_buffer("class_weight", class_weights)
        self.register_buffer(
            "positive_pair_frequency",
            positive_pair_frequencies,
        )
        self.dice_epsilon = float(dice_epsilon)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        sampling_importance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.components(
            logits,
            target,
            valid_mask,
            sampling_importance=sampling_importance,
        )["total"]

    def components(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        sampling_importance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return importance-corrected BCE and positive-pair Dice in FP32."""
        if logits.ndim != 4:
            raise ValueError("logits must have shape [B,C,H,W]")
        if target.shape != logits.shape or valid_mask.shape != logits.shape:
            raise ValueError("target and valid_mask must match logits")
        if logits.shape[1] != self.pos_weight.numel():
            raise ValueError("logit channels differ from pos_weight")
        logits_fp32 = logits.to(dtype=torch.float32)
        target_fp32 = target.to(device=logits.device, dtype=torch.float32)
        valid = valid_mask.to(device=logits.device, dtype=torch.bool)
        active = valid.any(dim=(2, 3))
        sample_active = active.any(dim=1)
        if not bool(sample_active.any()):
            zero = logits_fp32.sum() * 0.0
            return {"total": zero, "bce": zero, "dice": zero}
        if not bool(sample_active.all()):
            raise ValueError(
                "BEV loss cannot mix fully invalid and valid samples"
            )
        if sampling_importance is None:
            importance = torch.ones(
                logits.shape[0],
                device=logits.device,
                dtype=torch.float32,
            )
        else:
            importance = torch.as_tensor(
                sampling_importance,
                device=logits.device,
                dtype=torch.float32,
            ).reshape(-1)
            if (
                importance.shape != logits.shape[:1]
                or not torch.isfinite(importance).all()
                or bool((importance <= 0.0).any())
            ):
                raise ValueError(
                    "sampling_importance must be finite, positive, and "
                    "have shape [B]"
                )

        with torch.autocast(
            device_type=logits.device.type,
            enabled=False,
        ):
            mask = valid.to(torch.float32)
            safe_logits = torch.where(
                valid,
                logits_fp32,
                torch.zeros_like(logits_fp32),
            )
            safe_target = torch.where(
                valid,
                target_fp32,
                torch.zeros_like(target_fp32),
            )
            bce = F.binary_cross_entropy_with_logits(
                safe_logits,
                safe_target,
                pos_weight=self.pos_weight.view(1, -1, 1, 1),
                reduction="none",
            )
            valid_counts = mask.sum(dim=(2, 3)).clamp_min(1.0)
            sample_bce = (bce * mask).sum(dim=(2, 3)) / valid_counts
            active_pair_weight = (
                self.class_weight.view(1, -1)
                * active.to(torch.float32)
            )
            # Keep a fixed taxonomy objective. Renormalizing by the active
            # classes would make missing supervision amplify unrelated classes.
            sample_bce = (
                (sample_bce * active_pair_weight).sum(dim=1)
                / self.class_weight.sum()
            )

            probabilities = safe_logits.sigmoid()
            intersection = (
                probabilities * safe_target * mask
            ).sum(dim=(2, 3))
            denominator = (
                (probabilities + safe_target) * mask
            ).sum(dim=(2, 3))
            sample_dice = 1.0 - (
                2.0 * intersection + self.dice_epsilon
            ) / (denominator + self.dice_epsilon)
            active_sample_count = sample_active.to(torch.float32).sum()
            bce_loss = (
                sample_bce[sample_active]
                * importance[sample_active]
            ).sum() / active_sample_count
            positive_active = active & (
                ((safe_target >= 0.5) & valid).any(dim=(2, 3))
            )
            positive_pair_weight = (
                self.class_weight.view(1, -1)
                * positive_active.to(torch.float32)
            )
            dice_numerator = (
                sample_dice * positive_pair_weight
            ).sum(dim=1)
            # This fixed expectation keeps the Horvitz-Thompson estimator
            # unbiased after rare-positive repeat sampling.
            positive_pair_normalizer = (
                self.class_weight * self.positive_pair_frequency
            ).sum()
            dice_loss = (
                dice_numerator[sample_active]
                * importance[sample_active]
            ).sum() / (
                active_sample_count * positive_pair_normalizer
            )
            total_loss = 0.5 * bce_loss + 0.5 * dice_loss
            return {
                "total": total_loss,
                "bce": bce_loss,
                "dice": dice_loss,
            }
