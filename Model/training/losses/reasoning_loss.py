"""Multi-label, multi-horizon student↔teacher distillation loss (issue #98).

This module lives OUTSIDE the model (in ``Model/training/``) and is computed
in the training loop — matching Zain's explicit requirement that loss modules
for each band stay separate from the model forward pass (same principle as the
JEPA loss in #85).

The loss is binary-cross-entropy (BCE) with sigmoid targets, summed across
taxonomy groups and averaged across horizons, then scaled by a scalar weight.
Soft teacher targets (confidence scores in [0, 1]) are supported directly
because ``F.binary_cross_entropy_with_logits`` accepts float targets.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

# Type aliases matching reasoning_band.py and teachers/base.py
ReasoningOutput = Dict[str, List[torch.Tensor]]   # logits (raw, pre-sigmoid)
ReasoningTargets = Dict[str, List[torch.Tensor]]  # targets in [0, 1]


class ReasoningLoss:
    """Weighted student↔teacher distillation loss for the reasoning band.

    Computes BCE with logits between the student's per-group, per-horizon
    sigmoid heads and the teacher's multi-label targets.  Losses are averaged
    across horizons and summed across groups (so adding a new group increases
    the raw loss magnitude; callers should rescale ``weight`` accordingly).

    This class is intentionally **not** an ``nn.Module`` — it holds no
    trainable parameters (it is a pure stateless loss function) and lives in
    the training loop, not in the model graph.

    Args:
        weight: scalar multiplier applied to the final loss before returning
            (default 1.0, matching the JEPA equal-weight start policy from #85).
        reduction: ``"mean"`` (default) averages over the batch; ``"none"``
            returns per-sample losses of shape ``[B]`` for logging.
        loss_type: ``"bce"`` (default) or ``"asl"``.  ASL (Asymmetric Loss,
            Ridnik et al., arXiv:2009.14119) down-weights the flood of easy
            negatives that dominates an imbalanced multi-label problem — our
            scenario distribution is heavily skewed (intersection ~29.6% vs
            nighttime ~5.1%) — and reported 86.6 vs 84.0 mAP over plain
            cross-entropy on MS-COCO.  Soft teacher targets are supported by
            weighting the positive/negative terms with the target value.
        gamma_neg / gamma_pos / clip: ASL focusing/shift parameters (paper
            defaults 4 / 0 / 0.05); ignored for ``loss_type="bce"``.

    Example::

        loss_fn = ReasoningLoss(weight=1.0)
        loss = loss_fn(student_logits, teacher_targets)
        loss.backward()
    """

    def __init__(
        self,
        weight: float = 1.0,
        reduction: str = "mean",
        loss_type: str = "bce",
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
    ) -> None:
        if reduction not in ("mean", "none"):
            raise ValueError(
                f"Unsupported reduction '{reduction}'. Choose 'mean' or 'none'."
            )
        if loss_type not in ("bce", "asl"):
            raise ValueError(
                f"Unsupported loss_type '{loss_type}'. Choose 'bce' or 'asl'."
            )
        self.weight = weight
        self.reduction = reduction
        self.loss_type = loss_type
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def _asl_with_logits(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Asymmetric Loss (element-wise, ``[B, C]``), soft-target friendly."""
        eps = 1e-8
        p = torch.sigmoid(logits)
        # Probability shifting: hard-discard very easy negatives.
        p_shifted = (p - self.clip).clamp(min=0.0)
        loss_pos = ((1.0 - p) ** self.gamma_pos) * torch.log(p.clamp(min=eps))
        loss_neg = (p_shifted ** self.gamma_neg) * torch.log(
            (1.0 - p_shifted).clamp(min=eps)
        )
        return -(targets * loss_pos + (1.0 - targets) * loss_neg)

    def __call__(
        self,
        student_logits: ReasoningOutput,
        teacher_targets: ReasoningTargets,
    ) -> torch.Tensor:
        """Compute the reasoning distillation loss.

        Args:
            student_logits: dict mapping group name → list of per-horizon raw
                logits ``[B, num_classes]`` (output of :class:`ReasoningBand`).
            teacher_targets: dict mapping group name → list of per-horizon
                float targets ``[B, num_classes]`` in ``[0, 1]`` (output of a
                :class:`VLMTeacher`).

        Returns:
            Scalar loss (or ``[B]`` when ``reduction="none"``), multiplied by
            ``self.weight``.

        Raises:
            ValueError: if the set of groups or number of horizons does not
                match between logits and targets.
        """
        if student_logits.keys() != teacher_targets.keys():
            raise ValueError(
                f"Group mismatch: student has {set(student_logits.keys())}, "
                f"teacher has {set(teacher_targets.keys())}."
            )

        group_losses: list[torch.Tensor] = []

        for group_name in student_logits:
            s_horizons = student_logits[group_name]
            t_horizons = teacher_targets[group_name]
            if len(s_horizons) != len(t_horizons):
                raise ValueError(
                    f"Group '{group_name}': student has {len(s_horizons)} horizons, "
                    f"teacher has {len(t_horizons)}."
                )
            horizon_losses: list[torch.Tensor] = []
            for s_logit, t_target in zip(s_horizons, t_horizons):
                if self.loss_type == "asl":
                    elem = self._asl_with_logits(s_logit, t_target.float())
                    term = elem.mean() if self.reduction == "mean" else elem.mean(dim=-1)
                else:
                    # BCE with logits handles multi-label naturally (sigmoid implicit).
                    term = F.binary_cross_entropy_with_logits(
                        s_logit, t_target.float(), reduction=self.reduction
                    )
                    # When reduction="none", bce is [B, C]; average over classes.
                    if self.reduction == "none":
                        term = term.mean(dim=-1)   # [B]
                horizon_losses.append(term)

            # Average across horizons: [B] or scalar.
            group_losses.append(torch.stack(horizon_losses).mean(dim=0))

        # Sum across groups (each group contributes independently).
        total: torch.Tensor = torch.stack(group_losses).sum(dim=0)
        return self.weight * total
