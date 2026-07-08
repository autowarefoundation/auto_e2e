"""Horizon-aware reasoning loss (issue #98, v2).

Supervises :class:`HorizonReasoningPrediction` against offline teacher labels.
Lives OUTSIDE the model (in ``training/``) and is computed in the training loop,
matching the per-branch-loss-module policy (same as the JEPA loss, #85/#13).

Four groups of terms:
    * structured — per action-relevant group, per horizon: BCE (or Asymmetric
      Loss) for multi-label groups, cross-entropy for single-label groups;
    * confidence — Brier (squared error) on the per-horizon confidence head;
    * temporal — a weak KL / L1 smoothness across adjacent horizons;
    * alignment — optional cosine loss to a precomputed teacher embedding.

Every term is source-/confidence-weighted (``source_weight × label_confidence``)
and abstained horizons are masked out (weight 0), so an endpoint failure never
contributes a spurious all-zero target (R8/R9). Per-head weights make the
action-facing heads (response / hazard / relation) dominate generic context.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from model_components.reasoning.reasoning_taxonomy import (
    DEFAULT_TAXONOMY,
    LabelMode,
    ReasoningTaxonomy,
)
from model_components.reasoning.types import HorizonReasoningPrediction

IGNORE_INDEX = -100

# Prediction attribute name per taxonomy group (the core action-relevant heads).
_GROUP_TO_ATTR: Dict[str, str] = {
    "relation_to_ego": "relation_to_ego_logits",
    "hazard_event": "hazard_event_logits",
    "cause": "cause_logits",
    "longitudinal_response": "longitudinal_response_logits",
    "lateral_response": "lateral_response_logits",
    "tactical_response": "tactical_response_logits",
    "rule_response": "rule_response_logits",
}

# Per-head weights: action-facing heads dominate generic context (#98 Loss §4).
_DEFAULT_HEAD_WEIGHTS: Dict[str, float] = {
    "longitudinal_response": 1.0,
    "lateral_response": 1.0,
    "tactical_response": 1.0,
    "rule_response": 1.0,
    "hazard_event": 1.0,
    "relation_to_ego": 0.8,
    "cause": 0.5,
}

# Action-facing heads that get the weak temporal-consistency regulariser.
_TEMPORAL_HEADS = (
    "hazard_event",
    "relation_to_ego",
    "longitudinal_response",
    "lateral_response",
    "tactical_response",
)


def _asl_with_logits(
    logits: torch.Tensor, targets: torch.Tensor,
    gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05,
) -> torch.Tensor:
    """Asymmetric Loss (element-wise), soft-target friendly (arXiv:2009.14119)."""
    eps = 1e-8
    p = torch.sigmoid(logits)
    p_shift = (p - clip).clamp(min=0.0)
    loss_pos = ((1.0 - p) ** gamma_pos) * torch.log(p.clamp(min=eps))
    loss_neg = (p_shift ** gamma_neg) * torch.log((1.0 - p_shift).clamp(min=eps))
    return -(targets * loss_pos + (1.0 - targets) * loss_neg)


class HorizonReasoningLoss:
    """Weighted structured + confidence + temporal + alignment reasoning loss.

    Not an ``nn.Module`` — it holds no trainable parameters and lives in the
    training loop, not the model graph.

    Args:
        taxonomy: label registry (for per-group mode lookup).
        head_weights: per-group scalar weights (defaults to
            :data:`_DEFAULT_HEAD_WEIGHTS`; groups absent default to 1.0).
        lambda_structured / lambda_confidence / lambda_temporal / lambda_alignment:
            the four term weights (#98 defaults 0.5 / 0.05 / 0.1 / 0.0-or-0.5).
        multilabel_loss: ``"bce"`` (default) or ``"asl"`` for multi-label heads.

    Call:
        loss_fn(prediction, targets, source_weights, confidence_targets=None,
                teacher_embedding_targets=None) -> dict of scalar tensors
        (``"total"`` plus per-term breakdown for logging).
    """

    def __init__(
        self,
        taxonomy: Optional[ReasoningTaxonomy] = None,
        head_weights: Optional[Dict[str, float]] = None,
        lambda_structured: float = 0.5,
        lambda_confidence: float = 0.05,
        lambda_temporal: float = 0.1,
        lambda_alignment: float = 0.0,
        multilabel_loss: str = "bce",
    ) -> None:
        if multilabel_loss not in ("bce", "asl"):
            raise ValueError(f"multilabel_loss must be 'bce' or 'asl', got {multilabel_loss!r}.")
        self.taxonomy = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY
        self.head_weights = dict(_DEFAULT_HEAD_WEIGHTS)
        if head_weights:
            self.head_weights.update(head_weights)
        self.lambda_structured = lambda_structured
        self.lambda_confidence = lambda_confidence
        self.lambda_temporal = lambda_temporal
        self.lambda_alignment = lambda_alignment
        self.multilabel_loss = multilabel_loss

    # ------------------------------------------------------------------
    # Structured term
    # ------------------------------------------------------------------

    def _structured(
        self,
        prediction: HorizonReasoningPrediction,
        targets: Dict[str, torch.Tensor],
        weights: torch.Tensor,  # [B, 5] per-horizon source×confidence weight
    ) -> torch.Tensor:
        total = weights.new_zeros(())
        for group, attr in _GROUP_TO_ATTR.items():
            if group not in targets:
                continue
            logits = getattr(prediction, attr)          # [B, 5, C]
            target = targets[group]
            hw = self.head_weights.get(group, 1.0)
            if self.taxonomy.mode(group) is LabelMode.MULTI:
                total = total + hw * self._multilabel_term(logits, target, weights)
            else:
                total = total + hw * self._singlelabel_term(logits, target, weights)
        return total

    def _multilabel_term(
        self, logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Multi-label BCE/ASL, per-horizon weighted; target [B,5,C] float."""
        if self.multilabel_loss == "asl":
            elem = _asl_with_logits(logits, target.float())          # [B,5,C]
        else:
            elem = F.binary_cross_entropy_with_logits(
                logits, target.float(), reduction="none"
            )                                                        # [B,5,C]
        per_horizon = elem.mean(dim=-1)                              # [B,5]
        return _weighted_mean(per_horizon, weights)

    def _singlelabel_term(
        self, logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Single-label CE with ignore_index; target [B,5] long, weighted [B,5]."""
        B, H, C = logits.shape
        ce = F.cross_entropy(
            logits.reshape(B * H, C), target.reshape(B * H).long(),
            ignore_index=IGNORE_INDEX, reduction="none",
        ).reshape(B, H)                                              # [B,5]
        # Zero the weight where the target is ignored so masked horizons don't
        # dilute the mean (CE already returns 0 there, but the weight must too).
        valid = (target != IGNORE_INDEX).float()
        return _weighted_mean(ce, weights * valid)

    # ------------------------------------------------------------------
    # Confidence term (Brier)
    # ------------------------------------------------------------------

    def _confidence(
        self,
        prediction: HorizonReasoningPrediction,
        confidence_targets: torch.Tensor,  # [B, 5]
        weights: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.sigmoid(prediction.confidence_logits.squeeze(-1))  # [B,5]
        se = (pred - confidence_targets.float()) ** 2                   # [B,5]
        return _weighted_mean(se, weights)

    # ------------------------------------------------------------------
    # Temporal consistency (weak regulariser)
    # ------------------------------------------------------------------

    def _temporal(
        self,
        prediction: HorizonReasoningPrediction,
        confidence_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        total = prediction.confidence_logits.new_zeros(())
        # Confidence-gate adjacent horizons when a target is available; else 1.
        if confidence_targets is not None:
            c = confidence_targets.float()                              # [B,5]
            gate = c[:, :-1] * c[:, 1:]                                 # [B,4]
        else:
            gate = None
        for group in _TEMPORAL_HEADS:
            logits = getattr(prediction, _GROUP_TO_ATTR[group])         # [B,5,C]
            a, b = logits[:, :-1], logits[:, 1:]                        # [B,4,C]
            if self.taxonomy.mode(group) is LabelMode.MULTI:
                step = (torch.sigmoid(a) - torch.sigmoid(b)).abs().mean(dim=-1)  # [B,4]
            else:
                p = F.log_softmax(a, dim=-1)
                q = F.softmax(b, dim=-1)
                step = F.kl_div(p, q, reduction="none").sum(dim=-1)     # [B,4]
            if gate is not None:
                step = step * gate
            total = total + step.mean()
        return total

    # ------------------------------------------------------------------
    # Alignment term (optional, cosine)
    # ------------------------------------------------------------------

    def _alignment(
        self,
        prediction: HorizonReasoningPrediction,
        teacher_embedding_targets: torch.Tensor,  # [B,5,D]
        weights: torch.Tensor,
    ) -> torch.Tensor:
        student = prediction.student_reasoning_embedding
        if student is None:
            raise ValueError(
                "alignment loss requested but the head produced no "
                "student_reasoning_embedding (build it with teacher_embedding_dim)."
            )
        s = F.normalize(student, dim=-1)
        t = F.normalize(teacher_embedding_targets.float(), dim=-1)
        cos = (s * t).sum(dim=-1)                                       # [B,5]
        return _weighted_mean(1.0 - cos, weights)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        prediction: HorizonReasoningPrediction,
        targets: Dict[str, torch.Tensor],
        source_weights: torch.Tensor,                       # [B, 5]
        confidence_targets: Optional[torch.Tensor] = None,  # [B, 5]
        teacher_embedding_targets: Optional[torch.Tensor] = None,  # [B, 5, D]
    ) -> Dict[str, torch.Tensor]:
        """Compute the reasoning loss.

        Returns a dict with ``"total"`` and per-term scalars for logging. The
        planner (trajectory) loss is added by the training loop; this covers
        only the reasoning terms.
        """
        out: Dict[str, torch.Tensor] = {}
        structured = self._structured(prediction, targets, source_weights)
        out["structured"] = structured
        total = self.lambda_structured * structured

        if confidence_targets is not None:
            conf = self._confidence(prediction, confidence_targets, source_weights)
            out["confidence"] = conf
            total = total + self.lambda_confidence * conf

        if self.lambda_temporal > 0:
            temporal = self._temporal(prediction, confidence_targets)
            out["temporal"] = temporal
            total = total + self.lambda_temporal * temporal

        if self.lambda_alignment > 0 and teacher_embedding_targets is not None:
            align = self._alignment(prediction, teacher_embedding_targets, source_weights)
            out["alignment"] = align
            total = total + self.lambda_alignment * align

        out["total"] = total
        return out


def _weighted_mean(per_horizon: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean of ``[B, 5]`` values by ``[B, 5]`` weights.

    Normalizes by the weight sum (not the count), so abstained/masked horizons
    (weight 0) neither contribute nor dilute. Returns 0 if all weights are 0.
    """
    num = (per_horizon * weights).sum()
    den = weights.sum().clamp(min=1e-8)
    return num / den
