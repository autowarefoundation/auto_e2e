"""Typed output of the horizon-aware reasoning head (issue #98).

Kept in its own module (no nn import beyond torch tensors) so the planner and
the training loop can depend on the output contract without importing the head
implementation. The planner requires only ``reasoning_latent`` (and, in
horizon-cross-attention mode, ``horizon_tokens``); every other field is for
training, metrics, debugging, and visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HorizonReasoningPrediction:
    """One reasoning-head forward pass (1 Hz tick).

    Shapes use ``H = 5`` horizons (now, +1s, +2s, +3s, +4s) and ``embed=256``.

    Attributes:
        horizon_tokens: ``[B, 5, 256]`` per-horizon representation — the
            horizon-aware planner interface (preserves *when* a hazard matters).
        reasoning_latent: ``[B, 256]`` pooled latent — the compact planner
            interface for the ``pooled_latent`` coupling mode.
        *_logits: per-group structured logits ``[B, 5, C]`` (single-label
            groups → cross-entropy; multi-label → BCE/ASL). C comes from the
            taxonomy.
        confidence_logits: ``[B, 5, 1]`` per-horizon confidence (raw logits;
            ``sigmoid`` for a probability). Trained via a Brier target.
        student_reasoning_embedding: optional ``[B, 5, D_teacher]`` for the
            training-only teacher-embedding alignment loss (None unless enabled).
    """

    horizon_tokens: torch.Tensor
    reasoning_latent: torch.Tensor

    relation_to_ego_logits: torch.Tensor
    hazard_event_logits: torch.Tensor
    cause_logits: torch.Tensor
    longitudinal_response_logits: torch.Tensor
    lateral_response_logits: torch.Tensor
    tactical_response_logits: torch.Tensor
    rule_response_logits: torch.Tensor

    confidence_logits: torch.Tensor

    student_reasoning_embedding: Optional[torch.Tensor] = None
