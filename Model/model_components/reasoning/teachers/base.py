"""Abstract teacher interface for reasoning-band pseudo-label generation (issue #98).

Teachers are TRAIN-ONLY, offline autolabellers.  They are never instantiated
at inference time and must not appear in the model's forward pass.

Usage pattern in a training loop::

    teacher = Qwen2VLTeacher(taxonomy)
    # … for each batch …
    targets = teacher.label(frames, num_future_horizons=4)
    # targets["maneuver"][0] is a [B, 7] float tensor (multi-label, 0/1 or soft)
    loss = reasoning_loss(student_logits, targets, weight=1.0)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import torch

from ..scenario_taxonomy import ScenarioTaxonomy, DEFAULT_TAXONOMY

# ReasoningTargets[group_name][horizon_index] = float target tensor [B, num_classes]
# Values in [0, 1]: hard (0/1) for deterministic / soft for VLM confidence scores.
ReasoningTargets = Dict[str, List[torch.Tensor]]


class VLMTeacher(ABC):
    """Abstract base class for scenario pseudo-label teachers.

    All teachers share the same :meth:`label` interface so the training loop
    can swap backends without code changes.

    Args:
        taxonomy: the label registry used to determine group names and class
            counts.  Defaults to :data:`DEFAULT_TAXONOMY`.

    Extension point — Alpamayo CoC autolabeller (v2):
        Implement a subclass that calls the NVlabs CoC autolabeller API,
        mapping its chain-of-causation output onto the taxonomy groups.
        Register the class in
        ``model_components.reasoning.teachers._TEACHER_REGISTRY`` under the
        key ``"alpamayo_coc"`` so users can select it via a config string.
        **Do NOT implement CoC here** — this extension point is intentionally
        left as a docstring-only placeholder for a future contributor.
    """

    def __init__(
        self, taxonomy: Optional[ScenarioTaxonomy] = None
    ) -> None:
        self.taxonomy = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY

    @abstractmethod
    def label(
        self,
        frames: Sequence[torch.Tensor],
        num_future_horizons: int = 4,
    ) -> ReasoningTargets:
        """Produce multi-label targets for the current scene and future horizons.

        Args:
            frames: sequence of image tensors representing the current and
                (optionally) future front-camera frames.  The first element
                is the current frame; subsequent elements correspond to future
                horizons.  Each tensor is ``[B, C, H, W]`` or ``[B, 3, H, W]``.
            num_future_horizons: number of future horizons to label (must be
                consistent with the reasoning band's configuration).

        Returns:
            :data:`ReasoningTargets` — a dict mapping each group name to a
            list of ``1 + num_future_horizons`` float tensors of shape
            ``[B, num_classes]`` in ``[0, 1]``.  Hard labels use 0.0 / 1.0;
            soft labels from a VLM use confidence scores.
        """
        raise NotImplementedError
