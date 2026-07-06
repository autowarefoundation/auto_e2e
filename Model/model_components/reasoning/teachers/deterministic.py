"""Deterministic (keyword-rule) teacher for CI and offline testing (issue #98).

This teacher uses only CPU arithmetic — no neural network, no GPU, no heavy
dependencies.  It is the recommended backend for unit tests and CI pipelines.

The label logic is purely keyword-based: the caller supplies a dict of active
label names per group (``active_labels``), and the teacher constructs the
corresponding one-hot target tensors.  If no active labels are provided for
a group, the teacher returns all-zero targets (no class active).

This deterministic behaviour makes it trivial to construct expected outputs
in test assertions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..scenario_taxonomy import ScenarioTaxonomy
from .base import VLMTeacher, ReasoningTargets


class DeterministicTeacher(VLMTeacher):
    """Keyword/rule-based teacher that produces deterministic multi-label targets.

    Args:
        taxonomy: label registry.  Defaults to :data:`DEFAULT_TAXONOMY`.
        active_labels: dict mapping group names to lists of label strings that
            should be set to 1.0 in the target tensor.  Any group not present
            defaults to all-zero targets.  Labels not in the taxonomy group
            raise a ``KeyError``.

    Example::

        teacher = DeterministicTeacher(
            active_labels={
                "maneuver": ["turn_left"],
                "edge_case": ["avoid_roadworks"],
                "weather_env": ["rain_night"],
            }
        )
        # Returns the same targets regardless of the frames argument.
        targets = teacher.label(frames=[frame], num_future_horizons=4)
    """

    def __init__(
        self,
        taxonomy: Optional[ScenarioTaxonomy] = None,
        active_labels: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__(taxonomy)
        self.active_labels: Dict[str, List[str]] = active_labels or {}

        # Validate at construction time so errors surface early.
        for group_name, labels in self.active_labels.items():
            group = self.taxonomy[group_name]
            for label in labels:
                group.index(label)  # raises KeyError if not found

    def label(
        self,
        frames: Sequence[torch.Tensor],
        num_future_horizons: int = 4,
    ) -> ReasoningTargets:
        """Return deterministic targets derived from ``active_labels``.

        The targets are identical for all horizons and for all frames in the
        batch (keyword rules do not depend on image content).

        Args:
            frames: sequence of image tensors.  Shape ``[B, C, H, W]`` for
                each frame.  Only the batch size is read; pixel values are
                ignored.
            num_future_horizons: number of future horizons.

        Returns:
            :data:`ReasoningTargets` with ``1 + num_future_horizons`` horizon
            entries per group, each a ``[B, num_classes]`` float tensor in
            ``{0.0, 1.0}``.
        """
        if not frames:
            raise ValueError("frames must be non-empty.")

        B = frames[0].shape[0]
        device = frames[0].device
        total_horizons = 1 + num_future_horizons

        out: ReasoningTargets = {}
        for group in self.taxonomy.groups:
            active = self.active_labels.get(group.name, [])
            target = torch.zeros(B, len(group), device=device)
            for label in active:
                idx = group.index(label)
                target[:, idx] = 1.0
            out[group.name] = [target.clone() for _ in range(total_horizons)]

        return out
