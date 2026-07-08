"""Agreement-based fusion of several teachers (issue #98/#103).

Fusing multiple annotators with per-label agreement substantially improves
multi-label pseudo-label quality over the best single VLM (+47-55% F1 in a
dashcam multi-label study, arXiv:2510.01126; the multi-teacher pattern is also
how Hydra-MDP won the NAVSIM challenge, arXiv:2406.06978).  The fused target
is the **fraction of teachers that agree** on each label — which doubles as
the soft confidence signal for the per-horizon confidence head (#103), instead
of trusting any single teacher's (systematically over-confident) self-report.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..scenario_taxonomy import ScenarioTaxonomy
from .base import VLMTeacher, ReasoningTargets


class MultiTeacher(VLMTeacher):
    """Fuse the targets of several teachers into agreement-fraction targets.

    Args:
        teachers: two or more :class:`VLMTeacher` instances.  They must share
            the taxonomy (same groups, same class counts).
        taxonomy: label registry; defaults to the first teacher's taxonomy.

    Example::

        fused = MultiTeacher([qwen_teacher, videollama_teacher])
        targets = fused.label(frames, num_future_horizons=4)
        # targets[group][h] values are agreement fractions in {0, 0.5, 1.0}
        # for two teachers — usable both as soft labels and as confidence.
    """

    def __init__(
        self,
        teachers: Sequence[VLMTeacher],
        taxonomy: Optional[ScenarioTaxonomy] = None,
    ) -> None:
        if len(teachers) < 2:
            raise ValueError(
                f"MultiTeacher needs at least 2 teachers, got {len(teachers)}."
            )
        super().__init__(taxonomy if taxonomy is not None else teachers[0].taxonomy)
        for t in teachers:
            if t.taxonomy.group_names != self.taxonomy.group_names:
                raise ValueError(
                    "All teachers must share the same taxonomy groups; got "
                    f"{t.taxonomy.group_names} vs {self.taxonomy.group_names}."
                )
            # Group names matching is not enough: agreement is computed
            # index-by-index, so every teacher must use the SAME label tuple in
            # the SAME order — otherwise index i means a different class per
            # teacher and the fused target is semantically invalid.
            for group in self.taxonomy.groups:
                if t.taxonomy[group.name].labels != group.labels:
                    raise ValueError(
                        f"Teacher taxonomy mismatch in group '{group.name}': "
                        "label tuples differ in content or order "
                        f"({t.taxonomy[group.name].labels} vs {group.labels}). "
                        "Agreement fusion requires identical label ordering so "
                        "index i is the same class for every teacher."
                    )
        self.teachers = list(teachers)

    def label(
        self,
        frames: Sequence[torch.Tensor],
        num_future_horizons: int = 4,
    ) -> ReasoningTargets:
        """Average the member teachers' targets (per label, per horizon).

        Returns:
            :data:`ReasoningTargets` whose values are the mean of the member
            targets — for hard {0,1} members this is the agreement fraction.
        """
        all_targets = [
            t.label(frames, num_future_horizons=num_future_horizons)
            for t in self.teachers
        ]

        fused: ReasoningTargets = {}
        for group in self.taxonomy.groups:
            horizons = len(all_targets[0][group.name])
            fused[group.name] = [
                torch.stack(
                    [targets[group.name][h] for targets in all_targets]
                ).mean(dim=0)
                for h in range(horizons)
            ]
        return fused
