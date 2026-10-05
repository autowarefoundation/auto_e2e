"""Counterfactual cause labels from intervention deltas (#122).

Issue #122 splits in two: (1) intervention as an *evaluation* metric — already
in ``evaluation.faithfulness`` — and (2) using intervention magnitude to
*propose* ``cause`` labels. This module is a small, offline-friendly
implementation of (2).

Given per-candidate trajectory L2 deltas (from any intervention experiment —
horizon token ablations, object-token masks, etc.), it keeps the causes that
actually moved the plan and emits ``ReasoningHorizonLabel`` entries.

Default provenance is ``counterfactual_weak``, not ``counterfactual_gt``.
#122's circularity objection stands: labels from the model's own sensitivity
are not ground truth until they are scored against an external anchor
(KIT LiDAR / HD-map conflict). This PR does not have that data. The
synthetic geometry protocol in ``evaluation.cause_label_unit_protocol``
only smoke-tests the selector; it is not #122 validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .schema import HORIZON_SECONDS, ReasoningHorizonLabel


@dataclass(frozen=True)
class InterventionCauseCandidate:
    """One named intervention and how much it moved the trajectory."""

    name: str
    trajectory_l2: float
    horizon_sec: float = 0.0


def select_causes_by_intervention(
    candidates: Sequence[InterventionCauseCandidate],
    *,
    min_delta: float = 1e-3,
    top_k: Optional[int] = None,
    relative_to_max: Optional[float] = 0.5,
) -> List[str]:
    """Pick cause names whose intervention moved the trajectory enough.

    Args:
        candidates: scored interventions (any order).
        min_delta: absolute L2 floor; below this is treated as a no-op.
        top_k: keep at most this many (highest delta first). ``None`` = no cap.
        relative_to_max: if set, also require ``delta >= relative_to_max * max_delta``
            among candidates that clear ``min_delta``.

    Returns:
        Selected cause names, highest delta first (stable for ties by name).
    """
    ranked = sorted(
        candidates,
        key=lambda c: (-float(c.trajectory_l2), c.name),
    )
    above = [c for c in ranked if float(c.trajectory_l2) >= min_delta]
    if not above:
        return []
    if relative_to_max is not None:
        peak = float(above[0].trajectory_l2)
        floor = relative_to_max * peak
        above = [c for c in above if float(c.trajectory_l2) >= floor]
    if top_k is not None:
        above = above[: max(0, top_k)]
    return [c.name for c in above]


def horizon_labels_from_intervention_deltas(
    per_horizon: Mapping[float, Sequence[InterventionCauseCandidate]],
    *,
    min_delta: float = 1e-3,
    top_k: Optional[int] = 3,
    relative_to_max: Optional[float] = 0.5,
    confidence_scale: float = 1.0,
    provenance: str = "counterfactual_weak",
) -> List[ReasoningHorizonLabel]:
    """Build five (or fewer) horizon labels with counterfactual causes.

    Horizons missing from ``per_horizon`` get an empty cause list and low
    confidence. ``confidence`` is a simple monotone of the top delta
    (clamped to ``[0, 1]``) times ``confidence_scale``.

    Provenance defaults to ``counterfactual_weak``. Pass
    ``provenance="counterfactual_gt"`` only after a real KIT LiDAR /
    HD-map pass (#122) justifies treating the labeller as ground truth.
    The synthetic unit protocol in this PR is not that pass.
    """
    labels: List[ReasoningHorizonLabel] = []
    for h in HORIZON_SECONDS:
        cands = list(per_horizon.get(h, ()))
        causes = select_causes_by_intervention(
            cands, min_delta=min_delta, top_k=top_k, relative_to_max=relative_to_max
        )
        top_delta = max((float(c.trajectory_l2) for c in cands), default=0.0)
        conf = float(min(1.0, max(0.0, top_delta * confidence_scale))) if causes else 0.0
        labels.append(
            ReasoningHorizonLabel(
                horizon_sec=h,
                cause=causes,
                confidence=conf,
                provenance=provenance,
                evidence=(
                    f"intervention deltas: "
                    + ", ".join(f"{c.name}={c.trajectory_l2:.4f}" for c in sorted(
                        cands, key=lambda x: -x.trajectory_l2
                    )[:5])
                    if cands
                    else None
                ),
            )
        )
    return labels


def aggregate_candidate_deltas(
    named_deltas: Mapping[str, float],
    *,
    horizon_sec: float = 0.0,
) -> List[InterventionCauseCandidate]:
    """Convenience: ``{cause_name: trajectory_l2}`` → candidate list."""
    return [
        InterventionCauseCandidate(name=k, trajectory_l2=float(v), horizon_sec=horizon_sec)
        for k, v in named_deltas.items()
    ]
