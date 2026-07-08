"""Typed schema for reasoning label artifacts (issue #98, R4/R5).

Defines the offline label objects that a teacher produces and training
consumes: a per-horizon label (:class:`ReasoningHorizonLabel`), a per-sample
record of exactly five horizons plus provenance
(:class:`ReasoningLabelRecord`), and the batched training-target container
(:class:`ReasoningTargetBatch`).

These are plain dataclasses (no torch dependency at definition time) so the
schema can be validated, serialized to JSONL/Parquet, and unit-tested without
a GPU. Tensorization for the loss happens in :class:`ReasoningTargetBatch`.

Contract (validated in :mod:`.validators`):
    * every successful record has EXACTLY five horizons (0,1,2,3,4 s), in order;
    * labels are drawn only from the taxonomy (unknown_* to abstain within a
      group);
    * an abstained record carries ``abstained=True`` + ``teacher_error`` and is
      masked out of the loss (never silently turned into all-zero labels, R9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import torch

# The five horizons every successful record must carry, in seconds, in order.
HORIZON_SECONDS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)
NUM_HORIZONS: int = len(HORIZON_SECONDS)

SCHEMA_VERSION: str = "reasoning_label_v2"


@dataclass
class ReasoningHorizonLabel:
    """Action-relevant labels for one future horizon.

    The action-relevant core (required) answers what relates to ego, what the
    hazard is, why (cause), and how ego should respond. Multi-label groups hold
    a list of active label strings; single-label groups hold one string (or
    None to abstain). Optional v2 context/timing fields default to None so v1
    artifacts stay small.

    Attributes:
        horizon_sec: which horizon this is (one of :data:`HORIZON_SECONDS`).
        relation_to_ego: single-label — how the salient object relates to ego.
        hazard_event: multi-label — active hazards.
        cause: multi-label — active causes.
        longitudinal_response / lateral_response / tactical_response /
            rule_response: single-label response axes.
        confidence: teacher confidence for this horizon in [0, 1].
        provenance: label source (audited_gt / direct_gt / derived_gt /
            teacher_gt / weak_gt / counterfactual_gt / teacher_error).
        evidence: optional free-text rationale — audit/debug only, never a
            training signal.
        (optional v2 fields): scene/topology/actor context + timing regressions.
    """

    horizon_sec: float

    # Action-relevant core.
    relation_to_ego: Optional[str] = None
    hazard_event: List[str] = field(default_factory=list)
    cause: List[str] = field(default_factory=list)

    longitudinal_response: Optional[str] = None
    lateral_response: Optional[str] = None
    tactical_response: Optional[str] = None
    rule_response: Optional[str] = None

    confidence: float = 0.0
    provenance: str = "teacher_gt"

    evidence: Optional[str] = None

    # Optional v2 context (multi-label) — None means "not labelled".
    global_scene_context: Optional[List[str]] = None
    ego_mission_context: Optional[List[str]] = None
    road_topology: Optional[List[str]] = None
    lane_topology: Optional[List[str]] = None
    traffic_control: Optional[List[str]] = None
    right_of_way: Optional[List[str]] = None
    dynamic_actor_type: Optional[List[str]] = None
    actor_state: Optional[List[str]] = None
    actor_intent: Optional[List[str]] = None
    interaction_type: Optional[List[str]] = None

    # Optional v2 timing regressions (seconds) — None means "not labelled".
    time_to_conflict: Optional[float] = None
    time_to_collision: Optional[float] = None
    time_to_stop_line: Optional[float] = None


@dataclass
class ReasoningLabelRecord:
    """One sample's reasoning labels plus full artifact provenance (R4).

    A successful record holds exactly :data:`NUM_HORIZONS` horizons in order.
    An abstained record (endpoint/parse/schema failure under ``strict=False``)
    sets ``abstained=True`` and ``teacher_error``; its horizons are masked out
    of the loss rather than converted to all-zero labels (R9).
    """

    schema_version: str
    sample_id: str
    timestamp: float
    dataset_name: str

    teacher_provider: str
    teacher_model: str
    prompt_version: str
    request_mode: str

    horizons: List[ReasoningHorizonLabel]

    # Extended provenance (R4) — recorded so training can filter/weight.
    dataset_version: Optional[str] = None
    teacher_endpoint_type: Optional[str] = None
    labeler_version: Optional[str] = None
    provenance: str = "teacher_gt"
    created_at: Optional[str] = None

    abstained: bool = False
    teacher_error: Optional[str] = None

    @classmethod
    def abstain(
        cls,
        *,
        sample_id: str,
        dataset_name: str,
        teacher_provider: str,
        teacher_model: str,
        prompt_version: str,
        request_mode: str,
        teacher_error: str,
        timestamp: float = 0.0,
    ) -> "ReasoningLabelRecord":
        """Build an explicitly-abstained record (R9).

        Carries no horizons and marks the failure provenance, so downstream
        training masks the sample out instead of learning all-zero labels.
        """
        return cls(
            schema_version=SCHEMA_VERSION,
            sample_id=sample_id,
            timestamp=timestamp,
            dataset_name=dataset_name,
            teacher_provider=teacher_provider,
            teacher_model=teacher_model,
            prompt_version=prompt_version,
            request_mode=request_mode,
            horizons=[],
            provenance="teacher_error",
            abstained=True,
            teacher_error=teacher_error,
        )


@dataclass
class ReasoningTargetBatch:
    """Batched, tensorized training targets consumed by the loss.

    Produced by collating :class:`ReasoningLabelRecord` rows. Shapes mirror the
    student head's per-horizon logits so the loss can zip them directly:

        * multi-label groups: ``[B, 5, C]`` float in {0, 1} (or soft in [0,1]).
        * single-label groups: ``[B, 5]`` long class indices, ``ignore_index``
          (-100) where the target is missing/abstained.
        * confidence_targets: ``[B, 5]`` float in [0, 1].
        * source_weights: ``[B, 5]`` float — provenance × label-confidence,
          0.0 for abstained/masked horizons.
        * teacher_embedding_targets: optional ``[B, 5, D]`` for the alignment
          loss.

    Kept as a dict-of-tensors container rather than an nn.Module: it holds no
    parameters and is assembled in the data pipeline, not the model graph.
    """

    # group name -> [B, 5, C] (multi-label) or [B, 5] long (single-label)
    targets: Dict[str, "torch.Tensor"]
    confidence_targets: "torch.Tensor"          # [B, 5]
    source_weights: "torch.Tensor"              # [B, 5]
    teacher_embedding_targets: Optional["torch.Tensor"] = None  # [B, 5, D]

    IGNORE_INDEX: int = -100
