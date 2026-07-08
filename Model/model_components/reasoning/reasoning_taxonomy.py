"""Compositional, action-relevant reasoning taxonomy (issue #98, v2).

Single source of truth for the reasoning label space, shared by the runtime
student head (``model_components``) and the offline label-generation pipeline
(``data_processing``). It carries no teacher dependency, so importing it never
pulls in a VLM client.

Design (per the Horizon-Aware Action-Relevant Reasoning Head proposal in #98):

* The label space is NOT a flat scene-fact enum. It is a set of independent
  axes ("groups"), each answering a distinct question — what relates to ego,
  what the hazard is, why (cause), and how ego should respond
  (longitudinal / lateral / tactical / rule).
* Each group is either MULTI-label (several labels can be active — hazard,
  cause, scene context) or SINGLE-label (mutually exclusive — relation, the
  four response axes). The mode is part of the loss contract: multi-label
  groups train with BCE/ASL, single-label with cross-entropy.
* Label ORDER within a group is part of the loss contract: index ``i`` is a
  fixed class. Append only; never insert or reorder. This lets the taxonomy
  grow (new datasets, v2 optional axes) without invalidating trained weights
  or stored label artifacts.
* Every group carries an ``unknown_*`` label so a teacher can abstain WITHIN a
  group without emitting an all-zero (falsely-negative) row.

This supersedes the v1 three-axis taxonomy (``maneuver`` / ``edge_case`` /
``weather_env``): ``maneuver`` is the "action" label the WG flagged as least
additive (the trajectory already encodes it), and ``weather_env`` is not
action-relevant (what matters is ``low_friction_risk`` as a hazard, not "it is
raining"). Research phase — the old taxonomy is removed, not wrapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence


class LabelMode(str, Enum):
    """Whether a group's labels are mutually exclusive.

    * ``MULTI``  — several labels can be active at once (hazard, cause, scene
      context). Trained with BCE / Asymmetric Loss over per-class sigmoids.
    * ``SINGLE`` — exactly one label is the answer (relation-to-ego, the four
      response axes). Trained with cross-entropy over the class logits.
    """

    MULTI = "multi"
    SINGLE = "single"


@dataclass(frozen=True)
class TaxonomyGroup:
    """One axis in the reasoning taxonomy.

    Args:
        name: unique identifier (e.g. ``"cause"``).
        labels: ordered tuple of class names. Index order is part of the loss
            contract — append only, never insert or reorder.
        mode: :class:`LabelMode` — multi-label (BCE/ASL) or single-label (CE).

    Every group must contain at least one ``unknown_*`` (or ``no_*`` / ``none``)
    label so a teacher can abstain within the group without an all-zero row;
    this is validated at construction.
    """

    name: str
    labels: tuple[str, ...]
    mode: LabelMode = LabelMode.MULTI

    def __post_init__(self) -> None:
        if len(self.labels) == 0:
            raise ValueError(f"TaxonomyGroup '{self.name}' has no labels.")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(
                f"TaxonomyGroup '{self.name}' contains duplicate labels: {self.labels}"
            )
        # An abstain label is required so a teacher can decline within a group
        # rather than forcing an all-zero (falsely-negative) row.
        if not any(
            lbl.startswith("unknown") or lbl.startswith("no_") or lbl == "none"
            for lbl in self.labels
        ):
            raise ValueError(
                f"TaxonomyGroup '{self.name}' must include an abstain label "
                f"(unknown_*, no_*, or 'none'); got {self.labels}."
            )

    def __len__(self) -> int:
        return len(self.labels)

    def index(self, label: str) -> int:
        """Return the stable index of *label* (raises ``KeyError`` if absent)."""
        try:
            return self.labels.index(label)
        except ValueError:
            raise KeyError(
                f"Label '{label}' not found in group '{self.name}'. "
                f"Known labels: {self.labels}"
            )


# ---------------------------------------------------------------------------
# Minimum v1 label groups — the action-relevant core (do NOT reorder entries;
# index is part of the loss contract). Optional v2 context/timing axes append
# via register_group without touching these indices.
# ---------------------------------------------------------------------------

# How the salient object relates to the ego path (single-label).
_RELATION_TO_EGO: tuple[str, ...] = (
    "same_lane_ahead",
    "same_lane_behind",
    "left_adjacent",
    "right_adjacent",
    "crossing_path",
    "about_to_cross_path",
    "merging_into_ego_path",
    "cutting_into_ego_path",
    "oncoming_conflict",
    "intersection_conflict",
    "blocking_current_lane",
    "blocking_target_lane",
    "blocking_route",
    "occluded_near_path",
    "outside_path",
    "behind_ego",
    "unknown_relation",
)

# What risk is present (multi-label).
_HAZARD_EVENT: tuple[str, ...] = (
    "no_hazard",
    "collision_risk",
    "vru_collision_risk",
    "cut_in_risk",
    "merge_conflict_risk",
    "right_of_way_violation_risk",
    "red_light_violation_risk",
    "blocked_route_risk",
    "occlusion_risk",
    "low_friction_risk",
    "emergency_vehicle_risk",
    "unknown_hazard",
)

# Why the planner may need to change behaviour (multi-label; hardest to label).
_CAUSE: tuple[str, ...] = (
    "lead_vehicle",
    "slow_lead_vehicle",
    "stopped_lead_vehicle",
    "cut_in_vehicle",
    "cross_traffic",
    "oncoming_vehicle",
    "pedestrian_crossing",
    "pedestrian_about_to_cross",
    "vru_conflict",
    "red_light",
    "yellow_light",
    "stop_sign",
    "yield_sign",
    "human_direction",
    "route_turn",
    "route_merge",
    "route_lane_change",
    "lane_ending",
    "object_blocking_path",
    "blocked_lane",
    "road_closed",
    "construction_blocking_path",
    "occlusion",
    "poor_visibility",
    "slippery_road",
    "uncertainty_high",
    "unknown_cause",
)

# How ego should respond along the driving axis (single-label).
_LONGITUDINAL_RESPONSE: tuple[str, ...] = (
    "keep_speed",
    "accelerate",
    "coast",
    "slow_down",
    "prepare_stop",
    "stop",
    "stay_stopped",
    "creep",
    "yield",
    "follow_lead_vehicle",
    "increase_gap",
    "emergency_brake",
    "unknown_longitudinal",
)

# How ego should respond laterally (single-label).
_LATERAL_RESPONSE: tuple[str, ...] = (
    "keep_lane",
    "nudge_left",
    "nudge_right",
    "shift_left_within_lane",
    "shift_right_within_lane",
    "lane_change_left",
    "lane_change_right",
    "avoid_left",
    "avoid_right",
    "return_to_lane",
    "pull_over",
    "reverse",
    "unknown_lateral",
)

# The tactical decision (single-label).
_TACTICAL_RESPONSE: tuple[str, ...] = (
    "proceed",
    "proceed_with_caution",
    "wait",
    "wait_for_gap",
    "wait_for_actor",
    "wait_for_signal",
    "creep_for_visibility",
    "negotiate_merge",
    "negotiate_unprotected_turn",
    "yield_then_proceed",
    "stop_then_proceed",
    "reroute_or_wait",
    "unknown_tactical",
)

# The rule-level obligation (single-label).
_RULE_RESPONSE: tuple[str, ...] = (
    "none",
    "wait_for_green",
    "stop_at_stop_line",
    "stop_before_crosswalk",
    "yield_to_vru",
    "yield_to_oncoming",
    "yield_to_cross_traffic",
    "yield_to_emergency_vehicle",
    "obey_human_direction",
    "respect_speed_limit",
    "slow_for_school_zone",
    "slow_for_construction_zone",
    "do_not_enter",
    "do_not_turn",
    "unknown_rule",
)


# The canonical v1 groups in a fixed order. Each entry is (name, labels, mode).
_V1_GROUPS: tuple[tuple[str, tuple[str, ...], LabelMode], ...] = (
    ("relation_to_ego", _RELATION_TO_EGO, LabelMode.SINGLE),
    ("hazard_event", _HAZARD_EVENT, LabelMode.MULTI),
    ("cause", _CAUSE, LabelMode.MULTI),
    ("longitudinal_response", _LONGITUDINAL_RESPONSE, LabelMode.SINGLE),
    ("lateral_response", _LATERAL_RESPONSE, LabelMode.SINGLE),
    ("tactical_response", _TACTICAL_RESPONSE, LabelMode.SINGLE),
    ("rule_response", _RULE_RESPONSE, LabelMode.SINGLE),
)


class ReasoningTaxonomy:
    """Registry of reasoning label groups (the compositional label space).

    The seven canonical action-relevant groups are registered at construction.
    Optional v2 context/timing axes append via :meth:`register_group` or
    :meth:`extend` without breaking any existing index.

    Example::

        tax = ReasoningTaxonomy()
        tax.group_names()                    # ['relation_to_ego', 'hazard_event', ...]
        tax.num_classes("cause")             # 27
        tax.index("cause", "red_light")      # stable index
        tax.mode("hazard_event")             # LabelMode.MULTI
    """

    def __init__(self) -> None:
        self._groups: Dict[str, TaxonomyGroup] = {}
        for name, labels, mode in _V1_GROUPS:
            self.register_group(name, labels, mode)

    # ------------------------------------------------------------------
    # Extension API (append-only)
    # ------------------------------------------------------------------

    def register_group(
        self, name: str, labels: Sequence[str], mode: LabelMode = LabelMode.MULTI
    ) -> TaxonomyGroup:
        """Register a new label group.

        Raises:
            ValueError: if *name* is already registered.
        """
        if name in self._groups:
            raise ValueError(
                f"Group '{name}' is already registered. Use extend() to append labels."
            )
        group = TaxonomyGroup(name=name, labels=tuple(labels), mode=mode)
        self._groups[name] = group
        return group

    def extend(self, name: str, new_labels: Sequence[str]) -> TaxonomyGroup:
        """Append *new_labels* to an existing group (append-only, index-stable)."""
        if name not in self._groups:
            raise KeyError(f"Group '{name}' is not registered. Call register_group() first.")
        existing = self._groups[name]
        overlap = set(new_labels) & set(existing.labels)
        if overlap:
            raise ValueError(
                f"Labels {sorted(overlap)} already exist in group '{name}'."
            )
        updated = TaxonomyGroup(
            name=name, labels=existing.labels + tuple(new_labels), mode=existing.mode
        )
        self._groups[name] = updated
        return updated

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> TaxonomyGroup:
        try:
            return self._groups[name]
        except KeyError:
            raise KeyError(
                f"Group '{name}' not found. Registered groups: {list(self._groups)}"
            )

    def __contains__(self, name: object) -> bool:
        return name in self._groups

    @property
    def groups(self) -> List[TaxonomyGroup]:
        """All registered groups in insertion order."""
        return list(self._groups.values())

    def group_names(self) -> List[str]:
        """Names of all registered groups in insertion order."""
        return list(self._groups.keys())

    def labels(self, group: str) -> tuple[str, ...]:
        """Ordered label tuple for *group*."""
        return self[group].labels

    def mode(self, group: str) -> LabelMode:
        """Label mode (multi / single) for *group*."""
        return self[group].mode

    def num_classes(self, group: str) -> int:
        """Number of classes in *group*."""
        return len(self[group])

    def index(self, group: str, label: str) -> int:
        """Stable index of *label* within *group*."""
        return self[group].index(label)

    def total_classes(self) -> int:
        """Total classes across all groups."""
        return sum(len(g) for g in self._groups.values())

    def validate_exact_match(self, other: "ReasoningTaxonomy") -> None:
        """Assert *other* is label-for-label identical to this taxonomy.

        Required before fusing multiple teachers (issue #98 R10): group-name
        equality is not enough — agreement is computed index-by-index, so every
        taxonomy must share the SAME groups with the SAME labels in the SAME
        order and the SAME mode. Otherwise index ``i`` means a different class
        and the fused target is semantically invalid.

        Raises:
            ValueError: on any mismatch (groups, order, labels, or mode).
        """
        if self.group_names() != other.group_names():
            raise ValueError(
                f"Taxonomy group mismatch: {self.group_names()} vs {other.group_names()}."
            )
        for name in self.group_names():
            a, b = self[name], other[name]
            if a.labels != b.labels:
                raise ValueError(
                    f"Group '{name}' label mismatch (content or order): "
                    f"{a.labels} vs {b.labels}."
                )
            if a.mode != b.mode:
                raise ValueError(
                    f"Group '{name}' mode mismatch: {a.mode} vs {b.mode}."
                )


# Module-level default taxonomy instance — shared across the package.
DEFAULT_TAXONOMY: ReasoningTaxonomy = ReasoningTaxonomy()
