"""Source-preserving evidence normalization, label fusion, and event building."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .ontology import ONTOLOGY, ontology_sha256
from .schema import (
    CandidateValue,
    EventInstance,
    EventPhase,
    EvidenceReference,
    LabelEvidence,
    LabelObservation,
    LabelScope,
    Measurement,
    SemanticLabelerProvenance,
    canonical_json_bytes,
    coalesce_observations,
    content_sha256,
    make_observation,
)


FUSION_VERSION = "odd_source_fusion_v1"
EVENT_SEGMENTER_VERSION = "odd_event_segmenter_v2"
EVENT_JOIN_GAP_NS = 1_000_000_000
EVENT_ONSET_QUANTUM_NS = 100_000_000

UNION_KEYS = {
    "odd.road.edge_type_present",
    "odd.road.lane_type_present",
    "odd.road.special_structure",
    "odd.road.workzone_state",
    "odd.traffic_control.present",
    "odd.environment.visibility_degradation",
    "odd.environment.glare",
    "odd.dynamic.agent_type_present",
    "perception.occlusion.source",
    "perception.visual.lighting",
    "perception.visual.glare",
    "perception.image.weather_artifact",
    "perception.image.lens_contamination",
}

EVENT_BACKGROUND_VALUES = {
    "event.ego.motion_state": {"stopped", "moving", "creeping"},
    "event.ego.maneuver": {"lane_follow"},
    "event.ego.strong_response": {"none"},
    "event.vehicle.interaction": {"none"},
    "event.vru.interaction": {"none"},
    "event.traffic_control.response": {"no_response_required"},
    "event.right_of_way": {"not_applicable"},
    "event.hazard.type": {"none"},
    "event.hazard.response": {"none"},
    "event.traffic_flow": {"none"},
    "event.interaction.actor": {"none"},
}

EVENT_PRIMARY_PRIORITY = (
    "event.hazard.type",
    "event.vehicle.interaction",
    "event.vru.interaction",
    "event.ego.strong_response",
    "event.traffic_flow",
    "event.traffic_control.response",
    "event.ego.maneuver",
    "event.ego.motion_state",
)

EVENT_CONTEXT_KEYS = {
    "event.right_of_way",
    "event.hazard.response",
    "event.interaction.actor",
    "event.outcome",
}


@dataclasses.dataclass(frozen=True)
class EvidenceBuildContext:
    dataset_name: str
    dataset_version: str
    dataset_manifest_sha256: str
    capability_manifest_sha256: str
    source_artifact_uri: str
    source_artifact_sha256: str
    labeler_image_digest: str
    labeler_source_revision: str

    def __post_init__(self) -> None:
        if not self.source_artifact_uri:
            raise ValueError("source artifact URI is required")
        for value, name in (
            (self.dataset_manifest_sha256, "dataset manifest"),
            (self.capability_manifest_sha256, "capability manifest"),
            (self.source_artifact_sha256, "source artifact"),
        ):
            if len(value) != 64:
                raise ValueError(f"{name} SHA-256 is invalid")
        if not self.labeler_image_digest.startswith("sha256:"):
            raise ValueError("labeler image must be pinned by digest")
        if not self.labeler_source_revision:
            raise ValueError("labeler source revision is required")


@dataclasses.dataclass(frozen=True)
class ResolvedSceneLabels:
    evidence: tuple[LabelEvidence, ...]
    observations: tuple[LabelObservation, ...]
    events: tuple[EventInstance, ...]


def _measurement_unit(name: str) -> str:
    explicit = {
        "ego_speed_kph": "km/h",
        "ego_speed_mps": "m/s",
        "longitudinal_acceleration_mps2": "m/s^2",
        "yaw_rate_radps": "rad/s",
        "route_lateral_distance_m": "m",
        "input_frame_count": "count",
        "supporting_camera_count": "count",
    }
    if name in explicit:
        return explicit[name]
    if name.endswith("_count"):
        return "count"
    if name.endswith(("_fraction", "_ratio", "_density")):
        return "ratio"
    if name.endswith("_ns"):
        return "ns"
    return "scalar"


def _measurement_aggregation(source: str, name: str) -> str:
    if source == "image_qc":
        return "frame"
    if source == "gnss_ins" and name not in {
        "input_frame_count",
        "supporting_camera_count",
    }:
        return "interval_median"
    return "interval"


def _scope(
    observation: LabelObservation,
    context: EvidenceBuildContext,
) -> LabelScope:
    supporting_cameras = observation.provenance.get("supporting_cameras", ())
    if not isinstance(supporting_cameras, (list, tuple)):
        supporting_cameras = ()
    camera_ids = tuple(
        str(value)
        for value in supporting_cameras
        if isinstance(value, str) and value
    )
    if observation.camera_id:
        camera_ids = (*camera_ids, observation.camera_id)
    if observation.actor_track_uid and observation.camera_id:
        subject_type = "actor"
        subject_id = observation.actor_track_uid
    elif observation.actor_track_uid:
        subject_type = "actor"
        subject_id = observation.actor_track_uid
    elif observation.camera_id:
        subject_type = "camera"
        subject_id = observation.camera_id
    else:
        subject_type = "scene"
        subject_id = None
    return LabelScope(
        dataset_name=context.dataset_name,
        dataset_version=context.dataset_version,
        scene_uid=observation.scene_uid,
        start_timestamp_ns=observation.start_timestamp_ns,
        end_timestamp_ns=observation.end_timestamp_ns,
        subject_type=subject_type,
        subject_id=subject_id,
        anchor_timestamp_ns=observation.start_timestamp_ns,
        camera_ids=camera_ids,
        coordinate_frame=(
            "ego_flu"
            if observation.source in {"gnss_ins", "map_route", "can_optional"}
            else None
        ),
    )


def _candidate_values(
    observation: LabelObservation,
) -> tuple[CandidateValue, ...]:
    raw = observation.provenance.get("candidate_values", ())
    if not isinstance(raw, (list, tuple)):
        return ()
    candidates: list[CandidateValue] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        value = item.get("value")
        score = item.get("score")
        if isinstance(value, str) and isinstance(score, (float, int)):
            candidates.append(
                CandidateValue(
                    value=value,
                    score=float(score),
                    evidence_ref=(
                        str(item["evidence_ref"])
                        if item.get("evidence_ref")
                        else None
                    ),
                )
            )
    return tuple(candidates)


def _semantic_provenance(
    observation: LabelObservation,
    context: EvidenceBuildContext,
) -> SemanticLabelerProvenance:
    details = dict(observation.provenance)
    labeler_version = str(
        details.get("labeler_version")
        or details.get("prompt_version")
        or FUSION_VERSION
    )
    config_fields = {
        key: details[key]
        for key in (
            "schema_version",
            "labeler_version",
            "prompt_version",
            "task_bundle",
            "stationary_epsilon_kph",
            "model",
            "model_revision",
        )
        if key in details
    }
    config_fields["source"] = observation.source
    config_sha256 = content_sha256(config_fields)

    model_name = str(details.get("model") or "")
    model_revision = str(details.get("model_revision") or "")
    model_provider = str(details.get("model_provider") or "")
    if observation.source == "vlm" and not model_provider:
        model_provider = "openai_compatible"
    model_fields: dict[str, str | None] = {
        "model_provider": None,
        "model_name": None,
        "model_revision": None,
    }
    if model_provider and model_name and model_revision:
        model_fields = {
            "model_provider": model_provider,
            "model_name": model_name,
            "model_revision": model_revision,
        }
    prompt_sha256 = details.get("prompt_sha256")
    if (
        observation.source != "vlm"
        or not isinstance(prompt_sha256, str)
        or len(prompt_sha256) != 64
    ):
        prompt_sha256 = None
    decoding_config_sha256 = details.get("decoding_config_sha256")
    if (
        observation.source != "vlm"
        or not isinstance(decoding_config_sha256, str)
        or len(decoding_config_sha256) != 64
    ):
        decoding_config_sha256 = None

    return SemanticLabelerProvenance(
        labeler_name={
            "map_route": "map_route_labeler",
            "gnss_ins": "gnss_ins_labeler",
            "image_qc": "image_qc_labeler",
            "vlm": "openai_compatible_road_observer",
            "can_optional": "can_optional_labeler",
            "fusion": "fusion_labeler",
        }[observation.source],
        labeler_version=labeler_version,
        code_commit=context.labeler_source_revision,
        container_image_digest=context.labeler_image_digest,
        config_sha256=config_sha256,
        ontology_sha256=ontology_sha256(),
        input_artifact_sha256s=(
            context.dataset_manifest_sha256,
            context.source_artifact_sha256,
        ),
        prompt_sha256=prompt_sha256,
        lookback_ns=int(details.get("lookback_ns", 0) or 0),
        lookahead_ns=int(details.get("lookahead_ns", 0) or 0),
        details=details,
        **model_fields,
        decoding_config_sha256=decoding_config_sha256,
    )


def source_observations_to_evidence(
    observations: Iterable[LabelObservation],
    *,
    context: EvidenceBuildContext,
) -> tuple[LabelEvidence, ...]:
    output: list[LabelEvidence] = []
    source_manifest_uri = (
        f"{context.source_artifact_uri.rstrip('/')}/manifest.json"
    )
    for observation in observations:
        scope = _scope(observation, context)
        measurements = tuple(
            Measurement(
                name=name,
                value=value,
                unit=_measurement_unit(name),
                quality="valid",
                aggregation=_measurement_aggregation(
                    observation.source,
                    name,
                ),
            )
            for name, value in sorted(observation.measurements.items())
        )
        refs = tuple(
            EvidenceReference(
                artifact_uri=source_manifest_uri,
                artifact_sha256=context.source_artifact_sha256,
                timestamp_ns=scope.anchor_timestamp_ns,
                camera_id=camera_id,
            )
            for camera_id in (scope.camera_ids or (None,))
        )
        provenance = _semantic_provenance(observation, context)
        identity = {
            "observation_uid": observation.observation_uid,
            "source": observation.source,
            "scope": scope.to_dict(),
            "status": observation.status,
            "values": observation.values,
            "confidence": observation.confidence,
            "measurements": [dataclasses.asdict(item) for item in measurements],
            "provenance": provenance.to_dict(),
            "capability_manifest_sha256": (
                context.capability_manifest_sha256
            ),
        }
        output.append(
            LabelEvidence(
                evidence_uid=f"oddev-{content_sha256(identity)[:24]}",
                label_key=observation.key,
                cardinality=ONTOLOGY[observation.key].cardinality,
                values=observation.values,
                candidate_values=_candidate_values(observation),
                status=observation.status,
                confidence=observation.confidence,
                source=observation.source,
                scope=scope,
                measurements=measurements,
                evidence_refs=refs,
                provenance=provenance,
            )
        )
    return tuple(sorted(output, key=lambda item: item.evidence_uid))


def _group_identity(evidence: LabelEvidence) -> tuple[Any, ...]:
    scope = evidence.scope
    return (
        evidence.label_key,
        scope.subject_type,
        scope.subject_id or "",
        scope.camera_ids if scope.subject_type == "actor" else (),
    )


def _source_rank(key: str, source: str) -> int:
    sources = ONTOLOGY[key].primary_sources
    try:
        return sources.index(source)
    except ValueError:
        return len(sources) + 1


def _evidence_measurements(
    evidence: LabelEvidence,
) -> dict[str, float | int | str | bool]:
    return {item.name: item.value for item in evidence.measurements}


def _resolve_interval(
    active: list[LabelEvidence],
    *,
    start_timestamp_ns: int,
    end_timestamp_ns: int,
) -> LabelObservation:
    first = active[0]
    key = first.label_key
    valid = [item for item in active if item.status == "valid"]
    supporting: list[LabelEvidence]
    conflicting: list[LabelEvidence]
    policy: str

    if not valid:
        status_order = {"ambiguous": 0, "not_observable": 1, "unavailable": 2}
        selected_status = min(
            (item.status for item in active),
            key=lambda status: status_order[status],
        )
        supporting = [
            item for item in active if item.status == selected_status
        ]
        conflicting = [
            item for item in active if item.status != selected_status
        ]
        status = selected_status
        values: tuple[str, ...] = ()
        confidence = max(item.confidence for item in supporting)
        policy = f"status_precedence:{selected_status}"
    else:
        values_by_evidence = {item.evidence_uid: item.values for item in valid}
        distinct_values = set(values_by_evidence.values())
        best_rank = min(_source_rank(key, item.source) for item in valid)
        authoritative = [
            item
            for item in valid
            if _source_rank(key, item.source) == best_rank
        ]
        authoritative_values = {item.values for item in authoritative}

        if len(distinct_values) == 1:
            values = valid[0].values
            supporting = valid
            conflicting = []
            status = "valid"
            confidence = min(item.confidence for item in valid)
            policy = "cross_source_agreement"
        elif (
            key in UNION_KEYS
            and all("none" not in item.values for item in valid)
        ):
            values = tuple(
                sorted(
                    {
                        value
                        for item in valid
                        for value in item.values
                    }
                )
            )
            supporting = valid
            conflicting = []
            status = "valid"
            confidence = min(item.confidence for item in valid)
            policy = "multi_select_union"
        elif len(authoritative_values) == 1 and len(authoritative) < len(valid):
            values = authoritative[0].values
            supporting = [
                item for item in valid if item.values == values
            ]
            conflicting = [
                item
                for item in valid
                if item.values != values
            ]
            status = "valid"
            confidence = min(item.confidence for item in authoritative) * 0.85
            policy = "authoritative_source_override"
        else:
            values = ()
            supporting = []
            conflicting = valid
            status = "ambiguous"
            confidence = max(item.confidence for item in authoritative)
            policy = "unresolved_source_conflict"

    supporting_ids = tuple(item.evidence_uid for item in supporting)
    conflicting_ids = tuple(item.evidence_uid for item in conflicting)
    source = (
        supporting[0].source
        if len(active) == 1 and not conflicting
        else "fusion"
    )
    measurement_source = min(
        supporting or conflicting or active,
        key=lambda item: (
            _source_rank(key, item.source),
            -item.confidence,
            item.evidence_uid,
        ),
    )
    subject_id = first.scope.subject_id
    return make_observation(
        scene_uid=first.scope.scene_uid,
        key=key,
        status=status,
        values=values,
        confidence=confidence,
        source=source,
        start_timestamp_ns=start_timestamp_ns,
        end_timestamp_ns=end_timestamp_ns,
        evidence_uids=supporting_ids,
        conflicting_evidence_uids=conflicting_ids,
        measurements=_evidence_measurements(measurement_source),
        provenance={
            "fusion_version": FUSION_VERSION,
            "policy": policy,
            "active_evidence_uids": sorted(
                item.evidence_uid for item in active
            ),
        },
        camera_id=(
            subject_id if first.scope.subject_type == "camera" else None
        ),
        actor_track_uid=(
            subject_id if first.scope.subject_type == "actor" else None
        ),
    )


def resolve_evidence(
    evidence: Iterable[LabelEvidence],
) -> tuple[LabelObservation, ...]:
    grouped: dict[tuple[Any, ...], list[LabelEvidence]] = defaultdict(list)
    for item in evidence:
        grouped[_group_identity(item)].append(item)

    observations: list[LabelObservation] = []
    for identity in sorted(grouped, key=str):
        rows = grouped[identity]
        boundaries = sorted(
            {
                timestamp
                for row in rows
                for timestamp in (
                    row.scope.start_timestamp_ns,
                    row.scope.end_timestamp_ns,
                )
            }
        )
        for start_timestamp_ns, end_timestamp_ns in zip(
            boundaries,
            boundaries[1:],
        ):
            active = [
                row
                for row in rows
                if row.scope.start_timestamp_ns <= start_timestamp_ns
                and row.scope.end_timestamp_ns >= end_timestamp_ns
            ]
            if not active:
                continue
            observations.append(
                _resolve_interval(
                    sorted(active, key=lambda item: item.evidence_uid),
                    start_timestamp_ns=start_timestamp_ns,
                    end_timestamp_ns=end_timestamp_ns,
                )
            )
    return coalesce_observations(observations)


def _is_positive_event(observation: LabelObservation) -> bool:
    if observation.namespace != "event" or observation.status != "valid":
        return False
    if observation.key in {"event.phase", "event.outcome"}:
        return False
    background = EVENT_BACKGROUND_VALUES.get(observation.key, set())
    return bool(set(observation.values) - background)


def _phase_intervals(
    start_ns: int,
    end_ns: int,
    *,
    onset_observed: bool,
    resolution_observed: bool,
) -> tuple[EventPhase, ...]:
    duration = end_ns - start_ns
    if duration < 3:
        return (EventPhase("active", start_ns, end_ns),)
    edge = min(1_000_000_000, max(1, duration // 5))
    active_start = start_ns + edge if onset_observed else start_ns
    active_end = end_ns - edge if resolution_observed else end_ns
    if active_end <= active_start:
        return (EventPhase("active", start_ns, end_ns),)
    phases = []
    if onset_observed:
        phases.append(EventPhase("onset", start_ns, active_start))
    phases.append(EventPhase("active", active_start, active_end))
    if resolution_observed:
        phases.append(EventPhase("resolution", active_end, end_ns))
    return tuple(phases)


def _merge_event_seeds(
    observations: Iterable[LabelObservation],
) -> tuple[tuple[LabelObservation, ...], ...]:
    grouped: dict[tuple[Any, ...], list[LabelObservation]] = defaultdict(list)
    for observation in observations:
        if observation.key not in EVENT_PRIMARY_PRIORITY:
            continue
        if not _is_positive_event(observation):
            continue
        grouped[
            (
                observation.scene_uid,
                observation.key,
                observation.values,
                observation.actor_track_uid or "",
            )
        ].append(observation)

    segments: list[tuple[LabelObservation, ...]] = []
    for identity in sorted(grouped, key=str):
        current: list[LabelObservation] = []
        for observation in sorted(
            grouped[identity],
            key=lambda item: (
                item.start_timestamp_ns,
                item.end_timestamp_ns,
                item.observation_uid,
            ),
        ):
            if (
                current
                and observation.start_timestamp_ns
                > current[-1].end_timestamp_ns + EVENT_JOIN_GAP_NS
            ):
                segments.append(tuple(current))
                current = []
            current.append(observation)
        if current:
            segments.append(tuple(current))
    return tuple(segments)


def _segment_actor_uids(
    segment: Iterable[LabelObservation],
) -> set[str]:
    return {
        item.actor_track_uid
        for item in segment
        if item.actor_track_uid
    }


def _segments_are_related(
    left: tuple[LabelObservation, ...],
    right: tuple[LabelObservation, ...],
) -> bool:
    left_start = min(item.start_timestamp_ns for item in left)
    left_end = max(item.end_timestamp_ns for item in left)
    right_start = min(item.start_timestamp_ns for item in right)
    right_end = max(item.end_timestamp_ns for item in right)
    if left_start >= right_end or right_start >= left_end:
        return False
    left_actors = _segment_actor_uids(left)
    right_actors = _segment_actor_uids(right)
    return not left_actors or not right_actors or bool(
        left_actors & right_actors
    )


def _cluster_event_seeds(
    segments: Iterable[tuple[LabelObservation, ...]],
) -> tuple[tuple[LabelObservation, ...], ...]:
    clusters: list[list[LabelObservation]] = []
    ordered = sorted(
        segments,
        key=lambda segment: (
            min(item.start_timestamp_ns for item in segment),
            max(item.end_timestamp_ns for item in segment),
            tuple(item.observation_uid for item in segment),
        ),
    )
    for segment in ordered:
        related_indices = [
            index
            for index, cluster in enumerate(clusters)
            if _segments_are_related(tuple(cluster), segment)
        ]
        segment_actors = _segment_actor_uids(segment)
        related_actor_sets = {
            frozenset(_segment_actor_uids(clusters[index]))
            for index in related_indices
            if _segment_actor_uids(clusters[index])
        }
        if (
            not segment_actors
            and len(related_indices) > 1
            and len(related_actor_sets) > 1
        ):
            segment_start = min(
                item.start_timestamp_ns for item in segment
            )
            segment_end = max(item.end_timestamp_ns for item in segment)
            related_indices = [
                max(
                    related_indices,
                    key=lambda index: (
                        min(
                            segment_end,
                            max(
                                item.end_timestamp_ns
                                for item in clusters[index]
                            ),
                        )
                        - max(
                            segment_start,
                            min(
                                item.start_timestamp_ns
                                for item in clusters[index]
                            ),
                        ),
                        -index,
                    ),
                )
            ]
        if not related_indices:
            clusters.append(list(segment))
            continue
        merged = list(segment)
        for index in reversed(related_indices):
            merged.extend(clusters.pop(index))
        clusters.append(merged)
    return tuple(
        tuple(
            sorted(
                cluster,
                key=lambda item: (
                    item.start_timestamp_ns,
                    item.end_timestamp_ns,
                    item.key,
                    item.observation_uid,
                ),
            )
        )
        for cluster in sorted(
            clusters,
            key=lambda cluster: (
                min(item.start_timestamp_ns for item in cluster),
                max(item.end_timestamp_ns for item in cluster),
            ),
        )
    )


def segment_events(
    observations: Iterable[LabelObservation],
) -> tuple[EventInstance, ...]:
    ordered = tuple(observations)
    if not ordered:
        return ()
    scene_start_ns = min(item.start_timestamp_ns for item in ordered)
    scene_end_ns = max(item.end_timestamp_ns for item in ordered)
    events: list[EventInstance] = []
    primary_order = {
        key: index for index, key in enumerate(EVENT_PRIMARY_PRIORITY)
    }
    seed_clusters = _cluster_event_seeds(_merge_event_seeds(ordered))
    for seed_segment in seed_clusters:
        start_ns = min(item.start_timestamp_ns for item in seed_segment)
        end_ns = max(item.end_timestamp_ns for item in seed_segment)
        seed_actor_uids = _segment_actor_uids(seed_segment)
        seed = min(
            seed_segment,
            key=lambda item: (
                primary_order[item.key],
                item.start_timestamp_ns,
                item.observation_uid,
            ),
        )
        related = [
            item
            for item in ordered
            if item.namespace == "event"
            and item.status == "valid"
            and item.key != "event.phase"
            and item.start_timestamp_ns < end_ns
            and item.end_timestamp_ns > start_ns
            and (
                not seed_actor_uids
                or not item.actor_track_uid
                or item.actor_track_uid in seed_actor_uids
            )
            and (
                _is_positive_event(item)
                or item.key == "event.outcome"
            )
        ]
        actor_track_uids = tuple(
            sorted(
                {
                    item.actor_track_uid
                    for item in related
                    if item.actor_track_uid
                }
            )
        )
        quantized_onset = (
            start_ns // EVENT_ONSET_QUANTUM_NS * EVENT_ONSET_QUANTUM_NS
        )
        event_identity = {
            "scene_uid": seed.scene_uid,
            "primary_event_key": seed.key,
            "primary_values": seed.values,
            "actor_track_uids": actor_track_uids,
            "quantized_onset_ns": quantized_onset,
            "segmenter_version": EVENT_SEGMENTER_VERSION,
        }
        event_uid = f"oddevent-{content_sha256(event_identity)[:24]}"
        supporting_evidence = tuple(
            sorted(
                {
                    evidence_uid
                    for item in related
                    for evidence_uid in item.evidence_uids
                }
            )
        )
        explicit_outcomes = sorted(
            {
                value
                for item in related
                if item.key == "event.outcome"
                for value in item.values
            }
        )
        onset_observed = start_ns > scene_start_ns
        resolution_observed = end_ns < scene_end_ns
        if explicit_outcomes:
            inferred_outcome = explicit_outcomes[0]
        elif not resolution_observed:
            inferred_outcome = "unresolved"
        else:
            inferred_outcome = "not_observed"
        events.append(
            EventInstance(
                event_uid=event_uid,
                scene_uid=seed.scene_uid,
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                primary_event_key=seed.key,
                actor_track_uids=actor_track_uids,
                observation_uids=tuple(
                    sorted(item.observation_uid for item in related)
                ),
                phases=_phase_intervals(
                    start_ns,
                    end_ns,
                    onset_observed=onset_observed,
                    resolution_observed=resolution_observed,
                ),
                confidence=min(item.confidence for item in seed_segment),
                status="valid",
                supporting_evidence_uids=supporting_evidence,
                provenance={
                    "segmenter_version": EVENT_SEGMENTER_VERSION,
                    "join_gap_ns": EVENT_JOIN_GAP_NS,
                    "onset_quantum_ns": EVENT_ONSET_QUANTUM_NS,
                    "primary_values": list(seed.values),
                    "onset_observed": onset_observed,
                    "resolution_observed": resolution_observed,
                    "outcome": inferred_outcome,
                },
            )
        )
    event_ids = [event.event_uid for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event segmentation produced duplicate event_uid")
    return tuple(
        sorted(
            events,
            key=lambda item: (
                item.start_timestamp_ns,
                item.end_timestamp_ns,
                item.event_uid,
            ),
        )
    )


def build_resolved_scene_labels(
    source_observations: Iterable[LabelObservation],
    *,
    context: EvidenceBuildContext,
) -> ResolvedSceneLabels:
    evidence = source_observations_to_evidence(
        source_observations,
        context=context,
    )
    observations = resolve_evidence(evidence)
    events = segment_events(observations)
    canonical_json_bytes(
        {
            "evidence": [item.to_dict() for item in evidence],
            "observations": [item.to_dict() for item in observations],
            "events": [item.to_dict() for item in events],
        }
    )
    return ResolvedSceneLabels(
        evidence=evidence,
        observations=observations,
        events=events,
    )
