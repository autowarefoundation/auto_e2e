from __future__ import annotations

from data_processing.odd_labeling.fusion import (
    EVENT_SEGMENTER_VERSION,
    EvidenceBuildContext,
    build_resolved_scene_labels,
    fusion_config_sha256,
    resolve_evidence,
    segment_events,
    source_observations_to_evidence,
)
from data_processing.odd_labeling.schema import make_observation


def _context() -> EvidenceBuildContext:
    return EvidenceBuildContext(
        dataset_name="synthetic",
        dataset_version="v1",
        dataset_manifest_sha256="1" * 64,
        capability_manifest_sha256="2" * 64,
        source_artifact_uri="s3://fixture/scene-1",
        source_artifact_sha256="3" * 64,
        labeler_image_digest=f"sha256:{'4' * 64}",
        labeler_source_revision="5" * 40,
    )


def _observation(
    *,
    key: str,
    values: tuple[str, ...],
    source: str,
    start_ns: int = 100,
    end_ns: int = 200,
    confidence: float = 0.9,
    provenance: dict | None = None,
    actor_track_uid: str | None = None,
    camera_id: str | None = None,
):
    return make_observation(
        scene_uid="scene-1",
        key=key,
        status="valid",
        values=values,
        confidence=confidence,
        source=source,
        start_timestamp_ns=start_ns,
        end_timestamp_ns=end_ns,
        provenance=provenance or {"labeler_version": f"{source}_v1"},
        actor_track_uid=actor_track_uid,
        camera_id=camera_id,
    )


def test_fusion_policy_hash_is_stable() -> None:
    assert fusion_config_sha256() == (
        "fd1d8c2bd9303beb046e4e5cd9f352076d15a35f58fd5a74de4c363bbe1e4bc9"
    )


def test_source_claim_becomes_auditable_evidence() -> None:
    observation = _observation(
        key="odd.environment.sky",
        values=("clear",),
        source="vlm",
        provenance={
            "prompt_version": "road_scene_v1",
            "prompt_sha256": "6" * 64,
            "decoding_config_sha256": "9" * 64,
            "model": "road-observer",
            "model_revision": "model-v1",
            "request_sha256": "7" * 64,
            "response_sha256": "8" * 64,
            "supporting_cameras": ["front_center", "front_left"],
            "reason": "Visible sky.",
        },
    )

    evidence = source_observations_to_evidence(
        (observation,),
        context=_context(),
    )[0]

    assert evidence.evidence_uid.startswith("oddev-")
    assert evidence.source == "vlm"
    assert evidence.scope.camera_ids == ("front_center", "front_left")
    assert len(evidence.evidence_refs) == 2
    assert evidence.provenance.model_provider == "openai_compatible"
    assert evidence.provenance.prompt_sha256 == "6" * 64
    assert evidence.provenance.decoding_config_sha256 == "9" * 64
    assert evidence.provenance.details["request_sha256"] == "7" * 64


def test_source_policy_thresholds_change_evidence_config_identity() -> None:
    first = _observation(
        key="odd.ego.speed_bin",
        values=("low_speed",),
        source="gnss_ins",
        provenance={
            "labeler_version": "odd_deterministic_kinematics_v3",
            "kinematics_policy_version": "odd_gnss_ins_kinematics_v2",
            "maximum_gap_ns": 500_000_000,
            "stationary_epsilon_kph": 0.5,
            "stationary_dwell_ns": 1_000_000_000,
        },
    )
    second = _observation(
        key="odd.ego.speed_bin",
        values=("low_speed",),
        source="gnss_ins",
        provenance={
            **first.provenance,
            "maximum_gap_ns": 750_000_000,
        },
    )

    first_evidence = source_observations_to_evidence(
        (first,),
        context=_context(),
    )[0]
    second_evidence = source_observations_to_evidence(
        (second,),
        context=_context(),
    )[0]

    assert (
        first_evidence.provenance.config_sha256
        != second_evidence.provenance.config_sha256
    )

    sampled = _observation(
        key="perception.image.frame_status",
        values=("normal",),
        source="image_qc",
        provenance={
            "labeler_version": "odd_image_qc_v3",
            "image_qc_policy_version": "odd_image_qc_policy_v3",
            "frame_inventory_mode": "sampled_evidence",
            "frozen_ego_motion_threshold_m": 0.5,
        },
    )
    capture = _observation(
        key="perception.image.frame_status",
        values=("normal",),
        source="image_qc",
        provenance={
            **sampled.provenance,
            "frame_inventory_mode": "capture_timeline",
        },
    )
    sampled_evidence = source_observations_to_evidence(
        (sampled,),
        context=_context(),
    )[0]
    capture_evidence = source_observations_to_evidence(
        (capture,),
        context=_context(),
    )[0]

    assert (
        sampled_evidence.provenance.config_sha256
        != capture_evidence.provenance.config_sha256
    )


def test_map_evidence_retains_bedrock_model_identity() -> None:
    observation = _observation(
        key="odd.route.action",
        values=("turn_left",),
        source="map_route",
        provenance={
            "labeler_version": "bedrock_map_route_v1",
            "model_provider": "amazon_bedrock",
            "model": "us.anthropic.claude-sonnet-4-6",
            "model_revision": "claude-sonnet-4-6",
        },
    )

    evidence = source_observations_to_evidence(
        (observation,),
        context=_context(),
    )[0]

    assert evidence.source == "map_route"
    assert evidence.provenance.model_provider == "amazon_bedrock"
    assert evidence.provenance.model_name == (
        "us.anthropic.claude-sonnet-4-6"
    )
    assert evidence.provenance.model_revision == "claude-sonnet-4-6"


def test_fusion_retains_agreeing_source_evidence() -> None:
    evidence = source_observations_to_evidence(
        (
            _observation(
                key="odd.road.junction_control",
                values=("traffic_light",),
                source="map_route",
                confidence=0.95,
            ),
            _observation(
                key="odd.road.junction_control",
                values=("traffic_light",),
                source="vlm",
                confidence=0.8,
            ),
        ),
        context=_context(),
    )

    observation = resolve_evidence(evidence)[0]

    assert observation.status == "valid"
    assert observation.values == ("traffic_light",)
    assert observation.source == "fusion"
    assert observation.confidence == 0.8
    assert set(observation.evidence_uids) == {
        item.evidence_uid for item in evidence
    }
    assert observation.conflicting_evidence_uids == ()


def test_authoritative_source_wins_but_conflict_remains_visible() -> None:
    evidence = source_observations_to_evidence(
        (
            _observation(
                key="odd.road.junction_control",
                values=("traffic_light",),
                source="map_route",
                confidence=0.95,
            ),
            _observation(
                key="odd.road.junction_control",
                values=("stop_sign",),
                source="vlm",
                confidence=0.9,
            ),
        ),
        context=_context(),
    )

    observation = resolve_evidence(evidence)[0]

    assert observation.status == "valid"
    assert observation.values == ("traffic_light",)
    assert observation.source == "fusion"
    assert observation.confidence == 0.95 * 0.85
    assert len(observation.evidence_uids) == 1
    assert len(observation.conflicting_evidence_uids) == 1


def test_native_dropped_frame_overrides_visual_normal_interval() -> None:
    evidence = source_observations_to_evidence(
        (
            _observation(
                key="perception.image.frame_status",
                values=("normal",),
                source="vlm",
                start_ns=0,
                end_ns=1_000_000_000,
                camera_id="front_center",
            ),
            _observation(
                key="perception.image.frame_status",
                values=("dropped_frame",),
                source="image_qc",
                start_ns=100_000_000,
                end_ns=200_000_000,
                confidence=1.0,
                camera_id="front_center",
            ),
        ),
        context=_context(),
    )

    observations = resolve_evidence(evidence)
    dropped = next(
        item for item in observations if item.values == ("dropped_frame",)
    )

    assert (dropped.start_timestamp_ns, dropped.end_timestamp_ns) == (
        100_000_000,
        200_000_000,
    )
    assert dropped.provenance["policy"] == "authoritative_source_override"
    assert len(dropped.conflicting_evidence_uids) == 1


def test_same_authority_conflict_abstains() -> None:
    evidence = source_observations_to_evidence(
        (
            _observation(
                key="odd.environment.sky",
                values=("clear",),
                source="vlm",
            ),
            _observation(
                key="odd.environment.sky",
                values=("overcast",),
                source="vlm",
            ),
        ),
        context=_context(),
    )

    observation = resolve_evidence(evidence)[0]

    assert observation.status == "ambiguous"
    assert observation.values == ()
    assert observation.evidence_uids == ()
    assert set(observation.conflicting_evidence_uids) == {
        item.evidence_uid for item in evidence
    }


def test_multi_select_union_never_combines_neutral_values() -> None:
    union_evidence = source_observations_to_evidence(
        (
            _observation(
                key="odd.road.edge_type_present",
                values=("curb",),
                source="map_route",
            ),
            _observation(
                key="odd.road.edge_type_present",
                values=("grass",),
                source="vlm",
            ),
        ),
        context=_context(),
    )
    none_conflict = source_observations_to_evidence(
        (
            _observation(
                key="odd.road.edge_type_present",
                values=("none",),
                source="map_route",
            ),
            _observation(
                key="odd.road.edge_type_present",
                values=("grass",),
                source="map_route",
            ),
        ),
        context=_context(),
    )
    normal_conflict = source_observations_to_evidence(
        (
            _observation(
                key="perception.visual.lighting",
                values=("normal",),
                source="vlm",
                camera_id="front_center",
            ),
            _observation(
                key="perception.visual.lighting",
                values=("backlit",),
                source="vlm",
                camera_id="front_center",
            ),
        ),
        context=_context(),
    )

    union = resolve_evidence(union_evidence)[0]
    none_ambiguous = resolve_evidence(none_conflict)[0]
    normal_ambiguous = resolve_evidence(normal_conflict)[0]

    assert union.values == ("curb", "grass")
    assert none_ambiguous.status == "ambiguous"
    assert none_ambiguous.values == ()
    assert normal_ambiguous.status == "ambiguous"
    assert normal_ambiguous.values == ()
    assert set(normal_ambiguous.conflicting_evidence_uids) == {
        item.evidence_uid for item in normal_conflict
    }


def test_event_segmentation_is_stable_and_ignores_background() -> None:
    source = (
        _observation(
            key="event.ego.strong_response",
            values=("hard_brake",),
            source="gnss_ins",
            start_ns=100_000_000,
            end_ns=1_100_000_000,
        ),
        _observation(
            key="event.hazard.type",
            values=("obstacle_on_road",),
            source="vlm",
            start_ns=200_000_000,
            end_ns=1_200_000_000,
        ),
        _observation(
            key="event.ego.maneuver",
            values=("lane_follow",),
            source="gnss_ins",
            start_ns=0,
            end_ns=2_000_000_000,
        ),
    )

    first = build_resolved_scene_labels(source, context=_context())
    second = segment_events(first.observations)

    assert len(first.events) == 1
    assert first.events == second
    assert first.events[0].primary_event_key == "event.hazard.type"
    assert all(
        event.provenance["segmenter_version"] == EVENT_SEGMENTER_VERSION
        and tuple(phase.phase for phase in event.phases)
        == ("onset", "active", "resolution")
        and len(event.supporting_evidence_uids) == 2
        and event.provenance["outcome"] == "not_observed"
        for event in first.events
    )


def test_event_boundary_does_not_invent_onset_or_resolution() -> None:
    event = build_resolved_scene_labels(
        (
            _observation(
                key="event.ego.strong_response",
                values=("hard_brake",),
                source="gnss_ins",
                start_ns=100,
                end_ns=200,
            ),
        ),
        context=_context(),
    ).events[0]

    assert tuple(phase.phase for phase in event.phases) == ("active",)
    assert event.provenance["onset_observed"] is False
    assert event.provenance["resolution_observed"] is False
    assert event.provenance["outcome"] == "unresolved"


def test_overlapping_events_with_distinct_actors_remain_independent() -> None:
    events = build_resolved_scene_labels(
        (
            _observation(
                key="event.vehicle.interaction",
                values=("cut_in",),
                source="vlm",
                start_ns=100,
                end_ns=300,
                actor_track_uid="vehicle-a",
            ),
            _observation(
                key="event.vehicle.interaction",
                values=("cut_in",),
                source="vlm",
                start_ns=100,
                end_ns=300,
                actor_track_uid="vehicle-b",
            ),
        ),
        context=_context(),
    ).events

    assert len(events) == 2
    assert {event.actor_track_uids for event in events} == {
        ("vehicle-a",),
        ("vehicle-b",),
    }
