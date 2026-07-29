from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import data_processing.odd_labeling.ontology as ontology_module
from data_processing.odd_labeling.deterministic import _interval_slices
from data_processing.odd_labeling.ontology import (
    LABEL_STATUSES,
    ONTOLOGY,
    ONTOLOGY_VERSION,
    load_ontology_registry,
    ontology_document,
    ontology_sha256,
)
from data_processing.odd_labeling.schema import (
    CameraCapability,
    CandidateValue,
    ChannelCapability,
    DatasetCapabilityManifest,
    EventInstance,
    EventPhase,
    LabelEvidence,
    LabelScope,
    Measurement,
    ProviderExchange,
    SceneLabelRecord,
    SemanticLabelerProvenance,
    canonical_json_bytes,
    coalesce_observations,
    content_sha256,
    make_observation,
)


def test_interval_slices_carry_forward_final_sample() -> None:
    timestamps = np.array(
        [index * 100_000_000 for index in range(20)]
        + [1_950_000_000],
        dtype=np.int64,
    )

    slices = _interval_slices(timestamps)

    assert slices[-1] == (2_000_000_000, 2_050_000_000, 20, 21)


def test_ontology_contains_complete_scene_label_catalog() -> None:
    counts = {"odd": 0, "event": 0, "perception": 0}
    for definition in ONTOLOGY.values():
        counts[definition.namespace] += 1

    assert counts == {"odd": 32, "event": 13, "perception": 21}
    assert tuple(LABEL_STATUSES) == (
        "valid",
        "unavailable",
        "not_observable",
        "ambiguous",
    )
    document = ontology_document()
    assert ONTOLOGY_VERSION == "odd_ontology_v1.0.1"
    assert len(document["ontology_sha256"]) == 64
    assert len(document["labels"]) == 66


def test_ontology_registry_exposes_acquisition_and_candidate_semantics() -> None:
    document = ontology_document()
    backends = {
        item["backend"]: item["canonical_source"]
        for item in document["backend_definitions"]
    }
    capabilities = set(document["capability_definitions"])

    for label in document["labels"]:
        assert label["display_name"]
        assert label["description"]
        assert label["values"]
        assert all(value["display_name"] for value in label["values"])
        assert all(value["description"] for value in label["values"])
        assert set(label["authoritative_sources"]) <= set(
            label["allowed_sources"]
        )
        assert set(label["fallback_sources"]) <= set(label["allowed_sources"])
        assert set(label["quality_tier_by_source"]) == set(
            label["allowed_sources"]
        )
        assert label["acquisition"]["required_evidence"]
        assert label["acquisition"]["routing_policy"]
        assert label["acquisition"]["fallback_policy"]
        for backend in (
            label["acquisition"]["primary_backends"]
            + label["acquisition"]["fallback_backends"]
        ):
            assert backends[backend] in label["allowed_sources"]
        for alternative in label["required_capabilities"]["any_of"]:
            assert alternative
            assert set(alternative) <= capabilities

        values = {value["value"] for value in label["values"]}
        neutral = values & {"none", "normal"}
        if label["cardinality"] == "multi" and neutral:
            assert neutral == {label["neutral_value"]}
            assert label["none_semantics"]


def test_provider_exchange_retains_auditable_raw_response() -> None:
    raw_response = {
        "choices": [
            {
                "message": {
                    "content": "{\"observations\":{}}",
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }
    exchange = ProviderExchange(
        backend="ORV",
        provider="openai_compatible",
        model="nvidia/Cosmos3-Nano",
        model_revision="revision-1",
        request_sha256="a" * 64,
        response_sha256=content_sha256(raw_response),
        status="succeeded",
        attempt=1,
        latency_ms=1234.5,
        input_image_count=6,
        request_metadata={
            "task_bundle": "road_appearance",
            "frame_timestamps_ns": [100, 200],
        },
        raw_response=raw_response,
        usage={"input_tokens": 100, "output_tokens": 20},
    )

    assert exchange.to_dict()["raw_response"] == raw_response
    assert canonical_json_bytes(exchange.to_dict())

    with pytest.raises(ValueError, match="response digest"):
        ProviderExchange(
            **{
                **exchange.to_dict(),
                "response_sha256": "b" * 64,
            }
        )
    with pytest.raises(ValueError, match="latency"):
        ProviderExchange(
            **{
                **exchange.to_dict(),
                "latency_ms": -1.0,
            }
        )


def test_ontology_digest_covers_expanded_registry_semantics() -> None:
    document = ontology_document()
    digest = document.pop("ontology_sha256")

    expected = hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()

    assert digest == expected == ontology_sha256()


def test_ontology_loader_rejects_unknown_registry_fields(tmp_path: Path) -> None:
    registry_path = Path(ontology_module.__file__).with_name(
        "ontology_registry.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["labels"][0]["unreviewed_policy"] = "accept"
    invalid_path = tmp_path / "invalid-ontology.json"
    invalid_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown=.*unreviewed_policy"):
        load_ontology_registry(invalid_path)


def test_ontology_loader_rejects_backend_source_escalation(
    tmp_path: Path,
) -> None:
    registry_path = Path(ontology_module.__file__).with_name(
        "ontology_registry.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    road_type = next(
        label
        for label in registry["labels"]
        if label["key"] == "odd.road.type"
    )
    road_type["acquisition"]["fallback_backends"] = ["ORV"]
    invalid_path = tmp_path / "invalid-routing.json"
    invalid_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="emits disallowed source vlm"):
        load_ontology_registry(invalid_path)


def test_route_plan_and_actual_maneuver_remain_distinct() -> None:
    planned = ONTOLOGY["odd.route.action"]
    actual = ONTOLOGY["event.ego.maneuver"]

    assert "turn_left" in planned.values
    assert "turn_left" in actual.values
    assert planned.primary_sources == ("map_route",)
    assert "gnss_ins" in actual.primary_sources


def test_non_valid_status_cannot_encode_none() -> None:
    with pytest.raises(ValueError, match="must not carry resolved values"):
        make_observation(
            scene_uid="scene-1",
            key="odd.road.workzone_state",
            status="not_observable",
            values=("none",),
            confidence=0.0,
            source="vlm",
            start_timestamp_ns=1,
            end_timestamp_ns=2,
        )


def test_none_and_normal_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="none cannot coexist"):
        make_observation(
            scene_uid="scene-1",
            key="odd.road.workzone_state",
            status="valid",
            values=("none", "cones"),
            confidence=0.8,
            source="vlm",
            start_timestamp_ns=1,
            end_timestamp_ns=2,
        )
    with pytest.raises(ValueError, match="normal cannot coexist"):
        make_observation(
            scene_uid="scene-1",
            key="perception.visual.lighting",
            status="valid",
            values=("normal", "backlit"),
            confidence=0.8,
            source="image_qc",
            start_timestamp_ns=1,
            end_timestamp_ns=2,
            camera_id="front",
        )


def test_speed_observation_retains_continuous_measurement() -> None:
    observation = make_observation(
        scene_uid="scene-1",
        key="odd.ego.speed_bin",
        status="valid",
        values=("low_speed",),
        confidence=0.99,
        source="gnss_ins",
        start_timestamp_ns=1,
        end_timestamp_ns=2,
        measurements={"ego_speed_kph": 12.5},
    )

    assert observation.measurements["ego_speed_kph"] == 12.5


def test_adjacent_equal_observations_coalesce() -> None:
    observations = [
        make_observation(
            scene_uid="scene-1",
            key="odd.environment.sky",
            status="valid",
            values=("clear",),
            confidence=0.9,
            source="vlm",
            start_timestamp_ns=10,
            end_timestamp_ns=20,
        ),
        make_observation(
            scene_uid="scene-1",
            key="odd.environment.sky",
            status="valid",
            values=("clear",),
            confidence=0.8,
            source="vlm",
            start_timestamp_ns=20,
            end_timestamp_ns=30,
        ),
    ]

    merged = coalesce_observations(observations)

    assert len(merged) == 1
    assert merged[0].start_timestamp_ns == 10
    assert merged[0].end_timestamp_ns == 30
    assert merged[0].confidence == 0.8


def _absent_channel() -> ChannelCapability:
    return ChannelCapability(
        availability="absent",
        coverage_start_ns=None,
        coverage_end_ns=None,
        nominal_rate_hz=None,
        observed_count=0,
        missing_count=0,
        source_artifact_sha256=None,
    )


def _provenance() -> SemanticLabelerProvenance:
    return SemanticLabelerProvenance(
        labeler_name="synthetic_labeler",
        labeler_version="synthetic_v1",
        code_commit="1" * 40,
        container_image_digest=f"sha256:{'2' * 64}",
        config_sha256="3" * 64,
        ontology_sha256="4" * 64,
        input_artifact_sha256s=("5" * 64,),
    )


def test_semantic_provenance_retains_finite_audit_details() -> None:
    details = {"request_sha256": "6" * 64, "attempt": 1}
    provenance = SemanticLabelerProvenance(
        labeler_name="synthetic_labeler",
        labeler_version="synthetic_v1",
        code_commit="1" * 40,
        container_image_digest=f"sha256:{'2' * 64}",
        config_sha256="3" * 64,
        ontology_sha256="4" * 64,
        input_artifact_sha256s=("5" * 64,),
        details=details,
    )
    details["attempt"] = 2

    assert provenance.details["attempt"] == 1

    with pytest.raises(ValueError, match="Out of range float"):
        SemanticLabelerProvenance(
            labeler_name="synthetic_labeler",
            labeler_version="synthetic_v1",
            code_commit="1" * 40,
            container_image_digest=f"sha256:{'2' * 64}",
            config_sha256="3" * 64,
            ontology_sha256="4" * 64,
            input_artifact_sha256s=("5" * 64,),
            details={"confidence": float("nan")},
        )


def test_dataset_capability_manifest_distinguishes_absent_channels() -> None:
    camera_channel = ChannelCapability(
        availability="complete",
        coverage_start_ns=100,
        coverage_end_ns=200,
        nominal_rate_hz=10.0,
        observed_count=2,
        missing_count=0,
        source_artifact_sha256="6" * 64,
    )
    channels = {
        "map": _absent_channel(),
        "route": _absent_channel(),
        "gnss": camera_channel,
        "ins": camera_channel,
        "lidar": _absent_channel(),
        "object_tracks": _absent_channel(),
        "can": _absent_channel(),
    }
    manifest = DatasetCapabilityManifest(
        dataset_name="synthetic",
        dataset_version="v1",
        dataset_manifest_sha256="7" * 64,
        source_revision="source-v1",
        adapter_name="synthetic",
        adapter_version="adapter-v1",
        scene_inventory_sha256="8" * 64,
        canonical_clock="scene_monotonic_ns",
        absolute_time_available=False,
        timezone_resolution_available=False,
        cameras=(
            CameraCapability(
                camera_id="front",
                canonical_role="front_center",
                channel=camera_channel,
            ),
        ),
        channels=channels,
        coordinate_frames=("ego_flu",),
    )

    assert manifest.channels["map"].availability == "absent"
    assert manifest.channels["gnss"].availability == "complete"
    assert manifest.cameras[0].frame_inventory_mode == "unknown"
    assert len(manifest.semantic_sha256()) == 64

    with pytest.raises(ValueError, match="frame inventory mode"):
        CameraCapability(
            camera_id="front",
            canonical_role="front_center",
            channel=camera_channel,
            frame_inventory_mode="assumed_complete",
        )

    with pytest.raises(ValueError, match="absent channel"):
        ChannelCapability(
            availability="absent",
            coverage_start_ns=100,
            coverage_end_ns=200,
            nominal_rate_hz=1.0,
            observed_count=1,
            missing_count=0,
            source_artifact_sha256="9" * 64,
        )


def test_label_scope_requires_stable_subject_identity() -> None:
    scope = LabelScope(
        dataset_name="synthetic",
        dataset_version="v1",
        scene_uid="scene-1",
        start_timestamp_ns=100,
        end_timestamp_ns=200,
        subject_type="camera",
        subject_id="front",
        camera_ids=("front",),
    )

    assert scope.subject_id == "front"

    with pytest.raises(ValueError, match="actor scope requires subject_id"):
        LabelScope(
            dataset_name="synthetic",
            dataset_version="v1",
            scene_uid="scene-1",
            start_timestamp_ns=100,
            end_timestamp_ns=200,
            subject_type="actor",
        )


def test_evidence_keeps_candidates_separate_from_resolved_values() -> None:
    scope = LabelScope(
        dataset_name="synthetic",
        dataset_version="v1",
        scene_uid="scene-1",
        start_timestamp_ns=100,
        end_timestamp_ns=200,
    )
    evidence = LabelEvidence(
        evidence_uid="oddev-test",
        label_key="odd.road.junction_type",
        cardinality="single",
        values=(),
        candidate_values=(
            CandidateValue(value="t_junction", score=0.55),
            CandidateValue(value="crossroad", score=0.45),
        ),
        status="ambiguous",
        confidence=0.55,
        source="map_route",
        scope=scope,
        measurements=(
            Measurement(
                name="topology_margin",
                value=0.1,
                unit="ratio",
                quality="valid",
                aggregation="interval",
            ),
        ),
        evidence_refs=(),
        provenance=_provenance(),
    )

    assert evidence.values == ()
    assert [item.value for item in evidence.candidate_values] == [
        "t_junction",
        "crossroad",
    ]

    with pytest.raises(ValueError, match="non-valid evidence"):
        LabelEvidence(
            evidence_uid="oddev-invalid",
            label_key="odd.road.workzone_state",
            cardinality="multi",
            values=("none",),
            candidate_values=(),
            status="not_observable",
            confidence=0.0,
            source="vlm",
            scope=scope,
            measurements=(),
            evidence_refs=(),
            provenance=_provenance(),
        )


def test_scene_record_validates_event_and_evidence_references() -> None:
    scope = LabelScope(
        dataset_name="synthetic",
        dataset_version="v1",
        scene_uid="scene-1",
        start_timestamp_ns=100,
        end_timestamp_ns=300,
    )
    evidence = LabelEvidence(
        evidence_uid="oddev-1",
        label_key="event.ego.maneuver",
        cardinality="single",
        values=("turn_left",),
        candidate_values=(),
        status="valid",
        confidence=0.9,
        source="gnss_ins",
        scope=scope,
        measurements=(),
        evidence_refs=(),
        provenance=_provenance(),
    )
    observation = make_observation(
        scene_uid="scene-1",
        key="event.ego.maneuver",
        status="valid",
        values=("turn_left",),
        confidence=0.9,
        source="gnss_ins",
        start_timestamp_ns=100,
        end_timestamp_ns=300,
        evidence_uids=("oddev-1",),
        event_uid="oddevent-1",
    )
    event = EventInstance(
        event_uid="oddevent-1",
        scene_uid="scene-1",
        start_timestamp_ns=100,
        end_timestamp_ns=300,
        primary_event_key="event.ego.maneuver",
        actor_track_uids=(),
        observation_uids=(observation.observation_uid,),
        phases=(
            EventPhase("onset", 100, 150),
            EventPhase("active", 150, 250),
            EventPhase("resolution", 250, 300),
        ),
        confidence=0.9,
        status="valid",
        supporting_evidence_uids=("oddev-1",),
        provenance={"segmenter_version": "event_segmenter_v1"},
    )
    record = SceneLabelRecord(
        scene_uid="scene-1",
        dataset_name="synthetic",
        dataset_version="v1",
        dataset_manifest_sha256="a" * 64,
        start_timestamp_ns=100,
        end_timestamp_ns=300,
        distance_m=12.0,
        observations=(observation,),
        source_artifact_uri="s3://example/scene-1",
        source_artifact_sha256="b" * 64,
        evidence=(evidence,),
        events=(event,),
        capability_manifest_sha256="c" * 64,
    )

    assert record.events[0].event_uid == "oddevent-1"
    assert record.evidence[0].evidence_uid == "oddev-1"

    with pytest.raises(ValueError, match="ordered"):
        EventInstance(
            event_uid="oddevent-invalid",
            scene_uid="scene-1",
            start_timestamp_ns=100,
            end_timestamp_ns=300,
            primary_event_key="event.ego.maneuver",
            actor_track_uids=(),
            observation_uids=(observation.observation_uid,),
            phases=(
                EventPhase("active", 100, 200),
                EventPhase("onset", 200, 300),
            ),
            confidence=0.9,
            status="valid",
            supporting_evidence_uids=("oddev-1",),
            provenance={},
        )
