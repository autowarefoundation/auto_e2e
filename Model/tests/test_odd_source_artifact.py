from __future__ import annotations

import json

import pytest

from data_processing.odd_labeling.schema import (
    ProviderExchange,
    content_sha256,
    make_observation,
)
from data_processing.odd_labeling.source_artifact import (
    SourceObservationArtifact,
)


def _descriptor(scene_uid: str = "scene-a") -> str:
    return json.dumps(
        {
            "dataset_name": "synthetic",
            "dataset_version": "v1",
            "scene_uid": scene_uid,
            "source_uri": f"s3://example/{scene_uid}",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _observation(scene_uid: str = "scene-a"):
    return make_observation(
        scene_uid=scene_uid,
        key="odd.road.context",
        status="valid",
        values=("urban",),
        confidence=0.9,
        source="map_route",
        start_timestamp_ns=100,
        end_timestamp_ns=200,
    )


def _exchange(backend: str = "ORV") -> ProviderExchange:
    provider = (
        "openai_compatible" if backend == "ORV" else "amazon_bedrock"
    )
    raw_response = {"result": "urban"}
    return ProviderExchange(
        backend=backend,
        provider=provider,
        model="model",
        model_revision="revision",
        request_sha256="a" * 64,
        response_sha256=content_sha256(raw_response),
        status="succeeded",
        attempt=1,
        latency_ms=12.5,
        input_image_count=6 if backend == "ORV" else 1,
        request_metadata={"bundle": "road"},
        raw_response=raw_response,
    )


def test_source_artifact_round_trip_is_canonical() -> None:
    descriptor = _descriptor()
    artifact = SourceObservationArtifact.create(
        source_stage="map_route_deterministic",
        descriptor_json=descriptor,
        scene_uid="scene-a",
        observations=[_observation()],
    )

    restored = SourceObservationArtifact.from_bytes(
        artifact.to_bytes(),
        expected_descriptor_json=descriptor,
        expected_source_stage="map_route_deterministic",
    )

    assert restored == artifact
    assert len(restored.semantic_sha256()) == 64


def test_provider_exchanges_round_trip_only_in_their_source_stage() -> None:
    descriptor = _descriptor()
    artifact = SourceObservationArtifact.create(
        source_stage="openai_compatible_vlm",
        descriptor_json=descriptor,
        scene_uid="scene-a",
        observations=[_observation()],
        provider_exchanges=[_exchange()],
    )

    restored = SourceObservationArtifact.from_bytes(
        artifact.to_bytes(),
        expected_descriptor_json=descriptor,
        expected_source_stage="openai_compatible_vlm",
    )

    assert restored.provider_exchanges == (_exchange(),)
    with pytest.raises(ValueError, match="source stage"):
        SourceObservationArtifact.create(
            source_stage="map_route_deterministic",
            descriptor_json=descriptor,
            scene_uid="scene-a",
            observations=[],
            provider_exchanges=[_exchange()],
        )
    with pytest.raises(ValueError, match="source stage"):
        SourceObservationArtifact.create(
            source_stage="bedrock_map_route",
            descriptor_json=descriptor,
            scene_uid="scene-a",
            observations=[],
            provider_exchanges=[_exchange()],
        )


def test_source_artifact_rejects_cross_scene_and_descriptor_mix() -> None:
    with pytest.raises(ValueError, match="another scene"):
        SourceObservationArtifact.create(
            source_stage="gnss_ins",
            descriptor_json=_descriptor(),
            scene_uid="scene-a",
            observations=[_observation("scene-b")],
        )

    artifact = SourceObservationArtifact.create(
        source_stage="gnss_ins",
        descriptor_json=_descriptor(),
        scene_uid="scene-a",
        observations=[_observation()],
    )
    with pytest.raises(ValueError, match="descriptor digest differs"):
        SourceObservationArtifact.from_bytes(
            artifact.to_bytes(),
            expected_descriptor_json=_descriptor("scene-b"),
        )


def test_source_artifact_rejects_noncanonical_json() -> None:
    artifact = SourceObservationArtifact.create(
        source_stage="image_qc",
        descriptor_json=_descriptor(),
        scene_uid="scene-a",
        observations=[],
    )
    pretty = json.dumps(artifact.to_dict(), indent=2).encode()

    with pytest.raises(ValueError, match="not canonical"):
        SourceObservationArtifact.from_bytes(pretty)
