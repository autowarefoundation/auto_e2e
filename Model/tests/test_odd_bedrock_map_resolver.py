from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from data_processing.odd_labeling.bedrock_map_resolver import (
    BEDROCK_TOOL_NAME,
    BedrockMapRouteResolver,
    bedrock_map_decoding_config_sha256,
    bedrock_map_prompt_bundle_sha256,
    build_privacy_safe_request,
    resolve_ambiguous_map_route,
)
from data_processing.odd_labeling.published_snapshot import (
    CameraObject,
    CanonicalSceneEvidence,
    PublishedSceneDescriptor,
)
from data_processing.odd_labeling.schema import make_observation
from navigation.contracts import (
    Destination,
    DirectedLaneField,
    Maneuver,
    MapFrame,
    NavigationMap,
    NavigationRoute,
    PolygonPrimitive,
    RouteLaneSegment,
    RouteProvenance,
    RouteQuality,
)


def _scene() -> CanonicalSceneEvidence:
    frame = MapFrame("fixture-enu", 49.0, 8.0, "local ENU")
    route_points = np.asarray(
        [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0]]
    )
    lanes = (
        np.asarray([[-30.0, 0.0], [30.0, 0.0]]),
        np.asarray([[0.0, -30.0], [0.0, 30.0]]),
        np.asarray([[-30.0, -30.0], [30.0, 30.0]]),
        np.asarray([[-30.0, 30.0], [30.0, -30.0]]),
    )
    navigation_map = NavigationMap(
        map_version="fixture-map-v1",
        provider="fixture-provider",
        frame=frame,
        bounds_enu_m=(-100.0, -100.0, 100.0, 100.0),
        intersection_polygons=(
            PolygonPrimitive(
                "provider-intersection-id",
                np.asarray(
                    [
                        [-5.0, -5.0],
                        [5.0, -5.0],
                        [5.0, 5.0],
                        [-5.0, 5.0],
                    ]
                ),
            ),
        ),
        directed_lane_fields=tuple(
            DirectedLaneField(
                f"provider-lane-{index}",
                points,
            )
            for index, points in enumerate(lanes)
        ),
    )
    route = NavigationRoute(
        route_id="provider-route-id",
        revision=1,
        provider="fixture-provider",
        timestamp_ns=0,
        valid_from_ns=0,
        map_version=navigation_map.map_version,
        frame=frame,
        lane_sequence=(
            RouteLaneSegment(
                lane_id="provider-route-lane",
                provider_segment_id="provider-segment-id",
                centerline_enu_m=route_points,
                maneuver=Maneuver.UNKNOWN,
            ),
        ),
        destination=Destination(np.asarray([20.0, 20.0]), "fixture"),
        confidence=0.9,
        valid=True,
        quality=RouteQuality(1.0, 0.0, 0.0, 0.0, 0.0),
        estimated_destination=False,
        provenance=RouteProvenance(
            source_revision="source-revision",
            matcher_version="matcher-v1",
            matcher_config_sha256="1" * 64,
            map_sha256="2" * 64,
            trace_sha256="3" * 64,
        ),
    )
    descriptor = PublishedSceneDescriptor(
        dataset_name="synthetic",
        dataset_version="v1",
        dataset_manifest_uri="s3://fixture/manifest.json",
        dataset_manifest_sha256="4" * 64,
        partition_id="partition-1",
        scene_uid="private-scene-id",
        source_uri="s3://fixture/private-scene-id",
        source_manifest_sha256="5" * 64,
        shard_name="private-scene-id.tar",
        camera_count=6,
        endpoint_exclusion_frames=0,
    )
    path = np.asarray(
        [
            [49.0, 8.0, 90.0, 0.0],
            [49.0, 8.000001, 90.0, 100_000_000.0],
            [49.0, 8.000002, 90.0, 200_000_000.0],
        ]
    )
    return CanonicalSceneEvidence(
        descriptor=descriptor,
        path_latlon_heading_timestamp=path,
        navigation_map=navigation_map,
        navigation_route=route,
        navigation_quality={"valid": True},
        camera_objects=(
            CameraObject(
                frame_index=0,
                camera_index=0,
                camera_role="front_center",
                timestamp_ns=0,
                bucket="fixture",
                key="camera.jpg",
                byte_size=10,
            ),
        ),
    )


def _ambiguous_junction():
    return make_observation(
        scene_uid="private-scene-id",
        key="odd.road.junction_type",
        status="ambiguous",
        confidence=0.0,
        source="map_route",
        start_timestamp_ns=0,
        end_timestamp_ns=300_000_000,
        measurements={"junction_branch_count": 3},
        provenance={
            "labeler_version": "deterministic-v3",
            "matched_lane_id": "provider-lane-0",
            "bedrock_eligible": True,
            "candidate_values": ["t_junction", "y_junction"],
        },
    )


def test_bedrock_map_semantic_hashes_are_stable() -> None:
    assert bedrock_map_prompt_bundle_sha256() == (
        "a0c72c946a20c36b0942d3c6532437bea7cd0513e72dc195bfb74a908003904c"
    )
    assert bedrock_map_decoding_config_sha256(max_tokens=1024) == (
        "b5f42c6a13e7c29a8c624dbbb88ec4fc32eb3de368494db6b00e447a72c01050"
    )
    assert bedrock_map_decoding_config_sha256(
        max_tokens=512
    ) != bedrock_map_decoding_config_sha256(max_tokens=1024)


class _BedrockClient:
    def __init__(self, value: str) -> None:
        self.value = value
        self.requests = []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": BEDROCK_TOOL_NAME,
                                "input": {
                                    "status": "resolved",
                                    "value": self.value,
                                    "confidence": 0.88,
                                    "cited_primitive_ids": ["route-000"],
                                    "candidate_rejections": {
                                        "straight": "Heading changes left."
                                    },
                                },
                            }
                        }
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 100, "outputTokens": 20},
        }


def test_request_removes_geography_and_provider_identity() -> None:
    request = build_privacy_safe_request(_scene(), _ambiguous_junction())

    assert request is not None
    payload = request.payload()
    serialized = json.dumps(payload, sort_keys=True)
    assert "private-scene-id" not in serialized
    assert "fixture-provider" not in serialized
    assert "provider-" not in serialized
    assert "latitude" not in serialized
    assert "longitude" not in serialized
    assert "s3://" not in serialized
    assert request.render_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert payload["coordinate_convention"] == (
        "ego_flu_x_forward_y_left_meters"
    )
    assert {item["primitive_id"] for item in payload["primitives"]} >= {
        "route-000",
        "lane-000",
    }


def test_missing_navigation_never_calls_bedrock() -> None:
    scene = replace(
        _scene(),
        navigation_map=None,
        navigation_route=None,
        navigation_quality={},
    )
    client = _BedrockClient("turn_left")
    resolver = BedrockMapRouteResolver(
        client,
        model_id="us.anthropic.claude-opus-5",
        model_revision="claude-opus-5",
    )

    assert build_privacy_safe_request(
        scene,
        _ambiguous_junction(),
    ) is None
    assert resolve_ambiguous_map_route(
        resolver,
        scene,
        (_ambiguous_junction(),),
    ) == ()
    assert client.requests == []
    assert resolver.provider_exchanges == ()


def test_resolver_uses_bedrock_tool_and_geometry_validation() -> None:
    client = _BedrockClient("t_junction")
    resolver = BedrockMapRouteResolver(
        client,
        model_id="us.anthropic.claude-opus-5",
        model_revision="claude-opus-5",
    )

    observations = resolve_ambiguous_map_route(
        resolver,
        _scene(),
        (_ambiguous_junction(),),
    )

    assert len(observations) == 1
    assert observations[0].status == "valid"
    assert observations[0].values == ("t_junction",)
    assert observations[0].confidence == 0.88
    assert observations[0].provenance["model_provider"] == "amazon_bedrock"
    assert observations[0].provenance["selected_value"] == "t_junction"
    assert len(observations[0].provenance["request_sha256"]) == 64
    request = client.requests[0]
    assert request["modelId"] == "us.anthropic.claude-opus-5"
    assert request["toolConfig"]["toolChoice"] == {
        "tool": {"name": BEDROCK_TOOL_NAME}
    }
    assert request["messages"][0]["content"][1]["image"]["format"] == "png"
    assert len(resolver.provider_exchanges) == 1
    exchange = resolver.provider_exchanges[0]
    assert exchange.backend == "BMR"
    assert exchange.status == "succeeded"
    assert exchange.usage == {"input_tokens": 100, "output_tokens": 20}
    assert exchange.raw_response is not None
    exchange_request = json.dumps(
        exchange.request_metadata,
        sort_keys=True,
    )
    assert "private-scene-id" not in exchange_request
    assert "fixture-provider" not in exchange_request
    assert "latitude" not in exchange_request
    assert "longitude" not in exchange_request
    assert "PNG" not in exchange_request


def test_invalid_candidate_preserves_ambiguous_evidence() -> None:
    client = _BedrockClient("straight")
    resolver = BedrockMapRouteResolver(
        client,
        model_id="us.anthropic.claude-opus-5",
        model_revision="claude-opus-5",
    )

    observations = resolve_ambiguous_map_route(
        resolver,
        _scene(),
        (_ambiguous_junction(),),
    )

    assert observations[0].status == "ambiguous"
    assert observations[0].values == ()
    assert observations[0].provenance["error_type"] == "ValueError"
    assert {
        item["value"]
        for item in observations[0].provenance["candidate_values"]
    } == {"t_junction", "y_junction"}
    assert len(resolver.provider_exchanges) == 1
    assert resolver.provider_exchanges[0].status == "invalid_response"
    assert resolver.provider_exchanges[0].raw_response is not None
    assert resolver.provider_exchanges[0].error_type == "ValueError"


def test_route_action_is_never_sent_to_bedrock() -> None:
    observation = make_observation(
        scene_uid="private-scene-id",
        key="odd.route.action",
        status="ambiguous",
        confidence=0.0,
        source="map_route",
        start_timestamp_ns=0,
        end_timestamp_ns=300_000_000,
        provenance={
            "bedrock_eligible": True,
            "candidate_values": ["turn_left", "turn_right"],
        },
    )

    assert build_privacy_safe_request(_scene(), observation) is None
