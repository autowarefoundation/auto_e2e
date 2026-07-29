from __future__ import annotations

import json

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


def _ambiguous_route_action():
    return make_observation(
        scene_uid="private-scene-id",
        key="odd.route.action",
        status="ambiguous",
        confidence=0.0,
        source="map_route",
        start_timestamp_ns=0,
        end_timestamp_ns=300_000_000,
        provenance={"labeler_version": "deterministic-v1"},
    )


def test_bedrock_map_semantic_hashes_are_stable() -> None:
    assert bedrock_map_prompt_bundle_sha256() == (
        "b68bd2ec197bbb35e9bffb9e48e3a4d7a3d99e14821d92e8d5705b72c3752e8f"
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
    request = build_privacy_safe_request(_scene(), _ambiguous_route_action())

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


def test_resolver_uses_bedrock_tool_and_geometry_validation() -> None:
    client = _BedrockClient("turn_left")
    resolver = BedrockMapRouteResolver(
        client,
        model_id="us.anthropic.claude-sonnet-4-6",
        model_revision="claude-sonnet-4-6",
    )

    observations = resolve_ambiguous_map_route(
        resolver,
        _scene(),
        (_ambiguous_route_action(),),
    )

    assert len(observations) == 1
    assert observations[0].status == "valid"
    assert observations[0].values == ("turn_left",)
    assert observations[0].confidence == 0.88
    assert observations[0].provenance["model_provider"] == "amazon_bedrock"
    assert observations[0].provenance["selected_value"] == "turn_left"
    assert len(observations[0].provenance["request_sha256"]) == 64
    request = client.requests[0]
    assert request["modelId"] == "us.anthropic.claude-sonnet-4-6"
    assert request["toolConfig"]["toolChoice"] == {
        "tool": {"name": BEDROCK_TOOL_NAME}
    }
    assert request["messages"][0]["content"][1]["image"]["format"] == "png"


def test_geometry_rejection_preserves_ambiguous_evidence() -> None:
    client = _BedrockClient("turn_right")
    resolver = BedrockMapRouteResolver(
        client,
        model_id="us.anthropic.claude-sonnet-4-6",
        model_revision="claude-sonnet-4-6",
    )

    observations = resolve_ambiguous_map_route(
        resolver,
        _scene(),
        (_ambiguous_route_action(),),
    )

    assert observations[0].status == "ambiguous"
    assert observations[0].values == ()
    assert observations[0].provenance["error_type"] == "ValueError"
    assert {
        item["value"]
        for item in observations[0].provenance["candidate_values"]
    } == {"turn_left", "straight"}
