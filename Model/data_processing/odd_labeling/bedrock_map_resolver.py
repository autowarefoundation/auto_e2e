"""Privacy-filtered Bedrock fallback for ambiguous map/route topology."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import time
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw

from .ontology import ONTOLOGY
from .published_snapshot import CanonicalSceneEvidence
from .schema import (
    LabelObservation,
    ProviderExchange,
    canonical_json_bytes,
    content_sha256,
    make_observation,
)


BEDROCK_MAP_SCHEMA_VERSION = "bedrock_map_junction_request_v2"
BEDROCK_MAP_PROMPT_VERSION = "bedrock_map_junction_resolver_v2"
BEDROCK_TOOL_NAME = "resolve_map_route_topology"
MAX_LOCAL_RANGE_M = 120.0
RENDER_SIZE_PX = 512

SUPPORTED_KEYS = {
    "odd.road.junction_type",
}

FORBIDDEN_REQUEST_TOKENS = {
    "latitude",
    "longitude",
    "lat",
    "lon",
    "street",
    "provider_id",
    "scene_uid",
    "dataset",
    "source_uri",
    "camera",
    "image_uri",
}


class BedrockRuntimeClient(Protocol):
    def converse(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclasses.dataclass(frozen=True)
class MapRouteCandidate:
    value: str
    score: float
    rationale: str

    def __post_init__(self) -> None:
        if not self.value or not 0.0 <= self.score <= 1.0:
            raise ValueError("map/route candidate is invalid")


@dataclasses.dataclass(frozen=True)
class PrivacySafeMapRouteRequest:
    label_key: str
    interval_duration_ns: int
    candidates: tuple[MapRouteCandidate, ...]
    primitives: tuple[Mapping[str, Any], ...]
    topology_summary: Mapping[str, Any]
    geometry_checks: Mapping[str, Any]
    render_png: bytes
    geometry_id: str

    def __post_init__(self) -> None:
        if self.label_key not in SUPPORTED_KEYS:
            raise ValueError(f"unsupported Bedrock map key: {self.label_key}")
        if self.interval_duration_ns <= 0:
            raise ValueError("map/route request interval must be positive")
        allowed = set(ONTOLOGY[self.label_key].values)
        values = [candidate.value for candidate in self.candidates]
        if not values or len(values) != len(set(values)):
            raise ValueError("map/route candidates must be unique and non-empty")
        if set(values) - allowed:
            raise ValueError("map/route request contains unknown candidates")
        if not self.primitives or not self.render_png:
            raise ValueError("map/route request needs primitives and a render")
        _assert_privacy_safe(self.payload())

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": BEDROCK_MAP_SCHEMA_VERSION,
            "label_key": self.label_key,
            "allowed_values": [
                candidate.value for candidate in self.candidates
            ],
            "coordinate_convention": "ego_flu_x_forward_y_left_meters",
            "interval_duration_ns": self.interval_duration_ns,
            "geometry_id": self.geometry_id,
            "semantic_layers": [
                "lane_centerlines",
                "lane_directions",
                "intersection_polygons",
                "route_corridor",
                "route_transition",
            ],
            "primitives": [dict(item) for item in self.primitives],
            "topology_summary": dict(self.topology_summary),
            "deterministic_candidates": [
                dataclasses.asdict(candidate)
                for candidate in self.candidates
            ],
            "geometry_checks": dict(self.geometry_checks),
            "required_evidence": {
                "cite_ephemeral_primitive_ids": True,
                "select_only_deterministic_candidates": True,
                "camera_imagery_prohibited": True,
                "geographic_identifiers_prohibited": True,
            },
        }


def _assert_privacy_safe(value: Any, *, path: str = "request") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_REQUEST_TOKENS:
                raise ValueError(
                    f"privacy-prohibited field in Bedrock request: {path}.{key}"
                )
            _assert_privacy_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_privacy_safe(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith(("s3://", "http://", "https://", "data:image")):
            raise ValueError(
                f"privacy-prohibited reference in Bedrock request: {path}"
            )


def _local_xy(
    path: np.ndarray,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
) -> np.ndarray:
    radius = 6_371_008.8
    latitude = np.radians(path[:, 0])
    longitude = np.radians(path[:, 1])
    latitude_origin = math.radians(origin_latitude_deg)
    longitude_origin = math.radians(origin_longitude_deg)
    east = (
        radius
        * (longitude - longitude_origin)
        * math.cos(latitude_origin)
    )
    north = radius * (latitude - latitude_origin)
    return np.column_stack([east, north])


def _ego_flu(
    points_enu: np.ndarray,
    *,
    ego_position_enu: np.ndarray,
    ego_yaw_rad: float,
) -> np.ndarray:
    delta = np.asarray(points_enu, dtype=np.float64)[:, :2] - ego_position_enu
    cosine = math.cos(ego_yaw_rad)
    sine = math.sin(ego_yaw_rad)
    forward = delta[:, 0] * cosine + delta[:, 1] * sine
    left = -delta[:, 0] * sine + delta[:, 1] * cosine
    return np.column_stack([forward, left])


def _closest_segment(
    point: np.ndarray,
    polyline: np.ndarray,
) -> tuple[float, int]:
    points = np.asarray(polyline, dtype=np.float64)[:, :2]
    starts = points[:-1]
    vectors = points[1:] - starts
    squared = np.einsum("ij,ij->i", vectors, vectors)
    parameters = np.divide(
        np.einsum("ij,ij->i", point - starts, vectors),
        squared,
        out=np.zeros_like(squared),
        where=squared > 0.0,
    )
    parameters = np.clip(parameters, 0.0, 1.0)
    closest = starts + parameters[:, None] * vectors
    distances = np.linalg.norm(closest - point, axis=1)
    index = int(np.argmin(distances))
    return float(distances[index]), index


def _signed_heading_change_deg(points_flu: np.ndarray) -> float:
    points = np.asarray(points_flu, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    vectors = np.diff(points[:, :2], axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = vectors[lengths >= 0.5]
    if len(valid) < 2:
        return 0.0
    first = valid[0]
    last = valid[-1]
    cross = float(first[0] * last[1] - first[1] * last[0])
    dot = float(np.dot(first, last))
    return math.degrees(math.atan2(cross, dot))


def _junction_candidates(branch_count: int) -> tuple[MapRouteCandidate, ...]:
    if branch_count == 3:
        return (
            MapRouteCandidate("t_junction", 0.6, "three branches"),
            MapRouteCandidate("y_junction", 0.4, "three oblique branches"),
        )
    return ()


def _primitive_points(
    points_flu: np.ndarray,
) -> list[list[float]]:
    return [
        [round(float(point[0]), 3), round(float(point[1]), 3)]
        for point in points_flu
        if np.linalg.norm(point[:2]) <= MAX_LOCAL_RANGE_M
    ]


def _semantic_render(
    primitives: Iterable[Mapping[str, Any]],
) -> bytes:
    image = Image.new("RGB", (RENDER_SIZE_PX, RENDER_SIZE_PX), "white")
    draw = ImageDraw.Draw(image)
    scale = RENDER_SIZE_PX / (MAX_LOCAL_RANGE_M * 2.0)

    def pixel(point: Iterable[float]) -> tuple[float, float]:
        forward, left = point
        return (
            RENDER_SIZE_PX * 0.5 - float(left) * scale,
            RENDER_SIZE_PX * 0.75 - float(forward) * scale,
        )

    colors = {
        "lane_centerline": "#8a8f98",
        "route_segment": "#1f6feb",
        "intersection": "#d29922",
    }
    for primitive in primitives:
        points = primitive.get("points_flu_m")
        if not isinstance(points, list) or len(points) < 2:
            continue
        pixels = [pixel(point) for point in points]
        color = colors.get(str(primitive.get("kind")), "#57606a")
        if primitive.get("kind") == "intersection" and len(pixels) >= 3:
            draw.polygon(pixels, outline=color, width=3)
        else:
            draw.line(pixels, fill=color, width=4)
    ego = pixel((0.0, 0.0))
    draw.polygon(
        [
            (ego[0], ego[1] - 10),
            (ego[0] - 7, ego[1] + 8),
            (ego[0] + 7, ego[1] + 8),
        ],
        fill="#cf222e",
    )
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def build_privacy_safe_request(
    evidence: CanonicalSceneEvidence,
    observation: LabelObservation,
) -> PrivacySafeMapRouteRequest | None:
    if (
        observation.source != "map_route"
        or observation.status != "ambiguous"
        or observation.key not in SUPPORTED_KEYS
        or evidence.navigation_map is None
        or observation.provenance.get("bedrock_eligible") is not True
        or set(observation.provenance.get("candidate_values", ()))
        != {"t_junction", "y_junction"}
    ):
        return None
    path = evidence.path_latlon_heading_timestamp
    timestamps = path[:, 3].astype(np.int64)
    anchor_ns = (
        observation.start_timestamp_ns + observation.end_timestamp_ns
    ) // 2
    path_index = int(np.argmin(np.abs(timestamps - anchor_ns)))
    map_frame = evidence.navigation_map.frame
    path_enu = _local_xy(
        path,
        map_frame.origin_latitude_deg,
        map_frame.origin_longitude_deg,
    )
    ego_position = path_enu[path_index]
    ego_yaw = math.radians(90.0 - float(path[path_index, 2]))

    matched_lane_id = str(
        observation.provenance.get("matched_lane_id", "")
    )
    route_segment = None
    route_distance = math.inf
    for candidate in evidence.navigation_map.directed_lane_fields:
        distance, _ = _closest_segment(
            ego_position,
            candidate.centerline_enu_m,
        )
        if candidate.lane_id == matched_lane_id:
            route_segment = candidate
            route_distance = distance
            break
        if distance < route_distance:
            route_segment = candidate
            route_distance = distance
    if route_segment is None or route_distance > 15.0:
        return None

    primitives: list[dict[str, Any]] = []
    route_points_flu = _ego_flu(
        route_segment.centerline_enu_m,
        ego_position_enu=ego_position,
        ego_yaw_rad=ego_yaw,
    )
    route_points = _primitive_points(route_points_flu)
    if len(route_points) < 2:
        return None
    primitives.append(
        {
            "primitive_id": "route-000",
            "kind": "matched_lane",
            "points_flu_m": route_points,
        }
    )

    for lane_index, lane in enumerate(
        evidence.navigation_map.directed_lane_fields
    ):
        distance, _ = _closest_segment(
            ego_position,
            lane.centerline_enu_m,
        )
        if distance > 40.0:
            continue
        points = _primitive_points(
            _ego_flu(
                lane.centerline_enu_m,
                ego_position_enu=ego_position,
                ego_yaw_rad=ego_yaw,
            )
        )
        if len(points) < 2:
            continue
        primitives.append(
            {
                "primitive_id": f"lane-{lane_index:03d}",
                "kind": "lane_centerline",
                "points_flu_m": points,
            }
        )

    for polygon_index, polygon in enumerate(
        evidence.navigation_map.intersection_polygons
    ):
        points = _primitive_points(
            _ego_flu(
                polygon.points_enu_m,
                ego_position_enu=ego_position,
                ego_yaw_rad=ego_yaw,
            )
        )
        if len(points) < 3:
            continue
        primitives.append(
            {
                "primitive_id": f"intersection-{polygon_index:03d}",
                "kind": "intersection",
                "points_flu_m": points,
            }
        )

    heading_change_deg = _signed_heading_change_deg(route_points_flu)
    deterministic_branch_count = int(
        observation.measurements.get("junction_branch_count", 0)
    )
    if deterministic_branch_count != 3:
        return None
    candidates = _junction_candidates(deterministic_branch_count)
    if len(candidates) < 2:
        return None

    geometry_basis = {
        "label_key": observation.key,
        "primitives": primitives,
        "heading_change_deg": round(heading_change_deg, 3),
        "branch_count": deterministic_branch_count,
        "transition": "map_only",
    }
    geometry_id = hashlib.sha256(
        canonical_json_bytes(geometry_basis)
    ).hexdigest()
    request = PrivacySafeMapRouteRequest(
        label_key=observation.key,
        interval_duration_ns=(
            observation.end_timestamp_ns
            - observation.start_timestamp_ns
        ),
        candidates=candidates,
        primitives=tuple(primitives),
        topology_summary={
            "branch_count": deterministic_branch_count,
            "intersection_polygon_count": sum(
                primitive["kind"] == "intersection"
                for primitive in primitives
            ),
            "route_transition": "map_only",
        },
        geometry_checks={
            "signed_route_heading_change_deg": round(
                heading_change_deg,
                3,
            ),
            "branch_count": deterministic_branch_count,
        },
        render_png=_semantic_render(primitives),
        geometry_id=geometry_id,
    )
    return request


def _tool_schema(request: PrivacySafeMapRouteRequest) -> dict[str, Any]:
    candidate_values = [candidate.value for candidate in request.candidates]
    return {
        "toolSpec": {
            "name": BEDROCK_TOOL_NAME,
            "description": (
                "Resolve one map/route topology ambiguity using only the "
                "provided semantic primitives."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "status",
                        "value",
                        "confidence",
                        "cited_primitive_ids",
                        "candidate_rejections",
                    ],
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["resolved", "ambiguous"],
                        },
                        "value": {
                            "type": "string",
                            "enum": ["", *candidate_values],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "cited_primitive_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "candidate_rejections": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                }
            },
        }
    }


def _system_prompt() -> str:
    return (
        "You resolve only a T-junction versus Y-junction static map "
        "topology ambiguity. The input is an "
        "ego-local semantic map render plus structured primitives. It contains "
        "no camera imagery and no geographic identity. Select only an allowed "
        "deterministic candidate, cite primitive IDs, and use ambiguous when "
        "the supplied topology is insufficient. Do not infer visual road, "
        "weather, actor, surface, or traffic-light-state facts."
    )


def _prompt_sha256(request: PrivacySafeMapRouteRequest) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "prompt_version": BEDROCK_MAP_PROMPT_VERSION,
                "system": _system_prompt(),
                "tool": _tool_schema(request),
            }
        )
    ).hexdigest()


def bedrock_map_prompt_bundle_document() -> dict[str, Any]:
    return {
        "schema_version": "bedrock_map_prompt_bundle_v1",
        "request_schema_version": BEDROCK_MAP_SCHEMA_VERSION,
        "prompt_version": BEDROCK_MAP_PROMPT_VERSION,
        "tool_name": BEDROCK_TOOL_NAME,
        "system_prompt": _system_prompt(),
        "supported_labels": {
            key: list(ONTOLOGY[key].values)
            for key in sorted(SUPPORTED_KEYS)
        },
        "input_policy": {
            "coordinate_convention": "ego_flu_x_forward_y_left_meters",
            "maximum_local_range_m": MAX_LOCAL_RANGE_M,
            "render_size_px": RENDER_SIZE_PX,
            "forbidden_request_tokens": sorted(FORBIDDEN_REQUEST_TOKENS),
            "camera_imagery_prohibited": True,
            "geographic_identifiers_prohibited": True,
            "select_only_deterministic_candidates": True,
            "t_y_junction_only": True,
            "cite_ephemeral_primitive_ids": True,
        },
        "tool_output_contract": {
            "additional_properties": False,
            "required": [
                "status",
                "value",
                "confidence",
                "cited_primitive_ids",
                "candidate_rejections",
            ],
            "statuses": ["resolved", "ambiguous"],
        },
    }


def bedrock_map_prompt_bundle_sha256() -> str:
    return hashlib.sha256(
        canonical_json_bytes(bedrock_map_prompt_bundle_document())
    ).hexdigest()


def bedrock_map_decoding_config_sha256(*, max_tokens: int = 1024) -> str:
    if max_tokens <= 0:
        raise ValueError("Bedrock map max_tokens must be positive")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": "bedrock_map_decoding_config_v1",
                "maxTokens": max_tokens,
                "temperature": 0.0,
                "toolChoice": {"tool": {"name": BEDROCK_TOOL_NAME}},
            }
        )
    ).hexdigest()


def _extract_tool_input(response: Mapping[str, Any]) -> dict[str, Any]:
    try:
        content = response["output"]["message"]["content"]  # type: ignore[index]
    except (KeyError, TypeError):
        raise ValueError("Bedrock response has no message content") from None
    if not isinstance(content, list):
        raise ValueError("Bedrock response content must be a list")
    tools = [
        item["toolUse"]
        for item in content
        if isinstance(item, Mapping)
        and isinstance(item.get("toolUse"), Mapping)
        and item["toolUse"].get("name") == BEDROCK_TOOL_NAME
    ]
    if len(tools) != 1 or not isinstance(tools[0].get("input"), dict):
        raise ValueError("Bedrock response must contain one resolver tool use")
    return dict(tools[0]["input"])


def _provider_usage(response: Mapping[str, Any]) -> dict[str, int | float]:
    raw = response.get("usage")
    if not isinstance(raw, Mapping):
        return {}
    aliases = {
        "inputTokens": "input_tokens",
        "outputTokens": "output_tokens",
        "totalTokens": "total_tokens",
    }
    usage: dict[str, int | float] = {}
    for source_name, target_name in aliases.items():
        value = raw.get(source_name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value >= 0
        ):
            usage[target_name] = value
    return usage


def _raw_model_response(response: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: response[key]
        for key in ("output", "stopReason", "usage", "metrics")
        if key in response
    }


def _validate_geometry(
    request: PrivacySafeMapRouteRequest,
    value: str,
) -> bool:
    branches = int(request.geometry_checks.get("branch_count", 0))
    if value in {"t_junction", "y_junction"}:
        return branches == 3
    return False


def _validate_tool_result(
    request: PrivacySafeMapRouteRequest,
    result: Mapping[str, Any],
) -> tuple[str, str, tuple[str, ...], float]:
    expected_fields = {
        "status",
        "value",
        "confidence",
        "cited_primitive_ids",
        "candidate_rejections",
    }
    if set(result) != expected_fields:
        raise ValueError("Bedrock resolver fields differ from schema")
    status = str(result["status"])
    value = str(result["value"])
    confidence = float(result["confidence"])
    citations = result["cited_primitive_ids"]
    if status not in {"resolved", "ambiguous"}:
        raise ValueError("Bedrock resolver status is invalid")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("Bedrock resolver confidence is invalid")
    if not isinstance(citations, list) or not all(
        isinstance(item, str) for item in citations
    ):
        raise ValueError("Bedrock resolver citations are invalid")
    known_ids = {
        str(primitive["primitive_id"])
        for primitive in request.primitives
    }
    if not set(citations).issubset(known_ids):
        raise ValueError("Bedrock resolver cited an unknown primitive")
    allowed = {candidate.value for candidate in request.candidates}
    if status == "ambiguous":
        if value:
            raise ValueError("ambiguous Bedrock result must not select a value")
        return status, "", tuple(sorted(set(citations))), confidence
    if value not in allowed or not citations:
        raise ValueError("resolved Bedrock result lacks candidate or citation")
    if not _validate_geometry(request, value):
        raise ValueError("Bedrock result failed independent geometry validation")
    return status, value, tuple(sorted(set(citations))), confidence


class BedrockMapRouteResolver:
    def __init__(
        self,
        client: BedrockRuntimeClient,
        *,
        model_id: str,
        model_revision: str,
        max_tokens: int = 1024,
    ) -> None:
        if not model_id or not model_revision or max_tokens <= 0:
            raise ValueError("Bedrock model identity and token cap are required")
        self.client = client
        self.model_id = model_id
        self.model_revision = model_revision
        self.max_tokens = max_tokens
        self._provider_exchanges: list[ProviderExchange] = []

    @property
    def provider_exchanges(self) -> tuple[ProviderExchange, ...]:
        return tuple(self._provider_exchanges)

    def resolve(
        self,
        request: PrivacySafeMapRouteRequest,
    ) -> tuple[str, str, tuple[str, ...], float, dict[str, Any]]:
        payload = request.payload()
        prompt_sha256 = _prompt_sha256(request)
        decoding = {
            "maxTokens": self.max_tokens,
            "temperature": 0.0,
        }
        decoding_sha256 = hashlib.sha256(
            canonical_json_bytes(decoding)
        ).hexdigest()
        request_identity = {
            "model_id": self.model_id,
            "system": _system_prompt(),
            "payload": payload,
            "render_png_sha256": hashlib.sha256(
                request.render_png
            ).hexdigest(),
            "tool": _tool_schema(request),
            "decoding": decoding,
        }
        request_sha256 = hashlib.sha256(
            canonical_json_bytes(request_identity)
        ).hexdigest()
        request_metadata = {
            "schema_version": BEDROCK_MAP_SCHEMA_VERSION,
            "prompt_sha256": prompt_sha256,
            "decoding_config_sha256": decoding_sha256,
            "payload": payload,
            "render_png_sha256": request_identity["render_png_sha256"],
            "tool": request_identity["tool"],
        }
        _assert_privacy_safe(request_metadata)
        response: Mapping[str, Any] | None = None
        started_at = time.perf_counter()
        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": _system_prompt()}],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": json.dumps(
                                    payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                            },
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": request.render_png},
                                }
                            },
                        ],
                    }
                ],
                toolConfig={
                    "tools": [_tool_schema(request)],
                    "toolChoice": {"tool": {"name": BEDROCK_TOOL_NAME}},
                },
                inferenceConfig=decoding,
            )
            raw_response = _raw_model_response(response)
            result = _extract_tool_input(response)
            status, selected_value, citations, confidence = (
                _validate_tool_result(
                    request,
                    result,
                )
            )
            response_sha256 = content_sha256(raw_response)
            self._provider_exchanges.append(
                ProviderExchange(
                    backend="BMR",
                    provider="amazon_bedrock",
                    model=self.model_id,
                    model_revision=self.model_revision,
                    request_sha256=request_sha256,
                    response_sha256=response_sha256,
                    status="succeeded",
                    attempt=1,
                    latency_ms=(
                        time.perf_counter() - started_at
                    )
                    * 1000.0,
                    input_image_count=1,
                    request_metadata=request_metadata,
                    raw_response=raw_response,
                    usage=_provider_usage(response),
                )
            )
        except Exception as error:
            raw_response = (
                _raw_model_response(response)
                if response is not None
                else None
            )
            self._provider_exchanges.append(
                ProviderExchange(
                    backend="BMR",
                    provider="amazon_bedrock",
                    model=self.model_id,
                    model_revision=self.model_revision,
                    request_sha256=request_sha256,
                    response_sha256=(
                        content_sha256(raw_response)
                        if raw_response is not None
                        else None
                    ),
                    status=(
                        "geometry_rejected"
                        if "independent geometry validation" in str(error)
                        else (
                            "invalid_response"
                            if response is not None
                            else "transport_error"
                        )
                    ),
                    attempt=1,
                    latency_ms=(
                        time.perf_counter() - started_at
                    )
                    * 1000.0,
                    input_image_count=1,
                    request_metadata=request_metadata,
                    raw_response=raw_response,
                    usage=(
                        _provider_usage(response)
                        if response is not None
                        else {}
                    ),
                    error_type=type(error).__name__,
                )
            )
            raise
        return status, selected_value, citations, confidence, {
            "schema_version": BEDROCK_MAP_SCHEMA_VERSION,
            "prompt_version": BEDROCK_MAP_PROMPT_VERSION,
            "prompt_sha256": prompt_sha256,
            "decoding_config_sha256": decoding_sha256,
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            "model_provider": "amazon_bedrock",
            "model": self.model_id,
            "model_revision": self.model_revision,
            "geometry_id": request.geometry_id,
            "selected_value": selected_value,
            "cited_primitive_ids": list(citations),
            "candidate_values": [
                {
                    "value": candidate.value,
                    "score": candidate.score,
                }
                for candidate in request.candidates
            ],
            "bedrock_stop_reason": response.get("stopReason"),
            "bedrock_usage": response.get("usage", {}),
        }


def resolve_ambiguous_map_route(
    resolver: BedrockMapRouteResolver,
    evidence: CanonicalSceneEvidence,
    observations: Iterable[LabelObservation],
) -> tuple[LabelObservation, ...]:
    output: list[LabelObservation] = []
    for observation in observations:
        request = build_privacy_safe_request(evidence, observation)
        if request is None:
            continue
        base_provenance = {
            "schema_version": BEDROCK_MAP_SCHEMA_VERSION,
            "prompt_version": BEDROCK_MAP_PROMPT_VERSION,
            "model_provider": "amazon_bedrock",
            "model": resolver.model_id,
            "model_revision": resolver.model_revision,
            "geometry_id": request.geometry_id,
            "candidate_values": [
                {
                    "value": candidate.value,
                    "score": candidate.score,
                }
                for candidate in request.candidates
            ],
        }
        try:
            status, selected, _, confidence, provenance = resolver.resolve(
                request
            )
        except Exception as exc:  # noqa: BLE001
            exchange = resolver.provider_exchanges[-1]
            output.append(
                make_observation(
                    scene_uid=observation.scene_uid,
                    key=observation.key,
                    status="ambiguous",
                    confidence=observation.confidence,
                    source="map_route",
                    start_timestamp_ns=observation.start_timestamp_ns,
                    end_timestamp_ns=observation.end_timestamp_ns,
                    provenance={
                        **base_provenance,
                        "request_sha256": exchange.request_sha256,
                        "response_sha256": exchange.response_sha256,
                        "attempt": exchange.attempt,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                    },
                )
            )
            continue
        output.append(
            make_observation(
                scene_uid=observation.scene_uid,
                key=observation.key,
                status="valid" if status == "resolved" else "ambiguous",
                values=(selected,) if status == "resolved" else (),
                confidence=confidence,
                source="map_route",
                start_timestamp_ns=observation.start_timestamp_ns,
                end_timestamp_ns=observation.end_timestamp_ns,
                provenance=provenance,
            )
        )
    return tuple(output)
