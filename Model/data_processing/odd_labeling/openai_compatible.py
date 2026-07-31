"""Generic OpenAI-compatible multimodal observer for road-scene evidence."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .image_qc import CameraAnchor, CameraFrame
from .ontology import ONTOLOGY
from .schema import (
    LabelObservation,
    ProviderExchange,
    canonical_json_bytes,
    content_sha256,
    make_observation,
)


ROAD_VLM_SCHEMA_VERSION = "road_vlm_request_v4"
ROAD_VLM_PROMPT_VERSION = "road_scene_observer_v6"
ROAD_VLM_PROTOCOL_REPAIR_VERSION = "road_vlm_protocol_repair_v2"
DEFAULT_REFINEMENT_CONFIDENCE_THRESHOLD = 0.65
VLM_FRAME_STATUS_VALUES = (
    "normal",
    "partial_obstruction",
    "full_obstruction",
)


@dataclasses.dataclass(frozen=True)
class RoadVLMTaskBundle:
    name: str
    scene_keys: tuple[str, ...] = ()
    camera_keys: tuple[str, ...] = ()
    temporal_mode: str = "static"
    scene_camera_roles: tuple[str, ...] | None = None
    camera_roles: tuple[str, ...] | None = None
    trigger_only: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.scene_keys + self.camera_keys:
            raise ValueError("road VLM task bundle must have a name and keys")
        if self.temporal_mode not in {"static", "short", "event"}:
            raise ValueError(f"invalid temporal mode: {self.temporal_mode}")
        if (
            self.scene_camera_roles is not None
            and (
                not self.scene_camera_roles
                or len(self.scene_camera_roles)
                != len(set(self.scene_camera_roles))
            )
        ):
            raise ValueError("scene camera roles must be unique and non-empty")
        if self.camera_roles is not None and (
            not self.camera_roles
            or len(self.camera_roles) != len(set(self.camera_roles))
        ):
            raise ValueError("camera roles must be unique and non-empty")
        if self.camera_roles is not None and not self.camera_keys:
            raise ValueError("camera roles require camera-scoped keys")
        keys = self.scene_keys + self.camera_keys
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate key in road VLM task bundle: {self.name}")
        unknown = set(keys) - set(ONTOLOGY)
        if unknown:
            raise ValueError(
                f"unknown road VLM task bundle keys: {sorted(unknown)}"
            )


ROAD_VLM_TASK_BUNDLES = (
    RoadVLMTaskBundle(
        name="road_environment",
        scene_keys=(
            "odd.road.context",
            "odd.road.lane_marking_quality",
            "odd.road.surface_type",
            "odd.road.surface_state",
            "odd.road.edge_type_present",
            "odd.road.special_structure",
            "odd.road.workzone_state",
            "odd.environment.day_phase",
            "odd.environment.sky",
            "odd.environment.precipitation_visual",
            "odd.environment.visibility_degradation",
            "odd.environment.road_lighting",
            "odd.environment.glare",
        ),
        scene_camera_roles=("front_center",),
    ),
    RoadVLMTaskBundle(
        name="traffic_dynamic",
        scene_keys=(
            "odd.road.junction_control",
            "odd.traffic_control.present",
            "odd.traffic_light.state",
            "odd.dynamic.traffic_density",
            "odd.dynamic.vru_density",
            "odd.dynamic.parked_vehicle_density",
            "odd.dynamic.oncoming_traffic",
            "odd.dynamic.agent_type_present",
            "perception.mixed_traffic",
            "perception.map_element_condition",
            "perception.scene.complexity",
            "perception.temporary_traffic_control",
        ),
        scene_camera_roles=("front_center",),
    ),
    RoadVLMTaskBundle(
        name="temporal_event",
        scene_keys=(
            "event.vehicle.interaction",
            "event.vru.interaction",
            "event.traffic_control.response",
            "event.right_of_way",
            "event.hazard.type",
            "event.hazard.response",
            "event.traffic_flow",
            "event.interaction.actor",
        ),
        temporal_mode="event",
        scene_camera_roles=("front_center",),
        trigger_only=True,
    ),
    RoadVLMTaskBundle(
        name="forward_perception",
        camera_keys=(
            "perception.occlusion.source",
            "perception.occlusion.level",
            "perception.scene.clutter",
            "perception.visual.contrast",
            "perception.visual.lighting",
            "perception.visual.glare",
            "perception.image.blur",
            "perception.image.frame_status",
            "perception.image.weather_artifact",
            "perception.image.lens_contamination",
        ),
        temporal_mode="short",
        camera_roles=("front_center",),
    ),
)

SAFETY_RELEVANT_REFINEMENT_KEYS = frozenset(
    {
        "odd.road.workzone_state",
        "odd.traffic_light.state",
        "perception.temporary_traffic_control",
        *(
            key
            for bundle in ROAD_VLM_TASK_BUNDLES
            for key in bundle.scene_keys
            if key.startswith("event.")
        ),
    }
)
NEGATIVE_REFINEMENT_VALUES = frozenset(
    {
        "absent",
        "no_response_required",
        "none",
        "none_visible",
        "normal",
        "not_applicable",
    }
)

VISUAL_TRIGGER_KEYS = frozenset(
    {
        "odd.road.junction_position",
        "odd.road.junction_control",
        "odd.route.action",
        "event.ego.motion_state",
        "event.ego.maneuver",
        "event.ego.strong_response",
        "perception.image.frame_status",
        "perception.image.exposure",
        "perception.image.blur",
        "perception.visual.contrast",
        "perception.visual.lighting",
        "perception.visual.glare",
    }
)

VISUAL_TRIGGER_NEUTRAL_VALUES = frozenset(
    {
        "lane_follow",
        "midblock",
        "moving",
        "none",
        "normal",
        "straight",
    }
)


Transport = Callable[
    [str, dict[str, Any], dict[str, str]],
    dict[str, Any],
]


@dataclasses.dataclass(frozen=True)
class RoadVLMConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 180.0
    max_tokens: int = 4096
    retry_count: int = 2
    model_revision: str = ""
    prompt_version: str = ROAD_VLM_PROMPT_VERSION

    def __post_init__(self) -> None:
        if not self.base_url or not self.model:
            raise ValueError("OpenAI-compatible base_url and model are required")
        if self.timeout_s <= 0.0 or self.max_tokens <= 0:
            raise ValueError("timeout and max_tokens must be positive")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")


class RoadVLMRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provenance: Mapping[str, Any],
    ) -> None:
        super().__init__(message)
        self.provenance = dict(provenance)


def _urllib_transport(timeout_s: float) -> Transport:
    def post(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        import urllib.request

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))

    return post


def _extract_content(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping)
        )
    return ""


def _provider_usage(response: Mapping[str, Any]) -> dict[str, int | float]:
    raw = response.get("usage")
    if not isinstance(raw, Mapping):
        return {}
    aliases = {
        "prompt_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
        "total_tokens": "total_tokens",
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


def _parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("road VLM response must be a JSON object")
    return value


def _vlm_allowed_values(key: str) -> tuple[str, ...]:
    if key == "perception.image.frame_status":
        return VLM_FRAME_STATUS_VALUES
    return ONTOLOGY[key].values


def _response_schema(
    keys: Iterable[str],
    *,
    subject_scope: str,
) -> dict[str, Any]:
    if subject_scope not in {"scene", "camera"}:
        raise ValueError(f"invalid road VLM subject scope: {subject_scope}")
    key_list = list(keys)
    observation_properties: dict[str, Any] = {}
    for key in key_list:
        definition = ONTOLOGY[key]
        allowed_values = _vlm_allowed_values(key)
        shared_properties = {
            "key": {"type": "string", "const": key},
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "supporting_cameras": {
                "type": "array",
                "items": {"type": "string"},
            },
            "supporting_timestamps_ns": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
            },
            "camera_id": {
                "type": "null" if subject_scope == "scene" else "string",
            },
            "reason": {"type": "string"},
        }
        valid_observation = (
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "key",
                    "status",
                    "values",
                    "confidence",
                    "camera_id",
                    "supporting_cameras",
                    "supporting_timestamps_ns",
                    "reason",
                ],
                "properties": {
                    **shared_properties,
                    "status": {"type": "string", "const": "valid"},
                    "values": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": (
                            1
                            if definition.cardinality == "single"
                            else len(allowed_values)
                        ),
                        "items": {
                            "type": "string",
                            "enum": list(allowed_values),
                        },
                    },
                },
            }
        )
        abstained_observation = (
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "key",
                    "status",
                    "values",
                    "confidence",
                    "camera_id",
                    "supporting_cameras",
                    "supporting_timestamps_ns",
                    "reason",
                ],
                "properties": {
                    **shared_properties,
                    "status": {
                        "type": "string",
                        "enum": [
                            "unavailable",
                            "not_observable",
                            "ambiguous",
                        ],
                    },
                    "values": {
                        "type": "array",
                        "maxItems": 0,
                        "items": {
                            "type": "string",
                            "enum": list(allowed_values),
                        },
                    },
                },
            }
        )
        observation_properties[key] = {
            "oneOf": [valid_observation, abstained_observation],
        }
    return {
        "name": "road_scene_observations",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["observations"],
            "properties": {
                "observations": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": key_list,
                    "properties": observation_properties,
                }
            },
        },
    }


def _system_prompt() -> str:
    return (
        "You are an offline observer of vehicle-camera road scenes. "
        "Return only the requested JSON object. Use only visible evidence in "
        "the ordered multi-camera temporal clip. A value of none means the "
        "relevant region was observable and the condition was explicitly "
        "absent. Missing, occluded, clipped, or irrelevant evidence must be "
        "not_observable, never none. Use unavailable only when the supplied "
        "input cannot support the key. Use ambiguous for conflicting visible "
        "evidence. Status and values must agree. If the reason names or "
        "describes an allowed candidate, status MUST be valid and values MUST "
        "contain that candidate. Use not_observable only when the reason "
        "explains which required region or subject is not visible. Supplied "
        "images are available and must not be treated as a missing input. "
        "Cite only supplied camera roles and timestamps. For a camera-scoped "
        "request, camera_id must equal subject_camera_id. For a scene-scoped "
        "request, camera_id must be null. "
        "For a multi-select label, none or normal is exclusive: when any "
        "positive or abnormal candidate applies, omit the neutral candidate. "
        "For perception.image.frame_status, visual inference is limited to "
        "normal, partial_obstruction, or full_obstruction. Never infer black, "
        "frozen, dropped, or corrupted frame states; those require "
        "authoritative decoder and timing evidence. "
        "Do not infer hard braking, evasive steering, collision, "
        "near collision, or actor intent from a single image. Do not invent "
        "values outside the candidate list."
    )


def _task_prompt(
    *,
    scene_uid_hash: str,
    task_bundle: str,
    keys: tuple[str, ...],
    frames: tuple[CameraFrame, ...],
    start_timestamp_ns: int,
    end_timestamp_ns: int,
    target_camera_id: str | None,
    inference_pass: str,
    refinement_reasons: Mapping[str, str],
    sampling_parameters: Mapping[str, Any],
) -> str:
    definitions = []
    for key in keys:
        definition = ONTOLOGY[key]
        definitions.append(
            {
                "key": key,
                "description": definition.description,
                "cardinality": definition.cardinality,
                "allowed_values": list(_vlm_allowed_values(key)),
                "none_semantics": definition.none_semantics,
            }
        )
    frame_metadata = [
        {
            "ordinal": index,
            "camera_role": frame.camera_role,
            "timestamp_ns": frame.timestamp_ns,
            "frame_index": frame.frame_index,
        }
        for index, frame in enumerate(frames)
    ]
    request = {
        "schema_version": ROAD_VLM_SCHEMA_VERSION,
        "task_bundle": task_bundle,
        "scene_uid_hash": scene_uid_hash,
        "clip_start_timestamp_ns": start_timestamp_ns,
        "clip_end_timestamp_ns": end_timestamp_ns,
        "subject_scope": "camera" if target_camera_id else "scene",
        "subject_camera_id": target_camera_id,
        "inference_pass": inference_pass,
        "refinement_reasons": dict(sorted(refinement_reasons.items())),
        "sampling_parameters": dict(sorted(sampling_parameters.items())),
        "requested_labels": definitions,
        "camera_frames": frame_metadata,
        "requirements": {
            "one_observation_per_requested_key": True,
            "valid_requires_at_least_one_allowed_value": True,
            "non_valid_requires_empty_values": True,
            "multi_select_none_is_exclusive": True,
            "multi_select_neutral_is_exclusive": True,
            "events_require_temporal_evidence": True,
            "supporting_evidence_must_cite_supplied_frames": True,
        },
    }
    return (
        "Classify the following ordered scene clip. Each image after this text "
        "matches camera_frames by ordinal. Respond with exactly one observation "
        "for every requested key.\n"
        + json.dumps(request, sort_keys=True, separators=(",", ":"))
    )


def _prompt_bundle_sha256(
    task_bundle: str,
    keys: tuple[str, ...],
    subject_scope: str,
    prompt_version: str = ROAD_VLM_PROMPT_VERSION,
) -> str:
    definitions = [
        {
            "key": key,
            "description": ONTOLOGY[key].description,
            "cardinality": ONTOLOGY[key].cardinality,
            "allowed_values": list(_vlm_allowed_values(key)),
            "none_semantics": ONTOLOGY[key].none_semantics,
        }
        for key in keys
    ]
    bundle = {
        "schema_version": ROAD_VLM_SCHEMA_VERSION,
        "prompt_version": prompt_version,
        "system_prompt": _system_prompt(),
        "task_bundle": task_bundle,
        "subject_scope": subject_scope,
        "requested_labels": definitions,
        "response_schema": _response_schema(
            keys,
            subject_scope=subject_scope,
        ),
        "protocol_repair_version": ROAD_VLM_PROTOCOL_REPAIR_VERSION,
        "requirements": {
            "one_observation_per_requested_key": True,
            "valid_requires_at_least_one_allowed_value": True,
            "non_valid_requires_empty_values": True,
            "multi_select_none_is_exclusive": True,
            "multi_select_neutral_is_exclusive": True,
            "events_require_temporal_evidence": True,
            "supporting_evidence_must_cite_supplied_frames": True,
        },
    }
    return hashlib.sha256(canonical_json_bytes(bundle)).hexdigest()


def road_vlm_prompt_bundle_document(
    prompt_version: str = ROAD_VLM_PROMPT_VERSION,
) -> dict[str, Any]:
    """Return the complete immutable prompt contract used by a Full Run."""
    entries = []
    for bundle in ROAD_VLM_TASK_BUNDLES:
        for subject_scope, keys in (
            ("scene", bundle.scene_keys),
            ("camera", bundle.camera_keys),
        ):
            if not keys:
                continue
            entries.append(
                {
                    "task_bundle": bundle.name,
                    "subject_scope": subject_scope,
                    "temporal_mode": bundle.temporal_mode,
                    "scene_camera_roles": (
                        list(bundle.scene_camera_roles)
                        if bundle.scene_camera_roles is not None
                        else ["all_available"]
                    ),
                    "camera_roles": (
                        list(bundle.camera_roles)
                        if bundle.camera_roles is not None
                        else ["all_available"]
                    ),
                    "trigger_only": bundle.trigger_only,
                    "keys": list(keys),
                    "prompt_sha256": _prompt_bundle_sha256(
                        bundle.name,
                        keys,
                        subject_scope,
                        prompt_version,
                    ),
                }
            )
    return {
        "schema_version": "road_vlm_prompt_bundle_v1",
        "request_schema_version": ROAD_VLM_SCHEMA_VERSION,
        "prompt_version": prompt_version,
        "protocol_repair_version": ROAD_VLM_PROTOCOL_REPAIR_VERSION,
        "entries": entries,
        "refinement": {
            "safety_relevant_keys": sorted(
                SAFETY_RELEVANT_REFINEMENT_KEYS
            ),
            "negative_values": sorted(NEGATIVE_REFINEMENT_VALUES),
        },
    }


def road_vlm_prompt_bundle_sha256(
    prompt_version: str = ROAD_VLM_PROMPT_VERSION,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            road_vlm_prompt_bundle_document(prompt_version)
        )
    ).hexdigest()


def road_vlm_decoding_bundle_sha256(*, max_tokens: int = 4096) -> str:
    if max_tokens <= 0:
        raise ValueError("road VLM max_tokens must be positive")
    schemas = []
    for bundle in ROAD_VLM_TASK_BUNDLES:
        for subject_scope, keys in (
            ("scene", bundle.scene_keys),
            ("camera", bundle.camera_keys),
        ):
            if not keys:
                continue
            schemas.append(
                {
                    "task_bundle": bundle.name,
                    "subject_scope": subject_scope,
                    "response_schema": _response_schema(
                        keys,
                        subject_scope=subject_scope,
                    ),
                }
            )
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": "road_vlm_decoding_bundle_v1",
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "protocol_repair_version": ROAD_VLM_PROTOCOL_REPAIR_VERSION,
                "schemas": schemas,
            }
        )
    ).hexdigest()


def _repair_response_protocol(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    frames: tuple[CameraFrame, ...],
    target_camera_id: str | None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    repaired = copy.deepcopy(dict(payload))
    raw_observations = repaired.get("observations")
    if not isinstance(raw_observations, dict):
        return repaired, ()

    repairs: list[dict[str, Any]] = []
    allowed_cameras = {frame.camera_role for frame in frames}
    allowed_timestamps = {frame.timestamp_ns for frame in frames}
    for key in keys:
        raw = raw_observations.get(key)
        if not isinstance(raw, dict):
            continue

        camera_id = raw.get("camera_id")
        supporting_cameras = raw.get("supporting_cameras")
        if (
            target_camera_id is None
            and isinstance(camera_id, str)
            and camera_id in allowed_cameras
            and isinstance(supporting_cameras, list)
            and camera_id in supporting_cameras
        ):
            raw["camera_id"] = None
            repairs.append(
                {
                    "kind": "scene_camera_id_to_null",
                    "key": key,
                    "before": camera_id,
                    "after": None,
                }
            )

        values = raw.get("values")
        definition = ONTOLOGY[key]
        neutral = definition.neutral_value
        if (
            raw.get("status") == "valid"
            and definition.cardinality == "multi"
            and neutral is not None
            and isinstance(values, list)
            and all(isinstance(value, str) for value in values)
            and set(values) <= set(_vlm_allowed_values(key))
            and neutral in values
            and len(set(values)) > 1
        ):
            canonical_values = [
                value for value in values if value != neutral
            ]
            raw["values"] = canonical_values
            repairs.append(
                {
                    "kind": "exclusive_neutral_removed",
                    "key": key,
                    "before": values,
                    "after": canonical_values,
                }
            )

        supporting_timestamps = raw.get("supporting_timestamps_ns")
        if (
            isinstance(supporting_timestamps, list)
            and all(
                isinstance(timestamp, int)
                and not isinstance(timestamp, bool)
                for timestamp in supporting_timestamps
            )
        ):
            supported = [
                timestamp
                for timestamp in supporting_timestamps
                if timestamp in allowed_timestamps
            ]
            if supported and len(supported) != len(supporting_timestamps):
                raw["supporting_timestamps_ns"] = supported
                repairs.append(
                    {
                        "kind": "unsupported_timestamps_removed",
                        "key": key,
                        "before": supporting_timestamps,
                        "after": supported,
                    }
                )
    return repaired, tuple(repairs)


def _validate_response(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    frames: tuple[CameraFrame, ...],
    target_camera_id: str | None,
) -> tuple[dict[str, Any], ...]:
    raw_observations = payload.get("observations")
    if not isinstance(raw_observations, Mapping):
        raise ValueError("road VLM response has no observations object")
    if set(raw_observations) != set(keys):
        raise ValueError("road VLM observation keys differ from request")
    by_key: dict[str, dict[str, Any]] = {}
    for requested_key in keys:
        raw = raw_observations[requested_key]
        if not isinstance(raw, dict):
            raise ValueError("road VLM observation must be an object")
        if set(raw) != {
            "key",
            "status",
            "values",
            "confidence",
            "camera_id",
            "supporting_cameras",
            "supporting_timestamps_ns",
            "reason",
        }:
            raise ValueError("road VLM observation fields differ from schema")
        key = str(raw["key"])
        if key != requested_key:
            raise ValueError(f"road VLM observation key differs: {key}")
        status = str(raw["status"])
        if status not in {
            "valid",
            "unavailable",
            "not_observable",
            "ambiguous",
        }:
            raise ValueError(f"invalid road VLM status: {status}")
        values = raw["values"]
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"invalid road VLM values: {key}")
        allowed = set(_vlm_allowed_values(key))
        if set(values) - allowed:
            raise ValueError(f"road VLM returned an unknown value: {key}")
        if status == "valid":
            if not values:
                raise ValueError(f"valid road VLM observation has no value: {key}")
            if ONTOLOGY[key].cardinality == "single" and len(set(values)) != 1:
                raise ValueError(f"single road VLM observation has many values: {key}")
            if "none" in values and len(set(values)) != 1:
                raise ValueError(f"road VLM none coexists with another value: {key}")
            if "normal" in values and len(set(values)) != 1:
                raise ValueError(f"road VLM normal coexists with abnormal value: {key}")
        elif values:
            raise ValueError(f"non-valid road VLM observation has values: {key}")
        confidence = float(raw["confidence"])
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"invalid road VLM confidence: {key}")
        cameras = raw["supporting_cameras"]
        if not isinstance(cameras, list) or not all(
            isinstance(value, str) for value in cameras
        ):
            raise ValueError(f"invalid supporting cameras: {key}")
        allowed_cameras = {frame.camera_role for frame in frames}
        if set(cameras) - allowed_cameras:
            raise ValueError(f"road VLM cited an unknown camera: {key}")
        timestamps = raw["supporting_timestamps_ns"]
        if not isinstance(timestamps, list) or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in timestamps
        ):
            raise ValueError(f"invalid supporting timestamps: {key}")
        allowed_timestamps = {frame.timestamp_ns for frame in frames}
        if set(timestamps) - allowed_timestamps:
            raise ValueError(f"road VLM cited an unknown timestamp: {key}")
        camera_id = raw["camera_id"]
        if camera_id != target_camera_id:
            raise ValueError(f"road VLM camera identity differs: {key}")
        if status == "valid" and (not cameras or not timestamps):
            raise ValueError(f"valid road VLM output lacks frame evidence: {key}")
        if target_camera_id and set(cameras) - {target_camera_id}:
            raise ValueError(f"camera-scoped road VLM output crossed cameras: {key}")
        by_key[key] = {
            **raw,
            "values": sorted(set(values)),
            "supporting_cameras": sorted(set(cameras)),
            "supporting_timestamps_ns": sorted(set(timestamps)),
        }
    if set(by_key) != set(keys):
        missing = sorted(set(keys) - set(by_key))
        raise ValueError(f"road VLM response is missing keys: {missing}")
    return tuple(by_key[key] for key in keys)


class OpenAICompatibleRoadObserver:
    def __init__(
        self,
        config: RoadVLMConfig,
        *,
        transport: Transport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or _urllib_transport(config.timeout_s)
        self._provider_exchanges: list[ProviderExchange] = []

    @property
    def endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    @property
    def provider_exchanges(self) -> tuple[ProviderExchange, ...]:
        return tuple(self._provider_exchanges)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _request(
        self,
        *,
        scene_uid: str,
        task_bundle: str,
        keys: tuple[str, ...],
        frames: tuple[CameraFrame, ...],
        start_timestamp_ns: int,
        end_timestamp_ns: int,
        target_camera_id: str | None,
        inference_pass: str,
        refinement_reasons: Mapping[str, str],
        sampling_parameters: Mapping[str, Any],
    ) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
        scene_uid_hash = hashlib.sha256(scene_uid.encode("utf-8")).hexdigest()
        prompt = _task_prompt(
            scene_uid_hash=scene_uid_hash,
            task_bundle=task_bundle,
            keys=keys,
            frames=frames,
            start_timestamp_ns=start_timestamp_ns,
            end_timestamp_ns=end_timestamp_ns,
            target_camera_id=target_camera_id,
            inference_pass=inference_pass,
            refinement_reasons=refinement_reasons,
            sampling_parameters=sampling_parameters,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": frame.data_url()},
                }
            )
        request = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": self.config.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": _response_schema(
                    keys,
                    subject_scope=(
                        "camera" if target_camera_id else "scene"
                    ),
                ),
            },
        }
        decoding_config_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "temperature": request["temperature"],
                    "max_tokens": request["max_tokens"],
                    "response_format": request["response_format"],
                }
            )
        ).hexdigest()
        request_digest = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
        prompt_sha256 = _prompt_bundle_sha256(
            task_bundle,
            keys,
            "camera" if target_camera_id else "scene",
            self.config.prompt_version,
        )
        request_metadata = {
            "schema_version": ROAD_VLM_SCHEMA_VERSION,
            "scene_uid_sha256": scene_uid_hash,
            "task_bundle": task_bundle,
            "keys": list(keys),
            "subject_scope": "camera" if target_camera_id else "scene",
            "subject_camera_id": target_camera_id,
            "start_timestamp_ns": start_timestamp_ns,
            "end_timestamp_ns": end_timestamp_ns,
            "inference_pass": inference_pass,
            "refinement_reasons": dict(refinement_reasons),
            "sampling_parameters": dict(sampling_parameters),
            "prompt_sha256": prompt_sha256,
            "decoding_config_sha256": decoding_config_sha256,
            "frames": [
                {
                    "camera_role": frame.camera_role,
                    "timestamp_ns": frame.timestamp_ns,
                    "jpeg_sha256": hashlib.sha256(frame.jpeg).hexdigest(),
                }
                for frame in frames
            ],
        }
        last_error: Exception | None = None
        last_exchange: ProviderExchange | None = None
        for attempt in range(self.config.retry_count + 1):
            response: Mapping[str, Any] | None = None
            protocol_repairs: tuple[dict[str, Any], ...] = ()
            started_at = time.perf_counter()
            try:
                response = self._transport(
                    self.endpoint, request, self._headers()
                )
                content_text = _extract_content(response)
                if not content_text:
                    raise ValueError("OpenAI-compatible response content is empty")
                parsed = _parse_json_content(content_text)
                repaired, protocol_repairs = _repair_response_protocol(
                    parsed,
                    keys,
                    frames=frames,
                    target_camera_id=target_camera_id,
                )
                validated = _validate_response(
                    repaired,
                    keys,
                    frames=frames,
                    target_camera_id=target_camera_id,
                )
                response_sha256 = content_sha256(response)
                exchange = ProviderExchange(
                    backend="ORV",
                    provider="openai_compatible",
                    model=self.config.model,
                    model_revision=(
                        self.config.model_revision or "unversioned"
                    ),
                    request_sha256=request_digest,
                    response_sha256=response_sha256,
                    status="succeeded",
                    attempt=attempt + 1,
                    latency_ms=(
                        time.perf_counter() - started_at
                    )
                    * 1000.0,
                    input_image_count=len(frames),
                    request_metadata=request_metadata,
                    raw_response=response,
                    protocol_repairs=protocol_repairs,
                    usage=_provider_usage(response),
                )
                self._provider_exchanges.append(exchange)
                return validated, {
                    "prompt_sha256": prompt_sha256,
                    "decoding_config_sha256": decoding_config_sha256,
                    "request_sha256": request_digest,
                    "response_sha256": response_sha256,
                    "attempt": attempt + 1,
                    "protocol_repair_version": (
                        ROAD_VLM_PROTOCOL_REPAIR_VERSION
                    ),
                    "protocol_repairs": [
                        dict(repair) for repair in protocol_repairs
                    ],
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                response_sha256 = (
                    content_sha256(response)
                    if response is not None
                    else None
                )
                last_exchange = ProviderExchange(
                    backend="ORV",
                    provider="openai_compatible",
                    model=self.config.model,
                    model_revision=(
                        self.config.model_revision or "unversioned"
                    ),
                    request_sha256=request_digest,
                    response_sha256=response_sha256,
                    status=(
                        "invalid_response"
                        if response is not None
                        else "transport_error"
                    ),
                    attempt=attempt + 1,
                    latency_ms=(
                        time.perf_counter() - started_at
                    )
                    * 1000.0,
                    input_image_count=len(frames),
                    request_metadata=request_metadata,
                    raw_response=response,
                    protocol_repairs=protocol_repairs,
                    usage=(
                        _provider_usage(response)
                        if response is not None
                        else {}
                    ),
                    error_type=type(exc).__name__,
                )
                self._provider_exchanges.append(last_exchange)
                if attempt < self.config.retry_count:
                    time.sleep(min(8.0, 2.0**attempt))
        assert last_error is not None
        failure_provenance = {
            "prompt_sha256": prompt_sha256,
            "decoding_config_sha256": decoding_config_sha256,
            "request_sha256": request_digest,
            "attempt": self.config.retry_count + 1,
            "error_type": type(last_error).__name__,
        }
        if last_exchange is not None and last_exchange.response_sha256:
            failure_provenance["response_sha256"] = (
                last_exchange.response_sha256
            )
        failure_provenance["protocol_repair_version"] = (
            ROAD_VLM_PROTOCOL_REPAIR_VERSION
        )
        failure_provenance["protocol_repairs"] = (
            [
                dict(repair)
                for repair in last_exchange.protocol_repairs
            ]
            if last_exchange is not None
            else []
        )
        raise RoadVLMRequestError(
            (
                "OpenAI-compatible road observer failed after "
                f"{self.config.retry_count + 1} attempts: {last_error}"
            ),
            provenance=failure_provenance,
        ) from last_error

    def observe(
        self,
        *,
        scene_uid: str,
        task_bundle: str,
        keys: tuple[str, ...],
        frames: tuple[CameraFrame, ...],
        start_timestamp_ns: int,
        end_timestamp_ns: int,
        target_camera_id: str | None = None,
        inference_pass: str = "primary",
        refinement_reasons: Mapping[str, str] | None = None,
        sampling_parameters: Mapping[str, Any] | None = None,
    ) -> tuple[LabelObservation, ...]:
        if end_timestamp_ns <= start_timestamp_ns:
            raise ValueError("road VLM interval must be positive")
        frame_timestamps = [frame.timestamp_ns for frame in frames]
        if not frame_timestamps:
            raise ValueError("road VLM request requires camera frames")
        if inference_pass not in {"primary", "refinement"}:
            raise ValueError(f"invalid road VLM inference pass: {inference_pass}")
        if target_camera_id is not None and (
            not target_camera_id
            or any(
                frame.camera_role != target_camera_id
                for frame in frames
            )
        ):
            raise ValueError(
                "camera-scoped road VLM frames must match target_camera_id"
            )
        refinement_reasons = dict(refinement_reasons or {})
        sampling_parameters = dict(sampling_parameters or {})
        if set(refinement_reasons) - set(keys):
            raise ValueError("road VLM refinement reason has an unknown key")
        input_start_timestamp_ns = min(frame_timestamps)
        input_end_timestamp_ns = max(frame_timestamps)
        lookback_ns = max(
            0,
            start_timestamp_ns - input_start_timestamp_ns,
        )
        lookahead_ns = max(
            0,
            input_end_timestamp_ns - end_timestamp_ns,
        )
        subject_scope = "camera" if target_camera_id else "scene"
        prompt_sha256 = _prompt_bundle_sha256(
            task_bundle,
            keys,
            subject_scope,
            self.config.prompt_version,
        )
        try:
            response, call_provenance = self._request(
                scene_uid=scene_uid,
                task_bundle=task_bundle,
                keys=keys,
                frames=frames,
                start_timestamp_ns=start_timestamp_ns,
                end_timestamp_ns=end_timestamp_ns,
                target_camera_id=target_camera_id,
                inference_pass=inference_pass,
                refinement_reasons=refinement_reasons,
                sampling_parameters=sampling_parameters,
            )
        except RoadVLMRequestError as exc:
            return tuple(
                _make_vlm_observation(
                    request_identity=exc.provenance,
                    scene_uid=scene_uid,
                    key=key,
                    status="unavailable",
                    confidence=0.0,
                    source="vlm",
                    start_timestamp_ns=start_timestamp_ns,
                    end_timestamp_ns=end_timestamp_ns,
                    camera_id=target_camera_id,
                    provenance={
                        "schema_version": ROAD_VLM_SCHEMA_VERSION,
                        "prompt_version": self.config.prompt_version,
                        "prompt_sha256": prompt_sha256,
                        "model": self.config.model,
                        "model_revision": self.config.model_revision,
                        "task_bundle": task_bundle,
                        "subject_scope": subject_scope,
                        "inference_pass": inference_pass,
                        "refinement_reasons": refinement_reasons,
                        "sampling_parameters": sampling_parameters,
                        "input_start_timestamp_ns": input_start_timestamp_ns,
                        "input_end_timestamp_ns": input_end_timestamp_ns,
                        "request_frame_timestamps_ns": frame_timestamps,
                        "request_camera_roles": [
                            frame.camera_role for frame in frames
                        ],
                        "lookback_ns": lookback_ns,
                        "lookahead_ns": lookahead_ns,
                        **exc.provenance,
                    },
                )
                for key in keys
            )

        observations: list[LabelObservation] = []
        for item in response:
            item_call_provenance = {
                **call_provenance,
                "protocol_repairs": [
                    repair
                    for repair in call_provenance["protocol_repairs"]
                    if repair["key"] == item["key"]
                ],
            }
            observations.append(
                _make_vlm_observation(
                    request_identity=item_call_provenance,
                    scene_uid=scene_uid,
                    key=str(item["key"]),
                    status=str(item["status"]),
                    values=tuple(item["values"]),
                    confidence=float(item["confidence"]),
                    source="vlm",
                    start_timestamp_ns=start_timestamp_ns,
                    end_timestamp_ns=end_timestamp_ns,
                    camera_id=target_camera_id,
                    measurements={
                        "supporting_camera_count": len(
                            item["supporting_cameras"]
                        ),
                        "input_frame_count": len(frames),
                    },
                    provenance={
                        "schema_version": ROAD_VLM_SCHEMA_VERSION,
                        "prompt_version": self.config.prompt_version,
                        "model": self.config.model,
                        "model_revision": self.config.model_revision,
                        "task_bundle": task_bundle,
                        "subject_scope": subject_scope,
                        "inference_pass": inference_pass,
                        "refinement_reasons": refinement_reasons,
                        "sampling_parameters": sampling_parameters,
                        "supporting_cameras": item["supporting_cameras"],
                        "supporting_timestamps_ns": item[
                            "supporting_timestamps_ns"
                        ],
                        "input_start_timestamp_ns": input_start_timestamp_ns,
                        "input_end_timestamp_ns": input_end_timestamp_ns,
                        "request_frame_timestamps_ns": frame_timestamps,
                        "request_camera_roles": [
                            frame.camera_role for frame in frames
                        ],
                        "lookback_ns": lookback_ns,
                        "lookahead_ns": lookahead_ns,
                        "reason": str(item["reason"])[:1000],
                        **item_call_provenance,
                    },
                )
            )
        return tuple(observations)


def _make_vlm_observation(
    *,
    request_identity: Mapping[str, Any],
    **kwargs: Any,
) -> LabelObservation:
    observation = make_observation(**kwargs)
    identity = {
        "base_observation_uid": observation.observation_uid,
        "request_sha256": request_identity.get("request_sha256", ""),
        "response_sha256": request_identity.get("response_sha256", ""),
        "inference_pass": observation.provenance.get("inference_pass", ""),
    }
    return dataclasses.replace(
        observation,
        observation_uid=(
            "oddobs-"
            + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:24]
        ),
    )


def _anchor_indexes(
    anchor_count: int,
    index: int,
    temporal_mode: str,
) -> tuple[int, ...]:
    if temporal_mode == "static":
        candidates = (index,)
    elif temporal_mode == "short":
        candidates = (index - 1, index)
    elif temporal_mode == "event":
        candidates = (index - 1, index, index + 1)
    else:
        raise ValueError(f"invalid temporal mode: {temporal_mode}")
    return tuple(
        candidate
        for candidate in candidates
        if 0 <= candidate < anchor_count
    )


def _frames_for_indexes(
    anchors: tuple[CameraAnchor, ...],
    indexes: Iterable[int],
    *,
    camera_id: str | None = None,
    camera_roles: tuple[str, ...] | None = None,
) -> tuple[CameraFrame, ...]:
    if camera_id is not None and camera_roles is not None:
        raise ValueError("camera_id and camera_roles are mutually exclusive")
    return tuple(
        frame
        for index in indexes
        for frame in anchors[index].frames
        if (
            (camera_id is None or frame.camera_role == camera_id)
            and (
                camera_roles is None
                or frame.camera_role in camera_roles
            )
        )
    )


def _refinement_frames(
    anchors: tuple[CameraAnchor, ...],
    index: int,
    *,
    temporal_mode: str,
    camera_id: str | None = None,
    camera_roles: tuple[str, ...] | None = None,
) -> tuple[CameraFrame, ...]:
    if temporal_mode == "event":
        indexes = (
            index - 2,
            index - 1,
            index,
            index + 1,
            index + 2,
        )
    else:
        indexes = (index - 1, index, index + 1)
    bounded = tuple(
        candidate
        for candidate in indexes
        if 0 <= candidate < len(anchors)
    )
    return _frames_for_indexes(
        anchors,
        bounded,
        camera_id=camera_id,
        camera_roles=camera_roles,
    )


def _frame_identity(frames: Iterable[CameraFrame]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            frame.timestamp_ns,
            frame.camera_role,
            frame.frame_index,
            hashlib.sha256(frame.jpeg).hexdigest(),
        )
        for frame in frames
    )


def _refinement_plan(
    observations: Iterable[LabelObservation],
    *,
    confidence_threshold: float,
) -> dict[str, str]:
    plan: dict[str, str] = {}
    for observation in observations:
        if observation.status == "unavailable":
            continue
        if observation.status in {"ambiguous", "not_observable"}:
            plan[observation.key] = observation.status
            continue
        if observation.confidence < confidence_threshold:
            plan[observation.key] = "low_confidence"
            continue
        if (
            observation.key in SAFETY_RELEVANT_REFINEMENT_KEYS
            and observation.status == "valid"
            and set(observation.values) - NEGATIVE_REFINEMENT_VALUES
        ):
            plan[observation.key] = "positive_safety_relevant"
    return plan


def derive_visual_trigger_timestamps(
    *observation_groups: Iterable[LabelObservation],
) -> tuple[int, ...]:
    grouped: dict[tuple[str, str, str], list[LabelObservation]] = {}
    for observation in (
        item for group in observation_groups for item in group
    ):
        if (
            observation.key not in VISUAL_TRIGGER_KEYS
            or observation.status == "unavailable"
        ):
            continue
        identity = (
            observation.key,
            observation.camera_id or "",
            observation.actor_track_uid or "",
        )
        grouped.setdefault(identity, []).append(observation)

    triggers: set[int] = set()
    for rows in grouped.values():
        ordered = sorted(
            rows,
            key=lambda item: (
                item.start_timestamp_ns,
                item.end_timestamp_ns,
                item.observation_uid,
            ),
        )
        previous_signature: tuple[str, tuple[str, ...]] | None = None
        for observation in ordered:
            signature = (observation.status, observation.values)
            positive = (
                observation.status == "valid"
                and bool(
                    set(observation.values)
                    - VISUAL_TRIGGER_NEUTRAL_VALUES
                )
            )
            changed = (
                previous_signature is not None
                and signature != previous_signature
            )
            positive_onset = previous_signature is None and positive
            if changed or positive_onset:
                triggers.add(observation.start_timestamp_ns)
                if observation.namespace == "event":
                    triggers.add(observation.end_timestamp_ns - 1)
            previous_signature = signature
    return tuple(sorted(triggers))


def _trigger_anchor_indexes(
    anchors: tuple[CameraAnchor, ...],
    trigger_timestamps_ns: Iterable[int],
) -> frozenset[int]:
    if not anchors:
        return frozenset()
    timestamps = tuple(anchor.timestamp_ns for anchor in anchors)
    return frozenset(
        min(
            range(len(timestamps)),
            key=lambda index: (
                abs(timestamps[index] - int(trigger_timestamp_ns)),
                timestamps[index],
            ),
        )
        for trigger_timestamp_ns in set(trigger_timestamps_ns)
    )


def label_visual_scene(
    observer: OpenAICompatibleRoadObserver,
    *,
    scene_uid: str,
    scene_end_timestamp_ns: int,
    anchors: tuple[CameraAnchor, ...],
    event_trigger_timestamps_ns: tuple[int, ...] = (),
    refinement_confidence_threshold: float = (
        DEFAULT_REFINEMENT_CONFIDENCE_THRESHOLD
    ),
    sampling_parameters: Mapping[str, Any] | None = None,
) -> tuple[LabelObservation, ...]:
    if not 0.0 <= refinement_confidence_threshold <= 1.0:
        raise ValueError("refinement confidence threshold must be in [0,1]")
    if any(
        left.timestamp_ns >= right.timestamp_ns
        for left, right in zip(anchors, anchors[1:])
    ):
        raise ValueError("road VLM anchors must be strictly ordered")
    sampling_parameters = dict(sampling_parameters or {})
    trigger_anchor_indexes = _trigger_anchor_indexes(
        anchors,
        event_trigger_timestamps_ns,
    )

    observations: list[LabelObservation] = []
    for index, anchor in enumerate(anchors):
        end_timestamp_ns = (
            anchors[index + 1].timestamp_ns
            if index + 1 < len(anchors)
            else scene_end_timestamp_ns
        )
        if end_timestamp_ns <= anchor.timestamp_ns:
            continue
        for bundle in ROAD_VLM_TASK_BUNDLES:
            if bundle.trigger_only and index not in trigger_anchor_indexes:
                continue
            primary_indexes = _anchor_indexes(
                len(anchors),
                index,
                bundle.temporal_mode,
            )
            if bundle.scene_keys:
                primary_frames = _frames_for_indexes(
                    anchors,
                    primary_indexes,
                    camera_roles=bundle.scene_camera_roles,
                )
                primary = observer.observe(
                    scene_uid=scene_uid,
                    task_bundle=bundle.name,
                    keys=bundle.scene_keys,
                    frames=primary_frames,
                    start_timestamp_ns=anchor.timestamp_ns,
                    end_timestamp_ns=end_timestamp_ns,
                    sampling_parameters=sampling_parameters,
                )
                observations.extend(primary)
                refinement_plan = _refinement_plan(
                    primary,
                    confidence_threshold=(
                        refinement_confidence_threshold
                    ),
                )
                alternate_frames = _refinement_frames(
                    anchors,
                    index,
                    temporal_mode=bundle.temporal_mode,
                    camera_roles=bundle.scene_camera_roles,
                )
                if (
                    refinement_plan
                    and _frame_identity(alternate_frames)
                    != _frame_identity(primary_frames)
                ):
                    observations.extend(
                        observer.observe(
                            scene_uid=scene_uid,
                            task_bundle=bundle.name,
                            keys=tuple(refinement_plan),
                            frames=alternate_frames,
                            start_timestamp_ns=anchor.timestamp_ns,
                            end_timestamp_ns=end_timestamp_ns,
                            inference_pass="refinement",
                            refinement_reasons=refinement_plan,
                            sampling_parameters=sampling_parameters,
                        )
                    )

            if bundle.camera_keys:
                camera_ids = tuple(
                    dict.fromkeys(
                        frame.camera_role
                        for frame in anchor.frames
                        if (
                            bundle.camera_roles is None
                            or frame.camera_role in bundle.camera_roles
                        )
                    )
                )
                for camera_id in camera_ids:
                    primary_frames = _frames_for_indexes(
                        anchors,
                        primary_indexes,
                        camera_id=camera_id,
                    )
                    primary = observer.observe(
                        scene_uid=scene_uid,
                        task_bundle=bundle.name,
                        keys=bundle.camera_keys,
                        frames=primary_frames,
                        start_timestamp_ns=anchor.timestamp_ns,
                        end_timestamp_ns=end_timestamp_ns,
                        target_camera_id=camera_id,
                        sampling_parameters=sampling_parameters,
                    )
                    observations.extend(primary)
                    refinement_plan = _refinement_plan(
                        primary,
                        confidence_threshold=(
                            refinement_confidence_threshold
                        ),
                    )
                    alternate_frames = _refinement_frames(
                        anchors,
                        index,
                        temporal_mode=bundle.temporal_mode,
                        camera_id=camera_id,
                    )
                    if (
                        refinement_plan
                        and _frame_identity(alternate_frames)
                        != _frame_identity(primary_frames)
                    ):
                        observations.extend(
                            observer.observe(
                                scene_uid=scene_uid,
                                task_bundle=bundle.name,
                                keys=tuple(refinement_plan),
                                frames=alternate_frames,
                                start_timestamp_ns=anchor.timestamp_ns,
                                end_timestamp_ns=end_timestamp_ns,
                                target_camera_id=camera_id,
                                inference_pass="refinement",
                                refinement_reasons=refinement_plan,
                                sampling_parameters=sampling_parameters,
                            )
                        )
    return tuple(observations)
