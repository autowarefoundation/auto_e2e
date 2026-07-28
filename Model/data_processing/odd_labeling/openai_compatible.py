"""Generic OpenAI-compatible multimodal observer for road-scene evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .image_qc import CameraAnchor, CameraFrame
from .ontology import ONTOLOGY
from .schema import LabelObservation, canonical_json_bytes, make_observation


ROAD_VLM_SCHEMA_VERSION = "road_vlm_request_v2"
ROAD_VLM_PROMPT_VERSION = "road_scene_observer_v2"

STATIC_SCENE_KEYS = (
    "odd.road.context",
    "odd.road.lane_marking_quality",
    "odd.road.surface_type",
    "odd.road.surface_state",
    "odd.road.edge_type_present",
    "odd.road.special_structure",
    "odd.road.workzone_state",
    "odd.traffic_control.present",
    "odd.traffic_light.state",
    "odd.environment.day_phase",
    "odd.environment.sky",
    "odd.environment.precipitation_visual",
    "odd.environment.visibility_degradation",
    "odd.environment.road_lighting",
    "odd.environment.glare",
    "perception.occlusion.source",
    "perception.occlusion.level",
    "perception.scene.clutter",
    "perception.image.weather_artifact",
    "perception.image.lens_contamination",
    "perception.map_element_condition",
    "perception.scene.complexity",
    "perception.temporary_traffic_control",
)

DYNAMIC_SCENE_KEYS = (
    "odd.dynamic.traffic_density",
    "odd.dynamic.vru_density",
    "odd.dynamic.parked_vehicle_density",
    "odd.dynamic.oncoming_traffic",
    "odd.dynamic.agent_type_present",
    "event.vehicle.interaction",
    "event.vru.interaction",
    "event.hazard.type",
    "event.traffic_flow",
    "event.interaction.actor",
    "perception.mixed_traffic",
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


def _response_schema(keys: Iterable[str]) -> dict[str, Any]:
    key_list = list(keys)
    observation_properties: dict[str, Any] = {}
    for key in key_list:
        definition = ONTOLOGY[key]
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
                    "supporting_cameras",
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
                            else len(definition.values)
                        ),
                        "items": {
                            "type": "string",
                            "enum": list(definition.values),
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
                    "supporting_cameras",
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
                            "enum": list(definition.values),
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
) -> str:
    definitions = []
    for key in keys:
        definition = ONTOLOGY[key]
        definitions.append(
            {
                "key": key,
                "description": definition.description,
                "cardinality": definition.cardinality,
                "allowed_values": list(definition.values),
                "none_semantics": definition.none_semantics,
            }
        )
    frame_metadata = [
        {
            "ordinal": index,
            "camera_role": frame.camera_role,
            "timestamp_ns": frame.timestamp_ns,
        }
        for index, frame in enumerate(frames)
    ]
    request = {
        "schema_version": ROAD_VLM_SCHEMA_VERSION,
        "task_bundle": task_bundle,
        "scene_uid_hash": scene_uid_hash,
        "requested_labels": definitions,
        "camera_frames": frame_metadata,
        "requirements": {
            "one_observation_per_requested_key": True,
            "valid_requires_at_least_one_allowed_value": True,
            "non_valid_requires_empty_values": True,
            "multi_select_none_is_exclusive": True,
            "events_require_temporal_evidence": True,
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
    prompt_version: str = ROAD_VLM_PROMPT_VERSION,
) -> str:
    definitions = [
        {
            "key": key,
            "description": ONTOLOGY[key].description,
            "cardinality": ONTOLOGY[key].cardinality,
            "allowed_values": list(ONTOLOGY[key].values),
            "none_semantics": ONTOLOGY[key].none_semantics,
        }
        for key in keys
    ]
    bundle = {
        "schema_version": ROAD_VLM_SCHEMA_VERSION,
        "prompt_version": prompt_version,
        "system_prompt": _system_prompt(),
        "task_bundle": task_bundle,
        "requested_labels": definitions,
        "response_schema": _response_schema(keys),
        "requirements": {
            "one_observation_per_requested_key": True,
            "valid_requires_at_least_one_allowed_value": True,
            "non_valid_requires_empty_values": True,
            "multi_select_none_is_exclusive": True,
            "events_require_temporal_evidence": True,
        },
    }
    return hashlib.sha256(canonical_json_bytes(bundle)).hexdigest()


def _validate_response(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
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
            "supporting_cameras",
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
        allowed = set(ONTOLOGY[key].values)
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
        by_key[key] = {
            **raw,
            "values": sorted(set(values)),
            "supporting_cameras": sorted(set(cameras)),
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

    @property
    def endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

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
    ) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
        scene_uid_hash = hashlib.sha256(scene_uid.encode("utf-8")).hexdigest()
        prompt = _task_prompt(
            scene_uid_hash=scene_uid_hash,
            task_bundle=task_bundle,
            keys=keys,
            frames=frames,
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
                "json_schema": _response_schema(keys),
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
        last_error: Exception | None = None
        for attempt in range(self.config.retry_count + 1):
            try:
                response = self._transport(
                    self.endpoint, request, self._headers()
                )
                content_text = _extract_content(response)
                if not content_text:
                    raise ValueError("OpenAI-compatible response content is empty")
                parsed = _parse_json_content(content_text)
                return _validate_response(parsed, keys), {
                    "prompt_sha256": _prompt_bundle_sha256(
                        task_bundle,
                        keys,
                        self.config.prompt_version,
                    ),
                    "decoding_config_sha256": decoding_config_sha256,
                    "request_sha256": request_digest,
                    "response_sha256": hashlib.sha256(
                        content_text.encode("utf-8")
                    ).hexdigest(),
                    "attempt": attempt + 1,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.config.retry_count:
                    time.sleep(min(8.0, 2.0**attempt))
        assert last_error is not None
        raise RoadVLMRequestError(
            (
                "OpenAI-compatible road observer failed after "
                f"{self.config.retry_count + 1} attempts: {last_error}"
            ),
            provenance={
                "prompt_sha256": _prompt_bundle_sha256(
                    task_bundle,
                    keys,
                    self.config.prompt_version,
                ),
                "decoding_config_sha256": decoding_config_sha256,
                "request_sha256": request_digest,
                "attempt": self.config.retry_count + 1,
                "error_type": type(last_error).__name__,
            },
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
    ) -> tuple[LabelObservation, ...]:
        if end_timestamp_ns <= start_timestamp_ns:
            raise ValueError("road VLM interval must be positive")
        frame_timestamps = [frame.timestamp_ns for frame in frames]
        if not frame_timestamps:
            raise ValueError("road VLM request requires camera frames")
        input_start_timestamp_ns = min(frame_timestamps)
        input_end_timestamp_ns = max(frame_timestamps)
        lookback_ns = max(
            0,
            start_timestamp_ns - input_start_timestamp_ns,
        )
        prompt_sha256 = _prompt_bundle_sha256(
            task_bundle,
            keys,
            self.config.prompt_version,
        )
        try:
            response, call_provenance = self._request(
                scene_uid=scene_uid,
                task_bundle=task_bundle,
                keys=keys,
                frames=frames,
            )
        except RoadVLMRequestError as exc:
            return tuple(
                make_observation(
                    scene_uid=scene_uid,
                    key=key,
                    status="unavailable",
                    confidence=0.0,
                    source="vlm",
                    start_timestamp_ns=start_timestamp_ns,
                    end_timestamp_ns=end_timestamp_ns,
                    provenance={
                        "schema_version": ROAD_VLM_SCHEMA_VERSION,
                        "prompt_version": self.config.prompt_version,
                        "prompt_sha256": prompt_sha256,
                        "model": self.config.model,
                        "model_revision": self.config.model_revision,
                        "task_bundle": task_bundle,
                        "input_start_timestamp_ns": input_start_timestamp_ns,
                        "input_end_timestamp_ns": input_end_timestamp_ns,
                        "lookback_ns": lookback_ns,
                        "lookahead_ns": 0,
                        **exc.provenance,
                    },
                )
                for key in keys
            )

        observations: list[LabelObservation] = []
        for item in response:
            observations.append(
                make_observation(
                    scene_uid=scene_uid,
                    key=str(item["key"]),
                    status=str(item["status"]),
                    values=tuple(item["values"]),
                    confidence=float(item["confidence"]),
                    source="vlm",
                    start_timestamp_ns=start_timestamp_ns,
                    end_timestamp_ns=end_timestamp_ns,
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
                        "supporting_cameras": item["supporting_cameras"],
                        "input_start_timestamp_ns": input_start_timestamp_ns,
                        "input_end_timestamp_ns": input_end_timestamp_ns,
                        "lookback_ns": lookback_ns,
                        "lookahead_ns": 0,
                        "reason": str(item["reason"])[:1000],
                        **call_provenance,
                    },
                )
            )
        return tuple(observations)


def label_visual_scene(
    observer: OpenAICompatibleRoadObserver,
    *,
    scene_uid: str,
    scene_end_timestamp_ns: int,
    anchors: tuple[CameraAnchor, ...],
) -> tuple[LabelObservation, ...]:
    observations: list[LabelObservation] = []
    for index, anchor in enumerate(anchors):
        end_timestamp_ns = (
            anchors[index + 1].timestamp_ns
            if index + 1 < len(anchors)
            else scene_end_timestamp_ns
        )
        if end_timestamp_ns <= anchor.timestamp_ns:
            continue
        observations.extend(
            observer.observe(
                scene_uid=scene_uid,
                task_bundle="static_scene",
                keys=STATIC_SCENE_KEYS,
                frames=anchor.frames,
                start_timestamp_ns=anchor.timestamp_ns,
                end_timestamp_ns=end_timestamp_ns,
            )
        )
        temporal_frames = (
            anchors[index - 1].frames + anchor.frames
            if index > 0
            else anchor.frames
        )
        observations.extend(
            observer.observe(
                scene_uid=scene_uid,
                task_bundle="dynamic_scene",
                keys=DYNAMIC_SCENE_KEYS,
                frames=temporal_frames,
                start_timestamp_ns=anchor.timestamp_ns,
                end_timestamp_ns=end_timestamp_ns,
            )
        )
    return tuple(observations)
