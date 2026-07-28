"""Canonical scene-level records for automatic ODD labeling."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

from .ontology import LABEL_SOURCES, LABEL_STATUSES, ONTOLOGY, validate_values


SCHEMA_VERSION = "odd_labelset_v1"
CAPABILITY_SCHEMA_VERSION = "odd_dataset_capabilities_v1"
EVIDENCE_SCHEMA_VERSION = "odd_label_evidence_v1"
EVENT_SCHEMA_VERSION = "odd_event_instance_v1"
RECEIPT_SCHEMA_VERSION = "odd_execution_receipt_v1"

CHANNEL_AVAILABILITY = ("complete", "partial", "absent")
SUBJECT_TYPES = (
    "scene",
    "ego",
    "camera",
    "actor",
    "traffic_control",
    "route_segment",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _require_sha256(value: str, *, name: str, optional: bool = False) -> None:
    if optional and not value:
        return
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclasses.dataclass(frozen=True)
class ChannelCapability:
    availability: str
    coverage_start_ns: int | None
    coverage_end_ns: int | None
    nominal_rate_hz: float | None
    observed_count: int
    missing_count: int
    source_artifact_sha256: str | None
    quality_summary: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.availability not in CHANNEL_AVAILABILITY:
            raise ValueError(f"invalid channel availability: {self.availability}")
        if self.observed_count < 0 or self.missing_count < 0:
            raise ValueError("channel counts must be non-negative")
        if self.nominal_rate_hz is not None and (
            not math.isfinite(self.nominal_rate_hz)
            or self.nominal_rate_hz <= 0.0
        ):
            raise ValueError("nominal channel rate must be finite and positive")
        if (self.coverage_start_ns is None) != (self.coverage_end_ns is None):
            raise ValueError("channel coverage bounds must both be present or absent")
        if (
            self.coverage_start_ns is not None
            and self.coverage_end_ns is not None
            and (
                self.coverage_start_ns < 0
                or self.coverage_end_ns <= self.coverage_start_ns
            )
        ):
            raise ValueError("channel coverage interval must be positive")
        if self.availability == "absent":
            if self.observed_count != 0 or self.coverage_start_ns is not None:
                raise ValueError(
                    "absent channel cannot report observations or coverage"
                )
        if self.source_artifact_sha256 is not None:
            _require_sha256(
                self.source_artifact_sha256,
                name="channel source artifact",
            )

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclasses.dataclass(frozen=True)
class CameraCapability:
    camera_id: str
    canonical_role: str
    channel: ChannelCapability
    calibration_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.camera_id or not self.canonical_role:
            raise ValueError("camera identity and canonical role are required")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclasses.dataclass(frozen=True)
class DatasetCapabilityManifest:
    dataset_name: str
    dataset_version: str
    dataset_manifest_sha256: str
    source_revision: str
    adapter_name: str
    adapter_version: str
    scene_inventory_sha256: str
    canonical_clock: str
    absolute_time_available: bool
    timezone_resolution_available: bool
    cameras: tuple[CameraCapability, ...]
    channels: Mapping[str, ChannelCapability]
    coordinate_frames: tuple[str, ...]
    calibration_refs: tuple[str, ...] = ()
    known_limitations: tuple[str, ...] = ()
    schema_version: str = CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAPABILITY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported capability schema: {self.schema_version}"
            )
        for value, name in (
            (self.dataset_name, "dataset_name"),
            (self.dataset_version, "dataset_version"),
            (self.source_revision, "source_revision"),
            (self.adapter_name, "adapter_name"),
            (self.adapter_version, "adapter_version"),
            (self.canonical_clock, "canonical_clock"),
        ):
            if not value:
                raise ValueError(f"{name} is required")
        _require_sha256(
            self.dataset_manifest_sha256,
            name="dataset manifest",
        )
        _require_sha256(
            self.scene_inventory_sha256,
            name="scene inventory",
        )
        camera_ids = [camera.camera_id for camera in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera capability IDs must be unique")
        required_channels = {
            "map",
            "route",
            "gnss",
            "ins",
            "lidar",
            "object_tracks",
            "can",
        }
        if set(self.channels) != required_channels:
            raise ValueError(
                "capability channels must exactly cover canonical sources"
            )
        if not self.coordinate_frames:
            raise ValueError("at least one coordinate frame is required")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def semantic_sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclasses.dataclass(frozen=True)
class LabelScope:
    dataset_name: str
    dataset_version: str
    scene_uid: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    subject_type: str = "scene"
    subject_id: str | None = None
    anchor_timestamp_ns: int | None = None
    camera_ids: tuple[str, ...] = ()
    coordinate_frame: str | None = None
    spatial_roi: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.dataset_name or not self.dataset_version or not self.scene_uid:
            raise ValueError("scope dataset and scene identity are required")
        if self.subject_type not in SUBJECT_TYPES:
            raise ValueError(f"invalid subject type: {self.subject_type}")
        if (
            self.start_timestamp_ns < 0
            or self.end_timestamp_ns <= self.start_timestamp_ns
        ):
            raise ValueError("scope interval must be positive and ordered")
        if (
            self.anchor_timestamp_ns is not None
            and not self.start_timestamp_ns
            <= self.anchor_timestamp_ns
            < self.end_timestamp_ns
        ):
            raise ValueError("scope anchor must lie inside the interval")
        if self.subject_type in {
            "camera",
            "actor",
            "traffic_control",
            "route_segment",
        } and not self.subject_id:
            raise ValueError(f"{self.subject_type} scope requires subject_id")
        normalized_cameras = tuple(sorted(set(self.camera_ids)))
        object.__setattr__(self, "camera_ids", normalized_cameras)
        if self.subject_type == "camera" and self.subject_id not in normalized_cameras:
            raise ValueError("camera subject must be present in camera_ids")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclasses.dataclass(frozen=True)
class CandidateValue:
    value: str
    score: float
    evidence_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("candidate value is required")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("candidate score must be finite and in [0,1]")


@dataclasses.dataclass(frozen=True)
class Measurement:
    name: str
    value: float | int | str | bool
    unit: str
    quality: str
    aggregation: str

    def __post_init__(self) -> None:
        if not self.name or not self.unit or not self.quality or not self.aggregation:
            raise ValueError("measurement metadata is required")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError(f"measurement {self.name} must be finite")


@dataclasses.dataclass(frozen=True)
class EvidenceReference:
    artifact_uri: str
    artifact_sha256: str
    timestamp_ns: int | None = None
    camera_id: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_uri:
            raise ValueError("evidence artifact URI is required")
        _require_sha256(self.artifact_sha256, name="evidence artifact")
        if self.timestamp_ns is not None and self.timestamp_ns < 0:
            raise ValueError("evidence timestamp must be non-negative")


@dataclasses.dataclass(frozen=True)
class SemanticLabelerProvenance:
    labeler_name: str
    labeler_version: str
    code_commit: str
    container_image_digest: str
    config_sha256: str
    ontology_sha256: str
    input_artifact_sha256s: tuple[str, ...]
    model_provider: str | None = None
    model_name: str | None = None
    model_revision: str | None = None
    prompt_sha256: str | None = None
    decoding_config_sha256: str | None = None
    lookback_ns: int = 0
    lookahead_ns: int = 0

    def __post_init__(self) -> None:
        if not self.labeler_name or not self.labeler_version or not self.code_commit:
            raise ValueError("labeler identity and code revision are required")
        if not self.container_image_digest.startswith("sha256:"):
            raise ValueError("container image must be pinned by digest")
        _require_sha256(
            self.container_image_digest.removeprefix("sha256:"),
            name="container image",
        )
        for value, name in (
            (self.config_sha256, "labeler config"),
            (self.ontology_sha256, "ontology"),
        ):
            _require_sha256(value, name=name)
        for value in self.input_artifact_sha256s:
            _require_sha256(value, name="input artifact")
        if self.prompt_sha256 is not None:
            _require_sha256(self.prompt_sha256, name="prompt")
        if self.decoding_config_sha256 is not None:
            _require_sha256(
                self.decoding_config_sha256,
                name="decoding config",
            )
        if self.lookback_ns < 0 or self.lookahead_ns < 0:
            raise ValueError("retrospective context must be non-negative")
        model_fields = (
            self.model_provider,
            self.model_name,
            self.model_revision,
        )
        if any(model_fields) and not all(model_fields):
            raise ValueError(
                "model provider, name, and revision must be pinned together"
            )

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclasses.dataclass(frozen=True)
class LabelEvidence:
    evidence_uid: str
    label_key: str
    cardinality: str
    values: tuple[str, ...]
    candidate_values: tuple[CandidateValue, ...]
    status: str
    confidence: float
    source: str
    scope: LabelScope
    measurements: tuple[Measurement, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    provenance: SemanticLabelerProvenance
    schema_version: str = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence schema: {self.schema_version}")
        if not self.evidence_uid:
            raise ValueError("evidence_uid is required")
        definition = ONTOLOGY.get(self.label_key)
        if definition is None:
            raise ValueError(f"unknown ontology key: {self.label_key}")
        if self.cardinality != definition.cardinality:
            raise ValueError("evidence cardinality differs from ontology")
        if self.status not in LABEL_STATUSES:
            raise ValueError(f"invalid evidence status: {self.status}")
        if self.source not in LABEL_SOURCES:
            raise ValueError(f"invalid evidence source: {self.source}")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("evidence confidence must be finite and in [0,1]")
        normalized = tuple(sorted(set(self.values)))
        if self.status == "valid":
            normalized = validate_values(self.label_key, normalized)
        elif normalized:
            raise ValueError("non-valid evidence must not carry resolved values")
        object.__setattr__(self, "values", normalized)
        candidate_names = [candidate.value for candidate in self.candidate_values]
        if len(candidate_names) != len(set(candidate_names)):
            raise ValueError("evidence candidate values must be unique")
        unknown_candidates = set(candidate_names) - set(definition.values)
        if unknown_candidates:
            raise ValueError(
                f"unknown candidate values for {self.label_key}: "
                f"{sorted(unknown_candidates)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def semantic_sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclasses.dataclass(frozen=True)
class LabelObservation:
    observation_uid: str
    scene_uid: str
    key: str
    status: str
    values: tuple[str, ...]
    confidence: float
    source: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    evidence_uids: tuple[str, ...] = ()
    conflicting_evidence_uids: tuple[str, ...] = ()
    measurements: Mapping[str, float | int | str | bool] = dataclasses.field(
        default_factory=dict
    )
    provenance: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    camera_id: str | None = None
    actor_track_uid: str | None = None
    event_uid: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported ODD schema: {self.schema_version}")
        if not self.observation_uid or not self.scene_uid:
            raise ValueError("observation_uid and scene_uid are required")
        definition = ONTOLOGY.get(self.key)
        if definition is None:
            raise ValueError(f"unknown ontology key: {self.key}")
        if self.status not in LABEL_STATUSES:
            raise ValueError(f"invalid label status: {self.status}")
        if self.source not in LABEL_SOURCES:
            raise ValueError(f"invalid label source: {self.source}")
        if (
            self.start_timestamp_ns < 0
            or self.end_timestamp_ns <= self.start_timestamp_ns
        ):
            raise ValueError("observation interval must be positive and ordered")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0,1]")

        normalized = tuple(sorted(set(self.values)))
        if self.status == "valid":
            normalized = validate_values(self.key, normalized)
        elif normalized:
            raise ValueError(
                f"{self.status} observation must not carry resolved values: "
                f"{self.key}"
            )
        object.__setattr__(self, "values", normalized)
        object.__setattr__(
            self, "evidence_uids", tuple(sorted(set(self.evidence_uids)))
        )
        object.__setattr__(
            self,
            "conflicting_evidence_uids",
            tuple(sorted(set(self.conflicting_evidence_uids))),
        )
        if set(self.evidence_uids) & set(self.conflicting_evidence_uids):
            raise ValueError("supporting and conflicting evidence must be disjoint")

        if definition.subject == "actor" and not self.actor_track_uid:
            raise ValueError(f"actor-scoped label requires actor_track_uid: {self.key}")
        if definition.subject == "actor_camera":
            if not self.actor_track_uid or not self.camera_id:
                raise ValueError(
                    f"actor-camera label requires actor_track_uid and camera_id: "
                    f"{self.key}"
                )

        for key, value in self.measurements.items():
            if not key:
                raise ValueError("measurement name must not be empty")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"measurement {key} must be finite")

    @property
    def duration_ns(self) -> int:
        return self.end_timestamp_ns - self.start_timestamp_ns

    @property
    def namespace(self) -> str:
        return self.key.split(".", 1)[0]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclasses.dataclass(frozen=True)
class EventPhase:
    phase: str
    start_timestamp_ns: int
    end_timestamp_ns: int

    def __post_init__(self) -> None:
        if self.phase not in {"onset", "active", "resolution"}:
            raise ValueError(f"invalid event phase: {self.phase}")
        if (
            self.start_timestamp_ns < 0
            or self.end_timestamp_ns <= self.start_timestamp_ns
        ):
            raise ValueError("event phase interval must be positive")


@dataclasses.dataclass(frozen=True)
class EventInstance:
    event_uid: str
    scene_uid: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    primary_event_key: str
    actor_track_uids: tuple[str, ...]
    observation_uids: tuple[str, ...]
    phases: tuple[EventPhase, ...]
    confidence: float
    status: str
    supporting_evidence_uids: tuple[str, ...]
    provenance: Mapping[str, Any]
    schema_version: str = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported event schema: {self.schema_version}")
        if not self.event_uid or not self.scene_uid:
            raise ValueError("event and scene identity are required")
        definition = ONTOLOGY.get(self.primary_event_key)
        if definition is None or definition.namespace != "event":
            raise ValueError("primary event key must be an event ontology key")
        if (
            self.start_timestamp_ns < 0
            or self.end_timestamp_ns <= self.start_timestamp_ns
        ):
            raise ValueError("event interval must be positive")
        if self.status not in LABEL_STATUSES:
            raise ValueError(f"invalid event status: {self.status}")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("event confidence must be finite and in [0,1]")
        expected_phase_order = {"onset": 0, "active": 1, "resolution": 2}
        previous_order = -1
        previous_end = self.start_timestamp_ns
        for phase in self.phases:
            order = expected_phase_order[phase.phase]
            if order <= previous_order:
                raise ValueError("event phases must be unique and ordered")
            if (
                phase.start_timestamp_ns < self.start_timestamp_ns
                or phase.end_timestamp_ns > self.end_timestamp_ns
                or phase.start_timestamp_ns < previous_end
            ):
                raise ValueError("event phase is outside or overlaps prior phase")
            previous_order = order
            previous_end = phase.end_timestamp_ns
        object.__setattr__(
            self, "actor_track_uids", tuple(sorted(set(self.actor_track_uids)))
        )
        object.__setattr__(
            self, "observation_uids", tuple(sorted(set(self.observation_uids)))
        )
        object.__setattr__(
            self,
            "supporting_evidence_uids",
            tuple(sorted(set(self.supporting_evidence_uids))),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclasses.dataclass(frozen=True)
class ExecutionReceipt:
    semantic_partition_sha256: str
    created_at: str
    flyte_execution_id: str
    flyte_task_execution_id: str
    attempt: int
    runtime_metrics: Mapping[str, float | int]
    receipt_schema_version: str = RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.receipt_schema_version != RECEIPT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported receipt schema: {self.receipt_schema_version}"
            )
        _require_sha256(
            self.semantic_partition_sha256,
            name="semantic partition",
        )
        if (
            not self.created_at
            or not self.flyte_execution_id
            or not self.flyte_task_execution_id
        ):
            raise ValueError("receipt execution identity is required")
        if self.attempt <= 0:
            raise ValueError("receipt attempt must be positive")
        for name, value in self.runtime_metrics.items():
            if not name or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ValueError("receipt runtime metrics must be finite")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclasses.dataclass(frozen=True)
class SceneLabelRecord:
    scene_uid: str
    dataset_name: str
    dataset_version: str
    dataset_manifest_sha256: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    distance_m: float
    observations: tuple[LabelObservation, ...]
    source_artifact_uri: str
    source_artifact_sha256: str
    evidence: tuple[LabelEvidence, ...] = ()
    events: tuple[EventInstance, ...] = ()
    capability_manifest_sha256: str = ""
    provenance: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported ODD schema: {self.schema_version}")
        required = (
            self.scene_uid,
            self.dataset_name,
            self.dataset_version,
            self.dataset_manifest_sha256,
            self.source_artifact_uri,
            self.source_artifact_sha256,
        )
        if any(not value for value in required):
            raise ValueError("scene identity and provenance fields are required")
        if (
            self.start_timestamp_ns < 0
            or self.end_timestamp_ns <= self.start_timestamp_ns
        ):
            raise ValueError("scene interval must be positive and ordered")
        if not math.isfinite(self.distance_m) or self.distance_m < 0.0:
            raise ValueError("scene distance must be finite and non-negative")
        _require_sha256(
            self.capability_manifest_sha256,
            name="capability manifest",
            optional=True,
        )
        observations = tuple(
            sorted(
                self.observations,
                key=lambda item: (
                    item.start_timestamp_ns,
                    item.end_timestamp_ns,
                    item.key,
                    item.observation_uid,
                ),
            )
        )
        seen: set[str] = set()
        for observation in observations:
            if observation.observation_uid in seen:
                raise ValueError(
                    f"duplicate observation_uid: {observation.observation_uid}"
                )
            seen.add(observation.observation_uid)
            if observation.scene_uid != self.scene_uid:
                raise ValueError("observation belongs to another scene")
            if (
                observation.start_timestamp_ns < self.start_timestamp_ns
                or observation.end_timestamp_ns > self.end_timestamp_ns
            ):
                raise ValueError("observation interval exceeds scene interval")
        object.__setattr__(self, "observations", observations)
        evidence = tuple(sorted(self.evidence, key=lambda item: item.evidence_uid))
        evidence_ids = [item.evidence_uid for item in evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate evidence_uid")
        if any(item.scope.scene_uid != self.scene_uid for item in evidence):
            raise ValueError("evidence belongs to another scene")
        object.__setattr__(self, "evidence", evidence)
        events = tuple(sorted(self.events, key=lambda item: item.event_uid))
        event_ids = [item.event_uid for item in events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("duplicate event_uid")
        if any(item.scene_uid != self.scene_uid for item in events):
            raise ValueError("event belongs to another scene")
        object.__setattr__(self, "events", events)
        known_evidence = set(evidence_ids)
        if known_evidence:
            for observation in observations:
                referenced = set(observation.evidence_uids) | set(
                    observation.conflicting_evidence_uids
                )
                if not referenced.issubset(known_evidence):
                    raise ValueError("observation references unknown evidence")
            for event in events:
                if not set(event.supporting_evidence_uids).issubset(
                    known_evidence
                ):
                    raise ValueError("event references unknown evidence")
        known_observations = set(seen)
        for event in events:
            if not set(event.observation_uids).issubset(known_observations):
                raise ValueError("event references unknown observation")

    @property
    def duration_ns(self) -> int:
        return self.end_timestamp_ns - self.start_timestamp_ns

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def semantic_sha256(self) -> str:
        return content_sha256(self.to_dict())


def make_observation(
    *,
    scene_uid: str,
    key: str,
    status: str,
    values: Iterable[str] = (),
    confidence: float,
    source: str,
    start_timestamp_ns: int,
    end_timestamp_ns: int,
    evidence_uids: Iterable[str] = (),
    conflicting_evidence_uids: Iterable[str] = (),
    measurements: Mapping[str, float | int | str | bool] | None = None,
    provenance: Mapping[str, Any] | None = None,
    camera_id: str | None = None,
    actor_track_uid: str | None = None,
    event_uid: str | None = None,
) -> LabelObservation:
    identity = {
        "scene_uid": scene_uid,
        "key": key,
        "status": status,
        "values": sorted(set(values)),
        "source": source,
        "start_timestamp_ns": int(start_timestamp_ns),
        "end_timestamp_ns": int(end_timestamp_ns),
        "camera_id": camera_id,
        "actor_track_uid": actor_track_uid,
        "event_uid": event_uid,
    }
    observation_uid = f"oddobs-{content_sha256(identity)[:24]}"
    return LabelObservation(
        observation_uid=observation_uid,
        scene_uid=scene_uid,
        key=key,
        status=status,
        values=tuple(identity["values"]),
        confidence=float(confidence),
        source=source,
        start_timestamp_ns=int(start_timestamp_ns),
        end_timestamp_ns=int(end_timestamp_ns),
        evidence_uids=tuple(evidence_uids),
        conflicting_evidence_uids=tuple(conflicting_evidence_uids),
        measurements=dict(measurements or {}),
        provenance=dict(provenance or {}),
        camera_id=camera_id,
        actor_track_uid=actor_track_uid,
        event_uid=event_uid,
    )


def coalesce_observations(
    observations: Iterable[LabelObservation],
    *,
    maximum_gap_ns: int = 0,
) -> tuple[LabelObservation, ...]:
    if maximum_gap_ns < 0:
        raise ValueError("maximum_gap_ns must be non-negative")
    ordered = sorted(
        observations,
        key=lambda item: (
            item.key,
            item.status,
            item.values,
            item.source,
            item.camera_id or "",
            item.actor_track_uid or "",
            item.event_uid or "",
            item.start_timestamp_ns,
            item.end_timestamp_ns,
        ),
    )
    output: list[LabelObservation] = []
    for current in ordered:
        if not output:
            output.append(current)
            continue
        previous = output[-1]
        compatible = (
            previous.scene_uid == current.scene_uid
            and previous.key == current.key
            and previous.status == current.status
            and previous.values == current.values
            and previous.source == current.source
            and previous.camera_id == current.camera_id
            and previous.actor_track_uid == current.actor_track_uid
            and previous.event_uid == current.event_uid
            and current.start_timestamp_ns
            <= previous.end_timestamp_ns + maximum_gap_ns
        )
        if not compatible:
            output.append(current)
            continue
        merged_evidence = set(previous.evidence_uids) | set(current.evidence_uids)
        merged_conflicts = set(previous.conflicting_evidence_uids) | set(
            current.conflicting_evidence_uids
        )
        merged_measurements = dict(previous.measurements)
        merged_measurements.update(current.measurements)
        output[-1] = make_observation(
            scene_uid=previous.scene_uid,
            key=previous.key,
            status=previous.status,
            values=previous.values,
            confidence=min(previous.confidence, current.confidence),
            source=previous.source,
            start_timestamp_ns=previous.start_timestamp_ns,
            end_timestamp_ns=max(
                previous.end_timestamp_ns, current.end_timestamp_ns
            ),
            evidence_uids=merged_evidence,
            conflicting_evidence_uids=merged_conflicts,
            measurements=merged_measurements,
            provenance=previous.provenance,
            camera_id=previous.camera_id,
            actor_track_uid=previous.actor_track_uid,
            event_uid=previous.event_uid,
        )
    return tuple(
        sorted(
            output,
            key=lambda item: (
                item.start_timestamp_ns,
                item.end_timestamp_ns,
                item.key,
                item.observation_uid,
            ),
        )
    )
