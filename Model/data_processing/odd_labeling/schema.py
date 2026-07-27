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
