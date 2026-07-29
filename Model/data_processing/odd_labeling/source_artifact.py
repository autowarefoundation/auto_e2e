"""Canonical handoff artifact between independently cached source labelers."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from typing import Any

from .schema import (
    LabelObservation,
    ProviderExchange,
    canonical_json_bytes,
    content_sha256,
)


SOURCE_ARTIFACT_SCHEMA_VERSION = "odd_source_observations_v2"
SOURCE_STAGES = {
    "map_route_deterministic",
    "gnss_ins",
    "image_qc",
    "openai_compatible_vlm",
    "bedrock_map_route",
}
PROVIDER_BACKEND_BY_SOURCE_STAGE = {
    "openai_compatible_vlm": "ORV",
    "bedrock_map_route": "BMR",
}


def descriptor_sha256(descriptor_json: str) -> str:
    value = json.loads(descriptor_json)
    if not isinstance(value, Mapping):
        raise ValueError("scene descriptor must be a JSON object")
    return content_sha256(dict(value))


@dataclasses.dataclass(frozen=True)
class SourceObservationArtifact:
    source_stage: str
    scene_uid: str
    descriptor_sha256: str
    observations: tuple[LabelObservation, ...]
    provider_exchanges: tuple[ProviderExchange, ...] = ()
    schema_version: str = SOURCE_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported source artifact schema: {self.schema_version}"
            )
        if self.source_stage not in SOURCE_STAGES:
            raise ValueError(f"unknown source stage: {self.source_stage}")
        if not self.scene_uid:
            raise ValueError("source artifact scene_uid is required")
        if (
            len(self.descriptor_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.descriptor_sha256
            )
        ):
            raise ValueError("source artifact descriptor digest is invalid")
        ordered = tuple(
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
        if any(
            observation.scene_uid != self.scene_uid
            for observation in ordered
        ):
            raise ValueError("source observation belongs to another scene")
        observation_ids = [
            observation.observation_uid for observation in ordered
        ]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("source artifact has duplicate observations")
        expected_backend = PROVIDER_BACKEND_BY_SOURCE_STAGE.get(
            self.source_stage
        )
        if self.provider_exchanges and expected_backend is None:
            raise ValueError(
                "provider exchange is invalid for deterministic source stage"
            )
        exchanges = tuple(
            sorted(
                self.provider_exchanges,
                key=lambda item: (
                    item.backend,
                    item.request_sha256,
                    item.attempt,
                    item.response_sha256 or "",
                    item.status,
                ),
            )
        )
        if any(
            exchange.backend != expected_backend for exchange in exchanges
        ):
            raise ValueError("provider exchange backend differs from source stage")
        exchange_identities = [
            (
                exchange.backend,
                exchange.request_sha256,
                exchange.attempt,
                exchange.response_sha256,
                exchange.status,
            )
            for exchange in exchanges
        ]
        if len(exchange_identities) != len(set(exchange_identities)):
            raise ValueError("source artifact has duplicate provider exchanges")
        object.__setattr__(self, "observations", ordered)
        object.__setattr__(self, "provider_exchanges", exchanges)

    @classmethod
    def create(
        cls,
        *,
        source_stage: str,
        descriptor_json: str,
        scene_uid: str,
        observations: Iterable[LabelObservation],
        provider_exchanges: Iterable[ProviderExchange] = (),
    ) -> "SourceObservationArtifact":
        return cls(
            source_stage=source_stage,
            scene_uid=scene_uid,
            descriptor_sha256=descriptor_sha256(descriptor_json),
            observations=tuple(observations),
            provider_exchanges=tuple(provider_exchanges),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_stage": self.source_stage,
            "scene_uid": self.scene_uid,
            "descriptor_sha256": self.descriptor_sha256,
            "observations": [
                observation.to_dict()
                for observation in self.observations
            ],
            "provider_exchanges": [
                exchange.to_dict() for exchange in self.provider_exchanges
            ],
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def semantic_sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        expected_descriptor_json: str | None = None,
        expected_source_stage: str | None = None,
    ) -> "SourceObservationArtifact":
        value = json.loads(payload)
        if not isinstance(value, Mapping):
            raise ValueError("source artifact must be a JSON object")
        artifact = cls(
            schema_version=str(value["schema_version"]),
            source_stage=str(value["source_stage"]),
            scene_uid=str(value["scene_uid"]),
            descriptor_sha256=str(value["descriptor_sha256"]),
            observations=tuple(
                LabelObservation(**dict(observation))
                for observation in value.get("observations", [])
            ),
            provider_exchanges=tuple(
                ProviderExchange(**dict(exchange))
                for exchange in value.get("provider_exchanges", [])
            ),
        )
        if (
            expected_descriptor_json is not None
            and artifact.descriptor_sha256
            != descriptor_sha256(expected_descriptor_json)
        ):
            raise ValueError("source artifact descriptor digest differs")
        if (
            expected_source_stage is not None
            and artifact.source_stage != expected_source_stage
        ):
            raise ValueError("source artifact stage differs")
        if artifact.to_bytes() != payload:
            raise ValueError("source artifact is not canonical")
        return artifact
