"""Strict loader for the scene-level ODD ontology registry."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


_REGISTRY_PATH = Path(__file__).with_name("ontology_registry.json")
_REGISTRY_SCHEMA_VERSION = "odd_ontology_registry_v2"
_NAMESPACES = frozenset({"odd", "event", "perception"})
_CARDINALITIES = frozenset({"single", "multi"})
_SUBJECT_TYPES = frozenset(
    {"scene", "camera", "actor", "actor_camera", "map_element"}
)
_QUALITY_TIERS = frozenset({"certified", "experimental", "disabled"})
_VALUE_FIELDS = frozenset({"value", "display_name", "description"})
_ACQUISITION_FIELDS = frozenset(
    {
        "primary_backends",
        "fallback_backends",
        "required_evidence",
        "routing_policy",
        "fallback_policy",
    }
)
_LABEL_FIELDS = frozenset(
    {
        "key",
        "display_name",
        "description",
        "cardinality",
        "allowed_values",
        "neutral_value",
        "allowed_statuses",
        "allowed_sources",
        "authoritative_sources",
        "fallback_sources",
        "subject_types",
        "temporal_scope",
        "required_capabilities",
        "acquisition",
        "temporal_resolution",
        "spatial_context",
        "aggregation_rule",
        "conflict_rule",
        "minimum_duration_ns",
        "hysteresis",
        "quality_tier_by_source",
        "none_semantics",
        "introduced_in",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "ontology_version",
        "status_definitions",
        "source_definitions",
        "backend_definitions",
        "capability_definitions",
        "labels",
        "excluded_labels",
    }
)


@dataclasses.dataclass(frozen=True)
class ValueDefinition:
    value: str
    display_name: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class AcquisitionPolicy:
    primary_backends: tuple[str, ...]
    fallback_backends: tuple[str, ...]
    required_evidence: str
    routing_policy: str
    fallback_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_backends": list(self.primary_backends),
            "fallback_backends": list(self.fallback_backends),
            "required_evidence": self.required_evidence,
            "routing_policy": self.routing_policy,
            "fallback_policy": self.fallback_policy,
        }


@dataclasses.dataclass(frozen=True)
class LabelDefinition:
    key: str
    display_name: str
    description: str
    cardinality: str
    value_definitions: tuple[ValueDefinition, ...]
    neutral_value: str | None
    allowed_statuses: tuple[str, ...]
    primary_sources: tuple[str, ...]
    authoritative_sources: tuple[str, ...]
    fallback_sources: tuple[str, ...]
    subject_types: tuple[str, ...]
    temporal_scope: str
    required_capabilities: tuple[tuple[str, ...], ...]
    acquisition: AcquisitionPolicy
    temporal_resolution: str
    spatial_context: str
    aggregation_rule: str
    conflict_rule: str
    minimum_duration_ns: int
    hysteresis: str
    quality_tier_by_source: tuple[tuple[str, str], ...]
    none_semantics: str | None
    introduced_in: str

    @property
    def namespace(self) -> str:
        return self.key.split(".", 1)[0]

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(item.value for item in self.value_definitions)

    @property
    def backends(self) -> tuple[str, ...]:
        return self.acquisition.primary_backends + tuple(
            backend
            for backend in self.acquisition.fallback_backends
            if backend not in self.acquisition.primary_backends
        )

    @property
    def subject(self) -> str:
        if self.subject_types == ("camera", "actor", "actor_camera"):
            return "camera_or_actor"
        return self.subject_types[0]

    @property
    def quality_tier(self) -> str:
        tiers = {tier for _, tier in self.quality_tier_by_source}
        if "disabled" in tiers:
            return "disabled"
        if tiers == {"certified"}:
            return "certified"
        return "experimental"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "namespace": self.namespace,
            "display_name": self.display_name,
            "description": self.description,
            "cardinality": self.cardinality,
            "values": [item.to_dict() for item in self.value_definitions],
            "neutral_value": self.neutral_value,
            "allowed_statuses": list(self.allowed_statuses),
            # Keep these compatibility fields while exposing the complete policy.
            "primary_sources": list(self.primary_sources),
            "backends": list(self.backends),
            "subject": self.subject,
            "temporal_scope": self.temporal_scope,
            "quality_tier": self.quality_tier,
            "none_semantics": self.none_semantics,
            "allowed_sources": list(self.primary_sources),
            "authoritative_sources": list(self.authoritative_sources),
            "fallback_sources": list(self.fallback_sources),
            "subject_types": list(self.subject_types),
            "required_capabilities": {
                "any_of": [list(group) for group in self.required_capabilities]
            },
            "acquisition": self.acquisition.to_dict(),
            "temporal_resolution": self.temporal_resolution,
            "spatial_context": self.spatial_context,
            "aggregation_rule": self.aggregation_rule,
            "conflict_rule": self.conflict_rule,
            "minimum_duration_ns": self.minimum_duration_ns,
            "hysteresis": self.hysteresis,
            "quality_tier_by_source": dict(self.quality_tier_by_source),
            "introduced_in": self.introduced_in,
        }


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    context: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{context} fields differ: missing={missing}, unknown={unknown}"
        )


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_string_list(
    value: Any,
    context: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{context} must be a non-empty string array")
    items = tuple(_require_text(item, context) for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{context} contains duplicates")
    return items


def _parse_named_definitions(
    raw: Any,
    *,
    name_field: str,
    context: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{context} must be a non-empty array")
    definitions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        parsed = _require_object(item, f"{context}[{index}]")
        name = _require_text(
            parsed.get(name_field), f"{context}[{index}].{name_field}"
        )
        if name in seen:
            raise ValueError(f"duplicate {context} name: {name}")
        seen.add(name)
        definitions.append(parsed)
    return tuple(definitions)


def _parse_value_definitions(
    raw: Any,
    *,
    key: str,
) -> tuple[ValueDefinition, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{key}.allowed_values must be a non-empty array")
    values: list[ValueDefinition] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        parsed = _require_object(item, f"{key}.allowed_values[{index}]")
        _require_exact_fields(
            parsed,
            _VALUE_FIELDS,
            f"{key}.allowed_values[{index}]",
        )
        value = _require_text(
            parsed["value"], f"{key}.allowed_values[{index}].value"
        )
        if value in seen:
            raise ValueError(f"duplicate value for {key}: {value}")
        seen.add(value)
        values.append(
            ValueDefinition(
                value=value,
                display_name=_require_text(
                    parsed["display_name"],
                    f"{key}.allowed_values[{index}].display_name",
                ),
                description=_require_text(
                    parsed["description"],
                    f"{key}.allowed_values[{index}].description",
                ),
            )
        )
    return tuple(values)


def _parse_capability_requirements(
    raw: Any,
    *,
    key: str,
    known_capabilities: frozenset[str],
) -> tuple[tuple[str, ...], ...]:
    parsed = _require_object(raw, f"{key}.required_capabilities")
    _require_exact_fields(
        parsed, frozenset({"any_of"}), f"{key}.required_capabilities"
    )
    alternatives = parsed["any_of"]
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError(f"{key}.required_capabilities.any_of must be non-empty")
    result: list[tuple[str, ...]] = []
    for index, group in enumerate(alternatives):
        capabilities = _require_string_list(
            group, f"{key}.required_capabilities.any_of[{index}]"
        )
        unknown = set(capabilities) - known_capabilities
        if unknown:
            raise ValueError(
                f"{key} references unknown capabilities: {sorted(unknown)}"
            )
        result.append(tuple(sorted(capabilities)))
    normalized = tuple(sorted(set(result)))
    if len(normalized) != len(result):
        raise ValueError(f"{key} contains duplicate capability alternatives")
    return normalized


def _parse_acquisition(
    raw: Any,
    *,
    key: str,
    known_backends: frozenset[str],
) -> AcquisitionPolicy:
    parsed = _require_object(raw, f"{key}.acquisition")
    _require_exact_fields(parsed, _ACQUISITION_FIELDS, f"{key}.acquisition")
    primary = _require_string_list(
        parsed["primary_backends"], f"{key}.acquisition.primary_backends"
    )
    fallback = _require_string_list(
        parsed["fallback_backends"],
        f"{key}.acquisition.fallback_backends",
        allow_empty=True,
    )
    unknown = (set(primary) | set(fallback)) - known_backends
    if unknown:
        raise ValueError(f"{key} references unknown backends: {sorted(unknown)}")
    if set(primary) & set(fallback):
        raise ValueError(f"{key} repeats a primary backend as fallback")
    return AcquisitionPolicy(
        primary_backends=primary,
        fallback_backends=fallback,
        required_evidence=_require_text(
            parsed["required_evidence"],
            f"{key}.acquisition.required_evidence",
        ),
        routing_policy=_require_text(
            parsed["routing_policy"], f"{key}.acquisition.routing_policy"
        ),
        fallback_policy=_require_text(
            parsed["fallback_policy"], f"{key}.acquisition.fallback_policy"
        ),
    )


def _parse_label(
    raw: Any,
    *,
    index: int,
    known_statuses: frozenset[str],
    known_sources: frozenset[str],
    known_backends: frozenset[str],
    backend_sources: Mapping[str, str],
    known_capabilities: frozenset[str],
) -> LabelDefinition:
    parsed = _require_object(raw, f"labels[{index}]")
    _require_exact_fields(parsed, _LABEL_FIELDS, f"labels[{index}]")
    key = _require_text(parsed["key"], f"labels[{index}].key")
    namespace, separator, local_name = key.partition(".")
    if (
        not separator
        or not local_name
        or namespace not in _NAMESPACES
        or any(not part for part in local_name.split("."))
    ):
        raise ValueError(f"invalid ontology key: {key}")
    cardinality = _require_text(parsed["cardinality"], f"{key}.cardinality")
    if cardinality not in _CARDINALITIES:
        raise ValueError(f"invalid cardinality for {key}: {cardinality}")

    values = _parse_value_definitions(parsed["allowed_values"], key=key)
    value_names = {item.value for item in values}
    neutral_value = parsed["neutral_value"]
    if neutral_value is not None:
        neutral_value = _require_text(neutral_value, f"{key}.neutral_value")
        if neutral_value not in value_names:
            raise ValueError(f"{key}.neutral_value is not an allowed value")

    allowed_statuses = _require_string_list(
        parsed["allowed_statuses"], f"{key}.allowed_statuses"
    )
    if set(allowed_statuses) != known_statuses:
        raise ValueError(f"{key} must explicitly allow every canonical status")

    allowed_sources = _require_string_list(
        parsed["allowed_sources"], f"{key}.allowed_sources"
    )
    unknown_sources = set(allowed_sources) - known_sources
    if unknown_sources:
        raise ValueError(
            f"{key} references unknown sources: {sorted(unknown_sources)}"
        )
    authoritative_sources = _require_string_list(
        parsed["authoritative_sources"], f"{key}.authoritative_sources"
    )
    fallback_sources = _require_string_list(
        parsed["fallback_sources"],
        f"{key}.fallback_sources",
        allow_empty=True,
    )
    for field_name, sources in (
        ("authoritative_sources", authoritative_sources),
        ("fallback_sources", fallback_sources),
    ):
        unknown = set(sources) - set(allowed_sources)
        if unknown:
            raise ValueError(
                f"{key}.{field_name} is not allowed: {sorted(unknown)}"
            )

    subject_types = _require_string_list(
        parsed["subject_types"], f"{key}.subject_types"
    )
    invalid_subjects = set(subject_types) - _SUBJECT_TYPES
    if invalid_subjects:
        raise ValueError(
            f"{key} has invalid subject types: {sorted(invalid_subjects)}"
        )
    acquisition = _parse_acquisition(
        parsed["acquisition"],
        key=key,
        known_backends=known_backends,
    )
    for backend in (
        acquisition.primary_backends + acquisition.fallback_backends
    ):
        if backend_sources[backend] not in allowed_sources:
            raise ValueError(
                f"{key} backend {backend} emits disallowed source "
                f"{backend_sources[backend]}"
            )

    quality = _require_object(
        parsed["quality_tier_by_source"], f"{key}.quality_tier_by_source"
    )
    if set(quality) != set(allowed_sources):
        raise ValueError(
            f"{key}.quality_tier_by_source must cover allowed_sources exactly"
        )
    quality_pairs: list[tuple[str, str]] = []
    for source in sorted(quality):
        tier = _require_text(
            quality[source], f"{key}.quality_tier_by_source.{source}"
        )
        if tier not in _QUALITY_TIERS:
            raise ValueError(f"{key} has invalid quality tier: {tier}")
        quality_pairs.append((source, tier))

    none_semantics = parsed["none_semantics"]
    if none_semantics is not None:
        none_semantics = _require_text(none_semantics, f"{key}.none_semantics")
    semantic_neutrals = value_names & {"none", "normal"}
    if cardinality == "multi" and semantic_neutrals:
        if len(semantic_neutrals) != 1:
            raise ValueError(f"{key} has multiple exclusive neutral values")
        expected_neutral = next(iter(semantic_neutrals))
        if neutral_value != expected_neutral or none_semantics is None:
            raise ValueError(
                f"{key} must declare exclusive neutral semantics for "
                f"{expected_neutral}"
            )

    minimum_duration_ns = parsed["minimum_duration_ns"]
    if (
        isinstance(minimum_duration_ns, bool)
        or not isinstance(minimum_duration_ns, int)
        or minimum_duration_ns < 0
    ):
        raise ValueError(f"{key}.minimum_duration_ns must be a non-negative int")

    return LabelDefinition(
        key=key,
        display_name=_require_text(parsed["display_name"], f"{key}.display_name"),
        description=_require_text(parsed["description"], f"{key}.description"),
        cardinality=cardinality,
        value_definitions=values,
        neutral_value=neutral_value,
        allowed_statuses=allowed_statuses,
        primary_sources=allowed_sources,
        authoritative_sources=authoritative_sources,
        fallback_sources=fallback_sources,
        subject_types=subject_types,
        temporal_scope=_require_text(
            parsed["temporal_scope"], f"{key}.temporal_scope"
        ),
        required_capabilities=_parse_capability_requirements(
            parsed["required_capabilities"],
            key=key,
            known_capabilities=known_capabilities,
        ),
        acquisition=acquisition,
        temporal_resolution=_require_text(
            parsed["temporal_resolution"], f"{key}.temporal_resolution"
        ),
        spatial_context=_require_text(
            parsed["spatial_context"], f"{key}.spatial_context"
        ),
        aggregation_rule=_require_text(
            parsed["aggregation_rule"], f"{key}.aggregation_rule"
        ),
        conflict_rule=_require_text(
            parsed["conflict_rule"], f"{key}.conflict_rule"
        ),
        minimum_duration_ns=minimum_duration_ns,
        hysteresis=_require_text(parsed["hysteresis"], f"{key}.hysteresis"),
        quality_tier_by_source=tuple(quality_pairs),
        none_semantics=none_semantics,
        introduced_in=_require_text(
            parsed["introduced_in"], f"{key}.introduced_in"
        ),
    )


def _parse_registry(
    raw: Any,
) -> tuple[
    str,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[str, ...],
    tuple[LabelDefinition, ...],
    tuple[dict[str, Any], ...],
]:
    registry = _require_object(raw, "ontology registry")
    _require_exact_fields(registry, _TOP_LEVEL_FIELDS, "ontology registry")
    if registry["schema_version"] != _REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            "unsupported ontology registry schema: "
            f"{registry['schema_version']!r}"
        )
    ontology_version = _require_text(
        registry["ontology_version"], "ontology_version"
    )

    status_definitions = _parse_named_definitions(
        registry["status_definitions"],
        name_field="status",
        context="status_definitions",
    )
    source_definitions = _parse_named_definitions(
        registry["source_definitions"],
        name_field="source",
        context="source_definitions",
    )
    backend_definitions = _parse_named_definitions(
        registry["backend_definitions"],
        name_field="backend",
        context="backend_definitions",
    )
    for context, definitions, expected_fields in (
        (
            "status_definitions",
            status_definitions,
            frozenset({"status", "description"}),
        ),
        (
            "source_definitions",
            source_definitions,
            frozenset({"source", "description"}),
        ),
        (
            "backend_definitions",
            backend_definitions,
            frozenset(
                {
                    "backend",
                    "name",
                    "canonical_source",
                    "permitted_inputs",
                }
            ),
        ),
    ):
        for index, definition in enumerate(definitions):
            _require_exact_fields(
                definition, expected_fields, f"{context}[{index}]"
            )

    statuses = frozenset(item["status"] for item in status_definitions)
    expected_statuses = frozenset(
        {"valid", "unavailable", "not_observable", "ambiguous"}
    )
    if statuses != expected_statuses:
        raise ValueError("status definitions differ from the canonical contract")
    sources = frozenset(item["source"] for item in source_definitions)

    backend_sources: dict[str, str] = {}
    for definition in backend_definitions:
        source = _require_text(
            definition["canonical_source"],
            f"backend {definition['backend']} canonical_source",
        )
        if source not in sources:
            raise ValueError(
                f"backend {definition['backend']} has unknown source {source}"
            )
        _require_string_list(
            definition["permitted_inputs"],
            f"backend {definition['backend']} permitted_inputs",
        )
        backend_sources[definition["backend"]] = source

    capabilities = _require_string_list(
        registry["capability_definitions"], "capability_definitions"
    )
    labels_raw = registry["labels"]
    if not isinstance(labels_raw, list) or not labels_raw:
        raise ValueError("labels must be a non-empty array")
    definitions = tuple(
        _parse_label(
            item,
            index=index,
            known_statuses=statuses,
            known_sources=sources,
            known_backends=frozenset(backend_sources),
            backend_sources=backend_sources,
            known_capabilities=frozenset(capabilities),
        )
        for index, item in enumerate(labels_raw)
    )
    if len({definition.key for definition in definitions}) != len(definitions):
        raise ValueError("ontology registry contains duplicate label keys")

    excluded = _parse_named_definitions(
        registry["excluded_labels"],
        name_field="key",
        context="excluded_labels",
    )
    for index, definition in enumerate(excluded):
        _require_exact_fields(
            definition,
            frozenset({"key", "reason"}),
            f"excluded_labels[{index}]",
        )
        _require_text(definition["reason"], f"excluded_labels[{index}].reason")

    return (
        ontology_version,
        status_definitions,
        source_definitions,
        backend_definitions,
        tuple(sorted(capabilities)),
        tuple(sorted(definitions, key=lambda item: item.key)),
        tuple(sorted(excluded, key=lambda item: item["key"])),
    )


def load_ontology_registry(
    path: Path = _REGISTRY_PATH,
) -> tuple[
    str,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[str, ...],
    tuple[LabelDefinition, ...],
    tuple[dict[str, Any], ...],
]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load ontology registry {path}: {error}") from error
    return _parse_registry(raw)


(
    ONTOLOGY_VERSION,
    _STATUS_DEFINITIONS,
    _SOURCE_DEFINITIONS,
    _BACKEND_DEFINITIONS,
    _CAPABILITY_DEFINITIONS,
    _DEFINITIONS,
    _EXCLUDED_LABELS,
) = load_ontology_registry()

LABEL_STATUSES = tuple(item["status"] for item in _STATUS_DEFINITIONS)
LABEL_SOURCES = tuple(item["source"] for item in _SOURCE_DEFINITIONS)
ONTOLOGY = {definition.key: definition for definition in _DEFINITIONS}


def ontology_document() -> dict[str, Any]:
    body = {
        "schema_version": _REGISTRY_SCHEMA_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "statuses": list(LABEL_STATUSES),
        "sources": list(LABEL_SOURCES),
        "status_definitions": [dict(item) for item in _STATUS_DEFINITIONS],
        "source_definitions": [dict(item) for item in _SOURCE_DEFINITIONS],
        "backend_definitions": [dict(item) for item in _BACKEND_DEFINITIONS],
        "capability_definitions": list(_CAPABILITY_DEFINITIONS),
        "labels": [definition.to_dict() for definition in _DEFINITIONS],
        "excluded_labels": [dict(item) for item in _EXCLUDED_LABELS],
    }
    body["ontology_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return body


def ontology_sha256() -> str:
    return str(ontology_document()["ontology_sha256"])


def validate_values(key: str, values: Iterable[str]) -> tuple[str, ...]:
    definition = ONTOLOGY.get(key)
    if definition is None:
        raise ValueError(f"unknown ontology key: {key}")
    normalized = tuple(sorted(set(values)))
    if not normalized:
        raise ValueError(f"valid observation requires a value: {key}")
    unknown = set(normalized) - set(definition.values)
    if unknown:
        raise ValueError(f"unknown values for {key}: {sorted(unknown)}")
    if definition.cardinality == "single" and len(normalized) != 1:
        raise ValueError(f"single-select label has {len(normalized)} values: {key}")
    if definition.neutral_value in normalized and len(normalized) != 1:
        raise ValueError(
            f"{definition.neutral_value} cannot coexist with another value: {key}"
        )
    return normalized
