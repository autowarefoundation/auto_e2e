"""Structural validation and auditable quality documents for ODD LabelSets."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


QUALITY_SCHEMA_VERSION = "odd_quality_v1"
AUDIT_SAMPLE_TARGET = 50
CONFIDENCE_BANDS = (
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
)


def calibration_bundle_document() -> dict[str, Any]:
    return {
        "schema_version": "odd_calibration_bundle_v1",
        "mode": "identity_pending_human_audit",
        "partition_by": [
            "ontology_key",
            "source",
            "labeler_version",
        ],
        "raw_confidence_retained": True,
        "resolved_confidence_transform": "identity",
        "certification_status": "experimental",
        "certified": False,
        "audit_sample_target": AUDIT_SAMPLE_TARGET,
        "confidence_bands": [list(band) for band in CONFIDENCE_BANDS],
        "future_allowed_methods": [
            "isotonic_regression",
            "temperature_scaling",
        ],
    }


def calibration_bundle_sha256() -> str:
    return hashlib.sha256(
        _canonical_bytes(calibration_bundle_document())
    ).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _check_interval(
    value: Mapping[str, Any],
    *,
    scene_start_ns: int,
    scene_end_ns: int,
    name: str,
) -> None:
    start_ns = int(value["start_timestamp_ns"])
    end_ns = int(value["end_timestamp_ns"])
    if (
        start_ns < scene_start_ns
        or end_ns > scene_end_ns
        or end_ns <= start_ns
    ):
        raise ValueError(f"{name} interval exceeds its scene")


def _check_label(
    value: Mapping[str, Any],
    *,
    key_field: str,
    definitions: Mapping[str, Mapping[str, Any]],
    statuses: set[str],
    sources: set[str],
    name: str,
) -> None:
    key = str(value[key_field])
    definition = definitions.get(key)
    if definition is None:
        raise ValueError(f"{name} has unknown ontology key: {key}")
    status = str(value["status"])
    if status not in statuses:
        raise ValueError(f"{name} has invalid status: {status}")
    source = str(value["source"])
    if source not in sources:
        raise ValueError(f"{name} has invalid source: {source}")
    confidence = float(value["confidence"])
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{name} confidence must be finite and in [0,1]")
    values = [str(item) for item in value.get("values", [])]
    if len(values) != len(set(values)):
        raise ValueError(f"{name} has duplicate values")
    if status == "valid":
        cardinality = str(definition["cardinality"])
        if not values or (cardinality == "single" and len(values) != 1):
            raise ValueError(f"{name} values differ from cardinality")
        allowed = {
            str(candidate["value"]) for candidate in definition["values"]
        }
        if not set(values).issubset(allowed):
            raise ValueError(f"{name} has values outside the ontology")
        if len(values) > 1 and {"none", "normal"} & set(values):
            raise ValueError(f"{name} neutral value is not mutually exclusive")
    elif values:
        raise ValueError(f"{name} non-valid status carries resolved values")


def validate_labelset_records(
    records: Sequence[Mapping[str, Any]],
    ontology: Mapping[str, Any],
) -> dict[str, Any]:
    definitions = {
        str(definition["key"]): definition
        for definition in ontology["labels"]
    }
    statuses = {str(status) for status in ontology["statuses"]}
    sources = {str(source) for source in ontology["sources"]}
    scene_ids: set[str] = set()
    evidence_ids: set[str] = set()
    observation_ids: set[str] = set()
    event_ids: set[str] = set()
    evidence_count = 0
    observation_count = 0
    event_count = 0

    for record in records:
        scene_uid = str(record["scene_uid"])
        if not scene_uid or scene_uid in scene_ids:
            raise ValueError(f"duplicate or empty scene_uid: {scene_uid}")
        scene_ids.add(scene_uid)
        scene_start_ns = int(record["start_timestamp_ns"])
        scene_end_ns = int(record["end_timestamp_ns"])
        distance_m = float(record["distance_m"])
        if (
            scene_start_ns < 0
            or scene_end_ns <= scene_start_ns
            or not math.isfinite(distance_m)
            or distance_m < 0.0
        ):
            raise ValueError("scene temporal or distance contract is invalid")

        scene_evidence: set[str] = set()
        for evidence in record.get("evidence", []):
            evidence_uid = str(evidence["evidence_uid"])
            if not evidence_uid or evidence_uid in evidence_ids:
                raise ValueError(f"duplicate evidence_uid: {evidence_uid}")
            evidence_ids.add(evidence_uid)
            scene_evidence.add(evidence_uid)
            evidence_count += 1
            _check_label(
                evidence,
                key_field="label_key",
                definitions=definitions,
                statuses=statuses,
                sources=sources,
                name="evidence",
            )
            scope = evidence["scope"]
            if str(scope["scene_uid"]) != scene_uid:
                raise ValueError("evidence scope belongs to another scene")
            _check_interval(
                scope,
                scene_start_ns=scene_start_ns,
                scene_end_ns=scene_end_ns,
                name="evidence",
            )

        scene_observations: set[str] = set()
        observations_by_uid: dict[str, Mapping[str, Any]] = {}
        for observation in record.get("observations", []):
            observation_uid = str(observation["observation_uid"])
            if not observation_uid or observation_uid in observation_ids:
                raise ValueError(
                    f"duplicate observation_uid: {observation_uid}"
                )
            observation_ids.add(observation_uid)
            scene_observations.add(observation_uid)
            observations_by_uid[observation_uid] = observation
            observation_count += 1
            if str(observation["scene_uid"]) != scene_uid:
                raise ValueError("observation belongs to another scene")
            _check_label(
                observation,
                key_field="key",
                definitions=definitions,
                statuses=statuses,
                sources=sources,
                name="observation",
            )
            _check_interval(
                observation,
                scene_start_ns=scene_start_ns,
                scene_end_ns=scene_end_ns,
                name="observation",
            )
            supporting = {
                str(item) for item in observation.get("evidence_uids", [])
            }
            conflicting = {
                str(item)
                for item in observation.get(
                    "conflicting_evidence_uids",
                    [],
                )
            }
            if supporting & conflicting:
                raise ValueError(
                    "observation evidence is both supporting and conflicting"
                )
            if not (supporting | conflicting).issubset(scene_evidence):
                raise ValueError("observation references unknown evidence")

        for event in record.get("events", []):
            event_uid = str(event["event_uid"])
            if not event_uid or event_uid in event_ids:
                raise ValueError(f"duplicate event_uid: {event_uid}")
            event_ids.add(event_uid)
            event_count += 1
            if str(event["scene_uid"]) != scene_uid:
                raise ValueError("event belongs to another scene")
            _check_interval(
                event,
                scene_start_ns=scene_start_ns,
                scene_end_ns=scene_end_ns,
                name="event",
            )
            primary_key = str(event["primary_event_key"])
            definition = definitions.get(primary_key)
            if definition is None or definition["namespace"] != "event":
                raise ValueError("event primary key is not an event label")
            event_status = str(event["status"])
            event_confidence = float(event["confidence"])
            if event_status not in statuses:
                raise ValueError("event has invalid status")
            if (
                not math.isfinite(event_confidence)
                or not 0.0 <= event_confidence <= 1.0
            ):
                raise ValueError(
                    "event confidence must be finite and in [0,1]"
                )
            event_observations = {
                str(item) for item in event.get("observation_uids", [])
            }
            event_evidence = {
                str(item)
                for item in event.get("supporting_evidence_uids", [])
            }
            if not event_observations.issubset(scene_observations):
                raise ValueError("event references unknown observation")
            if not event_evidence.issubset(scene_evidence):
                raise ValueError("event references unknown evidence")
            previous_end = int(event["start_timestamp_ns"])
            phase_order = {"onset": 0, "active": 1, "resolution": 2}
            previous_order = -1
            for phase in event.get("phases", []):
                phase_name = str(phase["phase"])
                order = phase_order.get(phase_name)
                if order is None or order <= previous_order:
                    raise ValueError("event phases are not ordered")
                phase_start = int(phase["start_timestamp_ns"])
                phase_end = int(phase["end_timestamp_ns"])
                if (
                    phase_start < previous_end
                    or phase_end <= phase_start
                    or phase_end > int(event["end_timestamp_ns"])
                ):
                    raise ValueError("event phase interval is invalid")
                previous_order = order
                previous_end = phase_end

    return {
        "status": "passed",
        "checks": [
            "dataset_coordinate",
            "unique_identities",
            "bounded_intervals",
            "ontology_conformance",
            "status_value_invariants",
            "finite_confidence",
            "same_scene_references",
            "event_phase_order",
        ],
        "scene_count": len(scene_ids),
        "evidence_count": evidence_count,
        "observation_count": observation_count,
        "event_count": event_count,
    }


def _support_state(key_row: Mapping[str, Any]) -> str:
    successful_count = int(key_row["successful_count"])
    attempted_count = int(key_row["attempted_count"])
    status_counts = key_row["status_scene_counts"]
    unavailable_count = int(status_counts.get("unavailable", 0))
    ambiguous_count = int(status_counts.get("ambiguous", 0))
    not_observable_count = int(status_counts.get("not_observable", 0))
    if successful_count > 0 or ambiguous_count > 0 or not_observable_count > 0:
        return (
            "supported_certified"
            if key_row.get("quality_tier") == "certified"
            else "supported_experimental"
        )
    if attempted_count > 0 or unavailable_count > 0:
        return "unsupported_missing_source"
    return "disabled_pending_audit"


def _coverage_document(
    statistics: Mapping[str, Any],
    *,
    labelset_id: str,
    structural_validation: Mapping[str, Any],
) -> dict[str, Any]:
    keys = []
    for row in statistics["keys"]:
        keys.append(
            {
                "key": row["key"],
                "namespace": row["namespace"],
                "eligible_duration_ns": row["eligible_duration_ns"],
                "valid_duration_ns": row["valid_duration_ns"],
                "unavailable_duration_ns": row[
                    "status_duration_ns"
                ].get("unavailable", 0),
                "not_observable_duration_ns": row[
                    "status_duration_ns"
                ].get("not_observable", 0),
                "ambiguous_duration_ns": row[
                    "status_duration_ns"
                ].get("ambiguous", 0),
                "eligible_distance_m": row["eligible_distance_m"],
                "valid_distance_m": row["valid_distance_m"],
                "attempted_count": row["attempted_count"],
                "successful_count": row["successful_count"],
                "conflict_count": row["conflict_count"],
                "quality_tier": row["quality_tier"],
                "support_state": _support_state(row),
            }
        )
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "document_type": "coverage",
        "labelset_id": labelset_id,
        "scene_count": statistics["scene_count"],
        "structural_validation": dict(structural_validation),
        "publication_gate": {
            "structural_status": "passed",
            "semantic_coverage_policy": "report_without_imputation",
            "human_audit_required_for_certification": True,
        },
        "keys": keys,
    }


def _confidence_band(confidence: float) -> str:
    for lower, upper in CONFIDENCE_BANDS:
        if lower <= confidence < upper or (
            upper == 1.0 and confidence == 1.0
        ):
            return f"{lower:.1f}-{upper:.1f}"
    raise ValueError("confidence is outside audit bands")


def _ranked_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    labelset_id: str,
    stratum_id: str,
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: hashlib.sha256(
            (
                f"{labelset_id}\0{stratum_id}\0"
                f"{candidate['evidence_uid']}"
            ).encode("utf-8")
        ).hexdigest(),
    )[:AUDIT_SAMPLE_TARGET]


def _audit_document(
    records: Sequence[Mapping[str, Any]],
    *,
    labelset_id: str,
) -> dict[str, Any]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        scene_uid = str(record["scene_uid"])
        for evidence in record.get("evidence", []):
            confidence = float(evidence["confidence"])
            base = {
                "scene_uid": scene_uid,
                "evidence_uid": str(evidence["evidence_uid"]),
                "label_key": str(evidence["label_key"]),
                "source": str(evidence["source"]),
                "status": str(evidence["status"]),
                "values": list(evidence.get("values", [])),
                "confidence": confidence,
                "start_timestamp_ns": int(
                    evidence["scope"]["start_timestamp_ns"]
                ),
                "end_timestamp_ns": int(
                    evidence["scope"]["end_timestamp_ns"]
                ),
            }
            identities = {
                (
                    f"key={evidence['label_key']}|"
                    f"source={evidence['source']}"
                ),
                (
                    f"key={evidence['label_key']}|"
                    f"status={evidence['status']}"
                ),
                (
                    f"key={evidence['label_key']}|"
                    f"confidence={_confidence_band(confidence)}"
                ),
            }
            for value in evidence.get("values", []):
                identities.add(
                    f"key={evidence['label_key']}|value={value}"
                )
            if evidence.get("candidate_values"):
                identities.add(
                    f"key={evidence['label_key']}|candidate_ambiguity"
                )
            for identity in identities:
                strata[identity].append(base)
    rows = []
    for stratum_id in sorted(strata):
        candidates = strata[stratum_id]
        selected = _ranked_candidates(
            candidates,
            labelset_id=labelset_id,
            stratum_id=stratum_id,
        )
        rows.append(
            {
                "stratum_id": stratum_id,
                "population_count": len(candidates),
                "target_count": min(
                    AUDIT_SAMPLE_TARGET,
                    len(candidates),
                ),
                "selected": selected,
            }
        )
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "document_type": "audit_manifest",
        "labelset_id": labelset_id,
        "status": "pending_human_audit",
        "selection_policy": {
            "method": "sha256_rank_within_stratum",
            "target_per_stratum": AUDIT_SAMPLE_TARGET,
            "reviewer_blinding": "source_evidence_hidden_initially",
        },
        "strata": rows,
    }


def _calibration_document(
    records: Sequence[Mapping[str, Any]],
    *,
    labelset_id: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        for evidence in record.get("evidence", []):
            grouped[
                (
                    str(evidence["label_key"]),
                    str(evidence["source"]),
                )
            ].append(float(evidence["confidence"]))
    rows = []
    for (key, source), confidences in sorted(grouped.items()):
        rows.append(
            {
                "key": key,
                "source": source,
                "evidence_count": len(confidences),
                "raw_confidence_mean": sum(confidences) / len(confidences),
                "calibration_status": "pending_human_audit",
                "calibration_method": None,
                "calibration_dataset_sha256": None,
                "expected_calibration_error": None,
                "quality_tier": "experimental",
                "certified": False,
            }
        )
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "document_type": "calibration",
        "labelset_id": labelset_id,
        "confidence_policy": (
            "raw source confidence remains audit metadata until calibrated"
        ),
        "rows": rows,
    }


def build_quality_documents(
    records: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    ontology: Mapping[str, Any],
    *,
    labelset_id: str,
) -> dict[str, dict[str, Any]]:
    structural_validation = validate_labelset_records(records, ontology)
    return {
        "coverage": _coverage_document(
            statistics,
            labelset_id=labelset_id,
            structural_validation=structural_validation,
        ),
        "audit_manifest": _audit_document(
            records,
            labelset_id=labelset_id,
        ),
        "calibration": _calibration_document(
            records,
            labelset_id=labelset_id,
        ),
    }
