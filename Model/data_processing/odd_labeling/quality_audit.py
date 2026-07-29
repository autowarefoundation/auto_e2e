"""Human-audit annotation contract and measured ODD quality metrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


ANNOTATION_SCHEMA_VERSION = "odd_human_audit_annotations_v1"
RESULT_SCHEMA_VERSION = "odd_human_audit_results_v1"
LABEL_STATUSES = {
    "valid",
    "unavailable",
    "not_observable",
    "ambiguous",
}
EVENT_MATCH_STATUSES = {"matched", "spurious", "missed"}
ACTOR_CONTINUITY_STATUSES = {
    "correct",
    "fragmented",
    "switched",
    "not_applicable",
    "unreviewable",
}
AGREEMENT_STATUSES = {
    "unanimous",
    "majority",
    "adjudicated_disagreement",
}
CONFIDENCE_BANDS = (
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
)
MINIMUM_AUDIT_EXAMPLES = 50


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_document(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def _definitions(ontology: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["key"]): dict(item)
        for item in ontology["labels"]
    }


def _validate_status_values(
    value: Mapping[str, Any],
    *,
    allowed_values: set[str],
    cardinality: str,
    name: str,
    include_confidence: bool,
) -> None:
    expected = {"status", "values"}
    if include_confidence:
        expected |= {"confidence", "evidence_uid"}
    _require_exact_keys(value, expected, name)
    status = str(value["status"])
    if status not in LABEL_STATUSES:
        raise ValueError(f"{name} status is invalid")
    values = value["values"]
    if not isinstance(values, list) or not all(
        isinstance(item, str) and item for item in values
    ):
        raise ValueError(f"{name} values must be strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} values must be unique")
    if status == "valid":
        if not values or (cardinality == "single" and len(values) != 1):
            raise ValueError(f"{name} values differ from cardinality")
        if not set(values).issubset(allowed_values):
            raise ValueError(f"{name} values are outside the ontology")
        if len(values) > 1 and {"none", "normal"} & set(values):
            raise ValueError(f"{name} neutral value is not exclusive")
    elif values:
        raise ValueError(f"{name} non-valid status carries values")
    if include_confidence:
        confidence = float(value["confidence"])
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{name} confidence must be in [0,1]")
        evidence_uid = value["evidence_uid"]
        if evidence_uid is not None and (
            not isinstance(evidence_uid, str) or not evidence_uid
        ):
            raise ValueError(f"{name} evidence_uid is invalid")


def _validate_interval(
    start: object,
    end: object,
    name: str,
) -> tuple[int, int]:
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
    ):
        raise ValueError(f"{name} interval is invalid")
    return start, end


def validate_human_audit_annotations(
    document: Mapping[str, Any],
    ontology: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        document,
        {
            "schema_version",
            "labelset_id",
            "annotation_set_id",
            "audit_manifest_sha256",
            "adjudication_status",
            "reviewer_count",
            "label_units",
            "event_units",
            "ontology_questions",
        },
        "annotation document",
    )
    if document["schema_version"] != ANNOTATION_SCHEMA_VERSION:
        raise ValueError("annotation schema version is unsupported")
    if not isinstance(document["labelset_id"], str) or not document["labelset_id"]:
        raise ValueError("annotation labelset_id is required")
    _require_sha256(document["annotation_set_id"], "annotation_set_id")
    _require_sha256(
        document["audit_manifest_sha256"],
        "audit_manifest_sha256",
    )
    if document["adjudication_status"] not in {"draft", "adjudicated"}:
        raise ValueError("annotation adjudication_status is invalid")
    reviewer_count = document["reviewer_count"]
    if (
        not isinstance(reviewer_count, int)
        or isinstance(reviewer_count, bool)
        or reviewer_count < 1
    ):
        raise ValueError("annotation reviewer_count must be positive")
    if document["adjudication_status"] == "adjudicated" and reviewer_count < 2:
        raise ValueError("adjudicated audit requires at least two reviewers")
    if not isinstance(document["label_units"], list):
        raise ValueError("annotation label_units must be a list")
    if not isinstance(document["event_units"], list):
        raise ValueError("annotation event_units must be a list")
    if not isinstance(document["ontology_questions"], list):
        raise ValueError("annotation ontology_questions must be a list")

    definitions = _definitions(ontology)
    sources = {str(source) for source in ontology["sources"]}
    unit_ids: set[str] = set()
    evidence_ids: set[str] = set()
    for unit in document["label_units"]:
        if not isinstance(unit, Mapping):
            raise ValueError("label audit unit must be an object")
        _require_exact_keys(
            unit,
            {
                "unit_uid",
                "scene_uid",
                "label_key",
                "source",
                "start_timestamp_ns",
                "end_timestamp_ns",
                "sampling_weight",
                "agreement",
                "prediction",
                "reference",
            },
            "label audit unit",
        )
        unit_uid = str(unit["unit_uid"])
        if not unit_uid or unit_uid in unit_ids:
            raise ValueError("audit unit_uid must be non-empty and unique")
        unit_ids.add(unit_uid)
        if not isinstance(unit["scene_uid"], str) or not unit["scene_uid"]:
            raise ValueError("label audit scene_uid is required")
        key = str(unit["label_key"])
        definition = definitions.get(key)
        if definition is None:
            raise ValueError(f"label audit key is unknown: {key}")
        if str(unit["source"]) not in sources:
            raise ValueError("label audit source is invalid")
        if str(unit["agreement"]) not in AGREEMENT_STATUSES:
            raise ValueError("label audit agreement is invalid")
        _validate_interval(
            unit["start_timestamp_ns"],
            unit["end_timestamp_ns"],
            "label audit",
        )
        weight = float(unit["sampling_weight"])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("label audit sampling_weight must be positive")
        allowed_values = {
            str(item["value"]) for item in definition["values"]
        }
        _validate_status_values(
            unit["prediction"],
            allowed_values=allowed_values,
            cardinality=str(definition["cardinality"]),
            name="label prediction",
            include_confidence=True,
        )
        evidence_uid = unit["prediction"]["evidence_uid"]
        if evidence_uid is not None:
            if evidence_uid in evidence_ids:
                raise ValueError(
                    "predicted evidence_uid appears in multiple audit units"
                )
            evidence_ids.add(evidence_uid)
        _validate_status_values(
            unit["reference"],
            allowed_values=allowed_values,
            cardinality=str(definition["cardinality"]),
            name="label reference",
            include_confidence=False,
        )

    predicted_event_ids: set[str] = set()
    reference_event_ids: set[str] = set()
    for unit in document["event_units"]:
        if not isinstance(unit, Mapping):
            raise ValueError("event audit unit must be an object")
        _require_exact_keys(
            unit,
            {
                "unit_uid",
                "scene_uid",
                "primary_event_key",
                "source",
                "sampling_weight",
                "agreement",
                "match_status",
                "predicted_event_uid",
                "reference_event_uid",
                "predicted_start_timestamp_ns",
                "predicted_end_timestamp_ns",
                "reference_start_timestamp_ns",
                "reference_end_timestamp_ns",
                "actor_continuity",
            },
            "event audit unit",
        )
        unit_uid = str(unit["unit_uid"])
        if not unit_uid or unit_uid in unit_ids:
            raise ValueError("audit unit_uid must be non-empty and unique")
        unit_ids.add(unit_uid)
        if not isinstance(unit["scene_uid"], str) or not unit["scene_uid"]:
            raise ValueError("event audit scene_uid is required")
        key = str(unit["primary_event_key"])
        definition = definitions.get(key)
        if definition is None or definition["namespace"] != "event":
            raise ValueError("event audit key is not an event label")
        if str(unit["source"]) not in sources:
            raise ValueError("event audit source is invalid")
        if str(unit["agreement"]) not in AGREEMENT_STATUSES:
            raise ValueError("event audit agreement is invalid")
        weight = float(unit["sampling_weight"])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("event audit sampling_weight must be positive")
        match_status = str(unit["match_status"])
        if match_status not in EVENT_MATCH_STATUSES:
            raise ValueError("event match_status is invalid")
        actor_continuity = str(unit["actor_continuity"])
        if actor_continuity not in ACTOR_CONTINUITY_STATUSES:
            raise ValueError("event actor_continuity is invalid")
        predicted_present = match_status in {"matched", "spurious"}
        reference_present = match_status in {"matched", "missed"}
        _validate_optional_event_side(
            unit,
            prefix="predicted",
            required=predicted_present,
        )
        _validate_optional_event_side(
            unit,
            prefix="reference",
            required=reference_present,
        )
        if predicted_present:
            predicted_event_uid = str(unit["predicted_event_uid"])
            if predicted_event_uid in predicted_event_ids:
                raise ValueError(
                    "predicted event appears in multiple audit units"
                )
            predicted_event_ids.add(predicted_event_uid)
        if reference_present:
            reference_event_uid = str(unit["reference_event_uid"])
            if reference_event_uid in reference_event_ids:
                raise ValueError(
                    "reference event appears in multiple audit units"
                )
            reference_event_ids.add(reference_event_uid)
        if match_status != "matched" and actor_continuity != "not_applicable":
            raise ValueError(
                "unmatched event actor_continuity must be not_applicable"
            )

    question_ids: set[str] = set()
    for question in document["ontology_questions"]:
        if not isinstance(question, Mapping):
            raise ValueError("ontology question must be an object")
        _require_exact_keys(
            question,
            {
                "question_uid",
                "label_key",
                "question",
                "status",
                "resolution",
            },
            "ontology question",
        )
        question_uid = str(question["question_uid"])
        if not question_uid or question_uid in question_ids:
            raise ValueError("ontology question_uid must be unique")
        question_ids.add(question_uid)
        if str(question["label_key"]) not in definitions:
            raise ValueError("ontology question label_key is unknown")
        if not isinstance(question["question"], str) or not question["question"]:
            raise ValueError("ontology question text is required")
        status = str(question["status"])
        resolution = question["resolution"]
        if status == "open" and resolution is not None:
            raise ValueError("open ontology question cannot have a resolution")
        if status == "resolved" and (
            not isinstance(resolution, str) or not resolution
        ):
            raise ValueError("resolved ontology question needs a resolution")
        if status not in {"open", "resolved"}:
            raise ValueError("ontology question status is invalid")


def _validate_optional_event_side(
    unit: Mapping[str, Any],
    *,
    prefix: str,
    required: bool,
) -> None:
    event_uid = unit[f"{prefix}_event_uid"]
    start = unit[f"{prefix}_start_timestamp_ns"]
    end = unit[f"{prefix}_end_timestamp_ns"]
    if required:
        if not isinstance(event_uid, str) or not event_uid:
            raise ValueError(f"{prefix} event_uid is required")
        _validate_interval(start, end, f"{prefix} event")
    elif event_uid is not None or start is not None or end is not None:
        raise ValueError(f"{prefix} event fields must be null")


def _divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _confidence_band(confidence: float) -> str:
    for lower, upper in CONFIDENCE_BANDS:
        if lower <= confidence < upper or (
            upper == 1.0 and confidence == 1.0
        ):
            return f"{lower:.1f}-{upper:.1f}"
    raise ValueError("confidence is outside calibration bands")


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1),
    )
    return ordered[index]


def _label_metrics(
    document: Mapping[str, Any],
    ontology: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    definitions = _definitions(ontology)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for unit in document["label_units"]:
        grouped[(str(unit["label_key"]), str(unit["source"]))].append(unit)

    value_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for (key, source), units in sorted(grouped.items()):
        status_correct = 0
        status_confusion: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for unit in units:
            prediction = unit["prediction"]
            reference = unit["reference"]
            predicted_status = str(prediction["status"])
            reference_status = str(reference["status"])
            status_confusion[reference_status][predicted_status] += 1
            if (
                predicted_status == reference_status
                and set(prediction["values"]) == set(reference["values"])
            ):
                status_correct += 1
        status_rows.append(
            {
                "key": key,
                "source": source,
                "audited_units": len(units),
                "exact_status_value_accuracy": status_correct / len(units),
                "accuracy_ci95": _wilson(status_correct, len(units)),
                "confusion_reference_to_prediction": {
                    reference: dict(sorted(predictions.items()))
                    for reference, predictions in sorted(
                        status_confusion.items()
                    )
                },
            }
        )

        comparable = [
            unit for unit in units if unit["reference"]["status"] == "valid"
        ]
        for value in (
            str(item["value"]) for item in definitions[key]["values"]
        ):
            true_positive = false_positive = false_negative = 0
            weighted_true_positive = 0.0
            weighted_false_positive = 0.0
            weighted_false_negative = 0.0
            positive_references = 0
            for unit in comparable:
                predicted = (
                    set(unit["prediction"]["values"])
                    if unit["prediction"]["status"] == "valid"
                    else set()
                )
                reference = set(unit["reference"]["values"])
                weight = float(unit["sampling_weight"])
                predicted_positive = value in predicted
                reference_positive = value in reference
                positive_references += int(reference_positive)
                if predicted_positive and reference_positive:
                    true_positive += 1
                    weighted_true_positive += weight
                elif predicted_positive:
                    false_positive += 1
                    weighted_false_positive += weight
                elif reference_positive:
                    false_negative += 1
                    weighted_false_negative += weight
            precision_total = true_positive + false_positive
            recall_total = true_positive + false_negative
            negative_references = len(comparable) - positive_references
            value_rows.append(
                {
                    "key": key,
                    "source": source,
                    "value": value,
                    "audited_comparable_units": len(comparable),
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "precision": _divide(true_positive, precision_total),
                    "precision_ci95": _wilson(
                        true_positive,
                        precision_total,
                    ),
                    "recall": _divide(true_positive, recall_total),
                    "recall_ci95": _wilson(
                        true_positive,
                        recall_total,
                    ),
                    "weighted_precision": _divide(
                        weighted_true_positive,
                        weighted_true_positive + weighted_false_positive,
                    ),
                    "weighted_recall": _divide(
                        weighted_true_positive,
                        weighted_true_positive + weighted_false_negative,
                    ),
                    "positive_reference_count": positive_references,
                    "negative_reference_count": negative_references,
                    "sample_sufficiency": {
                        "minimum": MINIMUM_AUDIT_EXAMPLES,
                        "positive_sufficient": (
                            positive_references >= MINIMUM_AUDIT_EXAMPLES
                        ),
                        "negative_sufficient": (
                            negative_references >= MINIMUM_AUDIT_EXAMPLES
                        ),
                    },
                }
            )
    return value_rows, status_rows


def _calibration_metrics(
    document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for unit in document["label_units"]:
        if (
            unit["prediction"]["status"] == "valid"
            and unit["reference"]["status"] == "valid"
        ):
            grouped[(str(unit["label_key"]), str(unit["source"]))].append(unit)
    rows: list[dict[str, Any]] = []
    for (key, source), units in sorted(grouped.items()):
        bands: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for unit in units:
            bands[
                _confidence_band(float(unit["prediction"]["confidence"]))
            ].append(unit)
        band_rows = []
        weighted_error_sum = 0.0
        total_weight = sum(float(unit["sampling_weight"]) for unit in units)
        for band in (
            f"{lower:.1f}-{upper:.1f}"
            for lower, upper in CONFIDENCE_BANDS
        ):
            members = bands.get(band, [])
            if not members:
                continue
            weights = [float(unit["sampling_weight"]) for unit in members]
            correct = [
                (
                    unit["prediction"]["status"]
                    == unit["reference"]["status"]
                    and set(unit["prediction"]["values"])
                    == set(unit["reference"]["values"])
                )
                for unit in members
            ]
            weighted_count = sum(weights)
            mean_confidence = sum(
                float(unit["prediction"]["confidence"]) * weight
                for unit, weight in zip(members, weights, strict=True)
            ) / weighted_count
            empirical_accuracy = sum(
                int(is_correct) * weight
                for is_correct, weight in zip(correct, weights, strict=True)
            ) / weighted_count
            gap = abs(mean_confidence - empirical_accuracy)
            weighted_error_sum += weighted_count * gap
            band_rows.append(
                {
                    "band": band,
                    "count": len(members),
                    "weighted_count": weighted_count,
                    "mean_confidence": mean_confidence,
                    "empirical_accuracy": empirical_accuracy,
                    "absolute_calibration_gap": gap,
                }
            )
        rows.append(
            {
                "key": key,
                "source": source,
                "audited_predictions": len(units),
                "expected_calibration_error": (
                    weighted_error_sum / total_weight
                ),
                "bands": band_rows,
            }
        )
    return rows


def _event_metrics(
    document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for unit in document["event_units"]:
        grouped[
            (str(unit["primary_event_key"]), str(unit["source"]))
        ].append(unit)
    rows = []
    for (key, source), units in sorted(grouped.items()):
        matched = [unit for unit in units if unit["match_status"] == "matched"]
        spurious = [unit for unit in units if unit["match_status"] == "spurious"]
        missed = [unit for unit in units if unit["match_status"] == "missed"]
        onset_errors_ms = [
            abs(
                int(unit["predicted_start_timestamp_ns"])
                - int(unit["reference_start_timestamp_ns"])
            )
            / 1e6
            for unit in matched
        ]
        offset_errors_ms = [
            abs(
                int(unit["predicted_end_timestamp_ns"])
                - int(unit["reference_end_timestamp_ns"])
            )
            / 1e6
            for unit in matched
        ]
        temporal_ious = []
        for unit in matched:
            predicted_start = int(unit["predicted_start_timestamp_ns"])
            predicted_end = int(unit["predicted_end_timestamp_ns"])
            reference_start = int(unit["reference_start_timestamp_ns"])
            reference_end = int(unit["reference_end_timestamp_ns"])
            intersection = max(
                0,
                min(predicted_end, reference_end)
                - max(predicted_start, reference_start),
            )
            union = (
                max(predicted_end, reference_end)
                - min(predicted_start, reference_start)
            )
            temporal_ious.append(intersection / union)
        actor_reviewed = [
            unit
            for unit in matched
            if unit["actor_continuity"]
            in {"correct", "fragmented", "switched"}
        ]
        actor_errors = [
            unit
            for unit in actor_reviewed
            if unit["actor_continuity"] in {"fragmented", "switched"}
        ]
        precision_total = len(matched) + len(spurious)
        recall_total = len(matched) + len(missed)
        matched_weight = sum(
            float(unit["sampling_weight"]) for unit in matched
        )
        spurious_weight = sum(
            float(unit["sampling_weight"]) for unit in spurious
        )
        missed_weight = sum(
            float(unit["sampling_weight"]) for unit in missed
        )
        rows.append(
            {
                "key": key,
                "source": source,
                "matched": len(matched),
                "spurious": len(spurious),
                "missed": len(missed),
                "precision": _divide(len(matched), precision_total),
                "precision_ci95": _wilson(len(matched), precision_total),
                "recall": _divide(len(matched), recall_total),
                "recall_ci95": _wilson(len(matched), recall_total),
                "weighted_precision": _divide(
                    matched_weight,
                    matched_weight + spurious_weight,
                ),
                "weighted_recall": _divide(
                    matched_weight,
                    matched_weight + missed_weight,
                ),
                "boundary_error_ms": {
                    "onset_mean": (
                        sum(onset_errors_ms) / len(onset_errors_ms)
                        if onset_errors_ms
                        else None
                    ),
                    "onset_p50": _percentile(onset_errors_ms, 0.5),
                    "onset_p95": _percentile(onset_errors_ms, 0.95),
                    "offset_mean": (
                        sum(offset_errors_ms) / len(offset_errors_ms)
                        if offset_errors_ms
                        else None
                    ),
                    "offset_p50": _percentile(offset_errors_ms, 0.5),
                    "offset_p95": _percentile(offset_errors_ms, 0.95),
                    "temporal_iou_mean": (
                        sum(temporal_ious) / len(temporal_ious)
                        if temporal_ious
                        else None
                    ),
                },
                "actor_continuity": {
                    "reviewed_count": len(actor_reviewed),
                    "error_count": len(actor_errors),
                    "error_rate": _divide(
                        len(actor_errors),
                        len(actor_reviewed),
                    ),
                    "fragmented_count": sum(
                        unit["actor_continuity"] == "fragmented"
                        for unit in actor_reviewed
                    ),
                    "switched_count": sum(
                        unit["actor_continuity"] == "switched"
                        for unit in actor_reviewed
                    ),
                },
            }
        )
    return rows


def _agreement_summary(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    units = [*document["label_units"], *document["event_units"]]
    counts = {
        status: sum(unit["agreement"] == status for unit in units)
        for status in sorted(AGREEMENT_STATUSES)
    }
    return {
        "reviewed_units": len(units),
        "counts": counts,
        "unanimous_rate": _divide(counts["unanimous"], len(units)),
        "adjudicated_disagreement_rate": _divide(
            counts["adjudicated_disagreement"],
            len(units),
        ),
    }


def evaluate_human_audit_annotations(
    document: Mapping[str, Any],
    ontology: Mapping[str, Any],
) -> dict[str, Any]:
    validate_human_audit_annotations(document, ontology)
    if document["adjudication_status"] != "adjudicated":
        raise ValueError("quality metrics require adjudicated annotations")
    label_metrics, status_metrics = _label_metrics(document, ontology)
    calibration_metrics = _calibration_metrics(document)
    event_metrics = _event_metrics(document)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "document_type": "human_audit_results",
        "labelset_id": document["labelset_id"],
        "annotation_set_id": document["annotation_set_id"],
        "annotation_document_sha256": _sha256_document(document),
        "audit_manifest_sha256": document["audit_manifest_sha256"],
        "status": "measured",
        "certification_status": "experimental_human_metrics_available",
        "certified": False,
        "certification_policy": (
            "metrics never certify automatically; frozen per-family gates "
            "and sample sufficiency require explicit approval"
        ),
        "reviewer_count": document["reviewer_count"],
        "reviewer_agreement": _agreement_summary(document),
        "ontology_questions": {
            "total": len(document["ontology_questions"]),
            "open": sum(
                question["status"] == "open"
                for question in document["ontology_questions"]
            ),
            "resolved": sum(
                question["status"] == "resolved"
                for question in document["ontology_questions"]
            ),
        },
        "label_value_metrics": label_metrics,
        "label_status_metrics": status_metrics,
        "calibration_metrics": calibration_metrics,
        "event_metrics": event_metrics,
    }
