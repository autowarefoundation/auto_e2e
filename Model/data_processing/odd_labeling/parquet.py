"""Explicit Arrow schemas and deterministic Parquet artifacts for ODD labels."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


PARQUET_SCHEMA_VERSION = "odd_parquet_v1"


@dataclasses.dataclass(frozen=True)
class ParquetArtifact:
    payload: bytes
    row_count: int
    sha256: str
    schema_version: str = PARQUET_SCHEMA_VERSION


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _metadata(
    *,
    table_name: str,
    labelset_id: str,
    dataset_name: str,
    dataset_version: str,
    dataset_manifest_sha256: str,
    capability_manifest_sha256: str,
    ontology_sha256: str,
) -> dict[bytes, bytes]:
    values = {
        "odd.parquet_schema_version": PARQUET_SCHEMA_VERSION,
        "odd.table_name": table_name,
        "odd.labelset_id": labelset_id,
        "odd.dataset_name": dataset_name,
        "odd.dataset_version": dataset_version,
        "odd.dataset_manifest_sha256": dataset_manifest_sha256,
        "odd.capability_manifest_sha256": capability_manifest_sha256,
        "odd.ontology_sha256": ontology_sha256,
    }
    return {
        key.encode("ascii"): value.encode("ascii")
        for key, value in values.items()
    }


def _schemas(pa):
    string_list = pa.list_(pa.string())
    candidate_list = pa.list_(
        pa.struct(
            [
                pa.field("value", pa.string(), nullable=False),
                pa.field("score", pa.float32(), nullable=False),
                pa.field("evidence_ref", pa.string()),
            ]
        )
    )
    measurement_list = pa.list_(
        pa.struct(
            [
                pa.field("name", pa.string(), nullable=False),
                pa.field("value_json", pa.string(), nullable=False),
                pa.field("unit", pa.string(), nullable=False),
                pa.field("quality", pa.string(), nullable=False),
                pa.field("aggregation", pa.string(), nullable=False),
            ]
        )
    )
    evidence_ref_list = pa.list_(
        pa.struct(
            [
                pa.field("artifact_uri", pa.string(), nullable=False),
                pa.field("artifact_sha256", pa.string(), nullable=False),
                pa.field("timestamp_ns", pa.int64()),
                pa.field("camera_id", pa.string()),
            ]
        )
    )
    phase_list = pa.list_(
        pa.struct(
            [
                pa.field("phase", pa.string(), nullable=False),
                pa.field("start_timestamp_ns", pa.int64(), nullable=False),
                pa.field("end_timestamp_ns", pa.int64(), nullable=False),
            ]
        )
    )
    return {
        "scene_records": pa.schema(
            [
                pa.field("schema_version", pa.string(), nullable=False),
                pa.field("scene_uid", pa.string(), nullable=False),
                pa.field("dataset_name", pa.string(), nullable=False),
                pa.field("dataset_version", pa.string(), nullable=False),
                pa.field(
                    "dataset_manifest_sha256",
                    pa.string(),
                    nullable=False,
                ),
                pa.field(
                    "capability_manifest_sha256",
                    pa.string(),
                    nullable=False,
                ),
                pa.field("start_timestamp_ns", pa.int64(), nullable=False),
                pa.field("end_timestamp_ns", pa.int64(), nullable=False),
                pa.field("duration_ns", pa.int64(), nullable=False),
                pa.field("distance_m", pa.float64(), nullable=False),
                pa.field("source_artifact_uri", pa.string(), nullable=False),
                pa.field(
                    "source_artifact_sha256",
                    pa.string(),
                    nullable=False,
                ),
                pa.field("evidence_count", pa.int32(), nullable=False),
                pa.field("observation_count", pa.int32(), nullable=False),
                pa.field("event_count", pa.int32(), nullable=False),
                pa.field("provenance_json", pa.string(), nullable=False),
            ]
        ),
        "evidence": pa.schema(
            [
                pa.field("schema_version", pa.string(), nullable=False),
                pa.field("evidence_uid", pa.string(), nullable=False),
                pa.field("scene_uid", pa.string(), nullable=False),
                pa.field("label_key", pa.string(), nullable=False),
                pa.field("namespace", pa.string(), nullable=False),
                pa.field("cardinality", pa.string(), nullable=False),
                pa.field("status", pa.string(), nullable=False),
                pa.field("values", string_list, nullable=False),
                pa.field("candidate_values", candidate_list, nullable=False),
                pa.field("confidence", pa.float32(), nullable=False),
                pa.field("source", pa.string(), nullable=False),
                pa.field("start_timestamp_ns", pa.int64(), nullable=False),
                pa.field("end_timestamp_ns", pa.int64(), nullable=False),
                pa.field("duration_ns", pa.int64(), nullable=False),
                pa.field("subject_type", pa.string(), nullable=False),
                pa.field("subject_id", pa.string()),
                pa.field("anchor_timestamp_ns", pa.int64()),
                pa.field("camera_ids", string_list, nullable=False),
                pa.field("coordinate_frame", pa.string()),
                pa.field("spatial_roi_json", pa.string()),
                pa.field("measurements", measurement_list, nullable=False),
                pa.field("evidence_refs", evidence_ref_list, nullable=False),
                pa.field("labeler_name", pa.string(), nullable=False),
                pa.field("labeler_version", pa.string(), nullable=False),
                pa.field("code_commit", pa.string(), nullable=False),
                pa.field(
                    "container_image_digest",
                    pa.string(),
                    nullable=False,
                ),
                pa.field("config_sha256", pa.string(), nullable=False),
                pa.field("ontology_sha256", pa.string(), nullable=False),
                pa.field(
                    "input_artifact_sha256s",
                    string_list,
                    nullable=False,
                ),
                pa.field("model_provider", pa.string()),
                pa.field("model_name", pa.string()),
                pa.field("model_revision", pa.string()),
                pa.field("prompt_sha256", pa.string()),
                pa.field("decoding_config_sha256", pa.string()),
                pa.field("lookback_ns", pa.int64(), nullable=False),
                pa.field("lookahead_ns", pa.int64(), nullable=False),
                pa.field("provenance_details_json", pa.string(), nullable=False),
            ]
        ),
        "observations": pa.schema(
            [
                pa.field("schema_version", pa.string(), nullable=False),
                pa.field("observation_uid", pa.string(), nullable=False),
                pa.field("scene_uid", pa.string(), nullable=False),
                pa.field("label_key", pa.string(), nullable=False),
                pa.field("namespace", pa.string(), nullable=False),
                pa.field("status", pa.string(), nullable=False),
                pa.field("values", string_list, nullable=False),
                pa.field("confidence", pa.float32(), nullable=False),
                pa.field("source", pa.string(), nullable=False),
                pa.field("start_timestamp_ns", pa.int64(), nullable=False),
                pa.field("end_timestamp_ns", pa.int64(), nullable=False),
                pa.field("duration_ns", pa.int64(), nullable=False),
                pa.field("evidence_uids", string_list, nullable=False),
                pa.field(
                    "conflicting_evidence_uids",
                    string_list,
                    nullable=False,
                ),
                pa.field("measurements_json", pa.string(), nullable=False),
                pa.field("provenance_json", pa.string(), nullable=False),
                pa.field("camera_id", pa.string()),
                pa.field("actor_track_uid", pa.string()),
                pa.field("event_uid", pa.string()),
            ]
        ),
        "events": pa.schema(
            [
                pa.field("schema_version", pa.string(), nullable=False),
                pa.field("event_uid", pa.string(), nullable=False),
                pa.field("scene_uid", pa.string(), nullable=False),
                pa.field("start_timestamp_ns", pa.int64(), nullable=False),
                pa.field("end_timestamp_ns", pa.int64(), nullable=False),
                pa.field("duration_ns", pa.int64(), nullable=False),
                pa.field("primary_event_key", pa.string(), nullable=False),
                pa.field("actor_track_uids", string_list, nullable=False),
                pa.field("observation_uids", string_list, nullable=False),
                pa.field("phases", phase_list, nullable=False),
                pa.field("confidence", pa.float32(), nullable=False),
                pa.field("status", pa.string(), nullable=False),
                pa.field(
                    "supporting_evidence_uids",
                    string_list,
                    nullable=False,
                ),
                pa.field("provenance_json", pa.string(), nullable=False),
            ]
        ),
        "statistics": pa.schema(
            [
                pa.field("schema_version", pa.string(), nullable=False),
                pa.field("labelset_id", pa.string(), nullable=False),
                pa.field("label_key", pa.string(), nullable=False),
                pa.field("namespace", pa.string(), nullable=False),
                pa.field("value", pa.string(), nullable=False),
                pa.field("scene_count", pa.int64(), nullable=False),
                pa.field("scene_ratio", pa.float64(), nullable=False),
                pa.field("duration_ns", pa.int64(), nullable=False),
                pa.field("duration_ratio", pa.float64(), nullable=False),
                pa.field("valid_scene_count", pa.int64(), nullable=False),
                pa.field("eligible_scene_count", pa.int64(), nullable=False),
                pa.field(
                    "observable_scene_coverage",
                    pa.float64(),
                    nullable=False,
                ),
                pa.field("eligible_duration_ns", pa.int64(), nullable=False),
                pa.field("valid_duration_ns", pa.int64(), nullable=False),
                pa.field(
                    "valid_interval_count",
                    pa.int64(),
                    nullable=False,
                ),
                pa.field(
                    "status_scene_counts_json",
                    pa.string(),
                    nullable=False,
                ),
                pa.field(
                    "status_duration_ns_json",
                    pa.string(),
                    nullable=False,
                ),
                pa.field(
                    "source_scene_counts_json",
                    pa.string(),
                    nullable=False,
                ),
            ]
        ),
    }


def _scene_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "schema_version": str(record["schema_version"]),
                "scene_uid": str(record["scene_uid"]),
                "dataset_name": str(record["dataset_name"]),
                "dataset_version": str(record["dataset_version"]),
                "dataset_manifest_sha256": str(
                    record["dataset_manifest_sha256"]
                ),
                "capability_manifest_sha256": str(
                    record["capability_manifest_sha256"]
                ),
                "start_timestamp_ns": int(record["start_timestamp_ns"]),
                "end_timestamp_ns": int(record["end_timestamp_ns"]),
                "duration_ns": (
                    int(record["end_timestamp_ns"])
                    - int(record["start_timestamp_ns"])
                ),
                "distance_m": float(record["distance_m"]),
                "source_artifact_uri": str(record["source_artifact_uri"]),
                "source_artifact_sha256": str(
                    record["source_artifact_sha256"]
                ),
                "evidence_count": len(record.get("evidence", [])),
                "observation_count": len(record.get("observations", [])),
                "event_count": len(record.get("events", [])),
                "provenance_json": _json(record.get("provenance", {})),
            }
        )
    return sorted(rows, key=lambda item: item["scene_uid"])


def _evidence_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for evidence in record.get("evidence", []):
            scope = evidence["scope"]
            provenance = evidence["provenance"]
            rows.append(
                {
                    "schema_version": str(evidence["schema_version"]),
                    "evidence_uid": str(evidence["evidence_uid"]),
                    "scene_uid": str(scope["scene_uid"]),
                    "label_key": str(evidence["label_key"]),
                    "namespace": str(evidence["label_key"]).split(".", 1)[0],
                    "cardinality": str(evidence["cardinality"]),
                    "status": str(evidence["status"]),
                    "values": list(evidence["values"]),
                    "candidate_values": [
                        {
                            "value": str(candidate["value"]),
                            "score": float(candidate["score"]),
                            "evidence_ref": candidate.get("evidence_ref"),
                        }
                        for candidate in evidence["candidate_values"]
                    ],
                    "confidence": float(evidence["confidence"]),
                    "source": str(evidence["source"]),
                    "start_timestamp_ns": int(scope["start_timestamp_ns"]),
                    "end_timestamp_ns": int(scope["end_timestamp_ns"]),
                    "duration_ns": (
                        int(scope["end_timestamp_ns"])
                        - int(scope["start_timestamp_ns"])
                    ),
                    "subject_type": str(scope["subject_type"]),
                    "subject_id": scope.get("subject_id"),
                    "anchor_timestamp_ns": scope.get(
                        "anchor_timestamp_ns"
                    ),
                    "camera_ids": list(scope.get("camera_ids", [])),
                    "coordinate_frame": scope.get("coordinate_frame"),
                    "spatial_roi_json": (
                        _json(scope["spatial_roi"])
                        if scope.get("spatial_roi") is not None
                        else None
                    ),
                    "measurements": [
                        {
                            "name": str(measurement["name"]),
                            "value_json": _json(measurement["value"]),
                            "unit": str(measurement["unit"]),
                            "quality": str(measurement["quality"]),
                            "aggregation": str(
                                measurement["aggregation"]
                            ),
                        }
                        for measurement in evidence["measurements"]
                    ],
                    "evidence_refs": [
                        {
                            "artifact_uri": str(reference["artifact_uri"]),
                            "artifact_sha256": str(
                                reference["artifact_sha256"]
                            ),
                            "timestamp_ns": reference.get("timestamp_ns"),
                            "camera_id": reference.get("camera_id"),
                        }
                        for reference in evidence["evidence_refs"]
                    ],
                    "labeler_name": str(provenance["labeler_name"]),
                    "labeler_version": str(
                        provenance["labeler_version"]
                    ),
                    "code_commit": str(provenance["code_commit"]),
                    "container_image_digest": str(
                        provenance["container_image_digest"]
                    ),
                    "config_sha256": str(provenance["config_sha256"]),
                    "ontology_sha256": str(
                        provenance["ontology_sha256"]
                    ),
                    "input_artifact_sha256s": list(
                        provenance["input_artifact_sha256s"]
                    ),
                    "model_provider": provenance.get("model_provider"),
                    "model_name": provenance.get("model_name"),
                    "model_revision": provenance.get("model_revision"),
                    "prompt_sha256": provenance.get("prompt_sha256"),
                    "decoding_config_sha256": provenance.get(
                        "decoding_config_sha256"
                    ),
                    "lookback_ns": int(provenance["lookback_ns"]),
                    "lookahead_ns": int(provenance["lookahead_ns"]),
                    "provenance_details_json": _json(
                        provenance.get("details", {})
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda item: (
            item["scene_uid"],
            item["start_timestamp_ns"],
            item["label_key"],
            item["evidence_uid"],
        ),
    )


def _observation_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for observation in record.get("observations", []):
            rows.append(
                {
                    "schema_version": str(observation["schema_version"]),
                    "observation_uid": str(
                        observation["observation_uid"]
                    ),
                    "scene_uid": str(observation["scene_uid"]),
                    "label_key": str(observation["key"]),
                    "namespace": str(observation["key"]).split(".", 1)[0],
                    "status": str(observation["status"]),
                    "values": list(observation["values"]),
                    "confidence": float(observation["confidence"]),
                    "source": str(observation["source"]),
                    "start_timestamp_ns": int(
                        observation["start_timestamp_ns"]
                    ),
                    "end_timestamp_ns": int(
                        observation["end_timestamp_ns"]
                    ),
                    "duration_ns": (
                        int(observation["end_timestamp_ns"])
                        - int(observation["start_timestamp_ns"])
                    ),
                    "evidence_uids": list(
                        observation.get("evidence_uids", [])
                    ),
                    "conflicting_evidence_uids": list(
                        observation.get("conflicting_evidence_uids", [])
                    ),
                    "measurements_json": _json(
                        observation.get("measurements", {})
                    ),
                    "provenance_json": _json(
                        observation.get("provenance", {})
                    ),
                    "camera_id": observation.get("camera_id"),
                    "actor_track_uid": observation.get("actor_track_uid"),
                    "event_uid": observation.get("event_uid"),
                }
            )
    return sorted(
        rows,
        key=lambda item: (
            item["scene_uid"],
            item["start_timestamp_ns"],
            item["label_key"],
            item["observation_uid"],
        ),
    )


def _event_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for event in record.get("events", []):
            rows.append(
                {
                    "schema_version": str(event["schema_version"]),
                    "event_uid": str(event["event_uid"]),
                    "scene_uid": str(event["scene_uid"]),
                    "start_timestamp_ns": int(event["start_timestamp_ns"]),
                    "end_timestamp_ns": int(event["end_timestamp_ns"]),
                    "duration_ns": (
                        int(event["end_timestamp_ns"])
                        - int(event["start_timestamp_ns"])
                    ),
                    "primary_event_key": str(event["primary_event_key"]),
                    "actor_track_uids": list(event["actor_track_uids"]),
                    "observation_uids": list(event["observation_uids"]),
                    "phases": [
                        {
                            "phase": str(phase["phase"]),
                            "start_timestamp_ns": int(
                                phase["start_timestamp_ns"]
                            ),
                            "end_timestamp_ns": int(
                                phase["end_timestamp_ns"]
                            ),
                        }
                        for phase in event["phases"]
                    ],
                    "confidence": float(event["confidence"]),
                    "status": str(event["status"]),
                    "supporting_evidence_uids": list(
                        event["supporting_evidence_uids"]
                    ),
                    "provenance_json": _json(
                        event.get("provenance", {})
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda item: (
            item["scene_uid"],
            item["start_timestamp_ns"],
            item["event_uid"],
        ),
    )


def _statistics_rows(statistics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key_row in statistics["keys"]:
        for value_row in key_row["values"]:
            rows.append(
                {
                    "schema_version": str(statistics["schema_version"]),
                    "labelset_id": str(statistics["labelset_id"]),
                    "label_key": str(key_row["key"]),
                    "namespace": str(key_row["namespace"]),
                    "value": str(value_row["value"]),
                    "scene_count": int(value_row["scene_count"]),
                    "scene_ratio": float(value_row["scene_ratio"]),
                    "duration_ns": int(value_row["duration_ns"]),
                    "duration_ratio": float(value_row["duration_ratio"]),
                    "valid_scene_count": int(
                        key_row["valid_scene_count"]
                    ),
                    "eligible_scene_count": int(
                        key_row["eligible_scene_count"]
                    ),
                    "observable_scene_coverage": float(
                        key_row["observable_scene_coverage"]
                    ),
                    "eligible_duration_ns": int(
                        key_row["eligible_duration_ns"]
                    ),
                    "valid_duration_ns": int(
                        key_row["valid_duration_ns"]
                    ),
                    "valid_interval_count": int(
                        key_row.get("valid_interval_count", 0)
                    ),
                    "status_scene_counts_json": _json(
                        key_row["status_scene_counts"]
                    ),
                    "status_duration_ns_json": _json(
                        key_row["status_duration_ns"]
                    ),
                    "source_scene_counts_json": _json(
                        key_row["source_scene_counts"]
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda item: (item["label_key"], item["value"]),
    )


def _write_table(
    pa,
    pq,
    *,
    schema,
    rows: list[dict[str, Any]],
    metadata: Mapping[bytes, bytes],
    dictionary_columns: Sequence[str],
    group_column: str | None,
) -> ParquetArtifact:
    table = pa.Table.from_pylist(
        rows,
        schema=schema.with_metadata(dict(metadata)),
    )
    sink = io.BytesIO()
    writer = pq.ParquetWriter(
        sink,
        table.schema,
        compression="zstd",
        use_dictionary=list(dictionary_columns),
        write_statistics=True,
        data_page_version="2.0",
        version="2.6",
    )
    try:
        if group_column is None or not rows:
            writer.write_table(table, row_group_size=8192)
        else:
            group_index = table.schema.get_field_index(group_column)
            values = table.column(group_index).to_pylist()
            start = 0
            while start < len(values):
                end = start + 1
                while end < len(values) and values[end] == values[start]:
                    end += 1
                writer.write_table(
                    table.slice(start, end - start),
                    row_group_size=end - start,
                )
                start = end
    finally:
        writer.close()
    payload = sink.getvalue()
    reopened = pq.ParquetFile(io.BytesIO(payload))
    if reopened.metadata.num_rows != len(rows):
        raise ValueError("Parquet read-back row count differs")
    if reopened.schema_arrow != table.schema:
        raise ValueError("Parquet read-back schema differs")
    return ParquetArtifact(
        payload=payload,
        row_count=len(rows),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def build_parquet_artifacts(
    records: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    *,
    labelset_id: str,
    dataset_name: str,
    dataset_version: str,
    dataset_manifest_sha256: str,
    capability_manifest_sha256: str,
    ontology_sha256: str,
) -> dict[str, ParquetArtifact]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - image contract
        raise ImportError(
            "ODD LabelSet publication requires pyarrow"
        ) from exc

    if not records:
        raise ValueError("cannot publish empty ODD Parquet tables")
    if any(
        not math.isfinite(float(record["distance_m"]))
        for record in records
    ):
        raise ValueError("scene distance must be finite")
    schemas = _schemas(pa)
    row_builders = {
        "scene_records": lambda: _scene_rows(records),
        "evidence": lambda: _evidence_rows(records),
        "observations": lambda: _observation_rows(records),
        "events": lambda: _event_rows(records),
        "statistics": lambda: _statistics_rows(statistics),
    }
    dictionary_columns = {
        "scene_records": ("dataset_name", "dataset_version"),
        "evidence": (
            "label_key",
            "namespace",
            "cardinality",
            "status",
            "source",
            "subject_type",
            "model_provider",
        ),
        "observations": (
            "label_key",
            "namespace",
            "status",
            "source",
        ),
        "events": ("primary_event_key", "status"),
        "statistics": ("label_key", "namespace", "value"),
    }
    output = {}
    for table_name in (
        "scene_records",
        "evidence",
        "observations",
        "events",
        "statistics",
    ):
        output[table_name] = _write_table(
            pa,
            pq,
            schema=schemas[table_name],
            rows=row_builders[table_name](),
            metadata=_metadata(
                table_name=table_name,
                labelset_id=labelset_id,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                dataset_manifest_sha256=dataset_manifest_sha256,
                capability_manifest_sha256=capability_manifest_sha256,
                ontology_sha256=ontology_sha256,
            ),
            dictionary_columns=dictionary_columns[table_name],
            group_column=(
                "scene_uid"
                if table_name
                in {"scene_records", "evidence", "observations", "events"}
                else None
            ),
        )
    return output
