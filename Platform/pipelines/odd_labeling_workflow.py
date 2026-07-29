"""Standalone scene-level ODD Dataset Labeler and immutable publication."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import tempfile
from typing import List, NamedTuple

from flytekit import (
    LaunchPlan,
    Resources,
    Secret,
    dynamic,
    map_task,
    task,
    workflow,
)
from flytekit.types.file import FlyteFile


ECR_PREFIX = os.environ.get(
    "ECR_PREFIX",
    "381491877296.dkr.ecr.us-west-2.amazonaws.com",
)
DATA_PREP_IMAGE = os.environ.get(
    "AUTO_E2E_DATA_PREP_IMAGE",
    f"{ECR_PREFIX}/auto-e2e/data-prep:latest",
)
ODD_LABELER_VERSION = "odd_dataset_labeler_v5"
ODD_SCENE_INDEX_SCHEMA_VERSION = "odd_scene_index_v2"
MAX_ODD_ARTIFACT_BYTES = 64 << 20
MAX_ODD_PARQUET_BYTES = 512 << 20
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_REVISION_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
PUBLICATION_PREFIX_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,254}$")
ODD_EXECUTABLE_SOURCES = frozenset(
    {"map_route", "gnss_ins", "vlm", "image_qc", "fusion"}
)
ODD_SOURCE_POLICY_VERSIONS = {
    "map_route": "odd_map_route_policy_v1",
    "gnss_ins": "odd_gnss_ins_policy_v1",
    "vlm": "odd_road_vlm_policy_v3",
    "image_qc": "odd_image_qc_policy_v1",
    "fusion": "odd_source_fusion_v1",
}

OddPublication = NamedTuple(
    "OddPublication",
    labelset_id=str,
    manifest_key=str,
    manifest_sha256=str,
)
OddScenePlan = NamedTuple(
    "OddScenePlan",
    descriptors=List[str],
    capability_manifest_json=str,
)


def _scene_labeling_pod_template():
    """Keep long-running VLM calls on their assigned Karpenter node."""
    from flytekit import PodTemplate

    return PodTemplate(
        annotations={"karpenter.sh/do-not-disrupt": "true"},
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _without_labelset_id(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_labelset_id(item)
            for key, item in value.items()
            if key != "labelset_id"
        }
    if isinstance(value, (list, tuple)):
        return [_without_labelset_id(item) for item in value]
    return value


def _semantic_output_merkle_root(
    records: list[dict],
    statistics: dict,
    quality_documents: dict[str, dict],
) -> str:
    leaves: list[tuple[str, object]] = []
    for record in sorted(records, key=lambda item: str(item["scene_uid"])):
        scene_uid = str(record["scene_uid"])
        scene_root = {
            key: value
            for key, value in record.items()
            if key not in {"evidence", "observations", "events"}
        }
        leaves.append((f"scene\0{scene_uid}", scene_root))
        for kind, identity_key, values in (
            ("evidence", "evidence_uid", record.get("evidence", [])),
            (
                "observation",
                "observation_uid",
                record.get("observations", []),
            ),
            ("event", "event_uid", record.get("events", [])),
        ):
            for value in values:
                identity = str(value[identity_key])
                leaves.append((f"{kind}\0{identity}", value))
    leaves.append(("statistics\0dataset", statistics))
    for name, document in quality_documents.items():
        leaves.append((f"quality\0{name}", document))

    identities = [identity for identity, _ in leaves]
    if not leaves or len(identities) != len(set(identities)):
        raise ValueError("semantic output Merkle leaves are empty or duplicated")
    level = [
        hashlib.sha256(
            b"\x00"
            + _canonical_bytes(
                {
                    "identity": identity,
                    "payload": _without_labelset_id(payload),
                }
            )
        ).digest()
        for identity, payload in sorted(leaves, key=lambda item: item[0])
    ]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(b"\x01" + level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def _execution_receipt(
    semantic_partition_sha256: str,
    runtime_metrics: dict[str, float | int],
    *,
    environment: dict[str, str] | None = None,
    created_at: str | None = None,
) -> dict:
    from datetime import UTC, datetime

    from data_processing.odd_labeling.schema import ExecutionReceipt

    env = os.environ if environment is None else environment
    try:
        attempt = int(env.get("FLYTE_ATTEMPT_NUMBER", "0")) + 1
    except ValueError as error:
        raise ValueError("Flyte attempt number is invalid") from error
    execution_id = env.get("FLYTE_INTERNAL_EXECUTION_ID", "local-execution")
    task_execution_id = env.get("HOSTNAME") or (
        f"{env.get('FLYTE_INTERNAL_TASK_NAME', 'local-task')}:{attempt}"
    )
    timestamp = created_at or datetime.now(UTC).isoformat().replace(
        "+00:00",
        "Z",
    )
    return ExecutionReceipt(
        semantic_partition_sha256=semantic_partition_sha256,
        created_at=timestamp,
        flyte_execution_id=execution_id,
        flyte_task_execution_id=task_execution_id,
        attempt=attempt,
        runtime_metrics=runtime_metrics,
    ).to_dict()


def _execution_receipt_key(
    publication_prefix: str,
    receipt: dict,
) -> str:
    _validate_publication_prefix(publication_prefix)
    semantic_partition_sha256 = str(
        receipt["semantic_partition_sha256"]
    )
    _require_digest(
        semantic_partition_sha256,
        "semantic_partition_sha256",
    )
    receipt_sha256 = _sha256(_canonical_bytes(receipt))
    return (
        f"{publication_prefix}/execution-receipts/"
        f"semantic-partition={semantic_partition_sha256}/"
        f"receipt={receipt_sha256}.json"
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile input is invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _provider_report(exchanges: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for exchange in exchanges:
        grouped.setdefault(str(exchange["backend"]), []).append(exchange)
    backend_rows = []
    for backend, rows in sorted(grouped.items()):
        latency = [float(row["latency_ms"]) for row in rows]
        usage: dict[str, float | int] = {}
        for row in rows:
            for name, value in row.get("usage", {}).items():
                usage[name] = usage.get(name, 0) + value
        failures: dict[str, int] = {}
        for row in rows:
            if row["status"] == "succeeded":
                continue
            identity = (
                f"{row['status']}:{row.get('error_type') or 'unknown'}"
            )
            failures[identity] = failures.get(identity, 0) + 1
        backend_rows.append(
            {
                "backend": backend,
                "providers": sorted(
                    {str(row["provider"]) for row in rows}
                ),
                "models": sorted(
                    {
                        f"{row['model']}@{row['model_revision']}"
                        for row in rows
                    }
                ),
                "request_count": len(
                    {str(row["request_sha256"]) for row in rows}
                ),
                "attempt_count": len(rows),
                "successful_count": sum(
                    row["status"] == "succeeded" for row in rows
                ),
                "failure_count": sum(
                    row["status"] != "succeeded" for row in rows
                ),
                "input_image_count": sum(
                    int(row["input_image_count"]) for row in rows
                ),
                "latency_ms": {
                    "total": sum(latency),
                    "mean": sum(latency) / len(latency),
                    "p50": _percentile(latency, 0.5),
                    "p95": _percentile(latency, 0.95),
                    "max": max(latency),
                },
                "usage": dict(sorted(usage.items())),
                "failures": failures,
                "estimated_cost_usd": None,
                "cost_estimation_status": (
                    "unavailable_without_frozen_pricing"
                ),
            }
        )
    return {
        "schema_version": "odd_provider_report_v1",
        "totals": {
            "request_count": len(
                {
                    str(exchange["request_sha256"])
                    for exchange in exchanges
                }
            ),
            "attempt_count": len(exchanges),
            "successful_count": sum(
                exchange["status"] == "succeeded"
                for exchange in exchanges
            ),
            "failure_count": sum(
                exchange["status"] != "succeeded"
                for exchange in exchanges
            ),
            "input_image_count": sum(
                int(exchange["input_image_count"])
                for exchange in exchanges
            ),
        },
        "backends": backend_rows,
    }


def _provider_exchange_key(
    publication_prefix: str,
    labelset_id: str,
    exchange: dict,
) -> str:
    _validate_publication_prefix(publication_prefix)
    exchange_sha256 = _sha256(_canonical_bytes(exchange))
    return (
        f"{publication_prefix}/provider-artifacts/"
        f"labelset={labelset_id}/backend={exchange['backend']}/"
        f"request={exchange['request_sha256']}/"
        f"exchange={exchange_sha256}.json"
    )


def _provider_report_key(
    publication_prefix: str,
    labelset_id: str,
    report: dict,
) -> str:
    _validate_publication_prefix(publication_prefix)
    report_sha256 = _sha256(_canonical_bytes(report))
    return (
        f"{publication_prefix}/provider-reports/"
        f"labelset={labelset_id}/report={report_sha256}.json"
    )


def _normalized_enabled_sources(enabled_sources: List[str]) -> tuple[str, ...]:
    if not enabled_sources:
        raise ValueError("at least one ODD source must be enabled")
    normalized = tuple(sorted(set(enabled_sources)))
    if len(normalized) != len(enabled_sources):
        raise ValueError("enabled ODD sources must be unique")
    unknown = set(normalized) - ODD_EXECUTABLE_SOURCES
    if unknown:
        raise ValueError(f"unsupported enabled ODD sources: {sorted(unknown)}")
    if "fusion" not in normalized:
        raise ValueError("fusion must be enabled for resolved ODD labels")
    return normalized


def odd_labeler_config_document(enabled_sources: List[str]) -> dict:
    normalized = _normalized_enabled_sources(enabled_sources)
    return {
        "schema_version": "odd_labeler_config_v1",
        "labeler_bundle_version": ODD_LABELER_VERSION,
        "enabled_sources": list(normalized),
        "source_policy_versions": {
            source: ODD_SOURCE_POLICY_VERSIONS[source]
            for source in normalized
        },
        "road_vlm_runtime": {
            "timeout_s": 600,
            "max_tokens": 4096,
            "retry_count": 2,
            "temperature": 0.0,
        },
        "bedrock_map_runtime": {
            "max_tokens": 1024,
            "temperature": 0.0,
            "privacy_policy": "privacy_filtered_map_route_only",
        },
        "unresolved_label_policy": (
            "publish_explicit_status_without_value"
        ),
    }


def odd_labeler_config_sha256(enabled_sources: List[str]) -> str:
    return _sha256(_canonical_bytes(odd_labeler_config_document(enabled_sources)))


def _require_digest(value: str, name: str) -> None:
    if HEX_SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _validate_source_semantic_inputs(
    enabled_sources: List[str],
    ontology_sha256: str,
    labeler_config_sha256: str,
) -> tuple[str, ...]:
    from data_processing.odd_labeling.ontology import (
        ontology_sha256 as local_ontology_sha256,
    )

    normalized = _normalized_enabled_sources(enabled_sources)
    _require_digest(ontology_sha256, "ontology_sha256")
    _require_digest(labeler_config_sha256, "labeler_config_sha256")
    if ontology_sha256 != local_ontology_sha256():
        raise ValueError("ontology_sha256 differs from the task implementation")
    if labeler_config_sha256 != odd_labeler_config_sha256(list(normalized)):
        raise ValueError("labeler_config_sha256 differs from enabled sources")
    return normalized


def _validate_publication_prefix(value: str) -> str:
    if (
        PUBLICATION_PREFIX_RE.fullmatch(value) is None
        or value.endswith("/")
        or "//" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("publication_prefix is invalid")
    return value


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-semantic-contract-v1",
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="2", mem="4Gi"),
)
def validate_odd_semantic_contract(
    ontology_version: str,
    ontology_sha256: str,
    labeler_bundle_version: str,
    labeler_config_uri: str,
    labeler_config_sha256: str,
    enabled_sources: List[str],
    road_vlm_provider: str,
    road_vlm_model: str,
    road_vlm_model_revision: str,
    road_vlm_prompt_bundle_sha256: str,
    road_vlm_decoding_config_sha256: str,
    map_resolver_provider: str,
    map_resolver_model_id: str,
    map_resolver_model_revision: str,
    map_resolver_prompt_bundle_sha256: str,
    map_resolver_decoding_config_sha256: str,
    fusion_config_sha256: str,
    calibration_bundle_sha256: str,
    labeler_image_digest: str,
    labeler_source_revision: str,
    camera_anchor_interval_s: float,
    maximum_camera_anchors: int,
    trigger_context_s: float,
    refinement_confidence_threshold: float,
    publication_prefix: str,
) -> str:
    import boto3

    from data_processing.odd_labeling.bedrock_map_resolver import (
        bedrock_map_decoding_config_sha256 as local_bedrock_decoding_sha256,
    )
    from data_processing.odd_labeling.bedrock_map_resolver import (
        bedrock_map_prompt_bundle_sha256 as local_bedrock_prompt_sha256,
    )
    from data_processing.odd_labeling.fusion import (
        fusion_config_sha256 as local_fusion_sha256,
    )
    from data_processing.odd_labeling.ontology import ontology_document
    from data_processing.odd_labeling.openai_compatible import (
        road_vlm_decoding_bundle_sha256 as local_road_decoding_sha256,
    )
    from data_processing.odd_labeling.openai_compatible import (
        road_vlm_prompt_bundle_sha256 as local_road_prompt_sha256,
    )
    from data_processing.odd_labeling.published_snapshot import S3Location
    from data_processing.odd_labeling.quality import (
        calibration_bundle_sha256 as local_calibration_sha256,
    )

    digest_inputs = {
        "ontology_sha256": ontology_sha256,
        "labeler_config_sha256": labeler_config_sha256,
        "road_vlm_prompt_bundle_sha256": road_vlm_prompt_bundle_sha256,
        "road_vlm_decoding_config_sha256": (
            road_vlm_decoding_config_sha256
        ),
        "map_resolver_prompt_bundle_sha256": (
            map_resolver_prompt_bundle_sha256
        ),
        "map_resolver_decoding_config_sha256": (
            map_resolver_decoding_config_sha256
        ),
        "fusion_config_sha256": fusion_config_sha256,
        "calibration_bundle_sha256": calibration_bundle_sha256,
    }
    for name, value in digest_inputs.items():
        _require_digest(value, name)
    if SHA256_RE.fullmatch(labeler_image_digest) is None:
        raise ValueError("labeler_image_digest must be pinned by SHA-256")
    if SOURCE_REVISION_RE.fullmatch(labeler_source_revision) is None:
        raise ValueError("labeler_source_revision must be immutable")
    if (
        labeler_bundle_version != ODD_LABELER_VERSION
        or road_vlm_provider != "openai_compatible"
        or map_resolver_provider != "amazon_bedrock"
        or not road_vlm_model
        or not road_vlm_model_revision
        or not map_resolver_model_id
        or not map_resolver_model_revision
    ):
        raise ValueError("ODD provider or labeler identity is unsupported")
    normalized_sources = _normalized_enabled_sources(enabled_sources)
    if "vlm" not in normalized_sources:
        raise ValueError("the production ODD contract requires the road VLM")
    if (
        camera_anchor_interval_s <= 0
        or maximum_camera_anchors <= 0
        or trigger_context_s < 0
        or not 0 <= refinement_confidence_threshold <= 1
    ):
        raise ValueError("ODD sampling configuration is invalid")
    _validate_publication_prefix(publication_prefix)

    ontology = ontology_document()
    expected = {
        "ontology_version": ontology["ontology_version"],
        "ontology_sha256": ontology["ontology_sha256"],
        "road_vlm_prompt_bundle_sha256": local_road_prompt_sha256(),
        "road_vlm_decoding_config_sha256": (
            local_road_decoding_sha256(max_tokens=4096)
        ),
        "map_resolver_prompt_bundle_sha256": (
            local_bedrock_prompt_sha256()
        ),
        "map_resolver_decoding_config_sha256": (
            local_bedrock_decoding_sha256(max_tokens=1024)
        ),
        "fusion_config_sha256": local_fusion_sha256(),
        "calibration_bundle_sha256": local_calibration_sha256(),
    }
    actual = {
        "ontology_version": ontology_version,
        "ontology_sha256": ontology_sha256,
        "road_vlm_prompt_bundle_sha256": road_vlm_prompt_bundle_sha256,
        "road_vlm_decoding_config_sha256": (
            road_vlm_decoding_config_sha256
        ),
        "map_resolver_prompt_bundle_sha256": (
            map_resolver_prompt_bundle_sha256
        ),
        "map_resolver_decoding_config_sha256": (
            map_resolver_decoding_config_sha256
        ),
        "fusion_config_sha256": fusion_config_sha256,
        "calibration_bundle_sha256": calibration_bundle_sha256,
    }
    if actual != expected:
        differences = sorted(
            name for name in expected if actual[name] != expected[name]
        )
        raise ValueError(
            f"ODD semantic contract differs from implementation: {differences}"
        )

    config_location = S3Location.parse(labeler_config_uri)
    response = boto3.client("s3").get_object(
        Bucket=config_location.bucket,
        Key=config_location.key,
    )
    body = response["Body"]
    try:
        payload = body.read(MAX_ODD_ARTIFACT_BYTES + 1)
    finally:
        body.close()
    if len(payload) > MAX_ODD_ARTIFACT_BYTES:
        raise ValueError("ODD labeler config exceeds size cap")
    if _sha256(payload) != labeler_config_sha256:
        raise ValueError("ODD labeler config digest differs")
    config = json.loads(payload)
    expected_config = odd_labeler_config_document(list(normalized_sources))
    if payload != _canonical_bytes(config) or config != expected_config:
        raise ValueError("ODD labeler config is not the canonical contract")

    return json.dumps(
        {
            "schema_version": "odd_semantic_contract_v1",
            "ontology": {
                "version": ontology_version,
                "sha256": ontology_sha256,
            },
            "labeler": {
                "bundle_version": labeler_bundle_version,
                "config_uri": labeler_config_uri,
                "config_sha256": labeler_config_sha256,
                "image_digest": labeler_image_digest,
                "source_revision": labeler_source_revision,
                "enabled_sources": list(normalized_sources),
            },
            "road_vlm": {
                "provider": road_vlm_provider,
                "model": road_vlm_model,
                "model_revision": road_vlm_model_revision,
                "prompt_bundle_sha256": road_vlm_prompt_bundle_sha256,
                "decoding_config_sha256": (
                    road_vlm_decoding_config_sha256
                ),
            },
            "map_resolver": {
                "provider": map_resolver_provider,
                "model_id": map_resolver_model_id,
                "model_revision": map_resolver_model_revision,
                "prompt_bundle_sha256": (
                    map_resolver_prompt_bundle_sha256
                ),
                "decoding_config_sha256": (
                    map_resolver_decoding_config_sha256
                ),
            },
            "fusion_config_sha256": fusion_config_sha256,
            "calibration_bundle_sha256": calibration_bundle_sha256,
            "sampling": {
                "camera_anchor_interval_s": camera_anchor_interval_s,
                "maximum_camera_anchors": maximum_camera_anchors,
                "trigger_context_s": trigger_context_s,
                "refinement_confidence_threshold": (
                    refinement_confidence_threshold
                ),
            },
            "publication_prefix": publication_prefix,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _scene_summary(
    record: dict,
    *,
    shard_name: str,
    record_key: str,
    record_sha256: str,
    record_byte_size: int,
) -> dict:
    grouped: dict[tuple[object, ...], dict] = {}
    for observation in record["observations"]:
        identity = (
            observation["key"],
            observation["status"],
            tuple(observation["values"]),
            observation["source"],
            observation.get("camera_id"),
            observation.get("actor_track_uid"),
            observation.get("event_uid"),
        )
        current = grouped.get(identity)
        duration = (
            int(observation["end_timestamp_ns"])
            - int(observation["start_timestamp_ns"])
        )
        if current is None:
            grouped[identity] = {
                "key": observation["key"],
                "status": observation["status"],
                "values": observation["values"],
                "source": observation["source"],
                "confidence": float(observation["confidence"]),
                "duration_ns": duration,
                "first_timestamp_ns": int(observation["start_timestamp_ns"]),
                "interval_count": 1,
                "camera_id": observation.get("camera_id"),
                "actor_track_uid": observation.get("actor_track_uid"),
                "event_uid": observation.get("event_uid"),
            }
            continue
        current["confidence"] = max(
            current["confidence"], float(observation["confidence"])
        )
        current["duration_ns"] += duration
        current["first_timestamp_ns"] = min(
            current["first_timestamp_ns"],
            int(observation["start_timestamp_ns"]),
        )
        current["interval_count"] += 1
    return {
        "scene_uid": record["scene_uid"],
        "shard_name": shard_name,
        "record_key": record_key,
        "record_sha256": record_sha256,
        "record_byte_size": record_byte_size,
        "start_timestamp_ns": int(record["start_timestamp_ns"]),
        "end_timestamp_ns": int(record["end_timestamp_ns"]),
        "distance_m": float(record["distance_m"]),
        "observations": sorted(
            grouped.values(),
            key=lambda item: (
                item["key"],
                item["status"],
                item["values"],
                item["source"],
                item["camera_id"] or "",
                item["actor_track_uid"] or "",
            ),
        ),
        "events": sorted(
            [
                {
                    "event_uid": event["event_uid"],
                    "primary_event_key": event["primary_event_key"],
                    "primary_values": event.get("provenance", {}).get(
                        "primary_values",
                        [],
                    ),
                    "start_timestamp_ns": int(
                        event["start_timestamp_ns"]
                    ),
                    "end_timestamp_ns": int(event["end_timestamp_ns"]),
                    "status": event["status"],
                    "confidence": float(event["confidence"]),
                    "actor_track_uids": event["actor_track_uids"],
                    "outcome": event.get("provenance", {}).get(
                        "outcome",
                        "not_observed",
                    ),
                }
                for event in record.get("events", [])
            ],
            key=lambda item: (
                item["start_timestamp_ns"],
                item["end_timestamp_ns"],
                item["event_uid"],
            ),
        ),
    }


def _union_duration(intervals: list[tuple[int, int]]) -> int:
    from data_processing.odd_labeling.statistics import union_duration

    return union_duration(intervals)


def _publication_scope(
    scene_count: int,
    expected_scene_count: int,
    requested_scope: str,
) -> str:
    if scene_count <= 0 or expected_scene_count <= 0:
        raise ValueError("ODD publication scene counts must be positive")
    if requested_scope not in {"smoke", "full"}:
        raise ValueError("ODD publication scope must be smoke or full")
    if requested_scope == "full" and scene_count != expected_scene_count:
        raise ValueError(
            "latest ODD publication requires the complete scene inventory: "
            f"expected={expected_scene_count} actual={scene_count}"
        )
    return requested_scope


def _statistics(records: list[dict], ontology: dict, labelset_id: str) -> dict:
    from data_processing.odd_labeling.statistics import build_statistics

    return build_statistics(records, ontology, labelset_id)


def _put_immutable(
    s3,
    bucket: str,
    key: str,
    payload: bytes,
    *,
    content_type: str = "application/json",
    schema_version: str = "v1",
    maximum_bytes: int = MAX_ODD_ARTIFACT_BYTES,
) -> None:
    from botocore.exceptions import ClientError

    if len(payload) > maximum_bytes:
        raise ValueError(f"ODD artifact exceeds size cap: {key}")
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
            IfNoneMatch="*",
            Metadata={
                "sha256": _sha256(payload),
                "odd-schema": schema_version,
            },
        )
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status != 412:
            raise
        existing = s3.get_object(Bucket=bucket, Key=key)["Body"].read(
            maximum_bytes + 1
        )
        if existing != payload:
            raise ValueError(f"immutable ODD object differs: s3://{bucket}/{key}")


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-resolve-scenes-v3",
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="2", mem="4Gi"),
)
def resolve_odd_scenes(
    dataset_manifest_uri: str,
    dataset_manifest_sha256: str,
    maximum_scenes: int,
) -> OddScenePlan:
    import boto3

    from data_processing.odd_labeling.published_snapshot import (
        PublishedSnapshotAdapter,
    )

    adapter = PublishedSnapshotAdapter(
        boto3.client("s3"),
        dataset_manifest_uri=dataset_manifest_uri,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    descriptors = adapter.list_scenes()
    capability_manifest = adapter.describe_capabilities()
    if maximum_scenes < 0:
        raise ValueError("maximum_scenes must be non-negative")
    if maximum_scenes:
        descriptors = descriptors[:maximum_scenes]
    return OddScenePlan(
        descriptors=[descriptor.to_json() for descriptor in descriptors],
        capability_manifest_json=json.dumps(
            capability_manifest.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _load_canonical_scene(
    descriptor_json: str,
    capability_manifest_json: str,
) -> tuple[object, object, object]:
    import boto3

    from data_processing.odd_labeling.published_snapshot import (
        PublishedSceneDescriptor,
        load_scene_evidence,
    )
    from data_processing.odd_labeling.schema import DatasetCapabilityManifest

    descriptor = PublishedSceneDescriptor.from_json(descriptor_json)
    capability_manifest = DatasetCapabilityManifest.from_json(
        capability_manifest_json
    )
    if (
        descriptor.dataset_name != capability_manifest.dataset_name
        or descriptor.dataset_version != capability_manifest.dataset_version
        or descriptor.dataset_manifest_sha256
        != capability_manifest.dataset_manifest_sha256
    ):
        raise ValueError("scene descriptor differs from capability coordinate")
    evidence = load_scene_evidence(
        boto3.client("s3"),
        descriptor,
        capability_manifest=capability_manifest,
    )
    return descriptor, capability_manifest, evidence


def _source_artifact_file(
    *,
    source_stage: str,
    descriptor_json: str,
    scene_uid: str,
    observations: object,
    provider_exchanges: object = (),
) -> FlyteFile:
    from data_processing.odd_labeling.source_artifact import (
        SourceObservationArtifact,
    )

    artifact = SourceObservationArtifact.create(
        source_stage=source_stage,
        descriptor_json=descriptor_json,
        scene_uid=scene_uid,
        observations=observations,
        provider_exchanges=provider_exchanges,
    )
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f"odd-source-{source_stage}-",
        suffix=".json",
        delete=False,
    ) as stream:
        stream.write(artifact.to_bytes())
        output = stream.name
    return FlyteFile(output)


def _read_source_artifact(
    source_file: FlyteFile,
    *,
    descriptor_json: str,
    source_stage: str,
):
    from data_processing.odd_labeling.source_artifact import (
        SourceObservationArtifact,
    )

    path = source_file.download()
    with open(path, "rb") as stream:
        payload = stream.read(MAX_ODD_ARTIFACT_BYTES + 1)
    if len(payload) > MAX_ODD_ARTIFACT_BYTES:
        raise ValueError("source observation artifact exceeds size cap")
    return SourceObservationArtifact.from_bytes(
        payload,
        expected_descriptor_json=descriptor_json,
        expected_source_stage=source_stage,
    )


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-source-map-route-v2",
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
)
def label_odd_map_route(
    descriptor_json: str,
    capability_manifest_json: str,
    enabled_sources: List[str],
    ontology_sha256: str,
    labeler_config_sha256: str,
) -> FlyteFile:
    from data_processing.odd_labeling.deterministic import label_map_route

    normalized_sources = _validate_source_semantic_inputs(
        enabled_sources,
        ontology_sha256,
        labeler_config_sha256,
    )
    _, _, evidence = _load_canonical_scene(
        descriptor_json,
        capability_manifest_json,
    )
    return _source_artifact_file(
        source_stage="map_route_deterministic",
        descriptor_json=descriptor_json,
        scene_uid=evidence.scene_uid,
        observations=(
            label_map_route(evidence)
            if "map_route" in normalized_sources
            else ()
        ),
    )


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-source-gnss-ins-v2",
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
)
def label_odd_kinematics(
    descriptor_json: str,
    capability_manifest_json: str,
    enabled_sources: List[str],
    ontology_sha256: str,
    labeler_config_sha256: str,
) -> FlyteFile:
    from data_processing.odd_labeling.deterministic import label_kinematics

    normalized_sources = _validate_source_semantic_inputs(
        enabled_sources,
        ontology_sha256,
        labeler_config_sha256,
    )
    _, _, evidence = _load_canonical_scene(
        descriptor_json,
        capability_manifest_json,
    )
    return _source_artifact_file(
        source_stage="gnss_ins",
        descriptor_json=descriptor_json,
        scene_uid=evidence.scene_uid,
        observations=(
            label_kinematics(evidence)
            if "gnss_ins" in normalized_sources
            else ()
        ),
    )


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-source-image-qc-v2",
    requests=Resources(cpu="2", mem="6Gi"),
    limits=Resources(cpu="4", mem="10Gi"),
)
def label_odd_image_quality(
    descriptor_json: str,
    capability_manifest_json: str,
    enabled_sources: List[str],
    ontology_sha256: str,
    labeler_config_sha256: str,
    camera_anchor_interval_s: float,
    maximum_camera_anchors: int,
) -> FlyteFile:
    import boto3

    from data_processing.odd_labeling.image_qc import (
        label_image_quality,
        load_camera_anchors,
    )

    normalized_sources = _validate_source_semantic_inputs(
        enabled_sources,
        ontology_sha256,
        labeler_config_sha256,
    )
    if camera_anchor_interval_s <= 0 or maximum_camera_anchors <= 0:
        raise ValueError("camera anchor sampling must be positive")
    _, _, evidence = _load_canonical_scene(
        descriptor_json,
        capability_manifest_json,
    )
    anchors = ()
    if "image_qc" in normalized_sources:
        anchors = load_camera_anchors(
            boto3.client("s3"),
            evidence,
            interval_s=camera_anchor_interval_s,
            maximum_anchors=maximum_camera_anchors,
        )
    return _source_artifact_file(
        source_stage="image_qc",
        descriptor_json=descriptor_json,
        scene_uid=evidence.scene_uid,
        observations=(
            label_image_quality(evidence, anchors)
            if "image_qc" in normalized_sources
            else ()
        ),
    )


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-source-openai-compatible-v3",
    retries=2,
    pod_template=_scene_labeling_pod_template(),
    requests=Resources(cpu="2", mem="6Gi"),
    limits=Resources(cpu="4", mem="10Gi"),
    secret_requests=[
        Secret(
            group="odd-road-observer",
            key="OPENAI_COMPATIBLE_BASE_URL",
            mount_requirement=Secret.MountType.ENV_VAR,
        ),
        Secret(
            group="odd-road-observer",
            key="OPENAI_COMPATIBLE_API_KEY",
            mount_requirement=Secret.MountType.ENV_VAR,
        ),
    ],
)
def label_odd_visual(
    descriptor_json: str,
    capability_manifest_json: str,
    map_route_file: FlyteFile,
    kinematics_file: FlyteFile,
    image_quality_file: FlyteFile,
    enabled_sources: List[str],
    ontology_sha256: str,
    labeler_config_sha256: str,
    road_vlm_provider: str,
    road_vlm_model: str,
    road_vlm_model_revision: str,
    road_vlm_prompt_bundle_sha256: str,
    road_vlm_decoding_config_sha256: str,
    camera_anchor_interval_s: float,
    maximum_camera_anchors: int,
    trigger_context_s: float,
    refinement_confidence_threshold: float,
) -> FlyteFile:
    import boto3

    from data_processing.odd_labeling.image_qc import load_camera_anchors
    from data_processing.odd_labeling.openai_compatible import (
        OpenAICompatibleRoadObserver,
        RoadVLMConfig,
        derive_visual_trigger_timestamps,
        label_visual_scene,
        road_vlm_decoding_bundle_sha256,
        road_vlm_prompt_bundle_sha256 as local_prompt_bundle_sha256,
    )

    normalized_sources = _validate_source_semantic_inputs(
        enabled_sources,
        ontology_sha256,
        labeler_config_sha256,
    )
    _require_digest(
        road_vlm_prompt_bundle_sha256,
        "road_vlm_prompt_bundle_sha256",
    )
    _require_digest(
        road_vlm_decoding_config_sha256,
        "road_vlm_decoding_config_sha256",
    )
    if (
        camera_anchor_interval_s <= 0
        or maximum_camera_anchors <= 0
        or trigger_context_s < 0
        or not 0 <= refinement_confidence_threshold <= 1
    ):
        raise ValueError("camera anchor sampling must be positive")
    if (
        road_vlm_provider != "openai_compatible"
        or not road_vlm_model
        or not road_vlm_model_revision
        or road_vlm_prompt_bundle_sha256 != local_prompt_bundle_sha256()
        or road_vlm_decoding_config_sha256
        != road_vlm_decoding_bundle_sha256(max_tokens=4096)
    ):
        raise ValueError("road VLM semantic contract is invalid")
    _, _, evidence = _load_canonical_scene(
        descriptor_json,
        capability_manifest_json,
    )
    observations = ()
    provider_exchanges = ()
    if "vlm" in normalized_sources:
        map_route_artifact = _read_source_artifact(
            map_route_file,
            descriptor_json=descriptor_json,
            source_stage="map_route_deterministic",
        )
        kinematics_artifact = _read_source_artifact(
            kinematics_file,
            descriptor_json=descriptor_json,
            source_stage="gnss_ins",
        )
        image_quality_artifact = _read_source_artifact(
            image_quality_file,
            descriptor_json=descriptor_json,
            source_stage="image_qc",
        )
        trigger_timestamps_ns = derive_visual_trigger_timestamps(
            map_route_artifact.observations,
            kinematics_artifact.observations,
            image_quality_artifact.observations,
        )
        from flytekit import current_context

        base_url = current_context().secrets.get(
            "odd-road-observer",
            "OPENAI_COMPATIBLE_BASE_URL",
        )
        api_key = current_context().secrets.get(
            "odd-road-observer",
            "OPENAI_COMPATIBLE_API_KEY",
        )
        if not base_url:
            raise ValueError(
                "OpenAI-compatible labeling requires a configured base URL"
            )
        observer = OpenAICompatibleRoadObserver(
            RoadVLMConfig(
                base_url=base_url,
                model=road_vlm_model,
                api_key=api_key or None,
                timeout_s=600,
                max_tokens=4096,
                retry_count=2,
                model_revision=road_vlm_model_revision,
            )
        )
        anchors = load_camera_anchors(
            boto3.client("s3"),
            evidence,
            interval_s=camera_anchor_interval_s,
            maximum_anchors=maximum_camera_anchors,
            trigger_timestamps_ns=trigger_timestamps_ns,
            trigger_context_s=trigger_context_s,
        )
        sampling_parameters = {
            "regular_interval_s": camera_anchor_interval_s,
            "maximum_anchors": maximum_camera_anchors,
            "trigger_context_s": trigger_context_s,
            "trigger_count": len(trigger_timestamps_ns),
            "refinement_confidence_threshold": (
                refinement_confidence_threshold
            ),
        }
        observations = label_visual_scene(
            observer,
            scene_uid=evidence.scene_uid,
            scene_end_timestamp_ns=evidence.end_timestamp_ns,
            anchors=anchors,
            refinement_confidence_threshold=(
                refinement_confidence_threshold
            ),
            sampling_parameters=sampling_parameters,
        )
        provider_exchanges = observer.provider_exchanges
    return _source_artifact_file(
        source_stage="openai_compatible_vlm",
        descriptor_json=descriptor_json,
        scene_uid=evidence.scene_uid,
        observations=observations,
        provider_exchanges=provider_exchanges,
    )


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-source-bedrock-map-v2",
    retries=2,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
)
def label_odd_bedrock_map(
    descriptor_json: str,
    capability_manifest_json: str,
    map_route_file: FlyteFile,
    enabled_sources: List[str],
    ontology_sha256: str,
    labeler_config_sha256: str,
    map_resolver_provider: str,
    map_resolver_model_id: str,
    map_resolver_model_revision: str,
    map_resolver_prompt_bundle_sha256: str,
    map_resolver_decoding_config_sha256: str,
) -> FlyteFile:
    import boto3

    from data_processing.odd_labeling.bedrock_map_resolver import (
        BedrockMapRouteResolver,
        bedrock_map_decoding_config_sha256,
        bedrock_map_prompt_bundle_sha256,
        resolve_ambiguous_map_route,
    )

    normalized_sources = _validate_source_semantic_inputs(
        enabled_sources,
        ontology_sha256,
        labeler_config_sha256,
    )
    _require_digest(
        map_resolver_prompt_bundle_sha256,
        "map_resolver_prompt_bundle_sha256",
    )
    _require_digest(
        map_resolver_decoding_config_sha256,
        "map_resolver_decoding_config_sha256",
    )
    if (
        map_resolver_provider != "amazon_bedrock"
        or not map_resolver_model_id
        or not map_resolver_model_revision
        or map_resolver_prompt_bundle_sha256
        != bedrock_map_prompt_bundle_sha256()
        or map_resolver_decoding_config_sha256
        != bedrock_map_decoding_config_sha256(max_tokens=1024)
    ):
        raise ValueError("Bedrock map resolver semantic contract is invalid")
    map_artifact = _read_source_artifact(
        map_route_file,
        descriptor_json=descriptor_json,
        source_stage="map_route_deterministic",
    )
    _, _, evidence = _load_canonical_scene(
        descriptor_json,
        capability_manifest_json,
    )
    observations = ()
    provider_exchanges = ()
    if "map_route" in normalized_sources:
        resolver = BedrockMapRouteResolver(
            boto3.client(
                "bedrock-runtime",
                region_name=os.environ.get("AWS_REGION", "us-west-2"),
            ),
            model_id=map_resolver_model_id,
            model_revision=map_resolver_model_revision,
        )
        observations = resolve_ambiguous_map_route(
            resolver,
            evidence,
            map_artifact.observations,
        )
        provider_exchanges = resolver.provider_exchanges
    return _source_artifact_file(
        source_stage="bedrock_map_route",
        descriptor_json=descriptor_json,
        scene_uid=evidence.scene_uid,
        observations=observations,
        provider_exchanges=provider_exchanges,
    )


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-fuse-scene-v4",
    requests=Resources(cpu="2", mem="6Gi"),
    limits=Resources(cpu="4", mem="12Gi"),
)
def fuse_odd_scene(
    descriptor_json: str,
    capability_manifest_json: str,
    map_route_file: FlyteFile,
    kinematics_file: FlyteFile,
    image_quality_file: FlyteFile,
    visual_file: FlyteFile,
    bedrock_map_file: FlyteFile,
    enabled_sources: List[str],
    ontology_sha256: str,
    labeler_config_sha256: str,
    fusion_config_sha256: str,
    calibration_bundle_sha256: str,
    labeler_image_digest: str,
    labeler_source_revision: str,
    camera_anchor_interval_s: float,
    maximum_camera_anchors: int,
    trigger_context_s: float,
    refinement_confidence_threshold: float,
) -> FlyteFile:
    import time

    started_at = time.perf_counter()
    from data_processing.odd_labeling.fusion import (
        EvidenceBuildContext,
        build_resolved_scene_labels,
        fusion_config_sha256 as local_fusion_config_sha256,
    )
    from data_processing.odd_labeling.ontology import ONTOLOGY
    from data_processing.odd_labeling.quality import (
        calibration_bundle_sha256 as local_calibration_bundle_sha256,
    )
    from data_processing.odd_labeling.schema import (
        SceneLabelRecord,
        make_observation,
    )

    normalized_sources = _validate_source_semantic_inputs(
        enabled_sources,
        ontology_sha256,
        labeler_config_sha256,
    )
    _require_digest(fusion_config_sha256, "fusion_config_sha256")
    _require_digest(
        calibration_bundle_sha256,
        "calibration_bundle_sha256",
    )
    if "fusion" not in normalized_sources:
        raise ValueError("fusion source must be enabled")
    if (
        fusion_config_sha256 != local_fusion_config_sha256()
        or calibration_bundle_sha256
        != local_calibration_bundle_sha256()
    ):
        raise ValueError("fusion or calibration semantic contract differs")
    if SHA256_RE.fullmatch(labeler_image_digest) is None:
        raise ValueError("labeler_image_digest must be a sha256 digest")
    if SOURCE_REVISION_RE.fullmatch(labeler_source_revision) is None:
        raise ValueError("labeler_source_revision must be a full Git revision")
    if (
        camera_anchor_interval_s <= 0
        or maximum_camera_anchors <= 0
        or trigger_context_s < 0
        or not 0 <= refinement_confidence_threshold <= 1
    ):
        raise ValueError("VLM sampling configuration is invalid")
    descriptor, capability_manifest, evidence = _load_canonical_scene(
        descriptor_json,
        capability_manifest_json,
    )
    source_files = (
        (map_route_file, "map_route_deterministic"),
        (kinematics_file, "gnss_ins"),
        (image_quality_file, "image_qc"),
        (visual_file, "openai_compatible_vlm"),
        (bedrock_map_file, "bedrock_map_route"),
    )
    source_artifacts = [
        _read_source_artifact(
            source_file,
            descriptor_json=descriptor_json,
            source_stage=source_stage,
        )
        for source_file, source_stage in source_files
    ]
    observations = [
        observation
        for artifact in source_artifacts
        for observation in artifact.observations
    ]
    provider_exchanges = sorted(
        [
            exchange.to_dict()
            for artifact in source_artifacts
            for exchange in artifact.provider_exchanges
        ],
        key=lambda item: (
            item["backend"],
            item["request_sha256"],
            item["attempt"],
            item.get("response_sha256") or "",
            item["status"],
        ),
    )

    observed_keys = {observation.key for observation in observations}
    for key in ONTOLOGY:
        if key in observed_keys:
            continue
        if ONTOLOGY[key].subject in {"actor", "actor_camera"}:
            continue
        observations.append(
            make_observation(
                scene_uid=evidence.scene_uid,
                key=key,
                status="unavailable",
                confidence=0.0,
                source=ONTOLOGY[key].primary_sources[0],
                start_timestamp_ns=evidence.start_timestamp_ns,
                end_timestamp_ns=evidence.end_timestamp_ns,
                provenance={
                    "labeler_version": ODD_LABELER_VERSION,
                    "reason": "required source or implemented resolver unavailable",
                },
            )
        )
    resolved = build_resolved_scene_labels(
        observations,
        context=EvidenceBuildContext(
            dataset_name=descriptor.dataset_name,
            dataset_version=descriptor.dataset_version,
            dataset_manifest_sha256=descriptor.dataset_manifest_sha256,
            capability_manifest_sha256=(
                capability_manifest.semantic_sha256()
            ),
            source_artifact_uri=descriptor.source_uri,
            source_artifact_sha256=descriptor.source_manifest_sha256,
            labeler_image_digest=labeler_image_digest,
            labeler_source_revision=labeler_source_revision,
        ),
    )
    record = SceneLabelRecord(
        scene_uid=evidence.scene_uid,
        dataset_name=descriptor.dataset_name,
        dataset_version=descriptor.dataset_version,
        dataset_manifest_sha256=descriptor.dataset_manifest_sha256,
        start_timestamp_ns=evidence.start_timestamp_ns,
        end_timestamp_ns=evidence.end_timestamp_ns,
        distance_m=evidence.distance_m,
        observations=resolved.observations,
        source_artifact_uri=descriptor.source_uri,
        source_artifact_sha256=descriptor.source_manifest_sha256,
        evidence=resolved.evidence,
        events=resolved.events,
        capability_manifest_sha256=capability_manifest.semantic_sha256(),
        provenance={
            "labeler_version": ODD_LABELER_VERSION,
            "labeler_image_digest": labeler_image_digest,
            "labeler_source_revision": labeler_source_revision,
            "ontology_sha256": ontology_sha256,
            "labeler_config_sha256": labeler_config_sha256,
            "fusion_config_sha256": fusion_config_sha256,
            "calibration_bundle_sha256": calibration_bundle_sha256,
            "enabled_sources": list(normalized_sources),
            "road_vlm_sampling": {
                "regular_interval_s": camera_anchor_interval_s,
                "maximum_anchors": maximum_camera_anchors,
                "trigger_context_s": trigger_context_s,
                "refinement_confidence_threshold": (
                    refinement_confidence_threshold
                ),
            },
        },
    )
    record_sha256 = record.semantic_sha256()
    wrapper = {
        "record": record.to_dict(),
        "record_sha256": record_sha256,
        "shard_name": descriptor.shard_name,
        "partition_id": descriptor.partition_id,
        "receipt": _execution_receipt(
            record_sha256,
            {
                "wall_seconds": time.perf_counter() - started_at,
                "evidence_count": len(record.evidence),
                "observation_count": len(record.observations),
                "event_count": len(record.events),
                "provider_attempt_count": len(provider_exchanges),
                "provider_failure_count": sum(
                    exchange["status"] != "succeeded"
                    for exchange in provider_exchanges
                ),
            },
        ),
        "provider_exchanges": provider_exchanges,
    }
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="odd-scene-",
        suffix=".json",
        delete=False,
    ) as stream:
        stream.write(_canonical_bytes(wrapper))
        output = stream.name
    return FlyteFile(output)


@dynamic(
    container_image=DATA_PREP_IMAGE,
    environment={"AUTO_E2E_DATA_PREP_IMAGE": DATA_PREP_IMAGE},
)
def map_odd_scenes(
    descriptors: List[str],
    capability_manifest_json: str,
    semantic_contract_json: str,
    enabled_sources: List[str],
    ontology_sha256: str,
    labeler_config_sha256: str,
    road_vlm_provider: str,
    road_vlm_model: str,
    road_vlm_model_revision: str,
    road_vlm_prompt_bundle_sha256: str,
    road_vlm_decoding_config_sha256: str,
    map_resolver_provider: str,
    map_resolver_model_id: str,
    map_resolver_model_revision: str,
    map_resolver_prompt_bundle_sha256: str,
    map_resolver_decoding_config_sha256: str,
    fusion_config_sha256: str,
    calibration_bundle_sha256: str,
    labeler_image_digest: str,
    labeler_source_revision: str,
    camera_anchor_interval_s: float,
    maximum_camera_anchors: int,
    trigger_context_s: float,
    refinement_confidence_threshold: float,
    deterministic_concurrency: int,
    image_qc_concurrency: int,
    openai_concurrency: int,
    bedrock_concurrency: int,
    fusion_concurrency: int,
) -> List[FlyteFile]:
    semantic_contract = json.loads(semantic_contract_json)
    if semantic_contract.get("schema_version") != "odd_semantic_contract_v1":
        raise ValueError("validated ODD semantic contract is invalid")
    concurrency_values = (
        deterministic_concurrency,
        image_qc_concurrency,
        openai_concurrency,
        bedrock_concurrency,
        fusion_concurrency,
    )
    if any(value <= 0 for value in concurrency_values):
        raise ValueError("source and fusion concurrency must be positive")
    map_route_files = map_task(
        functools.partial(
            label_odd_map_route,
            capability_manifest_json=capability_manifest_json,
            enabled_sources=enabled_sources,
            ontology_sha256=ontology_sha256,
            labeler_config_sha256=labeler_config_sha256,
        ),
        concurrency=deterministic_concurrency,
    )(descriptor_json=descriptors)
    kinematics_files = map_task(
        functools.partial(
            label_odd_kinematics,
            capability_manifest_json=capability_manifest_json,
            enabled_sources=enabled_sources,
            ontology_sha256=ontology_sha256,
            labeler_config_sha256=labeler_config_sha256,
        ),
        concurrency=deterministic_concurrency,
    )(descriptor_json=descriptors)
    image_quality_files = map_task(
        functools.partial(
            label_odd_image_quality,
            capability_manifest_json=capability_manifest_json,
            enabled_sources=enabled_sources,
            ontology_sha256=ontology_sha256,
            labeler_config_sha256=labeler_config_sha256,
            camera_anchor_interval_s=camera_anchor_interval_s,
            maximum_camera_anchors=maximum_camera_anchors,
        ),
        concurrency=image_qc_concurrency,
    )(descriptor_json=descriptors)
    visual_files = map_task(
        functools.partial(
            label_odd_visual,
            capability_manifest_json=capability_manifest_json,
            enabled_sources=enabled_sources,
            ontology_sha256=ontology_sha256,
            labeler_config_sha256=labeler_config_sha256,
            road_vlm_provider=road_vlm_provider,
            road_vlm_model=road_vlm_model,
            road_vlm_model_revision=road_vlm_model_revision,
            road_vlm_prompt_bundle_sha256=(
                road_vlm_prompt_bundle_sha256
            ),
            road_vlm_decoding_config_sha256=(
                road_vlm_decoding_config_sha256
            ),
            camera_anchor_interval_s=camera_anchor_interval_s,
            maximum_camera_anchors=maximum_camera_anchors,
            trigger_context_s=trigger_context_s,
            refinement_confidence_threshold=(
                refinement_confidence_threshold
            ),
        ),
        concurrency=openai_concurrency,
    )(
        descriptor_json=descriptors,
        map_route_file=map_route_files,
        kinematics_file=kinematics_files,
        image_quality_file=image_quality_files,
    )
    bedrock_map_files = map_task(
        functools.partial(
            label_odd_bedrock_map,
            capability_manifest_json=capability_manifest_json,
            enabled_sources=enabled_sources,
            ontology_sha256=ontology_sha256,
            labeler_config_sha256=labeler_config_sha256,
            map_resolver_provider=map_resolver_provider,
            map_resolver_model_id=map_resolver_model_id,
            map_resolver_model_revision=map_resolver_model_revision,
            map_resolver_prompt_bundle_sha256=(
                map_resolver_prompt_bundle_sha256
            ),
            map_resolver_decoding_config_sha256=(
                map_resolver_decoding_config_sha256
            ),
        ),
        concurrency=bedrock_concurrency,
    )(
        descriptor_json=descriptors,
        map_route_file=map_route_files,
    )
    fuser = map_task(
        functools.partial(
            fuse_odd_scene,
            capability_manifest_json=capability_manifest_json,
            enabled_sources=enabled_sources,
            ontology_sha256=ontology_sha256,
            labeler_config_sha256=labeler_config_sha256,
            fusion_config_sha256=fusion_config_sha256,
            calibration_bundle_sha256=calibration_bundle_sha256,
            labeler_image_digest=labeler_image_digest,
            labeler_source_revision=labeler_source_revision,
            camera_anchor_interval_s=camera_anchor_interval_s,
            maximum_camera_anchors=maximum_camera_anchors,
            trigger_context_s=trigger_context_s,
            refinement_confidence_threshold=(
                refinement_confidence_threshold
            ),
        ),
        concurrency=fusion_concurrency,
    )
    return fuser(
        descriptor_json=descriptors,
        map_route_file=map_route_files,
        kinematics_file=kinematics_files,
        image_quality_file=image_quality_files,
        visual_file=visual_files,
        bedrock_map_file=bedrock_map_files,
    )


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-publish-labelset-v9",
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="4", mem="16Gi"),
)
def publish_odd_labelset(
    scene_files: List[FlyteFile],
    capability_manifest_json: str,
    semantic_contract_json: str,
    dataset_name: str,
    dataset_version: str,
    dataset_manifest_uri: str,
    dataset_manifest_sha256: str,
    datasets_bucket: str,
    ontology_version: str,
    ontology_sha256: str,
    labeler_bundle_version: str,
    labeler_config_uri: str,
    labeler_config_sha256: str,
    enabled_sources: List[str],
    road_vlm_provider: str,
    road_vlm_model: str,
    road_vlm_model_revision: str,
    road_vlm_prompt_bundle_sha256: str,
    road_vlm_decoding_config_sha256: str,
    map_resolver_provider: str,
    map_resolver_model_id: str,
    map_resolver_model_revision: str,
    map_resolver_prompt_bundle_sha256: str,
    map_resolver_decoding_config_sha256: str,
    fusion_config_sha256: str,
    calibration_bundle_sha256: str,
    labeler_image_digest: str,
    labeler_source_revision: str,
    camera_anchor_interval_s: float,
    maximum_camera_anchors: int,
    trigger_context_s: float,
    refinement_confidence_threshold: float,
    publication_scope: str,
    publication_prefix: str,
) -> OddPublication:
    import boto3

    from data_processing.odd_labeling.ontology import ontology_document
    from data_processing.odd_labeling.parquet import (
        PARQUET_SCHEMA_VERSION,
        build_parquet_artifacts,
    )
    from data_processing.odd_labeling.published_snapshot import S3Location
    from data_processing.odd_labeling.quality import (
        QUALITY_SCHEMA_VERSION,
        build_quality_documents,
    )
    from data_processing.odd_labeling.schema import (
        DatasetCapabilityManifest,
        ExecutionReceipt,
        ProviderExchange,
    )
    from data_processing.odd_labeling.statistics import (
        STATISTICS_SCHEMA_VERSION,
    )

    capability_manifest = DatasetCapabilityManifest.from_json(
        capability_manifest_json
    )
    capability_manifest_sha256 = capability_manifest.semantic_sha256()
    semantic_contract = json.loads(semantic_contract_json)
    if semantic_contract.get("schema_version") != "odd_semantic_contract_v1":
        raise ValueError("validated ODD semantic contract is invalid")
    normalized_sources = _validate_source_semantic_inputs(
        enabled_sources,
        ontology_sha256,
        labeler_config_sha256,
    )
    _validate_publication_prefix(publication_prefix)
    if (
        capability_manifest.dataset_name != dataset_name
        or capability_manifest.dataset_version != dataset_version
        or capability_manifest.dataset_manifest_sha256
        != dataset_manifest_sha256
    ):
        raise ValueError("capability manifest differs from publication coordinate")
    wrappers = []
    for scene_file in scene_files:
        path = scene_file.download()
        with open(path, "rb") as stream:
            payload = stream.read(MAX_ODD_ARTIFACT_BYTES + 1)
        if len(payload) > MAX_ODD_ARTIFACT_BYTES:
            raise ValueError("scene ODD record exceeds size cap")
        wrapper = json.loads(payload)
        record_sha256 = _sha256(_canonical_bytes(wrapper["record"]))
        if record_sha256 != wrapper.get("record_sha256"):
            raise ValueError("ODD scene record digest differs from wrapper")
        raw_receipt = wrapper.get("receipt")
        if not isinstance(raw_receipt, dict):
            raise ValueError("ODD scene wrapper has no execution receipt")
        receipt = ExecutionReceipt(**raw_receipt)
        if (
            receipt.semantic_partition_sha256 != record_sha256
            or receipt.to_dict() != raw_receipt
        ):
            raise ValueError("ODD execution receipt differs from scene record")
        raw_exchanges = wrapper.get("provider_exchanges")
        if not isinstance(raw_exchanges, list):
            raise ValueError("ODD scene wrapper has no provider exchanges")
        validated_exchanges = []
        for raw_exchange in raw_exchanges:
            if not isinstance(raw_exchange, dict):
                raise ValueError("ODD provider exchange must be an object")
            exchange = ProviderExchange(**raw_exchange)
            if exchange.to_dict() != raw_exchange:
                raise ValueError("ODD provider exchange is not canonical")
            if (
                exchange.backend == "ORV"
                and exchange.request_metadata.get("scene_uid_sha256")
                != _sha256(wrapper["record"]["scene_uid"].encode("utf-8"))
            ):
                raise ValueError(
                    "ODD road-observer exchange belongs to another scene"
                )
            validated_exchanges.append(exchange.to_dict())
        wrapper["provider_exchanges"] = validated_exchanges
        wrappers.append(wrapper)
    wrappers.sort(key=lambda item: item["record"]["scene_uid"])
    records = [item["record"] for item in wrappers]
    provider_exchanges = [
        exchange
        for wrapper in wrappers
        for exchange in wrapper["provider_exchanges"]
    ]
    scene_ids = [record["scene_uid"] for record in records]
    if not records or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("ODD publication has no scenes or duplicate scene ids")
    if any(
        record["dataset_name"] != dataset_name
        or record["dataset_version"] != dataset_version
        or record["dataset_manifest_sha256"] != dataset_manifest_sha256
        for record in records
    ):
        raise ValueError("ODD scene records differ from publication coordinate")
    if any(
        record.get("capability_manifest_sha256")
        != capability_manifest_sha256
        for record in records
    ):
        raise ValueError("ODD scene records differ from capability manifest")
    if SHA256_RE.fullmatch(labeler_image_digest) is None:
        raise ValueError("labeler_image_digest must be a sha256 digest")
    if SOURCE_REVISION_RE.fullmatch(labeler_source_revision) is None:
        raise ValueError("labeler_source_revision must be a full Git revision")
    if (
        camera_anchor_interval_s <= 0
        or maximum_camera_anchors <= 0
        or trigger_context_s < 0
        or not 0 <= refinement_confidence_threshold <= 1
    ):
        raise ValueError("VLM sampling configuration is invalid")

    manifest_location = S3Location.parse(dataset_manifest_uri)
    manifest_response = boto3.client("s3").get_object(
        Bucket=manifest_location.bucket,
        Key=manifest_location.key,
    )
    manifest_body = manifest_response["Body"]
    try:
        manifest_payload = manifest_body.read(MAX_ODD_ARTIFACT_BYTES + 1)
    finally:
        manifest_body.close()
    if len(manifest_payload) > MAX_ODD_ARTIFACT_BYTES:
        raise ValueError("dataset manifest exceeds ODD publication size cap")
    if _sha256(manifest_payload) != dataset_manifest_sha256:
        raise ValueError("dataset manifest digest differs during publication")
    dataset_manifest = json.loads(manifest_payload)
    if (
        dataset_manifest.get("status") != "ready"
        or dataset_manifest.get("dataset") != dataset_name
        or dataset_manifest.get("version") != dataset_version
    ):
        raise ValueError("dataset manifest differs from publication coordinate")
    expected_scene_count = int(dataset_manifest["episodes"])
    validated_publication_scope = _publication_scope(
        len(records), expected_scene_count, publication_scope
    )

    ontology = ontology_document()
    if (
        ontology["ontology_version"] != ontology_version
        or ontology["ontology_sha256"] != ontology_sha256
        or labeler_bundle_version != ODD_LABELER_VERSION
        or semantic_contract["ontology"]
        != {"version": ontology_version, "sha256": ontology_sha256}
        or semantic_contract["labeler"]["config_uri"]
        != labeler_config_uri
        or semantic_contract["labeler"]["config_sha256"]
        != labeler_config_sha256
        or semantic_contract["labeler"]["enabled_sources"]
        != list(normalized_sources)
        or semantic_contract["publication_prefix"] != publication_prefix
    ):
        raise ValueError("publication semantic contract differs from inputs")
    audit_selection_seed = _sha256(
        _canonical_bytes(
            {
                "policy": "odd_audit_selection_v1",
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "ontology_sha256": ontology_sha256,
                "scene_record_sha256": [
                    item["record_sha256"] for item in wrappers
                ],
            }
        )
    )
    provisional_statistics = _statistics(records, ontology, "")
    provisional_quality_documents = build_quality_documents(
        records,
        provisional_statistics,
        ontology,
        labelset_id="",
        audit_selection_seed=audit_selection_seed,
    )
    semantic_output_merkle_root = _semantic_output_merkle_root(
        records,
        provisional_statistics,
        provisional_quality_documents,
    )
    adapter_bundle_sha256 = _sha256(
        _canonical_bytes(
            {
                "adapter_name": capability_manifest.adapter_name,
                "adapter_version": capability_manifest.adapter_version,
                "scene_inventory_sha256": (
                    capability_manifest.scene_inventory_sha256
                ),
                "capability_manifest_sha256": capability_manifest_sha256,
            }
        )
    )
    labeler_bundle_sha256 = _sha256(
        _canonical_bytes(
            {
                "labeler_bundle_version": labeler_bundle_version,
                "labeler_config_sha256": labeler_config_sha256,
                "labeler_image_digest": labeler_image_digest,
                "labeler_source_revision": labeler_source_revision,
            }
        )
    )
    source_configuration_sha256 = _sha256(
        _canonical_bytes(
            {
                "enabled_sources": list(normalized_sources),
                "road_vlm": semantic_contract["road_vlm"],
                "map_resolver": semantic_contract["map_resolver"],
                "sampling": semantic_contract["sampling"],
            }
        )
    )
    identity = {
        "schema_version": "odd_labelset_identity_v3",
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "capability_manifest_sha256": capability_manifest_sha256,
        "adapter_bundle_sha256": adapter_bundle_sha256,
        "labeler_bundle_sha256": labeler_bundle_sha256,
        "source_configuration_sha256": source_configuration_sha256,
        "fusion_config_sha256": fusion_config_sha256,
        "calibration_bundle_sha256": calibration_bundle_sha256,
        "semantic_contract_sha256": _sha256(
            semantic_contract_json.encode("utf-8")
        ),
        "statistics_schema_version": STATISTICS_SCHEMA_VERSION,
        "parquet_schema_version": PARQUET_SCHEMA_VERSION,
        "quality_schema_version": QUALITY_SCHEMA_VERSION,
        "scene_index_schema_version": ODD_SCENE_INDEX_SCHEMA_VERSION,
        "publication_scope": validated_publication_scope,
        "scene_record_sha256": [
            item["record_sha256"] for item in wrappers
        ],
        "semantic_output_merkle_algorithm": "sha256_binary_dup_last_v1",
        "semantic_output_merkle_root": semantic_output_merkle_root,
    }
    labelset_id = f"oddls-{_sha256(_canonical_bytes(identity))[:32]}"
    root = f"{publication_prefix}/labelsets/{labelset_id}"
    s3 = boto3.client("s3")
    statistics = _statistics(records, ontology, labelset_id)
    quality_documents = build_quality_documents(
        records,
        statistics,
        ontology,
        labelset_id=labelset_id,
        audit_selection_seed=audit_selection_seed,
    )
    if (
        _semantic_output_merkle_root(
            records,
            statistics,
            quality_documents,
        )
        != semantic_output_merkle_root
    ):
        raise ValueError("final ODD outputs differ from semantic Merkle identity")
    parquet_artifacts = build_parquet_artifacts(
        records,
        statistics,
        labelset_id=labelset_id,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_manifest_sha256=dataset_manifest_sha256,
        capability_manifest_sha256=capability_manifest_sha256,
        ontology_sha256=ontology["ontology_sha256"],
    )

    artifacts = {}

    def publish(name: str, value: object, relative_key: str) -> None:
        payload = _canonical_bytes(value)
        key = f"{root}/{relative_key}"
        _put_immutable(s3, datasets_bucket, key, payload)
        artifacts[name] = {
            "key": key,
            "sha256": _sha256(payload),
            "byte_size": len(payload),
            "content_type": "application/json",
            "format": "json",
        }

    receipt_partitions = []
    for wrapper in wrappers:
        receipt = wrapper["receipt"]
        receipt_payload = _canonical_bytes(receipt)
        receipt_key = _execution_receipt_key(
            publication_prefix,
            receipt,
        )
        _put_immutable(
            s3,
            datasets_bucket,
            receipt_key,
            receipt_payload,
        )
        semantic_partition_sha256 = str(
            receipt["semantic_partition_sha256"]
        )
        receipt_partitions.append(
            {
                "scene_uid": wrapper["record"]["scene_uid"],
                "semantic_partition_sha256": semantic_partition_sha256,
                "receipt_prefix": (
                    f"{publication_prefix}/execution-receipts/"
                    f"semantic-partition={semantic_partition_sha256}/"
                ),
            }
        )
    publish(
        "execution_receipt_index",
        {
            "schema_version": "odd_execution_receipt_index_v1",
            "receipt_schema_version": "odd_execution_receipt_v1",
            "labelset_id": labelset_id,
            "partition_count": len(receipt_partitions),
            "partitions": receipt_partitions,
        },
        "receipts/index.json",
    )
    exchange_artifacts = []
    for exchange in provider_exchanges:
        exchange_payload = _canonical_bytes(exchange)
        exchange_key = _provider_exchange_key(
            publication_prefix,
            labelset_id,
            exchange,
        )
        _put_immutable(
            s3,
            datasets_bucket,
            exchange_key,
            exchange_payload,
            schema_version="odd_provider_exchange_v1",
        )
        exchange_artifacts.append(
            {
                "backend": exchange["backend"],
                "request_sha256": exchange["request_sha256"],
                "attempt": exchange["attempt"],
                "status": exchange["status"],
                "key": exchange_key,
                "sha256": _sha256(exchange_payload),
                "byte_size": len(exchange_payload),
            }
        )
    provider_report = {
        **_provider_report(provider_exchanges),
        "labelset_id": labelset_id,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "exchange_artifacts": exchange_artifacts,
    }
    provider_report_payload = _canonical_bytes(provider_report)
    provider_report_key = _provider_report_key(
        publication_prefix,
        labelset_id,
        provider_report,
    )
    _put_immutable(
        s3,
        datasets_bucket,
        provider_report_key,
        provider_report_payload,
        schema_version="odd_provider_report_v1",
    )
    publish(
        "capabilities",
        capability_manifest.to_dict(),
        "capabilities.json",
    )
    if artifacts["capabilities"]["sha256"] != capability_manifest_sha256:
        raise ValueError("published capability digest differs from semantic digest")
    publish("ontology", ontology, "ontology.json")
    publish(
        "statistics",
        statistics,
        "statistics.json",
    )
    publish(
        "quality_coverage",
        quality_documents["coverage"],
        "quality/coverage.json",
    )
    publish(
        "quality_audit_manifest",
        quality_documents["audit_manifest"],
        "quality/audit_manifest.json",
    )
    publish(
        "quality_calibration",
        quality_documents["calibration"],
        "quality/calibration.json",
    )
    parquet_paths = {
        "scene_records": "scene_records/part-00000.parquet",
        "evidence": "evidence/part-00000.parquet",
        "observations": "observations/part-00000.parquet",
        "events": "events/part-00000.parquet",
        "statistics": "statistics/values.parquet",
        "odd_cooccurrences": "statistics/odd_cooccurrences.parquet",
        "odd_event_cooccurrences": (
            "statistics/odd_event_cooccurrences.parquet"
        ),
        "conflicts": "quality/conflicts.parquet",
    }
    if set(parquet_artifacts) != set(parquet_paths):
        raise ValueError(
            "ODD Parquet artifacts differ from the publication layout"
        )
    for table_name, parquet_artifact in parquet_artifacts.items():
        relative_key = parquet_paths[table_name]
        key = f"{root}/{relative_key}"
        _put_immutable(
            s3,
            datasets_bucket,
            key,
            parquet_artifact.payload,
            content_type="application/vnd.apache.parquet",
            schema_version=parquet_artifact.schema_version,
            maximum_bytes=MAX_ODD_PARQUET_BYTES,
        )
        artifacts[f"{table_name}_parquet"] = {
            "key": key,
            "sha256": parquet_artifact.sha256,
            "byte_size": len(parquet_artifact.payload),
            "content_type": "application/vnd.apache.parquet",
            "format": "parquet",
            "row_count": parquet_artifact.row_count,
            "schema_version": parquet_artifact.schema_version,
            "authoritative": True,
        }

    scene_summaries = []
    for wrapper in wrappers:
        record = wrapper["record"]
        scene_digest = hashlib.sha256(
            record["scene_uid"].encode("utf-8")
        ).hexdigest()
        record_key = f"{root}/scenes/{scene_digest}.json"
        record_payload = _canonical_bytes(record)
        record_sha256 = _sha256(record_payload)
        _put_immutable(s3, datasets_bucket, record_key, record_payload)
        scene_summaries.append(
            _scene_summary(
                record,
                shard_name=wrapper["shard_name"],
                record_key=record_key,
                record_sha256=record_sha256,
                record_byte_size=len(record_payload),
            )
        )
    publish(
        "scene_index",
        {
            "schema_version": ODD_SCENE_INDEX_SCHEMA_VERSION,
            "labelset_id": labelset_id,
            "scenes": scene_summaries,
        },
        "scene_index.json",
    )

    manifest = {
        "schema_version": "odd_labelset_manifest_v1",
        "status": "ready",
        "labelset_id": labelset_id,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_manifest_uri": dataset_manifest_uri,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "capability_manifest_sha256": capability_manifest_sha256,
        "adapter": {
            "name": capability_manifest.adapter_name,
            "version": capability_manifest.adapter_version,
            "scene_inventory_sha256": (
                capability_manifest.scene_inventory_sha256
            ),
            "bundle_sha256": adapter_bundle_sha256,
        },
        "ontology_version": ontology["ontology_version"],
        "ontology_sha256": ontology["ontology_sha256"],
        "labeler_version": ODD_LABELER_VERSION,
        "labeler_image_digest": labeler_image_digest,
        "labeler_source_revision": labeler_source_revision,
        "labeler_bundle_sha256": labeler_bundle_sha256,
        "source_configuration_sha256": source_configuration_sha256,
        "fusion_config_sha256": fusion_config_sha256,
        "calibration_bundle_sha256": calibration_bundle_sha256,
        "semantic_output_merkle_algorithm": "sha256_binary_dup_last_v1",
        "semantic_output_merkle_root": semantic_output_merkle_root,
        "semantic_contract": semantic_contract,
        "semantic_contract_sha256": _sha256(
            semantic_contract_json.encode("utf-8")
        ),
        "publication_scope": validated_publication_scope,
        "expected_scene_count": expected_scene_count,
        "scene_count": len(records),
        "execution_receipts": {
            "schema_version": "odd_execution_receipt_index_v1",
            "artifact": "execution_receipt_index",
            "partition_count": len(receipt_partitions),
        },
        "provider_audit": {
            "report_schema_version": "odd_provider_report_v1",
            "report_prefix": (
                f"{publication_prefix}/provider-reports/"
                f"labelset={labelset_id}/"
            ),
            "exchange_schema_version": "odd_provider_exchange_v1",
            "exchange_prefixes": {
                backend: (
                    f"{publication_prefix}/provider-artifacts/"
                    f"labelset={labelset_id}/backend={backend}/"
                )
                for backend in ("ORV", "BMR")
            },
        },
        "openai_compatible": {
            "provider": road_vlm_provider,
            "model": road_vlm_model,
            "model_revision": road_vlm_model_revision,
            "prompt_bundle_sha256": road_vlm_prompt_bundle_sha256,
            "decoding_config_sha256": (
                road_vlm_decoding_config_sha256
            ),
            "sampling": {
                "regular_interval_s": camera_anchor_interval_s,
                "maximum_anchors": maximum_camera_anchors,
                "trigger_context_s": trigger_context_s,
                "refinement_confidence_threshold": (
                    refinement_confidence_threshold
                ),
            },
        },
        "bedrock_map_resolver": {
            "provider": map_resolver_provider,
            "model_id": map_resolver_model_id,
            "model_revision": map_resolver_model_revision,
            "prompt_bundle_sha256": (
                map_resolver_prompt_bundle_sha256
            ),
            "decoding_config_sha256": (
                map_resolver_decoding_config_sha256
            ),
            "input_policy": "privacy_filtered_map_route_only",
        },
        "quality": {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "structural_status": quality_documents["coverage"][
                "structural_validation"
            ]["status"],
            "audit_status": quality_documents["audit_manifest"]["status"],
            "certification_status": "experimental",
        },
        "artifacts": artifacts,
    }
    manifest_payload = _canonical_bytes(manifest)
    manifest_key = f"{root}/manifest.json"
    _put_immutable(s3, datasets_bucket, manifest_key, manifest_payload)
    manifest_sha256 = _sha256(manifest_payload)

    if validated_publication_scope == "full":
        pointer = {
            "schema_version": "odd_labelset_pointer_v1",
            "status": "ready",
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "labelset_id": labelset_id,
            "manifest_key": manifest_key,
            "manifest_sha256": manifest_sha256,
        }
        s3.put_object(
            Bucket=datasets_bucket,
            Key=f"{publication_prefix}/latest.json",
            Body=_canonical_bytes(pointer),
            ContentType="application/json",
            Metadata={"odd-schema": "v1", "status": "ready"},
        )
    return OddPublication(
        labelset_id=labelset_id,
        manifest_key=manifest_key,
        manifest_sha256=manifest_sha256,
    )


@workflow
def wf_generate_odd_labelset(
    dataset_name: str,
    dataset_version: str,
    dataset_manifest_uri: str,
    dataset_manifest_sha256: str,
    datasets_bucket: str,
    ontology_version: str,
    ontology_sha256: str,
    labeler_bundle_version: str,
    labeler_config_uri: str,
    labeler_config_sha256: str,
    enabled_sources: List[str],
    road_vlm_provider: str,
    road_vlm_model: str,
    road_vlm_model_revision: str,
    road_vlm_prompt_bundle_sha256: str,
    road_vlm_decoding_config_sha256: str,
    map_resolver_provider: str,
    map_resolver_model_id: str,
    map_resolver_model_revision: str,
    map_resolver_prompt_bundle_sha256: str,
    map_resolver_decoding_config_sha256: str,
    fusion_config_sha256: str,
    calibration_bundle_sha256: str,
    publication_prefix: str,
    labeler_image_digest: str,
    labeler_source_revision: str,
    camera_anchor_interval_s: float = 1.0,
    maximum_camera_anchors: int = 128,
    trigger_context_s: float = 1.0,
    refinement_confidence_threshold: float = 0.65,
    maximum_scenes: int = 0,
    scene_concurrency: int = 40,
    openai_concurrency: int = 10,
    bedrock_concurrency: int = 20,
    publication_scope: str = "full",
) -> OddPublication:
    """Label a published dataset independently from every training workflow."""
    semantic_contract_json = validate_odd_semantic_contract(
        ontology_version=ontology_version,
        ontology_sha256=ontology_sha256,
        labeler_bundle_version=labeler_bundle_version,
        labeler_config_uri=labeler_config_uri,
        labeler_config_sha256=labeler_config_sha256,
        enabled_sources=enabled_sources,
        road_vlm_provider=road_vlm_provider,
        road_vlm_model=road_vlm_model,
        road_vlm_model_revision=road_vlm_model_revision,
        road_vlm_prompt_bundle_sha256=road_vlm_prompt_bundle_sha256,
        road_vlm_decoding_config_sha256=(
            road_vlm_decoding_config_sha256
        ),
        map_resolver_provider=map_resolver_provider,
        map_resolver_model_id=map_resolver_model_id,
        map_resolver_model_revision=map_resolver_model_revision,
        map_resolver_prompt_bundle_sha256=(
            map_resolver_prompt_bundle_sha256
        ),
        map_resolver_decoding_config_sha256=(
            map_resolver_decoding_config_sha256
        ),
        fusion_config_sha256=fusion_config_sha256,
        calibration_bundle_sha256=calibration_bundle_sha256,
        labeler_image_digest=labeler_image_digest,
        labeler_source_revision=labeler_source_revision,
        camera_anchor_interval_s=camera_anchor_interval_s,
        maximum_camera_anchors=maximum_camera_anchors,
        trigger_context_s=trigger_context_s,
        refinement_confidence_threshold=refinement_confidence_threshold,
        publication_prefix=publication_prefix,
    )
    scene_plan = resolve_odd_scenes(
        dataset_manifest_uri=dataset_manifest_uri,
        dataset_manifest_sha256=dataset_manifest_sha256,
        maximum_scenes=maximum_scenes,
    )
    scene_files = map_odd_scenes(
        descriptors=scene_plan.descriptors,
        capability_manifest_json=scene_plan.capability_manifest_json,
        semantic_contract_json=semantic_contract_json,
        enabled_sources=enabled_sources,
        ontology_sha256=ontology_sha256,
        labeler_config_sha256=labeler_config_sha256,
        road_vlm_provider=road_vlm_provider,
        road_vlm_model=road_vlm_model,
        road_vlm_model_revision=road_vlm_model_revision,
        road_vlm_prompt_bundle_sha256=road_vlm_prompt_bundle_sha256,
        road_vlm_decoding_config_sha256=(
            road_vlm_decoding_config_sha256
        ),
        map_resolver_provider=map_resolver_provider,
        map_resolver_model_id=map_resolver_model_id,
        map_resolver_model_revision=map_resolver_model_revision,
        map_resolver_prompt_bundle_sha256=(
            map_resolver_prompt_bundle_sha256
        ),
        map_resolver_decoding_config_sha256=(
            map_resolver_decoding_config_sha256
        ),
        fusion_config_sha256=fusion_config_sha256,
        calibration_bundle_sha256=calibration_bundle_sha256,
        labeler_image_digest=labeler_image_digest,
        labeler_source_revision=labeler_source_revision,
        camera_anchor_interval_s=camera_anchor_interval_s,
        maximum_camera_anchors=maximum_camera_anchors,
        trigger_context_s=trigger_context_s,
        refinement_confidence_threshold=refinement_confidence_threshold,
        deterministic_concurrency=scene_concurrency,
        image_qc_concurrency=scene_concurrency,
        openai_concurrency=openai_concurrency,
        bedrock_concurrency=bedrock_concurrency,
        fusion_concurrency=scene_concurrency,
    )
    return publish_odd_labelset(
        scene_files=scene_files,
        capability_manifest_json=scene_plan.capability_manifest_json,
        semantic_contract_json=semantic_contract_json,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_manifest_uri=dataset_manifest_uri,
        dataset_manifest_sha256=dataset_manifest_sha256,
        datasets_bucket=datasets_bucket,
        ontology_version=ontology_version,
        ontology_sha256=ontology_sha256,
        labeler_bundle_version=labeler_bundle_version,
        labeler_config_uri=labeler_config_uri,
        labeler_config_sha256=labeler_config_sha256,
        enabled_sources=enabled_sources,
        road_vlm_provider=road_vlm_provider,
        road_vlm_model=road_vlm_model,
        road_vlm_model_revision=road_vlm_model_revision,
        road_vlm_prompt_bundle_sha256=road_vlm_prompt_bundle_sha256,
        road_vlm_decoding_config_sha256=(
            road_vlm_decoding_config_sha256
        ),
        map_resolver_provider=map_resolver_provider,
        map_resolver_model_id=map_resolver_model_id,
        map_resolver_model_revision=map_resolver_model_revision,
        map_resolver_prompt_bundle_sha256=(
            map_resolver_prompt_bundle_sha256
        ),
        map_resolver_decoding_config_sha256=(
            map_resolver_decoding_config_sha256
        ),
        fusion_config_sha256=fusion_config_sha256,
        calibration_bundle_sha256=calibration_bundle_sha256,
        labeler_image_digest=labeler_image_digest,
        labeler_source_revision=labeler_source_revision,
        camera_anchor_interval_s=camera_anchor_interval_s,
        maximum_camera_anchors=maximum_camera_anchors,
        trigger_context_s=trigger_context_s,
        refinement_confidence_threshold=refinement_confidence_threshold,
        publication_scope=publication_scope,
        publication_prefix=publication_prefix,
    )


odd_dataset_labeler_launch_plan = LaunchPlan.get_or_create(
    workflow=wf_generate_odd_labelset,
    name="odd-dataset-labeler",
)
