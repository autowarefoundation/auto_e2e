"""Standalone scene-level ODD Dataset Labeler and immutable publication."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from typing import List, NamedTuple

from flytekit import Resources, Secret, dynamic, map_task, task, workflow
from flytekit.types.file import FlyteFile


ECR_PREFIX = os.environ.get(
    "ECR_PREFIX",
    "381491877296.dkr.ecr.us-west-2.amazonaws.com",
)
DATA_PREP_IMAGE = os.environ.get(
    "AUTO_E2E_DATA_PREP_IMAGE",
    f"{ECR_PREFIX}/auto-e2e/data-prep:latest",
)
ODD_LABELER_VERSION = "odd_dataset_labeler_v2"
MAX_ODD_ARTIFACT_BYTES = 64 << 20
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SOURCE_REVISION_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")

OddPublication = NamedTuple(
    "OddPublication",
    labelset_id=str,
    manifest_key=str,
    manifest_sha256=str,
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


def _scene_summary(
    record: dict,
    *,
    shard_name: str,
    record_key: str,
    record_sha256: str,
    record_byte_size: int,
) -> dict:
    grouped: dict[tuple[str, str, tuple[str, ...], str], dict] = {}
    for observation in record["observations"]:
        identity = (
            observation["key"],
            observation["status"],
            tuple(observation["values"]),
            observation["source"],
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
            ),
        ),
    }


def _union_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    start, end = sorted(intervals)[0]
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start <= end:
            end = max(end, next_end)
            continue
        total += end - start
        start, end = next_start, next_end
    return total + end - start


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
    total_scenes = len(records)
    scene_duration = {
        record["scene_uid"]: (
            int(record["end_timestamp_ns"])
            - int(record["start_timestamp_ns"])
        )
        for record in records
    }
    rows = []
    for definition in ontology["labels"]:
        key = definition["key"]
        valid_scenes: set[str] = set()
        status_scenes: dict[str, set[str]] = defaultdict(set)
        value_scenes: dict[str, set[str]] = defaultdict(set)
        status_intervals: dict[str, dict[str, list[tuple[int, int]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        value_intervals: dict[str, dict[str, list[tuple[int, int]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        source_scenes: dict[str, set[str]] = defaultdict(set)
        for record in records:
            scene_uid = record["scene_uid"]
            for observation in record["observations"]:
                if observation["key"] != key:
                    continue
                status = observation["status"]
                status_scenes[status].add(scene_uid)
                interval = (
                    int(observation["start_timestamp_ns"]),
                    int(observation["end_timestamp_ns"]),
                )
                status_intervals[status][scene_uid].append(interval)
                source_scenes[observation["source"]].add(scene_uid)
                if status != "valid":
                    continue
                valid_scenes.add(scene_uid)
                for value in observation["values"]:
                    value_scenes[value].add(scene_uid)
                    value_intervals[value][scene_uid].append(interval)
        status_duration = {
            status: sum(
                _union_duration(intervals)
                for intervals in by_scene.values()
            )
            for status, by_scene in status_intervals.items()
        }
        value_duration = {
            value: sum(
                _union_duration(intervals)
                for intervals in by_scene.values()
            )
            for value, by_scene in value_intervals.items()
        }
        valid_duration = status_duration.get("valid", 0)
        values = []
        for candidate in definition["values"]:
            value = candidate["value"]
            count = len(value_scenes[value])
            values.append(
                {
                    "value": value,
                    "scene_count": count,
                    "scene_ratio": (
                        count / len(valid_scenes) if valid_scenes else 0.0
                    ),
                    "duration_ns": value_duration.get(value, 0),
                    "duration_ratio": (
                        value_duration.get(value, 0) / valid_duration
                        if valid_duration
                        else 0.0
                    ),
                }
            )
        rows.append(
            {
                "key": key,
                "namespace": definition["namespace"],
                "valid_scene_count": len(valid_scenes),
                "eligible_scene_count": total_scenes,
                "observable_scene_coverage": (
                    len(valid_scenes) / total_scenes if total_scenes else 0.0
                ),
                "eligible_duration_ns": sum(scene_duration.values()),
                "valid_duration_ns": valid_duration,
                "status_scene_counts": {
                    status: len(status_scenes[status])
                    for status in ontology["statuses"]
                },
                "status_duration_ns": {
                    status: status_duration.get(status, 0)
                    for status in ontology["statuses"]
                },
                "source_scene_counts": {
                    source: len(scenes)
                    for source, scenes in sorted(source_scenes.items())
                },
                "values": values,
            }
        )
    return {
        "schema_version": "odd_statistics_v1",
        "labelset_id": labelset_id,
        "scene_count": total_scenes,
        "scene_duration_ns": sum(scene_duration.values()),
        "keys": rows,
    }


def _put_immutable(s3, bucket: str, key: str, payload: bytes) -> None:
    from botocore.exceptions import ClientError

    if len(payload) > MAX_ODD_ARTIFACT_BYTES:
        raise ValueError(f"ODD artifact exceeds size cap: {key}")
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
            IfNoneMatch="*",
            Metadata={"sha256": _sha256(payload), "odd-schema": "v1"},
        )
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status != 412:
            raise
        existing = s3.get_object(Bucket=bucket, Key=key)["Body"].read(
            MAX_ODD_ARTIFACT_BYTES + 1
        )
        if existing != payload:
            raise ValueError(f"immutable ODD object differs: s3://{bucket}/{key}")


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-resolve-scenes-v2",
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="2", mem="4Gi"),
)
def resolve_odd_scenes(
    dataset_manifest_uri: str,
    dataset_manifest_sha256: str,
    maximum_scenes: int,
) -> List[str]:
    import boto3

    from data_processing.odd_labeling.published_snapshot import (
        resolve_scene_descriptors,
    )

    descriptors = resolve_scene_descriptors(
        boto3.client("s3"),
        dataset_manifest_uri=dataset_manifest_uri,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    if maximum_scenes < 0:
        raise ValueError("maximum_scenes must be non-negative")
    if maximum_scenes:
        descriptors = descriptors[:maximum_scenes]
    return [descriptor.to_json() for descriptor in descriptors]


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-label-scene-v4",
    retries=2,
    pod_template=_scene_labeling_pod_template(),
    requests=Resources(cpu="2", mem="6Gi"),
    limits=Resources(cpu="4", mem="12Gi"),
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
def label_odd_scene(
    descriptor_json: str,
    openai_model: str,
    openai_model_revision: str,
    camera_anchor_interval_s: float,
    maximum_camera_anchors: int,
) -> FlyteFile:
    import boto3

    from data_processing.odd_labeling.deterministic import (
        label_kinematics,
        label_map_route,
    )
    from data_processing.odd_labeling.image_qc import (
        label_image_quality,
        load_camera_anchors,
    )
    from data_processing.odd_labeling.ontology import ONTOLOGY
    from data_processing.odd_labeling.openai_compatible import (
        OpenAICompatibleRoadObserver,
        RoadVLMConfig,
        label_visual_scene,
    )
    from data_processing.odd_labeling.published_snapshot import (
        PublishedSceneDescriptor,
        load_scene_evidence,
    )
    from data_processing.odd_labeling.schema import (
        SceneLabelRecord,
        coalesce_observations,
        make_observation,
    )

    if camera_anchor_interval_s <= 0 or maximum_camera_anchors <= 0:
        raise ValueError("camera anchor sampling must be positive")
    descriptor = PublishedSceneDescriptor.from_json(descriptor_json)
    client = boto3.client("s3")
    evidence = load_scene_evidence(client, descriptor)
    anchors = load_camera_anchors(
        client,
        evidence,
        interval_s=camera_anchor_interval_s,
        maximum_anchors=maximum_camera_anchors,
    )
    observations = [
        *label_kinematics(evidence),
        *label_map_route(evidence),
        *label_image_quality(evidence, anchors),
    ]
    if openai_model:
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
                model=openai_model,
                api_key=api_key or None,
                timeout_s=600,
                max_tokens=4096,
                retry_count=2,
                model_revision=openai_model_revision,
            )
        )
        observations.extend(
            label_visual_scene(
                observer,
                scene_uid=evidence.scene_uid,
                scene_end_timestamp_ns=evidence.end_timestamp_ns,
                anchors=anchors,
            )
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
    record = SceneLabelRecord(
        scene_uid=evidence.scene_uid,
        dataset_name=descriptor.dataset_name,
        dataset_version=descriptor.dataset_version,
        dataset_manifest_sha256=descriptor.dataset_manifest_sha256,
        start_timestamp_ns=evidence.start_timestamp_ns,
        end_timestamp_ns=evidence.end_timestamp_ns,
        distance_m=evidence.distance_m,
        observations=coalesce_observations(observations),
        source_artifact_uri=descriptor.source_uri,
        source_artifact_sha256=descriptor.source_manifest_sha256,
    )
    wrapper = {
        "record": record.to_dict(),
        "record_sha256": record.semantic_sha256(),
        "shard_name": descriptor.shard_name,
        "partition_id": descriptor.partition_id,
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
    openai_model: str,
    openai_model_revision: str,
    camera_anchor_interval_s: float,
    maximum_camera_anchors: int,
    scene_concurrency: int,
) -> List[FlyteFile]:
    if scene_concurrency <= 0:
        raise ValueError("scene_concurrency must be positive")
    labeler = map_task(
        functools.partial(
            label_odd_scene,
            openai_model=openai_model,
            openai_model_revision=openai_model_revision,
            camera_anchor_interval_s=camera_anchor_interval_s,
            maximum_camera_anchors=maximum_camera_anchors,
        ),
        concurrency=scene_concurrency,
    )
    return labeler(descriptor_json=descriptors)


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-publish-labelset-v1",
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="4", mem="16Gi"),
)
def publish_odd_labelset(
    scene_files: List[FlyteFile],
    dataset_name: str,
    dataset_version: str,
    dataset_manifest_uri: str,
    dataset_manifest_sha256: str,
    datasets_bucket: str,
    openai_model: str,
    openai_model_revision: str,
    labeler_image_digest: str,
    labeler_source_revision: str,
    publication_scope: str,
) -> OddPublication:
    import boto3

    from data_processing.odd_labeling.ontology import ontology_document
    from data_processing.odd_labeling.published_snapshot import S3Location

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
        wrappers.append(wrapper)
    wrappers.sort(key=lambda item: item["record"]["scene_uid"])
    records = [item["record"] for item in wrappers]
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
    if SHA256_RE.fullmatch(labeler_image_digest) is None:
        raise ValueError("labeler_image_digest must be a sha256 digest")
    if SOURCE_REVISION_RE.fullmatch(labeler_source_revision) is None:
        raise ValueError("labeler_source_revision must be a full Git revision")

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
    identity = {
        "schema_version": "odd_labelset_identity_v1",
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "ontology_sha256": ontology["ontology_sha256"],
        "labeler_version": ODD_LABELER_VERSION,
        "labeler_image_digest": labeler_image_digest,
        "labeler_source_revision": labeler_source_revision,
        "openai_model": openai_model,
        "openai_model_revision": openai_model_revision,
        "publication_scope": validated_publication_scope,
        "scene_record_sha256": [
            item["record_sha256"] for item in wrappers
        ],
    }
    labelset_id = f"oddls-{_sha256(_canonical_bytes(identity))[:32]}"
    root = (
        f"{dataset_name}/{dataset_version}/odd/labelsets/{labelset_id}"
    )
    s3 = boto3.client("s3")

    artifacts = {}

    def publish(name: str, value: object, relative_key: str) -> None:
        payload = _canonical_bytes(value)
        key = f"{root}/{relative_key}"
        _put_immutable(s3, datasets_bucket, key, payload)
        artifacts[name] = {
            "key": key,
            "sha256": _sha256(payload),
            "byte_size": len(payload),
        }

    publish("ontology", ontology, "ontology.json")
    publish(
        "statistics",
        _statistics(records, ontology, labelset_id),
        "statistics.json",
    )

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
            "schema_version": "odd_scene_index_v1",
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
        "ontology_version": ontology["ontology_version"],
        "ontology_sha256": ontology["ontology_sha256"],
        "labeler_version": ODD_LABELER_VERSION,
        "labeler_image_digest": labeler_image_digest,
        "labeler_source_revision": labeler_source_revision,
        "publication_scope": validated_publication_scope,
        "expected_scene_count": expected_scene_count,
        "scene_count": len(records),
        "openai_compatible": {
            "model": openai_model,
            "model_revision": openai_model_revision,
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
            Key=f"{dataset_name}/{dataset_version}/odd/latest.json",
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
    openai_model: str,
    openai_model_revision: str,
    labeler_image_digest: str,
    labeler_source_revision: str,
    camera_anchor_interval_s: float = 4.0,
    maximum_camera_anchors: int = 12,
    maximum_scenes: int = 0,
    scene_concurrency: int = 40,
    publication_scope: str = "full",
) -> OddPublication:
    """Label a published dataset independently from every training workflow."""
    descriptors = resolve_odd_scenes(
        dataset_manifest_uri=dataset_manifest_uri,
        dataset_manifest_sha256=dataset_manifest_sha256,
        maximum_scenes=maximum_scenes,
    )
    scene_files = map_odd_scenes(
        descriptors=descriptors,
        openai_model=openai_model,
        openai_model_revision=openai_model_revision,
        camera_anchor_interval_s=camera_anchor_interval_s,
        maximum_camera_anchors=maximum_camera_anchors,
        scene_concurrency=scene_concurrency,
    )
    return publish_odd_labelset(
        scene_files=scene_files,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_manifest_uri=dataset_manifest_uri,
        dataset_manifest_sha256=dataset_manifest_sha256,
        datasets_bucket=datasets_bucket,
        openai_model=openai_model,
        openai_model_revision=openai_model_revision,
        labeler_image_digest=labeler_image_digest,
        labeler_source_revision=labeler_source_revision,
        publication_scope=publication_scope,
    )
