"""Standalone scene-level ODD Dataset Labeler and immutable publication."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import tempfile
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
ODD_LABELER_VERSION = "odd_dataset_labeler_v4"
MAX_ODD_ARTIFACT_BYTES = 64 << 20
MAX_ODD_PARQUET_BYTES = 512 << 20
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SOURCE_REVISION_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")

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
) -> FlyteFile:
    from data_processing.odd_labeling.source_artifact import (
        SourceObservationArtifact,
    )

    artifact = SourceObservationArtifact.create(
        source_stage=source_stage,
        descriptor_json=descriptor_json,
        scene_uid=scene_uid,
        observations=observations,
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
    cache_version="odd-source-map-route-v1",
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
)
def label_odd_map_route(
    descriptor_json: str,
    capability_manifest_json: str,
) -> FlyteFile:
    from data_processing.odd_labeling.deterministic import label_map_route

    _, _, evidence = _load_canonical_scene(
        descriptor_json,
        capability_manifest_json,
    )
    return _source_artifact_file(
        source_stage="map_route_deterministic",
        descriptor_json=descriptor_json,
        scene_uid=evidence.scene_uid,
        observations=label_map_route(evidence),
    )


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-source-gnss-ins-v1",
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
)
def label_odd_kinematics(
    descriptor_json: str,
    capability_manifest_json: str,
) -> FlyteFile:
    from data_processing.odd_labeling.deterministic import label_kinematics

    _, _, evidence = _load_canonical_scene(
        descriptor_json,
        capability_manifest_json,
    )
    return _source_artifact_file(
        source_stage="gnss_ins",
        descriptor_json=descriptor_json,
        scene_uid=evidence.scene_uid,
        observations=label_kinematics(evidence),
    )


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-source-image-qc-v1",
    requests=Resources(cpu="2", mem="6Gi"),
    limits=Resources(cpu="4", mem="10Gi"),
)
def label_odd_image_quality(
    descriptor_json: str,
    capability_manifest_json: str,
    camera_anchor_interval_s: float,
    maximum_camera_anchors: int,
) -> FlyteFile:
    import boto3

    from data_processing.odd_labeling.image_qc import (
        label_image_quality,
        load_camera_anchors,
    )

    if camera_anchor_interval_s <= 0 or maximum_camera_anchors <= 0:
        raise ValueError("camera anchor sampling must be positive")
    _, _, evidence = _load_canonical_scene(
        descriptor_json,
        capability_manifest_json,
    )
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
        observations=label_image_quality(evidence, anchors),
    )


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-source-openai-compatible-v1",
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
    openai_model: str,
    openai_model_revision: str,
    camera_anchor_interval_s: float,
    maximum_camera_anchors: int,
) -> FlyteFile:
    import boto3

    from data_processing.odd_labeling.image_qc import load_camera_anchors
    from data_processing.odd_labeling.openai_compatible import (
        OpenAICompatibleRoadObserver,
        RoadVLMConfig,
        label_visual_scene,
    )

    if camera_anchor_interval_s <= 0 or maximum_camera_anchors <= 0:
        raise ValueError("camera anchor sampling must be positive")
    if openai_model and not openai_model_revision:
        raise ValueError("OpenAI-compatible model revision must be pinned")
    _, _, evidence = _load_canonical_scene(
        descriptor_json,
        capability_manifest_json,
    )
    observations = ()
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
        anchors = load_camera_anchors(
            boto3.client("s3"),
            evidence,
            interval_s=camera_anchor_interval_s,
            maximum_anchors=maximum_camera_anchors,
        )
        observations = label_visual_scene(
            observer,
            scene_uid=evidence.scene_uid,
            scene_end_timestamp_ns=evidence.end_timestamp_ns,
            anchors=anchors,
        )
    return _source_artifact_file(
        source_stage="openai_compatible_vlm",
        descriptor_json=descriptor_json,
        scene_uid=evidence.scene_uid,
        observations=observations,
    )


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-source-bedrock-map-v1",
    retries=2,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
)
def label_odd_bedrock_map(
    descriptor_json: str,
    capability_manifest_json: str,
    map_route_file: FlyteFile,
    bedrock_map_model_id: str,
    bedrock_map_model_revision: str,
) -> FlyteFile:
    import boto3

    from data_processing.odd_labeling.bedrock_map_resolver import (
        BedrockMapRouteResolver,
        resolve_ambiguous_map_route,
    )

    if bool(bedrock_map_model_id) != bool(bedrock_map_model_revision):
        raise ValueError("Bedrock map model ID and revision must be paired")
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
    if bedrock_map_model_id:
        resolver = BedrockMapRouteResolver(
            boto3.client(
                "bedrock-runtime",
                region_name=os.environ.get("AWS_REGION", "us-west-2"),
            ),
            model_id=bedrock_map_model_id,
            model_revision=bedrock_map_model_revision,
        )
        observations = resolve_ambiguous_map_route(
            resolver,
            evidence,
            map_artifact.observations,
        )
    return _source_artifact_file(
        source_stage="bedrock_map_route",
        descriptor_json=descriptor_json,
        scene_uid=evidence.scene_uid,
        observations=observations,
    )


@task(
    container_image=DATA_PREP_IMAGE,
    cache=True,
    cache_version="odd-fuse-scene-v1",
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
    labeler_image_digest: str,
    labeler_source_revision: str,
) -> FlyteFile:
    from data_processing.odd_labeling.fusion import (
        EvidenceBuildContext,
        build_resolved_scene_labels,
    )
    from data_processing.odd_labeling.ontology import ONTOLOGY
    from data_processing.odd_labeling.schema import (
        SceneLabelRecord,
        make_observation,
    )

    if SHA256_RE.fullmatch(labeler_image_digest) is None:
        raise ValueError("labeler_image_digest must be a sha256 digest")
    if SOURCE_REVISION_RE.fullmatch(labeler_source_revision) is None:
        raise ValueError("labeler_source_revision must be a full Git revision")
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
    observations = [
        observation
        for source_file, source_stage in source_files
        for observation in _read_source_artifact(
            source_file,
            descriptor_json=descriptor_json,
            source_stage=source_stage,
        ).observations
    ]

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
        },
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
    capability_manifest_json: str,
    openai_model: str,
    openai_model_revision: str,
    bedrock_map_model_id: str,
    bedrock_map_model_revision: str,
    labeler_image_digest: str,
    labeler_source_revision: str,
    camera_anchor_interval_s: float,
    maximum_camera_anchors: int,
    deterministic_concurrency: int,
    image_qc_concurrency: int,
    openai_concurrency: int,
    bedrock_concurrency: int,
    fusion_concurrency: int,
) -> List[FlyteFile]:
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
        ),
        concurrency=deterministic_concurrency,
    )(descriptor_json=descriptors)
    kinematics_files = map_task(
        functools.partial(
            label_odd_kinematics,
            capability_manifest_json=capability_manifest_json,
        ),
        concurrency=deterministic_concurrency,
    )(descriptor_json=descriptors)
    image_quality_files = map_task(
        functools.partial(
            label_odd_image_quality,
            capability_manifest_json=capability_manifest_json,
            camera_anchor_interval_s=camera_anchor_interval_s,
            maximum_camera_anchors=maximum_camera_anchors,
        ),
        concurrency=image_qc_concurrency,
    )(descriptor_json=descriptors)
    visual_files = map_task(
        functools.partial(
            label_odd_visual,
            capability_manifest_json=capability_manifest_json,
            openai_model=openai_model,
            openai_model_revision=openai_model_revision,
            camera_anchor_interval_s=camera_anchor_interval_s,
            maximum_camera_anchors=maximum_camera_anchors,
        ),
        concurrency=openai_concurrency,
    )(descriptor_json=descriptors)
    bedrock_map_files = map_task(
        functools.partial(
            label_odd_bedrock_map,
            capability_manifest_json=capability_manifest_json,
            bedrock_map_model_id=bedrock_map_model_id,
            bedrock_map_model_revision=bedrock_map_model_revision,
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
            labeler_image_digest=labeler_image_digest,
            labeler_source_revision=labeler_source_revision,
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
    cache_version="odd-publish-labelset-v6",
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="4", mem="16Gi"),
)
def publish_odd_labelset(
    scene_files: List[FlyteFile],
    capability_manifest_json: str,
    dataset_name: str,
    dataset_version: str,
    dataset_manifest_uri: str,
    dataset_manifest_sha256: str,
    datasets_bucket: str,
    openai_model: str,
    openai_model_revision: str,
    bedrock_map_model_id: str,
    bedrock_map_model_revision: str,
    labeler_image_digest: str,
    labeler_source_revision: str,
    publication_scope: str,
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
    from data_processing.odd_labeling.schema import DatasetCapabilityManifest
    from data_processing.odd_labeling.statistics import (
        STATISTICS_SCHEMA_VERSION,
    )

    capability_manifest = DatasetCapabilityManifest.from_json(
        capability_manifest_json
    )
    capability_manifest_sha256 = capability_manifest.semantic_sha256()
    if (
        capability_manifest.dataset_name != dataset_name
        or capability_manifest.dataset_version != dataset_version
        or capability_manifest.dataset_manifest_sha256
        != dataset_manifest_sha256
    ):
        raise ValueError("capability manifest differs from publication coordinate")
    if bool(bedrock_map_model_id) != bool(bedrock_map_model_revision):
        raise ValueError("Bedrock map model ID and revision must be paired")
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
        "capability_manifest_sha256": capability_manifest_sha256,
        "ontology_sha256": ontology["ontology_sha256"],
        "labeler_version": ODD_LABELER_VERSION,
        "labeler_image_digest": labeler_image_digest,
        "labeler_source_revision": labeler_source_revision,
        "openai_model": openai_model,
        "openai_model_revision": openai_model_revision,
        "bedrock_map_model_id": bedrock_map_model_id,
        "bedrock_map_model_revision": bedrock_map_model_revision,
        "statistics_schema_version": STATISTICS_SCHEMA_VERSION,
        "parquet_schema_version": PARQUET_SCHEMA_VERSION,
        "quality_schema_version": QUALITY_SCHEMA_VERSION,
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
    statistics = _statistics(records, ontology, labelset_id)
    quality_documents = build_quality_documents(
        records,
        statistics,
        ontology,
        labelset_id=labelset_id,
    )
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
        "capability_manifest_sha256": capability_manifest_sha256,
        "adapter": {
            "name": capability_manifest.adapter_name,
            "version": capability_manifest.adapter_version,
            "scene_inventory_sha256": (
                capability_manifest.scene_inventory_sha256
            ),
        },
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
        "bedrock_map_resolver": {
            "model_id": bedrock_map_model_id,
            "model_revision": bedrock_map_model_revision,
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
    bedrock_map_model_id: str,
    bedrock_map_model_revision: str,
    labeler_image_digest: str,
    labeler_source_revision: str,
    camera_anchor_interval_s: float = 4.0,
    maximum_camera_anchors: int = 12,
    maximum_scenes: int = 0,
    scene_concurrency: int = 40,
    openai_concurrency: int = 10,
    bedrock_concurrency: int = 20,
    publication_scope: str = "full",
) -> OddPublication:
    """Label a published dataset independently from every training workflow."""
    scene_plan = resolve_odd_scenes(
        dataset_manifest_uri=dataset_manifest_uri,
        dataset_manifest_sha256=dataset_manifest_sha256,
        maximum_scenes=maximum_scenes,
    )
    scene_files = map_odd_scenes(
        descriptors=scene_plan.descriptors,
        capability_manifest_json=scene_plan.capability_manifest_json,
        openai_model=openai_model,
        openai_model_revision=openai_model_revision,
        bedrock_map_model_id=bedrock_map_model_id,
        bedrock_map_model_revision=bedrock_map_model_revision,
        labeler_image_digest=labeler_image_digest,
        labeler_source_revision=labeler_source_revision,
        camera_anchor_interval_s=camera_anchor_interval_s,
        maximum_camera_anchors=maximum_camera_anchors,
        deterministic_concurrency=scene_concurrency,
        image_qc_concurrency=scene_concurrency,
        openai_concurrency=openai_concurrency,
        bedrock_concurrency=bedrock_concurrency,
        fusion_concurrency=scene_concurrency,
    )
    return publish_odd_labelset(
        scene_files=scene_files,
        capability_manifest_json=scene_plan.capability_manifest_json,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_manifest_uri=dataset_manifest_uri,
        dataset_manifest_sha256=dataset_manifest_sha256,
        datasets_bucket=datasets_bucket,
        openai_model=openai_model,
        openai_model_revision=openai_model_revision,
        bedrock_map_model_id=bedrock_map_model_id,
        bedrock_map_model_revision=bedrock_map_model_revision,
        labeler_image_digest=labeler_image_digest,
        labeler_source_revision=labeler_source_revision,
        publication_scope=publication_scope,
    )
