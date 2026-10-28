"""Flyte data-stage tasks behind wf_autoe2e.

Raw bytes come only from S3 raw snapshots; no task here holds the HF token.
Labels and packs call the existing ``generate_reasoning_labels``,
``data_processing``, and ``pack_nuplan_snapshot_reactive_dataset`` task
functions on a local tree rebuilt from the snapshot, so parsers stay as they
are.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Dict, List, Optional

from flytekit import BatchSize, PodTemplate, Resources, Secret, current_context, dynamic, map_task, task
from flytekit.types.directory import FlyteDirectory

from Platform.pipelines.autoe2e_adapters import (
    SourcePlan,
    UnitRef,
    extract_unit_tar,
    pack_contracts,
    read_source,
)
from Platform.pipelines.autoe2e_dataset import (
    AutoE2EDataset,
    DatasetSpec,
    ManifestRef,
    canonical_dataset_spec,
    partition_view_prefix,
    units_key,
    views_manifest,
)
from Platform.pipelines.autoe2e_labels import LabelIdentity, label_attempt_prefix, label_id, label_identity
from Platform.pipelines.autoe2e_lineage import UnitLineage, dataset_lineage
from Platform.pipelines.autoe2e_env import PIPELINE_ENV
from Platform.pipelines.autoe2e_store import (
    DatasetHandle,
    claim_label_set,
    commit_label_set,
    lookup_label_set,
    resolve_published_dataset,
    write_dataset_index,
    write_dataset_lineage,
    write_views_manifest,
)
from Platform.pipelines.dataset_publication import canonical_json_bytes, sha256_bytes
from Platform.pipelines.dataset_publication_tasks import (
    DATA_PREP_IMAGE,
    _copy_immutable,
    _list_objects,
    _put_immutable,
    _remote_directory_uri,
    _split_s3_uri,
)

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
_DATA_PREP_ENV = {"AUTO_E2E_DATA_PREP_IMAGE": DATA_PREP_IMAGE}
# Read at registration and baked into the task, so dataset lineage names the code.
_GIT_ENV = {"AUTO_E2E_GIT_COMMIT": os.environ.get("AUTO_E2E_GIT_COMMIT", "unknown")}
# Long Cosmos calls and packs must not be consolidated away mid-run.
_DO_NOT_DISRUPT = PodTemplate(annotations={"karpenter.sh/do-not-disrupt": "true"})


def _teacher_model() -> str:
    """The model the teacher endpoint serves, from the cosmos-teacher secret."""
    try:
        return current_context().secrets.get("cosmos-teacher", "COSMOS_TEACHER_MODEL")
    except Exception:
        return os.environ.get("COSMOS_TEACHER_MODEL", "")


@dataclass
class UnitLabel:
    unit_id: str
    label_id: str = ""
    record_sha256: str = ""
    origin: str = ""
    label_count: int = 0
    artifact_uri: str = ""


@dataclass
class PartitionView:
    partition_id: str
    prefix: str
    manifest_sha256: str
    sample_count: int
    unit_ids: List[str] = field(default_factory=list)
    reasoning_label_count: int = 0


def _execution_id() -> str:
    try:
        return current_context().execution_id.name or "local"
    except Exception:
        return "local"


def _attempt_id() -> str:
    try:
        attempt = int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))
    except ValueError:
        attempt = 0
    return f"{_execution_id()}-{attempt}"


def _s3(region: str = AWS_REGION):
    import boto3

    return boto3.client("s3", region_name=region)


def _snapshot_payload(s3, spec: DatasetSpec) -> bytes:
    bucket, key = _split_s3_uri(spec.source_manifest.uri)
    response = s3.get_object(Bucket=bucket, Key=key, VersionId=spec.source_manifest.version_id)
    try:
        return response["Body"].read()
    finally:
        response["Body"].close()


def _label_identity(spec: DatasetSpec, plan: SourcePlan, unit_id: str) -> dict:
    from data_processing.contract_versions import (
        REASONING_LABEL_POLICY_VERSION,
        UID_SCHEMA_VERSION,
    )

    reasoning = spec.reasoning
    assert reasoning is not None
    return label_identity(
        spec.dataset,
        plan.revision,
        unit_id,
        LabelIdentity(
            teacher=reasoning.teacher,
            teacher_model=reasoning.teacher_model,
            prompt_version=reasoning.prompt_version,
            label_stride=reasoning.label_stride,
            label_policy_version=REASONING_LABEL_POLICY_VERSION,
            uid_schema_version=UID_SCHEMA_VERSION,
        ),
    )


def materialize_units(s3, bucket: str, units: List[UnitRef], root: Path) -> None:
    """Rebuild the upstream tree for ``units`` from their snapshot tars."""
    import hashlib

    class _Hashing(io.RawIOBase):
        def __init__(self, body):
            self.body, self.digest = body, hashlib.sha256()

        def readable(self):
            return True

        def readinto(self, buffer):
            data = self.body.read(len(buffer))
            self.digest.update(data)
            buffer[: len(data)] = data
            return len(data)

    root.mkdir(parents=True, exist_ok=True)
    for unit in units:
        response = s3.get_object(Bucket=bucket, Key=unit.tar_key)
        stream = _Hashing(response["Body"])
        try:
            extract_unit_tar(io.BufferedReader(stream, 8 * 1024**2), unit.unit_id, root)
            while stream.read(8 * 1024**2):
                pass
        finally:
            response["Body"].close()
        if stream.digest.hexdigest() != unit.tar_sha256:
            raise ValueError(f"{unit.unit_id}: snapshot tar differs from its manifest digest")


def download_prefix(s3, uri: str, root: Path) -> None:
    bucket, prefix = _split_s3_uri(uri)
    for item in _list_objects(s3, bucket, prefix):
        target = root.joinpath(*item["relative"].split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, item["key"], str(target))


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="1", mem="4Gi"),
    limits=Resources(cpu="1", mem="4Gi"),
    retries=2,
)
def plan_dataset(spec: DatasetSpec, datasets_bucket: str) -> SourcePlan:
    """Read the pinned snapshot manifest and plan units and partitions."""
    s3 = _s3()
    payload = _snapshot_payload(s3, spec)
    plan = read_source(spec, payload)
    if spec.dataset is AutoE2EDataset.NUPLAN:
        from data_parsing.nuplan.materialization import load_nuplan_snapshot_manifest

        from Platform.pipelines.nuplan_dataset import (
            NUPLAN_FULL_TRAIN_GROUP_COUNT,
            _build_nuplan_train_pack_plan,
            _resolve_nuplan_split_sensor_plan,
        )

        manifest = load_nuplan_snapshot_manifest(payload)
        sensor_groups, database_logs = _resolve_nuplan_split_sensor_plan(
            manifest, datasets_bucket=datasets_bucket, aws_region=AWS_REGION,
            split="train", group_count=NUPLAN_FULL_TRAIN_GROUP_COUNT,
        )
        archive_sets, limits = _build_nuplan_train_pack_plan(
            manifest, sensor_groups, database_logs,
            int(plan.pack_params["total_scenario_limit"]),
        )
        group_limit = int(plan.pack_params["group_limit"])
        if group_limit:
            archive_sets, limits = archive_sets[:group_limit], limits[:group_limit]
        plan.partitions = [list(archives) for archives in archive_sets]
        plan.partition_limits = list(limits)
        # Only archives some partition reads belong in the dataset lineage.
        used = {archive for archives in plan.partitions for archive in archives}
        plan.units = [unit for unit in plan.units if unit.unit_id in used]
    return plan


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi"),
    retries=2,
)
def resolve_dataset(spec: DatasetSpec, plan: SourcePlan, datasets_bucket: str) -> DatasetHandle:
    """Reuse a verified publication of ``spec`` or reserve its version."""
    return resolve_published_dataset(
        _s3(),
        bucket=datasets_bucket,
        spec=spec,
        published_name=plan.published_name,
        expected_contracts=pack_contracts(),
        execution_id=_execution_id(),
    )


@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_DO_NOT_DISRUPT,
    requests=Resources(cpu="15", mem="64Gi", ephemeral_storage="60Gi"),
    limits=Resources(cpu="15", mem="64Gi", ephemeral_storage="60Gi"),
    secret_requests=[
        Secret(group="cosmos-teacher", key="COSMOS_TEACHER_BASE_URL",
               mount_requirement=Secret.MountType.ENV_VAR),
        Secret(group="cosmos-teacher", key="COSMOS_TEACHER_MODEL",
               mount_requirement=Secret.MountType.ENV_VAR),
    ],
    retries=2,
    environment={"_F_P_WRITE_CHUNK_SIZE": "8388608"},
)
def label_unit(
    spec: DatasetSpec,
    plan: SourcePlan,
    unit: UnitRef,
    datasets_bucket: str,
    label_workers: int = 2,
) -> UnitLabel:
    """Reuse a unit's committed labels, or label it once and commit."""
    from Platform.pipelines.workflows import Dataset, generate_reasoning_labels

    s3 = _s3()
    identity = _label_identity(spec, plan, unit.unit_id)
    record = lookup_label_set(s3, bucket=datasets_bucket, dataset=spec.dataset, identity=identity)
    if record is None:
        served = _teacher_model()
        if served != spec.reasoning.teacher_model:
            raise ValueError(
                f"teacher endpoint serves {served!r}, spec names {spec.reasoning.teacher_model!r}"
            )
        claim_label_set(
            s3, bucket=datasets_bucket, dataset=spec.dataset, identity=identity,
            execution_id=_execution_id(),
            claimed_at=datetime.now(timezone.utc).isoformat(),
        )
        raw_root = Path(tempfile.mkdtemp(prefix="autoe2e-raw-"))
        materialize_units(s3, datasets_bucket, [unit], raw_root)
        output = generate_reasoning_labels.task_function(
            raw_data=FlyteDirectory(str(raw_root)),
            dataset=Dataset[spec.dataset.name],
            source_revision=plan.revision,
            episodes=0,
            split=plan.split,
            teacher=spec.reasoning.teacher,
            prompt_version=spec.reasoning.prompt_version,
            group_ids=[unit.unit_id],
            label_stride=spec.reasoning.label_stride,
            label_workers=label_workers,
        )
        local = Path(output.path)
        meta = json.loads((local / "meta.json").read_text())
        attempt = _attempt_id()
        prefix = label_attempt_prefix(spec.dataset, label_id(identity), attempt)
        for path in sorted(p for p in local.rglob("*") if p.is_file()):
            _put_immutable(
                s3, bucket=datasets_bucket,
                key=f"{prefix}/{path.relative_to(local).as_posix()}",
                payload=path.read_bytes(), content_type="application/octet-stream",
            )
        record = commit_label_set(
            s3, bucket=datasets_bucket, dataset=spec.dataset, identity=identity,
            attempt_id=attempt, label_count=int(meta["num_records"]),
        )
    # A fact of the committed record, identical on every rerun.
    origin = "adopted" if str(record["attempt_id"]).startswith("adopted-") else "created"
    return UnitLabel(
        unit_id=unit.unit_id,
        label_id=record["label_id"],
        record_sha256=sha256_bytes(canonical_json_bytes(record, pretty=True)),
        origin=origin,
        label_count=int(record["label_count"]),
        artifact_uri=record["artifact_uri"],
    )


# Clip the OSM extract this far around every pose and route waypoint: far
# wider than the raster, so map matching keeps the connectivity of the roads
# an episode drives near.
L2D_OSM_REGION_BUFFER_M = 1000.0


def _l2d_track_points(raw_root: Path, episodes: List[int]) -> list:
    """Every (longitude, latitude) of the episodes' poses and route waypoints."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted = set(episodes)
    points = []
    seen = set()
    for path in sorted((raw_root / "data").rglob("*.parquet")):
        table = pq.read_table(
            path, columns=["episode_index", "observation.state.vehicle", "observation.state.waypoints"],
        )
        table = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(sorted(wanted))))
        if not table.num_rows:
            continue
        seen.update(int(e) for e in pc.unique(table["episode_index"]).to_pylist())
        vehicle = np.asarray(table["observation.state.vehicle"].to_pylist(), dtype=np.float64)
        points.extend(zip(vehicle[:, 4], vehicle[:, 3]))
        for route in table["observation.state.waypoints"].to_pylist():
            points.extend((float(lon), float(lat)) for lon, lat in route)
    if seen != wanted:
        raise ValueError(f"L2D data files lack episodes {sorted(wanted - seen)[:5]}")
    return points


@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_DO_NOT_DISRUPT,
    requests=Resources(cpu="4", mem="16Gi", ephemeral_storage="60Gi"),
    limits=Resources(cpu="4", mem="16Gi", ephemeral_storage="60Gi"),
    retries=2,
    # Cached by the plan (which pins the snapshot), so every pack of a
    # rebuilt dataset sees the same region bytes.
    cache=True,
    cache_version=f"l2d-osm-region-v1-{sha256_bytes(DATA_PREP_IMAGE.encode())[:16]}",
)
def build_l2d_osm_region(plan: SourcePlan, datasets_bucket: str) -> ManifestRef:
    """Clip the snapshot's OSM extract to the episodes' tracks, once per plan."""
    from data_parsing.l2d.osm_pbf_region import clip_cells, file_sha256, read_region, region_bytes

    from Platform.pipelines.l2d_snapshot import L2D_OSM_UNIT

    s3 = _s3()
    by_id = {unit.unit_id: unit for unit in plan.units}
    raw_root = Path(tempfile.mkdtemp(prefix="autoe2e-l2d-region-"))
    data_units = sorted({u for partition in plan.partitions for u in partition if u.startswith("data.")})
    materialize_units(s3, datasets_bucket, [by_id[u] for u in [L2D_OSM_UNIT, *data_units]], raw_root)
    episodes = sorted(int(e) for group in plan.partition_groups for e in group)
    points = _l2d_track_points(raw_root, episodes)
    cells = clip_cells(points, buffer_m=L2D_OSM_REGION_BUFFER_M)
    (pbf,) = sorted((raw_root / "osm").glob("*.osm.pbf"))
    nodes, ways = read_region(pbf, cells)
    payload = region_bytes(nodes, ways, provenance={
        "snapshot_manifest_sha256": plan.snapshot_manifest_sha256,
        "osm_unit_digest": by_id[L2D_OSM_UNIT].tar_digest,
        "osm_extract": pbf.name,
        "osm_extract_sha256": file_sha256(pbf),
        "episodes": episodes,
        "buffer_m": L2D_OSM_REGION_BUFFER_M,
        "cell_count": len(cells),
    })
    digest = sha256_bytes(payload)
    key = f"{plan.published_name}/osm-regions/{digest}.json"
    _put_immutable(s3, bucket=datasets_bucket, key=key, payload=payload, content_type="application/json")
    head = s3.head_object(Bucket=datasets_bucket, Key=key)
    print(f"L2D OSM region: {len(ways)} ways, {len(nodes)} nodes, {len(payload)} bytes, sha256 {digest}")
    return ManifestRef(uri=f"s3://{datasets_bucket}/{key}", version_id=str(head.get("VersionId") or "null"), sha256=digest)


def _download_region(s3, ref: ManifestRef) -> Path:
    bucket, key = _split_s3_uri(ref.uri)
    payload = s3.get_object(Bucket=bucket, Key=key, VersionId=ref.version_id)["Body"].read()
    if sha256_bytes(payload) != ref.sha256:
        raise ValueError("OSM region differs from its pinned SHA-256")
    path = Path(tempfile.mkdtemp(prefix="autoe2e-osm-")) / "region.json"
    path.write_bytes(payload)
    return path


@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_DO_NOT_DISRUPT,
    requests=Resources(cpu="15", mem="64Gi", ephemeral_storage="240Gi"),
    limits=Resources(cpu="15", mem="64Gi", ephemeral_storage="240Gi"),
    retries=2,
    environment={"_F_P_WRITE_CHUNK_SIZE": "8388608"},
    # Cached so a rerun of the same spec reuses the same pack output URIs:
    # the write-once publication of a reserved version must see identical
    # bytes, or a failed build could never be finished.
    cache=True,
    # The image is part of the key, so a packer fix in a new image never
    # reuses a pack made by the old one.
    cache_version=f"autoe2e-pack-v1-{sha256_bytes(DATA_PREP_IMAGE.encode())[:16]}",
)
def pack_partition(
    spec: DatasetSpec,
    plan: SourcePlan,
    partition_index: int,
    label_uris: Dict[str, str],
    version: str,
    datasets_bucket: str,
    osm_region: Optional[ManifestRef] = None,
) -> Annotated[FlyteDirectory, BatchSize(4)]:
    """Pack one partition from snapshot units (and their labels).

    The cache key includes the spec (with its code-bound pack_version), the
    plan, and the labels, so any change to what is packed misses the cache.
    """
    s3 = _s3()
    unit_ids = plan.partitions[partition_index]
    if spec.dataset is AutoE2EDataset.NUPLAN:
        from flytekit.types.file import FlyteFile

        from Platform.pipelines.nuplan_dataset import pack_nuplan_snapshot_reactive_dataset

        return pack_nuplan_snapshot_reactive_dataset.task_function(
            snapshot_manifest=FlyteFile(spec.source_manifest.uri),
            datasets_bucket=datasets_bucket,
            archive_ids=list(unit_ids),
            limit_total_scenarios=int(plan.partition_limits[partition_index]),
            image_size=spec.image_size,
            samples_per_shard=int(plan.pack_params["samples_per_shard"]),
            max_rejection_fraction=float(plan.pack_params["max_rejection_fraction"]),
            aws_region=AWS_REGION,
        )

    from Platform.pipelines.workflows import Dataset, data_processing

    by_id = {unit.unit_id: unit for unit in plan.units}
    raw_root = Path(tempfile.mkdtemp(prefix="autoe2e-raw-"))
    materialize_units(s3, datasets_bucket, [by_id[u] for u in unit_ids], raw_root)
    labels_dir = None
    if plan.labels:
        labels_root = Path(tempfile.mkdtemp(prefix="autoe2e-labels-"))
        for unit_id in unit_ids:
            download_prefix(s3, label_uris[unit_id], labels_root / unit_id)
        labels_dir = FlyteDirectory(str(labels_root))
    training = plan.split == "train"
    groups = plan.partition_groups[partition_index] if plan.partition_groups else unit_ids
    region = {}
    if spec.dataset is AutoE2EDataset.L2D:
        if osm_region is None:
            raise ValueError("L2D packs need the OSM region")
        region = {"osm_region_path": str(_download_region(s3, osm_region)), "osm_region_sha256": osm_region.sha256}
    return data_processing.task_function(
        raw_data=FlyteDirectory(str(raw_root)),
        dataset=Dataset[spec.dataset.name],
        source_revision=plan.revision,
        dataset_version=version,
        hz=10,
        image_size=spec.image_size,
        episodes=0,
        world_model=training,
        reasoning_labels=labels_dir,
        group_ids=list(groups),
        sample_limit=0,
        source_split=plan.split,
        data_role="training" if training else "benchmark",
        **region,
    )


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="4", mem="8Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
    retries=2,
)
def publish_partition_view(
    packed: FlyteDirectory,
    unit_ids: List[str],
    published_name: str,
    version: str,
    datasets_bucket: str,
) -> PartitionView:
    """Copy one packed partition, byte for byte, into its training view."""
    from concurrent.futures import ThreadPoolExecutor

    s3 = _s3()
    source_bucket, source_prefix = _split_s3_uri(_remote_directory_uri(packed))
    objects = _list_objects(s3, source_bucket, source_prefix)
    manifest_object = next((o for o in objects if o["relative"] == "manifest.json"), None)
    if manifest_object is None:
        raise ValueError(f"packed partition has no manifest.json: {source_prefix}")
    manifest_payload = s3.get_object(Bucket=source_bucket, Key=manifest_object["key"])["Body"].read()
    manifest = json.loads(manifest_payload)
    prefix = partition_view_prefix(published_name, version, str(manifest["partition_id"]))
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(
            lambda source: _copy_immutable(
                s3, source, destination_bucket=datasets_bucket,
                destination_key=f"{prefix}/{source['relative']}",
            ),
            objects,
        ))
    copied = s3.get_object(Bucket=datasets_bucket, Key=f"{prefix}/manifest.json")["Body"].read()
    if copied != manifest_payload:
        raise RuntimeError(f"partition view manifest differs after copy: {prefix}")
    return PartitionView(
        partition_id=str(manifest["partition_id"]),
        prefix=prefix,
        manifest_sha256=sha256_bytes(manifest_payload),
        sample_count=int(manifest.get("total_samples", 0)),
        unit_ids=list(unit_ids),
        reasoning_label_count=int(manifest.get("reasoning_label_count", 0)),
    )


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="2", mem="4Gi"),
    retries=2,
    environment=_GIT_ENV,
)
def commit_dataset(
    spec: DatasetSpec,
    plan: SourcePlan,
    version: str,
    reserved_by: str,
    labels: List[UnitLabel],
    views: List[PartitionView],
    console_manifest_key: str,
    console_manifest_sha256: str,
    datasets_bucket: str,
    osm_region: Optional[ManifestRef] = None,
) -> DatasetHandle:
    """Write views manifest, lineage, and finally the by-id index."""
    from Platform.pipelines.autoe2e_dataset import compute_dataset_id

    s3 = _s3()
    dataset_id = compute_dataset_id(spec)
    contracts = pack_contracts()
    views_record = views_manifest(
        dataset_id=dataset_id,
        published_name=plan.published_name,
        version=version,
        contracts=contracts,
        image_size=spec.image_size,
        partitions=[
            {"partition_id": v.partition_id, "prefix": v.prefix,
             "manifest_sha256": v.manifest_sha256, "sample_count": v.sample_count}
            for v in views
        ],
    )
    views_sha256 = write_views_manifest(s3, bucket=datasets_bucket, views=views_record)
    rows = unit_rows(plan, labels, views)
    from Platform.pipelines.autoe2e_dataset import views_manifest_key

    record, units_payload = dataset_lineage(
        dataset_id=dataset_id,
        spec=canonical_dataset_spec(spec),
        published={"name": plan.published_name, "version": version},
        raw={
            "upstream": plan.upstream,
            "revision": plan.revision,
            "snapshot_manifest_uri": spec.source_manifest.uri,
            "snapshot_manifest_sha256": plan.snapshot_manifest_sha256,
        },
        labels=None if not plan.labels else {
            "teacher": spec.reasoning.teacher,
            "teacher_model": spec.reasoning.teacher_model,
            "prompt_version": spec.reasoning.prompt_version,
            "label_stride": spec.reasoning.label_stride,
        },
        units=rows,
        units_uri=f"s3://{datasets_bucket}/{units_key(plan.published_name, version)}",
        views={
            "key": views_manifest_key(plan.published_name, version),
            "sha256": views_sha256,
            "partition_ids": [v.partition_id for v in views],
        },
        console=None if not console_manifest_key else {
            "manifest_key": console_manifest_key,
            "manifest_sha256": console_manifest_sha256,
        },
        pack={
            "contracts": contracts,
            "image_digest": DATA_PREP_IMAGE,
            "git_commit": os.environ.get("AUTO_E2E_GIT_COMMIT", "unknown"),
            **({} if osm_region is None else {"osm_region": {
                "uri": osm_region.uri, "version_id": osm_region.version_id, "sha256": osm_region.sha256,
            }}),
        },
        # Only values that a rerun reproduces: the write-once lineage must come
        # out byte-identical when a failed commit is retried.
        build={"flyte_execution_id": reserved_by},
        packed_reasoning_labels=sum(v.reasoning_label_count for v in views),
    )
    _, lineage_sha256 = write_dataset_lineage(
        s3, bucket=datasets_bucket, published_name=plan.published_name, version=version,
        record=record, units_payload=units_payload,
        units_key_name=units_key(plan.published_name, version),
    )
    write_dataset_index(
        s3, bucket=datasets_bucket, spec=spec, published_name=plan.published_name,
        version=version, views_sha256=views_sha256, lineage_sha256=lineage_sha256,
    )
    return resolve_published_dataset(
        s3, bucket=datasets_bucket, spec=spec, published_name=plan.published_name,
        expected_contracts=contracts, execution_id=_execution_id(),
    )


def unit_rows(plan: SourcePlan, labels: List[UnitLabel], views: List[PartitionView]) -> List[UnitLineage]:
    """Join snapshot units, their labels, and the partition that packed them."""
    by_label = {label.unit_id: label for label in labels}
    partition_of = {}
    samples_of = {}
    for view in views:
        for unit_id in view.unit_ids:
            partition_of[unit_id] = view.partition_id
            samples_of[unit_id] = view.sample_count
    rows = []
    for unit in plan.units:
        label = by_label.get(unit.unit_id)
        if plan.labels and label is None:
            raise ValueError(f"{unit.unit_id}: labelled dataset is missing its label set")
        # nuPlan archives and L2D files and meta feed several partitions.
        shared = plan.dataset in (AutoE2EDataset.NUPLAN, AutoE2EDataset.L2D)
        if not shared and unit.unit_id not in partition_of:
            raise ValueError(f"{unit.unit_id}: no partition view packed this unit")
        rows.append(UnitLineage(
            unit_id=unit.unit_id,
            partition_id="*" if shared else partition_of[unit.unit_id],
            raw_tar_key=unit.tar_key,
            raw_tar_digest=unit.tar_digest,
            raw_index_sha256=unit.index_sha256 or None,
            label_id=label.label_id if label else None,
            label_record_sha256=label.record_sha256 if label else None,
            label_origin=label.origin if label else None,
            sample_count=0 if shared else samples_of[unit.unit_id],
            label_count=label.label_count if label else 0,
        ))
    return rows


@task(container_image=DATA_PREP_IMAGE, requests=Resources(cpu="1", mem="1Gi"))
def committed_label_uris(labels: List[UnitLabel]) -> Dict[str, str]:
    """Committed label prefixes per unit: the only label input a pack sees.

    A rerun reuses the same committed attempt, so this mapping, and the pack
    cache key built from it, is identical whether labels were just created
    or reused.
    """
    return {label.unit_id: label.artifact_uri for label in labels}


@dynamic(container_image=DATA_PREP_IMAGE, environment={**_DATA_PREP_ENV, **PIPELINE_ENV})
def build_dataset(
    spec: DatasetSpec,
    plan: SourcePlan,
    handle: DatasetHandle,
    datasets_bucket: str,
    dynamo_table: str,
    label_concurrency: int,
    pack_concurrency: int,
) -> DatasetHandle:
    """Label missing units, pack every partition, publish, and commit."""
    import functools

    from Platform.pipelines.dataset_publication_tasks import (
        finalize_dataset_publication,
        publish_dataset_partition,
    )

    labels: List[UnitLabel] = []
    label_uris: Dict[str, str] = {}
    # map_task turns every unbound defaulted input into a required list, so
    # every scalar is bound here explicitly.
    if plan.labels:
        labels = map_task(
            functools.partial(label_unit, spec=spec, plan=plan, datasets_bucket=datasets_bucket, label_workers=2),
            concurrency=label_concurrency,
        )(unit=plan.units)
        label_uris = committed_label_uris(labels=labels)
    osm_region = None
    if plan.dataset is AutoE2EDataset.L2D:
        osm_region = build_l2d_osm_region(plan=plan, datasets_bucket=datasets_bucket)
    packed = map_task(
        functools.partial(
            pack_partition, spec=spec, plan=plan, label_uris=label_uris,
            version=handle.version, datasets_bucket=datasets_bucket, osm_region=osm_region,
        ),
        concurrency=pack_concurrency,
    )(partition_index=list(range(len(plan.partitions))))
    if plan.dataset is AutoE2EDataset.NUPLAN:
        from Platform.pipelines.nuplan_dataset import NUPLAN_FULL_PACK_EPHEMERAL_STORAGE

        # A sensor group pack downloads and extracts whole 157-250 GB archives.
        packed = packed.with_overrides(
            requests=Resources(cpu="16", mem="64Gi", ephemeral_storage=NUPLAN_FULL_PACK_EPHEMERAL_STORAGE),
            limits=Resources(cpu="16", mem="64Gi", ephemeral_storage=NUPLAN_FULL_PACK_EPHEMERAL_STORAGE),
        )
    views = map_task(
        functools.partial(
            publish_partition_view, published_name=plan.published_name,
            version=handle.version, datasets_bucket=datasets_bucket,
        ),
        concurrency=pack_concurrency,
    )(packed=packed, unit_ids=plan.partitions)
    console_key, console_sha = "", ""
    if plan.console:
        results = map_task(
            functools.partial(
                publish_dataset_partition, published_dataset=plan.published_name,
                dataset_version=handle.version, datasets_bucket=datasets_bucket,
                aws_region=AWS_REGION, copy_workers=16,
            ),
            concurrency=pack_concurrency,
        )(shard_dir=packed)
        publication = finalize_dataset_publication(
            partition_results=results, published_dataset=plan.published_name,
            dataset_version=handle.version, datasets_bucket=datasets_bucket,
            dynamo_table=dynamo_table, aws_region=AWS_REGION,
        )
        console_key, console_sha = publication.manifest_key, publication.manifest_sha256
    return commit_dataset(
        spec=spec, plan=plan, version=handle.version, reserved_by=handle.reserved_by,
        labels=labels, views=views,
        console_manifest_key=console_key, console_manifest_sha256=console_sha,
        datasets_bucket=datasets_bucket, osm_region=osm_region,
    )
