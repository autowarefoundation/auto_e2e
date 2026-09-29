"""Immutable shard planning and rank-local staging for Reactive DDP."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS
from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_TAXONOMY_VERSION,
)
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_FRAME_OFFSETS,
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_INDEX,
)
from training.dataset_policy import (
    KITSCENES_DATASET_NAME,
    L2D_DATASET_NAME,
    NUPLAN_DATASET_NAME,
)
from training.reactive_multitask import ReactiveTrainingStage


@dataclass(frozen=True)
class ReactiveShardReference:
    """One immutable tar file in a packed Reactive dataset."""

    source_uri: str
    manifest_sha256: str
    partition_id: str
    split_group_uid: str | None
    shard_name: str
    shard_sha256: str
    sample_count: int

    @property
    def identity(self) -> str:
        return f"{self.source_uri.rstrip('/')}/{self.shard_name}"


@dataclass(frozen=True)
class ReactiveDatasetPlan:
    """Validated corpus identity shared by every DDP rank."""

    dataset: str
    dataset_manifest_sha256: str
    physical_camera_order: tuple[str, ...]
    camera_slots: tuple[str, ...]
    num_views: int
    total_samples: int
    shards: tuple[ReactiveShardReference, ...]
    source_revision: str
    dataset_version: str
    packed_contract_digest: str
    partition_count: int
    empty_partition_count: int
    split_group_uids: tuple[str, ...]


class RestartingIterator:
    """Repeat a finite loader while exposing restart evidence."""

    def __init__(self, source: Iterable[Any]) -> None:
        self._source = source
        self._iterator: Iterator[Any] | None = None
        self.restarts = 0

    def __iter__(self) -> RestartingIterator:
        return self

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self._source)
        try:
            return next(self._iterator)
        except StopIteration:
            self.restarts += 1
            self._iterator = iter(self._source)
            try:
                return next(self._iterator)
            except StopIteration as error:
                raise ValueError(
                    "rank-local Reactive training loader yielded no batches"
                ) from error


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _local_source_path(source_uri: str) -> Path | None:
    parsed = urlparse(source_uri)
    if parsed.scheme == "":
        return Path(source_uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError("file URI must refer to the local host")
        return Path(unquote(parsed.path))
    return None


def _s3_location(source_uri: str, relative_path: str) -> tuple[str, str]:
    parsed = urlparse(source_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(
            "distributed dataset source must be a local path or S3 URI"
        )
    prefix = parsed.path.lstrip("/").rstrip("/")
    key = "/".join(part for part in (prefix, relative_path) if part)
    return parsed.netloc, key


def read_source_file(source_uri: str, relative_path: str) -> bytes:
    """Read one small source file without materializing a FlyteDirectory."""
    local = _local_source_path(source_uri)
    if local is not None:
        return (local / relative_path).read_bytes()
    import boto3

    bucket, key = _s3_location(source_uri, relative_path)
    response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def _expected_dataset(stage: ReactiveTrainingStage) -> str:
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        return NUPLAN_DATASET_NAME
    if stage is ReactiveTrainingStage.KITSCENES_FINETUNE:
        return KITSCENES_DATASET_NAME
    return L2D_DATASET_NAME


def _validate_reactive_manifest(
    manifest: Mapping[str, Any],
    *,
    stage: ReactiveTrainingStage,
    source_uri: str,
    allow_mapless_kitscenes_evaluation: bool = False,
) -> None:
    expected_dataset = _expected_dataset(stage)
    if manifest.get("dataset") != expected_dataset:
        raise ValueError(
            f"{stage.value} requires dataset={expected_dataset}, "
            f"got {manifest.get('dataset')!r} from {source_uri}"
        )
    total_samples = int(manifest.get("total_samples", -1))
    if total_samples < 0:
        raise ValueError("Reactive DDP manifest has invalid total_samples")
    if total_samples == 0:
        if (
            manifest.get("shard_names") not in ([], None)
            or int(manifest.get("shards", 0)) != 0
        ):
            raise ValueError(
                f"empty Reactive partition contains shards in {source_uri}"
            )
        return
    if (
        allow_mapless_kitscenes_evaluation
        and stage is not ReactiveTrainingStage.KITSCENES_FINETUNE
    ):
        raise ValueError(
            "mapless evaluation is supported only for KITScenes"
        )
    required_flags = {"has_trajectory_xy": True}
    if allow_mapless_kitscenes_evaluation:
        required_flags.update({
            "has_gps": False,
            "has_map": False,
            "has_navigation": False,
            "has_reactive_navigation": False,
            "has_route_reconstruction": False,
        })
        if manifest.get("input_track") != "camera_only_missing_map_route":
            raise ValueError(
                "mapless KITScenes evaluation has the wrong input track"
            )
    else:
        required_flags.update({
            "has_reactive_navigation": True,
            "has_route_reconstruction": True,
        })
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        required_flags["has_bev_segmentation"] = True
    mismatches = {
        name: manifest.get(name)
        for name, expected in required_flags.items()
        if manifest.get(name) is not expected
    }
    if mismatches:
        raise ValueError(
            f"Reactive target coverage is incomplete in {source_uri}: "
            f"{mismatches}"
        )
    if not allow_mapless_kitscenes_evaluation:
        if (
            manifest.get("navigation_geometry")
            != AUTOE2E_NAVIGATION_GEOMETRY.contract()
        ):
            raise ValueError(
                f"navigation geometry differs in {source_uri}"
            )
    if int(manifest.get("map_context_channels", 0)) != 14:
        raise ValueError("Reactive DDP requires 14 map channels")
    if int(manifest.get("route_channels", 0)) != 2:
        raise ValueError("Reactive DDP requires two route channels")
    if int(manifest.get("num_views", 0)) <= 0:
        raise ValueError("Reactive DDP manifest has no camera views")
    camera_order = manifest.get("camera_order")
    camera_slots = manifest.get("camera_slots")
    if (
        not isinstance(camera_order, list)
        or camera_slots != list(CANONICAL_SIX_CAMERA_SLOTS)
        or len(camera_order) != len(CANONICAL_SIX_CAMERA_SLOTS)
        or any(
            not isinstance(camera, str) or not camera
            for camera in camera_order
        )
        or len(set(camera_order)) != len(camera_order)
        or int(manifest["num_views"]) != len(camera_order)
    ):
        raise ValueError(
            "Reactive DDP requires the canonical six-camera slot contract"
        )
    if int(manifest.get("image_size", 0)) != REACTIVE_CAMERA_IMAGE_SIZE:
        raise ValueError(
            "Reactive DDP camera image size differs from model contract"
        )
    if stage in {
        ReactiveTrainingStage.NUPLAN_FULL,
        ReactiveTrainingStage.KITSCENES_FINETUNE,
    }:
        if (
            manifest.get("front_camera_index")
            != REACTIVE_FRONT_CAMERA_INDEX
            or manifest.get("front_camera_image_size")
            != REACTIVE_FRONT_CAMERA_IMAGE_SIZE
        ):
            raise ValueError(
                "Reactive front camera dimensions differ from model contract"
            )
        if (
            manifest.get("temporal_frame_offsets")
            != list(REACTIVE_BEVFORMER_FRAME_OFFSETS)
            or manifest.get("temporal_frame_interval_us")
            != REACTIVE_BEVFORMER_FRAME_INTERVAL_US
        ):
            raise ValueError(
                "Reactive temporal camera history differs from T8 contract"
            )
    if stage is ReactiveTrainingStage.KITSCENES_FINETUNE:
        archive_name = manifest.get("frame_pool_archive")
        archive_count = int(manifest.get("frame_pool_frame_count", 0))
        if (
            archive_name != "frame_pool.tar"
            or manifest.get("frame_pool_archive_schema")
            != "frame_pool_archive_v1"
            or archive_count <= 0
        ):
            raise ValueError(
                "KITScenes T8 training requires an immutable frame-pool archive"
            )
        _validate_sha256(
            manifest.get("frame_pool_archive_sha256"),
            field="frame-pool archive digest",
        )
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        if (
            manifest.get("bev_taxonomy_version")
            != BEV_SEGMENTATION_TAXONOMY_VERSION
        ):
            raise ValueError(
                "Stage A requires the current BEV taxonomy"
            )
        if int(manifest.get("bev_statistics_count", 0)) != int(
            manifest.get("total_samples", 0)
        ):
            raise ValueError(
                "Stage A requires BEV statistics for every sample"
            )


def build_reactive_dataset_plan(
    source_uris: Sequence[str],
    *,
    stage: ReactiveTrainingStage,
    allow_mapless_kitscenes_evaluation: bool = False,
) -> ReactiveDatasetPlan:
    """Validate manifests and return a deterministic global shard inventory."""
    normalized_sources = tuple(
        sorted(source_uri.rstrip("/") for source_uri in source_uris)
    )
    if not normalized_sources:
        raise ValueError("at least one Reactive shard source is required")
    if len(set(normalized_sources)) != len(normalized_sources):
        raise ValueError("Reactive shard sources contain duplicates")

    references: list[ReactiveShardReference] = []
    manifest_identities: list[dict[str, Any]] = []
    physical_camera_orders: set[tuple[str, ...]] = set()
    camera_slot_orders: set[tuple[str, ...]] = set()
    view_counts: set[int] = set()
    source_revisions: set[str] = set()
    dataset_versions: set[str] = set()
    packed_contract_digests: set[str] = set()
    partition_ids: set[str] = set()
    split_group_uids: set[str] = set()
    empty_partition_count = 0
    for source_uri in normalized_sources:
        manifest_bytes = read_source_file(source_uri, "manifest.json")
        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid Reactive manifest at {source_uri}"
            ) from error
        if not isinstance(manifest, Mapping):
            raise ValueError(
                f"Reactive manifest must be an object at {source_uri}"
            )
        _validate_reactive_manifest(
            manifest,
            stage=stage,
            source_uri=source_uri,
            allow_mapless_kitscenes_evaluation=(
                allow_mapless_kitscenes_evaluation
            ),
        )
        source_revision = str(manifest.get("source_revision") or "")
        dataset_version = str(manifest.get("dataset_version") or "")
        contracts = manifest.get("contracts")
        if not source_revision or not dataset_version or not isinstance(
            contracts,
            Mapping,
        ):
            raise ValueError(
                f"Reactive manifest provenance is incomplete at {source_uri}"
            )
        source_revisions.add(source_revision)
        dataset_versions.add(dataset_version)
        packed_contract_digests.add(
            _sha256_bytes(_canonical_json_bytes(contracts))
        )
        shard_names = manifest.get("shard_names")
        shard_counts = manifest.get("shard_sample_counts")
        shard_hashes = manifest.get("shard_sha256")
        if (
            not isinstance(shard_names, list)
            or not isinstance(shard_counts, Mapping)
            or not isinstance(shard_hashes, Mapping)
        ):
            raise ValueError(
                "distributed Reactive training requires shard_names, "
                "shard_sample_counts, and shard_sha256"
            )
        total_samples = int(manifest.get("total_samples", 0))
        if total_samples == 0:
            empty_partition_count += 1
        elif not shard_names or len(set(shard_names)) != len(shard_names):
            raise ValueError(
                f"Reactive manifest has invalid shard names at {source_uri}"
            )
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        partition_id = str(manifest.get("partition_id") or "")
        if not partition_id or partition_id in partition_ids:
            raise ValueError(
                "Reactive partition IDs must be non-empty and unique"
            )
        partition_ids.add(partition_id)
        manifest_split_groups = manifest.get("split_group_uids")
        split_group_uid: str | None = None
        if stage is ReactiveTrainingStage.KITSCENES_FINETUNE:
            if (
                not isinstance(manifest_split_groups, list)
                or len(manifest_split_groups) != 1
                or not isinstance(manifest_split_groups[0], str)
                or not manifest_split_groups[0]
                or manifest_split_groups[0] in split_group_uids
            ):
                raise ValueError(
                    "KITScenes partitions require one unique split group"
                )
            split_group_uid = manifest_split_groups[0]
            split_group_uids.add(split_group_uid)
        counted_samples = 0
        for shard_name_value in shard_names:
            shard_name = str(shard_name_value)
            if (
                Path(shard_name).name != shard_name
                or not shard_name.endswith(".tar")
            ):
                raise ValueError(
                    f"invalid tar shard name {shard_name!r}"
                )
            sample_count = int(shard_counts.get(shard_name, 0))
            if sample_count <= 0:
                raise ValueError(
                    f"tar shard {shard_name} has no samples"
                )
            counted_samples += sample_count
            references.append(
                ReactiveShardReference(
                    source_uri=source_uri,
                    manifest_sha256=manifest_sha256,
                    partition_id=partition_id,
                    split_group_uid=split_group_uid,
                    shard_name=shard_name,
                    shard_sha256=_validate_sha256(
                        shard_hashes.get(shard_name),
                        field=f"{shard_name} digest",
                    ),
                    sample_count=sample_count,
                )
            )
        if counted_samples != total_samples:
            raise ValueError(
                "per-shard sample counts differ from total_samples in "
                f"{source_uri}: {counted_samples} != {total_samples}"
            )
        if total_samples:
            num_views = int(manifest["num_views"])
            view_counts.add(num_views)
            physical_camera_orders.add(tuple(
                str(camera) for camera in manifest["camera_order"]
            ))
            camera_slot_orders.add(tuple(
                str(slot) for slot in manifest["camera_slots"]
            ))
        manifest_identities.append({
            "camera_order": manifest["camera_order"],
            "camera_slots": manifest["camera_slots"],
            "dataset": manifest["dataset"],
            "manifest_sha256": manifest_sha256,
            "partition_id": partition_id,
            "source_revision": manifest.get("source_revision"),
            "source_uri": source_uri,
            "total_samples": total_samples,
        })

    all_empty_mapless = (
        allow_mapless_kitscenes_evaluation
        and not references
        and empty_partition_count == len(normalized_sources)
    )
    if all_empty_mapless:
        physical_camera_order: tuple[str, ...] = ()
        camera_slots = tuple(CANONICAL_SIX_CAMERA_SLOTS)
        num_views = 0
    else:
        if len(view_counts) != 1:
            raise ValueError(
                "Reactive DDP cannot mix camera counts: "
                f"{sorted(view_counts)}"
            )
        if len(physical_camera_orders) != 1:
            raise ValueError(
                "Reactive DDP cannot mix physical camera orders"
            )
        if len(camera_slot_orders) != 1:
            raise ValueError(
                "Reactive DDP cannot mix semantic camera slots"
            )
        physical_camera_order = next(iter(physical_camera_orders))
        camera_slots = next(iter(camera_slot_orders))
        num_views = next(iter(view_counts))
    if len(source_revisions) != 1:
        raise ValueError("Reactive DDP cannot mix source revisions")
    if len(dataset_versions) != 1:
        raise ValueError("Reactive DDP cannot mix dataset versions")
    if len(packed_contract_digests) != 1:
        raise ValueError("Reactive DDP cannot mix packed contracts")
    references.sort(
        key=lambda item: (
            item.source_uri,
            item.partition_id,
            item.shard_name,
        )
    )
    dataset_manifest_sha256 = _sha256_bytes(
        _canonical_json_bytes(manifest_identities)
    )
    return ReactiveDatasetPlan(
        dataset=_expected_dataset(stage),
        dataset_manifest_sha256=dataset_manifest_sha256,
        physical_camera_order=physical_camera_order,
        camera_slots=camera_slots,
        num_views=num_views,
        total_samples=sum(item.sample_count for item in references),
        shards=tuple(references),
        source_revision=next(iter(source_revisions)),
        dataset_version=next(iter(dataset_versions)),
        packed_contract_digest=next(iter(packed_contract_digests)),
        partition_count=len(normalized_sources),
        empty_partition_count=empty_partition_count,
        split_group_uids=tuple(sorted(split_group_uids)),
    )


def assign_reactive_shards(
    shards: Sequence[ReactiveShardReference],
    *,
    world_size: int,
    validation_group_uids: Sequence[str] | None = None,
) -> tuple[tuple[ReactiveShardReference, ...], ...]:
    """Balance complete tar files with deterministic LPT assignment."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if len(shards) < world_size:
        raise ValueError(
            "distributed Reactive training needs at least one tar shard "
            f"per rank: shards={len(shards)} world_size={world_size}"
        )
    identities = [shard.identity for shard in shards]
    if len(set(identities)) != len(identities):
        raise ValueError("Reactive shard inventory contains duplicates")

    assignments: list[list[ReactiveShardReference]] = [
        [] for _ in range(world_size)
    ]
    validation_groups = frozenset(
        str(value) for value in (validation_group_uids or ())
    )
    if validation_group_uids is not None and (
        not validation_groups
        or len(validation_groups) != len(validation_group_uids)
        or any(not value for value in validation_groups)
        or any(shard.split_group_uid is None for shard in shards)
    ):
        raise ValueError("validation-aware assignment has invalid groups")

    def ordered(values):
        return sorted(
            values,
            key=lambda shard: (
                -shard.sample_count,
                shard.source_uri,
                shard.partition_id,
                shard.shard_name,
            ),
        )

    def place(values) -> None:
        totals = [0] * world_size
        for shard in ordered(values):
            rank = min(
                range(world_size),
                key=lambda item: (totals[item], item),
            )
            assignments[rank].append(shard)
            totals[rank] += shard.sample_count

    if validation_group_uids is None:
        place(shards)
    else:
        train_shards = [
            shard
            for shard in shards
            if shard.split_group_uid not in validation_groups
        ]
        validation_shards = [
            shard
            for shard in shards
            if shard.split_group_uid in validation_groups
        ]
        if {
            shard.split_group_uid for shard in validation_shards
        } != validation_groups:
            raise ValueError(
                "validation-aware assignment is missing frozen groups"
            )
        place(train_shards)
        place(validation_shards)
    return tuple(
        tuple(
            sorted(
                rank_shards,
                key=lambda shard: (
                    shard.source_uri,
                    shard.partition_id,
                    shard.shard_name,
                ),
            )
        )
        for rank_shards in assignments
    )


def reactive_assignment_sha256(
    assignments: Sequence[Sequence[ReactiveShardReference]],
) -> str:
    payload = [
        {
            "rank": rank,
            "sample_count": sum(
                shard.sample_count for shard in rank_shards
            ),
            "shards": [
                {
                    "identity": shard.identity,
                    "sample_count": shard.sample_count,
                    "sha256": shard.shard_sha256,
                }
                for shard in rank_shards
            ],
        }
        for rank, rank_shards in enumerate(assignments)
    ]
    return _sha256_bytes(_canonical_json_bytes(payload))


def optimizer_steps_per_epoch(
    *,
    total_samples: int,
    val_fraction: float,
    world_size: int,
    per_rank_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    if (
        world_size <= 0
        or per_rank_batch_size <= 0
        or gradient_accumulation_steps <= 0
    ):
        raise ValueError("batch and world-size values must be positive")
    estimated_train_samples = max(
        1,
        math.ceil(total_samples * (1.0 - val_fraction)),
    )
    global_effective_batch = (
        world_size
        * per_rank_batch_size
        * gradient_accumulation_steps
    )
    return max(1, math.ceil(
        estimated_train_samples / global_effective_batch
    ))


def _copy_or_download_shard(
    shard: ReactiveShardReference,
    destination: Path,
) -> None:
    _copy_or_download_source_file(
        shard.source_uri,
        shard.shard_name,
        destination,
        expected_sha256=shard.shard_sha256,
    )


def _copy_or_download_source_file(
    source_uri: str,
    relative_path: str,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    local = _local_source_path(source_uri)
    if local is not None:
        source = local / relative_path
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
    else:
        import boto3

        bucket, key = _s3_location(source_uri, relative_path)
        boto3.client("s3").download_file(
            bucket,
            key,
            str(destination),
        )
    actual_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        destination.unlink(missing_ok=True)
        raise ValueError(
            "source file digest mismatch for "
            f"{source_uri.rstrip('/')}/{relative_path}"
        )


def stage_rank_reactive_shards(
    rank_shards: Sequence[ReactiveShardReference],
    *,
    cache_root: str | Path,
) -> tuple[str, ...]:
    """Materialize only one rank's immutable tar files."""
    if not rank_shards:
        raise ValueError("rank has no assigned Reactive tar shards")
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, list[ReactiveShardReference]] = {}
    for shard in rank_shards:
        by_source.setdefault(shard.source_uri, []).append(shard)

    local_directories: list[str] = []
    for source_uri, shards in sorted(by_source.items()):
        source_digest = hashlib.sha256(
            source_uri.encode("utf-8")
        ).hexdigest()[:16]
        destination = root / source_digest
        destination.mkdir(parents=True, exist_ok=True)
        manifest_bytes = read_source_file(source_uri, "manifest.json")
        expected_manifest = {shard.manifest_sha256 for shard in shards}
        if expected_manifest != {_sha256_bytes(manifest_bytes)}:
            raise ValueError(
                f"manifest changed while staging {source_uri}"
            )
        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid Reactive manifest at {source_uri}"
            ) from error
        if not isinstance(manifest, Mapping):
            raise ValueError(
                f"Reactive manifest must be an object at {source_uri}"
            )
        manifest_path = destination / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        archive_name = manifest.get("frame_pool_archive")
        if archive_name is not None:
            if archive_name != "frame_pool.tar":
                raise ValueError(
                    f"invalid frame-pool archive name at {source_uri}"
                )
            archive_sha256 = _validate_sha256(
                manifest.get("frame_pool_archive_sha256"),
                field="frame-pool archive digest",
            )
            archive_target = destination / archive_name
            if archive_target.is_file():
                actual = hashlib.sha256(
                    archive_target.read_bytes()
                ).hexdigest()
                if actual != archive_sha256:
                    archive_target.unlink()
            if not archive_target.is_file():
                _copy_or_download_source_file(
                    source_uri,
                    archive_name,
                    archive_target,
                    expected_sha256=archive_sha256,
                )
        for shard in sorted(shards, key=lambda item: item.shard_name):
            target = destination / shard.shard_name
            if target.is_file():
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
                if actual == shard.shard_sha256:
                    continue
                target.unlink()
            _copy_or_download_shard(shard, target)
        (destination / "rank_shards.json").write_text(
            json.dumps(
                {
                    "schema_version": "reactive_rank_shards_v1",
                    "manifest_sha256": _sha256_bytes(manifest_bytes),
                    "shard_names": sorted(
                        shard.shard_name for shard in shards
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="ascii",
        )
        local_directories.append(str(destination))
    return tuple(local_directories)
