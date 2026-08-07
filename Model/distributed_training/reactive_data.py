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

from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
from training.dataset_policy import (
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
    num_views: int
    total_samples: int
    shards: tuple[ReactiveShardReference, ...]


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
    return L2D_DATASET_NAME


def _validate_reactive_manifest(
    manifest: Mapping[str, Any],
    *,
    stage: ReactiveTrainingStage,
    source_uri: str,
) -> None:
    expected_dataset = _expected_dataset(stage)
    if manifest.get("dataset") != expected_dataset:
        raise ValueError(
            f"{stage.value} requires dataset={expected_dataset}, "
            f"got {manifest.get('dataset')!r} from {source_uri}"
        )
    required_flags = {
        "has_reactive_navigation": True,
        "has_route_reconstruction": True,
        "has_trajectory_xy": True,
    }
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


def build_reactive_dataset_plan(
    source_uris: Sequence[str],
    *,
    stage: ReactiveTrainingStage,
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
    view_counts: set[int] = set()
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
        if not shard_names or len(set(shard_names)) != len(shard_names):
            raise ValueError(
                f"Reactive manifest has invalid shard names at {source_uri}"
            )
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        partition_id = str(manifest.get("partition_id") or "")
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
                    shard_name=shard_name,
                    shard_sha256=_validate_sha256(
                        shard_hashes.get(shard_name),
                        field=f"{shard_name} digest",
                    ),
                    sample_count=sample_count,
                )
            )
        total_samples = int(manifest.get("total_samples", 0))
        if counted_samples != total_samples:
            raise ValueError(
                "per-shard sample counts differ from total_samples in "
                f"{source_uri}: {counted_samples} != {total_samples}"
            )
        num_views = int(manifest["num_views"])
        view_counts.add(num_views)
        manifest_identities.append({
            "dataset": manifest["dataset"],
            "manifest_sha256": manifest_sha256,
            "partition_id": partition_id,
            "source_revision": manifest.get("source_revision"),
            "source_uri": source_uri,
            "total_samples": total_samples,
        })

    if len(view_counts) != 1:
        raise ValueError(
            f"Reactive DDP cannot mix camera counts: {sorted(view_counts)}"
        )
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
        num_views=next(iter(view_counts)),
        total_samples=sum(item.sample_count for item in references),
        shards=tuple(references),
    )


def assign_reactive_shards(
    shards: Sequence[ReactiveShardReference],
    *,
    world_size: int,
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
    totals = [0] * world_size
    ordered = sorted(
        shards,
        key=lambda shard: (
            -shard.sample_count,
            shard.source_uri,
            shard.partition_id,
            shard.shard_name,
        ),
    )
    for shard in ordered:
        rank = min(range(world_size), key=lambda item: (totals[item], item))
        assignments[rank].append(shard)
        totals[rank] += shard.sample_count
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
    local = _local_source_path(shard.source_uri)
    if local is not None:
        source = local / shard.shard_name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
    else:
        import boto3

        bucket, key = _s3_location(
            shard.source_uri,
            shard.shard_name,
        )
        boto3.client("s3").download_file(
            bucket,
            key,
            str(destination),
        )
    actual_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual_sha256 != shard.shard_sha256:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"tar shard digest mismatch for {shard.identity}"
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
        manifest_path = destination / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        for shard in sorted(shards, key=lambda item: item.shard_name):
            target = destination / shard.shard_name
            if target.is_file():
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
                if actual == shard.shard_sha256:
                    continue
                target.unlink()
            _copy_or_download_shard(shard, target)
        local_directories.append(str(destination))
    return tuple(local_directories)
