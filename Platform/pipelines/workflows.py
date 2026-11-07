"""Data tasks of the earlier per-stage pipeline, still used by wf_autoe2e.

``autoe2e_tasks.py`` runs ``data_processing`` (pre-extract frames into
WebDataset shards) and ``generate_reasoning_labels`` (offline teacher labels)
for KITScenes and L2D. The other tasks ingest the raw sources, fan these
stages out per partition and audit the KITScenes inputs.
"""
import enum
import functools
import time
from pathlib import Path
from flytekit import (
    task, dynamic, map_task, Resources, Secret, BatchSize,
)
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from typing import Annotated, Any, List, Optional

from data_processing.contract_versions import (
    GEOMETRY_VERSION as _GEOM_V,
    PARSER_VERSION as _PARSER_V,
    REASONING_LABEL_POLICY_VERSION as _LABEL_POLICY_V,
    SHARD_SCHEMA_VERSION as _SHARD_V,
    UID_SCHEMA_VERSION as _UID_V,
)
from data_processing.source_revisions import L2D_DATA_REVISION
from data_parsing.kit_scenes.temporal_contract import (
    kitscenes_temporal_contract,
)
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_FRAME_OFFSETS,
    REACTIVE_BEVFORMER_HISTORY_FRAMES,
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_INDEX,
)

import os as _os

ECR_PREFIX = _os.environ.get("ECR_PREFIX", "381491877296.dkr.ecr.us-west-2.amazonaws.com")
DATA_PREP_IMAGE = _os.environ.get(
    "AUTO_E2E_DATA_PREP_IMAGE",
    f"{ECR_PREFIX}/auto-e2e/data-prep:latest",
)

DATASET_PACK_VERSION = "v2.4"
KITSCENES_TEST_DATASET_VERSION = "v3.5-test-camera-only-v1"
L2D_SOURCE_REVISION = L2D_DATA_REVISION
KITSCENES_SOURCE_REVISION = "6fde0034446669e2ed7235e4c7fe323cd23d599d"

# Flyte cache versions. The cache key is (task interface, input literals,
# cache_version); code-contract determinants (uid/parser/shard/geometry schema)
# can't be captured by inputs, so they go here. Sourced from
# Model/data_processing/contract_versions.py. Registration must put Model/ on
# PYTHONPATH; an import failure is fatal because guessing these values can
# silently reuse incompatible cache entries. Per-partition group_ids and
# source_revision travel as task inputs, so ranges are independently cacheable.
#
# Each stage's cache_version folds in only the contracts that determine its
# output: ingest depends on the parser enumeration; labels also on the uid
# format (the JOIN key) and sparse-selection policy; pack on the shard and
# geometry encoding.
_DEPLOYED_CACHE_VERSION_ALIASES = {
    # Map these exact ingest/reasoning contract tuples to the v1 cache strings
    # already present in deployed caches so those expensive KITScenes caches
    # remain reusable. Pack must follow geometry contract bumps; any other
    # contract tuple falls through to a contract-derived cache version.
    ("ingest", "v2"): "ingest-v1",
    ("label", "v2", "v1", "v2"): "label-v1-v1-v1",
}


def _cache_versions_for_contracts(
    *,
    uid: str,
    parser: str,
    shard: str,
    geometry: str,
    label_policy: str,
) -> dict[str, str]:
    def resolve(stage: str, *contracts: str) -> str:
        key = (stage, *contracts)
        return _DEPLOYED_CACHE_VERSION_ALIASES.get(key, "-".join(key))

    return {
        "ingest": resolve("ingest", parser),
        "label": resolve("label", parser, uid, label_policy),
        "pack": resolve("pack", parser, uid, shard, geometry),
    }


_CACHE_VERSIONS = _cache_versions_for_contracts(
    uid=_UID_V,
    parser=_PARSER_V,
    shard=_SHARD_V,
    geometry=_GEOM_V,
    label_policy=_LABEL_POLICY_V,
)
INGEST_CACHE_VERSION = _CACHE_VERSIONS["ingest"]
LABEL_CACHE_VERSION = _CACHE_VERSIONS["label"]
PACK_CACHE_VERSION = (
    f"{_CACHE_VERSIONS['pack']}-camera{REACTIVE_CAMERA_IMAGE_SIZE}"
)
KITSCENES_TEST_DIRECT_PACK_CACHE_VERSION = (
    f"kitscenes-test-direct-{INGEST_CACHE_VERSION}-{PACK_CACHE_VERSION}-v1"
)
NAVIGATION_QUALITY_CACHE_VERSION = "navigation-quality-v2"
_FLYTE_DOWNLOAD_GATHER_BATCH_SIZE = 16


def _packed_episode_count(
    episodes: int,
    group_ids: list[str] | None,
) -> int:
    """Return the number of source groups represented by one pack task."""
    return len(group_ids) if group_ids is not None else episodes


def _download_flyte_directory(
    directory: FlyteDirectory,
    *,
    attempts: int = 3,
    initial_delay_seconds: float = 15.0,
) -> str:
    """Bound concurrency and retry transient object-store downloads."""
    import fsspec
    from flytekit.exceptions.system import FlyteDownloadDataException

    if attempts <= 0 or initial_delay_seconds < 0:
        raise ValueError("Flyte directory retry policy is invalid")
    # fsspec derives its default from RLIMIT_NOFILE. Large pack pods can then
    # queue thousands of signed S3 requests long enough for the signatures to
    # expire before a connection becomes available.
    for key in ("gather_batch_size", "nofiles_gather_batch_size"):
        configured = fsspec.config.conf.get(key)
        if (
            configured is None
            or int(configured) == -1
            or int(configured) > _FLYTE_DOWNLOAD_GATHER_BATCH_SIZE
        ):
            fsspec.config.conf[key] = _FLYTE_DOWNLOAD_GATHER_BATCH_SIZE
    for attempt in range(1, attempts + 1):
        try:
            return directory.download()
        except FlyteDownloadDataException:
            if attempt == attempts:
                raise
            delay = initial_delay_seconds * (2 ** (attempt - 1))
            print(
                "Flyte directory download failed; retrying "
                f"attempt={attempt + 1}/{attempts} delay={delay:.0f}s"
            )
            time.sleep(delay)
    raise AssertionError("unreachable Flyte directory retry state")


def _bounded_group_coverage_indices(
    dataset,
    sample_limit: int,
) -> list[int]:
    """Select an exact sample budget while retaining every non-empty group."""
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive")
    by_group: dict[str, list[int]] = {}
    for sample_index in range(len(dataset)):
        group_id = str(dataset.split_group_uid(sample_index))
        by_group.setdefault(group_id, []).append(sample_index)
    if not by_group:
        raise ValueError(
            "bounded sampling found no valid samples"
        )
    if sample_limit < len(by_group):
        raise ValueError(
            "sample_limit is smaller than the number of non-empty groups: "
            f"limit={sample_limit} groups={len(by_group)}"
        )
    if sample_limit > len(dataset):
        raise ValueError(
            "sample_limit exceeds available samples: "
            f"limit={sample_limit} available={len(dataset)}"
        )
    if sample_limit == len(dataset):
        return list(range(len(dataset)))

    group_items = sorted(by_group.items(), key=lambda item: item[0])
    extra_budget = sample_limit - len(group_items)
    capacities = [
        len(indices) - 1 for _, indices in group_items
    ]
    total_capacity = sum(capacities)
    weighted = [
        extra_budget * capacity / total_capacity
        for capacity in capacities
    ]
    extras = [int(value) for value in weighted]
    unassigned = extra_budget - sum(extras)
    if unassigned < 0:
        raise AssertionError("bounded sample allocation is negative")
    remainder_order = sorted(
        range(len(group_items)),
        key=lambda index: (
            -(weighted[index] - extras[index]),
            group_items[index][0],
        ),
    )
    for index in remainder_order[:unassigned]:
        extras[index] += 1

    selected: list[int] = []
    for (_, indices), extra_count in zip(group_items, extras):
        quota = 1 + extra_count
        if quota == 1:
            positions = [len(indices) // 2]
        else:
            positions = [
                round(position * (len(indices) - 1) / (quota - 1))
                for position in range(quota)
            ]
        selected.extend(indices[position] for position in positions)
    selected.sort()
    if len(selected) != sample_limit or len(selected) != len(set(selected)):
        raise AssertionError(
            "bounded group-coverage selection produced an invalid budget"
        )
    return selected


def _allocate_partition_sample_limits(
    partitions: list[list[str]],
    total_sample_limit: int,
) -> list[int]:
    """Allocate an exact budget enforced by each partition pack task."""
    if not partitions or any(not partition for partition in partitions):
        raise ValueError("partitions must be non-empty")
    group_counts = [len(partition) for partition in partitions]
    total_groups = sum(group_counts)
    if total_sample_limit < total_groups:
        raise ValueError(
            "total_sample_limit must allow at least one sample per group"
        )
    weighted = [
        total_sample_limit * count / total_groups
        for count in group_counts
    ]
    limits = [int(value) for value in weighted]
    unassigned = total_sample_limit - sum(limits)
    if unassigned < 0:
        raise AssertionError("partition sample allocation is negative")
    remainder_order = sorted(
        range(len(partitions)),
        key=lambda index: (
            -(weighted[index] - limits[index]),
            index,
        ),
    )
    for index in remainder_order[:unassigned]:
        limits[index] += 1
    if sum(limits) != total_sample_limit:
        raise AssertionError("partition sample allocation is invalid")
    return limits


def _data_prep_pod_template():
    """Protect active data-prep pods from voluntary Karpenter disruption."""
    from flytekit import PodTemplate

    return PodTemplate(
        annotations={"karpenter.sh/do-not-disrupt": "true"},
    )


class Dataset(enum.Enum):
    L2D = "yaak-ai/L2D"
    KITSCENES = "KIT-MRT/KITScenes-Multimodal"
    NVIDIA_PHYSICAL_AI = "nvidia/PhysicalAI-Autonomous-Vehicles"


KITSCENES_TRAINING_SPLIT = "train"
KITSCENES_BENCHMARK_SPLITS = frozenset({"val", "overlap_train_val"})
KITSCENES_LABELED_EVALUATION_SPLITS = frozenset({
    *KITSCENES_BENCHMARK_SPLITS,
    "test",
})


def _validate_kitscenes_data_role(
    *,
    data_role: str,
    source_split: str,
) -> None:
    """Keep held-out KITScenes scenes outside every training workflow."""
    if data_role == "training":
        if source_split != KITSCENES_TRAINING_SPLIT:
            raise ValueError(
                "KITScenes training accepts only the official train split, "
                f"got {source_split!r}"
            )
        return
    if data_role == "benchmark":
        if source_split not in KITSCENES_LABELED_EVALUATION_SPLITS:
            raise ValueError(
                "KITScenes labeled evaluation preparation accepts only val, "
                f"overlap_train_val, and test, got {source_split!r}"
            )
        return
    raise ValueError(f"unsupported KITScenes data_role {data_role!r}")


def _row_decode_worker_count(dataset: Dataset, row_count: int) -> int:
    """Bound row decoders by each parser's per-process memory footprint."""
    # Each KITScenes child reparses the scene's Lanelet2 map and calibration.
    # Two children can exceed the 64 GiB pod limit on large scenes. An L2D
    # child decodes one episode video at a time, so the pod's 15 CPUs, not
    # memory, bound it.
    max_workers = {
        Dataset.KITSCENES: 1,
        Dataset.L2D: 12,
    }.get(dataset, 16)
    return max(1, min(max_workers, row_count))


def _use_parent_assembly_pack(
    dataset: Dataset,
    *,
    has_samples: bool,
    world_model: bool,
    reactive_targets: bool,
) -> bool:
    """Select the memory-bounded row-decode path for structured targets."""
    return (
        dataset != Dataset.NVIDIA_PHYSICAL_AI
        and has_samples
        and (
            world_model
            or dataset in (Dataset.KITSCENES, Dataset.L2D)
            or reactive_targets
        )
    )


def _l2d_episode_navigation(
    ds,
    episodes,
    *,
    osm_region_path: str,
    osm_region_sha256: str,
) -> tuple[dict, dict]:
    """Per-episode L2D navigation and its manifest summary.

    Every episode gets its own lane graph from the pinned OSM region, so a
    region that does not cover an episode fails the pack instead of
    publishing samples without map channels.
    """
    import hashlib

    from data_parsing.l2d.episode_navigation import (
        L2D_NAVIGATION_VERSION,
        L2DEpisodeNavigation,
        episode_lane_graph_from_region,
    )
    from data_parsing.l2d.osm_pbf_region import RegionIndex, parse_region
    from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
    from navigation.rasterizer import NativeNavigationRasterizer

    payload = Path(osm_region_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != osm_region_sha256:
        raise ValueError("OSM region differs from its pinned SHA-256")
    nodes, ways, _ = parse_region(payload)
    region = RegionIndex(nodes, ways)
    rasterizer = NativeNavigationRasterizer(AUTOE2E_NAVIGATION_GEOMETRY)
    navigation = {}
    records = []
    for episode in sorted(episodes):
        longitudes, latitudes, headings, timestamps = ds.episode_track(episode)
        graph, graph_sha256 = episode_lane_graph_from_region(
            region,
            longitudes_deg=longitudes,
            latitudes_deg=latitudes,
            region_sha256=osm_region_sha256,
            episode_id=str(episode),
        )
        episode_navigation = L2DEpisodeNavigation(
            episode_id=str(episode),
            longitudes_deg=longitudes,
            latitudes_deg=latitudes,
            headings_deg_cw_from_north=headings,
            timestamps_ns=timestamps,
            lane_graph=graph,
            lane_graph_sha256=graph_sha256,
            rasterizer=rasterizer,
        )
        navigation[episode] = episode_navigation
        quality = episode_navigation.route.quality
        records.append({
            "episode_index": int(episode),
            "lane_graph_sha256": graph_sha256,
            "lane_count": len(graph["lanes"]),
            "episode_navigation_sha256": (
                episode_navigation.episode_navigation_sha256
            ),
            "route_valid": bool(episode_navigation.route.valid),
            "route_matched_pose_ratio": float(quality.matched_pose_ratio),
            "route_p95_lateral_distance_m": float(
                quality.p95_lateral_distance_m
            ),
            "route_failure_reasons": list(quality.failure_reasons),
        })
    summary = {
        "schema_version": L2D_NAVIGATION_VERSION,
        "osm_region_sha256": osm_region_sha256,
        "episodes": records,
        "route_valid_episodes": sum(r["route_valid"] for r in records),
    }
    return navigation, summary


def _loader_download_dir(shard) -> str:
    """Download one shard FlyteDirectory and return its local path (merged path)."""
    return str(shard.download())


def _reasoning_label_indices(ds, label_stride: int) -> List[int]:
    """Select a stable sparse label set with supervision in every split group.

    The regular frame-index grid remains partition-independent. Its union with
    each group's earliest valid sample covers short scenes whose entire valid
    span falls between grid points. The extra sample costs at most one teacher
    call per scene/episode and prevents a non-empty shard with zero supervision.
    """
    if label_stride <= 1:
        return list(range(len(ds)))

    selected: set[int] = set()
    first_by_group: dict[str, tuple[int, int]] = {}
    for sample_index in range(len(ds)):
        frame_index = int(ds.frame_index(sample_index))
        group_id = str(ds.split_group_uid(sample_index))
        first = first_by_group.get(group_id)
        candidate = (frame_index, sample_index)
        if first is None or candidate < first:
            first_by_group[group_id] = candidate
        if frame_index % label_stride == 0:
            selected.add(sample_index)

    selected.update(sample_index for _, sample_index in first_by_group.values())
    return sorted(selected)


# ============================================================
# Task: Resolve the immutable fan-out inventory
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="1", mem="2Gi", ephemeral_storage="10Gi"),
    limits=Resources(cpu="1", mem="2Gi", ephemeral_storage="10Gi"),
    secret_requests=[Secret(group="hf-token", key="HF_TOKEN",
                            mount_requirement=Secret.MountType.ENV_VAR)],
    cache=True,
    cache_version=f"inventory-{_PARSER_V}",
    retries=2,
)
def plan_fanout_partitions(
    dataset: Dataset,
    source_revision: str,
    episodes: int,
    start_ep: int,
    end_ep: int,
    partition_size: int,
    max_partitions: int,
    max_missing_scenes: int = 1,
    split: str = "train",
    data_role: str = "training",
) -> List[List[str]]:
    """Resolve source groups once and return deterministic mapped-task inputs.

    KITScenes is one scene per partition. The pinned SDK's official split is
    reconciled with the pinned Hugging Face archive manifest before any large pod
    is launched; missing scenes beyond ``max_missing_scenes`` or any unexpected
    scene fail the workflow at preflight.
    """
    import json
    import os
    import tempfile

    from data_processing.partition_plan import plan_partitions

    if episodes < 0:
        raise ValueError(f"episodes must be >= 0, got {episodes}")
    if start_ep >= 0 and end_ep <= start_ep:
        raise ValueError(
            f"end_ep must be greater than start_ep, got [{start_ep}, {end_ep})"
        )

    token = ""
    try:
        from flytekit import current_context
        token = current_context().secrets.get("hf-token", "HF_TOKEN")
    except Exception:
        token = os.environ.get("HF_TOKEN", "")

    if dataset == Dataset.KITSCENES:
        _validate_kitscenes_data_role(
            data_role=data_role,
            source_split=split,
        )
        if source_revision != KITSCENES_SOURCE_REVISION:
            raise ValueError(
                "KITScenes source_revision must match the audited pinned "
                f"revision {KITSCENES_SOURCE_REVISION}, got {source_revision!r}"
            )
        if partition_size != 1:
            raise ValueError(
                "KITScenes requires partition_size=1 because calibration and "
                "map state are scene-scoped"
            )
        from data_parsing.kit_scenes.source import (
            fetch_archive_manifest,
            resolve_inventory,
        )

        with tempfile.TemporaryDirectory(prefix="kitscenes_inventory_") as tmp:
            archives = fetch_archive_manifest(
                tmp,
                revision=source_revision,
                token=token or None,
            )
        inventory = resolve_inventory(
            archives,
            split=split,
            source_revision=source_revision,
            max_missing_scenes=max_missing_scenes,
        )
        group_ids = list(inventory.selected_scene_ids)
        print(
            "KITScenes inventory preflight: "
            + json.dumps(inventory.metadata(), sort_keys=True)
        )
    elif dataset == Dataset.L2D:
        if data_role != "training" or split != "train":
            raise ValueError(
                "L2D fan-out supports only data_role='training', split='train'"
            )
        if source_revision != L2D_SOURCE_REVISION:
            raise ValueError(
                "L2D source_revision must match the audited pinned commit "
                f"{L2D_SOURCE_REVISION}; got {source_revision!r}"
            )
        if episodes == 0 or start_ep >= 0:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            except ModuleNotFoundError:
                from ledataset.datasets.lerobot_dataset import LeRobotDatasetMetadata
            from huggingface_hub import login

            if token:
                login(token=token)
            meta = LeRobotDatasetMetadata(
                repo_id=dataset.value,
                revision=source_revision,
            )
            total = int(meta.total_episodes)
        else:
            total = episodes
        group_ids = [str(index) for index in range(total)]
    else:
        if data_role != "training" or split != "train":
            raise ValueError(
                "non-KITScenes fan-out supports only "
                "data_role='training', split='train'"
            )
        raise NotImplementedError(
            "NVIDIA PhysicalAI fan-out remains deferred; use the existing "
            "single-dataset workflow for that source."
        )

    if start_ep >= 0:
        if end_ep > len(group_ids):
            raise ValueError(
                f"requested range [{start_ep}, {end_ep}) exceeds the resolved "
                f"{len(group_ids)} groups"
            )
        selected = group_ids[start_ep:end_ep]
    elif episodes > 0:
        if episodes > len(group_ids):
            raise ValueError(
                f"requested {episodes} groups but only {len(group_ids)} resolved"
            )
        selected = group_ids[:episodes]
    else:
        selected = group_ids

    plan = plan_partitions(
        selected,
        partition_size=partition_size,
        max_partitions=max_partitions,
    )
    print(f"Fan-out inventory: {plan.summary()}")
    return [list(partition.group_ids) for partition in plan.partitions]


# ============================================================
# Task: Data Ingest (download raw from HuggingFace)
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_data_prep_pod_template(),
    # KITScenes fan-out is one scene per pod. Its largest pinned archive is
    # 20.12 GiB; download+extract briefly holds about twice that, so 60Gi fits
    # the EKS Auto Mode default NodeClass (~70Gi allocatable disk). 15 vCPU stays
    # below a 16-vCPU node's kube-reserved allocatable boundary.
    requests=Resources(cpu="15", mem="64Gi", ephemeral_storage="60Gi"),
    limits=Resources(cpu="15", mem="64Gi", ephemeral_storage="60Gi"),
    secret_requests=[Secret(group="hf-token", key="HF_TOKEN",
                            mount_requirement=Secret.MountType.ENV_VAR)],
    # Cache on (dataset, group_ids, episodes, cache_version): a partition's raw
    # is fetched once, and its stable output URI lets the downstream label/pack
    # caches hit too (FlyteDirectory hashes by URI). The HF token is a secret
    # env, not an input, so it never enters the key.
    cache=True,
    cache_version=INGEST_CACHE_VERSION,
    # Retries absorb transient HF rate limits and node-placement failures in
    # large fan-outs.
    retries=2,
    # Cut the multipart-upload chunk from the flytekit default 25 MiB to 8 MiB.
    # s3fs holds one `chunksize` buffer plus an aiobotocore send buffer per
    # in-flight file; with BatchSize(4) on the return type this bounds peak
    # upload RSS to a few hundred MB.
    environment={"_F_P_WRITE_CHUNK_SIZE": "8388608"},
)
def data_ingest(
    dataset: Dataset = Dataset.L2D,
    source_revision: str = L2D_SOURCE_REVISION,
    episodes: int = 3,
    group_ids: Optional[List[str]] = None,
    source_split: str = "train",
    data_role: str = "training",
) -> Annotated[FlyteDirectory, BatchSize(4)]:
    """Download raw dataset from HuggingFace (lerobot for L2D, physical_ai_av for NVIDIA).

    HF token comes from the `hf-token` K8s Secret (injected as env var by Flyte),
    never from a workflow input, so it is not visible in the Flyte/MLflow UI.

    ``group_ids`` selects an explicit set of groups (L2D episode indices as
    strings, KITScenes scene IDs, or NVIDIA clip uuids) so a fan-out partition
    materializes only its slice. When None, the first ``episodes`` groups are
    used. The ids are global, which keeps the downstream ``sample_uid``
    partition-independent.
    """
    import os
    import shutil
    from huggingface_hub import login
    from flytekit import current_context

    token = ""
    try:
        token = current_context().secrets.get("hf-token", "HF_TOKEN")
    except Exception:
        token = os.environ.get("HF_TOKEN", "")
    if token:
        login(token=token)
        os.environ["HF_TOKEN"] = token

    out_dir = "/tmp/raw_data"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if dataset == Dataset.KITSCENES:
        _validate_kitscenes_data_role(
            data_role=data_role,
            source_split=source_split,
        )
        if source_revision != KITSCENES_SOURCE_REVISION:
            raise ValueError(
                "KITScenes ingest requires pinned source revision "
                f"{KITSCENES_SOURCE_REVISION}, got {source_revision!r}"
            )
        from data_parsing.kit_scenes.source import (
            PinnedKITScenesDownloader,
            resolve_inventory,
        )

        downloader = PinnedKITScenesDownloader(
            out_dir,
            revision=source_revision,
            token=token or None,
        )
        if group_ids is None:
            inventory = resolve_inventory(
                downloader.archives,
                split=source_split,
                source_revision=source_revision,
                max_missing_scenes=(
                    1 if data_role == "training" else 0
                ),
            )
            scene_ids = list(inventory.selected_scene_ids)
            if episodes > 0:
                scene_ids = scene_ids[:episodes]
        else:
            scene_ids = [str(scene_id) for scene_id in group_ids]
        downloader.download(scene_ids, expected_split=source_split)
        print(
            f"Ingested {dataset.value}@{source_revision}: "
            f"{len(scene_ids)} {source_split} scenes -> {out_dir}"
        )
        return FlyteDirectory(out_dir)

    if source_split != "train" or data_role != "training":
        raise ValueError(
            "non-KITScenes ingest supports only "
            "data_role='training', source_split='train'"
        )

    if dataset == Dataset.NVIDIA_PHYSICAL_AI:
        # NVIDIA PhysicalAI-AV: download via physical_ai_av SDK + unpack into the
        # parser layout (camera/<cam>/, labels/egomotion/) that NvidiaAVDataset reads.
        import pathlib
        from physical_ai_av import PhysicalAIAVDatasetInterface
        from data_parsing.nvidia_physical_ai.download_dataset import (
            CAMERAS, unpack_camera_zip, unpack_egomotion_zip,
        )
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ds = PhysicalAIAVDatasetInterface(
            local_dir=str(out / ".hf_cache"),
            confirm_download_threshold_gb=float("inf"),
        )
        if group_ids is not None:
            clip_ids = list(group_ids)
        else:
            clip_ids = ds.clip_index.index.tolist()[:episodes]
        feats = CAMERAS + ["egomotion"]
        # Native f-theta intrinsics + sensor extrinsics. The rig is shared across
        # the subset, so save calibration from the first clip that has it and
        # fall back to pseudo geometry downstream if none does.
        calib_saved = False
        for clip_id in clip_ids:
            ds.download_clip_features(clip_id, features=feats)
            for cam in CAMERAS:
                cf = ds.features.get_chunk_feature_filename(ds.get_clip_chunk(clip_id), cam)
                with ds.open_file(cf, maybe_stream=True) as f:
                    unpack_camera_zip(f.read(), clip_id, cam, out)
            cf = ds.features.get_chunk_feature_filename(ds.get_clip_chunk(clip_id), "egomotion")
            with ds.open_file(cf, maybe_stream=True) as f:
                unpack_egomotion_zip(f.read(), clip_id, out)
            if not calib_saved:
                try:
                    import pickle
                    ds.download_clip_features(
                        clip_id, features=["camera_intrinsics", "sensor_extrinsics"])
                    intr = ds.get_clip_feature(clip_id, "camera_intrinsics")
                    extr = ds.get_clip_feature(clip_id, "sensor_extrinsics")
                    calib_dir = out / "calibration"
                    calib_dir.mkdir(parents=True, exist_ok=True)
                    with open(calib_dir / "intrinsics.pkl", "wb") as f:
                        pickle.dump(intr, f)
                    with open(calib_dir / "extrinsics.pkl", "wb") as f:
                        pickle.dump(extr, f)
                    calib_saved = True
                    print(f"Saved NVIDIA calibration from clip {clip_id}")
                except Exception as e:
                    print(f"WARN: no calibration for clip {clip_id}: {e}")
        print(f"Ingested {dataset.value}: {len(clip_ids)} clips → {out_dir}")
        return FlyteDirectory(out_dir)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from ledataset.datasets.lerobot_dataset import LeRobotDataset

    if group_ids is not None:
        ep_list = [int(g) for g in group_ids]
    else:
        ep_list = list(range(episodes)) if episodes > 0 else None
    # Keep download_videos enabled. The label/pack pods re-open this dir with
    # LeRobotDataset(root=...); lerobot's _check_cached_episodes_sufficient
    # requires the requested episodes' video files on disk, else the offline pod
    # attempts a network re-download and fails.
    #
    # snapshot_download can silently under-fetch on transient Hub errors, which
    # surfaces later as an opaque "corresponds to no data" error. The explicit
    # pre-fetch below guarantees every parquet and video file is on disk before
    # LeRobotDataset runs, and fails with a RuntimeError otherwise.
    from ledataset.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from huggingface_hub import hf_hub_download
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    # The pinned main-branch commit avoids lerobot's default v3.0 tag, which
    # points to an incomplete yaak-ai/L2D snapshot (tasks.parquet has 1 row
    # instead of 4219) and causes downstream KeyError/IndexError.
    if source_revision != L2D_SOURCE_REVISION:
        raise ValueError(
            "L2D ingest requires the audited pinned commit "
            f"{L2D_SOURCE_REVISION}; got {source_revision!r}"
        )
    _meta = LeRobotDatasetMetadata(
        repo_id=dataset.value,
        revision=source_revision,
    )
    if ep_list is not None:
        # Compute the set of parquet+video paths lerobot would ask for, then
        # download each with hf_hub_download. Matches lerobot's own
        # dataset_reader.get_episodes_file_paths, but with per-file timeouts
        # so a single stalled HTTP connection can't hang the whole task.
        _data_paths = list({str(_meta.get_data_file_path(ep)) for ep in ep_list})
        _video_paths = list({
            str(_meta.get_video_file_path(ep, k))
            for k in _meta.video_keys for ep in ep_list
        })
        _all_files = _data_paths + _video_paths
        print(f"Pre-fetch: {len(_data_paths)} parquet + {len(_video_paths)} video "
              f"= {len(_all_files)} unique files for {len(ep_list)} episodes")

        def _one_file(rel_path: str, attempt_i: int) -> tuple[str, bool, str]:
            """Download one file into local_dir; return (path, success, note)."""
            try:
                hf_hub_download(
                    repo_id=dataset.value, repo_type="dataset",
                    revision=source_revision,
                    filename=rel_path, local_dir=str(_meta.root),
                    etag_timeout=30.0,
                )
                return (rel_path, True, "ok")
            except Exception as e:
                return (rel_path, False, f"{type(e).__name__}: {e}")

        # Up to 3 attempts; later attempts re-download only files still missing
        # on disk. 4 workers balance throughput against in-flight buffer memory.
        _missing = list(_all_files)
        for attempt in range(3):
            batch = _missing if attempt > 0 else _all_files
            print(f"Pre-fetch attempt {attempt+1}: downloading {len(batch)} files "
                  f"({4} workers)")
            t0 = time.time()
            n_ok = n_fail = 0
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_one_file, p, attempt): p for p in batch}
                for f in as_completed(futures):
                    p, ok, note = f.result()
                    if ok:
                        n_ok += 1
                    else:
                        n_fail += 1
                        print(f"  FAIL {p}: {note}")
            print(f"Pre-fetch attempt {attempt+1}: {n_ok} ok / {n_fail} fail "
                  f"in {time.time()-t0:.0f}s")
            # Every expected file must be on disk so downstream pods stay
            # offline (see the download_videos note above).
            _missing = [p for p in _all_files
                        if not (_meta.root / p).exists()]
            if not _missing:
                print(f"Pre-fetch attempt {attempt+1}: all "
                      f"{len(_all_files)} files present on disk")
                break
            _mp = [p for p in _missing if p in set(_data_paths)]
            print(f"Pre-fetch attempt {attempt+1}: "
                  f"{len(_missing)} files STILL missing "
                  f"({len(_mp)} parquets + {len(_missing)-len(_mp)} videos, "
                  f"first: {_missing[:2]}); retrying")
        else:
            _mp = [p for p in _missing if p in set(_data_paths)]
            raise RuntimeError(
                f"data_ingest: after 3 attempts, {len(_missing)} files "
                f"are still missing on disk ({len(_mp)} parquets + "
                f"{len(_missing)-len(_mp)} videos; first missing: "
                f"{_missing[:3]}). HF Hub may be transiently degraded — "
                f"retry the task.")
    ds = LeRobotDataset(
        repo_id=dataset.value,
        episodes=ep_list,
        revision=source_revision,
    )
    cache_dir = ds.root
    # Hardlink the whole cache tree (data/ + meta/ + videos/) into out_dir: a
    # byte copy doubles disk use and page-cache pressure. The full tree is what
    # lets the downstream root= reopen stay offline. Falls back to a real copy
    # across filesystem boundaries.
    try:
        shutil.copytree(str(cache_dir), out_dir, copy_function=os.link)
    except OSError:
        shutil.copytree(str(cache_dir), out_dir)

    print(f"Ingested {dataset.value}: {len(ds)} frames, {episodes} episodes → {out_dir}")
    return FlyteDirectory(out_dir)


# ============================================================
# Task: Data Processing (pre-extract frames into WebDataset shards)
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_data_prep_pod_template(),
    # Process-parallel pack workers use the pod's cores for camera decode/JPEG.
    # KITScenes one-scene partitions use the same schedulable Guaranteed profile
    # as ingest; storage is sized for the 512px camera contract.
    requests=Resources(cpu="15", mem="64Gi", ephemeral_storage="240Gi"),
    limits=Resources(cpu="15", mem="64Gi", ephemeral_storage="240Gi"),
    # Cache on (raw URI, labels URI, group_ids, world_model, image_size,
    # cache_version). Raw + labels inputs are FlyteDirectories hashed by URI, so
    # an upstream cache hit keeps their URIs stable and this task hits too.
    # PACK_CACHE_VERSION folds in the shard + geometry encoding.
    cache=True,
    cache_version=PACK_CACHE_VERSION,
    retries=2,
    # Same fsspec upload-chunk cap as data_ingest; pack output can hold tens of
    # thousands of files.
    environment={"_F_P_WRITE_CHUNK_SIZE": "8388608"},
)
def data_processing(
    raw_data: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    source_revision: str = L2D_SOURCE_REVISION,
    dataset_version: str = DATASET_PACK_VERSION,
    hz: int = 10,
    image_size: int = REACTIVE_CAMERA_IMAGE_SIZE,
    episodes: int = 3,
    world_model: bool = False,
    reasoning_labels: Optional[FlyteDirectory] = None,
    group_ids: Optional[List[str]] = None,
    sample_limit: int = 0,
    source_split: str = "train",
    data_role: str = "training",
    osm_region_path: str = "",
    osm_region_sha256: str = "",
) -> Annotated[FlyteDirectory, BatchSize(4)]:
    """Pre-extract aligned frames + egomotion → WebDataset shards.

    Deterministic packing with no video decode at training time; this task calls
    no external teacher. When ``reasoning_labels`` (the
    ``generate_reasoning_labels`` artifact) is provided, each sample's frozen
    label is joined by ``sample_id`` and embedded as a ``reasoning.json`` member.

    When ``world_model`` is set, each sample also gets the 1 Hz past/future
    multi-view windows for the JEPA loss. The window config (num_frames/stride)
    matches the online dataset so shards and on-the-fly windows are identical.

    L2D needs ``osm_region_path`` (an ``l2d_osm_region_v1`` file pinned by
    ``osm_region_sha256``): its 14-channel navigation is rendered from OSM and
    the dataset map image, since L2D ships no HD map.
    """
    import os
    import io
    import json
    import tarfile
    import tempfile

    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")
    if dataset == Dataset.L2D and not (osm_region_path and osm_region_sha256):
        raise ValueError(
            "L2D packs need osm_region_path and osm_region_sha256 for "
            "14-channel navigation"
        )
    if (
        dataset == Dataset.KITSCENES
        and image_size != REACTIVE_CAMERA_IMAGE_SIZE
    ):
        raise ValueError(
            "KITScenes T8 packing requires the 512px camera contract"
        )
    if dataset == Dataset.KITSCENES:
        _validate_kitscenes_data_role(
            data_role=data_role,
            source_split=source_split,
        )
    elif source_split != "train" or data_role != "training":
        raise ValueError(
            "non-KITScenes processing supports only "
            "data_role='training', source_split='train'"
        )
    benchmark_protocol = (
        dataset == Dataset.KITSCENES and data_role == "benchmark"
    )
    mapless_kitscenes_test = (
        dataset == Dataset.KITSCENES
        and data_role == "benchmark"
        and source_split == "test"
    )
    raw_path = _download_flyte_directory(raw_data)
    print(f"Processing raw data from: {raw_path} (dataset={dataset.value})")

    # Reasoning labels imply a full-loss run, and the JEPA loss needs the WM
    # window packed, so force WM on. The sample set does not depend on the WM
    # flag (egomotion margins 64/64 dominate the WM margins 30/40). The label set
    # is a 1 Hz subset of the 10 Hz packed set; unlabeled samples pack without
    # reasoning.json and mask out of the reasoning loss. NVIDIA has no WM windows
    # and no labels.
    if reasoning_labels is not None and dataset != Dataset.NVIDIA_PHYSICAL_AI and not world_model:
        print("reasoning_labels present → forcing world_model=True so the JEPA "
              "loss has its WM window (future frames) packed.")
        world_model = True

    # Datasets are raw pre-extraction sources that emit unmodified frames. The
    # shard packer below owns the single geometry-aware resize; the pre-extracted
    # loader owns the single ToTensor+Normalize, so the projection targets a
    # known plain-resized frame. Sample schema: visual_tiles (V,3,H,W), map_tile
    # (3,H,W), egomotion_history (256), trajectory_target (128).
    if dataset == Dataset.KITSCENES:
        if source_revision != KITSCENES_SOURCE_REVISION:
            raise ValueError(
                "KITScenes pack requires pinned source revision "
                f"{KITSCENES_SOURCE_REVISION}, got {source_revision!r}"
            )
        ep_list = (
            [str(group_id) for group_id in group_ids]
            if group_ids is not None
            else None
        )
    else:
        if dataset == Dataset.L2D and source_revision != L2D_SOURCE_REVISION:
            raise ValueError(
                "L2D pack requires the audited pinned commit "
                f"{L2D_SOURCE_REVISION}; got {source_revision!r}"
            )
        ep_list = ([int(g) for g in group_ids] if group_ids is not None
                   else (list(range(episodes)) if episodes > 0 else None))
    # A fan-out partition can legitimately hold no valid samples (a short
    # episode/clip below the egomotion margin). Treat the parser's "No valid
    # samples" error as success with an empty shard dir, matching the label task.
    try:
        if dataset == Dataset.NVIDIA_PHYSICAL_AI:
            from data_parsing.nvidia_physical_ai.dataset import NvidiaAVDataset
            # Discover from raw_path, which holds only this partition's clips, so
            # the packer enumerates the same set in the same order as the labeler.
            ds = NvidiaAVDataset(data_root=raw_path)
            if world_model:
                print("world_model requested but NVIDIA has no window support yet; "
                      "packing without JEPA windows.")
        elif dataset == Dataset.KITSCENES:
            from data_parsing.kit_scenes import KitScenesDataset
            ds = KitScenesDataset(
                data_root=raw_path,
                split=source_split,
                scene_ids=ep_list,
                image_size=image_size,
                include_world_model_windows=world_model,
                include_navigation=False,
                source_revision=source_revision,
                benchmark_protocol=benchmark_protocol,
                allow_mapless=mapless_kitscenes_test,
            )
        else:
            from data_parsing.l2d import L2DDataset
            # root=raw_path reads the partition's materialized raw without
            # re-hitting HF.
            ds = L2DDataset(
                repo_id=dataset.value,
                revision=source_revision,
                episodes=ep_list,
                include_world_model_windows=world_model,
                root=raw_path,
            )
        n_samples = len(ds)
        idx_iter = (
            _bounded_group_coverage_indices(ds, sample_limit)
            if sample_limit
            else range(n_samples)
        )
        if sample_limit:
            print(
                "Applied bounded group-coverage sampling: "
                f"available={n_samples} selected={len(idx_iter)}"
            )
    except ValueError as e:
        if sample_limit or "No valid samples" not in str(e):
            raise
        print(f"Partition has no valid samples ({e}); writing an EMPTY shard dir "
              f"(short episode/clip — nothing to pack).")
        ds = None
        n_samples = 0
        idx_iter = range(0)

    # Reasoning labels are joined from the generate_reasoning_labels artifact by
    # sample_id. Without them, shards carry no reasoning.json.
    labels_by_id = {}
    _record_to_json = None
    if reasoning_labels is not None:
        from data_processing.reasoning_label_generation.targets import (
            load_records_by_sample_id, record_to_json,
        )
        labels_dir = _download_flyte_directory(reasoning_labels)
        records_files = sorted(Path(labels_dir).rglob("records.jsonl"))
        if records_files:
            for rf in records_files:
                loaded = load_records_by_sample_id(str(rf))
                duplicate_ids = set(labels_by_id).intersection(loaded)
                if duplicate_ids:
                    raise ValueError(
                        "duplicate reasoning sample IDs across artifacts: "
                        f"{sorted(duplicate_ids)[:3]}"
                    )
                labels_by_id.update(loaded)
            _record_to_json = record_to_json
            print(f"Reasoning labels JOIN: {len(labels_by_id)} records from "
                  f"{[str(p) for p in records_files]}")
        else:
            print(f"WARN: reasoning_labels dir {labels_dir} has no records.jsonl; "
                  "packing without reasoning.json (imitation-only).")

    # Geometry is a per-dataset rig constant, computed once and stored in the
    # manifest plus rig/projection.json. Per-sample calib.json retains only the
    # small identity fields; every child of a merged loader carries its own
    # loader-level projection from that manifest.
    projection_spec = None
    build_spec = getattr(ds, "projection_spec", None)
    if callable(build_spec) and n_samples:
        projection_spec = build_spec(image_size)
    camera_order = (
        list(getattr(ds, "camera_names"))
        if ds is not None and hasattr(ds, "camera_names")
        else None
    )
    camera_slots = (
        list(getattr(ds, "camera_slots"))
        if ds is not None and hasattr(ds, "camera_slots")
        else None
    )
    sample_geometry_type = (projection_spec or {}).get("type", "pseudo")
    calib_bytes = json.dumps(
        {"dataset": dataset.value, "geometry_type": sample_geometry_type,
         "image_size": image_size}
    ).encode()

    out_dir = tempfile.mkdtemp()

    # Geo products are generated in the same pass that writes the shards. They
    # read only numeric parquet columns, so this adds no video decode. Each
    # fan-out partition emits its own episode paths + sample-pose parquet, which
    # publication merges without scanning the tar files.
    geo_summary = None
    if (
        ds is not None
        and dataset in (Dataset.L2D, Dataset.KITSCENES)
        and bool(getattr(ds, "has_geospatial", True))
    ):
        from data_processing.geospatial import write_geo_artifacts
        geo_summary = write_geo_artifacts(
            ds,
            out_dir,
            dataset_name=dataset.value,
            dataset_version=dataset_version,
        )

    # Projection/calibration is a rig constant. Publish one canonical rig-level
    # artifact for loaders and the console camera overlay.
    rig_dir = os.path.join(out_dir, "rig")
    os.makedirs(rig_dir, exist_ok=True)
    with open(os.path.join(rig_dir, "projection.json"), "w") as f:
        json.dump({
            "schema_version": "v1",
            "dataset": dataset.value,
            "geometry_type": sample_geometry_type,
            "image_size": image_size,
            "projection": projection_spec,
        }, f, sort_keys=True)

    # Fan-out pack tasks are flattened into one published dataset version, so
    # shard names are prefixed with the deterministic group-set identity to
    # avoid collisions. Non-fan-out workflows keep the compact name.
    from data_processing.dataset_snapshot import (
        published_shard_name,
        shard_partition_id,
    )
    partition_id = shard_partition_id(group_ids)

    shard_idx = 0
    shard_names: list[str] = []
    shard_sample_counts: dict[str, int] = {}
    current_shard_name: str | None = None
    sample_count = 0
    reasoning_label_count = 0
    samples_per_shard = 1000
    current_tar = None

    # Shared frame pool: window frames are content-addressed by a global
    # frame_id and written once, deduping cross-sample overlap. The loose
    # files are sealed into one deterministic archive before publication so a
    # distributed worker can stage and verify the complete pool with one object.
    pool_dir = os.path.join(out_dir, "pool")
    os.makedirs(pool_dir, exist_ok=True)
    seen_frame_ids: set = set()
    pool_frames_written = 0

    def _write_pool(frame_id, blob):
        nonlocal pool_frames_written
        if frame_id in seen_frame_ids:
            return
        seen_frame_ids.add(frame_id)
        with open(os.path.join(pool_dir, f"{frame_id}.jpg"), "wb") as pf:
            pf.write(blob)
        pool_frames_written += 1

    def open_new_shard():
        nonlocal current_tar, current_shard_name, shard_idx
        if current_tar:
            current_tar.close()
        shard_name = published_shard_name(group_ids, shard_idx)
        current_tar = tarfile.open(os.path.join(out_dir, shard_name), "w")
        shard_names.append(shard_name)
        shard_sample_counts[shard_name] = 0
        current_shard_name = shard_name
        shard_idx += 1

    # Decode+JPEG-encode happens in the pack workers (parallel_pack); the parent
    # only appends the returned byte blobs to the current tar (single-threaded).
    def _add_member(sample_key, suffix, blob):
        ti = tarfile.TarInfo(name=f"{sample_key}.{suffix}")
        ti.size = len(blob)
        current_tar.addfile(ti, io.BytesIO(blob))

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from data_processing.reasoning_label_generation import parallel_pack

    idx_list = list(idx_iter)
    ctx = mp.get_context("spawn")
    num_views = 0
    has_map = False
    has_wm = False
    has_bevformer_history = False
    navigation_artifact_summary = None
    trajectory_xy_count = 0
    bev_segmentation_count = 0
    reactive_navigation_count = 0
    geospatial_count = 0

    if _use_parent_assembly_pack(
        dataset,
        has_samples=bool(idx_list),
        world_model=world_model,
        reactive_targets=False,
    ):
        # Decode-dedup path: decode each unique physical row once.
        # Pass A: collect unique (group_id, frame_index) rows; row-level workers
        # decode each exactly once and write to pool/.
        # Pass B: assemble each sample's members (window_index, ego, meta, calib,
        # reasoning JOIN) from the pool with no video decode.
        print(f"Packing {len(idx_list)} samples, parent-assembly mode "
              f"(row-level camera workers, world_model={world_model})...")
        row_init = (
            dataset.value,
            ep_list,
            raw_path,
            image_size,
            source_split,
            source_revision,
            benchmark_protocol,
            mapless_kitscenes_test,
        )

        # Pass A: unique rows. ds is still alive here (not yet deleted).
        all_rows: set = set()
        # Collect the current-frame row (offset 0 = cam_*.jpg) first. Do not
        # catch IndexError from window_rows: enumeration excludes edge frames
        # (margins 64/64 dominate WM 30/40), so a raise means the invariant is
        # broken. Silently dropping the sample's cam_*.jpg would poison the shard
        # (the loader hits torch.stack([]) at train time).
        sample_cur_rows: dict = {}  # si -> (episode/scene, frame) current row
        for si in idx_list:
            if dataset == Dataset.KITSCENES:
                current_row = ds.row_identity(si)
            else:
                ep_idx_s, row_s = ds._samples[si]
                ep_start_s, _ = ds._episode_ranges[ep_idx_s]
                current_row = (ep_idx_s, row_s - ep_start_s)
            sample_cur_rows[si] = current_row
            all_rows.add(current_row)
            if dataset in (Dataset.KITSCENES, Dataset.L2D):
                for row_t in ds.bevformer_history_rows(si):
                    all_rows.add(row_t)
            if world_model:
                for row_t in ds.window_rows(si):
                    all_rows.add(row_t)

        del ds  # free before spawning workers

        # row_map contains base camera JPEGs and current-row extras. KITScenes
        # navigation is generated once in the parent assembly below.
        row_map: dict = {}
        row_workers = _row_decode_worker_count(dataset, len(all_rows))
        current_rows = set(sample_cur_rows.values())
        decode_tasks = [
            (group_id, frame_index, (group_id, frame_index) in current_rows)
            for group_id, frame_index in sorted(all_rows)
        ]
        import inspect
        inspect.signature(parallel_pack.init_row_worker).bind(*row_init)
        if dataset == Dataset.L2D:
            # One task per episode video; more processes would only start idle.
            row_workers = max(1, min(row_workers, len(parallel_pack.l2d_video_tasks(decode_tasks))))

        def _decoded_rows(pool):
            if dataset == Dataset.L2D:
                # One forward pass per episode video instead of a seek per
                # row; rows are assembled once every video is decoded.
                decoded = list(pool.map(
                    parallel_pack.decode_l2d_video,
                    parallel_pack.l2d_video_tasks(decode_tasks),
                ))
                yield from parallel_pack.assemble_l2d_rows(decoded, decode_tasks)
            else:
                yield from pool.map(parallel_pack.decode_row, decode_tasks)

        with ProcessPoolExecutor(max_workers=row_workers, mp_context=ctx,
                                 initializer=parallel_pack.init_row_worker,
                                 initargs=row_init) as rpool:
            for row_key, cam_jpegs, row_extras in _decoded_rows(rpool):
                row_map[row_key] = (cam_jpegs, row_extras)
                for fid, blob in cam_jpegs.items():
                    _write_pool(fid, blob)
                if dataset != Dataset.KITSCENES and row_extras is not None:
                    has_map = True
        num_views = len(next(iter(row_map.values()))[0]) if row_map else 0
        print(f"Frame pool: {pool_frames_written} unique frames decoded "
              f"(was ~{pool_frames_written * 8} with per-sample decode).")

        # Pass B: plain-mode dataset for window IDs and numeric members only.
        import numpy as np
        import torch
        if dataset == Dataset.KITSCENES:
            from data_parsing.kit_scenes import KitScenesDataset
            ds_asm = KitScenesDataset(
                data_root=raw_path,
                split=source_split,
                scene_ids=ep_list,
                image_size=image_size,
                include_world_model_windows=False,
                include_navigation=not mapless_kitscenes_test,
                source_revision=source_revision,
                benchmark_protocol=benchmark_protocol,
                allow_mapless=mapless_kitscenes_test,
            )
        else:
            from data_parsing.l2d import L2DDataset
            ds_asm = L2DDataset(
                repo_id=dataset.value,
                revision=source_revision,
                episodes=ep_list,
                include_world_model_windows=False,
                root=raw_path,
            )

        l2d_navigation: dict = {}
        l2d_map_image_registered = 0
        if dataset == Dataset.L2D:
            l2d_navigation, navigation_artifact_summary = (
                _l2d_episode_navigation(
                    ds_asm,
                    {row[0] for row in sample_cur_rows.values()},
                    osm_region_path=osm_region_path,
                    osm_region_sha256=osm_region_sha256,
                )
            )

        if dataset == Dataset.KITSCENES and not mapless_kitscenes_test:
            import hashlib

            artifact_records = []
            scene_artifacts = ds_asm.scene_navigation_artifacts()
            for scene_id, artifacts in sorted(scene_artifacts.items()):
                destination = (
                    out_dir
                    if len(scene_artifacts) == 1
                    else os.path.join(out_dir, "navigation", scene_id)
                )
                os.makedirs(destination, exist_ok=True)
                hashes = {}
                for filename, blob in sorted(artifacts.items()):
                    with open(os.path.join(destination, filename), "wb") as f:
                        f.write(blob)
                    hashes[filename] = hashlib.sha256(blob).hexdigest()
                quality = json.loads(artifacts["navigation_quality.json"])
                artifact_records.append({
                    "scene_id": scene_id,
                    "path": os.path.relpath(destination, out_dir),
                    "hashes": hashes,
                    "route_valid": bool(quality["route_valid"]),
                    "route_confidence": float(quality["route_confidence"]),
                    "geometry_id": quality["geometry_id"],
                })
            navigation_artifact_summary = {
                "schema_version": "scene_navigation_v1",
                "scenes": artifact_records,
            }

        for si in idx_list:
            if sample_count % samples_per_shard == 0:
                open_new_shard()
            uid = ds_asm.sample_uid(si)
            split_group = ds_asm.split_group_uid(si)
            from data_processing.dataset_snapshot import split_bucket
            members: dict = {}

            if world_model:
                # window_index contains pool frame IDs, never future navigation.
                ids = ds_asm.window_frame_ids(si)
                members["window_index.json"] = json.dumps(ids).encode()
                has_wm = True

            if dataset in (Dataset.KITSCENES, Dataset.L2D):
                history_ids = ds_asm.bevformer_history_frame_ids(si)
                missing_history_ids = sorted(
                    frame_id
                    for frame in history_ids
                    for frame_id in frame
                    if frame_id not in seen_frame_ids
                )
                if missing_history_ids:
                    raise ValueError(
                        f"{dataset.value} T8 history is missing frame-pool members: "
                        f"{missing_history_ids[:8]}"
                    )
                members["bev_history_index.json"] = json.dumps(
                    history_ids,
                    separators=(",", ":"),
                ).encode("ascii")
                has_bevformer_history = True

            # cam_*.jpg = current frame (offset 0), the same jpegs already
            # written to the pool.
            cur_key = sample_cur_rows.get(si)
            if cur_key and cur_key in row_map:
                cur_cams, row_extras = row_map[cur_key]
                # cam_cams is {frame_id: bytes}; sort by cam index embedded in fid.
                for fid, blob in sorted(cur_cams.items(),
                                        key=lambda kv: int(kv[0].rsplit("-c", 1)[-1])):
                    cam_i = int(fid.rsplit("-c", 1)[-1])
                    members[f"cam_{cam_i}.jpg"] = blob
                if dataset == Dataset.KITSCENES:
                    if not isinstance(row_extras, dict):
                        raise ValueError(
                            "KITScenes current row lacks native Front images"
                        )
                    members[
                        f"cam_{REACTIVE_FRONT_CAMERA_INDEX}.jpg"
                    ] = row_extras["front_camera_native_jpeg"]
                    members["front_camera_fpn.jpg"] = row_extras[
                        "front_camera_fpn_jpeg"
                    ]
                    if not mapless_kitscenes_test:
                        members.update(
                            ds_asm.navigation_members_for_row(*cur_key)
                        )
                        has_map = True
                elif dataset == Dataset.L2D:
                    if not isinstance(row_extras, dict):
                        raise ValueError("L2D current row lacks its front view")
                    members[
                        f"cam_{REACTIVE_FRONT_CAMERA_INDEX}.jpg"
                    ] = row_extras["front_camera_native_jpeg"]
                    members["front_camera_fpn.jpg"] = row_extras[
                        "front_camera_fpn_jpeg"
                    ]
                    map_rgb = None
                    if row_extras["map_png"] is not None:
                        members["map.jpg"] = row_extras["map_jpeg"]
                        from PIL import Image

                        map_rgb = np.asarray(
                            Image.open(
                                io.BytesIO(row_extras["map_png"])
                            ).convert("RGB")
                        )
                    navigation_members = l2d_navigation[
                        cur_key[0]
                    ].sample_members(
                        cur_key[1],
                        map_image=map_rgb,
                        waypoints_lon_lat=ds_asm.route_waypoints_for(si),
                    )
                    l2d_map_image_registered += int(json.loads(
                        navigation_members["navigation_meta.json"]
                    )["map_image_registered"])
                    members.update(navigation_members)
                    has_map = True
                elif row_extras is not None:
                    members["map.jpg"] = row_extras

            # ego + meta + calib (no video decode).
            ego_hist, traj = ds_asm.egomotion_for(si)
            ego_data = np.concatenate([
                ego_hist.numpy() if torch.is_tensor(ego_hist) else np.asarray(ego_hist),
                traj.numpy() if torch.is_tensor(traj) else np.asarray(traj),
            ]).astype(np.float32)
            members["ego.npy"] = ego_data.tobytes()
            if (
                dataset != Dataset.KITSCENES
                or bool(getattr(ds_asm, "has_geospatial", False))
            ):
                _, _, pose_current, gps_future = ds_asm.numeric_for(si)
                from data_processing.geospatial import geospatial_members

                members.update(geospatial_members({
                    "pose_current": pose_current,
                    "gps_future": gps_future,
                }))
                geospatial_count += 1
                if dataset == Dataset.L2D:
                    from data_processing.reactive_training_artifacts import (
                        TRAJECTORY_XY_MEMBER,
                        encode_trajectory_xy,
                        wgs84_future_to_ego_xy,
                    )

                    members[TRAJECTORY_XY_MEMBER] = encode_trajectory_xy(
                        *wgs84_future_to_ego_xy(
                            gps_future,
                            current_latitude_deg=pose_current["latitude_deg"],
                            current_longitude_deg=pose_current[
                                "longitude_deg"
                            ],
                            heading_deg_cw_from_north=pose_current[
                                "heading_deg_cw_from_north"
                            ],
                        )
                    )
            if dataset == Dataset.KITSCENES:
                from data_processing.reactive_training_artifacts import (
                    TRAJECTORY_XY_MEMBER,
                    encode_trajectory_xy,
                )

                trajectory_xy, trajectory_valid = (
                    ds_asm.trajectory_xy_for(si)
                )
                members[TRAJECTORY_XY_MEMBER] = encode_trajectory_xy(
                    trajectory_xy,
                    trajectory_valid,
                )
            members["meta.json"] = json.dumps({
                "idx": si, "dataset": dataset.value,
                "sample_uid": uid, "split_group_uid": split_group,
                "split_bucket": split_bucket(split_group),
                "frame_idx": ds_asm.frame_index(si),
            }).encode()
            members["calib.json"] = (
                json.dumps(
                    ds_asm.reactive_calibration_for(
                        si,
                        image_size=image_size,
                    ),
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
                if dataset in (Dataset.KITSCENES, Dataset.L2D)
                else calib_bytes
            )

            for suffix, blob in members.items():
                _add_member(uid, suffix, blob)
            trajectory_xy_count += int("trajectory_xy.npz" in members)
            bev_segmentation_count += int(
                "bev_segmentation.npz" in members
            )
            reactive_navigation_count += int(
                "navigation_meta.json" in members
            )
            if _record_to_json is not None:
                record = labels_by_id.get(uid)
                if record is not None:
                    _add_member(uid, "reasoning.json",
                                json.dumps(_record_to_json(record)).encode())
                    reasoning_label_count += 1
            if current_shard_name is None:
                raise AssertionError("sample was packed without an open shard")
            shard_sample_counts[current_shard_name] += 1
            sample_count += 1
        if dataset == Dataset.L2D:
            navigation_artifact_summary[
                "map_image_registered_samples"
            ] = l2d_map_image_registered

    else:
        # Per-sample decode path (NVIDIA, or an empty
        # partition).
        pack_workers = _row_decode_worker_count(dataset, len(idx_list))
        print(f"Packing {len(idx_list)} samples, legacy mode "
              f"(world_model={world_model}, per-sample decode)...")
        pack_init = (dataset.value, ep_list, raw_path, image_size, world_model, calib_bytes)
        del ds
        with ProcessPoolExecutor(max_workers=pack_workers, mp_context=ctx,
                                 initializer=parallel_pack.init_pack_worker,
                                 initargs=pack_init) as pool:
            for sample_key, nviews, members, frame_pool in pool.map(
                    parallel_pack.pack_sample, idx_list):
                if sample_count % samples_per_shard == 0:
                    open_new_shard()
                for suffix, blob in members.items():
                    _add_member(sample_key, suffix, blob)
                for frame_id, blob in frame_pool.items():
                    _write_pool(frame_id, blob)
                num_views = nviews
                has_map = has_map or (
                    "map.jpg" in members
                    or "map_semantic.npz" in members
                )
                has_wm = has_wm or ("window_index.json" in members)
                trajectory_xy_count += int(
                    "trajectory_xy.npz" in members
                )
                bev_segmentation_count += int(
                    "bev_segmentation.npz" in members
                )
                reactive_navigation_count += int(
                    "navigation_meta.json" in members
                )
                geospatial_count += int(
                    "pose.npy" in members and "gps.npy" in members
                )
                if _record_to_json is not None:
                    record = labels_by_id.get(sample_key)
                    if record is not None:
                        _add_member(sample_key, "reasoning.json",
                                    json.dumps(_record_to_json(record)).encode())
                        reasoning_label_count += 1
                if current_shard_name is None:
                    raise AssertionError(
                        "sample was packed without an open shard"
                    )
                shard_sample_counts[current_shard_name] += 1
                sample_count += 1

    if current_tar:
        current_tar.close()
    import hashlib

    frame_pool_archive_name = None
    frame_pool_archive_sha256 = None
    frame_pool_archive_schema = None
    if pool_frames_written:
        frame_pool_archive_name = "frame_pool.tar"
        frame_pool_archive_schema = "frame_pool_archive_v1"
        frame_pool_archive_path = Path(
            out_dir,
            frame_pool_archive_name,
        )
        with tarfile.open(frame_pool_archive_path, "w") as archive:
            pool_paths = sorted(Path(pool_dir).glob("*.jpg"))
            if len(pool_paths) != pool_frames_written:
                raise ValueError(
                    "frame-pool file count differs from decoded frame count"
                )
            for pool_path in pool_paths:
                frame_name = pool_path.name
                if (
                    Path(frame_name).name != frame_name
                    or not frame_name.endswith(".jpg")
                ):
                    raise ValueError(
                        f"invalid frame-pool member name {frame_name!r}"
                    )
                payload = pool_path.read_bytes()
                member = tarfile.TarInfo(frame_name)
                member.size = len(payload)
                member.mtime = 0
                member.mode = 0o444
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                archive.addfile(member, io.BytesIO(payload))
        frame_pool_archive_sha256 = hashlib.sha256(
            frame_pool_archive_path.read_bytes()
        ).hexdigest()
        import shutil

        shutil.rmtree(pool_dir)
    else:
        Path(pool_dir).rmdir()

    shard_sha256 = {
        name: hashlib.sha256(
            Path(out_dir, name).read_bytes()
        ).hexdigest()
        for name in shard_names
    }
    if (
        dataset in (Dataset.KITSCENES, Dataset.L2D)
        and sample_count
        and (
            trajectory_xy_count != sample_count
            or (
                not mapless_kitscenes_test
                and reactive_navigation_count != sample_count
            )
        )
    ):
        raise ValueError(
            f"{dataset.value} Reactive target packing was incomplete: "
            f"trajectory={trajectory_xy_count}/{sample_count} "
            f"navigation={reactive_navigation_count}/{sample_count}"
        )

    from data_processing.contract_versions import contract_versions
    from data_processing.geospatial import (
        EPISODE_PATH_SCHEMA_VERSION,
        GPS_SCHEMA_VERSION,
        POSE_SCHEMA_VERSION,
    )
    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_ARTIFACT_VERSION,
        REACTIVE_NAVIGATION_ARTIFACT_VERSION,
        TRAJECTORY_XY_ARTIFACT_VERSION,
    )
    from navigation.geometry import (
        AUTOE2E_NAVIGATION_GEOMETRY,
        DEFAULT_NAVIGATION_GEOMETRY,
    )
    from navigation.supervision import (
        ROUTE_SUPERVISION_ARTIFACT_VERSION,
    )

    manifest = {"total_samples": sample_count, "shards": shard_idx,
                "shard_names": shard_names,
                "shard_sample_counts": shard_sample_counts,
                "shard_sha256": shard_sha256,
                "partition_id": partition_id or None,
                "split_group_uids": (
                    [f"kitscenes-{group_id}" for group_id in group_ids]
                    if dataset == Dataset.KITSCENES
                    and group_ids is not None
                    else None
                ),
                "hz": hz, "image_size": image_size, "dataset": dataset.value,
                "source_revision": source_revision,
                "source_split": source_split,
                "data_role": data_role,
                "input_track": (
                    "camera_only_missing_map_route"
                    if mapless_kitscenes_test
                    else "camera_map_route"
                ),
                "dataset_version": dataset_version,
                "episodes": _packed_episode_count(episodes, group_ids),
                "temporal_sampling": (
                    kitscenes_temporal_contract(
                        benchmark_protocol=benchmark_protocol,
                    )
                    if dataset == Dataset.KITSCENES
                    else None
                ),
                "contracts": contract_versions(),
                # Real cameras only; the map view is stored under map.jpg.
                "num_views": num_views if sample_count else 0,
                "camera_order": camera_order if sample_count else None,
                "camera_slots": camera_slots if sample_count else None,
                "front_camera_index": (
                    REACTIVE_FRONT_CAMERA_INDEX
                    if sample_count and dataset in (Dataset.KITSCENES, Dataset.L2D)
                    else None
                ),
                "front_camera_image_size": (
                    REACTIVE_FRONT_CAMERA_IMAGE_SIZE
                    if sample_count and dataset in (Dataset.KITSCENES, Dataset.L2D)
                    else None
                ),
                "front_camera_fpn_image_size": (
                    REACTIVE_CAMERA_IMAGE_SIZE
                    if sample_count and dataset in (Dataset.KITSCENES, Dataset.L2D)
                    else None
                ),
                "has_map": bool(sample_count) and has_map,
                "has_navigation": (
                    bool(sample_count)
                    and navigation_artifact_summary is not None
                ),
                "has_reactive_navigation": (
                    bool(sample_count)
                    and reactive_navigation_count == sample_count
                ),
                "reactive_navigation_count": reactive_navigation_count,
                "reactive_navigation_version": (
                    REACTIVE_NAVIGATION_ARTIFACT_VERSION
                    if reactive_navigation_count
                    else None
                ),
                "has_route_supervision": (
                    bool(sample_count)
                    and navigation_artifact_summary is not None
                ),
                "has_route_reconstruction": (
                    bool(sample_count)
                    and reactive_navigation_count == sample_count
                ),
                "route_supervision_version": (
                    ROUTE_SUPERVISION_ARTIFACT_VERSION
                    if (
                        sample_count
                        and navigation_artifact_summary is not None
                    )
                    else None
                ),
                "navigation": navigation_artifact_summary,
                "navigation_geometry": (
                    (
                        AUTOE2E_NAVIGATION_GEOMETRY.contract()
                        if dataset in (Dataset.KITSCENES, Dataset.L2D)
                        else DEFAULT_NAVIGATION_GEOMETRY.contract()
                    )
                    if navigation_artifact_summary is not None
                    else None
                ),
                "map_context_channels": (
                    14
                    if dataset == Dataset.KITSCENES
                    or navigation_artifact_summary is not None
                    else 3
                ),
                "route_channels": 2,
                "has_trajectory_xy": (
                    bool(sample_count)
                    and trajectory_xy_count == sample_count
                ),
                "trajectory_xy_count": trajectory_xy_count,
                "trajectory_xy_version": (
                    TRAJECTORY_XY_ARTIFACT_VERSION
                    if trajectory_xy_count
                    else None
                ),
                "has_bev_segmentation": (
                    bool(sample_count)
                    and bev_segmentation_count == sample_count
                ),
                "bev_segmentation_count": bev_segmentation_count,
                "bev_segmentation_version": (
                    BEV_SEGMENTATION_ARTIFACT_VERSION
                    if bev_segmentation_count
                    else None
                ),
                "has_world_model": bool(sample_count) and has_wm,
                "has_bevformer_history": (
                    bool(sample_count) and has_bevformer_history
                ),
                "frame_pool_archive": frame_pool_archive_name,
                "frame_pool_archive_schema": frame_pool_archive_schema,
                "frame_pool_archive_sha256": frame_pool_archive_sha256,
                "frame_pool_frame_count": pool_frames_written,
                "bevformer_temporal_contract": (
                    {
                        "frame_count": len(
                            REACTIVE_BEVFORMER_FRAME_OFFSETS
                        ),
                        "history_frame_count": (
                            REACTIVE_BEVFORMER_HISTORY_FRAMES
                        ),
                        "frame_interval_us": (
                            REACTIVE_BEVFORMER_FRAME_INTERVAL_US
                        ),
                        "frame_offsets": list(
                            REACTIVE_BEVFORMER_FRAME_OFFSETS
                        ),
                        "history_reference_frame": "current_ego",
                    }
                    if sample_count and has_bevformer_history
                    else None
                ),
                "temporal_frame_offsets": (
                    list(REACTIVE_BEVFORMER_FRAME_OFFSETS)
                    if sample_count and has_bevformer_history
                    else None
                ),
                "temporal_frame_interval_us": (
                    REACTIVE_BEVFORMER_FRAME_INTERVAL_US
                    if sample_count and has_bevformer_history
                    else None
                ),
                "has_reasoning_labels": reasoning_label_count > 0,
                "reasoning_label_count": reasoning_label_count,
                "has_gps": (
                    bool(sample_count)
                    and geospatial_count == sample_count
                ),
                "geospatial": {
                    "pose_schema": POSE_SCHEMA_VERSION,
                    "gps_schema": GPS_SCHEMA_VERSION,
                    "episode_path_schema": EPISODE_PATH_SCHEMA_VERSION,
                    "source_coordinate_dtype": "float32",
                    "stored_coordinate_dtype": "float64",
                    "timestamp_dtype": "int64_ns",
                    "summary": geo_summary,
                } if geo_summary is not None else None}

    # Manifest carries the rig projection for both single and merged loaders.
    if projection_spec is not None:
        manifest["projection"] = projection_spec
        manifest["geometry_type"] = projection_spec.get("type", "pinhole")
    else:
        manifest["geometry_type"] = "pseudo"

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    print(f"Processed {dataset.value}: {sample_count} samples → {shard_idx} shards")
    return FlyteDirectory(out_dir)


# ============================================================
# Task: Direct KITScenes test ingest + pack
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_data_prep_pod_template(),
    # data_processing already fits raw download plus packed output in 240Gi.
    # The direct path runs ingest first and deletes its source archive before
    # packing, so the two stages do not add their peak storage requirements.
    requests=Resources(cpu="15", mem="64Gi", ephemeral_storage="240Gi"),
    limits=Resources(cpu="15", mem="64Gi", ephemeral_storage="240Gi"),
    secret_requests=[Secret(
        group="hf-token",
        key="HF_TOKEN",
        mount_requirement=Secret.MountType.ENV_VAR,
    )],
    cache=True,
    cache_version=KITSCENES_TEST_DIRECT_PACK_CACHE_VERSION,
    # A retry after packing starts must download the source scene again. This is
    # the deliberate tradeoff for not publishing the raw tree to object storage.
    retries=2,
    environment={"_F_P_WRITE_CHUNK_SIZE": "8388608"},
)
def ingest_and_process_kitscenes_test_partition(
    group_ids: List[str],
    source_revision: str,
    dataset_version: str,
    image_size: int,
) -> Annotated[FlyteDirectory, BatchSize(4)]:
    """Pack one standard-test scene without staging raw files in object storage."""
    if len(group_ids) != 1:
        raise ValueError(
            "KITScenes test direct pack requires exactly one scene"
        )
    if source_revision != KITSCENES_SOURCE_REVISION:
        raise ValueError(
            "KITScenes test direct pack requires pinned source revision "
            f"{KITSCENES_SOURCE_REVISION!r}"
        )
    if dataset_version != KITSCENES_TEST_DATASET_VERSION:
        raise ValueError(
            "KITScenes test direct pack requires dataset version "
            f"{KITSCENES_TEST_DATASET_VERSION!r}"
        )
    if image_size != REACTIVE_CAMERA_IMAGE_SIZE:
        raise ValueError(
            "KITScenes test direct pack requires the 512px camera contract"
        )

    raw_data = data_ingest.task_function(
        dataset=Dataset.KITSCENES,
        source_revision=source_revision,
        episodes=0,
        group_ids=group_ids,
        source_split="test",
        data_role="benchmark",
    )
    # Packing is offline. Do not expose the Hub token to row-worker subprocesses.
    import os

    os.environ.pop("HF_TOKEN", None)
    return data_processing.task_function(
        raw_data=raw_data,
        dataset=Dataset.KITSCENES,
        source_revision=source_revision,
        dataset_version=dataset_version,
        hz=10,
        image_size=image_size,
        episodes=0,
        world_model=False,
        reasoning_labels=None,
        group_ids=group_ids,
        sample_limit=0,
        source_split="test",
        data_role="benchmark",
    )


# ============================================================
# Task: KITScenes navigation quality audit
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_data_prep_pod_template(),
    requests=Resources(cpu="1", mem="2Gi", ephemeral_storage="2Gi"),
    limits=Resources(cpu="1", mem="2Gi", ephemeral_storage="2Gi"),
    cache=True,
    cache_version=NAVIGATION_QUALITY_CACHE_VERSION,
)
def audit_kitscenes_navigation_quality(
    shards: List[FlyteDirectory],
) -> FlyteFile:
    """Create a hash-bound route-quality gate before KITScenes training."""
    import json
    import os
    import shutil
    import tempfile
    from pathlib import Path, PurePosixPath

    from navigation.quality import audit_packed_navigation_quality

    view_root = Path(tempfile.mkdtemp(prefix="navigation-quality-input-"))
    shard_dirs = []
    for index, shard in enumerate(shards):
        remote_source = str(getattr(shard, "remote_source", "") or "")
        if "://" not in remote_source:
            shard_dirs.append(_loader_download_dir(shard))
            continue

        destination = view_root / f"partition-{index:06d}"
        destination.mkdir()
        base_uri = remote_source.rstrip("/")
        manifest_source = FlyteFile(
            f"{base_uri}/manifest.json"
        ).download()
        manifest_path = destination / "manifest.json"
        shutil.copyfile(manifest_source, manifest_path)
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if int(manifest.get("total_samples", 0)) > 0:
            scenes = (manifest.get("navigation") or {}).get("scenes")
            if not isinstance(scenes, list) or not scenes:
                raise ValueError(
                    "packed partition has no navigation scenes"
                )
            for scene in scenes:
                relative = PurePosixPath(str(scene.get("path", "")))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        "navigation quality path escapes packed partition"
                    )
                relative_parts = [
                    part for part in relative.parts if part != "."
                ]
                remote_parts = [
                    base_uri,
                    *relative_parts,
                    "navigation_quality.json",
                ]
                quality_source = FlyteFile(
                    "/".join(remote_parts)
                ).download()
                quality_destination = destination.joinpath(
                    *relative_parts,
                    "navigation_quality.json",
                )
                quality_destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                shutil.copyfile(
                    quality_source,
                    quality_destination,
                )
        shard_dirs.append(str(destination))

    report = audit_packed_navigation_quality(shard_dirs)
    output_dir = tempfile.mkdtemp(prefix="navigation-quality-")
    output_path = os.path.join(
        output_dir,
        "navigation_quality_audit.json",
    )
    with open(output_path, "w", encoding="ascii") as stream:
        json.dump(
            report,
            stream,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(
        "KITScenes navigation quality: "
        f"accepted={report['accepted_scene_count']} "
        f"excluded={report['excluded_scene_count']} "
        f"report={output_path}"
    )
    return FlyteFile(output_path)


# ============================================================
# Task: Reasoning label generation (offline teacher → versioned S3 artifact)
#
# This is the only place the teacher (Cosmos) is called. Each sample's global
# uid (parser.sample_uid) is the JOIN key to data_processing and is stable across
# episode-range shards. The partition's labels are written to one records.jsonl
# artifact; the Flyte task cache on the deterministic partition prevents
# re-billing on a re-run.
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_data_prep_pod_template(),
    # Process-parallel front-clip decode overlaps the remote teacher calls.
    # KITScenes uses two workers over one scene per pod, so 64Gi has ample decode
    # headroom while keeping each pod schedulable on a 16-vCPU node. The 60Gi
    # disk request holds the materialized scene throughout teacher calls.
    requests=Resources(cpu="15", mem="64Gi", ephemeral_storage="60Gi"),
    limits=Resources(cpu="15", mem="64Gi", ephemeral_storage="60Gi"),
    # The openai_compatible teacher endpoint (e.g. the Cosmos3-Nano vLLM ALB) is
    # injected from a K8s Secret so no concrete URL / account value is committed
    # to git or shown in the Flyte UI. Optional: only consumed when
    # teacher="openai_compatible" (mock/cached ignore it).
    secret_requests=[
        Secret(group="cosmos-teacher", key="COSMOS_TEACHER_BASE_URL",
               mount_requirement=Secret.MountType.ENV_VAR),
        Secret(group="cosmos-teacher", key="COSMOS_TEACHER_MODEL",
               mount_requirement=Secret.MountType.ENV_VAR),
    ],
    # Cache on (raw URI, group_ids, teacher, prompt_version, cache_version); this
    # is the only re-label protection, so an unchanged partition never re-bills
    # Cosmos. LABEL_CACHE_VERSION folds in the uid format (the JOIN key) and
    # sparse-selection policy. label_workers is pure parallelism
    # (output-invariant), so it is excluded from the key.
    cache=True,
    cache_version=LABEL_CACHE_VERSION,
    cache_ignore_input_vars=("label_workers",),
    # The teacher call is idempotent, so retrying a transient endpoint error is
    # safe.
    retries=2,
    # Same fsspec upload-chunk cap as data_ingest (see comment there).
    environment={"_F_P_WRITE_CHUNK_SIZE": "8388608"},
)
def generate_reasoning_labels(
    raw_data: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    source_revision: str = L2D_SOURCE_REVISION,
    episodes: int = 3,
    split: str = "train",
    teacher: str = "openai_compatible",
    prompt_version: str = "action_relevant_reasoning_v3_temporal_front256",
    group_ids: Optional[List[str]] = None,
    # Reasoning is a 1 Hz concern (horizons 0/1/2/3/4 s), so label the stable
    # frame_index % label_stride grid plus the first valid sample of every split
    # group. The one-sample bootstrap covers short scenes that fall entirely
    # between grid points while preserving partition independence. L2D and
    # KITScenes are 10 Hz, so stride 10 remains approximately 1 Hz and cuts
    # Cosmos calls by about 10x. Unlabeled samples decode as fully-masked targets.
    # stride=1 labels every sample.
    label_stride: int = 10,
    # Process-parallel worker count. The stage is bound by teacher latency
    # (~12s per call), so 6 workers overlap it well; lerobot reader memory scales
    # with loaded episode count, and more workers can exceed the pod limit.
    label_workers: int = 6,
) -> Annotated[FlyteDirectory, BatchSize(4)]:
    """Label sparse 1 Hz samples with a temporal front-camera clip, then write a
    versioned label artifact for the data_processing JOIN.

    The teacher sees one front-camera frame per horizon (0 s current + 1/2/3/4 s
    future). Datasets expose ``get_front_clip(idx)`` in a ``reasoning_clip_only``
    mode that decodes only those 5 front frames and does not change the sample
    set. Labels are keyed by the parser's global ``sample_uid``, so the JOIN to
    ``data_processing`` holds across different episode-range shards.

    Returns:
        FlyteDirectory with a whole-record ``records.jsonl`` (the JOIN
        interchange data_processing reads), the flattened
        ``reasoning_labels_v2.{parquet,jsonl}`` analytics export, and a
        provenance ``meta.json``.
    """
    import json
    import os
    import tempfile

    # The teacher and dataset live in the per-process workers (parallel_label);
    # teacher_kwargs is assembled here from the Flyte secret context.
    from data_processing.reasoning_label_generation.parquet_writer import (
        write_jsonl, write_parquet,
    )
    from data_processing.reasoning_label_generation.targets import write_records_jsonl

    raw_path = _download_flyte_directory(raw_data)
    print(f"Generating reasoning labels: dataset={dataset.value} split={split} "
          f"teacher={teacher} prompt={prompt_version} raw={raw_path}")

    # Build the dataset once (front-clip mode) to select sample indices.
    # Enumeration matches data_processing so sample_ids JOIN; workers rebuild
    # their own front-clip dataset in init_worker.
    if dataset == Dataset.KITSCENES:
        if source_revision != KITSCENES_SOURCE_REVISION:
            raise ValueError(
                "KITScenes labeling requires pinned source revision "
                f"{KITSCENES_SOURCE_REVISION}, got {source_revision!r}"
            )
        ep_list = (
            [str(group_id) for group_id in group_ids]
            if group_ids is not None
            else None
        )
    else:
        if dataset == Dataset.L2D and source_revision != L2D_SOURCE_REVISION:
            raise ValueError(
                "L2D labeling requires the audited pinned commit "
                f"{L2D_SOURCE_REVISION}; got {source_revision!r}"
            )
        ep_list = ([int(g) for g in group_ids] if group_ids is not None
                   else (list(range(episodes)) if episodes > 0 else None))
    # A fan-out partition can legitimately contain no valid samples, e.g. a
    # short L2D episode below the egomotion margin (64+64+1 frames). Treat it as
    # success with an empty label artifact rather than failing the fan-out.
    try:
        if dataset == Dataset.NVIDIA_PHYSICAL_AI:
            from data_parsing.nvidia_physical_ai.dataset import NvidiaAVDataset
            # Discover from raw_path (only this partition's clips) so this probe
            # and parallel_label.init_worker enumerate in the same order. Passing
            # clip_uuids in partition order would risk an order mismatch.
            ds = NvidiaAVDataset(data_root=raw_path, reasoning_clip_only=True)
        elif dataset == Dataset.KITSCENES:
            from data_parsing.kit_scenes import KitScenesDataset
            ds = KitScenesDataset(
                data_root=raw_path,
                split=split,
                scene_ids=ep_list,
                reasoning_clip_only=True,
            )
        else:
            from data_parsing.l2d import L2DDataset
            ds = L2DDataset(
                repo_id=dataset.value,
                revision=source_revision,
                episodes=ep_list,
                reasoning_clip_only=True,
                root=raw_path,
            )
        n_samples = len(ds)
        label_indices = _reasoning_label_indices(ds, label_stride)
    except ValueError as e:
        if "No valid samples" not in str(e):
            raise
        print(f"Partition has no valid samples ({e}); writing an EMPTY label "
              f"artifact (short episode/clip — nothing to label).")
        ds = None
        n_samples = 0
        label_indices = []

    # openai_compatible resolves base_url/model/api_key from the Secret (env
    # fallback); mock/cached need none of these.
    teacher_kwargs = {}
    if teacher == "openai_compatible":
        from flytekit import current_context

        def _secret(key, default=None):
            try:
                return current_context().secrets.get("cosmos-teacher", key)
            except Exception:
                return os.environ.get(key, default)

        base_url = _secret("COSMOS_TEACHER_BASE_URL")
        if not base_url:
            raise ValueError(
                "teacher='openai_compatible' needs COSMOS_TEACHER_BASE_URL "
                "(cosmos-teacher K8s Secret / env); none found."
            )
        teacher_kwargs = {
            "base_url": base_url,
            "model": _secret("COSMOS_TEACHER_MODEL", "nvidia/Cosmos3-Nano"),
        }
        api_key = _secret("COSMOS_TEACHER_API_KEY")
        if api_key:
            teacher_kwargs["api_key"] = api_key
    teacher_kwargs["prompt_version"] = prompt_version
    # strict=False for bulk offline labeling: a malformed response or one with
    # fewer than 5 horizons becomes an abstained record (masked out of the
    # reasoning loss) instead of failing the run. meta.json reports
    # num_abstained, and a high rate fails below.
    teacher_kwargs["strict"] = False

    # Free the parent's dataset handle: each worker process builds its own.
    del ds

    from data_processing.reasoning_label_generation.targets import record_from_json

    n_computed = n_abstain = 0
    if not label_indices:
        # Empty partition, or nothing selected by the stride: empty artifact.
        records = []
    else:
        # Process-parallel labeling, not threads: lerobot's reader is not
        # thread-safe. Each worker owns an independent dataset + reader, so decode
        # runs in parallel and the teacher calls overlap. Only the sample index
        # crosses the process boundary. Spawn context because torch is imported.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from data_processing.reasoning_label_generation import parallel_label

        workers = max(1, min(label_workers, len(label_indices)))
        print(f"Labeling {len(label_indices)}/{n_samples} samples (sparse subset, "
              f"stride={label_stride}) with {workers} parallel PROCESSES "
              f"(teacher={teacher})...")
        ctx = mp.get_context("spawn")
        # Order must match parallel_label.init_worker(repo_id, episodes,
        # dataset_name, teacher, teacher_kwargs, prompt_version, raw_path).
        init_args = (dataset.value, ep_list, dataset.value, teacher, teacher_kwargs,
                     prompt_version, raw_path)
        records = []
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx,
            initializer=parallel_label.init_worker, initargs=init_args,
        ) as pool:
            for si, rec_json, status in pool.map(parallel_label.label_sample,
                                                 label_indices):
                records.append(record_from_json(rec_json))
                if status == "abstained":
                    n_abstain += 1
                else:
                    n_computed += 1
    print(f"Labeled {len(records)} samples "
          f"(computed={n_computed}, abstained={n_abstain})")
    # A few abstentions are masked out of the reasoning loss; a high rate means a
    # systemic prompt/model problem.
    if records and n_abstain > 0.5 * len(records):
        raise RuntimeError(
            f"{n_abstain}/{len(records)} samples abstained (>50%) — the teacher "
            f"is failing systematically (prompt/model/endpoint), not just on a few "
            f"hard frames. Aborting so the problem is fixed rather than masked.")

    out_dir = tempfile.mkdtemp()
    layout = os.path.join(
        out_dir, f"dataset={dataset.value}", f"split={split}",
        "schema_version=reasoning_label_v2", f"teacher={teacher}",
    )
    os.makedirs(layout, exist_ok=True)
    # records.jsonl = whole-record JOIN interchange data_processing reads back.
    write_records_jsonl(records, os.path.join(layout, "records.jsonl"))
    # Flattened analytics export (per-horizon rows) for querying/diffing.
    write_jsonl(records, os.path.join(layout, "reasoning_labels_v2.jsonl"))
    write_parquet(records, os.path.join(layout, "reasoning_labels_v2.parquet"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"dataset": dataset.value, "split": split, "teacher": teacher,
                   "source_revision": source_revision,
                   "prompt_version": prompt_version, "num_records": len(records),
                   "label_policy_version": _LABEL_POLICY_V,
                   "computed": n_computed, "num_abstained": n_abstain,
                   "source": "offline teacher (generate_reasoning_labels); "
                             "records.jsonl artifact, Flyte task-cached per partition"}, f)
    print(f"Wrote reasoning label artifact → {layout}")
    return FlyteDirectory(out_dir)


@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_data_prep_pod_template(),
    requests=Resources(cpu="1", mem="2Gi", ephemeral_storage="2Gi"),
    limits=Resources(cpu="1", mem="2Gi", ephemeral_storage="2Gi"),
    secret_requests=[
        Secret(
            group="hf-token",
            key="HF_TOKEN",
            mount_requirement=Secret.MountType.ENV_VAR,
        )
    ],
    cache=True,
    cache_version="kitscenes-benchmark-inventory-v1",
    retries=2,
)
def audit_kitscenes_benchmark_inventory() -> FlyteFile:
    """Audit the pinned val/overlap archive inventory without downloading it."""
    import json
    import os
    import tempfile

    from flytekit import current_context

    from data_parsing.kit_scenes.source import (
        KITSCENES_SDK_REVISION,
        fetch_archive_manifest,
        resolve_inventory,
    )

    try:
        token = current_context().secrets.get("hf-token", "HF_TOKEN")
    except Exception:
        token = os.environ.get("HF_TOKEN", "")
    with tempfile.TemporaryDirectory(
        prefix="kitscenes_benchmark_inventory_"
    ) as tmp:
        archives = fetch_archive_manifest(
            tmp,
            revision=KITSCENES_SOURCE_REVISION,
            token=token or None,
        )

    splits = {}
    total_size_bytes = 0
    total_scene_count = 0
    for source_split in ("val", "overlap_train_val"):
        inventory = resolve_inventory(
            archives,
            split=source_split,
            source_revision=KITSCENES_SOURCE_REVISION,
            max_missing_scenes=0,
        )
        scene_records = [
            {
                "archive_path": archives[scene_id].filename,
                "archive_sha256": archives[scene_id].sha256,
                "archive_size_bytes": archives[scene_id].size_bytes,
                "scene_id": scene_id,
            }
            for scene_id in inventory.selected_scene_ids
        ]
        split_size = sum(
            int(record["archive_size_bytes"]) for record in scene_records
        )
        splits[source_split] = {
            **inventory.metadata(),
            "archives": scene_records,
            "total_size_bytes": split_size,
        }
        total_size_bytes += split_size
        total_scene_count += len(scene_records)

    report = {
        "dataset": Dataset.KITSCENES.value,
        "dataset_revision": KITSCENES_SOURCE_REVISION,
        "sdk_revision": KITSCENES_SDK_REVISION,
        "schema_version": "kitscenes_benchmark_inventory_v1",
        "splits": splits,
        "total_scene_count": total_scene_count,
        "total_size_bytes": total_size_bytes,
    }
    output_dir = Path("/tmp/kitscenes-benchmark-inventory")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "inventory.json"
    output_path.write_text(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(
        "KITScenes benchmark inventory: "
        f"scenes={total_scene_count} bytes={total_size_bytes}"
    )
    return FlyteFile(os.fspath(output_path))


# ============================================================
# Workflows
# ============================================================
@dynamic(
    container_image=DATA_PREP_IMAGE,
    environment={"AUTO_E2E_DATA_PREP_IMAGE": DATA_PREP_IMAGE},
)
def _map_dataset_partitions(
    partitions: List[List[str]],
    dataset: Dataset,
    source_revision: str,
    dataset_version: str,
    image_size: int,
    world_model: bool,
    reasoning_teacher: str,
    prompt_version: str,
    label_stride: int,
    label_workers: int,
    ingest_concurrency: int,
    label_concurrency: int,
    pack_concurrency: int,
    total_sample_limit: int,
    source_split: str = "train",
    data_role: str = "training",
) -> List[FlyteDirectory]:
    """Execute each data-prep stage as one bounded Flyte array node."""
    for name, value in (
        ("ingest_concurrency", ingest_concurrency),
        ("label_concurrency", label_concurrency),
        ("pack_concurrency", pack_concurrency),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if total_sample_limit < 0:
        raise ValueError(
            "total_sample_limit must be non-negative, got "
            f"{total_sample_limit}"
        )
    if total_sample_limit and dataset != Dataset.L2D:
        raise ValueError(
            "bounded full-pack sampling is currently supported only for L2D"
        )
    if total_sample_limit and reasoning_teacher != "none":
        raise ValueError(
            "bounded full-pack sampling requires reasoning_teacher='none'"
        )
    if reasoning_teacher != "none" and data_role != "training":
        raise ValueError(
            "benchmark dataset preparation must not invoke reasoning teachers"
        )
    sample_limits = (
        _allocate_partition_sample_limits(
            partitions,
            total_sample_limit,
        )
        if total_sample_limit
        else [0] * len(partitions)
    )

    ingest = map_task(
        functools.partial(
            data_ingest,
            dataset=dataset,
            source_revision=source_revision,
            episodes=0,
            source_split=source_split,
            data_role=data_role,
        ),
        concurrency=ingest_concurrency,
    )
    raw_dirs = ingest(group_ids=partitions)
    large_l2d_pack = (
        dataset == Dataset.L2D
        and max(map(len, partitions), default=0) >= 50
    )

    def apply_pack_resources(mapped_pack: Any) -> Any:
        if not large_l2d_pack:
            return mapped_pack
        return mapped_pack.with_overrides(
            requests=Resources(
                cpu="15",
                mem="128Gi",
                ephemeral_storage="800Gi",
            ),
            limits=Resources(
                cpu="15",
                mem="128Gi",
                ephemeral_storage="800Gi",
            ),
        )

    if reasoning_teacher != "none":
        label = map_task(
            functools.partial(
                generate_reasoning_labels,
                dataset=dataset,
                source_revision=source_revision,
                episodes=0,
                split="train",
                teacher=reasoning_teacher,
                prompt_version=prompt_version,
                label_stride=label_stride,
                label_workers=label_workers,
            ),
            concurrency=label_concurrency,
        )
        label_dirs = label(raw_data=raw_dirs, group_ids=partitions)
        pack = map_task(
            functools.partial(
                data_processing,
                dataset=dataset,
                source_revision=source_revision,
                dataset_version=dataset_version,
                hz=10,
                image_size=image_size,
                episodes=0,
                world_model=True,
                source_split=source_split,
                data_role=data_role,
                # map_task needs every input bound; the fan-out has no OSM
                # region, so an L2D pack here fails with a clear error.
                osm_region_path="",
                osm_region_sha256="",
            ),
            concurrency=pack_concurrency,
        )
        return apply_pack_resources(pack(
            raw_data=raw_dirs,
            reasoning_labels=label_dirs,
            group_ids=partitions,
            sample_limit=sample_limits,
        ))

    pack = map_task(
        functools.partial(
            data_processing,
            dataset=dataset,
            source_revision=source_revision,
            dataset_version=dataset_version,
            hz=10,
            image_size=image_size,
            episodes=0,
            world_model=world_model,
            reasoning_labels=None,
            source_split=source_split,
            data_role=data_role,
            osm_region_path="",
            osm_region_sha256="",
        ),
        concurrency=pack_concurrency,
    )
    return apply_pack_resources(pack(
        raw_data=raw_dirs,
        group_ids=partitions,
        sample_limit=sample_limits,
    ))


@dynamic(
    container_image=DATA_PREP_IMAGE,
    environment={"AUTO_E2E_DATA_PREP_IMAGE": DATA_PREP_IMAGE},
)
def _map_kitscenes_test_partitions(
    partitions: List[List[str]],
    dataset_version: str,
    image_size: int,
    ingest_concurrency: int,
    pack_concurrency: int,
) -> List[FlyteDirectory]:
    """Pack standard-test scenes in one task per scene."""
    for name, value in (
        ("ingest_concurrency", ingest_concurrency),
        ("pack_concurrency", pack_concurrency),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if not partitions or any(len(partition) != 1 for partition in partitions):
        raise ValueError(
            "KITScenes test partitions must contain exactly one scene each"
        )
    if dataset_version != KITSCENES_TEST_DATASET_VERSION:
        raise ValueError(
            "KITScenes test direct pack requires dataset version "
            f"{KITSCENES_TEST_DATASET_VERSION!r}"
        )
    if image_size != REACTIVE_CAMERA_IMAGE_SIZE:
        raise ValueError(
            "KITScenes test direct pack requires the 512px camera contract"
        )

    direct_pack = map_task(
        functools.partial(
            ingest_and_process_kitscenes_test_partition,
            source_revision=KITSCENES_SOURCE_REVISION,
            dataset_version=dataset_version,
            image_size=image_size,
        ),
        concurrency=min(ingest_concurrency, pack_concurrency),
    )
    return direct_pack(group_ids=partitions)

