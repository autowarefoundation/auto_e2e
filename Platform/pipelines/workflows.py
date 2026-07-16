"""AutoE2E Flyte-native workflows — Real Training Pipeline.

Architecture:
  data_ingest → data_processing → train_il → evaluate
                                      ↓
                              train_offline_rl → evaluate

MLflow: Only evaluate task logs. 2 experiments: imitation-learning, offline-rl.
"""
import enum
import functools
from flytekit import (
    task, workflow, dynamic, map_task, Resources, Secret, BatchSize,
)
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from typing import Annotated, NamedTuple, List, Optional

from Platform.pipelines.dataset_publication import DatasetPublication

import os as _os

ECR_PREFIX = _os.environ.get("ECR_PREFIX", "381491877296.dkr.ecr.us-west-2.amazonaws.com")
TRAINING_IMAGE = _os.environ.get(
    "AUTO_E2E_TRAINING_IMAGE",
    f"{ECR_PREFIX}/auto-e2e/training:latest",
)
EVAL_IMAGE = _os.environ.get(
    "AUTO_E2E_EVAL_IMAGE",
    f"{ECR_PREFIX}/auto-e2e/eval:latest",
)
OFFLINE_RL_IMAGE = _os.environ.get(
    "AUTO_E2E_OFFLINE_RL_IMAGE",
    f"{ECR_PREFIX}/auto-e2e/offline-rl:latest",
)
DATA_PREP_IMAGE = _os.environ.get(
    "AUTO_E2E_DATA_PREP_IMAGE",
    f"{ECR_PREFIX}/auto-e2e/data-prep:latest",
)

MLFLOW_URI = "http://mlflow.mlflow.svc.cluster.local:5000"
DATASET_PACK_VERSION = "v2.1"
L2D_SOURCE_REVISION = "main"
KITSCENES_SOURCE_REVISION = "6fde0034446669e2ed7235e4c7fe323cd23d599d"

# The per-sample S3 label cache is REMOVED (#121 §3.4): at full L2D it was ~10M
# tiny JSON objects (inode/quota/copy-rate blowup). The teacher is now called once
# per (deterministic) partition and its records aggregate into one records.jsonl;
# re-run protection is the Flyte task cache on the deterministic partition, so an
# unchanged range never re-bills Cosmos.

# Flyte cache versions (#121 §3.4a). The cache key is (task interface, input
# literals, cache_version); the CODE-contract determinants (uid/parser/shard/
# geometry schema) can't be captured by inputs, so they go here. Sourced from
# Model/data_processing/contract_versions.py (the single place any of these is
# bumped, §3.4c). Imported guarded: Model is on the path in the data-prep image
# and on the dev box, but NOT necessarily when this module is first imported at
# registration — the fallback keeps registration working (the real values load
# in the pod where the tasks actually run and cache). Per-partition group_ids and
# source_revision travel as task INPUTS, so ranges are independently cacheable.
try:
    from data_processing.contract_versions import (
        UID_SCHEMA_VERSION as _UID_V, PARSER_VERSION as _PARSER_V,
        SHARD_SCHEMA_VERSION as _SHARD_V, GEOMETRY_VERSION as _GEOM_V,
        REASONING_LABEL_POLICY_VERSION as _LABEL_POLICY_V,
    )
except Exception:  # pragma: no cover - registration-time fallback only
    _UID_V = _PARSER_V = _SHARD_V = _GEOM_V = "v1"
    _LABEL_POLICY_V = "v1"

# Each stage's cache_version folds in ONLY the contracts that actually determine
# its output (§3.4a): ingest depends on the parser enumeration; labels also on
# the uid format (the JOIN key) and sparse-selection policy; pack on the shard
# and geometry encoding.
INGEST_CACHE_VERSION = f"ingest-{_PARSER_V}"
LABEL_CACHE_VERSION = f"label-{_PARSER_V}-{_UID_V}-{_LABEL_POLICY_V}"
PACK_CACHE_VERSION = f"pack-{_PARSER_V}-{_UID_V}-{_SHARD_V}-{_GEOM_V}"


def _data_prep_pod_template():
    """Protect active data-prep pods from voluntary Karpenter disruption."""
    from flytekit import PodTemplate

    return PodTemplate(
        annotations={"karpenter.sh/do-not-disrupt": "true"},
    )


def _large_shm_pod_template():
    """PodTemplate that mounts a large tmpfs at /dev/shm (#121 P0).

    DataLoader workers (num_workers>0) transport batches to the parent through
    shared memory; the default Kubernetes pod /dev/shm is only ~64MB, so
    WM-window batches overflow it and workers die with "Bus error / worker killed
    by signal". A `Memory`-backed emptyDir at /dev/shm gives the workers real
    shared memory (sized from the pod's mem limit), which is the documented fix.
    Built lazily so importing this module never requires the k8s client models.
    """
    from flytekit import PodTemplate
    from kubernetes.client import (
        V1PodSpec, V1Container, V1Volume, V1VolumeMount, V1EmptyDirVolumeSource,
    )
    return PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="primary",
                    volume_mounts=[V1VolumeMount(name="dshm", mount_path="/dev/shm")],
                )
            ],
            volumes=[
                V1Volume(
                    name="dshm",
                    empty_dir=V1EmptyDirVolumeSource(
                        medium="Memory", size_limit="8Gi"),
                )
            ],
        ),
    )


# --- Enums ---
class Dataset(enum.Enum):
    L2D = "yaak-ai/L2D"
    KITSCENES = "KIT-MRT/KITScenes-Multimodal"
    NVIDIA_PHYSICAL_AI = "nvidia/PhysicalAI-Autonomous-Vehicles"


class Backbone(enum.Enum):
    SWIN_V2_TINY = "swin_v2_tiny"
    CONVNEXT_V2_TINY = "conv_next_v2_tiny"
    RESNET_50 = "res_net_50"


# NOTE: view fusion is no longer selectable. The reactive-refactor (PR #94)
# removed concat/cross_attn and hardcoded BEV fusion inside ReactiveE2E, and
# dropped the `fusion_mode` argument from AutoE2E.__init__. We keep the string
# "bev" only as a metadata label so MLflow runs stay comparable with old runs.
FUSION_LABEL = "bev"

TrainOutput = NamedTuple("TrainOutput", checkpoint=FlyteFile, metadata=FlyteFile)
EvalMetrics = NamedTuple("EvalMetrics", ade=float, fde=float, gate_pass=bool)
PublishedOverlayOutput = NamedTuple(
    "PublishedOverlayOutput",
    overlay_result=str,
    manifest_key=str,
    manifest_sha256=str,
)
# wf_create_dataset returns just the ready-to-train WebDataset shards (train_il
# reads reasoning supervision from in-shard reasoning.json members). The
# versioned reasoning-label artifact persists independently in S3 (the
# generate_reasoning_labels task output + the sample_id-keyed cache), so it is
# not a workflow return value.


def _model_kwargs(config: dict) -> dict:
    """Filter a saved checkpoint `config` down to kwargs the current AutoE2E
    accepts. The reactive refactor (PR #94) removed `fusion_mode`, but old
    checkpoints (and our own metadata) may still carry it, which would make
    `AutoE2E(**config)` raise. Drop any keys the constructor no longer takes.
    """
    import inspect
    from model_components.auto_e2e import AutoE2E
    valid = set(inspect.signature(AutoE2E.__init__).parameters) - {"self"}
    return {k: v for k, v in config.items() if k in valid}


def _select_shard_dir(shards, dataset) -> str:
    """Download all shard FlyteDirectories and return the local path of the one
    whose manifest matches `dataset`.

    All datasets are passed in (each a separately-packed WebDataset), but only
    the selected dataset is used for this run. Multi-dataset training of a single
    model is tracked in issue #77 (requires dynamic-num_views BEV fusion).
    """
    import os
    import json
    target = dataset.value
    fallback = None
    for sh in shards:
        d = sh.download()
        mpath = os.path.join(str(d), "manifest.json")
        if os.path.exists(mpath):
            try:
                manifest = json.load(open(mpath))
                if int(manifest.get("total_samples", 0)) <= 0:
                    continue
                fallback = fallback or d
                if manifest.get("dataset") == target:
                    print(f"Selected shards for dataset={target}: {d}")
                    return d
            except Exception:
                pass
    if fallback is None:
        raise RuntimeError(
            f"_select_shard_dir: no non-empty shard dir matched dataset={target}"
        )
    print(f"WARN: no shards matched dataset={target}; using first ({fallback})")
    return fallback


def _select_shard_dirs(shards, dataset) -> List[str]:
    """Download ALL shard FlyteDirectories whose manifest matches `dataset`.

    Sharded fan-out returns N per-partition dirs (one per partition), all with
    the same ``dataset`` in their manifest. The eval task must consume ALL of
    them so ADE/FDE reflects the whole held-out set, not partition 0 only
    (Flyte-review B2 fix — the single-dir _select_shard_dir was silently
    collapsing sharded eval to 1/N of val).
    """
    import os
    import json
    target = dataset.value
    matched: List[str] = []
    skipped_empty = 0
    for sh in shards:
        d = sh.download()
        mpath = os.path.join(str(d), "manifest.json")
        if os.path.exists(mpath):
            try:
                manifest = json.load(open(mpath))
                if manifest.get("dataset") != target:
                    continue
                if int(manifest.get("total_samples", 0)) <= 0:
                    skipped_empty += 1
                else:
                    matched.append(str(d))
            except Exception:
                pass
    if not matched:
        raise RuntimeError(
            f"_select_shard_dirs: no shard dirs matched dataset={target} "
            f"(had {len(shards)} shards)")
    print(
        f"Selected {len(matched)} non-empty shard dirs for dataset={target}; "
        f"skipped_empty={skipped_empty}"
    )
    return matched


def _loader_download_dir(shard) -> str:
    """Download one shard FlyteDirectory and return its local path (merged path)."""
    return str(shard.download())


def _loader_projection(loader, device):
    """Return the loader's per-dataset projection operator on ``device``.

    Geometry is a rig constant exposed on the loader (``.projection`` /
    ``.geometry_type``) by make_pre_extracted_loader, not per batch. Datasets
    without calibration expose ``projection=None`` + ``geometry_type='pseudo'``,
    so we run the explicit pseudo path — never a silent real-geometry claim.
    """
    projection = getattr(loader, "projection", None)
    geometry_type = getattr(loader, "geometry_type", "pseudo")
    if projection is not None:
        projection = projection.to(device)
    return projection, geometry_type


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
) -> List[List[str]]:
    """Resolve source groups once and return deterministic mapped-task inputs.

    KITScenes is intentionally one scene per partition. The pinned SDK's official
    split is reconciled with the pinned Hugging Face archive manifest before any
    large pod is launched. The v1.0.1 one-scene deficit is allowed only when it
    stays within ``max_missing_scenes``; any second deficit or unexpected scene
    fails the workflow at preflight.
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
        if source_revision != KITSCENES_SOURCE_REVISION:
            raise ValueError(
                "KITScenes source_revision must match the audited pinned "
                f"revision {KITSCENES_SOURCE_REVISION}, got {source_revision!r}"
            )
        if split != "train":
            raise ValueError(
                "The full training fan-out currently accepts only the official "
                f"KITScenes train split, got {split!r}"
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
        if source_revision != L2D_SOURCE_REVISION:
            raise ValueError(
                "L2D currently supports only revision='main' because the v3.0 "
                f"tag is stale; got {source_revision!r}"
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
    # KITScenes production fan-out is one scene per pod. Its largest pinned
    # archive is 20.12 GiB; download+extract briefly holds about twice that, so
    # 60Gi fits the EKS Auto Mode default NodeClass (~70Gi allocatable disk).
    # 15 vCPU stays below a 16-vCPU node's kube-reserved allocatable boundary;
    # 64Gi memory fits comfortably on that node. Sixty pods request 900 vCPU.
    # L2D's old multi-episode 128Gi/800Gi
    # profile is intentionally deferred with that dataset's full run.
    requests=Resources(cpu="15", mem="64Gi", ephemeral_storage="60Gi"),
    limits=Resources(cpu="15", mem="64Gi", ephemeral_storage="60Gi"),
    secret_requests=[Secret(group="hf-token", key="HF_TOKEN",
                            mount_requirement=Secret.MountType.ENV_VAR)],
    # "Ingest once, never again" (#121 §3.4a): cache on (dataset, group_ids,
    # episodes, cache_version). A partition's raw is fetched from HF/SDK exactly
    # once; a re-run of the same partition is a cache no-op, and its stable output
    # URI lets the downstream label/pack cache hit too (FlyteDirectory hashes by
    # URI). The HF token is a secret env, NOT an input, so it never enters the key.
    cache=True,
    cache_version=INGEST_CACHE_VERSION,
    # 100-partition fan-out: a single transient HF 503 or Karpenter provisioning
    # blip would abort the WHOLE workflow without retries. 2 attempts cover
    # ~all rate-limit / node-placement transients (Flyte-review H3 fix).
    retries=2,
    # Cut the multipart-upload chunk from the flytekit default 25 MiB to 8 MiB.
    # After data_ingest returns FlyteDirectory("/tmp/raw_data"), flytekit calls
    # fsspec/s3fs to upload the whole tree; s3fs holds one `chunksize` buffer +
    # aiobotocore send buffer PER in-flight file. Combined with BatchSize(4) on
    # the return type, peak upload RSS ≈ 4 × 8 MiB × (few multipart windows) ≈
    # a few hundred MB, comfortably inside 64Gi. Prior run a9rzqr9mfg5g4c2j7dmt
    # OOMKilled DURING this upload (127 GB / 264 files at PS=50, unbounded
    # concurrency), so the fix targets exactly that path.
    environment={"_F_P_WRITE_CHUNK_SIZE": "8388608"},
)
def data_ingest(
    dataset: Dataset = Dataset.L2D,
    source_revision: str = L2D_SOURCE_REVISION,
    episodes: int = 3,
    group_ids: Optional[List[str]] = None,
) -> Annotated[FlyteDirectory, BatchSize(4)]:
    """Download raw dataset from HuggingFace (lerobot for L2D, physical_ai_av for NVIDIA).

    HF token comes from the `hf-token` K8s Secret (injected as env var by Flyte),
    never from a workflow input — so it is not visible in the Flyte/MLflow UI.

    ``group_ids`` (#121 option B) selects an EXPLICIT set of groups — L2D episode
    indices (as strings) or NVIDIA clip uuids — so a fan-out partition materializes
    ONLY its slice. When None, the legacy first-``episodes`` path is used. The ids
    are GLOBAL (episode 12 is "12" in every partition), which is what keeps the
    downstream ``sample_uid`` partition-independent (§3.1).
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
                split="train",
                source_revision=source_revision,
                max_missing_scenes=1,
            )
            scene_ids = list(inventory.selected_scene_ids)
            if episodes > 0:
                scene_ids = scene_ids[:episodes]
        else:
            scene_ids = [str(scene_id) for scene_id in group_ids]
        downloader.download(scene_ids, expected_split="train")
        print(
            f"Ingested {dataset.value}@{source_revision}: "
            f"{len(scene_ids)} scenes -> {out_dir}"
        )
        return FlyteDirectory(out_dir)

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
        # Fan-out (option B): download EXACTLY this partition's clips (global clip
        # uuids), not the first-N. None → legacy first-``episodes`` slice.
        if group_ids is not None:
            clip_ids = list(group_ids)
        else:
            clip_ids = ds.clip_index.index.tolist()[:episodes]
        feats = CAMERAS + ["egomotion"]
        # Real calibration: native f-theta intrinsics + sensor extrinsics. Enables
        # geometrically-meaningful BEV projection (#77). The rig is shared across
        # the subset, so we save calibration from the first clip that has it and
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

    # L2D: lerobot
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from ledataset.datasets.lerobot_dataset import LeRobotDataset

    # Fan-out (option B): load EXACTLY this partition's episodes (global indices),
    # not the first-N. group_ids are strings ("12") → int episode indices. None →
    # legacy first-``episodes`` slice.
    if group_ids is not None:
        ep_list = [int(g) for g in group_ids]
    else:
        ep_list = list(range(episodes)) if episodes > 0 else None
    # download_videos defaults True — DO NOT disable it. The label/pack pods
    # re-open this dir with LeRobotDataset(root=…); lerobot's
    # _check_cached_episodes_sufficient requires the requested episodes' video
    # files to exist on disk, else the OFFLINE pod attempts a network re-download
    # and fails (#121 option B invariant, verified against lerobot v0.5.0 source).
    #
    # At partition_size=500 a prior run (ah4nmxpw2jv2fklqcnkr) saw only 491 of 602
    # expected files reach disk, followed by
    # "Instruction 'train' corresponds to no data!" — the LeRobotDataset chain
    # falls through the load_hf_dataset → download → load_hf_dataset retry loop
    # (lerobot_dataset.py:742-754) but if snapshot_download silently under-fetches
    # (e.g. transient Hub 5xx during multi-thread fetch), the second load has no
    # parquet to read. This EXPLICIT pre-fetch below GUARANTEES the parquet files
    # are on disk before LeRobotDataset touches its retry logic, and asserts the
    # count so a partial fetch surfaces as an explicit RuntimeError we can debug
    # instead of the opaque "no data" error.
    from ledataset.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from huggingface_hub import hf_hub_download
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    # revision="main" — lerobot 0.5.0 defaults to CODEBASE_VERSION="v3.0", but
    # yaak-ai/L2D's v3.0 TAG points to a stale/broken snapshot (tasks.parquet
    # is 1485 bytes / 1 row at v3.0 vs 135484 bytes / 4219 rows on main;
    # episodes/data parquets are ~20% smaller too). Reading v3.0 causes
    # downstream KeyError in _absolute_to_relative_idx and IndexError in
    # iloc[task_idx]. Pin to main so we always get the live L2D revision.
    if source_revision != L2D_SOURCE_REVISION:
        raise ValueError(
            "L2D ingest supports revision='main' only because its v3.0 tag is "
            f"stale; got {source_revision!r}"
        )
    _meta = LeRobotDatasetMetadata(
        repo_id=dataset.value,
        revision=source_revision,
    )
    if ep_list is not None:
        # Compute the set of parquet+video paths lerobot would ask for, then
        # download each with hf_hub_download.  Matches lerobot's own
        # dataset_reader.get_episodes_file_paths, but with per-file timeouts
        # so a single stalled HTTP connection can't hang the whole task
        # indefinitely (previous run algqrc6zqq5kn6bnq4sx hung for ~30 min at
        # 0-byte parquet + 470MB video with snapshot_download).
        _data_paths = list({str(_meta.get_data_file_path(ep)) for ep in ep_list})
        _video_paths = list({
            str(_meta.get_video_file_path(ep, k))
            for k in _meta.video_keys for ep in ep_list
        })
        _all_files = _data_paths + _video_paths
        print(f"Pre-fetch: {len(_data_paths)} parquet + {len(_video_paths)} video "
              f"= {len(_all_files)} unique files for {len(ep_list)} episodes")

        def _one_file(rel_path: str, attempt_i: int) -> tuple[str, bool, str]:
            """Download one file into local_dir with a timeout budget.  Returns
            (path, success, note).  Retries handled by outer loop; we just
            surface success/fail to the outer retry decision.
            """
            try:
                hf_hub_download(
                    repo_id=dataset.value, repo_type="dataset",
                    revision=source_revision,
                    filename=rel_path, local_dir=str(_meta.root),
                    # Timeout ONE file: 30 s to establish etag, 12 min to
                    # transfer (~700 MB @ 1 MB/s worst-case).
                    etag_timeout=30.0,
                )
                return (rel_path, True, "ok")
            except Exception as e:
                return (rel_path, False, f"{type(e).__name__}: {e}")

        # Retry the WHOLE set 3x; between attempts, only re-download the ones
        # still missing on disk.  4 concurrent workers: each holds ~500MB
        # in-flight buffer + HTTP TLS = ~2-3GB peak, comfortably under 64Gi.
        # Serial (1 worker) would take hours per partition; 8+ workers OOM.
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
            # Verify EACH expected file is now on disk.  Downstream pods do
            # LeRobotDataset(root=raw_path, episodes=…); lerobot's
            # _check_cached_episodes_sufficient checks video presence and
            # silently re-downloads if any is missing, which risks the same
            # partial-fetch problem in a pod without our retry harness.  We
            # MUST land 100% here so downstream stays offline.
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
    # Hardlink the WHOLE cache tree (data/ + meta/ + videos/) into out_dir instead
    # of a byte copy: at tens of episodes the copy doubles disk use and churns the
    # page cache, which (with the raw video already resident) pushed the pod over
    # its memory limit → OOMKilled. Hardlinks share the same inodes (no data
    # copied, no extra RAM), and FlyteDirectory uploads them normally. Copying the
    # full tree (incl. videos/) is what lets the downstream root= reopen stay
    # offline (see the download_videos invariant above). Falls back to a real copy
    # only across filesystem boundaries (cross-device link error).
    try:
        shutil.copytree(str(cache_dir), out_dir, copy_function=os.link)
    except OSError:
        shutil.copytree(str(cache_dir), out_dir)

    print(f"Ingested {dataset.value}: {len(ds)} frames, {episodes} episodes → {out_dir}")
    return FlyteDirectory(out_dir)


# ============================================================
# Task: Data Processing (Issue #30: pre-extract frames)
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_data_prep_pod_template(),
    # Process-parallel pack workers use the pod's available cores for camera
    # decode/JPEG. The deduplicated WM path decodes each physical row once.
    # KITScenes one-scene partitions use the same schedulable Guaranteed profile
    # as ingest. The raw scene plus deduplicated 256px camera pool stays below
    # the default NodeClass's allocatable ephemeral storage.
    requests=Resources(cpu="15", mem="64Gi", ephemeral_storage="60Gi"),
    limits=Resources(cpu="15", mem="64Gi", ephemeral_storage="60Gi"),
    # Cache on (raw URI, labels URI, group_ids, world_model, image_size,
    # cache_version) so "processing is rarely needed" holds (#121 §3.4a): an
    # unchanged partition re-uses its shards. Because the raw + labels inputs are
    # FlyteDirectories hashed by URI, a cache-hit upstream keeps their URIs stable
    # → this task hits too. PACK_CACHE_VERSION folds in the shard + geometry
    # encoding, so a shard-layout change correctly re-packs.
    cache=True,
    cache_version=PACK_CACHE_VERSION,
    # 100-partition fan-out: transient pack failures (OOM at bad seed, torn
    # ProcessPool worker) shouldn't abort the whole workflow (Flyte-review H3).
    retries=2,
    # Same fsspec upload-chunk cap as data_ingest — pack output includes the
    # sibling pool/ jpg tree plus the *.tar shards, so it can also hit tens of
    # thousands of files. See data_ingest env comment for the mechanism.
    environment={"_F_P_WRITE_CHUNK_SIZE": "8388608"},
)
def data_processing(
    raw_data: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    source_revision: str = L2D_SOURCE_REVISION,
    dataset_version: str = DATASET_PACK_VERSION,
    hz: int = 10,
    image_size: int = 256,
    episodes: int = 3,
    world_model: bool = False,
    reasoning_labels: Optional[FlyteDirectory] = None,
    group_ids: Optional[List[str]] = None,
) -> Annotated[FlyteDirectory, BatchSize(4)]:
    """Pre-extract aligned frames + egomotion → WebDataset shards.

    Solves Issue #30: no video decode at training time.

    Pure deterministic packing: this task calls NO external teacher. When
    ``reasoning_labels`` is provided (the artifact from
    ``generate_reasoning_labels``), each sample's frozen label is JOINed in by
    ``sample_id`` and embedded as a per-sample ``reasoning.json`` member (#98),
    the single source of truth train_il reads. Labels are generated (and
    S3-cached) once, upstream; re-packing never re-bills the teacher (#117).

    When ``world_model`` is set (L2D only for now), each sample also gets the 1 Hz
    past/future multi-view windows for the JEPA loss (#13): members
    ``hist_{t}_cam_{v}.jpg`` (oldest→newest, current last) and
    ``fut_{f}_cam_{v}.jpg`` (the frozen JEPA targets). The window config
    (num_frames/stride) matches the online dataset so shards and on-the-fly
    windows are identical.
    """
    import os
    import io
    import json
    import tarfile
    import tempfile

    raw_path = raw_data.download()
    print(f"Processing raw data from: {raw_path} (dataset={dataset.value})")

    # Reasoning labels present ⇒ this is a full-loss run, and the JEPA/world-model
    # loss needs the WM window (future frames) packed — so force WM on. Note the
    # sample SET does NOT depend on the WM flag (egomotion margins 64/64 dominate
    # the WM margins 30/40, so enumeration is identical), and the label set is now
    # a 1 Hz SUBSET of the 10 Hz packed set by design (§3.4d): the ~9/10 unlabeled
    # samples pack without reasoning.json and mask out of the reasoning loss, while
    # still training reactive + JEPA. The global sample_uid keeps the JOIN correct.
    # (NVIDIA has no WM windows and no labels.)
    if reasoning_labels is not None and dataset != Dataset.NVIDIA_PHYSICAL_AI and not world_model:
        print("reasoning_labels present → forcing world_model=True so the JEPA "
              "loss has its WM window (future frames) packed.")
        world_model = True

    # Build the appropriate Dataset. Both are RAW pre-extraction sources: they
    # emit unmodified frames (no backbone resize/crop/normalize). The shard packer
    # below owns the single, explicit, geometry-aware resize; the pre-extracted
    # loader owns the single ToTensor+Normalize. This avoids any double-normalize /
    # center-crop and keeps the projection ABI targeting a known (plain-resized)
    # frame. Sample schema: visual_tiles (V,3,H,W), map_tile (3,H,W),
    # egomotion_history (256), trajectory_target (128). See #77.
    # Fan-out (option B): group_ids selects this partition's groups (global L2D
    # episode indices / NVIDIA clip uuids). None → legacy first-``episodes``.
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
                "L2D pack supports revision='main' only because its v3.0 tag is "
                f"stale; got {source_revision!r}"
            )
        ep_list = ([int(g) for g in group_ids] if group_ids is not None
                   else (list(range(episodes)) if episodes > 0 else None))
    # A fan-out partition can legitimately hold NO valid samples (a short episode/
    # clip below the egomotion margin); the parser raises "No valid samples found".
    # Treat that as SUCCESS producing an EMPTY shard dir (nothing to pack) rather
    # than a failure that kills the @dynamic — matches the label task's guard.
    try:
        if dataset == Dataset.NVIDIA_PHYSICAL_AI:
            from data_parsing.nvidia_physical_ai.dataset import NvidiaAVDataset
            # DISCOVERY from raw_path (the partition's ingest materialized only this
            # partition's clips), so the packer enumerates exactly the partition set
            # in the SAME order the labeler used → the reasoning.json JOIN by uid holds.
            ds = NvidiaAVDataset(data_root=raw_path)
            if world_model:
                print("world_model requested but NVIDIA has no window support yet; "
                      "packing without JEPA windows.")
        elif dataset == Dataset.KITSCENES:
            from data_parsing.kit_scenes import KitScenesDataset
            ds = KitScenesDataset(
                data_root=raw_path,
                split="train",
                scene_ids=ep_list,
                image_size=image_size,
                include_world_model_windows=world_model,
            )
        else:
            from data_parsing.l2d import L2DDataset
            # World-Model windows (#16/#13) are only produced when requested, so the
            # imitation-only path stays cheap (no extra frame decode). root=raw_path:
            # read the partition's materialized raw, don't re-hit HF.
            ds = L2DDataset(repo_id=dataset.value, episodes=ep_list,
                            include_world_model_windows=world_model, root=raw_path)
        n_samples = len(ds)
        idx_iter = range(n_samples)
    except ValueError as e:
        if "No valid samples" not in str(e):
            raise
        print(f"Partition has no valid samples ({e}); writing an EMPTY shard dir "
              f"(short episode/clip — nothing to pack).")
        ds = None
        n_samples = 0
        idx_iter = range(0)

    # Reasoning labels (#98): JOINed in from the generate_reasoning_labels
    # artifact by sample_id — NO teacher call here (this task is pure packing).
    # None → shards carry no reasoning.json (training runs imitation-only). The
    # artifact's whole-record records.jsonl is read into a {sample_id: record}
    # map; each matching sample gets a frozen reasoning.json member.
    labels_by_id = {}
    _record_to_json = None
    if reasoning_labels is not None:
        from pathlib import Path
        from data_processing.reasoning_label_generation.targets import (
            load_records_by_sample_id, record_to_json,
        )
        labels_dir = reasoning_labels.download()
        records_files = sorted(Path(labels_dir).rglob("records.jsonl"))
        if records_files:
            for rf in records_files:
                labels_by_id.update(load_records_by_sample_id(str(rf)))
            _record_to_json = record_to_json
            print(f"Reasoning labels JOIN: {len(labels_by_id)} records from "
                  f"{[str(p) for p in records_files]}")
        else:
            print(f"WARN: reasoning_labels dir {labels_dir} has no records.jsonl; "
                  "packing without reasoning.json (imitation-only).")

    # Geometry is a per-dataset rig constant, computed once. It is written into
    # EACH sample's calib.json (self-describing shards) so datasets can later be
    # merged — a merged loader resolves geometry per sample/dataset rather than
    # from a single manifest. geometry_type "pseudo" when no calibration exists.
    projection_spec = None
    build_spec = getattr(ds, "projection_spec", None)
    if callable(build_spec) and n_samples:
        projection_spec = build_spec(image_size)
    sample_geometry_type = (projection_spec or {}).get("type", "pseudo")
    calib_bytes = json.dumps(
        {"dataset": dataset.value, "geometry_type": sample_geometry_type,
         "projection": projection_spec}
    ).encode()

    out_dir = tempfile.mkdtemp()

    # v2.1 geo products are generated in the SAME full repack that writes the
    # shards. They read only numeric parquet columns, so this adds no video
    # decode. Each fan-out partition emits its own episode paths + sample-pose
    # parquet; publication can merge the partition summaries without scanning
    # the tar files or DynamoDB.
    geo_summary = None
    if ds is not None and dataset in (Dataset.L2D, Dataset.KITSCENES):
        from data_processing.geospatial import write_geo_artifacts
        geo_summary = write_geo_artifacts(
            ds,
            out_dir,
            dataset_name=dataset.value,
            dataset_version=dataset_version,
        )

    # Projection/calibration is a rig constant. Keep the existing per-sample
    # calib member for current loaders, and also publish the canonical rig-level
    # artifact used by the console's camera overlay.
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

    # A sharded full run creates many independent pack tasks. A local name such
    # as train-000000.tar collides as soon as their outputs are flattened into
    # the published dataset version, so prefix it with the deterministic group
    # set identity. Non-fan-out workflows retain the compact historical name.
    from data_processing.dataset_snapshot import (
        published_shard_name,
        shard_partition_id,
    )
    partition_id = shard_partition_id(group_ids)

    shard_idx = 0
    shard_names: list[str] = []
    sample_count = 0
    reasoning_label_count = 0
    samples_per_shard = 1000
    current_tar = None

    # Shared frame pool (#121 §3.4d): WM window frames are content-addressed by a
    # global frame_id and written ONCE here, deduping the ~8x cross-sample overlap
    # (10Hz samples × 1Hz stride-10 window). The pool is a SIBLING pool/ DIRECTORY,
    # NOT inside the .tar shards, so the loader's glob("*.tar") + split_by_worker
    # never shards it away — every DataLoader worker reaches any frame_id by path.
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
        nonlocal current_tar, shard_idx
        if current_tar:
            current_tar.close()
        shard_name = published_shard_name(group_ids, shard_idx)
        current_tar = tarfile.open(os.path.join(out_dir, shard_name), "w")
        shard_names.append(shard_name)
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

    if world_model and dataset != Dataset.NVIDIA_PHYSICAL_AI and idx_list:
        # ── DECODE-DEDUP path: decode each UNIQUE physical row once ──
        # (#121 §3.4d) Previous approach decoded all 48 window frames per sample
        # (6 workers × ~8 sample overlap = ~8x redundant decode). This two-pass
        # approach decodes only the unique rows once per partition.
        #
        # Pass A: collect unique (group_id, frame_index) rows -> row-level workers
        # decode each exactly once → write to pool/.
        #
        # Pass B: assemble each sample's members (window_index, ego, meta, calib,
        # reasoning JOIN) from the pool — zero video decode.
        print(f"Packing {len(idx_list)} samples, decode-dedup mode "
              f"(row-level workers, world_model=True)...")
        row_init = (dataset.value, ep_list, raw_path, image_size)

        # Pass A: unique rows. ds is still alive here (not yet deleted).
        all_rows: set = set()
        # Collect the current-frame row (offset 0 = cam_*.jpg) FIRST so it's
        # tracked even if window_rows raises. Do NOT catch IndexError from
        # window_rows: enumeration excludes edge frames (margins 64/64 dominate
        # WM 30/40), so a raise here means the invariant has broken and we MUST
        # fail loudly rather than silently drop the sample's cam_*.jpg (which
        # would poison the shard: loader hits torch.stack([]) at train time).
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
            # window_rows raises IndexError only if the margin invariant is
            # broken — let it propagate (fail-loud on invariant violation).
            for row_t in ds.window_rows(si):
                all_rows.add(row_t)

        del ds  # free before spawning workers

        # row_map: (group_id, frame) -> {frame_id: blob} per cam + map_jpeg.
        # Only current rows need a map tile; history/future windows contain
        # camera pixels only. This is particularly important for KITScenes,
        # where every map tile runs a Lanelet2 query and rasterization.
        row_map: dict = {}
        row_workers = max(1, min(16, len(all_rows)))
        current_rows = set(sample_cur_rows.values())
        decode_tasks = [
            (group_id, frame_index, (group_id, frame_index) in current_rows)
            for group_id, frame_index in sorted(all_rows)
        ]
        with ProcessPoolExecutor(max_workers=row_workers, mp_context=ctx,
                                 initializer=parallel_pack.init_row_worker,
                                 initargs=row_init) as rpool:
            for row_key, cam_jpegs, map_jpeg in rpool.map(
                    parallel_pack.decode_row, decode_tasks):
                row_map[row_key] = (cam_jpegs, map_jpeg)
                for fid, blob in cam_jpegs.items():
                    _write_pool(fid, blob)
                if map_jpeg is not None:
                    has_map = True
        num_views = len(next(iter(row_map.values()))[0]) if row_map else 0
        print(f"Frame pool: {pool_frames_written} unique frames decoded "
              f"(was ~{pool_frames_written * 8} with per-sample decode).")

        # Pass B: assemble per-sample members — zero video decode.
        # Plain-mode dataset for window IDs and numeric members only.
        import numpy as np
        import torch
        if dataset == Dataset.KITSCENES:
            from data_parsing.kit_scenes import KitScenesDataset
            ds_asm = KitScenesDataset(
                data_root=raw_path,
                split="train",
                scene_ids=ep_list,
                image_size=image_size,
                include_world_model_windows=False,
            )
        else:
            from data_parsing.l2d import L2DDataset
            ds_asm = L2DDataset(
                repo_id=dataset.value,
                episodes=ep_list,
                include_world_model_windows=False,
                root=raw_path,
            )

        for si in idx_list:
            if sample_count % samples_per_shard == 0:
                open_new_shard()
            uid = ds_asm.sample_uid(si)
            split_group = ds_asm.split_group_uid(si)
            from data_processing.dataset_snapshot import split_bucket
            members: dict = {}

            # window_index — pool frame_ids, no decode.
            try:
                ids = ds_asm.window_frame_ids(si)
                members["window_index.json"] = json.dumps(ids).encode()
                has_wm = True
            except (IndexError, AttributeError):
                pass

            # cam_*.jpg = current frame (offset 0). The current-frame bytes are in
            # row_map[(ep_idx, cur_fi)][0] — the same jpegs already written to pool.
            cur_key = sample_cur_rows.get(si)
            if cur_key and cur_key in row_map:
                cur_cams, cur_map = row_map[cur_key]
                # cam_cams is {frame_id: bytes}; sort by cam index embedded in fid.
                for fid, blob in sorted(cur_cams.items(),
                                        key=lambda kv: int(kv[0].rsplit("-c", 1)[-1])):
                    cam_i = int(fid.rsplit("-c", 1)[-1])
                    members[f"cam_{cam_i}.jpg"] = blob
                if cur_map is not None:
                    members["map.jpg"] = cur_map

            # ego + meta + calib (no video decode).
            ego_hist, traj, pose_current, gps_future = ds_asm.numeric_for(si)
            ego_data = np.concatenate([
                ego_hist.numpy() if torch.is_tensor(ego_hist) else np.asarray(ego_hist),
                traj.numpy() if torch.is_tensor(traj) else np.asarray(traj),
            ]).astype(np.float32)
            members["ego.npy"] = ego_data.tobytes()
            from data_processing.geospatial import geospatial_members
            members.update(geospatial_members({
                "pose_current": pose_current,
                "gps_future": gps_future,
            }))
            members["meta.json"] = json.dumps({
                "idx": si, "dataset": dataset.value,
                "sample_uid": uid, "split_group_uid": split_group,
                "split_bucket": split_bucket(split_group),
                "frame_idx": ds_asm.frame_index(si),
            }).encode()
            members["calib.json"] = calib_bytes

            for suffix, blob in members.items():
                _add_member(uid, suffix, blob)
            if _record_to_json is not None:
                record = labels_by_id.get(uid)
                if record is not None:
                    _add_member(uid, "reasoning.json",
                                json.dumps(_record_to_json(record)).encode())
                    reasoning_label_count += 1
            sample_count += 1

    else:
        # ── Legacy path (imitation-only L2D, NVIDIA, or empty partition) ──
        # Per-sample full-window decode. For NVIDIA there are no WM windows.
        max_workers_cap = 16  # imitation-only samples are light
        pack_workers = max(1, min(max_workers_cap, len(idx_list)))
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
                has_map = has_map or ("map.jpg" in members)
                has_wm = has_wm or ("window_index.json" in members)
                if _record_to_json is not None:
                    record = labels_by_id.get(sample_key)
                    if record is not None:
                        _add_member(sample_key, "reasoning.json",
                                    json.dumps(_record_to_json(record)).encode())
                        reasoning_label_count += 1
                sample_count += 1

    if current_tar:
        current_tar.close()

    from data_processing.contract_versions import contract_versions
    from data_processing.geospatial import (
        EPISODE_PATH_SCHEMA_VERSION,
        GPS_SCHEMA_VERSION,
        POSE_SCHEMA_VERSION,
    )

    manifest = {"total_samples": sample_count, "shards": shard_idx,
                "shard_names": shard_names,
                "partition_id": partition_id or None,
                "hz": hz, "image_size": image_size, "dataset": dataset.value,
                "source_revision": source_revision,
                "dataset_version": dataset_version,
                "episodes": episodes,
                "contracts": contract_versions(),
                # num_views = real cameras only; the map view is stored under a
                # separate map.jpg key and is NOT counted here (#77).
                "num_views": num_views if sample_count else 0,
                "has_map": bool(sample_count) and has_map,
                # World-Model windows present when packed (enables JEPA training).
                "has_world_model": bool(sample_count) and has_wm,
                "has_reasoning_labels": reasoning_label_count > 0,
                "reasoning_label_count": reasoning_label_count,
                "has_gps": bool(sample_count) and dataset in (
                    Dataset.L2D, Dataset.KITSCENES,
                ),
                "geospatial": {
                    "pose_schema": POSE_SCHEMA_VERSION,
                    "gps_schema": GPS_SCHEMA_VERSION,
                    "episode_path_schema": EPISODE_PATH_SCHEMA_VERSION,
                    "source_coordinate_dtype": "float32",
                    "stored_coordinate_dtype": "float64",
                    "timestamp_dtype": "int64_ns",
                    "summary": geo_summary,
                } if dataset in (Dataset.L2D, Dataset.KITSCENES) else None}

    # Manifest also carries the projection spec (computed once above) for the
    # single-dataset loader path; the merged loader uses per-sample calib.json.
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
# Task: Reasoning label generation (offline teacher → versioned S3 artifact)
#
# This is the SINGLE place the teacher (Cosmos) is ever called. It enumerates
# samples straight from the raw dataset; each sample's GLOBAL uid
# (parser.sample_uid, #121 §3.1) is the JOIN key to data_processing — stable
# across episode-range shards. It labels every sample of the partition and writes
# ONE records.jsonl artifact (no per-sample S3 cache, §3.4); the Flyte task cache
# on the deterministic partition prevents re-billing on a re-run. data_processing
# later JOINs this artifact into the shards by sample_id — it does NOT call the
# teacher (#98/#117).
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
    # Cache on (raw URI, group_ids, teacher, prompt_version, cache_version) so a
    # re-run of an unchanged partition is a no-op (#121 §3.4a) — this is now the
    # SOLE re-label protection (the per-sample S3 cache is gone, §3.4): an unchanged
    # partition never re-bills Cosmos, a changed prompt_version / teacher correctly
    # misses. LABEL_CACHE_VERSION folds in the uid format (the JOIN key) and
    # sparse-selection policy. EXCLUDE the tuning knob from the key (§3.4c):
    # label_workers is pure parallelism (output-invariant), so a tweak must not
    # force a corpus re-label.
    cache=True,
    cache_version=LABEL_CACHE_VERSION,
    cache_ignore_input_vars=("label_workers",),
    # 100-partition fan-out: a single Cosmos vLLM 503 must not abort the whole
    # workflow. The teacher call is idempotent (labels are computed, not stored),
    # so a retry is safe (Flyte-review H3).
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
    # Process-parallel worker count. Front-clip mode decodes only 5 front frames
    # per sample, but at 20+ episodes 24 concurrent decoders + their lerobot
    # readers still OOM-killed the task at ~96/125. 12 workers still overlap the
    # ~12s teacher HTTP wait well (the stage is latency-bound, not CPU-bound) and
    # halve peak memory; combined with the raised 60Gi limit this clears the OOM.
    # 2026-07-14: at partition_size=50 (13k+ hf rows loaded via lerobot per worker),
    # 12 workers OOM at 60Gi (run a88ch58g5xqgj4sc8r4n dn1-1 exit 137 right after
    # "Labeling 705/7130 samples ..." print). Drop to 6 — lerobot memory scales
    # ~linearly with loaded episode count, and teacher latency (~12s) leaves 6
    # workers plenty of overlap. Cross-pod fan-out (Flyte map_task) is the real
    # scale fix.
    label_workers: int = 6,
) -> Annotated[FlyteDirectory, BatchSize(4)]:
    """Label each 1 Hz World-Model sample with a TEMPORAL front-camera clip, then
    write a versioned label artifact for the data_processing JOIN.

    Reasoning is a 1 Hz, temporal concern: the teacher is shown one FRONT-camera
    frame per horizon (0 s current + 1/2/3/4 s future) so it can reason about how
    the scene evolves (cut-ins, stops, yields) instead of guessing from a single
    instant with many cameras. Both datasets expose ``get_front_clip(idx)`` in a
    light-weight ``reasoning_clip_only`` mode that decodes ONLY those 5 front
    frames (L2D via lerobot delta_timestamps; NVIDIA via a sparse front-camera
    PyAV decode) — far cheaper than the full multi-view World-Model window.

    Labels are keyed by the parser's GLOBAL ``sample_uid`` (#121 §3.1), so the
    JOIN to ``data_processing`` holds even when labeling and packing run over
    different episode-range shards. Both L2D and NVIDIA are labelled (NVIDIA is no longer
    skipped): ``reasoning_clip_only`` does not change either dataset's sample set.

    There is NO per-sample S3 cache (#121 §3.4): the teacher is called once per
    sample of this partition and all records aggregate into ONE ``records.jsonl``.
    Re-label protection is the Flyte task cache on the deterministic partition —
    an unchanged partition is a task-cache no-op (no Cosmos call); a changed
    ``teacher`` / ``prompt_version`` (both in the cache key) correctly misses, so
    the temporal-clip / front-only / 256px prompt change re-labels cleanly.

    Returns:
        FlyteDirectory with a whole-record ``records.jsonl`` (the JOIN
        interchange data_processing reads), the flattened
        ``reasoning_labels_v2.{parquet,jsonl}`` analytics export, and a
        provenance ``meta.json``.
    """
    import json
    import os
    import tempfile

    # Parent only needs the artifact writers; the teacher/dataset/cache/clip live
    # in the per-process workers (see parallel_label). teacher_kwargs is still
    # assembled here (from the Flyte secret context) and passed to the workers.
    from data_processing.reasoning_label_generation.parquet_writer import (
        write_jsonl, write_parquet,
    )
    from data_processing.reasoning_label_generation.targets import write_records_jsonl

    raw_path = raw_data.download()
    print(f"Generating reasoning labels: dataset={dataset.value} split={split} "
          f"teacher={teacher} prompt={prompt_version} raw={raw_path}")

    # Sample count: build the dataset once (front-clip mode) just to get len().
    # Enumeration matches data_processing (WM-window sample set) so sample_ids
    # JOIN; workers rebuild their own front-clip dataset in init_worker.
    # Fan-out (option B): group_ids selects this partition's groups (global L2D
    # episode indices / NVIDIA clip uuids). None → legacy first-``episodes``.
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
                "L2D labeling supports revision='main' only because its v3.0 "
                f"tag is stale; got {source_revision!r}"
            )
        ep_list = ([int(g) for g in group_ids] if group_ids is not None
                   else (list(range(episodes)) if episodes > 0 else None))
    # A fan-out partition can legitimately contain NO valid samples — e.g. a
    # single short L2D episode with fewer than the egomotion margin (64+64+1)
    # frames. The parser raises "No valid samples found" in that case; in the
    # single-pod path that never happened because other episodes filled the set,
    # but per-episode partitioning exposes it. Treat an empty partition as a
    # SUCCESS that produces an empty label artifact (nothing to JOIN downstream) —
    # NOT a failure that kills the whole @dynamic fan-out.
    try:
        if dataset == Dataset.NVIDIA_PHYSICAL_AI:
            from data_parsing.nvidia_physical_ai.dataset import NvidiaAVDataset
            # NVIDIA: the partition's ingest materialized ONLY this partition's
            # clips into raw_path, so DISCOVERY (sorted) yields exactly the
            # partition set — and the worker (parallel_label.init_worker) also
            # discovers from raw_path, so probe and workers enumerate in the SAME
            # order (sample-index JOIN holds). Passing clip_uuids in partition
            # order here would risk an order mismatch.
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
            # root=raw_path: read the partition's materialized raw, don't re-hit HF.
            ds = L2DDataset(repo_id=dataset.value, episodes=ep_list,
                            reasoning_clip_only=True, root=raw_path)
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
    # strict=False for bulk offline labeling: a single sample whose response the
    # model returns malformed / with <5 horizons becomes an ABSTAINED record
    # (masked out of the reasoning loss, R9) instead of raising and killing the
    # whole 1000+-sample run. meta.json reports num_abstained so a systematically
    # high rate (bad prompt/model) is still visible.
    teacher_kwargs["strict"] = False

    # Free the parent's dataset handle: each worker process builds its own.
    del ds

    from data_processing.reasoning_label_generation.targets import record_from_json

    n_computed = n_abstain = 0
    if not label_indices:
        # Empty partition, or nothing selected by the stride: empty artifact.
        records = []
    else:
        # Process-parallel labeling (NOT threads): decode dominates and lerobot's
        # reader is not thread-safe, so a ThreadPool had to serialize decode under a
        # lock, leaving the scaled-out vLLM replicas idle. With processes, each worker
        # owns an independent dataset + reader, so decode runs truly in parallel across
        # CPU cores and the teacher calls overlap — finally using the extra GPUs. Only
        # the sample index crosses the process boundary; frames never do. Spawn context
        # (torch is imported) re-imports the worker module cleanly.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from data_processing.reasoning_label_generation import parallel_label

        workers = max(1, min(label_workers, len(label_indices)))
        print(f"Labeling {len(label_indices)}/{n_samples} samples (sparse subset, "
              f"stride={label_stride}) with {workers} parallel PROCESSES "
              f"(teacher={teacher})...")
        ctx = mp.get_context("spawn")
        # Order MUST match parallel_label.init_worker(repo_id, episodes, dataset_name,
        # teacher, teacher_kwargs, prompt_version, raw_path). No cache_bucket — the
        # per-sample S3 cache is gone (§3.4).
        init_args = (dataset.value, ep_list, dataset.value, teacher, teacher_kwargs,
                     prompt_version, raw_path)
        # Only the 1 Hz subset is labeled; records.jsonl carries just those. The
        # packer JOINs by uid, so the ~9/10 unlabeled 10 Hz samples get no
        # reasoning.json and are masked out of the reasoning loss at train time.
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
    # A few abstentions (malformed teacher JSON) are fine — they are masked out of
    # the reasoning loss. A HIGH rate means a systemic prompt/model problem, so
    # fail loudly rather than silently shipping a mostly-unlabeled dataset.
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


# ============================================================
# Task: IL Training (real AutoE2E)
# ============================================================
@task(
    container_image=TRAINING_IMAGE,
    # requests == limits (Guaranteed QoS). g6e.4xlarge has 16 vCPU / 44.7 GB
    # GPU-attached mem; keep pod at 16 GB so multiple non-GPU sidecars can
    # share the node if needed, but the whole GPU is reserved (gpu="1").
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(cpu="4", mem="16Gi", gpu="1"),
    pod_template=_large_shm_pod_template(),  # /dev/shm for DataLoader workers (#121 P0)
)
def train_il(
    shards: List[FlyteDirectory],
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs: int = 3,
    batch_size: int = 4,
    # Effective batch size = batch_size * grad_accum_steps. The World-Model
    # windows (T history + F future frames x V cams) blow up activation memory,
    # forcing batch_size=1 on the L40S; but the trajectory loss needs a larger
    # effective batch to descend past ~0.84 (the bs=1 per-sample SmoothL1 gradient
    # is too noisy — the bs=4 imitation run reached 0.36). Accumulating grads over
    # N micro-batches recovers the bs=4 signal at bs=1 memory: zero_grad at the
    # window start, step once at the window end. Default 1 = plain per-batch step.
    grad_accum_steps: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    # AMP off by default: with fp16 autocast the GradScaler detected inf/nan grads
    # every step (fp16 overflow somewhere in the BEV projection / Bezier basis /
    # backbone path) and skipped optimizer.step() FOREVER — weights never updated,
    # so the trajectory loss sat perfectly flat (~2.95) while fp32 learns in one
    # step (verified: control_head grad norm ~6.6, loss 6.30->5.00). Keep fp32
    # until the specific overflow op is isolated and kept in fp32 explicitly.
    amp: bool = False,
    enable_reasoning: bool = False,
    reasoning_mode: str = "pooled_latent",
    # Small default: the reasoning branch is zero-init coupled (alpha=0), so it
    # does not move the trajectory yet, and its structured-CE term sits at a
    # large near-constant floor (~ln(num_classes) per group) until real (non-mock)
    # labels + a non-zero visual history are available. A large weight only adds
    # a constant that masks the trajectory loss in the logged total. Keep it small
    # until the reasoning branch is actually learnable.
    reasoning_loss_weight: float = 0.05,
    enable_world_model: bool = False,
    jepa_loss_weight: float = 1.0,
    # Held-out split: train on the (1 - val_fraction) majority of samples, so the
    # separate eval task can score the disjoint val split and measure
    # GENERALIZATION rather than training-set memorization (which structurally
    # favours the lower-capacity imitation model). 0.0 = train on everything
    # (legacy in-sample behaviour). The split is a stable per-sample hash of
    # __key__, so train and eval never share a sample and both tasks agree.
    val_fraction: float = 0.0,
    # Parallel JPEG decode (#121 P0). num_workers=0 decodes every sample (~55
    # JPEGs/sample with WM windows) serially on the training process, stalling the
    # GPU — the dominant per-epoch cost at scale. >0 spreads decode across worker
    # processes (sharded over shards by split_by_worker), overlapping it with the
    # GPU step. Effective parallelism is capped by shard count, so scale needs more
    # (smaller) shards too.
    num_workers: int = 0,
) -> TrainOutput:
    """Train AutoE2E model on pre-extracted WebDataset shards.

    All datasets' shards are passed in; the one matching `dataset` is selected
    (single-dataset training; multi-dataset tracked in #77).

    When ``enable_reasoning`` is set, the horizon-aware reasoning branch (#98) is
    built with the given ``reasoning_mode`` (pooled_latent /
    horizon_cross_attention) and, if the shards carry per-sample reasoning labels
    (a ``reasoning.json`` member), its HorizonReasoningLoss is added to the
    imitation loss. If reasoning is on but a batch has no labels, only the
    trajectory loss is used for that batch (the branch still runs, zero-init so
    it does not perturb the trajectory until trained).

    When ``enable_world_model`` is set and the shards carry World-Model windows
    (packed via data_processing(world_model=True)), the JEPA future-feature
    reconstruction loss (#13) is added: the model runs the stateless windowed
    path (encode_history → aggregate → predict_future), and jepa_loss compares
    the prediction against the frozen target on the real future frames. The WM
    also supplies the Encoded Visual History to the planner and reasoning branch
    (otherwise visual_history is zeros).
    """
    import os
    import json
    import torch
    import numpy as np
    from flytekit import current_context

    # DataLoader workers (num_workers>0) transport batches to the parent via shared
    # memory (/dev/shm) by default; the Flyte pod's /dev/shm is tiny (~64MB), so
    # WM-window batches overflow it → "Bus error / worker killed by signal"
    # (#121 P0, documented in Platform/HowToUseFlyte.md). Switch torch's tensor
    # sharing to the file_system strategy, which passes tensors via mmap'd temp
    # files instead of /dev/shm — the standard fix for constrained-shm containers.
    # No-op when num_workers=0.
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")

    from model_components.auto_e2e import AutoE2E
    from model_components.losses import TrajectoryImitationLoss
    from data_parsing.pre_extracted import make_multi_dataset_loader
    # _loader_download_dir is a module-level helper in THIS file, not in
    # pre_extracted — call it directly (importing it from there is an ImportError).

    ctx = current_context()
    bb, fm = backbone.value, FUSION_LABEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training: backbone={bb} fusion={fm} epochs={epochs} bs={batch_size} device={device}")

    # MERGED DataLoader over ALL provided shard dirs. Each dataset keeps its own
    # geometry/num_views; batches are same-dataset (uniform), interleaved across
    # datasets, each carrying its projection — so L2D (6cam pseudo) and NVIDIA
    # (7cam f-theta) train together. The model is runtime-V-dynamic (projection
    # ABI, #77), so a single model consumes both. num_views only sizes defaults.
    all_shard_dirs = [_loader_download_dir(s) for s in shards]
    shard_dirs = []
    manifests = {}
    dataset_versions = set()
    skipped_empty = 0
    for shard_dir in all_shard_dirs:
        manifest_path = os.path.join(shard_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"packed shard manifest is missing: {manifest_path}")
        manifest = json.load(open(manifest_path))
        manifests[shard_dir] = manifest
        version = manifest.get("dataset_version")
        if version:
            dataset_versions.add(str(version))
        if int(manifest.get("total_samples", 0)) <= 0:
            skipped_empty += 1
        else:
            shard_dirs.append(shard_dir)
    if not shard_dirs:
        raise ValueError("all packed shard partitions are empty; nothing to train")
    if len(dataset_versions) > 1:
        raise ValueError(
            f"mixed dataset versions in one training run: {sorted(dataset_versions)}"
        )
    dataset_version = next(iter(dataset_versions), "unknown")
    # Train on the "train" split when a held-out fraction is requested (eval scores
    # the disjoint "val" split); val_fraction=0 keeps the legacy all-samples path.
    _split = "train" if val_fraction > 0.0 else "all"
    merged = make_multi_dataset_loader(shard_dirs, batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=(device.type == "cuda"),
                                       split=_split, val_fraction=val_fraction)
    print(f"Merged {len(shard_dirs)} non-empty partition(s) into one training stream "
          f"(skipped_empty={skipped_empty}, split={_split}, "
          f"val_fraction={val_fraction}, num_workers={num_workers}).")

    # Peek the first batch to size num_views defaults.
    _peek, _peek_proj, _peek_geom = next(iter(merged))
    num_views = int(_peek["visual_tiles"].shape[1])
    print(f"Detected num_views={num_views} (first dataset); geometry={_peek_geom}")

    # Consistency guard (packing ↔ training) across every non-empty partition.
    # Sparse reasoning targets are masked on unlabeled samples, so probing a
    # random first batch cannot distinguish an intentionally unlabeled sample
    # from a wholly unsupervised shard. The pack manifest records the exact join
    # count; validate that deterministic aggregate instead.
    total_reasoning_labels = 0
    for d in shard_dirs:
        manifest = manifests[d]
        dname = manifest.get("dataset", d)
        if enable_world_model and not manifest.get("has_world_model", False):
            raise ValueError(
                f"enable_world_model=True but dataset '{dname}' ({d}) has no "
                f"World-Model windows. Re-pack that dataset with world_model=True "
                f"(NVIDIA has no window support yet — exclude it or disable WM)."
            )
        if enable_reasoning:
            label_count = int(manifest.get("reasoning_label_count", 0))
            has_labels = bool(manifest.get("has_reasoning_labels", False))
            if has_labels != (label_count > 0):
                raise ValueError(
                    f"reasoning manifest flags disagree for dataset '{dname}' "
                    f"({d}): has_reasoning_labels={has_labels}, "
                    f"reasoning_label_count={label_count}"
                )
            if label_count <= 0:
                raise ValueError(
                    f"enable_reasoning=True but dataset '{dname}' ({d}) carries no "
                    f"reasoning labels. Re-pack it with reasoning_teacher set."
                )
            total_reasoning_labels += label_count
    if enable_reasoning:
        print(
            f"Reasoning supervision: {total_reasoning_labels} joined labels "
            f"across {len(shard_dirs)} non-empty partitions"
        )

    # Model. fusion_mode is gone (BEV hardcoded inside ReactiveE2E); the model
    # now also owns the map branch, so its forward requires a map_input tensor.
    model = AutoE2E(
        backbone=bb, num_views=num_views, embed_dim=256,
        is_pretrained=True,
        enable_reasoning=enable_reasoning, reasoning_mode=reasoning_mode,
        enable_world_model=enable_world_model,
    ).to(device)
    print(f"Reasoning: {'on' if enable_reasoning else 'off'}"
          + (f" (mode={reasoning_mode})" if enable_reasoning else ""))
    print(f"World Model: {'on' if enable_world_model else 'off'}")

    # Optimizer + Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = TrajectoryImitationLoss(loss_type="smooth_l1")
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)

    # Reasoning loss (#98): computed outside the model on the aux reasoning_pred
    # against the shard's per-sample labels. Built only when reasoning is on.
    reasoning_loss_fn = None
    target_batch_from_loader = None
    if enable_reasoning:
        from training.losses.horizon_reasoning_loss import HorizonReasoningLoss
        from data_processing.reasoning_label_generation.targets import (
            target_batch_from_loader as _tb_from_loader,
        )
        reasoning_loss_fn = HorizonReasoningLoss()
        target_batch_from_loader = _tb_from_loader

    # Training loop
    model.train()
    losses_per_epoch = []
    scaler = torch.amp.GradScaler(enabled=amp)

    _proj_cache = {}
    _first_step = True  # gate the one-time gradient-flow probe below
    accum = max(1, int(grad_accum_steps))
    if accum > 1:
        print(f"Gradient accumulation: {accum} micro-batches "
              f"(effective batch size = {batch_size * accum})")
    for epoch in range(epochs):
        epoch_losses = []
        traj_losses = []
        jepa_vals = []
        reason_vals = []
        micro_idx = 0  # position within the current accumulation window
        # Merged loader yields (batch, projection, geometry_type): each batch is
        # same-dataset (uniform num_views/geometry) but datasets are interleaved,
        # so the per-batch projection is applied to the batch it belongs to.
        for batch, batch_proj, batch_geom in merged:
            visual = batch["visual_tiles"].to(device)        # (B, V, 3, H, W)
            ego_hist = batch["egomotion_history"].to(device)  # (B, 256)
            vis_hist = batch["visual_history"].to(device)     # (B, 896)
            target = batch["trajectory_target"].to(device)    # (B, 128)
            map_input = batch["map_input"].to(device)

            # Per-batch geometry, moved to device once per operator (cached).
            if batch_proj is not None:
                key = id(batch_proj)
                if key not in _proj_cache:
                    _proj_cache[key] = batch_proj.to(device)
                proj_dev = _proj_cache[key]
            else:
                proj_dev = None

            # World-Model windows (#13): present only on world_model shards. The
            # windowed path makes JEPA loss differentiable and also supplies the
            # Encoded Visual History to the planner + reasoning branch.
            history_frames = batch.get("history_frames")
            future_frames = batch.get("future_frames")
            if history_frames is not None:
                history_frames = history_frames.to(device)
            if future_frames is not None:
                future_frames = future_frames.to(device)

            # Accumulation window: zero grads only at its start, step at its end.
            if micro_idx == 0:
                optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(visual, map_input, vis_hist, ego_hist,
                            projection=proj_dev, geometry_type=batch_geom,
                            mode="train", trajectory_target=target,
                            history_frames=history_frames, future_frames=future_frames)
                # Train mode returns (trajectory, aux) when a branch (reasoning /
                # world model) is on; otherwise just the trajectory tensor.
                trajectory, aux = out if isinstance(out, tuple) else (out, {})
                traj_loss = loss_fn(trajectory, target)
                loss = traj_loss

                # JEPA loss (#13): future-feature reconstruction, added when the
                # WM ran the windowed path AND this batch carries future frames.
                jepa_val = 0.0
                future_state_pred = aux.get("future_state_pred")
                if (enable_world_model and future_state_pred is not None
                        and future_frames is not None):
                    jepa = model.World_Action_Model_E2E.jepa_loss(
                        future_state_pred, future_frames)
                    loss = loss + jepa_loss_weight * jepa
                    jepa_val = float(jepa.item())

                # Add the reasoning loss when the branch is on AND this batch
                # carries labels (shards packed with a teacher). The branch is
                # zero-init, so with no labels the trajectory is unaffected.
                reason_val = 0.0
                reasoning_pred = aux.get("reasoning_pred")
                if reasoning_loss_fn is not None and reasoning_pred is not None:
                    tb = target_batch_from_loader(batch)
                    if tb is not None:
                        terms = reasoning_loss_fn(
                            reasoning_pred,
                            {g: t.to(device) for g, t in tb.targets.items()},
                            source_weights=tb.source_weights.to(device),
                            confidence_targets=tb.confidence_targets.to(device),
                        )
                        loss = loss + reasoning_loss_weight * terms["total"]
                        reason_val = float(terms["total"].item())

            # Divide by accum so summed micro-batch grads equal the MEAN gradient
            # of an effective batch of (batch_size * accum) — same scale as a plain
            # step, so lr/grad_clip keep their meaning. Log the unscaled loss.
            scaler.scale(loss / accum).backward()

            epoch_losses.append(loss.item())
            traj_losses.append(traj_loss.item())
            jepa_vals.append(jepa_val)
            reason_vals.append(reason_val)

            # Step only at the end of an accumulation window (or plain step when
            # accum==1). Grads persist across micro-batches until then.
            micro_idx += 1
            if micro_idx < accum:
                continue
            micro_idx = 0

            scaler.unscale_(optimizer)
            # One-time gradient-flow probe (very first optimizer step): prove each
            # enabled branch actually receives gradient (not just the trajectory
            # head). We report the grad-norm of a parameter unique to each branch —
            # a zero/None here means that branch is not training even though its
            # loss is being added.
            if _first_step:
                def _branch_gn(substr):
                    tot, n = 0.0, 0
                    for nm, p in model.named_parameters():
                        if substr in nm and p.grad is not None:
                            tot += float(p.grad.norm().item()) ** 2
                            n += 1
                    return (tot ** 0.5, n)
                planner_gn = _branch_gn("TrajectoryPlanner")
                probe = f"grad-flow probe: planner={planner_gn}"
                if enable_world_model:
                    probe += f" world_model={_branch_gn('World_Action_Model')}"
                if enable_reasoning:
                    probe += f" reasoning={_branch_gn('Reasoning')}"
                print(probe)
                _first_step = False
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        # Flush a trailing partial accumulation window (epoch batch count not a
        # multiple of accum) so its grads aren't silently dropped at epoch end.
        if micro_idx > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            micro_idx = 0

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_traj = np.mean(traj_losses) if traj_losses else 0.0
        avg_jepa = np.mean(jepa_vals) if jepa_vals else 0.0
        avg_reason = np.mean(reason_vals) if reason_vals else 0.0
        losses_per_epoch.append(float(avg_loss))
        # Log each branch's sub-loss separately: the total carries traj + JEPA
        # (world model) + reasoning aux terms, so per-branch values show whether
        # EACH branch is actually being optimized (not just the trajectory head).
        print(f"  Epoch {epoch+1}/{epochs} loss={avg_loss:.4f} "
              f"traj_loss={avg_traj:.4f} jepa={avg_jepa:.4f} reason={avg_reason:.4f}")

    # Save checkpoint
    os.makedirs("/tmp/train", exist_ok=True)
    ckpt_path = "/tmp/train/best.pt"
    # `config` must be reconstruction kwargs for AutoE2E(**config); fusion_mode
    # is no longer a constructor arg, so it lives only in metadata below. The
    # branch flags MUST be recorded so a later stage (offline RL / eval) rebuilds
    # the SAME architecture — otherwise load_state_dict fails on missing keys.
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "backbone": bb, "embed_dim": 256, "num_views": num_views,
            "enable_reasoning": enable_reasoning, "reasoning_mode": reasoning_mode,
            "enable_world_model": enable_world_model,
        },
        "epoch": epochs,
    }, ckpt_path)

    # Metadata
    meta = {
        "data": {"dataset": dataset.value, "dataset_version": dataset_version,
                 "shard_dirs": shard_dirs,
                 "merged_datasets": len(shard_dirs)},
        "model": {"backbone": bb, "fusion_mode": fm, "embed_dim": 256, "num_views": num_views},
        "training": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "grad_clip": grad_clip, "amp": amp,
            "optimizer": "AdamW", "final_loss": losses_per_epoch[-1] if losses_per_epoch else 0,
            "losses_per_epoch": losses_per_epoch,
            # Recorded so the (separate) eval task scores the SAME held-out split
            # this run trained around — eval reads this to pick split="val".
            "val_fraction": val_fraction,
        },
        "context": {
            "flyte_execution_id": ctx.execution_id.name if ctx.execution_id else "local",
            "docker_image": TRAINING_IMAGE,
        },
    }
    meta_path = "/tmp/train/metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return TrainOutput(checkpoint=FlyteFile(ckpt_path), metadata=FlyteFile(meta_path))


# ============================================================
# Task: Offline RL
# ============================================================
@task(
    container_image=OFFLINE_RL_IMAGE,
    # requests == limits (Guaranteed QoS).
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(cpu="4", mem="16Gi", gpu="1"),
)
def train_offline_rl(
    pretrained: FlyteFile,
    shards: List[FlyteDirectory],
    il_metadata: FlyteFile,
    dataset: Dataset = Dataset.L2D,
    epochs: int = 3,
    tau: float = 0.7,
    beta: float = 3.0,
) -> TrainOutput:
    """Offline RL refinement of the IL checkpoint via advantage-weighted regression
    against a frozen IL prior (AWR — not full IQL; no learned value network)."""
    import os
    import json
    import torch
    import numpy as np
    from flytekit import current_context

    ckpt_path = pretrained.download()
    shard_dir = _select_shard_dir(shards, dataset)
    il_meta = json.load(open(il_metadata.download()))
    ctx = current_context()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Offline RL (AWR, frozen prior): epochs={epochs} beta={beta}")

    # Load IL model
    from model_components.auto_e2e import AutoE2E
    from data_parsing.pre_extracted import make_pre_extracted_loader

    import copy

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = AutoE2E(**_model_kwargs(config)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # FROZEN behavior prior = the IL checkpoint at t=0, kept fixed. The advantage
    # must be measured against a policy that does NOT move with the one being
    # trained; using the LIVE model for both terms makes advantage identically 0
    # (a no-op that silently reduces to plain BC). This frozen prior gives a real
    # signal: "does the fine-tuned policy beat the IL prior on this sample?".
    baseline_model = copy.deepcopy(model).to(device).eval()
    for p in baseline_model.parameters():
        p.requires_grad_(False)

    loader = make_pre_extracted_loader(shard_dir, batch_size=4, num_workers=0)
    projection, geometry_type = _loader_projection(loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-3)

    # Advantage-weighted regression (AWR) against the frozen IL prior.
    model.train()
    losses_per_epoch = []
    for epoch in range(epochs):
        epoch_losses = []
        for batch in loader:
            # Reset the WM per-sequence rolling buffer per batch (see eval note):
            # avoids cross-batch history leakage and ragged-batch cat crashes.
            if hasattr(model, "reset_visual_history"):
                model.reset_visual_history()
            if hasattr(baseline_model, "reset_visual_history"):
                baseline_model.reset_visual_history()
            visual = batch["visual_tiles"].to(device)
            ego_hist = batch["egomotion_history"].to(device)
            vis_hist = batch["visual_history"].to(device)
            target = batch["trajectory_target"].to(device)
            map_input = batch["map_input"].to(device)

            optimizer.zero_grad()
            # Offline RL regresses only the trajectory; run mode="infer" so the
            # forward returns a bare trajectory tensor even when the checkpoint
            # was trained with reasoning / world-model branches on (mode="train"
            # would return a (trajectory, aux) tuple and break the arithmetic).
            # The inference forward is still differentiable for the policy grad.
            pred = model(visual, map_input, vis_hist, ego_hist,
                         projection=projection, geometry_type=geometry_type,
                         mode="infer")
            # Advantage-weighted regression against the FROZEN IL prior. advantage
            # > 0 where the trained policy is already closer to the logged action
            # than the prior; exp(beta*advantage) up-weights those samples. Using
            # the frozen prior (not the live model) makes the advantage real and
            # non-zero, and makes beta actually do something.
            with torch.no_grad():
                baseline_pred = baseline_model(visual, map_input, vis_hist, ego_hist,
                                               projection=projection, geometry_type=geometry_type,
                                               mode="infer")
            advantage = -(pred.detach() - target).pow(2).mean(dim=-1) \
                + (baseline_pred - target).pow(2).mean(dim=-1)
            weights = torch.exp(beta * advantage).clamp(max=100.0)
            loss = (weights * (pred - target).pow(2).mean(dim=-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        losses_per_epoch.append(float(avg_loss))
        print(f"  Epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")

    os.makedirs("/tmp/rl", exist_ok=True)
    out_path = "/tmp/rl/policy_rl.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epochs}, out_path)

    meta = {
        "base_model": {"il_metadata": il_meta, "il_checkpoint": str(ckpt_path)},
        # AWR against a frozen IL prior — NOT full IQL: there is no learned value
        # / expectile network, so tau is not used (recorded as null for honesty;
        # a true IQL value head is future work).
        "rl": {"method": "awr_frozen_prior", "epochs": epochs, "tau": None, "beta": beta,
                "losses_per_epoch": losses_per_epoch},
        "context": {
            "flyte_execution_id": ctx.execution_id.name if ctx.execution_id else "local",
            "docker_image": OFFLINE_RL_IMAGE,
        },
    }
    meta_path = "/tmp/rl/metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return TrainOutput(checkpoint=FlyteFile(out_path), metadata=FlyteFile(meta_path))


# ============================================================
# Task: Evaluate (THE ONLY MLflow logging point)
# ============================================================
def _run_evaluation(checkpoint, shards, train_metadata, dataset, experiment_name):
    """Shared open-loop evaluation + MLflow logging logic.

    Called by both evaluate_il_policy and evaluate_rl_policy. Kept as a plain
    module-level function (not a @task) so the two evaluation tasks share one
    implementation while appearing as distinct nodes in the Flyte UI.
    """
    import os
    import json
    import yaml
    import torch
    import numpy as np
    import mlflow
    from flytekit import current_context

    from model_components.auto_e2e import AutoE2E
    from data_parsing.pre_extracted import make_multi_dataset_loader
    from evaluation.metrics import integrate_trajectory

    # Eval uses num_workers=4; use the file_system sharing strategy so the small
    # pod /dev/shm doesn't bus-error on WM-window batches (same as train_il, #121).
    torch.multiprocessing.set_sharing_strategy("file_system")

    ckpt_path = checkpoint.download()
    from Platform.pipelines.inference import sha256_file
    checkpoint_sha256 = sha256_file(ckpt_path)
    # Sharded fan-out returns N per-partition dirs; eval over ALL of them so
    # ADE/FDE covers the full held-out set, not partition 0 only (Flyte-review B2).
    shard_dirs = _select_shard_dirs(shards, dataset)
    meta = json.load(open(train_metadata.download()))
    ctx = current_context()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = AutoE2E(**_model_kwargs(config)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate on the HELD-OUT split this checkpoint trained around (recorded in
    # the train metadata), so ADE/FDE measure generalization, not training-set
    # memorization. val_fraction=0 (legacy) → score all samples (in-sample).
    val_fraction = float(meta.get("training", {}).get("val_fraction", 0.0) or 0.0)
    eval_split = "val" if val_fraction > 0.0 else "all"
    loader = make_multi_dataset_loader(shard_dirs, batch_size=8, num_workers=4, shuffle=0,
                                       pin_memory=(device.type == "cuda"),
                                       split=eval_split, val_fraction=val_fraction)
    print(f"Eval split={eval_split} (val_fraction={val_fraction}, {len(shard_dirs)} partitions) — "
          f"{'held-out generalization' if eval_split == 'val' else 'in-sample'}")
    all_ade, all_fde = [], []

    with torch.no_grad():
        # MergedDatasetLoader yields (batch, projection, geometry_type) per batch,
        # so each partition's per-dataset geometry is preserved (a single rig for
        # L2D, but the shape is future-proof for multi-dataset eval).
        for batch, projection, geometry_type in loader:
            if projection is not None:
                projection = projection.to(device)
            # WM rolling buffer is per-sequence state; reset per batch so batch N's
            # planner history is not built from unrelated prior batches (leakage),
            # and a ragged final batch cannot crash torch.cat over mixed batch dims.
            if hasattr(model, "reset_visual_history"):
                model.reset_visual_history()
            visual = batch["visual_tiles"].to(device)
            ego_hist = batch["egomotion_history"].to(device)
            vis_hist = batch["visual_history"].to(device)
            target = batch["trajectory_target"]  # (B, 128) on CPU
            map_input = batch["map_input"].to(device)

            # Train/eval consistency (#13): if the shard carries World-Model
            # windows, feed them so eval runs the SAME windowed path the model
            # trained on — the planner sees the DENSE WM-derived visual_history it
            # learned to use, not the mostly-zero rolling-buffer vector. Without
            # this, a WM-trained model is evaluated out-of-distribution (the
            # planner's visual_history is 3/4 zeros) and ADE inflates. On shards
            # without WM windows (imitation-only) batch.get returns None and the
            # model takes its normal path — identical to before. Future prediction
            # is gated on mode=="train", so mode="infer" safely skips it.
            history_frames = batch.get("history_frames")
            future_frames = batch.get("future_frames")
            if history_frames is not None:
                history_frames = history_frames.to(device)
            if future_frames is not None:
                future_frames = future_frames.to(device)

            pred = model(visual, map_input, vis_hist, ego_hist,
                         projection=projection, geometry_type=geometry_type,
                         history_frames=history_frames, future_frames=future_frames,
                         mode="infer")
            pred = pred.cpu().numpy()  # (B, 128)
            target_np = target.numpy()

            for i in range(pred.shape[0]):
                # Reshape: (64, 2) = [accel_x, curvature]
                pred_signals = pred[i].reshape(64, 2)
                gt_signals = target_np[i].reshape(64, 2)
                # Get initial speed from egomotion history (first signal)
                ego_np = batch["egomotion_history"][i].numpy()
                v0 = float(ego_np[-4])  # last speed value in history

                pred_traj = integrate_trajectory(pred_signals[:, 0], pred_signals[:, 1], v0)
                gt_traj = integrate_trajectory(gt_signals[:, 0], gt_signals[:, 1], v0)

                ade = float(np.mean(np.linalg.norm(pred_traj - gt_traj, axis=1)))
                fde = float(np.linalg.norm(pred_traj[-1] - gt_traj[-1]))
                all_ade.append(ade)
                all_fde.append(fde)

    avg_ade = float(np.mean(all_ade)) if all_ade else 99.0
    avg_fde = float(np.mean(all_fde)) if all_fde else 99.0
    passed = avg_ade < 2.0 and avg_fde < 4.0

    # --- MLflow logging ---
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)

    model_info = meta.get("model", meta.get("base_model", {}).get("il_metadata", {}).get("model", {}))
    bb = model_info.get("backbone", "?")
    fm = model_info.get("fusion_mode", "?")
    training = meta.get("training", meta.get("base_model", {}).get("il_metadata", {}).get("training", {}))
    run_name = f"{bb}-{fm}-e{training.get('epochs','?')}"

    with mlflow.start_run(run_name=run_name):
        # Flatten params
        params = {}
        data = meta.get("data", meta.get("base_model", {}).get("il_metadata", {}).get("data", {}))
        params["data/dataset"] = data.get("dataset", "?")
        params["data/dataset_version"] = data.get("dataset_version", "?")
        params["model/backbone"] = bb
        params["model/fusion_mode"] = fm
        params["train/epochs"] = training.get("epochs", "?")
        params["train/batch_size"] = training.get("batch_size", "?")
        params["train/lr"] = training.get("lr", "?")
        params["train/weight_decay"] = training.get("weight_decay", "?")
        params["train/amp"] = training.get("amp", "?")
        params["train/final_loss"] = training.get("final_loss", "?")
        params["train/val_fraction"] = training.get("val_fraction", 0.0)
        params["model/checkpoint_sha256"] = checkpoint_sha256

        # RL params
        if "rl" in meta:
            rl = meta["rl"]
            params["rl/method"] = rl.get("method", "?")
            params["rl/tau"] = rl.get("tau", "?")
            params["rl/beta"] = rl.get("beta", "?")
            params["rl/epochs"] = rl.get("epochs", "?")

        # Context
        train_ctx = meta.get("context", {})
        params["ctx/train_execution_id"] = train_ctx.get("flyte_execution_id", "?")
        params["ctx/train_docker_image"] = train_ctx.get("docker_image", "?")
        params["ctx/eval_execution_id"] = ctx.execution_id.name if ctx.execution_id else "local"
        params["ctx/eval_docker_image"] = EVAL_IMAGE

        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.set_tags({
            "pipeline": experiment_name,
            "backbone": bb,
            "fusion": fm,
            "checkpoint_sha256": checkpoint_sha256,
        })

        # Training loss curve
        for i, loss_val in enumerate(training.get("losses_per_epoch", [])):
            mlflow.log_metric("train/loss", loss_val, step=i)

        # Eval metrics
        mlflow.log_metrics({"eval/ade": avg_ade, "eval/fde": avg_fde, "eval/gate_pass": 1.0 if passed else 0.0})

        # Artifacts
        os.makedirs("/tmp/eval-artifacts", exist_ok=True)
        with open("/tmp/eval-artifacts/config.yaml", "w") as f:
            yaml.dump(meta, f)
        mlflow.log_artifact("/tmp/eval-artifacts/config.yaml")
        mlflow.log_artifact(ckpt_path, artifact_path="model")

        # Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        try:
            registered = mlflow.register_model(
                model_uri, "auto-e2e-driving-policy"
            )
            mlflow.tracking.MlflowClient().set_model_version_tag(
                "auto-e2e-driving-policy",
                registered.version,
                "checkpoint_sha256",
                checkpoint_sha256,
            )
        except Exception as e:
            print(f"Registry: {e}")

    print(f"Eval: ADE={avg_ade:.3f} FDE={avg_fde:.3f} Gate={'PASS' if passed else 'FAIL'}")
    return EvalMetrics(ade=avg_ade, fde=avg_fde, gate_pass=passed)


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="2", mem="8Gi", gpu="1"),
    limits=Resources(cpu="2", mem="8Gi", gpu="1"),
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
    pod_template=_large_shm_pod_template(),  # /dev/shm for eval DataLoader workers (#121 P0)
)
def evaluate_il_policy(
    checkpoint: FlyteFile,
    shards: List[FlyteDirectory],
    train_metadata: FlyteFile,
    dataset: Dataset = Dataset.L2D,
) -> EvalMetrics:
    """Open-loop evaluation of the Imitation-Learning policy.

    Logs ADE/FDE, params, artifacts to the MLflow `imitation-learning` experiment
    and registers the checkpoint in the `auto-e2e-driving-policy` model registry.
    """
    return _run_evaluation(checkpoint, shards, train_metadata, dataset, "imitation-learning")


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="2", mem="8Gi", gpu="1"),
    limits=Resources(cpu="2", mem="8Gi", gpu="1"),
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
    pod_template=_large_shm_pod_template(),  # /dev/shm for eval DataLoader workers (#121 P0)
)
def evaluate_rl_policy(
    checkpoint: FlyteFile,
    shards: List[FlyteDirectory],
    train_metadata: FlyteFile,
    dataset: Dataset = Dataset.L2D,
) -> EvalMetrics:
    """Open-loop evaluation of the Offline-RL refined policy.

    Logs ADE/FDE, params (incl. rl/*), artifacts to the MLflow `offline-rl`
    experiment and registers the refined checkpoint in the model registry.
    """
    return _run_evaluation(checkpoint, shards, train_metadata, dataset, "offline-rl")



# ============================================================
# Workflows
# ============================================================
@workflow
def wf_data_ingest(
    dataset: Dataset = Dataset.L2D,
    source_revision: str = L2D_SOURCE_REVISION,
    episodes: int = 3,
) -> FlyteDirectory:
    """Download raw dataset from HuggingFace."""
    return data_ingest(
        dataset=dataset,
        source_revision=source_revision,
        episodes=episodes,
    )


@workflow
def wf_data_processing(
    raw_data: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    source_revision: str = L2D_SOURCE_REVISION,
    dataset_version: str = DATASET_PACK_VERSION,
    hz: int = 10,
    image_size: int = 256,
    episodes: int = 3,
    world_model: bool = False,
    reasoning_labels: Optional[FlyteDirectory] = None,
) -> FlyteDirectory:
    """Pre-process raw data → WebDataset shards.

    ``world_model`` packs the JEPA per-sample windows (#13). ``reasoning_labels``
    (the generate_reasoning_labels artifact) is JOINed into reasoning.json (#98).
    Both MUST match the branch flags used at ``train_il`` time or that branch
    trains unsupervised.
    """
    return data_processing(raw_data=raw_data, dataset=dataset,
                           source_revision=source_revision,
                           dataset_version=dataset_version,
                           hz=hz, image_size=image_size, episodes=episodes,
                           world_model=world_model, reasoning_labels=reasoning_labels)


@workflow
def wf_generate_reasoning_labels(
    raw_data: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    source_revision: str = L2D_SOURCE_REVISION,
    episodes: int = 3,
    split: str = "train",
    teacher: str = "openai_compatible",
    prompt_version: str = "action_relevant_reasoning_v3_temporal_front256",
) -> FlyteDirectory:
    """Label raw samples with the offline teacher (S3-cached) → versioned artifact."""
    return generate_reasoning_labels(
        raw_data=raw_data, dataset=dataset,
        source_revision=source_revision,
        episodes=episodes, split=split,
        teacher=teacher, prompt_version=prompt_version)


@workflow
def _pack_with_labels(
    raw: FlyteDirectory,
    dataset: Dataset,
    source_revision: str,
    dataset_version: str,
    episodes: int,
    image_size: int,
    world_model: bool,
    teacher: str,
    prompt_version: str,
) -> FlyteDirectory:
    """The 'with reasoning labels' branch of wf_create_dataset: label from raw
    (teacher, S3-cached) → pack shards with the labels JOINed in.

    A Flyte conditional branch is a single node, so the two-task label→pack chain
    lives in this sub-workflow.

    Reasoning labels are built from the 1 Hz World-Model window (temporal front
    clip), and ``len(L2DDataset)`` / sample ordering depend on
    ``include_world_model_windows``. So both generate and data_processing MUST run
    with world_model=True for the ``sample_id`` JOIN to align — we force it on
    here (the ``world_model`` arg is ignored on the labelled branch). Training can
    still ignore the JEPA windows if enable_world_model is off.
    """
    labels = generate_reasoning_labels(
        raw_data=raw, dataset=dataset, source_revision=source_revision,
        episodes=episodes, split="train",
        teacher=teacher, prompt_version=prompt_version)
    return data_processing(
        raw_data=raw, dataset=dataset, source_revision=source_revision,
        dataset_version=dataset_version,
        episodes=episodes, image_size=image_size,
        world_model=True, reasoning_labels=labels)


@workflow
def wf_create_dataset(
    dataset: Dataset = Dataset.L2D,
    source_revision: str = L2D_SOURCE_REVISION,
    dataset_version: str = DATASET_PACK_VERSION,
    episodes: int = 3,
    image_size: int = 256,
    world_model: bool = False,
    reasoning_teacher: str = "none",
    prompt_version: str = "action_relevant_reasoning_v3_temporal_front256",
) -> FlyteDirectory:
    """CreateDataset: raw → ready-to-train WebDataset shards.

    "Dataset" means data already in a form training consumes DIRECTLY: the
    WebDataset shards (frames + ego + optional WM windows + per-sample
    reasoning.json when a teacher is set). train_il reads its reasoning
    supervision from those in-shard members — the shards ARE the dataset.

    Reasoning labels are generated once by ``generate_reasoning_labels`` (the only
    place the teacher is called; each sample S3-cached so re-packing never
    re-bills it, #117) and JOINed into the shards by ``data_processing``. The
    versioned label artifact persists independently in S3 (task output + cache),
    so it need not be a workflow return value — the shards are the single output.

    Chains: data_ingest → [teacher != none] generate_reasoning_labels →
    data_processing (JOIN labels). With reasoning_teacher="none", no labels are
    generated and the shards carry no reasoning.json (imitation-only).
    """
    from flytekit import conditional

    raw = data_ingest(
        dataset=dataset,
        source_revision=source_revision,
        episodes=episodes,
    )
    return (
        conditional("reasoning_labels")
        .if_(reasoning_teacher != "none")
        .then(_pack_with_labels(
            raw=raw, dataset=dataset, source_revision=source_revision,
            episodes=episodes, image_size=image_size,
            dataset_version=dataset_version,
            world_model=world_model, teacher=reasoning_teacher,
            prompt_version=prompt_version))
        .else_()
        .then(data_processing(
            raw_data=raw, dataset=dataset, source_revision=source_revision,
            dataset_version=dataset_version,
            episodes=episodes,
            image_size=image_size, world_model=world_model))
    )


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
) -> List[FlyteDirectory]:
    """Execute each data-prep stage as one bounded Flyte array node."""
    for name, value in (
        ("ingest_concurrency", ingest_concurrency),
        ("label_concurrency", label_concurrency),
        ("pack_concurrency", pack_concurrency),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    ingest = map_task(
        functools.partial(
            data_ingest,
            dataset=dataset,
            source_revision=source_revision,
            episodes=0,
        ),
        concurrency=ingest_concurrency,
    )
    raw_dirs = ingest(group_ids=partitions)

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
            ),
            concurrency=pack_concurrency,
        )
        return pack(
            raw_data=raw_dirs,
            reasoning_labels=label_dirs,
            group_ids=partitions,
        )

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
        ),
        concurrency=pack_concurrency,
    )
    return pack(raw_data=raw_dirs, group_ids=partitions)


@workflow
def wf_create_dataset_sharded(
    dataset: Dataset = Dataset.KITSCENES,
    source_revision: str = KITSCENES_SOURCE_REVISION,
    dataset_version: str = DATASET_PACK_VERSION,
    episodes: int = 10,
    start_ep: int = -1,
    end_ep: int = -1,
    partition_size: int = 1,
    image_size: int = 256,
    world_model: bool = False,
    reasoning_teacher: str = "none",
    prompt_version: str = "action_relevant_reasoning_v3_temporal_front256",
    label_stride: int = 10,
    label_workers: int = 2,
    max_partitions: int = 600,
    max_missing_scenes: int = 1,
    ingest_concurrency: int = 60,
    label_concurrency: int = 5,
    pack_concurrency: int = 60,
) -> List[FlyteDirectory]:
    """Fan out immutable source groups through bounded ingest/label/pack arrays.

    KITScenes uses one scene per mapped pod. With ``episodes=0`` the preflight
    resolves all available official train scenes (currently 533/534 at the
    pinned v1.0.1 source revision), permits only the known one-scene deficit, and
    then runs 60 ingest pods, 5 label pods, and 60 pack pods concurrently.
    """
    partitions = plan_fanout_partitions(
        dataset=dataset,
        source_revision=source_revision,
        episodes=episodes,
        start_ep=start_ep,
        end_ep=end_ep,
        partition_size=partition_size,
        max_partitions=max_partitions,
        max_missing_scenes=max_missing_scenes,
        split="train",
    )
    return _map_dataset_partitions(
        partitions=partitions,
        dataset=dataset,
        source_revision=source_revision,
        dataset_version=dataset_version,
        image_size=image_size,
        world_model=world_model,
        reasoning_teacher=reasoning_teacher,
        prompt_version=prompt_version,
        label_stride=label_stride,
        label_workers=label_workers,
        ingest_concurrency=ingest_concurrency,
        label_concurrency=label_concurrency,
        pack_concurrency=pack_concurrency,
    )


@workflow
def wf_sharded_full_run(
    dataset: Dataset = Dataset.KITSCENES,
    source_revision: str = KITSCENES_SOURCE_REVISION,
    dataset_version: str = DATASET_PACK_VERSION,
    episodes: int = 10,
    partition_size: int = 1,
    image_size: int = 256,
    reasoning_teacher: str = "openai_compatible",
    prompt_version: str = "action_relevant_reasoning_v3_temporal_front256",
    label_stride: int = 10,
    label_workers: int = 2,
    max_partitions: int = 600,
    max_missing_scenes: int = 1,
    ingest_concurrency: int = 60,
    label_concurrency: int = 5,
    pack_concurrency: int = 60,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    lr: float = 1e-4,
    enable_reasoning: bool = True,
    reasoning_mode: str = "pooled_latent",
    enable_world_model: bool = True,
    val_fraction: float = 0.1,
    num_workers: int = 4,
) -> EvalMetrics:
    """End-to-end scaled run (#121): episode-sharded dataset fan-out → IL train
    (all three losses) → held-out eval, in ONE execution.

    Chains ``wf_create_dataset_sharded`` (option B fan-out producing per-partition
    deduped WM shards with 1 Hz reasoning labels) straight into ``train_il`` over
    the merged ``List[FlyteDirectory]`` and then ``evaluate_il_policy`` on the
    disjoint held-out split. Defaults turn on BOTH the reasoning and world-model
    branches (the full 3-branch objective) with WM-friendly batch_size=1 +
    grad_accum, and a 10% group-level val split so ADE/FDE measure generalization.

    This is the entry point for "train on ALL episodes": set episodes=0 (all) and a
    cost-appropriate partition_size. Training is serial (single GPU); only the data
    pipeline fans out.
    """
    shards = wf_create_dataset_sharded(
        dataset=dataset, source_revision=source_revision,
        dataset_version=dataset_version,
        episodes=episodes, partition_size=partition_size,
        image_size=image_size, world_model=True,
        reasoning_teacher=reasoning_teacher, prompt_version=prompt_version,
        label_stride=label_stride, label_workers=label_workers,
        max_partitions=max_partitions,
        max_missing_scenes=max_missing_scenes,
        ingest_concurrency=ingest_concurrency,
        label_concurrency=label_concurrency,
        pack_concurrency=pack_concurrency)
    out = train_il(
        shards=shards, dataset=dataset, backbone=backbone, epochs=epochs,
        batch_size=batch_size, grad_accum_steps=grad_accum_steps, lr=lr,
        enable_reasoning=enable_reasoning, reasoning_mode=reasoning_mode,
        enable_world_model=enable_world_model, val_fraction=val_fraction,
        num_workers=num_workers)
    return evaluate_il_policy(
        checkpoint=out.checkpoint, shards=shards, dataset=dataset,
        train_metadata=out.metadata)


@workflow
def wf_train_il(
    shards: List[FlyteDirectory],
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    lr: float = 1e-4,
    amp: bool = False,
    enable_reasoning: bool = False,
    reasoning_mode: str = "pooled_latent",
    enable_world_model: bool = False,
    val_fraction: float = 0.0,
    num_workers: int = 0,
) -> EvalMetrics:
    """IL Train → Evaluate. All datasets' shards passed in; `dataset` selects one.

    The branch flags must match how the shards were packed (see
    ``wf_data_processing``); train_il fails loudly if a branch is enabled but its
    shard data is missing rather than training it unsupervised. ``amp`` defaults
    off: fp16 autocast made the GradScaler skip every step (see train_il).
    ``grad_accum_steps`` recovers a larger effective batch when the World-Model
    windows force batch_size=1 (effective batch = batch_size * grad_accum_steps).
    ``val_fraction`` > 0 trains on a per-sample train split and evaluates on the
    disjoint held-out val split (generalization, not in-sample memorization).
    ``num_workers`` > 0 parallelizes JPEG decode across worker processes (#121 P0)
    — the dominant per-epoch cost once episodes scale up.
    """
    out = train_il(shards=shards, dataset=dataset, backbone=backbone,
                   epochs=epochs, batch_size=batch_size,
                   grad_accum_steps=grad_accum_steps, lr=lr, amp=amp,
                   enable_reasoning=enable_reasoning, reasoning_mode=reasoning_mode,
                   enable_world_model=enable_world_model, val_fraction=val_fraction,
                   num_workers=num_workers)
    return evaluate_il_policy(checkpoint=out.checkpoint, shards=shards, dataset=dataset,
                              train_metadata=out.metadata)


@workflow
def wf_train_offline_rl(
    pretrained: FlyteFile,
    shards: List[FlyteDirectory],
    il_metadata: FlyteFile,
    dataset: Dataset = Dataset.L2D,
    epochs: int = 3,
    tau: float = 0.7,
    beta: float = 3.0,
) -> EvalMetrics:
    """Offline RL → Evaluate. All datasets' shards passed in; `dataset` selects one."""
    out = train_offline_rl(pretrained=pretrained, shards=shards, dataset=dataset,
                           il_metadata=il_metadata, epochs=epochs, tau=tau, beta=beta)
    return evaluate_rl_policy(checkpoint=out.checkpoint, shards=shards, dataset=dataset,
                              train_metadata=out.metadata)


@workflow
def wf_full_pipeline(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs_il: int = 3,
    epochs_rl: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    tau: float = 0.7,
    beta: float = 3.0,
) -> EvalMetrics:
    """Full: Ingest+Process ALL datasets (separately packed) → IL Train+Eval → RL Train+Eval.

    Every dataset is ingested and processed into its own WebDataset shard dir, and
    all shard dirs are passed to the train/eval tasks. The `dataset` argument selects
    which one is actually used for this run (single-dataset training; multi-dataset
    on one model tracked in #77).
    """
    # Ingest + process every dataset into separate WebDataset shard dirs
    raw_l2d = data_ingest(dataset=Dataset.L2D, episodes=episodes)
    shards_l2d = data_processing(raw_data=raw_l2d, dataset=Dataset.L2D, episodes=episodes)

    raw_nv = data_ingest(dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)
    shards_nv = data_processing(raw_data=raw_nv, dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)

    all_shards = [shards_l2d, shards_nv]

    il_out = train_il(shards=all_shards, dataset=dataset, backbone=backbone,
                      epochs=epochs_il, batch_size=batch_size, lr=lr)
    evaluate_il_policy(checkpoint=il_out.checkpoint, shards=all_shards, dataset=dataset,
                       train_metadata=il_out.metadata)
    rl_out = train_offline_rl(pretrained=il_out.checkpoint, shards=all_shards, dataset=dataset,
                              il_metadata=il_out.metadata, epochs=epochs_rl, tau=tau, beta=beta)
    return evaluate_rl_policy(checkpoint=rl_out.checkpoint, shards=all_shards, dataset=dataset,
                              train_metadata=rl_out.metadata)


@workflow
def wf_ingest_train_eval(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs_il: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
) -> EvalMetrics:
    """Ingest+Process ALL datasets → IL Train → IL Eval (no offline RL).

    Same as wf_full_pipeline but stops after IL evaluation. Useful when you only
    want a supervised checkpoint + open-loop metrics, or when the offline-RL step
    is too memory-hungry to co-run at the current BEV resolution (#77).
    """
    raw_l2d = data_ingest(dataset=Dataset.L2D, episodes=episodes)
    shards_l2d = data_processing(raw_data=raw_l2d, dataset=Dataset.L2D, episodes=episodes)

    raw_nv = data_ingest(dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)
    shards_nv = data_processing(raw_data=raw_nv, dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)

    all_shards = [shards_l2d, shards_nv]

    il_out = train_il(shards=all_shards, dataset=dataset, backbone=backbone,
                      epochs=epochs_il, batch_size=batch_size, lr=lr)
    return evaluate_il_policy(checkpoint=il_out.checkpoint, shards=all_shards, dataset=dataset,
                              train_metadata=il_out.metadata)


@dynamic(
    container_image=EVAL_IMAGE,
    environment={"AUTO_E2E_EVAL_IMAGE": EVAL_IMAGE},
)
def wf_precompute_overlays(
    shards: List[FlyteDirectory],
    model_version: str,
    dataset_manifest_digest: str,
    preprocessing_contract_digest: str,
    model_inference_code_digest: str,
    container_image_digest: str,
    artifacts_bucket: str,
    expected_train_execution_id: str = "",
    registered_model_name: str = "auto-e2e-driving-policy",
    dataset: str = "l2d",
    dataset_version: str = DATASET_PACK_VERSION,
    dynamo_table: str = "auto-e2e-console",
    aws_region: str = "us-west-2",
    base_seeds: List[int] = [0],
    batch_size: int = 32,
    num_workers: int = 4,
    sampler: str = "model-default",
) -> str:
    """Ops-only canonical trajectory overlay precompute.

    The Console never invokes this workflow. It resolves one immutable MLflow
    model version, marks the overlay set ``building``, runs one coarse GPU task
    per packed partition (loading the checkpoint once for every tar in that
    partition), writes S3 bodies before Dynamo pointers, then publishes the
    audit manifest and flips ``OVLSET`` to ``ready`` last.
    """
    from Platform.pipelines.overlay_tasks import (
        finalize_overlay_set,
        precompute_overlay_partition,
        prepare_overlay_set,
        resolve_overlay_model,
    )

    resolved = resolve_overlay_model(
        registered_model_name=registered_model_name,
        model_version=model_version,
        expected_train_execution_id=expected_train_execution_id,
    )
    gate = prepare_overlay_set(
        resolved_metadata=resolved.metadata,
        dataset=dataset,
        dataset_version=dataset_version,
        dataset_manifest_digest=dataset_manifest_digest,
        preprocessing_contract_digest=preprocessing_contract_digest,
        model_inference_code_digest=model_inference_code_digest,
        container_image_digest=container_image_digest,
        artifacts_bucket=artifacts_bucket,
        dynamo_table=dynamo_table,
        aws_region=aws_region,
        base_seeds=base_seeds,
        sampler=sampler,
    )
    results: List[FlyteFile] = []
    for partition in shards:
        results.append(precompute_overlay_partition(
            checkpoint=resolved.checkpoint,
            model_metadata=resolved.metadata,
            prepare_gate=gate,
            shard_dir=partition,
            dataset=dataset,
            dataset_version=dataset_version,
            dataset_manifest_digest=dataset_manifest_digest,
            preprocessing_contract_digest=preprocessing_contract_digest,
            model_inference_code_digest=model_inference_code_digest,
            container_image_digest=container_image_digest,
            artifacts_bucket=artifacts_bucket,
            dynamo_table=dynamo_table,
            aws_region=aws_region,
            base_seeds=base_seeds,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
        ))
    return finalize_overlay_set(
        model_metadata=resolved.metadata,
        partition_results=results,
        prepare_gate=gate,
        dataset=dataset,
        dataset_version=dataset_version,
        dataset_manifest_digest=dataset_manifest_digest,
        artifacts_bucket=artifacts_bucket,
        dynamo_table=dynamo_table,
        aws_region=aws_region,
    )


@dynamic(
    container_image=DATA_PREP_IMAGE,
    environment={"AUTO_E2E_DATA_PREP_IMAGE": DATA_PREP_IMAGE},
)
def wf_publish_dataset_snapshot(
    shards: List[FlyteDirectory],
    published_dataset: str,
    datasets_bucket: str,
    dataset_version: str = DATASET_PACK_VERSION,
    dynamo_table: str = "auto-e2e-console",
    aws_region: str = "us-west-2",
    copy_workers: int = 16,
) -> DatasetPublication:
    """Publish packed partitions under one immutable Console dataset version.

    This is an ops-only workflow. It copies shard, frame-pool, and exact-route
    bodies from Flyte's artifact bucket with conditional S3 writes, merges the
    rig and privacy-filtered geographic products, writes the canonical manifest
    last, and only then updates the Console's ``GEO#`` pointer. The returned
    manifest digest is the dataset identity required by ``wf_precompute_overlays``.
    """
    from Platform.pipelines.dataset_publication_tasks import (
        finalize_dataset_publication,
        publish_dataset_partition,
    )

    results: List[FlyteFile] = []
    for partition in shards:
        results.append(publish_dataset_partition(
            shard_dir=partition,
            published_dataset=published_dataset,
            dataset_version=dataset_version,
            datasets_bucket=datasets_bucket,
            aws_region=aws_region,
            copy_workers=copy_workers,
        ))
    return finalize_dataset_publication(
        partition_results=results,
        published_dataset=published_dataset,
        dataset_version=dataset_version,
        datasets_bucket=datasets_bucket,
        dynamo_table=dynamo_table,
        aws_region=aws_region,
    )


@workflow
def wf_publish_and_precompute_overlays(
    shards: List[FlyteDirectory],
    model_version: str,
    preprocessing_contract_digest: str,
    model_inference_code_digest: str,
    container_image_digest: str,
    datasets_bucket: str,
    artifacts_bucket: str,
    expected_train_execution_id: str = "",
    published_dataset: str = "kitscenes",
    registered_model_name: str = "auto-e2e-driving-policy",
    dataset_version: str = DATASET_PACK_VERSION,
    dynamo_table: str = "auto-e2e-console",
    aws_region: str = "us-west-2",
    base_seeds: List[int] = [0],
    batch_size: int = 32,
    num_workers: int = 4,
    copy_workers: int = 16,
) -> PublishedOverlayOutput:
    """Publish one immutable snapshot, then precompute its model overlays.

    The dataset manifest digest is wired directly between Flyte nodes, so an
    operator cannot accidentally launch inference against a different snapshot.
    """
    publication = wf_publish_dataset_snapshot(
        shards=shards,
        published_dataset=published_dataset,
        datasets_bucket=datasets_bucket,
        dataset_version=dataset_version,
        dynamo_table=dynamo_table,
        aws_region=aws_region,
        copy_workers=copy_workers,
    )
    overlay_result = wf_precompute_overlays(
        shards=shards,
        model_version=model_version,
        dataset_manifest_digest=publication.manifest_sha256,
        preprocessing_contract_digest=preprocessing_contract_digest,
        model_inference_code_digest=model_inference_code_digest,
        container_image_digest=container_image_digest,
        artifacts_bucket=artifacts_bucket,
        expected_train_execution_id=expected_train_execution_id,
        registered_model_name=registered_model_name,
        dataset=published_dataset,
        dataset_version=dataset_version,
        dynamo_table=dynamo_table,
        aws_region=aws_region,
        base_seeds=base_seeds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler="model-default",
    )
    return PublishedOverlayOutput(
        overlay_result=overlay_result,
        manifest_key=publication.manifest_key,
        manifest_sha256=publication.manifest_sha256,
    )


@workflow
def wf_publish_full_run_overlays(
    shards: List[FlyteDirectory],
    full_run_execution_id: str,
    preprocessing_contract_digest: str,
    model_inference_code_digest: str,
    container_image_digest: str,
    datasets_bucket: str,
    artifacts_bucket: str,
    published_dataset: str = "kitscenes",
    registered_model_name: str = "auto-e2e-driving-policy",
    source_dataset: str = Dataset.KITSCENES.value,
    dataset_version: str = DATASET_PACK_VERSION,
    dynamo_table: str = "auto-e2e-console",
    aws_region: str = "us-west-2",
    base_seeds: List[int] = [0],
    batch_size: int = 32,
    num_workers: int = 4,
    copy_workers: int = 16,
) -> PublishedOverlayOutput:
    """Publish the labeled shards and model produced by one completed Full Run."""
    from Platform.pipelines.overlay_tasks import (
        resolve_overlay_model_version,
    )

    model_version = resolve_overlay_model_version(
        registered_model_name=registered_model_name,
        train_execution_id=full_run_execution_id,
        expected_dataset=source_dataset,
        expected_dataset_version=dataset_version,
    )
    return wf_publish_and_precompute_overlays(
        shards=shards,
        model_version=model_version,
        preprocessing_contract_digest=preprocessing_contract_digest,
        model_inference_code_digest=model_inference_code_digest,
        container_image_digest=container_image_digest,
        datasets_bucket=datasets_bucket,
        artifacts_bucket=artifacts_bucket,
        expected_train_execution_id=full_run_execution_id,
        published_dataset=published_dataset,
        registered_model_name=registered_model_name,
        dataset_version=dataset_version,
        dynamo_table=dynamo_table,
        aws_region=aws_region,
        base_seeds=base_seeds,
        batch_size=batch_size,
        num_workers=num_workers,
        copy_workers=copy_workers,
    )


@workflow
def wf_create_publish_and_precompute_overlays(
    model_version: str,
    preprocessing_contract_digest: str,
    model_inference_code_digest: str,
    container_image_digest: str,
    datasets_bucket: str,
    artifacts_bucket: str,
    dataset: Dataset = Dataset.KITSCENES,
    source_revision: str = KITSCENES_SOURCE_REVISION,
    published_dataset: str = "kitscenes",
    dataset_version: str = DATASET_PACK_VERSION,
    episodes: int = 0,
    start_ep: int = -1,
    end_ep: int = -1,
    partition_size: int = 1,
    image_size: int = 256,
    reasoning_teacher: str = "openai_compatible",
    prompt_version: str = "action_relevant_reasoning_v3_temporal_front256",
    label_stride: int = 10,
    label_workers: int = 2,
    max_partitions: int = 600,
    max_missing_scenes: int = 1,
    ingest_concurrency: int = 60,
    label_concurrency: int = 5,
    pack_concurrency: int = 60,
    registered_model_name: str = "auto-e2e-driving-policy",
    dynamo_table: str = "auto-e2e-console",
    aws_region: str = "us-west-2",
    base_seeds: List[int] = [0],
    batch_size: int = 32,
    num_workers: int = 4,
    copy_workers: int = 16,
) -> PublishedOverlayOutput:
    """Build, publish, and overlay one dataset without manual URI handoff."""
    shards = wf_create_dataset_sharded(
        dataset=dataset,
        source_revision=source_revision,
        dataset_version=dataset_version,
        episodes=episodes,
        start_ep=start_ep,
        end_ep=end_ep,
        partition_size=partition_size,
        image_size=image_size,
        world_model=True,
        reasoning_teacher=reasoning_teacher,
        prompt_version=prompt_version,
        label_stride=label_stride,
        label_workers=label_workers,
        max_partitions=max_partitions,
        max_missing_scenes=max_missing_scenes,
        ingest_concurrency=ingest_concurrency,
        label_concurrency=label_concurrency,
        pack_concurrency=pack_concurrency,
    )
    return wf_publish_and_precompute_overlays(
        shards=shards,
        model_version=model_version,
        preprocessing_contract_digest=preprocessing_contract_digest,
        model_inference_code_digest=model_inference_code_digest,
        container_image_digest=container_image_digest,
        datasets_bucket=datasets_bucket,
        artifacts_bucket=artifacts_bucket,
        published_dataset=published_dataset,
        registered_model_name=registered_model_name,
        dataset_version=dataset_version,
        dynamo_table=dynamo_table,
        aws_region=aws_region,
        base_seeds=base_seeds,
        batch_size=batch_size,
        num_workers=num_workers,
        copy_workers=copy_workers,
    )


@workflow
def wf_export_trajectory_report(
    shard: FlyteFile,
    overlay: FlyteFile,
    dataset_manifest: FlyteFile,
    overlay_manifest: FlyteFile,
    selection_manifest: Optional[FlyteFile] = None,
    scene_uids: List[str] = [],
    seed_index: int = 0,
    camera_index: int = 0,
    max_frames_per_scene: int = 300,
    fps: float = 10.0,
) -> FlyteDirectory:
    """Render a canonical shard overlay as per-scene MP4 artifacts."""
    from Platform.pipelines.trajectory_visualization_tasks import (
        export_trajectory_report,
    )

    return export_trajectory_report(
        shard=shard,
        overlay=overlay,
        dataset_manifest=dataset_manifest,
        overlay_manifest=overlay_manifest,
        selection_manifest=selection_manifest,
        scene_uids=scene_uids,
        seed_index=seed_index,
        camera_index=camera_index,
        max_frames_per_scene=max_frames_per_scene,
        fps=fps,
    )
