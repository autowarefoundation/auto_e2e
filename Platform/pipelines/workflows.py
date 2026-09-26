"""AutoE2E Flyte-native workflows — Real Training Pipeline.

Architecture:
  data_ingest → data_processing → train_il → evaluate
                                      ↓
                              train_offline_rl → evaluate

MLflow: Training logs epoch metrics; evaluation logs final metrics and registry
entries. Two experiments: imitation-learning and offline-rl.
"""
import enum
import functools
from flytekit import (
    task, workflow, dynamic, map_task, Resources, Secret, BatchSize,
)
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from typing import Annotated, NamedTuple, List, Optional

from data_processing.contract_versions import (
    GEOMETRY_VERSION as _GEOM_V,
    PARSER_VERSION as _PARSER_V,
    REASONING_LABEL_POLICY_VERSION as _LABEL_POLICY_V,
    SHARD_SCHEMA_VERSION as _SHARD_V,
    UID_SCHEMA_VERSION as _UID_V,
)
from Platform.pipelines.dataset_publication import DatasetPublication
from Platform.pipelines.overlay_tasks import (
    register_selected_overlay_checkpoint,
    resolve_overlay_model_version,
)
from Platform.pipelines.trajectory_visualization_tasks import (
    export_trajectory_report,
)

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
DATASET_PACK_VERSION = "v2.2"
KITSCENES_NAVIGATION_DATASET_VERSION = "v3.3"
BASELINE_TRAINING_OBJECTIVE_VERSION = "trajectory_imitation_v1"
KITSCENES_NAVIGATION_OBJECTIVE_VERSION = (
    "kitscenes_navigation_objective_v1"
)
ROLLOUT_ALIGNED_OBJECTIVE_VERSION = "rollout_aligned_planner_v1"
ROLLOUT_ALIGNED_CONTROL_OBJECTIVE_VERSION = "rollout_aligned_control_v1"
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
# bumped, §3.4c). Registration must put Model/ on PYTHONPATH; an import failure
# is fatal because guessing these values can silently reuse incompatible cache
# entries. Per-partition group_ids and source_revision travel as task INPUTS, so
# ranges are independently cacheable.

# Each stage's cache_version folds in ONLY the contracts that actually determine
# its output (§3.4a): ingest depends on the parser enumeration; labels also on
# the uid format (the JOIN key) and sparse-selection policy; pack on the shard
# and geometry encoding.
_DEPLOYED_CACHE_VERSION_ALIASES = {
    # Registrations before contract imports became fail-closed serialized the
    # fallback v1 strings even though these were the contracts in the task
    # images. Preserve only the exact ingest/reasoning tuples so those expensive
    # KITScenes caches remain reusable. Pack must follow geometry contract bumps;
    # any other contract tuple falls through to a contract-derived cache version.
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
PACK_CACHE_VERSION = _CACHE_VERSIONS["pack"]
NAVIGATION_QUALITY_CACHE_VERSION = "navigation-quality-v2"


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
        annotations={"karpenter.sh/do-not-disrupt": "true"},
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


# Map-BEV fusion is selectable in the model (MAP_FUSION_REGISTRY) and accepted by
# AutoE2E.__init__, but train_il never forwarded it, so every run trained the
# constructor default regardless of intent (#168).
#
# CROSS_ATTN is dense O(N^2) attention: MapCrossAttentionFusion refuses above
# 4096 BEV tokens and the KITScenes contract grid is 256x256 = 65536, so
# DEFORMABLE is the attention-based option at that resolution.
class MapFusion(enum.Enum):
    RESIDUAL = "residual"
    CROSS_ATTN = "cross_attn"
    DEFORMABLE = "deformable"


def _row_decode_worker_count(dataset: Dataset, row_count: int) -> int:
    """Bound row decoders by each parser's per-process memory footprint."""
    # Each KITScenes child reparses the scene's Lanelet2 map and calibration.
    # Large scenes exceeded the 64 GiB pod limit with the generic 16-worker cap.
    max_workers = 2 if dataset == Dataset.KITSCENES else 16
    return max(1, min(max_workers, row_count))


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
KITScenesBenchmarkOutput = NamedTuple(
    "KITScenesBenchmarkOutput",
    ade_3s=float,
    fde_3s=float,
    ade_5s=float,
    fde_5s=float,
    predictions=FlyteFile,
    report=FlyteFile,
)
ReconstructionAuditOutput = NamedTuple(
    "ReconstructionAuditOutput",
    thresholds_pass=bool,
    report_sha256=str,
    records_sha256=str,
    report=FlyteFile,
    records=FlyteFile,
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


def _verified_navigation_training_shard_dirs(
    shard_dirs: List[str],
    manifests: dict[str, dict],
    report: dict,
) -> tuple[List[str], dict]:
    """Verify the packed audit and select only policy-accepted partitions."""
    from navigation.quality import (
        verify_packed_navigation_quality_audit,
    )

    verified = verify_packed_navigation_quality_audit(
        report,
        shard_dirs,
    )
    path_by_partition = {}
    for shard_dir in shard_dirs:
        partition_id = manifests[shard_dir].get("partition_id")
        if not isinstance(partition_id, str) or not partition_id:
            raise ValueError(
                "KITScenes navigation training requires partition IDs"
            )
        if partition_id in path_by_partition:
            raise ValueError(
                "KITScenes navigation training has duplicate partition ID "
                f"{partition_id!r}"
            )
        path_by_partition[partition_id] = shard_dir

    accepted = set(verified["accepted_partition_ids"])
    excluded = set(verified["excluded_partition_ids"])
    if accepted & excluded or accepted | excluded != set(
        path_by_partition
    ):
        raise ValueError(
            "navigation quality audit partition coverage differs from shards"
        )
    selected = [
        path_by_partition[partition_id]
        for partition_id in sorted(accepted)
    ]
    if not selected:
        raise ValueError(
            "navigation quality policy accepted no training partitions"
        )
    return selected, verified


def _training_num_views_from_manifests(
    manifests: dict[str, dict],
    shard_dirs: List[str],
) -> int:
    """Validate partition camera counts without starting dataset loaders."""
    dataset_views: dict[str, int] = {}
    for shard_dir in shard_dirs:
        manifest = manifests[shard_dir]
        dataset_name = manifest.get("dataset")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError(
                f"packed shard manifest has no dataset name: {shard_dir}"
            )
        num_views = manifest.get("num_views")
        if (
            isinstance(num_views, bool)
            or not isinstance(num_views, int)
            or num_views <= 0
        ):
            raise ValueError(
                f"packed shard manifest has invalid num_views={num_views!r}: "
                f"{shard_dir}"
            )
        previous = dataset_views.setdefault(dataset_name, num_views)
        if previous != num_views:
            raise ValueError(
                f"inconsistent num_views for dataset {dataset_name!r}: "
                f"{previous} != {num_views} ({shard_dir})"
            )
    if not dataset_views:
        raise ValueError("no non-empty shard manifests supplied")
    # AutoE2E is runtime-V-dynamic. The construction value only sizes defaults,
    # so use the largest validated rig when a multi-dataset run mixes view counts.
    return max(dataset_views.values())


def _training_source_revision(
    manifests: dict[str, dict],
    *,
    require_single: bool,
) -> str:
    """Return one packed revision when the dataset contract requires it."""
    revisions = [
        manifest.get("source_revision")
        for manifest in manifests.values()
    ]
    normalized = {
        str(revision)
        for revision in revisions
        if isinstance(revision, str) and revision
    }
    if require_single and (
        len(normalized) != 1
        or not all(
            isinstance(revision, str) and revision
            for revision in revisions
        )
    ):
        raise ValueError(
            "training requires one explicit packed source revision, got "
            f"{sorted(normalized)}"
        )
    return next(iter(normalized)) if len(normalized) == 1 else ""


def _validate_evaluation_shard_provenance(
    shard_dirs: List[str],
    *,
    dataset_name: str,
    source_revision: str,
    dataset_version: str,
    contract_digest: str,
) -> None:
    """Require evaluation shards to match the checkpoint's packed corpus."""
    import json
    import os

    from Platform.pipelines.training_checkpoint import stable_digest

    expected = {
        "dataset": dataset_name,
        "source_revision": source_revision,
        "dataset_version": dataset_version,
        "contract_digest": contract_digest,
    }
    if any(not isinstance(value, str) or not value for value in expected.values()):
        raise ValueError(
            "checkpoint validation contract has incomplete shard provenance"
        )

    for shard_dir in shard_dirs:
        manifest_path = os.path.join(shard_dir, "manifest.json")
        try:
            with open(manifest_path) as stream:
                manifest = json.load(stream)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"could not read evaluation shard manifest {manifest_path}"
            ) from error
        actual = {
            "dataset": manifest.get("dataset"),
            "source_revision": manifest.get("source_revision"),
            "dataset_version": manifest.get("dataset_version"),
            "contract_digest": stable_digest(manifest.get("contracts")),
        }
        if actual != expected:
            raise ValueError(
                "evaluation shard provenance differs from the checkpoint: "
                f"expected={expected} actual={actual}"
            )


def _checkpoint_bucket_name(sts_client=None) -> str:
    """Resolve checkpoint storage without embedding an AWS account ID."""
    configured = _os.environ.get("AUTO_E2E_CHECKPOINT_BUCKET", "").strip()
    if configured:
        return configured
    if sts_client is None:
        import boto3

        sts_client = boto3.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    cluster_name = _os.environ.get(
        "AUTO_E2E_CLUSTER_NAME", "auto-e2e-platform"
    )
    return f"{cluster_name}-checkpoints-{account_id}"


def _resumed_checkpoint_record(payload: dict, path: str) -> dict:
    """Rebuild the current immutable-checkpoint record from saved state."""
    import os

    from Platform.pipelines.training_checkpoint import sha256_file

    epoch = int(payload["epoch"])
    state = dict(payload["training_state"])
    history = list(state.get("metric_history", []))
    if not history or int(history[-1].get("epoch", -1)) != epoch:
        raise ValueError(
            "resume checkpoint metric history does not end at its saved epoch"
        )
    checkpoint_uri = str(state.get("current_checkpoint_uri", ""))
    if not checkpoint_uri.startswith("s3://"):
        raise ValueError(
            "resume checkpoint has no immutable current checkpoint URI"
        )
    record = {
        "epoch": epoch,
        "ade": float(history[-1]["val_ade"]),
        "fde": float(history[-1]["val_fde"]),
        "uri": checkpoint_uri,
        "sha256": sha256_file(path),
        "size": os.path.getsize(path),
    }
    metric_contract = history[-1].get("validation_metric_contract")
    if metric_contract is not None:
        if not isinstance(metric_contract, dict):
            raise ValueError(
                "resume checkpoint metric contract must be a mapping"
            )
        record["metric_contract"] = dict(metric_contract)
    selection = history[-1].get("checkpoint_selection")
    if selection is not None:
        if not isinstance(selection, dict):
            raise ValueError(
                "resume checkpoint selection state must be a mapping"
            )
        record["selection"] = dict(selection)
    return record


def _resume_terminal_state(
    *,
    completed_epoch: int,
    bad_epochs: int,
    requested_epochs: int,
    patience: int,
) -> tuple[bool, bool]:
    """Return ``(terminal, stopped_early)`` for a resumable checkpoint."""
    if bad_epochs < 0:
        raise ValueError("resume checkpoint has negative bad_epochs")
    if completed_epoch > requested_epochs:
        raise ValueError(
            f"resume checkpoint completed epoch {completed_epoch}, beyond "
            f"requested total epochs={requested_epochs}"
        )
    stopped_early = bad_epochs >= patience
    return completed_epoch == requested_epochs or stopped_early, stopped_early


def _resume_policy_transition(
    *,
    saved_config: dict,
    requested_config: dict,
) -> dict:
    """Validate and describe the supported continuation transition."""
    saved_sampling = saved_config.get("junction_sampling")
    requested_sampling = requested_config.get("junction_sampling")
    if not isinstance(saved_sampling, dict) or not isinstance(
        requested_sampling, dict
    ):
        raise ValueError(
            "resume policy transition requires junction sampling metadata"
        )
    sampling_changed = saved_sampling != requested_sampling
    if sampling_changed and not (
        saved_sampling.get("enabled") is False
        and requested_sampling.get("enabled") is True
    ):
        raise ValueError(
            "resume policy transition only supports unchanged sampling or "
            "disabled-to-enabled junction sampling"
        )
    saved_patience = int(saved_config.get("early_stopping_patience", 0))
    requested_patience = int(
        requested_config.get("early_stopping_patience", 0)
    )
    if requested_patience <= saved_patience:
        raise ValueError(
            "resume policy transition requires early-stopping patience "
            "to increase"
        )
    return {
        "policy_version": "dual_best_resume_transition_v1",
        "junction_sampling": {
            "from": saved_sampling,
            "to": requested_sampling,
            "changed": sampling_changed,
        },
        "early_stopping_patience": {
            "from": saved_patience,
            "to": requested_patience,
        },
        "bad_epochs_before_reset": None,
        "bad_epochs_after_reset": 0,
        "scheduler_state_action": (
            "reset_plateau_state_preserve_optimizer_lr"
        ),
        "best_checkpoint_scope": "full_history",
    }


def _restore_resume_optimization_state(
    optimizer,
    scheduler,
    resume_payload: dict,
    *,
    transition: dict | None,
) -> dict:
    """Restore optimizer state and optionally the plateau scheduler state."""
    optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
    plateau_state_restored = transition is None
    if plateau_state_restored:
        scheduler.load_state_dict(
            resume_payload["scheduler_state_dict"]
        )
    return {
        "optimizer_lr": [
            float(group["lr"]) for group in optimizer.param_groups
        ],
        "optimizer_lr_preserved": True,
        "plateau_state_restored": plateau_state_restored,
    }


def _transition_resume_selection_state(
    transition: dict | None,
    *,
    bad_epochs: int,
    best_checkpoint: dict | None,
    best_trajectory_checkpoint: dict | None,
) -> tuple[int, dict | None, dict | None]:
    """Reset patience while preserving both historical best checkpoints."""
    if transition is None:
        return bad_epochs, best_checkpoint, best_trajectory_checkpoint
    if bad_epochs < 0:
        raise ValueError("resume checkpoint has negative bad_epochs")
    if best_checkpoint is None or best_trajectory_checkpoint is None:
        raise ValueError(
            "resume policy transition requires both source best checkpoints"
        )
    selection = best_checkpoint.get("selection")
    trajectory_selection = best_trajectory_checkpoint.get("selection")
    transition["bad_epochs_before_reset"] = bad_epochs
    transition["best_before_resume"] = {
        "epoch": int(best_checkpoint["epoch"]),
        "uri": str(best_checkpoint["uri"]),
        "sha256": str(best_checkpoint["sha256"]),
        "selection_score": (
            float(selection["score"])
            if isinstance(selection, dict) and "score" in selection
            else None
        ),
    }
    transition["best_trajectory_before_resume"] = {
        "epoch": int(best_trajectory_checkpoint["epoch"]),
        "uri": str(best_trajectory_checkpoint["uri"]),
        "sha256": str(best_trajectory_checkpoint["sha256"]),
        "trajectory_utility": (
            float(trajectory_selection["components"]["trajectory"])
            if isinstance(trajectory_selection, dict)
            and isinstance(trajectory_selection.get("components"), dict)
            and "trajectory" in trajectory_selection["components"]
            else None
        ),
    }
    return 0, best_checkpoint, best_trajectory_checkpoint


def _best_trajectory_checkpoint_from_history(
    metric_history: list[dict],
    *,
    expected_policy_version: str,
    min_delta: float,
) -> dict:
    """Reconstruct the immutable trajectory-best record from metric history."""
    import math

    best = None
    best_utility = None
    for entry in metric_history:
        selection = entry.get("checkpoint_selection")
        if (
            isinstance(selection, dict)
            and selection.get("policy_version") != expected_policy_version
        ):
            raise ValueError(
                "metric history trajectory selection policy differs from "
                "the requested selector"
            )
        components = (
            selection.get("components")
            if isinstance(selection, dict)
            else None
        )
        if not isinstance(components, dict) or "trajectory" not in components:
            continue
        utility = float(components["trajectory"])
        if not math.isfinite(utility):
            raise ValueError("metric history has non-finite trajectory utility")
        if best is not None and utility <= float(best_utility) + min_delta:
            continue
        uri = str(entry.get("checkpoint_uri", ""))
        sha256 = entry.get("checkpoint_sha256") or ""
        if (
            not uri.startswith("s3://")
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(
                "metric history trajectory best lacks immutable checkpoint "
                "identity"
            )
        best = {
            "epoch": int(entry["epoch"]),
            "ade": float(entry["val_ade"]),
            "fde": float(entry["val_fde"]),
            "uri": uri,
            "sha256": sha256,
            "selection": dict(selection),
            "metric_contract": dict(
                entry["validation_metric_contract"]
            ),
        }
        best_utility = utility
    if best is None:
        raise ValueError("metric history has no trajectory checkpoint")
    return best


def _dual_best_improvements(
    selection: dict,
    *,
    best_selection: dict | None,
    best_trajectory_selection: dict | None,
    min_delta: float,
) -> tuple[bool, bool]:
    """Return independent composite-score and trajectory improvements."""
    from evaluation.checkpoint_selection import score_is_better

    score = float(selection["score"])
    trajectory = float(selection["components"]["trajectory"])
    score_improved = (
        best_selection is None
        or score_is_better(
            score,
            float(best_selection["score"]),
            min_delta=min_delta,
        )
    )
    trajectory_improved = (
        best_trajectory_selection is None
        or score_is_better(
            trajectory,
            float(
                best_trajectory_selection["components"]["trajectory"]
            ),
            min_delta=min_delta,
        )
    )
    return score_improved, trajectory_improved


def _collated_metadata_value(
    metadata,
    key: str,
    sample_index: int,
    default=None,
):
    """Read one scalar/string from PyTorch's recursively collated metadata."""
    if not isinstance(metadata, dict) or key not in metadata:
        return default
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        return value[sample_index]
    if hasattr(value, "ndim") and getattr(value, "ndim", 0) > 0:
        value = value[sample_index]
    if hasattr(value, "item"):
        return value.item()
    return value


def _accumulate_rollout_epoch_terms(
    term_sums: dict[str, float],
    term_weights: dict[str, int],
    terms: dict,
    *,
    batch_sample_count: int,
) -> None:
    """Accumulate diagnostics using their actual eligible sample counts."""
    active_count_keys = {
        "map": "map_sample_count",
        "route": "route_sample_count",
        "drivable": "drivable_sample_count",
    }
    for name in term_sums:
        weight = (
            int(terms[active_count_keys[name]].item())
            if name in active_count_keys
            else batch_sample_count
        )
        term_sums[name] += float(terms[name].item()) * weight
        term_weights[name] += weight


def _stable_evaluation_noise(sample_uids, trajectory_width, dtype):
    """Create a batch-order-independent planner prior for paired evaluation."""
    import hashlib
    import torch

    noise = []
    for sample_uid in sample_uids:
        digest = hashlib.sha256(
            f"auto-e2e-open-loop-noise-v1:{sample_uid}".encode("utf-8")
        ).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int.from_bytes(digest[:8], "big") % (2**63 - 1)
        )
        noise.append(
            torch.randn(
                trajectory_width,
                dtype=dtype,
                generator=generator,
            )
        )
    return torch.stack(noise)


def _validate_selector_preflight_identity(
    validation: dict,
    *,
    expected_sample_count: int | None,
    expected_sample_uid_digest: str | None,
) -> None:
    """Reject availability evidence from outside the frozen validation set."""
    actual_count = validation.get("sample_count")
    actual_digest = validation.get("sample_uid_digest")
    if (
        expected_sample_count is None
        or not expected_sample_uid_digest
        or actual_count != expected_sample_count
        or actual_digest != expected_sample_uid_digest
    ):
        raise ValueError(
            "selector preflight validation identity differs from frozen "
            f"split: expected_count={expected_sample_count} "
            f"actual_count={actual_count} "
            f"expected_digest={expected_sample_uid_digest} "
            f"actual_digest={actual_digest}"
        )


def _evaluate_open_loop(
    model,
    loader,
    device,
    training_policy=None,
    navigation_geometry=None,
    route_swap_counterfactual: bool = False,
    include_navigation_records: bool = False,
    include_rollout_selector_records: bool = False,
    report_intervention: bool = False,
) -> dict:
    """Evaluate one fixed loader and return finite ADE/FDE plus its UID digest."""
    import hashlib
    import numpy as np
    import torch

    from evaluation.metrics import integrate_trajectory
    from training.dataset_policy import (
        AUTO_E2E_TIMESTEPS,
        adapt_egomotion_history,
    )

    was_training = model.training
    all_ade: list[float] = []
    all_fde: list[float] = []
    evaluation_steps = 30
    horizon_steps = {
        "1s": 10,
        "2s": 20,
        "3s": evaluation_steps,
    }
    horizon_ade: dict[str, list[float]] = {
        label: [] for label in horizon_steps
    }
    horizon_fde: dict[str, list[float]] = {
        label: [] for label in horizon_steps
    }
    sample_uids: list[str] = []
    navigation_records: list[dict] = []
    route_swap_records: list[dict] = []
    rollout_selector_records: list[dict] = []
    route_cache: dict[str, dict] = {}
    _reactive = getattr(model, "Reactive_E2E", None)
    report_delta = report_intervention and (
        getattr(_reactive, "ReasoningHead", None) is not None
    )
    intervention_deltas: list[float] = []
    model.eval()
    try:
        with torch.no_grad():
            for batch, projection, geometry_type in loader:
                if projection is not None:
                    projection = projection.to(device)
                if hasattr(model, "reset_visual_history"):
                    model.reset_visual_history()

                visual = batch["visual_tiles"].to(device)
                raw_ego_hist = batch["egomotion_history"]
                ego_hist = raw_ego_hist.to(device)
                if training_policy is not None:
                    ego_hist = adapt_egomotion_history(
                        ego_hist,
                        training_policy,
                    )
                vis_hist = batch["visual_history"].to(device)
                target = batch["trajectory_target"]
                raw_map_context = batch["map_context"]
                raw_route_mask = batch["route_mask"]
                map_context = raw_map_context.to(device)
                route_mask = raw_route_mask.to(device)
                map_valid = batch["map_valid"].to(device)
                route_valid = batch["route_valid"].to(device)
                batch_uids = batch.get("sample_uid", [])
                if isinstance(batch_uids, str):
                    batch_uids = [batch_uids]
                batch_uids = [str(uid) for uid in batch_uids]
                if len(batch_uids) != visual.shape[0]:
                    raise ValueError(
                        "evaluation batch lost sample identities: "
                        f"samples={visual.shape[0]} "
                        f"sample_uids={len(batch_uids)}"
                    )
                if any(not uid for uid in batch_uids):
                    raise ValueError(
                        "evaluation batch contains an empty sample UID"
                    )
                initial_noise = _stable_evaluation_noise(
                    batch_uids,
                    int(target.shape[-1]),
                    visual.dtype,
                ).to(device)
                navigation_metadata = batch.get(
                    "navigation_metadata",
                    {},
                )
                history_frames = batch.get("history_frames")
                future_frames = batch.get("future_frames")
                if history_frames is not None:
                    history_frames = history_frames.to(device)
                if future_frames is not None:
                    future_frames = future_frames.to(device)

                pred = model(
                    visual,
                    map_context,
                    vis_hist,
                    ego_hist,
                    route_mask=route_mask,
                    map_valid=map_valid,
                    route_valid=route_valid,
                    projection=projection,
                    geometry_type=geometry_type,
                    history_frames=history_frames,
                    future_frames=future_frames,
                    mode="infer",
                    initial_noise=initial_noise,
                )
                pred_np = pred.cpu().numpy()
                target_np = target.numpy()
                if len(batch_uids) != pred_np.shape[0]:
                    raise ValueError(
                        "evaluation batch lost sample identities: "
                        f"samples={pred_np.shape[0]} "
                        f"sample_uids={len(batch_uids)}"
                    )
                sample_uids.extend(str(uid) for uid in batch_uids)

                if include_rollout_selector_records:
                    from evaluation.kitscenes_benchmark import (
                        wgs84_trajectory_to_ego_xy,
                    )
                    from evaluation.rollout_validation import (
                        build_rollout_validation_records,
                    )

                    batch_group_uids = batch.get(
                        "split_group_uid",
                        [],
                    )
                    if isinstance(batch_group_uids, str):
                        batch_group_uids = [batch_group_uids]
                    batch_group_uids = [
                        str(uid) for uid in batch_group_uids
                    ]
                    if len(batch_group_uids) != pred_np.shape[0]:
                        raise ValueError(
                            "selector validation batch lost split group "
                            "identities"
                        )
                    pose_current = batch.get("pose_current")
                    gps_future = batch.get("gps_future")
                    route_supervision = batch.get("route_supervision")
                    if (
                        pose_current is None
                        or gps_future is None
                        or route_supervision is None
                    ):
                        raise ValueError(
                            "rollout selector requires packed pose, GPS, "
                            "and route supervision"
                        )
                    logged_xy = wgs84_trajectory_to_ego_xy(
                        gps_future.numpy(),
                        pose_current.numpy(),
                    )
                    route_intersections = [
                        bool(
                            _collated_metadata_value(
                                navigation_metadata,
                                "route_intersection",
                                sample_index,
                                False,
                            )
                        )
                        for sample_index in range(pred_np.shape[0])
                    ]
                    initial_speeds = raw_ego_hist.reshape(
                        raw_ego_hist.shape[0],
                        AUTO_E2E_TIMESTEPS,
                        -1,
                    )[:, -1, 0]
                    rollout_selector_records.extend(
                        build_rollout_validation_records(
                            pred,
                            target,
                            initial_speeds,
                            logged_xy,
                            route_supervision,
                            batch["map_valid"],
                            batch["route_valid"],
                            batch_uids,
                            batch_group_uids,
                            route_mask=route_mask,
                            route_intersections=route_intersections,
                        )
                    )

                swapped_pred_np = None
                swapped_indices: set[int] = set()
                swapped_route_entries: dict[int, dict] = {}
                if (
                    navigation_geometry is not None
                    and route_swap_counterfactual
                ):
                    swapped_route_mask = raw_route_mask.clone()
                    swapped_route_valid = batch["route_valid"].clone()
                    for sample_index in range(pred_np.shape[0]):
                        route_id = str(
                            _collated_metadata_value(
                                navigation_metadata,
                                "route_id",
                                sample_index,
                                "",
                            )
                        )
                        selected_maneuver = str(
                            _collated_metadata_value(
                                navigation_metadata,
                                "route_maneuver",
                                sample_index,
                                "unknown",
                            )
                        )
                        candidates = sorted(
                            (
                                candidate_id
                                for candidate_id in route_cache
                                if candidate_id != route_id
                            ),
                            key=lambda candidate_id: (
                                route_cache[candidate_id]["maneuver"]
                                == selected_maneuver,
                                route_cache[candidate_id]["maneuver"]
                                not in ("left", "right", "straight"),
                                candidate_id,
                            ),
                        )
                        if (
                            bool(route_valid[sample_index].item())
                            and route_id
                            and candidates
                        ):
                            candidate = route_cache[candidates[0]]
                            swapped_route_mask[sample_index] = (
                                candidate["mask"]
                            )
                            swapped_route_valid[sample_index] = True
                            swapped_indices.add(sample_index)
                            swapped_route_entries[sample_index] = candidate
                    if swapped_indices:
                        if hasattr(model, "reset_visual_history"):
                            model.reset_visual_history()
                        swapped_pred = model(
                            visual,
                            map_context,
                            vis_hist,
                            ego_hist,
                            route_mask=swapped_route_mask.to(device),
                            map_valid=map_valid,
                            route_valid=swapped_route_valid.to(device),
                            projection=projection,
                            geometry_type=geometry_type,
                            history_frames=history_frames,
                            future_frames=future_frames,
                            mode="infer",
                            initial_noise=initial_noise,
                        )
                        swapped_pred_np = swapped_pred.cpu().numpy()

                for sample_index in range(pred_np.shape[0]):
                    pred_signals = pred_np[sample_index].reshape(
                        AUTO_E2E_TIMESTEPS, 2
                    )
                    target_signals = target_np[sample_index].reshape(
                        AUTO_E2E_TIMESTEPS, 2
                    )
                    ego_np = raw_ego_hist[
                        sample_index
                    ].numpy()
                    v0 = float(ego_np[-4])
                    pred_traj = integrate_trajectory(
                        pred_signals[:, 0], pred_signals[:, 1], v0
                    )
                    target_traj = integrate_trajectory(
                        target_signals[:, 0], target_signals[:, 1], v0
                    )
                    errors = np.linalg.norm(
                        pred_traj - target_traj, axis=1
                    )
                    evaluation_errors = errors[:evaluation_steps]
                    all_ade.append(float(evaluation_errors.mean()))
                    all_fde.append(float(evaluation_errors[-1]))
                    for label, step_count in horizon_steps.items():
                        if step_count > len(errors):
                            raise ValueError(
                                f"evaluation horizon {label} exceeds "
                                f"trajectory length {len(errors)}"
                            )
                        horizon_errors = errors[:step_count]
                        horizon_ade[label].append(
                            float(horizon_errors.mean())
                        )
                        horizon_fde[label].append(
                            float(horizon_errors[-1])
                        )
                    if report_delta and len(intervention_deltas) < 50:
                        from evaluation.faithfulness import (
                            reasoning_intervention_delta,
                        )
                        intervention_deltas.append(
                            reasoning_intervention_delta(
                                model, visual, map_context, vis_hist,
                                ego_hist, projection=projection,
                                geometry_type=geometry_type,
                                route_mask=route_mask, map_valid=map_valid,
                                route_valid=route_valid,
                                history_frames=history_frames,
                                future_frames=future_frames,
                                initial_noise=initial_noise,
                            )["trajectory_l2"]
                        )
                    if navigation_geometry is not None:
                        from evaluation.navigation_metrics import (
                            ROUTE_QUALITY_FIELDS,
                            navigation_sample_metrics,
                            route_swap_sample_metrics,
                        )

                        metadata = {
                            key: _collated_metadata_value(
                                navigation_metadata,
                                key,
                                sample_index,
                                default,
                            )
                            for key, default in (
                                ("route_id", ""),
                                ("route_maneuver", "unknown"),
                                ("route_intersection", False),
                                ("destination_visible", False),
                            )
                        }
                        for key in ROUTE_QUALITY_FIELDS:
                            metadata[key] = _collated_metadata_value(
                                navigation_metadata,
                                key,
                                sample_index,
                                float("nan"),
                            )
                        navigation_record = navigation_sample_metrics(
                            pred_traj[:evaluation_steps],
                            target_traj[:evaluation_steps],
                            raw_route_mask[sample_index].numpy(),
                            raw_map_context[sample_index].numpy(),
                            route_valid=bool(
                                route_valid[sample_index].item()
                            ),
                            metadata=metadata,
                            geometry=navigation_geometry,
                        )
                        navigation_record["sample_uid"] = str(
                            batch_uids[sample_index]
                        )
                        navigation_records.append(navigation_record)
                        if (
                            swapped_pred_np is not None
                            and sample_index in swapped_indices
                        ):
                            swapped_signals = swapped_pred_np[
                                sample_index
                            ].reshape(AUTO_E2E_TIMESTEPS, 2)
                            swapped_traj = integrate_trajectory(
                                swapped_signals[:, 0],
                                swapped_signals[:, 1],
                                v0,
                            )
                            swapped_entry = swapped_route_entries[
                                sample_index
                            ]
                            route_swap_records.append(
                                route_swap_sample_metrics(
                                    pred_traj[:evaluation_steps],
                                    swapped_traj[:evaluation_steps],
                                    raw_route_mask[
                                        sample_index
                                    ].numpy(),
                                    swapped_route_mask=swapped_entry[
                                        "mask"
                                    ].numpy(),
                                    selected_maneuver=str(
                                        metadata["route_maneuver"]
                                    ),
                                    swapped_maneuver=str(
                                        swapped_entry["maneuver"]
                                    ),
                                    geometry=navigation_geometry,
                                )
                            )

                if navigation_geometry is not None:
                    for sample_index in range(pred_np.shape[0]):
                        if not bool(route_valid[sample_index].item()):
                            continue
                        route_id = str(
                            _collated_metadata_value(
                                navigation_metadata,
                                "route_id",
                                sample_index,
                                "",
                            )
                        )
                        if route_id:
                            route_cache[route_id] = {
                                "mask": (
                                    raw_route_mask[sample_index]
                                    .detach()
                                    .cpu()
                                    .clone()
                                ),
                                "maneuver": str(
                                    _collated_metadata_value(
                                        navigation_metadata,
                                        "route_maneuver",
                                        sample_index,
                                        "unknown",
                                    )
                                ),
                            }
                    while len(route_cache) > 8:
                        route_cache.pop(next(iter(route_cache)))
    finally:
        model.train(was_training)
        if hasattr(model, "reset_visual_history"):
            model.reset_visual_history()

    if not all_ade or len(sample_uids) != len(all_ade):
        raise ValueError(
            "validation produced no samples or lost sample identities: "
            f"metrics={len(all_ade)} sample_uids={len(sample_uids)}"
        )
    if any(not uid for uid in sample_uids):
        raise ValueError("validation contains an empty sample UID")
    if len(set(sample_uids)) != len(sample_uids):
        raise ValueError("validation contains duplicate sample UIDs")
    ade = float(np.mean(all_ade))
    fde = float(np.mean(all_fde))
    if not np.isfinite(ade) or not np.isfinite(fde):
        raise ValueError(f"non-finite validation metrics: ADE={ade}, FDE={fde}")
    uid_digest = hashlib.sha256(
        "\n".join(sorted(sample_uids)).encode("utf-8")
    ).hexdigest()
    result = {
        "ade": ade,
        "fde": fde,
        "evaluation_steps": evaluation_steps,
        "prediction_steps": AUTO_E2E_TIMESTEPS,
        "sample_count": len(all_ade),
        "sample_uid_digest": uid_digest,
        "metric_contract": {
            "version": "control_rollout_validation_v2",
            "horizon_seconds": 3.0,
            "horizon_steps": evaluation_steps,
            "target_source": "target_control_rollout",
            "aggregation": "sample_mean",
        },
        "horizons": {
            label: {
                "steps": horizon_steps[label],
                "ade": float(np.mean(horizon_ade[label])),
                "fde": float(np.mean(horizon_fde[label])),
            }
            for label in horizon_steps
        },
    }
    if intervention_deltas:
        result["reasoning_intervention_delta"] = float(
            np.mean(intervention_deltas)
        )
    if navigation_geometry is not None:
        from evaluation.navigation_metrics import (
            summarize_navigation_metrics,
        )

        if len(navigation_records) != len(all_ade):
            raise ValueError(
                "navigation metric coverage differs from displacement metrics"
            )
        result["navigation"] = summarize_navigation_metrics(
            navigation_records,
            route_swap_records=route_swap_records,
        )
        if include_navigation_records:
            result["navigation_records"] = navigation_records
    if include_rollout_selector_records:
        if len(rollout_selector_records) != len(all_ade):
            raise ValueError(
                "rollout selector coverage differs from displacement metrics"
            )
        result["rollout_selector_records"] = (
            rollout_selector_records
        )
        from evaluation.checkpoint_selection import (
            aggregate_validation_records,
        )
        from evaluation.rollout_validation import (
            ROLLOUT_VALIDATION_VERSION,
        )

        aggregates = aggregate_validation_records(
            rollout_selector_records
        )
        result["ade"] = float(
            aggregates["metrics"]["ade_3s_m"]["scene_balanced"]
        )
        result["fde"] = float(
            aggregates["metrics"]["fde_3s_m"]["scene_balanced"]
        )
        result["metric_contract"] = {
            "version": ROLLOUT_VALIDATION_VERSION,
            "horizon_seconds": 3.0,
            "horizon_steps": evaluation_steps,
            "target_source": "logged_xy",
            "aggregation": "scene_balanced",
        }
    return result


def _register_checkpoint_version(
    client,
    *,
    run_id: str,
    roles: List[str],
    epoch: int,
    checkpoint_uri: str,
    checkpoint_sha256: str,
    ade: float,
    fde: float,
    metric_contract: dict,
    selection: dict | None = None,
) -> str:
    """Register one immutable checkpoint idempotently for an MLflow run."""
    expected_metric_contract = {
        "horizon_seconds": 3.0,
        "horizon_steps": 30,
    }
    mismatched_metric_contract = {
        key: {
            "expected": value,
            "actual": metric_contract.get(key),
        }
        for key, value in expected_metric_contract.items()
        if metric_contract.get(key) != value
    }
    if mismatched_metric_contract:
        raise ValueError(
            "registry checkpoint metrics are not canonical: "
            f"{mismatched_metric_contract}"
        )
    model_name = "auto-e2e-driving-policy"
    normalized_roles = sorted(set(roles))
    if (
        not normalized_roles
        or not set(normalized_roles)
        <= {"best", "best_trajectory", "final"}
    ):
        raise ValueError(f"invalid checkpoint roles: {roles}")

    try:
        client.get_registered_model(model_name)
    except Exception:
        try:
            client.create_registered_model(model_name)
        except Exception:
            # A concurrent retry may have created it after the first read.
            client.get_registered_model(model_name)

    for existing in client.search_model_versions(f"name='{model_name}'"):
        if (
            existing.run_id == run_id
            and str(existing.source or "") == checkpoint_uri
        ):
            version = str(existing.version)
            break
    else:
        registered = client.create_model_version(
            name=model_name,
            source=checkpoint_uri,
            run_id=run_id,
        )
        version = str(registered.version)

    tags = {
        "checkpoint_role": ",".join(normalized_roles),
        "checkpoint_epoch": str(epoch),
        "checkpoint_s3_uri": checkpoint_uri,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_ade_3s_m": str(ade),
        "validation_fde_3s_m": str(fde),
        "validation_ade": str(ade),
        "validation_fde": str(fde),
        "validation_metric_version": str(metric_contract["version"]),
        "validation_metric_horizon_seconds": str(
            metric_contract["horizon_seconds"]
        ),
        "validation_metric_horizon_steps": str(
            metric_contract["horizon_steps"]
        ),
        "validation_metric_target_source": str(
            metric_contract["target_source"]
        ),
        "validation_metric_aggregation": str(
            metric_contract["aggregation"]
        ),
    }
    if selection is not None:
        tags.update({
            "checkpoint_selector_policy": str(
                selection["policy_version"]
            ),
            "checkpoint_composite_score": str(selection["score"]),
        })
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, value)
    return version


class _ProjectionDeviceCache:
    """Cache device projections only while their source calibration is alive."""

    def __init__(self, device):
        import weakref

        self._device = device
        self._values = weakref.WeakKeyDictionary()

    def get(self, projection):
        if projection is None:
            return None
        try:
            return self._values[projection]
        except KeyError:
            device_projection = projection.to(self._device)
            self._values[projection] = device_projection
            return device_projection

    def __len__(self):
        return len(self._values)


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
    expected_reasoning_label_count: Optional[int] = None,
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

    if expected_reasoning_label_count is not None:
        if expected_reasoning_label_count < 0:
            raise ValueError(
                "expected_reasoning_label_count must be non-negative"
            )
        if reasoning_labels is None:
            raise ValueError(
                "expected_reasoning_label_count requires reasoning_labels"
            )

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
                include_navigation=False,
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
        if (
            expected_reasoning_label_count is not None
            and len(records_files) != 1
        ):
            raise ValueError(
                "strict reasoning JOIN requires exactly one records.jsonl, "
                f"found {len(records_files)} in {labels_dir}"
            )
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
        if (
            expected_reasoning_label_count is not None
            and len(labels_by_id) != expected_reasoning_label_count
        ):
            raise ValueError(
                "reasoning label artifact count differs from the recovery "
                f"manifest: expected={expected_reasoning_label_count} "
                f"loaded={len(labels_by_id)}"
            )

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
    joined_reasoning_ids: set[str] = set()
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
    navigation_artifact_summary = None

    if (
        dataset != Dataset.NVIDIA_PHYSICAL_AI
        and idx_list
        and (world_model or dataset == Dataset.KITSCENES)
    ):
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
        print(f"Packing {len(idx_list)} samples, parent-assembly mode "
              f"(row-level camera workers, world_model={world_model})...")
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
            if world_model:
                # window_rows raises only if the margin invariant is broken.
                for row_t in ds.window_rows(si):
                    all_rows.add(row_t)

        del ds  # free before spawning workers

        # row_map contains camera JPEGs and an optional legacy map JPEG. KITScenes
        # navigation is generated once in the parent assembly below, never in a
        # camera worker.
        row_map: dict = {}
        row_workers = _row_decode_worker_count(dataset, len(all_rows))
        current_rows = set(sample_cur_rows.values())
        decode_tasks = [
            (group_id, frame_index, (group_id, frame_index) in current_rows)
            for group_id, frame_index in sorted(all_rows)
        ]
        with ProcessPoolExecutor(max_workers=row_workers, mp_context=ctx,
                                 initializer=parallel_pack.init_row_worker,
                                 initargs=row_init) as rpool:
            for row_key, cam_jpegs, legacy_map in rpool.map(
                    parallel_pack.decode_row, decode_tasks):
                row_map[row_key] = (cam_jpegs, legacy_map)
                for fid, blob in cam_jpegs.items():
                    _write_pool(fid, blob)
                if dataset != Dataset.KITSCENES and legacy_map is not None:
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
                include_navigation=True,
                source_revision=source_revision,
            )
        else:
            from data_parsing.l2d import L2DDataset
            ds_asm = L2DDataset(
                repo_id=dataset.value,
                episodes=ep_list,
                include_world_model_windows=False,
                root=raw_path,
            )

        if dataset == Dataset.KITSCENES:
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

            # cam_*.jpg = current frame (offset 0). The current-frame bytes are in
            # row_map[(ep_idx, cur_fi)][0] — the same jpegs already written to pool.
            cur_key = sample_cur_rows.get(si)
            if cur_key and cur_key in row_map:
                cur_cams, legacy_map = row_map[cur_key]
                # cam_cams is {frame_id: bytes}; sort by cam index embedded in fid.
                for fid, blob in sorted(cur_cams.items(),
                                        key=lambda kv: int(kv[0].rsplit("-c", 1)[-1])):
                    cam_i = int(fid.rsplit("-c", 1)[-1])
                    members[f"cam_{cam_i}.jpg"] = blob
                if dataset == Dataset.KITSCENES:
                    members.update(
                        ds_asm.navigation_members_for_row(*cur_key)
                    )
                    has_map = True
                elif legacy_map is not None:
                    members["map.jpg"] = legacy_map

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
                    joined_reasoning_ids.add(uid)
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
                has_map = has_map or (
                    "map.jpg" in members
                    or "map_semantic.npz" in members
                )
                has_wm = has_wm or ("window_index.json" in members)
                if _record_to_json is not None:
                    record = labels_by_id.get(sample_key)
                    if record is not None:
                        _add_member(sample_key, "reasoning.json",
                                    json.dumps(_record_to_json(record)).encode())
                        reasoning_label_count += 1
                        joined_reasoning_ids.add(sample_key)
                sample_count += 1

    if current_tar:
        current_tar.close()

    if expected_reasoning_label_count is not None:
        unjoined_ids = set(labels_by_id) - joined_reasoning_ids
        if unjoined_ids:
            raise ValueError(
                "reasoning labels did not join a packed sample: "
                f"{sorted(unjoined_ids)[:3]}"
            )
        if reasoning_label_count != expected_reasoning_label_count:
            raise ValueError(
                "packed reasoning JOIN count differs from the recovery "
                f"manifest: expected={expected_reasoning_label_count} "
                f"joined={reasoning_label_count}"
            )
        if expected_reasoning_label_count == 0 and sample_count > 0:
            raise ValueError(
                "a zero-label recovery partition produced "
                f"{sample_count} samples; refusing unsupervised packing"
            )

    from data_processing.contract_versions import contract_versions
    from data_processing.geospatial import (
        EPISODE_PATH_SCHEMA_VERSION,
        GPS_SCHEMA_VERSION,
        POSE_SCHEMA_VERSION,
    )
    from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY
    from navigation.supervision import (
        ROUTE_SUPERVISION_ARTIFACT_VERSION,
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
                "has_navigation": (
                    bool(sample_count)
                    and navigation_artifact_summary is not None
                ),
                "has_route_supervision": (
                    bool(sample_count)
                    and navigation_artifact_summary is not None
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
                    DEFAULT_NAVIGATION_GEOMETRY.contract()
                    if navigation_artifact_summary is not None
                    else None
                ),
                "map_context_channels": (
                    14 if navigation_artifact_summary is not None else 3
                ),
                "route_channels": 2,
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
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
)
def train_il(
    shards: List[FlyteDirectory],
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    # Defaults to the previously hardcoded value, so a run that does not pass it
    # is unchanged.
    map_fusion_mode: MapFusion = MapFusion.RESIDUAL,
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
    training_seed: int = 149,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    # AMP off by default: with fp16 autocast the GradScaler detected inf/nan grads
    # every step (fp16 overflow somewhere in the BEV projection / Bezier basis /
    # backbone path) and skipped optimizer.step() FOREVER — weights never updated,
    # so the trajectory loss sat perfectly flat (~2.95) while fp32 learns in one
    # step (verified: control_head grad norm ~6.6, loss 6.30->5.00). Keep fp32
    # until the specific overflow op is isolated and kept in fp32 explicitly.
    amp: bool = False,
    enable_route_conditioning: bool = True,
    training_objective_version: str = (
        BASELINE_TRAINING_OBJECTIVE_VERSION
    ),
    enable_junction_sampling: bool = False,
    enable_route_consistency: bool = False,
    route_consistency_weight: float = 0.10,
    navigation_quality_audit: Optional[FlyteFile] = None,
    reconstruction_audit: Optional[FlyteFile] = None,
    reconstruction_audit_decision: str = "",
    reconstruction_audit_rationale: str = "",
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
    # Held-out split: train on the (1 - val_fraction) majority of scene groups, so the
    # separate eval task can score the disjoint val split and measure
    # GENERALIZATION rather than training-set memorization (which structurally
    # favours the lower-capacity imitation model). 0.0 = train on everything
    # (legacy in-sample behaviour). KITScenes uses a frozen audited scene
    # manifest; L2D/NVIDIA retain stable split_group_uid hash buckets.
    val_fraction: float = 0.1,
    # Full KITScenes runs must match the frozen corpus manifest. Smoke runs may
    # explicitly select "subset", which pins the same corpus provenance while
    # deriving an exact deterministic holdout from the packed scene inventory.
    validation_scope: str = "full",
    # Parallel JPEG decode (#121 P0). num_workers=0 decodes every sample (~55
    # JPEGs/sample with WM windows) serially on the training process, stalling the
    # GPU — the dominant per-epoch cost at scale. >0 spreads decode across worker
    # processes (sharded over shards by split_by_worker), overlapping it with the
    # GPU step. Effective parallelism is capped by shard count, so scale needs more
    # (smaller) shards too.
    num_workers: int = 0,
    resume_from: Optional[FlyteFile] = None,
    early_stopping_patience: int = 5,
    allow_resume_policy_transition: bool = False,
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

    KITScenes requires the hash-bound output of
    ``audit_kitscenes_navigation_quality``. The frozen validation inventory is
    checked against every packed scene, while optimizer batches include only
    policy-accepted scene partitions.
    """
    import os
    import hashlib
    import json
    import random
    import time
    import torch
    import numpy as np
    import mlflow
    import boto3
    from flytekit import current_context

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be between 0 and 1, got {val_fraction}"
        )
    if not 3 <= early_stopping_patience <= 10:
        raise ValueError(
            "early_stopping_patience must be between 3 and 10"
        )
    if allow_resume_policy_transition and resume_from is None:
        raise ValueError(
            "resume policy transition requires a resume checkpoint"
        )
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if training_objective_version not in {
        BASELINE_TRAINING_OBJECTIVE_VERSION,
        KITSCENES_NAVIGATION_OBJECTIVE_VERSION,
        ROLLOUT_ALIGNED_CONTROL_OBJECTIVE_VERSION,
        ROLLOUT_ALIGNED_OBJECTIVE_VERSION,
    }:
        raise ValueError(
            "unsupported training_objective_version "
            f"{training_objective_version!r}"
        )
    objective_v1 = (
        training_objective_version
        == KITSCENES_NAVIGATION_OBJECTIVE_VERSION
    )
    objective_v2 = (
        training_objective_version == ROLLOUT_ALIGNED_OBJECTIVE_VERSION
    )
    objective_v2_control = (
        training_objective_version
        == ROLLOUT_ALIGNED_CONTROL_OBJECTIVE_VERSION
    )
    selector_enabled = objective_v2 or objective_v2_control
    if (
        objective_v1 or objective_v2 or objective_v2_control
    ) and dataset != Dataset.KITSCENES:
        raise ValueError(
            "navigation planner objectives are KITScenes-only"
        )
    if enable_junction_sampling and not (
        objective_v1 or objective_v2 or objective_v2_control
    ):
        raise ValueError(
            "navigation sampling requires a KITScenes navigation objective"
        )
    if enable_route_consistency and not objective_v1:
        raise ValueError(
            "route consistency requires kitscenes_navigation_objective_v1"
        )
    if enable_route_consistency and not enable_route_conditioning:
        raise ValueError(
            "route consistency requires Reactive route conditioning"
        )
    if enable_route_consistency and route_consistency_weight <= 0.0:
        raise ValueError(
            "enabled route consistency requires a positive weight"
        )
    if objective_v2 and enable_route_consistency:
        raise ValueError(
            "rollout_aligned_planner_v1 replaces legacy route consistency"
        )
    if objective_v2 and not enable_route_conditioning:
        raise ValueError(
            "rollout_aligned_planner_v1 requires Reactive route conditioning"
        )
    if objective_v2 and not enable_world_model:
        raise ValueError(
            "rollout-aligned matched experiments require the World Model"
        )
    if objective_v2_control and (
        not enable_route_conditioning or not enable_world_model
    ):
        raise ValueError(
            "rollout-aligned control requires Reactive route conditioning "
            "and the World Model"
        )
    if not 0 <= training_seed <= 2**32 - 1:
        raise ValueError(
            "training_seed must be between 0 and 2**32 - 1"
        )

    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
    from training.dataset_policy import (
        AUTO_E2E_TIMESTEPS,
        adapt_egomotion_history,
        group_uid_digest,
        training_policy_for_dataset,
        validation_group_uids as select_validation_group_uids,
        validation_sample_identity,
    )
    from data_parsing.pre_extracted import (
        NavigationRepeatPolicy,
        discover_split_inventory,
        discover_navigation_exposure,
        make_multi_dataset_loader,
    )
    # _loader_download_dir is a module-level helper in THIS file, not in
    # pre_extracted — call it directly (importing it from there is an ImportError).

    ctx = current_context()
    train_execution_id = (
        ctx.execution_id.name if ctx.execution_id else "local"
    )
    bb, fm = backbone.value, FUSION_LABEL
    mfm = map_fusion_mode.value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_policy = training_policy_for_dataset(
        dataset.value,
        validation_scope=validation_scope,
    )
    uses_exact_validation = (
        training_policy.validation_strategy != "hash_buckets"
    )

    print(f"Training: backbone={bb} fusion={fm} epochs={epochs} bs={batch_size} device={device}")
    print(f"Map BEV fusion: {mfm}")

    # MERGED DataLoader over ALL provided shard dirs. Each dataset keeps its own
    # geometry/num_views; batches are same-dataset (uniform), interleaved across
    # datasets, each carrying its projection — so L2D (6cam pseudo) and NVIDIA
    # (7cam f-theta) train together. The model is runtime-V-dynamic (projection
    # ABI, #77), so a single model consumes both. num_views only sizes defaults.
    all_shard_dirs = []
    shard_artifact_uris = {}
    for shard in shards:
        artifact_uri = str(
            getattr(shard, "remote_source", "")
            or getattr(shard, "path", "")
            or shard
        )
        shard_dir = _loader_download_dir(shard)
        all_shard_dirs.append(shard_dir)
        shard_artifact_uris[shard_dir] = artifact_uri
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
    packed_source_revision = _training_source_revision(
        manifests,
        require_single=uses_exact_validation,
    )
    all_shard_dirs.sort(
        key=lambda shard_dir: (
            str(manifests[shard_dir].get("partition_id", "")),
            shard_artifact_uris[shard_dir],
        )
    )
    shard_dirs.sort(
        key=lambda shard_dir: (
            str(manifests[shard_dir].get("partition_id", "")),
            shard_artifact_uris[shard_dir],
        )
    )
    num_views = _training_num_views_from_manifests(manifests, shard_dirs)
    map_context_channel_counts = {
        int(manifests[path].get("map_context_channels", 3))
        for path in shard_dirs
    }
    route_channel_counts = {
        int(manifests[path].get("route_channels", 2))
        for path in shard_dirs
    }
    if len(map_context_channel_counts) != 1:
        raise ValueError(
            "one training run cannot mix map-context channel contracts: "
            f"{sorted(map_context_channel_counts)}"
        )
    if len(route_channel_counts) != 1:
        raise ValueError(
            "one training run cannot mix route channel contracts: "
            f"{sorted(route_channel_counts)}"
        )
    map_context_channels = next(iter(map_context_channel_counts))
    route_channels = next(iter(route_channel_counts))
    navigation_geometry_id = None
    view_fusion_kwargs = None
    if dataset == Dataset.KITSCENES:
        from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY

        expected_geometry = DEFAULT_NAVIGATION_GEOMETRY.contract()
        mismatched_geometry = [
            path
            for path in shard_dirs
            if manifests[path].get("navigation_geometry")
            != expected_geometry
        ]
        if mismatched_geometry:
            raise ValueError(
                "KITScenes navigation geometry differs from the model "
                f"contract in shards: {mismatched_geometry[:3]}"
            )
        navigation_geometry_id = DEFAULT_NAVIGATION_GEOMETRY.geometry_id
        view_fusion_kwargs = (
            DEFAULT_NAVIGATION_GEOMETRY.camera_bev_kwargs()
        )

    from Platform.pipelines.training_checkpoint import stable_digest

    navigation_quality_report = None
    navigation_quality_audit_sha256 = None
    training_shard_dirs = list(shard_dirs)
    if dataset == Dataset.KITSCENES:
        if navigation_quality_audit is None:
            raise ValueError(
                "KITScenes training requires a navigation quality audit"
            )
        audit_path = navigation_quality_audit.download()
        try:
            with open(audit_path, encoding="ascii") as stream:
                supplied_audit = json.load(stream)
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(
                "could not read the KITScenes navigation quality audit"
            ) from error
        (
            training_shard_dirs,
            navigation_quality_report,
        ) = _verified_navigation_training_shard_dirs(
            shard_dirs,
            manifests,
            supplied_audit,
        )
        navigation_quality_audit_sha256 = stable_digest(
            navigation_quality_report
        )
    elif navigation_quality_audit is not None:
        raise ValueError(
            "navigation quality audits are supported only for KITScenes"
        )

    contract_digests = {
        stable_digest(manifest.get("contracts"))
        for manifest in manifests.values()
    }
    if uses_exact_validation and len(contract_digests) != 1:
        raise ValueError(
            "exact validation splitting requires one packed contract digest, "
            f"got {sorted(contract_digests)}"
        )
    packed_contract_digest = (
        next(iter(contract_digests))
        if len(contract_digests) == 1
        else ""
    )

    data_identity = []
    for shard_dir in all_shard_dirs:
        manifest = manifests[shard_dir]
        data_identity.append({
            "dataset": manifest.get("dataset"),
            "artifact_uri": shard_artifact_uris[shard_dir],
            "source_revision": manifest.get("source_revision"),
            "dataset_version": manifest.get("dataset_version"),
            "partition_id": manifest.get("partition_id"),
            "total_samples": int(manifest.get("total_samples", 0)),
            "shard_names": list(manifest.get("shard_names", [])),
            "contracts": manifest.get("contracts"),
            "num_views": int(manifest.get("num_views", 0)),
            "map_context_channels": int(
                manifest.get("map_context_channels", 3)
            ),
            "route_channels": int(manifest.get("route_channels", 2)),
            "has_route_supervision": bool(
                manifest.get("has_route_supervision", False)
            ),
            "route_supervision_version": manifest.get(
                "route_supervision_version"
            ),
            "navigation": manifest.get("navigation"),
            "navigation_geometry": manifest.get("navigation_geometry"),
            "has_world_model": bool(
                manifest.get("has_world_model", False)
            ),
            "reasoning_label_count": int(
                manifest.get("reasoning_label_count", 0)
            ),
        })
    data_identity.sort(
        key=lambda item: (
            str(item["partition_id"]),
            str(item["dataset"]),
        )
    )
    split_inventory = None
    available_split_groups: tuple[str, ...] = ()
    if uses_exact_validation:
        split_inventory = discover_split_inventory(shard_dirs)
        available_split_groups = split_inventory.group_uids
        expected_sample_count = sum(
            int(manifests[shard_dir].get("total_samples", 0))
            for shard_dir in shard_dirs
        )
        if split_inventory.sample_count != expected_sample_count:
            raise ValueError(
                "packed sample metadata coverage differs from manifests: "
                f"expected={expected_sample_count} "
                f"actual={split_inventory.sample_count}"
            )
    fixed_validation_groups = select_validation_group_uids(
        available_split_groups,
        val_fraction=val_fraction,
        policy=training_policy,
        source_revision=packed_source_revision,
        packed_dataset_version=dataset_version,
        packed_contract_digest=packed_contract_digest,
        packed_partition_count=len(manifests),
        empty_partition_count=skipped_empty,
        packed_sample_count=(
            split_inventory.sample_count
            if split_inventory is not None
            else None
        ),
        packed_sample_uid_digest=(
            split_inventory.sample_uid_digest
            if split_inventory is not None
            else None
        ),
    )
    available_group_digest = (
        group_uid_digest(available_split_groups)
        if fixed_validation_groups is not None
        else None
    )
    validation_group_digest = (
        group_uid_digest(fixed_validation_groups)
        if fixed_validation_groups is not None
        else None
    )
    selected_validation_sample_count = None
    selected_validation_sample_digest = None
    if fixed_validation_groups is not None:
        if split_inventory is None:
            raise RuntimeError(
                "exact validation splitting has no packed sample inventory"
            )
        (
            actual_validation_sample_count,
            actual_validation_sample_digest,
        ) = split_inventory.sample_identity_for_groups(
            fixed_validation_groups
        )
        if training_policy.validation_strategy == "exact_group_fraction":
            (
                frozen_validation_sample_count,
                frozen_validation_sample_digest,
            ) = validation_sample_identity(training_policy)
            if (
                actual_validation_sample_count
                != frozen_validation_sample_count
                or actual_validation_sample_digest
                != frozen_validation_sample_digest
            ):
                raise ValueError(
                    "packed validation sample identity differs from the frozen "
                    "split: "
                    f"expected_count={frozen_validation_sample_count} "
                    f"actual_count={actual_validation_sample_count} "
                    f"expected_digest={frozen_validation_sample_digest} "
                    f"actual_digest={actual_validation_sample_digest}"
                )
        (
            selected_validation_sample_count,
            selected_validation_sample_digest,
        ) = (
            actual_validation_sample_count,
            actual_validation_sample_digest,
        )
    reconstruction_audit_contract = None
    if selector_enabled:
        if reconstruction_audit is None:
            raise ValueError(
                "composite-selector training requires a reconstruction audit"
            )
        if reconstruction_audit_decision != "go":
            raise ValueError(
                "composite-selector training requires an explicit Go decision"
            )
        if len(reconstruction_audit_rationale.strip()) < 20:
            raise ValueError(
                "reconstruction audit Go rationale is too short"
            )
        audit_path = str(reconstruction_audit.download())
        with open(audit_path, "rb") as stream:
            audit_bytes = stream.read()
        try:
            audit_report = json.loads(audit_bytes)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "reconstruction audit is not valid JSON"
            ) from error
        if not isinstance(audit_report, dict):
            raise ValueError(
                "reconstruction audit must be a JSON object"
            )
        from evaluation.reconstruction_audit import AUDIT_SCHEMA_VERSION

        if (
            audit_report.get("schema_version")
            != AUDIT_SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported reconstruction audit schema"
            )
        heading_alignment = audit_report.get("heading_alignment")
        valid_heading_steps = (
            heading_alignment.get("valid_step_count")
            if isinstance(heading_alignment, dict)
            else None
        )
        if (
            not isinstance(heading_alignment, dict)
            or not isinstance(valid_heading_steps, int)
            or isinstance(valid_heading_steps, bool)
            or valid_heading_steps <= 0
            or heading_alignment.get("full_horizon") is None
        ):
            raise ValueError(
                "reconstruction audit has no usable heading alignment"
            )
        provenance = audit_report.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(
                "reconstruction audit has no provenance"
            )
        from training.losses.control_rollout import (
            ROLLOUT_POLICY_VERSION,
        )

        expected_audit_identity = {
            "dataset": Dataset.KITSCENES.value,
            "dataset_version": dataset_version,
            "source_revision": packed_source_revision,
            "packed_contract_digest": packed_contract_digest,
            "validation_group_uid_digest": validation_group_digest,
            "validation_sample_uid_digest": (
                selected_validation_sample_digest
            ),
            "rollout_policy_version": ROLLOUT_POLICY_VERSION,
        }
        mismatches = {
            key: {
                "expected": expected,
                "actual": provenance.get(key),
            }
            for key, expected in expected_audit_identity.items()
            if provenance.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "reconstruction audit identity differs from training: "
                f"{mismatches}"
            )
        if (
            int(audit_report.get("sample_count", -1))
            != selected_validation_sample_count
            or int(audit_report.get("scene_count", -1))
            != len(fixed_validation_groups or ())
        ):
            raise ValueError(
                "reconstruction audit coverage differs from validation"
            )
        from evaluation.reconstruction_audit import (
            P95_FDE_3S_LIMIT_M,
            P95_FDE_FULL_LIMIT_M,
        )

        expected_thresholds = {
            "p95_fde_3s_limit_m": P95_FDE_3S_LIMIT_M,
            "p95_fde_full_limit_m": P95_FDE_FULL_LIMIT_M,
        }
        if audit_report.get("thresholds") != expected_thresholds:
            raise ValueError(
                "reconstruction audit thresholds differ from training: "
                f"expected={expected_thresholds} "
                f"actual={audit_report.get('thresholds')}"
            )
        reconstruction_audit_contract = {
            "decision": reconstruction_audit_decision,
            "rationale": reconstruction_audit_rationale.strip(),
            "position_target_source": (
                "packed_logged_xy" if objective_v2 else "not_applicable"
            ),
            "report_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "thresholds_pass": bool(
                audit_report.get("thresholds_pass", False)
            ),
            "thresholds": expected_thresholds,
            "sample_count": int(audit_report["sample_count"]),
            "scene_count": int(audit_report["scene_count"]),
            "metrics": audit_report["metrics"],
            "heading_alignment": heading_alignment,
            "audit_code_revision": provenance.get(
                "audit_code_revision"
            ),
            "rollout_policy_version": provenance.get(
                "rollout_policy_version"
            ),
        }
    elif reconstruction_audit is not None:
        raise ValueError(
            "reconstruction audit is accepted only by composite-selector "
            "training"
        )
    navigation_repeat_policy = (
        NavigationRepeatPolicy()
        if enable_junction_sampling
        else None
    )
    navigation_exposure_audit = (
        discover_navigation_exposure(
            training_shard_dirs,
            policy=navigation_repeat_policy,
            validation_group_uids=fixed_validation_groups,
        )
        if navigation_repeat_policy is not None
        else None
    )
    navigation_exposure_metadata = (
        navigation_exposure_audit.metadata()
        if navigation_exposure_audit is not None
        else None
    )
    if navigation_exposure_metadata is not None:
        print(
            "Navigation exposure: "
            f"unique={navigation_exposure_audit.unique_sample_count} "
            f"effective={navigation_exposure_audit.effective_exposure_count} "
            f"digest={navigation_exposure_audit.exposure_digest}"
        )

    data_coverage = {
        "available_group_uid_digest": available_group_digest,
        "available_group_count": (
            len(available_split_groups)
            if fixed_validation_groups is not None
            else None
        ),
        "sample_uid_digest": (
            split_inventory.sample_uid_digest
            if split_inventory is not None
            else None
        ),
        "sample_count": (
            split_inventory.sample_count
            if split_inventory is not None
            else None
        ),
    }
    data_fingerprint_payload = {
        "partitions": data_identity,
        "coverage": data_coverage,
        "navigation_quality_audit_sha256": (
            navigation_quality_audit_sha256
        ),
        "training_partition_ids": (
            navigation_quality_report["accepted_partition_ids"]
            if navigation_quality_report is not None
            else None
        ),
        "navigation_exposure": navigation_exposure_metadata,
    }
    data_fingerprint = stable_digest(data_fingerprint_payload)
    data_fingerprint_without_navigation_repeat = stable_digest({
        **data_fingerprint_payload,
        "navigation_exposure": None,
    })
    validation_split_contract = {
        "strategy": training_policy.validation_strategy,
        "split_id": training_policy.validation_split_id,
        "val_fraction": val_fraction,
        "source_revision": packed_source_revision or None,
        "dataset_version": dataset_version,
        "packed_contract_digest": packed_contract_digest or None,
        "available_group_count": (
            len(available_split_groups)
            if fixed_validation_groups is not None
            else None
        ),
        "available_group_uid_digest": available_group_digest,
        "validation_group_count": (
            len(fixed_validation_groups)
            if fixed_validation_groups is not None
            else None
        ),
        "validation_group_uids": (
            list(fixed_validation_groups)
            if fixed_validation_groups is not None
            else None
        ),
        "validation_group_uid_digest": validation_group_digest,
        "packed_sample_count": data_coverage["sample_count"],
        "packed_sample_uid_digest": data_coverage["sample_uid_digest"],
        "validation_sample_count": selected_validation_sample_count,
        "validation_sample_uid_digest": selected_validation_sample_digest,
    }

    validation_loader = make_multi_dataset_loader(
        shard_dirs,
        batch_size=8,
        num_workers=min(num_workers, 1),
        shuffle=0,
        pin_memory=(device.type == "cuda"),
        split="val",
        val_fraction=val_fraction,
        validation_group_uids=fixed_validation_groups,
        max_active_loaders=1,
        prefetch_factor=1,
        decode_future_frames=False,
    )
    print(
        f"Selected {len(training_shard_dirs)}/{len(shard_dirs)} non-empty "
        "partition(s) for the optimizer "
        f"(skipped_empty={skipped_empty}, split=train, "
        f"val_fraction={val_fraction}, num_workers={num_workers}, "
        f"num_views={num_views}, data_fingerprint={data_fingerprint})."
    )
    print(
        "Validation split: "
        f"strategy={training_policy.validation_strategy} "
        f"split_id={training_policy.validation_split_id} "
        f"groups={validation_split_contract['validation_group_count']} "
        f"group_digest={validation_group_digest}"
    )

    from navigation.supervision import (
        ROUTE_SUPERVISION_ARTIFACT_VERSION,
    )

    # Consistency guard (packing ↔ training) across every non-empty partition.
    # Sparse reasoning targets are masked on unlabeled samples, so probing a
    # random first batch cannot distinguish an intentionally unlabeled sample
    # from a wholly unsupervised shard. The pack manifest records the exact join
    # count; validate that deterministic aggregate instead.
    for d in shard_dirs:
        manifest = manifests[d]
        dname = manifest.get("dataset", d)
        if (
            dataset == Dataset.KITSCENES
            and not manifest.get("has_navigation", False)
        ):
            raise ValueError(
                f"KITScenes shard '{dname}' ({d}) has no schema-v8 "
                "navigation artifacts"
            )
        if enable_route_consistency and (
            not manifest.get("has_route_supervision", False)
            or manifest.get("route_supervision_version")
            != ROUTE_SUPERVISION_ARTIFACT_VERSION
        ):
            raise ValueError(
                "route consistency requires "
                f"{ROUTE_SUPERVISION_ARTIFACT_VERSION} in "
                f"dataset '{dname}' ({d})"
            )
        if selector_enabled and (
            not manifest.get("has_route_supervision", False)
            or manifest.get("route_supervision_version")
            != ROUTE_SUPERVISION_ARTIFACT_VERSION
        ):
            raise ValueError(
                "rollout composite selector requires "
                f"{ROUTE_SUPERVISION_ARTIFACT_VERSION} "
                f"in dataset '{dname}' ({d})"
            )
        if selector_enabled and not manifest.get("has_gps", False):
            raise ValueError(
                "rollout composite selector requires packed pose and GPS "
                f"in dataset '{dname}' ({d})"
            )

    total_reasoning_labels = 0
    for d in training_shard_dirs:
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
            f"across {len(training_shard_dirs)} accepted partitions"
        )

    # Route is fused only through the Reactive navigation encoder. Reasoning
    # receives no route-derived argument.
    model = AutoE2E(
        backbone=bb, num_views=num_views, embed_dim=256,
        is_pretrained=True,
        view_fusion_kwargs=view_fusion_kwargs,
        map_context_channels=map_context_channels,
        route_channels=route_channels,
        enable_route_conditioning=enable_route_conditioning,
        map_fusion_mode=mfm,
        enable_reasoning=enable_reasoning, reasoning_mode=reasoning_mode,
        enable_world_model=enable_world_model,
    ).to(device)
    print(f"Reasoning: {'on' if enable_reasoning else 'off'}"
          + (f" (mode={reasoning_mode})" if enable_reasoning else ""))
    print(f"World Model: {'on' if enable_world_model else 'off'}")

    # Optimizer + scheduler + loss.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    selector_mode = "max" if selector_enabled else "min"
    selector_threshold = 0.0005 if selector_enabled else 1e-4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=selector_mode,
        factor=0.5,
        patience=1,
        threshold=selector_threshold,
        threshold_mode="abs",
    )
    loss_fn = TrajectoryImitationLoss(
        loss_type="smooth_l1",
        temporal_decay=training_policy.temporal_decay,
        temporal_weight_normalization=(
            training_policy.temporal_weight_normalization
        ),
        signal_scales=training_policy.signal_scales,
    )
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)
    print(
        "Dataset training policy: "
        f"auto_e2e_timesteps={AUTO_E2E_TIMESTEPS} "
        f"temporal_decay={training_policy.temporal_decay:.4g} "
        "temporal_weight_normalization="
        f"{training_policy.temporal_weight_normalization} "
        f"acceleration_scale={training_policy.signal_scales[0]:.4g} "
        f"curvature_scale={training_policy.signal_scales[1]:.4g}"
    )

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

    route_consistency_loss_fn = None
    route_consistency_config = {
        "enabled": enable_route_consistency,
        "weight": route_consistency_weight,
    }
    if enable_route_consistency:
        from training.losses import RouteConsistencyLoss

        route_consistency_loss_fn = RouteConsistencyLoss(
            temporal_decay=training_policy.temporal_decay,
        ).to(device)
        route_consistency_config.update(
            route_consistency_loss_fn.metadata()
        )
    rollout_aligned_loss_fn = None
    rollout_aligned_config = {
        "enabled": objective_v2,
        "rollout_weight": 0.5,
        "constraint_weight": 0.05,
    }
    if objective_v2:
        from evaluation.kitscenes_benchmark import (
            wgs84_trajectory_to_ego_xy,
        )
        from training.losses import RolloutAlignedLoss

        rollout_aligned_loss_fn = RolloutAlignedLoss().to(device)
        rollout_aligned_config.update(
            rollout_aligned_loss_fn.metadata()
        )
    if selector_enabled:
        from evaluation.checkpoint_selection import (
            SELECTOR_MIN_DELTA,
            SELECTOR_POLICY_VERSION,
            TOP_LEVEL_WEIGHTS,
            UTILITY_SCALES,
        )

        if selector_threshold != SELECTOR_MIN_DELTA:
            raise RuntimeError(
                "scheduler and checkpoint selector thresholds differ"
            )
        checkpoint_selection_config = {
            "enabled": True,
            "policy_version": SELECTOR_POLICY_VERSION,
            "min_delta": SELECTOR_MIN_DELTA,
            "top_level_weights": dict(TOP_LEVEL_WEIGHTS),
            "utility_scales": dict(UTILITY_SCALES),
        }
    else:
        checkpoint_selection_config = {
            "enabled": False,
            "policy_version": "ade_fde_lexicographic_v1",
            "min_delta": None,
            "top_level_weights": None,
            "utility_scales": None,
        }

    scaler = torch.amp.GradScaler(enabled=amp)
    checkpoint_config = {
        "backbone": bb,
        # Carried in the checkpoint so evaluation rebuilds the SAME architecture:
        # _model_kwargs feeds this dict into AutoE2E(**config), so without the key
        # a non-default run would be reconstructed with the constructor default
        # and load mismatched weights.
        "map_fusion_mode": mfm,
        "embed_dim": 256,
        "num_views": num_views,
        "view_fusion_kwargs": view_fusion_kwargs,
        "navigation_geometry_id": navigation_geometry_id,
        "map_context_channels": map_context_channels,
        "route_channels": route_channels,
        "enable_route_conditioning": enable_route_conditioning,
        "navigation_quality_audit_sha256": (
            navigation_quality_audit_sha256
        ),
        # Checkpoints contain the complete backbone. Reconstruction must not
        # download pretrained weights before loading that state.
        "is_pretrained": False,
        "enable_reasoning": enable_reasoning,
        "reasoning_mode": reasoning_mode,
        "enable_world_model": enable_world_model,
        "optimizer": "AdamW",
        "lr": lr,
        "training_seed": training_seed,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "amp": amp,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "num_workers": num_workers,
        "reasoning_loss_weight": reasoning_loss_weight,
        "jepa_loss_weight": jepa_loss_weight,
        "training_objective_version": training_objective_version,
        "junction_sampling": {
            "enabled": enable_junction_sampling,
            "policy": (
                navigation_repeat_policy.metadata()
                if navigation_repeat_policy is not None
                else None
            ),
        },
        "route_consistency": route_consistency_config,
        "rollout_aligned_loss": rollout_aligned_config,
        "checkpoint_selection": checkpoint_selection_config,
        "reconstruction_audit": reconstruction_audit_contract,
        "trajectory_training_policy": training_policy.metadata(),
        "val_fraction": val_fraction,
        "validation_scope": validation_scope,
        "validation_split": validation_split_contract,
        "early_stopping_patience": early_stopping_patience,
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": selector_mode,
            "factor": 0.5,
            "patience": 1,
            "threshold": selector_threshold,
            "threshold_mode": "abs",
        },
    }

    from Platform.pipelines.training_checkpoint import (
        CHECKPOINT_SCHEMA_VERSION,
        capture_rng_state,
        checkpoint_key,
        metric_pair_is_better,
        rescale_partial_accumulation_gradients,
        restore_rng_state,
        sha256_file,
        update_best_pointer,
        validate_immutable_checkpoint_record,
        upload_immutable_checkpoint,
        validate_resume_envelope,
        validate_resume_payload,
    )

    checkpoint_bucket = _checkpoint_bucket_name()
    s3_client = boto3.client("s3")
    metric_history: list[dict] = []
    best_checkpoint = None
    best_trajectory_checkpoint = None
    final_checkpoint = None
    selector_availability = None
    bad_epochs = 0
    expected_validation_digest = selected_validation_sample_digest
    validation_sample_count = selected_validation_sample_count
    start_epoch = 1
    best_local_path = None
    resumed = resume_from is not None
    resume_policy_transition = None
    resume_optimization_state = None
    terminal_resume = False
    stopped_early = False

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("imitation-learning")
    if resumed:
        resume_path = resume_from.download()
        # RNG tensors must stay on CPU for torch.set_rng_state. Checkpoints are
        # trusted pipeline artifacts and include NumPy RNG state, so opt out of
        # the PyTorch 2.6+ weights-only default explicitly.
        resume_payload = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        validate_resume_envelope(resume_payload)
        if allow_resume_policy_transition:
            resume_policy_transition = _resume_policy_transition(
                saved_config=dict(resume_payload["config"]),
                requested_config=checkpoint_config,
            )
        validate_resume_payload(
            resume_payload,
            expected_config=checkpoint_config,
            expected_data_fingerprint=data_fingerprint,
            allowed_config_changes=(
                frozenset({
                    "junction_sampling",
                    "early_stopping_patience",
                })
                if resume_policy_transition is not None
                else frozenset()
            ),
            compatible_data_fingerprints=(
                frozenset({
                    data_fingerprint_without_navigation_repeat,
                })
                if resume_policy_transition is not None
                else frozenset()
            ),
        )
        model.load_state_dict(resume_payload["model_state_dict"])
        resume_optimization_state = _restore_resume_optimization_state(
            optimizer,
            scheduler,
            resume_payload,
            transition=resume_policy_transition,
        )
        scaler.load_state_dict(resume_payload["scaler_state_dict"])
        state = dict(resume_payload["training_state"])
        run_id = str(state.get("run_id", ""))
        if not run_id:
            raise ValueError("resume checkpoint has no MLflow run ID")
        completed_epoch = int(resume_payload["epoch"])
        start_epoch = completed_epoch + 1
        metric_history = list(state.get("metric_history", []))
        resumed_checkpoint = _resumed_checkpoint_record(
            resume_payload, resume_path
        )
        metric_history[-1].setdefault(
            "checkpoint_uri", resumed_checkpoint["uri"]
        )
        metric_history[-1].setdefault(
            "checkpoint_sha256", resumed_checkpoint["sha256"]
        )
        final_checkpoint = resumed_checkpoint
        saved_best = state.get("best")
        best_checkpoint = dict(saved_best) if saved_best is not None else None
        saved_best_trajectory = state.get("best_trajectory")
        bad_epochs = int(state.get("bad_epochs", 0))
        if resume_policy_transition is not None:
            resume_policy_transition["source_checkpoint"] = {
                "epoch": completed_epoch,
                "uri": resumed_checkpoint["uri"],
                "sha256": resumed_checkpoint["sha256"],
            }
        saved_selector_availability = state.get(
            "checkpoint_selector_availability"
        )
        if selector_enabled:
            if not isinstance(saved_selector_availability, dict):
                raise ValueError(
                    "resume checkpoint has no frozen selector availability"
                )
            selector_availability = dict(
                saved_selector_availability
            )
        elif saved_selector_availability is not None:
            raise ValueError(
                "legacy checkpoint selection cannot restore composite state"
            )
        saved_validation_digest = state.get(
            "validation_sample_uid_digest"
        )
        saved_validation_count = state.get("validation_sample_count")
        if (
            selected_validation_sample_digest is not None
            and (
                saved_validation_digest
                != selected_validation_sample_digest
                or saved_validation_count
                != selected_validation_sample_count
            )
        ):
            raise ValueError(
                "resume checkpoint validation sample identity differs from "
                "the selected split"
            )
        expected_validation_digest = saved_validation_digest
        validation_sample_count = saved_validation_count
        if (
            best_checkpoint is not None
            and int(best_checkpoint["epoch"]) == completed_epoch
        ):
            best_local_path = resume_path
            saved_digest = best_checkpoint.get("sha256")
            if saved_digest not in (None, resumed_checkpoint["sha256"]):
                raise ValueError(
                    "resume checkpoint bytes differ from its saved best digest"
                )
            best_checkpoint = dict(resumed_checkpoint)
        if best_checkpoint is None:
            raise ValueError("resume checkpoint has no selected best checkpoint")
        if selector_enabled:
            best_selection = best_checkpoint.get("selection")
            if (
                not isinstance(best_selection, dict)
                or best_selection.get("policy_version")
                != checkpoint_selection_config["policy_version"]
            ):
                raise ValueError(
                    "resume checkpoint best selection policy differs "
                    "from the requested selector"
                )
            if saved_best_trajectory is None:
                best_trajectory_checkpoint = (
                    _best_trajectory_checkpoint_from_history(
                        metric_history,
                        expected_policy_version=str(
                            checkpoint_selection_config["policy_version"]
                        ),
                        min_delta=float(
                            checkpoint_selection_config["min_delta"]
                        ),
                    )
                )
            else:
                best_trajectory_checkpoint = dict(saved_best_trajectory)
            if (
                int(best_trajectory_checkpoint["epoch"])
                == completed_epoch
            ):
                saved_digest = best_trajectory_checkpoint.get("sha256")
                if saved_digest not in (
                    None,
                    resumed_checkpoint["sha256"],
                ):
                    raise ValueError(
                        "resume checkpoint bytes differ from its saved "
                        "trajectory-best digest"
                    )
                best_trajectory_checkpoint = dict(resumed_checkpoint)
            trajectory_selection = best_trajectory_checkpoint.get(
                "selection"
            )
            if (
                not isinstance(trajectory_selection, dict)
                or trajectory_selection.get("policy_version")
                != checkpoint_selection_config["policy_version"]
                or not isinstance(
                    trajectory_selection.get("components"), dict
                )
                or "trajectory"
                not in trajectory_selection["components"]
            ):
                raise ValueError(
                    "resume checkpoint trajectory-best selection differs "
                    "from the requested selector"
                )
        else:
            best_trajectory_checkpoint = dict(best_checkpoint)
        validate_immutable_checkpoint_record(s3_client, best_checkpoint)
        validate_immutable_checkpoint_record(
            s3_client,
            best_trajectory_checkpoint,
        )
        (
            bad_epochs,
            best_checkpoint,
            best_trajectory_checkpoint,
        ) = (
            _transition_resume_selection_state(
                resume_policy_transition,
                bad_epochs=bad_epochs,
                best_checkpoint=best_checkpoint,
                best_trajectory_checkpoint=best_trajectory_checkpoint,
            )
        )
        if resume_policy_transition is not None:
            resume_policy_transition["optimization_state"] = (
                resume_optimization_state
            )
        terminal_resume, stopped_early = _resume_terminal_state(
            completed_epoch=completed_epoch,
            bad_epochs=bad_epochs,
            requested_epochs=epochs,
            patience=early_stopping_patience,
        )
        if resume_policy_transition is not None and terminal_resume:
            raise ValueError(
                "resume policy transition requires at least one new epoch"
            )
        if best_checkpoint is not None:
            update_best_pointer(
                s3_client,
                bucket=checkpoint_bucket,
                run_id=run_id,
                epoch=int(best_checkpoint["epoch"]),
                checkpoint_uri=str(best_checkpoint["uri"]),
                checkpoint_sha256=str(best_checkpoint["sha256"]),
                ade=float(best_checkpoint["ade"]),
                fde=float(best_checkpoint["fde"]),
                selection=best_checkpoint.get("selection"),
                metric_contract=best_checkpoint.get("metric_contract"),
            )
        if best_trajectory_checkpoint is not None:
            update_best_pointer(
                s3_client,
                bucket=checkpoint_bucket,
                run_id=run_id,
                role="best_trajectory",
                epoch=int(best_trajectory_checkpoint["epoch"]),
                checkpoint_uri=str(best_trajectory_checkpoint["uri"]),
                checkpoint_sha256=str(
                    best_trajectory_checkpoint["sha256"]
                ),
                ade=float(best_trajectory_checkpoint["ade"]),
                fde=float(best_trajectory_checkpoint["fde"]),
                selection=best_trajectory_checkpoint.get("selection"),
                metric_contract=best_trajectory_checkpoint.get(
                    "metric_contract"
                ),
            )
        restore_rng_state(resume_payload["rng_state"])
        if resume_policy_transition is not None:
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tags({
                    "resume_policy_transition": (
                        resume_policy_transition["policy_version"]
                    ),
                    "resume_bad_epochs_reset": "true",
                    "resume_plateau_state_reset": "true",
                    "resume_optimizer_lr_preserved": "true",
                    "resume_best_checkpoints_preserved": "true",
                    "resume_navigation_repeat_enabled": str(
                        resume_policy_transition["junction_sampling"][
                            "to"
                        ]["enabled"]
                    ).lower(),
                    "resume_early_stopping_patience": str(
                        early_stopping_patience
                    ),
                })
        print(
            f"Resuming MLflow run {run_id} at epoch {start_epoch}; "
            f"bad_epochs={bad_epochs} terminal={terminal_resume}"
        )
    else:
        run_name = f"{bb}-{fm}-e{epochs}"
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.log_params({
                "data/dataset": dataset.value,
                "data/dataset_version": dataset_version,
                "data/fingerprint": data_fingerprint,
                "data/navigation_quality_audit_sha256": (
                    navigation_quality_audit_sha256 or "none"
                ),
                "data/training_partition_count": len(
                    training_shard_dirs
                ),
                "model/backbone": bb,
                "model/fusion_mode": fm,
                "model/map_fusion_mode": mfm,
                "model/num_views": num_views,
                "model/navigation_geometry_id": (
                    navigation_geometry_id or "legacy"
                ),
                "model/enable_route_conditioning": (
                    enable_route_conditioning
                ),
                "train/batch_size": batch_size,
                "train/grad_accum_steps": grad_accum_steps,
                "train/num_workers": num_workers,
                "train/epochs": epochs,
                "train/lr": lr,
                "train/seed": training_seed,
                "train/weight_decay": weight_decay,
                "train/amp": amp,
                "train/cudnn_benchmark": (
                    torch.backends.cudnn.benchmark
                ),
                "train/cudnn_deterministic": (
                    torch.backends.cudnn.deterministic
                ),
                "train/acceleration_signal_scale": (
                    training_policy.signal_scales[0]
                ),
                "train/curvature_signal_scale": (
                    training_policy.signal_scales[1]
                ),
                "model/trajectory_timesteps": AUTO_E2E_TIMESTEPS,
                "train/temporal_decay": (
                    training_policy.temporal_decay
                ),
                "train/temporal_weight_normalization": (
                    training_policy.temporal_weight_normalization
                ),
                "train/objective_version": training_objective_version,
                "train/reconstruction_audit_sha256": (
                    reconstruction_audit_contract["report_sha256"]
                    if reconstruction_audit_contract is not None
                    else "none"
                ),
                "train/junction_sampling_enabled": (
                    enable_junction_sampling
                ),
                "train/navigation_repeat_policy_version": (
                    navigation_repeat_policy.version
                    if navigation_repeat_policy is not None
                    else "none"
                ),
                "train/navigation_turn_repeat": (
                    navigation_repeat_policy.turn_repeat
                    if navigation_repeat_policy is not None
                    else 1
                ),
                "train/navigation_junction_repeat": (
                    navigation_repeat_policy.junction_repeat
                    if navigation_repeat_policy is not None
                    else 1
                ),
                "train/navigation_exposure_digest": (
                    navigation_exposure_audit.exposure_digest
                    if navigation_exposure_audit is not None
                    else "none"
                ),
                "train/navigation_unique_samples": (
                    navigation_exposure_audit.unique_sample_count
                    if navigation_exposure_audit is not None
                    else -1
                ),
                "train/navigation_effective_exposures": (
                    navigation_exposure_audit.effective_exposure_count
                    if navigation_exposure_audit is not None
                    else -1
                ),
                "train/route_consistency_enabled": (
                    enable_route_consistency
                ),
                "train/route_consistency_weight": (
                    route_consistency_weight
                ),
                "train/route_artifact_version": (
                    route_consistency_config.get(
                        "artifact_version",
                        "none",
                    )
                ),
                "train/route_target_compliance_threshold": (
                    route_consistency_config.get(
                        "target_compliance_threshold",
                        -1.0,
                    )
                ),
                **{
                    f"train/route_term_weight_{name}": value
                    for name, value in route_consistency_config.get(
                        "term_weights",
                        {},
                    ).items()
                },
                "train/val_fraction": val_fraction,
                "train/validation_scope": validation_scope,
                "train/validation_strategy": (
                    training_policy.validation_strategy
                ),
                "train/validation_split_id": (
                    training_policy.validation_split_id
                ),
                "train/validation_group_count": (
                    len(fixed_validation_groups)
                    if fixed_validation_groups is not None
                    else -1
                ),
                "train/validation_group_uid_digest": (
                    validation_group_digest or "hash_buckets"
                ),
                "train/early_stopping_patience": early_stopping_patience,
                "train/checkpoint_selector_policy": (
                    checkpoint_selection_config["policy_version"]
                ),
                "train/checkpoint_selector_min_delta": (
                    checkpoint_selection_config["min_delta"]
                    if selector_enabled
                    else -1.0
                ),
                **{
                    f"train/checkpoint_selector_weight_{name}": value
                    for name, value in (
                        checkpoint_selection_config[
                            "top_level_weights"
                        ] or {}
                    ).items()
                },
                **{
                    f"train/checkpoint_selector_scale_{name}": value
                    for name, value in (
                        checkpoint_selection_config[
                            "utility_scales"
                        ] or {}
                    ).items()
                },
                "ctx/train_execution_id": train_execution_id,
                "ctx/train_docker_image": TRAINING_IMAGE,
            })
            if reconstruction_audit_contract is not None:
                audit_metrics = {
                    "audit/reconstruction/sample_count": float(
                        reconstruction_audit_contract["sample_count"]
                    ),
                    "audit/reconstruction/scene_count": float(
                        reconstruction_audit_contract["scene_count"]
                    ),
                    "audit/reconstruction/thresholds_pass": float(
                        reconstruction_audit_contract["thresholds_pass"]
                    ),
                }
                for metric_name, aggregates in (
                    reconstruction_audit_contract["metrics"].items()
                ):
                    for aggregate_name, distribution in (
                        aggregates.items()
                    ):
                        for statistic, value in distribution.items():
                            audit_metrics[
                                "audit/reconstruction/"
                                f"{metric_name}/{aggregate_name}/{statistic}"
                            ] = float(value)
                mlflow.log_metrics(audit_metrics, step=0)

    if selector_enabled and not resumed:
        from evaluation.checkpoint_selection import (
            aggregate_validation_records,
            freeze_component_availability,
        )

        preflight_validation = _evaluate_open_loop(
            model,
            validation_loader,
            device,
            training_policy=training_policy,
            include_rollout_selector_records=True,
        )
        _validate_selector_preflight_identity(
            preflight_validation,
            expected_sample_count=validation_sample_count,
            expected_sample_uid_digest=expected_validation_digest,
        )
        preflight_aggregates = aggregate_validation_records(
            preflight_validation["rollout_selector_records"]
        )
        selector_availability = freeze_component_availability(
            preflight_aggregates
        )
        print(
            "Frozen checkpoint selector availability before epoch 1: "
            f"{selector_availability}"
        )

    # Training loop
    model.train()
    losses_per_epoch = [
        float(entry["train_loss"]) for entry in metric_history
    ]

    _proj_cache = _ProjectionDeviceCache(device)
    optimizer_step_count = 0
    route_valid_sample_count = 0
    route_sample_count = 0
    gradient_evidence = {
        "first_step": None,
        "navigation_encoder_first_nonzero_step": None,
        "route_input_first_nonzero_step": None,
        "route_loss_gradient_budget": None,
        "objective_term_gradient_norms": None,
    }
    # Watch EVERY trainable planner parameter, not just the first one
    # named_parameters() happens to yield. The probe asks "did the optimizer
    # move the planner at all", and a single parameter cannot answer that: a
    # zero-initialised projection fed an all-zero input is provably immovable
    # (its gradient is zero, and AdamW's decoupled decay scales the parameter,
    # so a zero parameter stays zero). BezierPlanner.visual_history_proj is
    # exactly that whenever the World Model is off — which is the default, and
    # which makes visual_history all-zero in the packed data. Probing it alone
    # fails a perfectly healthy run.
    optimizer_probe = [
        (name, parameter, parameter.detach().clone())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "TrajectoryPlanner" in name
    ]
    if not optimizer_probe:
        raise RuntimeError(
            "no trainable TrajectoryPlanner parameters to verify progress on"
        )

    def _branch_gradient_norm(name_fragment):
        total, count = 0.0, 0
        for parameter_name, parameter in model.named_parameters():
            if (
                name_fragment in parameter_name
                and parameter.grad is not None
            ):
                total += float(parameter.grad.norm().item()) ** 2
                count += 1
        return {"norm": total ** 0.5, "parameter_count": count}

    def _observe_gradient_flow(step_number):
        planner = _branch_gradient_norm("TrajectoryPlanner")
        navigation_fusion = _branch_gradient_norm("MapBEVFusion")
        navigation_encoder = _branch_gradient_norm("NavigationEncoder")
        if gradient_evidence["first_step"] is None:
            first = {
                "optimizer_step": step_number,
                "planner": planner,
                "navigation_fusion": navigation_fusion,
                "navigation_encoder": navigation_encoder,
            }
            if enable_world_model:
                first["world_model"] = _branch_gradient_norm(
                    "World_Action_Model"
                )
            if enable_reasoning:
                first["reasoning"] = _branch_gradient_norm("Reasoning")
            gradient_evidence["first_step"] = first
            print(f"grad-flow probe: {first}")
        if (
            navigation_encoder["norm"] > 0.0
            and gradient_evidence[
                "navigation_encoder_first_nonzero_step"
            ] is None
        ):
            gradient_evidence[
                "navigation_encoder_first_nonzero_step"
            ] = {
                "optimizer_step": step_number,
                **navigation_encoder,
            }
            print(
                "navigation encoder gradient became non-zero: "
                f"{gradient_evidence['navigation_encoder_first_nonzero_step']}"
            )

    def _gradient_list_norm(gradients):
        return sum(
            float(gradient.detach().norm().item()) ** 2
            for gradient in gradients
            if gradient is not None
        ) ** 0.5

    accum = max(1, int(grad_accum_steps))
    if accum > 1:
        print(f"Gradient accumulation: {accum} micro-batches "
              f"(effective batch size = {batch_size * accum})")
    os.makedirs("/tmp/train", exist_ok=True)
    epoch_range = range(0) if terminal_resume else range(
        start_epoch, epochs + 1
    )
    for epoch in epoch_range:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_compute_started = time.perf_counter()
        merged = make_multi_dataset_loader(
            training_shard_dirs,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            split="train",
            val_fraction=val_fraction,
            validation_group_uids=fixed_validation_groups,
            shuffle_seed=1729 + epoch * 1_000_003,
            navigation_repeat_policy=navigation_repeat_policy,
        )
        epoch_losses = []
        traj_losses = []
        jepa_vals = []
        reason_vals = []
        route_vals = []
        rollout_term_sums = {
            name: 0.0
            for name in (
                "rollout",
                "path",
                "final",
                "constraint",
                "comfort",
                "jerk",
                "lateral_acceleration",
                "map",
                "route",
                "drivable",
            )
        }
        rollout_term_weights = {
            name: 0
            for name in rollout_term_sums
        }
        rollout_term_counts = {
            "map_sample_count": 0,
            "route_sample_count": 0,
            "drivable_sample_count": 0,
        }
        route_term_vals = {
            "corridor": [],
            "branch": [],
            "destination": [],
            "heading": [],
        }
        route_epoch_counts = {
            "candidate_count": 0,
            "eligible_count": 0,
            "compliance_rejected_count": 0,
            "corridor_active_count": 0,
            "branch_active_count": 0,
            "destination_active_count": 0,
            "heading_active_count": 0,
        }
        route_target_compliance_sum = 0.0
        epoch_training_sample_count = 0
        epoch_optimizer_step_start = optimizer_step_count
        micro_idx = 0  # position within the current accumulation window
        # Merged loader yields (batch, projection, geometry_type): each batch is
        # same-dataset (uniform num_views/geometry) but datasets are interleaved,
        # so the per-batch projection is applied to the batch it belongs to.
        for batch, batch_proj, batch_geom in merged:
            visual = batch["visual_tiles"].to(device)        # (B, V, 3, H, W)
            epoch_training_sample_count += int(visual.shape[0])
            ego_hist = adapt_egomotion_history(
                batch["egomotion_history"].to(device),
                training_policy,
            )
            vis_hist = batch["visual_history"].to(device)     # (B, 896)
            target = batch["trajectory_target"].to(device)    # (B, 128)
            map_context = batch["map_context"].to(device)
            route_mask = batch["route_mask"].to(device)
            map_valid = batch["map_valid"].to(device)
            route_valid = batch["route_valid"].to(device)
            route_valid_sample_count += int(route_valid.sum().item())
            route_sample_count += int(route_valid.numel())
            route_supervision = None
            route_intersection = None
            logged_positions = None
            if (
                route_consistency_loss_fn is not None
                or rollout_aligned_loss_fn is not None
            ):
                route_supervision = {
                    key: value.to(device)
                    for key, value in batch["route_supervision"].items()
                }
                if rollout_aligned_loss_fn is not None:
                    pose_current = batch.get("pose_current")
                    gps_future = batch.get("gps_future")
                    if pose_current is None or gps_future is None:
                        raise ValueError(
                            "rollout-aligned loss requires packed pose and GPS"
                        )
                    logged_positions = torch.from_numpy(
                        wgs84_trajectory_to_ego_xy(
                            gps_future.detach().cpu().numpy(),
                            pose_current.detach().cpu().numpy(),
                        )
                    ).to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                if route_consistency_loss_fn is not None:
                    navigation_metadata = batch.get(
                        "navigation_metadata",
                        {},
                    )
                    route_intersection = torch.tensor(
                        [
                            bool(
                                _collated_metadata_value(
                                    navigation_metadata,
                                    "route_intersection",
                                    sample_index,
                                    False,
                                )
                            )
                            for sample_index in range(visual.shape[0])
                        ],
                        device=device,
                        dtype=torch.bool,
                    )
            probe_route_gradient = (
                enable_route_conditioning
                and gradient_evidence[
                    "route_input_first_nonzero_step"
                ] is None
                and bool(route_valid.any().item())
            )
            if probe_route_gradient:
                route_mask = route_mask.detach().requires_grad_(True)

            # A weak object-key cache cannot alias a newly opened scene when
            # Python reuses the identity of a projection from a retired loader.
            proj_dev = _proj_cache.get(batch_proj)

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
                out = model(visual, map_context, vis_hist, ego_hist,
                            route_mask=route_mask,
                            map_valid=map_valid,
                            route_valid=route_valid,
                            projection=proj_dev, geometry_type=batch_geom,
                            mode="train", trajectory_target=target,
                            history_frames=history_frames, future_frames=future_frames)
                # Train mode returns (trajectory, aux) when a branch (reasoning /
                # world model) is on; otherwise just the trajectory tensor.
                trajectory, aux = out if isinstance(out, tuple) else (out, {})
                traj_loss = loss_fn(trajectory, target)
                loss = traj_loss
                initial_speed = ego_hist.reshape(
                    ego_hist.shape[0],
                    AUTO_E2E_TIMESTEPS,
                    -1,
                )[:, -1, 0]

                route_terms = None
                if route_consistency_loss_fn is not None:
                    assert route_supervision is not None
                    assert route_intersection is not None
                    route_terms = route_consistency_loss_fn(
                        trajectory,
                        target,
                        initial_speed,
                        route_supervision,
                        route_valid,
                        route_intersection,
                    )
                    loss = (
                        loss
                        + route_consistency_weight
                        * route_terms["total"]
                    )
                rollout_terms = None
                if rollout_aligned_loss_fn is not None:
                    assert route_supervision is not None
                    assert logged_positions is not None
                    rollout_terms = rollout_aligned_loss_fn(
                        trajectory,
                        target,
                        initial_speed,
                        logged_positions,
                        route_supervision,
                        map_valid,
                        route_valid,
                    )
                    loss = (
                        loss
                        + 0.5 * rollout_terms["rollout"]
                        + 0.05 * rollout_terms["constraint"]
                    )

                # JEPA loss (#13): future-feature reconstruction, added when the
                # WM ran the windowed path AND this batch carries future frames.
                jepa_val = 0.0
                weighted_jepa = None
                future_state_pred = aux.get("future_state_pred")
                if (enable_world_model and future_state_pred is not None
                        and future_frames is not None):
                    jepa = model.World_Action_Model_E2E.jepa_loss(
                        future_state_pred, future_frames)
                    weighted_jepa = jepa_loss_weight * jepa
                    loss = loss + weighted_jepa
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

            if (
                objective_v2
                and rollout_terms is not None
                and weighted_jepa is not None
                and gradient_evidence[
                    "objective_term_gradient_norms"
                ] is None
            ):
                planner_parameters = [
                    parameter
                    for name, parameter in model.named_parameters()
                    if (
                        parameter.requires_grad
                        and "TrajectoryPlanner" in name
                    )
                ]
                world_model_parameters = [
                    parameter
                    for name, parameter in model.named_parameters()
                    if (
                        parameter.requires_grad
                        and "World_Action_Model_E2E" in name
                    )
                ]
                objective_terms = {
                    "action": (traj_loss, planner_parameters),
                    "weighted_rollout": (
                        0.5 * rollout_terms["rollout"],
                        planner_parameters,
                    ),
                    "weighted_constraint": (
                        0.05 * rollout_terms["constraint"],
                        planner_parameters,
                    ),
                    "weighted_jepa": (
                        weighted_jepa,
                        world_model_parameters,
                    ),
                }
                term_norms = {}
                for term_name, (
                    term_loss,
                    term_parameters,
                ) in objective_terms.items():
                    gradients = torch.autograd.grad(
                        term_loss,
                        term_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    term_norms[term_name] = _gradient_list_norm(
                        gradients
                    )
                if not all(
                    np.isfinite(value)
                    for value in term_norms.values()
                ):
                    raise RuntimeError(
                        "rollout-aligned objective produced non-finite "
                        f"gradient evidence: {term_norms}"
                    )
                if term_norms["weighted_jepa"] <= 0.0:
                    raise RuntimeError(
                        "weighted JEPA produced no World Model gradient"
                    )
                gradient_evidence[
                    "objective_term_gradient_norms"
                ] = term_norms
                print(
                    "rollout-aligned objective gradient norms: "
                    f"{term_norms}"
                )

            if (
                route_terms is not None
                and int(route_terms["eligible_count"].item()) > 0
                and gradient_evidence["route_loss_gradient_budget"] is None
            ):
                planner_parameters = [
                    parameter
                    for name, parameter in model.named_parameters()
                    if (
                        parameter.requires_grad
                        and "TrajectoryPlanner" in name
                    )
                ]
                trajectory_gradients = torch.autograd.grad(
                    traj_loss,
                    planner_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                weighted_route_gradients = torch.autograd.grad(
                    route_consistency_weight * route_terms["total"],
                    planner_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                trajectory_gradient_norm = _gradient_list_norm(
                    trajectory_gradients
                )
                route_gradient_norm = _gradient_list_norm(
                    weighted_route_gradients
                )
                if trajectory_gradient_norm <= 0.0:
                    raise RuntimeError(
                        "trajectory planner gradient budget reference is zero"
                    )
                gradient_ratio = (
                    route_gradient_norm / trajectory_gradient_norm
                )
                gradient_evidence["route_loss_gradient_budget"] = {
                    "trajectory_planner_norm": trajectory_gradient_norm,
                    "weighted_route_planner_norm": route_gradient_norm,
                    "route_to_trajectory_ratio": gradient_ratio,
                    "maximum_ratio": 2.0,
                }
                print(
                    "route loss gradient budget: "
                    f"{gradient_evidence['route_loss_gradient_budget']}"
                )
                if gradient_ratio > 2.0:
                    raise RuntimeError(
                        "weighted route planner gradient exceeds the 2x "
                        "trajectory gradient budget"
                    )

            # Divide by accum so summed micro-batch grads equal the MEAN gradient
            # of an effective batch of (batch_size * accum) — same scale as a plain
            # step, so lr/grad_clip keep their meaning. Log the unscaled loss.
            scaler.scale(loss / accum).backward()
            if probe_route_gradient and route_mask.grad is not None:
                route_gradient_norm = float(route_mask.grad.norm().item())
                if route_gradient_norm > 0.0:
                    gradient_evidence[
                        "route_input_first_nonzero_step"
                    ] = {
                        "optimizer_step": optimizer_step_count + 1,
                        "norm": route_gradient_norm,
                    }
                    print(
                        "route input gradient became non-zero: "
                        f"{gradient_evidence['route_input_first_nonzero_step']}"
                    )

            epoch_losses.append(loss.item())
            traj_losses.append(traj_loss.item())
            jepa_vals.append(jepa_val)
            reason_vals.append(reason_val)
            route_vals.append(
                float(route_terms["total"].item())
                if route_terms is not None
                else 0.0
            )
            if route_terms is not None:
                for term_name in route_term_vals:
                    route_term_vals[term_name].append(
                        float(route_terms[term_name].item())
                    )
                for count_name in route_epoch_counts:
                    route_epoch_counts[count_name] += int(
                        route_terms[count_name].item()
                    )
                route_target_compliance_sum += float(
                    route_terms["target_compliance_sum"].item()
                )
            if rollout_terms is not None:
                _accumulate_rollout_epoch_terms(
                    rollout_term_sums,
                    rollout_term_weights,
                    rollout_terms,
                    batch_sample_count=int(trajectory.shape[0]),
                )
                for count_name in rollout_term_counts:
                    rollout_term_counts[count_name] += int(
                        rollout_terms[count_name].item()
                    )

            # Step only at the end of an accumulation window (or plain step when
            # accum==1). Grads persist across micro-batches until then.
            micro_idx += 1
            if micro_idx < accum:
                continue
            micro_idx = 0

            scaler.unscale_(optimizer)
            _observe_gradient_flow(optimizer_step_count + 1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_count += 1

        # Flush a trailing partial accumulation window (epoch batch count not a
        # multiple of accum) so its grads aren't silently dropped at epoch end.
        if micro_idx > 0:
            scaler.unscale_(optimizer)
            rescale_partial_accumulation_gradients(
                model.parameters(),
                accumulation_steps=accum,
                partial_count=micro_idx,
            )
            _observe_gradient_flow(optimizer_step_count + 1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_count += 1
            micro_idx = 0

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_wall_seconds = (
            time.perf_counter() - epoch_compute_started
        )
        if not epoch_losses:
            raise ValueError(
                "the internal train split produced no batches"
            )
        if training_wall_seconds <= 0.0:
            raise RuntimeError("training wall time must be positive")
        epoch_optimizer_steps = (
            optimizer_step_count - epoch_optimizer_step_start
        )
        if epoch_optimizer_steps <= 0:
            raise RuntimeError(
                "epoch completed without an optimizer step"
            )
        avg_loss = float(np.mean(epoch_losses))
        avg_traj = float(np.mean(traj_losses))
        avg_jepa = float(np.mean(jepa_vals))
        avg_reason = float(np.mean(reason_vals))
        avg_route = float(np.mean(route_vals))
        avg_route_terms = {
            name: (
                float(np.mean(values))
                if values
                else 0.0
            )
            for name, values in route_term_vals.items()
        }
        avg_rollout_terms = {
            name: (
                rollout_term_sums[name]
                / rollout_term_weights[name]
                if rollout_term_weights[name] > 0
                else 0.0
            )
            for name in rollout_term_sums
        }
        if (
            enable_route_consistency
            and route_epoch_counts["eligible_count"] <= 0
        ):
            raise RuntimeError(
                "route-enabled epoch produced no eligible route sample"
            )
        if not all(
            np.isfinite(value)
            for value in (
                avg_loss,
                avg_traj,
                avg_jepa,
                avg_reason,
                avg_route,
                *avg_route_terms.values(),
                *avg_rollout_terms.values(),
            )
        ):
            raise ValueError(
                "non-finite training metrics at epoch "
                f"{epoch}: loss={avg_loss} traj={avg_traj} "
                f"route={avg_route} jepa={avg_jepa} reason={avg_reason}"
            )

        validation = _evaluate_open_loop(
            model,
            validation_loader,
            device,
            training_policy=training_policy,
            include_rollout_selector_records=selector_enabled,
        )
        validation_digest = validation["sample_uid_digest"]
        if expected_validation_digest is None:
            expected_validation_digest = validation_digest
            validation_sample_count = validation["sample_count"]
        elif (
            validation_digest != expected_validation_digest
            or validation["sample_count"] != validation_sample_count
        ):
            raise ValueError(
                "internal validation sample set changed between epochs: "
                f"expected_digest={expected_validation_digest} "
                f"actual_digest={validation_digest} "
                f"expected_count={validation_sample_count} "
                f"actual_count={validation['sample_count']}"
            )

        checkpoint_selection = None
        validation_aggregates = None
        validation_aggregate_summary = None
        if selector_enabled:
            from evaluation.checkpoint_selection import (
                aggregate_validation_records,
                build_selector_calibration_report,
                freeze_component_availability,
                score_checkpoint,
                validate_frozen_availability,
            )

            validation_aggregates = aggregate_validation_records(
                validation["rollout_selector_records"]
            )
            validation["ade"] = float(
                validation_aggregates["metrics"]["ade_3s_m"][
                    "scene_balanced"
                ]
            )
            validation["fde"] = float(
                validation_aggregates["metrics"]["fde_3s_m"][
                    "scene_balanced"
                ]
            )
            observed_availability = freeze_component_availability(
                validation_aggregates
            )
            if selector_availability is None:
                raise RuntimeError(
                    "checkpoint selector availability was not frozen "
                    "before training"
                )
            validate_frozen_availability(
                selector_availability,
                observed_availability,
            )
            checkpoint_selection = score_checkpoint(
                validation_aggregates,
                selector_availability,
            )
            checkpoint_selection["calibration_report"] = (
                build_selector_calibration_report([
                    *[
                        entry["checkpoint_selection"]
                        for entry in metric_history
                        if entry.get("checkpoint_selection") is not None
                    ],
                    checkpoint_selection,
                ])
            )
            validation_aggregate_summary = {
                "sample_count": validation_aggregates["sample_count"],
                "scene_count": validation_aggregates["scene_count"],
                "metrics": {
                    metric_name: {
                        key: value
                        for key, value in aggregate.items()
                        if key != "scene_means"
                    }
                    for metric_name, aggregate in validation_aggregates[
                        "metrics"
                    ].items()
                },
            }
            score_improved, trajectory_improved = (
                _dual_best_improvements(
                    checkpoint_selection,
                    best_selection=(
                        best_checkpoint["selection"]
                        if best_checkpoint is not None
                        else None
                    ),
                    best_trajectory_selection=(
                        best_trajectory_checkpoint["selection"]
                        if best_trajectory_checkpoint is not None
                        else None
                    ),
                    min_delta=float(
                        checkpoint_selection_config["min_delta"]
                    ),
                )
            )
            scheduler_metric = float(checkpoint_selection["score"])
        else:
            score_improved = (
                best_checkpoint is None
                or metric_pair_is_better(
                    validation["ade"],
                    validation["fde"],
                    float(best_checkpoint["ade"]),
                    float(best_checkpoint["fde"]),
                )
            )
            trajectory_improved = score_improved
            scheduler_metric = float(validation["ade"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_compute_wall_seconds = (
            time.perf_counter() - epoch_compute_started
        )
        throughput = {
            "train_wall_seconds": training_wall_seconds,
            "epoch_compute_wall_seconds": epoch_compute_wall_seconds,
            "sample_count": epoch_training_sample_count,
            "samples_per_second": (
                epoch_training_sample_count / training_wall_seconds
            ),
            "optimizer_step_count": epoch_optimizer_steps,
            "optimizer_steps_per_second": (
                epoch_optimizer_steps / training_wall_seconds
            ),
        }
        patience_improved = score_improved or trajectory_improved
        next_bad_epochs = 0 if patience_improved else bad_epochs + 1
        key = checkpoint_key(run_id, epoch)
        checkpoint_uri = f"s3://{checkpoint_bucket}/{key}"
        candidate_record = {
            "epoch": epoch,
            "ade": validation["ade"],
            "fde": validation["fde"],
            "uri": checkpoint_uri,
            "sha256": None,
            "metric_contract": validation["metric_contract"],
        }
        if checkpoint_selection is not None:
            candidate_record["selection"] = checkpoint_selection
        candidate_best = (
            candidate_record
            if score_improved
            else dict(best_checkpoint)
        )
        candidate_best_trajectory = (
            candidate_record
            if trajectory_improved
            else dict(best_trajectory_checkpoint)
        )
        history_entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "trajectory_loss": avg_traj,
            "jepa_loss": avg_jepa,
            "reasoning_loss": avg_reason,
            "route_loss": avg_route,
            "route_loss_terms": avg_route_terms,
            "route_loss_counts": route_epoch_counts,
            "route_target_compliance_sum": (
                route_target_compliance_sum
            ),
            "rollout_aligned_loss_terms": avg_rollout_terms,
            "rollout_aligned_loss_counts": rollout_term_counts,
            "navigation_exposure": navigation_exposure_metadata,
            "val_ade": validation["ade"],
            "val_fde": validation["fde"],
            "val_horizons": validation["horizons"],
            "validation_metric_contract": validation["metric_contract"],
            "validation_sample_count": validation["sample_count"],
            "validation_sample_uid_digest": validation_digest,
            "checkpoint_selection": checkpoint_selection,
            "validation_aggregates": validation_aggregate_summary,
            "throughput": throughput,
            "improved": patience_improved,
            "score_improved": score_improved,
            "trajectory_improved": trajectory_improved,
        }
        metric_history.append(history_entry)
        losses_per_epoch.append(avg_loss)
        scheduler.step(scheduler_metric)
        current_lr = float(optimizer.param_groups[0]["lr"])
        selector_mlflow_metrics = {}
        if (
            checkpoint_selection is not None
            and validation_aggregates is not None
        ):
            selector_mlflow_metrics = {
                "val/checkpoint_composite_score": float(
                    checkpoint_selection["score"]
                ),
                "val/ade_3s_scene_balanced_logged_xy": validation["ade"],
                "val/fde_3s_scene_balanced_logged_xy": validation["fde"],
                "selection/score": float(
                    checkpoint_selection["score"]
                ),
                **{
                    f"val/checkpoint_component_{name}": float(value)
                    for name, value in checkpoint_selection[
                        "components"
                    ].items()
                },
                **{
                    f"selection/component/{name}": float(value)
                    for name, value in checkpoint_selection[
                        "components"
                    ].items()
                },
                **{
                    f"selection/effective_weight/{name}": float(value)
                    for name, value in checkpoint_selection[
                        "effective_weights"
                    ].items()
                },
                "selection/bad_epochs": float(next_bad_epochs),
                "selection/score_improved": float(score_improved),
                "selection/trajectory_improved": float(
                    trajectory_improved
                ),
            }
            calibration_report = checkpoint_selection[
                "calibration_report"
            ]
            selector_mlflow_metrics[
                "selection/calibration/saturated_component_count"
            ] = float(len(
                calibration_report[
                    "almost_always_saturated_components"
                ]
            ))
            sensitivity = calibration_report["weight_sensitivity"]
            selector_mlflow_metrics[
                "selection/calibration/top1_stability"
            ] = float(np.mean([
                float(item["top_checkpoint_unchanged"])
                for item in sensitivity
            ]))
            rank_correlations = [
                float(item["spearman_rank_correlation"])
                for item in sensitivity
                if item["spearman_rank_correlation"] is not None
            ]
            if rank_correlations:
                selector_mlflow_metrics[
                    "selection/calibration/min_rank_correlation"
                ] = min(rank_correlations)
            for metric_name, aggregate in validation_aggregates[
                "metrics"
            ].items():
                for aggregate_name in ("natural", "scene_balanced"):
                    value = aggregate[aggregate_name]
                    if value is not None:
                        selector_mlflow_metrics[
                            f"val/{metric_name}_{aggregate_name}"
                        ] = float(value)
                        selector_mlflow_metrics[
                            f"validation/{aggregate_name}/{metric_name}"
                        ] = float(value)
                distribution = aggregate["scene_distribution"]
                for statistic in ("count", "mean", "p50", "p90"):
                    value = distribution[statistic]
                    if value is not None:
                        selector_mlflow_metrics[
                            f"val/{metric_name}_scene_{statistic}"
                        ] = float(value)
                        selector_mlflow_metrics[
                            "validation/scene_distribution/"
                            f"{metric_name}/{statistic}"
                        ] = float(value)
                selector_mlflow_metrics[
                    f"val/{metric_name}_eligible_samples"
                ] = float(aggregate["eligible_sample_count"])
                selector_mlflow_metrics[
                    f"val/{metric_name}_eligible_scenes"
                ] = float(aggregate["eligible_scene_count"])
                selector_mlflow_metrics[
                    f"validation/coverage/{metric_name}/eligible_samples"
                ] = float(aggregate["eligible_sample_count"])
                selector_mlflow_metrics[
                    f"validation/coverage/{metric_name}/eligible_scenes"
                ] = float(aggregate["eligible_scene_count"])

        # The same MLflow run is reopened for each epoch. A failed metric write
        # aborts before checkpointing, so resume cannot silently skip a metric.
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(
                {
                    "train/loss": avg_loss,
                    "train/trajectory_loss": avg_traj,
                    "train/jepa_loss": avg_jepa,
                    "train/reasoning_loss": avg_reason,
                    "train/route_loss": avg_route,
                    "train/weighted_route_loss": (
                        route_consistency_weight * avg_route
                    ),
                    "train/route_corridor_loss": (
                        avg_route_terms["corridor"]
                    ),
                    "train/route_branch_loss": (
                        avg_route_terms["branch"]
                    ),
                    "train/route_destination_loss": (
                        avg_route_terms["destination"]
                    ),
                    "train/route_heading_loss": (
                        avg_route_terms["heading"]
                    ),
                    "train/route_candidate_count": (
                        route_epoch_counts["candidate_count"]
                    ),
                    "train/route_eligible_count": (
                        route_epoch_counts["eligible_count"]
                    ),
                    "train/route_compliance_rejected_count": (
                        route_epoch_counts[
                            "compliance_rejected_count"
                        ]
                    ),
                    "train/route_gradient_ratio": (
                        gradient_evidence[
                            "route_loss_gradient_budget"
                        ]["route_to_trajectory_ratio"]
                        if gradient_evidence[
                            "route_loss_gradient_budget"
                        ] is not None
                        else -1.0
                    ),
                    "train/rollout_loss": (
                        avg_rollout_terms["rollout"]
                    ),
                    "train/loss_action": avg_traj,
                    "train/loss_rollout": (
                        avg_rollout_terms["rollout"]
                    ),
                    "train/rollout_path_loss": (
                        avg_rollout_terms["path"]
                    ),
                    "train/loss_rollout_path": (
                        avg_rollout_terms["path"]
                    ),
                    "train/rollout_final_loss": (
                        avg_rollout_terms["final"]
                    ),
                    "train/loss_rollout_final": (
                        avg_rollout_terms["final"]
                    ),
                    "train/constraint_loss": (
                        avg_rollout_terms["constraint"]
                    ),
                    "train/loss_constraint": (
                        avg_rollout_terms["constraint"]
                    ),
                    "train/comfort_loss": (
                        avg_rollout_terms["comfort"]
                    ),
                    "train/loss_comfort": (
                        avg_rollout_terms["comfort"]
                    ),
                    "train/loss_comfort_jerk": (
                        avg_rollout_terms["jerk"]
                    ),
                    "train/loss_comfort_lateral_acceleration": (
                        avg_rollout_terms["lateral_acceleration"]
                    ),
                    "train/loss_comfort_lateral": (
                        avg_rollout_terms["lateral_acceleration"]
                    ),
                    "train/map_loss": avg_rollout_terms["map"],
                    "train/loss_map": avg_rollout_terms["map"],
                    "train/route_relative_loss": (
                        avg_rollout_terms["route"]
                    ),
                    "train/loss_route_relative": (
                        avg_rollout_terms["route"]
                    ),
                    "train/loss_map_route": (
                        avg_rollout_terms["route"]
                    ),
                    "train/drivable_relative_loss": (
                        avg_rollout_terms["drivable"]
                    ),
                    "train/loss_drivable_relative": (
                        avg_rollout_terms["drivable"]
                    ),
                    "train/loss_map_drivable": (
                        avg_rollout_terms["drivable"]
                    ),
                    "train/loss_total": avg_loss,
                    "train/lr": current_lr,
                    "train/throughput/train_wall_seconds": (
                        throughput["train_wall_seconds"]
                    ),
                    "train/throughput/epoch_compute_wall_seconds": (
                        throughput["epoch_compute_wall_seconds"]
                    ),
                    "train/throughput/sample_count": float(
                        throughput["sample_count"]
                    ),
                    "train/throughput/samples_per_second": (
                        throughput["samples_per_second"]
                    ),
                    "train/throughput/optimizer_step_count": float(
                        throughput["optimizer_step_count"]
                    ),
                    "train/throughput/optimizer_steps_per_second": (
                        throughput["optimizer_steps_per_second"]
                    ),
                    "val/ade": validation["ade"],
                    "val/fde": validation["fde"],
                    **{
                        f"val/control_rollout_ade_{label}": values["ade"]
                        for label, values in validation[
                            "horizons"
                        ].items()
                    },
                    **{
                        f"val/control_rollout_fde_{label}": values["fde"]
                        for label, values in validation[
                            "horizons"
                        ].items()
                    },
                    **selector_mlflow_metrics,
                },
                step=epoch,
            )

        checkpoint_path = f"/tmp/train/epoch-{epoch:04d}.pt"
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng_state": capture_rng_state(),
                "epoch": epoch,
                "config": checkpoint_config,
                "training_state": {
                    "run_id": run_id,
                    "best": candidate_best,
                    "best_trajectory": candidate_best_trajectory,
                    "bad_epochs": next_bad_epochs,
                    "metric_history": metric_history,
                    "validation_sample_uid_digest": (
                        expected_validation_digest
                    ),
                    "validation_sample_count": validation_sample_count,
                    "validation_split": validation_split_contract,
                    "navigation_exposure": (
                        navigation_exposure_metadata
                    ),
                    "checkpoint_selector_availability": (
                        selector_availability
                    ),
                    "current_checkpoint_uri": checkpoint_uri,
                    "early_stopping_patience": (
                        early_stopping_patience
                    ),
                    "resume_policy_transition": resume_policy_transition,
                },
                "data_fingerprint": data_fingerprint,
            },
            checkpoint_path,
        )
        uploaded = upload_immutable_checkpoint(
            s3_client,
            bucket=checkpoint_bucket,
            key=key,
            path=checkpoint_path,
        )
        checkpoint_info = {
            "epoch": epoch,
            "ade": validation["ade"],
            "fde": validation["fde"],
            "uri": uploaded["uri"],
            "sha256": uploaded["sha256"],
            "size": uploaded["size"],
            "metric_contract": validation["metric_contract"],
        }
        if checkpoint_selection is not None:
            checkpoint_info["selection"] = checkpoint_selection
        history_entry["checkpoint_uri"] = uploaded["uri"]
        history_entry["checkpoint_sha256"] = uploaded["sha256"]
        previous_best_local_path = best_local_path
        if score_improved:
            best_checkpoint = checkpoint_info
            best_local_path = checkpoint_path
            update_best_pointer(
                s3_client,
                bucket=checkpoint_bucket,
                run_id=run_id,
                epoch=epoch,
                checkpoint_uri=uploaded["uri"],
                checkpoint_sha256=uploaded["sha256"],
                ade=validation["ade"],
                fde=validation["fde"],
                selection=checkpoint_selection,
                metric_contract=validation["metric_contract"],
            )
            if (
                previous_best_local_path
                and previous_best_local_path != best_local_path
                and os.path.abspath(previous_best_local_path).startswith(
                    "/tmp/train/"
                )
            ):
                os.remove(previous_best_local_path)
        if trajectory_improved:
            best_trajectory_checkpoint = checkpoint_info
            update_best_pointer(
                s3_client,
                bucket=checkpoint_bucket,
                run_id=run_id,
                role="best_trajectory",
                epoch=epoch,
                checkpoint_uri=uploaded["uri"],
                checkpoint_sha256=uploaded["sha256"],
                ade=validation["ade"],
                fde=validation["fde"],
                selection=checkpoint_selection,
                metric_contract=validation["metric_contract"],
            )
        if not score_improved:
            os.remove(checkpoint_path)
        bad_epochs = next_bad_epochs
        final_checkpoint = checkpoint_info

        selector_summary = (
            "composite_score="
            f"{float(checkpoint_selection['score']):.6f} "
            if checkpoint_selection is not None
            else ""
        )
        print(
            f"  Epoch {epoch}/{epochs} loss={avg_loss:.4f} "
            f"traj={avg_traj:.4f} route={avg_route:.4f} "
            f"jepa={avg_jepa:.4f} "
            f"reason={avg_reason:.4f} val_ADE={validation['ade']:.4f} "
            f"val_FDE={validation['fde']:.4f} "
            f"score_improved={score_improved} "
            f"trajectory_improved={trajectory_improved} "
            f"samples_per_second={throughput['samples_per_second']:.3f} "
            "optimizer_steps_per_second="
            f"{throughput['optimizer_steps_per_second']:.3f} "
            f"{selector_summary}"
            f"bad_epochs={bad_epochs} checkpoint={uploaded['uri']}"
        )
        if bad_epochs >= early_stopping_patience:
            stopped_early = True
            print(
                f"Early stopping after epoch {epoch}: no validation "
                f"improvement for {bad_epochs} epochs"
            )
            break

    # The planner moved if ANY of its parameters moved. Report the one that
    # moved most, so a passing run still names a parameter and a failing run
    # says how many were checked rather than pointing at one arbitrary name.
    optimizer_probe_deltas = [
        (
            name,
            float((parameter.detach() - before).norm().item()),
        )
        for name, parameter, before in optimizer_probe
    ]
    optimizer_probe_name, optimizer_parameter_delta_norm = max(
        optimizer_probe_deltas, key=lambda item: item[1]
    )
    if (
        not terminal_resume
        and (
            optimizer_step_count <= 0
            or optimizer_parameter_delta_norm <= 0.0
        )
    ):
        raise RuntimeError(
            "optimizer produced no parameter update: "
            f"steps={optimizer_step_count} "
            f"parameters_checked={len(optimizer_probe_deltas)} "
            f"largest_delta_parameter={optimizer_probe_name} "
            f"delta_norm={optimizer_parameter_delta_norm}"
        )
    first_gradient_evidence = gradient_evidence["first_step"] or {}
    if (
        not terminal_resume
        and (
            first_gradient_evidence.get(
                "navigation_fusion", {}
            ).get("norm", 0.0)
            <= 0.0
        )
    ):
        raise RuntimeError("Reactive navigation fusion received no gradient")
    if (
        not terminal_resume
        and gradient_evidence[
            "navigation_encoder_first_nonzero_step"
        ] is None
    ):
        raise RuntimeError("Reactive NavigationEncoder received no gradient")
    if (
        not terminal_resume
        and dataset == Dataset.KITSCENES
        and route_valid_sample_count <= 0
    ):
        raise RuntimeError(
            "KITScenes training saw no valid route-conditioned sample"
        )
    if (
        not terminal_resume
        and dataset == Dataset.KITSCENES
        and enable_route_conditioning
        and gradient_evidence["route_input_first_nonzero_step"] is None
    ):
        raise RuntimeError("Reactive planner received no route input gradient")
    if (
        not terminal_resume
        and enable_route_consistency
        and gradient_evidence["route_loss_gradient_budget"] is None
    ):
        raise RuntimeError(
            "route consistency produced no planner gradient budget evidence"
        )
    if (
        not terminal_resume
        and objective_v2
        and gradient_evidence["objective_term_gradient_norms"] is None
    ):
        raise RuntimeError(
            "rollout-aligned objective produced no term gradient evidence"
        )

    if (
        best_checkpoint is None
        or best_trajectory_checkpoint is None
        or final_checkpoint is None
    ):
        raise RuntimeError(
            "training completed without best/best-trajectory/final checkpoints"
        )

    if best_local_path is None:
        from urllib.parse import urlparse

        parsed = urlparse(best_checkpoint["uri"])
        best_local_path = (
            f"/tmp/train/best-epoch-{int(best_checkpoint['epoch']):04d}.pt"
        )
        s3_client.download_file(
            parsed.netloc,
            parsed.path.lstrip("/"),
            best_local_path,
        )
    best_digest = sha256_file(best_local_path)
    if best_digest != best_checkpoint["sha256"]:
        raise RuntimeError(
            "local best checkpoint differs from its immutable S3 object: "
            f"expected={best_checkpoint['sha256']} actual={best_digest}"
        )

    throughput_history = [
        dict(entry["throughput"])
        for entry in metric_history
        if isinstance(entry.get("throughput"), dict)
    ]
    throughput_train_wall_seconds = sum(
        float(item["train_wall_seconds"])
        for item in throughput_history
    )
    throughput_sample_count = sum(
        int(item["sample_count"])
        for item in throughput_history
    )
    throughput_optimizer_step_count = sum(
        int(item["optimizer_step_count"])
        for item in throughput_history
    )
    throughput_summary = {
        "epoch_count": len(throughput_history),
        "train_wall_seconds": throughput_train_wall_seconds,
        "epoch_compute_wall_seconds": sum(
            float(item["epoch_compute_wall_seconds"])
            for item in throughput_history
        ),
        "sample_count": throughput_sample_count,
        "samples_per_second": (
            throughput_sample_count / throughput_train_wall_seconds
            if throughput_train_wall_seconds > 0.0
            else None
        ),
        "optimizer_step_count": throughput_optimizer_step_count,
        "optimizer_steps_per_second": (
            throughput_optimizer_step_count
            / throughput_train_wall_seconds
            if throughput_train_wall_seconds > 0.0
            else None
        ),
        "per_epoch": throughput_history,
    }

    meta = {
        "data": {
            "dataset": dataset.value,
            "dataset_version": dataset_version,
            "data_fingerprint": data_fingerprint,
            "packed_partitions": len(manifests),
            "non_empty_partitions": len(shard_dirs),
            "training_partitions": len(training_shard_dirs),
            "empty_partitions": skipped_empty,
            "coverage": data_coverage,
            "navigation_quality": (
                {
                    "audit_sha256": navigation_quality_audit_sha256,
                    "schema_version": navigation_quality_report[
                        "schema_version"
                    ],
                    "policy": navigation_quality_report["policy"],
                    "accepted_scene_count": navigation_quality_report[
                        "accepted_scene_count"
                    ],
                    "excluded_scene_count": navigation_quality_report[
                        "excluded_scene_count"
                    ],
                    "accepted_partition_ids": navigation_quality_report[
                        "accepted_partition_ids"
                    ],
                    "excluded_partition_ids": navigation_quality_report[
                        "excluded_partition_ids"
                    ],
                }
                if navigation_quality_report is not None
                else None
            ),
            "navigation_exposure": navigation_exposure_metadata,
        },
        "model": {
            "backbone": bb,
            "fusion_mode": fm,
            "embed_dim": 256,
            "num_views": num_views,
            "view_fusion_kwargs": view_fusion_kwargs,
            "navigation_geometry_id": navigation_geometry_id,
            "map_context_channels": map_context_channels,
            "route_channels": route_channels,
            "enable_route_conditioning": enable_route_conditioning,
        },
        "training": {
            "epochs": epochs,
            "epochs_completed": int(final_checkpoint["epoch"]),
            "stopped_early": stopped_early,
            "early_stopping_patience": early_stopping_patience,
            "resume_policy_transition": resume_policy_transition,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "num_workers": num_workers,
            "lr": lr,
            "training_seed": training_seed,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "amp": amp,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "trajectory_training_policy": training_policy.metadata(),
            "training_objective_version": training_objective_version,
            "junction_sampling": {
                "enabled": enable_junction_sampling,
                "policy": (
                    navigation_repeat_policy.metadata()
                    if navigation_repeat_policy is not None
                    else None
                ),
                "exposure": navigation_exposure_metadata,
            },
            "route_consistency": route_consistency_config,
            "rollout_aligned_loss": rollout_aligned_config,
            "checkpoint_selection": {
                **checkpoint_selection_config,
                "availability": selector_availability,
                "best": best_checkpoint.get("selection"),
                "best_trajectory": (
                    best_trajectory_checkpoint.get("selection")
                ),
            },
            "reconstruction_audit": reconstruction_audit_contract,
            "final_loss": losses_per_epoch[-1],
            "losses_per_epoch": losses_per_epoch,
            "val_fraction": val_fraction,
            "validation_scope": validation_scope,
            "validation_split": validation_split_contract,
            "metric_history": metric_history,
            "optimizer_evidence": {
                "step_count": optimizer_step_count,
                "parameters_checked": len(optimizer_probe_deltas),
                # The planner parameter that moved most, not an arbitrary one:
                # the probe now watches every trainable planner parameter.
                "probe_parameter": optimizer_probe_name,
                "probe_parameter_delta_norm": (
                    optimizer_parameter_delta_norm
                ),
            },
            "throughput": throughput_summary,
            "gradient_evidence": gradient_evidence,
            "route_conditioning_evidence": {
                "enabled": enable_route_conditioning,
                "valid_sample_exposures": route_valid_sample_count,
                "conditioned_valid_sample_exposures": (
                    route_valid_sample_count
                    if enable_route_conditioning
                    else 0
                ),
                "sample_exposures": route_sample_count,
                "valid_fraction": (
                    route_valid_sample_count / route_sample_count
                    if route_sample_count
                    else 0.0
                ),
            },
        },
        "validation": {
            "sample_count": validation_sample_count,
            "sample_uid_digest": expected_validation_digest,
            "split": "internal_scene_holdout",
            "evaluation_steps": 30,
            "prediction_steps": AUTO_E2E_TIMESTEPS,
            "metric_contract": final_checkpoint["metric_contract"],
            **validation_split_contract,
        },
        "tracking": {
            "mlflow_experiment": "imitation-learning",
            "mlflow_run_id": run_id,
        },
        "checkpoints": {
            "best": best_checkpoint,
            "best_trajectory": best_trajectory_checkpoint,
            "final": final_checkpoint,
            "best_pointer_uri": (
                f"s3://{checkpoint_bucket}/imitation-learning/"
                f"{run_id}/best.json"
            ),
            "best_trajectory_pointer_uri": (
                f"s3://{checkpoint_bucket}/imitation-learning/"
                f"{run_id}/best-trajectory.json"
            ),
        },
        "context": {
            "flyte_execution_id": train_execution_id,
            "docker_image": TRAINING_IMAGE,
        },
    }
    meta_path = "/tmp/train/metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({
            "pipeline": "imitation-learning",
            "backbone": bb,
            "fusion": fm,
            "best_checkpoint_sha256": best_checkpoint["sha256"],
            "best_trajectory_checkpoint_sha256": (
                best_trajectory_checkpoint["sha256"]
            ),
            "final_checkpoint_sha256": final_checkpoint["sha256"],
            "validation_sample_uid_digest": expected_validation_digest,
            "validation_strategy": training_policy.validation_strategy,
            "validation_split_id": training_policy.validation_split_id,
            "validation_group_uid_digest": (
                validation_group_digest or "hash_buckets"
            ),
            "checkpoint_selector_policy": (
                checkpoint_selection_config["policy_version"]
            ),
            "validation_metric_version": str(
                final_checkpoint["metric_contract"]["version"]
            ),
            "validation_metric_horizon_seconds": "3.0",
            "validation_metric_horizon_steps": "30",
            "validation_metric_target_source": str(
                final_checkpoint["metric_contract"]["target_source"]
            ),
            "validation_metric_aggregation": str(
                final_checkpoint["metric_contract"]["aggregation"]
            ),
            "best_checkpoint_ade_3s_m": str(best_checkpoint["ade"]),
            "best_checkpoint_fde_3s_m": str(best_checkpoint["fde"]),
            "best_trajectory_checkpoint_ade_3s_m": str(
                best_trajectory_checkpoint["ade"]
            ),
            "best_trajectory_checkpoint_fde_3s_m": str(
                best_trajectory_checkpoint["fde"]
            ),
            "best_checkpoint_composite_score": (
                str(best_checkpoint["selection"]["score"])
                if selector_enabled
                else "not_applicable"
            ),
            "best_trajectory_checkpoint_utility": (
                str(
                    best_trajectory_checkpoint["selection"]["components"][
                        "trajectory"
                    ]
                )
                if selector_enabled
                else "not_applicable"
            ),
            "resume_policy_transition": (
                resume_policy_transition["policy_version"]
                if resume_policy_transition is not None
                else "none"
            ),
            "ctx/train_execution_id": train_execution_id,
            "ctx/train_docker_image": TRAINING_IMAGE,
        })
        mlflow.log_artifact(meta_path, artifact_path="training")

    return TrainOutput(
        checkpoint=FlyteFile(best_local_path),
        metadata=FlyteFile(meta_path),
    )


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
    il_meta = json.load(open(il_metadata.download()))
    ctx = current_context()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Offline RL (AWR, frozen prior): epochs={epochs} beta={beta}")

    # Load IL model
    from model_components.auto_e2e import AutoE2E
    from data_parsing.pre_extracted import make_pre_extracted_loader

    import copy

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )
    config = ckpt["config"]
    from training.dataset_policy import (
        adapt_egomotion_history,
        training_policy_from_config,
    )

    training_policy = training_policy_from_config(
        config,
        dataset.value,
    )
    if training_policy.validation_strategy != "hash_buckets":
        raise ValueError(
            "offline RL does not yet support an exact KITScenes train/holdout "
            "partition; refusing to train on one shard or leak validation scenes"
        )
    shard_dir = _select_shard_dir(shards, dataset)
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
            ego_hist = adapt_egomotion_history(
                batch["egomotion_history"].to(device),
                training_policy,
            )
            vis_hist = batch["visual_history"].to(device)
            target = batch["trajectory_target"].to(device)
            map_context = batch["map_context"].to(device)
            route_mask = batch["route_mask"].to(device)
            map_valid = batch["map_valid"].to(device)
            route_valid = batch["route_valid"].to(device)

            optimizer.zero_grad()
            # Offline RL regresses only the trajectory; run mode="infer" so the
            # forward returns a bare trajectory tensor even when the checkpoint
            # was trained with reasoning / world-model branches on (mode="train"
            # would return a (trajectory, aux) tuple and break the arithmetic).
            # The inference forward is still differentiable for the policy grad.
            pred = model(visual, map_context, vis_hist, ego_hist,
                         route_mask=route_mask,
                         map_valid=map_valid,
                         route_valid=route_valid,
                         projection=projection, geometry_type=geometry_type,
                         mode="infer")
            # Advantage-weighted regression against the FROZEN IL prior. advantage
            # > 0 where the trained policy is already closer to the logged action
            # than the prior; exp(beta*advantage) up-weights those samples. Using
            # the frozen prior (not the live model) makes the advantage real and
            # non-zero, and makes beta actually do something.
            with torch.no_grad():
                baseline_pred = baseline_model(
                                               visual, map_context, vis_hist, ego_hist,
                                               route_mask=route_mask,
                                               map_valid=map_valid,
                                               route_valid=route_valid,
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
def _run_evaluation(
    checkpoint,
    shards,
    train_metadata,
    dataset,
    experiment_name,
    *,
    navigation_records_output=None,
):
    """Shared open-loop evaluation + MLflow logging logic.

    Called by both evaluate_il_policy and evaluate_rl_policy. Kept as a plain
    module-level function (not a @task) so the two evaluation tasks share one
    implementation while appearing as distinct nodes in the Flyte UI.
    """
    import os
    import json
    import math
    import yaml
    import torch
    import mlflow
    from flytekit import current_context

    from model_components.auto_e2e import AutoE2E
    from data_parsing.pre_extracted import make_multi_dataset_loader

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
    saved_best = meta.get("checkpoints", {}).get("best", {})
    saved_best_digest = saved_best.get("sha256")
    if saved_best_digest and saved_best_digest != checkpoint_sha256:
        raise ValueError(
            "evaluated checkpoint differs from training's selected best: "
            f"expected={saved_best_digest} actual={checkpoint_sha256}"
        )
    ctx = current_context()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )
    config = ckpt["config"]
    from training.dataset_policy import (
        AUTO_E2E_TIMESTEPS,
        training_policy_from_config,
    )

    training_policy = training_policy_from_config(
        config,
        dataset.value,
    )
    model = AutoE2E(**_model_kwargs(config)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate on the exact HELD-OUT split this checkpoint trained around.
    # KITScenes records a frozen scene manifest; L2D/NVIDIA retain their legacy
    # deterministic group buckets.
    base_il_metadata = (
        meta.get("base_model", {}).get("il_metadata", {})
    )
    training_metadata = meta.get(
        "training",
        base_il_metadata.get("training", {}),
    )
    validation_metadata = meta.get(
        "validation",
        base_il_metadata.get("validation", {}),
    )
    val_fraction = float(
        training_metadata.get("val_fraction", 0.0) or 0.0
    )
    fixed_validation_groups = None
    if training_policy.validation_strategy != "hash_buckets":
        checkpoint_validation = config.get("validation_split")
        if not isinstance(checkpoint_validation, dict):
            raise ValueError(
                "checkpoint has no exact validation_split contract"
            )
        mismatched_keys = [
            key
            for key, value in checkpoint_validation.items()
            if validation_metadata.get(key) != value
        ]
        if mismatched_keys:
            raise ValueError(
                "training metadata differs from the checkpoint validation "
                f"contract: {sorted(mismatched_keys)}"
            )
        strategy = validation_metadata.get("strategy")
        split_id = validation_metadata.get("split_id")
        if strategy != training_policy.validation_strategy:
            raise ValueError(
                "training metadata has no exact validation-group contract: "
                f"expected={training_policy.validation_strategy!r} "
                f"actual={strategy!r}"
            )
        if split_id != training_policy.validation_split_id:
            raise ValueError(
                "validation split ID differs from the checkpoint policy: "
                f"expected={training_policy.validation_split_id!r} "
                f"actual={split_id!r}"
            )
        raw_groups = checkpoint_validation.get("validation_group_uids")
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError(
                "exact validation metadata has no validation_group_uids"
            )
        fixed_validation_groups = tuple(str(uid) for uid in raw_groups)
        if (
            list(fixed_validation_groups)
            != sorted(set(fixed_validation_groups))
            or any(not uid for uid in fixed_validation_groups)
        ):
            raise ValueError(
                "validation_group_uids must be sorted, unique, and non-empty"
            )
        from Platform.pipelines.training_checkpoint import stable_digest

        actual_group_digest = stable_digest(
            list(fixed_validation_groups)
        )
        expected_group_digest = validation_metadata.get(
            "validation_group_uid_digest"
        )
        if actual_group_digest != expected_group_digest:
            raise ValueError(
                "validation group manifest digest mismatch: "
                f"expected={expected_group_digest} "
                f"actual={actual_group_digest}"
            )
        _validate_evaluation_shard_provenance(
            shard_dirs,
            dataset_name=dataset.value,
            source_revision=str(
                checkpoint_validation.get("source_revision", "")
            ),
            dataset_version=str(
                checkpoint_validation.get("dataset_version", "")
            ),
            contract_digest=str(
                checkpoint_validation.get("packed_contract_digest", "")
            ),
        )

    eval_split = "val" if val_fraction > 0.0 else "all"
    loader = make_multi_dataset_loader(
        shard_dirs,
        batch_size=8,
        num_workers=4,
        shuffle=0,
        pin_memory=(device.type == "cuda"),
        split=eval_split,
        val_fraction=val_fraction,
        validation_group_uids=fixed_validation_groups,
        max_active_loaders=1,
        prefetch_factor=1,
        decode_future_frames=False,
    )
    print(f"Eval split={eval_split} (strategy={training_policy.validation_strategy}, "
          f"val_fraction={val_fraction}, {len(shard_dirs)} partitions) — "
          f"{'held-out generalization' if eval_split == 'val' else 'in-sample'}")
    navigation_geometry = None
    if dataset == Dataset.KITSCENES:
        from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY

        if config.get("navigation_geometry_id") != (
            DEFAULT_NAVIGATION_GEOMETRY.geometry_id
        ):
            raise ValueError(
                "KITScenes checkpoint navigation geometry differs from "
                "the route evaluation contract"
            )
        navigation_geometry = DEFAULT_NAVIGATION_GEOMETRY
    evaluation = _evaluate_open_loop(
        model,
        loader,
        device,
        training_policy=training_policy,
        navigation_geometry=navigation_geometry,
        report_intervention=True,
        route_swap_counterfactual=(navigation_geometry is not None),
        include_navigation_records=(
            navigation_records_output is not None
        ),
        include_rollout_selector_records=(
            navigation_geometry is not None
        ),
    )
    expected_digest = validation_metadata.get("sample_uid_digest")
    if expected_digest and evaluation["sample_uid_digest"] != expected_digest:
        raise ValueError(
            "standalone evaluation used a different internal validation set: "
            f"expected={expected_digest} "
            f"actual={evaluation['sample_uid_digest']}"
        )
    avg_ade = evaluation["ade"]
    avg_fde = evaluation["fde"]
    navigation_report = evaluation.get("navigation")
    if navigation_report is not None:
        navigation_report = {
            **navigation_report,
            "checkpoint_sha256": checkpoint_sha256,
            "dataset": dataset.value,
            "dataset_version": meta.get("data", {}).get(
                "dataset_version",
                "unknown",
            ),
            "enable_route_conditioning": bool(
                config.get("enable_route_conditioning", True)
            ),
            "navigation_geometry_id": config.get(
                "navigation_geometry_id"
            ),
            "sample_uid_digest": evaluation["sample_uid_digest"],
        }
    if navigation_records_output is not None:
        navigation_records = evaluation.get("navigation_records")
        if navigation_report is None or not isinstance(
            navigation_records,
            list,
        ):
            raise ValueError(
                "navigation record export requires KITScenes evaluation"
            )
        serializable_records = []
        for record in navigation_records:
            serializable_records.append({
                key: (
                    None
                    if isinstance(value, float)
                    and not math.isfinite(value)
                    else value
                )
                for key, value in record.items()
            })
        os.makedirs(
            os.path.dirname(navigation_records_output),
            exist_ok=True,
        )
        with open(
            navigation_records_output,
            "w",
            encoding="ascii",
        ) as stream:
            json.dump(
                {
                    "schema_version": (
                        "navigation_evaluation_records_v1"
                    ),
                    "checkpoint_sha256": checkpoint_sha256,
                    "dataset": dataset.value,
                    "dataset_version": navigation_report[
                        "dataset_version"
                    ],
                    "enable_route_conditioning": navigation_report[
                        "enable_route_conditioning"
                    ],
                    "navigation_geometry_id": navigation_report[
                        "navigation_geometry_id"
                    ],
                    "sample_uid_digest": evaluation[
                        "sample_uid_digest"
                    ],
                    "training_seed": int(
                        config.get("training_seed", -1)
                    ),
                    "mlflow_run_id": str(
                        meta.get("tracking", {}).get(
                            "mlflow_run_id",
                            "",
                        )
                    ),
                    "data_fingerprint": str(
                        meta.get("data", {}).get(
                            "data_fingerprint",
                            "",
                        )
                    ),
                    "navigation_quality_audit_sha256": str(
                        config.get(
                            "navigation_quality_audit_sha256",
                            "",
                        )
                    ),
                    "validation_group_uid_digest": str(
                        validation_metadata.get(
                            "validation_group_uid_digest",
                            "",
                        )
                    ),
                    "records": serializable_records,
                },
                stream,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
    passed = avg_ade < 2.0 and avg_fde < 4.0

    # --- MLflow logging ---
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)

    model_info = meta.get("model", meta.get("base_model", {}).get("il_metadata", {}).get("model", {}))
    bb = model_info.get("backbone", "?")
    fm = model_info.get("fusion_mode", "?")
    training = meta.get("training", meta.get("base_model", {}).get("il_metadata", {}).get("training", {}))
    run_name = f"{bb}-{fm}-e{training.get('epochs','?')}"
    existing_run_id = meta.get("tracking", {}).get("mlflow_run_id")
    run_context = (
        mlflow.start_run(run_id=existing_run_id)
        if existing_run_id
        else mlflow.start_run(run_name=run_name)
    )

    with run_context as active_run:
        run_id = active_run.info.run_id
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
        params["model/trajectory_timesteps"] = AUTO_E2E_TIMESTEPS
        params["train/val_fraction"] = training.get("val_fraction", 0.0)
        params["train/validation_strategy"] = (
            training_policy.validation_strategy
        )
        params["train/validation_split_id"] = (
            training_policy.validation_split_id
        )
        params["train/validation_group_uid_digest"] = (
            validation_metadata.get(
                "validation_group_uid_digest",
                "hash_buckets",
            )
        )
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

        # IL training already owns this run's immutable params. Re-logging
        # mutable results after a resumed continuation makes MLflow reject the
        # new value. Legacy/RL evaluation creates a fresh run and logs once.
        if not existing_run_id:
            mlflow.log_params({
                k: str(v)[:500] for k, v in params.items()
            })
        mlflow.set_tags({
            "pipeline": experiment_name,
            "backbone": bb,
            "fusion": fm,
            "checkpoint_sha256": checkpoint_sha256,
            "ctx/eval_execution_id": (
                ctx.execution_id.name if ctx.execution_id else "local"
            ),
            "ctx/eval_docker_image": EVAL_IMAGE,
            "train/epochs_completed": str(
                training.get("epochs_completed", training.get("epochs", "?"))
            ),
            "train/final_loss": str(training.get("final_loss", "?")),
            "validation_metric_version": str(
                evaluation["metric_contract"]["version"]
            ),
            "validation_metric_horizon_seconds": str(
                evaluation["metric_contract"]["horizon_seconds"]
            ),
            "validation_metric_horizon_steps": str(
                evaluation["metric_contract"]["horizon_steps"]
            ),
            "validation_metric_target_source": str(
                evaluation["metric_contract"]["target_source"]
            ),
            "validation_metric_aggregation": str(
                evaluation["metric_contract"]["aggregation"]
            ),
        })

        # Eval metrics
        logged_metrics = {
            "eval/ade": avg_ade,
            "eval/fde": avg_fde,
            "eval/gate_pass": 1.0 if passed else 0.0,
        }
        if evaluation["metric_contract"]["target_source"] == "logged_xy":
            logged_metrics.update({
                "eval/ade_3s_scene_balanced_logged_xy": avg_ade,
                "eval/fde_3s_scene_balanced_logged_xy": avg_fde,
            })
        if navigation_report is not None:
            slices = navigation_report["slices"]
            counterfactual = navigation_report[
                "route_swap_counterfactual"
            ]
            navigation_metrics = {
                "eval/navigation/route_compliance": slices[
                    "route_valid"
                ]["route_point_compliance"]["mean"],
                "eval/navigation/route_outside_distance_m": slices[
                    "route_valid"
                ]["route_outside_distance_m"]["mean"],
                "eval/navigation/wrong_branch_rate": slices[
                    "junction"
                ]["wrong_branch_rate"]["mean"],
                "eval/navigation/destination_error_m": slices[
                    "overall"
                ]["destination_distance_error_m"]["mean"],
                "eval/navigation/junction_ade_m": slices[
                    "junction"
                ]["ade_m"]["mean"],
                "eval/navigation/junction_fde_m": slices[
                    "junction"
                ]["fde_m"]["mean"],
                "eval/navigation/non_junction_ade_m": slices[
                    "non_junction"
                ]["ade_m"]["mean"],
                "eval/navigation/valid_route_ade_m": slices[
                    "route_valid"
                ]["ade_m"]["mean"],
                "eval/navigation/invalid_route_ade_m": slices[
                    "route_invalid"
                ]["ade_m"]["mean"],
                "eval/navigation/valid_minus_invalid_ade_m": (
                    navigation_report["route_valid_vs_invalid_delta"][
                        "ade_m"
                    ]["mean"]
                ),
                "eval/navigation/valid_minus_invalid_fde_m": (
                    navigation_report["route_valid_vs_invalid_delta"][
                        "fde_m"
                    ]["mean"]
                ),
                "eval/navigation/route_confidence_p50": slices[
                    "overall"
                ]["route_quality"]["route_confidence"]["p50"],
                "eval/navigation/swap_endpoint_delta_m": counterfactual[
                    "endpoint_delta_m"
                ]["mean"],
                "eval/navigation/swap_compliance_drop": counterfactual[
                    "selected_compliance_drop"
                ]["mean"],
                "eval/navigation/swap_direction_consistency": (
                    counterfactual[
                        "maneuver_direction_consistent"
                    ]["mean"]
                ),
            }
            for maneuver in ("left", "right", "straight"):
                maneuver_slice = slices[f"maneuver_{maneuver}"]
                navigation_metrics[
                    f"eval/navigation/{maneuver}_ade_m"
                ] = maneuver_slice["ade_m"]["mean"]
                navigation_metrics[
                    f"eval/navigation/{maneuver}_fde_m"
                ] = maneuver_slice["fde_m"]["mean"]
            logged_metrics.update({
                key: float(value)
                for key, value in navigation_metrics.items()
                if value is not None
            })
        if evaluation.get("reasoning_intervention_delta") is not None:
            logged_metrics["eval/reasoning_intervention_delta"] = float(
                evaluation["reasoning_intervention_delta"]
            )
        mlflow.log_metrics(logged_metrics)

        # Artifacts
        os.makedirs("/tmp/eval-artifacts", exist_ok=True)
        with open("/tmp/eval-artifacts/config.yaml", "w") as f:
            yaml.dump(meta, f)
        mlflow.log_artifact("/tmp/eval-artifacts/config.yaml")
        if navigation_report is not None:
            navigation_report_path = (
                "/tmp/eval-artifacts/navigation_evaluation.json"
            )
            with open(
                navigation_report_path,
                "w",
                encoding="ascii",
            ) as stream:
                json.dump(
                    navigation_report,
                    stream,
                    allow_nan=False,
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
            mlflow.log_artifact(navigation_report_path)

        # Register immutable selected/final checkpoints. Retry of the eval task
        # reuses the same run/source pair and therefore the same versions.
        saved_checkpoints = meta.get("checkpoints", {})
        grouped: dict[str, dict] = {}
        for role in ("best", "best_trajectory", "final"):
            record = saved_checkpoints.get(role)
            if not record:
                continue
            uri = str(record["uri"])
            grouped.setdefault(
                uri,
                {"record": record, "roles": []},
            )["roles"].append(role)

        if not grouped:
            # Legacy/offline-RL path: retain one explicitly-final model source.
            mlflow.log_artifact(ckpt_path, artifact_path="model/final")
            fallback_uri = (
                f"runs:/{run_id}/model/final/{os.path.basename(ckpt_path)}"
            )
            grouped[fallback_uri] = {
                "roles": ["final"],
                "record": {
                    "epoch": int(ckpt.get("epoch", 0)),
                    "uri": fallback_uri,
                    "sha256": checkpoint_sha256,
                    "ade": avg_ade,
                    "fde": avg_fde,
                    "metric_contract": evaluation["metric_contract"],
                },
            }

        client = mlflow.tracking.MlflowClient()
        for uri, item in grouped.items():
            record = item["record"]
            version = _register_checkpoint_version(
                client,
                run_id=run_id,
                roles=item["roles"],
                epoch=int(record["epoch"]),
                checkpoint_uri=uri,
                checkpoint_sha256=str(record["sha256"]),
                ade=float(record["ade"]),
                fde=float(record["fde"]),
                metric_contract=dict(record["metric_contract"]),
                selection=record.get("selection"),
            )
            print(
                f"Registry version {version}: roles={item['roles']} "
                f"checkpoint={uri}"
            )

    print(f"Eval: ADE={avg_ade:.3f} FDE={avg_fde:.3f} Gate={'PASS' if passed else 'FAIL'}")
    if navigation_report is not None:
        route_compliance = navigation_report["slices"][
            "route_valid"
        ]["route_point_compliance"]["mean"]
        swap_delta = navigation_report["route_swap_counterfactual"][
            "endpoint_delta_m"
        ]["mean"]
        print(
            "Navigation eval: "
            f"route_compliance={route_compliance} "
            f"swap_endpoint_delta_m={swap_delta}"
        )
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
    pod_template=_large_shm_pod_template(),
)
def evaluate_navigation_records(
    checkpoint: FlyteFile,
    shards: List[FlyteDirectory],
    train_metadata: FlyteFile,
    expected_route_conditioning: bool,
) -> FlyteFile:
    """Evaluate one KITScenes checkpoint and retain paired sample records."""
    import json
    import os

    output_path = os.path.join(
        "/tmp/navigation-evaluation-records",
        (
            "conditioned.json"
            if expected_route_conditioning
            else "baseline.json"
        ),
    )
    _run_evaluation(
        checkpoint,
        shards,
        train_metadata,
        Dataset.KITSCENES,
        "imitation-learning",
        navigation_records_output=output_path,
    )
    with open(output_path, encoding="ascii") as stream:
        payload = json.load(stream)
    actual = bool(payload["enable_route_conditioning"])
    if actual != expected_route_conditioning:
        raise ValueError(
            "navigation comparison checkpoint mode differs from its role: "
            f"expected={expected_route_conditioning} actual={actual}"
        )
    return FlyteFile(output_path)


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="2", mem="4Gi"),
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
)
def compare_navigation_record_artifacts(
    conditioned_records: FlyteFile,
    baseline_records: FlyteFile,
) -> FlyteFile:
    """Validate two evaluations and publish the frozen paired research gate."""
    import json
    import os

    import mlflow

    from evaluation.navigation_metrics import compare_navigation_records

    def load_records(artifact, label):
        with open(artifact.download(), encoding="ascii") as stream:
            payload = json.load(stream)
        if (
            payload.get("schema_version")
            != "navigation_evaluation_records_v1"
        ):
            raise ValueError(
                f"{label} has an unsupported navigation record schema"
            )
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{label} has no navigation records")
        return payload

    conditioned = load_records(conditioned_records, "conditioned")
    baseline = load_records(baseline_records, "baseline")
    if not bool(conditioned.get("enable_route_conditioning")):
        raise ValueError("conditioned artifact disabled route conditioning")
    if bool(baseline.get("enable_route_conditioning")):
        raise ValueError("baseline artifact enabled route conditioning")

    identity_fields = (
        "dataset",
        "dataset_version",
        "navigation_geometry_id",
        "sample_uid_digest",
        "training_seed",
        "data_fingerprint",
        "navigation_quality_audit_sha256",
        "validation_group_uid_digest",
    )
    mismatches = [
        field
        for field in identity_fields
        if conditioned.get(field) != baseline.get(field)
    ]
    if mismatches:
        raise ValueError(
            "navigation comparison provenance differs: "
            f"{mismatches}"
        )
    for field in identity_fields:
        if conditioned.get(field) in (None, "", -1):
            raise ValueError(
                f"navigation comparison provenance is missing {field}"
            )

    report = compare_navigation_records(
        conditioned["records"],
        baseline["records"],
    )
    report["provenance"] = {
        field: conditioned[field] for field in identity_fields
    }
    report["provenance"].update({
        "conditioned_checkpoint_sha256": conditioned[
            "checkpoint_sha256"
        ],
        "baseline_checkpoint_sha256": baseline[
            "checkpoint_sha256"
        ],
        "conditioned_mlflow_run_id": conditioned["mlflow_run_id"],
        "baseline_mlflow_run_id": baseline["mlflow_run_id"],
    })

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("navigation-comparison")
    with mlflow.start_run(
        run_name=(
            "reactive-route-vs-baseline-"
            f"seed-{conditioned['training_seed']}"
        )
    ) as active_run:
        report["provenance"]["comparison_mlflow_run_id"] = (
            active_run.info.run_id
        )
        output_dir = "/tmp/navigation-comparison"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            "navigation_comparison.json",
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

        primary = report["primary_metric"]
        guardrails = report["aggregate_guardrails"]
        decision = report["decision"]
        mlflow.log_params({
            "dataset": conditioned["dataset"],
            "dataset_version": conditioned["dataset_version"],
            "navigation_geometry_id": conditioned[
                "navigation_geometry_id"
            ],
            "training_seed": conditioned["training_seed"],
            "sample_uid_digest": conditioned["sample_uid_digest"],
            "conditioned_run_id": conditioned["mlflow_run_id"],
            "baseline_run_id": baseline["mlflow_run_id"],
            "primary_metric": primary["name"],
            "primary_minimum_sample_count": primary[
                "minimum_sample_count"
            ],
            "maximum_relative_regression": guardrails[
                "maximum_relative_regression"
            ],
            "verdict": decision["verdict"],
        })
        metrics = {
            "primary/eligible_count": primary["count"],
            "primary/conditioned_mean": primary["conditioned_mean"],
            "primary/baseline_mean": primary["baseline_mean"],
            "primary/difference_mean": primary["difference_mean"],
            "primary/difference_ci95_low": (
                primary["difference_ci95"][0]
                if primary["difference_ci95"] is not None
                else None
            ),
            "primary/difference_ci95_high": (
                primary["difference_ci95"][1]
                if primary["difference_ci95"] is not None
                else None
            ),
            "guardrail/ade_relative_regression": guardrails["ade_m"][
                "relative_regression"
            ],
            "guardrail/fde_relative_regression": guardrails["fde_m"][
                "relative_regression"
            ],
            "decision/supported": (
                1.0 if decision["verdict"] == "supported" else 0.0
            ),
        }
        mlflow.log_metrics({
            key: float(value)
            for key, value in metrics.items()
            if value is not None
        })
        mlflow.log_artifact(output_path)

    return FlyteFile(output_path)


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


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(cpu="4", mem="16Gi", gpu="1"),
    environment={
        "MLFLOW_TRACKING_URI": MLFLOW_URI,
        "AUTO_E2E_EVAL_IMAGE": EVAL_IMAGE,
    },
    pod_template=_large_shm_pod_template(),
)
def evaluate_kitscenes_benchmark_checkpoint(
    checkpoint: FlyteFile,
    benchmark_shards: List[FlyteDirectory],
    benchmark_manifest: FlyteFile,
    expected_manifest_sha256: str = "",
    mlflow_run_id: str = "",
    batch_size: int = 4,
) -> KITScenesBenchmarkOutput:
    """Score one immutable checkpoint against one fixed KITScenes manifest.

    This task is intentionally independent of training and data preparation.
    It computes the released displacement metrics and emits canonical trajectory
    predictions for future authority-side safety scoring. Unreleased
    drivable-surface, collision, centerline, and MMS metrics are not estimated.
    """
    import hashlib
    import json
    import os
    import re
    from pathlib import Path

    import mlflow
    import numpy as np
    import torch
    from flytekit import current_context

    from data_parsing.pre_extracted import make_multi_dataset_loader
    from evaluation.kitscenes_benchmark import (
        EVALUATOR_VERSION,
        PROTOCOL_ID,
        compute_displacement_metrics,
        limit_egomotion_history,
        load_benchmark_manifest,
        sample_uid_digest,
        wgs84_trajectory_to_ego_xy,
    )
    from model_components.auto_e2e import AutoE2E
    from Platform.pipelines.training_checkpoint import (
        sha256_file,
        stable_digest,
    )
    from training.dataset_policy import (
        adapt_egomotion_history,
        training_policy_from_config,
    )

    if not benchmark_shards:
        raise ValueError("benchmark_shards must not be empty")
    if not 0 < batch_size <= 8:
        raise ValueError("benchmark batch_size must be between 1 and 8")
    if expected_manifest_sha256 and not re.fullmatch(
        r"[0-9a-f]{64}", expected_manifest_sha256
    ):
        raise ValueError(
            "expected_manifest_sha256 must be a lowercase SHA-256"
        )

    manifest_path = str(benchmark_manifest.download())
    manifest, manifest_sha256 = load_benchmark_manifest(manifest_path)
    if (
        expected_manifest_sha256
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError(
            "benchmark manifest digest mismatch: "
            f"expected={expected_manifest_sha256} "
            f"actual={manifest_sha256}"
        )
    if manifest.protocol_status == "official" and not expected_manifest_sha256:
        raise ValueError(
            "official benchmark evaluation requires a pinned expected "
            "manifest SHA-256"
        )
    if "map" not in manifest.input_track.lower():
        raise ValueError(
            "this AutoE2E evaluator consumes a raster map; input_track must "
            "declare map use rather than silently substituting a camera-only "
            "benchmark track"
        )

    checkpoint_uri = str(
        getattr(checkpoint, "remote_source", "") or checkpoint
    )
    checkpoint_path = str(checkpoint.download())
    checkpoint_sha256 = sha256_file(checkpoint_path)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    required_checkpoint_fields = {
        "model_state_dict",
        "config",
        "epoch",
    }
    missing_checkpoint_fields = required_checkpoint_fields - set(payload)
    if missing_checkpoint_fields:
        raise ValueError(
            "benchmark checkpoint is missing required fields: "
            f"{sorted(missing_checkpoint_fields)}"
        )
    if not isinstance(payload["config"], dict):
        raise ValueError("benchmark checkpoint config must be an object")
    config = dict(payload["config"])
    training_policy = training_policy_from_config(
        config,
        Dataset.KITSCENES.value,
    )
    epoch = int(payload["epoch"])
    if epoch <= 0:
        raise ValueError(
            f"benchmark checkpoint epoch must be positive, got {epoch}"
        )
    training_state = payload.get("training_state", {})
    if training_state is None:
        training_state = {}
    if not isinstance(training_state, dict):
        raise ValueError("checkpoint training_state must be an object")
    checkpoint_run_id = str(training_state.get("run_id", ""))
    run_id = str(mlflow_run_id or checkpoint_run_id)
    if not run_id or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError(
            "checkpoint has no valid MLflow run ID; pass mlflow_run_id "
            "explicitly for a trusted legacy checkpoint"
        )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow_client = mlflow.tracking.MlflowClient()
    mlflow_client.get_run(run_id)

    shard_dirs: list[str] = []
    shard_identities: list[dict] = []
    dataset_versions: set[str] = set()
    contract_digests: set[str] = set()
    for shard in benchmark_shards:
        shard_uri = str(
            getattr(shard, "remote_source", "") or shard
        )
        shard_dir = str(shard.download())
        packed_manifest_path = Path(shard_dir) / "manifest.json"
        if not packed_manifest_path.is_file():
            raise FileNotFoundError(
                "packed benchmark manifest is missing: "
                f"{packed_manifest_path}"
            )
        try:
            packed_manifest_bytes = packed_manifest_path.read_bytes()
            packed_manifest = json.loads(packed_manifest_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid packed benchmark manifest {packed_manifest_path}"
            ) from error
        if not isinstance(packed_manifest, dict):
            raise ValueError(
                f"packed benchmark manifest is not an object: {shard_dir}"
            )
        if packed_manifest.get("dataset") != Dataset.KITSCENES.value:
            raise ValueError(
                "benchmark shard belongs to another dataset: "
                f"{packed_manifest.get('dataset')!r}"
            )
        if packed_manifest.get("source_revision") != manifest.dataset_revision:
            raise ValueError(
                "benchmark shard source revision differs from the fixed "
                f"manifest: shard={packed_manifest.get('source_revision')!r} "
                f"manifest={manifest.dataset_revision!r}"
            )
        if int(packed_manifest.get("hz", 0)) != manifest.frequency_hz:
            raise ValueError(
                "benchmark shard frequency differs from the protocol: "
                f"{packed_manifest.get('hz')!r}"
            )
        dataset_version = str(
            packed_manifest.get("dataset_version", "")
        )
        if not dataset_version:
            raise ValueError(
                f"benchmark shard has no dataset_version: {shard_dir}"
            )
        dataset_versions.add(dataset_version)
        contracts = packed_manifest.get("contracts")
        contract_digests.add(stable_digest(contracts))
        total_samples = int(packed_manifest.get("total_samples", 0))
        if total_samples < 0:
            raise ValueError(
                f"benchmark shard has negative sample count: {shard_dir}"
            )
        shard_identities.append({
            "contracts": contracts,
            "dataset": packed_manifest.get("dataset"),
            "dataset_version": dataset_version,
            "hz": int(packed_manifest.get("hz", 0)),
            "manifest_sha256": hashlib.sha256(
                packed_manifest_bytes
            ).hexdigest(),
            "num_views": int(packed_manifest.get("num_views", 0)),
            "partition_id": packed_manifest.get("partition_id"),
            "shard_names": list(
                packed_manifest.get("shard_names", [])
            ),
            "source_revision": packed_manifest.get("source_revision"),
            "total_samples": total_samples,
            "uri": shard_uri,
        })
        if total_samples > 0:
            if int(packed_manifest.get("num_views", 0)) <= 0:
                raise ValueError(
                    f"benchmark shard has no camera views: {shard_dir}"
                )
            if not bool(packed_manifest.get("has_map", False)):
                raise ValueError(
                    "map-conditioned benchmark shard has no raster map: "
                    f"{shard_dir}"
                )
            if not bool(packed_manifest.get("has_gps", False)):
                raise ValueError(
                    "benchmark shard has no pose-grounded trajectory: "
                    f"{shard_dir}"
                )
            if (
                config.get("enable_world_model", False)
                and not bool(
                    packed_manifest.get("has_world_model", False)
                )
            ):
                raise ValueError(
                    "world-model checkpoint requires benchmark history "
                    f"windows: {shard_dir}"
                )
            shard_dirs.append(shard_dir)

    if not shard_dirs:
        raise ValueError("all benchmark shard partitions are empty")
    if len(dataset_versions) != 1:
        raise ValueError(
            "benchmark shards mix dataset versions: "
            f"{sorted(dataset_versions)}"
        )
    if len(contract_digests) != 1:
        raise ValueError("benchmark shards mix packing contracts")
    shard_identities.sort(
        key=lambda item: (
            str(item["partition_id"]),
            str(item["shard_names"]),
        )
    )
    shard_manifest_digest = stable_digest(shard_identities)

    model_kwargs = _model_kwargs(config)
    model_kwargs["is_pretrained"] = False
    model = AutoE2E(**model_kwargs)
    model.load_state_dict(payload["model_state_dict"])
    del payload

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cudnn, "deterministic"):
        torch.backends.cudnn.deterministic = True
    torch.multiprocessing.set_sharing_strategy("file_system")

    loader = make_multi_dataset_loader(
        shard_dirs,
        batch_size=batch_size,
        num_workers=1,
        split="all",
        val_fraction=0.0,
        shuffle=0,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=1,
        max_active_loaders=1,
        sample_uids=manifest.sample_uids,
        decode_future_frames=False,
    )
    projection_cache = _ProjectionDeviceCache(device)
    observed_uids: list[str] = []
    predicted_batches: list[np.ndarray] = []
    current_pose_batches: list[np.ndarray] = []
    gps_batches: list[np.ndarray] = []
    speed_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch, projection, geometry_type in loader:
            batch_uids = batch.get("sample_uid", [])
            if isinstance(batch_uids, str):
                batch_uids = [batch_uids]
            batch_uids = [str(uid) for uid in batch_uids]
            if not batch_uids:
                raise ValueError("benchmark batch lost its sample UIDs")
            batch_count = len(batch_uids)
            observed_uids.extend(batch_uids)

            history = batch["egomotion_history"]
            if tuple(history.shape) != (batch_count, 64 * 4):
                raise ValueError(
                    "benchmark egomotion history has unexpected shape "
                    f"{tuple(history.shape)}"
                )
            initial_speeds = (
                history.reshape(batch_count, 64, 4)[:, -1, 0]
                .numpy()
                .astype(np.float64)
            )
            current_pose = batch.get("pose_current")
            gps_future = batch.get("gps_future")
            if (
                current_pose is None
                or tuple(current_pose.shape) != (batch_count, 3)
            ):
                raise ValueError(
                    "benchmark current pose has unexpected shape "
                    f"{getattr(current_pose, 'shape', None)}"
                )
            if (
                gps_future is None
                or tuple(gps_future.shape) != (batch_count, 65, 2)
            ):
                raise ValueError(
                    "benchmark GPS trajectory has unexpected shape "
                    f"{getattr(gps_future, 'shape', None)}"
                )
            policy_history = adapt_egomotion_history(
                history,
                training_policy,
            )
            limited_history = limit_egomotion_history(
                policy_history,
                observation_steps=manifest.observation_steps,
            )

            history_frames = batch.get("history_frames")
            if batch.get("future_frames") is not None:
                raise RuntimeError(
                    "benchmark loader exposed future camera frames"
                )
            if config.get("enable_world_model", False):
                if history_frames is None:
                    raise ValueError(
                        "world-model checkpoint requires packed benchmark "
                        "history frames"
                    )
                if history_frames.ndim != 6 or history_frames.shape[1] != 4:
                    raise ValueError(
                        "KITScenes protocol expects four 1 Hz world-model "
                        "history frames, got "
                        f"{tuple(history_frames.shape)}"
                    )
                history_frames = history_frames.to(device)
            else:
                history_frames = None

            stable_noise = []
            for uid in batch_uids:
                seed_bytes = hashlib.sha256(
                    f"{PROTOCOL_ID}:{uid}".encode("ascii")
                ).digest()[:8]
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    int.from_bytes(seed_bytes, "big") % (2**63 - 1)
                )
                stable_noise.append(
                    torch.randn(128, generator=generator)
                )
            initial_noise = torch.stack(stable_noise).to(device)

            if hasattr(model, "reset_visual_history"):
                model.reset_visual_history()
            prediction = model(
                batch["visual_tiles"].to(device),
                batch["map_context"].to(device),
                batch["visual_history"].to(device),
                limited_history.to(device),
                route_mask=batch["route_mask"].to(device),
                map_valid=batch["map_valid"].to(device),
                route_valid=batch["route_valid"].to(device),
                projection=projection_cache.get(projection),
                geometry_type=geometry_type,
                history_frames=history_frames,
                mode="infer",
                initial_noise=initial_noise,
            )
            if not torch.is_tensor(prediction):
                raise TypeError(
                    "benchmark inference must return one trajectory tensor"
                )
            if tuple(prediction.shape) != (batch_count, 64 * 2):
                raise ValueError(
                    "benchmark prediction has unexpected shape "
                    f"{tuple(prediction.shape)}"
                )
            predicted_batches.append(
                prediction.detach().cpu().numpy().reshape(
                    batch_count, 64, 2
                )
            )
            current_pose_batches.append(current_pose.numpy())
            gps_batches.append(gps_future.numpy())
            speed_batches.append(initial_speeds)

    if hasattr(model, "reset_visual_history"):
        model.reset_visual_history()
    if len(observed_uids) != len(set(observed_uids)):
        raise ValueError("benchmark shards contain duplicate sample UIDs")
    expected_uids = set(manifest.sample_uids)
    actual_uids = set(observed_uids)
    if actual_uids != expected_uids:
        raise ValueError(
            "benchmark observed UID set differs from the fixed manifest: "
            f"missing={sorted(expected_uids - actual_uids)[:5]} "
            f"unexpected={sorted(actual_uids - expected_uids)[:5]}"
        )
    observed_uid_digest = sample_uid_digest(observed_uids)
    expected_uid_digest = sample_uid_digest(manifest.sample_uids)
    if observed_uid_digest != expected_uid_digest:
        raise ValueError("benchmark sample UID digest mismatch")

    predicted_controls = np.concatenate(predicted_batches, axis=0)
    current_poses = np.concatenate(current_pose_batches, axis=0)
    gps_future = np.concatenate(gps_batches, axis=0)
    target_xy = wgs84_trajectory_to_ego_xy(
        gps_future,
        current_poses,
    )
    initial_speeds = np.concatenate(speed_batches, axis=0)
    metrics, predicted_xy = compute_displacement_metrics(
        predicted_controls,
        target_xy,
        initial_speeds,
        frequency_hz=manifest.frequency_hz,
        horizons_seconds=manifest.horizons_seconds,
    )

    output_dir = Path("/tmp/kitscenes-benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    report_path = output_dir / "metrics.json"
    logged_manifest_path = output_dir / "manifest.json"
    logged_manifest_path.write_bytes(Path(manifest_path).read_bytes())

    prediction_records = []
    for index, uid in enumerate(observed_uids):
        prediction_records.append({
            "acceleration_curvature": predicted_controls[
                index, :max(manifest.horizon_steps)
            ].tolist(),
            "frequency_hz": manifest.frequency_hz,
            "horizon_steps": max(manifest.horizon_steps),
            "sample_uid": uid,
            "schema_version": "kitscenes_e2e_prediction_v1",
            "trajectory_xy_m": predicted_xy[index].tolist(),
        })
    prediction_records.sort(key=lambda record: record["sample_uid"])
    with predictions_path.open("w", encoding="ascii") as stream:
        for record in prediction_records:
            stream.write(json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ))
            stream.write("\n")
    predictions_sha256 = sha256_file(predictions_path)

    ctx = current_context()
    report = {
        "artifacts": {
            "predictions_sha256": predictions_sha256,
        },
        "benchmark": {
            "authority": manifest.authority,
            "benchmark_id": manifest.benchmark_id,
            "dataset_revision": manifest.dataset_revision,
            "history_adapter": manifest.history_adapter,
            "input_track": manifest.input_track,
            "manifest_sha256": manifest_sha256,
            "protocol_id": PROTOCOL_ID,
            "protocol_source": manifest.protocol_source,
            "protocol_status": manifest.protocol_status,
            "release_id": manifest.release_id,
            "sample_count": len(observed_uids),
            "sample_uid_digest": observed_uid_digest,
            "sdk_revision": manifest.sdk_revision,
            "source_splits": list(manifest.source_splits),
        },
        "checkpoint": {
            "epoch": epoch,
            "mlflow_run_id": run_id,
            "recorded_mlflow_run_id": checkpoint_run_id or None,
            "sha256": checkpoint_sha256,
            "uri": checkpoint_uri,
        },
        "dataset": {
            "dataset_version": next(iter(dataset_versions)),
            "packed_manifest_digest": shard_manifest_digest,
            "partition_count": len(shard_identities),
            "partitions": shard_identities,
        },
        "evaluator": {
            "docker_image": EVAL_IMAGE,
            "flyte_execution_id": (
                ctx.execution_id.name if ctx.execution_id else "local"
            ),
            "prediction_noise": "sha256(protocol_id:sample_uid)",
            "prediction_trajectory": "integrated_acceleration_curvature",
            "target_trajectory": "packed_gps_to_utm32_ego_frame",
            "version": EVALUATOR_VERSION,
        },
        "model_inputs": {
            "camera_views": True,
            "egomotion_history_seconds": manifest.past_seconds,
            "raster_map": True,
            "world_model_history_frames": (
                4 if config.get("enable_world_model", False) else 0
            ),
        },
        "metric_availability": {
            "ade_3s": "computed",
            "ade_5s": "computed",
            "centerline_distance": "authority_assets_required",
            "collision_free_rate": "authority_assets_required",
            "drivable_surface_survival": "authority_assets_required",
            "fde_3s": "computed",
            "fde_5s": "computed",
            "mms": "authority_assets_required",
        },
        "metrics": metrics,
        "schema_version": "kitscenes_e2e_benchmark_report_v1",
    }
    report_path.write_text(
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

    metric_prefix = (
        f"benchmark/kitscenes/{manifest.protocol_status}"
    )
    artifact_path = (
        f"benchmark/kitscenes/{manifest.benchmark_id}/"
        f"{manifest_sha256[:12]}/"
        f"checkpoint-{checkpoint_sha256[:12]}"
    )
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                f"{metric_prefix}/{name}": value
                for name, value in metrics.items()
            },
            step=epoch,
        )
        mlflow.set_tags({
            "benchmark/kitscenes/authority": manifest.authority,
            "benchmark/kitscenes/checkpoint_sha256": checkpoint_sha256,
            "benchmark/kitscenes/input_track": manifest.input_track,
            "benchmark/kitscenes/manifest_sha256": manifest_sha256,
            "benchmark/kitscenes/protocol_source": (
                manifest.protocol_source
            ),
            "benchmark/kitscenes/protocol_status": (
                manifest.protocol_status
            ),
            "benchmark/kitscenes/release_id": manifest.release_id,
            "benchmark/kitscenes/sample_uid_digest": (
                observed_uid_digest
            ),
            "benchmark/kitscenes/source_splits": ",".join(
                manifest.source_splits
            ),
        })
        mlflow.log_artifacts(str(output_dir), artifact_path=artifact_path)

    print(
        "KITScenes benchmark: "
        f"status={manifest.protocol_status} epoch={epoch} "
        f"samples={len(observed_uids)} metrics={metrics}"
    )
    return KITScenesBenchmarkOutput(
        ade_3s=metrics["ade_3s"],
        fde_3s=metrics["fde_3s"],
        ade_5s=metrics["ade_5s"],
        fde_5s=metrics["fde_5s"],
        predictions=FlyteFile(str(predictions_path)),
        report=FlyteFile(str(report_path)),
    )


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="4", mem="16Gi"),
    limits=Resources(cpu="4", mem="16Gi"),
    environment={
        "AUTO_E2E_EVAL_IMAGE": EVAL_IMAGE,
    },
)
def audit_kitscenes_target_reconstruction(
    packed_shards: List[FlyteDirectory],
    audit_code_revision: str,
    expected_dataset_version: str = KITSCENES_NAVIGATION_DATASET_VERSION,
    val_fraction: float = 0.1,
    validation_scope: str = "full",
) -> ReconstructionAuditOutput:
    """Derive the training holdout and audit target controls against GPS."""
    import hashlib
    import json
    import os
    import re
    from pathlib import Path

    from evaluation.reconstruction_audit import (
        audit_packed_target_rollout_reconstruction,
        load_packed_reconstruction_inputs,
    )
    from Platform.pipelines.training_checkpoint import (
        sha256_file,
        stable_digest,
    )
    from training.losses.control_rollout import ROLLOUT_POLICY_VERSION
    from training.dataset_policy import (
        group_uid_digest,
        training_policy_for_dataset,
        validation_group_uids as select_validation_group_uids,
    )
    from data_parsing.pre_extracted import discover_split_inventory

    if not packed_shards:
        raise ValueError("packed_shards must not be empty")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be between 0 and 1, got {val_fraction}"
        )
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", audit_code_revision):
        raise ValueError(
            "audit_code_revision must be a 40- or 64-character revision"
        )
    if not expected_dataset_version:
        raise ValueError("expected_dataset_version must not be empty")

    shard_dirs: list[str] = []
    shard_identities: list[dict] = []
    dataset_versions: set[str] = set()
    source_revisions: set[str] = set()
    contract_digests: set[str] = set()
    for shard in packed_shards:
        shard_uri = str(getattr(shard, "remote_source", "") or shard)
        shard_dir = str(shard.download())
        manifest_path = Path(shard_dir) / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"packed shard manifest is missing: {manifest_path}"
            )
        manifest_bytes = manifest_path.read_bytes()
        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid packed shard manifest: {manifest_path}"
            ) from error
        if not isinstance(manifest, dict):
            raise ValueError(
                f"packed shard manifest must be an object: {manifest_path}"
            )
        if manifest.get("dataset") != Dataset.KITSCENES.value:
            raise ValueError(
                "reconstruction audit only accepts KITScenes shards"
            )
        total_samples = int(manifest.get("total_samples", 0))
        if total_samples > 0 and not bool(manifest.get("has_gps", False)):
            raise ValueError(
                f"packed shard has no pose-grounded trajectory: {shard_dir}"
            )
        dataset_version = str(manifest.get("dataset_version", ""))
        source_revision = str(manifest.get("source_revision", ""))
        if not dataset_version or not source_revision:
            raise ValueError(
                f"packed shard has incomplete provenance: {shard_dir}"
            )
        dataset_versions.add(dataset_version)
        source_revisions.add(source_revision)
        contract_digests.add(stable_digest(manifest.get("contracts")))
        identity = {
            "contracts": manifest.get("contracts"),
            "dataset": manifest.get("dataset"),
            "dataset_version": dataset_version,
            "hz": int(manifest.get("hz", 0)),
            "manifest_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
            "partition_id": manifest.get("partition_id"),
            "shard_names": list(manifest.get("shard_names", [])),
            "source_revision": source_revision,
            "total_samples": total_samples,
            "uri": shard_uri,
        }
        shard_identities.append(identity)
        if identity["total_samples"] > 0:
            shard_dirs.append(shard_dir)

    if not shard_dirs:
        raise ValueError("all packed shard partitions are empty")
    if dataset_versions != {expected_dataset_version}:
        raise ValueError(
            "packed dataset version mismatch: "
            f"expected={expected_dataset_version!r} "
            f"actual={sorted(dataset_versions)}"
        )
    if len(source_revisions) != 1:
        raise ValueError(
            f"packed shards mix source revisions: {sorted(source_revisions)}"
        )
    if len(contract_digests) != 1:
        raise ValueError("packed shards mix packing contracts")
    shard_identities.sort(
        key=lambda item: (
            str(item["partition_id"]),
            str(item["shard_names"]),
        )
    )

    source_revision = next(iter(source_revisions))
    packed_contract_digest = next(iter(contract_digests))
    split_inventory = discover_split_inventory(shard_dirs)
    expected_sample_count = sum(
        int(identity["total_samples"])
        for identity in shard_identities
    )
    if split_inventory.sample_count != expected_sample_count:
        raise ValueError(
            "packed sample metadata coverage differs from manifests: "
            f"expected={expected_sample_count} "
            f"actual={split_inventory.sample_count}"
        )
    training_policy = training_policy_for_dataset(
        Dataset.KITSCENES.value,
        validation_scope=validation_scope,
    )
    validation_group_uids = select_validation_group_uids(
        split_inventory.group_uids,
        val_fraction=val_fraction,
        policy=training_policy,
        source_revision=source_revision,
        packed_dataset_version=expected_dataset_version,
        packed_contract_digest=packed_contract_digest,
        packed_partition_count=len(shard_identities),
        empty_partition_count=(
            len(shard_identities) - len(shard_dirs)
        ),
        packed_sample_count=split_inventory.sample_count,
        packed_sample_uid_digest=split_inventory.sample_uid_digest,
    )
    if validation_group_uids is None:
        raise ValueError(
            "reconstruction audit requires an exact validation split"
        )
    (
        expected_validation_sample_count,
        expected_validation_sample_uid_digest,
    ) = split_inventory.sample_identity_for_groups(
        validation_group_uids
    )

    inputs = load_packed_reconstruction_inputs(
        shard_dirs,
        validation_group_uids=validation_group_uids,
    )
    report = audit_packed_target_rollout_reconstruction(inputs)
    actual_sample_digest = str(report["sample_uid_digest"])
    if actual_sample_digest != expected_validation_sample_uid_digest:
        raise ValueError(
            "validation snapshot digest mismatch: "
            f"expected={expected_validation_sample_uid_digest} "
            f"actual={actual_sample_digest}"
        )
    if int(report["sample_count"]) != expected_validation_sample_count:
        raise ValueError(
            "validation snapshot sample count mismatch: "
            f"expected={expected_validation_sample_count} "
            f"actual={report['sample_count']}"
        )

    records = report.pop("records")
    output_dir = Path("/tmp/target-rollout-reconstruction-audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "sample_metrics.jsonl"
    with records_path.open("w", encoding="ascii") as stream:
        for record in records:
            stream.write(json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ))
            stream.write("\n")
    records_sha256 = sha256_file(records_path)

    group_digest = group_uid_digest(validation_group_uids)
    report["artifacts"] = {
        "sample_metrics_sha256": records_sha256,
    }
    report["metric_availability"] = {
        "current_model_pose_grounded_error": (
            "not_computed_by_target_reconstruction_audit"
        ),
        "target_rollout_reconstruction": "computed",
    }
    report["provenance"] = {
        "audit_code_revision": audit_code_revision,
        "container_image": os.environ["AUTO_E2E_EVAL_IMAGE"],
        "dataset": Dataset.KITSCENES.value,
        "dataset_version": next(iter(dataset_versions)),
        "packed_contract_digest": packed_contract_digest,
        "packed_manifest_digest": stable_digest(shard_identities),
        "partition_count": len(shard_identities),
        "rollout_policy_version": ROLLOUT_POLICY_VERSION,
        "source_revision": source_revision,
        "validation_scope": validation_scope,
        "validation_fraction": val_fraction,
        "validation_group_count": len(validation_group_uids),
        "validation_group_uid_digest": group_digest,
        "validation_sample_uid_digest": actual_sample_digest,
        "validation_split_id": training_policy.validation_split_id,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
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
    report_sha256 = sha256_file(report_path)

    print(
        "Target rollout reconstruction audit: "
        f"samples={report['sample_count']} "
        f"scenes={report['scene_count']} "
        f"thresholds_pass={report['thresholds_pass']} "
        f"report_sha256={report_sha256}"
    )
    return ReconstructionAuditOutput(
        thresholds_pass=bool(report["thresholds_pass"]),
        report_sha256=report_sha256,
        records_sha256=records_sha256,
        report=FlyteFile(str(report_path)),
        records=FlyteFile(str(records_path)),
    )



# ============================================================
# Workflows
# ============================================================
@workflow
def wf_evaluate_kitscenes_benchmark(
    checkpoint: FlyteFile,
    benchmark_shards: List[FlyteDirectory],
    benchmark_manifest: FlyteFile,
    expected_manifest_sha256: str = "",
    mlflow_run_id: str = "",
    batch_size: int = 4,
) -> KITScenesBenchmarkOutput:
    """Retrospectively score one checkpoint without invoking training."""
    return evaluate_kitscenes_benchmark_checkpoint(
        checkpoint=checkpoint,
        benchmark_shards=benchmark_shards,
        benchmark_manifest=benchmark_manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        mlflow_run_id=mlflow_run_id,
        batch_size=batch_size,
    )


@workflow
def wf_audit_kitscenes_target_reconstruction(
    packed_shards: List[FlyteDirectory],
    audit_code_revision: str,
    expected_dataset_version: str = KITSCENES_NAVIGATION_DATASET_VERSION,
    val_fraction: float = 0.1,
    validation_scope: str = "full",
) -> ReconstructionAuditOutput:
    """Run the immutable preflight gate for rollout-aligned training."""
    return audit_kitscenes_target_reconstruction(
        packed_shards=packed_shards,
        audit_code_revision=audit_code_revision,
        expected_dataset_version=expected_dataset_version,
        val_fraction=val_fraction,
        validation_scope=validation_scope,
    )


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
                expected_reasoning_label_count=None,
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
            expected_reasoning_label_count=None,
        ),
        concurrency=pack_concurrency,
    )
    return pack(raw_data=raw_dirs, group_ids=partitions)


@dynamic(
    container_image=DATA_PREP_IMAGE,
    environment={"AUTO_E2E_DATA_PREP_IMAGE": DATA_PREP_IMAGE},
)
def _map_recovered_kitscenes_artifacts(
    recovery_manifest: FlyteFile,
    artifact_set_sha256: str,
    dataset_version: str,
    image_size: int,
    pack_concurrency: int,
    max_partitions: int,
) -> List[FlyteDirectory]:
    """Map only pack tasks over an audited raw/label artifact set."""
    if pack_concurrency <= 0:
        raise ValueError(
            f"pack_concurrency must be positive, got {pack_concurrency}"
        )
    if max_partitions < 0:
        raise ValueError(
            f"max_partitions must be non-negative, got {max_partitions}"
        )

    from data_parsing.kit_scenes.source import sdk_split_scene_ids
    from Platform.pipelines.kitscenes_recovery import (
        AUDITED_EMPTY_SCENE_COUNT,
        AUDITED_LABEL_COUNT,
        KNOWN_MISSING_TRAIN_SCENE,
        load_recovery_manifest,
    )

    official_train = sdk_split_scene_ids("train")
    expected_scenes = [
        scene_id
        for scene_id in official_train
        if scene_id != KNOWN_MISSING_TRAIN_SCENE
    ]
    if (
        len(official_train) != 534
        or len(expected_scenes) != 533
        or KNOWN_MISSING_TRAIN_SCENE not in official_train
    ):
        raise ValueError(
            "pinned KITScenes train inventory no longer has the audited "
            "534-scene/one-missing contract"
        )

    entries = load_recovery_manifest(
        recovery_manifest.download(),
        expected_artifact_set_sha256=artifact_set_sha256,
        expected_dataset=Dataset.KITSCENES.value,
        expected_source_revision=KITSCENES_SOURCE_REVISION,
        expected_scene_ids=expected_scenes,
        expected_label_count=AUDITED_LABEL_COUNT,
        expected_empty_scene_count=AUDITED_EMPTY_SCENE_COUNT,
    )
    if max_partitions:
        if not 2 <= max_partitions < len(entries):
            raise ValueError(
                "recovery subset max_partitions must select at least two "
                "but fewer than the full manifest"
            )
        entries = entries[:max_partitions]
    raw_dirs = [
        FlyteDirectory(entry["raw_uri"]) for entry in entries
    ]
    label_dirs = [
        FlyteDirectory(entry["label_uri"]) for entry in entries
    ]
    partitions = [[entry["scene_id"]] for entry in entries]
    expected_label_counts = [
        entry["expected_label_count"] for entry in entries
    ]

    pack = map_task(
        functools.partial(
            data_processing,
            dataset=Dataset.KITSCENES,
            source_revision=KITSCENES_SOURCE_REVISION,
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
        expected_reasoning_label_count=expected_label_counts,
    )


@workflow
def wf_repack_existing_kitscenes(
    recovery_manifest: FlyteFile,
    artifact_set_sha256: str,
    dataset_version: str = KITSCENES_NAVIGATION_DATASET_VERSION,
    image_size: int = 256,
    pack_concurrency: int = 60,
    max_partitions: int = 0,
) -> List[FlyteDirectory]:
    """Repack audited raw/Cosmos artifacts without ingest or teacher calls."""
    return _map_recovered_kitscenes_artifacts(
        recovery_manifest=recovery_manifest,
        artifact_set_sha256=artifact_set_sha256,
        dataset_version=dataset_version,
        image_size=image_size,
        pack_concurrency=pack_concurrency,
        max_partitions=max_partitions,
    )


@workflow
def wf_audit_recovered_kitscenes_target_reconstruction(
    recovery_manifest: FlyteFile,
    artifact_set_sha256: str,
    audit_code_revision: str,
    dataset_version: str = KITSCENES_NAVIGATION_DATASET_VERSION,
    image_size: int = 256,
    pack_concurrency: int = 60,
    max_partitions: int = 0,
    val_fraction: float = 0.1,
    validation_scope: str = "full",
) -> ReconstructionAuditOutput:
    """Repack the exact recovery scope and audit its derived holdout."""
    shards = wf_repack_existing_kitscenes(
        recovery_manifest=recovery_manifest,
        artifact_set_sha256=artifact_set_sha256,
        dataset_version=dataset_version,
        image_size=image_size,
        pack_concurrency=pack_concurrency,
        max_partitions=max_partitions,
    )
    return audit_kitscenes_target_reconstruction(
        packed_shards=shards,
        audit_code_revision=audit_code_revision,
        expected_dataset_version=dataset_version,
        val_fraction=val_fraction,
        validation_scope=validation_scope,
    )


@workflow
def wf_create_dataset_sharded(
    dataset: Dataset = Dataset.KITSCENES,
    source_revision: str = KITSCENES_SOURCE_REVISION,
    dataset_version: str = KITSCENES_NAVIGATION_DATASET_VERSION,
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
    dataset_version: str = KITSCENES_NAVIGATION_DATASET_VERSION,
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
    epochs: int = 10,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    lr: float = 1e-4,
    training_seed: int = 149,
    enable_route_conditioning: bool = True,
    training_objective_version: str = (
        BASELINE_TRAINING_OBJECTIVE_VERSION
    ),
    enable_junction_sampling: bool = False,
    enable_route_consistency: bool = False,
    route_consistency_weight: float = 0.10,
    reconstruction_audit: Optional[FlyteFile] = None,
    reconstruction_audit_decision: str = "",
    reconstruction_audit_rationale: str = "",
    enable_reasoning: bool = True,
    reasoning_mode: str = "pooled_latent",
    enable_world_model: bool = True,
    val_fraction: float = 0.1,
    validation_scope: str = "full",
    num_workers: int = 4,
    resume_from: Optional[FlyteFile] = None,
    early_stopping_patience: int = 5,
    allow_resume_policy_transition: bool = False,
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
    navigation_quality_audit = audit_kitscenes_navigation_quality(
        shards=shards,
    )
    out = train_il(
        shards=shards, dataset=dataset, backbone=backbone, epochs=epochs,
        batch_size=batch_size, grad_accum_steps=grad_accum_steps, lr=lr,
        training_seed=training_seed,
        enable_route_conditioning=enable_route_conditioning,
        training_objective_version=training_objective_version,
        enable_junction_sampling=enable_junction_sampling,
        enable_route_consistency=enable_route_consistency,
        route_consistency_weight=route_consistency_weight,
        navigation_quality_audit=navigation_quality_audit,
        reconstruction_audit=reconstruction_audit,
        reconstruction_audit_decision=reconstruction_audit_decision,
        reconstruction_audit_rationale=reconstruction_audit_rationale,
        enable_reasoning=enable_reasoning, reasoning_mode=reasoning_mode,
        enable_world_model=enable_world_model, val_fraction=val_fraction,
        validation_scope=validation_scope,
        num_workers=num_workers, resume_from=resume_from,
        early_stopping_patience=early_stopping_patience,
        allow_resume_policy_transition=allow_resume_policy_transition)
    return evaluate_il_policy(
        checkpoint=out.checkpoint, shards=shards, dataset=dataset,
        train_metadata=out.metadata)


@workflow
def wf_recovered_kitscenes_full_run(
    recovery_manifest: FlyteFile,
    artifact_set_sha256: str,
    dataset_version: str = KITSCENES_NAVIGATION_DATASET_VERSION,
    image_size: int = 256,
    pack_concurrency: int = 60,
    max_partitions: int = 0,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs: int = 20,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    lr: float = 1e-4,
    training_seed: int = 149,
    enable_route_conditioning: bool = True,
    training_objective_version: str = (
        ROLLOUT_ALIGNED_OBJECTIVE_VERSION
    ),
    enable_junction_sampling: bool = False,
    enable_route_consistency: bool = False,
    route_consistency_weight: float = 0.10,
    reconstruction_audit: Optional[FlyteFile] = None,
    reconstruction_audit_decision: str = "",
    reconstruction_audit_rationale: str = "",
    reasoning_mode: str = "pooled_latent",
    val_fraction: float = 0.1,
    validation_scope: str = "full",
    num_workers: int = 4,
    resume_from: Optional[FlyteFile] = None,
    early_stopping_patience: int = 5,
    allow_resume_policy_transition: bool = False,
) -> EvalMetrics:
    """Repack audited artifacts, then train/evaluate without ingest or Cosmos."""
    shards = wf_repack_existing_kitscenes(
        recovery_manifest=recovery_manifest,
        artifact_set_sha256=artifact_set_sha256,
        dataset_version=dataset_version,
        image_size=image_size,
        pack_concurrency=pack_concurrency,
        max_partitions=max_partitions,
    )
    navigation_quality_audit = audit_kitscenes_navigation_quality(
        shards=shards,
    )
    out = train_il(
        shards=shards,
        dataset=Dataset.KITSCENES,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        training_seed=training_seed,
        enable_route_conditioning=enable_route_conditioning,
        training_objective_version=training_objective_version,
        enable_junction_sampling=enable_junction_sampling,
        enable_route_consistency=enable_route_consistency,
        route_consistency_weight=route_consistency_weight,
        navigation_quality_audit=navigation_quality_audit,
        reconstruction_audit=reconstruction_audit,
        reconstruction_audit_decision=reconstruction_audit_decision,
        reconstruction_audit_rationale=reconstruction_audit_rationale,
        enable_reasoning=True,
        reasoning_mode=reasoning_mode,
        enable_world_model=True,
        val_fraction=val_fraction,
        validation_scope=validation_scope,
        num_workers=num_workers,
        resume_from=resume_from,
        early_stopping_patience=early_stopping_patience,
        allow_resume_policy_transition=allow_resume_policy_transition,
    )
    return evaluate_il_policy(
        checkpoint=out.checkpoint,
        shards=shards,
        dataset=Dataset.KITSCENES,
        train_metadata=out.metadata,
    )


@workflow
def wf_compare_recovered_kitscenes_navigation(
    recovery_manifest: FlyteFile,
    artifact_set_sha256: str,
    conditioned_checkpoint: FlyteFile,
    conditioned_train_metadata: FlyteFile,
    baseline_checkpoint: FlyteFile,
    baseline_train_metadata: FlyteFile,
    dataset_version: str = KITSCENES_NAVIGATION_DATASET_VERSION,
    image_size: int = 256,
    pack_concurrency: int = 60,
) -> FlyteFile:
    """Run the frozen paired comparison on the cached KITScenes v3 corpus."""
    shards = wf_repack_existing_kitscenes(
        recovery_manifest=recovery_manifest,
        artifact_set_sha256=artifact_set_sha256,
        dataset_version=dataset_version,
        image_size=image_size,
        pack_concurrency=pack_concurrency,
    )
    conditioned_records = evaluate_navigation_records(
        checkpoint=conditioned_checkpoint,
        shards=shards,
        train_metadata=conditioned_train_metadata,
        expected_route_conditioning=True,
    )
    baseline_records = evaluate_navigation_records(
        checkpoint=baseline_checkpoint,
        shards=shards,
        train_metadata=baseline_train_metadata,
        expected_route_conditioning=False,
    )
    return compare_navigation_record_artifacts(
        conditioned_records=conditioned_records,
        baseline_records=baseline_records,
    )


@workflow
def wf_train_il(
    shards: List[FlyteDirectory],
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    map_fusion_mode: MapFusion = MapFusion.RESIDUAL,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    lr: float = 1e-4,
    training_seed: int = 149,
    amp: bool = False,
    enable_route_conditioning: bool = True,
    training_objective_version: str = (
        BASELINE_TRAINING_OBJECTIVE_VERSION
    ),
    enable_junction_sampling: bool = False,
    enable_route_consistency: bool = False,
    route_consistency_weight: float = 0.10,
    navigation_quality_audit: Optional[FlyteFile] = None,
    reconstruction_audit: Optional[FlyteFile] = None,
    reconstruction_audit_decision: str = "",
    reconstruction_audit_rationale: str = "",
    enable_reasoning: bool = False,
    reasoning_mode: str = "pooled_latent",
    enable_world_model: bool = False,
    val_fraction: float = 0.1,
    validation_scope: str = "full",
    num_workers: int = 0,
    resume_from: Optional[FlyteFile] = None,
    early_stopping_patience: int = 5,
    allow_resume_policy_transition: bool = False,
) -> EvalMetrics:
    """IL Train → Evaluate. All datasets' shards passed in; `dataset` selects one.

    The branch flags must match how the shards were packed (see
    ``wf_data_processing``); train_il fails loudly if a branch is enabled but its
    shard data is missing rather than training it unsupervised. ``amp`` defaults
    off: fp16 autocast made the GradScaler skip every step (see train_il).
    ``grad_accum_steps`` recovers a larger effective batch when the World-Model
    windows force batch_size=1 (effective batch = batch_size * grad_accum_steps).
    ``val_fraction`` > 0 trains on a group-level train split and evaluates on the
    disjoint held-out val split (generalization, not in-sample memorization).
    KITScenes pins full runs to an audited scene manifest; ``validation_scope``
    may explicitly select a deterministic subset split for smoke runs.
    ``num_workers`` > 0 parallelizes JPEG decode across worker processes (#121 P0)
    — the dominant per-epoch cost once episodes scale up.
    """
    out = train_il(shards=shards, dataset=dataset, backbone=backbone,
                   map_fusion_mode=map_fusion_mode,
                   epochs=epochs, batch_size=batch_size,
                   grad_accum_steps=grad_accum_steps, lr=lr,
                   training_seed=training_seed, amp=amp,
                   enable_route_conditioning=enable_route_conditioning,
                   training_objective_version=training_objective_version,
                   enable_junction_sampling=enable_junction_sampling,
                   enable_route_consistency=enable_route_consistency,
                   route_consistency_weight=route_consistency_weight,
                   navigation_quality_audit=navigation_quality_audit,
                   reconstruction_audit=reconstruction_audit,
                   reconstruction_audit_decision=reconstruction_audit_decision,
                   reconstruction_audit_rationale=reconstruction_audit_rationale,
                   enable_reasoning=enable_reasoning, reasoning_mode=reasoning_mode,
                   enable_world_model=enable_world_model, val_fraction=val_fraction,
                   validation_scope=validation_scope,
                   num_workers=num_workers, resume_from=resume_from,
                   early_stopping_patience=early_stopping_patience,
                   allow_resume_policy_transition=allow_resume_policy_transition)
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
    model version, marks the overlay set ``building``, then runs one resumable
    GPU task over every packed partition so the checkpoint is loaded once for
    the FullSet. It writes S3 bodies before Dynamo pointers, then publishes the
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
    result = precompute_overlay_partition(
        checkpoint=resolved.checkpoint,
        model_metadata=resolved.metadata,
        prepare_gate=gate,
        shard_dirs=shards,
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
    )
    return finalize_overlay_set(
        model_metadata=resolved.metadata,
        partition_results=[result],
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
def wf_publish_selected_checkpoint_overlays(
    shards: List[FlyteDirectory],
    full_run_execution_id: str,
    mlflow_run_id: str,
    checkpoint_uri: str,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
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
    """Publish a verified checkpoint while its parent Training still runs."""
    model_version = register_selected_overlay_checkpoint(
        registered_model_name=registered_model_name,
        run_id=mlflow_run_id,
        checkpoint_uri=checkpoint_uri,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=checkpoint_epoch,
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
