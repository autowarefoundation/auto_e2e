"""Flyte Ray tasks for distributed AutoE2E training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, NamedTuple, Optional
from urllib.parse import urlparse

from flytekit import (
    PodTemplate,
    Resources,
    current_context,
    dynamic,
    task,
    workflow,
)
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from flytekitplugins.ray import (
    HeadNodeConfig,
    RayJobConfig,
    WorkerNodeConfig,
)
from kubernetes.client import (
    V1Affinity,
    V1Container,
    V1EmptyDirVolumeSource,
    V1EnvVar,
    V1LabelSelector,
    V1LabelSelectorRequirement,
    V1PodAffinity,
    V1PodAffinityTerm,
    V1PodAntiAffinity,
    V1PodSpec,
    V1ResourceRequirements,
    V1Toleration,
    V1Volume,
    V1VolumeMount,
)

from data_parsing.kit_scenes.source import (
    KITSCENES_DATA_REVISION,
    KITSCENES_REPO_ID,
    KITSCENES_STANDARD_TEST_SCENE_COUNT,
    KITSCENES_STANDARD_TEST_SCENE_UID_SHA256,
    KITSCENES_STANDARD_VAL_SCENE_COUNT,
    KITSCENES_STANDARD_VAL_SCENE_UID_SHA256,
    sdk_split_scene_ids,
)

if TYPE_CHECKING:
    from data_parsing.pre_extracted import PackedSplitInventory

TRAINING_IMAGE = os.environ.get(
    "AUTO_E2E_TRAINING_IMAGE",
    "auto-e2e/training:latest",
)
RAY_STORAGE_PATH = os.environ.get(
    "AUTO_E2E_RAY_STORAGE_PATH",
    "s3://auto-e2e-platform-checkpoints/ray-train",
)
MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://mlflow.mlflow.svc.cluster.local:5000",
)
REACTIVE_MLFLOW_EXPERIMENT = "reactive-training"
REACTIVE_BEV_EVALUATION_MLFLOW_EXPERIMENT = (
    "reactive-bev-evaluation"
)
REACTIVE_POLICY_EVALUATION_MLFLOW_EXPERIMENT = (
    "reactive-policy-evaluation"
)
REACTIVE_BEV_REGISTERED_MODEL = "auto-e2e-bev-segmentation"
REACTIVE_POLICY_REGISTERED_MODEL = "auto-e2e-driving-policy"
RAY_TASK_ENVIRONMENT = {
    "AWS_DEFAULT_REGION": "us-west-2",
    "AUTO_E2E_RAY_STORAGE_PATH": RAY_STORAGE_PATH,
    "MLFLOW_TRACKING_URI": MLFLOW_URI,
    "RAY_TRAIN_V2_ENABLED": "1",
}
BEV_POS_WEIGHT_CAP = 64.0
BEV_CANARY_MIN_SELECTION_GAIN = 1e-3
BEV_CANARY_AP_LIFT_ABSOLUTE_REGRESSION_TOLERANCE = 1e-3
BEV_CANARY_RARE_AP_LIFT_RELATIVE_REGRESSION_TOLERANCE = 0.1
P5EN_SMOKE_MAX_RUNTIME = timedelta(minutes=30)
NUPLAN_EPOCH5_TRAJECTORY_ROUTE_CHECKPOINT_SHA256 = (
    "ca8b43d7a777d6fd9195bb253d1452f3cd645f7aab3bd0e36b1f74ffb31df29b"
)
KITSCENES_OFFICIAL_TEST_SCENE_COUNT = (
    KITSCENES_STANDARD_TEST_SCENE_COUNT
)
KITSCENES_OFFICIAL_TEST_SCENE_UID_SHA256 = (
    KITSCENES_STANDARD_TEST_SCENE_UID_SHA256
)
KITSCENES_OFFICIAL_VAL_SCENE_COUNT = KITSCENES_STANDARD_VAL_SCENE_COUNT
KITSCENES_OFFICIAL_VAL_SCENE_UID_SHA256 = (
    KITSCENES_STANDARD_VAL_SCENE_UID_SHA256
)
KITSCENES_OFFICIAL_VAL_SAMPLE_COUNT = 11_035
KITSCENES_OFFICIAL_TEST_SAMPLE_COUNT = 23_690
KITSCENES_OFFICIAL_VAL_DATASET_VERSION = "v3.5"
KITSCENES_OFFICIAL_TEST_DATASET_VERSION = "v3.5-test-camera-only-v1"
KITSCENES_PRIMARY_VAL_BATCH_SIZE = 1
KITSCENES_STATEFUL_CAMERA_FPN_BATCH_SIZE = 5
KITSCENES_STATEFUL_CAMERA_FPN_POLICY = (
    "stateful_camera_fpn_5lane_v1"
)
KITSCENES_TRAJECTORY_INFERENCE_POLICY = (
    "deterministic_gru_no_noise_v1"
)
KITSCENES_PRIMARY_VAL_ROUTE_USAGE_POLICY = (
    "route_zero_counterfactual_no_grad_v1"
)
KITSCENES_CAMERA_ONLY_ROUTE_USAGE_POLICY = (
    "not_applicable_camera_only_v1"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kitscenes_route_usage_evaluation_policy(
    *,
    mapless_test: bool,
) -> dict[str, object]:
    return {
        "counterfactuals_enabled": not mapless_test,
        "input_gradient_enabled": False,
        "version": (
            KITSCENES_CAMERA_ONLY_ROUTE_USAGE_POLICY
            if mapless_test
            else KITSCENES_PRIMARY_VAL_ROUTE_USAGE_POLICY
        ),
    }


def _discover_kitscenes_evaluation_inventory(
    shard_directories: list[str],
) -> PackedSplitInventory:
    from data_parsing.pre_extracted import discover_split_inventory

    # Evaluation canaries may intentionally cover one exact scene.
    return discover_split_inventory(
        shard_directories,
        allow_single_group=True,
    )


def _validate_kitscenes_test_inventory_groups(
    inventory_group_uids: tuple[str, ...],
    manifest_group_uids: list[str],
) -> None:
    expected_group_uids = tuple(sorted(manifest_group_uids))
    if inventory_group_uids != expected_group_uids:
        raise ValueError(
            "KITScenes test packed scene identities differ from manifests"
        )


def _kitscenes_test_scene_identity(
    scene_uids: list[str],
    *,
    expected_partition_count: int,
) -> tuple[str, bool]:
    if (
        isinstance(expected_partition_count, bool)
        or not isinstance(expected_partition_count, int)
        or expected_partition_count < 0
    ):
        raise ValueError(
            "KITScenes test expected partition count must be non-negative"
        )
    if len(scene_uids) != len(set(scene_uids)):
        raise ValueError(
            "KITScenes test manifests contain duplicate scene identities"
        )
    scene_uid_sha256 = hashlib.sha256(
        "\n".join(sorted(scene_uids)).encode("utf-8")
    ).hexdigest()
    if (
        expected_partition_count
        and len(scene_uids) != expected_partition_count
    ):
        raise ValueError(
            "KITScenes test evaluation does not cover the expected "
            f"{expected_partition_count} partitions"
        )
    is_official_test = (
        len(scene_uids) == KITSCENES_OFFICIAL_TEST_SCENE_COUNT
        and scene_uid_sha256
        == KITSCENES_OFFICIAL_TEST_SCENE_UID_SHA256
    )
    if (
        expected_partition_count
        == KITSCENES_OFFICIAL_TEST_SCENE_COUNT
        and not is_official_test
    ):
        raise ValueError(
            "KITScenes test manifests do not cover the official scene set"
        )
    return scene_uid_sha256, is_official_test


def _kitscenes_val_scene_identity(
    scene_uids: list[str],
    *,
    expected_partition_count: int,
) -> tuple[str, bool]:
    if (
        isinstance(expected_partition_count, bool)
        or not isinstance(expected_partition_count, int)
        or expected_partition_count < 0
    ):
        raise ValueError(
            "KITScenes val expected partition count must be non-negative"
        )
    if len(scene_uids) != len(set(scene_uids)):
        raise ValueError(
            "KITScenes val manifests contain duplicate scene identities"
        )
    scene_uid_sha256 = hashlib.sha256(
        "\n".join(sorted(scene_uids)).encode("utf-8")
    ).hexdigest()
    if (
        expected_partition_count
        and len(scene_uids) != expected_partition_count
    ):
        raise ValueError(
            "KITScenes val evaluation does not cover the expected "
            f"{expected_partition_count} partitions"
    )
    is_official_val = (
        expected_partition_count == KITSCENES_OFFICIAL_VAL_SCENE_COUNT
        and len(scene_uids) == KITSCENES_OFFICIAL_VAL_SCENE_COUNT
        and scene_uid_sha256
        == KITSCENES_OFFICIAL_VAL_SCENE_UID_SHA256
    )
    if (
        expected_partition_count == KITSCENES_OFFICIAL_VAL_SCENE_COUNT
        and not is_official_val
    ):
        raise ValueError(
            "KITScenes val manifests do not cover the official scene set"
        )
    return scene_uid_sha256, is_official_val


def _validate_kitscenes_official_val_sdk_split(
    scene_uids: list[str],
    *,
    expected_partition_count: int,
) -> None:
    if expected_partition_count != KITSCENES_OFFICIAL_VAL_SCENE_COUNT:
        return
    sdk_val_scene_uids = sorted(
        f"kitscenes-{scene_id}"
        for scene_id in sdk_split_scene_ids("val")
    )
    sdk_train_scene_uids = {
        f"kitscenes-{scene_id}"
        for scene_id in sdk_split_scene_ids("train")
    }
    sdk_val_sha256 = hashlib.sha256(
        "\n".join(sdk_val_scene_uids).encode("utf-8")
    ).hexdigest()
    if (
        len(sdk_val_scene_uids) != KITSCENES_OFFICIAL_VAL_SCENE_COUNT
        or sdk_val_sha256 != KITSCENES_OFFICIAL_VAL_SCENE_UID_SHA256
    ):
        raise ValueError(
            "pinned KITScenes SDK val split differs from the reviewed "
            "official identity"
        )
    actual_scene_uids = sorted(scene_uids)
    if actual_scene_uids != sdk_val_scene_uids:
        raise ValueError(
            "KITScenes official val manifests differ from the pinned SDK "
            "val split"
        )
    overlap = sorted(set(actual_scene_uids) & sdk_train_scene_uids)
    if overlap:
        raise ValueError(
            "KITScenes official val split overlaps the pinned SDK train "
            f"split: {overlap[:5]}"
        )


def _validate_kitscenes_official_sample_inventory(
    *,
    expected_partition_count: int,
    empty_partition_count: int,
    manifest_sample_count: int,
    inventory_sample_count: int | None,
) -> None:
    if expected_partition_count != KITSCENES_OFFICIAL_TEST_SCENE_COUNT:
        return
    if (
        empty_partition_count != 0
        or manifest_sample_count != KITSCENES_OFFICIAL_TEST_SAMPLE_COUNT
        or inventory_sample_count
        != KITSCENES_OFFICIAL_TEST_SAMPLE_COUNT
    ):
        raise ValueError(
            "KITScenes official test sample inventory differs from "
            f"{KITSCENES_OFFICIAL_TEST_SAMPLE_COUNT}"
        )


def _validate_kitscenes_official_val_sample_inventory(
    *,
    expected_partition_count: int,
    empty_partition_count: int,
    manifest_sample_count: int,
    inventory_sample_count: int | None,
) -> None:
    if expected_partition_count != KITSCENES_OFFICIAL_VAL_SCENE_COUNT:
        return
    if (
        empty_partition_count != 0
        or manifest_sample_count != KITSCENES_OFFICIAL_VAL_SAMPLE_COUNT
        or inventory_sample_count != KITSCENES_OFFICIAL_VAL_SAMPLE_COUNT
    ):
        raise ValueError(
            "KITScenes official val sample inventory differs from "
            f"{KITSCENES_OFFICIAL_VAL_SAMPLE_COUNT}"
        )


def _validate_kitscenes_val_manifest_contract(
    manifest: Mapping[str, Any],
) -> None:
    expected_flags = {
        "has_map": True,
        "has_navigation": True,
        "has_reactive_navigation": True,
        "has_route_reconstruction": True,
        "has_trajectory_xy": True,
    }
    mismatches = {
        key: manifest.get(key)
        for key, expected in expected_flags.items()
        if manifest.get(key) is not expected
    }
    input_track = manifest.get("input_track")
    if input_track not in {None, "camera_map_route"}:
        mismatches["input_track"] = input_track
    if mismatches:
        raise ValueError(
            "KITScenes val manifest violates the camera-map-route "
            f"contract: {mismatches}"
        )


def _resolve_nuplan_validation_sample_limit(
    config: Mapping[str, Any],
    requested_sample_limit: int,
) -> tuple[int, str]:
    """Resolve legacy checkpoint provenance from an explicit workflow input."""
    if (
        isinstance(requested_sample_limit, bool)
        or not isinstance(requested_sample_limit, int)
        or requested_sample_limit < 0
    ):
        raise ValueError(
            "nuPlan evaluation validation_sample_limit is invalid"
        )
    checkpoint_sample_limit = config.get("validation_sample_limit")
    if checkpoint_sample_limit is None:
        if requested_sample_limit <= 0:
            raise ValueError(
                "nuPlan checkpoint lacks validation_sample_limit provenance; "
                "set the workflow input from the source training execution"
            )
        return requested_sample_limit, "workflow_input_legacy_checkpoint"
    if (
        isinstance(checkpoint_sample_limit, bool)
        or not isinstance(checkpoint_sample_limit, int)
        or checkpoint_sample_limit <= 0
    ):
        raise ValueError(
            "nuPlan checkpoint validation_sample_limit is invalid"
        )
    if (
        requested_sample_limit > 0
        and requested_sample_limit != checkpoint_sample_limit
    ):
        raise ValueError(
            "nuPlan evaluation validation_sample_limit differs from the "
            "checkpoint"
        )
    return checkpoint_sample_limit, "checkpoint"


class RaySmokeOutput(NamedTuple):
    report: FlyteFile


class ReactiveRayOutput(NamedTuple):
    checkpoint: FlyteFile
    metadata: FlyteFile
    checkpoint_uri: str
    checkpoint_sha256: str


class ReactiveDistributedProgramOutput(NamedTuple):
    stage_a_checkpoint: FlyteFile
    stage_a_metadata: FlyteFile
    stage_a_checkpoint_uri: str
    stage_a_checkpoint_sha256: str
    stage_b_checkpoint: FlyteFile
    stage_b_metadata: FlyteFile
    stage_b_checkpoint_uri: str
    stage_b_checkpoint_sha256: str


class ReactiveCanaryOutput(NamedTuple):
    stage_a_checkpoint: FlyteFile
    stage_b_checkpoint: FlyteFile
    stage_a_metadata: FlyteFile
    stage_b_metadata: FlyteFile
    gate_report: FlyteFile


class ReactiveBEVCanaryOutput(NamedTuple):
    checkpoint: FlyteFile
    metadata: FlyteFile
    checkpoint_uri: str
    checkpoint_sha256: str
    gate_report: FlyteFile


class ReactiveBEVEvaluationOutput(NamedTuple):
    report: FlyteFile
    report_sha256: str
    checkpoint_sha256: str
    checkpoint_epoch: int


class ReactiveKITScenesEvaluationOutput(NamedTuple):
    report: FlyteFile
    report_sha256: str
    checkpoint_sha256: str
    checkpoint_epoch: int


class ReactiveKITScenesEvaluationWorkflowOutput(NamedTuple):
    report: FlyteFile
    report_sha256: str
    checkpoint_sha256: str
    checkpoint_epoch: int
    mlflow_run_id: str
    registered_model_name: str
    registered_model_version: str


class ReactiveBEVPublicationOutput(NamedTuple):
    mlflow_run_id: str
    registered_model_name: str
    registered_model_version: str


class ReactivePolicyPublicationOutput(NamedTuple):
    mlflow_run_id: str
    registered_model_name: str
    registered_model_version: str


class ReactiveBEVEvaluationWorkflowOutput(NamedTuple):
    report: FlyteFile
    report_sha256: str
    checkpoint_sha256: str
    checkpoint_epoch: int
    mlflow_run_id: str
    registered_model_name: str
    registered_model_version: str


def _head_pod_template() -> PodTemplate:
    return PodTemplate(
        primary_container_name="ray-head",
        labels={"auto-e2e.training/role": "ray-head"},
        annotations={"karpenter.sh/do-not-disrupt": "true"},
        pod_spec=V1PodSpec(
            service_account_name="default",
            containers=[
                V1Container(
                    name="ray-head",
                    resources=V1ResourceRequirements(
                        requests={"cpu": "2", "memory": "16Gi"},
                        limits={"cpu": "2", "memory": "16Gi"},
                    ),
                    volume_mounts=[
                        V1VolumeMount(
                            name="dshm",
                            mount_path="/dev/shm",
                        ),
                    ],
                ),
            ],
            volumes=[
                V1Volume(
                        name="dshm",
                        empty_dir=V1EmptyDirVolumeSource(
                            medium="Memory",
                            size_limit="8Gi",
                        ),
                ),
            ],
        ),
    )


def _bev_evaluation_pod_template() -> PodTemplate:
    return PodTemplate(
        primary_container_name="primary",
        annotations={"karpenter.sh/do-not-disrupt": "true"},
        pod_spec=V1PodSpec(
            service_account_name="default",
            node_selector={"workload-type": "gpu-validation"},
            tolerations=[
                V1Toleration(
                    key="nvidia.com/gpu",
                    operator="Exists",
                    effect="NoSchedule",
                ),
            ],
            containers=[
                V1Container(
                    name="primary",
                    volume_mounts=[
                        V1VolumeMount(
                            name="dshm",
                            mount_path="/dev/shm",
                        ),
                    ],
                ),
            ],
            volumes=[
                V1Volume(
                    name="dshm",
                    empty_dir=V1EmptyDirVolumeSource(
                        medium="Memory",
                        size_limit="16Gi",
                    ),
                ),
            ],
        ),
    )


def _worker_pod_template(
    *,
    workload_type: str,
    cpu: str,
    memory: str,
    shm_size: str,
    gpu_count: int,
    ephemeral_storage: str | None = None,
    require_distinct_hosts: bool = True,
    protect_from_disruption: bool = True,
) -> PodTemplate:
    worker_labels = {
        "auto-e2e.training/role": "ray-gpu-worker",
        "auto-e2e.training/workload-type": workload_type,
    }
    affinity = None
    if require_distinct_hosts:
        affinity = V1Affinity(
            pod_affinity=V1PodAffinity(
                required_during_scheduling_ignored_during_execution=[
                    V1PodAffinityTerm(
                        label_selector=V1LabelSelector(
                            match_expressions=[
                                V1LabelSelectorRequirement(
                                    key=(
                                        "auto-e2e.training/workload-type"
                                    ),
                                    operator="In",
                                    values=[workload_type],
                                ),
                            ],
                        ),
                        topology_key="topology.kubernetes.io/zone",
                    ),
                ],
            ),
            pod_anti_affinity=V1PodAntiAffinity(
                required_during_scheduling_ignored_during_execution=[
                    V1PodAffinityTerm(
                        label_selector=V1LabelSelector(
                            match_expressions=[
                                V1LabelSelectorRequirement(
                                    key=(
                                        "auto-e2e.training/workload-type"
                                    ),
                                    operator="In",
                                    values=[workload_type],
                                ),
                            ],
                        ),
                        topology_key="kubernetes.io/hostname",
                    ),
                ],
            ),
        )
    resources = {
        "cpu": cpu,
        "memory": memory,
        "nvidia.com/gpu": str(gpu_count),
    }
    if ephemeral_storage is not None:
        resources["ephemeral-storage"] = ephemeral_storage
    return PodTemplate(
        primary_container_name="ray-worker",
        labels=worker_labels,
        annotations=(
            {"karpenter.sh/do-not-disrupt": "true"}
            if protect_from_disruption
            else {}
        ),
        pod_spec=V1PodSpec(
            service_account_name="default",
            node_selector={"workload-type": workload_type},
            tolerations=[
                V1Toleration(
                    key="nvidia.com/gpu",
                    operator="Exists",
                    effect="NoSchedule",
                ),
            ],
            affinity=affinity,
            containers=[
                V1Container(
                    name="ray-worker",
                    env=[
                        V1EnvVar(name="NCCL_DEBUG", value="INFO"),
                        V1EnvVar(
                            name="TORCH_DISTRIBUTED_DEBUG",
                            value="INFO",
                        ),
                    ],
                    resources=V1ResourceRequirements(
                        requests=dict(resources),
                        limits=dict(resources),
                    ),
                    volume_mounts=[
                        V1VolumeMount(
                            name="dshm",
                            mount_path="/dev/shm",
                        ),
                    ],
                ),
            ],
            volumes=[
                V1Volume(
                    name="dshm",
                    empty_dir=V1EmptyDirVolumeSource(
                        medium="Memory",
                        size_limit=shm_size,
                    ),
                ),
            ],
        ),
    )


def _ray_job_config(
    pod_replicas: int,
    *,
    worker_workload_type: str,
    worker_cpu: str = "4",
    worker_memory: str = "16Gi",
    worker_shm_size: str = "8Gi",
    gpus_per_pod: int = 1,
    worker_ephemeral_storage: str | None = None,
    require_distinct_hosts: bool = True,
    protect_workers_from_disruption: bool = True,
) -> RayJobConfig:
    return RayJobConfig(
        head_node_config=HeadNodeConfig(
            ray_start_params={
                "dashboard-host": "0.0.0.0",
                "num-cpus": "0",
            },
            pod_template=_head_pod_template(),
        ),
        worker_node_config=[
            WorkerNodeConfig(
                group_name="gpu-workers",
                replicas=pod_replicas,
                min_replicas=pod_replicas,
                max_replicas=pod_replicas,
                ray_start_params={
                    "num-cpus": worker_cpu,
                    "num-gpus": str(gpus_per_pod),
                },
                pod_template=_worker_pod_template(
                    workload_type=worker_workload_type,
                    cpu=worker_cpu,
                    memory=worker_memory,
                    shm_size=worker_shm_size,
                    gpu_count=gpus_per_pod,
                    ephemeral_storage=worker_ephemeral_storage,
                    require_distinct_hosts=require_distinct_hosts,
                    protect_from_disruption=(
                        protect_workers_from_disruption
                    ),
                ),
            ),
        ],
        enable_autoscaling=False,
        address="auto",
        shutdown_after_job_finishes=True,
        ttl_seconds_after_finished=300,
    )


RAY_1 = _ray_job_config(
    1,
    worker_workload_type="gpu-validation",
    worker_cpu="3",
    worker_memory="24Gi",
    worker_shm_size="4Gi",
    worker_ephemeral_storage="100Gi",
)
RAY_2 = _ray_job_config(
    2,
    worker_workload_type="gpu-validation",
    worker_cpu="3",
    worker_memory="12Gi",
    worker_shm_size="4Gi",
)
RAY_4 = _ray_job_config(
    1,
    worker_workload_type="p5en-capacity-block",
    worker_cpu="48",
    worker_memory="512Gi",
    worker_shm_size="64Gi",
    gpus_per_pod=4,
    worker_ephemeral_storage="500Gi",
    require_distinct_hosts=False,
)
RAY_REACTIVE_4 = _ray_job_config(
    1,
    worker_workload_type="p5en-capacity-block",
    worker_cpu="48",
    worker_memory="512Gi",
    worker_shm_size="64Gi",
    gpus_per_pod=4,
    worker_ephemeral_storage="500Gi",
    require_distinct_hosts=False,
)
RAY_8 = _ray_job_config(
    1,
    worker_workload_type="p5en-capacity-block",
    worker_cpu="96",
    worker_memory="1Ti",
    worker_shm_size="128Gi",
    gpus_per_pod=8,
    worker_ephemeral_storage="500Gi",
    require_distinct_hosts=False,
)


@task(
    task_config=RAY_4,
    container_image=TRAINING_IMAGE,
    retries=1,
    timeout=P5EN_SMOKE_MAX_RUNTIME,
    labels={
        "kueue.x-k8s.io/queue-name": "p5en-capacity-block",
        "kueue.x-k8s.io/priority-class": "research-low",
    },
    environment=RAY_TASK_ENVIRONMENT,
)
def ray_ddp_smoke_4(
    capacity_block_end_utc: str,
    steps: int = 4,
) -> RaySmokeOutput:
    from distributed_training.ray_smoke import run_smoke

    context = current_context()
    execution_name = (
        context.execution_id.name
        if context.execution_id is not None
        else "local"
    )
    run_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "-",
        f"{execution_name}-ray-ddp-smoke-4",
    )
    result = run_smoke(
        num_workers=4,
        steps=steps,
        storage_path=RAY_STORAGE_PATH,
        run_name=run_name,
        capacity_block_end_utc=capacity_block_end_utc,
    )
    report_path = Path("/tmp/ray-ddp-smoke/report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return RaySmokeOutput(report=FlyteFile(str(report_path)))


def _flyte_remote_uri(value: FlyteDirectory | FlyteFile) -> str:
    remote_source = str(getattr(value, "remote_source", "") or "")
    uri = remote_source or str(value)
    if not uri.startswith("s3://"):
        raise ValueError(
            "distributed Ray workers require immutable S3 inputs"
        )
    return uri.rstrip("/")


def _reactive_worker_cpus(num_workers: int) -> int:
    """Allocate the equal CPU share available to each Ray actor."""
    return 3 if num_workers <= 2 else 12


def _reactive_run_name(
    *,
    execution_name: str,
    stage: str,
    num_workers: int,
) -> str:
    return re.sub(
        r"[^a-zA-Z0-9_-]",
        "-",
        f"{execution_name}-{stage}-ray-{num_workers}-full",
    )


def _reactive_mlflow_config_sha256(
    config: Mapping[str, Any],
) -> str:
    payload = json.dumps(
        dict(config),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reactive_mlflow_params(
    config: Mapping[str, Any],
    *,
    execution_name: str,
) -> dict[str, Any]:
    return {
        "ctx/flyte_execution_id": execution_name,
        "data/stage": config["stage"],
        "data/source_partition_count": len(config["source_uris"]),
        "model/backbone": config["backbone"],
        "model/freeze_bevformer": config["freeze_bevformer"],
        "model/is_pretrained": config["is_pretrained"],
        "train/bev_encoder_learning_rate": config[
            "bev_encoder_learning_rate"
        ],
        "train/bev_weight": config["bev_weight"],
        "train/checkpoint_interval_steps": config[
            "checkpoint_interval_steps"
        ],
        "train/epochs": config["epochs"],
        "train/gradient_accumulation_steps": config[
            "gradient_accumulation_steps"
        ],
        "train/learning_rate": config["learning_rate"],
        "train/num_loader_workers": config["num_loader_workers"],
        "train/per_rank_batch_size": config["per_rank_batch_size"],
        "train/precision": config["precision"],
        "train/route_weight": config["route_weight"],
        "train/seed": config["training_seed"],
        "train/training_scope": config["training_scope"],
        "train/trajectory_weight": config["trajectory_weight"],
        "train/validation_fraction": config["val_fraction"],
        "train/validation_sample_limit": config[
            "validation_sample_limit"
        ],
        "train/weight_decay": config["weight_decay"],
        "train/world_size": config["num_workers"],
    }


def _reactive_mlflow_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, float]:
    result = {}
    for name, value in metrics.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            continue
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            result[str(name)] = numeric_value
    return result


def _start_reactive_mlflow_run(
    config: Mapping[str, Any],
    *,
    execution_name: str,
    run_name: str,
):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.set_experiment(REACTIVE_MLFLOW_EXPERIMENT)
    client = MlflowClient()
    config_sha256 = _reactive_mlflow_config_sha256(config)
    matches = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.reactive_run_key = '{run_name}'",
        max_results=2,
    )
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple MLflow runs use Reactive key {run_name}"
        )
    if matches:
        run = matches[0]
        recorded_sha256 = run.data.tags.get(
            "reactive_config_sha256"
        )
        if recorded_sha256 != config_sha256:
            raise ValueError(
                "Reactive MLflow run config differs from the Flyte retry"
            )
        run_id = run.info.run_id
        client.set_tag(run_id, "task_status", "RUNNING")
        client.set_tag(run_id, "flyte_retry_reused", "true")
        return client, run_id

    tags = {
        "mlflow.runName": run_name,
        "pipeline": REACTIVE_MLFLOW_EXPERIMENT,
        "reactive_run_key": run_name,
        "reactive_config_sha256": config_sha256,
        "flyte_execution_id": execution_name,
        "stage": str(config["stage"]),
        "training_scope": str(config["training_scope"]),
        "task_status": "RUNNING",
    }
    run = client.create_run(
        experiment_id=experiment.experiment_id,
        start_time=int(time.time() * 1000),
        tags=tags,
    )
    run_id = run.info.run_id
    for name, value in _reactive_mlflow_params(
        config,
        execution_name=execution_name,
    ).items():
        client.log_param(run_id, name, value)
    return client, run_id


def _log_reactive_mlflow_result(
    client,
    run_id: str,
    result: Mapping[str, Any],
    *,
    metadata_path: Path,
) -> None:
    _log_reactive_mlflow_history(
        client,
        run_id,
        result["history"],
    )
    selected_metrics = result["metrics"]
    final_metrics = result["final_metrics"]
    tags = {
        "checkpoint_sha256": str(
            selected_metrics["checkpoint_sha256"]
        ),
        "checkpoint_uri": str(result["checkpoint_file_uri"]),
        "dataset_manifest_sha256": str(
            selected_metrics["dataset_manifest_sha256"]
        ),
        "final_checkpoint_sha256": str(
            final_metrics["checkpoint_sha256"]
        ),
        "selected_checkpoint_epoch": str(result["selected_epoch"]),
        "task_status": "FINISHED",
    }
    for name, value in tags.items():
        client.set_tag(run_id, name, value)
    client.log_artifact(
        run_id,
        str(metadata_path),
        artifact_path="training",
    )
    client.set_terminated(run_id, status="FINISHED")


def _log_reactive_mlflow_history(
    client,
    run_id: str,
    history: list[Mapping[str, Any]],
) -> None:
    from mlflow.entities import Metric

    timestamp = int(time.time() * 1000)
    for row in history:
        step = int(row["epoch"])
        entries = [
            Metric(
                key=name,
                value=value,
                timestamp=timestamp,
                step=step,
            )
            for name, value in _reactive_mlflow_metrics(row).items()
        ]
        for offset in range(0, len(entries), 500):
            client.log_batch(
                run_id,
                metrics=entries[offset : offset + 500],
            )


def _recover_reactive_mlflow_checkpoint(
    config: Mapping[str, Any],
    *,
    s3_client=None,
) -> dict[str, Any] | None:
    storage_path = str(config["storage_path"]).rstrip("/")
    parsed = urlparse(storage_path)
    if parsed.scheme != "s3" or not parsed.netloc:
        return None
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")
    prefix = "/".join(
        part
        for part in (
            parsed.path.strip("/"),
            str(config["run_name"]),
        )
        if part
    )

    def read_json(key: str):
        body = s3_client.get_object(
            Bucket=parsed.netloc,
            Key=key,
        )["Body"]
        return json.loads(body.read())

    try:
        snapshot = read_json(
            f"{prefix}/checkpoint_manager_snapshot.json"
        )
    except Exception:
        return None
    latest = snapshot.get("latest_checkpoint_result")
    if not isinstance(latest, Mapping):
        return None
    checkpoint_directory = latest.get("checkpoint_dir_name")
    metrics = latest.get("metrics")
    if (
        not isinstance(checkpoint_directory, str)
        or not checkpoint_directory
        or not isinstance(metrics, Mapping)
    ):
        return None
    history = read_json(
        f"{prefix}/{checkpoint_directory}/history.json"
    )
    if (
        not isinstance(history, list)
        or any(not isinstance(row, Mapping) for row in history)
    ):
        raise ValueError("Reactive S3 recovery history is invalid")
    return {
        "checkpoint_uri": (
            f"{storage_path}/{config['run_name']}/"
            f"{checkpoint_directory}/checkpoint.pt"
        ),
        "history": history,
        "metrics": dict(metrics),
    }


def _mark_reactive_mlflow_failed(
    client,
    run_id: str,
    error: Exception,
    *,
    config: Mapping[str, Any],
    run_name: str,
) -> None:
    failure_path = (
        Path("/tmp/reactive-ray")
        / run_name
        / "failure.json"
    )
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(
        json.dumps(
            {
                "error_message": str(error),
                "error_type": type(error).__name__,
                "run_name": run_name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    client.set_tag(run_id, "error_type", type(error).__name__)
    client.set_tag(run_id, "error_message", str(error)[:4000])
    client.set_tag(run_id, "task_status", "FAILED")
    try:
        recovered = _recover_reactive_mlflow_checkpoint(config)
        if recovered is not None:
            _log_reactive_mlflow_history(
                client,
                run_id,
                recovered["history"],
            )
            recovered_metrics = recovered["metrics"]
            recovered_tags = {
                "completed_epoch": str(
                    len(recovered["history"])
                ),
                "latest_recovery_checkpoint_sha256": str(
                    recovered_metrics["checkpoint_sha256"]
                ),
                "latest_recovery_checkpoint_uri": str(
                    recovered["checkpoint_uri"]
                ),
                "latest_recovery_epoch": str(
                    recovered_metrics["epoch"]
                ),
                "latest_recovery_optimizer_step": str(
                    recovered_metrics.get(
                        "executed_optimizer_steps",
                        0,
                    )
                ),
            }
            for name, value in recovered_tags.items():
                client.set_tag(run_id, name, value)
    except Exception as recovery_error:
        client.set_tag(
            run_id,
            "history_recovery_error",
            str(recovery_error)[:4000],
        )
    client.log_artifact(
        run_id,
        str(failure_path),
        artifact_path="training",
    )
    client.set_terminated(run_id, status="FAILED")


def _reactive_bev_evaluation_metrics(
    report: Mapping[str, Any],
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for child_name, child_value in value.items():
                child_prefix = (
                    f"{prefix}/{child_name}"
                    if prefix
                    else str(child_name)
                )
                visit(child_prefix, child_value)
            return
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            return
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            metrics[f"eval/{prefix}"] = numeric_value

    visit("", report)
    return metrics


def _register_reactive_bev_model_version(
    client,
    *,
    run_id: str,
    checkpoint_artifact_uri: str,
    checkpoint_source_uri: str,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
    report: Mapping[str, Any],
    report_sha256: str,
    source_training_mlflow_run_id: str,
) -> str:
    try:
        client.get_registered_model(REACTIVE_BEV_REGISTERED_MODEL)
    except Exception:
        try:
            client.create_registered_model(
                REACTIVE_BEV_REGISTERED_MODEL
            )
        except Exception:
            client.get_registered_model(
                REACTIVE_BEV_REGISTERED_MODEL
            )

    version = None
    for existing in client.search_model_versions(
        f"name='{REACTIVE_BEV_REGISTERED_MODEL}'"
    ):
        existing_tags = getattr(existing, "tags", {}) or {}
        if (
            str(existing_tags.get("checkpoint_sha256", ""))
            == checkpoint_sha256
        ) or (
            str(getattr(existing, "run_id", "")) == run_id
            and str(getattr(existing, "source", ""))
            == checkpoint_artifact_uri
        ):
            version = str(existing.version)
            break
    if version is None:
        registered = client.create_model_version(
            name=REACTIVE_BEV_REGISTERED_MODEL,
            source=checkpoint_artifact_uri,
            run_id=run_id,
        )
        version = str(registered.version)

    dataset = str(report["dataset"])
    dataset_tag_prefix = re.sub(
        r"[^a-z0-9]+",
        "_",
        dataset.lower(),
    ).strip("_")
    version_tags = {
        "checkpoint_artifact_uri": checkpoint_artifact_uri,
        "checkpoint_epoch": str(checkpoint_epoch),
        "checkpoint_s3_uri": checkpoint_source_uri,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_dataset": dataset,
        "evaluation_report_sha256": report_sha256,
        "evaluation_schema_version": str(
            report["schema_version"]
        ),
        "model_role": "bev_segmentation_candidate",
        (
            f"{dataset_tag_prefix}_macro_average_precision"
        ): str(
            report["macro_average_precision_supported_classes"]
        ),
        (
            f"{dataset_tag_prefix}_evaluation_sample_count"
        ): str(report["sample_count"]),
    }
    if source_training_mlflow_run_id:
        version_tags["source_training_mlflow_run_id"] = (
            source_training_mlflow_run_id
        )
    for name, value in version_tags.items():
        client.set_model_version_tag(
            REACTIVE_BEV_REGISTERED_MODEL,
            version,
            name,
            value,
        )
    return version


def _register_reactive_policy_model_version(
    client,
    *,
    checkpoint_uri: str,
    checkpoint_sha256: str,
    evaluation_run_id: str,
    expected_model_version: str = "",
) -> str:
    """Resolve one checkpoint to exactly one policy model version."""
    try:
        client.get_registered_model(REACTIVE_POLICY_REGISTERED_MODEL)
    except Exception:
        try:
            client.create_registered_model(
                REACTIVE_POLICY_REGISTERED_MODEL
            )
        except Exception:
            client.get_registered_model(
                REACTIVE_POLICY_REGISTERED_MODEL
            )

    if expected_model_version:
        existing = client.get_model_version(
            REACTIVE_POLICY_REGISTERED_MODEL,
            expected_model_version,
        )
        tags = getattr(existing, "tags", {}) or {}
        existing_sha256 = str(tags.get("checkpoint_sha256", ""))
        existing_source = str(getattr(existing, "source", "") or "")
        if (
            existing_source != checkpoint_uri
            or existing_sha256 != checkpoint_sha256
        ):
            raise RuntimeError(
                "expected policy model version does not match the checkpoint"
            )
        return str(existing.version)

    matches = []
    for existing in client.search_model_versions(
        f"name='{REACTIVE_POLICY_REGISTERED_MODEL}'"
    ):
        tags = getattr(existing, "tags", {}) or {}
        existing_sha256 = str(tags.get("checkpoint_sha256", ""))
        existing_source = str(getattr(existing, "source", "") or "")
        source_matches = existing_source == checkpoint_uri
        digest_matches = existing_sha256 == checkpoint_sha256
        if source_matches and existing_sha256 not in (
            "",
            checkpoint_sha256,
        ):
            raise RuntimeError(
                "policy checkpoint source has a conflicting SHA-256"
            )
        if source_matches or digest_matches:
            matches.append(existing)

    if len(matches) > 1:
        raise RuntimeError(
            "policy checkpoint resolves to multiple model versions"
        )
    if matches:
        return str(matches[0].version)

    registered = client.create_model_version(
        name=REACTIVE_POLICY_REGISTERED_MODEL,
        source=checkpoint_uri,
        run_id=evaluation_run_id,
    )
    return str(registered.version)


def _validate_kitscenes_publication_binding(
    report_payload: dict[str, object],
    *,
    expected_model_version: str,
) -> None:
    source_split = report_payload.get("source_split")
    evaluation_role = report_payload.get("evaluation_role")
    if source_split == "val" and (
        evaluation_role != "official_val_camera_map_route"
    ):
        return
    if source_split not in {"val", "test"}:
        return
    if not expected_model_version:
        raise ValueError(
            "KITScenes official publication requires an existing model "
            "version"
        )
    metrics = report_payload.get("metrics")
    route_metrics = (
        metrics.get("route")
        if isinstance(metrics, dict)
        else None
    )
    evaluation_batch_size = report_payload.get("evaluation_batch_size")
    has_integral_batch_size = (
        isinstance(evaluation_batch_size, int)
        and not isinstance(evaluation_batch_size, bool)
    )
    if source_split == "val":
        valid = (
            evaluation_role == "official_val_camera_map_route"
            and report_payload.get("dataset") == KITSCENES_REPO_ID
            and report_payload.get("dataset_version")
            == KITSCENES_OFFICIAL_VAL_DATASET_VERSION
            and report_payload.get("source_revision")
            == KITSCENES_DATA_REVISION
            and report_payload.get("input_track") == "camera_map_route"
            and report_payload.get("partition_count")
            == KITSCENES_OFFICIAL_VAL_SCENE_COUNT
            and report_payload.get("expected_partition_count")
            == KITSCENES_OFFICIAL_VAL_SCENE_COUNT
            and report_payload.get("scene_uid_sha256")
            == KITSCENES_OFFICIAL_VAL_SCENE_UID_SHA256
            and report_payload.get("inference_cache_policy")
            == "stateless"
            and has_integral_batch_size
            and evaluation_batch_size
            == KITSCENES_PRIMARY_VAL_BATCH_SIZE
            and report_payload.get("maximum_labeled_horizon_steps") == 50
            and isinstance(
                report_payload.get("trajectory_inference_policy"),
                dict,
            )
            and report_payload["trajectory_inference_policy"].get("version")
            == KITSCENES_TRAJECTORY_INFERENCE_POLICY
            and report_payload["trajectory_inference_policy"].get("planner")
            == "gru"
            and report_payload["trajectory_inference_policy"].get(
                "stochastic_noise"
            )
            is False
            and report_payload.get("route_usage_evaluation_policy")
            == _kitscenes_route_usage_evaluation_policy(
                mapless_test=False,
            )
            and isinstance(metrics, dict)
            and isinstance(metrics.get("sample_count"), int)
            and not isinstance(metrics.get("sample_count"), bool)
            and metrics["sample_count"]
            == KITSCENES_OFFICIAL_VAL_SAMPLE_COUNT
            and report_payload.get("expected_sample_count")
            == KITSCENES_OFFICIAL_VAL_SAMPLE_COUNT
            and isinstance(route_metrics, dict)
            and route_metrics.get("route_input_gradient_mean_abs") is None
            and isinstance(
                route_metrics.get("route_zero_sample_count"),
                int,
            )
            and not isinstance(
                route_metrics.get("route_zero_sample_count"),
                bool,
            )
            and route_metrics["route_zero_sample_count"]
            == KITSCENES_OFFICIAL_VAL_SAMPLE_COUNT
            and isinstance(
                route_metrics.get("route_zero_trajectory_delta_m"),
                (int, float),
            )
            and not isinstance(
                route_metrics.get("route_zero_trajectory_delta_m"),
                bool,
            )
            and math.isfinite(
                float(
                    route_metrics["route_zero_trajectory_delta_m"]
                )
            )
        )
    else:
        valid = (
            evaluation_role
            == "official_test_camera_only_missing_map_route"
            and report_payload.get("dataset") == KITSCENES_REPO_ID
            and report_payload.get("dataset_version")
            == KITSCENES_OFFICIAL_TEST_DATASET_VERSION
            and report_payload.get("source_revision")
            == KITSCENES_DATA_REVISION
            and report_payload.get("input_track")
            == "camera_only_missing_map_route"
            and report_payload.get("partition_count")
            == KITSCENES_OFFICIAL_TEST_SCENE_COUNT
            and report_payload.get("expected_partition_count")
            == KITSCENES_OFFICIAL_TEST_SCENE_COUNT
            and report_payload.get("scene_uid_sha256")
            == KITSCENES_OFFICIAL_TEST_SCENE_UID_SHA256
            and report_payload.get("inference_cache_policy")
            == KITSCENES_STATEFUL_CAMERA_FPN_POLICY
            and has_integral_batch_size
            and evaluation_batch_size
            == KITSCENES_STATEFUL_CAMERA_FPN_BATCH_SIZE
            and report_payload.get("maximum_labeled_horizon_steps") == 50
            and isinstance(
                report_payload.get("trajectory_inference_policy"),
                dict,
            )
            and report_payload["trajectory_inference_policy"].get("version")
            == KITSCENES_TRAJECTORY_INFERENCE_POLICY
            and report_payload["trajectory_inference_policy"].get("planner")
            == "gru"
            and report_payload["trajectory_inference_policy"].get(
                "stochastic_noise"
            )
            is False
            and report_payload.get("route_usage_evaluation_policy")
            == _kitscenes_route_usage_evaluation_policy(
                mapless_test=True,
            )
            and isinstance(metrics, dict)
            and metrics.get("sample_count")
            == KITSCENES_OFFICIAL_TEST_SAMPLE_COUNT
        )
    if not valid:
        raise ValueError(
            "KITScenes publication requires the exact official inventory"
        )


def _log_reactive_bev_evaluation_to_mlflow(
    *,
    report: Mapping[str, Any],
    report_path: Path,
    report_sha256: str,
    checkpoint_path: Path,
    checkpoint_source_uri: str,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
    source_training_mlflow_run_id: str,
) -> tuple[str, str]:
    import mlflow
    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.set_experiment(
        REACTIVE_BEV_EVALUATION_MLFLOW_EXPERIMENT
    )
    client = MlflowClient()
    evaluation_key = (
        f"{checkpoint_sha256}:{report['dataset']}:{report_sha256}"
    )
    matches = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.reactive_bev_evaluation_key = "
            f"'{evaluation_key}'"
        ),
        max_results=2,
    )
    if len(matches) > 1:
        raise RuntimeError(
            "multiple MLflow runs use one Reactive BEV evaluation key"
        )
    if matches:
        run_id = matches[0].info.run_id
        client.set_tag(run_id, "flyte_retry_reused", "true")
    else:
        run_name = (
            f"bev-{str(report['dataset']).split('/')[-1]}-"
            f"e{checkpoint_epoch}-{checkpoint_sha256[:12]}"
        )
        tags = {
            "mlflow.runName": run_name,
            "pipeline": REACTIVE_BEV_EVALUATION_MLFLOW_EXPERIMENT,
            "reactive_bev_evaluation_key": evaluation_key,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_s3_uri": checkpoint_source_uri,
            "evaluation_dataset": str(report["dataset"]),
            "evaluation_report_sha256": report_sha256,
            "task_status": "RUNNING",
        }
        if source_training_mlflow_run_id:
            tags["source_training_mlflow_run_id"] = (
                source_training_mlflow_run_id
            )
        run = client.create_run(
            experiment_id=experiment.experiment_id,
            start_time=int(time.time() * 1000),
            tags=tags,
        )
        run_id = run.info.run_id
        params = {
            "data/dataset": report["dataset"],
            "data/split": report["split"],
            "eval/batch_size": report["evaluation_batch_size_counts"],
            "eval/precision": report["evaluation_precision"],
            "eval/probability_bins": report[
                "training_probability_bins"
            ],
            "eval/sample_count": report["sample_count"],
            "model/checkpoint_epoch": checkpoint_epoch,
            "model/checkpoint_sha256": checkpoint_sha256,
            "model/checkpoint_source_uri": checkpoint_source_uri,
        }
        for name, value in params.items():
            client.log_param(run_id, name, str(value)[:500])

    timestamp = int(time.time() * 1000)
    metrics = [
        Metric(
            key=name,
            value=value,
            timestamp=timestamp,
            step=checkpoint_epoch,
        )
        for name, value in _reactive_bev_evaluation_metrics(
            report
        ).items()
    ]
    for offset in range(0, len(metrics), 500):
        client.log_batch(
            run_id,
            metrics=metrics[offset : offset + 500],
        )
    client.log_artifact(
        run_id,
        str(report_path),
        artifact_path="evaluation",
    )
    client.log_artifact(
        run_id,
        str(checkpoint_path),
        artifact_path="model",
    )
    checkpoint_artifact_uri = (
        f"runs:/{run_id}/model/{checkpoint_path.name}"
    )
    version = _register_reactive_bev_model_version(
        client,
        run_id=run_id,
        checkpoint_artifact_uri=checkpoint_artifact_uri,
        checkpoint_source_uri=checkpoint_source_uri,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=checkpoint_epoch,
        report=report,
        report_sha256=report_sha256,
        source_training_mlflow_run_id=(
            source_training_mlflow_run_id
        ),
    )
    client.set_tag(run_id, "registered_model_name", (
        REACTIVE_BEV_REGISTERED_MODEL
    ))
    client.set_tag(run_id, "registered_model_version", version)
    client.set_tag(run_id, "task_status", "FINISHED")
    client.set_terminated(run_id, status="FINISHED")
    return run_id, version


def _run_reactive_stage_task(
    *,
    shards: List[FlyteDirectory],
    stage: str,
    num_workers: int,
    parent_checkpoint: Optional[FlyteFile],
    resume_checkpoint: Optional[FlyteDirectory],
    backbone: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    val_fraction: float,
    num_loader_workers: int,
    per_rank_batch_size: int,
    training_seed: int,
    precision: str,
    gradient_accumulation_steps: int,
    steps_per_epoch: int,
    checkpoint_interval_steps: int,
    shuffle_buffer: int,
    is_pretrained: bool,
    trajectory_weight: float,
    bev_weight: float,
    route_weight: float,
    corridor_pos_weight: float,
    bev_pos_weight_cap: float,
    bev_repeat_frequency_threshold: float,
    bev_max_repeat: int,
    bev_min_positive_samples: int,
    bev_min_positive_cells: int,
    freeze_bevformer: bool,
    validation_sample_limit: int,
    capacity_block_end_utc: str,
    training_scope: str = "multitask",
    bev_encoder_learning_rate: float = 1e-5,
    allow_random_bevformer_init: bool = False,
    allow_single_worker_smoke: bool = False,
    allow_bounded_bev_canary: bool = False,
    parent_profile: str = "",
    parent_checkpoint_sha256: str = "",
    parent_checkpoint_epoch: int = 0,
) -> ReactiveRayOutput:
    from distributed_training.reactive_stage import run_reactive_stage
    from model_components.bevformer_v2_pretrained import (
        BEVFORMER_V2_T8_CHECKPOINT_MIRROR_KEY,
        BEVFORMER_V2_T8_CHECKPOINT_SHA256,
        bevformer_v2_t8_checkpoint_mirror_uri,
    )

    context = current_context()
    execution_name = (
        context.execution_id.name
        if context.execution_id is not None
        else "local"
    )
    run_name = _reactive_run_name(
        execution_name=execution_name,
        stage=stage,
        num_workers=num_workers,
    )
    source_uris = [_flyte_remote_uri(shard) for shard in shards]
    parent_uri = (
        _flyte_remote_uri(parent_checkpoint)
        if parent_checkpoint is not None
        else ""
    )
    resume_uri = (
        _flyte_remote_uri(resume_checkpoint)
        if resume_checkpoint is not None
        else ""
    )
    pretrained_uri = ""
    if is_pretrained and not parent_uri:
        configured_bucket = os.environ.get(
            "AUTO_E2E_CHECKPOINT_BUCKET",
            "",
        ).strip()
        if configured_bucket:
            pretrained_uri = (
                f"s3://{configured_bucket}/"
                f"{BEVFORMER_V2_T8_CHECKPOINT_MIRROR_KEY}"
            )
        else:
            import boto3

            account_id = str(
                boto3.client("sts").get_caller_identity()["Account"]
            )
            pretrained_uri = bevformer_v2_t8_checkpoint_mirror_uri(
                account_id,
                cluster_name=os.environ.get(
                    "AUTO_E2E_CLUSTER_NAME",
                    "auto-e2e-platform",
                ),
            )
    training_config = {
        "allow_bounded_bev_canary": allow_bounded_bev_canary,
        "allow_random_bevformer_init": allow_random_bevformer_init,
        "allow_single_worker_smoke": allow_single_worker_smoke,
        "backbone": backbone,
        "bev_ap_bins": 1024,
        "bev_max_repeat": bev_max_repeat,
        "bev_min_positive_cells": bev_min_positive_cells,
        "bev_min_positive_samples": bev_min_positive_samples,
        "bev_pos_weight_cap": bev_pos_weight_cap,
        "bev_repeat_frequency_threshold": (
            bev_repeat_frequency_threshold
        ),
        "bev_weight": bev_weight,
        "bev_encoder_learning_rate": bev_encoder_learning_rate,
        "bevformer_pretrained_checkpoint_sha256": (
            BEVFORMER_V2_T8_CHECKPOINT_SHA256
        ),
        "bevformer_pretrained_checkpoint_uri": pretrained_uri,
        "corridor_pos_weight": corridor_pos_weight,
        "capacity_block_end_utc": capacity_block_end_utc,
        "checkpoint_interval_steps": checkpoint_interval_steps,
        "epochs": epochs,
        "grad_clip": grad_clip,
        "gradient_accumulation_steps": (
            gradient_accumulation_steps
        ),
        "is_pretrained": is_pretrained,
        "freeze_bevformer": freeze_bevformer,
        "learning_rate": learning_rate,
        "local_cache_root": "/tmp/auto-e2e-reactive",
        "num_loader_workers": num_loader_workers,
        "num_workers": num_workers,
        "parent_checkpoint_uri": parent_uri,
        "parent_profile": parent_profile,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_checkpoint_epoch": parent_checkpoint_epoch,
        "resume_checkpoint_uri": resume_uri,
        "per_rank_batch_size": per_rank_batch_size,
        "precision": precision,
        "route_weight": route_weight,
        "run_name": run_name,
        "selection_ade_regression_margin_m": 0.5,
        "selection_ade_scale_m": 5.0,
        "shuffle_buffer": shuffle_buffer,
        "source_uris": source_uris,
        "stage": stage,
        "steps_per_epoch": steps_per_epoch,
        "storage_path": RAY_STORAGE_PATH,
        "training_seed": training_seed,
        "training_scope": training_scope,
        "trajectory_weight": trajectory_weight,
        "use_gpu": True,
        "val_fraction": val_fraction,
        "validation_sample_limit": validation_sample_limit,
        "weight_decay": weight_decay,
        "worker_cpus": _reactive_worker_cpus(num_workers),
    }
    mlflow_client, mlflow_run_id = _start_reactive_mlflow_run(
        training_config,
        execution_name=execution_name,
        run_name=run_name,
    )
    try:
        result = run_reactive_stage(training_config)
    except Exception as error:
        _mark_reactive_mlflow_failed(
            mlflow_client,
            mlflow_run_id,
            error,
            config=training_config,
            run_name=run_name,
        )
        raise
    result["tracking"] = {
        "mlflow_experiment": REACTIVE_MLFLOW_EXPERIMENT,
        "mlflow_run_id": mlflow_run_id,
    }
    metadata_path = (
        Path("/tmp/reactive-ray")
        / run_name
        / "metadata.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    _log_reactive_mlflow_result(
        mlflow_client,
        mlflow_run_id,
        result,
        metadata_path=metadata_path,
    )
    metrics = result["metrics"]
    return ReactiveRayOutput(
        checkpoint=FlyteFile(result["checkpoint_file_uri"]),
        metadata=FlyteFile(str(metadata_path)),
        checkpoint_uri=str(result["checkpoint_file_uri"]),
        checkpoint_sha256=str(metrics["checkpoint_sha256"]),
    )


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="2", mem="8Gi"),
)
def build_reactive_canary_dataset(stage: str) -> FlyteDirectory:
    """Create deterministic production-schema shards for the GPU gate."""
    import tempfile
    from pathlib import Path

    from distributed_training.reactive_canary_data import (
        write_reactive_canary_dataset,
    )
    from training.reactive_multitask import ReactiveTrainingStage

    training_stage = ReactiveTrainingStage(stage)
    output = Path(tempfile.mkdtemp(prefix=f"reactive-{stage}-"))
    write_reactive_canary_dataset(
        output,
        stage=training_stage,
        shard_count=2,
        train_samples_per_shard=2,
        validation_samples_per_shard=1,
    )
    return FlyteDirectory(str(output))


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi"),
)
def verify_reactive_canary_training(
    stage_a_metadata: FlyteFile,
    stage_b_metadata: FlyteFile,
) -> FlyteFile:
    """Fail when the real-model two-stage GPU canary is not learning."""
    import math
    import tempfile
    from pathlib import Path

    reports = {}
    for stage_name, source in (
        ("stage_a", stage_a_metadata),
        ("stage_b", stage_b_metadata),
    ):
        payload = json.loads(Path(source.download()).read_text())
        history = payload.get("history")
        if not isinstance(history, list) or len(history) < 2:
            raise ValueError(
                f"{stage_name} canary needs at least two reported epochs"
            )
        required = (
            "train_bev_segmentation",
            "train_bev_segmentation_bce",
            "train_bev_segmentation_dice",
            "train_route_reconstruction",
            "train_total",
            "train_trajectory",
            "train_gradient_front_gate_pre_clip_norm",
            "validation_ade_6p4s_m",
            "validation_selection_score",
        )
        for epoch in history:
            if any(
                name not in epoch
                or not math.isfinite(float(epoch[name]))
                for name in required
            ):
                raise ValueError(
                    f"{stage_name} canary emitted non-finite metrics"
                )
        reports[stage_name] = history

    stage_a = reports["stage_a"]
    stage_b = reports["stage_b"]
    if float(stage_a[0]["train_bev_segmentation"]) <= 0.0:
        raise ValueError("Stage A canary did not execute the BEV loss")
    if any(
        float(epoch["train_gradient_front_gate_pre_clip_norm"]) <= 0.0
        for epoch in stage_a
    ):
        raise ValueError(
            "Stage A canary did not use the native front residual"
        )
    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_CLASSES,
    )

    for epoch in stage_a:
        for class_name in BEV_SEGMENTATION_CLASSES:
            for suffix in (
                "average_precision",
                "positive_cells",
                "recall_at_0p5",
            ):
                name = f"validation_bev_{class_name}_{suffix}"
                if (
                    name not in epoch
                    or not math.isfinite(float(epoch[name]))
                ):
                    raise ValueError(
                        f"Stage A canary omitted class metric {name}"
                    )
            if float(
                epoch[
                    f"validation_bev_{class_name}_positive_cells"
                ]
            ) <= 0.0:
                raise ValueError(
                    f"Stage A canary has no {class_name} positives"
                )
    if all(
        abs(float(stage_a[0][f"bev_pos_weight_{index}"]) - 1.0)
        <= 1e-12
        for index in range(8)
    ):
        raise ValueError("Stage A canary derived only unit BEV weights")
    if any(
        abs(float(epoch["train_bev_segmentation"])) > 1e-12
        for epoch in stage_b
    ):
        raise ValueError("Stage B canary executed the BEV loss")
    initial_total = float(stage_a[0]["train_total"])
    minimum_later_total = min(
        float(epoch["train_total"]) for epoch in stage_a[1:]
    )
    if minimum_later_total >= initial_total:
        raise ValueError(
            "Stage A canary total loss did not decrease: "
            f"initial={initial_total} later_min={minimum_later_total}"
        )
    initial_bev = float(stage_a[0]["train_bev_segmentation"])
    minimum_later_bev = min(
        float(epoch["train_bev_segmentation"])
        for epoch in stage_a[1:]
    )
    if minimum_later_bev >= initial_bev:
        raise ValueError(
            "Stage A canary BEV loss did not decrease: "
            f"initial={initial_bev} later_min={minimum_later_bev}"
        )
    initial_bev_bce = float(
        stage_a[0]["train_bev_segmentation_bce"]
    )
    minimum_later_bev_bce = min(
        float(epoch["train_bev_segmentation_bce"])
        for epoch in stage_a[1:]
    )
    initial_bev_dice = float(
        stage_a[0]["train_bev_segmentation_dice"]
    )
    minimum_later_bev_dice = min(
        float(epoch["train_bev_segmentation_dice"])
        for epoch in stage_a[1:]
    )

    report = {
        "schema_version": "reactive_ddp_canary_report_v2",
        "stage_a_initial_bev": initial_bev,
        "stage_a_initial_bev_bce": initial_bev_bce,
        "stage_a_initial_bev_dice": initial_bev_dice,
        "stage_a_minimum_later_bev": minimum_later_bev,
        "stage_a_minimum_later_bev_bce": minimum_later_bev_bce,
        "stage_a_minimum_later_bev_dice": minimum_later_bev_dice,
        "stage_a_initial_total": initial_total,
        "stage_a_front_gate_gradient_verified": True,
        "stage_a_minimum_later_total": minimum_later_total,
        "stage_a_epochs": len(stage_a),
        "stage_b_epochs": len(stage_b),
        "stage_b_bev_loss_disabled": True,
        "thresholds_pass": True,
    }
    output = (
        Path(tempfile.mkdtemp(prefix="reactive-canary-report-"))
        / "report.json"
    )
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return FlyteFile(str(output))


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi"),
)
def verify_reactive_bev_canary_training(
    metadata: FlyteFile,
) -> FlyteFile:
    """Fail unless a real-data BEV-only canary is finite and learning."""
    import math
    import tempfile
    from pathlib import Path

    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_CLASSES,
    )
    payload = json.loads(Path(metadata.download()).read_text())
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != 2:
        raise ValueError("BEV canary requires exactly two reported epochs")
    epoch_numbers = []
    for epoch in history:
        raw_epoch = epoch.get("epoch")
        try:
            epoch_number = int(raw_epoch)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "BEV canary emitted an invalid epoch number"
            ) from error
        if isinstance(raw_epoch, bool) or float(raw_epoch) != epoch_number:
            raise ValueError("BEV canary emitted an invalid epoch number")
        epoch_numbers.append(epoch_number)
    expected_epoch_numbers = list(range(1, len(history) + 1))
    if epoch_numbers != expected_epoch_numbers:
        raise ValueError(
            "BEV canary epochs must be unique and contiguous from one: "
            f"actual={epoch_numbers}"
        )

    common_metrics = (
        "bounded_bev_canary",
        "bev_rank_max_full_microbatch_capacity",
        "bev_rank_max_drop_last_fraction",
        "bev_rank_max_importance_scale",
        "bev_rank_max_optimizer_tail_fraction",
        "bev_rank_max_truncation_fraction",
        "bev_rank_min_full_microbatch_capacity",
        "bev_rank_min_drop_last_fraction",
        "bev_rank_min_importance_scale",
        "bev_rank_min_optimizer_tail_fraction",
        "bev_rank_min_truncation_fraction",
        "train_bev_segmentation",
        "train_bev_segmentation_bce",
        "train_bev_segmentation_dice",
        "train_bev_logit_gradient_diagnostic_batches",
        "train_gradient_camera_pre_clip_norm",
        "train_gradient_front_gate_pre_clip_norm",
        "train_loader_restarts",
        "validation_bev_all_classes_supported",
        "validation_bev_dynamic_macro_ap_lift",
        "validation_bev_min_ap_lift",
        "validation_bev_static_macro_ap_lift",
        "validation_selection_score",
    )
    class_suffixes = (
        "ap_lift",
        "ap_lift_bootstrap_lower_95",
        "ap_lift_bootstrap_upper_95",
        "average_precision",
        "best_iou_on_validation_set",
        "best_iou_precision_on_validation_set",
        "best_iou_recall_on_validation_set",
        "best_iou_threshold_on_validation_set",
        "positive_cells",
        "positive_prevalence",
        "supported",
    )
    for epoch in history:
        required = list(common_metrics)
        required.extend(
            f"bev_pos_weight_{index}"
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        )
        required.extend(
            f"bev_class_weight_{index}"
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        )
        required.extend(
            f"bev_positive_pair_frequency_{index}"
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        )
        required.extend(
            f"bev_repeat_factor_{index}"
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        )
        required.extend(
            f"train_bev_logit_gradient_l1_class_{index}"
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        )
        required.extend(
            f"train_bev_logit_gradient_share_class_{index}"
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        )
        required.extend(
            f"validation_bev_{class_name}_{suffix}"
            for class_name in BEV_SEGMENTATION_CLASSES
            for suffix in class_suffixes
        )
        if any(
            name not in epoch or not math.isfinite(float(epoch[name]))
            for name in required
        ):
            raise ValueError("BEV canary emitted missing or non-finite metrics")
        if "validation_ade_6p4s_m" in epoch:
            raise ValueError("BEV-only canary unexpectedly evaluated trajectory")
        if float(epoch["validation_bev_all_classes_supported"]) != 1.0:
            raise ValueError("BEV canary validation omitted a class")
        if float(epoch["bounded_bev_canary"]) != 1.0:
            raise ValueError("BEV canary did not use the bounded-run contract")
        if float(epoch["train_gradient_camera_pre_clip_norm"]) <= 0.0:
            raise ValueError("BEV canary did not update camera BEV parameters")
        if float(epoch["train_gradient_front_gate_pre_clip_norm"]) <= 0.0:
            raise ValueError("BEV canary did not update the Front residual gate")
        if float(epoch["train_loader_restarts"]) != 0.0:
            raise ValueError("BEV canary restarted an exhausted train loader")
        rank_min_capacity = float(
            epoch["bev_rank_min_full_microbatch_capacity"]
        )
        rank_max_capacity = float(
            epoch["bev_rank_max_full_microbatch_capacity"]
        )
        rank_min_importance = float(
            epoch["bev_rank_min_importance_scale"]
        )
        rank_max_importance = float(
            epoch["bev_rank_max_importance_scale"]
        )
        rank_min_drop_last = float(
            epoch["bev_rank_min_drop_last_fraction"]
        )
        rank_max_drop_last = float(
            epoch["bev_rank_max_drop_last_fraction"]
        )
        rank_min_optimizer_tail = float(
            epoch["bev_rank_min_optimizer_tail_fraction"]
        )
        rank_max_optimizer_tail = float(
            epoch["bev_rank_max_optimizer_tail_fraction"]
        )
        rank_min_truncation = float(
            epoch["bev_rank_min_truncation_fraction"]
        )
        rank_max_truncation = float(
            epoch["bev_rank_max_truncation_fraction"]
        )
        if (
            rank_min_capacity <= 0.0
            or rank_max_capacity < rank_min_capacity
            or rank_min_importance <= 0.0
            or rank_max_importance < rank_min_importance
            or rank_min_drop_last < 0.0
            or rank_max_drop_last < rank_min_drop_last
            or rank_min_optimizer_tail < 0.0
            or rank_max_optimizer_tail < rank_min_optimizer_tail
            or rank_min_truncation < 0.0
            or rank_max_truncation < rank_min_truncation
            or rank_max_truncation < rank_max_drop_last
            or rank_max_truncation < rank_max_optimizer_tail
            or rank_max_truncation >= 1.0
        ):
            raise ValueError("BEV canary emitted invalid rank sampling evidence")
        if (
            float(
                epoch["train_bev_logit_gradient_diagnostic_batches"]
            )
            <= 0.0
        ):
            raise ValueError(
                "BEV canary did not measure class gradient budgets"
            )
        gradient_shares = [
            float(
                epoch[
                    f"train_bev_logit_gradient_share_class_{index}"
                ]
            )
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        ]
        gradient_magnitudes = [
            float(
                epoch[
                    f"train_bev_logit_gradient_l1_class_{index}"
                ]
            )
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        ]
        if (
            any(value <= 0.0 for value in gradient_shares)
            or any(value <= 0.0 for value in gradient_magnitudes)
            or not math.isclose(
                sum(gradient_shares),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
        ):
            raise ValueError(
                "BEV canary emitted invalid class gradient budgets"
            )
        for class_name in BEV_SEGMENTATION_CLASSES:
            prefix = f"validation_bev_{class_name}"
            if float(epoch[f"{prefix}_supported"]) != 1.0:
                raise ValueError(
                    f"BEV canary does not support class {class_name}"
                )
            if float(epoch[f"{prefix}_positive_cells"]) <= 0.0:
                raise ValueError(
                    f"BEV canary has no positive cells for {class_name}"
                )
            calibrated_threshold = float(
                epoch[
                    f"{prefix}_best_iou_threshold_on_validation_set"
                ]
            )
            if not 0.0 <= calibrated_threshold <= 1.0:
                raise ValueError(
                    "BEV canary emitted an invalid calibrated threshold "
                    f"for {class_name}"
                )

    initial_loss = float(history[0]["train_bev_segmentation"])
    minimum_later_loss = min(
        float(epoch["train_bev_segmentation"])
        for epoch in history[1:]
    )
    if initial_loss <= 0.0 or minimum_later_loss >= initial_loss:
        raise ValueError(
            "BEV canary loss did not decrease: "
            f"initial={initial_loss} later_min={minimum_later_loss}"
        )
    pos_weights = [
        float(history[0][f"bev_pos_weight_{index}"])
        for index in range(len(BEV_SEGMENTATION_CLASSES))
    ]
    if (
        any(value < 1.0 or value > BEV_POS_WEIGHT_CAP for value in pos_weights)
        or all(abs(value - 1.0) <= 1e-12 for value in pos_weights)
    ):
        raise ValueError("BEV canary derived invalid positive weights")
    class_weights = [
        float(history[0][f"bev_class_weight_{index}"])
        for index in range(len(BEV_SEGMENTATION_CLASSES))
    ]
    if (
        any(value <= 0.0 for value in class_weights)
        or not math.isclose(
            sum(class_weights) / len(class_weights),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        )
        or max(class_weights) <= 1.0
    ):
        raise ValueError("BEV canary derived invalid class weights")
    positive_pair_frequencies = [
        float(
            history[0][f"bev_positive_pair_frequency_{index}"]
        )
        for index in range(len(BEV_SEGMENTATION_CLASSES))
    ]
    if any(
        value <= 0.0 or value > 1.0
        for value in positive_pair_frequencies
    ):
        raise ValueError(
            "BEV canary derived invalid positive-pair frequencies"
        )
    repeat_factors = [
        int(history[0][f"bev_repeat_factor_{index}"])
        for index in range(len(BEV_SEGMENTATION_CLASSES))
    ]
    if any(value < 1 or value > 4 for value in repeat_factors):
        raise ValueError(
            "BEV canary derived invalid class repeat factors"
        )
    if not any(int(epoch.get("is_best", 0)) == 1 for epoch in history):
        raise ValueError("BEV canary did not select a best checkpoint")
    selected_epoch = max(
        history,
        key=lambda epoch: float(epoch["validation_selection_score"]),
    )
    if int(selected_epoch.get("is_best", 0)) != 1:
        raise ValueError(
            "BEV canary selected checkpoint is not marked as best"
        )
    classes_without_lift = [
        class_name
        for class_name in BEV_SEGMENTATION_CLASSES
        if float(
            selected_epoch[
                f"validation_bev_{class_name}_"
                "ap_lift_bootstrap_lower_95"
            ]
        )
        <= 0.0
    ]
    if classes_without_lift:
        raise ValueError(
            "BEV canary did not beat prevalence for classes: "
            f"{classes_without_lift}"
        )
    invalid_operating_points = [
        class_name
        for class_name in BEV_SEGMENTATION_CLASSES
        if (
            float(
                selected_epoch[
                    f"validation_bev_{class_name}_"
                    "best_iou_on_validation_set"
                ]
            )
            <= float(
                selected_epoch[
                    f"validation_bev_{class_name}_positive_prevalence"
                ]
            )
            or float(
                selected_epoch[
                    f"validation_bev_{class_name}_"
                    "best_iou_precision_on_validation_set"
                ]
            )
            <= float(
                selected_epoch[
                    f"validation_bev_{class_name}_positive_prevalence"
                ]
            )
            or float(
                selected_epoch[
                    f"validation_bev_{class_name}_"
                    "best_iou_recall_on_validation_set"
                ]
            )
            <= 0.0
        )
    ]
    if invalid_operating_points:
        raise ValueError(
            "BEV canary has unusable calibrated operating points for: "
            f"{invalid_operating_points}"
        )

    first_epoch = history[0]
    second_epoch = history[1]
    selection_gain = float(
        second_epoch["validation_selection_score"]
    ) - float(first_epoch["validation_selection_score"])
    if selection_gain < BEV_CANARY_MIN_SELECTION_GAIN:
        raise ValueError(
            "BEV canary selection score did not improve enough: "
            f"gain={selection_gain} "
            f"required={BEV_CANARY_MIN_SELECTION_GAIN}"
        )
    trend_classes = tuple(BEV_SEGMENTATION_CLASSES)
    rare_trend_classes = {
        "vulnerable_road_user",
        "other_obstacle",
    }
    first_epoch_ap_lift = {
        class_name: float(
            first_epoch[
                f"validation_bev_{class_name}_ap_lift"
            ]
        )
        for class_name in trend_classes
    }
    ap_lift_regression_tolerance = {
        class_name: (
            min(
                BEV_CANARY_AP_LIFT_ABSOLUTE_REGRESSION_TOLERANCE,
                BEV_CANARY_RARE_AP_LIFT_RELATIVE_REGRESSION_TOLERANCE
                * max(first_epoch_ap_lift[class_name], 0.0)
            )
            if class_name in rare_trend_classes
            else BEV_CANARY_AP_LIFT_ABSOLUTE_REGRESSION_TOLERANCE
        )
        for class_name in trend_classes
    }
    regressed_classes = [
        class_name
        for class_name in trend_classes
        if float(
            second_epoch[
                f"validation_bev_{class_name}_ap_lift"
            ]
        )
        < (
            first_epoch_ap_lift[class_name]
            - ap_lift_regression_tolerance[class_name]
        )
    ]
    if regressed_classes:
        raise ValueError(
            "BEV canary AP lift regressed for classes: "
            f"{regressed_classes}"
        )
    threshold_classes = tuple(BEV_SEGMENTATION_CLASSES)
    threshold_drift = {
        class_name: abs(
            float(
                second_epoch[
                    f"validation_bev_{class_name}_"
                    "best_iou_threshold_on_validation_set"
                ]
            )
            - float(
                first_epoch[
                    f"validation_bev_{class_name}_"
                    "best_iou_threshold_on_validation_set"
                ]
            )
        )
        for class_name in threshold_classes
    }
    report = {
        "schema_version": "reactive_bev_canary_report_v7",
        "epochs": len(history),
        "initial_bev_loss": initial_loss,
        "minimum_later_bev_loss": minimum_later_loss,
        "best_selection_score": max(
            float(epoch["validation_selection_score"])
            for epoch in history
        ),
        "selected_checkpoint_epoch": int(selected_epoch["epoch"]),
        "camera_gradient_verified": True,
        "front_gate_gradient_verified": True,
        "all_classes_supported": True,
        "all_classes_beat_prevalence": True,
        "all_classes_have_useful_operating_points": True,
        "operating_point_requirement": (
            "beats_prevalence_with_positive_recall_v1"
        ),
        "production_quality_guard_deferred": True,
        "class_order": list(BEV_SEGMENTATION_CLASSES),
        "positive_weights": pos_weights,
        "positive_weights_by_class": dict(
            zip(BEV_SEGMENTATION_CLASSES, pos_weights, strict=True)
        ),
        "class_weights": class_weights,
        "class_weights_by_class": dict(
            zip(BEV_SEGMENTATION_CLASSES, class_weights, strict=True)
        ),
        "repeat_factors": repeat_factors,
        "repeat_factors_by_class": dict(
            zip(BEV_SEGMENTATION_CLASSES, repeat_factors, strict=True)
        ),
        "selection_gain_minimum": BEV_CANARY_MIN_SELECTION_GAIN,
        "second_epoch_selection_gain": selection_gain,
        "second_epoch_trend_classes": list(trend_classes),
        "ap_lift_absolute_regression_tolerance": (
            BEV_CANARY_AP_LIFT_ABSOLUTE_REGRESSION_TOLERANCE
        ),
        "rare_ap_lift_relative_regression_tolerance": (
            BEV_CANARY_RARE_AP_LIFT_RELATIVE_REGRESSION_TOLERANCE
        ),
        "ap_lift_regression_tolerance_by_class": (
            ap_lift_regression_tolerance
        ),
        "key_threshold_drift": threshold_drift,
        "threshold_stability_gate_enforced": False,
        "thresholds_pass": True,
    }
    output = (
        Path(tempfile.mkdtemp(prefix="reactive-bev-canary-report-"))
        / "report.json"
    )
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return FlyteFile(str(output))


@task(
    task_config=RAY_1,
    container_image=TRAINING_IMAGE,
    retries=1,
    labels={
        "kueue.x-k8s.io/queue-name": "gpu-validation",
        "kueue.x-k8s.io/priority-class": "research-low",
    },
    environment=RAY_TASK_ENVIRONMENT,
)
def train_reactive_nuplan_bev_ray_1_smoke(
    shards: List[FlyteDirectory],
    epochs: int = 1,
    steps_per_epoch: int = 8,
) -> ReactiveRayOutput:
    """Run one-rank BEV training when only one validation GPU is available."""
    return _run_reactive_stage_task(
        shards=shards,
        stage="nuplan_full",
        num_workers=1,
        parent_checkpoint=None,
        resume_checkpoint=None,
        backbone="res_net_50",
        epochs=epochs,
        learning_rate=1e-4,
        weight_decay=1e-2,
        grad_clip=1.0,
        val_fraction=0.1,
        num_loader_workers=2,
        per_rank_batch_size=1,
        training_seed=149,
        precision="bf16",
        gradient_accumulation_steps=1,
        steps_per_epoch=steps_per_epoch,
        checkpoint_interval_steps=min(4, steps_per_epoch),
        shuffle_buffer=64,
        is_pretrained=True,
        trajectory_weight=0.0,
        bev_weight=1.0,
        route_weight=0.0,
        corridor_pos_weight=1.0,
        bev_pos_weight_cap=BEV_POS_WEIGHT_CAP,
        bev_repeat_frequency_threshold=0.05,
        bev_max_repeat=4,
        bev_min_positive_samples=1,
        bev_min_positive_cells=1,
        freeze_bevformer=False,
        validation_sample_limit=128,
        capacity_block_end_utc="",
        training_scope="bev_only",
        bev_encoder_learning_rate=1e-5,
        allow_single_worker_smoke=True,
    )


@task(
    task_config=RAY_2,
    container_image=TRAINING_IMAGE,
    retries=1,
    labels={
        "kueue.x-k8s.io/queue-name": "gpu-validation",
        "kueue.x-k8s.io/priority-class": "research-low",
    },
    environment=RAY_TASK_ENVIRONMENT,
)
def train_reactive_stage_ray_2(
    shards: List[FlyteDirectory],
    stage: str,
    parent_checkpoint: Optional[FlyteFile] = None,
    resume_checkpoint: Optional[FlyteDirectory] = None,
    backbone: str = "res_net_50",
    epochs: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    val_fraction: float = 0.1,
    num_loader_workers: int = 2,
    per_rank_batch_size: int = 1,
    training_seed: int = 149,
    precision: str = "fp32",
    gradient_accumulation_steps: int = 1,
    steps_per_epoch: int = 2,
    checkpoint_interval_steps: int = 1,
    shuffle_buffer: int = 64,
    is_pretrained: bool = True,
    trajectory_weight: float = 1.0,
    bev_weight: float = 1.0,
    route_weight: float = 1.0,
    corridor_pos_weight: float = 1.0,
    freeze_bevformer: bool = True,
    training_scope: str = "multitask",
    bev_encoder_learning_rate: float = 1e-5,
    allow_random_bevformer_init: bool = False,
) -> ReactiveRayOutput:
    """Run two-rank training with the production pretrained default."""
    return _run_reactive_stage_task(
        shards=shards,
        stage=stage,
        num_workers=2,
        parent_checkpoint=parent_checkpoint,
        resume_checkpoint=resume_checkpoint,
        backbone=backbone,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        per_rank_batch_size=per_rank_batch_size,
        training_seed=training_seed,
        precision=precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        steps_per_epoch=steps_per_epoch,
        checkpoint_interval_steps=checkpoint_interval_steps,
        shuffle_buffer=shuffle_buffer,
        is_pretrained=is_pretrained,
        trajectory_weight=trajectory_weight,
        bev_weight=bev_weight,
        route_weight=route_weight,
        corridor_pos_weight=corridor_pos_weight,
        bev_pos_weight_cap=BEV_POS_WEIGHT_CAP,
        bev_repeat_frequency_threshold=0.05,
        bev_max_repeat=4,
        bev_min_positive_samples=1,
        bev_min_positive_cells=1,
        freeze_bevformer=freeze_bevformer,
        validation_sample_limit=256,
        capacity_block_end_utc="",
        training_scope=training_scope,
        bev_encoder_learning_rate=bev_encoder_learning_rate,
        allow_random_bevformer_init=allow_random_bevformer_init,
    )


@task(
    task_config=RAY_REACTIVE_4,
    container_image=TRAINING_IMAGE,
    retries=2,
    labels={
        "kueue.x-k8s.io/queue-name": "p5en-capacity-block",
        "kueue.x-k8s.io/priority-class": "production-high",
    },
    environment=RAY_TASK_ENVIRONMENT,
)
def train_reactive_stage_ray_4(
    shards: List[FlyteDirectory],
    stage: str,
    parent_checkpoint: Optional[FlyteFile] = None,
    resume_checkpoint: Optional[FlyteDirectory] = None,
    backbone: str = "res_net_50",
    epochs: int = 3,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    val_fraction: float = 0.2,
    num_loader_workers: int = 2,
    per_rank_batch_size: int = 1,
    training_seed: int = 149,
    precision: str = "bf16",
    gradient_accumulation_steps: int = 1,
    steps_per_epoch: int = 0,
    checkpoint_interval_steps: int = 256,
    shuffle_buffer: int = 256,
    is_pretrained: bool = True,
    trajectory_weight: float = 1.0,
    bev_weight: float = 1.0,
    route_weight: float = 1.0,
    corridor_pos_weight: float = 1.0,
    freeze_bevformer: bool = True,
    capacity_block_end_utc: str = "",
    training_scope: str = "multitask",
    bev_encoder_learning_rate: float = 1e-5,
) -> ReactiveRayOutput:
    """Run a four-rank Reactive performance training stage."""
    return _run_reactive_stage_task(
        shards=shards,
        stage=stage,
        num_workers=4,
        parent_checkpoint=parent_checkpoint,
        resume_checkpoint=resume_checkpoint,
        backbone=backbone,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        per_rank_batch_size=per_rank_batch_size,
        training_seed=training_seed,
        precision=precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        steps_per_epoch=steps_per_epoch,
        checkpoint_interval_steps=checkpoint_interval_steps,
        shuffle_buffer=shuffle_buffer,
        is_pretrained=is_pretrained,
        trajectory_weight=trajectory_weight,
        bev_weight=bev_weight,
        route_weight=route_weight,
        corridor_pos_weight=corridor_pos_weight,
        bev_pos_weight_cap=BEV_POS_WEIGHT_CAP,
        bev_repeat_frequency_threshold=0.05,
        bev_max_repeat=4,
        bev_min_positive_samples=20,
        bev_min_positive_cells=2000,
        freeze_bevformer=freeze_bevformer,
        validation_sample_limit=1024,
        capacity_block_end_utc=capacity_block_end_utc,
        training_scope=training_scope,
        bev_encoder_learning_rate=bev_encoder_learning_rate,
    )


@task(
    task_config=RAY_8,
    container_image=TRAINING_IMAGE,
    retries=1,
    labels={
        "kueue.x-k8s.io/queue-name": "p5en-capacity-block",
        "kueue.x-k8s.io/priority-class": "production-high",
    },
    environment=RAY_TASK_ENVIRONMENT,
)
def train_reactive_stage_ray_8(
    shards: List[FlyteDirectory],
    stage: str,
    parent_checkpoint: Optional[FlyteFile] = None,
    resume_checkpoint: Optional[FlyteDirectory] = None,
    backbone: str = "res_net_50",
    epochs: int = 3,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    val_fraction: float = 0.1,
    num_loader_workers: int = 2,
    per_rank_batch_size: int = 4,
    training_seed: int = 149,
    precision: str = "bf16",
    gradient_accumulation_steps: int = 1,
    steps_per_epoch: int = 0,
    checkpoint_interval_steps: int = 256,
    shuffle_buffer: int = 256,
    is_pretrained: bool = True,
    trajectory_weight: float = 1.0,
    bev_weight: float = 1.0,
    route_weight: float = 1.0,
    corridor_pos_weight: float = 1.0,
    freeze_bevformer: bool = True,
    capacity_block_end_utc: str = "",
    training_scope: str = "multitask",
    bev_encoder_learning_rate: float = 1e-5,
    bev_repeat_frequency_threshold: float = 0.05,
    validation_sample_limit: int = 1024,
    allow_bounded_bev_canary: bool = False,
    parent_profile: str = "",
    parent_checkpoint_sha256: str = "",
    parent_checkpoint_epoch: int = 0,
) -> ReactiveRayOutput:
    """Run one production-size Reactive DDP stage."""
    return _run_reactive_stage_task(
        shards=shards,
        stage=stage,
        num_workers=8,
        parent_checkpoint=parent_checkpoint,
        resume_checkpoint=resume_checkpoint,
        backbone=backbone,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        per_rank_batch_size=per_rank_batch_size,
        training_seed=training_seed,
        precision=precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        steps_per_epoch=steps_per_epoch,
        checkpoint_interval_steps=checkpoint_interval_steps,
        shuffle_buffer=shuffle_buffer,
        is_pretrained=is_pretrained,
        trajectory_weight=trajectory_weight,
        bev_weight=bev_weight,
        route_weight=route_weight,
        corridor_pos_weight=corridor_pos_weight,
        bev_pos_weight_cap=BEV_POS_WEIGHT_CAP,
        bev_repeat_frequency_threshold=bev_repeat_frequency_threshold,
        bev_max_repeat=4,
        bev_min_positive_samples=20,
        bev_min_positive_cells=2000,
        freeze_bevformer=freeze_bevformer,
        validation_sample_limit=validation_sample_limit,
        capacity_block_end_utc=capacity_block_end_utc,
        training_scope=training_scope,
        bev_encoder_learning_rate=bev_encoder_learning_rate,
        allow_bounded_bev_canary=allow_bounded_bev_canary,
        parent_profile=parent_profile,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        parent_checkpoint_epoch=parent_checkpoint_epoch,
    )


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(
        cpu="6",
        mem="28Gi",
        gpu="1",
        ephemeral_storage="420Gi",
    ),
    limits=Resources(
        cpu="6",
        mem="28Gi",
        gpu="1",
        ephemeral_storage="420Gi",
    ),
    retries=1,
    labels={
        "kueue.x-k8s.io/queue-name": "gpu-validation",
        "kueue.x-k8s.io/priority-class": "research-low",
    },
    pod_template=_bev_evaluation_pod_template(),
)
def evaluate_reactive_bev_checkpoint(
    checkpoint: FlyteFile,
    shards: List[FlyteDirectory],
    dataset: str,
    benchmark_inventory: Optional[FlyteFile] = None,
    split: str = "validation_holdout",
    val_fraction: float = 0.1,
    batch_size: int = 1,
    num_loader_workers: int = 2,
    probability_bins: int = 1024,
    validation_sample_limit: int = 0,
) -> ReactiveBEVEvaluationOutput:
    """Evaluate one Reactive checkpoint with exact T8 class supervision."""
    import hashlib
    import tempfile

    import torch

    from data_parsing.pre_extracted import (
        discover_bev_sample_statistics,
        make_multi_dataset_loader,
        select_distributed_bev_validation_sample_uids,
        select_bev_validation_holdout_sample_uids,
    )
    from distributed_training.reactive_data import (
        assign_reactive_shards,
        build_reactive_dataset_plan,
        reactive_assignment_sha256,
    )
    from evaluation.reactive_bev_checkpoint import (
        KITSCENES_BENCHMARK_SPLITS,
        KITSCENES_DATASET,
        NUPLAN_DATASET,
        checkpoint_bev_probability_bins,
        checkpoint_validation_thresholds_with_protocol,
        evaluate_reactive_bev_model,
        load_reactive_bev_checkpoint,
        validate_kitscenes_benchmark_inventory_coverage,
        validate_reactive_bev_evaluation_manifest,
    )
    from training.reactive_stage_runner import reactive_config_sha256
    from training.reactive_multitask import ReactiveTrainingStage

    if dataset not in (NUPLAN_DATASET, KITSCENES_DATASET):
        raise ValueError("unsupported Reactive BEV evaluation dataset")
    if dataset == NUPLAN_DATASET and (
        split != "validation_holdout"
        or not 0.0 < val_fraction < 1.0
    ):
        raise ValueError(
            "nuPlan BEV evaluation requires the validation holdout split"
        )
    if dataset == KITSCENES_DATASET and (
        split != "all" or val_fraction != 0.0
    ):
        raise ValueError(
            "KITScenes BEV evaluation consumes the complete official "
            "held-out split without an additional hash split"
        )
    if dataset == KITSCENES_DATASET and benchmark_inventory is None:
        raise ValueError(
            "KITScenes BEV evaluation requires the pinned benchmark inventory"
        )
    if dataset == NUPLAN_DATASET and benchmark_inventory is not None:
        raise ValueError(
            "nuPlan BEV evaluation does not accept a KITScenes inventory"
        )
    if not 1 <= batch_size <= 4:
        raise ValueError("BEV evaluation batch size must be between one and four")
    if not 0 <= num_loader_workers <= 4:
        raise ValueError("BEV evaluation loader workers must be between zero and four")
    if probability_bins < 256:
        raise ValueError("BEV evaluation needs at least 256 probability bins")
    if not shards:
        raise ValueError("BEV evaluation shards must not be empty")

    shard_directories = []
    manifest_identities = []
    source_directories = {}
    inventory_payload = None
    inventory_sha256 = None
    if benchmark_inventory is not None:
        inventory_path = Path(benchmark_inventory.download())
        inventory_bytes = inventory_path.read_bytes()
        try:
            inventory_payload = json.loads(inventory_bytes)
        except json.JSONDecodeError as error:
            raise ValueError(
                "KITScenes benchmark inventory is invalid JSON"
            ) from error
        if not isinstance(inventory_payload, dict):
            raise ValueError(
                "KITScenes benchmark inventory must be an object"
            )
        inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    for shard in shards:
        remote_uri = _flyte_remote_uri(shard)
        directory = Path(shard.download())
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"BEV evaluation manifest is missing from {directory}"
            )
        manifest_bytes = manifest_path.read_bytes()
        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as error:
            raise ValueError("BEV evaluation manifest is invalid JSON") from error
        if not isinstance(manifest, dict):
            raise ValueError("BEV evaluation manifest must be an object")
        validate_reactive_bev_evaluation_manifest(
            manifest,
            dataset=dataset,
        )
        if remote_uri in source_directories:
            raise ValueError(
                "BEV evaluation shard sources contain duplicates"
            )
        source_directories[remote_uri] = directory
        if int(manifest["total_samples"]) > 0:
            shard_directories.append(str(directory))
        manifest_identities.append({
            "data_role": str(manifest.get("data_role", "")),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "partition_id": str(manifest.get("partition_id", "")),
            "source_revision": str(manifest.get("source_revision", "")),
            "source_split": str(manifest.get("source_split", "")),
            "total_samples": int(manifest["total_samples"]),
            "uri": remote_uri,
        })
    if not shard_directories:
        raise ValueError("BEV evaluation has no non-empty shard directories")

    kitscenes_inventory_contract = None
    if dataset == KITSCENES_DATASET:
        assert inventory_payload is not None
        kitscenes_inventory_contract = (
            validate_kitscenes_benchmark_inventory_coverage(
                inventory_payload,
                manifest_identities,
            )
        )

    checkpoint_path = Path(checkpoint.download())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Reactive BEV evaluation requires a GPU")
    (
        model,
        config,
        checkpoint_metrics,
        checkpoint_sha256,
        checkpoint_epoch,
    ) = load_reactive_bev_checkpoint(checkpoint_path, device=device)
    (
        reference_thresholds,
        threshold_protocol,
    ) = checkpoint_validation_thresholds_with_protocol(
        checkpoint_metrics
    )
    training_probability_bins = checkpoint_bev_probability_bins(config)
    if probability_bins != training_probability_bins:
        raise ValueError(
            "BEV evaluation probability_bins differs from the checkpoint: "
            f"evaluation={probability_bins} "
            f"checkpoint={training_probability_bins}"
        )
    sample_uids = None
    loader_split = "all"
    split_contract: dict[str, object]
    if dataset == NUPLAN_DATASET:
        checkpoint_val_fraction = float(
            config.get(
                "validation_fraction",
                checkpoint_metrics.get(
                    "configured_validation_fraction",
                    -1.0,
                ),
            )
        )
        if not math.isclose(
            val_fraction,
            checkpoint_val_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "nuPlan evaluation val_fraction differs from the checkpoint"
            )
        (
            resolved_validation_sample_limit,
            validation_sample_limit_source,
        ) = _resolve_nuplan_validation_sample_limit(
            config,
            validation_sample_limit,
        )
        world_size = int(config.get("distributed_world_size", 0))
        plan = build_reactive_dataset_plan(
            tuple(source_directories),
            stage=ReactiveTrainingStage.NUPLAN_FULL,
        )
        assignments = assign_reactive_shards(
            plan.shards,
            world_size=world_size,
        )
        assignment_digest = reactive_assignment_sha256(assignments)
        expected_assignment_digest = str(
            config.get("distributed_assignment_sha256") or ""
        )
        if assignment_digest != expected_assignment_digest:
            raise ValueError(
                "nuPlan evaluation shard assignment differs from the "
                "checkpoint"
            )
        records_by_rank = []
        for rank_shards in assignments:
            rank_directories = sorted({
                source_directories[shard.source_uri]
                for shard in rank_shards
            })
            rank_files = [
                source_directories[shard.source_uri] / shard.shard_name
                for shard in rank_shards
            ]
            records_by_rank.append(discover_bev_sample_statistics(
                rank_directories,
                shard_files=rank_files,
            ))
        records = tuple(
            record
            for rank_records in records_by_rank
            for record in rank_records
        )
        if len({record.sample_uid for record in records}) != len(records):
            raise ValueError(
                "nuPlan evaluation shard assignment duplicates samples"
            )
        calibration_uids = (
            select_distributed_bev_validation_sample_uids(
                records_by_rank,
                val_fraction=val_fraction,
                sample_limit=resolved_validation_sample_limit,
            )
        )
        calibration_digest = hashlib.sha256(
            "\n".join(calibration_uids).encode("utf-8")
        ).hexdigest()
        expected_calibration_digest = str(
            config.get("validation_sample_uid_sha256")
            or checkpoint_metrics.get("validation_sample_uid_sha256")
            or ""
        )
        if calibration_digest != expected_calibration_digest:
            raise ValueError(
                "nuPlan validation calibration subset differs from the "
                "checkpoint"
            )
        sample_uids = select_bev_validation_holdout_sample_uids(
            records,
            val_fraction=val_fraction,
            excluded_sample_uids=calibration_uids,
        )
        records_by_uid = {
            record.sample_uid: record
            for record in records
        }
        calibration_group_uids = tuple(sorted({
            records_by_uid[sample_uid].split_group_uid
            for sample_uid in calibration_uids
        }))
        holdout_group_uids = tuple(sorted({
            records_by_uid[sample_uid].split_group_uid
            for sample_uid in sample_uids
        }))
        if set(calibration_group_uids) & set(holdout_group_uids):
            raise ValueError(
                "nuPlan calibration and holdout split groups overlap"
            )
        holdout_digest = hashlib.sha256(
            "\n".join(sample_uids).encode("utf-8")
        ).hexdigest()
        calibration_group_digest = hashlib.sha256(
            "\n".join(calibration_group_uids).encode("utf-8")
        ).hexdigest()
        holdout_group_digest = hashlib.sha256(
            "\n".join(holdout_group_uids).encode("utf-8")
        ).hexdigest()
        loader_split = "val"
        split_contract = {
            "role": "validation_holdout",
            "validation_fraction": val_fraction,
            "validation_sample_limit": resolved_validation_sample_limit,
            "validation_sample_limit_source": (
                validation_sample_limit_source
            ),
            "calibration_sample_count": len(calibration_uids),
            "calibration_sample_uid_sha256": calibration_digest,
            "calibration_split_group_count": len(
                calibration_group_uids
            ),
            "calibration_split_group_sha256": (
                calibration_group_digest
            ),
            "distributed_assignment_sha256": assignment_digest,
            "distributed_world_size": world_size,
            "evaluation_sample_count": len(sample_uids),
            "evaluation_sample_uid_sha256": holdout_digest,
            "evaluation_split_group_count": len(holdout_group_uids),
            "evaluation_split_group_sha256": holdout_group_digest,
            "calibration_evaluation_disjoint": True,
            "group_disjoint_from_calibration": True,
            "training_evaluation_disjoint": True,
        }
        reference_threshold_source = (
            "checkpoint_validation_calibration_subset:"
            + threshold_protocol
            + ":"
            + calibration_digest
        )
        expected_sample_uids = sample_uids
        expected_sample_count = len(sample_uids)
    else:
        expected_sample_uids = None
        expected_sample_count = 0
        for identity in manifest_identities:
            total_samples = identity.get("total_samples")
            if (
                isinstance(total_samples, bool)
                or not isinstance(total_samples, int)
                or total_samples < 0
            ):
                raise ValueError(
                    "KITScenes manifest total_samples is invalid"
                )
            expected_sample_count += total_samples
        split_contract = {
            "role": "official_benchmark",
            "source_splits": sorted(KITSCENES_BENCHMARK_SPLITS),
            "evaluation_sample_count": expected_sample_count,
            "benchmark_inventory_sha256": inventory_sha256,
            **dict(kitscenes_inventory_contract or {}),
            "training_evaluation_disjoint": True,
        }
        checkpoint_calibration_digest = str(
            config.get("validation_sample_uid_sha256")
            or checkpoint_metrics.get("validation_sample_uid_sha256")
            or ""
        )
        if (
            len(checkpoint_calibration_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in checkpoint_calibration_digest
            )
        ):
            raise ValueError(
                "checkpoint lacks nuPlan validation calibration provenance"
            )
        reference_threshold_source = (
            "checkpoint_nuplan_validation_calibration_subset:"
            + threshold_protocol
            + ":"
            + checkpoint_calibration_digest
        )
    if "distributed_precision" not in config:
        raise ValueError(
            "Reactive BEV checkpoint lacks precision provenance"
        )
    evaluation_precision = str(config["distributed_precision"])
    loader = make_multi_dataset_loader(
        shard_directories,
        batch_size=batch_size,
        num_workers=num_loader_workers,
        split=loader_split,
        val_fraction=val_fraction,
        shuffle=0,
        pin_memory=True,
        prefetch_factor=1,
        max_active_loaders=1,
        sample_uids=sample_uids,
        decode_history_frames=False,
        decode_future_frames=False,
    )
    report = evaluate_reactive_bev_model(
        model,
        loader,
        dataset=dataset,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        probability_bins=probability_bins,
        reference_thresholds=reference_thresholds,
        reference_threshold_source=reference_threshold_source,
        evaluation_split_contract=split_contract,
        precision=evaluation_precision,
        expected_sample_uids=expected_sample_uids,
        expected_sample_count=expected_sample_count,
    )
    if report.get("evaluation_valid") is not True:
        raise FloatingPointError(
            "Reactive BEV evaluation rejected non-finite model outputs"
        )
    if dataset == KITSCENES_DATASET:
        report_split_contract = report.get(
            "evaluation_split_contract"
        )
        if not isinstance(report_split_contract, dict):
            raise ValueError(
                "KITScenes report lacks its evaluation split contract"
            )
        report_split_contract["evaluation_sample_uid_sha256"] = report[
            "sample_uid_set_sha256"
        ]
    report.update({
        "checkpoint_config_sha256": reactive_config_sha256(config),
        "checkpoint_epoch": checkpoint_epoch,
        "manifest_identities": sorted(
            manifest_identities,
            key=lambda value: (
                value["partition_id"],
                value["manifest_sha256"],
            ),
        ),
        "split": split,
        "val_fraction": val_fraction,
        "training_probability_bins": training_probability_bins,
    })
    report_bytes = (
        json.dumps(
            report,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    output = (
        Path(tempfile.mkdtemp(prefix="reactive-bev-evaluation-"))
        / "report.json"
    )
    output.write_bytes(report_bytes)
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    return ReactiveBEVEvaluationOutput(
        report=FlyteFile(str(output)),
        report_sha256=report_sha256,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=checkpoint_epoch,
    )


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(
        cpu="6",
        mem="28Gi",
        gpu="1",
        ephemeral_storage="420Gi",
    ),
    limits=Resources(
        cpu="6",
        mem="28Gi",
        gpu="1",
        ephemeral_storage="420Gi",
    ),
    retries=1,
    labels={
        "kueue.x-k8s.io/queue-name": "gpu-validation",
        "kueue.x-k8s.io/priority-class": "research-low",
    },
    pod_template=_bev_evaluation_pod_template(),
)
def evaluate_reactive_kitscenes_checkpoint(
    checkpoint: FlyteFile,
    shards: List[FlyteDirectory],
    source_split: str = "val",
    batch_size: int = 1,
    num_loader_workers: int = 4,
    expected_test_partition_count: int = 0,
    expected_val_partition_count: int = 0,
    use_stateful_camera_fpn_cache: bool = False,
) -> ReactiveKITScenesEvaluationOutput:
    """Evaluate a KITScenes fine-tuned checkpoint without checkpoint selection."""
    import tempfile

    import torch

    from data_parsing.pre_extracted import make_multi_dataset_loader
    from data_parsing.kit_scenes.temporal_contract import (
        KITSCENES_BENCHMARK_FUTURE_STEPS,
        kitscenes_temporal_contract,
    )
    from distributed_training.reactive_data import (
        build_reactive_dataset_plan,
    )
    from model_components.auto_e2e import AutoE2E
    from model_components.trajectory_planning.gru_planner import GRUPlanner
    from training.reactive_multitask import (
        ReactiveTrainingStage,
        reactive_model_kwargs,
    )
    from training.reactive_stage_runner import (
        evaluate_reactive_multitask,
        inspect_reactive_checkpoint_identity,
    )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ValueError(
            "KITScenes evaluation batch size must be an integer"
        )
    if source_split not in {"val", "overlap_train_val", "test"}:
        raise ValueError(
            "KITScenes labeled evaluation requires val, overlap_train_val, "
            "or test"
        )
    maximum_batch_size = (
        KITSCENES_STATEFUL_CAMERA_FPN_BATCH_SIZE
        if use_stateful_camera_fpn_cache
        else 4
    )
    if not 1 <= batch_size <= maximum_batch_size:
        raise ValueError(
            "KITScenes evaluation batch size exceeds the selected "
            "inference policy"
        )
    if not 0 <= num_loader_workers <= 4:
        raise ValueError(
            "KITScenes evaluation loader workers must be between zero and four"
        )
    if not shards:
        raise ValueError("KITScenes evaluation shards must not be empty")

    mapless_test = source_split == "test"
    if use_stateful_camera_fpn_cache and (
        not mapless_test
        or batch_size != KITSCENES_STATEFUL_CAMERA_FPN_BATCH_SIZE
    ):
        raise ValueError(
            "stateful camera FPN evaluation requires test data and "
            f"batch size {KITSCENES_STATEFUL_CAMERA_FPN_BATCH_SIZE}"
        )
    if not mapless_test and expected_test_partition_count:
        raise ValueError(
            "expected_test_partition_count is only valid for test data"
        )
    if source_split != "val" and expected_val_partition_count:
        raise ValueError(
            "expected_val_partition_count is only valid for val data"
        )
    if (
        expected_val_partition_count
        == KITSCENES_OFFICIAL_VAL_SCENE_COUNT
        and batch_size != KITSCENES_PRIMARY_VAL_BATCH_SIZE
    ):
        raise ValueError(
            "official KITScenes val evaluation requires batch size "
            f"{KITSCENES_PRIMARY_VAL_BATCH_SIZE}"
        )
    if mapless_test and expected_val_partition_count:
        raise ValueError(
            "KITScenes test evaluation cannot require val partitions"
        )
    remote_uris = [_flyte_remote_uri(shard) for shard in shards]
    plan = build_reactive_dataset_plan(
        remote_uris,
        stage=ReactiveTrainingStage.KITSCENES_FINETUNE,
        allow_mapless_kitscenes_evaluation=mapless_test,
    )
    expected_manifest_sha256s: dict[str, str] = {}
    expected_shard_sha256s: dict[tuple[str, str], str] = {}
    for reference in plan.shards:
        previous_manifest_sha256 = expected_manifest_sha256s.setdefault(
            reference.partition_id,
            reference.manifest_sha256,
        )
        if previous_manifest_sha256 != reference.manifest_sha256:
            raise ValueError(
                "KITScenes dataset plan mixes partition manifests"
            )
        expected_shard_sha256s[
            (reference.partition_id, reference.shard_name)
        ] = reference.shard_sha256
    shard_directories: list[str] = []
    manifest_identities: list[dict[str, object]] = []
    dataset_names: set[str] = set()
    dataset_versions: set[str] = set()
    source_revisions: set[str] = set()
    manifest_group_uids: list[str] = []
    scene_uids: list[str] = []
    expected_sample_count = 0
    for shard in shards:
        directory = Path(shard.download())
        manifest_path = directory / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if (
            not isinstance(manifest, dict)
            or manifest.get("source_split") != source_split
            or manifest.get("data_role") != "benchmark"
            or manifest.get("temporal_sampling")
            != kitscenes_temporal_contract(benchmark_protocol=True)
        ):
            raise ValueError(
                "KITScenes evaluation manifest has the wrong split, role, "
                "or temporal contract"
            )
        partition_id = str(manifest.get("partition_id") or "")
        total_samples = int(manifest.get("total_samples", 0))
        if total_samples < 0:
            raise ValueError(
                "KITScenes evaluation manifest has invalid sample count"
            )
        if mapless_test and total_samples > 0:
            expected_test_contract = {
                "has_gps": False,
                "has_map": False,
                "has_navigation": False,
                "has_reactive_navigation": False,
                "has_route_reconstruction": False,
                "has_trajectory_xy": True,
                "input_track": "camera_only_missing_map_route",
            }
            mismatches = {
                key: manifest.get(key)
                for key, expected in expected_test_contract.items()
                if manifest.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    "KITScenes test manifest violates the camera-only "
                    f"missing-map contract: {mismatches}"
                )
        if source_split == "val" and total_samples > 0:
            _validate_kitscenes_val_manifest_contract(manifest)
        if mapless_test and total_samples == 0 and any(
            bool(manifest.get(field, False))
            for field in (
                "has_gps",
                "has_map",
                "has_navigation",
                "has_reactive_navigation",
                "has_route_reconstruction",
                "has_trajectory_xy",
            )
        ):
            raise ValueError(
                "empty KITScenes test partition advertises target data"
            )
        if total_samples > 0:
            shard_directories.append(str(directory))
            actual_manifest_sha256 = hashlib.sha256(
                manifest_bytes
            ).hexdigest()
            if (
                expected_manifest_sha256s.get(partition_id)
                != actual_manifest_sha256
            ):
                raise ValueError(
                    "downloaded KITScenes manifest differs from the plan"
                )
            shard_names = manifest.get("shard_names")
            if not isinstance(shard_names, list) or not shard_names:
                raise ValueError(
                    "KITScenes evaluation manifest has no shards"
                )
            for shard_name_value in shard_names:
                shard_name = str(shard_name_value)
                shard_path = directory / shard_name
                expected_shard_sha256 = expected_shard_sha256s.get(
                    (partition_id, shard_name)
                )
                if (
                    expected_shard_sha256 is None
                    or not shard_path.is_file()
                    or _sha256_file(shard_path)
                    != expected_shard_sha256
                ):
                    raise ValueError(
                        "downloaded KITScenes shard differs from the plan"
                    )
        expected_sample_count += total_samples
        dataset_names.add(str(manifest.get("dataset") or ""))
        dataset_versions.add(str(manifest.get("dataset_version") or ""))
        source_revisions.add(str(manifest.get("source_revision") or ""))
        split_group_uids = manifest.get("split_group_uids")
        if (
            source_split in {"val", "test"}
            and (
                not isinstance(split_group_uids, list)
                or len(split_group_uids) != 1
                or not isinstance(split_group_uids[0], str)
                or not split_group_uids[0].startswith("kitscenes-")
            )
        ):
            raise ValueError(
                f"KITScenes {source_split} partition lacks one scene identity"
            )
        if source_split in {"val", "test"} and total_samples > 0:
            manifest_group_uids.append(split_group_uids[0])
        if source_split in {"val", "test"}:
            scene_uids.append(split_group_uids[0])
        manifest_identities.append({
            "manifest_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
            "partition_id": partition_id,
            "split_group_uids": split_group_uids,
            "total_samples": int(manifest.get("total_samples", 0)),
        })
    if (
        len(dataset_names) != 1
        or "" in dataset_names
        or len(dataset_versions) != 1
        or "" in dataset_versions
        or len(source_revisions) != 1
        or "" in source_revisions
    ):
        raise ValueError(
            "KITScenes evaluation manifests mix dataset provenance"
        )
    inventory = (
        _discover_kitscenes_evaluation_inventory(
            shard_directories,
        )
        if shard_directories
        else None
    )
    if expected_sample_count != plan.total_samples or (
        inventory is not None
        and inventory.sample_count != plan.total_samples
    ):
        raise ValueError(
            "KITScenes evaluation sample inventory differs from manifests"
        )
    if inventory is None and (not mapless_test or expected_sample_count != 0):
        raise ValueError("KITScenes evaluation has no non-empty shards")
    if inventory is not None and source_split in {"val", "test"}:
        _validate_kitscenes_test_inventory_groups(
            inventory.group_uids,
            manifest_group_uids,
        )
    scene_uid_sha256 = ""
    is_official_test = False
    is_official_val = False
    if mapless_test:
        scene_uid_sha256, is_official_test = (
            _kitscenes_test_scene_identity(
                scene_uids,
                expected_partition_count=expected_test_partition_count,
            )
        )
    elif source_split == "val":
        scene_uid_sha256, is_official_val = (
            _kitscenes_val_scene_identity(
                scene_uids,
                expected_partition_count=expected_val_partition_count,
            )
        )
        _validate_kitscenes_official_val_sdk_split(
            scene_uids,
            expected_partition_count=expected_val_partition_count,
        )
    _validate_kitscenes_official_sample_inventory(
        expected_partition_count=expected_test_partition_count,
        empty_partition_count=plan.empty_partition_count,
        manifest_sample_count=expected_sample_count,
        inventory_sample_count=(
            inventory.sample_count if inventory is not None else None
        ),
    )
    _validate_kitscenes_official_val_sample_inventory(
        expected_partition_count=expected_val_partition_count,
        empty_partition_count=plan.empty_partition_count,
        manifest_sample_count=expected_sample_count,
        inventory_sample_count=(
            inventory.sample_count if inventory is not None else None
        ),
    )

    checkpoint_path = Path(checkpoint.download())
    checkpoint_identity = inspect_reactive_checkpoint_identity(
        checkpoint_path
    )
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = payload.get("config")
    state_dict = payload.get("model_state_dict")
    checkpoint_epoch = payload.get("epoch")
    if (
        not isinstance(config, Mapping)
        or not isinstance(state_dict, Mapping)
        or isinstance(checkpoint_epoch, bool)
        or not isinstance(checkpoint_epoch, int)
        or checkpoint_epoch <= 0
        or config.get("training_stage")
        != ReactiveTrainingStage.KITSCENES_FINETUNE.value
    ):
        raise ValueError(
            "checkpoint is not a reviewed KITScenes fine-tuning checkpoint"
        )
    route_usage_evaluation_policy = (
        _kitscenes_route_usage_evaluation_policy(
            mapless_test=mapless_test,
        )
    )
    if inventory is None:
        trajectory_metrics: dict[str, float | int | None] = {
            "valid_timestep_count": 0,
            "total_timestep_count": 0,
            "valid_horizon_coverage": None,
            "mean_abs_longitudinal_error_m": None,
            "mean_abs_lateral_error_m": None,
            "nonfinite_prediction_count": 0,
            "nonfinite_prediction_rate": None,
        }
        for horizon in ("1s", "2s", "3s", "5s", "6p4s"):
            trajectory_metrics[f"ade_{horizon}_m"] = None
            trajectory_metrics[f"fde_{horizon}_m"] = None
            trajectory_metrics[f"ade_{horizon}_sample_count"] = 0
            trajectory_metrics[f"fde_{horizon}_sample_count"] = 0
        metrics = {
            "schema_version": "reactive_multitask_evaluation_v1",
            "sample_count": 0,
            "sample_uid_sha256": hashlib.sha256(b"").hexdigest(),
            "trajectory": trajectory_metrics,
            "bev_segmentation": {
                "available": False,
                "reason": "targets_unavailable",
            },
            "route": {
                "available": False,
                "reason": "input_and_targets_unavailable",
            },
        }
    else:
        stage = ReactiveTrainingStage.KITSCENES_FINETUNE
        model = AutoE2E(
            backbone=str(config["backbone"]),
            embed_dim=int(config["embed_dim"]),
            is_pretrained=False,
            **reactive_model_kwargs(
                stage,
                num_views=int(config["num_views"]),
            ),
        )
        model.load_state_dict(state_dict)
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("KITScenes evaluation requires a GPU")
        model.to(device)
        planner = model.Reactive_E2E.TrajectoryPlanner
        if not isinstance(planner, GRUPlanner):
            raise ValueError(
                "KITScenes evaluation requires the reviewed deterministic "
                "GRU planner"
            )

        camera_fpn_cache = (
            model.create_stateful_camera_fpn_cache()
            if use_stateful_camera_fpn_cache
            else None
        )
        loader = make_multi_dataset_loader(
            shard_directories,
            batch_size=batch_size,
            num_workers=num_loader_workers,
            split="all",
            val_fraction=0.0,
            shuffle=0,
            pin_memory=True,
            prefetch_factor=1,
            max_active_loaders=1,
            decode_future_frames=False,
            decode_front_camera_fpn=use_stateful_camera_fpn_cache,
        )
        metrics = evaluate_reactive_multitask(
            model,
            loader,
            stage=stage,
            device=device,
            include_counterfactuals=bool(
                route_usage_evaluation_policy[
                    "counterfactuals_enabled"
                ]
            ),
            include_route_gradient=bool(
                route_usage_evaluation_policy[
                    "input_gradient_enabled"
                ]
            ),
            camera_fpn_cache=camera_fpn_cache,
        )
    trajectory_metrics = metrics.get("trajectory")
    if not isinstance(trajectory_metrics, dict):
        raise ValueError("KITScenes evaluation omitted trajectory metrics")
    trajectory_metrics["ade_6p4s_m"] = None
    trajectory_metrics["fde_6p4s_m"] = None
    trajectory_metrics["ade_6p4s_sample_count"] = 0
    trajectory_metrics["fde_6p4s_sample_count"] = 0
    if inventory is not None and (
        metrics.get("sample_count") != inventory.sample_count
        or metrics.get("sample_uid_sha256")
        != inventory.sample_uid_digest
    ):
        raise ValueError(
            "KITScenes evaluation did not cover the exact packed inventory"
        )
    report = {
        "schema_version": "reactive_kitscenes_labeled_evaluation_v1",
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
        "dataset": next(iter(dataset_names)),
        "dataset_version": next(iter(dataset_versions)),
        "evaluation_role": (
            "official_test_camera_only_missing_map_route"
            if is_official_test
            else "official_val_camera_map_route"
            if is_official_val
            else "test_subset_camera_only_missing_map_route"
            if mapless_test
            else "labeled_external_benchmark"
        ),
        "input_track": (
            "camera_only_missing_map_route"
            if mapless_test
            else "camera_map_route"
        ),
        "inference_cache_policy": (
            KITSCENES_STATEFUL_CAMERA_FPN_POLICY
            if use_stateful_camera_fpn_cache
            else "stateless"
        ),
        "evaluation_batch_size": batch_size,
        "trajectory_inference_policy": {
            "planner": "gru",
            "stochastic_noise": False,
            "version": KITSCENES_TRAJECTORY_INFERENCE_POLICY,
        },
        "route_usage_evaluation_policy": (
            route_usage_evaluation_policy
        ),
        "maximum_labeled_horizon_steps": (
            KITSCENES_BENCHMARK_FUTURE_STEPS
        ),
        "partition_count": len(manifest_identities),
        "expected_partition_count": (
            expected_test_partition_count
            if mapless_test
            else expected_val_partition_count
            if source_split == "val"
            else 0
        ),
        "expected_sample_count": (
            KITSCENES_OFFICIAL_TEST_SAMPLE_COUNT
            if is_official_test
            else KITSCENES_OFFICIAL_VAL_SAMPLE_COUNT
            if is_official_val
            else 0
        ),
        "scene_uid_sha256": (
            scene_uid_sha256
            if source_split in {"val", "test"}
            else None
        ),
        "manifest_identities": sorted(
            manifest_identities,
            key=lambda item: str(item["partition_id"]),
        ),
        "metrics": metrics,
        "source_revision": next(iter(source_revisions)),
        "source_split": source_split,
    }
    report_bytes = (
        json.dumps(
            report,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    report_path = (
        Path(tempfile.mkdtemp(prefix="reactive-kitscenes-eval-"))
        / "report.json"
    )
    report_path.write_bytes(report_bytes)
    return ReactiveKITScenesEvaluationOutput(
        report=FlyteFile(str(report_path)),
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        checkpoint_sha256=checkpoint_identity["checkpoint_sha256"],
        checkpoint_epoch=checkpoint_epoch,
    )


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(
        cpu="2",
        mem="4Gi",
        ephemeral_storage="4Gi",
    ),
    limits=Resources(
        cpu="2",
        mem="4Gi",
        ephemeral_storage="4Gi",
    ),
    retries=2,
)
def aggregate_reactive_kitscenes_test_evaluations(
    reports: List[FlyteFile],
    report_sha256s: List[str],
    checkpoint_sha256s: List[str],
    checkpoint_epochs: List[int],
    expected_partition_count: int,
) -> ReactiveKITScenesEvaluationOutput:
    """Aggregate exact per-scene KITScenes test trajectory statistics."""
    import tempfile

    lengths = {
        len(reports),
        len(report_sha256s),
        len(checkpoint_sha256s),
        len(checkpoint_epochs),
    }
    if len(lengths) != 1 or not reports:
        raise ValueError(
            "KITScenes test aggregation inputs must be non-empty and aligned"
        )
    if (
        isinstance(expected_partition_count, bool)
        or not isinstance(expected_partition_count, int)
        or expected_partition_count <= 0
    ):
        raise ValueError(
            "KITScenes test expected partition count must be positive"
        )
    if len(reports) != expected_partition_count:
        raise ValueError(
            "KITScenes test aggregation does not cover the expected "
            f"{expected_partition_count} partitions"
        )
    if len(set(checkpoint_sha256s)) != 1:
        raise ValueError(
            "KITScenes test partitions used different checkpoints"
        )
    if len(set(checkpoint_epochs)) != 1:
        raise ValueError(
            "KITScenes test partitions used different checkpoint epochs"
        )

    payloads = []
    seen_partitions: set[str] = set()
    seen_scene_uids: set[str] = set()
    for report, expected_sha256 in zip(
        reports,
        report_sha256s,
        strict=True,
    ):
        report_path = Path(report.download())
        report_bytes = report_path.read_bytes()
        if hashlib.sha256(report_bytes).hexdigest() != expected_sha256:
            raise ValueError(
                "KITScenes test partition report digest differs"
            )
        payload = json.loads(report_bytes)
        if (
            not isinstance(payload, dict)
            or payload.get("source_split") != "test"
            or payload.get("input_track")
            != "camera_only_missing_map_route"
            or payload.get("evaluation_role") not in {
                "official_test_camera_only_missing_map_route",
                "test_subset_camera_only_missing_map_route",
            }
            or payload.get("maximum_labeled_horizon_steps") != 50
            or payload.get("inference_cache_policy")
            != KITSCENES_STATEFUL_CAMERA_FPN_POLICY
            or not isinstance(
                payload.get("evaluation_batch_size"),
                int,
            )
            or isinstance(payload.get("evaluation_batch_size"), bool)
            or payload["evaluation_batch_size"]
            != KITSCENES_STATEFUL_CAMERA_FPN_BATCH_SIZE
            or not isinstance(
                payload.get("trajectory_inference_policy"),
                dict,
            )
            or payload["trajectory_inference_policy"].get("version")
            != KITSCENES_TRAJECTORY_INFERENCE_POLICY
            or payload["trajectory_inference_policy"].get("planner")
            != "gru"
            or payload["trajectory_inference_policy"].get(
                "stochastic_noise"
            )
            is not False
            or payload.get("route_usage_evaluation_policy")
            != _kitscenes_route_usage_evaluation_policy(
                mapless_test=True,
            )
            or payload.get("checkpoint_sha256")
            != checkpoint_sha256s[0]
            or payload.get("checkpoint_epoch") != checkpoint_epochs[0]
        ):
            raise ValueError(
                "KITScenes test partition report identity differs"
            )
        identities = payload.get("manifest_identities")
        if not isinstance(identities, list) or len(identities) != 1:
            raise ValueError(
                "KITScenes test report must cover exactly one partition"
            )
        partition_id = str(identities[0].get("partition_id") or "")
        if not partition_id or partition_id in seen_partitions:
            raise ValueError(
                "KITScenes test reports contain duplicate partitions"
            )
        split_group_uids = identities[0].get("split_group_uids")
        if (
            not isinstance(split_group_uids, list)
            or len(split_group_uids) != 1
            or not isinstance(split_group_uids[0], str)
            or not split_group_uids[0].startswith("kitscenes-")
            or split_group_uids[0] in seen_scene_uids
        ):
            raise ValueError(
                "KITScenes test reports contain invalid scene identities"
            )
        seen_partitions.add(partition_id)
        seen_scene_uids.add(split_group_uids[0])
        payloads.append(payload)

    datasets = {str(payload.get("dataset") or "") for payload in payloads}
    dataset_versions = {
        str(payload.get("dataset_version") or "")
        for payload in payloads
    }
    source_revisions = {
        str(payload.get("source_revision") or "")
        for payload in payloads
    }
    inference_policy_identities = {
        json.dumps(
            payload["trajectory_inference_policy"],
            sort_keys=True,
        )
        for payload in payloads
    }
    if (
        len(datasets) != 1
        or "" in datasets
        or len(dataset_versions) != 1
        or "" in dataset_versions
        or len(source_revisions) != 1
        or "" in source_revisions
        or len(inference_policy_identities) != 1
    ):
        raise ValueError(
            "KITScenes test partition reports mix dataset provenance"
        )

    trajectory_rows = []
    sample_count = 0
    partition_uid_identities = []
    for payload in payloads:
        metrics = payload.get("metrics")
        trajectory = (
            metrics.get("trajectory")
            if isinstance(metrics, dict)
            else None
        )
        partition_sample_count = (
            metrics.get("sample_count")
            if isinstance(metrics, dict)
            else None
        )
        sample_uid_sha256 = (
            metrics.get("sample_uid_sha256")
            if isinstance(metrics, dict)
            else None
        )
        if (
            not isinstance(trajectory, dict)
            or isinstance(partition_sample_count, bool)
            or not isinstance(partition_sample_count, int)
            or partition_sample_count < 0
            or not isinstance(sample_uid_sha256, str)
            or len(sample_uid_sha256) != 64
            or int(
                payload["manifest_identities"][0]["total_samples"]
            )
            != partition_sample_count
        ):
            raise ValueError(
                "KITScenes test partition metrics are incomplete"
            )
        sample_count += partition_sample_count
        trajectory_rows.append(trajectory)
        partition_id = str(
            payload["manifest_identities"][0]["partition_id"]
        )
        partition_uid_identities.append(
            f"{partition_id}:{sample_uid_sha256}"
        )
    if sample_count <= 0:
        raise ValueError("KITScenes test evaluation has no labeled samples")
    scene_uid_sha256 = hashlib.sha256(
        "\n".join(sorted(seen_scene_uids)).encode("utf-8")
    ).hexdigest()
    if (
        expected_partition_count
        == KITSCENES_OFFICIAL_TEST_SCENE_COUNT
        and scene_uid_sha256
        != KITSCENES_OFFICIAL_TEST_SCENE_UID_SHA256
    ):
        raise ValueError(
            "KITScenes test reports do not cover the official scene set"
        )

    def sum_integer(name: str) -> int:
        values = [row.get(name) for row in trajectory_rows]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise ValueError(
                f"KITScenes trajectory metric {name!r} is not integral"
            )
        return sum(values)

    def weighted_mean(
        value_name: str,
        count_name: str,
    ) -> float | None:
        numerator = 0.0
        denominator = 0
        for row in trajectory_rows:
            value = row.get(value_name)
            count = row.get(count_name)
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError(
                    f"KITScenes trajectory count {count_name!r} is invalid"
                )
            if count == 0:
                if value is not None:
                    raise ValueError(
                        f"KITScenes metric {value_name!r} has no samples"
                    )
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"KITScenes metric {value_name!r} is invalid"
                )
            numerator += float(value) * count
            denominator += count
        return numerator / denominator if denominator else None

    valid_timestep_count = sum_integer("valid_timestep_count")
    total_timestep_count = sum_integer("total_timestep_count")
    nonfinite_prediction_count = sum_integer(
        "nonfinite_prediction_count"
    )
    trajectory_metrics: dict[str, float | int | None] = {
        "valid_timestep_count": valid_timestep_count,
        "total_timestep_count": total_timestep_count,
        "valid_horizon_coverage": (
            valid_timestep_count / total_timestep_count
            if total_timestep_count
            else None
        ),
        "mean_abs_longitudinal_error_m": weighted_mean(
            "mean_abs_longitudinal_error_m",
            "valid_timestep_count",
        ),
        "mean_abs_lateral_error_m": weighted_mean(
            "mean_abs_lateral_error_m",
            "valid_timestep_count",
        ),
        "nonfinite_prediction_count": nonfinite_prediction_count,
        "nonfinite_prediction_rate": (
            nonfinite_prediction_count / sample_count
        ),
    }
    for horizon in ("1s", "2s", "3s", "5s", "6p4s"):
        ade_count_name = f"ade_{horizon}_sample_count"
        fde_count_name = f"fde_{horizon}_sample_count"
        trajectory_metrics[f"ade_{horizon}_m"] = weighted_mean(
            f"ade_{horizon}_m",
            ade_count_name,
        )
        trajectory_metrics[f"fde_{horizon}_m"] = weighted_mean(
            f"fde_{horizon}_m",
            fde_count_name,
        )
        trajectory_metrics[ade_count_name] = sum_integer(ade_count_name)
        trajectory_metrics[fde_count_name] = sum_integer(fde_count_name)

    metrics = {
        "schema_version": "reactive_multitask_evaluation_v1",
        "sample_count": sample_count,
        "sample_uid_sha256": None,
        "partition_sample_uid_sha256": hashlib.sha256(
            "\n".join(sorted(partition_uid_identities)).encode("utf-8")
        ).hexdigest(),
        "trajectory": trajectory_metrics,
        "bev_segmentation": {
            "available": False,
            "reason": "targets_unavailable",
        },
        "route": {
            "available": False,
            "reason": "input_and_targets_unavailable",
        },
    }

    manifest_identities = [
        identity
        for payload in payloads
        for identity in payload["manifest_identities"]
    ]
    is_official_test = (
        expected_partition_count == KITSCENES_OFFICIAL_TEST_SCENE_COUNT
    )
    report_payload = {
        "schema_version": (
            "reactive_kitscenes_test_sharded_evaluation_v1"
        ),
        "checkpoint_epoch": checkpoint_epochs[0],
        "checkpoint_sha256": checkpoint_sha256s[0],
        "dataset": next(iter(datasets)),
        "dataset_version": next(iter(dataset_versions)),
        "evaluation_role": (
            "official_test_camera_only_missing_map_route"
            if is_official_test
            else "test_subset_camera_only_missing_map_route"
        ),
        "input_track": "camera_only_missing_map_route",
        "inference_cache_policy": KITSCENES_STATEFUL_CAMERA_FPN_POLICY,
        "evaluation_batch_size": (
            KITSCENES_STATEFUL_CAMERA_FPN_BATCH_SIZE
        ),
        "manifest_identities": sorted(
            manifest_identities,
            key=lambda item: str(item["partition_id"]),
        ),
        "maximum_labeled_horizon_steps": 50,
        "metrics": metrics,
        "partition_count": len(payloads),
        "expected_partition_count": expected_partition_count,
        "scene_uid_sha256": scene_uid_sha256,
        "partition_report_sha256s": sorted(report_sha256s),
        "source_revision": next(iter(source_revisions)),
        "source_split": "test",
        "trajectory_inference_policy": json.loads(
            next(iter(inference_policy_identities))
        ),
        "route_usage_evaluation_policy": (
            _kitscenes_route_usage_evaluation_policy(
                mapless_test=True,
            )
        ),
    }
    report_bytes = (
        json.dumps(
            report_payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    output = (
        Path(tempfile.mkdtemp(prefix="reactive-kitscenes-test-aggregate-"))
        / "report.json"
    )
    output.write_bytes(report_bytes)
    return ReactiveKITScenesEvaluationOutput(
        report=FlyteFile(str(output)),
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        checkpoint_sha256=checkpoint_sha256s[0],
        checkpoint_epoch=checkpoint_epochs[0],
    )


@dynamic(
    container_image=TRAINING_IMAGE,
    environment={"AUTO_E2E_TRAINING_IMAGE": TRAINING_IMAGE},
)
def evaluate_reactive_kitscenes_test_partitions(
    checkpoint: FlyteFile,
    shards: List[FlyteDirectory],
    expected_partition_count: int = KITSCENES_OFFICIAL_TEST_SCENE_COUNT,
) -> ReactiveKITScenesEvaluationOutput:
    """Evaluate each KITScenes test scene within one G6 disk budget."""
    reports: List[FlyteFile] = []
    report_sha256s: List[str] = []
    checkpoint_sha256s: List[str] = []
    checkpoint_epochs: List[int] = []
    for shard in shards:
        evaluation = evaluate_reactive_kitscenes_checkpoint(
            checkpoint=checkpoint,
            shards=[shard],
            source_split="test",
            batch_size=KITSCENES_STATEFUL_CAMERA_FPN_BATCH_SIZE,
            num_loader_workers=4,
            use_stateful_camera_fpn_cache=True,
        )
        reports.append(evaluation.report)
        report_sha256s.append(evaluation.report_sha256)
        checkpoint_sha256s.append(evaluation.checkpoint_sha256)
        checkpoint_epochs.append(evaluation.checkpoint_epoch)
    return aggregate_reactive_kitscenes_test_evaluations(
        reports=reports,
        report_sha256s=report_sha256s,
        checkpoint_sha256s=checkpoint_sha256s,
        checkpoint_epochs=checkpoint_epochs,
        expected_partition_count=expected_partition_count,
    )


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(
        cpu="2",
        mem="4Gi",
        ephemeral_storage="4Gi",
    ),
    limits=Resources(
        cpu="2",
        mem="4Gi",
        ephemeral_storage="4Gi",
    ),
    retries=2,
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
)
def publish_reactive_kitscenes_evaluation(
    checkpoint: FlyteFile,
    report: FlyteFile,
    report_sha256: str,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
    expected_model_version: str = "",
) -> ReactivePolicyPublicationOutput:
    """Register one exact KITScenes evaluation against its checkpoint."""
    import mlflow
    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient

    report_path = Path(report.download())
    report_bytes = report_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != report_sha256:
        raise ValueError("KITScenes evaluation report digest differs")
    report_payload = json.loads(report_bytes)
    if (
        not isinstance(report_payload, dict)
        or report_payload.get("checkpoint_sha256") != checkpoint_sha256
        or report_payload.get("checkpoint_epoch") != checkpoint_epoch
    ):
        raise ValueError("KITScenes evaluation report identity differs")
    _validate_kitscenes_publication_binding(
        report_payload,
        expected_model_version=expected_model_version,
    )

    checkpoint_path = Path(checkpoint.download())
    checkpoint_hasher = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checkpoint_hasher.update(chunk)
    if checkpoint_hasher.hexdigest() != checkpoint_sha256:
        raise ValueError("KITScenes evaluation checkpoint digest differs")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.set_experiment(
        REACTIVE_POLICY_EVALUATION_MLFLOW_EXPERIMENT
    )
    client = MlflowClient()
    evaluation_key = (
        f"{checkpoint_sha256}:{report_payload['source_split']}:"
        f"{report_sha256}"
    )
    matches = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.reactive_policy_evaluation_key = "
            f"'{evaluation_key}'"
        ),
        max_results=2,
    )
    if len(matches) > 1:
        raise RuntimeError(
            "multiple MLflow runs use one KITScenes evaluation key"
        )
    split = str(report_payload["source_split"])
    route_usage_policy = report_payload.get(
        "route_usage_evaluation_policy"
    )
    if not isinstance(route_usage_policy, dict):
        route_usage_policy = {}
    evaluation_priority = (
        "primary"
        if report_payload.get("evaluation_role")
        == "official_val_camera_map_route"
        else "secondary_camera_only"
        if split == "test"
        else "supplemental"
    )
    if matches:
        run_id = matches[0].info.run_id
        client.set_tag(run_id, "flyte_retry_reused", "true")
    else:
        run_name = (
            f"kitscenes-{split}-e{checkpoint_epoch}-"
            f"{checkpoint_sha256[:12]}"
        )
        run = client.create_run(
            experiment_id=experiment.experiment_id,
            start_time=int(time.time() * 1000),
            tags={
                "mlflow.runName": run_name,
                "pipeline": REACTIVE_POLICY_EVALUATION_MLFLOW_EXPERIMENT,
                "reactive_policy_evaluation_key": evaluation_key,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_s3_uri": _flyte_remote_uri(checkpoint),
                "evaluation_dataset": str(report_payload["dataset"]),
                "evaluation_input_track": str(
                    report_payload["input_track"]
                ),
                "evaluation_batch_size": str(
                    report_payload["evaluation_batch_size"]
                ),
                "evaluation_report_sha256": report_sha256,
                "evaluation_role": str(
                    report_payload["evaluation_role"]
                ),
                "route_usage_evaluation_policy": str(
                    route_usage_policy.get("version", "")
                ),
                "evaluation_split": split,
                "evaluation_priority": evaluation_priority,
                "task_status": "RUNNING",
            },
        )
        run_id = run.info.run_id
        for name, value in {
            "data/dataset": report_payload["dataset"],
            "data/dataset_version": report_payload["dataset_version"],
            "data/source_revision": report_payload["source_revision"],
            "data/source_split": split,
            "eval/evaluation_role": report_payload["evaluation_role"],
            "eval/batch_size": report_payload["evaluation_batch_size"],
            "eval/expected_partition_count": report_payload.get(
                "expected_partition_count",
                0,
            ),
            "eval/expected_sample_count": report_payload.get(
                "expected_sample_count",
                0,
            ),
            "eval/inference_cache_policy": report_payload.get(
                "inference_cache_policy",
                "",
            ),
            "eval/input_track": report_payload["input_track"],
            "eval/priority": evaluation_priority,
            "eval/partition_count": report_payload.get(
                "partition_count",
                0,
            ),
            "eval/scene_uid_sha256": report_payload.get(
                "scene_uid_sha256",
                "",
            ),
            "eval/trajectory_inference_policy": (
                report_payload.get("trajectory_inference_policy", {})
                .get("version", "")
                if isinstance(
                    report_payload.get("trajectory_inference_policy"),
                    dict,
                )
                else ""
            ),
            "eval/trajectory_planner": (
                report_payload.get("trajectory_inference_policy", {})
                .get("planner", "")
                if isinstance(
                    report_payload.get("trajectory_inference_policy"),
                    dict,
                )
                else ""
            ),
            "eval/trajectory_stochastic_noise": (
                report_payload.get("trajectory_inference_policy", {})
                .get("stochastic_noise", "")
                if isinstance(
                    report_payload.get("trajectory_inference_policy"),
                    dict,
                )
                else ""
            ),
            "eval/route_usage_evaluation_policy": (
                route_usage_policy.get("version", "")
            ),
            "eval/route_counterfactuals_enabled": (
                route_usage_policy.get("counterfactuals_enabled", "")
            ),
            "eval/route_input_gradient_enabled": (
                route_usage_policy.get("input_gradient_enabled", "")
            ),
            "model/checkpoint_epoch": checkpoint_epoch,
            "model/checkpoint_sha256": checkpoint_sha256,
        }.items():
            client.log_param(run_id, name, str(value)[:500])

    metrics_payload = report_payload.get("metrics")
    trajectory = (
        metrics_payload.get("trajectory")
        if isinstance(metrics_payload, dict)
        else None
    )
    if not isinstance(trajectory, dict):
        raise ValueError("KITScenes report has no trajectory metrics")
    timestamp = int(time.time() * 1000)
    numeric_metrics = {
        f"{split}/trajectory/{name}": float(value)
        for name, value in trajectory.items()
        if (
            value is not None
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )
    }
    numeric_metrics[f"{split}/sample_count"] = float(
        metrics_payload["sample_count"]
    )
    route = metrics_payload.get("route")
    if isinstance(route, dict):
        numeric_metrics.update({
            f"{split}/route/{name}": float(value)
            for name, value in route.items()
            if (
                value is not None
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            )
        })
    client.log_batch(
        run_id,
        metrics=[
            Metric(
                key=name,
                value=value,
                timestamp=timestamp,
                step=checkpoint_epoch,
            )
            for name, value in sorted(numeric_metrics.items())
        ],
    )
    client.log_artifact(
        run_id,
        str(report_path),
        artifact_path="evaluation",
    )

    checkpoint_uri = _flyte_remote_uri(checkpoint)
    version = _register_reactive_policy_model_version(
        client,
        checkpoint_uri=checkpoint_uri,
        checkpoint_sha256=checkpoint_sha256,
        evaluation_run_id=run_id,
        expected_model_version=expected_model_version,
    )

    split_tag = re.sub(
        r"[^a-z0-9]+",
        "_",
        split.lower(),
    ).strip("_")
    trajectory_tags = {
        name: value
        for name, value in trajectory.items()
        if name in {
            "ade_1s_m",
            "fde_1s_m",
            "ade_3s_m",
            "fde_3s_m",
            "ade_5s_m",
            "fde_5s_m",
            "nonfinite_prediction_rate",
        }
        and value is not None
    }
    route_tags = {
        name: value
        for name, value in (
            route.items() if isinstance(route, dict) else ()
        )
        if name in {
            "route_zero_trajectory_delta_m",
            "route_zero_sample_count",
            "route_swap_trajectory_delta_m",
            "route_swap_sample_count",
        }
        and value is not None
    }
    version_tags = {
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_s3_uri": checkpoint_uri,
        "checkpoint_sha256": checkpoint_sha256,
        f"{split_tag}_evaluation_input_track": (
            report_payload["input_track"]
        ),
        f"{split_tag}_evaluation_batch_size": (
            report_payload["evaluation_batch_size"]
        ),
        f"{split_tag}_inference_cache_policy": (
            report_payload.get("inference_cache_policy", "")
        ),
        f"{split_tag}_evaluation_role": (
            report_payload["evaluation_role"]
        ),
        f"{split_tag}_evaluation_priority": evaluation_priority,
        f"{split_tag}_evaluation_report_sha256": report_sha256,
        f"{split_tag}_evaluation_run_id": run_id,
        f"{split_tag}_expected_partition_count": (
            report_payload.get("expected_partition_count", 0)
        ),
        f"{split_tag}_expected_sample_count": (
            report_payload.get("expected_sample_count", 0)
        ),
        f"{split_tag}_partition_count": (
            report_payload.get("partition_count", 0)
        ),
        f"{split_tag}_scene_uid_sha256": (
            report_payload.get("scene_uid_sha256", "")
        ),
        f"{split_tag}_evaluation_sample_count": (
            metrics_payload["sample_count"]
        ),
        f"{split_tag}_trajectory_inference_policy": (
            report_payload.get("trajectory_inference_policy", {}).get(
                "version",
                "",
            )
            if isinstance(
                report_payload.get("trajectory_inference_policy"),
                dict,
            )
            else ""
        ),
        f"{split_tag}_trajectory_planner": (
            report_payload.get("trajectory_inference_policy", {}).get(
                "planner",
                "",
            )
            if isinstance(
                report_payload.get("trajectory_inference_policy"),
                dict,
            )
            else ""
        ),
        f"{split_tag}_trajectory_stochastic_noise": (
            report_payload.get("trajectory_inference_policy", {}).get(
                "stochastic_noise",
                "",
            )
            if isinstance(
                report_payload.get("trajectory_inference_policy"),
                dict,
            )
            else ""
        ),
        f"{split_tag}_route_usage_evaluation_policy": (
            route_usage_policy.get("version", "")
        ),
        f"{split_tag}_route_counterfactuals_enabled": (
            route_usage_policy.get("counterfactuals_enabled", "")
        ),
        f"{split_tag}_route_input_gradient_enabled": (
            route_usage_policy.get("input_gradient_enabled", "")
        ),
        **{
            f"{split_tag}_{name}": value
            for name, value in trajectory_tags.items()
        },
        **{
            f"{split_tag}_{name}": value
            for name, value in route_tags.items()
        },
    }
    if evaluation_priority == "primary":
        version_tags.update({
            "primary_evaluation_split": split,
            "primary_evaluation_input_track": (
                report_payload["input_track"]
            ),
            "primary_evaluation_batch_size": (
                report_payload["evaluation_batch_size"]
            ),
            "primary_evaluation_report_sha256": report_sha256,
            "primary_evaluation_run_id": run_id,
            "primary_evaluation_sample_count": (
                metrics_payload["sample_count"]
            ),
            "primary_route_usage_evaluation_policy": (
                route_usage_policy.get("version", "")
            ),
            **{
                f"primary_{name}": value
                for name, value in trajectory_tags.items()
            },
            **{
                f"primary_{name}": value
                for name, value in route_tags.items()
            },
        })
    for name, value in version_tags.items():
        client.set_model_version_tag(
            REACTIVE_POLICY_REGISTERED_MODEL,
            version,
            name,
            str(value),
        )
    client.set_tag(
        run_id,
        "registered_model_name",
        REACTIVE_POLICY_REGISTERED_MODEL,
    )
    client.set_tag(run_id, "registered_model_version", version)
    client.set_tag(run_id, "task_status", "FINISHED")
    client.set_terminated(run_id, status="FINISHED")
    return ReactivePolicyPublicationOutput(
        mlflow_run_id=run_id,
        registered_model_name=REACTIVE_POLICY_REGISTERED_MODEL,
        registered_model_version=version,
    )


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(
        cpu="2",
        mem="4Gi",
        ephemeral_storage="4Gi",
    ),
    limits=Resources(
        cpu="2",
        mem="4Gi",
        ephemeral_storage="4Gi",
    ),
    retries=2,
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
)
def publish_reactive_bev_evaluation(
    checkpoint: FlyteFile,
    report: FlyteFile,
    report_sha256: str,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
    source_training_mlflow_run_id: str = "",
) -> ReactiveBEVPublicationOutput:
    """Persist one verified BEV report and checkpoint in MLflow."""
    report_path = Path(report.download())
    report_bytes = report_path.read_bytes()
    actual_report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    if actual_report_sha256 != report_sha256:
        raise ValueError(
            "Reactive BEV report digest differs before MLflow publish"
        )
    try:
        report_payload = json.loads(report_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Reactive BEV report is invalid JSON"
        ) from error
    if (
        not isinstance(report_payload, dict)
        or report_payload.get("evaluation_valid") is not True
        or report_payload.get("checkpoint_sha256")
        != checkpoint_sha256
        or report_payload.get("checkpoint_epoch")
        != checkpoint_epoch
    ):
        raise ValueError(
            "Reactive BEV report identity differs before MLflow publish"
        )

    checkpoint_source_uri = _flyte_remote_uri(checkpoint)
    checkpoint_path = Path(checkpoint.download())
    checkpoint_hasher = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checkpoint_hasher.update(chunk)
    if checkpoint_hasher.hexdigest() != checkpoint_sha256:
        raise ValueError(
            "Reactive BEV checkpoint digest differs before MLflow publish"
        )

    mlflow_run_id, registered_model_version = (
        _log_reactive_bev_evaluation_to_mlflow(
            report=report_payload,
            report_path=report_path,
            report_sha256=report_sha256,
            checkpoint_path=checkpoint_path,
            checkpoint_source_uri=checkpoint_source_uri,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_epoch=checkpoint_epoch,
            source_training_mlflow_run_id=(
                source_training_mlflow_run_id
            ),
        )
    )
    return ReactiveBEVPublicationOutput(
        mlflow_run_id=mlflow_run_id,
        registered_model_name=REACTIVE_BEV_REGISTERED_MODEL,
        registered_model_version=registered_model_version,
    )


@workflow
def wf_ray_ddp_smoke_4(
    capacity_block_end_utc: str,
    steps: int = 4,
) -> FlyteFile:
    return ray_ddp_smoke_4(
        capacity_block_end_utc=capacity_block_end_utc,
        steps=steps,
    ).report


@workflow
def wf_evaluate_reactive_bev_checkpoint(
    checkpoint: FlyteFile,
    shards: List[FlyteDirectory],
    dataset: str,
    benchmark_inventory: Optional[FlyteFile] = None,
    split: str = "validation_holdout",
    val_fraction: float = 0.1,
    batch_size: int = 1,
    num_loader_workers: int = 2,
    probability_bins: int = 1024,
    validation_sample_limit: int = 0,
    source_training_mlflow_run_id: str = "",
) -> ReactiveBEVEvaluationWorkflowOutput:
    evaluation = evaluate_reactive_bev_checkpoint(
        checkpoint=checkpoint,
        shards=shards,
        dataset=dataset,
        benchmark_inventory=benchmark_inventory,
        split=split,
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_loader_workers=num_loader_workers,
        probability_bins=probability_bins,
        validation_sample_limit=validation_sample_limit,
    )
    publication = publish_reactive_bev_evaluation(
        checkpoint=checkpoint,
        report=evaluation.report,
        report_sha256=evaluation.report_sha256,
        checkpoint_sha256=evaluation.checkpoint_sha256,
        checkpoint_epoch=evaluation.checkpoint_epoch,
        source_training_mlflow_run_id=(
            source_training_mlflow_run_id
        ),
    )
    return ReactiveBEVEvaluationWorkflowOutput(
        report=evaluation.report,
        report_sha256=evaluation.report_sha256,
        checkpoint_sha256=evaluation.checkpoint_sha256,
        checkpoint_epoch=evaluation.checkpoint_epoch,
        mlflow_run_id=publication.mlflow_run_id,
        registered_model_name=publication.registered_model_name,
        registered_model_version=(
            publication.registered_model_version
        ),
    )


@workflow
def wf_train_reactive_nuplan_ray_4(
    nuplan_shards: List[FlyteDirectory],
    capacity_block_end_utc: str,
    resume_checkpoint: Optional[FlyteDirectory] = None,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    val_fraction: float = 0.2,
    num_loader_workers: int = 2,
    per_rank_batch_size: int = 1,
    training_seed: int = 149,
    precision: str = "bf16",
    trajectory_weight: float = 1.0,
    bev_weight: float = 1.0,
    route_weight: float = 1.0,
    checkpoint_interval_steps: int = 256,
) -> ReactiveRayOutput:
    """Train Stage A while keeping the pretrained camera BEV frozen."""
    return train_reactive_stage_ray_4(
        shards=nuplan_shards,
        stage="nuplan_full",
        parent_checkpoint=None,
        resume_checkpoint=resume_checkpoint,
        epochs=epochs,
        learning_rate=learning_rate,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        per_rank_batch_size=per_rank_batch_size,
        training_seed=training_seed,
        precision=precision,
        steps_per_epoch=0,
        checkpoint_interval_steps=checkpoint_interval_steps,
        shuffle_buffer=256,
        is_pretrained=True,
        trajectory_weight=trajectory_weight,
        bev_weight=bev_weight,
        route_weight=route_weight,
        freeze_bevformer=True,
        capacity_block_end_utc=capacity_block_end_utc,
    )


@workflow
def wf_train_reactive_nuplan_ray_8(
    nuplan_shards: List[FlyteDirectory],
    capacity_block_end_utc: str,
    resume_checkpoint: Optional[FlyteDirectory] = None,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    val_fraction: float = 0.1,
    num_loader_workers: int = 2,
    per_rank_batch_size: int = 4,
    training_seed: int = 149,
    precision: str = "bf16",
    trajectory_weight: float = 1.0,
    bev_weight: float = 0.0,
    route_weight: float = 1.0,
    checkpoint_interval_steps: int = 256,
) -> ReactiveRayOutput:
    """Train nuPlan on one eight-GPU node with the camera BEV frozen."""
    return train_reactive_stage_ray_8(
        shards=nuplan_shards,
        stage="nuplan_full",
        parent_checkpoint=None,
        resume_checkpoint=resume_checkpoint,
        epochs=epochs,
        learning_rate=learning_rate,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        per_rank_batch_size=per_rank_batch_size,
        training_seed=training_seed,
        precision=precision,
        steps_per_epoch=0,
        checkpoint_interval_steps=checkpoint_interval_steps,
        shuffle_buffer=256,
        is_pretrained=True,
        trajectory_weight=trajectory_weight,
        bev_weight=bev_weight,
        route_weight=route_weight,
        freeze_bevformer=True,
        capacity_block_end_utc=capacity_block_end_utc,
    )


@workflow
def wf_train_reactive_kitscenes_ray_8(
    kitscenes_shards: List[FlyteDirectory],
    nuplan_epoch5_checkpoint: FlyteFile,
    capacity_block_end_utc: str,
    epochs: int = 10,
    learning_rate: float = 3e-5,
    val_fraction: float = 0.1,
    num_loader_workers: int = 4,
    per_rank_batch_size: int = 2,
    training_seed: int = 149,
    precision: str = "bf16",
    checkpoint_interval_steps: int = 256,
) -> ReactiveRayOutput:
    """Fine-tune trajectory and route on the frozen KITScenes split."""
    return train_reactive_stage_ray_8(
        shards=kitscenes_shards,
        stage="kitscenes_finetune",
        parent_checkpoint=nuplan_epoch5_checkpoint,
        resume_checkpoint=None,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=1e-2,
        grad_clip=1.0,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        per_rank_batch_size=per_rank_batch_size,
        training_seed=training_seed,
        precision=precision,
        gradient_accumulation_steps=1,
        steps_per_epoch=0,
        checkpoint_interval_steps=checkpoint_interval_steps,
        shuffle_buffer=512,
        is_pretrained=True,
        trajectory_weight=1.0,
        bev_weight=0.0,
        route_weight=1.0,
        corridor_pos_weight=1.0,
        freeze_bevformer=True,
        capacity_block_end_utc=capacity_block_end_utc,
        training_scope="multitask",
        validation_sample_limit=3820,
        parent_profile="nuplan_trajectory_route_v1",
        parent_checkpoint_sha256=(
            NUPLAN_EPOCH5_TRAJECTORY_ROUTE_CHECKPOINT_SHA256
        ),
        parent_checkpoint_epoch=5,
    )


@workflow
def wf_evaluate_reactive_kitscenes(
    checkpoint: FlyteFile,
    evaluation_shards: List[FlyteDirectory],
    source_split: str = "val",
    expected_model_version: str = "",
) -> ReactiveKITScenesEvaluationWorkflowOutput:
    """Evaluate the selected checkpoint on an official labeled split."""
    evaluation = evaluate_reactive_kitscenes_checkpoint(
        checkpoint=checkpoint,
        shards=evaluation_shards,
        source_split=source_split,
        batch_size=1,
        num_loader_workers=4,
    )
    publication = publish_reactive_kitscenes_evaluation(
        checkpoint=checkpoint,
        report=evaluation.report,
        report_sha256=evaluation.report_sha256,
        checkpoint_sha256=evaluation.checkpoint_sha256,
        checkpoint_epoch=evaluation.checkpoint_epoch,
        expected_model_version=expected_model_version,
    )
    return ReactiveKITScenesEvaluationWorkflowOutput(
        report=evaluation.report,
        report_sha256=evaluation.report_sha256,
        checkpoint_sha256=evaluation.checkpoint_sha256,
        checkpoint_epoch=evaluation.checkpoint_epoch,
        mlflow_run_id=publication.mlflow_run_id,
        registered_model_name=publication.registered_model_name,
        registered_model_version=publication.registered_model_version,
    )


@workflow
def wf_evaluate_reactive_kitscenes_val(
    checkpoint: FlyteFile,
    evaluation_shards: List[FlyteDirectory],
    expected_model_version: str,
) -> ReactiveKITScenesEvaluationWorkflowOutput:
    """Evaluate the official KITScenes val split with map and route."""
    evaluation = evaluate_reactive_kitscenes_checkpoint(
        checkpoint=checkpoint,
        shards=evaluation_shards,
        source_split="val",
        batch_size=KITSCENES_PRIMARY_VAL_BATCH_SIZE,
        num_loader_workers=4,
        expected_val_partition_count=KITSCENES_OFFICIAL_VAL_SCENE_COUNT,
    )
    publication = publish_reactive_kitscenes_evaluation(
        checkpoint=checkpoint,
        report=evaluation.report,
        report_sha256=evaluation.report_sha256,
        checkpoint_sha256=evaluation.checkpoint_sha256,
        checkpoint_epoch=evaluation.checkpoint_epoch,
        expected_model_version=expected_model_version,
    )
    return ReactiveKITScenesEvaluationWorkflowOutput(
        report=evaluation.report,
        report_sha256=evaluation.report_sha256,
        checkpoint_sha256=evaluation.checkpoint_sha256,
        checkpoint_epoch=evaluation.checkpoint_epoch,
        mlflow_run_id=publication.mlflow_run_id,
        registered_model_name=publication.registered_model_name,
        registered_model_version=publication.registered_model_version,
    )


@workflow
def wf_evaluate_reactive_kitscenes_test_sharded(
    checkpoint: FlyteFile,
    evaluation_shards: List[FlyteDirectory],
    expected_model_version: str,
) -> ReactiveKITScenesEvaluationWorkflowOutput:
    """Evaluate the official KITScenes test set with one model load."""
    evaluation = evaluate_reactive_kitscenes_checkpoint(
        checkpoint=checkpoint,
        shards=evaluation_shards,
        source_split="test",
        batch_size=KITSCENES_STATEFUL_CAMERA_FPN_BATCH_SIZE,
        num_loader_workers=4,
        expected_test_partition_count=(
            KITSCENES_OFFICIAL_TEST_SCENE_COUNT
        ),
        use_stateful_camera_fpn_cache=True,
    )
    publication = publish_reactive_kitscenes_evaluation(
        checkpoint=checkpoint,
        report=evaluation.report,
        report_sha256=evaluation.report_sha256,
        checkpoint_sha256=evaluation.checkpoint_sha256,
        checkpoint_epoch=evaluation.checkpoint_epoch,
        expected_model_version=expected_model_version,
    )
    return ReactiveKITScenesEvaluationWorkflowOutput(
        report=evaluation.report,
        report_sha256=evaluation.report_sha256,
        checkpoint_sha256=evaluation.checkpoint_sha256,
        checkpoint_epoch=evaluation.checkpoint_epoch,
        mlflow_run_id=publication.mlflow_run_id,
        registered_model_name=publication.registered_model_name,
        registered_model_version=publication.registered_model_version,
    )


@workflow
def wf_train_reactive_nuplan_bev_ray_8(
    nuplan_shards: List[FlyteDirectory],
    capacity_block_end_utc: str,
    resume_checkpoint: Optional[FlyteDirectory] = None,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    bev_encoder_learning_rate: float = 1e-5,
    val_fraction: float = 0.1,
    num_loader_workers: int = 4,
    per_rank_batch_size: int = 4,
    training_seed: int = 149,
    precision: str = "bf16",
    checkpoint_interval_steps: int = 256,
) -> ReactiveRayOutput:
    """Fine-tune BEVFormer V2 and its nuPlan segmentation head only."""
    return train_reactive_stage_ray_8(
        shards=nuplan_shards,
        stage="nuplan_full",
        parent_checkpoint=None,
        resume_checkpoint=resume_checkpoint,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=1e-2,
        grad_clip=1.0,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        per_rank_batch_size=per_rank_batch_size,
        training_seed=training_seed,
        precision=precision,
        gradient_accumulation_steps=1,
        steps_per_epoch=0,
        checkpoint_interval_steps=checkpoint_interval_steps,
        shuffle_buffer=512,
        is_pretrained=True,
        trajectory_weight=0.0,
        bev_weight=1.0,
        route_weight=0.0,
        corridor_pos_weight=1.0,
        freeze_bevformer=False,
        capacity_block_end_utc=capacity_block_end_utc,
        training_scope="bev_only",
        bev_encoder_learning_rate=bev_encoder_learning_rate,
        bev_repeat_frequency_threshold=0.05,
        validation_sample_limit=4096,
    )


@workflow
def wf_train_reactive_nuplan_bev_ray_1_smoke(
    nuplan_shards: List[FlyteDirectory],
    epochs: int = 1,
    steps_per_epoch: int = 8,
) -> ReactiveRayOutput:
    """Exercise the production BEV objective on one validation GPU."""
    return train_reactive_nuplan_bev_ray_1_smoke(
        shards=nuplan_shards,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )


@workflow
def wf_train_reactive_nuplan_bev_ray_2_canary(
    nuplan_shards: List[FlyteDirectory],
    epochs: int = 2,
    steps_per_epoch: int = 16,
) -> ReactiveRayOutput:
    """Smoke-test the BEV-only objective without enforcing the 8-GPU gate."""
    return train_reactive_stage_ray_2(
        shards=nuplan_shards,
        stage="nuplan_full",
        parent_checkpoint=None,
        resume_checkpoint=None,
        epochs=epochs,
        learning_rate=1e-4,
        weight_decay=1e-2,
        grad_clip=1.0,
        val_fraction=0.1,
        num_loader_workers=2,
        per_rank_batch_size=1,
        training_seed=149,
        precision="bf16",
        gradient_accumulation_steps=1,
        steps_per_epoch=steps_per_epoch,
        checkpoint_interval_steps=8,
        shuffle_buffer=64,
        is_pretrained=True,
        trajectory_weight=0.0,
        bev_weight=1.0,
        route_weight=0.0,
        corridor_pos_weight=1.0,
        freeze_bevformer=False,
        training_scope="bev_only",
        bev_encoder_learning_rate=1e-5,
    )


@workflow
def wf_train_reactive_nuplan_bev_ray_8_canary(
    nuplan_shards: List[FlyteDirectory],
    capacity_block_end_utc: str,
    epochs: int = 2,
    steps_per_epoch: int = 128,
    validation_sample_limit: int = 1024,
) -> ReactiveBEVCanaryOutput:
    """Exercise the exact production BEV-only topology on real nuPlan data."""
    training = train_reactive_stage_ray_8(
        shards=nuplan_shards,
        stage="nuplan_full",
        parent_checkpoint=None,
        resume_checkpoint=None,
        epochs=epochs,
        learning_rate=1e-4,
        weight_decay=1e-2,
        grad_clip=1.0,
        val_fraction=0.1,
        num_loader_workers=4,
        per_rank_batch_size=4,
        training_seed=149,
        precision="bf16",
        gradient_accumulation_steps=1,
        steps_per_epoch=steps_per_epoch,
        checkpoint_interval_steps=32,
        shuffle_buffer=512,
        is_pretrained=True,
        trajectory_weight=0.0,
        bev_weight=1.0,
        route_weight=0.0,
        corridor_pos_weight=1.0,
        freeze_bevformer=False,
        capacity_block_end_utc=capacity_block_end_utc,
        training_scope="bev_only",
        bev_encoder_learning_rate=1e-5,
        bev_repeat_frequency_threshold=0.05,
        validation_sample_limit=validation_sample_limit,
        allow_bounded_bev_canary=True,
    )
    gate_report = verify_reactive_bev_canary_training(
        metadata=training.metadata,
    )
    return ReactiveBEVCanaryOutput(
        checkpoint=training.checkpoint,
        metadata=training.metadata,
        checkpoint_uri=training.checkpoint_uri,
        checkpoint_sha256=training.checkpoint_sha256,
        gate_report=gate_report,
    )


@workflow
def wf_train_reactive_nuplan_from_bev_ray_8(
    nuplan_shards: List[FlyteDirectory],
    bev_checkpoint: FlyteFile,
    capacity_block_end_utc: str,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    val_fraction: float = 0.1,
    num_loader_workers: int = 4,
    per_rank_batch_size: int = 4,
    training_seed: int = 149,
    precision: str = "bf16",
    checkpoint_interval_steps: int = 256,
) -> ReactiveRayOutput:
    """Train nuPlan trajectory and route from a frozen BEV-only parent."""
    return train_reactive_stage_ray_8(
        shards=nuplan_shards,
        stage="nuplan_full",
        parent_checkpoint=bev_checkpoint,
        resume_checkpoint=None,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=1e-2,
        grad_clip=1.0,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        per_rank_batch_size=per_rank_batch_size,
        training_seed=training_seed,
        precision=precision,
        gradient_accumulation_steps=1,
        steps_per_epoch=0,
        checkpoint_interval_steps=checkpoint_interval_steps,
        shuffle_buffer=512,
        is_pretrained=True,
        trajectory_weight=1.0,
        bev_weight=0.0,
        route_weight=1.0,
        corridor_pos_weight=1.0,
        freeze_bevformer=True,
        capacity_block_end_utc=capacity_block_end_utc,
        training_scope="multitask",
    )


@workflow
def wf_train_reactive_nuplan_l2d_ray_8(
    nuplan_shards: List[FlyteDirectory],
    l2d_shards: List[FlyteDirectory],
    capacity_block_end_utc: str,
    stage_a_epochs: int = 3,
    stage_b_epochs: int = 3,
    stage_a_learning_rate: float = 1e-4,
    stage_b_learning_rate: float = 3e-5,
    val_fraction: float = 0.1,
    num_loader_workers: int = 2,
    per_rank_batch_size: int = 4,
    training_seed: int = 149,
    precision: str = "bf16",
    trajectory_weight: float = 1.0,
    bev_weight: float = 1.0,
    route_weight: float = 1.0,
) -> ReactiveDistributedProgramOutput:
    """Run both production stages with the camera BEV frozen."""
    stage_a = train_reactive_stage_ray_8(
        shards=nuplan_shards,
        stage="nuplan_full",
        parent_checkpoint=None,
        resume_checkpoint=None,
        epochs=stage_a_epochs,
        learning_rate=stage_a_learning_rate,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        per_rank_batch_size=per_rank_batch_size,
        training_seed=training_seed,
        precision=precision,
        trajectory_weight=trajectory_weight,
        bev_weight=bev_weight,
        route_weight=route_weight,
        freeze_bevformer=True,
        capacity_block_end_utc=capacity_block_end_utc,
    )
    stage_b = train_reactive_stage_ray_8(
        shards=l2d_shards,
        stage="l2d_continuation",
        parent_checkpoint=stage_a.checkpoint,
        resume_checkpoint=None,
        epochs=stage_b_epochs,
        learning_rate=stage_b_learning_rate,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        per_rank_batch_size=per_rank_batch_size,
        training_seed=training_seed,
        precision=precision,
        trajectory_weight=trajectory_weight,
        bev_weight=0.0,
        route_weight=route_weight,
        freeze_bevformer=True,
        capacity_block_end_utc=capacity_block_end_utc,
    )
    return ReactiveDistributedProgramOutput(
        stage_a_checkpoint=stage_a.checkpoint,
        stage_a_metadata=stage_a.metadata,
        stage_a_checkpoint_uri=stage_a.checkpoint_uri,
        stage_a_checkpoint_sha256=stage_a.checkpoint_sha256,
        stage_b_checkpoint=stage_b.checkpoint,
        stage_b_metadata=stage_b.metadata,
        stage_b_checkpoint_uri=stage_b.checkpoint_uri,
        stage_b_checkpoint_sha256=stage_b.checkpoint_sha256,
    )


@workflow
def wf_reactive_multistage_ray_2_canary() -> ReactiveCanaryOutput:
    """Run a fast random-init plumbing canary through two RayJobs."""
    stage_a_data = build_reactive_canary_dataset(
        stage="nuplan_full"
    )
    stage_b_data = build_reactive_canary_dataset(
        stage="l2d_continuation"
    )
    stage_a = train_reactive_stage_ray_2(
        shards=[stage_a_data],
        stage="nuplan_full",
        parent_checkpoint=None,
        epochs=3,
        learning_rate=3e-4,
        val_fraction=0.5,
        num_loader_workers=1,
        precision="fp32",
        steps_per_epoch=4,
        shuffle_buffer=8,
        is_pretrained=False,
        allow_random_bevformer_init=True,
        trajectory_weight=0.1,
        bev_weight=1.0,
        route_weight=0.1,
        freeze_bevformer=True,
    )
    stage_b = train_reactive_stage_ray_2(
        shards=[stage_b_data],
        stage="l2d_continuation",
        parent_checkpoint=stage_a.checkpoint,
        epochs=2,
        learning_rate=1e-4,
        val_fraction=0.5,
        num_loader_workers=1,
        precision="fp32",
        steps_per_epoch=2,
        shuffle_buffer=8,
        is_pretrained=False,
        allow_random_bevformer_init=True,
        bev_weight=0.0,
        route_weight=0.1,
        freeze_bevformer=True,
    )
    gate_report = verify_reactive_canary_training(
        stage_a_metadata=stage_a.metadata,
        stage_b_metadata=stage_b.metadata,
    )
    return ReactiveCanaryOutput(
        stage_a_checkpoint=stage_a.checkpoint,
        stage_b_checkpoint=stage_b.checkpoint,
        stage_a_metadata=stage_a.metadata,
        stage_b_metadata=stage_b.metadata,
        gate_report=gate_report,
    )
