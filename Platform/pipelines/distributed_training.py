"""Flyte Ray tasks for distributed AutoE2E training."""

from __future__ import annotations

import json
import os
import re
from datetime import timedelta
from pathlib import Path
from typing import List, NamedTuple, Optional

from flytekit import (
    PodTemplate,
    Resources,
    current_context,
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

TRAINING_IMAGE = os.environ.get(
    "AUTO_E2E_TRAINING_IMAGE",
    "auto-e2e/training:latest",
)
RAY_STORAGE_PATH = os.environ.get(
    "AUTO_E2E_RAY_STORAGE_PATH",
    "s3://auto-e2e-platform-checkpoints/ray-train",
)
RAY_TASK_ENVIRONMENT = {
    "AWS_DEFAULT_REGION": "us-west-2",
    "AUTO_E2E_RAY_STORAGE_PATH": RAY_STORAGE_PATH,
    "RAY_TRAIN_V2_ENABLED": "1",
}
BEV_POS_WEIGHT_CAP = 2048.0
P5EN_TASK_MAX_RUNTIME = timedelta(hours=20)
P5EN_SMOKE_MAX_RUNTIME = timedelta(minutes=30)


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
                            value="DETAIL",
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
    return 3 if num_workers == 2 else 12


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


def _run_reactive_stage_task(
    *,
    shards: List[FlyteDirectory],
    stage: str,
    num_workers: int,
    parent_checkpoint: Optional[FlyteFile],
    backbone: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    val_fraction: float,
    num_loader_workers: int,
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
    allow_random_bevformer_init: bool = False,
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
    result = run_reactive_stage({
        "allow_random_bevformer_init": allow_random_bevformer_init,
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
        "per_rank_batch_size": 1,
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
        "trajectory_weight": trajectory_weight,
        "use_gpu": True,
        "val_fraction": val_fraction,
        "validation_sample_limit": validation_sample_limit,
        "weight_decay": weight_decay,
        "worker_cpus": _reactive_worker_cpus(num_workers),
    })
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
                "recall",
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
    backbone: str = "res_net_50",
    epochs: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    val_fraction: float = 0.1,
    num_loader_workers: int = 2,
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
    allow_random_bevformer_init: bool = False,
) -> ReactiveRayOutput:
    """Run two-rank training with the production pretrained default."""
    return _run_reactive_stage_task(
        shards=shards,
        stage=stage,
        num_workers=2,
        parent_checkpoint=parent_checkpoint,
        backbone=backbone,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
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
        allow_random_bevformer_init=allow_random_bevformer_init,
    )


@task(
    task_config=RAY_REACTIVE_4,
    container_image=TRAINING_IMAGE,
    retries=2,
    timeout=P5EN_TASK_MAX_RUNTIME,
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
    backbone: str = "res_net_50",
    epochs: int = 3,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    val_fraction: float = 0.2,
    num_loader_workers: int = 2,
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
) -> ReactiveRayOutput:
    """Run a four-rank Reactive performance training stage."""
    return _run_reactive_stage_task(
        shards=shards,
        stage=stage,
        num_workers=4,
        parent_checkpoint=parent_checkpoint,
        backbone=backbone,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
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
    )


@task(
    task_config=RAY_8,
    container_image=TRAINING_IMAGE,
    retries=1,
    timeout=P5EN_TASK_MAX_RUNTIME,
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
    backbone: str = "res_net_50",
    epochs: int = 3,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    val_fraction: float = 0.1,
    num_loader_workers: int = 2,
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
) -> ReactiveRayOutput:
    """Run one production-size Reactive DDP stage."""
    return _run_reactive_stage_task(
        shards=shards,
        stage=stage,
        num_workers=8,
        parent_checkpoint=parent_checkpoint,
        backbone=backbone,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
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
def wf_train_reactive_nuplan_ray_4(
    nuplan_shards: List[FlyteDirectory],
    capacity_block_end_utc: str,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    val_fraction: float = 0.2,
    num_loader_workers: int = 2,
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
        epochs=epochs,
        learning_rate=learning_rate,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
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
    epochs: int = 3,
    learning_rate: float = 1e-4,
    val_fraction: float = 0.1,
    num_loader_workers: int = 2,
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
        epochs=epochs,
        learning_rate=learning_rate,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
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
        epochs=stage_a_epochs,
        learning_rate=stage_a_learning_rate,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
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
        epochs=stage_b_epochs,
        learning_rate=stage_b_learning_rate,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
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
        shuffle_buffer=0,
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
        shuffle_buffer=0,
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
