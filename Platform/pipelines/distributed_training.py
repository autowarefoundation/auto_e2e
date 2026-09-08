"""Flyte Ray tasks for distributed AutoE2E training."""

from __future__ import annotations

import json
import math
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
BEV_POS_WEIGHT_CAP = 64.0
BEV_CANARY_MIN_SELECTION_GAIN = 1e-3
BEV_CANARY_AP_LIFT_ABSOLUTE_REGRESSION_TOLERANCE = 1e-3
BEV_CANARY_RARE_AP_LIFT_RELATIVE_REGRESSION_TOLERANCE = 0.1
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
    result = run_reactive_stage({
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
    )


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="8", mem="64Gi", gpu="1"),
    limits=Resources(cpu="8", mem="64Gi", gpu="1"),
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
        if "validation_sample_limit" not in config:
            raise ValueError(
                "nuPlan checkpoint lacks validation_sample_limit provenance"
            )
        validation_sample_limit = int(config["validation_sample_limit"])
        if validation_sample_limit <= 0:
            raise ValueError(
                "nuPlan checkpoint validation_sample_limit is invalid"
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
                sample_limit=validation_sample_limit,
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
    return ReactiveBEVEvaluationOutput(
        report=FlyteFile(str(output)),
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        checkpoint_sha256=checkpoint_sha256,
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
) -> ReactiveBEVEvaluationOutput:
    return evaluate_reactive_bev_checkpoint(
        checkpoint=checkpoint,
        shards=shards,
        dataset=dataset,
        benchmark_inventory=benchmark_inventory,
        split=split,
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_loader_workers=num_loader_workers,
        probability_bins=probability_bins,
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
