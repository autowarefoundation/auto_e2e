"""Flyte tasks that train an AutoE2E stage and validate its checkpoints alongside.

``train`` (a dynamic node of ``wf_autoe2e``) starts two tasks that do not
depend on each other, so Flyte runs them at the same time:

- the Ray task trains on the GPU profile's worker group and reports a
  checkpoint about every 30 minutes of training to ``run_uri``;
- ``validate_checkpoints`` runs on one validation GPU, picks up each
  checkpoint as it lands, writes its ``validation.json`` and tells the trainer
  to lower the AutoBEV learning rates when the BEV quality drops.

``select_checkpoints`` then names the best and the final checkpoint for
registration and the formal evaluation that follows. The lightweight
validation results are not recorded on any MLflow model version.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import List

from flytekit import PodTemplate, Resources, task
from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig
from kubernetes.client import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1PodSpec,
    V1ResourceRequirements,
    V1Toleration,
    V1Volume,
    V1VolumeMount,
)

TRAINING_IMAGE = os.environ.get("AUTO_E2E_TRAINING_IMAGE", "auto-e2e/training:latest")
RAY_STORAGE_PATH = os.environ.get("AUTO_E2E_RAY_STORAGE_PATH", "s3://auto-e2e-platform-checkpoints/ray-train")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.mlflow.svc.cluster.local:5000")
TRAIN_ENVIRONMENT = {
    "AWS_DEFAULT_REGION": "us-west-2",
    "MLFLOW_TRACKING_URI": MLFLOW_URI,
    "RAY_TRAIN_V2_ENABLED": "1",
    "AUTO_E2E_GIT_COMMIT": os.environ.get("AUTO_E2E_GIT_COMMIT", "unknown"),
    "AUTO_E2E_IMAGE_DIGEST": TRAINING_IMAGE,
}
QUEUES = {
    "gpu-validation": {"kueue.x-k8s.io/queue-name": "gpu-validation", "kueue.x-k8s.io/priority-class": "research-low"},
    "p5en": {"kueue.x-k8s.io/queue-name": "p5en-capacity-block", "kueue.x-k8s.io/priority-class": "production-high"},
}


class GpuProfile(enum.Enum):
    RAY_1 = "ray_1"  # one validation GPU; smoke only
    RAY_8 = "ray_8"  # one 8-GPU node


@dataclass
class TrainingSpec:
    """What a run trains; every other training value is a constant in Model/training."""

    stage: int = 2
    gpu_profile: GpuProfile = GpuProfile.RAY_8
    # 0 evaluates the parent without training.
    steps: int = 0
    learning_rate: float = 2e-4
    per_rank_batch_size: int = 1
    # About 30 minutes: 8 ranks at about 6 s per stage 2 sample.
    checkpoint_interval_steps: int = 300
    seed: int = 0


@dataclass
class TrainedRun:
    trained: bool
    run_uri: str = ""
    run_name: str = ""


@dataclass
class TrainingData:
    """Partition URIs per role and the dataset ids recorded in every checkpoint."""

    primary: List[str] = field(default_factory=list)
    mix: List[str] = field(default_factory=list)
    early: List[str] = field(default_factory=list)
    dataset_ids: List[str] = field(default_factory=list)


def _shm(size: str) -> tuple[V1VolumeMount, V1Volume]:
    return (V1VolumeMount(name="dshm", mount_path="/dev/shm"),
            V1Volume(name="dshm", empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit=size)))


def _pod(name: str, *, workload: str | None, resources: dict[str, str], shm: str) -> PodTemplate:
    mount, volume = _shm(shm)
    return PodTemplate(
        primary_container_name=name,
        annotations={"karpenter.sh/do-not-disrupt": "true"},
        pod_spec=V1PodSpec(
            service_account_name="default",
            node_selector={"workload-type": workload} if workload else None,
            tolerations=[V1Toleration(key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")]
            if workload else None,
            containers=[V1Container(
                name=name, volume_mounts=[mount],
                resources=V1ResourceRequirements(requests=dict(resources), limits=dict(resources)),
            )],
            volumes=[volume],
        ),
    )


def _ray(workload: str, *, gpus: int, cpu: str, memory: str, shm: str, storage: str) -> RayJobConfig:
    return RayJobConfig(
        head_node_config=HeadNodeConfig(
            ray_start_params={"dashboard-host": "0.0.0.0", "num-cpus": "0"},
            pod_template=_pod("ray-head", workload=None, resources={"cpu": "2", "memory": "16Gi"}, shm="8Gi"),
        ),
        worker_node_config=[WorkerNodeConfig(
            group_name="gpu-workers", replicas=1, min_replicas=1, max_replicas=1,
            ray_start_params={"num-cpus": cpu, "num-gpus": str(gpus)},
            pod_template=_pod("ray-worker", workload=workload, shm=shm, resources={
                "cpu": cpu, "memory": memory, "nvidia.com/gpu": str(gpus), "ephemeral-storage": storage}),
        )],
        enable_autoscaling=False,
        address="auto",
        shutdown_after_job_finishes=True,
        ttl_seconds_after_finished=300,
    )


RAY_1 = _ray("gpu-validation", gpus=1, cpu="6", memory="48Gi", shm="8Gi", storage="200Gi")
RAY_8 = _ray("p5en-capacity-block", gpus=8, cpu="96", memory="1Ti", shm="128Gi", storage="1500Gi")
VALIDATION_POD = _pod("primary", workload="gpu-validation", shm="16Gi",
                      resources={"cpu": "6", "memory": "48Gi", "nvidia.com/gpu": "1", "ephemeral-storage": "400Gi"})


def _run(spec: TrainingSpec, data: TrainingData, run: TrainedRun, parent_uri: str, world_size: int):
    from training.trainer import TrainRun

    return TrainRun(
        stage=spec.stage, run_name=run.run_name, storage_uri=RAY_STORAGE_PATH,
        primary_shards=list(data.primary), mix_shards=list(data.mix), early_shards=list(data.early),
        dataset_ids=list(data.dataset_ids), parent_checkpoint_uri=parent_uri, steps=spec.steps,
        learning_rate=spec.learning_rate, per_rank_batch_size=spec.per_rank_batch_size,
        checkpoint_interval_steps=spec.checkpoint_interval_steps, seed=spec.seed, world_size=world_size,
    )


def _train(spec: TrainingSpec, data: TrainingData, run: TrainedRun, parent_uri: str, world_size: int) -> TrainedRun:
    from training.trainer import run_training

    run_training(_run(spec, data, run, parent_uri, world_size))
    return run


@task(task_config=RAY_1, container_image=TRAINING_IMAGE, labels=QUEUES["gpu-validation"],
      environment=TRAIN_ENVIRONMENT, retries=1)
def train_ray_1(spec: TrainingSpec, data: TrainingData, run: TrainedRun, parent_uri: str) -> TrainedRun:
    return _train(spec, data, run, parent_uri, 1)


@task(task_config=RAY_8, container_image=TRAINING_IMAGE, labels=QUEUES["p5en"],
      environment=TRAIN_ENVIRONMENT, retries=1)
def train_ray_8(spec: TrainingSpec, data: TrainingData, run: TrainedRun, parent_uri: str) -> TrainedRun:
    return _train(spec, data, run, parent_uri, 8)


TRAIN_TASKS = {GpuProfile.RAY_1: train_ray_1, GpuProfile.RAY_8: train_ray_8}


@task(container_image=TRAINING_IMAGE, labels=QUEUES["gpu-validation"], pod_template=VALIDATION_POD,
      requests=Resources(cpu="6", mem="48Gi", gpu="1", ephemeral_storage="400Gi"),
      limits=Resources(cpu="6", mem="48Gi", gpu="1", ephemeral_storage="400Gi"),
      environment=TRAIN_ENVIRONMENT, retries=2)
def validate_checkpoints(spec: TrainingSpec, data: TrainingData, run: TrainedRun, parent_uri: str) -> int:
    """Validate the run's checkpoints as they appear; returns how many this attempt validated."""
    from training.autobev_data import autobev_validation_batches
    from evaluation.checkpoint_validation import CALIBRATION_SAMPLES, VALIDATION_SAMPLES, watch

    results = watch(
        _run(spec, data, run, parent_uri, 1),
        calibration=lambda: autobev_validation_batches(data.primary, split="calibration", limit=CALIBRATION_SAMPLES),
        validation=lambda: autobev_validation_batches(data.primary, split="validation", limit=VALIDATION_SAMPLES),
    )
    return len(results)


@dataclass
class TrainedCheckpoints:
    """The checkpoint files the run produced: the best by validation and the last."""

    trained: bool
    run_id: str = ""
    best_uri: str = ""
    best_sha256: str = ""
    best_step: int = 0
    final_uri: str = ""
    final_sha256: str = ""
    final_step: int = 0
    best_metrics_json: str = "{}"
    final_metrics_json: str = "{}"


# On the calibration split (not the held-out Pittsburgh split the formal
# evaluation reports): the stage 2 checkpoint with the lowest 6.4 s ADE, and the
# stage 1 checkpoint with the best drivable-area IoU over the forward bands.
SELECTION_METRIC = {2: ("ade/6.4s", min), 1: ("drivable", max)}


@task(container_image=TRAINING_IMAGE, requests=Resources(cpu="2", mem="8Gi"), environment=TRAIN_ENVIRONMENT,
      retries=2)
def select_checkpoints(spec: TrainingSpec, run: TrainedRun, validated: int) -> TrainedCheckpoints:
    """Pick the best and the final checkpoint from the validation results and open the run's MLflow run."""
    import json

    from mlflow import MlflowClient

    from evaluation.checkpoint_validation import RESULT_FILE, list_checkpoints, read_json, drivable_score
    from training.storage import sha256
    from training.trainer import CHECKPOINT_FILE

    results = [read_json(f"{checkpoint}/{RESULT_FILE}") for checkpoint in list_checkpoints(run.run_uri)]
    final = max(results, key=lambda result: result["step"])
    if final["step"] != spec.steps:
        raise RuntimeError(f"{run.run_uri}: the last validated step is {final['step']}, not {spec.steps}")
    for result in results:
        result["calibration_metrics"]["drivable"] = drivable_score(result["calibration_metrics"])
    name, choose = SELECTION_METRIC[spec.stage]
    best = choose((r for r in results if name in r["calibration_metrics"]),
                  key=lambda result: result["calibration_metrics"][name])

    client = MlflowClient(tracking_uri=MLFLOW_URI)
    experiment = client.get_experiment_by_name(os.environ.get("AUTO_E2E_MLFLOW_EXPERIMENT", "auto-e2e"))
    mlflow_run = client.create_run(experiment.experiment_id, run_name=run.run_name,
                                   tags={"run_uri": run.run_uri, "stage": str(spec.stage)})
    client.set_terminated(mlflow_run.info.run_id)
    best_uri, final_uri = (f"{r['checkpoint']}/{CHECKPOINT_FILE}" for r in (best, final))
    return TrainedCheckpoints(
        trained=True, run_id=mlflow_run.info.run_id,
        best_uri=best_uri, best_sha256=sha256(best_uri), best_step=int(best["step"]),
        final_uri=final_uri, final_sha256=sha256(final_uri), final_step=int(final["step"]),
        best_metrics_json=json.dumps(best["metrics"], sort_keys=True),
        final_metrics_json=json.dumps(final["metrics"], sort_keys=True),
    )
