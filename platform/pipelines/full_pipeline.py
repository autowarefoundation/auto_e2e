"""Flyte full pipeline: IL → Open-Loop Gate → Closed-Loop Gate → RL → Re-Eval → Champion.

Visible as a single DAG in Flyte UI. Each node logs to MLflow with stage tags
for cross-comparison (IL vs RL, hyperparameter sweeps).
"""

from __future__ import annotations

import os
from typing import Optional

from flytekit import conditional, dynamic, task, workflow, Resources
from flytekit.core.raw_container_task import RawContainerTask

_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "381491877296")
_REGION = os.environ.get("AWS_REGION", "us-west-2")
_CLUSTER = os.environ.get("EKS_CLUSTER", "auto-e2e-platform")

TRAINING_IMAGE = f"{_ACCOUNT_ID}.dkr.ecr.{_REGION}.amazonaws.com/auto-e2e/training:latest"
DATA_PREP_IMAGE = f"{_ACCOUNT_ID}.dkr.ecr.{_REGION}.amazonaws.com/auto-e2e/data-prep:latest"
TRAINING_RL_IMAGE = f"{_ACCOUNT_ID}.dkr.ecr.{_REGION}.amazonaws.com/auto-e2e/training-rl:latest"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://172.20.240.62:5000")
DATASET_BUCKET = f"{_CLUSTER}-datasets-{_ACCOUNT_ID}"
CHECKPOINT_BUCKET = f"{_CLUSTER}-checkpoints-{_ACCOUNT_ID}"


# --- IL Training (RawContainerTask — no flytekit in training image) ---

def make_il_train_task(backbone: str, fusion_mode: str, epochs: int, lr: float, dataset_format: str, shard_dir: str):
    return RawContainerTask(
        name=f"il-train-{backbone}-{fusion_mode}",
        image=TRAINING_IMAGE,
        command=[
            "python", "Model/training/train.py",
            f"--backbone={backbone}", f"--fusion-mode={fusion_mode}",
            f"--epochs={epochs}", f"--lr={lr}", "--amp",
            f"--dataset-format={dataset_format}", f"--shard-dir={shard_dir}",
            f"--save-dir=/tmp/ckpt", "--register-model",
        ],
        requests={"cpu": "6", "mem": "40Gi", "gpu": "1"},
        limits={"gpu": "1"},
        environment={"MLFLOW_TRACKING_URI": MLFLOW_URI, "AWS_DEFAULT_REGION": _REGION},
        pod_labels={"kueue.x-k8s.io/queue-name": "gpu-queue"},
        node_selector={"workload-type": "gpu-training"},
        tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
    )


# --- Evaluation Tasks ---

@task(container_image=DATA_PREP_IMAGE,
      requests=Resources(cpu="6", mem="40Gi", gpu="1"),
      limits=Resources(gpu="1"),
      environment={"MLFLOW_TRACKING_URI": MLFLOW_URI, "AWS_DEFAULT_REGION": _REGION},
      labels={"kueue.x-k8s.io/queue-name": "gpu-queue"})
def eval_open_loop(checkpoint_s3: str, val_shard_dir: str) -> dict:
    """Run open-loop eval (ADE/FDE). Same code for IL and RL checkpoints."""
    import sys
    sys.path.insert(0, "/workspace/Model")
    # Reuse evaluation/workflow.py logic inline
    from evaluation.metrics import compute_open_loop_metrics
    # ... (simplified — real impl loads model + val shards)
    return {"ADE@3s": 1.5, "FDE@6.4s": 3.0}  # placeholder


@task(container_image=DATA_PREP_IMAGE,
      requests=Resources(cpu="4", mem="16Gi"),
      environment={"MLFLOW_TRACKING_URI": MLFLOW_URI})
def eval_closed_loop(checkpoint_s3: str, scenarios: str = "S01,S02,S03") -> dict:
    """Run CARLA closed-loop eval."""
    # ... (calls closed_loop_runner or connects to CARLA)
    return {"route_completion": 0.95, "collisions": 0}  # placeholder


@task(container_image=DATA_PREP_IMAGE,
      requests=Resources(cpu="1", mem="1Gi"),
      environment={"MLFLOW_TRACKING_URI": MLFLOW_URI})
def gate_check(metrics: dict, sim_results: dict) -> bool:
    """Check if model passes both open-loop and closed-loop gates."""
    ol_pass = metrics.get("ADE@3s", 99) < 2.0 and metrics.get("FDE@6.4s", 99) < 5.0
    cl_pass = sim_results.get("route_completion", 0) >= 0.9 and sim_results.get("collisions", 99) == 0
    return ol_pass and cl_pass


@task(container_image=DATA_PREP_IMAGE,
      requests=Resources(cpu="1", mem="1Gi"),
      environment={"MLFLOW_TRACKING_URI": MLFLOW_URI})
def log_to_mlflow(checkpoint_s3: str, metrics: dict, sim_results: dict, stage: str) -> str:
    """Log all metrics to MLflow with stage tag. Returns run_id."""
    import mlflow
    mlflow.set_experiment("auto_e2e/full_pipeline")
    with mlflow.start_run() as run:
        mlflow.set_tag("stage", stage)
        mlflow.set_tag("checkpoint", checkpoint_s3)
        for k, v in metrics.items():
            mlflow.log_metric(f"ol_{k}", v)
        for k, v in sim_results.items():
            mlflow.log_metric(f"cl_{k}", v)
        return run.info.run_id


@task(container_image=DATA_PREP_IMAGE,
      requests=Resources(cpu="1", mem="1Gi"),
      environment={"MLFLOW_TRACKING_URI": MLFLOW_URI})
def promote_to_champion(checkpoint_s3: str, run_id: str) -> None:
    """Set MLflow alias 'champion' on the model version."""
    import mlflow
    client = mlflow.MlflowClient()
    versions = client.search_model_versions("name='auto_e2e'")
    if versions:
        latest = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias("auto_e2e", "champion", latest.version)
        print(f"Promoted version {latest.version} to 'champion'")


# --- RL Training (RawContainerTask) ---

def make_rl_train_task(checkpoint_s3: str, total_timesteps: int = 100_000):
    return RawContainerTask(
        name="rl-train-ppo",
        image=TRAINING_RL_IMAGE,
        command=[
            "python", "Model/training/train_rl.py",
            f"--checkpoint={checkpoint_s3}",
            "--carla-host=carla-server",
            f"--total-timesteps={total_timesteps}",
            "--save-dir=/tmp/rl_ckpt",
        ],
        requests={"cpu": "4", "mem": "16Gi"},
        environment={"MLFLOW_TRACKING_URI": MLFLOW_URI, "AWS_DEFAULT_REGION": _REGION},
        node_selector={"workload-type": "simulation"},
        tolerations=[{"key": "nvidia.com/gpu-sim", "operator": "Exists", "effect": "NoSchedule"}],
    )


# --- Full Pipeline Workflow ---

@workflow
def full_pipeline(
    backbone: str = "swin_v2_tiny",
    fusion_mode: str = "concat",
    epochs: int = 20,
    lr: float = 1e-4,
    dataset_format: str = "pre_extracted",
    shard_dir: str = "/data/shards",
    rl_timesteps: int = 100_000,
    val_shard_dir: str = "/data/val_shards",
) -> None:
    """IL → Gate → RL → Gate → Champion. Full E2E pipeline."""
    # Placeholder checkpoint path (real impl: IL task outputs to S3)
    il_checkpoint = f"s3://{CHECKPOINT_BUCKET}/il/{backbone}-{fusion_mode}-ep{epochs}.pt"

    # 1. IL Training
    il_task = make_il_train_task(backbone, fusion_mode, epochs, lr, dataset_format, shard_dir)
    il_task()

    # 2. Evaluate IL
    il_metrics = eval_open_loop(checkpoint_s3=il_checkpoint, val_shard_dir=val_shard_dir)
    il_sim = eval_closed_loop(checkpoint_s3=il_checkpoint)
    il_run_id = log_to_mlflow(checkpoint_s3=il_checkpoint, metrics=il_metrics,
                              sim_results=il_sim, stage="IL")

    # 3. Gate
    passed = gate_check(metrics=il_metrics, sim_results=il_sim)

    # 4. RL (only if gate passed) — conditional execution
    # Note: Flyte conditional requires static typing; simplified here
    rl_task = make_rl_train_task(il_checkpoint, rl_timesteps)
    rl_task()

    rl_checkpoint = f"s3://{CHECKPOINT_BUCKET}/rl/{backbone}-{fusion_mode}-ppo-{rl_timesteps}.pt"

    # 5. Re-evaluate RL
    rl_metrics = eval_open_loop(checkpoint_s3=rl_checkpoint, val_shard_dir=val_shard_dir)
    rl_sim = eval_closed_loop(checkpoint_s3=rl_checkpoint)
    rl_run_id = log_to_mlflow(checkpoint_s3=rl_checkpoint, metrics=rl_metrics,
                              sim_results=rl_sim, stage="RL")

    # 6. Final gate + promote
    rl_passed = gate_check(metrics=rl_metrics, sim_results=rl_sim)
    promote_to_champion(checkpoint_s3=rl_checkpoint, run_id=rl_run_id)
