"""Flyte training workflows for AutoE2E.

Provides:
- train_one: single kfpytorch task (PyTorchJob via Kueue)
- train_single: workflow wrapping one training run
- sweep: @dynamic workflow iterating over backbone x fusion combos

Enum choices are built from the model component registries at registration
time. Re-register (pyflyte register) when a new backbone/fusion is added.
"""

from enum import Enum
from itertools import product
from typing import List

from flytekit import dynamic, task, workflow, Resources, ImageSpec
from flytekit.models.literals import Blob
from flytekitplugins.kfpytorch import PyTorch, Master, Worker

# ---------------------------------------------------------------------------
# Dynamic enums from component registries
# ---------------------------------------------------------------------------

# These are materialized at `pyflyte register` time. The actual keys come from
# the model code — keep in sync by re-registering after registry changes.
Backbone = Enum("Backbone", {
    "swin_v2_tiny": "swin_v2_tiny",
    "conv_next_v2_tiny": "conv_next_v2_tiny",
    "res_net_50": "res_net_50",
})

FusionMode = Enum("FusionMode", {
    "concat": "concat",
    "cross_attn": "cross_attn",
    "bev": "bev",
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAINING_IMAGE = (
    "{ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/auto-e2e/training:latest"
)
MLFLOW_URI = "http://mlflow.mlflow.svc.cluster.local:5000"


# ---------------------------------------------------------------------------
# kfpytorch task: single training run
# ---------------------------------------------------------------------------

@task(
    task_config=PyTorch(
        master=Master(replicas=1),
    ),
    requests=Resources(cpu="6", mem="40Gi", gpu="1"),
    limits=Resources(gpu="1"),
    container_image=TRAINING_IMAGE,
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI, "AWS_DEFAULT_REGION": "us-west-2"},
    labels={
        "kueue.x-k8s.io/queue-name": "gpu-queue",
        "kueue.x-k8s.io/priority-class": "research-low",
    },
    pod_template_name="training-pod",
)
def train_one(
    backbone: str,
    fusion_mode: str,
    batch_size: int = 8,
    epochs: int = 20,
    lr: float = 1e-4,
    dataset: str = "yaak-ai/L2D",
) -> str:
    """Run train.py as a PyTorchJob on the GPU node."""
    import subprocess
    import sys

    cmd = [
        sys.executable, "Model/training/train.py",
        f"--backbone={backbone}",
        f"--fusion-mode={fusion_mode}",
        f"--batch-size={batch_size}",
        f"--epochs={epochs}",
        f"--lr={lr}",
        f"--dataset={dataset}",
        "--amp",
        "--save-dir=/tmp/ckpt",
        "--register-model",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\n{result.stderr}")
    return result.stdout


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------

@workflow
def train_single(
    backbone: Backbone = Backbone.swin_v2_tiny,
    fusion_mode: FusionMode = FusionMode.concat,
    batch_size: int = 8,
    epochs: int = 20,
    lr: float = 1e-4,
    dataset: str = "yaak-ai/L2D",
) -> str:
    """Single training run with enum dropdowns in Flyte UI."""
    return train_one(
        backbone=backbone.value,
        fusion_mode=fusion_mode.value,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        dataset=dataset,
    )


@dynamic
def sweep(
    backbones: List[str] = ["swin_v2_tiny", "conv_next_v2_tiny"],
    fusion_modes: List[str] = ["concat", "cross_attn"],
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 1e-4,
    dataset: str = "yaak-ai/L2D",
) -> List[str]:
    """Fan out backbone x fusion_mode combos. Kueue serializes on 1 GPU."""
    results = []
    for bb, fm in product(backbones, fusion_modes):
        out = train_one(
            backbone=bb,
            fusion_mode=fm,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            dataset=dataset,
        )
        results.append(out)
    return results
