"""Flyte training workflows for AutoE2E.

Provides:
- train_one: RawContainerTask (PyTorchJob rendered by Flyte, no flytekit in training image)
- train_single: workflow wrapping one training run
- sweep: @dynamic workflow iterating over backbone x fusion combos

Training image does NOT contain flytekit — Flyte propeller renders the
PyTorchJob CRD and the pod runs plain `python train.py`. This keeps the
training image minimal (torch + timm + webdataset + mlflow-skinny only).
"""

from enum import Enum
from itertools import product
from typing import List

from flytekit import dynamic, workflow
from flytekit.core.raw_container_task import RawContainerTask

# ---------------------------------------------------------------------------
# Dynamic enums from component registries
# ---------------------------------------------------------------------------

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
# Constants — resolved from environment at registration time
# ---------------------------------------------------------------------------

import os

_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "381491877296")
_REGION = os.environ.get("AWS_REGION", "us-west-2")

TRAINING_IMAGE = f"{_ACCOUNT_ID}.dkr.ecr.{_REGION}.amazonaws.com/auto-e2e/training:latest"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://172.20.240.62:5000")


# ---------------------------------------------------------------------------
# RawContainerTask: training image has no flytekit dependency
# ---------------------------------------------------------------------------

def make_train_task(
    backbone: str,
    fusion_mode: str,
    batch_size: int = 8,
    epochs: int = 20,
    lr: float = 1e-4,
    dataset: str = "yaak-ai/L2D",
    dataset_format: str = "pre_extracted",
    shard_dir: str = "/data/shards",
    priority: str = "research-low",
) -> RawContainerTask:
    """Create a RawContainerTask for one training run.

    The training pod runs plain `python train.py` — no flytekit installed.
    Flyte propeller handles scheduling, monitoring, and output collection.
    """
    name = f"train-{backbone}-{fusion_mode}".replace("_", "-")

    args = [
        "python", "Model/training/train.py",
        f"--backbone={backbone}",
        f"--fusion-mode={fusion_mode}",
        f"--batch-size={batch_size}",
        f"--epochs={epochs}",
        f"--lr={lr}",
        f"--dataset={dataset}",
        f"--dataset-format={dataset_format}",
        f"--shard-dir={shard_dir}",
        "--amp",
        "--save-dir=/tmp/ckpt",
        "--register-model",
    ]

    return RawContainerTask(
        name=name,
        image=TRAINING_IMAGE,
        command=args,
        requests={"cpu": "6", "mem": "40Gi", "gpu": "1"},
        limits={"gpu": "1"},
        environment={
            "MLFLOW_TRACKING_URI": MLFLOW_URI,
            "AWS_DEFAULT_REGION": "us-west-2",
        },
        pod_labels={
            "kueue.x-k8s.io/queue-name": "gpu-queue",
            "kueue.x-k8s.io/priority-class": priority,
        },
        pod_annotations={
            "karpenter.sh/do-not-disrupt": "true",
        },
        node_selector={"workload-type": "gpu-training"},
        tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
    )


# Pre-defined default task for Flyte UI
train_default = make_train_task(
    backbone="swin_v2_tiny",
    fusion_mode="concat",
)


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
) -> None:
    """Single training run with enum dropdowns in Flyte UI."""
    task = make_train_task(
        backbone=backbone.value,
        fusion_mode=fusion_mode.value,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        dataset=dataset,
    )
    task()


@dynamic
def sweep(
    backbones: List[str] = ["swin_v2_tiny", "conv_next_v2_tiny"],
    fusion_modes: List[str] = ["concat", "cross_attn"],
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 1e-4,
    dataset: str = "yaak-ai/L2D",
) -> None:
    """Fan out backbone x fusion_mode combos. Kueue serializes on 1 GPU."""
    for bb, fm in product(backbones, fusion_modes):
        task = make_train_task(
            backbone=bb, fusion_mode=fm,
            batch_size=batch_size, epochs=epochs, lr=lr, dataset=dataset,
        )
        task()
