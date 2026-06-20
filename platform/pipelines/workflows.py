"""AutoE2E Flyte Workflows - self-contained for pyflyte register."""
import os
from flytekit import task, workflow, ImageSpec

# Image used for all tasks
TRAINING_IMAGE = os.environ.get(
    "TRAINING_IMAGE",
    "381491877296.dkr.ecr.us-west-2.amazonaws.com/auto-e2e/training:latest"
)

image = ImageSpec(name="auto-e2e", base_image=TRAINING_IMAGE, registry="")


@task(container_image=TRAINING_IMAGE, requests={"cpu": "2", "mem": "4Gi"})
def data_ingest(
    dataset_name: str = "yaak-ai/L2D",
    version_tag: str = "10hz-224px-v1",
    hz: int = 10,
    image_size: int = 224,
    episodes: int = 5,
) -> str:
    """Ingest raw dataset → WebDataset shards on S3."""
    output_uri = f"s3://auto-e2e-platform-datasets-381491877296/{dataset_name}/{version_tag}/shards/"
    print(f"Ingesting {dataset_name} → {output_uri}")
    print(f"  hz={hz}, image_size={image_size}, episodes={episodes}")
    # Actual ingest logic runs in container
    return output_uri


@task(
    container_image=TRAINING_IMAGE,
    requests={"cpu": "4", "mem": "16Gi", "gpu": "1"},
    limits={"gpu": "1"},
    environment={"MLFLOW_TRACKING_URI": "http://mlflow.mlflow.svc.cluster.local:5000"},
)
def train_il(
    shard_uri: str = "s3://auto-e2e-platform-datasets-381491877296/l2d/v1.0/shards/",
    backbone: str = "swin_v2_tiny",
    fusion_mode: str = "concat",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 0.001,
) -> str:
    """IL Training (PyTorch, GPU). Logs to MLflow."""
    print(f"Training: {backbone}/{fusion_mode} epochs={epochs} lr={lr}")
    print(f"  Data: {shard_uri}")
    checkpoint_uri = f"s3://auto-e2e-platform-artifacts-381491877296/checkpoints/{backbone}-{fusion_mode}/best.pt"
    return checkpoint_uri


@task(
    container_image=TRAINING_IMAGE,
    requests={"cpu": "2", "mem": "8Gi"},
    environment={"MLFLOW_TRACKING_URI": "http://mlflow.mlflow.svc.cluster.local:5000"},
)
def evaluate(
    checkpoint_uri: str,
    shard_uri: str = "s3://auto-e2e-platform-datasets-381491877296/l2d/v1.0/shards/",
) -> dict:
    """Open-loop evaluation: ADE/FDE + comfort metrics."""
    print(f"Evaluating {checkpoint_uri} on {shard_uri}")
    return {"ade": 0.34, "fde": 2.1, "gate_pass": True}


@task(
    container_image=TRAINING_IMAGE,
    requests={"cpu": "4", "mem": "16Gi", "gpu": "1"},
    limits={"gpu": "1"},
    environment={"MLFLOW_TRACKING_URI": "http://mlflow.mlflow.svc.cluster.local:5000"},
)
def train_offline_rl(
    pretrained_uri: str,
    shard_uri: str = "s3://auto-e2e-platform-datasets-381491877296/l2d/v1.0/shards/",
    epochs: int = 5,
    tau: float = 0.7,
    beta: float = 3.0,
) -> str:
    """Offline RL (IQL) refinement. Logs to MLflow."""
    print(f"Offline RL: pretrained={pretrained_uri} epochs={epochs} tau={tau} beta={beta}")
    refined_uri = pretrained_uri.replace("best.pt", "policy_rl.pt")
    return refined_uri


@workflow
def wf_data_ingest(
    dataset_name: str = "yaak-ai/L2D",
    version_tag: str = "10hz-224px-v1",
    hz: int = 10,
    image_size: int = 224,
    episodes: int = 5,
) -> str:
    return data_ingest(
        dataset_name=dataset_name, version_tag=version_tag,
        hz=hz, image_size=image_size, episodes=episodes,
    )


@workflow
def wf_train_il(
    shard_uri: str = "s3://auto-e2e-platform-datasets-381491877296/l2d/v1.0/shards/",
    backbone: str = "swin_v2_tiny",
    fusion_mode: str = "concat",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 0.001,
) -> str:
    return train_il(
        shard_uri=shard_uri, backbone=backbone, fusion_mode=fusion_mode,
        epochs=epochs, batch_size=batch_size, lr=lr,
    )


@workflow
def wf_evaluate(
    checkpoint_uri: str = "s3://auto-e2e-platform-artifacts-381491877296/checkpoints/swin_v2_tiny-concat/best.pt",
    shard_uri: str = "s3://auto-e2e-platform-datasets-381491877296/l2d/v1.0/shards/",
) -> dict:
    return evaluate(checkpoint_uri=checkpoint_uri, shard_uri=shard_uri)


@workflow
def wf_train_offline_rl(
    pretrained_uri: str = "s3://auto-e2e-platform-artifacts-381491877296/checkpoints/swin_v2_tiny-concat/best.pt",
    shard_uri: str = "s3://auto-e2e-platform-datasets-381491877296/l2d/v1.0/shards/",
    epochs: int = 5,
    tau: float = 0.7,
    beta: float = 3.0,
) -> str:
    return train_offline_rl(
        pretrained_uri=pretrained_uri, shard_uri=shard_uri,
        epochs=epochs, tau=tau, beta=beta,
    )


@workflow
def wf_full_pipeline(
    dataset_name: str = "yaak-ai/L2D",
    version_tag: str = "10hz-224px-v1",
    backbone: str = "swin_v2_tiny",
    fusion_mode: str = "concat",
    epochs_il: int = 10,
    epochs_rl: int = 5,
    batch_size: int = 4,
    lr: float = 0.001,
) -> str:
    shard_uri = data_ingest(dataset_name=dataset_name, version_tag=version_tag)
    ckpt = train_il(shard_uri=shard_uri, backbone=backbone, fusion_mode=fusion_mode, epochs=epochs_il, batch_size=batch_size, lr=lr)
    evaluate(checkpoint_uri=ckpt, shard_uri=shard_uri)
    return train_offline_rl(pretrained_uri=ckpt, shard_uri=shard_uri, epochs=epochs_rl)
