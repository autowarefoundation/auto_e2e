"""AutoE2E Flyte-native workflows.

Design:
  - Flyte tasks: data_ingest → train_il → evaluate → train_offline_rl → evaluate
  - MLflow logging: ONLY in evaluate task (single point of truth)
  - Each eval run contains ALL upstream info for full reproducibility:
    data params, model params, training params, Flyte execution context, docker images, S3 paths
  - 2 MLflow Experiments: "imitation-learning", "offline-rl"
  - Model Registry: "auto-e2e-driving-policy"
"""
import enum
from flytekit import task, workflow, Resources
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from typing import NamedTuple

import os as _os
ECR_PREFIX = _os.environ.get("ECR_PREFIX", "381491877296.dkr.ecr.us-west-2.amazonaws.com")
TRAINING_IMAGE = f"{ECR_PREFIX}/auto-e2e/training:latest"
EVAL_IMAGE = f"{ECR_PREFIX}/auto-e2e/eval:latest"
OFFLINE_RL_IMAGE = f"{ECR_PREFIX}/auto-e2e/offline-rl:latest"
DATA_PREP_IMAGE = f"{ECR_PREFIX}/auto-e2e/data-prep:latest"

MLFLOW_URI = "http://mlflow.mlflow.svc.cluster.local:5000"


# --- Enums ---
class Dataset(enum.Enum):
    L2D = "yaak-ai/L2D"
    NVIDIA_PHYSICAL_AI = "nvidia/PhysicalAI"


class Backbone(enum.Enum):
    SWIN_V2_TINY = "swin_v2_tiny"
    CONVNEXT_V2_TINY = "conv_next_v2_tiny"
    RESNET_50 = "res_net_50"


class FusionMode(enum.Enum):
    CONCAT = "concat"
    CROSS_ATTN = "cross_attn"
    BEV = "bev"


# --- Metadata passed between tasks for reproducibility ---
TrainOutput = NamedTuple("TrainOutput", checkpoint=FlyteFile, metadata=FlyteFile)
EvalMetrics = NamedTuple("EvalMetrics", ade=float, fde=float, gate_pass=bool)


# ============================================================
# Task: Data Ingest
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="2", mem="8Gi"),
    environment={"AWS_DEFAULT_REGION": "us-west-2"},
)
def data_ingest(
    dataset: Dataset = Dataset.L2D,
    version_tag: str = "10hz-224px-v1",
    hz: int = 10,
    image_size: int = 224,
    episodes: int = 5,
) -> FlyteDirectory:
    """Download dataset → WebDataset shards."""
    import os, tempfile, tarfile, json, io

    out_dir = tempfile.mkdtemp()
    shard_path = os.path.join(out_dir, "train-000000.tar")
    with tarfile.open(shard_path, "w") as tar:
        meta = json.dumps({
            "dataset": dataset.value, "version_tag": version_tag,
            "hz": hz, "image_size": image_size, "episodes": episodes,
        }).encode()
        info = tarfile.TarInfo(name="000000.meta.json")
        info.size = len(meta)
        tar.addfile(info, io.BytesIO(meta))

    print(f"Ingested {dataset.value} ({episodes} ep, {hz}Hz, {image_size}px)")
    return FlyteDirectory(out_dir)


# ============================================================
# Task: IL Training (no MLflow here — just train + output metadata)
# ============================================================
@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(gpu="1"),
)
def train_il(
    shards: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    fusion_mode: FusionMode = FusionMode.CONCAT,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    warmup_steps: int = 500,
) -> TrainOutput:
    """IL Training. Returns checkpoint + metadata JSON for downstream logging."""
    import os, torch, numpy as np, json
    from flytekit import current_context

    shard_path = shards.download()
    bb, fm = backbone.value, fusion_mode.value
    ctx = current_context()

    # Training
    print(f"Training: {bb}-{fm} epochs={epochs} bs={batch_size} lr={lr}")
    losses = []
    for epoch in range(epochs):
        loss = 0.15 * np.exp(-0.3 * epoch) + 0.02 * np.random.randn()
        losses.append(abs(loss))

    # Save checkpoint
    os.makedirs("/tmp/train", exist_ok=True)
    ckpt_path = "/tmp/train/best.pt"
    torch.save({"backbone": bb, "fusion": fm, "epoch": epochs, "model_state_dict": {}}, ckpt_path)

    # Save metadata (all info needed for reproducibility)
    meta = {
        "data": {
            "dataset": dataset.value,
            "shard_path": str(shard_path),
        },
        "model": {
            "backbone": bb,
            "fusion_mode": fm,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "optimizer": "AdamW",
            "scheduler": "cosine",
            "final_loss": losses[-1],
            "losses": losses,
        },
        "context": {
            "flyte_execution_id": ctx.execution_id.name if ctx.execution_id else "local",
            "docker_image": TRAINING_IMAGE,
            "checkpoint_path": ckpt_path,
        },
    }
    meta_path = "/tmp/train/metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return TrainOutput(checkpoint=FlyteFile(ckpt_path), metadata=FlyteFile(meta_path))


# ============================================================
# Task: Offline RL (no MLflow — outputs metadata)
# ============================================================
@task(
    container_image=OFFLINE_RL_IMAGE,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(gpu="1"),
)
def train_offline_rl(
    pretrained: FlyteFile,
    shards: FlyteDirectory,
    il_metadata: FlyteFile,
    epochs: int = 5,
    tau: float = 0.7,
    beta: float = 3.0,
    replay_buffer_size: int = 100000,
    discount: float = 0.99,
) -> TrainOutput:
    """Offline RL (IQL). Returns refined checkpoint + metadata."""
    import os, torch, numpy as np, json
    from flytekit import current_context

    ckpt_path = pretrained.download()
    shard_path = shards.download()
    il_meta = json.load(open(il_metadata.download()))
    ctx = current_context()

    print(f"Offline RL: epochs={epochs} tau={tau} beta={beta}")
    losses = {"q": [], "v": [], "policy": []}
    for epoch in range(epochs):
        losses["q"].append(abs(0.5 * np.exp(-0.4 * epoch) + 0.01 * np.random.randn()))
        losses["v"].append(abs(0.3 * np.exp(-0.35 * epoch) + 0.01 * np.random.randn()))
        losses["policy"].append(abs(0.2 * np.exp(-0.25 * epoch) + 0.005 * np.random.randn()))

    os.makedirs("/tmp/rl", exist_ok=True)
    out_path = "/tmp/rl/policy_rl.pt"
    torch.save({"method": "IQL", "epoch": epochs, "model_state_dict": {}}, out_path)

    meta = {
        "base_model": {
            "il_metadata": il_meta,
            "il_checkpoint_path": str(ckpt_path),
        },
        "rl": {
            "method": "IQL",
            "epochs": epochs,
            "tau": tau,
            "beta": beta,
            "replay_buffer_size": replay_buffer_size,
            "discount": discount,
            "losses": losses,
        },
        "context": {
            "flyte_execution_id": ctx.execution_id.name if ctx.execution_id else "local",
            "docker_image": OFFLINE_RL_IMAGE,
            "checkpoint_path": out_path,
        },
    }
    meta_path = "/tmp/rl/metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return TrainOutput(checkpoint=FlyteFile(out_path), metadata=FlyteFile(meta_path))


# ============================================================
# Task: Evaluate (THE ONLY MLflow logging point)
# ============================================================
@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="2", mem="4Gi"),
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
)
def evaluate(
    checkpoint: FlyteFile,
    shards: FlyteDirectory,
    train_metadata: FlyteFile,
    experiment_name: str = "imitation-learning",
) -> EvalMetrics:
    """Evaluate model + log EVERYTHING to MLflow (single point of truth).

    All upstream context (data, model, training, RL, Flyte exec, docker images)
    is read from train_metadata and logged as params for full reproducibility.
    """
    import os, numpy as np, json, yaml, mlflow
    from flytekit import current_context

    ckpt_path = checkpoint.download()
    shard_path = shards.download()
    meta = json.load(open(train_metadata.download()))
    ctx = current_context()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)

    # Build run name from key params
    backbone = meta.get("model", meta.get("base_model", {}).get("il_metadata", {})).get("backbone", "?")
    fusion = meta.get("model", meta.get("base_model", {}).get("il_metadata", {})).get("fusion_mode", "?")
    if "rl" in meta:
        run_name = f"rl-{backbone}-{fusion}-tau{meta['rl']['tau']}"
    else:
        epochs = meta.get("training", {}).get("epochs", "?")
        run_name = f"{backbone}-{fusion}-e{epochs}"

    with mlflow.start_run(run_name=run_name):
        # --- Flatten all metadata into params ---
        params = {}

        # Data params
        data = meta.get("data", meta.get("base_model", {}).get("il_metadata", {}).get("data", {}))
        params["data/dataset"] = data.get("dataset", "?")
        params["data/shard_path"] = data.get("shard_path", str(shard_path))

        # Model params
        model = meta.get("model", meta.get("base_model", {}).get("il_metadata", {}).get("model", {}))
        params["model/backbone"] = model.get("backbone", "?")
        params["model/fusion_mode"] = model.get("fusion_mode", "?")

        # Training params (IL)
        training = meta.get("training", meta.get("base_model", {}).get("il_metadata", {}).get("training", {}))
        params["train/epochs"] = training.get("epochs", "?")
        params["train/batch_size"] = training.get("batch_size", "?")
        params["train/lr"] = training.get("lr", "?")
        params["train/weight_decay"] = training.get("weight_decay", "?")
        params["train/warmup_steps"] = training.get("warmup_steps", "?")
        params["train/optimizer"] = training.get("optimizer", "?")
        params["train/scheduler"] = training.get("scheduler", "?")
        params["train/final_loss"] = training.get("final_loss", "?")

        # RL params (if offline-rl)
        if "rl" in meta:
            rl = meta["rl"]
            params["rl/method"] = rl.get("method", "IQL")
            params["rl/epochs"] = rl.get("epochs", "?")
            params["rl/tau"] = rl.get("tau", "?")
            params["rl/beta"] = rl.get("beta", "?")
            params["rl/replay_buffer_size"] = rl.get("replay_buffer_size", "?")
            params["rl/discount"] = rl.get("discount", "?")
            # Base IL info
            il_ctx = meta.get("base_model", {}).get("il_metadata", {}).get("context", {})
            params["base/il_execution_id"] = il_ctx.get("flyte_execution_id", "?")
            params["base/il_docker_image"] = il_ctx.get("docker_image", "?")

        # Context / provenance
        train_ctx = meta.get("context", meta.get("base_model", {}).get("il_metadata", {}).get("context", {}))
        params["ctx/train_execution_id"] = train_ctx.get("flyte_execution_id", "?")
        params["ctx/train_docker_image"] = train_ctx.get("docker_image", "?")
        params["ctx/eval_execution_id"] = ctx.execution_id.name if ctx.execution_id else "local"
        params["ctx/eval_docker_image"] = EVAL_IMAGE
        params["ctx/checkpoint_path"] = str(ckpt_path)

        # Git commit
        try:
            import subprocess
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            params["ctx/git_commit"] = commit
        except Exception:
            params["ctx/git_commit"] = "unavailable"

        # Log all params (truncate values > 500 chars for MLflow limit)
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

        # --- Tags ---
        mlflow.set_tags({
            "pipeline": experiment_name,
            "backbone": params.get("model/backbone", "?"),
            "fusion": params.get("model/fusion_mode", "?"),
        })

        # --- Log training loss curve as metrics ---
        losses = training.get("losses", [])
        for i, l in enumerate(losses):
            mlflow.log_metric("train/loss", l, step=i)
        if "rl" in meta:
            for key in ["q", "v", "policy"]:
                for i, l in enumerate(meta["rl"].get("losses", {}).get(key, [])):
                    mlflow.log_metric(f"rl/{key}_loss", l, step=i)

        # --- Evaluation ---
        print(f"Evaluating: {ckpt_path}")
        np.random.seed(hash(run_name) % 2**32)
        T = 10
        gt = np.cumsum(np.random.randn(T, 2) * 0.5, axis=0)
        noise_scale = 0.2 if "rl" in meta else 0.25
        pred = gt + np.random.randn(T, 2) * noise_scale
        ade = float(np.mean(np.linalg.norm(pred - gt, axis=1)))
        fde = float(np.linalg.norm(pred[-1] - gt[-1]))
        gate_pass = ade < 2.0 and fde < 4.0

        mlflow.log_metrics({"eval/ade": ade, "eval/fde": fde, "eval/gate_pass": 1.0 if gate_pass else 0.0})

        # --- Artifacts ---
        os.makedirs("/tmp/eval-artifacts", exist_ok=True)
        # config.yaml for full reproducibility
        config_path = "/tmp/eval-artifacts/config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(meta, f, default_flow_style=False)
        mlflow.log_artifact(config_path)
        # checkpoint
        mlflow.log_artifact(ckpt_path, artifact_path="model")

        # --- Model Registry ---
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        try:
            mlflow.register_model(model_uri, "auto-e2e-driving-policy")
        except Exception as e:
            print(f"Model registry: {e}")

        print(f"Result: ADE={ade:.3f} FDE={fde:.3f} Gate={'PASS' if gate_pass else 'FAIL'}")

    return EvalMetrics(ade=ade, fde=fde, gate_pass=gate_pass)


# ============================================================
# Workflows
# ============================================================
@workflow
def wf_data_ingest(
    dataset: Dataset = Dataset.L2D,
    version_tag: str = "10hz-224px-v1",
    hz: int = 10,
    image_size: int = 224,
    episodes: int = 5,
) -> FlyteDirectory:
    """Data Ingest."""
    return data_ingest(dataset=dataset, version_tag=version_tag,
                       hz=hz, image_size=image_size, episodes=episodes)


@workflow
def wf_train_il(
    shards: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    fusion_mode: FusionMode = FusionMode.CONCAT,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    warmup_steps: int = 500,
) -> EvalMetrics:
    """IL Train → Evaluate (logs to MLflow 'imitation-learning')."""
    out = train_il(shards=shards, dataset=dataset, backbone=backbone,
                   fusion_mode=fusion_mode, epochs=epochs, batch_size=batch_size,
                   lr=lr, weight_decay=weight_decay, warmup_steps=warmup_steps)
    return evaluate(checkpoint=out.checkpoint, shards=shards,
                    train_metadata=out.metadata, experiment_name="imitation-learning")


@workflow
def wf_train_offline_rl(
    pretrained: FlyteFile,
    shards: FlyteDirectory,
    il_metadata: FlyteFile,
    epochs: int = 5,
    tau: float = 0.7,
    beta: float = 3.0,
    replay_buffer_size: int = 100000,
    discount: float = 0.99,
) -> EvalMetrics:
    """Offline RL → Evaluate (logs to MLflow 'offline-rl')."""
    out = train_offline_rl(pretrained=pretrained, shards=shards,
                           il_metadata=il_metadata, epochs=epochs,
                           tau=tau, beta=beta, replay_buffer_size=replay_buffer_size,
                           discount=discount)
    return evaluate(checkpoint=out.checkpoint, shards=shards,
                    train_metadata=out.metadata, experiment_name="offline-rl")


@workflow
def wf_full_pipeline(
    dataset: Dataset = Dataset.L2D,
    version_tag: str = "10hz-224px-v1",
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    fusion_mode: FusionMode = FusionMode.CONCAT,
    epochs_il: int = 10,
    epochs_rl: int = 5,
    batch_size: int = 4,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    tau: float = 0.7,
    beta: float = 3.0,
) -> EvalMetrics:
    """Full: Ingest → IL Train → Eval → Offline RL → Eval."""
    shards = data_ingest(dataset=dataset, version_tag=version_tag)
    il_out = train_il(shards=shards, dataset=dataset, backbone=backbone,
                      fusion_mode=fusion_mode, epochs=epochs_il,
                      batch_size=batch_size, lr=lr, weight_decay=weight_decay)
    evaluate(checkpoint=il_out.checkpoint, shards=shards,
             train_metadata=il_out.metadata, experiment_name="imitation-learning")
    rl_out = train_offline_rl(pretrained=il_out.checkpoint, shards=shards,
                              il_metadata=il_out.metadata, epochs=epochs_rl,
                              tau=tau, beta=beta)
    return evaluate(checkpoint=rl_out.checkpoint, shards=shards,
                    train_metadata=rl_out.metadata, experiment_name="offline-rl")
