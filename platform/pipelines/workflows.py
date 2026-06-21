"""AutoE2E Flyte-native workflows — Real Training Pipeline.

Architecture:
  data_ingest → data_processing → train_il → evaluate
                                      ↓
                              train_offline_rl → evaluate

MLflow: Only evaluate task logs. 2 experiments: imitation-learning, offline-rl.
"""
import enum
from flytekit import task, workflow, Resources, Secret
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from typing import NamedTuple, Optional

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
    NVIDIA_PHYSICAL_AI = "nvidia/PhysicalAI-Autonomous-Vehicles"


class Backbone(enum.Enum):
    SWIN_V2_TINY = "swin_v2_tiny"
    CONVNEXT_V2_TINY = "conv_next_v2_tiny"
    RESNET_50 = "res_net_50"


class FusionMode(enum.Enum):
    CONCAT = "concat"
    CROSS_ATTN = "cross_attn"
    BEV = "bev"


TrainOutput = NamedTuple("TrainOutput", checkpoint=FlyteFile, metadata=FlyteFile)
EvalMetrics = NamedTuple("EvalMetrics", ade=float, fde=float, gate_pass=bool)


# ============================================================
# Task: Data Ingest (download raw from HuggingFace)
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="2", mem="24Gi"),
)
def data_ingest(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
    hf_token: str = "",
) -> FlyteDirectory:
    """Download raw dataset from HuggingFace (lerobot for L2D, physical_ai_av for NVIDIA)."""
    import os, shutil
    from huggingface_hub import login

    token = hf_token or os.environ.get("HF_TOKEN", "")
    if token:
        login(token=token)
        os.environ["HF_TOKEN"] = token

    out_dir = "/tmp/raw_data"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

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
        clip_ids = ds.clip_index.index.tolist()[:episodes]
        feats = CAMERAS + ["egomotion"]
        for clip_id in clip_ids:
            ds.download_clip_features(clip_id, features=feats)
            for cam in CAMERAS:
                cf = ds.features.get_chunk_feature_filename(ds.get_clip_chunk(clip_id), cam)
                with ds.open_file(cf, maybe_stream=True) as f:
                    unpack_camera_zip(f.read(), clip_id, cam, out)
            cf = ds.features.get_chunk_feature_filename(ds.get_clip_chunk(clip_id), "egomotion")
            with ds.open_file(cf, maybe_stream=True) as f:
                unpack_egomotion_zip(f.read(), clip_id, out)
        print(f"Ingested {dataset.value}: {len(clip_ids)} clips → {out_dir}")
        return FlyteDirectory(out_dir)

    # L2D: lerobot
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from ledataset.datasets.lerobot_dataset import LeRobotDataset

    ep_list = list(range(episodes)) if episodes > 0 else None
    ds = LeRobotDataset(repo_id=dataset.value, episodes=ep_list)
    cache_dir = ds.root
    shutil.copytree(str(cache_dir), out_dir)

    print(f"Ingested {dataset.value}: {len(ds)} frames, {episodes} episodes → {out_dir}")
    return FlyteDirectory(out_dir)


# ============================================================
# Task: Data Processing (Issue #30: pre-extract frames)
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="4", mem="16Gi"),
)
def data_processing(
    raw_data: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    hz: int = 10,
    image_size: int = 256,
    episodes: int = 3,
) -> FlyteDirectory:
    """Pre-extract aligned frames + egomotion → WebDataset shards.

    Solves Issue #30: no video decode at training time.
    """
    import os, io, json, tarfile, tempfile
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    raw_path = raw_data.download()
    print(f"Processing raw data from: {raw_path} (dataset={dataset.value})")

    # Build the appropriate Dataset (both emit the same sample schema:
    # visual_tiles (V,3,H,W), egomotion_history (256), trajectory_target (128)).
    ep_list = list(range(episodes)) if episodes > 0 else None
    if dataset == Dataset.NVIDIA_PHYSICAL_AI:
        from data_parsing.nvidia_physical_ai.dataset import NvidiaAVDataset
        ds = NvidiaAVDataset(data_root=raw_path)
        n_samples = len(ds)
        idx_iter = range(n_samples)
    else:
        from data_parsing.l2d import L2DDataset
        ds = L2DDataset(repo_id=dataset.value, episodes=ep_list)
        n_samples = len(ds)
        idx_iter = range(n_samples)

    to_pil = transforms.ToPILImage()
    resize = transforms.Resize((image_size, image_size))
    out_dir = tempfile.mkdtemp()
    shard_idx = 0
    sample_count = 0
    samples_per_shard = 1000
    current_tar = None

    def open_new_shard():
        nonlocal current_tar, shard_idx
        if current_tar:
            current_tar.close()
        current_tar = tarfile.open(os.path.join(out_dir, f"train-{shard_idx:06d}.tar"), "w")
        shard_idx += 1

    open_new_shard()

    for si in idx_iter:
        sample = ds[si]
        visual = sample["visual_tiles"]            # (V, 3, H, W) tensor
        ego_hist = sample["egomotion_history"]     # (256,)
        traj = sample["trajectory_target"]         # (128,)
        ego_data = np.concatenate([
            ego_hist.numpy() if torch.is_tensor(ego_hist) else np.asarray(ego_hist),
            traj.numpy() if torch.is_tensor(traj) else np.asarray(traj),
        ]).astype(np.float32)

        sample_key = f"s{si:08d}"
        for cam_i in range(visual.shape[0]):
            frame = to_pil(visual[cam_i].cpu().clamp(0, 1) if visual[cam_i].dtype.is_floating_point else visual[cam_i])
            frame = resize(frame)
            buf = io.BytesIO()
            frame.save(buf, format="JPEG", quality=90)
            jpg = buf.getvalue()
            info = tarfile.TarInfo(name=f"{sample_key}.cam_{cam_i}.jpg")
            info.size = len(jpg)
            current_tar.addfile(info, io.BytesIO(jpg))

        eb = ego_data.tobytes()
        info = tarfile.TarInfo(name=f"{sample_key}.ego.npy")
        info.size = len(eb)
        current_tar.addfile(info, io.BytesIO(eb))

        m = json.dumps({"idx": si, "dataset": dataset.value}).encode()
        info = tarfile.TarInfo(name=f"{sample_key}.meta.json")
        info.size = len(m)
        current_tar.addfile(info, io.BytesIO(m))

        sample_count += 1
        if sample_count % samples_per_shard == 0:
            open_new_shard()

    if current_tar:
        current_tar.close()

    manifest = {"total_samples": sample_count, "shards": shard_idx,
                "hz": hz, "image_size": image_size, "dataset": dataset.value,
                "episodes": episodes, "num_views": int(visual.shape[0]) if sample_count else 0}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    print(f"Processed {dataset.value}: {sample_count} samples → {shard_idx} shards")
    return FlyteDirectory(out_dir)


# ============================================================
# Task: IL Training (real AutoE2E)
# ============================================================
@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(gpu="1"),
)
def train_il(
    shards: FlyteDirectory,
    shards2: Optional[FlyteDirectory] = None,
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    fusion_mode: FusionMode = FusionMode.CONCAT,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    amp: bool = True,
) -> TrainOutput:
    """Train AutoE2E model on pre-extracted WebDataset shards."""
    import os, json, torch
    import numpy as np
    from flytekit import current_context

    from model_components.auto_e2e import AutoE2E
    from model_components.losses import TrajectoryImitationLoss
    from data_parsing.pre_extracted import make_pre_extracted_loader

    _dirs = [shards.download()]
    if shards2 is not None:
        _dirs.append(shards2.download())
    # Merge tars from all dataset dirs into one (image's loader expects a single dir)
    import glob as _glob, shutil as _shutil
    merged_dir = "/tmp/merged_shards"
    os.makedirs(merged_dir, exist_ok=True)
    for di, d in enumerate(_dirs):
        for tar in _glob.glob(os.path.join(str(d), "*.tar")):
            _shutil.copy(tar, os.path.join(merged_dir, f"ds{di}_{os.path.basename(tar)}"))
    shard_dirs = _dirs
    ctx = current_context()
    bb, fm = backbone.value, fusion_mode.value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training: backbone={bb} fusion={fm} epochs={epochs} bs={batch_size} device={device} datasets={len(shard_dirs)}")

    # DataLoader
    loader = make_pre_extracted_loader(merged_dir, batch_size=batch_size, num_workers=0)

    # Detect num_views from the data (L2D=7, NVIDIA=8) so the model matches.
    _peek = next(iter(loader))
    num_views = int(_peek["visual_tiles"].shape[1])
    print(f"Detected num_views={num_views}")

    # Model
    model = AutoE2E(
        backbone=bb, num_views=num_views, embed_dim=256,
        fusion_mode=fm, is_pretrained=True,
    ).to(device)

    # Optimizer + Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = TrajectoryImitationLoss(loss_type="smooth_l1")
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)

    # Training loop
    model.train()
    losses_per_epoch = []
    scaler = torch.amp.GradScaler(enabled=amp)

    for epoch in range(epochs):
        epoch_losses = []
        for batch in loader:
            visual = batch["visual_tiles"].to(device)        # (B, 7, 3, H, W)
            ego_hist = batch["egomotion_history"].to(device)  # (B, 256)
            vis_hist = batch["visual_history"].to(device)     # (B, 896)
            target = batch["trajectory_target"].to(device)    # (B, 128)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                pred, _, _ = model(visual, vis_hist, ego_hist)
                loss = loss_fn(pred, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        losses_per_epoch.append(float(avg_loss))
        print(f"  Epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")

    # Save checkpoint
    os.makedirs("/tmp/train", exist_ok=True)
    ckpt_path = "/tmp/train/best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"backbone": bb, "fusion_mode": fm, "embed_dim": 256, "num_views": num_views},
        "epoch": epochs,
    }, ckpt_path)

    # Metadata
    meta = {
        "data": {"dataset": dataset.value, "shard_dirs": [str(d) for d in shard_dirs]},
        "model": {"backbone": bb, "fusion_mode": fm, "embed_dim": 256, "num_views": num_views},
        "training": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "grad_clip": grad_clip, "amp": amp,
            "optimizer": "AdamW", "final_loss": losses_per_epoch[-1] if losses_per_epoch else 0,
            "losses_per_epoch": losses_per_epoch,
        },
        "context": {
            "flyte_execution_id": ctx.execution_id.name if ctx.execution_id else "local",
            "docker_image": TRAINING_IMAGE,
        },
    }
    meta_path = "/tmp/train/metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return TrainOutput(checkpoint=FlyteFile(ckpt_path), metadata=FlyteFile(meta_path))


# ============================================================
# Task: Offline RL
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
    epochs: int = 3,
    tau: float = 0.7,
    beta: float = 3.0,
) -> TrainOutput:
    """Offline RL (IQL) refinement of IL checkpoint."""
    import os, json, torch
    import numpy as np
    from flytekit import current_context

    ckpt_path = pretrained.download()
    shard_dir = shards.download()
    il_meta = json.load(open(il_metadata.download()))
    ctx = current_context()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Offline RL: epochs={epochs} tau={tau} beta={beta}")

    # Load IL model
    from model_components.auto_e2e import AutoE2E
    from data_parsing.pre_extracted import make_pre_extracted_loader

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = AutoE2E(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    loader = make_pre_extracted_loader(shard_dir, batch_size=4, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-3)

    # Simplified IQL-style training
    model.train()
    losses_per_epoch = []
    for epoch in range(epochs):
        epoch_losses = []
        for batch in loader:
            visual = batch["visual_tiles"].to(device)
            ego_hist = batch["egomotion_history"].to(device)
            vis_hist = batch["visual_history"].to(device)
            target = batch["trajectory_target"].to(device)

            optimizer.zero_grad()
            pred, _, _ = model(visual, vis_hist, ego_hist)
            # IQL advantage-weighted regression
            with torch.no_grad():
                baseline_pred, _, _ = model(visual, vis_hist, ego_hist)
            advantage = -(pred - target).pow(2).mean(dim=-1) + (baseline_pred - target).pow(2).mean(dim=-1)
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
        "rl": {"method": "IQL", "epochs": epochs, "tau": tau, "beta": beta,
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
@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="2", mem="8Gi", gpu="1"),
    limits=Resources(gpu="1"),
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
)
def evaluate(
    checkpoint: FlyteFile,
    shards: FlyteDirectory,
    train_metadata: FlyteFile,
    experiment_name: str = "imitation-learning",
) -> EvalMetrics:
    """Evaluate + log everything to MLflow."""
    import os, json, yaml, torch
    import numpy as np, mlflow
    from flytekit import current_context

    from model_components.auto_e2e import AutoE2E
    from data_parsing.pre_extracted import make_pre_extracted_loader
    from evaluation.metrics import integrate_trajectory, gate_check

    ckpt_path = checkpoint.download()
    shard_dir = shards.download()
    meta = json.load(open(train_metadata.download()))
    ctx = current_context()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = AutoE2E(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate
    loader = make_pre_extracted_loader(shard_dir, batch_size=8, num_workers=0, shuffle=0)
    all_ade, all_fde = [], []

    with torch.no_grad():
        for batch in loader:
            visual = batch["visual_tiles"].to(device)
            ego_hist = batch["egomotion_history"].to(device)
            vis_hist = batch["visual_history"].to(device)
            target = batch["trajectory_target"]  # (B, 128) on CPU

            pred, _, _ = model(visual, vis_hist, ego_hist)
            pred = pred.cpu().numpy()  # (B, 128)
            target_np = target.numpy()

            for i in range(pred.shape[0]):
                # Reshape: (64, 2) = [accel_x, curvature]
                pred_signals = pred[i].reshape(64, 2)
                gt_signals = target_np[i].reshape(64, 2)
                # Get initial speed from egomotion history (first signal)
                ego_np = batch["egomotion_history"][i].numpy()
                v0 = float(ego_np[-4])  # last speed value in history

                pred_traj = integrate_trajectory(pred_signals[:, 0], pred_signals[:, 1], v0)
                gt_traj = integrate_trajectory(gt_signals[:, 0], gt_signals[:, 1], v0)

                ade = float(np.mean(np.linalg.norm(pred_traj - gt_traj, axis=1)))
                fde = float(np.linalg.norm(pred_traj[-1] - gt_traj[-1]))
                all_ade.append(ade)
                all_fde.append(fde)

    avg_ade = float(np.mean(all_ade)) if all_ade else 99.0
    avg_fde = float(np.mean(all_fde)) if all_fde else 99.0
    passed = avg_ade < 2.0 and avg_fde < 4.0

    # --- MLflow logging ---
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)

    model_info = meta.get("model", meta.get("base_model", {}).get("il_metadata", {}).get("model", {}))
    bb = model_info.get("backbone", "?")
    fm = model_info.get("fusion_mode", "?")
    training = meta.get("training", meta.get("base_model", {}).get("il_metadata", {}).get("training", {}))
    run_name = f"{bb}-{fm}-e{training.get('epochs','?')}"

    with mlflow.start_run(run_name=run_name):
        # Flatten params
        params = {}
        data = meta.get("data", meta.get("base_model", {}).get("il_metadata", {}).get("data", {}))
        params["data/dataset"] = data.get("dataset", "?")
        params["model/backbone"] = bb
        params["model/fusion_mode"] = fm
        params["train/epochs"] = training.get("epochs", "?")
        params["train/batch_size"] = training.get("batch_size", "?")
        params["train/lr"] = training.get("lr", "?")
        params["train/weight_decay"] = training.get("weight_decay", "?")
        params["train/amp"] = training.get("amp", "?")
        params["train/final_loss"] = training.get("final_loss", "?")

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
        params["ctx/eval_execution_id"] = ctx.execution_id.name if ctx.execution_id else "local"
        params["ctx/eval_docker_image"] = EVAL_IMAGE

        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.set_tags({"pipeline": experiment_name, "backbone": bb, "fusion": fm})

        # Training loss curve
        for i, l in enumerate(training.get("losses_per_epoch", [])):
            mlflow.log_metric("train/loss", l, step=i)

        # Eval metrics
        mlflow.log_metrics({"eval/ade": avg_ade, "eval/fde": avg_fde, "eval/gate_pass": 1.0 if passed else 0.0})

        # Artifacts
        os.makedirs("/tmp/eval-artifacts", exist_ok=True)
        with open("/tmp/eval-artifacts/config.yaml", "w") as f:
            yaml.dump(meta, f)
        mlflow.log_artifact("/tmp/eval-artifacts/config.yaml")
        mlflow.log_artifact(ckpt_path, artifact_path="model")

        # Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        try:
            mlflow.register_model(model_uri, "auto-e2e-driving-policy")
        except Exception as e:
            print(f"Registry: {e}")

    print(f"Eval: ADE={avg_ade:.3f} FDE={avg_fde:.3f} Gate={'PASS' if passed else 'FAIL'}")
    return EvalMetrics(ade=avg_ade, fde=avg_fde, gate_pass=passed)


# ============================================================
# Workflows
# ============================================================
@workflow
def wf_data_ingest(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
    hf_token: str = "",
) -> FlyteDirectory:
    """Download raw dataset from HuggingFace."""
    return data_ingest(dataset=dataset, episodes=episodes, hf_token=hf_token)


@workflow
def wf_data_processing(
    raw_data: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    hz: int = 10,
    image_size: int = 256,
    episodes: int = 3,
) -> FlyteDirectory:
    """Pre-process raw data → WebDataset shards."""
    return data_processing(raw_data=raw_data, dataset=dataset,
                           hz=hz, image_size=image_size, episodes=episodes)


@workflow
def wf_train_il(
    shards: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    fusion_mode: FusionMode = FusionMode.CONCAT,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
) -> EvalMetrics:
    """IL Train → Evaluate (logs to MLflow 'imitation-learning')."""
    out = train_il(shards=shards, dataset=dataset, backbone=backbone,
                   fusion_mode=fusion_mode, epochs=epochs, batch_size=batch_size, lr=lr)
    return evaluate(checkpoint=out.checkpoint, shards=shards,
                    train_metadata=out.metadata, experiment_name="imitation-learning")


@workflow
def wf_train_offline_rl(
    pretrained: FlyteFile,
    shards: FlyteDirectory,
    il_metadata: FlyteFile,
    epochs: int = 3,
    tau: float = 0.7,
    beta: float = 3.0,
) -> EvalMetrics:
    """Offline RL → Evaluate (logs to MLflow 'offline-rl')."""
    out = train_offline_rl(pretrained=pretrained, shards=shards,
                           il_metadata=il_metadata, epochs=epochs, tau=tau, beta=beta)
    return evaluate(checkpoint=out.checkpoint, shards=shards,
                    train_metadata=out.metadata, experiment_name="offline-rl")


@workflow
def wf_full_pipeline(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    fusion_mode: FusionMode = FusionMode.CONCAT,
    epochs_il: int = 3,
    epochs_rl: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    tau: float = 0.7,
    beta: float = 3.0,
    hf_token: str = "",
) -> EvalMetrics:
    """Full: Ingest → Process → IL Train+Eval → RL Train+Eval."""
    raw = data_ingest(dataset=dataset, episodes=episodes, hf_token=hf_token)
    shards = data_processing(raw_data=raw, dataset=dataset, episodes=episodes)
    il_out = train_il(shards=shards, dataset=dataset, backbone=backbone,
                      fusion_mode=fusion_mode, epochs=epochs_il,
                      batch_size=batch_size, lr=lr)
    evaluate(checkpoint=il_out.checkpoint, shards=shards,
             train_metadata=il_out.metadata, experiment_name="imitation-learning")
    rl_out = train_offline_rl(pretrained=il_out.checkpoint, shards=shards,
                              il_metadata=il_out.metadata, epochs=epochs_rl, tau=tau, beta=beta)
    return evaluate(checkpoint=rl_out.checkpoint, shards=shards,
                    train_metadata=rl_out.metadata, experiment_name="offline-rl")


@workflow
def wf_full_pipeline_all_datasets(
    episodes: int = 3,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    fusion_mode: FusionMode = FusionMode.CONCAT,
    epochs_il: int = 3,
    epochs_rl: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    tau: float = 0.7,
    beta: float = 3.0,
    hf_token: str = "",
) -> EvalMetrics:
    """Full pipeline using ALL datasets (L2D + NVIDIA PhysicalAI).

    Both datasets are ingested + processed in parallel into the same BEV-feature
    WebDataset format, then training consumes shards from both.
    """
    # Ingest + process both datasets in parallel
    raw_l2d = data_ingest(dataset=Dataset.L2D, episodes=episodes, hf_token=hf_token)
    shards_l2d = data_processing(raw_data=raw_l2d, dataset=Dataset.L2D, episodes=episodes)

    raw_nv = data_ingest(dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes, hf_token=hf_token)
    shards_nv = data_processing(raw_data=raw_nv, dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)

    # Train on both datasets
    il_out = train_il(shards=shards_l2d, shards2=shards_nv, dataset=Dataset.L2D,
                      backbone=backbone, fusion_mode=fusion_mode,
                      epochs=epochs_il, batch_size=batch_size, lr=lr)
    evaluate(checkpoint=il_out.checkpoint, shards=shards_l2d,
             train_metadata=il_out.metadata, experiment_name="imitation-learning")
    rl_out = train_offline_rl(pretrained=il_out.checkpoint, shards=shards_l2d,
                              il_metadata=il_out.metadata, epochs=epochs_rl, tau=tau, beta=beta)
    return evaluate(checkpoint=rl_out.checkpoint, shards=shards_l2d,
                    train_metadata=rl_out.metadata, experiment_name="offline-rl")
