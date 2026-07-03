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
from typing import NamedTuple, List

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


# NOTE: view fusion is no longer selectable. The reactive-refactor (PR #94)
# removed concat/cross_attn and hardcoded BEV fusion inside ReactiveE2E, and
# dropped the `fusion_mode` argument from AutoE2E.__init__. We keep the string
# "bev" only as a metadata label so MLflow runs stay comparable with old runs.
FUSION_LABEL = "bev"

TrainOutput = NamedTuple("TrainOutput", checkpoint=FlyteFile, metadata=FlyteFile)
EvalMetrics = NamedTuple("EvalMetrics", ade=float, fde=float, gate_pass=bool)


def _model_kwargs(config: dict) -> dict:
    """Filter a saved checkpoint `config` down to kwargs the current AutoE2E
    accepts. The reactive refactor (PR #94) removed `fusion_mode`, but old
    checkpoints (and our own metadata) may still carry it, which would make
    `AutoE2E(**config)` raise. Drop any keys the constructor no longer takes.
    """
    import inspect
    from model_components.auto_e2e import AutoE2E
    valid = set(inspect.signature(AutoE2E.__init__).parameters) - {"self"}
    return {k: v for k, v in config.items() if k in valid}


def _select_shard_dir(shards, dataset) -> str:
    """Download all shard FlyteDirectories and return the local path of the one
    whose manifest matches `dataset`.

    All datasets are passed in (each a separately-packed WebDataset), but only
    the selected dataset is used for this run. Multi-dataset training of a single
    model is tracked in issue #77 (requires dynamic-num_views BEV fusion).
    """
    import os
    import json
    target = dataset.value
    fallback = None
    for sh in shards:
        d = sh.download()
        fallback = fallback or d
        mpath = os.path.join(str(d), "manifest.json")
        if os.path.exists(mpath):
            try:
                if json.load(open(mpath)).get("dataset") == target:
                    print(f"Selected shards for dataset={target}: {d}")
                    return d
            except Exception:
                pass
    print(f"WARN: no shards matched dataset={target}; using first ({fallback})")
    return fallback


def _loader_projection(loader, device):
    """Return the loader's per-dataset projection operator on ``device``.

    Geometry is a rig constant exposed on the loader (``.projection`` /
    ``.geometry_type``) by make_pre_extracted_loader, not per batch. Datasets
    without calibration expose ``projection=None`` + ``geometry_type='pseudo'``,
    so we run the explicit pseudo path — never a silent real-geometry claim.
    """
    projection = getattr(loader, "projection", None)
    geometry_type = getattr(loader, "geometry_type", "pseudo")
    if projection is not None:
        projection = projection.to(device)
    return projection, geometry_type


# ============================================================
# Task: Data Ingest (download raw from HuggingFace)
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="2", mem="24Gi", ephemeral_storage="50Gi"),
    secret_requests=[Secret(group="hf-token", key="HF_TOKEN",
                            mount_requirement=Secret.MountType.ENV_VAR)],
)
def data_ingest(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
) -> FlyteDirectory:
    """Download raw dataset from HuggingFace (lerobot for L2D, physical_ai_av for NVIDIA).

    HF token comes from the `hf-token` K8s Secret (injected as env var by Flyte),
    never from a workflow input — so it is not visible in the Flyte/MLflow UI.
    """
    import os
    import shutil
    from huggingface_hub import login
    from flytekit import current_context

    token = ""
    try:
        token = current_context().secrets.get("hf-token", "HF_TOKEN")
    except Exception:
        token = os.environ.get("HF_TOKEN", "")
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
    requests=Resources(cpu="4", mem="16Gi", ephemeral_storage="50Gi"),
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
    import os
    import io
    import json
    import tarfile
    import tempfile
    import numpy as np
    import torch
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

    def _write_jpeg(sample_key, member, frame_tensor):
        """Resize a (3,H,W) tensor to a JPEG and add it to the current shard."""
        f = to_pil(frame_tensor.cpu().clamp(0, 1)
                   if frame_tensor.dtype.is_floating_point else frame_tensor.cpu())
        f = resize(f)
        b = io.BytesIO()
        f.save(b, format="JPEG", quality=90)
        jpg = b.getvalue()
        ti = tarfile.TarInfo(name=f"{sample_key}.{member}")
        ti.size = len(jpg)
        current_tar.addfile(ti, io.BytesIO(jpg))

    for si in idx_iter:
        sample = ds[si]
        visual = sample["visual_tiles"]            # (V, 3, H, W) real cameras
        map_tile = sample.get("map_tile")          # (3, H, W) nav-map, if any
        ego_hist = sample["egomotion_history"]     # (256,)
        traj = sample["trajectory_target"]         # (128,)
        ego_data = np.concatenate([
            ego_hist.numpy() if torch.is_tensor(ego_hist) else np.asarray(ego_hist),
            traj.numpy() if torch.is_tensor(traj) else np.asarray(traj),
        ]).astype(np.float32)

        sample_key = f"s{si:08d}"
        # Real camera views: cam_<i>.jpg (these define V / num_views).
        for cam_i in range(visual.shape[0]):
            _write_jpeg(sample_key, f"cam_{cam_i}.jpg", visual[cam_i])

        # Nav-map view: map.jpg, kept as a DISTINCT key so the loader routes it
        # to the map branch and never counts it as a camera.
        if map_tile is not None:
            _write_jpeg(sample_key, "map.jpg", map_tile)

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
                "episodes": episodes,
                # num_views = real cameras only; the map view is stored under a
                # separate map.jpg key and is NOT counted here (#77).
                "num_views": int(visual.shape[0]) if sample_count else 0,
                "has_map": bool(sample_count) and (map_tile is not None)}

    # Per-dataset camera calibration (rig constant): scaled ego-to-pixel
    # [V, 3, 4] matrices, stored once. NVIDIA/KITScenes provide real geometry;
    # L2D has no published intrinsics so it carries none (pseudo path). The
    # dataset exposes this via a `camera_params` attribute when available.
    cam_params = getattr(ds, "camera_params", None)
    if cam_params is not None and sample_count:
        cp = cam_params.cpu().numpy() if torch.is_tensor(cam_params) else np.asarray(cam_params)
        manifest["camera_params"] = cp.astype(float).tolist()
        manifest["geometry_type"] = getattr(ds, "geometry_type", "pinhole")
    else:
        manifest["geometry_type"] = "pseudo"

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
    shards: List[FlyteDirectory],
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    amp: bool = True,
) -> TrainOutput:
    """Train AutoE2E model on pre-extracted WebDataset shards.

    All datasets' shards are passed in; the one matching `dataset` is selected
    (single-dataset training; multi-dataset tracked in #77).
    """
    import os
    import json
    import torch
    import numpy as np
    from flytekit import current_context

    from model_components.auto_e2e import AutoE2E
    from model_components.losses import TrajectoryImitationLoss
    from data_parsing.pre_extracted import make_pre_extracted_loader

    shard_dir = _select_shard_dir(shards, dataset)
    ctx = current_context()
    bb, fm = backbone.value, FUSION_LABEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training: backbone={bb} fusion={fm} epochs={epochs} bs={batch_size} device={device}")

    # DataLoader
    loader = make_pre_extracted_loader(shard_dir, batch_size=batch_size, num_workers=0)
    projection, geometry_type = _loader_projection(loader, device)
    print(f"Geometry: {geometry_type} (projection={'real' if projection else 'pseudo'})")

    # Detect num_views from the data so the model matches the dataset.
    _peek = next(iter(loader))
    num_views = int(_peek["visual_tiles"].shape[1])
    print(f"Detected num_views={num_views}")

    # Model. fusion_mode is gone (BEV hardcoded inside ReactiveE2E); the model
    # now also owns the map branch, so its forward requires a map_input tensor.
    model = AutoE2E(
        backbone=bb, num_views=num_views, embed_dim=256,
        is_pretrained=True,
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
            visual = batch["visual_tiles"].to(device)        # (B, V, 3, H, W)
            ego_hist = batch["egomotion_history"].to(device)  # (B, 256)
            vis_hist = batch["visual_history"].to(device)     # (B, 896)
            target = batch["trajectory_target"].to(device)    # (B, 128)
            map_input = batch["map_input"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                pred = model(visual, map_input, vis_hist, ego_hist,
                             projection=projection, geometry_type=geometry_type,
                             mode="train", trajectory_target=target)
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
    # `config` must be reconstruction kwargs for AutoE2E(**config); fusion_mode
    # is no longer a constructor arg, so it lives only in metadata below.
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"backbone": bb, "embed_dim": 256, "num_views": num_views},
        "epoch": epochs,
    }, ckpt_path)

    # Metadata
    meta = {
        "data": {"dataset": dataset.value, "shard_dir": str(shard_dir)},
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
    shards: List[FlyteDirectory],
    il_metadata: FlyteFile,
    dataset: Dataset = Dataset.L2D,
    epochs: int = 3,
    tau: float = 0.7,
    beta: float = 3.0,
) -> TrainOutput:
    """Offline RL (IQL) refinement of IL checkpoint."""
    import os
    import json
    import torch
    import numpy as np
    from flytekit import current_context

    ckpt_path = pretrained.download()
    shard_dir = _select_shard_dir(shards, dataset)
    il_meta = json.load(open(il_metadata.download()))
    ctx = current_context()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Offline RL: epochs={epochs} tau={tau} beta={beta}")

    # Load IL model
    from model_components.auto_e2e import AutoE2E
    from data_parsing.pre_extracted import make_pre_extracted_loader

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = AutoE2E(**_model_kwargs(config)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    loader = make_pre_extracted_loader(shard_dir, batch_size=4, num_workers=0)
    projection, geometry_type = _loader_projection(loader, device)
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
            map_input = batch["map_input"].to(device)

            optimizer.zero_grad()
            pred = model(visual, map_input, vis_hist, ego_hist,
                         projection=projection, geometry_type=geometry_type,
                         mode="train", trajectory_target=target)
            # IQL advantage-weighted regression
            with torch.no_grad():
                baseline_pred = model(visual, map_input, vis_hist, ego_hist,
                                      projection=projection, geometry_type=geometry_type,
                                      mode="infer")
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
def _run_evaluation(checkpoint, shards, train_metadata, dataset, experiment_name):
    """Shared open-loop evaluation + MLflow logging logic.

    Called by both evaluate_il_policy and evaluate_rl_policy. Kept as a plain
    module-level function (not a @task) so the two evaluation tasks share one
    implementation while appearing as distinct nodes in the Flyte UI.
    """
    import os
    import json
    import yaml
    import torch
    import numpy as np
    import mlflow
    from flytekit import current_context

    from model_components.auto_e2e import AutoE2E
    from data_parsing.pre_extracted import make_pre_extracted_loader
    from evaluation.metrics import integrate_trajectory

    ckpt_path = checkpoint.download()
    shard_dir = _select_shard_dir(shards, dataset)
    meta = json.load(open(train_metadata.download()))
    ctx = current_context()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = AutoE2E(**_model_kwargs(config)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate
    loader = make_pre_extracted_loader(shard_dir, batch_size=8, num_workers=0, shuffle=0)
    projection, geometry_type = _loader_projection(loader, device)
    all_ade, all_fde = [], []

    with torch.no_grad():
        for batch in loader:
            visual = batch["visual_tiles"].to(device)
            ego_hist = batch["egomotion_history"].to(device)
            vis_hist = batch["visual_history"].to(device)
            target = batch["trajectory_target"]  # (B, 128) on CPU
            map_input = batch["map_input"].to(device)

            pred = model(visual, map_input, vis_hist, ego_hist,
                         projection=projection, geometry_type=geometry_type, mode="infer")
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
        for i, loss_val in enumerate(training.get("losses_per_epoch", [])):
            mlflow.log_metric("train/loss", loss_val, step=i)

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


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="2", mem="8Gi", gpu="1"),
    limits=Resources(gpu="1"),
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
)
def evaluate_il_policy(
    checkpoint: FlyteFile,
    shards: List[FlyteDirectory],
    train_metadata: FlyteFile,
    dataset: Dataset = Dataset.L2D,
) -> EvalMetrics:
    """Open-loop evaluation of the Imitation-Learning policy.

    Logs ADE/FDE, params, artifacts to the MLflow `imitation-learning` experiment
    and registers the checkpoint in the `auto-e2e-driving-policy` model registry.
    """
    return _run_evaluation(checkpoint, shards, train_metadata, dataset, "imitation-learning")


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="2", mem="8Gi", gpu="1"),
    limits=Resources(gpu="1"),
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
)
def evaluate_rl_policy(
    checkpoint: FlyteFile,
    shards: List[FlyteDirectory],
    train_metadata: FlyteFile,
    dataset: Dataset = Dataset.L2D,
) -> EvalMetrics:
    """Open-loop evaluation of the Offline-RL refined policy.

    Logs ADE/FDE, params (incl. rl/*), artifacts to the MLflow `offline-rl`
    experiment and registers the refined checkpoint in the model registry.
    """
    return _run_evaluation(checkpoint, shards, train_metadata, dataset, "offline-rl")



# ============================================================
# Workflows
# ============================================================
@workflow
def wf_data_ingest(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
) -> FlyteDirectory:
    """Download raw dataset from HuggingFace."""
    return data_ingest(dataset=dataset, episodes=episodes)


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
    shards: List[FlyteDirectory],
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
) -> EvalMetrics:
    """IL Train → Evaluate. All datasets' shards passed in; `dataset` selects one."""
    out = train_il(shards=shards, dataset=dataset, backbone=backbone,
                   epochs=epochs, batch_size=batch_size, lr=lr)
    return evaluate_il_policy(checkpoint=out.checkpoint, shards=shards, dataset=dataset,
                              train_metadata=out.metadata)


@workflow
def wf_train_offline_rl(
    pretrained: FlyteFile,
    shards: List[FlyteDirectory],
    il_metadata: FlyteFile,
    dataset: Dataset = Dataset.L2D,
    epochs: int = 3,
    tau: float = 0.7,
    beta: float = 3.0,
) -> EvalMetrics:
    """Offline RL → Evaluate. All datasets' shards passed in; `dataset` selects one."""
    out = train_offline_rl(pretrained=pretrained, shards=shards, dataset=dataset,
                           il_metadata=il_metadata, epochs=epochs, tau=tau, beta=beta)
    return evaluate_rl_policy(checkpoint=out.checkpoint, shards=shards, dataset=dataset,
                              train_metadata=out.metadata)


@workflow
def wf_full_pipeline(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs_il: int = 3,
    epochs_rl: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    tau: float = 0.7,
    beta: float = 3.0,
) -> EvalMetrics:
    """Full: Ingest+Process ALL datasets (separately packed) → IL Train+Eval → RL Train+Eval.

    Every dataset is ingested and processed into its own WebDataset shard dir, and
    all shard dirs are passed to the train/eval tasks. The `dataset` argument selects
    which one is actually used for this run (single-dataset training; multi-dataset
    on one model tracked in #77).
    """
    # Ingest + process every dataset into separate WebDataset shard dirs
    raw_l2d = data_ingest(dataset=Dataset.L2D, episodes=episodes)
    shards_l2d = data_processing(raw_data=raw_l2d, dataset=Dataset.L2D, episodes=episodes)

    raw_nv = data_ingest(dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)
    shards_nv = data_processing(raw_data=raw_nv, dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)

    all_shards = [shards_l2d, shards_nv]

    il_out = train_il(shards=all_shards, dataset=dataset, backbone=backbone,
                      epochs=epochs_il, batch_size=batch_size, lr=lr)
    evaluate_il_policy(checkpoint=il_out.checkpoint, shards=all_shards, dataset=dataset,
                       train_metadata=il_out.metadata)
    rl_out = train_offline_rl(pretrained=il_out.checkpoint, shards=all_shards, dataset=dataset,
                              il_metadata=il_out.metadata, epochs=epochs_rl, tau=tau, beta=beta)
    return evaluate_rl_policy(checkpoint=rl_out.checkpoint, shards=all_shards, dataset=dataset,
                              train_metadata=rl_out.metadata)


@workflow
def wf_ingest_train_eval(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs_il: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
) -> EvalMetrics:
    """Ingest+Process ALL datasets → IL Train → IL Eval (no offline RL).

    Same as wf_full_pipeline but stops after IL evaluation. Useful when you only
    want a supervised checkpoint + open-loop metrics, or when the offline-RL step
    is too memory-hungry to co-run at the current BEV resolution (#77).
    """
    raw_l2d = data_ingest(dataset=Dataset.L2D, episodes=episodes)
    shards_l2d = data_processing(raw_data=raw_l2d, dataset=Dataset.L2D, episodes=episodes)

    raw_nv = data_ingest(dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)
    shards_nv = data_processing(raw_data=raw_nv, dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)

    all_shards = [shards_l2d, shards_nv]

    il_out = train_il(shards=all_shards, dataset=dataset, backbone=backbone,
                      epochs=epochs_il, batch_size=batch_size, lr=lr)
    return evaluate_il_policy(checkpoint=il_out.checkpoint, shards=all_shards, dataset=dataset,
                              train_metadata=il_out.metadata)
