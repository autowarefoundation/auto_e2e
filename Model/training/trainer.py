"""Ray Train loop of both AutoE2E stages.

Stage 1 trains ``AutoBEV`` on every published BEV dataset; stage 2 trains the
whole ``AutoE2E`` from a stage 1 checkpoint (``Design/autobev_integration.md``).

The loop does not evaluate. Every ``checkpoint_interval_steps`` (about 30
minutes of training) rank 0 reports a checkpoint to Ray, which uploads it to
``storage_uri``; a separate one-GPU task validates the checkpoints as they
appear. When that validation finds the Pittsburgh BEV quality fell, it writes
``FEEDBACK_FILE`` and the next checkpoint step halves the AutoBEV learning
rates. A preempted or failed worker group resumes from the last checkpoint.

Checkpoints carry the model, optimizer and schedule state, the RNG state, and
``metadata.json``: the stage, the commit, the image digest, the dataset ids,
the AutoBEV geometry contract and every constant that shapes the training, so
a run is reproduced from its commit and data without any flag.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import torch

CHECKPOINT_FILE: Final = "checkpoint.pt"
METADATA_FILE: Final = "metadata.json"
FEEDBACK_FILE: Final = "validation_feedback.json"
GRADIENT_CLIP: Final = 1.0
DDP_TIMEOUT_S: Final = 1800
MAX_FAILURES: Final = 3
LOG_EVERY_STEPS: Final = 10


def bf16_autocast():
    """bf16 autocast for network forwards."""
    return torch.autocast("cuda", dtype=torch.bfloat16)


@dataclass(frozen=True)
class TrainRun:
    """What one training run reads and how long it trains; everything else is a constant."""

    stage: int
    run_name: str
    storage_uri: str
    primary_shards: list[str]
    mix_shards: list[str] = field(default_factory=list)
    early_shards: list[str] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)
    # The stage 1 checkpoint file (``.../checkpoint.pt`` or a registered ``<sha>.pt``).
    parent_checkpoint_uri: str = ""
    steps: int = 1000
    learning_rate: float = 2e-4
    per_rank_batch_size: int = 1
    checkpoint_interval_steps: int = 100
    seed: int = 0
    world_size: int = 8

    def __post_init__(self) -> None:
        if self.stage not in (1, 2):
            raise ValueError("stage is 1 (AutoBEV) or 2 (AutoE2E)")
        if (self.stage == 2) != bool(self.parent_checkpoint_uri):
            raise ValueError("stage 2, and only stage 2, starts from a stage 1 checkpoint file")
        if self.stage == 2 and self.early_shards:
            raise ValueError("early shards (SimBEV) join only the first half of stage 1")
        if min(self.steps, self.per_rank_batch_size, self.checkpoint_interval_steps, self.world_size) <= 0:
            raise ValueError("steps, batch size, checkpoint interval and world size must be positive")


def training_constants() -> dict[str, Any]:
    """Every module constant that shapes training, recorded with each checkpoint."""
    from model_components import reactive_e2e
    from model_components.autobev import encoder, geometry, losses
    from navigation import map_matching
    from training import distortion_mix, localization_noise, objective, schedule
    from training import autobev_augment

    modules = (geometry, encoder, losses, reactive_e2e, map_matching, distortion_mix, localization_noise,
               objective, schedule, autobev_augment)
    constants: dict[str, Any] = {}
    for module in modules:
        for name, value in vars(module).items():
            if name.isupper() and isinstance(value, (int, float, str, tuple, list, dict)):
                constants[f"{module.__name__}.{name}"] = value
    return json.loads(json.dumps(constants, default=str))


def _metadata(run: TrainRun, step: int) -> dict[str, Any]:
    from model_components.autobev.geometry import AUTOBEV_CONTRACT

    return {
        "stage": run.stage,
        "step": step,
        "run": asdict(run),
        "code_commit": os.environ.get("AUTO_E2E_GIT_COMMIT", "unknown"),
        "image_digest": os.environ.get("AUTO_E2E_IMAGE_DIGEST", "unknown"),
        "autobev_contract": AUTOBEV_CONTRACT,
        "constants": training_constants(),
    }


def build_model(run: TrainRun) -> torch.nn.Module:
    """Stage 1: AutoBEV from the pretrained DINOv2; stage 2: AutoE2E around the stage 1 AutoBEV."""
    from model_components import AutoE2E
    from model_components.autobev import AutoBEV, check_autobev_contract

    if run.stage == 1:
        return AutoBEV(pretrained_backbone=True)
    from training import storage

    model = AutoE2E(pretrained_backbone=False)
    with tempfile.TemporaryDirectory() as directory:
        local = storage.download(run.parent_checkpoint_uri, Path(directory) / CHECKPOINT_FILE)
        state = torch.load(local, map_location="cpu", weights_only=False)
    check_autobev_contract(state["metadata"].get("autobev_contract"), source=run.parent_checkpoint_uri)
    if state["metadata"]["stage"] != 1:
        raise ValueError(f"{run.parent_checkpoint_uri} is not a stage 1 checkpoint")
    model.Reactive_E2E.AutoBEV.load_state_dict(state["model"])
    return model


def _feedback_backoffs(storage_uri: str, run_name: str) -> int:
    """How many AutoBEV back-offs the validation task has asked for so far."""
    from training import storage

    uri = f"{storage_uri.rstrip('/')}/{run_name}/{FEEDBACK_FILE}"
    return int(json.loads(storage.read_bytes(uri))["autobev_backoffs"]) if storage.exists(uri) else 0


def model_inputs(batch: dict[str, Any], localization) -> dict[str, torch.Tensor]:
    return {
        "images": batch["visual_tiles"],
        "history_images": batch["camera_history_tiles"],
        "projection": batch["camera_projection_matrix"],
        "history_projection": batch["camera_history_projection_matrix"],
        "map_context": batch["map_context"],
        "route_mask": batch["route_mask"],
        "map_valid": batch["map_valid"],
        "route_valid": batch["route_valid"],
        "egomotion_history": batch["egomotion_history"],
        "map_pose": localization.pose,
        "map_pose_covariance": localization.covariance,
        "map_matching": localization.map_matching,
    }


def _stage2_terms(model, objective, batch, generator, schedule, step):
    """One stage 2 forward: localization noise, a distortion per sample, the clean teacher and the losses."""
    from model_components.autobev.geometry import BEV_H, BEV_W, EMBED_DIM
    from training.autobev_augment import restrict_to_visible
    from training.distortion_mix import aligned_teacher_bev, apply_distortions
    from training.localization_noise import simulate_localization

    localization = simulate_localization(batch["egomotion_history"], generator)
    distorted, distortion = apply_distortions(batch, generator)
    distorted = restrict_to_visible(distorted)
    distorted["autobev_dense_supervised"] = distorted["planner_supervised"]
    size = batch["visual_tiles"].shape[0]
    device = batch["visual_tiles"].device
    teacher = torch.zeros(size, EMBED_DIM, BEV_H, BEV_W, device=device)
    teacher_mask = torch.zeros(size, BEV_H, BEV_W, dtype=torch.bool, device=device)
    # The clean teacher pass is only needed when some sample was distorted.
    if bool(distortion.distorted.any()):
        network = model.module
        with torch.no_grad(), bf16_autocast():
            _, clean = network(**model_inputs(batch, localization))
        teacher, teacher_mask = aligned_teacher_bev(clean["planner_bev"], distortion)
        del clean
    with bf16_autocast():
        controls, auxiliary = model(**model_inputs(distorted, localization))
    return objective(
        controls, auxiliary, distorted, auxiliary_weight=schedule.auxiliary_weight(step),
        teacher_bev=teacher, teacher_mask=teacher_mask,
    )


def _stage1_terms(model, objective, batch, generator):
    """One stage 1 forward: zoom, photometric and BEV-space augmentation on every sample."""
    from training.autobev_augment import bev_augment, restrict_to_visible
    from training.distortion_mix import camera_augment, photometric

    every = torch.ones(batch["visual_tiles"].shape[0], dtype=torch.bool)
    batch = restrict_to_visible(bev_augment(photometric(camera_augment(batch, every, generator), every, generator),
                                            generator))
    batch["autobev_dense_supervised"] = every.to(batch["visual_tiles"].device)
    with bf16_autocast():
        outputs = model(batch["visual_tiles"], batch["camera_history_tiles"], batch["camera_projection_matrix"],
                        batch["camera_history_projection_matrix"], batch["autobev_image_projection"])
    return objective(outputs, batch)


def train_loop(config: dict[str, Any]) -> None:
    import ray.train as ray_train
    import torch.distributed as dist
    from ray.train import Checkpoint

    from training.autobev_data import autobev_training_stream, to_device
    from training.objective import AutoBEVObjective, AutoE2EObjective
    from training.schedule import MAX_BACKOFFS, Schedule, optimizer as build_optimizer, stage1_groups, stage2_groups

    run = TrainRun(**config["run"])
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_model(run).to(device)
    groups = stage1_groups(model) if run.stage == 1 else stage2_groups(model)
    optimizer = build_optimizer(groups, run.learning_rate)
    schedule = Schedule(run.steps, run.stage, groups, run.learning_rate)
    clip_sets = [[p for g in groups if g.autobev == autobev for p in g.parameters] for autobev in (True, False)]
    clip_sets = [group for group in clip_sets if group]
    objective = (AutoBEVObjective() if run.stage == 1 else AutoE2EObjective()).to(device)
    generator = torch.Generator().manual_seed(run.seed * 1000 + rank)
    start = 0
    resumed = ray_train.get_checkpoint()
    if resumed is not None:
        with resumed.as_directory() as directory:
            state = torch.load(Path(directory) / CHECKPOINT_FILE, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.autobev_backoff = int(state["autobev_backoff"])
        generator.set_state(state["generators"][rank])
        torch.set_rng_state(state["torch_rng"][rank])
        torch.cuda.set_rng_state(state["cuda_rng"][rank])
        start = int(state["step"])
    # The same starting weights on every rank: new layers are initialized per process.
    for tensor in [*model.parameters(), *model.buffers()]:
        dist.broadcast(tensor.data, 0)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], static_graph=True)
    stream = autobev_training_stream(
        stage=run.stage, primary=run.primary_shards, mix=run.mix_shards, early=run.early_shards,
        early_until_step=run.steps // 2, batch_size=run.per_rank_batch_size, seed=run.seed, rank=rank,
        world=world, start_step=start, workers=max(1, int(config["cpus_per_rank"]) - 2),
        root=Path(tempfile.gettempdir()) / "autobev-shards" / run.run_name / str(rank),
    )
    running: dict[str, float] = {}
    logged = 0
    for step in range(start, run.steps):
        schedule.apply(optimizer, step)
        batch = to_device(next(stream), device)
        # Geometry (augmentation, projections, targets) and the losses stay in
        # float32; only the network forward runs in bf16.
        terms = (_stage1_terms(ddp, objective, batch, generator) if run.stage == 1
                 else _stage2_terms(ddp, objective, batch, generator, schedule, step))
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        # AutoBEV and the planner side clip separately, so AutoBEV's auxiliary
        # gradients (taken even while its rate is zero) do not shrink the planner's.
        for group in clip_sets:
            torch.nn.utils.clip_grad_norm_(group, GRADIENT_CLIP)
        optimizer.step()
        for name, value in terms.items():
            running[name] = running.get(name, 0.0) + float(value.detach())
        logged += 1
        done = step + 1
        checkpointing = done % run.checkpoint_interval_steps == 0 or done == run.steps
        if done % LOG_EVERY_STEPS and not checkpointing:
            continue
        report = {"step": done, **{f"train/{k}": v / logged for k, v in running.items()}}
        running, logged = {}, 0
        if not checkpointing:
            ray_train.report(report)
            continue
        # Rank 0 reads the validation feedback, so every rank applies the same back-off.
        requested = torch.tensor(_feedback_backoffs(run.storage_uri, run.run_name) if rank == 0 else 0,
                                 device=device)
        dist.broadcast(requested, 0)
        schedule.autobev_backoff = max(schedule.autobev_backoff, min(int(requested), MAX_BACKOFFS))
        states = [None] * world
        dist.all_gather_object(states, (generator.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()))
        generators, torch_rng, cuda_rng = (list(column) for column in zip(*states))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = None
            if rank == 0:
                torch.save({
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": done,
                    "autobev_backoff": schedule.autobev_backoff, "generators": generators,
                    "torch_rng": torch_rng, "cuda_rng": cuda_rng,
                    "metadata": _metadata(run, done),
                }, Path(directory) / CHECKPOINT_FILE)
                (Path(directory) / METADATA_FILE).write_text(json.dumps(_metadata(run, done), indent=1))
                checkpoint = Checkpoint.from_directory(directory)
            ray_train.report(report, checkpoint=checkpoint)


def run_training(run: TrainRun) -> None:
    """Run (or resume) the worker group; checkpoints land under ``storage_uri/run_name``."""
    import ray
    from ray import train
    from ray.train.torch import TorchTrainer

    from distributed_training.ray_torch_backend import PreparedCudaTorchConfig

    if not ray.is_initialized():
        ray.init(address="auto")
    # The node's CPUs split evenly across the ranks, mostly for their loaders.
    cpus = int(ray.cluster_resources()["CPU"]) // run.world_size
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config={"run": asdict(run), "cpus_per_rank": cpus},
        torch_config=PreparedCudaTorchConfig(init_method="tcp", timeout_s=DDP_TIMEOUT_S),
        scaling_config=train.ScalingConfig(
            num_workers=run.world_size, use_gpu=True, resources_per_worker={"CPU": cpus, "GPU": 1},
            placement_strategy="PACK",
        ),
        run_config=train.RunConfig(
            name=run.run_name,
            storage_path=run.storage_uri,
            failure_config=train.FailureConfig(max_failures=MAX_FAILURES),
            # Every checkpoint stays: the validation task reads them all.
            checkpoint_config=train.CheckpointConfig(num_to_keep=None),
        ),
    )
    trainer.fit()
