"""Ray Train DDP runner for nuPlan and L2D Reactive stages."""

from __future__ import annotations

import contextlib
import json
import math
import random
import re
import socket
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from distributed_training.reactive_data import (
    RestartingIterator,
    assign_reactive_shards,
    build_reactive_dataset_plan,
    optimizer_steps_per_epoch,
    reactive_assignment_sha256,
    stage_rank_reactive_shards,
)
from training.reactive_multitask import ReactiveTrainingStage


SUPPORTED_WORLD_SIZES = frozenset({2, 4, 8})
SUPPORTED_PRECISIONS = frozenset({"fp32", "bf16"})
_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def validate_reactive_stage_config(config: Mapping[str, Any]) -> None:
    """Validate the JSON-safe contract before starting a Ray cluster."""
    try:
        stage = ReactiveTrainingStage(str(config["stage"]))
    except (KeyError, ValueError) as error:
        raise ValueError("unsupported Reactive distributed stage") from error
    world_size = int(config.get("num_workers", 0))
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"num_workers must be one of {sorted(SUPPORTED_WORLD_SIZES)}"
        )
    if int(config.get("epochs", 0)) <= 0:
        raise ValueError("epochs must be positive")
    if int(config.get("per_rank_batch_size", 0)) != 1:
        raise ValueError(
            "Reactive DDP v1 requires per_rank_batch_size=1"
        )
    if int(config.get("gradient_accumulation_steps", 0)) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(config.get("num_loader_workers", -1)) < 0:
        raise ValueError("num_loader_workers must be non-negative")
    if not 0.0 < float(config.get("val_fraction", 0.0)) < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    if float(config.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(config.get("weight_decay", -1.0)) < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if float(config.get("grad_clip", 0.0)) <= 0.0:
        raise ValueError("grad_clip must be positive")
    precision = str(config.get("precision", ""))
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"precision must be one of {sorted(SUPPORTED_PRECISIONS)}"
        )
    if precision == "bf16" and not bool(config.get("use_gpu", True)):
        raise ValueError("bf16 Reactive DDP requires GPU workers")
    sources = config.get("source_uris")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source_uris must be a non-empty list")
    if len(set(str(item) for item in sources)) != len(sources):
        raise ValueError("source_uris contains duplicates")
    run_name = str(config.get("run_name", ""))
    if _RUN_NAME_PATTERN.fullmatch(run_name) is None:
        raise ValueError("run_name contains unsupported characters")
    storage_path = str(config.get("storage_path", ""))
    if not storage_path.startswith("s3://"):
        raise ValueError("storage_path must be an S3 URI")
    parent_uri = str(config.get("parent_checkpoint_uri") or "")
    if stage is ReactiveTrainingStage.NUPLAN_FULL and parent_uri:
        raise ValueError("Stage A cannot load a parent checkpoint")
    if stage is ReactiveTrainingStage.L2D_CONTINUATION and not parent_uri:
        raise ValueError("Stage B requires the exact Stage A checkpoint")
    override = int(config.get("steps_per_epoch", 0))
    if override < 0:
        raise ValueError("steps_per_epoch cannot be negative")
    bev_weights = config.get("bev_pos_weights")
    if (
        not isinstance(bev_weights, list)
        or len(bev_weights) != 8
        or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in bev_weights
        )
    ):
        raise ValueError(
            "bev_pos_weights must contain eight positive values"
        )


def _base_model(model):
    from torch.nn.parallel import DistributedDataParallel

    return (
        model.module
        if isinstance(model, DistributedDataParallel)
        else model
    )


def _batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def _loader_item(item: Any) -> tuple[Mapping[str, Any], Any, str]:
    if isinstance(item, tuple):
        if len(item) != 3:
            raise ValueError(
                "Reactive loader items must be "
                "(batch, projection, geometry_type)"
            )
        batch, projection, geometry_type = item
        return batch, projection, str(geometry_type)
    return item, None, "pseudo"


def _collective_true(value: bool, device) -> bool:
    import torch
    import torch.distributed as dist

    flag = torch.tensor(
        int(value),
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _parameter_sample(model, *, sample_size: int = 2048):
    import torch

    samples = []
    remaining = sample_size
    for parameter in _base_model(model).parameters():
        if not parameter.requires_grad:
            continue
        flattened = parameter.detach().reshape(-1)
        take = min(remaining, flattened.numel())
        if take:
            samples.append(flattened[:take].float())
            remaining -= take
        if remaining == 0:
            break
    if not samples:
        raise RuntimeError("Reactive model has no trainable parameters")
    return torch.cat(samples)


def _maximum_parameter_delta(model, *, world_size: int) -> float:
    import torch
    import torch.distributed as dist

    local = _parameter_sample(model)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return max(
        float((candidate - gathered[0]).abs().max().item())
        for candidate in gathered
    )


def _download_checkpoint(checkpoint_uri: str, destination: Path) -> None:
    parsed = urlparse(checkpoint_uri)
    if parsed.scheme == "":
        source = Path(checkpoint_uri)
        destination.write_bytes(source.read_bytes())
        return
    if parsed.scheme == "file":
        source = Path(unquote(parsed.path))
        destination.write_bytes(source.read_bytes())
        return
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("checkpoint must be a local path or S3 URI")
    import boto3

    boto3.client("s3").download_file(
        parsed.netloc,
        parsed.path.lstrip("/"),
        str(destination),
    )


def normalize_ray_checkpoint_uri(
    checkpoint_path: str,
    storage_path: str,
) -> str:
    """Restore the S3 scheme Ray omits from filesystem-backed paths."""
    raw_path = checkpoint_path.rstrip("/")
    storage_uri = storage_path.rstrip("/")
    storage = urlparse(storage_uri)
    if storage.scheme != "s3" or not storage.netloc:
        raise ValueError("Reactive Ray checkpoint storage must be an S3 URI")
    storage_without_scheme = (
        f"{storage.netloc}/{storage.path.lstrip('/')}".rstrip("/")
    )
    parsed = urlparse(raw_path)
    if parsed.scheme == "s3":
        checkpoint_uri = raw_path
    elif parsed.scheme == "" and (
        raw_path == storage_without_scheme
        or raw_path.startswith(f"{storage_without_scheme}/")
    ):
        checkpoint_uri = f"s3://{raw_path}"
    else:
        raise ValueError(
            "Ray checkpoint path is outside the configured S3 storage"
        )
    if not (
        checkpoint_uri == storage_uri
        or checkpoint_uri.startswith(f"{storage_uri}/")
    ):
        raise ValueError(
            "Ray checkpoint URI is outside the configured S3 storage"
        )
    return checkpoint_uri


def _seed_epoch(seed: int, rank: int, epoch: int) -> None:
    import torch

    epoch_seed = seed + rank * 10_007 + epoch * 1_000_003
    random.seed(epoch_seed)
    np.random.seed(epoch_seed % (2**32))
    torch.manual_seed(epoch_seed)
    torch.cuda.manual_seed_all(epoch_seed)


def clip_finite_gradients_float64(
    parameters,
    max_norm: float,
):
    """Clip finite gradients without overflowing the global norm."""
    import torch

    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.zeros((), dtype=torch.float64), True
    device = gradients[0].device
    norm_squared = torch.zeros(
        (),
        dtype=torch.float64,
        device=device,
    )
    finite = True
    for gradient in gradients:
        if gradient.device != device:
            raise ValueError("all gradients must be on the same device")
        finite = finite and bool(torch.isfinite(gradient).all().item())
        norm_squared += gradient.detach().to(torch.float64).square().sum()
    gradient_norm = torch.sqrt(norm_squared)
    finite = finite and bool(torch.isfinite(gradient_norm).item())
    if finite:
        scale = torch.clamp(
            torch.as_tensor(
                max_norm,
                dtype=torch.float64,
                device=device,
            )
            / gradient_norm.clamp_min(torch.finfo(torch.float64).tiny),
            max=1.0,
        )
        for gradient in gradients:
            gradient.mul_(scale.to(dtype=gradient.dtype))
    return gradient_norm, finite


def _train_fixed_steps(
    model,
    loader,
    objective,
    optimizer,
    *,
    device,
    optimizer_steps: int,
    gradient_accumulation_steps: int,
    grad_clip: float,
    precision: str,
) -> dict[str, float]:
    import torch
    import torch.distributed as dist

    from training.reactive_stage_runner import (
        resolve_reactive_batch_projection,
    )

    model.train()
    iterator = RestartingIterator(loader)
    totals = {
        "total": 0.0,
        "trajectory": 0.0,
        "bev_segmentation": 0.0,
        "route_reconstruction": 0.0,
    }
    consumed_samples = 0
    micro_steps = optimizer_steps * gradient_accumulation_steps
    for _ in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        for accumulation_index in range(gradient_accumulation_steps):
            raw_batch, fallback_projection, fallback_geometry_type = (
                _loader_item(next(iterator))
            )
            batch = _batch_to_device(raw_batch, device)
            consumed_samples += int(batch["visual_tiles"].shape[0])
            projection, geometry_type = (
                resolve_reactive_batch_projection(
                    batch,
                    fallback_projection,
                    fallback_geometry_type,
                    device=device,
                )
            )
            synchronize = (
                accumulation_index
                == gradient_accumulation_steps - 1
            )
            sync_context = (
                contextlib.nullcontext()
                if synchronize
                else model.no_sync()
            )
            with sync_context:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16",
                ):
                    output = model(
                        batch["visual_tiles"],
                        batch["map_context"],
                        batch["visual_history"],
                        batch["egomotion_history"],
                        route_mask=batch["route_mask"],
                        map_valid=batch["map_valid"],
                        route_valid=batch["route_valid"],
                        projection=projection,
                        geometry_type=geometry_type,
                        mode="train",
                        compute_bev_segmentation=(
                            objective.compute_bev_segmentation
                        ),
                        compute_route_reconstruction=True,
                    )
                    if not isinstance(output, tuple):
                        raise RuntimeError(
                            "Reactive DDP model omitted auxiliary outputs"
                        )
                    predicted_controls, auxiliary = output
                    terms = objective(
                        predicted_controls,
                        auxiliary,
                        batch,
                    )
                    finite_loss = bool(
                        torch.isfinite(terms["total"]).item()
                    )
                    if not _collective_true(finite_loss, device):
                        raise FloatingPointError(
                            "a Reactive DDP rank produced non-finite loss"
                        )
                    scaled_loss = (
                        terms["total"]
                        / gradient_accumulation_steps
                    )
                scaled_loss.backward()
            for name in totals:
                totals[name] += float(terms[name].detach().item())

        trainable = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        gradient_norm, finite_gradient = clip_finite_gradients_float64(
            trainable,
            grad_clip,
        )
        if not _collective_true(finite_gradient, device):
            raise FloatingPointError(
                "a Reactive DDP rank produced non-finite gradients"
            )
        optimizer.step()

    packed = torch.tensor(
        [
            totals["total"],
            totals["trajectory"],
            totals["bev_segmentation"],
            totals["route_reconstruction"],
            float(consumed_samples),
            float(iterator.restarts),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    denominator = dist.get_world_size() * micro_steps
    return {
        "total": float(packed[0].item() / denominator),
        "trajectory": float(packed[1].item() / denominator),
        "bev_segmentation": float(packed[2].item() / denominator),
        "route_reconstruction": float(
            packed[3].item() / denominator
        ),
        "consumed_samples": float(packed[4].item()),
        "loader_restarts": float(packed[5].item()),
        "local_consumed_samples": float(consumed_samples),
        "local_loader_restarts": float(iterator.restarts),
    }


def _evaluate_global_trajectory(model, loader, *, device) -> dict[str, float]:
    import torch
    import torch.distributed as dist

    from training.losses.control_rollout import integrate_controls_torch
    from training.reactive_stage_runner import (
        resolve_reactive_batch_projection,
    )

    base = _base_model(model)
    was_training = base.training
    base.eval()
    ade_sum = 0.0
    fde_sum = 0.0
    sample_count = 0
    try:
        with torch.no_grad():
            for item in loader:
                raw_batch, fallback_projection, fallback_geometry_type = (
                    _loader_item(item)
                )
                batch = _batch_to_device(raw_batch, device)
                projection, geometry_type = (
                    resolve_reactive_batch_projection(
                        batch,
                        fallback_projection,
                        fallback_geometry_type,
                        device=device,
                    )
                )
                controls = base(
                    batch["visual_tiles"],
                    batch["map_context"],
                    batch["visual_history"],
                    batch["egomotion_history"],
                    route_mask=batch["route_mask"],
                    map_valid=batch["map_valid"],
                    route_valid=batch["route_valid"],
                    projection=projection,
                    geometry_type=geometry_type,
                    mode="infer",
                    compute_bev_segmentation=False,
                    compute_route_reconstruction=False,
                )
                if isinstance(controls, tuple):
                    controls = controls[0]
                batch_size = int(controls.shape[0])
                controls_3d = controls.reshape(batch_size, -1, 2)
                finite_controls = torch.isfinite(controls_3d).all(
                    dim=(1, 2)
                )
                safe_controls = torch.where(
                    finite_controls[:, None, None],
                    controls_3d,
                    torch.zeros_like(controls_3d),
                )
                predicted_xy, _, _ = integrate_controls_torch(
                    safe_controls,
                    batch["initial_speed_mps"],
                )
                target_xy = batch["trajectory_xy_m"].to(torch.float32)
                valid = (
                    batch["trajectory_valid"].to(dtype=torch.bool)
                    & finite_controls[:, None]
                    & torch.isfinite(target_xy).all(dim=-1)
                )
                if predicted_xy.shape != target_xy.shape:
                    raise ValueError(
                        "trajectory target shape differs from rollout"
                    )
                complete = valid.all(dim=1)
                if not bool(complete.any()):
                    continue
                errors = torch.linalg.vector_norm(
                    predicted_xy - target_xy,
                    dim=-1,
                )
                ade_sum += float(
                    errors[complete].mean(dim=1).sum().item()
                )
                fde_sum += float(errors[complete, -1].sum().item())
                sample_count += int(complete.sum().item())
    finally:
        base.train(was_training)

    values = torch.tensor(
        [ade_sum, fde_sum, float(sample_count)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    global_count = int(values[2].item())
    if global_count <= 0:
        raise ValueError(
            "Reactive distributed validation has no complete trajectories"
        )
    return {
        "ade_6p4s_m": float(values[0].item() / global_count),
        "fde_6p4s_m": float(values[1].item() / global_count),
        "complete_samples": float(global_count),
    }


def _load_resume_checkpoint(
    checkpoint_directory: str,
    *,
    model,
    optimizer,
    scheduler,
    expected: Mapping[str, Any],
) -> tuple[int, float]:
    import torch

    payload = torch.load(
        Path(checkpoint_directory) / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Reactive DDP resume checkpoint has no config")
    mismatches = {
        name: (config.get(name), value)
        for name, value in expected.items()
        if config.get(name) != value
    }
    if mismatches:
        raise ValueError(
            f"Reactive DDP resume contract differs: {mismatches}"
        )
    _base_model(model).load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    training_state = payload.get("training_state") or {}
    return (
        int(payload["epoch"]) + 1,
        float(training_state.get("best_ade_6p4s_m", float("inf"))),
    )


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """Train one fixed Reactive stage on every Ray worker."""
    import torch
    import torch.distributed as dist
    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import get_device, prepare_model

    from data_parsing.pre_extracted import (
        make_multi_dataset_loader,
        passthrough_nodesplitter,
    )
    from model_components.auto_e2e import AutoE2E
    from training.reactive_multitask import (
        ReactiveMultitaskObjective,
        configure_model_for_stage,
        reactive_model_kwargs,
    )
    from training.reactive_stage_runner import (
        load_stage_a_parent,
        save_reactive_checkpoint,
    )

    validate_reactive_stage_config(config)
    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    if world_size != int(config["num_workers"]):
        raise RuntimeError(
            f"Ray world size {world_size} differs from requested workers"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Reactive distributed training requires CUDA")
    device = get_device()
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    stage = ReactiveTrainingStage(config["stage"])
    plan = build_reactive_dataset_plan(
        list(config["source_uris"]),
        stage=stage,
    )
    assignments = assign_reactive_shards(
        plan.shards,
        world_size=world_size,
    )
    assignment_sha256 = reactive_assignment_sha256(assignments)
    rank_shards = assignments[rank]
    rank_sample_count = sum(
        shard.sample_count for shard in rank_shards
    )
    cache_root = (
        Path(config.get("local_cache_root") or "/tmp/auto-e2e-reactive")
        / config["run_name"]
        / f"rank-{rank:03d}"
    )
    local_directories = stage_rank_reactive_shards(
        rank_shards,
        cache_root=cache_root,
    )

    seed = int(config["training_seed"])
    _seed_epoch(seed, rank, 0)
    constructor_kwargs = reactive_model_kwargs(
        stage,
        num_views=plan.num_views,
    )
    model = AutoE2E(
        backbone=str(config["backbone"]),
        embed_dim=256,
        is_pretrained=bool(config["is_pretrained"]),
        **constructor_kwargs,
    ).to(device)
    lineage: dict[str, str] = {}
    parent_uri = str(config.get("parent_checkpoint_uri") or "")
    if parent_uri:
        parent_path = cache_root / "stage-a-parent.pt"
        parent_path.parent.mkdir(parents=True, exist_ok=True)
        _download_checkpoint(parent_uri, parent_path)
        lineage.update(load_stage_a_parent(model, parent_path))
    configure_model_for_stage(model, stage)
    objective = ReactiveMultitaskObjective(
        stage,
        bev_pos_weight=list(config["bev_pos_weights"]),
        bev_weight=float(config["bev_weight"]),
        route_weight=float(config["route_weight"]),
        corridor_pos_weight=float(config["corridor_pos_weight"]),
    ).to(device)

    model = prepare_model(
        model,
        parallel_strategy="ddp",
        parallel_strategy_kwargs={
            "find_unused_parameters": True,
            "gradient_as_bucket_view": True,
        },
    )
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        threshold=1e-4,
        threshold_mode="abs",
    )

    calculated_steps = optimizer_steps_per_epoch(
        total_samples=plan.total_samples,
        val_fraction=float(config["val_fraction"]),
        world_size=world_size,
        per_rank_batch_size=int(config["per_rank_batch_size"]),
        gradient_accumulation_steps=int(
            config["gradient_accumulation_steps"]
        ),
    )
    optimizer_steps = int(config["steps_per_epoch"]) or calculated_steps
    global_batch = (
        world_size
        * int(config["per_rank_batch_size"])
        * int(config["gradient_accumulation_steps"])
    )
    expected_resume = {
        "dataset_manifest_sha256": plan.dataset_manifest_sha256,
        "distributed_assignment_sha256": assignment_sha256,
        "distributed_global_batch": global_batch,
        "distributed_precision": str(config["precision"]),
        "distributed_world_size": world_size,
        "training_stage": stage.value,
    }
    start_epoch = 1
    best_ade = float("inf")
    restored = train.get_checkpoint()
    if restored is not None:
        with restored.as_directory() as checkpoint_directory:
            start_epoch, best_ade = _load_resume_checkpoint(
                checkpoint_directory,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected=expected_resume,
            )
    if start_epoch > int(config["epochs"]):
        raise ValueError(
            "resume checkpoint already completed the requested epochs"
        )

    hostnames: list[str | None] = [None] * world_size
    dist.all_gather_object(hostnames, socket.gethostname())
    unique_hostnames = sorted({str(host) for host in hostnames})
    if len(unique_hostnames) != world_size:
        raise RuntimeError(
            "one-GPU-per-node invariant failed: "
            f"world_size={world_size} hosts={unique_hostnames}"
        )

    model_config = {
        "backbone": str(config["backbone"]),
        "embed_dim": 256,
        "is_pretrained": False,
        **constructor_kwargs,
        "distributed_assignment_sha256": assignment_sha256,
        "distributed_global_batch": global_batch,
        "distributed_precision": str(config["precision"]),
        "distributed_world_size": world_size,
    }
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        _seed_epoch(seed, rank, epoch)
        train_loader = make_multi_dataset_loader(
            local_directories,
            batch_size=int(config["per_rank_batch_size"]),
            num_workers=int(config["num_loader_workers"]),
            split="train",
            val_fraction=float(config["val_fraction"]),
            shuffle=int(config["shuffle_buffer"]),
            shuffle_seed=seed + epoch,
            pin_memory=True,
            decode_future_frames=False,
            nodesplitter=passthrough_nodesplitter,
        )
        train_metrics = _train_fixed_steps(
            model,
            train_loader,
            objective,
            optimizer,
            device=device,
            optimizer_steps=optimizer_steps,
            gradient_accumulation_steps=int(
                config["gradient_accumulation_steps"]
            ),
            grad_clip=float(config["grad_clip"]),
            precision=str(config["precision"]),
        )
        validation_loader = make_multi_dataset_loader(
            local_directories,
            batch_size=int(config["per_rank_batch_size"]),
            num_workers=min(int(config["num_loader_workers"]), 1),
            split="val",
            val_fraction=float(config["val_fraction"]),
            shuffle=0,
            pin_memory=True,
            max_active_loaders=1,
            decode_future_frames=False,
            nodesplitter=passthrough_nodesplitter,
        )
        validation = _evaluate_global_trajectory(
            model,
            validation_loader,
            device=device,
        )
        scheduler.step(validation["ade_6p4s_m"])
        maximum_delta = _maximum_parameter_delta(
            model,
            world_size=world_size,
        )
        if maximum_delta > 1e-6:
            raise RuntimeError(
                "Reactive DDP replicas diverged: "
                f"maximum_parameter_delta={maximum_delta}"
            )
        rank_evidence: list[dict[str, Any] | None] = [
            None
        ] * world_size
        dist.all_gather_object(
            rank_evidence,
            {
                "assigned_samples": rank_sample_count,
                "consumed_samples": int(
                    train_metrics["local_consumed_samples"]
                ),
                "hostname": socket.gethostname(),
                "loader_restarts": int(
                    train_metrics["local_loader_restarts"]
                ),
                "rank": rank,
                "shards": [shard.identity for shard in rank_shards],
            },
        )
        is_best = validation["ade_6p4s_m"] < best_ade
        best_ade = min(best_ade, validation["ade_6p4s_m"])
        checkpoint_sha256: str | None = None
        with tempfile.TemporaryDirectory() as checkpoint_directory:
            checkpoint = None
            if rank == 0:
                checkpoint_path = (
                    Path(checkpoint_directory) / "checkpoint.pt"
                )
                checkpoint_sha256 = save_reactive_checkpoint(
                    checkpoint_path,
                    _base_model(model),
                    stage=stage,
                    dataset_manifest_sha256=(
                        plan.dataset_manifest_sha256
                    ),
                    epoch=epoch,
                    model_config=model_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=validation,
                    training_state={
                        "assignment_sha256": assignment_sha256,
                        "best_ade_6p4s_m": best_ade,
                        "global_batch": global_batch,
                        "optimizer_steps_per_epoch": optimizer_steps,
                        "rank_evidence": rank_evidence,
                        "world_size": world_size,
                    },
                    lineage=lineage,
                )
                checkpoint = Checkpoint.from_directory(
                    checkpoint_directory
                )
            checkpoint_digest: list[str | None] = [checkpoint_sha256]
            dist.broadcast_object_list(checkpoint_digest, src=0)
            metrics = {
                "checkpoint_sha256": str(checkpoint_digest[0]),
                "elapsed_seconds": time.perf_counter() - started,
                "epoch": epoch,
                "is_best": int(is_best),
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
                "maximum_parameter_delta": maximum_delta,
                "optimizer_steps_per_epoch": optimizer_steps,
                "train_bev_segmentation": train_metrics[
                    "bev_segmentation"
                ],
                "train_loader_restarts": train_metrics[
                    "loader_restarts"
                ],
                "train_route_reconstruction": train_metrics[
                    "route_reconstruction"
                ],
                "train_total": train_metrics["total"],
                "train_trajectory": train_metrics["trajectory"],
                "validation_ade_6p4s_m": validation["ade_6p4s_m"],
                "validation_complete_samples": validation[
                    "complete_samples"
                ],
                "validation_fde_6p4s_m": validation["fde_6p4s_m"],
                "world_size": world_size,
            }
            train.report(metrics, checkpoint=checkpoint)


def run_reactive_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    """Launch the fixed-size Ray Train worker group."""
    validate_reactive_stage_config(config)
    import ray
    from ray import train
    from ray.train.torch import TorchTrainer

    if not ray.is_initialized():
        ray.init(address="auto")
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=dict(config),
        scaling_config=train.ScalingConfig(
            num_workers=int(config["num_workers"]),
            use_gpu=True,
            resources_per_worker={"CPU": 4, "GPU": 1},
            placement_strategy="SPREAD",
        ),
        run_config=train.RunConfig(
            name=str(config["run_name"]),
            storage_path=str(config["storage_path"]),
            failure_config=train.FailureConfig(max_failures=2),
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="validation_ade_6p4s_m",
                checkpoint_score_order="min",
            ),
        ),
    )
    result = trainer.fit()
    if result.checkpoint is None:
        raise RuntimeError("Reactive Ray training returned no checkpoint")
    checkpoint_uri = normalize_ray_checkpoint_uri(
        str(result.checkpoint.path),
        str(config["storage_path"]),
    )
    metrics = dict(result.metrics)
    if int(metrics.get("world_size", 0)) != int(config["num_workers"]):
        raise RuntimeError(
            f"Reactive Ray result has unexpected world size: {metrics}"
        )
    history = []
    metrics_dataframe = getattr(result, "metrics_dataframe", None)
    if metrics_dataframe is not None:
        history = json.loads(
            metrics_dataframe.to_json(
                orient="records",
                double_precision=15,
            )
        )
    return {
        "checkpoint_file_uri": f"{checkpoint_uri}/checkpoint.pt",
        "checkpoint_uri": checkpoint_uri,
        "history": history,
        "metrics": metrics,
        "run_name": str(config["run_name"]),
        "storage_path": str(config["storage_path"]),
    }
