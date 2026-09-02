"""Ray Train DDP runner for nuPlan and L2D Reactive stages."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import re
import socket
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_TAXONOMY_VERSION,
)
from distributed_training.reactive_data import (
    RestartingIterator,
    assign_reactive_shards,
    build_reactive_dataset_plan,
    optimizer_steps_per_epoch,
    reactive_assignment_sha256,
    stage_rank_reactive_shards,
)
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
from training.reactive_multitask import ReactiveTrainingStage


SUPPORTED_WORLD_SIZES = frozenset({2, 4, 8})
SUPPORTED_PRECISIONS = frozenset({"fp32", "bf16"})
BEV_LANE_NEAR_RADIUS_M = 30.0
CAMERA_FEATURE_SCALE_WEIGHT_METRIC_PREFIX = (
    "camera_feature_scale_weight_"
)
PEAK_CUDA_ALLOCATED_BYTES_METRIC_PREFIX = (
    "peak_cuda_allocated_bytes_rank_"
)
PEAK_CUDA_RESERVED_BYTES_METRIC_PREFIX = (
    "peak_cuda_reserved_bytes_rank_"
)
BEV_LANE_RANGE_METRIC_PREFIX = "bev_lane_boundary_"
ROUTE_DESTINATION_LOGIT_RANGE_EPSILON = 1e-6
ROUTE_VALIDATION_METRICS_VERSION = "route_validation_v1"
REACTIVE_STEP_CHECKPOINT_VERSION = "reactive_step_checkpoint_v1"
REACTIVE_DDP_BUCKET_CAP_MB = 16
EPOCH_CHECKPOINT_RETENTION_SCORE_BASE = 1_000_000_000_000.0
P5EN_MINIMUM_REMAINING_RUNTIME = timedelta(hours=22)
_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def expected_reactive_hostname_count(world_size: int) -> int:
    """Return the reviewed Ray worker host count for each DDP topology."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            "world_size must be one of "
            f"{sorted(SUPPORTED_WORLD_SIZES)}, got {world_size}"
        )
    return 2 if world_size == 2 else 1


@dataclass(frozen=True)
class ReactiveResumeState:
    """Validated epoch and optimizer position restored by every rank."""

    epoch: int
    optimizer_step_in_epoch: int
    best_selection_score: float
    best_ade_6p4s_m: float
    epoch_history: list[dict[str, Any]]
    rank_train_states: tuple[Mapping[str, Any], ...] | None = None
    rank_rng_states: tuple[Mapping[str, Any], ...] | None = None


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
    capacity_block_end_utc = str(
        config.get("capacity_block_end_utc") or ""
    )
    if world_size > 2:
        if not capacity_block_end_utc:
            raise ValueError(
                "p5en training requires capacity_block_end_utc"
            )
        try:
            capacity_block_end = datetime.fromisoformat(
                capacity_block_end_utc.replace("Z", "+00:00")
            )
        except ValueError as error:
            raise ValueError(
                "capacity_block_end_utc must be an ISO-8601 timestamp"
            ) from error
        if capacity_block_end.tzinfo is None:
            raise ValueError(
                "capacity_block_end_utc must include a UTC offset"
            )
        minimum_end = (
            datetime.now(timezone.utc)
            + P5EN_MINIMUM_REMAINING_RUNTIME
        )
        if capacity_block_end.astimezone(timezone.utc) < minimum_end:
            raise ValueError(
                "p5en Capacity Block must have at least 22 hours "
                "remaining"
            )
    if int(config.get("worker_cpus", 0)) <= 0:
        raise ValueError("worker_cpus must be positive")
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
    for name in ("trajectory_weight", "bev_weight", "route_weight"):
        try:
            weight = float(config[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{name} must be configured") from error
        if not math.isfinite(weight) or not 0.0 <= weight <= 1_000.0:
            raise ValueError(
                f"{name} must be finite and between zero and 1000"
            )
    try:
        corridor_pos_weight = float(config["corridor_pos_weight"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("corridor_pos_weight must be configured") from error
    if (
        not math.isfinite(corridor_pos_weight)
        or not 1.0 <= corridor_pos_weight <= 1_000.0
    ):
        raise ValueError(
            "corridor_pos_weight must be finite and between one and 1000"
        )
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
    is_pretrained = config.get("is_pretrained")
    allow_random_init = config.get(
        "allow_random_bevformer_init",
        False,
    )
    if not isinstance(is_pretrained, bool) or not isinstance(
        allow_random_init,
        bool,
    ):
        raise ValueError("BEVFormer initialization flags must be boolean")
    if is_pretrained == allow_random_init:
        raise ValueError(
            "exactly one BEVFormer initialization mode must be selected"
        )
    if is_pretrained and not parent_uri:
        pretrained_uri = str(
            config.get("bevformer_pretrained_checkpoint_uri") or ""
        )
        pretrained_sha256 = str(
            config.get("bevformer_pretrained_checkpoint_sha256") or ""
        )
        if urlparse(pretrained_uri).scheme not in {
            "",
            "file",
            "http",
            "https",
            "s3",
        }:
            raise ValueError("unsupported BEVFormer checkpoint URI")
        if not pretrained_uri:
            raise ValueError("BEVFormer checkpoint URI is required")
        if re.fullmatch(r"[0-9a-f]{64}", pretrained_sha256) is None:
            raise ValueError(
                "BEVFormer checkpoint SHA-256 must be 64 lowercase hex chars"
            )
    override = int(config.get("steps_per_epoch", 0))
    if override < 0:
        raise ValueError("steps_per_epoch cannot be negative")
    if int(config.get("checkpoint_interval_steps", 0)) <= 0:
        raise ValueError("checkpoint_interval_steps must be positive")
    freeze_bevformer = config.get("freeze_bevformer")
    if not isinstance(freeze_bevformer, bool):
        raise ValueError("freeze_bevformer must be a boolean")
    if "bev_pos_weights" in config:
        raise ValueError(
            "bev_pos_weights is derived from train statistics and cannot "
            "be configured"
        )
    if float(config.get("bev_pos_weight_cap", 0.0)) < 1.0:
        raise ValueError("bev_pos_weight_cap must be at least one")
    repeat_threshold = float(
        config.get("bev_repeat_frequency_threshold", 0.0)
    )
    if not 0.0 < repeat_threshold <= 1.0:
        raise ValueError(
            "bev_repeat_frequency_threshold must be in (0,1]"
        )
    if int(config.get("bev_max_repeat", 0)) < 1:
        raise ValueError("bev_max_repeat must be positive")
    if int(config.get("bev_min_positive_samples", 0)) < 1:
        raise ValueError("bev_min_positive_samples must be positive")
    if int(config.get("bev_min_positive_cells", 0)) < 1:
        raise ValueError("bev_min_positive_cells must be positive")
    if int(config.get("bev_ap_bins", 0)) < 256:
        raise ValueError("bev_ap_bins must be at least 256")
    if float(config.get("selection_ade_scale_m", 0.0)) <= 0.0:
        raise ValueError("selection_ade_scale_m must be positive")
    if int(config.get("validation_sample_limit", 1024)) < world_size:
        raise ValueError(
            "validation_sample_limit must be at least num_workers"
        )
    if float(
        config.get("selection_ade_regression_margin_m", -1.0)
    ) < 0.0:
        raise ValueError(
            "selection_ade_regression_margin_m must be non-negative"
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


def _collective_true(value, device) -> bool:
    import torch
    import torch.distributed as dist

    flag = torch.as_tensor(value, device=device)
    if flag.numel() != 1:
        raise ValueError("collective boolean flag must be scalar")
    flag = flag.to(dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _all_reduce_bev_statistics(local_statistics, device):
    """Combine exact rank-local BEV counts into one global contract."""
    import torch
    import torch.distributed as dist

    from data_parsing.pre_extracted import BEVTrainingStatistics

    packed = torch.tensor(
        [
            float(local_statistics.sample_count),
            float(local_statistics.effective_exposure_count),
            *local_statistics.positive_sample_count,
            *local_statistics.positive_cell_count,
            *local_statistics.positive_mass,
            *local_statistics.valid_cell_count,
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    class_count = 8
    offset = 2

    def take(count: int):
        nonlocal offset
        values = packed[offset:offset + count]
        offset += count
        return values

    positive_samples = take(class_count)
    positive_cells = take(class_count)
    positive_mass = take(class_count)
    valid_cells = take(class_count)
    rank_digests: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(
        rank_digests,
        local_statistics.exposure_digest,
    )
    exposure_digest = hashlib.sha256(
        "\n".join(
            f"{rank}:{digest}"
            for rank, digest in enumerate(rank_digests)
        ).encode("ascii")
    ).hexdigest()
    return BEVTrainingStatistics(
        sample_count=int(round(float(packed[0].item()))),
        effective_exposure_count=int(round(float(packed[1].item()))),
        positive_sample_count=tuple(
            int(round(float(value)))
            for value in positive_samples.tolist()
        ),
        positive_cell_count=tuple(
            int(round(float(value)))
            for value in positive_cells.tolist()
        ),
        positive_mass=tuple(
            float(value) for value in positive_mass.tolist()
        ),
        valid_cell_count=tuple(
            int(round(float(value)))
            for value in valid_cells.tolist()
        ),
        exposure_digest=exposure_digest,
    )


def _camera_feature_scale_weights(model) -> tuple[float, ...]:
    import torch

    feature_fusion = _base_model(model).Reactive_E2E.FeatureFusion
    if getattr(feature_fusion, "architecture", None) in {
        "bevformer_v2_t1",
        "bevformer_v2_t8",
    }:
        level_count = int(feature_fusion.view_fusion.num_levels)
        if level_count <= 0:
            raise ValueError("BEVFormer feature level count is invalid")
        return (1.0 / level_count,) * level_count
    try:
        scale_logits = feature_fusion.scale_logits
    except AttributeError as error:
        raise ValueError(
            "Reactive model omitted camera feature scale logits"
        ) from error
    if scale_logits.ndim != 1 or scale_logits.numel() == 0:
        raise ValueError("camera feature scale logits have invalid shape")
    weights = scale_logits.detach().float().softmax(dim=0)
    if not bool(torch.isfinite(weights).all()):
        raise FloatingPointError("camera feature scale weights are non-finite")
    return tuple(float(value) for value in weights.cpu().tolist())


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
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        from urllib.request import urlopen

        temporary = destination.with_suffix(destination.suffix + ".download")
        with urlopen(checkpoint_uri, timeout=120) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        temporary.replace(destination)
        return
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(
            "checkpoint must be a local path, HTTP(S), or S3 URI"
        )
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


def _model_state_sha256(model) -> str:
    import torch

    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not torch.is_tensor(value):
            raise TypeError(f"model state {name} is not a tensor")
        tensor = value.detach()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        tensor = tensor.contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(metadata.encode("ascii"))
        byte_view = (
            tensor.reshape(-1)
            .view(torch.uint8)
            .cpu()
            .numpy()
        )
        digest.update(memoryview(byte_view))
    return digest.hexdigest()


def _assert_ddp_model_state_consistent(model) -> str:
    import torch.distributed as dist

    local_digest = _model_state_sha256(model)
    rank_digests: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(rank_digests, local_digest)
    if len(set(rank_digests)) != 1:
        raise RuntimeError(
            "DDP model state differs across ranks: "
            f"{rank_digests}"
        )
    return local_digest


def _capture_rng_state() -> dict[str, Any]:
    import torch

    return {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [
            state.cpu() for state in torch.cuda.get_rng_state_all()
        ],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    required = {"numpy", "python", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Reactive resume RNG state is missing {missing}")
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(list(state["torch_cuda"]))


def _checkpoint_step_due(
    completed_optimizer_steps: int,
    *,
    optimizer_steps: int,
    checkpoint_interval_steps: int,
) -> bool:
    if not 0 < completed_optimizer_steps <= optimizer_steps:
        raise ValueError("completed optimizer steps are out of range")
    if checkpoint_interval_steps <= 0:
        raise ValueError("checkpoint interval must be positive")
    return completed_optimizer_steps % checkpoint_interval_steps == 0


def _validate_checkpoint_interval(
    checkpoint_interval_steps: int,
    optimizer_steps: int,
) -> None:
    if checkpoint_interval_steps <= 0:
        raise ValueError("checkpoint interval must be positive")
    if checkpoint_interval_steps > optimizer_steps:
        raise ValueError(
            "checkpoint interval must not exceed optimizer steps per epoch"
        )


def _rank_resume_value(
    values: tuple[Mapping[str, Any], ...] | None,
    *,
    rank: int,
    world_size: int,
    name: str,
) -> Mapping[str, Any] | None:
    if values is None:
        return None
    if len(values) != world_size:
        raise ValueError(
            f"Reactive resume {name} count differs from world size"
        )
    value = values[rank]
    if int(value.get("rank", -1)) != rank:
        raise ValueError(f"Reactive resume {name} rank order is invalid")
    return value


def _rank_resume_value_for_epoch(
    values: tuple[Mapping[str, Any], ...] | None,
    *,
    current_epoch: int,
    resume_epoch: int,
    start_optimizer_step: int,
    rank: int,
    world_size: int,
    name: str,
) -> Mapping[str, Any] | None:
    if current_epoch != resume_epoch or start_optimizer_step == 0:
        return None
    return _rank_resume_value(
        values,
        rank=rank,
        world_size=world_size,
        name=name,
    )


def _replay_loader_position(
    iterator,
    *,
    skipped_micro_steps: int,
    expected_restarts: int,
    expected_samples: int,
) -> None:
    replayed_samples = 0
    for _ in range(skipped_micro_steps):
        raw_batch, _, _ = _loader_item(next(iterator))
        replayed_samples += int(raw_batch["visual_tiles"].shape[0])
    if (
        iterator.restarts != expected_restarts
        or replayed_samples != expected_samples
    ):
        raise ValueError(
            "Reactive resume loader position is not deterministic"
        )


def _resume_completed_requested_epochs(
    state: ReactiveResumeState,
    requested_epochs: int,
) -> bool:
    return state.epoch > requested_epochs


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
        return (
            torch.zeros((), dtype=torch.float64),
            torch.ones((), dtype=torch.bool),
        )
    device = gradients[0].device
    for gradient in gradients:
        if gradient.device != device:
            raise ValueError("all gradients must be on the same device")
    per_tensor_norms = [
        torch.linalg.vector_norm(
            gradient.detach(),
            ord=2,
            dtype=torch.float64,
        )
        for gradient in gradients
    ]
    gradient_norm = torch.linalg.vector_norm(torch.stack([
        value.detach()
        for value in per_tensor_norms
    ]))
    finite = torch.isfinite(gradient_norm)
    scale = torch.clamp(
        torch.as_tensor(
            max_norm,
            dtype=torch.float64,
            device=device,
        )
        / gradient_norm.clamp_min(torch.finfo(torch.float64).tiny),
        max=1.0,
    )
    scale = torch.where(finite, scale, torch.ones_like(scale))
    for gradient in gradients:
        gradient.mul_(scale.to(dtype=gradient.dtype))
    return gradient_norm, finite


def reactive_gradient_parameter_groups(model) -> dict[str, list[Any]]:
    """Partition every trainable parameter into one clipping group."""
    base = _base_model(model)
    try:
        reactive = base.Reactive_E2E
    except AttributeError as error:
        raise ValueError("Reactive model is missing Reactive_E2E") from error
    grouped_modules = {
        "camera": (
            reactive.Backbone,
            reactive.FeatureFusion,
            reactive.BEVSegmentationHead,
        ),
        "navigation": (
            reactive.NavigationEncoder,
            reactive.MapBEVFusion,
            reactive.RouteReconstructionHead,
        ),
    }
    groups: dict[str, list[Any]] = {
        "camera": [],
        "front_gate": [],
        "navigation": [],
        "planner": [],
    }
    assigned: set[int] = set()
    front_gate = getattr(
        getattr(reactive.FeatureFusion, "view_fusion", None),
        "front_residual_gate",
        None,
    )
    if front_gate is not None and front_gate.requires_grad:
        assigned.add(id(front_gate))
        groups["front_gate"].append(front_gate)
    for group_name, modules in grouped_modules.items():
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    continue
                identity = id(parameter)
                if parameter is front_gate:
                    continue
                if identity in assigned:
                    raise ValueError(
                        "Reactive gradient parameter groups overlap"
                    )
                assigned.add(identity)
                groups[group_name].append(parameter)
    for parameter in base.parameters():
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity not in assigned:
            assigned.add(identity)
            groups["planner"].append(parameter)
    trainable_count = sum(
        1 for parameter in base.parameters() if parameter.requires_grad
    )
    if len(assigned) != trainable_count:
        raise ValueError("Reactive gradient parameter grouping is incomplete")
    return groups


def _build_reactive_scheduler(
    optimizer,
):
    import torch

    return (
        "selection_plateau_v1",
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            threshold=1e-3,
            threshold_mode="rel",
            cooldown=1,
            min_lr=1e-5,
        ),
    )


def _synchronize_t8_temporal_batch_norm(model) -> int:
    """Use global rank statistics for T8 temporal normalization."""
    import torch.nn as nn

    feature_fusion = _base_model(model).Reactive_E2E.FeatureFusion
    if getattr(feature_fusion, "architecture", None) != "bevformer_v2_t8":
        return 0
    temporal_fusion = getattr(feature_fusion, "temporal_fusion", None)
    if temporal_fusion is None:
        raise ValueError("BEVFormer V2 T8 temporal fusion is missing")
    batch_norm_count = sum(
        isinstance(module, nn.BatchNorm2d)
        for module in temporal_fusion.modules()
    )
    if batch_norm_count <= 0:
        raise ValueError("BEVFormer V2 T8 temporal fusion has no BatchNorm")
    feature_fusion.temporal_fusion = (
        nn.SyncBatchNorm.convert_sync_batchnorm(temporal_fusion)
    )
    sync_count = sum(
        isinstance(module, nn.SyncBatchNorm)
        for module in feature_fusion.temporal_fusion.modules()
    )
    if sync_count != batch_norm_count:
        raise RuntimeError("T8 BatchNorm conversion was incomplete")
    return sync_count


def _synchronize_gradient_micro_step(
    optimizer_step_index: int,
    accumulation_index: int,
    gradient_accumulation_steps: int,
) -> bool:
    """Keep static DDP out of no_sync until its reducer is initialized."""
    return (
        optimizer_step_index == 0
        or accumulation_index == gradient_accumulation_steps - 1
    )


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
    start_optimizer_step: int = 0,
    resume_rank_state: Mapping[str, Any] | None = None,
    resume_rng_state: Mapping[str, Any] | None = None,
    checkpoint_interval_steps: int = 0,
    checkpoint_callback: (
        Callable[[int, Mapping[str, Any]], None] | None
    ) = None,
) -> dict[str, float]:
    import torch
    import torch.distributed as dist

    from training.reactive_stage_runner import (
        resolve_reactive_batch_projection,
        resolve_reactive_camera_history,
        resolve_reactive_front_projection,
    )

    if not 0 <= start_optimizer_step <= optimizer_steps:
        raise ValueError("resume optimizer step is out of range")
    if checkpoint_callback is not None and checkpoint_interval_steps <= 0:
        raise ValueError("checkpoint interval must be positive")
    if start_optimizer_step > 0 and (
        resume_rank_state is None or resume_rng_state is None
    ):
        raise ValueError("step resume requires rank and RNG state")
    if start_optimizer_step == 0 and (
        resume_rank_state is not None or resume_rng_state is not None
    ):
        raise ValueError("fresh epoch cannot include step resume state")

    model.train()
    iterator = RestartingIterator(loader)
    term_names = (
        "total",
        "trajectory",
        "bev_segmentation",
        "bev_segmentation_bce",
        "bev_segmentation_dice",
        "route_reconstruction",
    )
    if resume_rank_state is None:
        totals = torch.zeros(
            len(term_names),
            dtype=torch.float64,
            device=device,
        )
    else:
        totals = torch.as_tensor(
            resume_rank_state["term_totals"],
            dtype=torch.float64,
            device=device,
        )
        if totals.shape != (len(term_names),):
            raise ValueError("Reactive resume term totals are invalid")
    gradient_group_names = (
        "camera",
        "front_gate",
        "navigation",
        "planner",
    )
    gradient_groups = reactive_gradient_parameter_groups(model)
    if resume_rank_state is None:
        gradient_totals = torch.zeros(
            len(gradient_group_names) * 2,
            dtype=torch.float64,
            device=device,
        )
    else:
        gradient_totals = torch.as_tensor(
            resume_rank_state["gradient_totals"],
            dtype=torch.float64,
            device=device,
        )
        if gradient_totals.shape != (len(gradient_group_names) * 2,):
            raise ValueError("Reactive resume gradient totals are invalid")
    require_stage_a_camera_context = (
        objective.stage is ReactiveTrainingStage.NUPLAN_FULL
    )
    consumed_samples = (
        0
        if resume_rank_state is None
        else int(resume_rank_state["consumed_samples"])
    )
    skipped_micro_steps = (
        start_optimizer_step * gradient_accumulation_steps
    )
    if skipped_micro_steps:
        assert resume_rank_state is not None
        assert resume_rng_state is not None
        if start_optimizer_step < optimizer_steps:
            _replay_loader_position(
                iterator,
                skipped_micro_steps=skipped_micro_steps,
                expected_restarts=int(
                    resume_rank_state["loader_restarts"]
                ),
                expected_samples=consumed_samples,
            )
        else:
            iterator.restarts = int(
                resume_rank_state["loader_restarts"]
            )
        _restore_rng_state(resume_rng_state)
    micro_steps = optimizer_steps * gradient_accumulation_steps
    rank = dist.get_rank()

    def report_first_step_phase(
        phase: str,
        *,
        step_started: float,
    ) -> None:
        print(
            "Reactive first-step phase "
            f"rank={rank} phase={phase} "
            f"elapsed_seconds={time.perf_counter() - step_started:.3f}",
            flush=True,
        )

    for optimizer_step_index in range(
        start_optimizer_step,
        optimizer_steps,
    ):
        first_optimizer_step = (
            optimizer_step_index == start_optimizer_step
        )
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        finite_step = torch.ones((), dtype=torch.bool, device=device)
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
            front_projection = resolve_reactive_front_projection(
                batch,
                geometry_type,
                device=device,
                required=require_stage_a_camera_context,
            )
            camera_history_tiles, history_projections = (
                resolve_reactive_camera_history(
                    batch,
                    geometry_type,
                    device=device,
                    required=require_stage_a_camera_context,
                )
            )
            synchronize = _synchronize_gradient_micro_step(
                optimizer_step_index - start_optimizer_step,
                accumulation_index,
                gradient_accumulation_steps,
            )
            sync_context = (
                contextlib.nullcontext()
                if synchronize
                else model.no_sync()
            )
            if first_optimizer_step and accumulation_index == 0:
                report_first_step_phase(
                    "forward_start",
                    step_started=step_started,
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
                        camera_history_tiles=camera_history_tiles,
                        history_projections=history_projections,
                        front_camera_tile=batch.get("front_camera_tile"),
                        front_projection=front_projection,
                        mode="train",
                        compute_bev_segmentation=(
                            objective.compute_bev_segmentation
                        ),
                        compute_route_reconstruction=(
                            objective.compute_route_reconstruction
                        ),
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
                    finite_step.logical_and_(
                        torch.isfinite(terms["total"].detach())
                    )
                    scaled_loss = (
                        terms["total"]
                        / gradient_accumulation_steps
                    )
                if first_optimizer_step and accumulation_index == 0:
                    report_first_step_phase(
                        "forward_complete",
                        step_started=step_started,
                    )
                scaled_loss.backward()
                if first_optimizer_step and accumulation_index == 0:
                    report_first_step_phase(
                        "backward_complete",
                        step_started=step_started,
                    )
            totals += torch.stack([
                terms[name].detach().to(torch.float64)
                for name in term_names
            ])

        for group_index, group_name in enumerate(gradient_group_names):
            parameters = gradient_groups[group_name]
            if parameters:
                gradient_norm, finite_gradient = (
                    clip_finite_gradients_float64(
                        parameters,
                        grad_clip,
                    )
                )
                clip_scale = torch.clamp(
                    torch.as_tensor(
                        grad_clip,
                        dtype=torch.float64,
                        device=device,
                    )
                    / gradient_norm.clamp_min(
                        torch.finfo(torch.float64).tiny
                    ),
                    max=1.0,
                )
                clip_scale = torch.where(
                    finite_gradient,
                    clip_scale,
                    torch.ones_like(clip_scale),
                )
            else:
                gradient_norm = torch.zeros(
                    (),
                    dtype=torch.float64,
                    device=device,
                )
                finite_gradient = torch.ones(
                    (),
                    dtype=torch.bool,
                    device=device,
                )
                clip_scale = torch.ones(
                    (),
                    dtype=torch.float64,
                    device=device,
                )
            finite_step.logical_and_(finite_gradient)
            gradient_totals[group_index * 2] += gradient_norm
            gradient_totals[group_index * 2 + 1] += clip_scale
        if first_optimizer_step:
            report_first_step_phase(
                "gradient_check_complete",
                step_started=step_started,
            )
        if not _collective_true(finite_step, device):
            raise FloatingPointError(
                "a Reactive DDP rank produced non-finite loss or gradients"
            )
        if first_optimizer_step:
            report_first_step_phase(
                "finite_collective_complete",
                step_started=step_started,
            )
        optimizer.step()
        if first_optimizer_step:
            report_first_step_phase(
                "optimizer_complete",
                step_started=step_started,
            )
        completed_optimizer_steps = optimizer_step_index + 1
        if (
            checkpoint_callback is not None
            and _checkpoint_step_due(
                completed_optimizer_steps,
                optimizer_steps=optimizer_steps,
                checkpoint_interval_steps=checkpoint_interval_steps,
            )
        ):
            checkpoint_callback(
                completed_optimizer_steps,
                {
                    "consumed_samples": consumed_samples,
                    "gradient_totals": (
                        gradient_totals.detach().cpu().tolist()
                    ),
                    "loader_restarts": iterator.restarts,
                    "rank": dist.get_rank(),
                    "term_totals": totals.detach().cpu().tolist(),
                },
            )

    packed = torch.cat([
        totals,
        gradient_totals,
        torch.tensor(
            [float(consumed_samples), float(iterator.restarts)],
            dtype=torch.float64,
            device=device,
        ),
    ])
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    denominator = dist.get_world_size() * micro_steps
    gradient_denominator = dist.get_world_size() * optimizer_steps
    consumed_offset = len(term_names) + len(gradient_group_names) * 2
    metrics = {
        "total": float(packed[0].item() / denominator),
        "trajectory": float(packed[1].item() / denominator),
        "bev_segmentation": float(packed[2].item() / denominator),
        "bev_segmentation_bce": float(
            packed[3].item() / denominator
        ),
        "bev_segmentation_dice": float(
            packed[4].item() / denominator
        ),
        "route_reconstruction": float(
            packed[5].item() / denominator
        ),
        "consumed_samples": float(packed[consumed_offset].item()),
        "loader_restarts": float(packed[consumed_offset + 1].item()),
        "local_consumed_samples": float(consumed_samples),
        "local_loader_restarts": float(iterator.restarts),
    }
    for group_index, group_name in enumerate(gradient_group_names):
        metrics[f"gradient_{group_name}_pre_clip_norm"] = float(
            packed[len(term_names) + group_index * 2].item()
            / gradient_denominator
        )
        metrics[f"gradient_{group_name}_clip_scale"] = float(
            packed[len(term_names) + group_index * 2 + 1].item()
            / gradient_denominator
        )
    return metrics


def _histogram_average_precision(
    positive_histogram,
    negative_histogram,
) -> float:
    import torch

    positive_total = positive_histogram.sum()
    if float(positive_total.item()) <= 0.0:
        raise ValueError("BEV validation class has no positive cells")
    cumulative_positive = torch.cumsum(
        positive_histogram.flip(0),
        dim=0,
    )
    cumulative_negative = torch.cumsum(
        negative_histogram.flip(0),
        dim=0,
    )
    precision = cumulative_positive / (
        cumulative_positive + cumulative_negative
    ).clamp_min(1.0)
    recall = cumulative_positive / positive_total
    recall_delta = torch.diff(
        torch.cat([recall.new_zeros(1), recall])
    )
    return float((recall_delta * precision).sum().item())


def _bev_lane_range_masks(
    height: int,
    width: int,
    *,
    device,
):
    import torch

    if height <= 0 or width <= 0:
        raise ValueError("BEV lane range mask dimensions must be positive")
    geometry = AUTOE2E_NAVIGATION_GEOMETRY
    rows = (
        torch.arange(height, device=device, dtype=torch.float64) + 0.5
    ) / height
    cols = (
        torch.arange(width, device=device, dtype=torch.float64) + 0.5
    ) / width
    x_forward = geometry.x_max_m - rows * (
        geometry.x_max_m - geometry.x_min_m
    )
    y_left = geometry.y_max_m - cols * (
        geometry.y_max_m - geometry.y_min_m
    )
    distance_squared = (
        x_forward[:, None].square() + y_left[None, :].square()
    )
    near = distance_squared <= BEV_LANE_NEAR_RADIUS_M**2
    return torch.stack((near, ~near))


def _metric_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _route_validation_statistics(
    route_logits,
    route_target,
    route_channel_valid,
    route_valid,
    route_loss,
):
    import torch

    statistics = torch.zeros(
        20,
        dtype=torch.float64,
        device=route_logits.device,
    )
    if route_logits.ndim != 4 or route_logits.shape[1] != 2:
        statistics[14] = 1.0
        return statistics
    if route_target.shape != route_logits.shape:
        statistics[15] = 1.0
        return statistics
    if route_channel_valid.shape != route_logits.shape[:2]:
        statistics[16] = 1.0
        return statistics
    if route_valid.shape != route_logits.shape[:1]:
        statistics[17] = 1.0
        return statistics

    logits = route_logits.to(dtype=torch.float32)
    target = route_target.to(
        device=route_logits.device,
        dtype=torch.float32,
    )
    channel_valid = route_channel_valid.to(
        device=route_logits.device,
        dtype=torch.bool,
    )
    sample_valid = route_valid.to(
        device=route_logits.device,
        dtype=torch.bool,
    )
    active_samples = channel_valid.any(dim=1)
    if not torch.equal(active_samples, sample_valid):
        statistics[18] = 1.0
        return statistics
    active_values = channel_valid[:, :, None, None]
    nonfinite_samples = (
        (
            (~torch.isfinite(logits) | ~torch.isfinite(target))
            & active_values
        )
        .any(dim=(1, 2, 3))
    )
    statistics[19] = nonfinite_samples.sum()
    channel_valid = channel_valid & ~nonfinite_samples[:, None]
    active_samples = channel_valid.any(dim=1)

    zero = logits.new_zeros(())
    route_loss_sum = zero
    corridor_bce_sum = zero
    corridor_dice_sum = zero
    destination_focal_sum = zero
    active_count = active_samples.sum()
    components = route_loss.components(logits, target, channel_valid)
    if bool(active_samples.any()):
        route_loss_sum = components["total"] * active_count

    corridor_valid = channel_valid[:, 0]
    corridor_count = corridor_valid.sum()
    true_positive = zero
    false_positive = zero
    false_negative = zero
    if bool(corridor_valid.any()):
        corridor_bce_sum = components["corridor_bce"] * corridor_count
        corridor_dice_sum = components["corridor_dice"] * corridor_count
        corridor_prediction = logits[corridor_valid, 0] >= 0.0
        corridor_target = target[corridor_valid, 0] >= 0.5
        true_positive = (corridor_prediction & corridor_target).sum()
        false_positive = (corridor_prediction & ~corridor_target).sum()
        false_negative = (~corridor_prediction & corridor_target).sum()

    destination_valid = channel_valid[:, 1]
    destination_count = destination_valid.sum()
    destination_error_sum = zero
    destination_localized_count = zero
    destination_logit_range_sum = zero
    destination_ambiguous_count = zero
    if bool(destination_valid.any()):
        destination_focal_sum = (
            components["destination_focal"] * destination_count
        )
        destination_logits = logits[destination_valid, 1]
        destination_target = target[destination_valid, 1]
        _, _, width = destination_logits.shape
        flattened_logits = destination_logits.flatten(1)
        logit_range = (
            flattened_logits.max(dim=1).values
            - flattened_logits.min(dim=1).values
        )
        localized = (
            logit_range > ROUTE_DESTINATION_LOGIT_RANGE_EPSILON
        )
        destination_localized_count = localized.sum()
        destination_logit_range_sum = logit_range.sum()
        destination_ambiguous_count = (~localized).sum()
        if bool(localized.any()):
            predicted_index = flattened_logits[localized].argmax(dim=1)
            target_index = destination_target[
                localized
            ].flatten(1).argmax(dim=1)
            predicted_cells = torch.stack(
                (
                    torch.div(
                        predicted_index,
                        width,
                        rounding_mode="floor",
                    ),
                    predicted_index.remainder(width),
                ),
                dim=1,
            ).to(torch.float64)
            target_cells = torch.stack(
                (
                    torch.div(
                        target_index,
                        width,
                        rounding_mode="floor",
                    ),
                    target_index.remainder(width),
                ),
                dim=1,
            ).to(torch.float64)
            destination_error_sum = torch.linalg.vector_norm(
                predicted_cells - target_cells,
                dim=1,
            ).sum()

    statistics[:14] = torch.stack([
        route_loss_sum,
        active_count,
        corridor_bce_sum,
        corridor_dice_sum,
        true_positive,
        false_positive,
        false_negative,
        corridor_count,
        destination_focal_sum,
        destination_error_sum,
        destination_count,
        destination_localized_count,
        destination_logit_range_sum,
        destination_ambiguous_count,
    ]).to(dtype=torch.float64)
    if not torch.isfinite(statistics[:14]).all():
        failed_count = active_samples.sum()
        statistics[:14].zero_()
        statistics[19] += failed_count
    return statistics


def _evaluate_global_reactive(
    model,
    loader,
    objective,
    *,
    stage: ReactiveTrainingStage,
    device,
    probability_bins: int,
    ade_scale_m: float,
) -> dict[str, float]:
    import torch
    import torch.distributed as dist

    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_CLASSES,
    )
    from training.losses.control_rollout import integrate_controls_torch
    from training.reactive_stage_runner import (
        resolve_reactive_batch_projection,
        resolve_reactive_camera_history,
        resolve_reactive_front_projection,
    )

    base = _base_model(model)
    was_training = base.training
    base.eval()
    ade_sum = 0.0
    fde_sum = 0.0
    sample_count = 0
    class_count = len(BEV_SEGMENTATION_CLASSES)
    bev_counts = torch.zeros(
        (class_count, 5),
        dtype=torch.float64,
        device=device,
    )
    positive_histogram = torch.zeros(
        (class_count, probability_bins),
        dtype=torch.float64,
        device=device,
    )
    negative_histogram = torch.zeros_like(positive_histogram)
    lane_range_counts = torch.zeros(
        (2, 5),
        dtype=torch.float64,
        device=device,
    )
    lane_range_positive_histogram = torch.zeros(
        (2, probability_bins),
        dtype=torch.float64,
        device=device,
    )
    lane_range_negative_histogram = torch.zeros_like(
        lane_range_positive_histogram
    )
    lane_range_masks = None
    lane_class_index = BEV_SEGMENTATION_CLASSES.index("lane_boundary")
    bev_loss_sum = 0.0
    bev_bce_sum = 0.0
    bev_dice_sum = 0.0
    bev_loss_batches = 0
    route_values = torch.zeros(20, dtype=torch.float64, device=device)
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
                front_projection = resolve_reactive_front_projection(
                    batch,
                    geometry_type,
                    device=device,
                    required=(
                        stage is ReactiveTrainingStage.NUPLAN_FULL
                    ),
                )
                camera_history_tiles, history_projections = (
                    resolve_reactive_camera_history(
                        batch,
                        geometry_type,
                        device=device,
                        required=(
                            stage is ReactiveTrainingStage.NUPLAN_FULL
                        ),
                    )
                )
                output = base(
                    batch["visual_tiles"],
                    batch["map_context"],
                    batch["visual_history"],
                    batch["egomotion_history"],
                    route_mask=batch["route_mask"],
                    map_valid=batch["map_valid"],
                    route_valid=batch["route_valid"],
                    projection=projection,
                    geometry_type=geometry_type,
                    camera_history_tiles=camera_history_tiles,
                    history_projections=history_projections,
                    front_camera_tile=batch.get("front_camera_tile"),
                    front_projection=front_projection,
                    mode="infer",
                    return_auxiliary=(
                        stage is ReactiveTrainingStage.NUPLAN_FULL
                        or objective.compute_route_reconstruction
                    ),
                    compute_bev_segmentation=(
                        stage is ReactiveTrainingStage.NUPLAN_FULL
                    ),
                    compute_route_reconstruction=(
                        objective.compute_route_reconstruction
                    ),
                )
                if isinstance(output, tuple):
                    controls, auxiliary = output
                else:
                    controls = output
                    auxiliary = {}
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
                if objective.compute_route_reconstruction:
                    route_logits = auxiliary.get(
                        "route_reconstruction_logits"
                    )
                    if not torch.is_tensor(route_logits):
                        raise RuntimeError(
                            "Reactive validation omitted route logits"
                        )
                    # Route supervision is independent of trajectory completeness.
                    route_values += _route_validation_statistics(
                        route_logits,
                        batch["route_mask"],
                        batch["route_channel_valid"],
                        batch["route_valid"],
                        objective.route_loss,
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

                if stage is ReactiveTrainingStage.NUPLAN_FULL:
                    bev_logits = auxiliary.get(
                        "bev_segmentation_logits"
                    )
                    if not torch.is_tensor(bev_logits):
                        raise RuntimeError(
                            "Stage A validation omitted BEV logits"
                        )
                    target = batch["bev_segmentation_target"].to(
                        device=device,
                        dtype=torch.float32,
                    )
                    valid_mask = batch[
                        "bev_segmentation_valid"
                    ].to(device=device, dtype=torch.bool)
                    if (
                        target.shape != bev_logits.shape
                        or valid_mask.shape != bev_logits.shape
                    ):
                        raise ValueError(
                            "BEV validation target shape differs"
                        )
                    bev_components = objective.bev_loss.components(
                        bev_logits,
                        target,
                        valid_mask,
                    )
                    bev_loss_sum += float(
                        bev_components["total"].item()
                    )
                    bev_bce_sum += float(
                        bev_components["bce"].item()
                    )
                    bev_dice_sum += float(
                        bev_components["dice"].item()
                    )
                    bev_loss_batches += 1
                    probability = bev_logits.float().sigmoid()
                    binary_target = target >= 0.5
                    binary_prediction = probability >= 0.5
                    for class_index in range(class_count):
                        class_valid = valid_mask[:, class_index]
                        if not bool(class_valid.any()):
                            continue
                        class_target = binary_target[
                            :, class_index
                        ][class_valid]
                        class_prediction = binary_prediction[
                            :, class_index
                        ][class_valid]
                        class_probability = probability[
                            :, class_index
                        ][class_valid]
                        bev_counts[class_index, 0] += (
                            class_prediction & class_target
                        ).sum()
                        bev_counts[class_index, 1] += (
                            class_prediction & ~class_target
                        ).sum()
                        bev_counts[class_index, 2] += (
                            ~class_prediction & class_target
                        ).sum()
                        bev_counts[class_index, 3] += class_target.sum()
                        bev_counts[class_index, 4] += class_valid.sum()
                        bins = torch.clamp(
                            (
                                class_probability * probability_bins
                            ).to(torch.int64),
                            max=probability_bins - 1,
                        )
                        positive_histogram[class_index] += (
                            torch.bincount(
                                bins[class_target],
                                minlength=probability_bins,
                            )
                        )
                        negative_histogram[class_index] += (
                            torch.bincount(
                                bins[~class_target],
                                minlength=probability_bins,
                            )
                        )
                    if lane_range_masks is None:
                        lane_range_masks = _bev_lane_range_masks(
                            target.shape[-2],
                            target.shape[-1],
                            device=device,
                        )
                    elif lane_range_masks.shape[-2:] != target.shape[-2:]:
                        raise ValueError(
                            "BEV validation lane range shape changed"
                        )
                    for range_index in range(2):
                        lane_valid = (
                            valid_mask[:, lane_class_index]
                            & lane_range_masks[range_index][None]
                        )
                        if not bool(lane_valid.any()):
                            continue
                        lane_target = binary_target[
                            :, lane_class_index
                        ][lane_valid]
                        lane_prediction = binary_prediction[
                            :, lane_class_index
                        ][lane_valid]
                        lane_probability = probability[
                            :, lane_class_index
                        ][lane_valid]
                        lane_range_counts[range_index, 0] += (
                            lane_prediction & lane_target
                        ).sum()
                        lane_range_counts[range_index, 1] += (
                            lane_prediction & ~lane_target
                        ).sum()
                        lane_range_counts[range_index, 2] += (
                            ~lane_prediction & lane_target
                        ).sum()
                        lane_range_counts[range_index, 3] += (
                            lane_target.sum()
                        )
                        lane_range_counts[range_index, 4] += (
                            lane_valid.sum()
                        )
                        bins = torch.clamp(
                            (
                                lane_probability * probability_bins
                            ).to(torch.int64),
                            max=probability_bins - 1,
                        )
                        lane_range_positive_histogram[range_index] += (
                            torch.bincount(
                                bins[lane_target],
                                minlength=probability_bins,
                            )
                        )
                        lane_range_negative_histogram[range_index] += (
                            torch.bincount(
                                bins[~lane_target],
                                minlength=probability_bins,
                            )
                        )
    finally:
        base.train(was_training)

    values = torch.tensor(
        [ade_sum, fde_sum, float(sample_count)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    dist.all_reduce(bev_counts, op=dist.ReduceOp.SUM)
    dist.all_reduce(positive_histogram, op=dist.ReduceOp.SUM)
    dist.all_reduce(negative_histogram, op=dist.ReduceOp.SUM)
    dist.all_reduce(lane_range_counts, op=dist.ReduceOp.SUM)
    dist.all_reduce(
        lane_range_positive_histogram,
        op=dist.ReduceOp.SUM,
    )
    dist.all_reduce(
        lane_range_negative_histogram,
        op=dist.ReduceOp.SUM,
    )
    dist.all_reduce(route_values, op=dist.ReduceOp.SUM)
    bev_loss_values = torch.tensor(
        [
            bev_loss_sum,
            bev_bce_sum,
            bev_dice_sum,
            float(bev_loss_batches),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(bev_loss_values, op=dist.ReduceOp.SUM)
    route_contract_names = (
        "logit_shape",
        "target_shape",
        "channel_valid_shape",
        "sample_valid_shape",
        "sample_channel_validity_drift",
    )
    route_contract_violations = {
        name: int(route_values[14 + index].item())
        for index, name in enumerate(route_contract_names)
        if float(route_values[14 + index].item()) > 0.0
    }
    if route_contract_violations:
        raise ValueError(
            "Reactive distributed validation has invalid route contracts: "
            f"{route_contract_violations}"
        )
    global_count = int(values[2].item())
    if global_count <= 0:
        raise ValueError(
            "Reactive distributed validation has no complete trajectories"
        )
    metrics = {
        "ade_6p4s_m": float(values[0].item() / global_count),
        "fde_6p4s_m": float(values[1].item() / global_count),
        "complete_samples": float(global_count),
    }
    trajectory_quality = math.exp(
        -metrics["ade_6p4s_m"] / ade_scale_m
    )
    metrics["trajectory_quality"] = trajectory_quality
    (
        route_loss_sum,
        route_supported,
        corridor_bce_sum,
        corridor_dice_sum,
        corridor_true_positive,
        corridor_false_positive,
        corridor_false_negative,
        corridor_supported,
        destination_focal_sum,
        destination_error_sum,
        destination_supported,
        destination_localized,
        destination_logit_range_sum,
        destination_ambiguous,
    ) = (float(value) for value in route_values[:14].tolist())
    route_nonfinite_samples = float(route_values[19].item())
    metrics.update({
        "route_loss": _metric_ratio(
            route_loss_sum,
            route_supported,
        ),
        "route_supported_samples": route_supported,
        "route_corridor_bce": _metric_ratio(
            corridor_bce_sum,
            corridor_supported,
        ),
        "route_corridor_dice": _metric_ratio(
            corridor_dice_sum,
            corridor_supported,
        ),
        "route_corridor_iou": _metric_ratio(
            corridor_true_positive,
            corridor_true_positive
            + corridor_false_positive
            + corridor_false_negative,
        ),
        "route_corridor_precision": _metric_ratio(
            corridor_true_positive,
            corridor_true_positive + corridor_false_positive,
        ),
        "route_corridor_recall": _metric_ratio(
            corridor_true_positive,
            corridor_true_positive + corridor_false_negative,
        ),
        "route_corridor_positive_cells": (
            corridor_true_positive + corridor_false_negative
        ),
        "route_corridor_supported_samples": corridor_supported,
        "route_destination_focal": _metric_ratio(
            destination_focal_sum,
            destination_supported,
        ),
        "route_destination_error_cells": _metric_ratio(
            destination_error_sum,
            destination_localized,
        ),
        "route_destination_logit_range": _metric_ratio(
            destination_logit_range_sum,
            destination_supported,
        ),
        "route_destination_supported_samples": destination_supported,
        "route_destination_localized_samples": destination_localized,
        "route_destination_ambiguous_samples": destination_ambiguous,
        "route_nonfinite_samples": route_nonfinite_samples,
    })
    if stage is not ReactiveTrainingStage.NUPLAN_FULL:
        metrics["selection_score"] = trajectory_quality
        return metrics

    if float(bev_loss_values[3].item()) <= 0.0:
        raise ValueError("Stage A validation has no BEV batches")
    metrics["bev_loss"] = float(
        bev_loss_values[0].item() / bev_loss_values[3].item()
    )
    metrics["bev_bce"] = float(
        bev_loss_values[1].item() / bev_loss_values[3].item()
    )
    metrics["bev_dice"] = float(
        bev_loss_values[2].item() / bev_loss_values[3].item()
    )
    average_precisions = [0.0] * class_count
    ap_lifts = [0.0] * class_count
    class_support = [False] * class_count
    for class_index, class_name in enumerate(BEV_SEGMENTATION_CLASSES):
        true_positive, false_positive, false_negative, positive, valid = (
            float(value)
            for value in bev_counts[class_index].tolist()
        )
        supported = positive > 0.0 and valid > 0.0
        prevalence = _metric_ratio(positive, valid)
        average_precision = 0.0
        ap_lift = 0.0
        if supported:
            average_precision = _histogram_average_precision(
                positive_histogram[class_index],
                negative_histogram[class_index],
            )
            if prevalence < 1.0:
                ap_lift = max(
                    0.0,
                    min(
                        1.0,
                        (average_precision - prevalence)
                        / (1.0 - prevalence),
                    ),
                )
        class_support[class_index] = supported
        ap_lifts[class_index] = ap_lift
        average_precisions[class_index] = average_precision
        prefix = f"bev_{class_name}"
        metrics[f"{prefix}_iou"] = _metric_ratio(
            true_positive,
            true_positive + false_positive + false_negative,
        )
        metrics[f"{prefix}_precision"] = _metric_ratio(
            true_positive,
            true_positive + false_positive,
        )
        metrics[f"{prefix}_recall"] = _metric_ratio(
            true_positive,
            true_positive + false_negative,
        )
        metrics[f"{prefix}_average_precision"] = average_precision
        metrics[f"{prefix}_ap_lift"] = ap_lift
        metrics[f"{prefix}_positive_prevalence"] = prevalence
        metrics[f"{prefix}_positive_cells"] = positive
        metrics[f"{prefix}_supported"] = float(supported)
        metrics[f"{prefix}_valid_cells"] = valid

    for range_index, range_name in enumerate(("near", "far")):
        true_positive, false_positive, false_negative, positive, valid = (
            float(value)
            for value in lane_range_counts[range_index].tolist()
        )
        supported = positive > 0.0 and valid > 0.0
        average_precision = 0.0
        if supported:
            average_precision = _histogram_average_precision(
                lane_range_positive_histogram[range_index],
                lane_range_negative_histogram[range_index],
            )
        prefix = f"{BEV_LANE_RANGE_METRIC_PREFIX}{range_name}"
        metrics[f"{prefix}_average_precision"] = average_precision
        metrics[f"{prefix}_precision"] = _metric_ratio(
            true_positive,
            true_positive + false_positive,
        )
        metrics[f"{prefix}_recall"] = _metric_ratio(
            true_positive,
            true_positive + false_negative,
        )
        metrics[f"{prefix}_positive_cells"] = positive
        metrics[f"{prefix}_supported"] = float(supported)
        metrics[f"{prefix}_valid_cells"] = valid

    def supported_mean(values, indices) -> float:
        selected = [
            values[index]
            for index in indices
            if class_support[index]
        ]
        return float(np.mean(selected)) if selected else 0.0

    static_indices = range(5)
    dynamic_indices = range(5, class_count)
    static_macro_ap_lift = supported_mean(ap_lifts, static_indices)
    dynamic_macro_ap_lift = supported_mean(ap_lifts, dynamic_indices)
    metrics["bev_static_macro_average_precision"] = supported_mean(
        average_precisions,
        static_indices,
    )
    metrics["bev_dynamic_macro_average_precision"] = supported_mean(
        average_precisions,
        dynamic_indices,
    )
    metrics["bev_static_macro_ap_lift"] = static_macro_ap_lift
    metrics["bev_dynamic_macro_ap_lift"] = dynamic_macro_ap_lift
    metrics["bev_supported_class_count"] = float(sum(class_support))
    metrics["bev_all_classes_supported"] = float(all(class_support))
    metrics["selection_score"] = (
        0.25 * trajectory_quality
        + 0.25 * static_macro_ap_lift
        + 0.50 * dynamic_macro_ap_lift
    )
    return metrics


def _load_resume_checkpoint(
    checkpoint_directory: str,
    *,
    model,
    optimizer,
    scheduler,
    expected: Mapping[str, Any],
) -> ReactiveResumeState:
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
    history_path = Path(checkpoint_directory) / "history.json"
    history = json.loads(history_path.read_text(encoding="ascii"))
    if (
        not isinstance(history, list)
        or any(not isinstance(item, dict) for item in history)
    ):
        raise ValueError("Reactive DDP resume checkpoint has invalid history")
    checkpoint_kind = str(
        training_state.get("checkpoint_kind", "epoch")
    )
    if checkpoint_kind == "epoch":
        if not history:
            raise ValueError(
                "Reactive epoch checkpoint has empty history"
            )
        return ReactiveResumeState(
            epoch=int(payload["epoch"]) + 1,
            optimizer_step_in_epoch=0,
            best_selection_score=float(
                training_state.get(
                    "best_selection_score",
                    -float("inf"),
                )
            ),
            best_ade_6p4s_m=float(
                training_state.get("best_ade_6p4s_m", float("inf"))
            ),
            epoch_history=history,
        )
    if checkpoint_kind != "step":
        raise ValueError(
            f"unsupported Reactive checkpoint kind {checkpoint_kind}"
        )
    if (
        training_state.get("step_checkpoint_version")
        != REACTIVE_STEP_CHECKPOINT_VERSION
    ):
        raise ValueError("Reactive step checkpoint version differs")
    optimizer_steps = int(expected["optimizer_steps_per_epoch"])
    optimizer_step_in_epoch = int(
        training_state.get("optimizer_step_in_epoch", -1)
    )
    if not 0 < optimizer_step_in_epoch <= optimizer_steps:
        raise ValueError(
            "Reactive step checkpoint optimizer position is invalid"
        )
    rank_train_states = training_state.get("rank_train_states")
    rank_rng_states = training_state.get("rank_rng_states")
    world_size = int(expected["distributed_world_size"])
    required_train_state_keys = {
        "consumed_samples",
        "gradient_totals",
        "loader_restarts",
        "rank",
        "term_totals",
    }
    if (
        not isinstance(rank_train_states, list)
        or not isinstance(rank_rng_states, list)
        or len(rank_train_states) != world_size
        or len(rank_rng_states) != world_size
        or any(
            not isinstance(item, Mapping)
            for item in rank_train_states + rank_rng_states
        )
    ):
        raise ValueError("Reactive step checkpoint rank state is invalid")
    if any(
        not required_train_state_keys.issubset(item)
        for item in rank_train_states
    ):
        raise ValueError(
            "Reactive step checkpoint train state is incomplete"
        )
    return ReactiveResumeState(
        epoch=int(payload["epoch"]),
        optimizer_step_in_epoch=optimizer_step_in_epoch,
        best_selection_score=float(
            training_state.get(
                "best_selection_score",
                -float("inf"),
            )
        ),
        best_ade_6p4s_m=float(
            training_state.get("best_ade_6p4s_m", float("inf"))
        ),
        epoch_history=history,
        rank_train_states=tuple(rank_train_states),
        rank_rng_states=tuple(rank_rng_states),
    )


def _report_reactive_epoch(
    report,
    metrics: Mapping[str, Any],
    *,
    checkpoint,
    failure_message: str | None = None,
) -> None:
    report(metrics, checkpoint=checkpoint)
    if failure_message is not None:
        raise RuntimeError(failure_message)


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """Train one fixed Reactive stage on every Ray worker."""
    import torch
    import torch.distributed as dist
    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import get_device, prepare_model

    from data_parsing.pre_extracted import (
        BEVClassRepeatPolicy,
        derive_bev_pos_weights,
        derive_bev_repeat_factors,
        discover_bev_sample_statistics,
        discover_validation_sample_uids,
        make_multi_dataset_loader,
        passthrough_nodesplitter,
        select_bev_validation_sample_uids,
        summarize_bev_training_statistics,
    )
    from model_components.auto_e2e import AutoE2E
    from model_components.bevformer_v2_pretrained import (
        load_bevformer_v2_t8_checkpoint,
    )
    from training.reactive_multitask import (
        REACTIVE_MODEL_ARCHITECTURE_VERSION,
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
    local_bev_records = (
        discover_bev_sample_statistics(local_directories)
        if stage is ReactiveTrainingStage.NUPLAN_FULL
        else None
    )
    validation_sample_uids: tuple[str, ...] | None = None
    validation_sample_uid_sha256 = ""
    validation_sample_count = 0
    local_validation_limit = math.ceil(
        int(config.get("validation_sample_limit", 1024))
        / world_size
    )
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        assert local_bev_records is not None
        validation_sample_uids = select_bev_validation_sample_uids(
            local_bev_records,
            val_fraction=float(config["val_fraction"]),
            sample_limit=local_validation_limit,
        )
    else:
        validation_sample_uids = discover_validation_sample_uids(
            local_directories,
            val_fraction=float(config["val_fraction"]),
            sample_limit=local_validation_limit,
        )
    rank_validation_uids: list[Any] = [None] * world_size
    dist.all_gather_object(
        rank_validation_uids,
        validation_sample_uids,
    )
    global_validation_uids = sorted(
        str(sample_uid)
        for rank_uids in rank_validation_uids
        for sample_uid in rank_uids
    )
    if len(set(global_validation_uids)) != len(global_validation_uids):
        raise ValueError(
            "Reactive validation subset contains duplicate samples"
        )
    validation_sample_count = len(global_validation_uids)
    validation_sample_uid_sha256 = hashlib.sha256(
        "\n".join(global_validation_uids).encode("utf-8")
    ).hexdigest()
    bev_pos_weights: tuple[float, ...] = (1.0,) * 8
    bev_repeat_factors: tuple[int, ...] = (1,) * 8
    bev_repeat_policy = None
    raw_bev_statistics = None
    effective_bev_statistics = None
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        assert local_bev_records is not None
        local_raw_statistics = summarize_bev_training_statistics(
            local_bev_records,
            val_fraction=float(config["val_fraction"]),
        )
        raw_bev_statistics = _all_reduce_bev_statistics(
            local_raw_statistics,
            device,
        )
        # Validate support before repetition can inflate sample counts.
        derive_bev_pos_weights(
            raw_bev_statistics,
            max_weight=float(config["bev_pos_weight_cap"]),
            min_positive_samples=int(
                config["bev_min_positive_samples"]
            ),
            min_positive_cells=int(
                config["bev_min_positive_cells"]
            ),
        )
        bev_repeat_factors = derive_bev_repeat_factors(
            raw_bev_statistics,
            frequency_threshold=float(
                config["bev_repeat_frequency_threshold"]
            ),
            max_repeat=int(config["bev_max_repeat"]),
        )
        local_effective_statistics = (
            summarize_bev_training_statistics(
                local_bev_records,
                val_fraction=float(config["val_fraction"]),
                repeat_factors=bev_repeat_factors,
            )
        )
        effective_bev_statistics = _all_reduce_bev_statistics(
            local_effective_statistics,
            device,
        )
        bev_pos_weights = tuple(round(value, 6) for value in (
            derive_bev_pos_weights(
                effective_bev_statistics,
                max_weight=float(config["bev_pos_weight_cap"]),
            )
        ))
        mean_repeat = (
            effective_bev_statistics.effective_exposure_count
            / effective_bev_statistics.sample_count
        )
        bev_repeat_policy = BEVClassRepeatPolicy(
            repeat_factors=bev_repeat_factors,
            mean_repeat=mean_repeat,
        )

    seed = int(config["training_seed"])
    _seed_epoch(seed, 0, 0)
    constructor_kwargs = reactive_model_kwargs(
        stage,
        num_views=plan.num_views,
    )
    parent_uri = str(config.get("parent_checkpoint_uri") or "")
    restored = train.get_checkpoint()
    initialize_bevformer = (
        bool(config["is_pretrained"])
        and not parent_uri
        and restored is None
    )
    model = AutoE2E(
        backbone=str(config["backbone"]),
        embed_dim=256,
        # The official BEVFormer checkpoint owns both R50 and encoder init.
        is_pretrained=False,
        **constructor_kwargs,
    ).to(device)
    initialization_metadata: dict[str, object] | None = None
    lineage: dict[str, Any] = {}
    if initialize_bevformer:
        pretrained_path = cache_root / "bevformer-v2-r50-t8.pth"
        pretrained_path.parent.mkdir(parents=True, exist_ok=True)
        _download_checkpoint(
            str(config["bevformer_pretrained_checkpoint_uri"]),
            pretrained_path,
        )
        initialization_report = load_bevformer_v2_t8_checkpoint(
            model,
            pretrained_path,
            expected_sha256=str(
                config["bevformer_pretrained_checkpoint_sha256"]
            ),
        )
        initialization_metadata = initialization_report.metadata()
        lineage["bevformer_v2_parent_checkpoint_sha256"] = (
            initialization_report.source_sha256
        )
    if parent_uri and restored is None:
        parent_path = cache_root / "stage-a-parent.pt"
        parent_path.parent.mkdir(parents=True, exist_ok=True)
        _download_checkpoint(parent_uri, parent_path)
        lineage.update(
            load_stage_a_parent(
                model,
                parent_path,
                target_camera_slots=plan.camera_slots,
            )
        )
        inherited_initialization = lineage.get(
            "bevformer_v2_initialization"
        )
        if inherited_initialization is not None:
            initialization_metadata = dict(inherited_initialization)
    configure_model_for_stage(
        model,
        stage,
        freeze_bevformer=bool(config["freeze_bevformer"]),
        train_bev_head=float(config["bev_weight"]) > 0.0,
    )
    freeze_bevformer = bool(config["freeze_bevformer"])
    synchronized_temporal_batch_norm_count = 0
    if not freeze_bevformer:
        synchronized_temporal_batch_norm_count = (
            _synchronize_t8_temporal_batch_norm(model)
        )
    objective = ReactiveMultitaskObjective(
        stage,
        bev_pos_weight=bev_pos_weights,
        trajectory_weight=float(config["trajectory_weight"]),
        bev_weight=float(config["bev_weight"]),
        route_weight=float(config["route_weight"]),
        corridor_pos_weight=float(config["corridor_pos_weight"]),
    ).to(device)

    ddp_model_state_sha256 = _assert_ddp_model_state_consistent(model)
    if rank == 0:
        print(
            "Verified deterministic DDP model state: "
            f"{ddp_model_state_sha256}",
            flush=True,
        )
    model = prepare_model(
        model,
        parallel_strategy="ddp",
        parallel_strategy_kwargs={
            # State equivalence is verified before DDP and after resume.
            "broadcast_buffers": not freeze_bevformer,
            "bucket_cap_mb": REACTIVE_DDP_BUCKET_CAP_MB,
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
            "init_sync": False,
            "static_graph": True,
        },
    )
    _seed_epoch(seed, rank, 0)
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler_identity, scheduler = _build_reactive_scheduler(optimizer)
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
    checkpoint_interval_steps = int(config["checkpoint_interval_steps"])
    _validate_checkpoint_interval(
        checkpoint_interval_steps,
        optimizer_steps,
    )
    global_batch = (
        world_size
        * int(config["per_rank_batch_size"])
        * int(config["gradient_accumulation_steps"])
    )
    expected_resume = {
        "model_architecture_version": REACTIVE_MODEL_ARCHITECTURE_VERSION,
        "dataset_manifest_sha256": plan.dataset_manifest_sha256,
        "camera_slots": list(plan.camera_slots),
        "physical_camera_order": list(plan.physical_camera_order),
        "distributed_assignment_sha256": assignment_sha256,
        "distributed_global_batch": global_batch,
        "distributed_precision": str(config["precision"]),
        "distributed_world_size": world_size,
        "capacity_block_end_utc": str(
            config.get("capacity_block_end_utc") or ""
        ),
        "gradient_clip_max_norm": float(config["grad_clip"]),
        "gradient_clip_mode": "branch_v1",
        "epochs": int(config["epochs"]),
        "optimizer_steps_per_epoch": optimizer_steps,
        "route_metrics_version": ROUTE_VALIDATION_METRICS_VERSION,
        "step_checkpoint_version": REACTIVE_STEP_CHECKPOINT_VERSION,
        "trajectory_weight": float(config["trajectory_weight"]),
        "bev_weight": float(config["bev_weight"]),
        "route_weight": float(config["route_weight"]),
        "corridor_pos_weight": float(config["corridor_pos_weight"]),
        "training_seed": seed,
        "scheduler_identity": scheduler_identity,
        "temporal_normalization_identity": (
            "sync_batch_norm_running_stats_v1"
        ),
        "synchronized_temporal_batch_norm_count": (
            synchronized_temporal_batch_norm_count
        ),
        "freeze_bevformer": bool(config["freeze_bevformer"]),
        "training_stage": stage.value,
        "bev_pos_weights": list(bev_pos_weights),
        "bev_repeat_factors": list(bev_repeat_factors),
        "bev_taxonomy_version": BEV_SEGMENTATION_TAXONOMY_VERSION,
        "validation_sample_count": validation_sample_count,
        "validation_sample_uid_sha256": validation_sample_uid_sha256,
        "allow_random_bevformer_init": bool(
            config.get("allow_random_bevformer_init", False)
        ),
        "bevformer_v2_initialization": initialization_metadata,
    }
    resume_state = ReactiveResumeState(
        epoch=1,
        optimizer_step_in_epoch=0,
        best_selection_score=-float("inf"),
        best_ade_6p4s_m=float("inf"),
        epoch_history=[],
    )
    if restored is not None:
        with restored.as_directory() as checkpoint_directory:
            resume_payload = torch.load(
                Path(checkpoint_directory) / "checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            resume_config = resume_payload.get("config")
            if not isinstance(resume_config, Mapping):
                raise ValueError(
                    "Reactive DDP resume checkpoint has no config"
                )
            resume_initialization = resume_config.get(
                "bevformer_v2_initialization"
            )
            if bool(config["is_pretrained"]):
                expected_source_sha256 = str(
                    config["bevformer_pretrained_checkpoint_sha256"]
                )
                if (
                    not isinstance(resume_initialization, Mapping)
                    or resume_initialization.get("source_sha256")
                    != expected_source_sha256
                    or resume_config.get(
                        "bevformer_v2_parent_checkpoint_sha256"
                    )
                    != expected_source_sha256
                ):
                    raise ValueError(
                        "Reactive DDP resume checkpoint has invalid "
                        "BEVFormer initialization provenance"
                    )
                initialization_metadata = dict(resume_initialization)
                lineage["bevformer_v2_parent_checkpoint_sha256"] = (
                    expected_source_sha256
                )
            elif resume_initialization is not None:
                raise ValueError(
                    "random-init resume unexpectedly has BEVFormer "
                    "initialization provenance"
                )
            expected_resume["bevformer_v2_initialization"] = (
                initialization_metadata
            )
            for lineage_key in (
                "stage_a_parent_checkpoint_sha256",
                "stage_a_config_digest",
                "stage_a_model_state_sha256",
                "stage_a_freeze_bevformer",
                "stage_a_camera_embedding_transfer",
                "bevformer_v2_initialization_mode",
            ):
                lineage_value = resume_config.get(lineage_key)
                if lineage_value is not None:
                    lineage[lineage_key] = lineage_value
            resume_state = _load_resume_checkpoint(
                checkpoint_directory,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected=expected_resume,
            )
            resume_model_state_sha256 = (
                _assert_ddp_model_state_consistent(model)
            )
            if rank == 0:
                print(
                    "Verified resumed DDP model state: "
                    f"{resume_model_state_sha256}",
                    flush=True,
                )
    if _resume_completed_requested_epochs(
        resume_state,
        int(config["epochs"]),
    ):
        return
    best_selection_score = resume_state.best_selection_score
    best_ade = resume_state.best_ade_6p4s_m
    epoch_history = list(resume_state.epoch_history)

    hostnames: list[str | None] = [None] * world_size
    dist.all_gather_object(hostnames, socket.gethostname())
    unique_hostnames = sorted({str(host) for host in hostnames})
    expected_hostname_count = expected_reactive_hostname_count(world_size)
    if len(unique_hostnames) != expected_hostname_count:
        raise RuntimeError(
            "Ray worker placement invariant failed: "
            f"world_size={world_size} "
            f"expected_hostname_count={expected_hostname_count} "
            f"hosts={unique_hostnames}"
        )

    model_config = {
        "model_architecture_version": REACTIVE_MODEL_ARCHITECTURE_VERSION,
        "backbone": str(config["backbone"]),
        "embed_dim": 256,
        "camera_slots": list(plan.camera_slots),
        "physical_camera_order": list(plan.physical_camera_order),
        "is_pretrained": bool(config["is_pretrained"]),
        **constructor_kwargs,
        "distributed_assignment_sha256": assignment_sha256,
        "distributed_global_batch": global_batch,
        "distributed_precision": str(config["precision"]),
        "distributed_world_size": world_size,
        "gradient_clip_max_norm": float(config["grad_clip"]),
        "gradient_clip_mode": "branch_v1",
        "epochs": int(config["epochs"]),
        "optimizer_steps_per_epoch": optimizer_steps,
        "route_metrics_version": ROUTE_VALIDATION_METRICS_VERSION,
        "step_checkpoint_version": REACTIVE_STEP_CHECKPOINT_VERSION,
        "trajectory_weight": float(config["trajectory_weight"]),
        "bev_weight": float(config["bev_weight"]),
        "route_weight": float(config["route_weight"]),
        "corridor_pos_weight": float(config["corridor_pos_weight"]),
        "training_seed": seed,
        "scheduler_identity": scheduler_identity,
        "temporal_normalization_identity": (
            "sync_batch_norm_running_stats_v1"
        ),
        "synchronized_temporal_batch_norm_count": (
            synchronized_temporal_batch_norm_count
        ),
        "freeze_bevformer": bool(config["freeze_bevformer"]),
        "bev_pos_weights": list(bev_pos_weights),
        "bev_repeat_factors": list(bev_repeat_factors),
        "bev_taxonomy_version": BEV_SEGMENTATION_TAXONOMY_VERSION,
        "validation_sample_count": validation_sample_count,
        "validation_sample_uid_sha256": validation_sample_uid_sha256,
        "allow_random_bevformer_init": bool(
            config.get("allow_random_bevformer_init", False)
        ),
        "bevformer_v2_initialization": initialization_metadata,
    }
    if raw_bev_statistics is not None:
        model_config["bev_raw_statistics"] = (
            raw_bev_statistics.metadata()
        )
    if effective_bev_statistics is not None:
        model_config["bev_effective_statistics"] = (
            effective_bev_statistics.metadata()
        )
    started = time.perf_counter()
    for epoch in range(
        resume_state.epoch,
        int(config["epochs"]) + 1,
    ):
        _seed_epoch(seed, rank, epoch)
        torch.cuda.reset_peak_memory_stats(device)
        start_optimizer_step = (
            resume_state.optimizer_step_in_epoch
            if epoch == resume_state.epoch
            else 0
        )
        resume_rank_train_state = _rank_resume_value_for_epoch(
            resume_state.rank_train_states,
            current_epoch=epoch,
            resume_epoch=resume_state.epoch,
            start_optimizer_step=start_optimizer_step,
            rank=rank,
            world_size=world_size,
            name="train state",
        )
        resume_rank_rng_state = _rank_resume_value_for_epoch(
            resume_state.rank_rng_states,
            current_epoch=epoch,
            resume_epoch=resume_state.epoch,
            start_optimizer_step=start_optimizer_step,
            rank=rank,
            world_size=world_size,
            name="RNG state",
        )
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
            bev_repeat_policy=bev_repeat_policy,
            nodesplitter=passthrough_nodesplitter,
        )

        def report_step_checkpoint(
            completed_optimizer_steps: int,
            local_train_state: Mapping[str, Any],
        ) -> None:
            local_rng_state = {
                "rank": rank,
                **_capture_rng_state(),
            }
            rank_payloads: list[dict[str, Any] | None] = [
                None
            ] * world_size
            dist.all_gather_object(
                rank_payloads,
                {
                    "rank": rank,
                    "rng_state": local_rng_state,
                    "train_state": dict(local_train_state),
                },
            )
            if any(
                not isinstance(payload, Mapping)
                for payload in rank_payloads
            ):
                raise RuntimeError(
                    "Reactive step checkpoint rank state is incomplete"
                )
            validated_payloads = [
                dict(payload)
                for payload in rank_payloads
                if isinstance(payload, Mapping)
            ]
            ordered_payloads = sorted(
                validated_payloads,
                key=lambda payload: int(payload["rank"]),
            )
            if [
                int(payload["rank"])
                for payload in ordered_payloads
            ] != list(range(world_size)):
                raise RuntimeError(
                    "Reactive step checkpoint rank order is invalid"
                )
            rank_train_states = [
                dict(payload["train_state"])
                for payload in ordered_payloads
            ]
            rank_rng_states = [
                dict(payload["rng_state"])
                for payload in ordered_payloads
            ]
            executed_optimizer_steps = (
                (epoch - 1) * optimizer_steps
                + completed_optimizer_steps
            )
            step_metrics: dict[str, Any] = {
                "checkpoint_kind": "step",
                "checkpoint_retention_score": float(
                    executed_optimizer_steps
                ),
                "checkpoint_selection_score": -1.0,
                "dataset_manifest_sha256": (
                    plan.dataset_manifest_sha256
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "epoch": epoch,
                "executed_optimizer_steps": (
                    executed_optimizer_steps
                ),
                "is_best": 0,
                "optimizer_step_in_epoch": (
                    completed_optimizer_steps
                ),
                "optimizer_steps_per_epoch": optimizer_steps,
                "step_checkpoint_version": (
                    REACTIVE_STEP_CHECKPOINT_VERSION
                ),
                "world_size": world_size,
            }
            checkpoint_sha256: str | None = None
            with tempfile.TemporaryDirectory() as checkpoint_directory:
                checkpoint = None
                if rank == 0:
                    checkpoint_sha256 = save_reactive_checkpoint(
                        Path(checkpoint_directory) / "checkpoint.pt",
                        _base_model(model),
                        stage=stage,
                        dataset_manifest_sha256=(
                            plan.dataset_manifest_sha256
                        ),
                        epoch=epoch,
                        model_config=model_config,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        metrics=step_metrics,
                        training_state={
                            "assignment_sha256": assignment_sha256,
                            "best_ade_6p4s_m": best_ade,
                            "best_selection_score": (
                                best_selection_score
                            ),
                            "checkpoint_kind": "step",
                            "global_batch": global_batch,
                            "optimizer_step_in_epoch": (
                                completed_optimizer_steps
                            ),
                            "optimizer_steps_per_epoch": optimizer_steps,
                            "rank_rng_states": rank_rng_states,
                            "rank_train_states": rank_train_states,
                            "step_checkpoint_version": (
                                REACTIVE_STEP_CHECKPOINT_VERSION
                            ),
                            "world_size": world_size,
                        },
                        lineage=lineage,
                    )
                    (
                        Path(checkpoint_directory) / "history.json"
                    ).write_text(
                        json.dumps(
                            epoch_history,
                            allow_nan=False,
                            indent=2,
                            sort_keys=True,
                        ) + "\n",
                        encoding="ascii",
                    )
                    checkpoint = Checkpoint.from_directory(
                        checkpoint_directory
                    )
                checkpoint_digest: list[str | None] = [
                    checkpoint_sha256
                ]
                dist.broadcast_object_list(checkpoint_digest, src=0)
                step_metrics["checkpoint_sha256"] = str(
                    checkpoint_digest[0]
                )
                train.report(step_metrics, checkpoint=checkpoint)

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
            start_optimizer_step=start_optimizer_step,
            resume_rank_state=resume_rank_train_state,
            resume_rng_state=resume_rank_rng_state,
            checkpoint_interval_steps=checkpoint_interval_steps,
            checkpoint_callback=report_step_checkpoint,
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
            sample_uids=validation_sample_uids,
            decode_future_frames=False,
            nodesplitter=passthrough_nodesplitter,
        )
        validation = _evaluate_global_reactive(
            model,
            validation_loader,
            objective,
            stage=stage,
            device=device,
            probability_bins=int(config["bev_ap_bins"]),
            ade_scale_m=float(config["selection_ade_scale_m"]),
        )
        scheduler.step(validation["selection_score"])
        executed_optimizer_steps = epoch * optimizer_steps
        maximum_delta = _maximum_parameter_delta(
            model,
            world_size=world_size,
        )
        if maximum_delta > 1e-6:
            raise RuntimeError(
                "Reactive DDP replicas diverged: "
                f"maximum_parameter_delta={maximum_delta}"
            )
        feature_scale_weights = _camera_feature_scale_weights(model)
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
                "peak_cuda_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_cuda_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
                "rank": rank,
                "shards": [shard.identity for shard in rank_shards],
            },
        )
        ade_within_guard = validation["ade_6p4s_m"] <= (
            best_ade
            + float(config["selection_ade_regression_margin_m"])
        )
        is_best = (
            validation["selection_score"] > best_selection_score
            and ade_within_guard
        )
        checkpoint_selection_score = (
            validation["selection_score"]
            if ade_within_guard
            else -1.0
        )
        checkpoint_retention_score = (
            EPOCH_CHECKPOINT_RETENTION_SCORE_BASE
            + epoch
        )
        if is_best:
            best_selection_score = validation["selection_score"]
        best_ade = min(best_ade, validation["ade_6p4s_m"])
        diagnostic_metrics: dict[str, float | int] = {
            f"{CAMERA_FEATURE_SCALE_WEIGHT_METRIC_PREFIX}{index}": weight
            for index, weight in enumerate(feature_scale_weights)
        }
        diagnostic_metrics["synchronized_temporal_batch_norm_count"] = (
            synchronized_temporal_batch_norm_count
        )
        for evidence in rank_evidence:
            if not isinstance(evidence, Mapping):
                raise RuntimeError("Reactive rank evidence is incomplete")
            evidence_rank = int(evidence["rank"])
            for memory_name in (
                "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes",
            ):
                metric_prefix = {
                    "peak_cuda_allocated_bytes": (
                        PEAK_CUDA_ALLOCATED_BYTES_METRIC_PREFIX
                    ),
                    "peak_cuda_reserved_bytes": (
                        PEAK_CUDA_RESERVED_BYTES_METRIC_PREFIX
                    ),
                }[memory_name]
                diagnostic_metrics[
                    f"{metric_prefix}{evidence_rank}"
                ] = int(evidence[memory_name])
        checkpoint_metrics: dict[str, Any] = dict(validation)
        checkpoint_metrics.update(diagnostic_metrics)
        checkpoint_metrics["executed_optimizer_steps"] = (
            executed_optimizer_steps
        )
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
                    metrics=checkpoint_metrics,
                    training_state={
                        "assignment_sha256": assignment_sha256,
                        "best_ade_6p4s_m": best_ade,
                        "best_selection_score": best_selection_score,
                        "checkpoint_kind": "epoch",
                        "global_batch": global_batch,
                        "optimizer_steps_per_epoch": optimizer_steps,
                        "rank_evidence": rank_evidence,
                        "world_size": world_size,
                    },
                    lineage=lineage,
                )
            checkpoint_digest: list[str | None] = [checkpoint_sha256]
            dist.broadcast_object_list(checkpoint_digest, src=0)
            metrics = {
                **diagnostic_metrics,
                "checkpoint_kind": "epoch",
                "checkpoint_sha256": str(checkpoint_digest[0]),
                "checkpoint_selection_score": (
                    checkpoint_selection_score
                ),
                "checkpoint_retention_score": (
                    checkpoint_retention_score
                ),
                "bev_weight": float(config["bev_weight"]),
                "corridor_pos_weight": float(
                    config["corridor_pos_weight"]
                ),
                "dataset_manifest_sha256": (
                    plan.dataset_manifest_sha256
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "epoch": epoch,
                "executed_optimizer_steps": executed_optimizer_steps,
                "gradient_clip_max_norm": float(config["grad_clip"]),
                "gradient_clip_mode": "branch_v1",
                "route_metrics_version": (
                    ROUTE_VALIDATION_METRICS_VERSION
                ),
                "is_best": int(is_best),
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
                "maximum_parameter_delta": maximum_delta,
                "optimizer_steps_per_epoch": optimizer_steps,
                "freeze_bevformer": int(
                    bool(config["freeze_bevformer"])
                ),
                "route_weight": float(config["route_weight"]),
                "scheduler_identity": scheduler_identity,
                "train_bev_segmentation": train_metrics[
                    "bev_segmentation"
                ],
                "train_bev_segmentation_bce": train_metrics[
                    "bev_segmentation_bce"
                ],
                "train_bev_segmentation_dice": train_metrics[
                    "bev_segmentation_dice"
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
                "validation_selection_score": validation[
                    "selection_score"
                ],
                "validation_sample_count": validation_sample_count,
                "validation_sample_uid_sha256": (
                    validation_sample_uid_sha256
                ),
                "training_seed": seed,
                "trajectory_weight": float(
                    config["trajectory_weight"]
                ),
                "world_size": world_size,
            }
            for name, value in validation.items():
                if name not in {
                    "ade_6p4s_m",
                    "complete_samples",
                    "fde_6p4s_m",
                    "selection_score",
                }:
                    metrics[f"validation_{name}"] = value
            for class_index, value in enumerate(bev_pos_weights):
                metrics[f"bev_pos_weight_{class_index}"] = value
                metrics[f"bev_repeat_factor_{class_index}"] = (
                    bev_repeat_factors[class_index]
                )
            for group_name in (
                "camera",
                "front_gate",
                "navigation",
                "planner",
            ):
                metrics[
                    f"train_gradient_{group_name}_pre_clip_norm"
                ] = train_metrics[
                    f"gradient_{group_name}_pre_clip_norm"
                ]
                metrics[f"train_gradient_{group_name}_clip_scale"] = (
                    train_metrics[
                        f"gradient_{group_name}_clip_scale"
                    ]
                )
            epoch_history.append(metrics)
            if rank == 0:
                history_path = (
                    Path(checkpoint_directory) / "history.json"
                )
                history_path.write_text(
                    json.dumps(
                        epoch_history,
                        allow_nan=False,
                        indent=2,
                        sort_keys=True,
                    ) + "\n",
                    encoding="ascii",
                )
                checkpoint = Checkpoint.from_directory(
                    checkpoint_directory
                )
            _report_reactive_epoch(
                train.report,
                metrics,
                checkpoint=checkpoint,
            )


def _result_checkpoint_entry(entry) -> tuple[Any, dict[str, Any]]:
    if isinstance(entry, tuple) and len(entry) == 2:
        checkpoint, metrics = entry
    else:
        checkpoint = getattr(entry, "checkpoint", None)
        metrics = getattr(entry, "metrics", None)
    if checkpoint is None or not isinstance(metrics, Mapping):
        raise ValueError("Ray best checkpoint entry is invalid")
    return checkpoint, dict(metrics)


def _select_result_checkpoint(
    result,
) -> tuple[Any, dict[str, Any]]:
    """Select the best ADE-guarded checkpoint."""
    if result.checkpoint is None:
        raise RuntimeError("Reactive Ray training returned no checkpoint")

    candidates = []
    for entry in getattr(result, "best_checkpoints", ()) or ():
        checkpoint, metrics = _result_checkpoint_entry(entry)
        if int(metrics.get("is_best", 0)) != 1:
            continue
        score = float(metrics.get("checkpoint_selection_score", -1.0))
        if not math.isfinite(score) or score < 0.0:
            continue
        candidates.append((
            score,
            int(metrics.get("epoch", 0)),
            checkpoint,
            metrics,
        ))
    if not candidates:
        raise RuntimeError(
            "Reactive Ray training retained no ADE-guarded best checkpoint"
        )
    _, _, checkpoint, metrics = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return checkpoint, metrics


def _checkpoint_history(checkpoint) -> list[dict[str, Any]]:
    with checkpoint.as_directory() as checkpoint_directory:
        path = Path(checkpoint_directory) / "history.json"
        history = json.loads(path.read_text(encoding="ascii"))
    if (
        not isinstance(history, list)
        or not history
        or any(not isinstance(item, dict) for item in history)
    ):
        raise ValueError("Reactive Ray checkpoint history is invalid")
    return history


def run_reactive_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    """Launch the fixed-size Ray Train worker group."""
    validate_reactive_stage_config(config)
    import ray
    from ray import train
    from ray.train.torch import TorchTrainer

    from distributed_training.ray_torch_backend import (
        PreparedCudaTorchConfig,
    )

    if not ray.is_initialized():
        ray.init(address="auto")
    scaling_config = train.ScalingConfig(
        num_workers=int(config["num_workers"]),
        use_gpu=True,
        resources_per_worker={
            "CPU": int(config["worker_cpus"]),
            "GPU": 1,
        },
        placement_strategy=(
            "SPREAD"
            if int(config["num_workers"]) == 2
            else "PACK"
        ),
    )
    # Train V2 reloads checkpoint_manager_snapshot.json when a Flyte retry
    # recreates this trainer with the same stable experiment directory.
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=dict(config),
        torch_config=PreparedCudaTorchConfig(
            init_method="tcp",
            timeout_s=300,
        ),
        scaling_config=scaling_config,
        run_config=train.RunConfig(
            name=str(config["run_name"]),
            storage_path=str(config["storage_path"]),
            failure_config=train.FailureConfig(max_failures=2),
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=int(config["epochs"]) + 2,
                checkpoint_score_attribute=(
                    "checkpoint_retention_score"
                ),
                checkpoint_score_order="max",
            ),
        ),
    )
    result = trainer.fit()
    if result.checkpoint is None:
        raise RuntimeError("Reactive Ray training returned no checkpoint")
    history = _checkpoint_history(result.checkpoint)
    checkpoint, metrics = _select_result_checkpoint(result)
    checkpoint_uri = normalize_ray_checkpoint_uri(
        str(checkpoint.path),
        str(config["storage_path"]),
    )
    if int(metrics.get("world_size", 0)) != int(config["num_workers"]):
        raise RuntimeError(
            f"Reactive Ray result has unexpected world size: {metrics}"
        )
    selected_digest = str(metrics.get("checkpoint_sha256", ""))
    selected_rows = [
        row
        for row in history
        if str(row.get("checkpoint_sha256", "")) == selected_digest
    ]
    if (
        len(selected_rows) != 1
        or int(selected_rows[0].get("epoch", -1))
        != int(metrics.get("epoch", -2))
    ):
        raise RuntimeError(
            "selected Reactive checkpoint does not match epoch history"
        )
    final_metrics = dict(result.metrics)
    if (
        str(history[-1].get("checkpoint_sha256", ""))
        != str(final_metrics.get("checkpoint_sha256", ""))
    ):
        raise RuntimeError(
            "final Reactive checkpoint does not match epoch history"
        )
    return {
        "checkpoint_file_uri": f"{checkpoint_uri}/checkpoint.pt",
        "checkpoint_uri": checkpoint_uri,
        "final_metrics": final_metrics,
        "history": history,
        "metrics": metrics,
        "selected_epoch": int(metrics["epoch"]),
        "run_name": str(config["run_name"]),
        "storage_path": str(config["storage_path"]),
    }
