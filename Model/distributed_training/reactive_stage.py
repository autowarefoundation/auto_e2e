"""Ray Train DDP runner for nuPlan and L2D Reactive stages."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import re
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_CLASSES,
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
from evaluation.bev_segmentation import (
    BEV_AP_BOOTSTRAP_BINS,
    BEV_AP_BOOTSTRAP_MAX_WEIGHT,
    BEV_AP_BOOTSTRAP_REPLICATES,
    BEV_AP_BOOTSTRAP_VERSION,
    BEV_AP_BOOTSTRAP_WEIGHT_SCALE,
    accumulate_fixed_point_bootstrap_histogram,
    fixed_point_bayesian_bootstrap_weights,
    validate_fixed_point_bootstrap_capacity,
)
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
from reactive_training_contracts import (
    BEV_CHECKPOINT_MIN_CLASS_IOU,
    BEV_CHECKPOINT_MIN_CLASS_PRECISION,
    BEV_CHECKPOINT_QUALITY_GUARD_VERSION,
)
from training.reactive_multitask import (
    BEV_HEAD_INITIALIZATION_VERSION,
    BEV_SAMPLING_IMPORTANCE_CORRECTION_VERSION,
    ReactiveTrainingScope,
    ReactiveTrainingStage,
)


PRODUCTION_WORLD_SIZES = frozenset({2, 4, 8})
SUPPORTED_WORLD_SIZES = PRODUCTION_WORLD_SIZES | {1}
SUPPORTED_PER_RANK_BATCH_SIZES = frozenset({1, 2, 4})
SUPPORTED_PRECISIONS = frozenset({"fp32", "bf16"})
SINGLE_WORKER_SMOKE_MAX_STEPS = 16
SINGLE_WORKER_SMOKE_MAX_SOURCES = 2
SINGLE_WORKER_SMOKE_MAX_VALIDATION_SAMPLES = 256
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
REACTIVE_STEP_CHECKPOINT_VERSION = "reactive_step_checkpoint_v3"
REACTIVE_SAMPLE_STREAM_DIGEST_VERSION = "reactive_sample_stream_v1"
REACTIVE_SAMPLE_STREAM_INITIAL_SHA256 = hashlib.sha256(
    REACTIVE_SAMPLE_STREAM_DIGEST_VERSION.encode("ascii")
).hexdigest()
REACTIVE_DDP_BUCKET_CAP_MB = 16
REACTIVE_DDP_TIMEOUT_SECONDS = 1800
REACTIVE_PERFORMANCE_LOG_INTERVAL_STEPS = 64
REACTIVE_PERFORMANCE_LOG_VERSION = "reactive_performance_v1"
BEV_GRADIENT_BUDGET_DIAGNOSTIC_STEPS = 8
BEV_MAX_RANK_TRUNCATION_FRACTION = 0.125
BEV_HEAD_CLASSIFIER_WEIGHT_STD = 0.01
LEGACY_FROZEN_BEV_EPOCH_RESUME_FIELDS = frozenset({
    "bev_ap_bins",
    "bev_checkpoint_min_class_iou",
    "bev_checkpoint_min_class_precision",
    "bev_checkpoint_quality_guard_version",
    "bev_class_weights",
    "bev_encoder_learning_rate",
    "bev_head_initialization",
    "bev_loss_version",
    "bev_positive_pair_frequencies",
    "bev_rank_sampling_evidence",
    "bev_sampling_importance_correction",
    "bounded_bev_canary",
    "dataset_split_metrics",
    "num_loader_workers",
    "optimizer_identity",
    "sample_stream_digest_version",
    "shuffle_buffer",
    "step_checkpoint_version",
    "training_scope",
    "validation_fraction",
    "validation_positive_sample_counts",
    "validation_sample_limit",
})
REACTIVE_PERFORMANCE_PHASES = (
    "loader_wait",
    "host_to_device",
    "input_prepare",
    "forward",
    "backward",
    "gradient",
    "finite_collective",
    "optimizer",
    "step",
)
EPOCH_CHECKPOINT_RETENTION_SCORE_BASE = 1_000_000_000_000.0
P5EN_MINIMUM_REMAINING_RUNTIME = timedelta(hours=22)
P5EN_RESUME_MINIMUM_REMAINING_RUNTIME = timedelta(hours=2)
_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def expected_reactive_hostname_count(world_size: int) -> int:
    """Return the reviewed Ray worker host count for each DDP topology."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            "world_size must be one of "
            f"{sorted(SUPPORTED_WORLD_SIZES)}, got {world_size}"
        )
    return 2 if world_size == 2 else 1


def _validate_bev_rank_truncation(
    maximum_truncation_fraction: float,
    *,
    single_worker_smoke: bool,
    bounded_bev_canary: bool,
) -> None:
    """Enforce full-epoch coverage outside bounded pre-flight runs."""
    if (
        not math.isfinite(maximum_truncation_fraction)
        or not 0.0 <= maximum_truncation_fraction <= 1.0
    ):
        raise RuntimeError("BEV rank truncation fraction is invalid")
    if (
        not single_worker_smoke
        and not bounded_bev_canary
        and maximum_truncation_fraction
        > BEV_MAX_RANK_TRUNCATION_FRACTION
    ):
        raise ValueError(
            "BEV rank truncation exceeds the supported fraction: "
            f"{maximum_truncation_fraction:.6f} > "
            f"{BEV_MAX_RANK_TRUNCATION_FRACTION:.6f}"
        )


def _should_enforce_bev_checkpoint_quality_guard(
    *,
    single_worker_smoke: bool,
    bounded_bev_canary: bool,
) -> bool:
    """Reserve absolute class-quality floors for production training."""
    return not single_worker_smoke and not bounded_bev_canary


def _required_parent_training_scope(
    stage: ReactiveTrainingStage,
) -> str | None:
    """Return the only checkpoint scope accepted by a continuation stage."""
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        return ReactiveTrainingScope.BEV_ONLY.value
    if stage is ReactiveTrainingStage.L2D_CONTINUATION:
        return ReactiveTrainingScope.MULTITASK.value
    return None


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
    try:
        training_scope = ReactiveTrainingScope(
            str(
                config.get(
                    "training_scope",
                    ReactiveTrainingScope.MULTITASK.value,
                )
            )
        )
    except ValueError as error:
        raise ValueError("unsupported Reactive training scope") from error
    world_size = int(config.get("num_workers", 0))
    allow_single_worker_smoke = config.get(
        "allow_single_worker_smoke",
        False,
    )
    allow_bounded_bev_canary = config.get(
        "allow_bounded_bev_canary",
        False,
    )
    if not isinstance(allow_single_worker_smoke, bool):
        raise ValueError("allow_single_worker_smoke must be a boolean")
    if not isinstance(allow_bounded_bev_canary, bool):
        raise ValueError("allow_bounded_bev_canary must be a boolean")
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"num_workers must be one of {sorted(SUPPORTED_WORLD_SIZES)}"
        )
    if world_size == 1 and not allow_single_worker_smoke:
        raise ValueError(
            "num_workers=1 requires allow_single_worker_smoke"
        )
    if world_size != 1 and allow_single_worker_smoke:
        raise ValueError(
            "allow_single_worker_smoke requires num_workers=1"
        )
    if allow_bounded_bev_canary and (
        world_size != 8
        or stage is not ReactiveTrainingStage.NUPLAN_FULL
        or training_scope is not ReactiveTrainingScope.BEV_ONLY
        or int(config.get("epochs", 0)) != 2
        or not 1 <= int(config.get("steps_per_epoch", 0)) <= 256
        or int(config.get("per_rank_batch_size", 0)) != 4
        or str(config.get("precision", "")) != "bf16"
        or allow_single_worker_smoke
        or bool(config.get("freeze_bevformer", True))
        or str(config.get("parent_checkpoint_uri") or "")
        or str(config.get("resume_checkpoint_uri") or "")
    ):
        raise ValueError(
            "allow_bounded_bev_canary requires an eight-rank, fresh, "
            "two-epoch nuPlan BEV-only run with at most 256 steps"
        )
    resume_uri = str(config.get("resume_checkpoint_uri") or "")
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
        minimum_runtime = (
            P5EN_RESUME_MINIMUM_REMAINING_RUNTIME
            if resume_uri
            else P5EN_MINIMUM_REMAINING_RUNTIME
        )
        minimum_end = datetime.now(timezone.utc) + minimum_runtime
        if capacity_block_end.astimezone(timezone.utc) < minimum_end:
            minimum_hours = int(
                minimum_runtime.total_seconds() // 3600
            )
            raise ValueError(
                "p5en Capacity Block must have at least "
                f"{minimum_hours} hours remaining"
            )
    if int(config.get("worker_cpus", 0)) <= 0:
        raise ValueError("worker_cpus must be positive")
    if int(config.get("epochs", 0)) <= 0:
        raise ValueError("epochs must be positive")
    per_rank_batch_size = int(
        config.get("per_rank_batch_size", 0)
    )
    if per_rank_batch_size not in SUPPORTED_PER_RANK_BATCH_SIZES:
        raise ValueError(
            "per_rank_batch_size must be one of "
            f"{sorted(SUPPORTED_PER_RANK_BATCH_SIZES)}"
        )
    if int(config.get("gradient_accumulation_steps", 0)) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(config.get("num_loader_workers", -1)) < 0:
        raise ValueError("num_loader_workers must be non-negative")
    if int(config.get("shuffle_buffer", 0)) < 2:
        raise ValueError("shuffle_buffer must be at least two")
    val_fraction = float(config.get("val_fraction", 0.0))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    if not math.isclose(
        val_fraction * 10.0,
        round(val_fraction * 10.0),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "val_fraction must be representable by the ten-bucket split"
        )
    if float(config.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(config.get("bev_encoder_learning_rate", 1e-5)) <= 0.0:
        raise ValueError("bev_encoder_learning_rate must be positive")
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
    if training_scope is ReactiveTrainingScope.BEV_ONLY and (
        stage is not ReactiveTrainingStage.NUPLAN_FULL
        or float(config["trajectory_weight"]) != 0.0
        or float(config["bev_weight"]) <= 0.0
        or float(config["route_weight"]) != 0.0
        or bool(config.get("freeze_bevformer", True))
    ):
        raise ValueError(
            "BEV-only scope requires nuPlan full, only BEV loss, and an "
            "unfrozen BEVFormer"
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
    if parent_uri and resume_uri:
        raise ValueError(
            "parent and resume checkpoints are mutually exclusive"
        )
    if resume_uri and not resume_uri.startswith("s3://"):
        raise ValueError("resume checkpoint must be an S3 URI")
    if (
        stage is ReactiveTrainingStage.NUPLAN_FULL
        and parent_uri
        and (
            training_scope is not ReactiveTrainingScope.MULTITASK
            or not bool(config.get("freeze_bevformer", False))
        )
    ):
        raise ValueError(
            "nuPlan parent checkpoints require multitask training with a "
            "frozen BEVFormer"
        )
    if (
        stage is ReactiveTrainingStage.L2D_CONTINUATION
        and not parent_uri
        and not resume_uri
    ):
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
    if is_pretrained and not parent_uri and not resume_uri:
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
    if int(
        config.get(
            "performance_log_interval_steps",
            REACTIVE_PERFORMANCE_LOG_INTERVAL_STEPS,
        )
    ) <= 0:
        raise ValueError(
            "performance_log_interval_steps must be positive"
        )
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
    if world_size == 1:
        smoke_steps = int(config.get("steps_per_epoch", 0))
        smoke_checkpoint_interval = int(
            config.get("checkpoint_interval_steps", 0)
        )
        if (
            stage is not ReactiveTrainingStage.NUPLAN_FULL
            or training_scope is not ReactiveTrainingScope.BEV_ONLY
            or int(config.get("epochs", 0)) != 1
            or not 2 <= smoke_steps <= SINGLE_WORKER_SMOKE_MAX_STEPS
            or smoke_checkpoint_interval > smoke_steps
            or per_rank_batch_size != 1
            or int(config.get("gradient_accumulation_steps", 0)) != 1
            or str(config.get("backbone", "")) != "res_net_50"
            or precision != "bf16"
            or parent_uri
            or resume_uri
            or capacity_block_end_utc
            or len(sources) > SINGLE_WORKER_SMOKE_MAX_SOURCES
            or int(config.get("validation_sample_limit", 0))
            > SINGLE_WORKER_SMOKE_MAX_VALIDATION_SAMPLES
            or not is_pretrained
            or allow_random_init
            or freeze_bevformer
            or float(config["trajectory_weight"]) != 0.0
            or float(config["bev_weight"]) <= 0.0
            or float(config["route_weight"]) != 0.0
        ):
            raise ValueError(
                "single-worker smoke must be a one-epoch, 2-16 step, "
                "batch-one pretrained nuPlan BEV-only run over at most "
                "two sources with no checkpoint lineage or Capacity Block"
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


def _bev_gradient_diagnostic_step_indices(
    optimizer_steps: int,
) -> frozenset[int]:
    """Spread the fixed diagnostic budget across the entire epoch."""
    if optimizer_steps <= 0:
        raise ValueError("optimizer_steps must be positive")
    sample_count = min(
        BEV_GRADIENT_BUDGET_DIAGNOSTIC_STEPS,
        optimizer_steps,
    )
    if sample_count == 1:
        return frozenset({0})
    return frozenset(
        round(
            sample_index
            * (optimizer_steps - 1)
            / (sample_count - 1)
        )
        for sample_index in range(sample_count)
    )


def _performance_sample_due(
    completed_optimizer_steps: int,
    *,
    interval_steps: int,
) -> bool:
    if completed_optimizer_steps <= 0:
        raise ValueError("completed optimizer steps must be positive")
    if interval_steps <= 0:
        raise ValueError("performance log interval must be positive")
    return completed_optimizer_steps % interval_steps == 0


def _parse_nvidia_smi_csv(output: str) -> dict[str, float]:
    columns = (
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "memory_used_mib",
        "memory_total_mib",
        "power_watts",
    )
    values: dict[str, list[float]] = {name: [] for name in columns}
    for raw_line in output.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != len(columns):
            continue
        try:
            parsed = [float(field) for field in fields]
        except ValueError:
            continue
        for name, value in zip(columns, parsed, strict=True):
            values[name].append(value)
    if not values[columns[0]]:
        return {}
    metrics: dict[str, float] = {
        "performance_gpu_count": float(len(values[columns[0]])),
    }
    for name in columns:
        metrics[f"performance_{name}_min"] = min(values[name])
        metrics[f"performance_{name}_avg"] = (
            sum(values[name]) / len(values[name])
        )
        metrics[f"performance_{name}_max"] = max(values[name])
    return metrics


def _nvidia_smi_snapshot() -> dict[str, float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu="
                "utilization.gpu,utilization.memory,"
                "memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    return _parse_nvidia_smi_csv(result.stdout)


def _aggregate_reactive_performance(
    local_phase_seconds: Mapping[str, float],
    *,
    local_samples: int,
    device,
) -> dict[str, float]:
    import torch
    import torch.distributed as dist

    missing = [
        name
        for name in REACTIVE_PERFORMANCE_PHASES
        if name not in local_phase_seconds
    ]
    if missing:
        raise ValueError(
            f"Reactive performance phases are missing {missing}"
        )
    local_values = torch.tensor(
        [
            *[
                float(local_phase_seconds[name])
                for name in REACTIVE_PERFORMANCE_PHASES
            ],
            float(local_samples),
            float(
                torch.cuda.memory_allocated(device)
                if device.type == "cuda"
                else 0
            ),
            float(
                torch.cuda.memory_reserved(device)
                if device.type == "cuda"
                else 0
            ),
        ],
        dtype=torch.float64,
        device=device,
    )
    summed = local_values.clone()
    maximum = local_values.clone()
    dist.all_reduce(summed, op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    world_size = dist.get_world_size()
    metrics: dict[str, float] = {}
    for index, name in enumerate(REACTIVE_PERFORMANCE_PHASES):
        metrics[f"performance_{name}_seconds_avg"] = (
            float(summed[index].item()) / world_size
        )
        metrics[f"performance_{name}_seconds_max"] = float(
            maximum[index].item()
        )
    sample_index = len(REACTIVE_PERFORMANCE_PHASES)
    allocated_index = sample_index + 1
    reserved_index = sample_index + 2
    maximum_step_seconds = float(
        metrics["performance_step_seconds_max"]
    )
    minimum_positive_seconds = float(np.finfo(np.float64).tiny)
    metrics["performance_global_samples_per_second"] = (
        float(summed[sample_index].item())
        / max(maximum_step_seconds, minimum_positive_seconds)
    )
    metrics["performance_gpu_allocated_bytes_max"] = float(
        maximum[allocated_index].item()
    )
    metrics["performance_gpu_reserved_bytes_max"] = float(
        maximum[reserved_index].item()
    )
    accounted_seconds = sum(
        metrics[f"performance_{name}_seconds_avg"]
        for name in REACTIVE_PERFORMANCE_PHASES
        if name != "step"
    )
    metrics["performance_unaccounted_seconds_avg"] = max(
        0.0,
        metrics["performance_step_seconds_avg"] - accounted_seconds,
    )
    hardware_payload: list[dict[str, float] | None] = [
        (
            _nvidia_smi_snapshot()
            if dist.get_rank() == 0 and device.type == "cuda"
            else None
        )
    ]
    if device.type == "cuda":
        dist.broadcast_object_list(hardware_payload, src=0)
    hardware_metrics = hardware_payload[0]
    if hardware_metrics:
        metrics.update(hardware_metrics)
    return metrics


def _distributed_max_seconds(value: float, device) -> float:
    import torch
    import torch.distributed as dist

    duration = torch.tensor(
        float(value),
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(duration, op=dist.ReduceOp.MAX)
    return float(duration.item())


def _all_reduce_bev_statistics(local_statistics, device):
    """Combine exact rank-local BEV counts into one global contract."""
    import torch
    import torch.distributed as dist

    from data_parsing.pre_extracted import BEVTrainingStatistics

    packed = torch.tensor(
        [
            float(local_statistics.sample_count),
            float(local_statistics.effective_exposure_count),
            *local_statistics.active_sample_count,
            *local_statistics.positive_sample_count,
            *local_statistics.positive_cell_count,
            *local_statistics.positive_fraction_sum,
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

    active_samples = take(class_count)
    positive_samples = take(class_count)
    positive_cells = take(class_count)
    positive_fraction_sum = take(class_count)
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
        active_sample_count=tuple(
            int(round(float(value)))
            for value in active_samples.tolist()
        ),
        positive_sample_count=tuple(
            int(round(float(value)))
            for value in positive_samples.tolist()
        ),
        positive_cell_count=tuple(
            int(round(float(value)))
            for value in positive_cells.tolist()
        ),
        positive_fraction_sum=tuple(
            float(value) for value in positive_fraction_sum.tolist()
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


def _reduce_reactive_validation_state(
    tensors: Sequence[Any],
    *,
    local_sample_uids: Sequence[str] | None = None,
    expected_sample_count: int | None = None,
    expected_sample_uid_sha256: str | None = None,
) -> tuple[str, ...]:
    """Reduce validation tensors and verify global sample identity coverage."""
    import torch
    import torch.distributed as dist

    if not tensors or any(not torch.is_tensor(tensor) for tensor in tensors):
        raise ValueError("Reactive validation reduction needs tensors")
    coverage_inputs = (
        local_sample_uids,
        expected_sample_count,
        expected_sample_uid_sha256,
    )
    if any(value is None for value in coverage_inputs) and not all(
        value is None for value in coverage_inputs
    ):
        raise ValueError(
            "Reactive validation sample coverage inputs must be paired"
        )
    global_sample_uids: tuple[str, ...] = ()
    if local_sample_uids is not None:
        resolved_local_uids = tuple(str(value) for value in local_sample_uids)
        if (
            expected_sample_count is None
            or expected_sample_count <= 0
            or expected_sample_uid_sha256 is None
            or len(expected_sample_uid_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sample_uid_sha256
            )
            or any(not value for value in resolved_local_uids)
            or len(set(resolved_local_uids)) != len(resolved_local_uids)
        ):
            raise ValueError(
                "Reactive validation sample coverage inputs are invalid"
            )
        rank_sample_uids: list[list[str] | None] = [
            None
        ] * dist.get_world_size()
        dist.all_gather_object(
            rank_sample_uids,
            list(resolved_local_uids),
        )
        global_sample_uids = tuple(sorted(
            sample_uid
            for rank_uids in rank_sample_uids
            if rank_uids is not None
            for sample_uid in rank_uids
        ))
        actual_sample_uid_sha256 = hashlib.sha256(
            "\n".join(global_sample_uids).encode("utf-8")
        ).hexdigest()
        if (
            len(global_sample_uids) != expected_sample_count
            or len(set(global_sample_uids)) != len(global_sample_uids)
            or actual_sample_uid_sha256 != expected_sample_uid_sha256
        ):
            raise ValueError(
                "BEV validation sample UID coverage differs from the "
                "frozen manifest"
            )
    for tensor in tensors:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return global_sample_uids


def _raise_distributed_validation_contract_errors(
    local_errors: Sequence[str],
) -> None:
    """Make rank-local validation contract failures fail every rank."""
    import torch.distributed as dist

    resolved = tuple(str(error) for error in local_errors)
    if any(not error for error in resolved):
        raise ValueError("Reactive validation contract errors must be non-empty")
    if not dist.is_initialized():
        if resolved:
            raise ValueError(
                "Reactive validation contract failed: "
                f"{{0: {resolved}}}"
            )
        return
    rank_errors: list[list[str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(rank_errors, list(resolved))
    failures = {
        rank: tuple(errors)
        for rank, errors in enumerate(rank_errors)
        if errors
    }
    if failures:
        raise ValueError(
            "Reactive distributed validation contract failed: "
            f"{failures}"
        )


def _bev_validation_positive_sample_counts(
    records,
    sample_uids,
) -> tuple[int, ...]:
    """Count class-positive samples in one rank's validation subset."""
    requested = tuple(str(value) for value in sample_uids)
    allowed = frozenset(requested)
    if not allowed or len(allowed) != len(requested):
        raise ValueError(
            "BEV validation sample subset must be non-empty and unique"
        )
    selected = {
        record.sample_uid: record
        for record in records
        if record.sample_uid in allowed
    }
    if set(selected) != allowed:
        missing = sorted(allowed - set(selected))
        raise ValueError(
            "BEV validation statistics are missing selected samples: "
            f"{missing[:8]}"
        )
    counts = np.zeros(len(BEV_SEGMENTATION_CLASSES), dtype=np.int64)
    for record in selected.values():
        counts += (
            np.asarray(record.positive_cell_count, dtype=np.int64) > 0
        ).astype(np.int64)
    return tuple(int(value) for value in counts)


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


def _resolve_resume_sources(
    ray_checkpoint: Any | None,
    explicit_resume_uri: str,
) -> tuple[Any | None, str, bool]:
    """Use the Ray recovery state after an explicitly resumed run starts."""
    resume_uri = explicit_resume_uri.rstrip("/")
    if ray_checkpoint is not None:
        return ray_checkpoint, "", bool(resume_uri)
    return None, resume_uri, False


def _should_initialize_bev_head_from_training_statistics(
    training_scope: ReactiveTrainingScope | str,
    *,
    parent_uri: str,
    restored_checkpoint: Any | None,
) -> bool:
    """Initialize the BEV classifier only for a fresh BEV-only run."""
    return (
        ReactiveTrainingScope(training_scope)
        is ReactiveTrainingScope.BEV_ONLY
        and not parent_uri
        and restored_checkpoint is None
    )


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


def _extend_sample_stream_sha256(
    previous_sha256: str,
    sample_uids: Sequence[str],
) -> str:
    """Extend an order-sensitive digest with one loader microbatch."""
    if (
        len(previous_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in previous_sha256
        )
    ):
        raise ValueError("Reactive sample stream digest is invalid")
    resolved_uids = tuple(str(value) for value in sample_uids)
    if not resolved_uids or any(not value for value in resolved_uids):
        raise ValueError("Reactive sample stream UIDs are invalid")
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(previous_sha256))
    for sample_uid in resolved_uids:
        encoded_uid = sample_uid.encode("utf-8")
        digest.update(len(encoded_uid).to_bytes(8, byteorder="big"))
        digest.update(encoded_uid)
    return digest.hexdigest()


def _batch_sample_uids(raw_batch: Mapping[str, Any]) -> tuple[str, ...]:
    """Read the ordered sample identities from one collated batch."""
    raw_uids = raw_batch.get("sample_uid")
    if isinstance(raw_uids, str):
        resolved = (raw_uids,)
    elif isinstance(raw_uids, Sequence):
        resolved = tuple(str(value) for value in raw_uids)
    else:
        raise ValueError("Reactive training batch has no sample UIDs")
    batch_size = int(raw_batch["visual_tiles"].shape[0])
    if (
        len(resolved) != batch_size
        or any(not sample_uid for sample_uid in resolved)
    ):
        raise ValueError("Reactive training batch sample UIDs are invalid")
    return resolved


def _replay_loader_position(
    iterator,
    *,
    skipped_micro_steps: int,
    expected_restarts: int,
    expected_samples: int,
    expected_sample_stream_sha256: str,
) -> str:
    replayed_samples = 0
    replayed_sha256 = REACTIVE_SAMPLE_STREAM_INITIAL_SHA256
    for _ in range(skipped_micro_steps):
        raw_batch, _, _ = _loader_item(next(iterator))
        replayed_samples += int(raw_batch["visual_tiles"].shape[0])
        replayed_sha256 = _extend_sample_stream_sha256(
            replayed_sha256,
            _batch_sample_uids(raw_batch),
        )
    if (
        iterator.restarts != expected_restarts
        or replayed_samples != expected_samples
        or replayed_sha256 != expected_sample_stream_sha256
    ):
        raise ValueError(
            "Reactive resume loader position is not deterministic"
        )
    return replayed_sha256


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
    *,
    training_scope: ReactiveTrainingScope | str,
    total_optimizer_steps: int | None = None,
):
    import torch

    scope = ReactiveTrainingScope(training_scope)
    if scope is ReactiveTrainingScope.BEV_ONLY:
        if total_optimizer_steps is None or total_optimizer_steps < 2:
            raise ValueError(
                "BEV scheduler needs at least two optimizer steps"
            )
        warmup_steps = max(
            1,
            min(
                total_optimizer_steps - 1,
                math.ceil(total_optimizer_steps * 0.02),
            ),
        )
        decay_steps = total_optimizer_steps - warmup_steps
        minimum_factor = 0.1

        def learning_rate_factor(next_step_index: int) -> float:
            if next_step_index < warmup_steps:
                return (next_step_index + 1) / warmup_steps
            progress = min(
                1.0,
                (next_step_index - warmup_steps + 1) / decay_steps,
            )
            return minimum_factor + 0.5 * (
                1.0 - minimum_factor
            ) * (1.0 + math.cos(math.pi * progress))

        return (
            "bev_linear_warmup_cosine_v1",
            torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=learning_rate_factor,
            ),
        )
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


def _reactive_optimizer_parameter_groups(
    model,
    *,
    training_scope: ReactiveTrainingScope | str,
    learning_rate: float,
    bev_encoder_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Build disjoint AdamW groups with explicit BEV fine-tuning rates."""
    import torch

    scope = ReactiveTrainingScope(training_scope)
    if scope is ReactiveTrainingScope.MULTITASK:
        parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("Reactive optimizer has no trainable parameters")
        return [{
            "name": "multitask",
            "params": parameters,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }]

    base = _base_model(model)
    reactive = base.Reactive_E2E
    module_groups = (
        (
            "bev_encoder",
            (reactive.Backbone, reactive.FeatureFusion),
        ),
        (
            "bev_head",
            (reactive.BEVSegmentationHead,),
        ),
    )
    grouped: dict[tuple[str, bool], list[Any]] = {}
    assigned: set[int] = set()
    for group_name, modules in module_groups:
        for module in modules:
            if module is None:
                raise ValueError(f"{group_name} module is missing")
            embedding_parameter_ids = {
                id(parameter)
                for child in module.modules()
                if isinstance(child, torch.nn.Embedding)
                for parameter in child.parameters(recurse=False)
            }
            for parameter_name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue
                identity = id(parameter)
                if identity in assigned:
                    raise ValueError(
                        "Reactive optimizer parameter groups overlap"
                    )
                assigned.add(identity)
                use_decay = (
                    parameter.ndim > 1
                    and not parameter_name.endswith("bias")
                    and identity not in embedding_parameter_ids
                    and parameter_name.rsplit(".", 1)[-1] not in {
                        "camera_embeddings",
                        "level_embeddings",
                    }
                )
                grouped.setdefault(
                    (group_name, use_decay),
                    [],
                ).append(parameter)

    trainable = {
        id(parameter)
        for parameter in base.parameters()
        if parameter.requires_grad
    }
    if assigned != trainable:
        raise ValueError(
            "BEV-only optimizer does not cover every trainable parameter"
        )
    learning_rates = {
        "bev_encoder": bev_encoder_learning_rate,
        "bev_head": learning_rate,
    }
    parameter_groups = []
    for (group_name, use_decay), parameters in grouped.items():
        parameter_groups.append({
            "name": (
                f"{group_name}_{'decay' if use_decay else 'no_decay'}"
            ),
            "params": parameters,
            "lr": learning_rates[group_name],
            "weight_decay": weight_decay if use_decay else 0.0,
        })
    if not parameter_groups:
        raise ValueError("BEV-only optimizer has no trainable parameters")
    return parameter_groups


def _bev_checkpoint_class_guard(
    validation: Mapping[str, Any],
) -> bool:
    """Accept a BEV checkpoint only when every class is useful."""
    values = []
    for class_name in BEV_SEGMENTATION_CLASSES:
        supported_name = f"bev_{class_name}_supported"
        lower_name = (
            f"bev_{class_name}_ap_lift_bootstrap_lower_95"
        )
        prevalence_name = f"bev_{class_name}_positive_prevalence"
        best_iou_name = f"bev_{class_name}_best_iou_on_validation_set"
        precision_name = (
            f"bev_{class_name}_best_iou_precision_on_validation_set"
        )
        recall_name = (
            f"bev_{class_name}_best_iou_recall_on_validation_set"
        )
        metric_names = (
            supported_name,
            lower_name,
            prevalence_name,
            best_iou_name,
            precision_name,
            recall_name,
        )
        if any(name not in validation for name in metric_names):
            raise ValueError(
                "BEV checkpoint class guard metrics are incomplete"
            )
        supported = float(validation[supported_name])
        lower = float(validation[lower_name])
        prevalence = float(validation[prevalence_name])
        best_iou = float(validation[best_iou_name])
        precision = float(validation[precision_name])
        recall = float(validation[recall_name])
        if not all(
            math.isfinite(value)
            for value in (
                supported,
                lower,
                prevalence,
                best_iou,
                precision,
                recall,
            )
        ):
            raise ValueError(
                "BEV checkpoint class guard metrics are non-finite"
            )
        values.append(
            supported == 1.0
            and lower > 0.0
            and 0.0 < prevalence < 1.0
            and best_iou > prevalence
            and best_iou >= BEV_CHECKPOINT_MIN_CLASS_IOU
            and precision > prevalence
            and precision >= BEV_CHECKPOINT_MIN_CLASS_PRECISION
            and recall > 0.0
        )
    minimum_lift = float(validation.get("bev_min_ap_lift", float("nan")))
    if not math.isfinite(minimum_lift):
        raise ValueError("BEV checkpoint minimum AP lift is non-finite")
    return all(values) and minimum_lift > 0.0


def _weighted_bev_prior_logit_biases(
    statistics,
    pos_weights,
) -> tuple[float, ...]:
    """Return constant-logit optima for the configured weighted BCE."""
    positive_fraction_sum = np.asarray(
        statistics.positive_fraction_sum,
        dtype=np.float64,
    )
    active_sample_count = np.asarray(
        statistics.active_sample_count,
        dtype=np.float64,
    )
    weights = np.asarray(pos_weights, dtype=np.float64)
    class_count = len(BEV_SEGMENTATION_CLASSES)
    if (
        positive_fraction_sum.shape != (class_count,)
        or active_sample_count.shape != (class_count,)
        or weights.shape != (class_count,)
        or not np.isfinite(positive_fraction_sum).all()
        or not np.isfinite(active_sample_count).all()
        or not np.isfinite(weights).all()
        or np.any(positive_fraction_sum <= 0.0)
        or np.any(positive_fraction_sum >= active_sample_count)
        or np.any(weights < 1.0)
    ):
        raise ValueError("BEV prior statistics are invalid")
    prevalence = positive_fraction_sum / active_sample_count
    weighted_prevalence = (
        weights * prevalence
        / (1.0 - prevalence + weights * prevalence)
    )
    weighted_prevalence = np.clip(
        weighted_prevalence,
        1e-4,
        1.0 - 1e-4,
    )
    return tuple(
        round(float(value), 8)
        for value in np.log(
            weighted_prevalence / (1.0 - weighted_prevalence)
        )
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


def _configure_t8_temporal_normalization(
    model,
    *,
    freeze_bevformer: bool,
) -> tuple[str, int]:
    """Select the T8 normalization contract for this training run."""
    import torch.nn as nn

    feature_fusion = _base_model(model).Reactive_E2E.FeatureFusion
    if getattr(feature_fusion, "architecture", None) != "bevformer_v2_t8":
        return "not_applicable_v1", 0
    temporal_fusion = getattr(feature_fusion, "temporal_fusion", None)
    if temporal_fusion is None:
        raise ValueError("BEVFormer V2 T8 temporal fusion is missing")
    if freeze_bevformer:
        if any(
            isinstance(module, nn.SyncBatchNorm)
            for module in temporal_fusion.modules()
        ):
            raise ValueError("frozen T8 temporal fusion must not use SyncBatchNorm")
        if not any(
            type(module) is nn.BatchNorm2d
            for module in temporal_fusion.modules()
        ):
            raise ValueError("BEVFormer V2 T8 temporal fusion has no BatchNorm")
        return "frozen_pretrained_running_stats_v1", 0
    return (
        "sync_batch_norm_running_stats_v1",
        _synchronize_t8_temporal_batch_norm(model),
    )


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
    step_scheduler=None,
    start_optimizer_step: int = 0,
    resume_rank_state: Mapping[str, Any] | None = None,
    resume_rng_state: Mapping[str, Any] | None = None,
    checkpoint_interval_steps: int = 0,
    checkpoint_callback: (
        Callable[[int, Mapping[str, Any]], None] | None
    ) = None,
    performance_log_interval_steps: int = (
        REACTIVE_PERFORMANCE_LOG_INTERVAL_STEPS
    ),
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
    if performance_log_interval_steps <= 0:
        raise ValueError("performance log interval must be positive")
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
    bev_loss_module = getattr(objective, "bev_loss", None)
    bev_class_count = (
        int(bev_loss_module.pos_weight.numel())
        if bev_loss_module is not None
        and bool(getattr(objective, "compute_bev_segmentation", False))
        else 0
    )
    if resume_rank_state is None:
        bev_logit_gradient_totals = torch.zeros(
            bev_class_count,
            dtype=torch.float64,
            device=device,
        )
        bev_logit_gradient_batches = 0
    else:
        bev_logit_gradient_totals = torch.as_tensor(
            resume_rank_state["bev_logit_gradient_totals"],
            dtype=torch.float64,
            device=device,
        )
        bev_logit_gradient_batches = int(
            resume_rank_state["bev_logit_gradient_batches"]
        )
        if (
            bev_logit_gradient_totals.shape != (bev_class_count,)
            or not 0 <= bev_logit_gradient_batches <= min(
                BEV_GRADIENT_BUDGET_DIAGNOSTIC_STEPS,
                optimizer_steps,
            )
        ):
            raise ValueError(
                "Reactive resume BEV gradient diagnostics are invalid"
            )
    require_stage_a_camera_context = (
        objective.stage is ReactiveTrainingStage.NUPLAN_FULL
    )
    consumed_samples = (
        0
        if resume_rank_state is None
        else int(resume_rank_state["consumed_samples"])
    )
    track_sample_stream = (
        checkpoint_callback is not None or resume_rank_state is not None
    )
    sample_stream_sha256 = REACTIVE_SAMPLE_STREAM_INITIAL_SHA256
    if resume_rank_state is not None:
        expected_sample_stream_sha256 = str(
            resume_rank_state["sample_stream_sha256"]
        )
        if (
            len(expected_sample_stream_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sample_stream_sha256
            )
        ):
            raise ValueError(
                "Reactive resume sample stream digest is invalid"
            )
    else:
        expected_sample_stream_sha256 = (
            REACTIVE_SAMPLE_STREAM_INITIAL_SHA256
        )
    skipped_micro_steps = (
        start_optimizer_step * gradient_accumulation_steps
    )
    if skipped_micro_steps:
        assert resume_rank_state is not None
        assert resume_rng_state is not None
        if start_optimizer_step < optimizer_steps:
            sample_stream_sha256 = _replay_loader_position(
                iterator,
                skipped_micro_steps=skipped_micro_steps,
                expected_restarts=int(
                    resume_rank_state["loader_restarts"]
                ),
                expected_samples=consumed_samples,
                expected_sample_stream_sha256=(
                    expected_sample_stream_sha256
                ),
            )
        else:
            iterator.restarts = int(
                resume_rank_state["loader_restarts"]
            )
            sample_stream_sha256 = expected_sample_stream_sha256
        _restore_rng_state(resume_rng_state)
    micro_steps = optimizer_steps * gradient_accumulation_steps
    bev_gradient_diagnostic_steps = (
        _bev_gradient_diagnostic_step_indices(optimizer_steps)
        if bev_class_count
        else frozenset()
    )
    rank = dist.get_rank()
    latest_performance_metrics: dict[str, float] = {}

    def report_first_step_phase(
        phase: str,
        *,
        step_started: float,
    ) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            "Reactive first-step phase "
            f"rank={rank} phase={phase} "
            f"elapsed_seconds={time.perf_counter() - step_started:.3f}",
            flush=True,
        )

    def record_performance_phase(
        enabled: bool,
        phase_seconds: dict[str, float],
        phase: str,
        phase_started: float,
    ) -> None:
        if not enabled:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        phase_seconds[phase] += time.perf_counter() - phase_started

    for optimizer_step_index in range(
        start_optimizer_step,
        optimizer_steps,
    ):
        completed_optimizer_steps = optimizer_step_index + 1
        capture_performance = _performance_sample_due(
            completed_optimizer_steps,
            interval_steps=performance_log_interval_steps,
        )
        if capture_performance and device.type == "cuda":
            torch.cuda.synchronize(device)
        first_optimizer_step = (
            optimizer_step_index == start_optimizer_step
        )
        step_started = time.perf_counter()
        step_consumed_samples = consumed_samples
        performance_seconds = {
            name: 0.0 for name in REACTIVE_PERFORMANCE_PHASES
        }
        performance_term_totals = (
            torch.zeros(
                len(term_names),
                dtype=torch.float64,
                device=device,
            )
            if capture_performance
            else None
        )
        optimizer.zero_grad(set_to_none=True)
        finite_step = torch.ones((), dtype=torch.bool, device=device)
        for accumulation_index in range(gradient_accumulation_steps):
            phase_started = time.perf_counter()
            raw_batch, fallback_projection, fallback_geometry_type = (
                _loader_item(next(iterator))
            )
            if track_sample_stream:
                sample_stream_sha256 = _extend_sample_stream_sha256(
                    sample_stream_sha256,
                    _batch_sample_uids(raw_batch),
                )
            record_performance_phase(
                capture_performance,
                performance_seconds,
                "loader_wait",
                phase_started,
            )
            phase_started = time.perf_counter()
            batch = _batch_to_device(raw_batch, device)
            record_performance_phase(
                capture_performance,
                performance_seconds,
                "host_to_device",
                phase_started,
            )
            consumed_samples += int(batch["visual_tiles"].shape[0])
            phase_started = time.perf_counter()
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
            record_performance_phase(
                capture_performance,
                performance_seconds,
                "input_prepare",
                phase_started,
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
            phase_started = time.perf_counter()
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
                        bev_only=bool(
                            getattr(objective, "is_bev_only", False)
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
                    if (
                        bev_class_count
                        and accumulation_index == 0
                        and optimizer_step_index
                        in bev_gradient_diagnostic_steps
                    ):
                        bev_logits = auxiliary.get(
                            "bev_segmentation_logits"
                        )
                        if not torch.is_tensor(bev_logits):
                            raise RuntimeError(
                                "BEV gradient diagnostics require logits"
                            )
                        bev_logit_gradient = torch.autograd.grad(
                            terms["bev_segmentation"],
                            bev_logits,
                            retain_graph=True,
                            create_graph=False,
                        )[0]
                        class_gradient = (
                            bev_logit_gradient.detach()
                            .abs()
                            .sum(dim=(0, 2, 3))
                            .to(torch.float64)
                        )
                        if (
                            class_gradient.shape != (bev_class_count,)
                            or not bool(torch.isfinite(class_gradient).all())
                        ):
                            finite_step.fill_(False)
                        else:
                            bev_logit_gradient_totals += class_gradient
                            bev_logit_gradient_batches += 1
                record_performance_phase(
                    capture_performance,
                    performance_seconds,
                    "forward",
                    phase_started,
                )
                if first_optimizer_step and accumulation_index == 0:
                    report_first_step_phase(
                        "forward_complete",
                        step_started=step_started,
                    )
                phase_started = time.perf_counter()
                scaled_loss.backward()
                record_performance_phase(
                    capture_performance,
                    performance_seconds,
                    "backward",
                    phase_started,
                )
                if first_optimizer_step and accumulation_index == 0:
                    report_first_step_phase(
                        "backward_complete",
                        step_started=step_started,
                    )
            totals += torch.stack([
                terms[name].detach().to(torch.float64)
                for name in term_names
            ])
            if capture_performance:
                assert performance_term_totals is not None
                performance_term_totals += torch.stack([
                    terms[name].detach().to(torch.float64)
                    for name in term_names
                ])

        phase_started = time.perf_counter()
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
        record_performance_phase(
            capture_performance,
            performance_seconds,
            "gradient",
            phase_started,
        )
        if first_optimizer_step:
            report_first_step_phase(
                "gradient_check_complete",
                step_started=step_started,
            )
        phase_started = time.perf_counter()
        if not _collective_true(finite_step, device):
            raise FloatingPointError(
                "a Reactive DDP rank produced non-finite loss or gradients"
            )
        record_performance_phase(
            capture_performance,
            performance_seconds,
            "finite_collective",
            phase_started,
        )
        if first_optimizer_step:
            report_first_step_phase(
                "finite_collective_complete",
                step_started=step_started,
            )
        phase_started = time.perf_counter()
        optimizer.step()
        if step_scheduler is not None:
            step_scheduler.step()
        record_performance_phase(
            capture_performance,
            performance_seconds,
            "optimizer",
            phase_started,
        )
        if first_optimizer_step:
            report_first_step_phase(
                "optimizer_complete",
                step_started=step_started,
            )
        if capture_performance:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            performance_seconds["step"] = (
                time.perf_counter() - step_started
            )
            latest_performance_metrics = (
                _aggregate_reactive_performance(
                    performance_seconds,
                    local_samples=(
                        consumed_samples - step_consumed_samples
                    ),
                    device=device,
                )
            )
            assert performance_term_totals is not None
            dist.all_reduce(
                performance_term_totals,
                op=dist.ReduceOp.SUM,
            )
            loss_denominator = (
                dist.get_world_size() * gradient_accumulation_steps
            )
            latest_performance_metrics.update({
                f"performance_loss_{name}": float(
                    performance_term_totals[index].item()
                    / loss_denominator
                )
                for index, name in enumerate(term_names)
            })
            if rank == 0:
                print(
                    "Reactive performance "
                    + json.dumps(
                        {
                            "completed_optimizer_steps": (
                                completed_optimizer_steps
                            ),
                            "version": (
                                REACTIVE_PERFORMANCE_LOG_VERSION
                            ),
                            **latest_performance_metrics,
                        },
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
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
                    "bev_logit_gradient_batches": (
                        bev_logit_gradient_batches
                    ),
                    "bev_logit_gradient_totals": (
                        bev_logit_gradient_totals.detach().cpu().tolist()
                    ),
                    "gradient_totals": (
                        gradient_totals.detach().cpu().tolist()
                    ),
                    "loader_restarts": iterator.restarts,
                    "performance_metrics": dict(
                        latest_performance_metrics
                    ),
                    "rank": dist.get_rank(),
                    "sample_stream_sha256": sample_stream_sha256,
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
    if bev_class_count:
        dist.all_reduce(
            bev_logit_gradient_totals,
            op=dist.ReduceOp.SUM,
        )
        gradient_budget_total = bev_logit_gradient_totals.sum()
        if (
            bev_logit_gradient_batches <= 0
            or not bool(torch.isfinite(gradient_budget_total))
            or float(gradient_budget_total.item()) <= 0.0
        ):
            raise FloatingPointError(
                "BEV logit gradient budget is empty or non-finite"
            )
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
    if bev_class_count:
        global_diagnostic_batches = (
            dist.get_world_size() * bev_logit_gradient_batches
        )
        metrics["bev_logit_gradient_diagnostic_batches"] = float(
            global_diagnostic_batches
        )
        for class_index in range(bev_class_count):
            class_total = bev_logit_gradient_totals[class_index]
            metrics[
                f"bev_logit_gradient_l1_class_{class_index}"
            ] = float(
                class_total.item() / global_diagnostic_batches
            )
            metrics[
                f"bev_logit_gradient_share_class_{class_index}"
            ] = float(
                (class_total / gradient_budget_total).item()
            )
    metrics.update(latest_performance_metrics)
    return metrics


def _histogram_average_precision(
    positive_histogram,
    negative_histogram,
) -> float:
    import torch

    if (
        positive_histogram.ndim != 1
        or negative_histogram.shape != positive_histogram.shape
        or positive_histogram.numel() <= 1
    ):
        raise ValueError("BEV score histograms must be matching 1D tensors")
    positive_histogram = positive_histogram.to(torch.float64)
    negative_histogram = negative_histogram.to(torch.float64)
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


def _histogram_best_iou_operating_point(
    positive_histogram,
    negative_histogram,
) -> tuple[float, float, float, float]:
    """Return threshold, IoU, precision, and recall from score histograms."""
    import torch

    if (
        positive_histogram.ndim != 1
        or negative_histogram.shape != positive_histogram.shape
        or positive_histogram.numel() <= 1
    ):
        raise ValueError("BEV score histograms must be matching 1D tensors")
    positive_histogram = positive_histogram.to(torch.float64)
    negative_histogram = negative_histogram.to(torch.float64)
    positive_total = positive_histogram.sum()
    if float(positive_total.item()) <= 0.0:
        raise ValueError("BEV validation class has no positive cells")
    true_positive = torch.cumsum(positive_histogram.flip(0), dim=0)
    false_positive = torch.cumsum(negative_histogram.flip(0), dim=0)
    false_negative = positive_total - true_positive
    iou = true_positive / (
        true_positive + false_positive + false_negative
    ).clamp_min(1.0)
    best_reversed_index = int(iou.argmax().item())
    threshold_bin = (
        positive_histogram.numel() - 1 - best_reversed_index
    )
    selected_true_positive = true_positive[best_reversed_index]
    selected_false_positive = false_positive[best_reversed_index]
    selected_false_negative = false_negative[best_reversed_index]
    precision = selected_true_positive / (
        selected_true_positive + selected_false_positive
    ).clamp_min(1.0)
    recall = selected_true_positive / (
        selected_true_positive + selected_false_negative
    ).clamp_min(1.0)
    return (
        threshold_bin / positive_histogram.numel(),
        float(iou[best_reversed_index].item()),
        float(precision.item()),
        float(recall.item()),
    )


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


def _reactive_dataset_split_metrics(
    *,
    total_samples: int,
    train_split_sample_count: int,
    validation_evaluated_sample_count: int,
    configured_validation_fraction: float,
) -> dict[str, float | int]:
    if total_samples <= 0:
        raise ValueError("dataset split requires positive total samples")
    validation_pool_sample_count = (
        total_samples - train_split_sample_count
    )
    if (
        train_split_sample_count <= 0
        or validation_pool_sample_count <= 0
    ):
        raise ValueError("dataset split must contain train and validation")
    if not 0.0 < configured_validation_fraction < 1.0:
        raise ValueError("configured validation fraction must be in (0, 1)")
    if not (
        0
        <= validation_evaluated_sample_count
        <= validation_pool_sample_count
    ):
        raise ValueError(
            "validation subset exceeds the validation split"
        )
    return {
        "dataset_total_samples": total_samples,
        "train_split_sample_count": train_split_sample_count,
        "validation_pool_sample_count": validation_pool_sample_count,
        "configured_train_fraction": (
            1.0 - configured_validation_fraction
        ),
        "configured_validation_fraction": (
            configured_validation_fraction
        ),
        "actual_train_fraction": (
            train_split_sample_count / total_samples
        ),
        "actual_validation_fraction": (
            validation_pool_sample_count / total_samples
        ),
        "validation_evaluated_sample_count": (
            validation_evaluated_sample_count
        ),
        "validation_evaluated_fraction_of_pool": (
            validation_evaluated_sample_count
            / validation_pool_sample_count
        ),
    }


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


def _accumulate_bev_validation_statistics(
    auxiliary,
    batch,
    objective,
    *,
    device,
    class_count: int,
    probability_bins: int,
    bev_counts,
    positive_histogram,
    negative_histogram,
    lane_range_counts,
    lane_range_positive_histogram,
    lane_range_negative_histogram,
    lane_range_masks,
    lane_class_index: int,
    bootstrap_positive_histogram,
    bootstrap_negative_histogram,
):
    """Accumulate BEV metrics independently of trajectory completeness."""
    import torch

    bev_logits = auxiliary.get("bev_segmentation_logits")
    if not torch.is_tensor(bev_logits):
        raise ValueError("Stage A validation omitted BEV logits")
    target = batch["bev_segmentation_target"].to(
        device=device,
        dtype=torch.float32,
    )
    valid_mask = batch["bev_segmentation_valid"].to(
        device=device,
        dtype=torch.bool,
    )
    if target.shape != bev_logits.shape or valid_mask.shape != bev_logits.shape:
        raise ValueError("BEV validation target shape differs")
    bev_components = objective.bev_loss.components(
        bev_logits,
        target,
        valid_mask,
    )
    relevant_nonfinite = valid_mask & (
        ~torch.isfinite(bev_logits) | ~torch.isfinite(target)
    )
    if bool(relevant_nonfinite.any()):
        raise FloatingPointError(
            "BEV validation has non-finite values in valid cells"
        )
    probability = bev_logits.float().sigmoid()
    binary_target = target >= 0.5
    binary_prediction = probability >= 0.5
    sample_uids = batch.get("sample_uid")
    if (
        not isinstance(sample_uids, (list, tuple))
        or len(sample_uids) != target.shape[0]
        or any(
            not isinstance(sample_uid, str) or not sample_uid
            for sample_uid in sample_uids
        )
    ):
        raise ValueError(
            "BEV validation bootstrap requires one sample UID per item"
        )
    bootstrap_weights = fixed_point_bayesian_bootstrap_weights(
        sample_uids,
        device=device,
    )
    bootstrap_probability = torch.where(
        valid_mask,
        probability,
        torch.zeros_like(probability),
    )
    bootstrap_bins = torch.clamp(
        (
            bootstrap_probability * BEV_AP_BOOTSTRAP_BINS
        ).to(torch.int64),
        min=0,
        max=BEV_AP_BOOTSTRAP_BINS - 1,
    ).flatten(2)
    bootstrap_valid = valid_mask.flatten(2)
    bootstrap_target = binary_target.flatten(2)
    per_sample_positive_histogram = torch.zeros(
        (
            target.shape[0],
            class_count,
            BEV_AP_BOOTSTRAP_BINS,
        ),
        dtype=torch.int64,
        device=device,
    )
    per_sample_negative_histogram = torch.zeros_like(
        per_sample_positive_histogram
    )
    per_sample_positive_histogram.scatter_add_(
        2,
        bootstrap_bins,
        (bootstrap_valid & bootstrap_target).to(torch.int64),
    )
    per_sample_negative_histogram.scatter_add_(
        2,
        bootstrap_bins,
        (bootstrap_valid & ~bootstrap_target).to(torch.int64),
    )
    accumulate_fixed_point_bootstrap_histogram(
        bootstrap_positive_histogram,
        bootstrap_weights,
        per_sample_positive_histogram,
    )
    accumulate_fixed_point_bootstrap_histogram(
        bootstrap_negative_histogram,
        bootstrap_weights,
        per_sample_negative_histogram,
    )
    for class_index in range(class_count):
        class_valid = valid_mask[:, class_index]
        if not bool(class_valid.any()):
            continue
        class_target = binary_target[:, class_index][class_valid]
        class_prediction = binary_prediction[:, class_index][class_valid]
        class_probability = probability[:, class_index][class_valid]
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
            (class_probability * probability_bins).to(torch.int64),
            min=0,
            max=probability_bins - 1,
        )
        positive_histogram[class_index] += torch.bincount(
            bins[class_target],
            minlength=probability_bins,
        )
        negative_histogram[class_index] += torch.bincount(
            bins[~class_target],
            minlength=probability_bins,
        )

    if lane_range_masks is None:
        lane_range_masks = _bev_lane_range_masks(
            target.shape[-2],
            target.shape[-1],
            device=device,
        )
    elif lane_range_masks.shape[-2:] != target.shape[-2:]:
        raise ValueError("BEV validation lane range shape changed")
    for range_index in range(2):
        lane_valid = (
            valid_mask[:, lane_class_index]
            & lane_range_masks[range_index][None]
        )
        if not bool(lane_valid.any()):
            continue
        lane_target = binary_target[:, lane_class_index][lane_valid]
        lane_prediction = binary_prediction[:, lane_class_index][lane_valid]
        lane_probability = probability[:, lane_class_index][lane_valid]
        lane_range_counts[range_index, 0] += (
            lane_prediction & lane_target
        ).sum()
        lane_range_counts[range_index, 1] += (
            lane_prediction & ~lane_target
        ).sum()
        lane_range_counts[range_index, 2] += (
            ~lane_prediction & lane_target
        ).sum()
        lane_range_counts[range_index, 3] += lane_target.sum()
        lane_range_counts[range_index, 4] += lane_valid.sum()
        bins = torch.clamp(
            (lane_probability * probability_bins).to(torch.int64),
            max=probability_bins - 1,
        )
        lane_range_positive_histogram[range_index] += torch.bincount(
            bins[lane_target],
            minlength=probability_bins,
        )
        lane_range_negative_histogram[range_index] += torch.bincount(
            bins[~lane_target],
            minlength=probability_bins,
        )
    return (
        float(bev_components["total"].item()),
        float(bev_components["bce"].item()),
        float(bev_components["dice"].item()),
        lane_range_masks,
    )


def _evaluate_global_reactive(
    model,
    loader,
    objective,
    *,
    stage: ReactiveTrainingStage,
    device,
    probability_bins: int,
    ade_scale_m: float,
    precision: str = "fp32",
    expected_sample_count: int | None = None,
    expected_sample_uid_sha256: str | None = None,
) -> dict[str, Any]:
    import torch

    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_CLASSES,
    )
    from evaluation.reactive_open_loop import (
        OPEN_LOOP_STATISTIC_NAMES,
        open_loop_metrics_from_statistics,
        reactive_open_loop_statistics,
    )
    from training.losses.control_rollout import integrate_controls_torch
    from training.reactive_stage_runner import (
        resolve_reactive_batch_projection,
        resolve_reactive_camera_history,
        resolve_reactive_front_projection,
    )

    base = _base_model(model)
    if precision not in {"bf16", "fp32"}:
        raise ValueError("Reactive validation precision is unsupported")
    if (expected_sample_count is None) != (
        expected_sample_uid_sha256 is None
    ):
        raise ValueError(
            "Reactive validation sample count and digest must be paired"
        )
    if expected_sample_uid_sha256 is not None and (
        len(expected_sample_uid_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_sample_uid_sha256
        )
    ):
        raise ValueError("Reactive validation sample digest is invalid")
    was_training = base.training
    is_bev_only = bool(getattr(objective, "is_bev_only", False))
    if (
        stage is ReactiveTrainingStage.NUPLAN_FULL
        and expected_sample_count is not None
    ):
        validate_fixed_point_bootstrap_capacity(
            expected_sample_count,
            maximum_cells_per_sample=(
                AUTOE2E_NAVIGATION_GEOMETRY.height_px
                * AUTOE2E_NAVIGATION_GEOMETRY.width_px
            ),
        )
    base.eval()
    ade_sum = 0.0
    fde_sum = 0.0
    sample_count = 0
    class_count = len(BEV_SEGMENTATION_CLASSES)
    bev_counts = torch.zeros(
        (class_count, 5),
        dtype=torch.int64,
        device=device,
    )
    positive_histogram = torch.zeros(
        (class_count, probability_bins),
        dtype=torch.float64,
        device=device,
    )
    negative_histogram = torch.zeros_like(positive_histogram)
    bootstrap_positive_histogram = torch.zeros(
        (
            BEV_AP_BOOTSTRAP_REPLICATES,
            class_count,
            BEV_AP_BOOTSTRAP_BINS,
        ),
        dtype=torch.int64,
        device=device,
    )
    bootstrap_negative_histogram = torch.zeros_like(
        bootstrap_positive_histogram
    )
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
    bev_loss_samples = 0
    seen_validation_sample_uids: set[str] = set()
    local_bev_contract_errors: list[str] = []
    route_values = torch.zeros(20, dtype=torch.float64, device=device)
    open_loop_values = torch.zeros(
        len(OPEN_LOOP_STATISTIC_NAMES),
        dtype=torch.float64,
        device=device,
    )
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
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16",
                ):
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
                        bev_only=is_bev_only,
                    )
                if isinstance(output, tuple):
                    controls, auxiliary = output
                else:
                    controls = output
                    auxiliary = {}
                if stage is ReactiveTrainingStage.NUPLAN_FULL:
                    try:
                        batch_sample_uids = batch.get("sample_uid")
                        if (
                            not isinstance(batch_sample_uids, (list, tuple))
                            or len(batch_sample_uids)
                            != int(batch["visual_tiles"].shape[0])
                            or any(
                                not isinstance(sample_uid, str)
                                or not sample_uid
                                for sample_uid in batch_sample_uids
                            )
                        ):
                            raise ValueError(
                                "BEV validation requires one sample UID "
                                "per item"
                            )
                        duplicate_uids = (
                            seen_validation_sample_uids
                            & set(batch_sample_uids)
                        )
                        if (
                            len(set(batch_sample_uids))
                            != len(batch_sample_uids)
                            or duplicate_uids
                        ):
                            raise ValueError(
                                "BEV validation sample UIDs contain "
                                f"duplicates: {sorted(duplicate_uids)}"
                            )
                        seen_validation_sample_uids.update(batch_sample_uids)
                        (
                            batch_bev_loss,
                            batch_bev_bce,
                            batch_bev_dice,
                            lane_range_masks,
                        ) = _accumulate_bev_validation_statistics(
                            auxiliary,
                            batch,
                            objective,
                            device=device,
                            class_count=class_count,
                            probability_bins=probability_bins,
                            bev_counts=bev_counts,
                            positive_histogram=positive_histogram,
                            negative_histogram=negative_histogram,
                            lane_range_counts=lane_range_counts,
                            lane_range_positive_histogram=(
                                lane_range_positive_histogram
                            ),
                            lane_range_negative_histogram=(
                                lane_range_negative_histogram
                            ),
                            lane_range_masks=lane_range_masks,
                            lane_class_index=lane_class_index,
                            bootstrap_positive_histogram=(
                                bootstrap_positive_histogram
                            ),
                            bootstrap_negative_histogram=(
                                bootstrap_negative_histogram
                            ),
                        )
                        bev_batch_size = int(
                            batch["visual_tiles"].shape[0]
                        )
                        bev_loss_sum += batch_bev_loss * bev_batch_size
                        bev_bce_sum += batch_bev_bce * bev_batch_size
                        bev_dice_sum += batch_bev_dice * bev_batch_size
                        bev_loss_samples += bev_batch_size
                    except (
                        FloatingPointError,
                        OverflowError,
                        ValueError,
                    ) as error:
                        if len(local_bev_contract_errors) < 8:
                            local_bev_contract_errors.append(
                                f"{type(error).__name__}: {error}"
                            )
                if is_bev_only:
                    continue
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
                (
                    predicted_xy,
                    predicted_headings,
                    predicted_speeds,
                ) = integrate_controls_torch(
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
                open_loop_values += reactive_open_loop_statistics(
                    safe_controls,
                    predicted_xy,
                    predicted_headings,
                    predicted_speeds,
                    target_xy,
                    complete,
                    batch["map_context"],
                    batch["map_valid"],
                    batch["route_mask"],
                    batch["route_valid"],
                )
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

    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        _raise_distributed_validation_contract_errors(
            local_bev_contract_errors
        )
    values = torch.tensor(
        [ade_sum, fde_sum, float(sample_count)],
        dtype=torch.float64,
        device=device,
    )
    bev_loss_values = torch.tensor(
        [
            bev_loss_sum,
            bev_bce_sum,
            bev_dice_sum,
            float(bev_loss_samples),
        ],
        dtype=torch.float64,
        device=device,
    )
    verify_sample_coverage = (
        stage is ReactiveTrainingStage.NUPLAN_FULL
        and expected_sample_count is not None
    )
    _reduce_reactive_validation_state(
        (
            values,
            bev_counts,
            positive_histogram,
            negative_histogram,
            bootstrap_positive_histogram,
            bootstrap_negative_histogram,
            lane_range_counts,
            lane_range_positive_histogram,
            lane_range_negative_histogram,
            route_values,
            open_loop_values,
            bev_loss_values,
        ),
        local_sample_uids=(
            sorted(seen_validation_sample_uids)
            if verify_sample_coverage
            else None
        ),
        expected_sample_count=(
            expected_sample_count if verify_sample_coverage else None
        ),
        expected_sample_uid_sha256=(
            expected_sample_uid_sha256
            if verify_sample_coverage
            else None
        ),
    )
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
    metrics: dict[str, Any] = {
        "evaluation_precision": precision,
    }
    trajectory_quality = 0.0
    if not is_bev_only:
        global_count = int(values[2].item())
        if global_count <= 0:
            raise ValueError(
                "Reactive distributed validation has no complete trajectories"
            )
        metrics.update({
            "ade_6p4s_m": float(values[0].item() / global_count),
            "fde_6p4s_m": float(values[1].item() / global_count),
            "complete_samples": float(global_count),
        })
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
        metrics.update(open_loop_metrics_from_statistics(open_loop_values))
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
        best_iou_threshold = 0.5
        best_iou = 0.0
        best_iou_precision = 0.0
        best_iou_recall = 0.0
        ap_lift_bootstrap_lower_95 = 0.0
        ap_lift_bootstrap_upper_95 = 0.0
        if supported:
            average_precision = _histogram_average_precision(
                positive_histogram[class_index],
                negative_histogram[class_index],
            )
            (
                best_iou_threshold,
                best_iou,
                best_iou_precision,
                best_iou_recall,
            ) = _histogram_best_iou_operating_point(
                positive_histogram[class_index],
                negative_histogram[class_index],
            )
            if prevalence < 1.0:
                ap_lift = (
                    (average_precision - prevalence)
                    / (1.0 - prevalence)
                )
                bootstrap_lifts = []
                for replicate_index in range(
                    BEV_AP_BOOTSTRAP_REPLICATES
                ):
                    replicate_positive_histogram = (
                        bootstrap_positive_histogram[
                            replicate_index,
                            class_index,
                        ]
                    )
                    replicate_negative_histogram = (
                        bootstrap_negative_histogram[
                            replicate_index,
                            class_index,
                        ]
                    )
                    replicate_positive = float(
                        replicate_positive_histogram.sum().item()
                    )
                    replicate_valid = replicate_positive + float(
                        replicate_negative_histogram.sum().item()
                    )
                    if (
                        replicate_positive <= 0.0
                        or replicate_valid <= replicate_positive
                    ):
                        continue
                    replicate_prevalence = (
                        replicate_positive / replicate_valid
                    )
                    replicate_ap = _histogram_average_precision(
                        replicate_positive_histogram,
                        replicate_negative_histogram,
                    )
                    bootstrap_lifts.append(
                        (replicate_ap - replicate_prevalence)
                        / (1.0 - replicate_prevalence)
                    )
                if len(bootstrap_lifts) < (
                    BEV_AP_BOOTSTRAP_REPLICATES * 0.9
                ):
                    raise ValueError(
                        "BEV validation bootstrap has insufficient class "
                        f"support for {class_name}"
                    )
                bootstrap_tensor = torch.as_tensor(
                    bootstrap_lifts,
                    dtype=torch.float64,
                    device=device,
                )
                ap_lift_bootstrap_lower_95 = float(
                    torch.quantile(bootstrap_tensor, 0.025).item()
                )
                ap_lift_bootstrap_upper_95 = float(
                    torch.quantile(bootstrap_tensor, 0.975).item()
                )
        class_support[class_index] = supported
        ap_lifts[class_index] = ap_lift
        average_precisions[class_index] = average_precision
        prefix = f"bev_{class_name}"
        fixed_iou = _metric_ratio(
            true_positive,
            true_positive + false_positive + false_negative,
        )
        fixed_precision = _metric_ratio(
            true_positive,
            true_positive + false_positive,
        )
        fixed_recall = _metric_ratio(
            true_positive,
            true_positive + false_negative,
        )
        metrics[f"{prefix}_iou_at_0p5"] = fixed_iou
        metrics[f"{prefix}_precision_at_0p5"] = fixed_precision
        metrics[f"{prefix}_recall_at_0p5"] = fixed_recall
        metrics[f"{prefix}_average_precision"] = average_precision
        metrics[f"{prefix}_ap_lift"] = ap_lift
        metrics[
            f"{prefix}_ap_lift_bootstrap_lower_95"
        ] = ap_lift_bootstrap_lower_95
        metrics[
            f"{prefix}_ap_lift_bootstrap_upper_95"
        ] = ap_lift_bootstrap_upper_95
        metrics[
            f"{prefix}_best_iou_threshold_on_validation_set"
        ] = best_iou_threshold
        metrics[f"{prefix}_best_iou_on_validation_set"] = best_iou
        metrics[
            f"{prefix}_best_iou_precision_on_validation_set"
        ] = best_iou_precision
        metrics[
            f"{prefix}_best_iou_recall_on_validation_set"
        ] = best_iou_recall
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
    supported_ap_lifts = [
        value
        for value, supported in zip(ap_lifts, class_support)
        if supported
    ]
    if not supported_ap_lifts:
        raise ValueError("BEV validation supports no segmentation classes")
    metrics["bev_min_ap_lift"] = min(supported_ap_lifts)
    metrics["bev_supported_class_count"] = float(sum(class_support))
    metrics["bev_all_classes_supported"] = float(all(class_support))
    metrics["bev_threshold_selection"] = "same_validation_set_oracle"
    metrics["bev_ap_bootstrap_version"] = BEV_AP_BOOTSTRAP_VERSION
    metrics["bev_ap_bootstrap_replicates"] = float(
        BEV_AP_BOOTSTRAP_REPLICATES
    )
    metrics["bev_ap_bootstrap_bins"] = float(BEV_AP_BOOTSTRAP_BINS)
    metrics["bev_ap_bootstrap_weight_scale"] = float(
        BEV_AP_BOOTSTRAP_WEIGHT_SCALE
    )
    metrics["bev_ap_bootstrap_max_weight"] = (
        BEV_AP_BOOTSTRAP_MAX_WEIGHT
    )
    if is_bev_only:
        if not all(class_support):
            raise ValueError(
                "BEV-only validation must support every segmentation class"
            )
        metrics["selection_score"] = (
            0.5 * static_macro_ap_lift
            + 0.5 * dynamic_macro_ap_lift
        )
    else:
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
    checkpoint_single_worker_smoke = bool(
        config.get("single_worker_smoke", False)
    )
    expected_single_worker_smoke = bool(
        expected.get("single_worker_smoke", False)
    )
    if checkpoint_single_worker_smoke != expected_single_worker_smoke:
        raise ValueError(
            "Reactive DDP resume checkpoint smoke provenance differs"
        )
    training_state = payload.get("training_state") or {}
    checkpoint_kind = str(
        training_state.get("checkpoint_kind", "epoch")
    )
    checkpoint_epoch = int(payload.get("epoch", -1))
    if checkpoint_epoch < 1:
        raise ValueError("Reactive resume checkpoint epoch is invalid")
    epoch_boundary_batch_change = (
        checkpoint_kind == "epoch"
        and config.get("distributed_global_batch")
        != expected.get("distributed_global_batch")
    )
    fixed_step_scheduler = (
        expected.get("scheduler_identity")
        == "bev_linear_warmup_cosine_v1"
    )
    allowed_mismatches = set(
        {
            "distributed_global_batch",
            "optimizer_steps_per_epoch",
        }
        if epoch_boundary_batch_change and not fixed_step_scheduler
        else set()
    )
    requested_epochs = int(expected.get("epochs", 0))
    if (
        checkpoint_kind == "epoch"
        and not fixed_step_scheduler
        and requested_epochs > checkpoint_epoch
        and config.get("epochs") != requested_epochs
    ):
        allowed_mismatches.add("epochs")
    legacy_frozen_bev_epoch_resume = (
        checkpoint_kind == "epoch"
        and config.get("training_scope") is None
        and bool(config.get("freeze_bevformer", False))
        and float(config.get("bev_weight", -1.0)) == 0.0
        and expected.get("training_scope")
        == ReactiveTrainingScope.MULTITASK.value
        and bool(expected.get("freeze_bevformer", False))
        and float(expected.get("bev_weight", -1.0)) == 0.0
        and expected.get("optimizer_identity") == "adamw_v1"
        and int(expected.get("num_loader_workers", -1)) == 2
        and int(expected.get("shuffle_buffer", -1)) == 256
    )
    if legacy_frozen_bev_epoch_resume:
        allowed_mismatches.update(
            LEGACY_FROZEN_BEV_EPOCH_RESUME_FIELDS
        )
    mismatches = {}
    for name, value in expected.items():
        checkpoint_value = (
            checkpoint_single_worker_smoke
            if name == "single_worker_smoke"
            else config.get(name)
        )
        if (
            checkpoint_value != value
            and name not in allowed_mismatches
        ):
            mismatches[name] = (checkpoint_value, value)
    if mismatches:
        raise ValueError(
            f"Reactive DDP resume contract differs: {mismatches}"
        )
    if epoch_boundary_batch_change and not fixed_step_scheduler:
        print(
            "Reactive epoch-boundary batch change "
            + json.dumps(
                {
                    name: {
                        "checkpoint": config.get(name),
                        "requested": expected.get(name),
                    }
                    for name in sorted(allowed_mismatches)
                },
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    _base_model(model).load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    history_path = Path(checkpoint_directory) / "history.json"
    history = json.loads(history_path.read_text(encoding="ascii"))
    if (
        not isinstance(history, list)
        or any(not isinstance(item, dict) for item in history)
    ):
        raise ValueError("Reactive DDP resume checkpoint has invalid history")
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
        "bev_logit_gradient_batches",
        "bev_logit_gradient_totals",
        "consumed_samples",
        "gradient_totals",
        "loader_restarts",
        "rank",
        "sample_stream_sha256",
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
        bev_rank_full_microbatch_capacity,
        derive_bev_gradient_budget_weights,
        derive_bev_positive_pair_frequencies,
        derive_bev_pos_weights,
        derive_bev_rank_importance_scale,
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
    from model_components.losses import (
        BEV_SEGMENTATION_AUXILIARY_LOSS_VERSION,
    )
    from training.reactive_stage_runner import (
        load_stage_a_parent,
        save_reactive_checkpoint,
    )

    validate_reactive_stage_config(config)
    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    single_worker_smoke = bool(
        config.get("allow_single_worker_smoke", False)
    )
    bounded_bev_canary = bool(
        config.get("allow_bounded_bev_canary", False)
    )
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
    training_scope = ReactiveTrainingScope(
        str(
            config.get(
                "training_scope",
                ReactiveTrainingScope.MULTITASK.value,
            )
        )
    )
    bev_encoder_learning_rate = float(
        config.get("bev_encoder_learning_rate", 1e-5)
    )
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
    local_bev_records_by_directory = (
        tuple(
            discover_bev_sample_statistics((directory,))
            for directory in local_directories
        )
        if stage is ReactiveTrainingStage.NUPLAN_FULL
        else None
    )
    local_bev_records = (
        tuple(
            record
            for records in local_bev_records_by_directory
            for record in records
        )
        if local_bev_records_by_directory is not None
        else None
    )
    validation_sample_uids: tuple[str, ...] | None = None
    validation_sample_uid_sha256 = ""
    validation_sample_count = 0
    validation_positive_sample_counts: tuple[int, ...] | None = None
    local_validation_limit = math.ceil(
        int(config.get("validation_sample_limit", 1024))
        / world_size
    )
    local_validation_selection_errors: list[str] = []
    try:
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
    except ValueError as error:
        validation_sample_uids = ()
        local_validation_selection_errors.append(
            f"{type(error).__name__}: {error}"
        )
    _raise_distributed_validation_contract_errors(
        local_validation_selection_errors
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
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        assert local_bev_records is not None
        assert validation_sample_uids is not None
        local_validation_positive_counts = torch.tensor(
            _bev_validation_positive_sample_counts(
                local_bev_records,
                validation_sample_uids,
            ),
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(
            local_validation_positive_counts,
            op=dist.ReduceOp.SUM,
        )
        validation_positive_sample_counts = tuple(
            int(value)
            for value in local_validation_positive_counts.tolist()
        )
        if (
            training_scope is ReactiveTrainingScope.BEV_ONLY
            and any(
                value <= 0
                for value in validation_positive_sample_counts
            )
        ):
            missing_classes = [
                class_name
                for class_name, count in zip(
                    BEV_SEGMENTATION_CLASSES,
                    validation_positive_sample_counts,
                )
                if count <= 0
            ]
            raise ValueError(
                "BEV-only validation subset lacks positive samples for: "
                f"{missing_classes}"
            )
    bev_pos_weights: tuple[float, ...] = (1.0,) * 8
    bev_class_weights: tuple[float, ...] = (1.0,) * 8
    bev_positive_pair_frequencies: tuple[float, ...] = (1.0,) * 8
    bev_repeat_factors: tuple[int, ...] = (1,) * 8
    bev_repeat_policy = None
    raw_bev_statistics = None
    effective_bev_statistics = None
    local_effective_bev_statistics = None
    bev_rank_importance_scale = 1.0
    dataset_split_metrics: dict[str, float | int] = {}
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
        if float(config["bev_weight"]) > 0.0:
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
            if training_scope is ReactiveTrainingScope.BEV_ONLY:
                bev_repeat_factors = derive_bev_repeat_factors(
                    raw_bev_statistics,
                    frequency_threshold=float(
                        config["bev_repeat_frequency_threshold"]
                    ),
                    max_repeat=int(config["bev_max_repeat"]),
                )
                local_effective_bev_statistics = (
                    summarize_bev_training_statistics(
                        local_bev_records,
                        val_fraction=float(config["val_fraction"]),
                        repeat_factors=bev_repeat_factors,
                    )
                )
                effective_bev_statistics = _all_reduce_bev_statistics(
                    local_effective_bev_statistics,
                    device,
                )
                bev_rank_importance_scale = (
                    derive_bev_rank_importance_scale(
                        local_effective_exposure_count=(
                            local_effective_bev_statistics
                            .effective_exposure_count
                        ),
                        global_sample_count=raw_bev_statistics.sample_count,
                        world_size=world_size,
                    )
                )
                bev_repeat_policy = BEVClassRepeatPolicy(
                    repeat_factors=bev_repeat_factors,
                    importance_scale=bev_rank_importance_scale,
                )
            else:
                effective_bev_statistics = raw_bev_statistics
            bev_pos_weights = tuple(round(value, 6) for value in (
                derive_bev_pos_weights(
                    raw_bev_statistics,
                    max_weight=float(config["bev_pos_weight_cap"]),
                )
            ))
            bev_class_weights = tuple(round(value, 6) for value in (
                derive_bev_gradient_budget_weights(
                    raw_bev_statistics,
                    bev_pos_weights,
                )
            ))
            bev_positive_pair_frequencies = (
                derive_bev_positive_pair_frequencies(
                    raw_bev_statistics,
                )
            )
        dataset_split_metrics = _reactive_dataset_split_metrics(
            total_samples=plan.total_samples,
            train_split_sample_count=raw_bev_statistics.sample_count,
            validation_evaluated_sample_count=validation_sample_count,
            configured_validation_fraction=float(
                config["val_fraction"]
            ),
        )
    bev_head_initialization: dict[str, object] | None = None
    if (
        training_scope is ReactiveTrainingScope.BEV_ONLY
        and raw_bev_statistics is not None
    ):
        class_logit_bias = _weighted_bev_prior_logit_biases(
            raw_bev_statistics,
            bev_pos_weights,
        )
        bev_head_initialization = {
            "class_logit_bias": list(class_logit_bias),
            "classifier_weight_std": BEV_HEAD_CLASSIFIER_WEIGHT_STD,
            "version": BEV_HEAD_INITIALIZATION_VERSION,
        }

    seed = int(config["training_seed"])
    _seed_epoch(seed, 0, 0)
    constructor_kwargs = reactive_model_kwargs(
        stage,
        num_views=plan.num_views,
    )
    parent_uri = str(config.get("parent_checkpoint_uri") or "")
    restored, resume_uri, ignored_explicit_resume = (
        _resolve_resume_sources(
            train.get_checkpoint(),
            str(config.get("resume_checkpoint_uri") or ""),
        )
    )
    if ignored_explicit_resume and rank == 0:
        print(
            "Using the Ray-managed recovery checkpoint; the explicit "
            "resume URI only seeds a run before Ray recovery state exists.",
            flush=True,
        )
    if restored is None and resume_uri:
        resume_directory = cache_root / "explicit-resume"
        resume_directory.mkdir(parents=True, exist_ok=True)
        for filename in ("checkpoint.pt", "history.json"):
            _download_checkpoint(
                f"{resume_uri}/{filename}",
                resume_directory / filename,
            )
        restored = Checkpoint.from_directory(str(resume_directory))
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
                required_training_scope=_required_parent_training_scope(stage),
            )
        )
        inherited_initialization = lineage.get(
            "bevformer_v2_initialization"
        )
        if inherited_initialization is not None:
            initialization_metadata = dict(inherited_initialization)
    if _should_initialize_bev_head_from_training_statistics(
        training_scope,
        parent_uri=parent_uri,
        restored_checkpoint=restored,
    ):
        if bev_head_initialization is None:
            raise ValueError("BEV-only head initialization is missing")
        classifier_weight_std = bev_head_initialization[
            "classifier_weight_std"
        ]
        if (
            isinstance(classifier_weight_std, bool)
            or not isinstance(classifier_weight_std, (int, float))
        ):
            raise ValueError(
                "BEV head classifier weight standard deviation is invalid"
            )
        model.Reactive_E2E.BEVSegmentationHead.initialize_output_bias(
            bev_head_initialization["class_logit_bias"],
            classifier_weight_std=float(classifier_weight_std),
        )
    configure_model_for_stage(
        model,
        stage,
        freeze_bevformer=bool(config["freeze_bevformer"]),
        train_bev_head=float(config["bev_weight"]) > 0.0,
        training_scope=training_scope,
    )
    freeze_bevformer = bool(config["freeze_bevformer"])
    (
        temporal_normalization_identity,
        synchronized_temporal_batch_norm_count,
    ) = _configure_t8_temporal_normalization(
        model,
        freeze_bevformer=freeze_bevformer,
    )
    objective = ReactiveMultitaskObjective(
        stage,
        bev_pos_weight=bev_pos_weights,
        bev_class_weight=bev_class_weights,
        bev_positive_pair_frequency=bev_positive_pair_frequencies,
        trajectory_weight=float(config["trajectory_weight"]),
        bev_weight=float(config["bev_weight"]),
        route_weight=float(config["route_weight"]),
        corridor_pos_weight=float(config["corridor_pos_weight"]),
        training_scope=training_scope,
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
        _reactive_optimizer_parameter_groups(
            model,
            training_scope=training_scope,
            learning_rate=float(config["learning_rate"]),
            bev_encoder_learning_rate=bev_encoder_learning_rate,
            weight_decay=float(config["weight_decay"]),
        ),
    )
    local_train_microbatch_capacity = 0
    bev_optimizer_step_capacity = 0
    if (
        training_scope is ReactiveTrainingScope.BEV_ONLY
        and effective_bev_statistics is not None
    ):
        assert local_bev_records_by_directory is not None
        local_train_microbatch_capacity = (
            bev_rank_full_microbatch_capacity(
                local_bev_records_by_directory,
                val_fraction=float(config["val_fraction"]),
                repeat_factors=bev_repeat_factors,
                batch_size=int(config["per_rank_batch_size"]),
            )
        )
        capacity_tensor = torch.tensor(
            local_train_microbatch_capacity,
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(capacity_tensor, op=dist.ReduceOp.MIN)
        minimum_rank_microbatch_capacity = int(capacity_tensor.item())
        bev_optimizer_step_capacity = (
            minimum_rank_microbatch_capacity
            // int(config["gradient_accumulation_steps"])
        )
        if bev_optimizer_step_capacity <= 0:
            raise ValueError(
                "BEV-only rank-local loaders cannot supply one optimizer step"
            )
        calculated_steps = bev_optimizer_step_capacity
    else:
        calculated_steps = optimizer_steps_per_epoch(
            total_samples=plan.total_samples,
            val_fraction=float(config["val_fraction"]),
            world_size=world_size,
            per_rank_batch_size=int(config["per_rank_batch_size"]),
            gradient_accumulation_steps=int(
                config["gradient_accumulation_steps"]
            ),
        )
    configured_steps = int(config["steps_per_epoch"])
    if (
        training_scope is ReactiveTrainingScope.BEV_ONLY
        and configured_steps > bev_optimizer_step_capacity
    ):
        raise ValueError(
            "configured BEV-only steps exceed rank-local loader capacity"
        )
    optimizer_steps = configured_steps or calculated_steps
    consumed_microbatch_capacity = (
        optimizer_steps * int(config["gradient_accumulation_steps"])
    )
    rank_sampling_evidence: list[dict[str, Any] | None] = [
        None
    ] * world_size
    if training_scope is ReactiveTrainingScope.BEV_ONLY:
        assert local_effective_bev_statistics is not None
        batch_size = int(config["per_rank_batch_size"])
        total_exposure_count = (
            local_effective_bev_statistics.effective_exposure_count
        )
        retained_sample_capacity = (
            local_train_microbatch_capacity * batch_size
        )
        consumed_sample_capacity = (
            consumed_microbatch_capacity * batch_size
        )
        if not (
            0
            < consumed_sample_capacity
            <= retained_sample_capacity
            <= total_exposure_count
        ):
            raise RuntimeError("BEV rank sample capacity is invalid")
        local_drop_last_fraction = (
            total_exposure_count - retained_sample_capacity
        ) / total_exposure_count
        local_optimizer_tail_fraction = (
            retained_sample_capacity - consumed_sample_capacity
        ) / retained_sample_capacity
        local_truncation_fraction = (
            total_exposure_count - consumed_sample_capacity
        ) / total_exposure_count
        if not (
            0.0 <= local_drop_last_fraction <= 1.0
            and 0.0 <= local_optimizer_tail_fraction <= 1.0
            and 0.0 <= local_truncation_fraction <= 1.0
        ):
            raise RuntimeError("BEV rank exclusion fraction is invalid")
        if (
            local_truncation_fraction + 1e-12
            < local_drop_last_fraction
        ):
            raise RuntimeError("BEV rank exclusion accounting is invalid")
        dist.all_gather_object(
            rank_sampling_evidence,
            {
                "consumed_microbatch_capacity": (
                    consumed_microbatch_capacity
                ),
                "consumed_sample_capacity": consumed_sample_capacity,
                "drop_last_fraction": local_drop_last_fraction,
                "importance_scale": bev_rank_importance_scale,
                "microbatch_capacity": local_train_microbatch_capacity,
                "optimizer_tail_fraction": (
                    local_optimizer_tail_fraction
                ),
                "rank": rank,
                "retained_sample_capacity": retained_sample_capacity,
                "single_worker_smoke": single_worker_smoke,
                "bounded_bev_canary": bounded_bev_canary,
                "total_exposure_count": total_exposure_count,
                "truncation_guard_enforced": (
                    not single_worker_smoke
                    and not bounded_bev_canary
                ),
                "truncation_fraction": local_truncation_fraction,
            },
        )
        if any(
            not isinstance(evidence, Mapping)
            for evidence in rank_sampling_evidence
        ):
            raise RuntimeError("BEV rank sampling evidence is incomplete")
        maximum_truncation_fraction = max(
            float(evidence["truncation_fraction"])
            for evidence in rank_sampling_evidence
            if isinstance(evidence, Mapping)
        )
        _validate_bev_rank_truncation(
            maximum_truncation_fraction,
            single_worker_smoke=single_worker_smoke,
            bounded_bev_canary=bounded_bev_canary,
        )
    else:
        rank_sampling_evidence = []
    scheduler_identity, scheduler = _build_reactive_scheduler(
        optimizer,
        training_scope=training_scope,
        total_optimizer_steps=(
            int(config["epochs"]) * optimizer_steps
            if training_scope is ReactiveTrainingScope.BEV_ONLY
            else None
        ),
    )
    checkpoint_interval_steps = int(config["checkpoint_interval_steps"])
    performance_log_interval_steps = int(
        config.get(
            "performance_log_interval_steps",
            REACTIVE_PERFORMANCE_LOG_INTERVAL_STEPS,
        )
    )
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
        "single_worker_smoke": single_worker_smoke,
        "bounded_bev_canary": bounded_bev_canary,
        "gradient_clip_max_norm": float(config["grad_clip"]),
        "gradient_clip_mode": "branch_v1",
        "epochs": int(config["epochs"]),
        "optimizer_steps_per_epoch": optimizer_steps,
        "route_metrics_version": ROUTE_VALIDATION_METRICS_VERSION,
        "step_checkpoint_version": REACTIVE_STEP_CHECKPOINT_VERSION,
        "sample_stream_digest_version": (
            REACTIVE_SAMPLE_STREAM_DIGEST_VERSION
        ),
        "num_loader_workers": int(config["num_loader_workers"]),
        "shuffle_buffer": int(config["shuffle_buffer"]),
        "trajectory_weight": float(config["trajectory_weight"]),
        "bev_weight": float(config["bev_weight"]),
        "route_weight": float(config["route_weight"]),
        "training_scope": training_scope.value,
        "bev_encoder_learning_rate": bev_encoder_learning_rate,
        "optimizer_identity": (
            "bev_discriminative_adamw_v1"
            if training_scope is ReactiveTrainingScope.BEV_ONLY
            else "adamw_v1"
        ),
        "corridor_pos_weight": float(config["corridor_pos_weight"]),
        "training_seed": seed,
        "scheduler_identity": scheduler_identity,
        "temporal_normalization_identity": (
            temporal_normalization_identity
        ),
        "synchronized_temporal_batch_norm_count": (
            synchronized_temporal_batch_norm_count
        ),
        "freeze_bevformer": bool(config["freeze_bevformer"]),
        "training_stage": stage.value,
        "bev_pos_weights": list(bev_pos_weights),
        "bev_class_weights": list(bev_class_weights),
        "bev_positive_pair_frequencies": list(
            bev_positive_pair_frequencies
        ),
        "bev_repeat_factors": list(bev_repeat_factors),
        "bev_sampling_importance_correction": (
            BEV_SAMPLING_IMPORTANCE_CORRECTION_VERSION
        ),
        "bev_rank_sampling_evidence": rank_sampling_evidence,
        "bev_taxonomy_version": BEV_SEGMENTATION_TAXONOMY_VERSION,
        "bev_loss_version": BEV_SEGMENTATION_AUXILIARY_LOSS_VERSION,
        "bev_checkpoint_quality_guard_version": (
            BEV_CHECKPOINT_QUALITY_GUARD_VERSION
        ),
        "bev_checkpoint_min_class_iou": (
            BEV_CHECKPOINT_MIN_CLASS_IOU
        ),
        "bev_checkpoint_min_class_precision": (
            BEV_CHECKPOINT_MIN_CLASS_PRECISION
        ),
        "bev_head_initialization": bev_head_initialization,
        "bev_ap_bins": int(config["bev_ap_bins"]),
        "validation_sample_count": validation_sample_count,
        "validation_sample_uid_sha256": validation_sample_uid_sha256,
        "validation_fraction": float(config["val_fraction"]),
        "validation_sample_limit": int(
            config.get("validation_sample_limit", 1024)
        ),
        "validation_positive_sample_counts": (
            list(validation_positive_sample_counts)
            if validation_positive_sample_counts is not None
            else None
        ),
        "dataset_split_metrics": dataset_split_metrics,
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
                "stage_a_training_scope",
                "stage_a_weight_transfer_scope",
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
    if rank == 0:
        base_model = _base_model(model)
        gradient_groups = reactive_gradient_parameter_groups(model)
        device_properties = torch.cuda.get_device_properties(device)
        print(
            "Reactive runtime "
            + json.dumps(
                {
                    "cuda_device_name": device_properties.name,
                    "cuda_total_memory_bytes": (
                        device_properties.total_memory
                    ),
                    "freeze_bevformer": freeze_bevformer,
                    "training_scope": training_scope.value,
                    "global_batch": global_batch,
                    "gradient_accumulation_steps": int(
                        config["gradient_accumulation_steps"]
                    ),
                    "loader_workers_per_rank": int(
                        config["num_loader_workers"]
                    ),
                    "optimizer_steps_per_epoch": optimizer_steps,
                    "optimizer_parameter_groups": [
                        {
                            "learning_rate": float(group["lr"]),
                            "name": str(group.get("name", "")),
                            "parameter_count": sum(
                                parameter.numel()
                                for parameter in group["params"]
                            ),
                            "weight_decay": float(
                                group["weight_decay"]
                            ),
                        }
                        for group in optimizer.param_groups
                    ],
                    "parameter_count": sum(
                        parameter.numel()
                        for parameter in base_model.parameters()
                    ),
                    "performance_log_interval_steps": (
                        performance_log_interval_steps
                    ),
                    "performance_log_version": (
                        REACTIVE_PERFORMANCE_LOG_VERSION
                    ),
                    "per_rank_batch_size": int(
                        config["per_rank_batch_size"]
                    ),
                    "precision": str(config["precision"]),
                    "trainable_parameter_count": sum(
                        parameter.numel()
                        for parameter in base_model.parameters()
                        if parameter.requires_grad
                    ),
                    "trainable_parameter_count_by_group": {
                        name: sum(
                            parameter.numel()
                            for parameter in parameters
                        )
                        for name, parameters in gradient_groups.items()
                    },
                    "world_size": world_size,
                },
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
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
        "single_worker_smoke": single_worker_smoke,
        "bounded_bev_canary": bounded_bev_canary,
        "capacity_block_end_utc": str(
            config.get("capacity_block_end_utc") or ""
        ),
        "gradient_clip_max_norm": float(config["grad_clip"]),
        "gradient_clip_mode": "branch_v1",
        "epochs": int(config["epochs"]),
        "optimizer_steps_per_epoch": optimizer_steps,
        "performance_log_interval_steps": (
            performance_log_interval_steps
        ),
        "performance_log_version": REACTIVE_PERFORMANCE_LOG_VERSION,
        "route_metrics_version": ROUTE_VALIDATION_METRICS_VERSION,
        "step_checkpoint_version": REACTIVE_STEP_CHECKPOINT_VERSION,
        "sample_stream_digest_version": (
            REACTIVE_SAMPLE_STREAM_DIGEST_VERSION
        ),
        "num_loader_workers": int(config["num_loader_workers"]),
        "shuffle_buffer": int(config["shuffle_buffer"]),
        "trajectory_weight": float(config["trajectory_weight"]),
        "bev_weight": float(config["bev_weight"]),
        "route_weight": float(config["route_weight"]),
        "training_scope": training_scope.value,
        "bev_encoder_learning_rate": bev_encoder_learning_rate,
        "optimizer_identity": (
            "bev_discriminative_adamw_v1"
            if training_scope is ReactiveTrainingScope.BEV_ONLY
            else "adamw_v1"
        ),
        "corridor_pos_weight": float(config["corridor_pos_weight"]),
        "training_seed": seed,
        "scheduler_identity": scheduler_identity,
        "temporal_normalization_identity": (
            temporal_normalization_identity
        ),
        "synchronized_temporal_batch_norm_count": (
            synchronized_temporal_batch_norm_count
        ),
        "freeze_bevformer": bool(config["freeze_bevformer"]),
        "bev_pos_weights": list(bev_pos_weights),
        "bev_class_weights": list(bev_class_weights),
        "bev_positive_pair_frequencies": list(
            bev_positive_pair_frequencies
        ),
        "bev_repeat_factors": list(bev_repeat_factors),
        "bev_sampling_importance_correction": (
            BEV_SAMPLING_IMPORTANCE_CORRECTION_VERSION
        ),
        "bev_rank_sampling_evidence": rank_sampling_evidence,
        "bev_taxonomy_version": BEV_SEGMENTATION_TAXONOMY_VERSION,
        "bev_loss_version": BEV_SEGMENTATION_AUXILIARY_LOSS_VERSION,
        "bev_checkpoint_quality_guard_version": (
            BEV_CHECKPOINT_QUALITY_GUARD_VERSION
        ),
        "bev_checkpoint_min_class_iou": (
            BEV_CHECKPOINT_MIN_CLASS_IOU
        ),
        "bev_checkpoint_min_class_precision": (
            BEV_CHECKPOINT_MIN_CLASS_PRECISION
        ),
        "bev_head_initialization": bev_head_initialization,
        "bev_ap_bins": int(config["bev_ap_bins"]),
        "validation_sample_count": validation_sample_count,
        "validation_sample_uid_sha256": validation_sample_uid_sha256,
        "validation_fraction": float(config["val_fraction"]),
        "validation_sample_limit": int(
            config.get("validation_sample_limit", 1024)
        ),
        "validation_positive_sample_counts": (
            list(validation_positive_sample_counts)
            if validation_positive_sample_counts is not None
            else None
        ),
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
            drop_last=True,
            nodesplitter=passthrough_nodesplitter,
        )

        def report_step_checkpoint(
            completed_optimizer_steps: int,
            local_train_state: Mapping[str, Any],
        ) -> None:
            checkpoint_started = time.perf_counter()
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
            performance_payloads = [
                state.get("performance_metrics")
                for state in rank_train_states
            ]
            available_performance = [
                dict(payload)
                for payload in performance_payloads
                if isinstance(payload, Mapping) and payload
            ]
            if available_performance:
                if len(available_performance) != world_size:
                    raise RuntimeError(
                        "Reactive performance metrics are incomplete"
                    )
                canonical_performance = {
                    json.dumps(
                        payload,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for payload in available_performance
                }
                if len(canonical_performance) != 1:
                    raise RuntimeError(
                        "Reactive performance metrics differ across ranks"
                    )
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
                "performance_log_version": (
                    REACTIVE_PERFORMANCE_LOG_VERSION
                ),
                "step_checkpoint_version": (
                    REACTIVE_STEP_CHECKPOINT_VERSION
                ),
                "world_size": world_size,
            }
            if available_performance:
                step_metrics.update(available_performance[0])
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
                checkpoint_timing: list[dict[str, float] | None] = [
                    (
                        {
                            "checkpoint_bytes": float(
                                (
                                    Path(checkpoint_directory)
                                    / "checkpoint.pt"
                                ).stat().st_size
                            ),
                            "checkpoint_prepare_seconds": (
                                time.perf_counter()
                                - checkpoint_started
                            ),
                        }
                        if rank == 0
                        else None
                    )
                ]
                dist.broadcast_object_list(checkpoint_timing, src=0)
                if checkpoint_timing[0] is not None:
                    step_metrics.update(checkpoint_timing[0])
                report_started = time.perf_counter()
                train.report(step_metrics, checkpoint=checkpoint)
                if rank == 0:
                    print(
                        "Reactive checkpoint performance "
                        + json.dumps(
                            {
                                "completed_optimizer_steps": (
                                    completed_optimizer_steps
                                ),
                                "report_seconds": (
                                    time.perf_counter()
                                    - report_started
                                ),
                                "total_seconds": (
                                    time.perf_counter()
                                    - checkpoint_started
                                ),
                                **(
                                    checkpoint_timing[0]
                                    if checkpoint_timing[0] is not None
                                    else {}
                                ),
                            },
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )

        epoch_train_started = time.perf_counter()
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
            step_scheduler=(
                scheduler
                if training_scope is ReactiveTrainingScope.BEV_ONLY
                else None
            ),
            start_optimizer_step=start_optimizer_step,
            resume_rank_state=resume_rank_train_state,
            resume_rng_state=resume_rank_rng_state,
            checkpoint_interval_steps=checkpoint_interval_steps,
            checkpoint_callback=report_step_checkpoint,
            performance_log_interval_steps=(
                performance_log_interval_steps
            ),
        )
        if (
            objective.is_bev_only
            and float(train_metrics["loader_restarts"]) != 0.0
        ):
            raise RuntimeError(
                "BEV-only training exhausted a rank-local loader"
            )
        train_epoch_seconds = _distributed_max_seconds(
            time.perf_counter() - epoch_train_started,
            device,
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
        validation_started = time.perf_counter()
        validation = _evaluate_global_reactive(
            model,
            validation_loader,
            objective,
            stage=stage,
            device=device,
            probability_bins=int(config["bev_ap_bins"]),
            ade_scale_m=float(config["selection_ade_scale_m"]),
            precision=str(config["precision"]),
            expected_sample_count=validation_sample_count,
            expected_sample_uid_sha256=validation_sample_uid_sha256,
        )
        validation_seconds = _distributed_max_seconds(
            time.perf_counter() - validation_started,
            device,
        )
        if training_scope is not ReactiveTrainingScope.BEV_ONLY:
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
                "full_microbatch_capacity": (
                    local_train_microbatch_capacity
                ),
                "sampling_importance_scale": (
                    bev_rank_importance_scale
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
        ade_within_guard = (
            True
            if objective.is_bev_only
            else validation["ade_6p4s_m"] <= (
                best_ade
                + float(config["selection_ade_regression_margin_m"])
            )
        )
        bev_class_guard_pass = (
            _bev_checkpoint_class_guard(validation)
            if objective.is_bev_only
            else True
        )
        checkpoint_quality_guard_enforced = (
            _should_enforce_bev_checkpoint_quality_guard(
                single_worker_smoke=single_worker_smoke,
                bounded_bev_canary=bounded_bev_canary,
            )
        )
        checkpoint_quality_eligible = (
            bev_class_guard_pass
            or not checkpoint_quality_guard_enforced
        )
        is_best = (
            validation["selection_score"] > best_selection_score
            and ade_within_guard
            and checkpoint_quality_eligible
        )
        checkpoint_selection_score = (
            validation["selection_score"]
            if ade_within_guard and checkpoint_quality_eligible
            else -1.0
        )
        checkpoint_retention_score = (
            EPOCH_CHECKPOINT_RETENTION_SCORE_BASE
            + epoch
        )
        if is_best:
            best_selection_score = validation["selection_score"]
        if not objective.is_bev_only:
            best_ade = min(best_ade, validation["ade_6p4s_m"])
        diagnostic_metrics: dict[str, float | int] = {
            f"{CAMERA_FEATURE_SCALE_WEIGHT_METRIC_PREFIX}{index}": weight
            for index, weight in enumerate(feature_scale_weights)
        }
        diagnostic_metrics["synchronized_temporal_batch_norm_count"] = (
            synchronized_temporal_batch_norm_count
        )
        diagnostic_metrics["temporal_normalization_frozen_pretrained"] = int(
            temporal_normalization_identity
            == "frozen_pretrained_running_stats_v1"
        )
        diagnostic_metrics["bev_checkpoint_class_guard_pass"] = int(
            bev_class_guard_pass
        )
        diagnostic_metrics["checkpoint_quality_guard_enforced"] = int(
            checkpoint_quality_guard_enforced
        )
        diagnostic_metrics["single_worker_smoke"] = int(
            single_worker_smoke
        )
        diagnostic_metrics["bounded_bev_canary"] = int(
            bounded_bev_canary
        )
        diagnostic_metrics["bev_parent_promotion_eligible"] = int(
            objective.is_bev_only
            and checkpoint_quality_guard_enforced
            and bev_class_guard_pass
            and not single_worker_smoke
            and not bounded_bev_canary
        )
        if rank_sampling_evidence:
            sampling_evidence = [
                evidence
                for evidence in rank_sampling_evidence
                if isinstance(evidence, Mapping)
            ]
            diagnostic_metrics[
                "bev_rank_min_full_microbatch_capacity"
            ] = min(
                int(evidence["microbatch_capacity"])
                for evidence in sampling_evidence
            )
            diagnostic_metrics[
                "bev_rank_max_full_microbatch_capacity"
            ] = max(
                int(evidence["microbatch_capacity"])
                for evidence in sampling_evidence
            )
            diagnostic_metrics[
                "bev_rank_min_importance_scale"
            ] = min(
                float(evidence["importance_scale"])
                for evidence in sampling_evidence
            )
            diagnostic_metrics[
                "bev_rank_max_importance_scale"
            ] = max(
                float(evidence["importance_scale"])
                for evidence in sampling_evidence
            )
            diagnostic_metrics[
                "bev_rank_min_drop_last_fraction"
            ] = min(
                float(evidence["drop_last_fraction"])
                for evidence in sampling_evidence
            )
            diagnostic_metrics[
                "bev_rank_max_drop_last_fraction"
            ] = max(
                float(evidence["drop_last_fraction"])
                for evidence in sampling_evidence
            )
            diagnostic_metrics[
                "bev_rank_min_optimizer_tail_fraction"
            ] = min(
                float(evidence["optimizer_tail_fraction"])
                for evidence in sampling_evidence
            )
            diagnostic_metrics[
                "bev_rank_max_optimizer_tail_fraction"
            ] = max(
                float(evidence["optimizer_tail_fraction"])
                for evidence in sampling_evidence
            )
            diagnostic_metrics[
                "bev_rank_min_truncation_fraction"
            ] = min(
                float(evidence["truncation_fraction"])
                for evidence in sampling_evidence
            )
            diagnostic_metrics[
                "bev_rank_max_truncation_fraction"
            ] = max(
                float(evidence["truncation_fraction"])
                for evidence in sampling_evidence
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
        checkpoint_metrics.update(dataset_split_metrics)
        checkpoint_metrics["executed_optimizer_steps"] = (
            executed_optimizer_steps
        )
        checkpoint_metrics["train_epoch_seconds"] = (
            train_epoch_seconds
        )
        checkpoint_metrics["validation_seconds"] = validation_seconds
        checkpoint_metrics["performance_log_version"] = (
            REACTIVE_PERFORMANCE_LOG_VERSION
        )
        checkpoint_metrics.update({
            f"train_{name}": value
            for name, value in train_metrics.items()
            if name.startswith("performance_")
        })
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
                **dataset_split_metrics,
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
                "learning_rate": max(
                    float(group["lr"])
                    for group in optimizer.param_groups
                ),
                "maximum_parameter_delta": maximum_delta,
                "optimizer_steps_per_epoch": optimizer_steps,
                "performance_log_version": (
                    REACTIVE_PERFORMANCE_LOG_VERSION
                ),
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
                "train_epoch_seconds": train_epoch_seconds,
                "validation_selection_score": validation[
                    "selection_score"
                ],
                "validation_sample_count": validation_sample_count,
                "validation_sample_uid_sha256": (
                    validation_sample_uid_sha256
                ),
                "validation_seconds": validation_seconds,
                "training_seed": seed,
                "trajectory_weight": float(
                    config["trajectory_weight"]
                ),
                "world_size": world_size,
            }
            if not objective.is_bev_only:
                metrics.update({
                    "validation_ade_6p4s_m": validation["ade_6p4s_m"],
                    "validation_complete_samples": validation[
                        "complete_samples"
                    ],
                    "validation_fde_6p4s_m": validation["fde_6p4s_m"],
                })
            for group in optimizer.param_groups:
                group_name = str(group.get("name", ""))
                if group_name:
                    metrics[
                        f"learning_rate_{group_name}"
                    ] = float(group["lr"])
            metrics.update({
                f"train_{name}": value
                for name, value in train_metrics.items()
                if name.startswith("performance_")
            })
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
                metrics[f"bev_class_weight_{class_index}"] = (
                    bev_class_weights[class_index]
                )
                metrics[
                    f"bev_positive_pair_frequency_{class_index}"
                ] = bev_positive_pair_frequencies[class_index]
                metrics[f"bev_repeat_factor_{class_index}"] = (
                    bev_repeat_factors[class_index]
                )
                if objective.compute_bev_segmentation:
                    metrics[
                        f"train_bev_logit_gradient_l1_class_{class_index}"
                    ] = train_metrics[
                        f"bev_logit_gradient_l1_class_{class_index}"
                    ]
                    metrics[
                        f"train_bev_logit_gradient_share_class_{class_index}"
                    ] = train_metrics[
                        f"bev_logit_gradient_share_class_{class_index}"
                    ]
            if objective.compute_bev_segmentation:
                metrics[
                    "train_bev_logit_gradient_diagnostic_batches"
                ] = train_metrics[
                    "bev_logit_gradient_diagnostic_batches"
                ]
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
    """Select the best checkpoint accepted by the active training scope."""
    if result.checkpoint is None:
        raise RuntimeError("Reactive Ray training returned no checkpoint")

    entries = [
        _result_checkpoint_entry(entry)
        for entry in getattr(result, "best_checkpoints", ()) or ()
    ]
    candidates = []
    for checkpoint, metrics in entries:
        if str(metrics.get("checkpoint_kind", "epoch")) != "epoch":
            continue
        if int(metrics.get("is_best", 0)) != 1:
            continue
        score = float(metrics.get("checkpoint_selection_score", -1.0))
        single_worker_smoke = (
            int(metrics.get("single_worker_smoke", 0)) == 1
        )
        if (
            not math.isfinite(score)
            or (score < 0.0 and not single_worker_smoke)
        ):
            continue
        candidates.append((
            score,
            int(metrics.get("epoch", 0)),
            checkpoint,
            metrics,
        ))
    if candidates:
        _, _, checkpoint, metrics = max(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        return checkpoint, metrics

    evaluation_candidates = []
    for checkpoint, metrics in entries:
        if str(metrics.get("checkpoint_kind", "")) != "epoch":
            continue
        if int(metrics.get("single_worker_smoke", 0)) != 0:
            continue
        if int(metrics.get("bounded_bev_canary", 0)) != 0:
            continue
        if int(metrics.get("checkpoint_quality_guard_enforced", 0)) != 1:
            continue
        if int(metrics.get("bev_checkpoint_class_guard_pass", 0)) != 0:
            continue
        score = float(
            metrics.get("validation_selection_score", float("nan"))
        )
        if not math.isfinite(score):
            continue
        evaluation_candidates.append((
            score,
            int(metrics.get("epoch", 0)),
            checkpoint,
            metrics,
        ))
    if not evaluation_candidates:
        raise RuntimeError(
            "Reactive Ray training retained no accepted best checkpoint "
            "or production BEV checkpoint for evaluation"
        )
    _, _, checkpoint, metrics = max(
        evaluation_candidates,
        key=lambda item: (item[0], item[1]),
    )
    selected_metrics = dict(metrics)
    selected_metrics["selected_for_evaluation_only"] = 1
    selected_metrics["bev_parent_promotion_eligible"] = 0
    return checkpoint, selected_metrics


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
            timeout_s=REACTIVE_DDP_TIMEOUT_SECONDS,
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
