"""Distributed Reactive dataset and fixed-step contracts."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import random
import re
import tarfile
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import distributed_training.ray_torch_backend as ray_torch_backend
import distributed_training.reactive_stage as reactive_stage_module
from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS
from data_parsing.pre_extracted import (
    BEVClassRepeatPolicy,
    BEVSampleStatistics,
    BEVTrainingStatistics,
    bev_rank_full_microbatch_capacity,
    derive_bev_gradient_budget_weights,
    derive_bev_positive_pair_frequencies,
    derive_bev_pos_weights,
    derive_bev_rank_importance_scale,
    derive_bev_repeat_factors,
    discover_bev_sample_statistics,
    discover_bev_training_statistics,
    make_pre_extracted_loader,
    passthrough_nodesplitter,
    select_bev_validation_holdout_sample_uids,
    select_distributed_bev_validation_sample_uids,
)
from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_CLASSES,
    BEV_SEGMENTATION_STATS_MEMBER,
    BEV_SEGMENTATION_TAXONOMY_VERSION,
    encode_bev_segmentation_stats,
)
from distributed_training.reactive_canary_data import (
    write_reactive_canary_dataset,
)
from distributed_training.reactive_data import (
    RestartingIterator,
    assign_reactive_shards,
    build_reactive_dataset_plan,
    optimizer_steps_per_epoch,
    reactive_assignment_sha256,
    stage_rank_reactive_shards,
)
from distributed_training.reactive_stage import (
    BEV_AP_BOOTSTRAP_MAX_WEIGHT,
    BEV_AP_BOOTSTRAP_VERSION,
    BEV_AP_BOOTSTRAP_WEIGHT_SCALE,
    BEV_LANE_NEAR_RADIUS_M,
    REACTIVE_DDP_TIMEOUT_SECONDS,
    REACTIVE_PERFORMANCE_LOG_INTERVAL_STEPS,
    REACTIVE_PERFORMANCE_LOG_VERSION,
    REACTIVE_SAMPLE_STREAM_DIGEST_VERSION,
    REACTIVE_SAMPLE_STREAM_INITIAL_SHA256,
    REACTIVE_STEP_CHECKPOINT_VERSION,
    ReactiveResumeState,
    expected_reactive_hostname_count,
    _all_reduce_bev_statistics,
    _aggregate_reactive_performance,
    _bev_lane_range_masks,
    _bev_checkpoint_class_guard,
    _bev_validation_positive_sample_counts,
    _build_reactive_scheduler,
    _camera_feature_scale_weights,
    _capture_rng_state,
    _checkpoint_history,
    _checkpoint_step_due,
    _configure_t8_temporal_normalization,
    _histogram_average_precision,
    _histogram_best_iou_operating_point,
    _evaluate_global_reactive,
    _load_resume_checkpoint,
    _model_state_sha256,
    _parse_nvidia_smi_csv,
    _performance_sample_due,
    _rank_resume_value,
    _rank_resume_value_for_epoch,
    _reactive_dataset_split_metrics,
    _reactive_optimizer_parameter_groups,
    _extend_sample_stream_sha256,
    _reduce_reactive_validation_state,
    _raise_distributed_validation_contract_errors,
    _required_parent_training_scope,
    _weighted_bev_prior_logit_biases,
    _replay_loader_position,
    _resolve_resume_sources,
    _resume_completed_requested_epochs,
    _restore_rng_state,
    _route_validation_statistics,
    _select_result_checkpoint,
    _should_initialize_bev_head_from_training_statistics,
    _should_enforce_bev_checkpoint_quality_guard,
    _synchronize_gradient_micro_step,
    _train_fixed_steps,
    _validate_bev_rank_truncation,
    _validate_checkpoint_interval,
    clip_finite_gradients_float64,
    normalize_ray_checkpoint_uri,
    reactive_gradient_parameter_groups,
    run_reactive_stage,
    train_loop_per_worker,
    validate_reactive_stage_config,
)
from model_components.losses import (
    BEVSegmentationAuxiliaryLoss,
    RouteReconstructionLoss,
)
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_FRAME_OFFSETS,
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_INDEX,
)
from training.dataset_policy import L2D_DATASET_NAME
from training.reactive_multitask import (
    ReactiveMultitaskObjective,
    ReactiveTrainingScope,
    ReactiveTrainingStage,
)


@pytest.mark.parametrize(
    ("stage", "expected"),
    (
        (ReactiveTrainingStage.NUPLAN_FULL, "bev_only"),
        (ReactiveTrainingStage.L2D_CONTINUATION, "multitask"),
    ),
)
def test_continuation_stage_requires_reviewed_parent_scope(stage, expected):
    assert _required_parent_training_scope(stage) == expected


def test_bev_taxonomy_version_is_single_sourced():
    root = Path(__file__).parents[2]
    definition = (
        root
        / "Model"
        / "data_processing"
        / "reactive_training_artifacts.py"
    )
    offenders = []
    version_literal = re.compile(
        r"""["']bev_segmentation_""" + r"""v\d+["']"""
    )
    for directory in ("Model", "Platform"):
        for path in (root / directory).rglob("*.py"):
            if path == definition:
                continue
            if version_literal.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


def _write_source(
    root,
    *,
    dataset: str,
    shard_counts: list[int],
    include_bev: bool,
    num_views: int,
):
    root.mkdir()
    names = []
    hashes = {}
    counts = {}
    for index, sample_count in enumerate(shard_counts):
        name = f"part-{index:03d}.tar"
        payload = f"{root.name}:{index}:{sample_count}".encode()
        (root / name).write_bytes(payload)
        names.append(name)
        hashes[name] = hashlib.sha256(payload).hexdigest()
        counts[name] = sample_count
    manifest = {
        "bev_statistics_count": (
            sum(shard_counts) if include_bev else 0
        ),
        "bev_taxonomy_version": (
            BEV_SEGMENTATION_TAXONOMY_VERSION if include_bev else None
        ),
        "camera_order": [
            f"camera_{index}" for index in range(num_views)
        ],
        "camera_slots": list(CANONICAL_SIX_CAMERA_SLOTS),
        "dataset": dataset,
        "has_bev_segmentation": include_bev,
        "has_reactive_navigation": True,
        "has_route_reconstruction": True,
        "has_trajectory_xy": True,
        "image_size": REACTIVE_CAMERA_IMAGE_SIZE,
        "map_context_channels": 14,
        "navigation_geometry": (
            AUTOE2E_NAVIGATION_GEOMETRY.contract()
        ),
        "num_views": num_views,
        "partition_id": root.name,
        "route_channels": 2,
        "shard_names": names,
        "shard_sample_counts": counts,
        "shard_sha256": hashes,
        "source_revision": "test-revision",
        "total_samples": sum(shard_counts),
    }
    if include_bev:
        manifest.update({
            "front_camera_image_size": (
                REACTIVE_FRONT_CAMERA_IMAGE_SIZE
            ),
            "front_camera_index": REACTIVE_FRONT_CAMERA_INDEX,
            "temporal_frame_interval_us": (
                REACTIVE_BEVFORMER_FRAME_INTERVAL_US
            ),
            "temporal_frame_offsets": list(
                REACTIVE_BEVFORMER_FRAME_OFFSETS
            ),
        })
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="ascii",
    )
    return manifest


def test_dataset_plan_and_assignment_are_deterministic(tmp_path):
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    _write_source(
        source_a,
        dataset="nuplan/nuplan-v1.1",
        shard_counts=[9, 4],
        include_bev=True,
        num_views=6,
    )
    _write_source(
        source_b,
        dataset="nuplan/nuplan-v1.1",
        shard_counts=[8, 5],
        include_bev=True,
        num_views=6,
    )

    plan = build_reactive_dataset_plan(
        [str(source_b), str(source_a)],
        stage=ReactiveTrainingStage.NUPLAN_FULL,
    )
    assignments = assign_reactive_shards(
        plan.shards,
        world_size=2,
    )

    assert plan.total_samples == 26
    assert plan.num_views == 6
    assert plan.camera_slots == CANONICAL_SIX_CAMERA_SLOTS
    assert plan.physical_camera_order == tuple(
        f"camera_{index}" for index in range(6)
    )
    assert [sum(item.sample_count for item in rank) for rank in assignments] == [
        13,
        13,
    ]
    assert assignments == assign_reactive_shards(
        tuple(reversed(plan.shards)),
        world_size=2,
    )
    assert reactive_assignment_sha256(assignments) == (
        reactive_assignment_sha256(assignments)
    )


def test_dataset_plan_rejects_mixed_physical_camera_orders(tmp_path):
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    _write_source(
        source_a,
        dataset="nuplan/nuplan-v1.1",
        shard_counts=[4],
        include_bev=True,
        num_views=6,
    )
    manifest_b = _write_source(
        source_b,
        dataset="nuplan/nuplan-v1.1",
        shard_counts=[4],
        include_bev=True,
        num_views=6,
    )
    manifest_b["camera_order"] = list(reversed(
        manifest_b["camera_order"]
    ))
    (source_b / "manifest.json").write_text(
        json.dumps(manifest_b, sort_keys=True),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="physical camera orders"):
        build_reactive_dataset_plan(
            [str(source_a), str(source_b)],
            stage=ReactiveTrainingStage.NUPLAN_FULL,
        )


def test_stage_a_dataset_plan_rejects_previous_bev_taxonomy(tmp_path):
    source = tmp_path / "previous-taxonomy"
    manifest = _write_source(
        source,
        dataset="nuplan/nuplan-v1.1",
        shard_counts=[4],
        include_bev=True,
        num_views=6,
    )
    version_prefix, separator, version_number = (
        BEV_SEGMENTATION_TAXONOMY_VERSION.rpartition("v")
    )
    assert separator and int(version_number) > 1
    manifest["bev_taxonomy_version"] = (
        f"{version_prefix}v{int(version_number) - 1}"
    )
    (source / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="current BEV taxonomy"):
        build_reactive_dataset_plan(
            [str(source)],
            stage=ReactiveTrainingStage.NUPLAN_FULL,
        )


def test_dataset_plan_rejects_previous_camera_image_size(tmp_path):
    source = tmp_path / "camera-256"
    manifest = _write_source(
        source,
        dataset="nuplan/nuplan-v1.1",
        shard_counts=[4],
        include_bev=True,
        num_views=6,
    )
    manifest["image_size"] = 256
    (source / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="camera image size"):
        build_reactive_dataset_plan(
            [str(source)],
            stage=ReactiveTrainingStage.NUPLAN_FULL,
        )


def test_stage_b_rejects_previous_camera_image_size(tmp_path):
    source = tmp_path / "l2d-camera-256"
    manifest = _write_source(
        source,
        dataset=L2D_DATASET_NAME,
        shard_counts=[4],
        include_bev=False,
        num_views=6,
    )
    manifest["image_size"] = 256
    (source / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="camera image size"):
        build_reactive_dataset_plan(
            [str(source)],
            stage=ReactiveTrainingStage.L2D_CONTINUATION,
        )


def test_reactive_ddp_uses_static_graph_and_frozen_buffers():
    source = inspect.getsource(train_loop_per_worker)
    fixed_step_source = inspect.getsource(_train_fixed_steps)
    synchronization_source = inspect.getsource(
        _synchronize_gradient_micro_step
    )

    assert '"broadcast_buffers": not freeze_bevformer' in source
    assert '"bucket_cap_mb": REACTIVE_DDP_BUCKET_CAP_MB' in source
    assert '"find_unused_parameters": False' in source
    assert '"init_sync": False' in source
    assert '"static_graph": True' in source
    assert source.count("_assert_ddp_model_state_consistent(model)") == 2
    assert "_synchronize_gradient_micro_step" in fixed_step_source
    assert "Reactive first-step phase" in fixed_step_source
    assert "Reactive performance" in fixed_step_source
    assert "_aggregate_reactive_performance" in fixed_step_source
    assert "performance_metrics" in fixed_step_source
    synchronize_position = fixed_step_source.index(
        "torch.cuda.synchronize(device)"
    )
    timestamp_position = fixed_step_source.index(
        "time.perf_counter() - step_started"
    )
    assert synchronize_position < timestamp_position
    assert "optimizer_step_index == 0" in synchronization_source
    expected_resume_source = source.split(
        "expected_resume = {",
        maxsplit=1,
    )[1].split(
        "resume_state = ReactiveResumeState(",
        maxsplit=1,
    )[0]
    model_config_source = source.split(
        "model_config = {",
        maxsplit=1,
    )[1].split(
        "if raw_bev_statistics is not None:",
        maxsplit=1,
    )[0]
    assert '"capacity_block_end_utc"' not in expected_resume_source
    assert '"capacity_block_end_utc"' in model_config_source
    assert '"bev_ap_bins": int(config["bev_ap_bins"])' in (
        expected_resume_source
    )
    assert '"bev_ap_bins": int(config["bev_ap_bins"])' in (
        model_config_source
    )
    for field in (
        '"num_loader_workers": int(config["num_loader_workers"])',
        '"shuffle_buffer": int(config["shuffle_buffer"])',
        '"sample_stream_digest_version"',
        '"bev_checkpoint_quality_guard_version"',
        '"bev_checkpoint_min_class_iou"',
        '"bev_checkpoint_min_class_precision"',
    ):
        assert field in expected_resume_source
        assert field in model_config_source


def test_reactive_performance_sampling_and_nvidia_metrics():
    assert REACTIVE_PERFORMANCE_LOG_INTERVAL_STEPS == 64
    assert REACTIVE_PERFORMANCE_LOG_VERSION == "reactive_performance_v1"
    assert (
        REACTIVE_SAMPLE_STREAM_DIGEST_VERSION
        == "reactive_sample_stream_v1"
    )
    assert not _performance_sample_due(63, interval_steps=64)
    assert _performance_sample_due(64, interval_steps=64)
    with pytest.raises(ValueError, match="positive"):
        _performance_sample_due(0, interval_steps=64)
    with pytest.raises(ValueError, match="positive"):
        _performance_sample_due(1, interval_steps=0)

    metrics = _parse_nvidia_smi_csv(
        "50, 16, 7166, 81559, 235.0\n"
        "70, 20, 8192, 81559, 260.0\n"
    )

    assert metrics["performance_gpu_count"] == 2.0
    assert metrics["performance_gpu_utilization_percent_min"] == 50.0
    assert metrics["performance_gpu_utilization_percent_avg"] == 60.0
    assert metrics["performance_gpu_utilization_percent_max"] == 70.0
    assert metrics["performance_memory_used_mib_max"] == 8192.0
    assert metrics["performance_power_watts_avg"] == pytest.approx(247.5)
    assert _parse_nvidia_smi_csv("not supported") == {}


def test_reactive_performance_aggregation_reports_rank_skew(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    monkeypatch.setattr(dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    phase_seconds = {
        name: float(index + 1) / 10.0
        for index, name in enumerate(
            reactive_stage_module.REACTIVE_PERFORMANCE_PHASES
        )
    }

    metrics = _aggregate_reactive_performance(
        phase_seconds,
        local_samples=8,
        device=torch.device("cpu"),
    )

    assert metrics["performance_loader_wait_seconds_avg"] == 0.1
    assert metrics["performance_loader_wait_seconds_max"] == 0.1
    assert metrics["performance_step_seconds_max"] == 0.9
    assert metrics["performance_global_samples_per_second"] == (
        pytest.approx(8.0 / 0.9)
    )
    assert metrics["performance_gpu_allocated_bytes_max"] == 0.0


def test_model_state_sha256_covers_parameters_and_scalar_buffers():
    torch = pytest.importorskip("torch")

    class StateModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(4.0))
            self.register_buffer(
                "scalar",
                torch.tensor(2.0, dtype=torch.bfloat16),
            )

    left = StateModule()
    right = StateModule()

    assert _model_state_sha256(left) == _model_state_sha256(right)
    with torch.no_grad():
        right.weight[0] = 9.0
    assert _model_state_sha256(left) != _model_state_sha256(right)


def _run_static_graph_gloo_worker(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.checkpoint import checkpoint

    class ReentrantCheckpointModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = torch.nn.Linear(4, 4)
            self.deliberately_unused = torch.nn.Linear(4, 4)

        def forward(self, value):
            return checkpoint(
                self.used,
                value,
                use_reentrant=True,
            )

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = DistributedDataParallel(
            ReentrantCheckpointModel(),
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        observed_schedule = []
        for optimizer_step_index in range(2):
            optimizer.zero_grad(set_to_none=True)
            for accumulation_index in range(2):
                synchronize = _synchronize_gradient_micro_step(
                    optimizer_step_index,
                    accumulation_index,
                    2,
                )
                observed_schedule.append(synchronize)
                sync_context = (
                    nullcontext()
                    if synchronize
                    else model.no_sync()
                )
                value = torch.full(
                    (2, 4),
                    float(rank + accumulation_index + 1),
                    requires_grad=True,
                )
                with sync_context:
                    output = model(value)
                    loss = output.square().mean()
                    diagnostic_gradient = torch.autograd.grad(
                        loss,
                        output,
                        retain_graph=True,
                    )[0]
                    assert torch.isfinite(diagnostic_gradient).all()
                    loss.backward()
            gradient = model.module.used.weight.grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            optimizer.step()
        assert observed_schedule == [True, True, False, True]
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_static_graph_gloo_executes_production_no_sync_warmup(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("PyTorch was built without Gloo")

    torch.multiprocessing.spawn(
        _run_static_graph_gloo_worker,
        args=(2, str(tmp_path / "gloo-init")),
        nprocs=2,
        join=True,
    )


def _run_reactive_validation_reduction_gloo_worker(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        expected_uids = tuple(
            f"sample-{index}" for index in range(world_size)
        )
        expected_digest = hashlib.sha256(
            "\n".join(expected_uids).encode("utf-8")
        ).hexdigest()
        integer_state = torch.tensor(
            [rank + 1, (rank + 1) * 10],
            dtype=torch.int64,
        )
        floating_state = torch.tensor(
            [rank + 0.25],
            dtype=torch.float64,
        )
        sample_uids = _reduce_reactive_validation_state(
            (integer_state, floating_state),
            local_sample_uids=(f"sample-{rank}",),
            expected_sample_count=world_size,
            expected_sample_uid_sha256=expected_digest,
        )

        assert integer_state.tolist() == [3, 30]
        assert floating_state.tolist() == pytest.approx([1.5])
        assert sample_uids == ("sample-0", "sample-1")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_reactive_validation_reduction_uses_real_gloo_collectives(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("PyTorch was built without Gloo")

    torch.multiprocessing.spawn(
        _run_reactive_validation_reduction_gloo_worker,
        args=(2, str(tmp_path / "validation-gloo-init")),
        nprocs=2,
        join=True,
    )


def _run_reactive_validation_uid_mismatch_gloo_worker(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        expected_uids = tuple(
            f"sample-{index}" for index in range(world_size)
        )
        expected_digest = hashlib.sha256(
            "\n".join(expected_uids).encode("utf-8")
        ).hexdigest()
        local_uid = "replacement" if rank == 1 else f"sample-{rank}"
        with pytest.raises(ValueError, match="frozen manifest"):
            _reduce_reactive_validation_state(
                (torch.ones(1, dtype=torch.float64),),
                local_sample_uids=(local_uid,),
                expected_sample_count=world_size,
                expected_sample_uid_sha256=expected_digest,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_reactive_validation_reduction_rejects_equal_count_uid_substitution(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("PyTorch was built without Gloo")

    torch.multiprocessing.spawn(
        _run_reactive_validation_uid_mismatch_gloo_worker,
        args=(2, str(tmp_path / "validation-uid-mismatch-gloo-init")),
        nprocs=2,
        join=True,
    )


def _run_reactive_validation_error_gloo_worker(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        local_errors = (
            ["FloatingPointError: rank-local non-finite BEV logits"]
            if rank == 1
            else []
        )
        with pytest.raises(
            ValueError,
            match="rank-local non-finite BEV logits",
        ):
            _raise_distributed_validation_contract_errors(local_errors)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_reactive_validation_errors_use_real_gloo_collectives(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("PyTorch was built without Gloo")

    torch.multiprocessing.spawn(
        _run_reactive_validation_error_gloo_worker,
        args=(2, str(tmp_path / "validation-error-gloo-init")),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    ("optimizer_steps", "expected"),
    (
        (1, {0}),
        (4, {0, 1, 2, 3}),
        (8, set(range(8))),
        (15, {0, 2, 4, 6, 8, 10, 12, 14}),
    ),
)
def test_bev_gradient_diagnostics_span_the_epoch(
    optimizer_steps,
    expected,
):
    assert (
        reactive_stage_module._bev_gradient_diagnostic_step_indices(
            optimizer_steps
        )
        == expected
    )


def test_dataset_plan_rejects_legacy_manifest_without_per_shard_counts(
    tmp_path,
):
    source = tmp_path / "legacy"
    manifest = _write_source(
        source,
        dataset="yaak-ai/L2D",
        shard_counts=[2],
        include_bev=False,
        num_views=6,
    )
    manifest.pop("shard_sample_counts")
    (source / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="shard_sample_counts"):
        build_reactive_dataset_plan(
            [str(source)],
            stage=ReactiveTrainingStage.L2D_CONTINUATION,
        )


def test_rank_staging_verifies_tar_and_manifest_digests(tmp_path):
    source = tmp_path / "source"
    _write_source(
        source,
        dataset="yaak-ai/L2D",
        shard_counts=[3, 2],
        include_bev=False,
        num_views=6,
    )
    plan = build_reactive_dataset_plan(
        [str(source)],
        stage=ReactiveTrainingStage.L2D_CONTINUATION,
    )
    assignments = assign_reactive_shards(
        plan.shards,
        world_size=2,
    )

    local_directories = stage_rank_reactive_shards(
        assignments[0],
        cache_root=tmp_path / "cache",
    )

    assert len(local_directories) == 1
    staged = tmp_path / "cache" / hashlib.sha256(
        str(source).encode()
    ).hexdigest()[:16]
    assert (staged / "manifest.json").is_file()
    assert sorted(path.name for path in staged.glob("*.tar")) == [
        assignments[0][0].shard_name
    ]

    source_shard = source / assignments[1][0].shard_name
    source_shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        stage_rank_reactive_shards(
            assignments[1],
            cache_root=tmp_path / "cache-corrupt",
        )


def test_fixed_step_count_uses_global_effective_batch():
    assert optimizer_steps_per_epoch(
        total_samples=100,
        val_fraction=0.1,
        world_size=8,
        per_rank_batch_size=1,
        gradient_accumulation_steps=1,
    ) == 12
    assert optimizer_steps_per_epoch(
        total_samples=100,
        val_fraction=0.1,
        world_size=4,
        per_rank_batch_size=1,
        gradient_accumulation_steps=4,
    ) == 6


def test_restarting_iterator_repeats_finite_loader():
    iterator = RestartingIterator([1, 2])

    assert [next(iterator) for _ in range(5)] == [1, 2, 1, 2, 1]
    assert iterator.restarts == 2

    with pytest.raises(ValueError, match="yielded no batches"):
        next(RestartingIterator([]))


def test_step_checkpoint_schedule_uses_fixed_intervals():
    due_steps = [
        step
        for step in range(1, 1_025)
        if _checkpoint_step_due(
            step,
            optimizer_steps=1_024,
            checkpoint_interval_steps=512,
        )
    ]

    assert due_steps == [512, 1_024]
    assert not _checkpoint_step_due(
        7,
        optimizer_steps=7,
        checkpoint_interval_steps=512,
    )


def test_checkpoint_interval_must_fit_inside_epoch():
    _validate_checkpoint_interval(4, 10)
    with pytest.raises(ValueError, match="positive"):
        _validate_checkpoint_interval(0, 10)
    with pytest.raises(ValueError, match="must not exceed"):
        _validate_checkpoint_interval(11, 10)


def test_fixed_step_resume_matches_uninterrupted_training(
    monkeypatch,
    capsys,
):
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    class ResumeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, visual_tiles, *_args, **_kwargs):
            controls = (
                visual_tiles.reshape(visual_tiles.shape[0], -1).mean(dim=1)
                * self.weight
            )
            return controls[:, None], {}

    class ResumeObjective:
        stage = ReactiveTrainingStage.L2D_CONTINUATION
        compute_bev_segmentation = False
        compute_route_reconstruction = False

        def __call__(self, predicted_controls, _auxiliary, _batch):
            total = predicted_controls.mean()
            zero = total * 0.0
            return {
                "total": total,
                "trajectory": total,
                "bev_segmentation": zero,
                "bev_segmentation_bce": zero,
                "bev_segmentation_dice": zero,
                "route_reconstruction": zero,
            }

    batches = [
        {
            "sample_uid": [f"resume-sample-{index}"],
            "visual_tiles": torch.full((1, 1, 1, 1, 1), value),
            "map_context": torch.zeros(1, 1, 1, 1),
            "visual_history": torch.zeros(1, 1),
            "egomotion_history": torch.zeros(1, 1),
            "route_mask": torch.zeros(1, 1, 1, 1),
            "map_valid": torch.ones(1, dtype=torch.bool),
            "route_valid": torch.ones(1, dtype=torch.bool),
        }
        for index, value in enumerate((1.0, 2.0))
    ]
    monkeypatch.setattr(dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        reactive_stage_module,
        "reactive_gradient_parameter_groups",
        lambda model: {
            "camera": [],
            "front_gate": [],
            "navigation": [],
            "planner": [model.weight],
        },
    )

    uninterrupted_model = ResumeModel()
    uninterrupted_optimizer = torch.optim.SGD(
        uninterrupted_model.parameters(),
        lr=0.1,
    )
    saved = {}

    def capture_first_step(step, train_state):
        if step == 1:
            saved["model"] = {
                name: value.detach().clone()
                for name, value in uninterrupted_model.state_dict().items()
            }
            saved["optimizer"] = uninterrupted_optimizer.state_dict()
            saved["rng"] = _capture_rng_state()
            saved["train"] = dict(train_state)

    uninterrupted_metrics = _train_fixed_steps(
        uninterrupted_model,
        batches,
        ResumeObjective(),
        uninterrupted_optimizer,
        device=torch.device("cpu"),
        optimizer_steps=2,
        gradient_accumulation_steps=1,
        grad_clip=10.0,
        precision="fp32",
        checkpoint_interval_steps=1,
        checkpoint_callback=capture_first_step,
    )
    expected_phases = [
        "forward_start",
        "forward_complete",
        "backward_complete",
        "gradient_check_complete",
        "finite_collective_complete",
        "optimizer_complete",
    ]
    assert re.findall(
        r"Reactive first-step phase rank=0 phase=([a-z_]+)",
        capsys.readouterr().out,
    ) == expected_phases

    resumed_model = ResumeModel()
    resumed_model.load_state_dict(saved["model"])
    resumed_optimizer = torch.optim.SGD(
        resumed_model.parameters(),
        lr=0.1,
    )
    resumed_optimizer.load_state_dict(saved["optimizer"])
    resumed_metrics = _train_fixed_steps(
        resumed_model,
        batches,
        ResumeObjective(),
        resumed_optimizer,
        device=torch.device("cpu"),
        optimizer_steps=2,
        gradient_accumulation_steps=1,
        grad_clip=10.0,
        precision="fp32",
        start_optimizer_step=1,
        resume_rank_state=saved["train"],
        resume_rng_state=saved["rng"],
        checkpoint_interval_steps=1,
    )
    assert re.findall(
        r"Reactive first-step phase rank=0 phase=([a-z_]+)",
        capsys.readouterr().out,
    ) == expected_phases

    assert resumed_model.weight.item() == pytest.approx(
        uninterrupted_model.weight.item()
    )
    assert resumed_metrics == pytest.approx(uninterrupted_metrics)


def test_replay_loader_position_checks_samples_and_restarts():
    batch = {
        "sample_uid": ["sample-a", "sample-b"],
        "visual_tiles": np.zeros((2, 1, 1, 1, 1)),
    }
    iterator = RestartingIterator([batch])
    expected_sha256 = REACTIVE_SAMPLE_STREAM_INITIAL_SHA256
    for _ in range(2):
        expected_sha256 = _extend_sample_stream_sha256(
            expected_sha256,
            batch["sample_uid"],
        )

    assert _replay_loader_position(
        iterator,
        skipped_micro_steps=2,
        expected_restarts=1,
        expected_samples=4,
        expected_sample_stream_sha256=expected_sha256,
    ) == expected_sha256

    with pytest.raises(ValueError, match="loader position"):
        _replay_loader_position(
            RestartingIterator([batch]),
            skipped_micro_steps=2,
            expected_restarts=1,
            expected_samples=3,
            expected_sample_stream_sha256=expected_sha256,
        )

    reversed_batch = {
        **batch,
        "sample_uid": list(reversed(batch["sample_uid"])),
    }
    with pytest.raises(ValueError, match="loader position"):
        _replay_loader_position(
            RestartingIterator([reversed_batch]),
            skipped_micro_steps=2,
            expected_restarts=1,
            expected_samples=4,
            expected_sample_stream_sha256=expected_sha256,
        )


def test_reactive_rng_state_round_trips():
    torch = pytest.importorskip("torch")
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    state = _capture_rng_state()
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    _restore_rng_state(state)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_float64_gradient_clipping_handles_large_finite_values():
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([5.0e23, -2.0e23])

    norm, finite = clip_finite_gradients_float64([parameter], 1.0)

    assert finite
    assert torch.isfinite(norm)
    assert norm.item() > 1.0e23
    assert torch.linalg.vector_norm(parameter.grad).item() == pytest.approx(
        1.0,
        rel=1e-6,
    )


def test_float64_gradient_clipping_rejects_non_finite_values():
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([float("nan"), 1.0])

    _, finite = clip_finite_gradients_float64([parameter], 1.0)

    assert not finite
    assert torch.isnan(parameter.grad[0])


def test_reactive_gradient_groups_clip_branches_independently():
    torch = pytest.importorskip("torch")

    class Reactive(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Backbone = torch.nn.Linear(1, 1, bias=False)
            self.FeatureFusion = torch.nn.Linear(1, 1, bias=False)
            self.BEVSegmentationHead = torch.nn.Linear(1, 1, bias=False)
            self.NavigationEncoder = torch.nn.Linear(1, 1, bias=False)
            self.MapBEVFusion = torch.nn.Linear(1, 1, bias=False)
            self.RouteReconstructionHead = torch.nn.Linear(
                1,
                1,
                bias=False,
            )
            self.TemporalMemory = torch.nn.Linear(1, 1, bias=False)
            self.TrajectoryPlanner = torch.nn.Linear(1, 1, bias=False)

    model = torch.nn.Module()
    model.Reactive_E2E = Reactive()
    groups = reactive_gradient_parameter_groups(model)

    expected_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    grouped_ids = {
        id(parameter)
        for parameters in groups.values()
        for parameter in parameters
    }
    assert set(groups) == {
        "camera",
        "front_gate",
        "navigation",
        "planner",
    }
    assert grouped_ids == expected_ids
    assert sum(len(parameters) for parameters in groups.values()) == len(
        expected_ids
    )

    gradient_values = {
        "camera": 10.0,
        "front_gate": 0.0,
        "navigation": 0.5 / len(groups["navigation"]) ** 0.5,
        "planner": 4.0,
    }
    post_clip_norms = {}
    for group_name, parameters in groups.items():
        if not parameters:
            continue
        for parameter in parameters:
            parameter.grad = torch.full_like(
                parameter,
                gradient_values[group_name],
            )
        clip_finite_gradients_float64(parameters, 1.0)
        post_clip_norms[group_name] = torch.linalg.vector_norm(torch.stack([
            parameter.grad.reshape(-1)
            for parameter in parameters
        ]))

    assert post_clip_norms["camera"].item() == pytest.approx(1.0)
    assert "front_gate" not in post_clip_norms
    assert post_clip_norms["navigation"].item() == pytest.approx(0.5)
    assert post_clip_norms["planner"].item() == pytest.approx(1.0)


def test_ray_checkpoint_uri_restores_s3_scheme_within_storage():
    storage = "s3://checkpoints/ray-train"

    assert normalize_ray_checkpoint_uri(
        "checkpoints/ray-train/run/checkpoint_0001",
        storage,
    ) == "s3://checkpoints/ray-train/run/checkpoint_0001"
    assert normalize_ray_checkpoint_uri(
        "s3://checkpoints/ray-train/run/checkpoint_0001/",
        storage,
    ) == "s3://checkpoints/ray-train/run/checkpoint_0001"


def test_ray_checkpoint_uri_rejects_paths_outside_storage():
    with pytest.raises(ValueError, match="outside"):
        normalize_ray_checkpoint_uri(
            "other-bucket/ray-train/run/checkpoint_0001",
            "s3://checkpoints/ray-train",
        )


def test_ray_managed_recovery_checkpoint_wins_after_explicit_resume():
    ray_checkpoint = object()

    restored, explicit_uri, ignored_explicit = (
        _resolve_resume_sources(
            ray_checkpoint,
            "s3://checkpoints/original/checkpoint_0001/",
        )
    )

    assert restored is ray_checkpoint
    assert explicit_uri == ""
    assert ignored_explicit is True


def test_explicit_checkpoint_is_used_without_ray_checkpoint():
    restored, explicit_uri, ignored_explicit = (
        _resolve_resume_sources(
            None,
            "s3://checkpoints/original/checkpoint_0001/",
        )
    )

    assert restored is None
    assert explicit_uri == "s3://checkpoints/original/checkpoint_0001"
    assert ignored_explicit is False


@pytest.mark.parametrize(
    (
        "training_scope",
        "parent_uri",
        "restored_checkpoint",
        "expected",
    ),
    (
        (ReactiveTrainingScope.BEV_ONLY, "", None, True),
        (
            ReactiveTrainingScope.BEV_ONLY,
            "s3://checkpoints/parent.pt",
            None,
            False,
        ),
        (ReactiveTrainingScope.BEV_ONLY, "", object(), False),
        (ReactiveTrainingScope.MULTITASK, "", None, False),
    ),
)
def test_bev_head_dataset_prior_initialization_is_fresh_start_only(
    training_scope,
    parent_uri,
    restored_checkpoint,
    expected,
):
    assert (
        _should_initialize_bev_head_from_training_statistics(
            training_scope,
            parent_uri=parent_uri,
            restored_checkpoint=restored_checkpoint,
        )
        is expected
    )


def test_rank_owned_nodesplitter_preserves_every_assigned_shard():
    urls = ["rank-000-part-000.tar", "rank-000-part-003.tar"]

    assert list(passthrough_nodesplitter(iter(urls))) == urls


def _stage_config(stage: str) -> dict[str, object]:
    return {
        "backbone": "swin_v2_tiny",
        "bev_ap_bins": 1024,
        "bev_max_repeat": 4,
        "bev_min_positive_cells": 1,
        "bev_min_positive_samples": 1,
        "bev_pos_weight_cap": 64.0,
        "bev_repeat_frequency_threshold": 0.05,
        "bev_weight": 1.0,
        "allow_random_bevformer_init": True,
        "corridor_pos_weight": 1.0,
        "capacity_block_end_utc": "2099-01-01T00:00:00Z",
        "checkpoint_interval_steps": 512,
        "epochs": 2,
        "grad_clip": 1.0,
        "gradient_accumulation_steps": 1,
        "is_pretrained": False,
        "learning_rate": 1e-4,
        "num_loader_workers": 1,
        "num_workers": 8,
        "freeze_bevformer": True,
        "parent_checkpoint_uri": (
            "s3://checkpoints/stage-a/checkpoint.pt"
            if stage == "l2d_continuation"
            else ""
        ),
        "resume_checkpoint_uri": "",
        "per_rank_batch_size": 1,
        "precision": "bf16",
        "route_weight": 1.0,
        "run_name": "reactive-stage-test",
        "selection_ade_regression_margin_m": 0.5,
        "selection_ade_scale_m": 5.0,
        "source_uris": ["s3://datasets/reactive"],
        "stage": stage,
        "steps_per_epoch": 0,
        "storage_path": "s3://checkpoints/ray-train",
        "shuffle_buffer": 64,
        "training_seed": 149,
        "trajectory_weight": 1.0,
        "val_fraction": 0.1,
        "validation_sample_limit": 1024,
        "weight_decay": 0.01,
        "worker_cpus": 3,
    }


def _single_worker_smoke_config() -> dict[str, object]:
    config = _stage_config("nuplan_full")
    config.update({
        "allow_random_bevformer_init": False,
        "allow_single_worker_smoke": True,
        "backbone": "res_net_50",
        "bev_encoder_learning_rate": 1e-5,
        "bevformer_pretrained_checkpoint_sha256": "a" * 64,
        "bevformer_pretrained_checkpoint_uri": "s3://bucket/model.pth",
        "capacity_block_end_utc": "",
        "checkpoint_interval_steps": 4,
        "epochs": 1,
        "freeze_bevformer": False,
        "is_pretrained": True,
        "num_workers": 1,
        "parent_checkpoint_uri": "",
        "per_rank_batch_size": 1,
        "precision": "bf16",
        "resume_checkpoint_uri": "",
        "route_weight": 0.0,
        "source_uris": [
            "s3://datasets/reactive-000",
            "s3://datasets/reactive-001",
        ],
        "steps_per_epoch": 8,
        "training_scope": "bev_only",
        "trajectory_weight": 0.0,
        "validation_sample_limit": 128,
        "worker_cpus": 3,
    })
    return config


@pytest.mark.parametrize("stage", ["nuplan_full", "l2d_continuation"])
def test_validate_stage_config_accepts_locked_program(stage):
    validate_reactive_stage_config(_stage_config(stage))


def test_validate_stage_config_accepts_restricted_single_worker_smoke():
    validate_reactive_stage_config(_single_worker_smoke_config())


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"allow_single_worker_smoke": False}, "requires"),
        ({"epochs": 2}, "single-worker smoke"),
        ({"steps_per_epoch": 0}, "single-worker smoke"),
        ({"steps_per_epoch": 1}, "single-worker smoke"),
        ({"steps_per_epoch": 17}, "single-worker smoke"),
        ({"checkpoint_interval_steps": 9}, "single-worker smoke"),
        ({"per_rank_batch_size": 2}, "single-worker smoke"),
        ({"gradient_accumulation_steps": 2}, "single-worker smoke"),
        ({"backbone": "swin_v2_tiny"}, "single-worker smoke"),
        (
            {"capacity_block_end_utc": "2099-01-01T00:00:00Z"},
            "single-worker smoke",
        ),
        ({"validation_sample_limit": 257}, "single-worker smoke"),
        (
            {
                "source_uris": [
                    "s3://datasets/reactive-000",
                    "s3://datasets/reactive-001",
                    "s3://datasets/reactive-002",
                ],
            },
            "single-worker smoke",
        ),
    ],
)
def test_validate_stage_config_rejects_broadened_single_worker_smoke(
    override,
    match,
):
    config = _single_worker_smoke_config()
    config.update(override)

    with pytest.raises(ValueError, match=match):
        validate_reactive_stage_config(config)


def test_validate_stage_config_rejects_smoke_flag_on_production_topology():
    config = _stage_config("nuplan_full")
    config["allow_single_worker_smoke"] = True

    with pytest.raises(ValueError, match="requires num_workers=1"):
        validate_reactive_stage_config(config)


@pytest.mark.parametrize("fraction", [0.0, 0.125])
def test_bev_rank_truncation_guard_accepts_production_limit(fraction):
    _validate_bev_rank_truncation(
        fraction,
        single_worker_smoke=False,
        bounded_bev_canary=False,
    )


def test_bev_rank_truncation_guard_rejects_production_tail():
    with pytest.raises(ValueError, match="truncation exceeds"):
        _validate_bev_rank_truncation(
            0.125001,
            single_worker_smoke=False,
            bounded_bev_canary=False,
        )


def test_bev_rank_truncation_guard_allows_bounded_smoke_prefix():
    _validate_bev_rank_truncation(
        0.999,
        single_worker_smoke=True,
        bounded_bev_canary=False,
    )


def test_bev_rank_truncation_guard_allows_bounded_canary_prefix():
    _validate_bev_rank_truncation(
        0.999,
        single_worker_smoke=False,
        bounded_bev_canary=True,
    )


@pytest.mark.parametrize(
    (
        "single_worker_smoke",
        "bounded_bev_canary",
        "expected",
    ),
    (
        (False, False, True),
        (True, False, False),
        (False, True, False),
    ),
)
def test_bev_checkpoint_quality_guard_is_production_only(
    single_worker_smoke,
    bounded_bev_canary,
    expected,
):
    assert (
        _should_enforce_bev_checkpoint_quality_guard(
            single_worker_smoke=single_worker_smoke,
            bounded_bev_canary=bounded_bev_canary,
        )
        is expected
    )


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
def test_bev_rank_truncation_guard_rejects_invalid_fraction(fraction):
    with pytest.raises(RuntimeError, match="fraction is invalid"):
        _validate_bev_rank_truncation(
            fraction,
            single_worker_smoke=True,
            bounded_bev_canary=False,
        )


def test_validate_stage_config_accepts_bev_only_scope():
    config = _stage_config("nuplan_full")
    config.update({
        "bev_encoder_learning_rate": 1e-5,
        "bev_repeat_frequency_threshold": 0.01,
        "freeze_bevformer": False,
        "training_scope": "bev_only",
        "trajectory_weight": 0.0,
        "bev_weight": 1.0,
        "route_weight": 0.0,
        "validation_sample_limit": 4096,
    })

    validate_reactive_stage_config(config)


def test_validate_stage_config_accepts_bounded_eight_rank_bev_canary():
    config = _stage_config("nuplan_full")
    config.update({
        "allow_bounded_bev_canary": True,
        "bev_encoder_learning_rate": 1e-5,
        "bev_repeat_frequency_threshold": 0.05,
        "capacity_block_end_utc": (
            datetime.now(timezone.utc) + timedelta(hours=24)
        ).isoformat(),
        "epochs": 2,
        "freeze_bevformer": False,
        "num_workers": 8,
        "per_rank_batch_size": 4,
        "precision": "bf16",
        "steps_per_epoch": 128,
        "training_scope": "bev_only",
        "trajectory_weight": 0.0,
        "bev_weight": 1.0,
        "route_weight": 0.0,
    })

    validate_reactive_stage_config(config)


@pytest.mark.parametrize(
    "override",
    (
        {"num_workers": 4},
        {"epochs": 3},
        {"steps_per_epoch": 0},
        {"steps_per_epoch": 257},
        {"per_rank_batch_size": 2},
        {"precision": "fp32"},
        {"freeze_bevformer": True},
        {"training_scope": "multitask"},
    ),
)
def test_validate_stage_config_rejects_broadened_bounded_bev_canary(
    override,
):
    config = _stage_config("nuplan_full")
    config.update({
        "allow_bounded_bev_canary": True,
        "bev_encoder_learning_rate": 1e-5,
        "bev_repeat_frequency_threshold": 0.05,
        "capacity_block_end_utc": (
            datetime.now(timezone.utc) + timedelta(hours=24)
        ).isoformat(),
        "epochs": 2,
        "freeze_bevformer": False,
        "num_workers": 8,
        "per_rank_batch_size": 4,
        "precision": "bf16",
        "steps_per_epoch": 128,
        "training_scope": "bev_only",
        "trajectory_weight": 0.0,
        "bev_weight": 1.0,
        "route_weight": 0.0,
    })
    config.update(override)

    with pytest.raises(ValueError, match="allow_bounded_bev_canary"):
        validate_reactive_stage_config(config)


def test_validate_stage_config_rejects_unrepresentable_validation_fraction():
    config = _stage_config("nuplan_full")
    config["val_fraction"] = 0.15

    with pytest.raises(ValueError, match="ten-bucket split"):
        validate_reactive_stage_config(config)


@pytest.mark.parametrize("shuffle_buffer", (0, 1))
def test_validate_stage_config_rejects_disabled_shuffle(shuffle_buffer):
    config = _stage_config("nuplan_full")
    config["shuffle_buffer"] = shuffle_buffer

    with pytest.raises(ValueError, match="shuffle_buffer"):
        validate_reactive_stage_config(config)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"trajectory_weight": 1.0}, "only BEV loss"),
        ({"route_weight": 1.0}, "only BEV loss"),
        ({"bev_weight": 0.0}, "only BEV loss"),
        ({"freeze_bevformer": True}, "unfrozen BEVFormer"),
        ({"stage": "l2d_continuation"}, "nuPlan full"),
    ],
)
def test_validate_stage_config_rejects_invalid_bev_only_scope(
    override,
    match,
):
    config = _stage_config("nuplan_full")
    config.update({
        "freeze_bevformer": False,
        "training_scope": "bev_only",
        "trajectory_weight": 0.0,
        "bev_weight": 1.0,
        "route_weight": 0.0,
    })
    config.update(override)

    with pytest.raises(ValueError, match=match):
        validate_reactive_stage_config(config)


@pytest.mark.parametrize("batch_size", [2, 4])
def test_validate_stage_config_accepts_production_batch_sizes(batch_size):
    config = _stage_config("nuplan_full")
    config["per_rank_batch_size"] = batch_size

    validate_reactive_stage_config(config)


@pytest.mark.parametrize(
    ("world_size", "hostname_count"),
    ((1, 1), (2, 2), (4, 1), (8, 1)),
)
def test_expected_reactive_hostname_count_matches_ray_topology(
    world_size,
    hostname_count,
):
    assert expected_reactive_hostname_count(world_size) == hostname_count


def test_validate_stage_config_rejects_parent_and_batch_contract_changes():
    stage_a = _stage_config("nuplan_full")
    stage_a["parent_checkpoint_uri"] = "s3://checkpoints/parent.pt"
    validate_reactive_stage_config(stage_a)

    trainable_parent = dict(stage_a)
    trainable_parent["freeze_bevformer"] = False
    with pytest.raises(ValueError, match="frozen BEVFormer"):
        validate_reactive_stage_config(trainable_parent)

    bev_only_parent = dict(stage_a)
    bev_only_parent.update({
        "freeze_bevformer": False,
        "training_scope": "bev_only",
        "trajectory_weight": 0.0,
        "route_weight": 0.0,
    })
    with pytest.raises(ValueError, match="multitask"):
        validate_reactive_stage_config(bev_only_parent)

    stage_b = _stage_config("l2d_continuation")
    stage_b["per_rank_batch_size"] = 3
    with pytest.raises(ValueError, match="per_rank_batch_size"):
        validate_reactive_stage_config(stage_b)

    conflicting_resume = _stage_config("l2d_continuation")
    conflicting_resume["resume_checkpoint_uri"] = (
        "s3://checkpoints/resume"
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_reactive_stage_config(conflicting_resume)

    caller_weighted = _stage_config("nuplan_full")
    caller_weighted["bev_pos_weights"] = [1.0] * 8
    with pytest.raises(ValueError, match="derived"):
        validate_reactive_stage_config(caller_weighted)

    invalid_interval = _stage_config("nuplan_full")
    invalid_interval["checkpoint_interval_steps"] = 0
    with pytest.raises(ValueError, match="checkpoint_interval_steps"):
        validate_reactive_stage_config(invalid_interval)

    invalid_performance_interval = _stage_config("nuplan_full")
    invalid_performance_interval["performance_log_interval_steps"] = 0
    with pytest.raises(ValueError, match="performance_log_interval_steps"):
        validate_reactive_stage_config(invalid_performance_interval)


def test_p5en_stage_requires_a_safe_capacity_block_window():
    missing = _stage_config("nuplan_full")
    missing["capacity_block_end_utc"] = ""
    with pytest.raises(ValueError, match="capacity_block_end_utc"):
        validate_reactive_stage_config(missing)

    missing_offset = _stage_config("nuplan_full")
    missing_offset["capacity_block_end_utc"] = "2099-01-01T00:00:00"
    with pytest.raises(ValueError, match="UTC offset"):
        validate_reactive_stage_config(missing_offset)

    expired = _stage_config("nuplan_full")
    expired["capacity_block_end_utc"] = "2000-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="at least 22 hours"):
        validate_reactive_stage_config(expired)

    expired_resume = _stage_config("nuplan_full")
    expired_resume["resume_checkpoint_uri"] = "s3://checkpoints/resume"
    expired_resume["capacity_block_end_utc"] = "2000-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="at least 2 hours"):
        validate_reactive_stage_config(expired_resume)

    short_resume = _stage_config("nuplan_full")
    short_resume["resume_checkpoint_uri"] = "s3://checkpoints/resume"
    short_resume["capacity_block_end_utc"] = (
        datetime.now(timezone.utc) + timedelta(hours=3)
    ).isoformat()
    validate_reactive_stage_config(short_resume)

    validation = _stage_config("nuplan_full")
    validation["num_workers"] = 2
    validation["capacity_block_end_utc"] = ""
    validate_reactive_stage_config(validation)


def test_validate_stage_config_requires_one_initialization_mode():
    config = _stage_config("nuplan_full")
    config["allow_random_bevformer_init"] = False

    with pytest.raises(ValueError, match="exactly one"):
        validate_reactive_stage_config(config)

    config["is_pretrained"] = True
    config["bevformer_pretrained_checkpoint_uri"] = "s3://bucket/model.pth"
    config["bevformer_pretrained_checkpoint_sha256"] = "a" * 64
    validate_reactive_stage_config(config)


def _write_resume_checkpoint(
    directory,
    *,
    config,
    model,
    optimizer,
    scheduler,
    epoch=1,
    training_state=None,
    history=None,
):
    torch = pytest.importorskip("torch")
    directory.mkdir()
    torch.save(
        {
            "config": config,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_state": training_state or {
                "best_ade_6p4s_m": 2.0,
                "best_selection_score": 0.5,
            },
        },
        directory / "checkpoint.pt",
    )
    (directory / "history.json").write_text(
        json.dumps(
            [{"epoch": epoch}] if history is None else history
        )
    )


def test_step_checkpoint_restores_epoch_position_and_rank_state(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    expected = {
        "distributed_world_size": 2,
        "optimizer_steps_per_epoch": 8,
        "step_checkpoint_version": REACTIVE_STEP_CHECKPOINT_VERSION,
    }
    rank_train_states = [
        {
            "bev_logit_gradient_batches": 0,
            "bev_logit_gradient_totals": [],
            "consumed_samples": 3,
            "gradient_totals": [0.0] * 8,
            "loader_restarts": 0,
            "rank": rank,
            "sample_stream_sha256": (
                REACTIVE_SAMPLE_STREAM_INITIAL_SHA256
            ),
            "term_totals": [0.0] * 6,
        }
        for rank in range(2)
    ]
    rank_rng_states = [
        {"rank": rank, **_capture_rng_state()}
        for rank in range(2)
    ]
    checkpoint = tmp_path / "step-checkpoint"
    _write_resume_checkpoint(
        checkpoint,
        config=expected,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        training_state={
            "best_ade_6p4s_m": 1.5,
            "best_selection_score": 0.7,
            "checkpoint_kind": "step",
            "optimizer_step_in_epoch": 5,
            "rank_rng_states": rank_rng_states,
            "rank_train_states": rank_train_states,
            "step_checkpoint_version": (
                REACTIVE_STEP_CHECKPOINT_VERSION
            ),
        },
        history=[{"epoch": 1}],
    )

    state = _load_resume_checkpoint(
        str(checkpoint),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected=expected,
    )

    assert state.epoch == 2
    assert state.optimizer_step_in_epoch == 5
    assert state.best_selection_score == 0.7
    assert state.best_ade_6p4s_m == 1.5
    assert state.epoch_history == [{"epoch": 1}]
    assert state.rank_train_states == tuple(rank_train_states)
    assert state.rank_rng_states is not None
    assert [item["rank"] for item in state.rank_rng_states] == [0, 1]


def test_step_checkpoint_rejects_incomplete_rank_train_state(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    expected = {
        "distributed_world_size": 1,
        "optimizer_steps_per_epoch": 8,
        "step_checkpoint_version": REACTIVE_STEP_CHECKPOINT_VERSION,
    }
    checkpoint = tmp_path / "invalid-step-checkpoint"
    _write_resume_checkpoint(
        checkpoint,
        config=expected,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        training_state={
            "checkpoint_kind": "step",
            "optimizer_step_in_epoch": 5,
            "rank_rng_states": [{"rank": 0, **_capture_rng_state()}],
            "rank_train_states": [{"rank": 0}],
            "step_checkpoint_version": REACTIVE_STEP_CHECKPOINT_VERSION,
        },
        history=[{"epoch": 1}],
    )

    with pytest.raises(ValueError, match="train state is incomplete"):
        _load_resume_checkpoint(
            str(checkpoint),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected=expected,
        )


@pytest.mark.parametrize(
    ("field", "checkpoint_value", "requested_value"),
    (
        ("num_loader_workers", 2, 4),
        ("shuffle_buffer", 64, 512),
    ),
)
def test_resume_rejects_loader_topology_drift(
    tmp_path,
    field,
    checkpoint_value,
    requested_value,
):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    checkpoint_config = {
        "distributed_world_size": 1,
        "num_loader_workers": 2,
        "optimizer_steps_per_epoch": 8,
        "shuffle_buffer": 64,
    }
    checkpoint_config[field] = checkpoint_value
    expected = {
        **checkpoint_config,
        field: requested_value,
    }
    checkpoint = tmp_path / f"loader-drift-{field}"
    _write_resume_checkpoint(
        checkpoint,
        config=checkpoint_config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    with pytest.raises(ValueError, match="resume contract differs"):
        _load_resume_checkpoint(
            str(checkpoint),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected=expected,
        )


def test_rank_resume_state_is_used_only_for_the_resumed_epoch():
    values = ({"rank": 0, "value": 7},)

    active = _rank_resume_value_for_epoch(
        values,
        current_epoch=2,
        resume_epoch=2,
        start_optimizer_step=5,
        rank=0,
        world_size=1,
        name="train state",
    )
    next_epoch = _rank_resume_value_for_epoch(
        values,
        current_epoch=3,
        resume_epoch=2,
        start_optimizer_step=0,
        rank=0,
        world_size=1,
        name="train state",
    )

    assert active == {"rank": 0, "value": 7}
    assert next_epoch is None


def test_rank_resume_state_rejects_world_size_and_order_mismatch():
    with pytest.raises(ValueError, match="count"):
        _rank_resume_value(
            ({"rank": 0},),
            rank=0,
            world_size=2,
            name="train state",
        )
    with pytest.raises(ValueError, match="rank order"):
        _rank_resume_value(
            ({"rank": 1},),
            rank=0,
            world_size=1,
            name="train state",
        )


def test_completed_resume_skips_worker_training():
    state = ReactiveResumeState(
        epoch=4,
        optimizer_step_in_epoch=0,
        best_selection_score=0.5,
        best_ade_6p4s_m=1.0,
        epoch_history=[{"epoch": 3}],
    )

    assert _resume_completed_requested_epochs(state, 3)
    assert not _resume_completed_requested_epochs(state, 4)


def test_ray_actor_cpu_reservation_matches_worker_config(
    monkeypatch,
    tmp_path,
):
    ray = pytest.importorskip("ray")
    from ray.train import torch as ray_train_torch

    captured = {}
    latest_directory = tmp_path / "latest"
    latest_directory.mkdir()
    best_metrics = {
        "checkpoint_selection_score": 0.6,
        "checkpoint_sha256": "a" * 64,
        "epoch": 1,
        "is_best": 1,
        "world_size": 4,
    }
    final_metrics = {
        "checkpoint_selection_score": 0.5,
        "checkpoint_sha256": "b" * 64,
        "epoch": 2,
        "is_best": 0,
        "world_size": 4,
    }
    (latest_directory / "history.json").write_text(json.dumps([
        best_metrics,
        final_metrics,
    ]))

    class FakeCheckpoint:
        def __init__(self, path, directory):
            self.path = path
            self.directory = directory

        def as_directory(self):
            return nullcontext(str(self.directory))

    latest_checkpoint = FakeCheckpoint(
        (
            "s3://checkpoints/ray-train/"
            "reactive-stage-test/checkpoint_0002"
        ),
        latest_directory,
    )
    best_checkpoint = FakeCheckpoint(
        (
            "s3://checkpoints/ray-train/"
            "reactive-stage-test/checkpoint_0001"
        ),
        latest_directory,
    )

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self):
            return SimpleNamespace(
                best_checkpoints=[(best_checkpoint, best_metrics)],
                checkpoint=latest_checkpoint,
                metrics=final_metrics,
            )

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray_train_torch, "TorchTrainer", FakeTrainer)
    config = _stage_config("nuplan_full")
    config["num_workers"] = 4
    config["worker_cpus"] = 3

    result = run_reactive_stage(config)

    assert captured["scaling_config"].resources_per_worker == {
        "CPU": 3,
        "GPU": 1,
    }
    assert captured["torch_config"].init_method == "tcp"
    assert (
        captured["torch_config"].timeout_s
        == REACTIVE_DDP_TIMEOUT_SECONDS
    )
    assert (
        type(captured["torch_config"]).__name__
        == "PreparedCudaTorchConfig"
    )
    assert (
        captured["run_config"].checkpoint_config.num_to_keep
        == config["epochs"] + 2
    )
    assert result["selected_epoch"] == 1
    assert result["metrics"]["checkpoint_sha256"] == "a" * 64


def test_prepared_cuda_backend_initializes_rank_environment(monkeypatch):
    for name in (
        "ACCELERATE_TORCH_DEVICE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "NODE_RANK",
        "RANK",
        "WORLD_SIZE",
    ):
        monkeypatch.setenv(name, "test-sentinel")
    context = SimpleNamespace(
        get_local_rank=lambda: 2,
        get_local_world_size=lambda: 8,
        get_node_rank=lambda: 0,
        get_world_rank=lambda: 2,
        get_world_size=lambda: 8,
    )
    device = ray_torch_backend.torch.device("cuda:0")
    calls = []

    monkeypatch.setattr(
        ray_torch_backend.train,
        "get_context",
        lambda: context,
    )
    monkeypatch.setattr(
        ray_torch_backend,
        "get_device",
        lambda: device,
    )
    monkeypatch.setattr(
        ray_torch_backend.torch.cuda,
        "set_device",
        lambda value: calls.append(("set_device", value)),
    )
    monkeypatch.setattr(
        ray_torch_backend.torch,
        "empty",
        lambda *args, **kwargs: calls.append(
            ("empty", args, kwargs)
        ),
    )
    monkeypatch.setattr(
        ray_torch_backend.torch.cuda,
        "synchronize",
        lambda value: calls.append(("synchronize", value)),
    )

    ray_torch_backend._prepare_torch_worker()

    assert ray_torch_backend.os.environ["LOCAL_RANK"] == "2"
    assert ray_torch_backend.os.environ["LOCAL_WORLD_SIZE"] == "8"
    assert ray_torch_backend.os.environ["NODE_RANK"] == "0"
    assert ray_torch_backend.os.environ["RANK"] == "2"
    assert ray_torch_backend.os.environ["WORLD_SIZE"] == "8"
    assert ray_torch_backend.os.environ["ACCELERATE_TORCH_DEVICE"] == (
        "cuda:0"
    )
    assert calls == [
        ("set_device", device),
        ("empty", (1,), {"device": device}),
        ("synchronize", device),
    ]


def test_prepared_cuda_backend_binds_nccl_device(monkeypatch):
    device = ray_torch_backend.torch.device("cuda:0")
    captured = {}

    monkeypatch.setattr(
        ray_torch_backend,
        "get_device",
        lambda: device,
    )
    monkeypatch.setattr(
        ray_torch_backend.dist,
        "init_process_group",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.delenv(
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        raising=False,
    )
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)

    ray_torch_backend._setup_prepared_torch_process_group(
        backend="nccl",
        world_rank=0,
        world_size=8,
        init_method="tcp://127.0.0.1:29500",
        timeout_s=300,
    )

    assert captured["backend"] == "nccl"
    assert captured["rank"] == 0
    assert captured["world_size"] == 8
    assert captured["device_id"] == device
    assert captured["timeout"].total_seconds() == 300
    assert (
        ray_torch_backend.os.environ[
            "TORCH_NCCL_ASYNC_ERROR_HANDLING"
        ]
        == "1"
    )
    assert ray_torch_backend.os.environ["NCCL_NVLS_ENABLE"] == "0"


def test_validate_stage_config_rejects_missing_worker_cpu_contract():
    config = _stage_config("nuplan_full")
    config.pop("worker_cpus")

    with pytest.raises(ValueError, match="worker_cpus"):
        validate_reactive_stage_config(config)


@pytest.mark.parametrize(
    ("stage", "expected_views", "expected_bev"),
    [
        (ReactiveTrainingStage.NUPLAN_FULL, 6, True),
        (ReactiveTrainingStage.L2D_CONTINUATION, 6, False),
    ],
)
def test_canary_dataset_uses_production_loader_contract(
    tmp_path,
    stage,
    expected_views,
    expected_bev,
):
    dataset = tmp_path / stage.value
    manifest = write_reactive_canary_dataset(
        dataset,
        stage=stage,
    )
    plan = build_reactive_dataset_plan(
        [str(dataset)],
        stage=stage,
    )
    train_loader = make_pre_extracted_loader(
        str(dataset),
        batch_size=1,
        num_workers=0,
        split="train",
        val_fraction=0.5,
        shuffle=0,
        decode_future_frames=False,
        nodesplitter=passthrough_nodesplitter,
    )
    train_batches = list(train_loader)
    validation_batches = list(make_pre_extracted_loader(
        str(dataset),
        batch_size=1,
        num_workers=0,
        split="val",
        val_fraction=0.5,
        shuffle=0,
        decode_future_frames=False,
        nodesplitter=passthrough_nodesplitter,
    ))

    assert manifest["total_samples"] == 6
    assert plan.total_samples == 6
    assert len(train_batches) == 4
    assert len(validation_batches) == 2
    sample = train_batches[0]
    assert sample["visual_tiles"].shape == (
        1,
        expected_views,
        3,
        REACTIVE_CAMERA_IMAGE_SIZE,
        REACTIVE_CAMERA_IMAGE_SIZE,
    )
    assert sample["map_context"].shape == (1, 14, 450, 300)
    assert sample["route_mask"].shape == (1, 2, 450, 300)
    assert bool(sample["bev_segmentation_available"][0]) is expected_bev
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        assert sample["front_camera_tile"].shape == (
            1,
            3,
            REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
            REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
        )
        statistics = discover_bev_training_statistics(
            [str(dataset)],
            val_fraction=0.5,
        )
        weights = derive_bev_pos_weights(statistics)
        assert len(weights) == 8
        assert all(value >= 1.0 for value in weights)
        assert statistics.positive_sample_count == (4,) * 8
    else:
        assert train_loader.geometry_type == "pinhole"
        assert train_loader.projection is not None
        assert "projection" in manifest
        with tarfile.open(dataset / manifest["shard_names"][0]) as archive:
            calibration_member = next(
                member
                for member in archive
                if member.name.endswith(".calib.json")
            )
            calibration_stream = archive.extractfile(calibration_member)
            assert calibration_stream is not None
            calibration = json.loads(calibration_stream.read())
        assert "projection" not in calibration


def test_t8_temporal_batch_norm_is_synchronized():
    torch = pytest.importorskip("torch")
    temporal_fusion = torch.nn.Sequential(
        torch.nn.Conv2d(4, 4, 1),
        torch.nn.BatchNorm2d(4),
        torch.nn.Sequential(torch.nn.BatchNorm2d(4)),
    )
    feature_fusion = SimpleNamespace(
        architecture="bevformer_v2_t8",
        temporal_fusion=temporal_fusion,
    )
    model = SimpleNamespace(
        Reactive_E2E=SimpleNamespace(FeatureFusion=feature_fusion),
    )
    for parameter in temporal_fusion.parameters():
        parameter.requires_grad_(False)

    identity, count = _configure_t8_temporal_normalization(
        model,
        freeze_bevformer=False,
    )

    assert identity == "sync_batch_norm_running_stats_v1"
    assert count == 2
    assert sum(
        isinstance(module, torch.nn.SyncBatchNorm)
        for module in feature_fusion.temporal_fusion.modules()
    ) == 2
    assert not any(
        type(module) is torch.nn.BatchNorm2d
        for module in feature_fusion.temporal_fusion.modules()
    )
    assert all(
        not parameter.requires_grad
        for parameter in feature_fusion.temporal_fusion.parameters()
    )


def test_frozen_t8_temporal_batch_norm_is_not_synchronized():
    torch = pytest.importorskip("torch")
    temporal_fusion = torch.nn.Sequential(
        torch.nn.Conv2d(4, 4, 1),
        torch.nn.BatchNorm2d(4),
    )
    feature_fusion = SimpleNamespace(
        architecture="bevformer_v2_t8",
        temporal_fusion=temporal_fusion,
    )
    model = SimpleNamespace(
        Reactive_E2E=SimpleNamespace(FeatureFusion=feature_fusion),
    )

    identity, count = _configure_t8_temporal_normalization(
        model,
        freeze_bevformer=True,
    )

    assert identity == "frozen_pretrained_running_stats_v1"
    assert count == 0
    assert sum(
        isinstance(module, torch.nn.SyncBatchNorm)
        for module in temporal_fusion.modules()
    ) == 0
    assert sum(
        type(module) is torch.nn.BatchNorm2d
        for module in temporal_fusion.modules()
    ) == 1


def test_non_t8_temporal_normalization_is_not_applicable():
    feature_fusion = SimpleNamespace(
        architecture="bevformer_v2_t1",
        temporal_fusion=None,
    )
    model = SimpleNamespace(
        Reactive_E2E=SimpleNamespace(FeatureFusion=feature_fusion),
    )

    assert _configure_t8_temporal_normalization(
        model,
        freeze_bevformer=True,
    ) == ("not_applicable_v1", 0)


def test_camera_feature_scale_diagnostics_are_normalized():
    torch = pytest.importorskip("torch")
    model = SimpleNamespace(
        Reactive_E2E=SimpleNamespace(
            FeatureFusion=SimpleNamespace(
                scale_logits=torch.tensor([0.0, 1.0, 2.0, 3.0]),
            ),
        ),
    )

    weights = _camera_feature_scale_weights(model)

    assert weights == pytest.approx(
        torch.tensor([0.0, 1.0, 2.0, 3.0]).softmax(0).tolist()
    )
    assert sum(weights) == pytest.approx(1.0)

    model.Reactive_E2E.FeatureFusion.scale_logits[0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        _camera_feature_scale_weights(model)

    bevformer = SimpleNamespace(
        Reactive_E2E=SimpleNamespace(
            FeatureFusion=SimpleNamespace(
                architecture="bevformer_v2_t1",
                view_fusion=SimpleNamespace(num_levels=4),
            ),
        ),
    )
    assert _camera_feature_scale_weights(bevformer) == (0.25,) * 4


def test_bev_statistics_all_reduce_preserves_vector_offsets(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    local = BEVTrainingStatistics(
        sample_count=2,
        effective_exposure_count=10,
        active_sample_count=(10,) * 8,
        positive_sample_count=tuple(range(1, 9)),
        positive_cell_count=tuple(range(11, 19)),
        positive_fraction_sum=tuple(
            (index + 1) / 10.0 for index in range(8)
        ),
        positive_mass=tuple(index + 0.5 for index in range(21, 29)),
        valid_cell_count=tuple(range(101, 109)),
        exposure_digest="a" * 64,
    )

    def all_reduce(tensor, op):
        assert op == dist.ReduceOp.SUM
        tensor.mul_(2)

    def all_gather_object(output, _value):
        output[:] = ["a" * 64, "b" * 64]

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    combined = _all_reduce_bev_statistics(local, torch.device("cpu"))

    assert combined.sample_count == 4
    assert combined.effective_exposure_count == 20
    assert combined.active_sample_count == (20,) * 8
    assert combined.positive_sample_count == tuple(
        2 * value for value in range(1, 9)
    )
    assert combined.positive_cell_count == tuple(
        2 * value for value in range(11, 19)
    )
    assert combined.positive_fraction_sum == pytest.approx(tuple(
        2 * (index + 1) / 10.0 for index in range(8)
    ))
    assert combined.positive_mass == pytest.approx(tuple(
        2 * (index + 0.5) for index in range(21, 29)
    ))
    assert combined.valid_cell_count == tuple(
        2 * value for value in range(101, 109)
    )


def test_histogram_average_precision_matches_hand_calculation():
    torch = pytest.importorskip("torch")

    average_precision = _histogram_average_precision(
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 0.0]),
    )

    assert average_precision == pytest.approx(5.0 / 6.0)


def test_histogram_best_iou_operating_point_prefers_clean_threshold():
    torch = pytest.importorskip("torch")

    threshold, iou, precision, recall = (
        _histogram_best_iou_operating_point(
            torch.tensor([0.0, 0.0, 1.0, 2.0]),
            torch.tensor([3.0, 1.0, 0.0, 0.0]),
        )
    )

    assert threshold == pytest.approx(0.5)
    assert iou == 1.0
    assert precision == 1.0
    assert recall == 1.0


def test_bev_only_optimizer_uses_discriminative_rates_and_no_decay_norms():
    torch = pytest.importorskip("torch")

    class ReactiveModules(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Backbone = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, 1),
                torch.nn.BatchNorm2d(4),
            )
            self.FeatureFusion = torch.nn.Sequential(
                torch.nn.Conv2d(4, 4, 1),
                torch.nn.GroupNorm(1, 4),
            )
            self.FeatureFusion.add_module(
                "bev_queries",
                torch.nn.Embedding(4, 4),
            )
            self.FeatureFusion.register_parameter(
                "camera_embeddings",
                torch.nn.Parameter(torch.ones(2, 4)),
            )
            self.FeatureFusion.register_parameter(
                "level_embeddings",
                torch.nn.Parameter(torch.ones(2, 4)),
            )
            self.BEVSegmentationHead = torch.nn.Conv2d(4, 8, 1)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Reactive_E2E = ReactiveModules()

    model = Model()
    groups = _reactive_optimizer_parameter_groups(
        model,
        training_scope=ReactiveTrainingScope.BEV_ONLY,
        learning_rate=1e-4,
        bev_encoder_learning_rate=1e-5,
        weight_decay=1e-2,
    )

    assert {group["name"] for group in groups} == {
        "bev_encoder_decay",
        "bev_encoder_no_decay",
        "bev_head_decay",
        "bev_head_no_decay",
    }
    assert {
        group["lr"]
        for group in groups
        if group["name"].startswith("bev_encoder")
    } == {1e-5}
    assert {
        group["lr"]
        for group in groups
        if group["name"].startswith("bev_head")
    } == {1e-4}
    assert {
        group["weight_decay"]
        for group in groups
        if group["name"].endswith("no_decay")
    } == {0.0}
    no_decay_ids = {
        id(parameter)
        for group in groups
        if group["name"].endswith("no_decay")
        for parameter in group["params"]
    }
    assert id(model.Reactive_E2E.FeatureFusion.bev_queries.weight) in (
        no_decay_ids
    )
    assert id(model.Reactive_E2E.FeatureFusion.camera_embeddings) in (
        no_decay_ids
    )
    assert id(model.Reactive_E2E.FeatureFusion.level_embeddings) in (
        no_decay_ids
    )
    assigned = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }


def test_bev_checkpoint_class_guard_requires_every_class():
    validation = {"bev_min_ap_lift": 0.01}
    for class_name in BEV_SEGMENTATION_CLASSES:
        validation[f"bev_{class_name}_supported"] = 1.0
        validation[
            f"bev_{class_name}_ap_lift_bootstrap_lower_95"
        ] = 0.001
        validation[f"bev_{class_name}_positive_prevalence"] = 0.01
        validation[
            f"bev_{class_name}_best_iou_on_validation_set"
        ] = 0.1
        validation[
            f"bev_{class_name}_best_iou_precision_on_validation_set"
        ] = 0.2
        validation[
            f"bev_{class_name}_best_iou_recall_on_validation_set"
        ] = 0.3

    assert _bev_checkpoint_class_guard(validation)

    validation[
        "bev_other_obstacle_ap_lift_bootstrap_lower_95"
    ] = -0.001
    assert not _bev_checkpoint_class_guard(validation)

    validation[
        "bev_other_obstacle_ap_lift_bootstrap_lower_95"
    ] = 0.001
    validation[
        "bev_other_obstacle_best_iou_precision_on_validation_set"
    ] = 0.01
    assert not _bev_checkpoint_class_guard(validation)

    validation[
        "bev_other_obstacle_best_iou_precision_on_validation_set"
    ] = 0.149
    assert not _bev_checkpoint_class_guard(validation)

    validation[
        "bev_other_obstacle_best_iou_precision_on_validation_set"
    ] = 0.2
    validation[
        "bev_other_obstacle_best_iou_on_validation_set"
    ] = 0.099
    assert not _bev_checkpoint_class_guard(validation)


def test_weighted_bev_prior_bias_matches_bce_constant_optimum():
    statistics = BEVTrainingStatistics(
        sample_count=10,
        effective_exposure_count=10,
        active_sample_count=(10,) * 8,
        positive_sample_count=(10,) * 8,
        positive_cell_count=(10,) * 8,
        positive_fraction_sum=(1.0,) * 8,
        positive_mass=(1.0,) * 8,
        valid_cell_count=(100,) * 8,
        exposure_digest="a" * 64,
    )

    biases = _weighted_bev_prior_logit_biases(
        statistics,
        (9.0,) * 8,
    )

    expected_probability = 0.5
    expected_bias = math.log(
        expected_probability / (1.0 - expected_probability)
    )
    assert biases == pytest.approx((expected_bias,) * 8)


def test_bev_weighting_uses_sample_normalized_prevalence():
    sample_prevalence = (0.5 + 1.0 / 64.0) / 2.0
    statistics = BEVTrainingStatistics(
        sample_count=2,
        effective_exposure_count=2,
        active_sample_count=(2,) * 8,
        positive_sample_count=(2,) * 8,
        positive_cell_count=(3,) * 8,
        positive_fraction_sum=(
            2.0 * sample_prevalence,
        ) * 8,
        positive_mass=(3.0,) * 8,
        valid_cell_count=(68,) * 8,
        exposure_digest="a" * 64,
    )

    pos_weights = derive_bev_pos_weights(
        statistics,
        max_weight=64.0,
    )
    expected_pos_weight = (
        1.0 - sample_prevalence
    ) / sample_prevalence
    assert pos_weights == pytest.approx((expected_pos_weight,) * 8)
    assert pos_weights[0] != pytest.approx((68.0 - 3.0) / 3.0)

    biases = _weighted_bev_prior_logit_biases(
        statistics,
        (8.0,) * 8,
    )
    expected_probability = (
        8.0 * sample_prevalence
        / (1.0 - sample_prevalence + 8.0 * sample_prevalence)
    )
    expected_bias = math.log(
        expected_probability / (1.0 - expected_probability)
    )
    assert biases == pytest.approx((expected_bias,) * 8)


def test_bev_gradient_budget_weights_equalize_capped_bce_mass():
    prevalence = (
        0.35,
        0.05,
        0.17,
        0.014,
        0.01,
        0.019,
        0.001,
        0.0008,
    )
    valid_count = 1_000_000
    statistics = BEVTrainingStatistics(
        sample_count=100,
        effective_exposure_count=100,
        active_sample_count=(100,) * 8,
        positive_sample_count=(100,) * 8,
        positive_cell_count=tuple(
            int(value * valid_count) for value in prevalence
        ),
        positive_fraction_sum=tuple(
            value * 100 for value in prevalence
        ),
        positive_mass=tuple(
            value * valid_count for value in prevalence
        ),
        valid_cell_count=(valid_count,) * 8,
        exposure_digest="a" * 64,
    )
    pos_weights = derive_bev_pos_weights(statistics, max_weight=64.0)
    class_weights = derive_bev_gradient_budget_weights(
        statistics,
        pos_weights,
    )
    probability = np.asarray(prevalence)
    positive_weight = np.asarray(pos_weights)
    weighted_prior = (
        positive_weight * probability
        / (
            1.0
            - probability
            + positive_weight * probability
        )
    )
    gradient_mass = (
        positive_weight * probability * (1.0 - weighted_prior)
        + (1.0 - probability) * weighted_prior
    ) * np.asarray(class_weights)

    assert np.mean(class_weights) == pytest.approx(1.0)
    assert class_weights[6] > 1.0
    assert class_weights[7] > class_weights[6]
    assert gradient_mass.max() / gradient_mass.min() < 1.15


def test_bev_gradient_budget_weights_include_class_activity_rate():
    statistics = BEVTrainingStatistics(
        sample_count=100,
        effective_exposure_count=100,
        active_sample_count=(25, 100, 100, 100, 100, 100, 100, 100),
        positive_sample_count=(10,) * 8,
        positive_cell_count=(100,) * 8,
        positive_fraction_sum=(2.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        positive_mass=(100.0,) * 8,
        valid_cell_count=(1_000,) * 8,
        exposure_digest="a" * 64,
    )

    class_weights = derive_bev_gradient_budget_weights(
        statistics,
        (9.0,) * 8,
        max_relative_weight=16.0,
    )

    assert class_weights[0] == pytest.approx(
        4.0 * class_weights[1]
    )


def test_bev_scheduler_warms_up_and_cosine_decays_per_step():
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)

    identity, scheduler = _build_reactive_scheduler(
        optimizer,
        training_scope=ReactiveTrainingScope.BEV_ONLY,
        total_optimizer_steps=100,
    )
    learning_rates = [float(optimizer.param_groups[0]["lr"])]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))

    assert identity == "bev_linear_warmup_cosine_v1"
    assert learning_rates[0] == pytest.approx(5e-5)
    assert max(learning_rates) == pytest.approx(1e-4)
    assert learning_rates[-1] == pytest.approx(1e-5)
    assert all(
        earlier >= later
        for earlier, later in zip(
            learning_rates[2:],
            learning_rates[3:],
        )
    )


def test_reactive_dataset_split_metrics_report_actual_sample_ratio():
    metrics = _reactive_dataset_split_metrics(
        total_samples=131_072,
        train_split_sample_count=121_750,
        validation_evaluated_sample_count=1_024,
        configured_validation_fraction=0.1,
    )

    assert metrics["validation_pool_sample_count"] == 9_322
    assert metrics["configured_train_fraction"] == pytest.approx(0.9)
    assert metrics["actual_train_fraction"] == pytest.approx(
        0.9288787841796875
    )
    assert metrics["actual_validation_fraction"] == pytest.approx(
        0.0711212158203125
    )
    assert metrics[
        "validation_evaluated_fraction_of_pool"
    ] == pytest.approx(1_024 / 9_322)


def test_lane_range_masks_partition_physical_bev():
    torch = pytest.importorskip("torch")
    geometry = AUTOE2E_NAVIGATION_GEOMETRY

    masks = _bev_lane_range_masks(
        geometry.height_px,
        geometry.width_px,
        device=torch.device("cpu"),
    )
    ego_row = round(geometry.ego_anchor_row)
    ego_col = round(geometry.ego_anchor_col)

    assert masks.shape == (2, geometry.height_px, geometry.width_px)
    assert bool(masks[0, ego_row, ego_col])
    assert not bool(masks[1, ego_row, ego_col])
    assert not bool((masks[0] & masks[1]).any())
    assert bool((masks[0] | masks[1]).all())
    assert BEV_LANE_NEAR_RADIUS_M == pytest.approx(30.0)


def test_route_validation_statistics_are_batch_and_mask_independent():
    torch = pytest.importorskip("torch")
    target = torch.zeros(2, 2, 3, 5)
    target[0, 0, 1, 1:4] = 1.0
    target[1, 1, 2, 4] = 1.0
    logits = torch.full_like(target, -20.0)
    logits[0, 0, 1, 1:4] = 20.0
    logits[1, 1, 0, 1] = 20.0
    channel_valid = torch.tensor([
        [True, False],
        [False, True],
    ])
    route_valid = torch.ones(2, dtype=torch.bool)
    route_loss = RouteReconstructionLoss()

    combined = _route_validation_statistics(
        logits,
        target,
        channel_valid,
        route_valid,
        route_loss,
    )
    split = sum(
        (
            _route_validation_statistics(
                logits[index:index + 1],
                target[index:index + 1],
                channel_valid[index:index + 1],
                route_valid[index:index + 1],
                route_loss,
            )
            for index in range(2)
        ),
        torch.zeros_like(combined),
    )

    torch.testing.assert_close(combined, split)
    assert combined[1].item() == 2
    assert combined[4].item() == 3
    assert combined[5].item() == 0
    assert combined[6].item() == 0
    assert combined[7].item() == 1
    assert combined[9].item() == pytest.approx(13**0.5)
    assert combined[10].item() == 1


def test_route_validation_statistics_empty_case_is_finite():
    torch = pytest.importorskip("torch")
    logits = torch.full((2, 2, 3, 5), float("nan"))
    target = torch.zeros_like(logits)
    channel_valid = torch.zeros(2, 2, dtype=torch.bool)
    route_valid = torch.zeros(2, dtype=torch.bool)

    statistics = _route_validation_statistics(
        logits,
        target,
        channel_valid,
        route_valid,
        RouteReconstructionLoss(),
    )

    assert torch.equal(statistics, torch.zeros_like(statistics))
    assert torch.isfinite(statistics).all()
    json.dumps(statistics.tolist(), allow_nan=False)


def test_route_validation_statistics_flags_validity_drift():
    torch = pytest.importorskip("torch")

    statistics = _route_validation_statistics(
        torch.zeros(1, 2, 3, 5),
        torch.zeros(1, 2, 3, 5),
        torch.tensor([[True, False]]),
        torch.tensor([False]),
        RouteReconstructionLoss(),
    )

    assert torch.equal(statistics[:14], torch.zeros(14))
    assert torch.equal(statistics[14:18], torch.zeros(4))
    assert statistics[18].item() == 1.0
    assert statistics[19].item() == 0.0


def test_route_validation_statistics_flags_nonfinite_active_values():
    torch = pytest.importorskip("torch")
    logits = torch.zeros(1, 2, 3, 5)
    logits[0, 0, 1, 2] = float("nan")

    statistics = _route_validation_statistics(
        logits,
        torch.zeros_like(logits),
        torch.tensor([[True, False]]),
        torch.tensor([True]),
        RouteReconstructionLoss(),
    )

    assert torch.equal(statistics[:19], torch.zeros(19))
    assert statistics[19].item() == 1.0


def test_route_validation_statistics_excludes_ambiguous_destination():
    torch = pytest.importorskip("torch")
    target = torch.zeros(1, 2, 3, 5)
    target[0, 1, 2, 4] = 1.0

    statistics = _route_validation_statistics(
        torch.zeros_like(target),
        target,
        torch.tensor([[False, True]]),
        torch.tensor([True]),
        RouteReconstructionLoss(),
    )

    assert statistics[9].item() == 0.0
    assert statistics[10].item() == 1.0
    assert statistics[11].item() == 0.0
    assert statistics[12].item() == 0.0
    assert statistics[13].item() == 1.0


def test_stage_b_validation_emits_route_metrics(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    geometry = AUTOE2E_NAVIGATION_GEOMETRY
    target = torch.zeros(
        1,
        2,
        geometry.height_px,
        geometry.width_px,
    )
    target[0, 0, 1, 1:4] = 1.0
    target[0, 1, 2, 4] = 1.0
    logits = torch.full_like(target, -20.0)
    logits[0, 0, 1, 1:4] = 20.0
    logits[0, 1, 2, 4] = 20.0

    class RouteValidationModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_options = None

        def forward(self, visual_tiles, *args, **kwargs):
            self.forward_options = kwargs
            controls = visual_tiles.new_zeros(
                visual_tiles.shape[0],
                128,
            )
            return controls, {
                "route_reconstruction_logits": logits,
            }

    model = RouteValidationModel()
    batch = {
        "visual_tiles": torch.zeros(1, 1, 3, 2, 2),
        "map_context": torch.zeros(
            1,
            14,
            geometry.height_px,
            geometry.width_px,
        ),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": target,
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "route_channel_valid": torch.ones(1, 2, dtype=torch.bool),
        "trajectory_xy_m": torch.zeros(1, 64, 2),
        "trajectory_valid": torch.ones(1, 64, dtype=torch.bool),
        "initial_speed_mps": torch.zeros(1),
    }
    objective = SimpleNamespace(
        compute_route_reconstruction=True,
        route_loss=RouteReconstructionLoss(),
    )
    monkeypatch.setattr(dist, "all_reduce", lambda *_args, **_kwargs: None)

    metrics = _evaluate_global_reactive(
        model,
        [batch],
        objective,
        stage=ReactiveTrainingStage.L2D_CONTINUATION,
        device=torch.device("cpu"),
        probability_bins=8,
        ade_scale_m=5.0,
    )

    assert metrics["route_corridor_iou"] == 1.0
    assert metrics["route_destination_error_cells"] == 0.0
    assert metrics["route_supported_samples"] == 1.0
    assert metrics["route_destination_localized_samples"] == 1.0
    assert metrics["route_destination_ambiguous_samples"] == 0.0
    assert metrics["route_destination_logit_range"] == 40.0
    assert metrics["route_nonfinite_samples"] == 0.0
    assert metrics["selection_score"] == 1.0
    assert "bev_loss" not in metrics
    assert model.forward_options["return_auxiliary"] is True
    assert model.forward_options["compute_bev_segmentation"] is False
    assert model.forward_options["compute_route_reconstruction"] is True


def test_stage_a_validation_skips_disabled_route_decoder(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    geometry = AUTOE2E_NAVIGATION_GEOMETRY

    class CapacityValidationModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_options = None

        def forward(self, visual_tiles, *args, **kwargs):
            self.forward_options = kwargs
            controls = visual_tiles.new_zeros(
                visual_tiles.shape[0],
                128,
            )
            return controls, {
                "bev_segmentation_logits": visual_tiles.new_full(
                    (
                        visual_tiles.shape[0],
                        8,
                        geometry.height_px,
                        geometry.width_px,
                    ),
                    20.0,
                ),
            }

    model = CapacityValidationModel()
    batch = {
        "visual_tiles": torch.zeros(1, 1, 3, 2, 2),
        "map_context": torch.zeros(
            1,
            14,
            geometry.height_px,
            geometry.width_px,
        ),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": torch.zeros(
            1,
            2,
            geometry.height_px,
            geometry.width_px,
        ),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "route_channel_valid": torch.ones(1, 2, dtype=torch.bool),
        "trajectory_xy_m": torch.zeros(1, 64, 2),
        "trajectory_valid": torch.ones(1, 64, dtype=torch.bool),
        "initial_speed_mps": torch.zeros(1),
        "sample_uid": ["stage-a-validation"],
        "bev_segmentation_target": torch.ones(
            1,
            8,
            geometry.height_px,
            geometry.width_px,
        ),
        "bev_segmentation_valid": torch.ones(
            1,
            8,
            geometry.height_px,
            geometry.width_px,
            dtype=torch.bool,
        ),
    }
    objective = SimpleNamespace(
        compute_route_reconstruction=False,
        route_loss=RouteReconstructionLoss(),
        bev_loss=BEVSegmentationAuxiliaryLoss([1.0] * 8),
    )
    monkeypatch.setattr(dist, "all_reduce", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="native front camera"):
        _evaluate_global_reactive(
            model,
            [batch],
            objective,
            stage=ReactiveTrainingStage.NUPLAN_FULL,
            device=torch.device("cpu"),
            probability_bins=8,
            ade_scale_m=5.0,
        )

    projection = torch.zeros(1, 1, 3, 4)
    projection[:, :, 2, 0] = 1.0
    batch.update({
        "camera_projection_matrix": projection,
        "camera_geometry_type": "rectified_pinhole",
        "front_camera_tile": torch.zeros(1, 3, 4, 4),
        "front_camera_projection_matrix": projection.clone(),
        "camera_history_tiles": torch.zeros(1, 7, 1, 3, 2, 2),
        "camera_history_projection_matrix": (
            projection[:, None].repeat(1, 7, 1, 1, 1)
        ),
    })
    metrics = _evaluate_global_reactive(
        model,
        [batch],
        objective,
        stage=ReactiveTrainingStage.NUPLAN_FULL,
        device=torch.device("cpu"),
        probability_bins=8,
        ade_scale_m=5.0,
    )

    assert metrics["route_supported_samples"] == 0.0
    assert metrics["route_destination_localized_samples"] == 0.0
    for range_name in ("near", "far"):
        assert (
            metrics[
                f"bev_lane_boundary_{range_name}_average_precision"
            ]
            == 1.0
        )
        assert metrics[
            f"bev_lane_boundary_{range_name}_precision"
        ] == 1.0
        assert metrics[f"bev_lane_boundary_{range_name}_recall"] == 1.0
    assert metrics["bev_lane_boundary_positive_cells"] == pytest.approx(
        sum(
            metrics[
                f"bev_lane_boundary_{range_name}_positive_cells"
            ]
            for range_name in ("near", "far")
        )
    )
    assert metrics["bev_lane_boundary_valid_cells"] == pytest.approx(
        sum(
            metrics[f"bev_lane_boundary_{range_name}_valid_cells"]
            for range_name in ("near", "far")
        )
    )
    assert model.forward_options["return_auxiliary"] is True
    assert model.forward_options["compute_bev_segmentation"] is True
    assert model.forward_options["compute_route_reconstruction"] is False


def test_bev_only_validation_does_not_require_trajectory(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    class BEVOnlyValidationModel(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = logits
            self.forward_options = None

        def forward(self, visual_tiles, *args, **kwargs):
            self.forward_options = kwargs
            return visual_tiles.new_zeros(
                visual_tiles.shape[0],
                0,
            ), {
                "bev_segmentation_logits": self.logits,
            }

    target = torch.zeros(1, 8, 4, 4)
    for class_index in range(8):
        target[0, class_index, class_index // 4, class_index % 4] = 1.0
    logits = torch.where(
        target > 0.5,
        torch.full_like(target, 20.0),
        torch.full_like(target, -20.0),
    )
    model = BEVOnlyValidationModel(logits)
    projection = torch.zeros(1, 1, 3, 4)
    projection[:, :, 2, 0] = 1.0
    batch = {
        "visual_tiles": torch.zeros(1, 1, 3, 2, 2),
        "map_context": torch.zeros(1, 14, 4, 4),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": torch.zeros(1, 2, 4, 4),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "route_channel_valid": torch.ones(1, 2, dtype=torch.bool),
        "sample_uid": ["bev-only-validation"],
        "bev_segmentation_target": target,
        "bev_segmentation_valid": torch.ones_like(
            target,
            dtype=torch.bool,
        ),
        "camera_projection_matrix": projection,
        "camera_geometry_type": "rectified_pinhole",
        "front_camera_tile": torch.zeros(1, 3, 4, 4),
        "front_camera_projection_matrix": projection.clone(),
        "camera_history_tiles": torch.zeros(1, 7, 1, 3, 2, 2),
        "camera_history_projection_matrix": (
            projection[:, None].repeat(1, 7, 1, 1, 1)
        ),
    }
    objective = ReactiveMultitaskObjective(
        ReactiveTrainingStage.NUPLAN_FULL,
        bev_pos_weight=[1.0] * 8,
        trajectory_weight=0.0,
        bev_weight=1.0,
        route_weight=0.0,
        training_scope=ReactiveTrainingScope.BEV_ONLY,
    )
    monkeypatch.setattr(dist, "all_reduce", lambda *_args, **_kwargs: None)

    metrics = _evaluate_global_reactive(
        model,
        [batch],
        objective,
        stage=ReactiveTrainingStage.NUPLAN_FULL,
        device=torch.device("cpu"),
        probability_bins=8,
        ade_scale_m=5.0,
    )

    assert "ade_6p4s_m" not in metrics
    assert metrics["bev_all_classes_supported"] == 1.0
    assert metrics["selection_score"] == pytest.approx(1.0)
    for class_name in BEV_SEGMENTATION_CLASSES:
        assert metrics[f"bev_{class_name}_average_precision"] == 1.0
        assert (
            metrics[
                f"bev_{class_name}_best_iou_on_validation_set"
            ]
            == 1.0
        )
        assert metrics[f"bev_{class_name}_iou_at_0p5"] == 1.0
        assert metrics[f"bev_{class_name}_precision_at_0p5"] == 1.0
        assert metrics[f"bev_{class_name}_recall_at_0p5"] == 1.0
    assert metrics["bev_threshold_selection"] == (
        "same_validation_set_oracle"
    )
    assert metrics["bev_ap_bootstrap_version"] == (
        BEV_AP_BOOTSTRAP_VERSION
    )
    assert metrics["bev_ap_bootstrap_weight_scale"] == float(
        BEV_AP_BOOTSTRAP_WEIGHT_SCALE
    )
    assert metrics["bev_ap_bootstrap_max_weight"] == (
        BEV_AP_BOOTSTRAP_MAX_WEIGHT
    )
    assert model.forward_options["bev_only"] is True
    assert model.forward_options["compute_route_reconstruction"] is False


def test_result_checkpoint_selection_honors_ade_guard(tmp_path):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    history = [{"checkpoint_sha256": "a" * 64, "epoch": 1}]
    (directory / "history.json").write_text(json.dumps(history))
    checkpoint = SimpleNamespace(
        path="s3://checkpoints/ray-train/run/checkpoint_0001",
        as_directory=lambda: nullcontext(str(directory)),
    )
    rejected = {
        "checkpoint_selection_score": -1.0,
        "checkpoint_sha256": "b" * 64,
        "epoch": 2,
        "is_best": 0,
    }
    accepted = {
        "checkpoint_selection_score": 0.6,
        "checkpoint_sha256": "a" * 64,
        "epoch": 1,
        "is_best": 1,
    }
    result = SimpleNamespace(
        best_checkpoints=[
            (SimpleNamespace(path="rejected"), rejected),
            (checkpoint, accepted),
        ],
        checkpoint=SimpleNamespace(path="latest"),
        metrics=rejected,
    )

    selected, metrics = _select_result_checkpoint(result)

    assert selected is checkpoint
    assert metrics == accepted
    assert _checkpoint_history(checkpoint) == history


def test_result_checkpoint_selection_accepts_smoke_without_quality_gate():
    checkpoint = SimpleNamespace(path="smoke")
    metrics = {
        "checkpoint_selection_score": -0.2,
        "checkpoint_sha256": "a" * 64,
        "epoch": 1,
        "is_best": 1,
        "single_worker_smoke": 1,
    }
    result = SimpleNamespace(
        best_checkpoints=[(checkpoint, metrics)],
        checkpoint=checkpoint,
        metrics=metrics,
    )

    selected, selected_metrics = _select_result_checkpoint(result)

    assert selected is checkpoint
    assert selected_metrics == metrics


def test_result_checkpoint_selection_retains_best_failed_quality_epoch():
    lower_checkpoint = SimpleNamespace(path="epoch-1")
    higher_checkpoint = SimpleNamespace(path="epoch-2")
    common = {
        "checkpoint_kind": "epoch",
        "checkpoint_selection_score": -1.0,
        "is_best": 0,
        "single_worker_smoke": 0,
        "bounded_bev_canary": 0,
        "checkpoint_quality_guard_enforced": 1,
        "bev_checkpoint_class_guard_pass": 0,
        "bev_parent_promotion_eligible": 0,
    }
    lower = {
        **common,
        "checkpoint_sha256": "a" * 64,
        "epoch": 1,
        "validation_selection_score": 0.35,
    }
    higher = {
        **common,
        "checkpoint_sha256": "b" * 64,
        "epoch": 2,
        "validation_selection_score": 0.42,
    }
    result = SimpleNamespace(
        best_checkpoints=[
            (lower_checkpoint, lower),
            (higher_checkpoint, higher),
        ],
        checkpoint=higher_checkpoint,
        metrics=higher,
    )

    selected, selected_metrics = _select_result_checkpoint(result)

    assert selected is higher_checkpoint
    assert selected_metrics["selected_for_evaluation_only"] == 1
    assert selected_metrics["bev_parent_promotion_eligible"] == 0
    assert selected_metrics["checkpoint_sha256"] == "b" * 64


def test_result_checkpoint_selection_prefers_accepted_quality_epoch():
    accepted_checkpoint = SimpleNamespace(path="accepted")
    failed_checkpoint = SimpleNamespace(path="failed")
    accepted = {
        "checkpoint_kind": "epoch",
        "checkpoint_selection_score": 0.30,
        "validation_selection_score": 0.30,
        "checkpoint_sha256": "a" * 64,
        "epoch": 1,
        "is_best": 1,
        "single_worker_smoke": 0,
        "bounded_bev_canary": 0,
        "checkpoint_quality_guard_enforced": 1,
        "bev_checkpoint_class_guard_pass": 1,
        "bev_parent_promotion_eligible": 1,
    }
    failed = {
        **accepted,
        "checkpoint_selection_score": -1.0,
        "validation_selection_score": 0.50,
        "checkpoint_sha256": "b" * 64,
        "epoch": 2,
        "is_best": 0,
        "bev_checkpoint_class_guard_pass": 0,
        "bev_parent_promotion_eligible": 0,
    }
    result = SimpleNamespace(
        best_checkpoints=[
            (accepted_checkpoint, accepted),
            (failed_checkpoint, failed),
        ],
        checkpoint=failed_checkpoint,
        metrics=failed,
    )

    selected, selected_metrics = _select_result_checkpoint(result)

    assert selected is accepted_checkpoint
    assert selected_metrics == accepted


@pytest.mark.parametrize(
    "invalid_metrics",
    (
        {"checkpoint_kind": "step"},
        {"single_worker_smoke": 1},
        {"bounded_bev_canary": 1},
        {"checkpoint_quality_guard_enforced": 0},
        {"validation_selection_score": float("nan")},
    ),
)
def test_result_checkpoint_selection_rejects_invalid_evaluation_fallback(
    invalid_metrics,
):
    checkpoint = SimpleNamespace(path="invalid")
    metrics = {
        "checkpoint_kind": "epoch",
        "checkpoint_selection_score": -1.0,
        "validation_selection_score": 0.4,
        "checkpoint_sha256": "a" * 64,
        "epoch": 1,
        "is_best": 0,
        "single_worker_smoke": 0,
        "bounded_bev_canary": 0,
        "checkpoint_quality_guard_enforced": 1,
        "bev_checkpoint_class_guard_pass": 0,
        "bev_parent_promotion_eligible": 0,
        **invalid_metrics,
    }
    result = SimpleNamespace(
        best_checkpoints=[(checkpoint, metrics)],
        checkpoint=checkpoint,
        metrics=metrics,
    )

    with pytest.raises(RuntimeError, match="no accepted best checkpoint"):
        _select_result_checkpoint(result)


def test_resume_rejects_smoke_provenance_change(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    checkpoint = tmp_path / "smoke-resume"
    _write_resume_checkpoint(
        checkpoint,
        config={
            "distributed_world_size": 1,
            "single_worker_smoke": True,
        },
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    with pytest.raises(ValueError, match="smoke provenance differs"):
        _load_resume_checkpoint(
            str(checkpoint),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected={
                "distributed_world_size": 1,
                "single_worker_smoke": False,
            },
        )


def test_epoch_resume_allows_batch_change_but_step_resume_rejects_it(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    expected = {
        "distributed_global_batch": 32,
        "optimizer_steps_per_epoch": 4,
    }

    def write_checkpoint(kind):
        directory = tmp_path / kind
        directory.mkdir()
        torch.save(
            {
                "config": {
                    "distributed_global_batch": 8,
                    "optimizer_steps_per_epoch": 16,
                },
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "training_state": {
                    "checkpoint_kind": kind,
                    "optimizer_step_in_epoch": 1,
                    "step_checkpoint_version": (
                        REACTIVE_STEP_CHECKPOINT_VERSION
                    ),
                },
            },
            directory / "checkpoint.pt",
        )
        (directory / "history.json").write_text(
            json.dumps([{"epoch": 1}]),
            encoding="ascii",
        )
        return directory

    state = _load_resume_checkpoint(
        str(write_checkpoint("epoch")),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected=expected,
    )

    assert state.epoch == 2
    with pytest.raises(ValueError, match="resume contract differs"):
        _load_resume_checkpoint(
            str(write_checkpoint("step")),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected=expected,
        )


def test_legacy_frozen_bev_epoch_resume_allows_batch_and_epoch_extension(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    checkpoint = tmp_path / "legacy-epoch"
    checkpoint.mkdir()
    torch.save(
        {
            "config": {
                "bev_weight": 0.0,
                "distributed_global_batch": 32,
                "epochs": 3,
                "freeze_bevformer": True,
                "optimizer_steps_per_epoch": 3687,
                "route_weight": 1.0,
                "scheduler_identity": "selection_plateau_v1",
                "trajectory_weight": 1.0,
            },
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_state": {
                "checkpoint_kind": "epoch",
            },
        },
        checkpoint / "checkpoint.pt",
    )
    (checkpoint / "history.json").write_text(
        json.dumps([{"epoch": epoch} for epoch in range(1, 4)]),
        encoding="ascii",
    )

    state = _load_resume_checkpoint(
        str(checkpoint),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected={
            "bev_ap_bins": 1024,
            "bev_checkpoint_min_class_iou": 0.1,
            "bev_checkpoint_min_class_precision": 0.15,
            "bev_checkpoint_quality_guard_version": "guard-v1",
            "bev_class_weights": [1.0] * 8,
            "bev_encoder_learning_rate": None,
            "bev_head_initialization": "head-v1",
            "bev_loss_version": "loss-v1",
            "bev_positive_pair_frequencies": [1.0] * 8,
            "bev_rank_sampling_evidence": [],
            "bev_sampling_importance_correction": "sampling-v1",
            "bev_weight": 0.0,
            "bounded_bev_canary": False,
            "dataset_split_metrics": {},
            "distributed_global_batch": 16,
            "epochs": 10,
            "freeze_bevformer": True,
            "num_loader_workers": 2,
            "optimizer_identity": "adamw_v1",
            "optimizer_steps_per_epoch": 7374,
            "route_weight": 1.0,
            "sample_stream_digest_version": "stream-v1",
            "scheduler_identity": "selection_plateau_v1",
            "shuffle_buffer": 256,
            "step_checkpoint_version": "step-v3",
            "training_scope": ReactiveTrainingScope.MULTITASK.value,
            "trajectory_weight": 1.0,
            "validation_fraction": 0.1,
            "validation_positive_sample_counts": [1] * 8,
            "validation_sample_limit": 1024,
        },
    )

    assert state.epoch == 4
    assert [item["epoch"] for item in state.epoch_history] == [1, 2, 3]


def test_legacy_epoch_resume_does_not_ignore_active_contract_changes(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    checkpoint = tmp_path / "legacy-epoch"
    checkpoint.mkdir()
    torch.save(
        {
            "config": {
                "bev_weight": 0.0,
                "distributed_global_batch": 32,
                "epochs": 3,
                "freeze_bevformer": True,
                "optimizer_steps_per_epoch": 3687,
                "route_weight": 1.0,
                "scheduler_identity": "selection_plateau_v1",
                "trajectory_weight": 1.0,
            },
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_state": {
                "checkpoint_kind": "epoch",
            },
        },
        checkpoint / "checkpoint.pt",
    )
    (checkpoint / "history.json").write_text(
        json.dumps([{"epoch": epoch} for epoch in range(1, 4)]),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="route_weight"):
        _load_resume_checkpoint(
            str(checkpoint),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected={
                "bev_weight": 0.0,
                "distributed_global_batch": 16,
                "epochs": 10,
                "freeze_bevformer": True,
                "num_loader_workers": 2,
                "optimizer_identity": "adamw_v1",
                "optimizer_steps_per_epoch": 7374,
                "route_weight": 0.0,
                "scheduler_identity": "selection_plateau_v1",
                "shuffle_buffer": 256,
                "training_scope": ReactiveTrainingScope.MULTITASK.value,
                "trajectory_weight": 1.0,
            },
        )


def test_bev_fixed_step_scheduler_rejects_epoch_batch_change(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )
    directory = tmp_path / "epoch"
    directory.mkdir()
    torch.save(
        {
            "config": {
                "distributed_global_batch": 8,
                "optimizer_steps_per_epoch": 16,
                "scheduler_identity": "bev_linear_warmup_cosine_v1",
            },
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_state": {
                "checkpoint_kind": "epoch",
            },
        },
        directory / "checkpoint.pt",
    )
    (directory / "history.json").write_text(
        json.dumps([{"epoch": 1}]),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="resume contract differs"):
        _load_resume_checkpoint(
            str(directory),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected={
                "distributed_global_batch": 32,
                "optimizer_steps_per_epoch": 4,
                "scheduler_identity": "bev_linear_warmup_cosine_v1",
            },
        )


def test_bev_repeat_factors_are_frequency_aware_and_clipped():
    positive_samples = (100, 25, 4, 1, 100, 25, 4, 1)
    statistics = BEVTrainingStatistics(
        sample_count=100,
        effective_exposure_count=100,
        active_sample_count=(100,) * 8,
        positive_sample_count=positive_samples,
        positive_cell_count=positive_samples,
        positive_fraction_sum=tuple(
            value / 100.0 for value in positive_samples
        ),
        positive_mass=tuple(float(value) for value in positive_samples),
        valid_cell_count=(100,) * 8,
        exposure_digest="a" * 64,
    )

    assert derive_bev_repeat_factors(
        statistics,
        frequency_threshold=0.25,
        max_repeat=4,
    ) == (1, 1, 3, 4, 1, 1, 3, 4)


def test_bev_repeat_factors_do_not_repeat_common_small_objects():
    statistics = BEVTrainingStatistics(
        sample_count=100,
        effective_exposure_count=100,
        active_sample_count=(100,) * 8,
        positive_sample_count=(90,) * 8,
        positive_cell_count=(90,) * 8,
        positive_fraction_sum=(1.0,) * 8,
        positive_mass=(1.0,) * 8,
        valid_cell_count=(100_000,) * 8,
        exposure_digest="a" * 64,
    )

    assert derive_bev_repeat_factors(
        statistics,
        frequency_threshold=0.05,
        max_repeat=4,
    ) == (1,) * 8


def test_bev_positive_pair_frequencies_use_raw_sample_support():
    statistics = BEVTrainingStatistics(
        sample_count=100,
        effective_exposure_count=132,
        active_sample_count=(132,) * 8,
        positive_sample_count=(100, 80, 60, 40, 20, 10, 4, 1),
        positive_cell_count=(100,) * 8,
        positive_fraction_sum=(10.0,) * 8,
        positive_mass=(100.0,) * 8,
        valid_cell_count=(1_000,) * 8,
        exposure_digest="a" * 64,
    )

    assert derive_bev_positive_pair_frequencies(
        statistics
    ) == pytest.approx(
        (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.04, 0.01)
    )


def test_bev_validation_support_is_checked_before_training():
    records = [
        SimpleNamespace(
            sample_uid="sample-a",
            positive_cell_count=(1, 0, 1, 0, 1, 0, 1, 0),
        ),
        SimpleNamespace(
            sample_uid="sample-b",
            positive_cell_count=(0, 1, 0, 1, 0, 1, 0, 1),
        ),
    ]

    assert _bev_validation_positive_sample_counts(
        records,
        ("sample-a", "sample-b"),
    ) == (1,) * len(BEV_SEGMENTATION_CLASSES)

    with pytest.raises(ValueError, match="missing selected samples"):
        _bev_validation_positive_sample_counts(
            records,
            ("sample-a", "missing"),
        )


def test_distributed_bev_validation_reproduces_rank_local_limits():
    records_by_rank = []
    for rank in range(2):
        records_by_rank.append(tuple(
            BEVSampleStatistics(
                sample_uid=f"rank-{rank}-sample-{index}",
                split_group_uid="group-8",
                positive_cell_count=(1, 0, 0, 0, 0, 0, 0, 0),
                positive_mass=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                valid_cell_count=(4,) * 8,
            )
            for index in range(2)
        ))

    selected = select_distributed_bev_validation_sample_uids(
        records_by_rank,
        val_fraction=0.1,
        sample_limit=3,
    )

    assert selected == (
        "rank-0-sample-0",
        "rank-0-sample-1",
        "rank-1-sample-0",
        "rank-1-sample-1",
    )


def test_bev_validation_holdout_excludes_calibration_split_groups():
    records = (
        BEVSampleStatistics(
            sample_uid="calibration",
            split_group_uid="group-8",
            positive_cell_count=(1,) * 8,
            positive_mass=(1.0,) * 8,
            valid_cell_count=(4,) * 8,
        ),
        BEVSampleStatistics(
            sample_uid="adjacent-frame",
            split_group_uid="group-8",
            positive_cell_count=(1,) * 8,
            positive_mass=(1.0,) * 8,
            valid_cell_count=(4,) * 8,
        ),
        BEVSampleStatistics(
            sample_uid="independent-holdout",
            split_group_uid="group-17",
            positive_cell_count=(1,) * 8,
            positive_mass=(1.0,) * 8,
            valid_cell_count=(4,) * 8,
        ),
    )

    assert select_bev_validation_holdout_sample_uids(
        records,
        val_fraction=0.1,
        excluded_sample_uids=("calibration",),
    ) == ("independent-holdout",)


def test_bev_sample_statistics_can_scan_explicit_rank_shards(tmp_path):
    def write_shard(path, sample_uid):
        target = np.zeros((8, 2, 2), dtype=np.float32)
        target[0, 0, 0] = 1.0
        members = {
            f"{sample_uid}.meta.json": json.dumps({
                "sample_uid": sample_uid,
                "split_group_uid": "group-8",
            }).encode("ascii"),
            f"{sample_uid}.{BEV_SEGMENTATION_STATS_MEMBER}": (
                encode_bev_segmentation_stats(
                    target,
                    np.ones_like(target, dtype=np.bool_),
                )
            ),
        }
        with tarfile.open(path, "w") as archive:
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    shard_a = tmp_path / "a.tar"
    shard_b = tmp_path / "b.tar"
    write_shard(shard_a, "sample-a")
    write_shard(shard_b, "sample-b")

    records = discover_bev_sample_statistics(
        [tmp_path],
        shard_files=[shard_b],
    )

    assert [record.sample_uid for record in records] == ["sample-b"]


def test_bev_statistics_reject_fully_invalid_samples():
    target = np.zeros((8, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one valid cell"):
        encode_bev_segmentation_stats(
            target,
            np.zeros_like(target, dtype=np.bool_),
        )


def test_bev_repeat_policy_preserves_non_bev_importance_mass():
    target = np.zeros((8, 2, 2), dtype=np.float32)
    target[3, 0, 0] = 1.0
    payload = encode_bev_segmentation_stats(
        target,
        np.ones_like(target, dtype=np.bool_),
    )
    policy = BEVClassRepeatPolicy(
        repeat_factors=(1, 1, 1, 4, 1, 1, 1, 1),
        importance_scale=2.0,
    )

    repeated = list(policy([{
        BEV_SEGMENTATION_STATS_MEMBER: payload,
        "sample": "rare",
    }]))

    assert len(repeated) == 4
    assert {
        item["__bev_repeat_factor__"] for item in repeated
    } == {4}
    assert sum(
        float(item["__bev_sampling_importance__"])
        for item in repeated
    ) == pytest.approx(2.0)


def test_rank_corrected_bev_importance_recovers_global_raw_mean():
    global_values = (
        ((1.0, 1), (3.0, 3)),
        ((5.0, 1),),
    )
    global_sample_count = sum(len(values) for values in global_values)
    rank_losses = []
    for rank_values in global_values:
        effective_count = sum(repeat for _, repeat in rank_values)
        importance_scale = derive_bev_rank_importance_scale(
            local_effective_exposure_count=effective_count,
            global_sample_count=global_sample_count,
            world_size=len(global_values),
        )
        weighted_exposures = []
        for value, repeat in rank_values:
            weighted_exposures.extend(
                [value * importance_scale / repeat] * repeat
            )
        rank_losses.append(
            sum(weighted_exposures) / len(weighted_exposures)
        )

    assert sum(rank_losses) / len(rank_losses) == pytest.approx(3.0)


def test_bev_rank_capacity_counts_drop_last_per_directory():
    def records(prefix, count):
        return tuple(
            BEVSampleStatistics(
                sample_uid=f"{prefix}-{index}",
                split_group_uid=f"{prefix}-group-{index}",
                positive_cell_count=(1,) * 8,
                positive_mass=(1.0,) * 8,
                valid_cell_count=(4,) * 8,
            )
            for index in range(count)
        )

    capacity = bev_rank_full_microbatch_capacity(
        (records("a", 5), records("b", 3)),
        val_fraction=0.1,
        repeat_factors=(1,) * 8,
        batch_size=4,
    )

    assert capacity == 1


def test_bev_rank_capacity_ignores_validation_only_directory():
    train_record = BEVSampleStatistics(
        sample_uid="train",
        split_group_uid="group-0",
        positive_cell_count=(1,) * 8,
        positive_mass=(1.0,) * 8,
        valid_cell_count=(4,) * 8,
    )
    validation_record = BEVSampleStatistics(
        sample_uid="validation",
        split_group_uid="group-8",
        positive_cell_count=(1,) * 8,
        positive_mass=(1.0,) * 8,
        valid_cell_count=(4,) * 8,
    )

    capacity = bev_rank_full_microbatch_capacity(
        ((validation_record,), (train_record,) * 4),
        val_fraction=0.1,
        repeat_factors=(1,) * 8,
        batch_size=4,
    )

    assert capacity == 1
