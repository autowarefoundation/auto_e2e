"""Distributed Reactive dataset and fixed-step contracts."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import re
import tarfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import distributed_training.ray_torch_backend as ray_torch_backend
import distributed_training.reactive_stage as reactive_stage_module
from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS
from data_parsing.pre_extracted import (
    BEVClassRepeatPolicy,
    BEVTrainingStatistics,
    derive_bev_pos_weights,
    derive_bev_repeat_factors,
    discover_bev_training_statistics,
    make_pre_extracted_loader,
    passthrough_nodesplitter,
)
from data_processing.reactive_training_artifacts import (
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
    BEV_LANE_NEAR_RADIUS_M,
    REACTIVE_STEP_CHECKPOINT_VERSION,
    ReactiveResumeState,
    expected_reactive_hostname_count,
    _all_reduce_bev_statistics,
    _bev_lane_range_masks,
    _camera_feature_scale_weights,
    _capture_rng_state,
    _checkpoint_history,
    _checkpoint_step_due,
    _histogram_average_precision,
    _evaluate_global_reactive,
    _load_resume_checkpoint,
    _rank_resume_value,
    _rank_resume_value_for_epoch,
    _replay_loader_position,
    _resume_completed_requested_epochs,
    _restore_rng_state,
    _route_validation_statistics,
    _select_result_checkpoint,
    _synchronize_gradient_micro_step,
    _synchronize_t8_temporal_batch_norm,
    _train_fixed_steps,
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
from training.reactive_multitask import ReactiveTrainingStage


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


def test_reactive_ddp_uses_static_graph_for_reentrant_checkpoints():
    source = inspect.getsource(train_loop_per_worker)
    fixed_step_source = inspect.getsource(_train_fixed_steps)
    synchronization_source = inspect.getsource(
        _synchronize_gradient_micro_step
    )

    assert '"find_unused_parameters": False' in source
    assert '"static_graph": True' in source
    assert "_synchronize_gradient_micro_step" in fixed_step_source
    assert "optimizer_step_index == 0" in synchronization_source


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
                    model(value).square().mean().backward()
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


def test_fixed_step_resume_matches_uninterrupted_training(monkeypatch):
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
            "visual_tiles": torch.full((1, 1, 1, 1, 1), value),
            "map_context": torch.zeros(1, 1, 1, 1),
            "visual_history": torch.zeros(1, 1),
            "egomotion_history": torch.zeros(1, 1),
            "route_mask": torch.zeros(1, 1, 1, 1),
            "map_valid": torch.ones(1, dtype=torch.bool),
            "route_valid": torch.ones(1, dtype=torch.bool),
        }
        for value in (1.0, 2.0)
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

    assert resumed_model.weight.item() == pytest.approx(
        uninterrupted_model.weight.item()
    )
    assert resumed_metrics == pytest.approx(uninterrupted_metrics)


def test_replay_loader_position_checks_samples_and_restarts():
    batch = {
        "visual_tiles": np.zeros((2, 1, 1, 1, 1)),
    }
    iterator = RestartingIterator([batch])

    _replay_loader_position(
        iterator,
        skipped_micro_steps=2,
        expected_restarts=1,
        expected_samples=4,
    )

    with pytest.raises(ValueError, match="loader position"):
        _replay_loader_position(
            RestartingIterator([batch]),
            skipped_micro_steps=2,
            expected_restarts=1,
            expected_samples=3,
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
        "training_seed": 149,
        "trajectory_weight": 1.0,
        "val_fraction": 0.1,
        "validation_sample_limit": 1024,
        "weight_decay": 0.01,
        "worker_cpus": 3,
    }


@pytest.mark.parametrize("stage", ["nuplan_full", "l2d_continuation"])
def test_validate_stage_config_accepts_locked_program(stage):
    validate_reactive_stage_config(_stage_config(stage))


@pytest.mark.parametrize(
    ("world_size", "hostname_count"),
    ((2, 2), (4, 1), (8, 1)),
)
def test_expected_reactive_hostname_count_matches_ray_topology(
    world_size,
    hostname_count,
):
    assert expected_reactive_hostname_count(world_size) == hostname_count


def test_validate_stage_config_rejects_parent_and_batch_contract_changes():
    stage_a = _stage_config("nuplan_full")
    stage_a["parent_checkpoint_uri"] = "s3://checkpoints/parent.pt"
    with pytest.raises(ValueError, match="Stage A"):
        validate_reactive_stage_config(stage_a)

    stage_b = _stage_config("l2d_continuation")
    stage_b["per_rank_batch_size"] = 2
    with pytest.raises(ValueError, match="per_rank_batch_size"):
        validate_reactive_stage_config(stage_b)

    caller_weighted = _stage_config("nuplan_full")
    caller_weighted["bev_pos_weights"] = [1.0] * 8
    with pytest.raises(ValueError, match="derived"):
        validate_reactive_stage_config(caller_weighted)

    invalid_interval = _stage_config("nuplan_full")
    invalid_interval["checkpoint_interval_steps"] = 0
    with pytest.raises(ValueError, match="checkpoint_interval_steps"):
        validate_reactive_stage_config(invalid_interval)


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
            "consumed_samples": 3,
            "gradient_totals": [0.0] * 8,
            "loader_restarts": 0,
            "rank": rank,
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
    assert captured["torch_config"].timeout_s == 300
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

    count = _synchronize_t8_temporal_batch_norm(model)

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
        effective_exposure_count=3,
        positive_sample_count=tuple(range(1, 9)),
        positive_cell_count=tuple(range(11, 19)),
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
    assert combined.effective_exposure_count == 6
    assert combined.positive_sample_count == tuple(
        2 * value for value in range(1, 9)
    )
    assert combined.positive_cell_count == tuple(
        2 * value for value in range(11, 19)
    )
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

    target = torch.zeros(1, 2, 3, 5)
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
        "map_context": torch.zeros(1, 14, 3, 5),
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
                    (visual_tiles.shape[0], 8, 3, 5),
                    20.0,
                ),
            }

    model = CapacityValidationModel()
    batch = {
        "visual_tiles": torch.zeros(1, 1, 3, 2, 2),
        "map_context": torch.zeros(1, 14, 3, 5),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": torch.zeros(1, 2, 3, 5),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "route_channel_valid": torch.ones(1, 2, dtype=torch.bool),
        "trajectory_xy_m": torch.zeros(1, 64, 2),
        "trajectory_valid": torch.ones(1, 64, dtype=torch.bool),
        "initial_speed_mps": torch.zeros(1),
        "bev_segmentation_target": torch.ones(1, 8, 3, 5),
        "bev_segmentation_valid": torch.ones(
            1,
            8,
            3,
            5,
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


def test_bev_repeat_factors_are_frequency_aware_and_clipped():
    positive_samples = (100, 25, 4, 1, 100, 25, 4, 1)
    statistics = BEVTrainingStatistics(
        sample_count=100,
        effective_exposure_count=100,
        positive_sample_count=positive_samples,
        positive_cell_count=positive_samples,
        positive_mass=tuple(float(value) for value in positive_samples),
        valid_cell_count=(100,) * 8,
        exposure_digest="a" * 64,
    )

    assert derive_bev_repeat_factors(
        statistics,
        frequency_threshold=0.25,
        max_repeat=4,
    ) == (1, 1, 3, 4, 1, 1, 3, 4)


def test_bev_repeat_policy_preserves_non_bev_importance_mass():
    target = np.zeros((8, 2, 2), dtype=np.float32)
    target[3, 0, 0] = 1.0
    payload = encode_bev_segmentation_stats(
        target,
        np.ones_like(target, dtype=np.bool_),
    )
    policy = BEVClassRepeatPolicy(
        repeat_factors=(1, 1, 1, 4, 1, 1, 1, 1),
        mean_repeat=2.0,
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
