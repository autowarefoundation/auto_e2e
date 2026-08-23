"""Distributed Reactive dataset and fixed-step contracts."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_parsing.pre_extracted import (
    BEVClassRepeatPolicy,
    BEVSampleStatistics,
    BEVTrainingStatistics,
    derive_bev_pos_weights,
    derive_bev_repeat_factors,
    discover_bev_training_statistics,
    make_pre_extracted_loader,
    passthrough_nodesplitter,
    summarize_bev_positive_samples,
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
    BEV_LANE_NEAR_RADIUS_M,
    BEV_OVERFIT_MIN_POSITIVE_SAMPLES,
    MIN_OVERFIT_OPTIMIZER_STEPS,
    ROUTE_VALIDATION_METRICS_VERSION,
    _all_reduce_bev_statistics,
    _bev_lane_range_masks,
    _bev_overfit_gate_result,
    _bev_overfit_selected_support,
    _build_reactive_scheduler,
    _camera_feature_scale_weights,
    _checkpoint_history,
    _histogram_average_precision,
    _load_resume_checkpoint,
    _overfit_gate_passed,
    _evaluate_global_reactive,
    _route_validation_statistics,
    _select_bev_overfit_subset,
    _select_result_checkpoint,
    _report_reactive_epoch,
    clip_finite_gradients_float64,
    normalize_ray_checkpoint_uri,
    reactive_gradient_parameter_groups,
    run_reactive_stage,
    validate_reactive_stage_config,
)
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
from model_components.losses import (
    BEVSegmentationAuxiliaryLoss,
    RouteReconstructionLoss,
)
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
        "dataset": dataset,
        "has_bev_segmentation": include_bev,
        "has_reactive_navigation": True,
        "has_route_reconstruction": True,
        "has_trajectory_xy": True,
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
        num_views=8,
    )
    _write_source(
        source_b,
        dataset="nuplan/nuplan-v1.1",
        shard_counts=[8, 5],
        include_bev=True,
        num_views=8,
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
    assert plan.num_views == 8
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


def test_stage_a_dataset_plan_rejects_previous_bev_taxonomy(tmp_path):
    source = tmp_path / "previous-taxonomy"
    manifest = _write_source(
        source,
        dataset="nuplan/nuplan-v1.1",
        shard_counts=[4],
        include_bev=True,
        num_views=8,
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
    assert set(groups) == {"camera", "navigation", "planner"}
    assert grouped_ids == expected_ids
    assert sum(len(parameters) for parameters in groups.values()) == len(
        expected_ids
    )

    gradient_values = {
        "camera": 10.0,
        "navigation": 0.5 / len(groups["navigation"]) ** 0.5,
        "planner": 4.0,
    }
    post_clip_norms = {}
    for group_name, parameters in groups.items():
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
        "epochs": 2,
        "grad_clip": 1.0,
        "gradient_accumulation_steps": 1,
        "is_pretrained": False,
        "learning_rate": 1e-4,
        "num_loader_workers": 1,
        "num_workers": 8,
        "overfit_min_ap": 0.9,
        "overfit_min_recall": 0.9,
        "overfit_bev_only": False,
        "overfit_fixed_lr": False,
        "overfit_sample_count": 0,
        "overfit_shard_limit": 0,
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


def test_validate_stage_config_requires_one_initialization_mode():
    config = _stage_config("nuplan_full")
    config["allow_random_bevformer_init"] = False

    with pytest.raises(ValueError, match="exactly one"):
        validate_reactive_stage_config(config)

    config["is_pretrained"] = True
    config["bevformer_pretrained_checkpoint_uri"] = "s3://bucket/model.pth"
    config["bevformer_pretrained_checkpoint_sha256"] = "a" * 64
    validate_reactive_stage_config(config)


def test_validate_stage_config_accepts_bev_only_capacity_probe():
    config = _stage_config("nuplan_full")
    config.update({
        "epochs": 10,
        "overfit_bev_only": True,
        "overfit_fixed_lr": True,
        "overfit_sample_count": 64,
        "route_weight": 0.0,
        "steps_per_epoch": 500,
        "trajectory_weight": 0.0,
        "weight_decay": 0.0,
    })

    validate_reactive_stage_config(config)


def test_validate_stage_config_enforces_joint_gate_step_floor():
    config = _stage_config("nuplan_full")
    config.update({
        "epochs": 10,
        "overfit_fixed_lr": True,
        "overfit_sample_count": 64,
        "steps_per_epoch": 500,
    })
    validate_reactive_stage_config(config)

    config["steps_per_epoch"] = 499
    with pytest.raises(ValueError, match="5000"):
        validate_reactive_stage_config(config)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"overfit_fixed_lr": False}, "fixed learning rate"),
        ({"route_weight": 1.0}, "objective weights"),
        ({"steps_per_epoch": 499}, "5000"),
        ({"weight_decay": 0.01}, "weight_decay"),
    ],
)
def test_validate_stage_config_rejects_invalid_bev_only_probe(
    override,
    match,
):
    config = _stage_config("nuplan_full")
    config.update({
        "epochs": 10,
        "overfit_bev_only": True,
        "overfit_fixed_lr": True,
        "overfit_sample_count": 64,
        "route_weight": 0.0,
        "steps_per_epoch": 500,
        "trajectory_weight": 0.0,
        "weight_decay": 0.0,
        **override,
    })

    with pytest.raises(ValueError, match=match):
        validate_reactive_stage_config(config)


def test_fixed_overfit_scheduler_preserves_lr_and_state():
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    identity, scheduler = _build_reactive_scheduler(
        optimizer,
        fixed_lr=True,
    )

    for _ in range(10):
        optimizer.step()
        scheduler.step()

    restored_optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    restored_identity, restored_scheduler = _build_reactive_scheduler(
        restored_optimizer,
        fixed_lr=True,
    )
    restored_scheduler.load_state_dict(scheduler.state_dict())

    assert identity == restored_identity == "constant_v1"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def _write_resume_checkpoint(
    directory,
    *,
    config,
    model,
    optimizer,
    scheduler,
):
    torch = pytest.importorskip("torch")
    directory.mkdir()
    torch.save(
        {
            "config": config,
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_state": {
                "best_ade_6p4s_m": 2.0,
                "best_selection_score": 0.5,
            },
        },
        directory / "checkpoint.pt",
    )
    (directory / "history.json").write_text(json.dumps([{"epoch": 1}]))


def test_resume_checkpoint_round_trips_fixed_scheduler(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    _, scheduler = _build_reactive_scheduler(optimizer, fixed_lr=True)
    expected = {
        "trajectory_weight": 0.0,
        "bev_weight": 1.0,
        "route_weight": 0.0,
        "corridor_pos_weight": 1.0,
        "gradient_clip_max_norm": 1.0,
        "gradient_clip_mode": "branch_v1",
        "route_metrics_version": ROUTE_VALIDATION_METRICS_VERSION,
        "training_seed": 149,
        "scheduler_identity": "constant_v1",
        "overfit_bev_only": True,
        "overfit_fixed_lr": True,
    }
    checkpoint = tmp_path / "checkpoint"
    _write_resume_checkpoint(
        checkpoint,
        config=expected,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    restored = _load_resume_checkpoint(
        str(checkpoint),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected=expected,
    )

    assert restored == (2, 0.5, 2.0, [{"epoch": 1}])


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("trajectory_weight", 0.5),
        ("bev_weight", 2.0),
        ("route_weight", 0.25),
        ("corridor_pos_weight", 2.0),
        ("gradient_clip_max_norm", 0.5),
        ("gradient_clip_mode", "global_v1"),
        ("route_metrics_version", "route_validation_v2"),
        ("training_seed", 150),
        ("scheduler_identity", "selection_plateau_v1"),
        ("overfit_bev_only", False),
        ("overfit_fixed_lr", False),
    ],
)
def test_resume_checkpoint_rejects_objective_mismatch(
    tmp_path,
    field,
    different_value,
):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    _, scheduler = _build_reactive_scheduler(optimizer, fixed_lr=True)
    checkpoint_config = {
        "trajectory_weight": 0.0,
        "bev_weight": 1.0,
        "route_weight": 0.0,
        "corridor_pos_weight": 1.0,
        "gradient_clip_max_norm": 1.0,
        "gradient_clip_mode": "branch_v1",
        "route_metrics_version": ROUTE_VALIDATION_METRICS_VERSION,
        "training_seed": 149,
        "scheduler_identity": "constant_v1",
        "overfit_bev_only": True,
        "overfit_fixed_lr": True,
    }
    checkpoint = tmp_path / "checkpoint"
    _write_resume_checkpoint(
        checkpoint,
        config=checkpoint_config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    expected = {**checkpoint_config, field: different_value}

    with pytest.raises(ValueError, match=field):
        _load_resume_checkpoint(
            str(checkpoint),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected=expected,
        )


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
    assert captured["run_config"].checkpoint_config.num_to_keep is None
    assert result["selected_epoch"] == 1
    assert result["metrics"]["checkpoint_sha256"] == "a" * 64


def test_validate_stage_config_rejects_missing_worker_cpu_contract():
    config = _stage_config("nuplan_full")
    config.pop("worker_cpus")

    with pytest.raises(ValueError, match="worker_cpus"):
        validate_reactive_stage_config(config)


def test_validate_stage_config_rejects_invalid_gate_dataset_digest():
    config = _stage_config("nuplan_full")
    config["required_gate_dataset_manifest_sha256"] = "not-a-digest"

    with pytest.raises(ValueError, match="required_gate_dataset"):
        validate_reactive_stage_config(config)


@pytest.mark.parametrize(
    ("stage", "expected_views", "expected_bev"),
    [
        (ReactiveTrainingStage.NUPLAN_FULL, 8, True),
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
    train_batches = list(make_pre_extracted_loader(
        str(dataset),
        batch_size=1,
        num_workers=0,
        split="train",
        val_fraction=0.5,
        shuffle=0,
        decode_future_frames=False,
        nodesplitter=passthrough_nodesplitter,
    ))
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
        256,
        256,
    )
    assert sample["map_context"].shape == (1, 14, 450, 300)
    assert sample["route_mask"].shape == (1, 2, 450, 300)
    assert bool(sample["bev_segmentation_available"][0]) is expected_bev
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        statistics = discover_bev_training_statistics(
            [str(dataset)],
            val_fraction=0.5,
        )
        weights = derive_bev_pos_weights(statistics)
        assert len(weights) == 8
        assert all(value >= 1.0 for value in weights)
        assert statistics.positive_sample_count == (4,) * 8


def test_positive_sample_summary_retains_per_class_cell_counts():
    records = tuple(
        BEVSampleStatistics(
            sample_uid=f"sample-{index:03d}",
            split_group_uid=f"group-{index:03d}",
            positive_cell_count=(
                index + 10,
                0,
                3,
                0,
                0,
                1,
                0,
                0,
            ),
            positive_mass=(
                float(index + 10),
                0.0,
                3.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ),
            valid_cell_count=(100,) * 8,
        )
        for index in range(20)
    )
    expected = {
        record.sample_uid: record.positive_cell_count
        for record in records
    }

    summaries = summarize_bev_positive_samples(
        records,
        val_fraction=0.1,
    )

    assert summaries
    assert all(cell_counts == expected[uid] for uid, cell_counts in summaries)


def test_overfit_subset_has_class_support_and_exact_rank_quotas():
    rank_summaries = tuple(
        tuple(
            (
                f"rank-{rank}-sample-{index:03d}",
                tuple(
                    (
                        100 + index
                        if index == 0 or class_index == index % 8
                        else 0
                    )
                    for class_index in range(8)
                ),
            )
            for index in range(20)
        )
        for rank in range(4)
    )

    selected = _select_bev_overfit_subset(
        rank_summaries,
        sample_count=64,
    )

    assert len(selected) == 64
    selected_set = set(selected)
    class_support = [
        sum(
            uid in selected_set and cell_counts[class_index] > 0
            for summaries in rank_summaries
            for uid, cell_counts in summaries
        )
        for class_index in range(8)
    ]
    assert _bev_overfit_selected_support(
        rank_summaries,
        selected,
    ) == tuple(class_support)
    assert all(
        support >= BEV_OVERFIT_MIN_POSITIVE_SAMPLES
        for support in class_support
    )
    assert [
        sum(uid.startswith(f"rank-{rank}-") for uid in selected)
        for rank in range(4)
    ] == [16, 16, 16, 16]
    assert selected == _select_bev_overfit_subset(
        tuple(reversed(rank_summaries)),
        sample_count=64,
    )


def test_overfit_subset_rejects_insufficient_rare_class_support():
    rank_summaries = tuple(
        tuple(
            (
                f"rank-{rank}-sample-{index:03d}",
                tuple(
                    (
                        1
                        if class_index < 7
                        or rank * 20 + index < 7
                        else 0
                    )
                    for class_index in range(8)
                ),
            )
            for index in range(20)
        )
        for rank in range(4)
    )

    with pytest.raises(
        ValueError,
        match="insufficient positive sample support",
    ):
        _select_bev_overfit_subset(
            rank_summaries,
            sample_count=64,
        )


def test_overfit_subset_reports_joint_rank_quota_conflict():
    rank_zero = tuple(
        (
            f"rank-0-class-0-{index:03d}",
            (1, 0, 1, 1, 1, 1, 1, 0),
        )
        for index in range(8)
    ) + tuple(
        (
            f"rank-0-class-1-{index:03d}",
            (0, 1, 0, 0, 0, 0, 0, 0),
        )
        for index in range(8)
    ) + tuple(
        (
            f"rank-0-class-7-{index:03d}",
            (0, 0, 0, 0, 0, 0, 0, 1),
        )
        for index in range(8)
    )
    filler = tuple(
        tuple(
            (f"rank-{rank}-filler-{index:03d}", (0,) * 8)
            for index in range(16)
        )
        for rank in range(1, 4)
    )

    with pytest.raises(
        ValueError,
        match=(
            r"support for class 1: .*"
            r"available_by_rank=\(8, 0, 0, 0\), .*"
            r"remaining_support_by_rank=\(3, 0, 0, 0\), .*"
            r"remaining_capacity_by_rank=\(0, 16, 16, 16\)"
        ),
    ):
        _select_bev_overfit_subset(
            (rank_zero, *filler),
            sample_count=64,
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


def test_bev_overfit_gate_result_reports_weakest_classes():
    validation = {
        f"bev_{class_name}_{suffix}": 0.95
        for class_name in BEV_SEGMENTATION_CLASSES
        for suffix in ("average_precision", "recall")
    }
    validation["bev_vehicle_average_precision"] = 0.72
    validation["bev_vulnerable_road_user_recall"] = 0.61

    result = _bev_overfit_gate_result(
        validation,
        minimum_ap=0.9,
        minimum_recall=0.9,
    )

    assert not result["passed"]
    assert result["minimum_ap"] == pytest.approx(0.72)
    assert result["minimum_ap_class"] == "vehicle"
    assert result["minimum_recall"] == pytest.approx(0.61)
    assert result["minimum_recall_class"] == "vulnerable_road_user"


def test_overfit_gate_pass_requires_executed_step_floor():
    assert not _overfit_gate_passed(
        thresholds_passed=False,
        executed_optimizer_steps=MIN_OVERFIT_OPTIMIZER_STEPS,
    )
    assert not _overfit_gate_passed(
        thresholds_passed=True,
        executed_optimizer_steps=MIN_OVERFIT_OPTIMIZER_STEPS - 1,
    )
    assert _overfit_gate_passed(
        thresholds_passed=True,
        executed_optimizer_steps=MIN_OVERFIT_OPTIMIZER_STEPS,
    )


def test_reactive_epoch_reports_checkpoint_before_gate_failure():
    reported = []
    checkpoint = object()
    metrics = {
        "epoch": 10,
        "overfit_gate_pass": 0,
        "overfit_minimum_ap": 0.2,
    }

    def report(values, *, checkpoint):
        reported.append((values, checkpoint))

    with pytest.raises(RuntimeError, match="gate failed"):
        _report_reactive_epoch(
            report,
            metrics,
            checkpoint=checkpoint,
            failure_message="gate failed",
        )

    assert reported == [(metrics, checkpoint)]


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

    selected, metrics = _select_result_checkpoint(
        result,
        overfit_mode=False,
    )

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
