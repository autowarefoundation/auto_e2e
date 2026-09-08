"""Cross-dataset BEV segmentation evaluation contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_CLASSES,
    BEV_SEGMENTATION_TAXONOMY_VERSION,
)
from data_processing.dataset_snapshot import shard_partition_id
from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS
from evaluation.bev_segmentation import (
    BEV_AP_BOOTSTRAP_BINS,
    BEV_AP_BOOTSTRAP_MAX_WEIGHT,
    BEV_AP_BOOTSTRAP_REPLICATES,
    BEV_AP_BOOTSTRAP_VERSION,
    BEV_AP_BOOTSTRAP_WEIGHT_SCALE,
    BEVSegmentationMetricAccumulator,
    KITSCENES_DYNAMIC_UNAVAILABLE_REASON,
    KITSCENES_THIN_CLASS_UNAVAILABLE_REASON,
    _reverse_cumulative_histogram,
    accumulate_fixed_point_bootstrap_histogram,
    fixed_point_bayesian_bootstrap_weights,
    kitscenes_static_bev_targets,
    validate_fixed_point_bootstrap_capacity,
)
from evaluation.reactive_bev_checkpoint import (
    KITSCENES_DATASET,
    NUPLAN_DATASET,
    _evaluation_targets,
    checkpoint_bev_probability_bins,
    checkpoint_validation_thresholds,
    checkpoint_validation_thresholds_with_protocol,
    evaluate_reactive_bev_model,
    load_reactive_bev_checkpoint,
    validate_kitscenes_benchmark_inventory_coverage,
    validate_reactive_bev_evaluation_manifest,
)
import evaluation.reactive_bev_checkpoint as checkpoint_evaluation
import distributed_training.reactive_stage as reactive_stage
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    DEFAULT_NAVIGATION_GEOMETRY,
    MapChannel,
)
from training.reactive_multitask import ReactiveTrainingStage
from training.reactive_multitask import (
    BEV_HEAD_INITIALIZATION_VERSION,
    BEV_SAMPLING_IMPORTANCE_CORRECTION_VERSION,
)
from training.reactive_stage_runner import (
    reactive_config_sha256,
    save_reactive_checkpoint,
)
from reactive_training_contracts import (
    BEV_CHECKPOINT_MIN_CLASS_IOU,
    BEV_CHECKPOINT_MIN_CLASS_PRECISION,
    BEV_CHECKPOINT_QUALITY_GUARD_VERSION,
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_FRAME_OFFSETS,
    REACTIVE_BEVFORMER_HISTORY_FRAMES,
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_INDEX,
)


def test_kitscenes_static_targets_use_physical_geometry_and_known_map():
    source = DEFAULT_NAVIGATION_GEOMETRY
    target = AUTOE2E_NAVIGATION_GEOMETRY
    map_context = torch.zeros(
        1,
        len(MapChannel),
        source.height_px,
        source.width_px,
    )
    map_valid = torch.ones(1, dtype=torch.bool)
    source_pixel = source.ego_to_pixel(np.asarray([[0.0, 0.0]]))[0]
    source_row = int(round(float(source_pixel[0])))
    source_col = int(round(float(source_pixel[1])))
    map_context[
        0,
        MapChannel.KNOWN_MAP_AREA,
        source_row,
        source_col,
    ] = 1.0
    map_context[
        0,
        MapChannel.DRIVABLE_AREA,
        source_row,
        source_col,
    ] = 1.0

    bev_target, bev_valid, availability = kitscenes_static_bev_targets(
        map_context,
        map_valid,
    )

    target_pixel = target.ego_to_pixel(np.asarray([[0.0, 0.0]]))[0]
    target_row = int(round(float(target_pixel[0])))
    target_col = int(round(float(target_pixel[1])))
    assert bev_target.shape == (
        1,
        len(BEV_SEGMENTATION_CLASSES),
        target.height_px,
        target.width_px,
    )
    assert bool(
        bev_target[0, 0, target_row, target_col]
    )
    assert bool(
        bev_valid[0, 0, target_row, target_col]
    )
    assert not bool(bev_valid[:, 5:].any())
    assert availability == (
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    )


def test_training_and_evaluation_share_bootstrap_protocol():
    assert (
        reactive_stage.BEV_AP_BOOTSTRAP_REPLICATES
        == BEV_AP_BOOTSTRAP_REPLICATES
    )
    assert reactive_stage.BEV_AP_BOOTSTRAP_BINS == BEV_AP_BOOTSTRAP_BINS
    assert reactive_stage.BEV_AP_BOOTSTRAP_VERSION == BEV_AP_BOOTSTRAP_VERSION
    assert (
        reactive_stage.BEV_AP_BOOTSTRAP_WEIGHT_SCALE
        == BEV_AP_BOOTSTRAP_WEIGHT_SCALE
    )
    assert (
        reactive_stage.BEV_AP_BOOTSTRAP_MAX_WEIGHT
        == BEV_AP_BOOTSTRAP_MAX_WEIGHT
    )


def test_fixed_point_bootstrap_weights_are_deterministic_and_bounded():
    first = fixed_point_bayesian_bootstrap_weights(
        ("sample-a", "sample-b"),
        device=torch.device("cpu"),
    )
    second = fixed_point_bayesian_bootstrap_weights(
        ("sample-a", "sample-b"),
        device=torch.device("cpu"),
    )

    assert torch.equal(first, second)
    assert first.dtype is torch.int64
    assert first.shape == (BEV_AP_BOOTSTRAP_REPLICATES, 2)
    assert int(first.min().item()) >= 1
    assert int(first.max().item()) <= int(
        BEV_AP_BOOTSTRAP_MAX_WEIGHT * BEV_AP_BOOTSTRAP_WEIGHT_SCALE
    )
    assert not torch.equal(first[:, 0], first[:, 1])

    with pytest.raises(ValueError, match="unique"):
        fixed_point_bayesian_bootstrap_weights(
            ("duplicate", "duplicate"),
            device=torch.device("cpu"),
        )


def test_fixed_point_bootstrap_weights_do_not_use_numpy_rng(monkeypatch):
    def reject_rng(*_args, **_kwargs):
        raise AssertionError("bootstrap protocol must not use NumPy RNG")

    monkeypatch.setattr(np.random, "default_rng", reject_rng)

    weights = fixed_point_bayesian_bootstrap_weights(
        ("sample-a",),
        device=torch.device("cpu"),
    )

    assert weights.shape == (BEV_AP_BOOTSTRAP_REPLICATES, 1)


def test_fixed_point_bootstrap_rejects_int64_overflow():
    accumulator = torch.full(
        (1, 1, 1),
        torch.iinfo(torch.int64).max,
        dtype=torch.int64,
    )

    with pytest.raises(OverflowError, match="overflow"):
        accumulate_fixed_point_bootstrap_histogram(
            accumulator,
            torch.ones(1, 1, dtype=torch.int64),
            torch.ones(1, 1, 1, dtype=torch.int64),
        )


def test_fixed_point_bootstrap_capacity_fails_before_accumulation():
    validate_fixed_point_bootstrap_capacity(
        131_072,
        maximum_cells_per_sample=450 * 300,
    )

    with pytest.raises(OverflowError, match="overflow"):
        validate_fixed_point_bootstrap_capacity(
            10_000_000,
            maximum_cells_per_sample=450 * 300,
        )


def test_kitscenes_static_targets_mask_invalid_map():
    source = DEFAULT_NAVIGATION_GEOMETRY
    map_context = torch.ones(
        1,
        len(MapChannel),
        source.height_px,
        source.width_px,
    )

    _, valid, _ = kitscenes_static_bev_targets(
        map_context,
        torch.zeros(1, dtype=torch.bool),
    )

    assert not bool(valid.any())


def test_metric_accumulator_reports_static_and_marks_dynamic_unsupported():
    logits = torch.full((1, 8, 2, 2), -8.0)
    target = torch.zeros_like(logits)
    valid = torch.zeros_like(logits, dtype=torch.bool)
    supported_indices = (0, 2, 3)
    target[:, supported_indices, 0, 0] = 1.0
    valid[:, supported_indices] = True
    logits[:, supported_indices, 0, 0] = 8.0
    unavailable = {
        class_name: KITSCENES_DYNAMIC_UNAVAILABLE_REASON
        for class_name in BEV_SEGMENTATION_CLASSES[5:]
    }
    unavailable.update({
        "lane_boundary": KITSCENES_THIN_CLASS_UNAVAILABLE_REASON,
        "stop_line": KITSCENES_THIN_CLASS_UNAVAILABLE_REASON,
    })
    accumulator = BEVSegmentationMetricAccumulator(
        probability_bins=32,
        class_availability=(
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
        ),
        unavailable_reasons=unavailable,
        reference_thresholds=(0.5,) * 8,
        reference_threshold_source="nuplan_validation",
    )

    accumulator.update(
        logits,
        target,
        valid,
        sample_uids=("kitscenes-static",),
    )
    report = accumulator.report(
        dataset="KIT-MRT/KITScenes-Multimodal",
        checkpoint_sha256="a" * 64,
        label_source="lanelet2_static_map",
        temporal_input="exact_t8_0p5s",
    )

    assert report["computed_class_count"] == 3
    assert report["supported_class_count"] == 3
    assert report[
        "macro_average_precision_supported_classes"
    ] == pytest.approx(1.0)
    assert report["threshold_selection"] == "same_evaluation_set_oracle"
    assert report["classes"]["drivable_area"][
        "best_iou_on_evaluation_set"
    ] == pytest.approx(1.0)
    assert report["classes"]["drivable_area"][
        "iou_at_reference_threshold"
    ] == pytest.approx(1.0)
    assert "calibrated_iou" not in report["classes"]["drivable_area"]
    lane = report["classes"]["lane_boundary"]
    assert lane["availability"] == "unsupported"
    assert lane["reason"] == KITSCENES_THIN_CLASS_UNAVAILABLE_REASON
    dynamic = report["classes"]["vehicle"]
    assert dynamic["availability"] == "unsupported"
    assert dynamic["reason"] == KITSCENES_DYNAMIC_UNAVAILABLE_REASON


def test_constant_predictor_cannot_beat_class_prevalence():
    accumulator = BEVSegmentationMetricAccumulator(probability_bins=32)
    target = torch.zeros(4, 8, 4, 4)
    valid = torch.ones_like(target, dtype=torch.bool)
    for sample_index in range(target.shape[0]):
        for class_index in range(target.shape[1]):
            target[
                sample_index,
                class_index,
                class_index // 4,
                class_index % 4,
            ] = 1.0

    accumulator.update(
        torch.zeros_like(target),
        target,
        valid,
        sample_uids=tuple(
            f"constant-{index}" for index in range(target.shape[0])
        ),
    )
    report = accumulator.report(
        dataset=NUPLAN_DATASET,
        checkpoint_sha256="a" * 64,
        label_source="test",
        temporal_input="test",
    )

    for class_name in BEV_SEGMENTATION_CLASSES:
        class_report = report["classes"][class_name]
        assert class_report["ap_lift"] == pytest.approx(0.0)
        assert class_report[
            "best_iou_on_evaluation_set"
        ] == pytest.approx(class_report["positive_prevalence"])


def test_metric_accumulator_rejects_valid_cells_for_unavailable_class():
    accumulator = BEVSegmentationMetricAccumulator(
        class_availability=(
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
        )
    )
    logits = torch.zeros(1, 8, 1, 1)
    target = torch.zeros_like(logits)
    valid = torch.zeros_like(logits, dtype=torch.bool)
    valid[:, 5] = True

    with pytest.raises(ValueError, match="unavailable"):
        accumulator.update(logits, target, valid)


def test_metric_accumulator_rejects_non_boolean_valid_mask():
    accumulator = BEVSegmentationMetricAccumulator()

    with pytest.raises(ValueError, match="must be boolean"):
        accumulator.update(
            torch.zeros(1, 8, 1, 1),
            torch.zeros(1, 8, 1, 1),
            torch.ones(1, 8, 1, 1),
        )


def test_metric_accumulator_requires_stable_sample_uids():
    accumulator = BEVSegmentationMetricAccumulator()

    with pytest.raises(ValueError, match="stable sample UIDs"):
        accumulator.update(
            torch.zeros(1, 8, 1, 1),
            torch.zeros(1, 8, 1, 1),
            torch.ones(1, 8, 1, 1, dtype=torch.bool),
        )


def test_reverse_cumulative_histogram_avoids_int64_overflow():
    histogram = torch.tensor(
        [5_000_000_000_000_000_000] * 2,
        dtype=torch.int64,
    )

    cumulative = _reverse_cumulative_histogram(histogram)

    assert cumulative.dtype is torch.float64
    assert cumulative.tolist() == pytest.approx(
        [5.0e18, 1.0e19]
    )


def test_metric_accumulator_masks_nonfinite_invalid_cells_before_binning():
    accumulator = BEVSegmentationMetricAccumulator(probability_bins=32)
    logits = torch.zeros(1, 8, 2, 2)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    target[:, :, 0, 0] = 1.0
    logits[:, :, 0, 0] = 8.0
    logits[:, :, 1, 1] = float("nan")
    valid[:, :, 1, 1] = False

    accumulator.update(
        logits,
        target,
        valid,
        sample_uids=("unique-sample",),
    )
    report = accumulator.report(
        dataset=NUPLAN_DATASET,
        checkpoint_sha256="a" * 64,
        label_source="test",
        temporal_input="test",
    )

    assert report["evaluation_valid"] is True
    assert report["sample_count"] == 1


def test_metric_accumulator_rejects_duplicate_sample_uids():
    accumulator = BEVSegmentationMetricAccumulator()
    logits = torch.zeros(1, 8, 1, 1)
    target = torch.ones_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)

    accumulator.update(
        logits,
        target,
        valid,
        sample_uids=("duplicate",),
    )
    with pytest.raises(ValueError, match="duplicates"):
        accumulator.update(
            logits,
            target,
            valid,
            sample_uids=("duplicate",),
        )


def test_metric_accumulator_discloses_order_and_batch_partition():
    logits = torch.linspace(-3.0, 3.0, 2 * 8 * 3 * 3).reshape(
        2,
        8,
        3,
        3,
    )
    target = torch.zeros_like(logits)
    target[:, :, 0, 0] = 1.0
    target[1, :, 1, 1] = 1.0
    valid = torch.ones_like(logits, dtype=torch.bool)

    one_batch = BEVSegmentationMetricAccumulator()
    one_batch.update(
        logits,
        target,
        valid,
        sample_uids=("sample-a", "sample-b"),
    )
    split_batches = BEVSegmentationMetricAccumulator()
    split_batches.update(
        logits[:1],
        target[:1],
        valid[:1],
        sample_uids=("sample-a",),
    )
    split_batches.update(
        logits[1:],
        target[1:],
        valid[1:],
        sample_uids=("sample-b",),
    )

    one_report = one_batch.report(
        dataset=NUPLAN_DATASET,
        checkpoint_sha256="a" * 64,
        label_source="test",
        temporal_input="test",
    )
    split_report = split_batches.report(
        dataset=NUPLAN_DATASET,
        checkpoint_sha256="a" * 64,
        label_source="test",
        temporal_input="test",
    )

    assert (
        one_report["sample_uid_order_sha256"]
        == split_report["sample_uid_order_sha256"]
    )
    assert (
        one_report["evaluation_batch_partition_sha256"]
        != split_report["evaluation_batch_partition_sha256"]
    )
    assert one_report["evaluation_batch_count"] == 1
    assert split_report["evaluation_batch_count"] == 2
    for class_name in BEV_SEGMENTATION_CLASSES:
        one_class = one_report["classes"][class_name]
        split_class = split_report["classes"][class_name]
        assert (
            one_class["ap_lift_bootstrap_lower_95"]
            == split_class["ap_lift_bootstrap_lower_95"]
        )
        assert (
            one_class["ap_lift_bootstrap_upper_95"]
            == split_class["ap_lift_bootstrap_upper_95"]
        )


def test_checkpoint_threshold_protocol_rejects_mixed_metric_versions():
    current_metrics = {
        f"bev_{class_name}_best_iou_threshold_on_validation_set": 0.25
        for class_name in BEV_SEGMENTATION_CLASSES
    }
    thresholds, protocol = checkpoint_validation_thresholds_with_protocol(
        current_metrics
    )

    assert thresholds == (0.25,) * len(BEV_SEGMENTATION_CLASSES)
    assert protocol == "current_best_iou_on_validation_set"
    assert checkpoint_validation_thresholds(current_metrics) == thresholds

    mixed_metrics = dict(current_metrics)
    first_class = BEV_SEGMENTATION_CLASSES[0]
    del mixed_metrics[
        f"bev_{first_class}_best_iou_threshold_on_validation_set"
    ]
    mixed_metrics[f"bev_{first_class}_calibrated_threshold"] = 0.25
    with pytest.raises(ValueError, match="mixes"):
        checkpoint_validation_thresholds_with_protocol(mixed_metrics)

    ambiguous_metrics = dict(current_metrics)
    ambiguous_metrics[f"bev_{first_class}_calibrated_threshold"] = 0.25
    with pytest.raises(ValueError, match="ambiguous"):
        checkpoint_validation_thresholds_with_protocol(ambiguous_metrics)

    boolean_metrics = {
        f"bev_{class_name}_best_iou_threshold_on_validation_set": True
        for class_name in BEV_SEGMENTATION_CLASSES
    }
    with pytest.raises(ValueError, match="threshold is invalid"):
        checkpoint_validation_thresholds(boolean_metrics)


def test_reactive_bev_checkpoint_roundtrip_and_integrity_guards(
    tmp_path,
    monkeypatch,
):
    class TinyAutoE2E(torch.nn.Module):
        def __init__(
            self,
            bev_segmentation_classes=8,
            is_pretrained=False,
        ):
            super().__init__()
            assert is_pretrained is False
            self.weight = torch.nn.Parameter(
                torch.arange(
                    bev_segmentation_classes,
                    dtype=torch.float32,
                )
            )

    monkeypatch.setattr(
        "model_components.auto_e2e.AutoE2E",
        TinyAutoE2E,
    )
    model = TinyAutoE2E()
    metrics = {
        f"bev_{class_name}_best_iou_threshold_on_validation_set": 0.5
        for class_name in BEV_SEGMENTATION_CLASSES
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_reactive_checkpoint(
        checkpoint_path,
        model,
        stage=ReactiveTrainingStage.NUPLAN_FULL,
        dataset_manifest_sha256="a" * 64,
        epoch=2,
        model_config={
            "bev_segmentation_classes": len(BEV_SEGMENTATION_CLASSES),
            "bev_taxonomy_version": BEV_SEGMENTATION_TAXONOMY_VERSION,
            "distributed_precision": "bf16",
            "enable_bev_segmentation": True,
            "training_scope": "bev_only",
            "bev_ap_bins": 1024,
            "bev_class_weights": [1.0] * 8,
            "bev_positive_pair_frequencies": [0.5] * 8,
                "bev_checkpoint_quality_guard_version": (
                    BEV_CHECKPOINT_QUALITY_GUARD_VERSION
                ),
                "bev_checkpoint_min_class_iou": (
                    BEV_CHECKPOINT_MIN_CLASS_IOU
                ),
                "bev_checkpoint_min_class_precision": (
                    BEV_CHECKPOINT_MIN_CLASS_PRECISION
                ),
                "bev_sampling_importance_correction": (
                    BEV_SAMPLING_IMPORTANCE_CORRECTION_VERSION
                ),
                "bev_head_initialization": {
                    "version": BEV_HEAD_INITIALIZATION_VERSION,
                },
                "bev_raw_statistics": {
                    "effective_exposure_count": 10,
                    "active_sample_count": [10] * 8,
                    "positive_fraction_sum": [1.0] * 8,
                },
                "validation_sample_limit": 32,
        },
        metrics=metrics,
    )

    loaded, config, loaded_metrics, digest, epoch = (
        load_reactive_bev_checkpoint(
            checkpoint_path,
            device=torch.device("cpu"),
        )
    )

    torch.testing.assert_close(loaded.weight, model.weight)
    assert config["training_stage"] == "nuplan_full"
    assert checkpoint_bev_probability_bins(config) == 1024
    assert loaded_metrics == metrics
    assert len(digest) == 64
    assert epoch == 2

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    wrong_stage = dict(payload)
    wrong_stage["config"] = {
        **payload["config"],
        "training_stage": "l2d_continuation",
    }
    wrong_stage["config_sha256"] = reactive_config_sha256(
        wrong_stage["config"]
    )
    wrong_stage_path = tmp_path / "wrong-stage.pt"
    torch.save(wrong_stage, wrong_stage_path)
    with pytest.raises(ValueError, match="reviewed BEV contract"):
        load_reactive_bev_checkpoint(
            wrong_stage_path,
            device=torch.device("cpu"),
        )

    missing_bins = dict(payload)
    missing_bins["config"] = dict(payload["config"])
    del missing_bins["config"]["bev_ap_bins"]
    missing_bins["config_sha256"] = reactive_config_sha256(
        missing_bins["config"]
    )
    missing_bins_path = tmp_path / "missing-bins.pt"
    torch.save(missing_bins, missing_bins_path)
    with pytest.raises(ValueError, match="bev_ap_bins provenance"):
        load_reactive_bev_checkpoint(
            missing_bins_path,
            device=torch.device("cpu"),
        )

    wrong_quality_guard = dict(payload)
    wrong_quality_guard["config"] = {
        **payload["config"],
        "bev_checkpoint_min_class_precision": 0.01,
    }
    wrong_quality_guard["config_sha256"] = reactive_config_sha256(
        wrong_quality_guard["config"]
    )
    wrong_quality_guard_path = tmp_path / "wrong-quality-guard.pt"
    torch.save(wrong_quality_guard, wrong_quality_guard_path)
    with pytest.raises(ValueError, match="quality provenance"):
        load_reactive_bev_checkpoint(
            wrong_quality_guard_path,
            device=torch.device("cpu"),
        )

    corrupt = dict(payload)
    corrupt["model_state_dict"] = {
        **payload["model_state_dict"],
        "weight": payload["model_state_dict"]["weight"] + 1.0,
    }
    corrupt_path = tmp_path / "corrupt.pt"
    torch.save(corrupt, corrupt_path)
    with pytest.raises(ValueError, match="model digest"):
        load_reactive_bev_checkpoint(
            corrupt_path,
            device=torch.device("cpu"),
        )


def test_evaluate_reactive_bev_model_executes_prediction_path(monkeypatch):
    class EvaluationModel(torch.nn.Module):
        def forward(self, visual_tiles, *_args, **_kwargs):
            logits = torch.full(
                (visual_tiles.shape[0], 8, 2, 2),
                -8.0,
                device=visual_tiles.device,
            )
            logits[:, :, 0, 0] = 8.0
            return visual_tiles.new_zeros((visual_tiles.shape[0], 0)), {
                "bev_segmentation_logits": logits,
            }

    monkeypatch.setattr(
        checkpoint_evaluation,
        "resolve_reactive_batch_projection",
        lambda *_args, **_kwargs: (None, "rectified_pinhole"),
    )
    monkeypatch.setattr(
        checkpoint_evaluation,
        "resolve_reactive_front_projection",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        checkpoint_evaluation,
        "resolve_reactive_camera_history",
        lambda *_args, **_kwargs: (None, None),
    )
    target = torch.zeros(1, 8, 2, 2)
    target[:, :, 0, 0] = 1.0
    batch = {
        "sample_uid": ["evaluation-sample"],
        "visual_tiles": torch.zeros(1, 6, 3, 4, 4),
        "map_context": torch.zeros(1, 14, 4, 4),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": torch.zeros(1, 2, 4, 4),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "bev_segmentation_available": torch.ones(1, dtype=torch.bool),
        "bev_segmentation_target": target,
        "bev_segmentation_valid": torch.ones_like(
            target,
            dtype=torch.bool,
        ),
    }

    report = evaluate_reactive_bev_model(
        EvaluationModel(),
        [batch],
        dataset=NUPLAN_DATASET,
        device=torch.device("cpu"),
        checkpoint_sha256="a" * 64,
        probability_bins=32,
        reference_thresholds=(0.5,) * 8,
        reference_threshold_source="test",
        precision="fp32",
        expected_sample_uids=("evaluation-sample",),
        expected_sample_count=1,
    )

    assert report["evaluation_valid"] is True
    assert report["sample_count"] == 1
    assert report["supported_class_count"] == 8
    assert report["sample_coverage_complete"] is True
    assert report["expected_sample_count"] == 1
    assert report["evaluation_batch_size_counts"] == {"1": 1}
    assert report["evaluation_batch_count"] == 1
    assert len(report["evaluation_batch_partition_sha256"]) == 64
    assert len(report["sample_uid_order_sha256"]) == 64
    assert len(report["sample_uid_set_sha256"]) == 64
    assert report["bootstrap_reduction_reproducibility"] == (
        "fixed_point_sample_uid_weights_partition_invariant_v3"
    )
    assert report["ap_bootstrap_weight_scale"] == (
        BEV_AP_BOOTSTRAP_WEIGHT_SCALE
    )
    assert report["ap_bootstrap_max_weight"] == (
        BEV_AP_BOOTSTRAP_MAX_WEIGHT
    )
    assert report["prediction_provenance"] == {
        "segmentation_head_input": (
            "image_bev_before_navigation_fusion"
        ),
        "map_context_consumed_by_segmentation_head": False,
        "route_mask_consumed_by_segmentation_head": False,
        "bev_only_returns_before_navigation_encoder": True,
    }

    with pytest.raises(ValueError, match="sample coverage"):
        evaluate_reactive_bev_model(
            EvaluationModel(),
            [batch],
            dataset=NUPLAN_DATASET,
            device=torch.device("cpu"),
            checkpoint_sha256="a" * 64,
            probability_bins=32,
            precision="fp32",
            expected_sample_uids=(
                "evaluation-sample",
                "missing-sample",
            ),
            expected_sample_count=2,
        )

    with pytest.raises(ValueError, match="front-camera companion"):
        evaluate_reactive_bev_model(
            EvaluationModel(),
            [{
                **batch,
                "front_camera_tile": torch.zeros(1, 3, 8, 8),
            }],
            dataset=KITSCENES_DATASET,
            device=torch.device("cpu"),
            checkpoint_sha256="a" * 64,
            probability_bins=32,
            precision="fp32",
        )


def test_dataset_targets_keep_nuplan_and_kitscenes_contracts_distinct():
    nuplan_batch = {
        "bev_segmentation_available": torch.ones(1, dtype=torch.bool),
        "bev_segmentation_target": torch.zeros(1, 8, 2, 2),
        "bev_segmentation_valid": torch.ones(
            1,
            8,
            2,
            2,
            dtype=torch.bool,
        ),
    }
    target, valid, availability = _evaluation_targets(
        nuplan_batch,
        dataset=NUPLAN_DATASET,
    )
    assert target.shape == valid.shape == (1, 8, 2, 2)
    assert availability == (True,) * 8

    source = DEFAULT_NAVIGATION_GEOMETRY
    kitscenes_batch = {
        "map_context": torch.zeros(
            1,
            len(MapChannel),
            source.height_px,
            source.width_px,
        ),
        "map_valid": torch.ones(1, dtype=torch.bool),
    }
    _, kitscenes_valid, kitscenes_availability = _evaluation_targets(
        kitscenes_batch,
        dataset=KITSCENES_DATASET,
    )
    assert not bool(kitscenes_valid[:, 5:].any())
    assert kitscenes_availability == (
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    )


def test_metric_accumulator_reports_and_skips_nonfinite_samples():
    accumulator = BEVSegmentationMetricAccumulator()
    logits = torch.zeros(2, 8, 1, 1)
    logits[0, 0, 0, 0] = float("inf")
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)

    accumulator.update(
        logits,
        target,
        valid,
        sample_uids=("nonfinite", "finite"),
    )
    report = accumulator.report(
        dataset=NUPLAN_DATASET,
        checkpoint_sha256="a" * 64,
        label_source="test",
        temporal_input="test",
    )

    assert report["observed_sample_count"] == 2
    assert report["sample_count"] == 1
    assert report["nonfinite_sample_count"] == 1
    assert report["evaluation_valid"] is False


def _evaluation_manifest(dataset):
    return {
        "dataset": dataset,
        "total_samples": 2,
        "num_views": len(CANONICAL_SIX_CAMERA_SLOTS),
        "camera_slots": list(CANONICAL_SIX_CAMERA_SLOTS),
        "image_size": REACTIVE_CAMERA_IMAGE_SIZE,
        "map_context_channels": 14,
    }


def test_nuplan_evaluation_manifest_requires_training_input_contract():
    manifest = _evaluation_manifest(NUPLAN_DATASET)
    manifest.update({
        "has_bev_segmentation": True,
        "front_camera_index": REACTIVE_FRONT_CAMERA_INDEX,
        "front_camera_image_size": REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
        "temporal_frame_offsets": list(REACTIVE_BEVFORMER_FRAME_OFFSETS),
        "temporal_frame_interval_us": (
            REACTIVE_BEVFORMER_FRAME_INTERVAL_US
        ),
    })

    validate_reactive_bev_evaluation_manifest(
        manifest,
        dataset=NUPLAN_DATASET,
    )
    manifest["temporal_frame_interval_us"] = 100_000
    with pytest.raises(ValueError, match="nuPlan"):
        validate_reactive_bev_evaluation_manifest(
            manifest,
            dataset=NUPLAN_DATASET,
        )


def test_kitscenes_evaluation_manifest_requires_v12_exact_t8():
    manifest = _evaluation_manifest(KITSCENES_DATASET)
    manifest.update({
        "data_role": "benchmark",
        "dataset_version": "v3.3-benchmark-v3",
        "partition_id": "scene-a",
        "source_revision": (
            "6fde0034446669e2ed7235e4c7fe323cd23d599d"
        ),
        "source_split": "val",
        "contracts": {"shard_schema_version": "v12"},
        "has_bevformer_history": True,
        "bevformer_temporal_contract": {
            "frame_count": len(REACTIVE_BEVFORMER_FRAME_OFFSETS),
            "history_frame_count": REACTIVE_BEVFORMER_HISTORY_FRAMES,
            "frame_interval_us": REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
            "frame_offsets": list(REACTIVE_BEVFORMER_FRAME_OFFSETS),
            "history_reference_frame": "current_ego",
        },
    })

    validate_reactive_bev_evaluation_manifest(
        manifest,
        dataset=KITSCENES_DATASET,
    )
    manifest["contracts"]["shard_schema_version"] = "v11"
    with pytest.raises(ValueError, match="v12"):
        validate_reactive_bev_evaluation_manifest(
            manifest,
            dataset=KITSCENES_DATASET,
        )


def test_kitscenes_evaluation_manifest_rejects_training_split():
    manifest = _evaluation_manifest(KITSCENES_DATASET)
    manifest.update({
        "data_role": "training",
        "dataset_version": "v3.4",
        "partition_id": "scene-a",
        "source_revision": (
            "6fde0034446669e2ed7235e4c7fe323cd23d599d"
        ),
        "source_split": "train",
        "contracts": {"shard_schema_version": "v12"},
        "has_bevformer_history": True,
        "bevformer_temporal_contract": {
            "frame_count": len(REACTIVE_BEVFORMER_FRAME_OFFSETS),
            "history_frame_count": REACTIVE_BEVFORMER_HISTORY_FRAMES,
            "frame_interval_us": REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
            "frame_offsets": list(REACTIVE_BEVFORMER_FRAME_OFFSETS),
            "history_reference_frame": "current_ego",
        },
    })

    with pytest.raises(ValueError, match="pinned benchmark data"):
        validate_reactive_bev_evaluation_manifest(
            manifest,
            dataset=KITSCENES_DATASET,
        )


def test_kitscenes_inventory_requires_every_official_partition():
    revision = "6fde0034446669e2ed7235e4c7fe323cd23d599d"
    val_partition_id = shard_partition_id(["scene-val"])
    overlap_partition_id = shard_partition_id(["scene-overlap"])
    inventory = {
        "schema_version": "kitscenes_benchmark_inventory_v1",
        "dataset": KITSCENES_DATASET,
        "dataset_revision": revision,
        "sdk_revision": "7765cdec5490894266070ab46e23724b58b3da42",
        "total_scene_count": 2,
        "splits": {
            "val": {
                "split": "val",
                "source_revision": revision,
                "expected_scene_count": 1,
                "selected_scene_count": 1,
                "missing_scene_ids": [],
                "archives": [{
                    "scene_id": "scene-val",
                    "archive_sha256": "a" * 64,
                    "archive_size_bytes": 10,
                }],
            },
            "overlap_train_val": {
                "split": "overlap_train_val",
                "source_revision": revision,
                "expected_scene_count": 1,
                "selected_scene_count": 1,
                "missing_scene_ids": [],
                "archives": [{
                    "scene_id": "scene-overlap",
                    "archive_sha256": "b" * 64,
                    "archive_size_bytes": 20,
                }],
            },
        },
    }
    identities = [
        {
            "data_role": "benchmark",
            "partition_id": val_partition_id,
            "source_revision": revision,
            "source_split": "val",
            "total_samples": 0,
        },
        {
            "data_role": "benchmark",
            "partition_id": overlap_partition_id,
            "source_revision": revision,
            "source_split": "overlap_train_val",
            "total_samples": 5,
        },
    ]

    contract = validate_kitscenes_benchmark_inventory_coverage(
        inventory,
        identities,
    )

    assert contract["inventory_complete"] is True
    assert contract["official_scene_count"] == 2
    assert contract["official_scene_count_by_split"] == {
        "overlap_train_val": 1,
        "val": 1,
    }
    assert len(contract["partition_identity_sha256"]) == 64

    with pytest.raises(ValueError, match="partition coverage"):
        validate_kitscenes_benchmark_inventory_coverage(
            inventory,
            identities[1:],
        )

    raw_scene_identities = [
        {**identity, "partition_id": scene_id}
        for identity, scene_id in zip(
            identities,
            ("scene-val", "scene-overlap"),
            strict=True,
        )
    ]
    with pytest.raises(ValueError, match="partition coverage"):
        validate_kitscenes_benchmark_inventory_coverage(
            inventory,
            raw_scene_identities,
        )


def test_kitscenes_empty_manifest_is_valid_inventory_evidence():
    manifest = {
        "dataset": KITSCENES_DATASET,
        "total_samples": 0,
        "data_role": "benchmark",
        "dataset_version": "v3.3-benchmark-v3",
        "partition_id": "empty-scene",
        "source_revision": (
            "6fde0034446669e2ed7235e4c7fe323cd23d599d"
        ),
        "source_split": "val",
        "contracts": {"shard_schema_version": "v12"},
        "num_views": 0,
        "shards": 0,
        "shard_names": [],
        "shard_sample_counts": {},
        "has_map": False,
        "has_gps": False,
        "has_navigation": False,
    }

    validate_reactive_bev_evaluation_manifest(
        manifest,
        dataset=KITSCENES_DATASET,
    )

    manifest["shards"] = 1
    with pytest.raises(ValueError, match="empty contract"):
        validate_reactive_bev_evaluation_manifest(
            manifest,
            dataset=KITSCENES_DATASET,
        )
