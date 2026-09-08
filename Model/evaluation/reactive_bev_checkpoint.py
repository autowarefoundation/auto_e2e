"""Reproducible class-level evaluation for Reactive BEV checkpoints."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from data_parsing.kit_scenes.source import (
    KITSCENES_DATA_REVISION,
    KITSCENES_SDK_REVISION,
)
from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_CLASSES,
    BEV_SEGMENTATION_TAXONOMY_VERSION,
)
from data_processing.dataset_snapshot import shard_partition_id
from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS
from evaluation.bev_segmentation import (
    BEVSegmentationMetricAccumulator,
    KITSCENES_DYNAMIC_UNAVAILABLE_REASON,
    KITSCENES_THIN_CLASS_UNAVAILABLE_REASON,
    kitscenes_static_bev_targets,
    validate_fixed_point_bootstrap_capacity,
)
from model_components.losses import (
    BEV_SEGMENTATION_AUXILIARY_LOSS_VERSION,
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
from training.reactive_multitask import (
    BEV_HEAD_INITIALIZATION_VERSION,
    BEV_SAMPLING_IMPORTANCE_CORRECTION_VERSION,
    REACTIVE_MODEL_ARCHITECTURE_VERSION,
)
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    DEFAULT_NAVIGATION_GEOMETRY,
)
from training.reactive_stage_runner import (
    _batch_to_device,
    _loader_item,
    reactive_config_sha256,
    reactive_model_state_sha256,
    resolve_reactive_batch_projection,
    resolve_reactive_camera_history,
    resolve_reactive_front_projection,
)


NUPLAN_DATASET: Final = "nuplan/nuplan-v1.1"
KITSCENES_DATASET: Final = "KIT-MRT/KITScenes-Multimodal"
KITSCENES_BENCHMARK_DATASET_VERSION: Final = "v3.3-benchmark-v3"
KITSCENES_BENCHMARK_SPLITS: Final = frozenset({
    "val",
    "overlap_train_val",
})
SUPPORTED_DATASETS: Final = frozenset({
    NUPLAN_DATASET,
    KITSCENES_DATASET,
})
LEGACY_BEV_SEGMENTATION_AUXILIARY_LOSS_VERSION: Final = (
    "weighted_bce_positive_pair_dice_v1"
)
LEGACY_GRADIENT_BUDGET_BEV_LOSS_VERSION: Final = (
    "weighted_bce_gradient_budget_positive_pair_dice_v2"
)
LEGACY_RANK_CORRECTED_BEV_LOSS_VERSION: Final = (
    "rank_corrected_weighted_bce_quantized_positive_pair_dice_v5"
)
LEGACY_SAMPLE_NORMALIZED_BEV_LOSS_VERSION: Final = (
    "rank_corrected_sample_normalized_weighted_bce_"
    "quantized_positive_pair_dice_v6"
)
SUPPORTED_BEV_SEGMENTATION_AUXILIARY_LOSS_VERSIONS: Final = frozenset({
    BEV_SEGMENTATION_AUXILIARY_LOSS_VERSION,
    LEGACY_SAMPLE_NORMALIZED_BEV_LOSS_VERSION,
    LEGACY_RANK_CORRECTED_BEV_LOSS_VERSION,
    LEGACY_GRADIENT_BUDGET_BEV_LOSS_VERSION,
    LEGACY_BEV_SEGMENTATION_AUXILIARY_LOSS_VERSION,
})


def checkpoint_bev_probability_bins(config: Mapping[str, Any]) -> int:
    """Resolve the training histogram grid without hiding current provenance."""
    raw_value = config.get("bev_ap_bins")
    if raw_value is None:
        if (
            config.get("bev_loss_version")
            == BEV_SEGMENTATION_AUXILIARY_LOSS_VERSION
        ):
            raise ValueError(
                "current BEV checkpoint lacks bev_ap_bins provenance"
            )
        return 1024
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError("checkpoint bev_ap_bins is invalid")
    probability_bins = raw_value
    if probability_bins < 256:
        raise ValueError("checkpoint bev_ap_bins is invalid")
    return probability_bins


def validate_reactive_bev_evaluation_manifest(
    manifest: Mapping[str, Any],
    *,
    dataset: str,
) -> None:
    """Reject packed data that cannot reproduce the checkpoint input ABI."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported BEV evaluation dataset {dataset!r}")
    if manifest.get("dataset") != dataset:
        raise ValueError("BEV evaluation manifest belongs to another dataset")
    total_samples = manifest.get("total_samples")
    if (
        isinstance(total_samples, bool)
        or not isinstance(total_samples, int)
        or total_samples < 0
    ):
        raise ValueError("BEV evaluation manifest has an invalid sample count")
    if dataset == NUPLAN_DATASET and total_samples <= 0:
        raise ValueError("BEV evaluation manifest has no samples")
    if dataset == KITSCENES_DATASET:
        contracts = manifest.get("contracts")
        if (
            not isinstance(contracts, Mapping)
            or contracts.get("shard_schema_version") != "v12"
        ):
            raise ValueError(
                "KITScenes BEV evaluation requires exact v12 data"
            )
        if (
            manifest.get("data_role") != "benchmark"
            or manifest.get("source_split") not in KITSCENES_BENCHMARK_SPLITS
            or manifest.get("source_revision") != KITSCENES_DATA_REVISION
            or manifest.get("dataset_version")
            != KITSCENES_BENCHMARK_DATASET_VERSION
            or not isinstance(manifest.get("partition_id"), str)
            or not manifest.get("partition_id")
        ):
            raise ValueError(
                "KITScenes BEV evaluation requires pinned benchmark data"
            )
        if total_samples == 0:
            empty_contract = {
                "num_views": 0,
                "shards": 0,
                "shard_names": [],
                "shard_sample_counts": {},
                "has_map": False,
                "has_gps": False,
                "has_navigation": False,
            }
            actual_empty_contract = {
                key: manifest.get(key) for key in empty_contract
            }
            if actual_empty_contract != empty_contract:
                raise ValueError(
                    "KITScenes empty BEV evaluation partition differs from "
                    "the audited empty contract"
                )
            return
    if (
        int(manifest.get("num_views", 0))
        != len(CANONICAL_SIX_CAMERA_SLOTS)
        or manifest.get("camera_slots")
        != list(CANONICAL_SIX_CAMERA_SLOTS)
        or int(manifest.get("image_size", 0))
        != REACTIVE_CAMERA_IMAGE_SIZE
        or int(manifest.get("map_context_channels", 0)) != 14
    ):
        raise ValueError("BEV evaluation manifest differs from model input")
    if dataset == NUPLAN_DATASET:
        if (
            manifest.get("has_bev_segmentation") is not True
            or manifest.get("front_camera_index")
            != REACTIVE_FRONT_CAMERA_INDEX
            or manifest.get("front_camera_image_size")
            != REACTIVE_FRONT_CAMERA_IMAGE_SIZE
            or manifest.get("temporal_frame_offsets")
            != list(REACTIVE_BEVFORMER_FRAME_OFFSETS)
            or manifest.get("temporal_frame_interval_us")
            != REACTIVE_BEVFORMER_FRAME_INTERVAL_US
        ):
            raise ValueError("nuPlan BEV evaluation contract is incomplete")
        return

    temporal_contract = manifest.get("bevformer_temporal_contract")
    if (
        manifest.get("has_bevformer_history") is not True
        or temporal_contract
        != {
            "frame_count": len(REACTIVE_BEVFORMER_FRAME_OFFSETS),
            "history_frame_count": REACTIVE_BEVFORMER_HISTORY_FRAMES,
            "frame_interval_us": REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
            "frame_offsets": list(REACTIVE_BEVFORMER_FRAME_OFFSETS),
            "history_reference_frame": "current_ego",
        }
    ):
        raise ValueError("KITScenes BEV evaluation requires exact v12 T8 data")


def validate_kitscenes_benchmark_inventory_coverage(
    inventory: Mapping[str, Any],
    manifest_identities: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Bind KITScenes evaluation partitions to the pinned official inventory."""
    splits = inventory.get("splits")
    if (
        inventory.get("schema_version")
        != "kitscenes_benchmark_inventory_v1"
        or inventory.get("dataset") != KITSCENES_DATASET
        or inventory.get("dataset_revision") != KITSCENES_DATA_REVISION
        or inventory.get("sdk_revision") != KITSCENES_SDK_REVISION
        or not isinstance(splits, Mapping)
        or set(splits) != KITSCENES_BENCHMARK_SPLITS
    ):
        raise ValueError("KITScenes benchmark inventory contract differs")

    expected_pairs: set[tuple[str, str]] = set()
    expected_counts: dict[str, int] = {}
    for source_split in sorted(KITSCENES_BENCHMARK_SPLITS):
        split_payload = splits[source_split]
        if not isinstance(split_payload, Mapping):
            raise ValueError("KITScenes benchmark split inventory is invalid")
        archives = split_payload.get("archives")
        expected_count = split_payload.get("expected_scene_count")
        selected_count = split_payload.get("selected_scene_count")
        if (
            split_payload.get("split") != source_split
            or split_payload.get("source_revision")
            != KITSCENES_DATA_REVISION
            or split_payload.get("missing_scene_ids") != []
            or isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count <= 0
            or selected_count != expected_count
            or not isinstance(archives, list)
            or len(archives) != expected_count
        ):
            raise ValueError(
                "KITScenes benchmark split inventory is incomplete"
            )
        scene_ids = []
        for archive in archives:
            if not isinstance(archive, Mapping):
                raise ValueError(
                    "KITScenes benchmark archive inventory is invalid"
                )
            scene_id = archive.get("scene_id")
            archive_sha256 = archive.get("archive_sha256")
            archive_size_bytes = archive.get("archive_size_bytes")
            if (
                not isinstance(scene_id, str)
                or not scene_id
                or not isinstance(archive_sha256, str)
                or len(archive_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in archive_sha256
                )
                or isinstance(archive_size_bytes, bool)
                or not isinstance(archive_size_bytes, int)
                or archive_size_bytes <= 0
            ):
                raise ValueError(
                    "KITScenes benchmark archive inventory is invalid"
                )
            scene_ids.append(scene_id)
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError(
                "KITScenes benchmark split inventory duplicates scenes"
            )
        expected_pairs.update(
            (source_split, shard_partition_id([scene_id]))
            for scene_id in scene_ids
        )
        expected_counts[source_split] = len(scene_ids)

    total_scene_count = inventory.get("total_scene_count")
    if (
        isinstance(total_scene_count, bool)
        or not isinstance(total_scene_count, int)
        or total_scene_count != len(expected_pairs)
    ):
        raise ValueError("KITScenes benchmark inventory total is invalid")

    observed_pairs: set[tuple[str, str]] = set()
    for identity in manifest_identities:
        observed_source_split = identity.get("source_split")
        partition_id = identity.get("partition_id")
        total_samples = identity.get("total_samples")
        if (
            identity.get("data_role") != "benchmark"
            or identity.get("source_revision") != KITSCENES_DATA_REVISION
            or observed_source_split not in KITSCENES_BENCHMARK_SPLITS
            or not isinstance(partition_id, str)
            or not partition_id
            or isinstance(total_samples, bool)
            or not isinstance(total_samples, int)
            or total_samples < 0
        ):
            raise ValueError(
                "KITScenes benchmark manifest identity is invalid"
            )
        pair = (str(observed_source_split), partition_id)
        if pair in observed_pairs:
            raise ValueError(
                "KITScenes benchmark manifests duplicate a partition"
            )
        observed_pairs.add(pair)
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        unexpected = sorted(observed_pairs - expected_pairs)
        raise ValueError(
            "KITScenes benchmark partition coverage differs from the pinned "
            f"inventory: missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    ordered_pairs = [
        f"{source_split}:{partition_id}"
        for source_split, partition_id in sorted(expected_pairs)
    ]
    return {
        "inventory_complete": True,
        "official_scene_count": len(expected_pairs),
        "official_scene_count_by_split": expected_counts,
        "partition_identity_sha256": hashlib.sha256(
            "\n".join(ordered_pairs).encode("utf-8")
        ).hexdigest(),
    }


def load_reactive_bev_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    dict[str, Any],
    dict[str, Any],
    str,
    int,
]:
    """Load and verify one Reactive checkpoint without pretrained side effects."""
    from model_components.auto_e2e import AutoE2E

    path = Path(checkpoint_path)
    checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Reactive BEV checkpoint must contain an object")
    config = payload.get("config")
    metrics = payload.get("metrics")
    state_dict = payload.get("model_state_dict")
    if (
        not isinstance(config, Mapping)
        or not isinstance(metrics, Mapping)
        or not isinstance(state_dict, Mapping)
    ):
        raise ValueError(
            "Reactive BEV checkpoint is missing config, metrics, or weights"
        )
    config = dict(config)
    metrics = dict(metrics)
    checkpoint_epoch = payload.get("epoch")
    if (
        isinstance(checkpoint_epoch, bool)
        or not isinstance(checkpoint_epoch, int)
        or checkpoint_epoch <= 0
    ):
        raise ValueError("Reactive BEV checkpoint has an invalid epoch")
    if (
        config.get("model_architecture_version")
        != REACTIVE_MODEL_ARCHITECTURE_VERSION
        or config.get("bev_taxonomy_version")
        != BEV_SEGMENTATION_TAXONOMY_VERSION
        or config.get("bev_loss_version")
        not in SUPPORTED_BEV_SEGMENTATION_AUXILIARY_LOSS_VERSIONS
        or config.get("training_stage") != "nuplan_full"
        or config.get("enable_bev_segmentation") is not True
        or int(config.get("bev_segmentation_classes", 0))
        != len(BEV_SEGMENTATION_CLASSES)
    ):
        raise ValueError("checkpoint does not implement the reviewed BEV contract")
    if (
        config.get("bev_loss_version")
        == BEV_SEGMENTATION_AUXILIARY_LOSS_VERSION
    ):
        frequencies = torch.as_tensor(
            config.get("bev_positive_pair_frequencies", ()),
            dtype=torch.float64,
        )
        class_weights = torch.as_tensor(
            config.get("bev_class_weights", ()),
            dtype=torch.float64,
        )
        class_count = len(BEV_SEGMENTATION_CLASSES)
        if (
            frequencies.shape != (class_count,)
            or not torch.isfinite(frequencies).all()
            or bool((frequencies <= 0.0).any())
            or bool((frequencies > 1.0).any())
            or class_weights.shape != (class_count,)
            or not torch.isfinite(class_weights).all()
            or bool((class_weights <= 0.0).any())
            or config.get("bev_sampling_importance_correction")
            != BEV_SAMPLING_IMPORTANCE_CORRECTION_VERSION
            or config.get("bev_checkpoint_quality_guard_version")
            != BEV_CHECKPOINT_QUALITY_GUARD_VERSION
            or config.get("bev_checkpoint_min_class_iou")
            != BEV_CHECKPOINT_MIN_CLASS_IOU
            or config.get("bev_checkpoint_min_class_precision")
            != BEV_CHECKPOINT_MIN_CLASS_PRECISION
        ):
            raise ValueError(
                "current BEV checkpoint lacks sampling or quality provenance"
            )
        if config.get("training_scope") == "bev_only":
            head_initialization = config.get("bev_head_initialization")
            raw_statistics = config.get("bev_raw_statistics")
            if (
                not isinstance(head_initialization, Mapping)
                or head_initialization.get("version")
                != BEV_HEAD_INITIALIZATION_VERSION
                or not isinstance(raw_statistics, Mapping)
            ):
                raise ValueError(
                    "current BEV checkpoint lacks sample-normalized "
                    "initialization provenance"
                )
            active_samples = torch.as_tensor(
                raw_statistics.get("active_sample_count", ()),
                dtype=torch.float64,
            )
            effective_exposure_count = raw_statistics.get(
                "effective_exposure_count"
            )
            positive_fraction_sum = torch.as_tensor(
                raw_statistics.get("positive_fraction_sum", ()),
                dtype=torch.float64,
            )
            if (
                isinstance(effective_exposure_count, bool)
                or not isinstance(effective_exposure_count, int)
                or effective_exposure_count <= 0
                or active_samples.shape != (class_count,)
                or positive_fraction_sum.shape != (class_count,)
                or not torch.isfinite(active_samples).all()
                or not torch.isfinite(positive_fraction_sum).all()
                or bool((active_samples <= 0.0).any())
                or bool(
                    (active_samples > effective_exposure_count).any()
                )
                or bool((positive_fraction_sum <= 0.0).any())
                or bool(
                    (positive_fraction_sum >= active_samples).any()
                )
            ):
                raise ValueError(
                    "current BEV checkpoint has invalid sample-normalized "
                    "statistics"
                )
    checkpoint_bev_probability_bins(config)
    expected_config_sha256 = payload.get("config_sha256")
    actual_config_sha256 = reactive_config_sha256(config)
    if expected_config_sha256 != actual_config_sha256:
        raise ValueError("Reactive BEV checkpoint config digest differs")
    expected_state_sha256 = payload.get("model_state_sha256")
    actual_state_sha256 = reactive_model_state_sha256(state_dict)
    if expected_state_sha256 != actual_state_sha256:
        raise ValueError("Reactive BEV checkpoint model digest differs")

    valid_constructor_keys = (
        set(inspect.signature(AutoE2E.__init__).parameters) - {"self"}
    )
    model_kwargs = {
        key: value
        for key, value in config.items()
        if key in valid_constructor_keys
    }
    model_kwargs["is_pretrained"] = False
    model = AutoE2E(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, metrics, checkpoint_sha256, checkpoint_epoch


def checkpoint_validation_thresholds(
    metrics: Mapping[str, Any],
) -> tuple[float, ...]:
    """Read validation-selected thresholds from current or legacy checkpoints."""
    thresholds, _ = checkpoint_validation_thresholds_with_protocol(metrics)
    return thresholds


def checkpoint_validation_thresholds_with_protocol(
    metrics: Mapping[str, Any],
) -> tuple[tuple[float, ...], str]:
    """Read thresholds and disclose the checkpoint metric-key protocol."""
    thresholds = []
    protocols = set()
    for class_name in BEV_SEGMENTATION_CLASSES:
        current_key = (
            f"bev_{class_name}_best_iou_threshold_on_validation_set"
        )
        legacy_key = f"bev_{class_name}_calibrated_threshold"
        if current_key in metrics and legacy_key in metrics:
            raise ValueError(
                "checkpoint contains ambiguous BEV threshold metrics"
            )
        if current_key in metrics:
            raw_value = metrics[current_key]
            protocols.add("current_best_iou_on_validation_set")
        else:
            raw_value = metrics.get(legacy_key)
            protocols.add("legacy_calibrated_threshold")
        if isinstance(raw_value, bool):
            raise ValueError("checkpoint BEV threshold is invalid")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "checkpoint lacks a validation-selected BEV threshold for "
                f"{class_name}"
            ) from error
        if not torch.isfinite(torch.tensor(value)) or not 0.0 <= value <= 1.0:
            raise ValueError("checkpoint BEV threshold is invalid")
        thresholds.append(value)
    if len(protocols) != 1:
        raise ValueError("checkpoint mixes BEV threshold metric protocols")
    return tuple(thresholds), protocols.pop()


def _nuplan_targets(
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, tuple[bool, ...]]:
    available = batch.get("bev_segmentation_available")
    target = batch.get("bev_segmentation_target")
    valid = batch.get("bev_segmentation_valid")
    if (
        not torch.is_tensor(available)
        or not bool(available.to(dtype=torch.bool).all())
        or not torch.is_tensor(target)
        or not torch.is_tensor(valid)
    ):
        raise ValueError("nuPlan evaluation requires a BEV teacher per sample")
    return (
        target.to(dtype=torch.float32),
        valid.to(dtype=torch.bool),
        (True,) * len(BEV_SEGMENTATION_CLASSES),
    )


def _evaluation_targets(
    batch: Mapping[str, Any],
    *,
    dataset: str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[bool, ...]]:
    if dataset == NUPLAN_DATASET:
        return _nuplan_targets(batch)
    if dataset == KITSCENES_DATASET:
        return kitscenes_static_bev_targets(
            batch["map_context"],
            batch["map_valid"],
        )
    raise ValueError(f"unsupported BEV evaluation dataset {dataset!r}")


def evaluate_reactive_bev_model(
    model: torch.nn.Module,
    loader: Iterable[Any],
    *,
    dataset: str,
    device: torch.device,
    checkpoint_sha256: str,
    probability_bins: int = 1024,
    reference_thresholds: tuple[float, ...] | None = None,
    reference_threshold_source: str | None = None,
    evaluation_split_contract: Mapping[str, object] | None = None,
    precision: str = "bf16",
    expected_sample_uids: Sequence[str] | None = None,
    expected_sample_count: int | None = None,
) -> dict[str, object]:
    """Evaluate one checkpoint with exact T8 inputs and dataset-valid labels."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported BEV evaluation dataset {dataset!r}")
    if precision not in {"bf16", "fp32"}:
        raise ValueError("unsupported BEV evaluation precision")
    resolved_expected_uids = (
        tuple(str(value) for value in expected_sample_uids)
        if expected_sample_uids is not None
        else None
    )
    if resolved_expected_uids is not None and (
        not resolved_expected_uids
        or any(not value for value in resolved_expected_uids)
        or len(set(resolved_expected_uids)) != len(resolved_expected_uids)
    ):
        raise ValueError(
            "expected BEV evaluation sample UIDs must be non-empty and unique"
        )
    if expected_sample_count is not None and (
        isinstance(expected_sample_count, bool)
        or not isinstance(expected_sample_count, int)
        or expected_sample_count <= 0
    ):
        raise ValueError(
            "expected BEV evaluation sample count must be positive"
        )
    if (
        resolved_expected_uids is not None
        and expected_sample_count is not None
        and len(resolved_expected_uids) != expected_sample_count
    ):
        raise ValueError(
            "expected BEV evaluation sample count differs from its UID set"
        )
    resolved_capacity_sample_count = (
        expected_sample_count
        if expected_sample_count is not None
        else (
            len(resolved_expected_uids)
            if resolved_expected_uids is not None
            else None
        )
    )
    if resolved_capacity_sample_count is not None:
        validate_fixed_point_bootstrap_capacity(
            resolved_capacity_sample_count,
            maximum_cells_per_sample=(
                AUTOE2E_NAVIGATION_GEOMETRY.height_px
                * AUTOE2E_NAVIGATION_GEOMETRY.width_px
            ),
        )
    class_availability = (
        (True,) * len(BEV_SEGMENTATION_CLASSES)
        if dataset == NUPLAN_DATASET
        else (True, False, True, True, False, False, False, False)
    )
    unavailable_reasons = (
        {}
        if dataset == NUPLAN_DATASET
        else {
            **{
                class_name: KITSCENES_DYNAMIC_UNAVAILABLE_REASON
                for class_name in BEV_SEGMENTATION_CLASSES[5:]
            },
            "lane_boundary": KITSCENES_THIN_CLASS_UNAVAILABLE_REASON,
            "stop_line": KITSCENES_THIN_CLASS_UNAVAILABLE_REASON,
        }
    )
    accumulator = BEVSegmentationMetricAccumulator(
        probability_bins=probability_bins,
        class_availability=class_availability,
        unavailable_reasons=unavailable_reasons,
        reference_thresholds=reference_thresholds,
        reference_threshold_source=reference_threshold_source,
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for item in loader:
                raw_batch, fallback_projection, fallback_geometry = (
                    _loader_item(item)
                )
                batch = _batch_to_device(raw_batch, device)
                if dataset == KITSCENES_DATASET:
                    forbidden_front_fields = sorted(
                        field
                        for field in (
                            "front_camera_tile",
                            "front_camera_projection_matrix",
                            "front_camera_fpn_tile",
                            "front_camera_fpn_available",
                        )
                        if field in batch
                    )
                    if forbidden_front_fields:
                        raise ValueError(
                            "KITScenes evaluation must not contain native "
                            "front-camera companion fields: "
                            f"{forbidden_front_fields}"
                        )
                projection, geometry_type = (
                    resolve_reactive_batch_projection(
                        batch,
                        fallback_projection,
                        fallback_geometry,
                        device=device,
                    )
                )
                front_projection = resolve_reactive_front_projection(
                    batch,
                    geometry_type,
                    device=device,
                    required=(dataset == NUPLAN_DATASET),
                )
                camera_history_tiles, history_projections = (
                    resolve_reactive_camera_history(
                        batch,
                        geometry_type,
                        device=device,
                        required=True,
                    )
                )
                reset_visual_history = getattr(
                    model,
                    "reset_visual_history",
                    None,
                )
                if callable(reset_visual_history):
                    reset_visual_history()
                with torch.autocast(
                    device_type=device.type,
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
                        mode="infer",
                        return_auxiliary=True,
                        compute_bev_segmentation=True,
                        compute_route_reconstruction=False,
                        bev_only=True,
                    )
                if not isinstance(output, tuple):
                    raise RuntimeError("BEV evaluator received no auxiliary output")
                _, auxiliary = output
                logits = auxiliary.get("bev_segmentation_logits")
                if not torch.is_tensor(logits):
                    raise RuntimeError("BEV evaluator received no logits")
                target, valid, availability = _evaluation_targets(
                    batch,
                    dataset=dataset,
                )
                if availability != class_availability:
                    raise RuntimeError("BEV evaluation class availability changed")
                accumulator.update(
                    logits,
                    target,
                    valid,
                    sample_uids=batch.get("sample_uid"),
                )
    finally:
        model.train(was_training)

    observed_sample_uids = accumulator.seen_sample_uids
    if resolved_expected_uids is not None:
        expected_uid_set = set(resolved_expected_uids)
        if observed_sample_uids != expected_uid_set:
            missing = sorted(expected_uid_set - observed_sample_uids)
            unexpected = sorted(observed_sample_uids - expected_uid_set)
            raise ValueError(
                "BEV evaluation sample coverage differs from the intended "
                f"split: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
    resolved_expected_count = (
        expected_sample_count
        if expected_sample_count is not None
        else (
            len(resolved_expected_uids)
            if resolved_expected_uids is not None
            else None
        )
    )
    if (
        resolved_expected_count is not None
        and accumulator.observed_sample_count != resolved_expected_count
    ):
        raise ValueError(
            "BEV evaluation observed sample count differs from the intended "
            f"split: expected={resolved_expected_count} "
            f"observed={accumulator.observed_sample_count}"
        )
    report = accumulator.report(
        dataset=dataset,
        checkpoint_sha256=checkpoint_sha256,
        label_source=(
            "nuplan_lidar_and_map_teacher"
            if dataset == NUPLAN_DATASET
            else "kitscenes_lanelet2_static_map"
        ),
        temporal_input=(
            "exact_t8_0p5s_current_ego_aligned"
            if dataset == NUPLAN_DATASET
            else "t8_target_0p5s_max_error_0p05s_current_ego_aligned"
        ),
        label_provenance=(
            {
                "geometry_id": AUTOE2E_NAVIGATION_GEOMETRY.geometry_id,
                "meters_per_pixel": (
                    AUTOE2E_NAVIGATION_GEOMETRY.meters_per_pixel
                ),
                "resample_mode": "native",
                "class_definitions": "nuplan_native_v1",
            }
            if dataset == NUPLAN_DATASET
            else {
                "source_geometry_id": (
                    DEFAULT_NAVIGATION_GEOMETRY.geometry_id
                ),
                "source_meters_per_pixel": (
                    DEFAULT_NAVIGATION_GEOMETRY.meters_per_pixel
                ),
                "target_geometry_id": (
                    AUTOE2E_NAVIGATION_GEOMETRY.geometry_id
                ),
                "target_meters_per_pixel": (
                    AUTOE2E_NAVIGATION_GEOMETRY.meters_per_pixel
                ),
                "resample_mode": "nearest",
                "supported_class_primitives": {
                    "drivable_area": "lanelet_polygon",
                    "intersection": "lanelet_polygon_heuristic",
                    "crosswalk": "lanelet_polygon",
                },
                "unsupported_thin_class_primitive": "1.0_m_polyline",
                "prediction_label_input_circularity": "none",
                "circularity_evidence": (
                    "bev_only_segmentation_logits_are_emitted_from_image_bev_"
                    "before_navigation_encoder"
                ),
            }
        ),
        camera_input_abi=(
            {
                "base_camera_count": 6,
                "base_camera_image_size": REACTIVE_CAMERA_IMAGE_SIZE,
                "native_front_companion_present": True,
                "native_front_image_size": (
                    REACTIVE_FRONT_CAMERA_IMAGE_SIZE
                ),
                "front_residual_branch_active": True,
                "matches_training_input": True,
            }
            if dataset == NUPLAN_DATASET
            else {
                "base_camera_count": 6,
                "base_camera_image_size": REACTIVE_CAMERA_IMAGE_SIZE,
                "native_front_companion_present": False,
                "native_front_image_size": None,
                "front_residual_branch_active": False,
                "matches_training_input": False,
            }
        ),
        evaluation_split_contract=evaluation_split_contract,
    )
    report["evaluation_precision"] = precision
    report["expected_sample_count"] = resolved_expected_count
    report["sample_coverage_complete"] = (
        resolved_expected_count is not None
        and accumulator.observed_sample_count == resolved_expected_count
    )
    report["prediction_provenance"] = {
        "segmentation_head_input": (
            "image_bev_before_navigation_fusion"
        ),
        "map_context_consumed_by_segmentation_head": False,
        "route_mask_consumed_by_segmentation_head": False,
        "bev_only_returns_before_navigation_encoder": True,
    }
    return report
