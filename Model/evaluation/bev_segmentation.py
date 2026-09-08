"""Cross-dataset BEV segmentation targets and class-level metrics."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F

from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_CLASSES,
    BEV_SEGMENTATION_TAXONOMY_VERSION,
)
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    DEFAULT_NAVIGATION_GEOMETRY,
    MapChannel,
    NavigationRasterGeometry,
)


BEV_SEGMENTATION_EVALUATION_VERSION: Final = "bev_segmentation_evaluation_v7"
BEV_AP_BOOTSTRAP_REPLICATES: Final = 256
BEV_AP_BOOTSTRAP_BINS: Final = 1024
BEV_AP_BOOTSTRAP_WEIGHT_SCALE: Final = 1_000_000
BEV_AP_BOOTSTRAP_MAX_WEIGHT: Final = 32.0
BEV_AP_BOOTSTRAP_VERSION: Final = (
    "sample_bayesian_hash_fixed_point_bootstrap_v5"
)
KITSCENES_STATIC_BEV_CLASS_CHANNELS: Final = {
    "drivable_area": MapChannel.DRIVABLE_AREA,
    "intersection": MapChannel.INTERSECTION,
    "crosswalk": MapChannel.CROSSWALK,
}
KITSCENES_DYNAMIC_UNAVAILABLE_REASON: Final = (
    "KITScenes packed annotations do not provide dynamic object occupancy "
    "ground truth for this taxonomy"
)
KITSCENES_THIN_CLASS_UNAVAILABLE_REASON: Final = (
    "KITScenes packed map rasters use 1.0 m polylines at 1.0 m/px, which "
    "does not match the nuPlan thin-class target definition at 0.4 m/px"
)


def _metric_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _ordered_string_sequence_sha256(values: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, byteorder="big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def fixed_point_bayesian_bootstrap_weights(
    sample_uids: Sequence[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return deterministic positive Bayesian weights as fixed-point units."""
    resolved_sample_uids = tuple(str(value) for value in sample_uids)
    if (
        not resolved_sample_uids
        or any(not value for value in resolved_sample_uids)
        or len(set(resolved_sample_uids)) != len(resolved_sample_uids)
    ):
        raise ValueError(
            "BEV bootstrap sample UIDs must be non-empty and unique"
        )
    maximum_units = int(
        BEV_AP_BOOTSTRAP_MAX_WEIGHT * BEV_AP_BOOTSTRAP_WEIGHT_SCALE
    )
    weight_units = np.empty(
        (
            len(resolved_sample_uids),
            BEV_AP_BOOTSTRAP_REPLICATES,
        ),
        dtype=np.int64,
    )
    uniform_scale = math.ldexp(1.0, -52)
    for sample_index, sample_uid in enumerate(resolved_sample_uids):
        for replicate_index in range(BEV_AP_BOOTSTRAP_REPLICATES):
            digest = hashlib.sha256(
                (
                    BEV_AP_BOOTSTRAP_VERSION
                    + "\0"
                    + sample_uid
                    + "\0"
                    + str(replicate_index)
                ).encode("utf-8")
            ).digest()
            mantissa = int.from_bytes(
                digest[:8],
                byteorder="big",
                signed=False,
            ) >> 12
            uniform = (float(mantissa) + 0.5) * uniform_scale
            exponential_weight = min(
                -math.log1p(-uniform),
                BEV_AP_BOOTSTRAP_MAX_WEIGHT,
            )
            weight_units[sample_index, replicate_index] = min(
                maximum_units,
                max(
                    1,
                    int(
                        math.floor(
                            exponential_weight
                            * BEV_AP_BOOTSTRAP_WEIGHT_SCALE
                            + 0.5
                        )
                    ),
                ),
            )
    return torch.as_tensor(
        weight_units,
        dtype=torch.int64,
        device=device,
    ).transpose(0, 1)


def validate_fixed_point_bootstrap_capacity(
    expected_sample_count: int,
    *,
    maximum_cells_per_sample: int,
) -> None:
    """Fail before evaluation when worst-case fixed-point counts can overflow."""
    if (
        isinstance(expected_sample_count, bool)
        or not isinstance(expected_sample_count, int)
        or expected_sample_count <= 0
        or isinstance(maximum_cells_per_sample, bool)
        or not isinstance(maximum_cells_per_sample, int)
        or maximum_cells_per_sample <= 0
    ):
        raise ValueError("BEV bootstrap capacity inputs must be positive")
    maximum_units = int(
        BEV_AP_BOOTSTRAP_MAX_WEIGHT * BEV_AP_BOOTSTRAP_WEIGHT_SCALE
    )
    maximum_count = (
        expected_sample_count
        * maximum_cells_per_sample
        * maximum_units
    )
    if maximum_count > torch.iinfo(torch.int64).max:
        raise OverflowError(
            "BEV fixed-point bootstrap can overflow for the expected "
            "evaluation size"
        )


def _reverse_cumulative_histogram(
    histogram: torch.Tensor,
) -> torch.Tensor:
    return torch.cumsum(
        histogram.to(torch.float64).flip(0),
        dim=0,
    )


def accumulate_fixed_point_bootstrap_histogram(
    accumulator: torch.Tensor,
    weight_units: torch.Tensor,
    per_sample_histogram: torch.Tensor,
) -> None:
    """Add one batch exactly, independent of batch partition and order."""
    if (
        accumulator.dtype is not torch.int64
        or weight_units.dtype is not torch.int64
        or per_sample_histogram.dtype is not torch.int64
        or accumulator.ndim != 3
        or weight_units.ndim != 2
        or per_sample_histogram.ndim != 3
        or accumulator.shape[0] != weight_units.shape[0]
        or accumulator.shape[1:] != per_sample_histogram.shape[1:]
        or weight_units.shape[1] != per_sample_histogram.shape[0]
        or accumulator.device != weight_units.device
        or accumulator.device != per_sample_histogram.device
    ):
        raise ValueError("BEV fixed-point bootstrap tensors are invalid")
    if bool((weight_units <= 0).any()) or bool(
        (per_sample_histogram < 0).any()
    ):
        raise ValueError(
            "BEV fixed-point bootstrap values must be non-negative"
        )

    maximum_increment = 0
    sample_maxima = (
        per_sample_histogram.reshape(
            per_sample_histogram.shape[0],
            -1,
        )
        .max(dim=1)
        .values.cpu()
        .tolist()
    )
    for replicate_weights in weight_units.cpu().tolist():
        maximum_increment = max(
            maximum_increment,
            sum(
                int(weight) * int(sample_maximum)
                for weight, sample_maximum in zip(
                    replicate_weights,
                    sample_maxima,
                    strict=True,
                )
            ),
        )
    maximum_accumulated = int(accumulator.max().item())
    if maximum_increment > (
        torch.iinfo(torch.int64).max - maximum_accumulated
    ):
        raise OverflowError("BEV fixed-point bootstrap histogram overflow")

    for sample_index in range(per_sample_histogram.shape[0]):
        accumulator.add_(
            weight_units[:, sample_index, None, None]
            * per_sample_histogram[sample_index][None, :, :]
        )


def _histogram_average_precision(
    positive_histogram: torch.Tensor,
    negative_histogram: torch.Tensor,
) -> float:
    if (
        positive_histogram.ndim != 1
        or negative_histogram.shape != positive_histogram.shape
        or positive_histogram.numel() <= 1
    ):
        raise ValueError("BEV score histograms must be matching 1D tensors")
    positive_total = positive_histogram.sum(dtype=torch.int64).to(
        torch.float64
    ) if positive_histogram.dtype is torch.int64 else (
        positive_histogram.to(torch.float64).sum()
    )
    if float(positive_total.item()) <= 0.0:
        raise ValueError("BEV evaluation class has no positive cells")
    cumulative_positive = _reverse_cumulative_histogram(
        positive_histogram
    )
    cumulative_negative = _reverse_cumulative_histogram(
        negative_histogram
    )
    precision = cumulative_positive / (
        cumulative_positive + cumulative_negative
    ).clamp_min(1.0)
    recall = cumulative_positive / positive_total
    recall_delta = torch.diff(
        torch.cat((recall.new_zeros(1), recall))
    )
    return float((recall_delta * precision).sum().item())


def _histogram_best_iou_operating_point(
    positive_histogram: torch.Tensor,
    negative_histogram: torch.Tensor,
) -> tuple[float, float, float, float]:
    positive_total = positive_histogram.sum(dtype=torch.int64).to(
        torch.float64
    ) if positive_histogram.dtype is torch.int64 else (
        positive_histogram.to(torch.float64).sum()
    )
    if float(positive_total.item()) <= 0.0:
        raise ValueError("BEV evaluation class has no positive cells")
    true_positive = _reverse_cumulative_histogram(positive_histogram)
    false_positive = _reverse_cumulative_histogram(negative_histogram)
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


def _histogram_operating_point_at_threshold(
    positive_histogram: torch.Tensor,
    negative_histogram: torch.Tensor,
    threshold: float,
) -> tuple[float, float, float]:
    if (
        positive_histogram.ndim != 1
        or negative_histogram.shape != positive_histogram.shape
        or positive_histogram.numel() <= 1
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("BEV threshold operating point is invalid")
    positive_histogram = positive_histogram.to(torch.float64)
    negative_histogram = negative_histogram.to(torch.float64)
    bins = positive_histogram.numel()
    first_positive_bin = min(
        bins,
        max(0, math.ceil(threshold * bins - 1e-12)),
    )
    if first_positive_bin == bins:
        true_positive = positive_histogram.new_zeros(())
        false_positive = negative_histogram.new_zeros(())
    else:
        true_positive = positive_histogram[first_positive_bin:].sum()
        false_positive = negative_histogram[first_positive_bin:].sum()
    positive_total = positive_histogram.sum()
    false_negative = positive_total - true_positive
    iou = true_positive / (
        true_positive + false_positive + false_negative
    ).clamp_min(1.0)
    precision = true_positive / (
        true_positive + false_positive
    ).clamp_min(1.0)
    recall = true_positive / positive_total.clamp_min(1.0)
    return (
        float(iou.item()),
        float(precision.item()),
        float(recall.item()),
    )


def _target_to_source_grid(
    source_geometry: NavigationRasterGeometry,
    target_geometry: NavigationRasterGeometry,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    target_x, target_y = target_geometry.pixel_center_grids()
    target_points = torch.as_tensor(
        np.column_stack(
            (target_x.reshape(-1), target_y.reshape(-1))
        ),
        device=device,
        dtype=dtype,
    )
    source_rows = (
        source_geometry.x_max_m - target_points[:, 0]
    ) / source_geometry.meters_per_pixel - 0.5
    source_columns = (
        source_geometry.y_max_m - target_points[:, 1]
    ) / source_geometry.meters_per_pixel - 0.5
    normalized_x = (
        2.0 * (source_columns + 0.5) / source_geometry.width_px - 1.0
    )
    normalized_y = (
        2.0 * (source_rows + 0.5) / source_geometry.height_px - 1.0
    )
    return torch.stack((normalized_x, normalized_y), dim=1).reshape(
        target_geometry.height_px,
        target_geometry.width_px,
        2,
    )


def kitscenes_static_bev_targets(
    map_context: torch.Tensor,
    map_valid: torch.Tensor,
    *,
    source_geometry: NavigationRasterGeometry = (
        DEFAULT_NAVIGATION_GEOMETRY
    ),
    target_geometry: NavigationRasterGeometry = (
        AUTOE2E_NAVIGATION_GEOMETRY
    ),
) -> tuple[torch.Tensor, torch.Tensor, tuple[bool, ...]]:
    """Reproject KITScenes static map labels into the nuPlan BEV taxonomy."""
    if map_context.ndim != 4:
        raise ValueError("KITScenes map context must have shape [B,C,H,W]")
    if map_context.shape[1] != len(MapChannel):
        raise ValueError("KITScenes map context channel count differs")
    if tuple(map_context.shape[-2:]) != (
        source_geometry.height_px,
        source_geometry.width_px,
    ):
        raise ValueError("KITScenes map context geometry differs")
    if map_valid.shape != (map_context.shape[0],):
        raise ValueError("KITScenes map validity must have shape [B]")
    if not bool(torch.isfinite(map_context).all()):
        raise ValueError("KITScenes map context contains non-finite values")

    supported_class_names = tuple(KITSCENES_STATIC_BEV_CLASS_CHANNELS)
    selected_channels = [
        int(KITSCENES_STATIC_BEV_CLASS_CHANNELS[class_name])
        for class_name in supported_class_names
    ]
    source = torch.cat(
        (
            map_context[:, selected_channels].to(dtype=torch.float32),
            map_context[
                :, int(MapChannel.KNOWN_MAP_AREA):(
                    int(MapChannel.KNOWN_MAP_AREA) + 1
                )
            ].to(dtype=torch.float32),
        ),
        dim=1,
    )
    grid = _target_to_source_grid(
        source_geometry,
        target_geometry,
        device=source.device,
        dtype=source.dtype,
    )[None].expand(source.shape[0], -1, -1, -1)
    reprojected = F.grid_sample(
        source,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    static_target = (
        reprojected[:, :len(supported_class_names)] >= 0.5
    ).to(torch.float32)
    known_map = (
        (reprojected[:, -1:] >= 0.5)
        & map_valid.to(device=source.device, dtype=torch.bool)[:, None, None, None]
    )
    target = source.new_zeros(
        (
            source.shape[0],
            len(BEV_SEGMENTATION_CLASSES),
            target_geometry.height_px,
            target_geometry.width_px,
        )
    )
    valid = torch.zeros_like(target, dtype=torch.bool)
    for source_index, class_name in enumerate(supported_class_names):
        class_index = BEV_SEGMENTATION_CLASSES.index(class_name)
        target[:, class_index] = static_target[:, source_index]
        valid[:, class_index] = known_map[:, 0]
    supported = tuple(
        class_name in KITSCENES_STATIC_BEV_CLASS_CHANNELS
        for class_name in BEV_SEGMENTATION_CLASSES
    )
    return target, valid, supported


class BEVSegmentationMetricAccumulator:
    """Accumulate class metrics and disclosed threshold operating points."""

    def __init__(
        self,
        *,
        probability_bins: int = 1024,
        class_availability: Sequence[bool] | None = None,
        unavailable_reasons: Mapping[str, str] | None = None,
        reference_thresholds: Sequence[float] | None = None,
        reference_threshold_source: str | None = None,
    ) -> None:
        if probability_bins <= 1:
            raise ValueError("probability_bins must be greater than one")
        class_count = len(BEV_SEGMENTATION_CLASSES)
        availability = (
            tuple(bool(value) for value in class_availability)
            if class_availability is not None
            else (True,) * class_count
        )
        if len(availability) != class_count:
            raise ValueError("class availability differs from taxonomy")
        self.probability_bins = int(probability_bins)
        self.class_availability = availability
        self.unavailable_reasons = dict(unavailable_reasons or {})
        if reference_thresholds is None:
            self.reference_thresholds = None
        else:
            thresholds = tuple(float(value) for value in reference_thresholds)
            if (
                len(thresholds) != class_count
                or any(
                    not math.isfinite(value) or not 0.0 <= value <= 1.0
                    for value in thresholds
                )
            ):
                raise ValueError("reference thresholds differ from taxonomy")
            self.reference_thresholds = thresholds
        if (
            self.reference_thresholds is None
        ) != (reference_threshold_source is None):
            raise ValueError(
                "reference thresholds and their source must be configured "
                "together"
            )
        self.reference_threshold_source = reference_threshold_source
        self.counts = torch.zeros((class_count, 5), dtype=torch.int64)
        self.positive_histogram = torch.zeros(
            (class_count, probability_bins),
            dtype=torch.int64,
        )
        self.negative_histogram = torch.zeros_like(
            self.positive_histogram
        )
        self.bootstrap_positive_histogram = torch.zeros(
            (
                BEV_AP_BOOTSTRAP_REPLICATES,
                class_count,
                BEV_AP_BOOTSTRAP_BINS,
            ),
            dtype=torch.int64,
        )
        self.bootstrap_negative_histogram = torch.zeros_like(
            self.bootstrap_positive_histogram
        )
        self.sample_count = 0
        self.observed_sample_count = 0
        self.nonfinite_sample_count = 0
        self.seen_sample_uids: set[str] = set()
        self.sample_uid_order: list[str] = []
        self.batch_size_counts: Counter[int] = Counter()
        self.batch_size_sequence: list[int] = []

    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_uids: Sequence[str] | None = None,
    ) -> None:
        if logits.shape != target.shape or logits.shape != valid_mask.shape:
            raise ValueError("BEV evaluation tensors differ in shape")
        if logits.ndim != 4 or logits.shape[1] != len(
            BEV_SEGMENTATION_CLASSES
        ):
            raise ValueError("BEV evaluation tensors must have shape [B,8,H,W]")
        if not bool(torch.isfinite(target).all()):
            raise ValueError("BEV evaluation targets contain non-finite values")
        if valid_mask.dtype is not torch.bool:
            raise ValueError("BEV evaluation valid mask must be boolean")
        valid_targets = target[valid_mask]
        if bool(valid_targets.numel()) and (
            bool((valid_targets < 0.0).any())
            or bool((valid_targets > 1.0).any())
        ):
            raise ValueError("BEV evaluation targets must be in [0,1]")
        availability = torch.as_tensor(
            self.class_availability,
            device=valid_mask.device,
            dtype=torch.bool,
        ).view(1, -1, 1, 1)
        if bool((valid_mask & ~availability).any()):
            raise ValueError("unavailable BEV class received valid cells")
        relevant_nonfinite = (
            ~torch.isfinite(logits)
            & valid_mask
            & availability
        )
        finite_samples = ~relevant_nonfinite.flatten(1).any(dim=1)
        batch_size = int(logits.shape[0])
        if sample_uids is None:
            raise ValueError("BEV evaluation requires stable sample UIDs")
        resolved_sample_uids = tuple(str(value) for value in sample_uids)
        if (
            len(resolved_sample_uids) != batch_size
            or any(not value for value in resolved_sample_uids)
        ):
            raise ValueError(
                "BEV evaluation sample UIDs must match the batch"
            )
        uid_counts = Counter(resolved_sample_uids)
        duplicate_uids = {
            sample_uid
            for sample_uid, count in uid_counts.items()
            if count > 1
        }
        duplicate_uids.update(
            set(resolved_sample_uids) & self.seen_sample_uids
        )
        if duplicate_uids:
            raise ValueError(
                "BEV evaluation sample UIDs contain duplicates: "
                f"{sorted(duplicate_uids)[:5]}"
            )
        self.batch_size_counts[batch_size] += 1
        self.batch_size_sequence.append(batch_size)
        self.sample_uid_order.extend(resolved_sample_uids)
        self.seen_sample_uids.update(resolved_sample_uids)
        self.observed_sample_count += batch_size
        self.nonfinite_sample_count += int((~finite_samples).sum().item())
        if not bool(finite_samples.any()):
            return
        finite_indices = finite_samples.nonzero(
            as_tuple=False
        ).flatten().tolist()
        resolved_sample_uids = tuple(
            resolved_sample_uids[index] for index in finite_indices
        )
        logits = logits[finite_samples]
        target = target[finite_samples]
        valid_mask = valid_mask[finite_samples]
        probability = logits.float().sigmoid()
        binary_target = target >= 0.5
        binary_prediction = probability >= 0.5
        bootstrap_weights = fixed_point_bayesian_bootstrap_weights(
            resolved_sample_uids,
            device=logits.device,
        )
        bootstrap_valid_mask = valid_mask & availability
        bootstrap_probability = torch.where(
            bootstrap_valid_mask,
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
        bootstrap_valid = bootstrap_valid_mask.flatten(2)
        bootstrap_target = binary_target.flatten(2)
        per_sample_positive_histogram = torch.zeros(
            (
                logits.shape[0],
                len(BEV_SEGMENTATION_CLASSES),
                BEV_AP_BOOTSTRAP_BINS,
            ),
            dtype=torch.int64,
            device=logits.device,
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
            self.bootstrap_positive_histogram,
            bootstrap_weights.cpu(),
            per_sample_positive_histogram.cpu(),
        )
        accumulate_fixed_point_bootstrap_histogram(
            self.bootstrap_negative_histogram,
            bootstrap_weights.cpu(),
            per_sample_negative_histogram.cpu(),
        )
        for class_index, available in enumerate(self.class_availability):
            class_valid = valid_mask[:, class_index].to(dtype=torch.bool)
            if not available:
                if bool(class_valid.any()):
                    raise ValueError(
                        "unavailable BEV class received valid cells"
                    )
                continue
            if not bool(class_valid.any()):
                continue
            class_target = binary_target[:, class_index][class_valid]
            class_prediction = binary_prediction[:, class_index][class_valid]
            class_probability = probability[:, class_index][class_valid]
            values = torch.stack((
                (class_prediction & class_target).sum(),
                (class_prediction & ~class_target).sum(),
                (~class_prediction & class_target).sum(),
                class_target.sum(),
                class_valid.sum(),
            )).to(device="cpu", dtype=torch.int64)
            self.counts[class_index] += values
            bins = torch.clamp(
                (
                    class_probability * self.probability_bins
                ).to(torch.int64),
                min=0,
                max=self.probability_bins - 1,
            )
            self.positive_histogram[class_index] += torch.bincount(
                bins[class_target],
                minlength=self.probability_bins,
            ).to(device="cpu", dtype=torch.int64)
            self.negative_histogram[class_index] += torch.bincount(
                bins[~class_target],
                minlength=self.probability_bins,
            ).to(device="cpu", dtype=torch.int64)
        self.sample_count += int(logits.shape[0])

    def report(
        self,
        *,
        dataset: str,
        checkpoint_sha256: str,
        label_source: str,
        temporal_input: str,
        label_provenance: Mapping[str, object] | None = None,
        camera_input_abi: Mapping[str, object] | None = None,
        evaluation_split_contract: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if self.sample_count <= 0:
            raise ValueError("BEV evaluation has no samples")
        classes: dict[str, dict[str, object]] = {}
        supported_average_precisions = []
        supported_ap_lifts = []
        supported_class_count = 0
        for class_index, class_name in enumerate(
            BEV_SEGMENTATION_CLASSES
        ):
            if not self.class_availability[class_index]:
                classes[class_name] = {
                    "availability": "unsupported",
                    "reason": self.unavailable_reasons.get(
                        class_name,
                        "ground truth is unavailable",
                    ),
                }
                continue
            (
                true_positive,
                false_positive,
                false_negative,
                positive,
                valid,
            ) = (
                float(value)
                for value in self.counts[class_index].tolist()
            )
            supported = positive > 0.0 and valid > 0.0
            prevalence = _metric_ratio(positive, valid)
            average_precision = 0.0
            ap_lift = 0.0
            best_iou_threshold = 0.5
            best_iou = 0.0
            best_iou_precision = 0.0
            best_iou_recall = 0.0
            reference_iou = 0.0
            reference_precision = 0.0
            reference_recall = 0.0
            ap_lift_bootstrap_lower_95 = 0.0
            ap_lift_bootstrap_upper_95 = 0.0
            if supported:
                supported_class_count += 1
                average_precision = _histogram_average_precision(
                    self.positive_histogram[class_index],
                    self.negative_histogram[class_index],
                )
                (
                    best_iou_threshold,
                    best_iou,
                    best_iou_precision,
                    best_iou_recall,
                ) = _histogram_best_iou_operating_point(
                    self.positive_histogram[class_index],
                    self.negative_histogram[class_index],
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
                            self.bootstrap_positive_histogram[
                                replicate_index,
                                class_index,
                            ]
                        )
                        replicate_negative_histogram = (
                            self.bootstrap_negative_histogram[
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
                            "BEV evaluation bootstrap has insufficient "
                            f"class support for {class_name}"
                        )
                    bootstrap_tensor = torch.as_tensor(
                        bootstrap_lifts,
                        dtype=torch.float64,
                    )
                    ap_lift_bootstrap_lower_95 = float(
                        torch.quantile(
                            bootstrap_tensor,
                            0.025,
                        ).item()
                    )
                    ap_lift_bootstrap_upper_95 = float(
                        torch.quantile(
                            bootstrap_tensor,
                            0.975,
                        ).item()
                    )
                supported_average_precisions.append(average_precision)
                supported_ap_lifts.append(ap_lift)
                if self.reference_thresholds is not None:
                    (
                        reference_iou,
                        reference_precision,
                        reference_recall,
                    ) = _histogram_operating_point_at_threshold(
                        self.positive_histogram[class_index],
                        self.negative_histogram[class_index],
                        self.reference_thresholds[class_index],
                    )
            classes[class_name] = {
                "availability": "computed",
                "supported": supported,
                "valid_cells": int(valid),
                "positive_cells": int(positive),
                "positive_prevalence": prevalence,
                "iou_at_0p5": _metric_ratio(
                    true_positive,
                    true_positive + false_positive + false_negative,
                ),
                "precision_at_0p5": _metric_ratio(
                    true_positive,
                    true_positive + false_positive,
                ),
                "recall_at_0p5": _metric_ratio(
                    true_positive,
                    true_positive + false_negative,
                ),
                "average_precision": average_precision,
                "ap_lift": ap_lift,
                "ap_lift_bootstrap_lower_95": (
                    ap_lift_bootstrap_lower_95
                ),
                "ap_lift_bootstrap_upper_95": (
                    ap_lift_bootstrap_upper_95
                ),
                "best_iou_threshold_on_evaluation_set": (
                    best_iou_threshold
                ),
                "best_iou_on_evaluation_set": best_iou,
                "best_iou_precision_on_evaluation_set": (
                    best_iou_precision
                ),
                "best_iou_recall_on_evaluation_set": (
                    best_iou_recall
                ),
            }
            if self.reference_thresholds is not None:
                classes[class_name].update({
                    "reference_threshold": (
                        self.reference_thresholds[class_index]
                    ),
                    "iou_at_reference_threshold": reference_iou,
                    "precision_at_reference_threshold": (
                        reference_precision
                    ),
                    "recall_at_reference_threshold": reference_recall,
                })
        return {
            "schema_version": BEV_SEGMENTATION_EVALUATION_VERSION,
            "taxonomy_version": BEV_SEGMENTATION_TAXONOMY_VERSION,
            "class_order": list(BEV_SEGMENTATION_CLASSES),
            "checkpoint_sha256": checkpoint_sha256,
            "dataset": dataset,
            "label_source": label_source,
            "temporal_input": temporal_input,
            "sample_count": self.sample_count,
            "observed_sample_count": self.observed_sample_count,
            "nonfinite_sample_count": self.nonfinite_sample_count,
            "evaluation_valid": self.nonfinite_sample_count == 0,
            "evaluation_batch_size_counts": {
                str(batch_size): count
                for batch_size, count in sorted(
                    self.batch_size_counts.items()
                )
            },
            "evaluation_batch_count": len(self.batch_size_sequence),
            "evaluation_batch_partition_sha256": (
                _ordered_string_sequence_sha256([
                    str(batch_size)
                    for batch_size in self.batch_size_sequence
                ])
            ),
            "sample_uid_order_sha256": _ordered_string_sequence_sha256(
                self.sample_uid_order
            ),
            "sample_uid_set_sha256": _ordered_string_sequence_sha256(
                sorted(self.seen_sample_uids)
            ),
            "bootstrap_reduction_reproducibility": (
                "fixed_point_sample_uid_weights_partition_invariant_v3"
            ),
            "computed_class_count": sum(self.class_availability),
            "supported_class_count": supported_class_count,
            "probability_bins": self.probability_bins,
            "ap_bootstrap_version": BEV_AP_BOOTSTRAP_VERSION,
            "ap_bootstrap_replicates": BEV_AP_BOOTSTRAP_REPLICATES,
            "ap_bootstrap_bins": BEV_AP_BOOTSTRAP_BINS,
            "ap_bootstrap_weight_scale": BEV_AP_BOOTSTRAP_WEIGHT_SCALE,
            "ap_bootstrap_max_weight": BEV_AP_BOOTSTRAP_MAX_WEIGHT,
            "threshold_selection": "same_evaluation_set_oracle",
            "reference_threshold_source": self.reference_threshold_source,
            "macro_average_precision_supported_classes": (
                sum(supported_average_precisions)
                / len(supported_average_precisions)
                if supported_average_precisions
                else 0.0
            ),
            "macro_ap_lift_supported_classes": (
                sum(supported_ap_lifts) / len(supported_ap_lifts)
                if supported_ap_lifts
                else 0.0
            ),
            "label_provenance": dict(label_provenance or {}),
            "camera_input_abi": dict(camera_input_abi or {}),
            "evaluation_split_contract": dict(
                evaluation_split_contract or {}
            ),
            "classes": classes,
        }
