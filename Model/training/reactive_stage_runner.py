"""Reusable runner for the nuPlan -> L2D Reactive training stages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_CLASSES,
)
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
from training.reactive_multitask import (
    SIMPLE_XY_IMITATION_OBJECTIVE_VERSION,
    ReactiveMultitaskObjective,
    ReactiveTrainingStage,
    configure_model_for_stage,
)


def _batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def _loader_item(
    item: Any,
) -> tuple[Mapping[str, Any], Any, str]:
    if isinstance(item, tuple):
        if len(item) != 3:
            raise ValueError(
                "multi-dataset loader items must be "
                "(batch, projection, geometry_type)"
            )
        batch, projection, geometry_type = item
        return batch, projection, str(geometry_type)
    return item, None, "pseudo"


def resolve_reactive_batch_projection(
    batch: Mapping[str, Any],
    fallback_projection: Any,
    fallback_geometry_type: str,
    *,
    device: torch.device,
) -> tuple[Any, str]:
    """Prefer a pose-compensated per-sample pinhole projection when packed."""
    matrix = batch.get("camera_projection_matrix")
    if matrix is None:
        projection = (
            fallback_projection.to(device)
            if fallback_projection is not None
            else None
        )
        return projection, fallback_geometry_type
    if not torch.is_tensor(matrix) or matrix.ndim != 4:
        raise ValueError(
            "camera_projection_matrix must have shape [B,V,3,4]"
        )
    geometry_value = batch.get(
        "camera_geometry_type",
        "rectified_pinhole",
    )
    if isinstance(geometry_value, str):
        geometry_types = [geometry_value]
    else:
        geometry_types = [str(value) for value in geometry_value]
    if (
        len(set(geometry_types)) != 1
        or geometry_types[0] not in ("pinhole", "rectified_pinhole")
    ):
        raise ValueError(
            "one Reactive batch must use one supported pinhole geometry"
        )
    from model_components.view_fusion.projection import PinholeProjection

    geometry_type = geometry_types[0]
    return (
        PinholeProjection(
            matrix.to(device),
            geometry_type=geometry_type,
        ),
        geometry_type,
    )


def _assert_reactive_only(model: torch.nn.Module) -> None:
    if getattr(model, "World_Action_Model_E2E", None) is not None:
        raise ValueError("Reactive multi-stage training requires WM OFF")
    reactive = getattr(model, "Reactive_E2E", None)
    if reactive is None or getattr(reactive, "ReasoningHead", None) is not None:
        raise ValueError(
            "Reactive multi-stage training requires Reasoning OFF"
        )


def reactive_config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(config),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def reactive_model_state_sha256(
    state_dict: Mapping[str, Any],
) -> str:
    """Hash tensor identity and bytes independently of torch serialization."""
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        if not torch.is_tensor(value):
            raise ValueError(
                f"model state value {name!r} is not a tensor"
            )
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "shape": list(tensor.shape),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def inspect_reactive_checkpoint_identity(
    checkpoint_path: str | Path,
) -> dict[str, str]:
    path = Path(checkpoint_path)
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    config = payload.get("config")
    state_dict = payload.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(
        state_dict, Mapping
    ):
        raise ValueError("Reactive checkpoint identity fields are missing")
    actual = {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "config_sha256": reactive_config_sha256(config),
        "model_state_sha256": reactive_model_state_sha256(state_dict),
    }
    for field in ("config_sha256", "model_state_sha256"):
        recorded = payload.get(field)
        if recorded != actual[field]:
            raise ValueError(
                f"Reactive checkpoint {field} does not match its payload"
            )
    return actual


def load_stage_a_parent(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load only Stage A model weights and validate Stage B lineage."""
    path = Path(checkpoint_path)
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Stage A checkpoint has no config mapping")
    required = {
        "training_objective_version": (
            SIMPLE_XY_IMITATION_OBJECTIVE_VERSION
        ),
        "training_stage": ReactiveTrainingStage.NUPLAN_FULL.value,
        "navigation_geometry_id": (
            AUTOE2E_NAVIGATION_GEOMETRY.geometry_id
        ),
        "enable_world_model": False,
        "enable_reasoning": False,
        "planner_mode": "gru",
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Stage A parent checkpoint contract differs: {mismatches}"
        )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Stage A checkpoint has no model state")
    config_sha256 = reactive_config_sha256(config)
    model_state_sha256 = reactive_model_state_sha256(state_dict)
    if payload.get("config_sha256") != config_sha256:
        raise ValueError("Stage A checkpoint config digest is invalid")
    if payload.get("model_state_sha256") != model_state_sha256:
        raise ValueError("Stage A checkpoint model-state digest is invalid")
    model.load_state_dict(state_dict)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "stage_a_parent_checkpoint_sha256": digest,
        "stage_a_config_digest": config_sha256,
        "stage_a_model_state_sha256": model_state_sha256,
    }


def run_reactive_epoch(
    model: torch.nn.Module,
    loader: Iterable[Any],
    objective: ReactiveMultitaskObjective,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Run one optimizer epoch with the locked stage objective."""
    if grad_clip <= 0.0:
        raise ValueError("grad_clip must be positive")
    _assert_reactive_only(model)
    configure_model_for_stage(model, objective.stage)
    model.train()
    totals: dict[str, list[float]] = {
        "total": [],
        "trajectory": [],
        "bev_segmentation": [],
        "route_reconstruction": [],
    }
    for item in loader:
        raw_batch, projection, geometry_type = _loader_item(item)
        batch = _batch_to_device(raw_batch, device)
        projection, geometry_type = resolve_reactive_batch_projection(
            batch,
            projection,
            geometry_type,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
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
                "multi-stage model did not return auxiliary outputs"
            )
        predicted_controls, auxiliary = output
        terms = objective(predicted_controls, auxiliary, batch)
        if not bool(torch.isfinite(terms["total"])):
            raise FloatingPointError(
                "Reactive multi-stage objective became non-finite"
            )
        terms["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            grad_clip,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(
                "Reactive multi-stage gradients became non-finite"
            )
        optimizer.step()
        for name in totals:
            totals[name].append(float(terms[name].detach().item()))
    if not totals["total"]:
        raise ValueError("Reactive training loader yielded no batches")
    return {
        name: float(np.mean(values))
        for name, values in totals.items()
    }


def evaluate_reactive_xy(
    model: torch.nn.Module,
    loader: Iterable[Any],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Return lightweight 6.4-second checkpoint-selection metrics."""
    from training.losses.control_rollout import integrate_controls_torch

    _assert_reactive_only(model)
    was_training = model.training
    ade_sum = 0.0
    fde_sum = 0.0
    complete_sample_count = 0
    model.eval()
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
                controls = model(
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
                ade_sum += float(errors[complete].mean(dim=1).sum().item())
                fde_sum += float(errors[complete, -1].sum().item())
                complete_sample_count += int(complete.sum().item())
    finally:
        model.train(was_training)
    if complete_sample_count <= 0:
        raise ValueError(
            "validation has no complete finite 6.4 second XY targets"
        )
    return {
        "ade_6p4s_m": ade_sum / complete_sample_count,
        "fde_6p4s_m": fde_sum / complete_sample_count,
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _mean_or_none(total: float, count: float) -> float | None:
    return _safe_ratio(total, count)


def _binary_metrics(
    true_positive: float,
    false_positive: float,
    false_negative: float,
) -> dict[str, float | None]:
    return {
        "iou": _safe_ratio(
            true_positive,
            true_positive + false_positive + false_negative,
        ),
        "dice": _safe_ratio(
            2.0 * true_positive,
            2.0 * true_positive + false_positive + false_negative,
        ),
        "precision": _safe_ratio(
            true_positive,
            true_positive + false_positive,
        ),
        "recall": _safe_ratio(
            true_positive,
            true_positive + false_negative,
        ),
    }


def _macro_metric(
    per_class: Mapping[str, Mapping[str, float | None]],
    name: str,
) -> float | None:
    values = []
    for metrics in per_class.values():
        value = metrics[name]
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else None


def _average_precision_from_histogram(
    positive_histogram: np.ndarray,
    negative_histogram: np.ndarray,
) -> float | None:
    positive_total = float(positive_histogram.sum())
    if positive_total <= 0.0:
        return None
    cumulative_positive = np.cumsum(positive_histogram[::-1])
    cumulative_negative = np.cumsum(negative_histogram[::-1])
    precision = cumulative_positive / np.maximum(
        cumulative_positive + cumulative_negative,
        1.0,
    )
    recall = cumulative_positive / positive_total
    recall_delta = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(recall_delta * precision))


def _calibration_error(
    confidence_sum: np.ndarray,
    positive_sum: np.ndarray,
    count: np.ndarray,
) -> float | None:
    total = float(count.sum())
    if total <= 0.0:
        return None
    active = count > 0.0
    confidence = confidence_sum[active] / count[active]
    accuracy = positive_sum[active] / count[active]
    return float(
        np.sum(np.abs(confidence - accuracy) * count[active]) / total
    )


def _destination_xy(
    heatmap: torch.Tensor,
) -> torch.Tensor:
    if heatmap.ndim != 3:
        raise ValueError("destination heatmap must have shape [B,H,W]")
    batch_size, height, width = heatmap.shape
    indices = heatmap.reshape(batch_size, -1).argmax(dim=1)
    rows = torch.div(indices, width, rounding_mode="floor")
    columns = indices.remainder(width)
    geometry = AUTOE2E_NAVIGATION_GEOMETRY
    x = geometry.x_max_m - (
        rows.to(torch.float32) + 0.5
    ) * ((geometry.x_max_m - geometry.x_min_m) / height)
    y = geometry.y_max_m - (
        columns.to(torch.float32) + 0.5
    ) * ((geometry.y_max_m - geometry.y_min_m) / width)
    return torch.stack((x, y), dim=-1)


def _trajectory_delta(
    first_xy: torch.Tensor,
    second_xy: torch.Tensor,
) -> torch.Tensor:
    return torch.linalg.vector_norm(first_xy - second_xy, dim=-1).mean(
        dim=1
    )


def _route_gradient_evidence(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    projection: Any,
    geometry_type: str,
) -> float | None:
    route_valid = batch["route_valid"].to(dtype=torch.bool)
    if not bool(route_valid.any()):
        return None
    route = batch["route_mask"].detach().clone().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        controls = model(
            batch["visual_tiles"],
            batch["map_context"],
            batch["visual_history"],
            batch["egomotion_history"],
            route_mask=route,
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
        gradient = torch.autograd.grad(
            controls.to(torch.float32).square().mean(),
            route,
            allow_unused=True,
        )[0]
    if gradient is None:
        return 0.0
    valid_gradient = gradient[route_valid]
    return float(valid_gradient.abs().mean().detach().cpu().item())


def evaluate_reactive_multitask(
    model: torch.nn.Module,
    loader: Iterable[Any],
    *,
    device: torch.device,
    include_counterfactuals: bool = True,
    include_route_gradient: bool = True,
    probability_bins: int = 100,
) -> dict[str, Any]:
    """Evaluate trajectory, BEV semantics, and route retention/use."""
    from training.losses.control_rollout import integrate_controls_torch

    if probability_bins < 10:
        raise ValueError("probability_bins must be at least 10")
    _assert_reactive_only(model)
    was_training = model.training
    horizon_steps = {
        "1s": 10,
        "2s": 20,
        "3s": 30,
        "5s": 50,
        "6p4s": 64,
    }
    trajectory_ade_sum = {name: 0.0 for name in horizon_steps}
    trajectory_ade_count = {name: 0 for name in horizon_steps}
    trajectory_fde_sum = {name: 0.0 for name in horizon_steps}
    trajectory_fde_count = {name: 0 for name in horizon_steps}
    longitudinal_sum = 0.0
    lateral_sum = 0.0
    trajectory_valid_count = 0
    trajectory_cell_count = 0
    nonfinite_samples = 0
    sample_count = 0
    sample_uids: list[str] = []

    class_count = len(BEV_SEGMENTATION_CLASSES)
    bev_true_positive = np.zeros(class_count, dtype=np.float64)
    bev_false_positive = np.zeros(class_count, dtype=np.float64)
    bev_false_negative = np.zeros(class_count, dtype=np.float64)
    bev_brier_sum = np.zeros(class_count, dtype=np.float64)
    bev_valid_count = np.zeros(class_count, dtype=np.float64)
    bev_positive_count = np.zeros(class_count, dtype=np.float64)
    bev_total_cells = np.zeros(class_count, dtype=np.float64)
    bev_positive_histogram = np.zeros(
        (class_count, probability_bins),
        dtype=np.float64,
    )
    bev_negative_histogram = np.zeros_like(bev_positive_histogram)
    bev_confidence_sum = np.zeros_like(bev_positive_histogram)
    bev_calibration_positive = np.zeros_like(bev_positive_histogram)
    bev_calibration_count = np.zeros_like(bev_positive_histogram)
    distance_band_names = ("0_to_30m", "30_to_60m", "60m_plus")
    distance_true_positive = np.zeros(
        len(distance_band_names),
        dtype=np.float64,
    )
    distance_false_positive = np.zeros_like(distance_true_positive)
    distance_false_negative = np.zeros_like(distance_true_positive)
    distance_positive_count = np.zeros_like(distance_true_positive)
    distance_valid_count = np.zeros_like(distance_true_positive)
    distance_total_cells = np.zeros_like(distance_true_positive)
    distance_masks: tuple[torch.Tensor, ...] | None = None

    route_true_positive = 0.0
    route_false_positive = 0.0
    route_false_negative = 0.0
    route_corridor_valid = 0
    route_destination_valid = 0
    route_destination_error_sum = 0.0
    route_channel_valid_count = np.zeros(2, dtype=np.int64)
    route_zero_delta_sum = 0.0
    route_zero_delta_count = 0
    route_swap_delta_sum = 0.0
    route_swap_delta_count = 0
    route_swap_directional_correct = 0
    route_swap_directional_count = 0
    route_gradient_l1: float | None = None

    model.eval()
    try:
        for item in loader:
            raw_batch, fallback_projection, fallback_geometry_type = (
                _loader_item(item)
            )
            batch = _batch_to_device(raw_batch, device)
            projection, geometry_type = resolve_reactive_batch_projection(
                batch,
                fallback_projection,
                fallback_geometry_type,
                device=device,
            )
            batch_size = int(batch["visual_tiles"].shape[0])
            sample_count += batch_size
            raw_uids = batch.get("sample_uid")
            if isinstance(raw_uids, str):
                batch_uids = [raw_uids]
            elif raw_uids is None:
                batch_uids = [
                    f"missing-sample-uid-{sample_count - batch_size + index}"
                    for index in range(batch_size)
                ]
            else:
                batch_uids = [str(value) for value in raw_uids]
            if len(batch_uids) != batch_size:
                raise ValueError("sample UID count differs from batch size")
            sample_uids.extend(batch_uids)

            if include_route_gradient and route_gradient_l1 is None:
                with torch.enable_grad():
                    route_gradient_l1 = _route_gradient_evidence(
                        model,
                        batch,
                        projection,
                        geometry_type,
                    )

            bev_available = batch.get("bev_segmentation_available")
            compute_bev = (
                bev_available is not None
                and bool(
                    torch.as_tensor(
                        bev_available,
                        device=device,
                    ).any()
                )
            )
            with torch.no_grad():
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
                    mode="infer",
                    return_auxiliary=True,
                    compute_bev_segmentation=compute_bev,
                    compute_route_reconstruction=True,
                )
                if not isinstance(output, tuple):
                    raise RuntimeError(
                        "Reactive evaluator requires auxiliary outputs"
                    )
                controls, auxiliary = output
                controls_3d = controls.reshape(batch_size, -1, 2)
                finite_samples = torch.isfinite(controls_3d).all(
                    dim=(1, 2)
                )
                nonfinite_samples += int((~finite_samples).sum().item())
                safe_controls = torch.where(
                    finite_samples[:, None, None],
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
                    & finite_samples[:, None]
                    & torch.isfinite(target_xy).all(dim=-1)
                )
                if target_xy.shape != predicted_xy.shape:
                    raise ValueError(
                        "trajectory target shape differs from rollout"
                    )
                displacement = predicted_xy - target_xy
                errors = torch.linalg.vector_norm(displacement, dim=-1)
                trajectory_cell_count += int(valid.numel())
                trajectory_valid_count += int(valid.sum().item())
                longitudinal_sum += float(
                    displacement[..., 0].abs()[valid].sum().item()
                )
                lateral_sum += float(
                    displacement[..., 1].abs()[valid].sum().item()
                )

                for name, step_count in horizon_steps.items():
                    usable = min(step_count, errors.shape[1])
                    horizon_valid = valid[:, :usable]
                    per_sample_count = horizon_valid.sum(dim=1)
                    eligible_ade = per_sample_count > 0
                    if bool(eligible_ade.any()):
                        per_sample_error = (
                            (errors[:, :usable] * horizon_valid).sum(dim=1)
                            / per_sample_count.clamp_min(1)
                        )
                        trajectory_ade_sum[name] += float(
                            per_sample_error[eligible_ade].sum().item()
                        )
                        trajectory_ade_count[name] += int(
                            eligible_ade.sum().item()
                        )
                    if step_count <= errors.shape[1]:
                        eligible_fde = valid[:, step_count - 1]
                        if bool(eligible_fde.any()):
                            trajectory_fde_sum[name] += float(
                                errors[eligible_fde, step_count - 1]
                                .sum()
                                .item()
                            )
                            trajectory_fde_count[name] += int(
                                eligible_fde.sum().item()
                            )

                bev_logits = auxiliary.get("bev_segmentation_logits")
                if compute_bev:
                    if not torch.is_tensor(bev_logits):
                        raise RuntimeError(
                            "BEV teacher is present but logits are missing"
                        )
                    bev_target = batch["bev_segmentation_target"].to(
                        dtype=bev_logits.dtype
                    )
                    bev_valid = batch["bev_segmentation_valid"].to(
                        dtype=torch.bool
                    )
                    if (
                        bev_target.shape != bev_logits.shape
                        or bev_valid.shape != bev_logits.shape
                    ):
                        raise ValueError(
                            "BEV prediction and target shapes differ"
                        )
                    probability = bev_logits.sigmoid()
                    binary_target = bev_target >= 0.5
                    binary_prediction = probability >= 0.5
                    height, width = probability.shape[-2:]
                    if distance_masks is None:
                        geometry = AUTOE2E_NAVIGATION_GEOMETRY
                        rows = torch.arange(
                            height,
                            device=device,
                            dtype=torch.float32,
                        )
                        columns = torch.arange(
                            width,
                            device=device,
                            dtype=torch.float32,
                        )
                        x = geometry.x_max_m - (
                            rows + 0.5
                        ) * (
                            (geometry.x_max_m - geometry.x_min_m)
                            / height
                        )
                        y = geometry.y_max_m - (
                            columns + 0.5
                        ) * (
                            (geometry.y_max_m - geometry.y_min_m)
                            / width
                        )
                        distance = torch.sqrt(
                            x[:, None].square() + y[None, :].square()
                        )
                        distance_masks = (
                            distance < 30.0,
                            (distance >= 30.0) & (distance < 60.0),
                            distance >= 60.0,
                        )
                    elif distance_masks[0].shape != (height, width):
                        raise ValueError(
                            "BEV metric batches use inconsistent geometry"
                        )
                    for class_index in range(class_count):
                        class_valid = bev_valid[:, class_index]
                        bev_total_cells[class_index] += float(
                            class_valid.numel()
                        )
                        for band_index, distance_mask in enumerate(
                            distance_masks
                        ):
                            distance_total_cells[band_index] += float(
                                batch_size * distance_mask.sum().item()
                            )
                        if not bool(class_valid.any()):
                            continue
                        class_probability = probability[
                            :, class_index
                        ][class_valid]
                        class_target = binary_target[:, class_index][
                            class_valid
                        ]
                        class_prediction = binary_prediction[
                            :, class_index
                        ][class_valid]
                        class_count_valid = float(class_valid.sum().item())
                        bev_valid_count[class_index] += class_count_valid
                        bev_positive_count[class_index] += float(
                            class_target.sum().item()
                        )
                        bev_true_positive[class_index] += float(
                            (class_prediction & class_target).sum().item()
                        )
                        bev_false_positive[class_index] += float(
                            (class_prediction & ~class_target).sum().item()
                        )
                        bev_false_negative[class_index] += float(
                            (~class_prediction & class_target).sum().item()
                        )
                        bev_brier_sum[class_index] += float(
                            (
                                class_probability
                                - class_target.to(class_probability.dtype)
                            )
                            .square()
                            .sum()
                            .item()
                        )
                        bins = torch.clamp(
                            (
                                class_probability * probability_bins
                            ).to(torch.int64),
                            max=probability_bins - 1,
                        )
                        probability_cpu = (
                            class_probability.detach().cpu().numpy()
                        )
                        target_cpu = class_target.detach().cpu().numpy()
                        bins_cpu = bins.detach().cpu().numpy()
                        positive_bins = np.bincount(
                            bins_cpu[target_cpu],
                            minlength=probability_bins,
                        )
                        negative_bins = np.bincount(
                            bins_cpu[~target_cpu],
                            minlength=probability_bins,
                        )
                        bev_positive_histogram[
                            class_index
                        ] += positive_bins
                        bev_negative_histogram[
                            class_index
                        ] += negative_bins
                        bev_confidence_sum[class_index] += np.bincount(
                            bins_cpu,
                            weights=probability_cpu,
                            minlength=probability_bins,
                        )
                        bev_calibration_positive[
                            class_index
                        ] += positive_bins
                        bev_calibration_count[class_index] += np.bincount(
                            bins_cpu,
                            minlength=probability_bins,
                        )
                        for band_index, distance_mask in enumerate(
                            distance_masks
                        ):
                            band_valid = class_valid & distance_mask
                            if not bool(band_valid.any()):
                                continue
                            band_target = binary_target[
                                :, class_index
                            ][band_valid]
                            band_prediction = binary_prediction[
                                :, class_index
                            ][band_valid]
                            distance_valid_count[band_index] += float(
                                band_valid.sum().item()
                            )
                            distance_positive_count[band_index] += float(
                                band_target.sum().item()
                            )
                            distance_true_positive[band_index] += float(
                                (
                                    band_prediction & band_target
                                ).sum().item()
                            )
                            distance_false_positive[band_index] += float(
                                (
                                    band_prediction & ~band_target
                                ).sum().item()
                            )
                            distance_false_negative[band_index] += float(
                                (
                                    ~band_prediction & band_target
                                ).sum().item()
                            )

                route_logits = auxiliary.get(
                    "route_reconstruction_logits"
                )
                if not torch.is_tensor(route_logits):
                    raise RuntimeError(
                        "route reconstruction logits are missing"
                    )
                route_target = batch["route_mask"].to(
                    dtype=route_logits.dtype
                )
                route_channel_valid = batch[
                    "route_channel_valid"
                ].to(dtype=torch.bool)
                if route_logits.shape != route_target.shape:
                    raise ValueError(
                        "route prediction and target shapes differ"
                    )
                route_channel_valid_count += (
                    route_channel_valid.sum(dim=0).cpu().numpy()
                )
                corridor_valid = route_channel_valid[:, 0]
                if bool(corridor_valid.any()):
                    corridor_probability = route_logits[
                        corridor_valid, 0
                    ].sigmoid()
                    corridor_target = route_target[
                        corridor_valid, 0
                    ] >= 0.5
                    corridor_prediction = corridor_probability >= 0.5
                    route_true_positive += float(
                        (corridor_prediction & corridor_target).sum().item()
                    )
                    route_false_positive += float(
                        (corridor_prediction & ~corridor_target).sum().item()
                    )
                    route_false_negative += float(
                        (~corridor_prediction & corridor_target).sum().item()
                    )
                    route_corridor_valid += int(
                        corridor_valid.sum().item()
                    )
                destination_valid = route_channel_valid[:, 1]
                if bool(destination_valid.any()):
                    predicted_destination = _destination_xy(
                        route_logits[destination_valid, 1]
                    )
                    target_destination = _destination_xy(
                        route_target[destination_valid, 1]
                    )
                    route_destination_error_sum += float(
                        torch.linalg.vector_norm(
                            predicted_destination - target_destination,
                            dim=-1,
                        )
                        .sum()
                        .item()
                    )
                    route_destination_valid += int(
                        destination_valid.sum().item()
                    )

                if include_counterfactuals:
                    zero_controls = model(
                        batch["visual_tiles"],
                        batch["map_context"],
                        batch["visual_history"],
                        batch["egomotion_history"],
                        route_mask=torch.zeros_like(batch["route_mask"]),
                        map_valid=batch["map_valid"],
                        route_valid=torch.zeros_like(
                            batch["route_valid"],
                            dtype=torch.bool,
                        ),
                        projection=projection,
                        geometry_type=geometry_type,
                        mode="infer",
                        compute_bev_segmentation=False,
                        compute_route_reconstruction=False,
                    )
                    if isinstance(zero_controls, tuple):
                        zero_controls = zero_controls[0]
                    zero_xy, _, _ = integrate_controls_torch(
                        zero_controls,
                        batch["initial_speed_mps"],
                    )
                    zero_eligible = (
                        batch["route_valid"].to(dtype=torch.bool)
                        & finite_samples
                        & torch.isfinite(zero_xy).all(dim=(1, 2))
                    )
                    if bool(zero_eligible.any()):
                        route_zero_delta_sum += float(
                            _trajectory_delta(
                                predicted_xy,
                                zero_xy,
                            )[zero_eligible]
                            .sum()
                            .item()
                        )
                        route_zero_delta_count += int(
                            zero_eligible.sum().item()
                        )

                    if batch_size > 1:
                        donor_indices = torch.roll(
                            torch.arange(batch_size, device=device),
                            shifts=1,
                        )
                        swapped_route = batch["route_mask"][
                            donor_indices
                        ]
                        swapped_valid = batch["route_valid"][
                            donor_indices
                        ]
                        swap_controls = model(
                            batch["visual_tiles"],
                            batch["map_context"],
                            batch["visual_history"],
                            batch["egomotion_history"],
                            route_mask=swapped_route,
                            map_valid=batch["map_valid"],
                            route_valid=swapped_valid,
                            projection=projection,
                            geometry_type=geometry_type,
                            mode="infer",
                            compute_bev_segmentation=False,
                            compute_route_reconstruction=False,
                        )
                        if isinstance(swap_controls, tuple):
                            swap_controls = swap_controls[0]
                        swap_xy, _, _ = integrate_controls_torch(
                            swap_controls,
                            batch["initial_speed_mps"],
                        )
                        swap_eligible = (
                            batch["route_valid"].to(dtype=torch.bool)
                            & swapped_valid.to(dtype=torch.bool)
                            & finite_samples
                            & torch.isfinite(swap_xy).all(dim=(1, 2))
                        )
                        if bool(swap_eligible.any()):
                            route_swap_delta_sum += float(
                                _trajectory_delta(
                                    predicted_xy,
                                    swap_xy,
                                )[swap_eligible]
                                .sum()
                                .item()
                            )
                            route_swap_delta_count += int(
                                swap_eligible.sum().item()
                            )
                        donor_destination_valid = route_channel_valid[
                            donor_indices, 1
                        ]
                        directional_eligible = (
                            swap_eligible & donor_destination_valid
                        )
                        if bool(directional_eligible.any()):
                            donor_destination = _destination_xy(
                                swapped_route[:, 1]
                            )
                            baseline_distance = torch.linalg.vector_norm(
                                predicted_xy[:, -1] - donor_destination,
                                dim=-1,
                            )
                            swap_distance = torch.linalg.vector_norm(
                                swap_xy[:, -1] - donor_destination,
                                dim=-1,
                            )
                            route_swap_directional_correct += int(
                                (
                                    swap_distance[directional_eligible]
                                    < baseline_distance[
                                        directional_eligible
                                    ]
                                )
                                .sum()
                                .item()
                            )
                            route_swap_directional_count += int(
                                directional_eligible.sum().item()
                            )
    finally:
        model.train(was_training)

    if sample_count <= 0:
        raise ValueError("Reactive validation loader yielded no batches")
    if len(set(sample_uids)) != len(sample_uids):
        raise ValueError("Reactive validation sample UIDs are not unique")

    trajectory_metrics: dict[str, float | int | None] = {
        "valid_timestep_count": trajectory_valid_count,
        "total_timestep_count": trajectory_cell_count,
        "valid_horizon_coverage": _safe_ratio(
            trajectory_valid_count,
            trajectory_cell_count,
        ),
        "mean_abs_longitudinal_error_m": _mean_or_none(
            longitudinal_sum,
            trajectory_valid_count,
        ),
        "mean_abs_lateral_error_m": _mean_or_none(
            lateral_sum,
            trajectory_valid_count,
        ),
        "nonfinite_prediction_count": nonfinite_samples,
        "nonfinite_prediction_rate": _safe_ratio(
            nonfinite_samples,
            sample_count,
        ),
    }
    for name in horizon_steps:
        trajectory_metrics[f"ade_{name}_m"] = _mean_or_none(
            trajectory_ade_sum[name],
            trajectory_ade_count[name],
        )
        trajectory_metrics[f"fde_{name}_m"] = _mean_or_none(
            trajectory_fde_sum[name],
            trajectory_fde_count[name],
        )
        trajectory_metrics[f"ade_{name}_sample_count"] = (
            trajectory_ade_count[name]
        )
        trajectory_metrics[f"fde_{name}_sample_count"] = (
            trajectory_fde_count[name]
        )

    bev_per_class: dict[str, dict[str, float | None]] = {}
    for class_index, class_name in enumerate(BEV_SEGMENTATION_CLASSES):
        metrics = _binary_metrics(
            bev_true_positive[class_index],
            bev_false_positive[class_index],
            bev_false_negative[class_index],
        )
        metrics.update({
            "average_precision_histogram": (
                _average_precision_from_histogram(
                    bev_positive_histogram[class_index],
                    bev_negative_histogram[class_index],
                )
            ),
            "brier_score": _mean_or_none(
                bev_brier_sum[class_index],
                bev_valid_count[class_index],
            ),
            "expected_calibration_error": _calibration_error(
                bev_confidence_sum[class_index],
                bev_calibration_positive[class_index],
                bev_calibration_count[class_index],
            ),
            "positive_prevalence": _safe_ratio(
                bev_positive_count[class_index],
                bev_valid_count[class_index],
            ),
            "valid_cell_coverage": _safe_ratio(
                bev_valid_count[class_index],
                bev_total_cells[class_index],
            ),
        })
        bev_per_class[class_name] = metrics
    bev_macro = {
        name: _macro_metric(bev_per_class, name)
        for name in (
            "iou",
            "dice",
            "precision",
            "recall",
            "average_precision_histogram",
            "brier_score",
            "expected_calibration_error",
            "positive_prevalence",
            "valid_cell_coverage",
        )
    }
    bev_class_groups = {
        "static": tuple(range(5)),
        "dynamic": tuple(range(5, class_count)),
    }
    bev_group_metrics: dict[str, dict[str, float | None]] = {}
    for group_name, class_indices in bev_class_groups.items():
        metrics = _binary_metrics(
            float(bev_true_positive[list(class_indices)].sum()),
            float(bev_false_positive[list(class_indices)].sum()),
            float(bev_false_negative[list(class_indices)].sum()),
        )
        metrics.update({
            "average_precision_histogram": (
                _average_precision_from_histogram(
                    bev_positive_histogram[list(class_indices)].sum(axis=0),
                    bev_negative_histogram[list(class_indices)].sum(axis=0),
                )
            ),
            "brier_score": _mean_or_none(
                float(bev_brier_sum[list(class_indices)].sum()),
                float(bev_valid_count[list(class_indices)].sum()),
            ),
            "expected_calibration_error": _calibration_error(
                bev_confidence_sum[list(class_indices)].sum(axis=0),
                bev_calibration_positive[list(class_indices)].sum(axis=0),
                bev_calibration_count[list(class_indices)].sum(axis=0),
            ),
            "positive_prevalence": _safe_ratio(
                float(bev_positive_count[list(class_indices)].sum()),
                float(bev_valid_count[list(class_indices)].sum()),
            ),
            "valid_cell_coverage": _safe_ratio(
                float(bev_valid_count[list(class_indices)].sum()),
                float(bev_total_cells[list(class_indices)].sum()),
            ),
        })
        bev_group_metrics[group_name] = metrics
    bev_distance_metrics = {}
    for band_index, band_name in enumerate(distance_band_names):
        metrics = _binary_metrics(
            distance_true_positive[band_index],
            distance_false_positive[band_index],
            distance_false_negative[band_index],
        )
        metrics.update({
            "positive_prevalence": _safe_ratio(
                distance_positive_count[band_index],
                distance_valid_count[band_index],
            ),
            "valid_cell_coverage": _safe_ratio(
                distance_valid_count[band_index],
                distance_total_cells[band_index],
            ),
        })
        bev_distance_metrics[band_name] = metrics

    route_metrics: dict[str, Any] = _binary_metrics(
        route_true_positive,
        route_false_positive,
        route_false_negative,
    )
    alpha = getattr(
        getattr(model, "Reactive_E2E").MapBEVFusion,
        "alpha",
        None,
    )
    route_metrics.update({
        "corridor_valid_sample_count": route_corridor_valid,
        "destination_valid_sample_count": route_destination_valid,
        "destination_localization_error_m": _mean_or_none(
            route_destination_error_sum,
            route_destination_valid,
        ),
        "channel_valid_sample_count": [
            int(value) for value in route_channel_valid_count
        ],
        "fusion_gate_mean_abs": (
            float(alpha.detach().abs().mean().cpu().item())
            if torch.is_tensor(alpha)
            else None
        ),
        "route_input_gradient_mean_abs": route_gradient_l1,
        "route_zero_trajectory_delta_m": _mean_or_none(
            route_zero_delta_sum,
            route_zero_delta_count,
        ),
        "route_zero_sample_count": route_zero_delta_count,
        "route_swap_trajectory_delta_m": _mean_or_none(
            route_swap_delta_sum,
            route_swap_delta_count,
        ),
        "route_swap_sample_count": route_swap_delta_count,
        "route_swap_directional_correctness": _safe_ratio(
            route_swap_directional_correct,
            route_swap_directional_count,
        ),
        "route_swap_directional_sample_count": (
            route_swap_directional_count
        ),
    })
    return {
        "schema_version": "reactive_multitask_evaluation_v1",
        "sample_count": sample_count,
        "sample_uid_sha256": hashlib.sha256(
            "\n".join(sorted(sample_uids)).encode("utf-8")
        ).hexdigest(),
        "trajectory": trajectory_metrics,
        "bev_segmentation": {
            "available": bool(bev_valid_count.sum() > 0.0),
            "probability_bins": probability_bins,
            "per_class": bev_per_class,
            "macro": bev_macro,
            "class_groups": bev_group_metrics,
            "distance_bands": bev_distance_metrics,
            "unavailable_stratifications": [
                "city",
                "day_night",
            ],
        },
        "route": route_metrics,
    }


def evaluate_reactive_transfer_matrix_models(
    stage_a_model: torch.nn.Module,
    stage_b_model: torch.nn.Module,
    loader_factories: Mapping[str, Callable[[], Iterable[Any]]],
    *,
    device: torch.device,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate two checkpoints on identical per-dataset sample sets."""
    if set(loader_factories) != {"nuplan", "l2d"}:
        raise ValueError(
            "retention loader factories must contain nuplan and l2d"
        )
    matrix: dict[str, dict[str, dict[str, Any]]] = {
        "stage_a": {},
        "stage_b": {},
    }
    for checkpoint_name, model in (
        ("stage_a", stage_a_model),
        ("stage_b", stage_b_model),
    ):
        for dataset_name in ("nuplan", "l2d"):
            matrix[checkpoint_name][dataset_name] = (
                evaluate_reactive_multitask(
                    model,
                    loader_factories[dataset_name](),
                    device=device,
                )
            )
    for dataset_name in ("nuplan", "l2d"):
        stage_a_identity = matrix["stage_a"][dataset_name]
        stage_b_identity = matrix["stage_b"][dataset_name]
        if (
            stage_a_identity["sample_count"]
            != stage_b_identity["sample_count"]
            or stage_a_identity["sample_uid_sha256"]
            != stage_b_identity["sample_uid_sha256"]
        ):
            raise ValueError(
                "Stage A and Stage B retention cells used different "
                f"{dataset_name} validation samples"
            )
    return matrix


def save_reactive_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    stage: ReactiveTrainingStage,
    dataset_manifest_sha256: str,
    epoch: int,
    model_config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    metrics: Mapping[str, float] | None = None,
    training_state: Mapping[str, Any] | None = None,
    lineage: Mapping[str, str] | None = None,
) -> str:
    """Write a stage checkpoint with immutable lineage fields."""
    if (
        len(dataset_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in dataset_manifest_sha256
        )
    ):
        raise ValueError("dataset manifest digest must be SHA-256")
    if epoch <= 0:
        raise ValueError("checkpoint epoch must be positive")
    config: dict[str, Any] = {
        **dict(model_config),
        "training_objective_version": (
            SIMPLE_XY_IMITATION_OBJECTIVE_VERSION
        ),
        "training_stage": stage.value,
        "navigation_geometry_id": (
            AUTOE2E_NAVIGATION_GEOMETRY.geometry_id
        ),
        "enable_world_model": False,
        "enable_reasoning": False,
        "planner_mode": "gru",
        "dataset_manifest_sha256": dataset_manifest_sha256,
    }
    if stage is ReactiveTrainingStage.L2D_CONTINUATION:
        config["stage_b_dataset_manifest_sha256"] = (
            dataset_manifest_sha256
        )
    config.update(dict(lineage or {}))
    state_dict = model.state_dict()
    payload: dict[str, Any] = {
        "model_state_dict": state_dict,
        "config": config,
        "config_sha256": reactive_config_sha256(config),
        "model_state_sha256": reactive_model_state_sha256(state_dict),
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
        "training_state": dict(training_state or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return hashlib.sha256(output_path.read_bytes()).hexdigest()
