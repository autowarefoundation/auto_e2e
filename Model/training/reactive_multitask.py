"""Stage-aware objective for nuPlan and L2D Reactive training."""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from model_components.losses import (
    BEVSegmentationAuxiliaryLoss,
    RouteReconstructionLoss,
    TrajectoryXYImitationLoss,
)
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    ROUTE_CHANNEL_COUNT,
)


SIMPLE_XY_IMITATION_OBJECTIVE_VERSION = "simple_xy_imitation_v1"
REACTIVE_MODEL_ARCHITECTURE_VERSION = (
    "bevformer_v2_t1_split_navigation_v1"
)


class ReactiveTrainingStage(str, enum.Enum):
    NUPLAN_FULL = "nuplan_full"
    L2D_CONTINUATION = "l2d_continuation"


def reactive_model_kwargs(
    stage: ReactiveTrainingStage,
    *,
    num_views: int,
) -> dict[str, Any]:
    """Return the locked Reactive-only model configuration."""
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    view_fusion_kwargs = (
        AUTOE2E_NAVIGATION_GEOMETRY.camera_bev_kwargs()
    )
    view_fusion_kwargs.update({
        "architecture": "bevformer_v2_t1",
        "bev_h": 256,
        "bev_w": 256,
        "image_size": 256,
        "num_heads": 8,
        "num_levels": 4,
        "num_points": 8,
        "num_encoder_layers": 6,
        "feedforward_channels": 512,
        "query_chunk_size": 4096,
        "activation_checkpointing": True,
    })
    return {
        "num_views": num_views,
        "view_fusion_kwargs": view_fusion_kwargs,
        "map_context_channels": MAP_CHANNEL_COUNT,
        "route_channels": ROUTE_CHANNEL_COUNT,
        "route_encoder_hidden_channels": 96,
        "map_type": "semantic_raster",
        "enable_route_conditioning": True,
        "map_fusion_mode": "deformable",
        "map_fusion_kwargs": {
            "num_sample_points": 8,
            "num_heads": 8,
            "query_chunk_size": 4096,
        },
        "temporal_memory_mode": "no_memory",
        "planner_mode": "gru",
        "enable_world_model": False,
        "enable_reasoning": False,
        # Stage B retains and loads the Stage A head but does not execute it.
        "enable_bev_segmentation": True,
        "bev_segmentation_classes": 8,
        "enable_route_reconstruction": True,
        "auxiliary_output_size": (
            AUTOE2E_NAVIGATION_GEOMETRY.height_px,
            AUTOE2E_NAVIGATION_GEOMETRY.width_px,
        ),
    }


def configure_model_for_stage(
    model: nn.Module,
    stage: ReactiveTrainingStage,
    *,
    bev_only: bool = False,
) -> None:
    """Apply trainability rules after loading the stage checkpoint."""
    if bev_only and stage is not ReactiveTrainingStage.NUPLAN_FULL:
        raise ValueError("BEV-only training is valid only for Stage A")
    try:
        reactive = getattr(model, "Reactive_E2E")
        bev_head = getattr(reactive, "BEVSegmentationHead")
    except AttributeError as exc:
        raise ValueError(
            "model does not expose the Reactive auxiliary heads"
        ) from exc
    if not isinstance(bev_head, nn.Module):
        raise ValueError("multi-stage training requires the BEV head")
    if bev_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in (
            reactive.Backbone,
            reactive.FeatureFusion,
            bev_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
            module.train()
        return
    train_bev = stage is ReactiveTrainingStage.NUPLAN_FULL
    for parameter in bev_head.parameters():
        parameter.requires_grad_(train_bev)
    bev_head.train(train_bev)


class ReactiveMultitaskObjective(nn.Module):
    """Compute the exact Stage A or Stage B objective."""

    def __init__(
        self,
        stage: ReactiveTrainingStage,
        *,
        bev_pos_weight: Sequence[float] | torch.Tensor,
        trajectory_weight: float = 1.0,
        bev_weight: float = 1.0,
        route_weight: float = 1.0,
        corridor_pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        task_weights = {
            "trajectory_weight": trajectory_weight,
            "bev_weight": bev_weight,
            "route_weight": route_weight,
        }
        for name, value in task_weights.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1_000.0:
                raise ValueError(
                    f"{name} must be finite and between zero and 1000"
                )
        if (
            not math.isfinite(corridor_pos_weight)
            or not 1.0 <= corridor_pos_weight <= 1_000.0
        ):
            raise ValueError(
                "corridor_pos_weight must be finite and between one and 1000"
            )
        self.stage = stage
        self.trajectory_weight = float(trajectory_weight)
        self.bev_weight = float(bev_weight)
        self.route_weight = float(route_weight)
        self.trajectory_loss = TrajectoryXYImitationLoss()
        self.bev_loss = (
            BEVSegmentationAuxiliaryLoss(bev_pos_weight)
            if stage is ReactiveTrainingStage.NUPLAN_FULL
            else None
        )
        self.route_loss = RouteReconstructionLoss(
            corridor_pos_weight=corridor_pos_weight,
        )

    @property
    def compute_bev_segmentation(self) -> bool:
        return self.bev_loss is not None and self.bev_weight > 0.0

    @property
    def compute_route_reconstruction(self) -> bool:
        return self.route_weight > 0.0

    def forward(
        self,
        predicted_controls: torch.Tensor,
        auxiliary: Mapping[str, Any],
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        zero = predicted_controls.sum() * 0.0
        trajectory = zero
        if self.trajectory_weight > 0.0:
            trajectory = self.trajectory_loss(
                predicted_controls,
                batch["trajectory_xy_m"],
                batch["trajectory_valid"],
                batch["initial_speed_mps"],
            )
        route = zero
        if self.compute_route_reconstruction:
            route_logits = auxiliary.get("route_reconstruction_logits")
            if not torch.is_tensor(route_logits):
                raise ValueError(
                    "route reconstruction logits are required when enabled"
                )
            route = self.route_loss(
                route_logits,
                batch["route_mask"].detach(),
                batch["route_channel_valid"],
            )
        importance = batch.get("bev_sampling_importance")
        if importance is not None:
            importance = torch.as_tensor(
                importance,
                device=predicted_controls.device,
                dtype=predicted_controls.dtype,
            ).reshape(-1)
            if (
                importance.shape != predicted_controls.shape[:1]
                or not torch.isfinite(importance).all()
                or bool((importance <= 0.0).any())
            ):
                raise ValueError(
                    "bev_sampling_importance must be finite, positive, "
                    "and have shape [B]"
                )
            if importance.numel() > 1 and not torch.allclose(
                importance,
                importance[:1].expand_as(importance),
            ):
                raise ValueError(
                    "non-uniform sampling importance requires batch size one"
                )
            non_bev_importance = importance.mean()
            trajectory = trajectory * non_bev_importance
            route = route * non_bev_importance
        bev = zero
        bev_bce = zero
        bev_dice = zero
        bev_loss = self.bev_loss
        if bev_loss is not None and self.bev_weight > 0.0:
            available = batch["bev_segmentation_available"].to(
                dtype=torch.bool
            )
            if not bool(available.all()):
                raise ValueError(
                    "nuPlan full training requires a BEV target per sample"
                )
            bev_logits = auxiliary.get("bev_segmentation_logits")
            if not torch.is_tensor(bev_logits):
                raise ValueError(
                    "nuPlan full training requires BEV segmentation logits"
                )
            bev_components = bev_loss.components(
                bev_logits,
                batch["bev_segmentation_target"],
                batch["bev_segmentation_valid"],
            )
            bev = bev_components["total"]
            bev_bce = bev_components["bce"]
            bev_dice = bev_components["dice"]
        total = (
            self.trajectory_weight * trajectory
            + self.bev_weight * bev
            + self.route_weight * route
        )
        return {
            "total": total,
            "trajectory": trajectory,
            "bev_segmentation": bev,
            "bev_segmentation_bce": bev_bce,
            "bev_segmentation_dice": bev_dice,
            "route_reconstruction": route,
        }
