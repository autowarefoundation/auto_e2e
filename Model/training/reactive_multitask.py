"""Stage-aware objective for nuPlan and L2D Reactive training."""

from __future__ import annotations

import enum
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
    return {
        "num_views": num_views,
        "view_fusion_kwargs": (
            AUTOE2E_NAVIGATION_GEOMETRY.camera_bev_kwargs()
        ),
        "map_context_channels": MAP_CHANNEL_COUNT,
        "route_channels": ROUTE_CHANNEL_COUNT,
        "map_type": "semantic_raster",
        "enable_route_conditioning": True,
        "map_fusion_mode": "residual",
        "temporal_memory_mode": "no_memory",
        "planner_mode": "gru",
        "enable_world_model": False,
        "enable_reasoning": False,
        # Stage B retains and loads the Stage A head but does not execute it.
        "enable_bev_segmentation": True,
        "bev_segmentation_classes": 8,
        "enable_route_reconstruction": True,
    }


def configure_model_for_stage(
    model: nn.Module,
    stage: ReactiveTrainingStage,
) -> None:
    """Apply trainability rules after loading the stage checkpoint."""
    try:
        reactive = getattr(model, "Reactive_E2E")
        bev_head = getattr(reactive, "BEVSegmentationHead")
    except AttributeError as exc:
        raise ValueError(
            "model does not expose the Reactive auxiliary heads"
        ) from exc
    if not isinstance(bev_head, nn.Module):
        raise ValueError("multi-stage training requires the BEV head")
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
        bev_weight: float = 1.0,
        route_weight: float = 1.0,
        corridor_pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if bev_weight < 0.0 or route_weight < 0.0:
            raise ValueError("auxiliary loss weights must be non-negative")
        self.stage = stage
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
        return self.bev_loss is not None

    def forward(
        self,
        predicted_controls: torch.Tensor,
        auxiliary: Mapping[str, Any],
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        trajectory = self.trajectory_loss(
            predicted_controls,
            batch["trajectory_xy_m"],
            batch["trajectory_valid"],
            batch["initial_speed_mps"],
        )
        route_logits = auxiliary.get("route_reconstruction_logits")
        if not torch.is_tensor(route_logits):
            raise ValueError(
                "route reconstruction logits are required for both stages"
            )
        route = self.route_loss(
            route_logits,
            batch["route_mask"].detach(),
            batch["route_channel_valid"],
        )
        zero = predicted_controls.sum() * 0.0
        bev = zero
        if self.bev_loss is not None:
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
            bev = self.bev_loss(
                bev_logits,
                batch["bev_segmentation_target"],
                batch["bev_segmentation_valid"],
            )
        total = (
            trajectory
            + self.bev_weight * bev
            + self.route_weight * route
        )
        return {
            "total": total,
            "trajectory": trajectory,
            "bev_segmentation": bev,
            "route_reconstruction": route,
        }
