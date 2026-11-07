"""Training objectives of the two AutoE2E stages.

Stage 1 trains AutoBEV alone with every AutoBEV term (``STAGE1_WEIGHTS``).

Stage 2 trains the whole model:

- trajectory imitation in (x, y) on samples with ``planner_supervised``;
- route reconstruction of the resampled route raster;
- the crop-pose error regressed by ``PoseErrorHead`` (yaw scaled to metres);
- feature consistency between each distorted sample's planner BEV and its
  clean teacher (PR #235);
- the AutoBEV segmentation, depth and centre terms as an auxiliary loss whose
  weight the schedule lowers from 1.0 to 0.3. BEV-only samples from the other
  datasets carry no depth and no boxes in this stage, so they add only the
  segmentation term.

Samples without a planner target carry ``trajectory_valid`` and
``route_channel_valid`` of False, so one expression covers both kinds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_components.autobev.losses import STAGE1_WEIGHTS, STAGE2_AUX_WEIGHTS, AutoBEVLoss
from model_components.losses import RouteReconstructionLoss, TrajectoryXYImitationLoss
from training.lateral_recovery import feature_consistency_kl

TRAJECTORY_WEIGHT: Final = 1.0
ROUTE_WEIGHT: Final = 0.1
POSE_WEIGHT: Final = 0.1
# A 0.1 rad heading error weighs like a 1 m offset.
POSE_YAW_SCALE_M: Final = 10.0
CONSISTENCY_WEIGHT: Final = 0.1


class AutoBEVObjective(nn.Module):
    """Stage 1: every AutoBEV term on the AutoBEV outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.autobev = AutoBEVLoss(STAGE1_WEIGHTS)

    def forward(self, outputs: Mapping[str, torch.Tensor], batch: Mapping) -> dict[str, torch.Tensor]:
        return self.autobev(dict(outputs), dict(batch))


class AutoE2EObjective(nn.Module):
    """Stage 2: planner terms plus the scheduled AutoBEV auxiliary terms."""

    def __init__(self) -> None:
        super().__init__()
        self.trajectory = TrajectoryXYImitationLoss()
        self.route = RouteReconstructionLoss()
        self.autobev = AutoBEVLoss(STAGE2_AUX_WEIGHTS)

    def forward(
        self,
        controls: torch.Tensor,
        auxiliary: Mapping[str, torch.Tensor],
        batch: Mapping,
        *,
        auxiliary_weight: float,
        teacher_bev: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        supervised = batch["planner_supervised"].bool()
        trajectory = self.trajectory(
            controls, batch["trajectory_xy_m"], batch["trajectory_valid"].bool() & supervised[:, None],
            batch["initial_speed_mps"],
        )
        route = self.route(
            auxiliary["route_reconstruction_logits"], auxiliary["route_raster"].detach(),
            batch["route_channel_valid"].bool() & supervised[:, None],
        )
        scale = torch.tensor([1.0, 1.0, POSE_YAW_SCALE_M], device=controls.device)
        pose_error = F.smooth_l1_loss(
            auxiliary["map_pose_prediction"].float() * scale, auxiliary["map_pose"].float() * scale,
            reduction="none",
        ).sum(-1)
        pose = (pose_error * supervised).sum() / supervised.sum().clamp_min(1)
        consistency = feature_consistency_kl(auxiliary["planner_bev"], teacher_bev, teacher_mask)
        autobev = self.autobev(dict(auxiliary), dict(batch))
        total = (
            TRAJECTORY_WEIGHT * trajectory + ROUTE_WEIGHT * route + POSE_WEIGHT * pose
            + CONSISTENCY_WEIGHT * consistency + auxiliary_weight * autobev["total"]
        )
        return {
            "total": total,
            "trajectory": trajectory.detach(),
            "route": route.detach(),
            "pose": pose.detach(),
            "consistency": consistency.detach(),
            "autobev": autobev["total"].detach(),
            "autobev_bev": autobev["bev"],
            "autobev_depth": autobev["depth"],
            "autobev_center": autobev["center_heat"] + autobev["center_offset"],
        }
