"""Reactive AutoE2E: AutoBEV camera BEV, map fusion at an estimated pose, GRU planner.

One forward, no modes:

1. AutoBEV encodes the six cameras and their T8 history into the camera BEV
   (220x160, 0.5 m, true ego frame) and its auxiliary outputs.
2. The map raster is cropped at the localization estimate ``map_pose``. For
   samples with ``map_matching`` the AutoBEV map classes refine that pose
   against the map first (``navigation.map_matching``); others keep the EKF
   pose. The raster is resampled onto the camera grid at the resulting pose
   (``navigation.resample``), not pooled to its shape.
3. Map and route are encoded separately and fused into the camera BEV by
   deformable cross-attention, which can sample the map where it really lies,
   so the network learns the remaining alignment. ``PoseErrorHead`` regresses
   that remaining crop pose and gives the alignment a direct signal.
4. A GRU planner decodes 64 steps of (x, y) from the fused BEV and the ego state.

At inference the raster arrives cropped at the onboard estimate, so
``map_pose`` is zero and ``map_pose_covariance`` is the EKF covariance.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

from navigation.geometry import MAP_CHANNEL_COUNT, ROUTE_CHANNEL_COUNT
from navigation.map_matching import match_pose
from navigation.resample import resample_map_context, resample_route

from .auxiliary_heads import RouteReconstructionHead
from .autobev import AutoBEV
from .autobev.geometry import AUTOBEV_GEOMETRY, EMBED_DIM
from .map_encoder import build_map_bev_fusion, build_split_navigation_encoder
from .trajectory_planning import GRUPlanner

NUM_TIMESTEPS: Final = 64
NUM_SIGNALS: Final = 2
EGOMOTION_DIM: Final = 256
ROUTE_HIDDEN_CHANNELS: Final = 96
MAP_FUSION_SAMPLE_POINTS: Final = 8
MAP_FUSION_HEADS: Final = 8
QUERY_CHUNK_SIZE: Final = 4096
PLANNER_ATTENTION_POINTS: Final = 16


class PoseErrorHead(nn.Module):
    """Regress the crop pose (x, y in m, yaw in rad) from the camera and map BEVs.

    Reading both BEVs side by side lets the head see how the map is displaced
    against what the cameras show. The last layer starts at zero.
    """

    def __init__(self, channels: int = EMBED_DIM, hidden: int = 64) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(2 * channels, hidden, 3, stride=2, padding=1)]
        for _ in range(3):
            layers += [nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv2d(hidden, hidden, 3, stride=2, padding=1)]
        self.tower = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.output = nn.Linear(hidden, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, camera_bev: torch.Tensor, map_bev: torch.Tensor) -> torch.Tensor:
        return self.output(self.tower(torch.cat([camera_bev, map_bev], dim=1)))


class ReactiveE2E(nn.Module):
    def __init__(self, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.AutoBEV = AutoBEV(pretrained_backbone=pretrained_backbone)
        self.NavigationEncoder = build_split_navigation_encoder(
            "semantic_raster",
            map_channels=MAP_CHANNEL_COUNT,
            route_channels=ROUTE_CHANNEL_COUNT,
            embed_dim=EMBED_DIM,
            output_h=AUTOBEV_GEOMETRY.height_px,
            output_w=AUTOBEV_GEOMETRY.width_px,
            route_hidden_channels=ROUTE_HIDDEN_CHANNELS,
        )
        self.MapBEVFusion = build_map_bev_fusion(
            "deformable",
            embed_dim=EMBED_DIM,
            num_sample_points=MAP_FUSION_SAMPLE_POINTS,
            num_heads=MAP_FUSION_HEADS,
            query_chunk_size=QUERY_CHUNK_SIZE,
        )
        self.PoseErrorHead = PoseErrorHead()
        self.RouteReconstructionHead = RouteReconstructionHead(
            embed_dim=EMBED_DIM, route_channels=ROUTE_CHANNEL_COUNT,
        )
        self.TrajectoryPlanner = GRUPlanner(
            embed_dim=EMBED_DIM,
            num_timesteps=NUM_TIMESTEPS,
            num_signals=NUM_SIGNALS,
            num_points=PLANNER_ATTENTION_POINTS,
            egomotion_dim=EGOMOTION_DIM,
        )

    def forward(
        self,
        images: torch.Tensor,
        history_images: torch.Tensor,
        projection: torch.Tensor,
        history_projection: torch.Tensor,
        map_context: torch.Tensor,
        route_mask: torch.Tensor,
        map_valid: torch.Tensor,
        route_valid: torch.Tensor,
        egomotion_history: torch.Tensor,
        map_pose: torch.Tensor,
        map_pose_covariance: torch.Tensor,
        map_matching: torch.Tensor,
        image_projection: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict ``[B, 128]`` (x, y) controls and the auxiliary outputs.

        ``map_context`` [B, 14, 450, 300] and ``route_mask`` [B, 2, 450, 300]
        are the navigation rasters in their own frame; ``map_pose`` [B, 3] is
        the crop pose ``T_M<-Ê`` in that frame, ``map_pose_covariance``
        [B, 3, 3] its covariance and ``map_matching`` [B] which samples refine
        it with the map. ``map_valid`` and ``route_valid`` [B] zero the
        navigation inputs of samples that have none.
        """
        camera = self.AutoBEV(images, history_images, projection, history_projection, image_projection)
        camera_bev = camera.pop("bev")
        with torch.no_grad():
            matched_pose, _, matched = match_pose(
                camera["bev_logits"].detach().float().sigmoid(), map_context, map_pose, map_pose_covariance,
            )
            use_match = matched & map_matching.bool()
            crop_pose = torch.where(use_match[:, None], matched_pose, map_pose.float())
            map_raster = resample_map_context(map_context, crop_pose, AUTOBEV_GEOMETRY)
            route_raster = resample_route(route_mask, crop_pose, AUTOBEV_GEOMETRY)
        map_gate = map_valid.reshape(-1, 1, 1, 1).to(map_raster.dtype)
        route_gate = route_valid.reshape(-1, 1, 1, 1).to(route_raster.dtype)
        navigation_bev, route_contribution = self.NavigationEncoder(
            map_raster * map_gate, route_raster * route_gate, return_route_contribution=True,
        )
        planner_bev = self.MapBEVFusion(camera_bev, navigation_bev)
        trajectory = self.TrajectoryPlanner(planner_bev, egomotion_history)
        return trajectory, {
            **camera,
            "planner_bev": planner_bev,
            "route_reconstruction_logits": self.RouteReconstructionHead(route_contribution),
            "route_raster": route_raster,
            "map_pose": crop_pose,
            "map_pose_prediction": self.PoseErrorHead(camera_bev, navigation_bev),
            "map_matching_used": use_match,
        }
