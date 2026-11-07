"""AutoBEV: BEVFormer v2 T8 on six 1344x768 cameras with perspective supervision.

Ported from ``experiment/autobev/model.py``. The encoder reads a DINOv2 ViT-B/14
with registers (Oquab et al. 2023; Darcet et al. 2024, Apache-2.0), whose
stride-14 features are resampled to the stride 8, 16 and 32 levels of the
BEVFormer pyramid, and lifts them into the 220x160 grid of
``geometry.AUTOBEV_GEOMETRY`` with the six-layer t1 encoder and T8 temporal
fusion.

Heads:

- ``bev_head``: eight BEV classes on the fused BEV;
- ``center``: object centre heatmaps and sub-cell offsets on the fused BEV
  (FIERY / CenterNet style);
- ``perspective`` on the stride-8 level of the current images: LiDAR depth as
  classification over log-spaced bins in a focal-normalized space (DG-BEV,
  Lu et al., CVPR 2023; d_n = d * F_REF / f), an FCOS3D-style box head, and
  image-space map segmentation.

The predicted depth also weights the spatial cross-attention: each pillar
point's sampling weight is scaled by how likely its depth is at its pixel
(DFA3D-style lifting, Li et al., ICCV 2023).

The fused BEV before the heads, ``[B, 256, 220, 160]``, is the AutoE2E camera
BEV. The seven history frames go through the encoder without gradient.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..auxiliary_heads import BEVSegmentationHead
from ..feature_pyramid import BEVFormerFeaturePyramid
from ..view_fusion.bevformer_v2_t1 import BEVFormerV2T1ViewFusion
from ..view_fusion.bevformer_v2_t8 import BEVFormerV2T8TemporalFusion
from ..view_fusion.projection import ImageTransform, PinholeProjection
from .geometry import BEV_H, BEV_W, EMBED_DIM, IMAGE_H, IMAGE_W, NUM_VIEWS, PC_RANGE

NUM_BEV_CLASSES = 8
NUM_BOX_CLASSES = 4
# Focal-normalized depth: bins cover 1-120 m for focal lengths 0.7 to 1.5
# times F_REF, so 60-80 m targets do not pile up in the last bins.
F_REF = 1000.0
DEPTH_BINS = 112
DEPTH_MIN_M, DEPTH_MAX_M = 0.7, 180.0
DEPTH_WEIGHT_FLOOR = 0.2
PV_SEG_CLASSES = ("drivable_area", "lane_boundary", "intersection", "crosswalk", "stop_line")
CENTER_GROUPS = ("vehicle", "vulnerable_road_user")


def focal_scale(projection: torch.Tensor) -> torch.Tensor:
    """F_REF / f per view for [..., 3, 4] projections P = K [R | t] (zero skew).

    Row 2 of P[:, :3] is the camera's unit optical axis; removing it from
    rows 0 and 1 leaves fx and fy times unit vectors. f is the focal length
    whose pixel size matches DG-BEV's sqrt(1/fx^2 + 1/fy^2) / sqrt(2).
    """
    m = projection[..., :3].float()
    axis = m[..., 2, :]
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    fx = (m[..., 0, :] - (m[..., 0, :] * axis).sum(-1, keepdim=True) * axis).norm(dim=-1)
    fy = (m[..., 1, :] - (m[..., 1, :] * axis).sum(-1, keepdim=True) * axis).norm(dim=-1)
    focal = (2.0 / (fx.clamp(min=1.0).pow(-2) + fy.clamp(min=1.0).pow(-2))).sqrt()
    return F_REF / focal


def depth_bin_centers(device=None) -> torch.Tensor:
    """Log-spaced depth bin centres in metres."""
    edges = torch.logspace(
        math.log10(DEPTH_MIN_M), math.log10(DEPTH_MAX_M), DEPTH_BINS + 1, device=device
    )
    return (edges[:-1] * edges[1:]).sqrt()


def depth_to_bin(depth: torch.Tensor) -> torch.Tensor:
    position = (depth.clamp(DEPTH_MIN_M, DEPTH_MAX_M - 1e-3).log() - math.log(DEPTH_MIN_M)) / (
        math.log(DEPTH_MAX_M) - math.log(DEPTH_MIN_M)
    )
    return (position * DEPTH_BINS).long().clamp(0, DEPTH_BINS - 1)


def _tower(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
    )


class PerspectiveHeads(nn.Module):
    """Dense depth, FCOS3D-style box and map-segmentation outputs on one FPN level."""

    def __init__(self, channels: int = EMBED_DIM) -> None:
        super().__init__()
        self.depth_tower = _tower(channels)
        self.depth = nn.Conv2d(channels, DEPTH_BINS, 1)
        self.box_tower = _tower(channels)
        self.box_cls = nn.Conv2d(channels, NUM_BOX_CLASSES, 3, padding=1)
        # centerness 1, offset 2, log depth 1, log size 3, yaw sin/cos 2
        self.box_reg = nn.Conv2d(channels, 9, 3, padding=1)
        nn.init.constant_(self.box_cls.bias, -math.log(99.0))
        self.seg_tower = _tower(channels)
        self.seg = nn.Conv2d(channels, len(PV_SEG_CLASSES), 1)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        box = self.box_tower(feature)
        return {
            "depth_logits": self.depth(self.depth_tower(feature)),
            "box_cls": self.box_cls(box),
            "box_reg": self.box_reg(box),
            "pv_seg_logits": self.seg(self.seg_tower(feature)),
        }


class DinoV2Backbone(nn.Module):
    """DINOv2 ViT-B/14 (registers) with stride 8/16/32 feature maps from three depths.

    Blocks 4, 8 and 12 are read out (as in ViTDet-style adapters), resampled
    bilinearly from stride 14 to strides 8, 16 and 32, and refined by a 3x3
    convolution. LayerNorm only, so nothing needs freezing for small batches.
    """

    MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"
    BLOCKS = (3, 7, 11)
    STRIDES = (8, 16, 32)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.vit = timm.create_model(self.MODEL, pretrained=pretrained, dynamic_img_size=True,
                                     dynamic_img_pad=True, num_classes=0)
        self.vit.set_grad_checkpointing(True)
        width = self.vit.embed_dim
        self.refine = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(32, width), nn.GELU())
            for _ in self.STRIDES)
        self.feature_channels = [width] * len(self.STRIDES)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        height, width = images.shape[-2:]
        maps = self.vit.forward_intermediates(images, indices=list(self.BLOCKS), output_fmt="NCHW",
                                              intermediates_only=True)
        return [refine(F.interpolate(feature, size=(height // stride, width // stride), mode="bilinear",
                                     align_corners=False))
                for refine, feature, stride in zip(self.refine, maps, self.STRIDES)]


class CenterHead(nn.Module):
    """Object centre heatmaps and sub-cell offsets on the fused BEV.

    Outputs ``center_logits`` [B, len(CENTER_GROUPS), H, W] and
    ``center_offset`` [B, 2, H, W] (row, column fraction of the centre cell).
    A pedestrian covers one or two 0.5 m cells, so its segmentation mask gives
    little signal; a peaked centre target gives the same object a clear one.
    """

    def __init__(self, channels: int = EMBED_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.tower = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU())
        self.heatmap = nn.Conv2d(hidden, len(CENTER_GROUPS), 1)
        self.offset = nn.Conv2d(hidden, 2, 1)
        nn.init.constant_(self.heatmap.bias, -math.log(99.0))

    def forward(self, bev: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.tower(bev)
        return {"center_logits": self.heatmap(feature), "center_offset": self.offset(feature).sigmoid()}


class AutoBEV(nn.Module):
    """Camera BEV encoder of AutoE2E; see the module docstring."""

    def __init__(self, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = DinoV2Backbone(pretrained=pretrained_backbone)
        self.pyramid = BEVFormerFeaturePyramid(self.backbone.feature_channels, embed_dim=EMBED_DIM)
        self.view_fusion = BEVFormerV2T1ViewFusion(
            num_views=NUM_VIEWS, embed_dim=EMBED_DIM, bev_h=BEV_H, bev_w=BEV_W, pc_range=PC_RANGE,
            # Six encoder layers over 35,200 queries do not fit otherwise.
            activation_checkpointing=True,
        )
        self.temporal_fusion = BEVFormerV2T8TemporalFusion(embed_dim=EMBED_DIM)
        self.image_transform = ImageTransform(model_input_size=(IMAGE_W, IMAGE_H))
        self.bev_head = BEVSegmentationHead(embed_dim=EMBED_DIM, num_classes=NUM_BEV_CLASSES)
        self.perspective = PerspectiveHeads()
        self.center = CenterHead()

    def _depth_weight(self, depth_logits: torch.Tensor, matrix: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # Projecting pillar points is geometry: bf16 would move them by pixels.
        with torch.autocast(matrix.device.type, enabled=False):
            return self._depth_weight_fp32(depth_logits, matrix, scale)

    def _depth_weight_fp32(self, depth_logits: torch.Tensor, matrix: torch.Tensor,
                           scale: torch.Tensor) -> torch.Tensor:
        """How likely each pillar point lies at its depth, per view: [B, V, N, Z].

        The predicted depth distribution is divided by its per-pixel maximum,
        so a flat (untrained) prediction gives weights near 1 and the encoder
        starts as plain BEVFormer. The value at (u, v, depth) is read with a
        trilinear 3D grid_sample over (log-depth bin, row, column); ``scale``
        [B, V] is F_REF / f, which maps metric depth to the predicted space.
        """
        view = self.view_fusion
        batch, views = matrix.shape[:2]
        homo = view._ego_reference_homo(view.reference_points_3d).to(matrix.device, torch.float32)
        cam = torch.einsum("bvij,mj->bvmi", matrix.float(), homo)
        depth = cam[..., 2]
        u = cam[..., 0] / depth.clamp(min=1e-3)
        v = cam[..., 1] / depth.clamp(min=1e-3)
        log_range = math.log(DEPTH_MAX_M) - math.log(DEPTH_MIN_M)
        normalized = depth * scale[:, :, None].to(depth.dtype)
        d = (normalized.clamp(min=DEPTH_MIN_M).log() - math.log(DEPTH_MIN_M)) / log_range
        grid = torch.stack([u / IMAGE_W * 2 - 1, v / IMAGE_H * 2 - 1, d * 2 - 1], dim=-1)
        # Detached: the depth head learns from its LiDAR loss, and the BEV loss
        # would otherwise push the distribution flat to open every gate.
        prob = depth_logits.detach().float().softmax(dim=1)
        relative = prob / prob.amax(dim=1, keepdim=True).clamp(min=1e-6)
        sampled = F.grid_sample(
            relative.unsqueeze(1),  # [B*V, 1, D, h, w]
            grid.reshape(batch * views, 1, 1, -1, 3),
            mode="bilinear", padding_mode="border", align_corners=False,
        ).reshape(batch, views, -1)
        # Floor the gate so occluded ground and object interiors, which the BEV
        # labels still need, keep some attention. Points behind the camera are
        # masked by the encoder; keep them neutral.
        sampled = DEPTH_WEIGHT_FLOOR + (1.0 - DEPTH_WEIGHT_FLOOR) * sampled
        sampled = torch.where(depth > DEPTH_MIN_M, sampled, torch.ones_like(sampled))
        return sampled.reshape(batch, views, view.bev_h * view.bev_w, view.num_points_in_pillar)

    def _frame_bev(self, images: torch.Tensor, matrix: torch.Tensor, scale: torch.Tensor):
        batch, views = images.shape[:2]
        features = self.backbone(images.flatten(0, 1))
        pyramid = tuple(self.pyramid(features))
        perspective = self.perspective(pyramid[0])
        weight = self._depth_weight(perspective["depth_logits"], matrix, scale)
        bev = self.view_fusion(
            pyramid, batch, views,
            projection=PinholeProjection(matrix, geometry_type="rectified_pinhole"),
            image_transform=self.image_transform,
            reference_depth_weight=weight,
        )
        return bev, perspective

    def forward(
        self,
        images: torch.Tensor,
        history_images: torch.Tensor,
        projection: torch.Tensor,
        history_projection: torch.Tensor,
        image_projection: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode ``images`` [B, 6, 3, 768, 1344] with ``history_images`` [B, 7, 6, 3, 768, 1344].

        ``projection`` [B, 6, 3, 4] and ``history_projection`` [B, 7, 6, 3, 4]
        map current-ego points into the current and past images.
        ``image_projection`` is the projection of the unaugmented ego frame
        when a BEV-space augmentation changed ``projection``; focal lengths are
        read from it because scaling the ego frame changes the row norms of P.

        Returns ``bev`` [B, 256, 220, 160] with ``bev_logits``,
        ``center_logits``, ``center_offset`` and the perspective outputs.
        """
        if images.shape[1:] != (NUM_VIEWS, 3, IMAGE_H, IMAGE_W):
            raise ValueError(f"AutoBEV images must be [B,{NUM_VIEWS},3,{IMAGE_H},{IMAGE_W}]")
        if history_images.shape[1] != 7 or history_images.shape[2:] != images.shape[1:]:
            raise ValueError("AutoBEV history must be seven frames shaped like the current images")
        scale = focal_scale(projection if image_projection is None else image_projection)
        history = []
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for index in range(history_images.shape[1]):
                bev, _ = self._frame_bev(history_images[:, index], history_projection[:, index], scale)
                history.append(bev.detach())
        self.train(was_training)
        # Weights cast to bf16 under no_grad stay in autocast's cache and would
        # be reused for the current frame, so convolutions (backbone, depth and
        # box heads) got no gradient. Drop the cache before the trained pass.
        torch.clear_autocast_cache()
        current, perspective = self._frame_bev(images, projection, scale)
        # Oldest first, current last, as in BEVFormer v2's ResNetFusion.
        fused = self.temporal_fusion([*history, current])
        return {"bev": fused, "bev_logits": self.bev_head(fused), **self.center(fused), **perspective}
