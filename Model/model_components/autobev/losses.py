"""AutoBEV losses: BEV segmentation, LiDAR depth, FCOS3D-style boxes, image-space map segmentation, centres.

Ported from ``experiment/autobev/losses.py``. Image-space map segmentation
(``pv_seg``) is taught by:

- the BEV map labels projected onto the ground seen by each stride-8 cell
  (``projected_map_targets``); cells whose LiDAR depth is clearly shorter than
  the ground hit are occluded by something above the road and are ignored, as
  in the perspective map supervision of MapTRv2 (Liao et al. 2023);
- 2D pseudo labels and the BEV-to-image render loss of the experiment are not
  ported yet: they need a pseudo-label layer in the data pipeline first.

Every dataset's ego origin is on the road (``geometry.EGO_ORIGIN``), so the
prior road height is 0 m; with LiDAR it is refined per sample.

``autobev_dense_supervised`` [B] selects the samples whose depth and centre
terms count. The other batch keys are those of an AutoBEV sample: ``bev_target`` and ``bev_valid``
[B, 8, 220, 160], ``depth`` [B, 6, 96, 168] (metres, 0 = no return),
``boxes`` and ``box_labels`` (lists of [N, 9] and [N]) and ``projection``
[B, 6, 3, 4]; augmentation adds ``image_projection``, ``image_boxes`` and
``image_valid``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from data_processing.reactive_training_artifacts import BEV_SEGMENTATION_CLASSES

from ..losses.bev_segmentation_loss import BEVSegmentationAuxiliaryLoss
from .encoder import (
    CENTER_GROUPS,
    NUM_BOX_CLASSES,
    PV_SEG_CLASSES,
    depth_to_bin,
    focal_scale,
)
from .geometry import FEATURE_STRIDE, IMAGE_H, IMAGE_W, PC_RANGE

# Per-class positive weights measured on the nuPlan pack for ft5
# (history.json bev_pos_weight_*), reused until the subset has its own.
BEV_POS_WEIGHT = (1.911607, 20.74171, 4.963441, 64.0, 64.0, 53.499984, 64.0, 64.0)
BOX_MAX_DEPTH_M = 80.0
# Box classes (pack.BOX_CLASSES: VEHICLE, PEDESTRIAN, BICYCLE, OTHER) per center group.
CENTER_GROUP_OF_BOX = {0: 0, 1: 1, 2: 1}
OBJECT_CLASS_INDEX = [BEV_SEGMENTATION_CLASSES.index(name) for name in CENTER_GROUPS]
BOX_RADIUS_CELLS = 1.5
PV_CLASS_INDEX = [BEV_SEGMENTATION_CLASSES.index(name) for name in PV_SEG_CLASSES]
PV_POS_WEIGHT = tuple(BEV_POS_WEIGHT[i] for i in PV_CLASS_INDEX)
LOSS_TERMS = ("bev", "depth", "box", "pv_seg", "object", "center")
# Stage 1 trains AutoBEV with every term. ``object`` adds a separate BCE+Dice
# term on vehicle and vulnerable_road_user, so the map-side terms cannot crowd
# out the object classes (BEVFusion reports that tasks sharing an encoder pull
# each other down).
STAGE1_WEIGHTS = {"bev": 1.0, "depth": 1.0, "box": 1.0, "pv_seg": 0.5,
                  "object": 0.5, "center": 1.0}
# Stage 2 keeps segmentation, depth and centres as auxiliary terms; their sum
# is scaled by the stage 2 schedule.
STAGE2_AUX_WEIGHTS = {"bev": 1.0, "depth": 1.0, "box": 0.0, "pv_seg": 0.0,
                      "object": 0.0, "center": 1.0}
# A LiDAR return blocks a cell's ground ray when it is this far above the
# road and at least OCCLUSION_MARGIN_M nearer than the ground hit.
OCCLUDER_HEIGHT_M, OCCLUSION_MARGIN_M = 0.25, 1.0
GROUND_SEARCH_M = 1.0  # LiDAR returns within this of the prior road height estimate the road


def _ground_rays(projection: torch.Tensor, ground_z: torch.Tensor, feature_hw, offsets=(0.0,)):
    """Ground hits of rays through each stride-8 cell at sub-pixel ``offsets`` (fractions of a cell).

    projection [B, V, 3, 4] maps the (augmented) ego frame to pixels. Returns
    camera centre [B, V, 3], ray directions [B, V, S, h, w, 3] with unit
    camera depth, ground depth t [B, V, S, h, w] and the BEV grid_sample grid
    [B, V, S, h, w, 2] of the ground points, S = len(offsets) ** 2.
    """
    batch, views = projection.shape[:2]
    height, width = feature_hw
    device = projection.device
    m = projection[..., :3].double()
    m_inv = torch.linalg.inv(m)
    center = -(m_inv @ projection[..., 3].double().unsqueeze(-1)).squeeze(-1)
    shifts = torch.tensor(offsets, device=device, dtype=torch.float64)
    dv, du = torch.meshgrid(shifts, shifts, indexing="ij")
    rows = torch.arange(height, device=device, dtype=torch.float64)
    cols = torch.arange(width, device=device, dtype=torch.float64)
    v = (rows.view(1, -1, 1) + 0.5 + dv.reshape(-1, 1, 1)) * FEATURE_STRIDE  # [S, h, 1]
    u = (cols.view(1, 1, -1) + 0.5 + du.reshape(-1, 1, 1)) * FEATURE_STRIDE  # [S, 1, w]
    v, u = torch.broadcast_tensors(v, u)
    pixels = torch.stack([u, v, torch.ones_like(u)], -1)  # [S, h, w, 3]
    direction = torch.einsum("bvij,shwj->bvshwi", m_inv, pixels)
    # P (C + t D) = t [u, v, 1], so t is the camera depth of the ground point.
    t = (ground_z.double().view(batch, 1, 1, 1, 1) - center[..., 2].view(batch, views, 1, 1, 1)) / direction[..., 2]
    point = center.view(batch, views, 1, 1, 1, 3) + t[..., None] * direction
    x_min, y_min, x_max, y_max = PC_RANGE[0], PC_RANGE[1], PC_RANGE[3], PC_RANGE[4]
    grid = torch.stack([(y_max - point[..., 1]) / (y_max - y_min) * 2 - 1,
                        (x_max - point[..., 0]) / (x_max - x_min) * 2 - 1], -1)
    return center, direction, t.float(), grid.float()


def ground_lookup(projection: torch.Tensor, ground_z: torch.Tensor, lidar_depth: torch.Tensor | None,
                  feature_hw, image_valid: torch.Tensor | None = None):
    """Where each stride-8 cell's central ray meets the road, as a BEV grid_sample grid.

    Returns grid [B, V, h, w, 2], a mask of cells whose ray hits the ground
    ahead of the camera inside the BEV grid, inside the (augmented) image and,
    with ``lidar_depth`` [B, V, h, w], not behind an occluder, and the road
    height [B] used. With LiDAR the road height is the median height of the
    returns within GROUND_SEARCH_M of the prior ``ground_z`` (nuPlan's origin
    sits about 0.3 m above its road, for example). An occluder is a return
    OCCLUDER_HEIGHT_M above that road and nearer than the ground hit; testing
    height rather than depth keeps far ground, whose nearest return in an
    8x8 cell is the cell's lower edge, from counting as blocked.
    """
    batch, views = projection.shape[:2]
    height, width = feature_hw
    center, direction, t, grid = _ground_rays(projection, ground_z, feature_hw)
    if lidar_depth is not None:
        hit_z = (center[..., 2].view(batch, views, 1, 1) + lidar_depth.double() * direction[:, :, 0, ..., 2]).float()
        has = lidar_depth > 0
        road = []
        for index in range(batch):
            near = hit_z[index][has[index] & ((hit_z[index] - ground_z[index]).abs() < GROUND_SEARCH_M)]
            # A low quantile: car bumpers and kerbs also fall within the window.
            road.append(near.quantile(0.3) if near.numel() > 100 else ground_z[index].float())
        ground_z = torch.stack(road).to(ground_z.dtype)
        center, direction, t, grid = _ground_rays(projection, ground_z, feature_hw)
    t, grid, direction = t[:, :, 0], grid[:, :, 0], direction[:, :, 0]
    mask = (t > 0.5) & (grid.abs() <= 1).all(-1)
    if image_valid is not None:
        mask &= image_valid
    if lidar_depth is not None:
        above = hit_z > ground_z.view(batch, 1, 1, 1).float() + OCCLUDER_HEIGHT_M
        blocked = (lidar_depth > 0) & above & (lidar_depth < t - OCCLUSION_MARGIN_M)
        # LiDAR returns are sparse (32 to 128 beams), so a car shows up as
        # rows of blocked cells; widen them, mostly vertically, to cover it.
        blocked = F.max_pool2d(blocked.flatten(0, 1)[:, None].float(), (11, 3), stride=1, padding=(5, 1))
        mask &= ~(blocked.view(batch, views, height, width) > 0)
    return grid, mask, ground_z


def _sample_bev(maps: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    """Sample [B, C, H, W] BEV maps at grid [B, V, h, w, 2]; returns [B, V, C, h, w]."""
    batch, views, height, width = grid.shape[:4]
    sampled = F.grid_sample(maps.float(), grid.reshape(batch, views * height, width, 2), mode=mode,
                            padding_mode="zeros", align_corners=False)
    return sampled.view(batch, -1, views, height, width).permute(0, 2, 1, 3, 4)


def _sample_valid_target(target: torch.Tensor, valid: torch.Tensor, grid: torch.Tensor):
    """Bilinear target from valid cells only (normalized by the valid weight), and where that weight is > 0.5."""
    weight = _sample_bev(valid.float(), grid, "bilinear")
    value = _sample_bev(target.float() * valid.float(), grid, "bilinear") / weight.clamp(min=1e-6)
    return value.clamp(0.0, 1.0), weight > 0.5


def projected_map_targets(batch: dict, feature_hw):
    """BEV map labels on the ground seen by each stride-8 cell: target, valid [B, V, C, h, w] and the ray grid."""
    projection = batch["camera_projection_matrix"]
    grid, mask, road = ground_lookup(projection, projection.new_zeros(projection.shape[0]),
                                     batch["autobev_depth"], feature_hw, batch.get("autobev_image_valid"))
    target, valid = _sample_valid_target(batch["bev_segmentation_target"][:, PV_CLASS_INDEX],
                                         batch["bev_segmentation_valid"][:, PV_CLASS_INDEX], grid)
    return target, valid & mask[:, :, None], grid, mask, road


def masked_bce_dice(logits, target, valid, pos_weight=None, probabilities: bool = False) -> torch.Tensor:
    """BCE (pos_weight per class, or plain) plus soft Dice over valid cells, per class then averaged.

    ``logits`` [N, C, h, w] are logits, or probabilities with ``probabilities``.
    Soft pseudo targets use plain BCE: a large pos_weight would move the
    optimum of a 20 % paint cell to over 0.8.
    """
    mask = valid.float()
    target = target.float() * mask
    if probabilities:
        bce = F.binary_cross_entropy(logits.clamp(1e-6, 1 - 1e-6), target, reduction="none")
        probability = logits * mask
    else:
        weight = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device).view(1, -1, 1, 1)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight, reduction="none")
        probability = logits.sigmoid() * mask
    count = mask.sum((0, 2, 3))
    bce = (bce * mask).sum((0, 2, 3)) / count.clamp_min(1.0)
    dice = 1 - (2 * (probability * target).sum((0, 2, 3)) + 1e-5) / ((probability + target).sum((0, 2, 3)) + 1e-5)
    present = (count > 0).float()
    return ((bce + dice) * present).sum() / present.sum().clamp_min(1.0)


def center_targets(boxes: torch.Tensor, labels: torch.Tensor, height: int, width: int):
    """Gaussian center heatmaps [G, H, W], offsets [2, H, W] and center mask [H, W] for one sample.

    ``boxes`` are in the (augmented) ego frame, as the BEV labels. The Gaussian
    sigma follows the footprint (a sixth of its mean side in cells, at least
    0.8 cell), as in CenterNet's size-adaptive radius.
    """
    device = boxes.device
    heat = torch.zeros(len(CENTER_GROUPS), height, width, device=device)
    offset = torch.zeros(2, height, width, device=device)
    peak = torch.zeros(height, width, dtype=torch.bool, device=device)
    if boxes.numel() == 0:
        return heat, offset, peak
    x_max, y_max = PC_RANGE[3], PC_RANGE[4]
    mpp = (PC_RANGE[3] - PC_RANGE[0]) / height
    rows_f = (x_max - boxes[:, 0]) / mpp
    cols_f = (y_max - boxes[:, 1]) / mpp
    rows, cols = rows_f.floor().long(), cols_f.floor().long()
    group = torch.tensor([CENTER_GROUP_OF_BOX.get(int(c), -1) for c in labels.tolist()], device=device)
    keep = (group >= 0) & (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if not keep.any():
        return heat, offset, peak
    rows, cols, rows_f, cols_f, group = rows[keep], cols[keep], rows_f[keep], cols_f[keep], group[keep]
    sigma = ((boxes[keep, 3] + boxes[keep, 4]) / 2 / mpp / 6).clamp(min=0.8)
    grid_r = torch.arange(height, device=device, dtype=torch.float32).view(1, -1, 1)
    grid_c = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, -1)
    gauss = torch.exp(-((grid_r - rows.view(-1, 1, 1)) ** 2 + (grid_c - cols.view(-1, 1, 1)) ** 2)
                      / (2 * sigma.view(-1, 1, 1) ** 2))
    for g in range(len(CENTER_GROUPS)):
        chosen = group == g
        if chosen.any():
            heat[g] = gauss[chosen].amax(0)
    heat[group, rows, cols] = 1.0
    offset[0, rows, cols] = rows_f - rows
    offset[1, rows, cols] = cols_f - cols
    peak[rows, cols] = True
    return heat, offset, peak


def center_focal_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss over valid cells, normalized by the number of peaks."""
    probability = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    positive = target >= 1.0
    pos_loss = -((1 - probability) ** 2) * probability.log() * positive
    neg_loss = -((1 - target) ** 4) * probability ** 2 * (1 - probability).log() * (~positive)
    mask = valid.float()
    return ((pos_loss + neg_loss) * mask).sum() / (positive & valid).sum().clamp(min=1)


def box_targets(boxes: torch.Tensor, labels: torch.Tensor, projection: torch.Tensor, feature_hw,
                scale: torch.Tensor | None = None):
    """Assign each stride-8 cell of each camera to the nearest projected box center.

    Returns per-camera tensors: class index (-1 = background), centerness,
    and the regression target [offset_u, offset_v, log depth, log l, log w,
    log h, sin alpha, cos alpha], where alpha is the yaw relative to the
    ray from the camera center so that one head serves all cameras. With
    ``scale`` [V] (F_REF / f) the depth target is focal-normalized, like the
    virtual depth of Cube R-CNN (Brazil et al., CVPR 2023).
    """
    views = projection.shape[0]
    height, width = feature_hw
    device = projection.device
    cls = torch.full((views, height, width), -1, dtype=torch.long, device=device)
    centerness = torch.zeros((views, height, width), device=device)
    regression = torch.zeros((views, 8, height, width), device=device)
    if boxes.numel() == 0:
        return cls, centerness, regression
    centers = torch.cat([boxes[:, :3], torch.ones_like(boxes[:, :1])], 1)
    rows = (torch.arange(height, device=device) + 0.5) * FEATURE_STRIDE
    cols = (torch.arange(width, device=device) + 0.5) * FEATURE_STRIDE
    grid_v, grid_u = torch.meshgrid(rows, cols, indexing="ij")
    for view in range(views):
        # Yaw relative to the ray from this camera's center (P = [M | p],
        # center = -M^-1 p), so one head sees the same target for the same look.
        camera = -torch.linalg.solve(projection[view][:, :3].double(), projection[view][:, 3].double()).float()
        alpha = boxes[:, 6] - torch.atan2(boxes[:, 1] - camera[1], boxes[:, 0] - camera[0])
        uvw = centers @ projection[view].T
        depth = uvw[:, 2]
        keep = (depth > 0.5) & (depth < BOX_MAX_DEPTH_M)
        u, v = uvw[:, 0] / depth.clamp(min=1e-3), uvw[:, 1] / depth.clamp(min=1e-3)
        keep &= (u >= 0) & (u < IMAGE_W) & (v >= 0) & (v < IMAGE_H)
        if not keep.any():
            continue
        index = keep.nonzero().squeeze(1)
        du = (grid_u[None] - u[index, None, None]) / FEATURE_STRIDE
        dv = (grid_v[None] - v[index, None, None]) / FEATURE_STRIDE
        distance = (du.square() + dv.square()).sqrt()
        cost = torch.where(distance <= BOX_RADIUS_CELLS, depth[index, None, None], torch.inf)
        best_cost, best = cost.min(0)
        positive = torch.isfinite(best_cost)
        chosen = index[best]
        cls[view][positive] = labels[chosen][positive]
        near = distance.gather(0, best[None])[0]
        centerness[view] = torch.where(positive, torch.exp(-near.square() / 2.0), 0.0)
        target = torch.stack([
            -du.gather(0, best[None])[0], -dv.gather(0, best[None])[0],
            (depth[chosen] * (1.0 if scale is None else scale[view])).clamp(min=0.5).log(),
            boxes[chosen, 3].clamp(min=0.1).log(), boxes[chosen, 4].clamp(min=0.1).log(),
            boxes[chosen, 5].clamp(min=0.1).log(),
            alpha[chosen].sin(), alpha[chosen].cos(),
        ])
        regression[view] = torch.where(positive[None], target, 0.0)
    return cls, centerness, regression


BEV_CLASS_NAMES = BEV_SEGMENTATION_CLASSES


@torch.no_grad()
def bev_class_terms(logits, target, valid) -> dict[str, torch.Tensor]:
    """Per-class weighted BCE and soft Dice, for diagnosing which classes learn."""
    weight = torch.tensor(BEV_POS_WEIGHT, device=logits.device).view(1, -1, 1, 1)
    mask = valid.float()
    target = target.float() * mask
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight, reduction="none")
    bce = (bce * mask).sum((0, 2, 3)) / mask.sum((0, 2, 3)).clamp_min(1.0)
    probability = logits.sigmoid() * mask
    dice = 1 - (2 * (probability * target).sum((0, 2, 3)) + 1e-5) / ((probability + target).sum((0, 2, 3)) + 1e-5)
    out = {}
    for index, name in enumerate(BEV_CLASS_NAMES):
        out[f"bce/{name}"] = bce[index]
        out[f"dice/{name}"] = dice[index]
    return out


class AutoBEVLoss(torch.nn.Module):
    def __init__(self, weights: Mapping[str, float]) -> None:
        super().__init__()
        if set(weights) != set(LOSS_TERMS):
            raise ValueError(f"AutoBEV loss weights must name exactly {LOSS_TERMS}")
        self.bev = BEVSegmentationAuxiliaryLoss(BEV_POS_WEIGHT)
        self.weights = dict(weights)

    def forward(self, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
        bev_parts = self.bev.components(outputs["bev_logits"].float(), batch["bev_segmentation_target"], batch["bev_segmentation_valid"])
        bev = bev_parts["total"]
        per_class = bev_class_terms(outputs["bev_logits"].detach().float(), batch["bev_segmentation_target"], batch["bev_segmentation_valid"])

        image_projection = batch.get("autobev_image_projection", batch["camera_projection_matrix"])
        scale = focal_scale(image_projection)  # [B, V]
        logits = outputs["depth_logits"].float()  # [B*V, D, h, w]
        # ``autobev_dense_supervised`` [B] marks the samples whose depth and
        # centres are supervised; stage 2 gives the mixed-in datasets only the
        # segmentation term.
        dense = batch["autobev_dense_supervised"].bool()
        views = batch["autobev_depth"].shape[1]
        depth = batch["autobev_depth"].flatten(0, 1)
        valid = (depth > 0) & dense.repeat_interleave(views)[:, None, None]
        normalized = depth * scale.flatten()[:, None, None]
        depth_loss = (
            F.cross_entropy(logits.permute(0, 2, 3, 1)[valid], depth_to_bin(normalized[valid]))
            if valid.any() else logits.sum() * 0.0
        )

        pv_logits = outputs["pv_seg_logits"].float()  # [B*V, C, h, w]
        feature_hw = pv_logits.shape[-2:]
        pv_target, pv_valid, grid, mask, road = projected_map_targets(batch, feature_hw)
        pv_loss = masked_bce_dice(pv_logits, pv_target.flatten(0, 1), pv_valid.flatten(0, 1), PV_POS_WEIGHT)
        cls_logits = outputs["box_cls"].float()
        reg = outputs["box_reg"].float()
        boxes = batch.get("autobev_image_boxes", batch["autobev_boxes"])
        targets = [
            box_targets(box, label, projection, feature_hw, factor)
            for box, label, projection, factor in zip(boxes, batch["autobev_box_labels"], image_projection, scale)
        ]
        cls_t = torch.cat([t[0] for t in targets])
        center_t = torch.cat([t[1] for t in targets])
        reg_t = torch.cat([t[2] for t in targets])
        positive = cls_t >= 0
        num_positive = positive.sum().clamp(min=1).float()
        one_hot = F.one_hot(cls_t.clamp(min=0), NUM_BOX_CLASSES).permute(0, 3, 1, 2).float()
        one_hot = one_hot * positive[:, None]
        cls_loss = sigmoid_focal_loss(cls_logits, one_hot, reduction="sum") / num_positive
        if positive.any():
            pred = reg.permute(0, 2, 3, 1)[positive]
            center_loss = F.binary_cross_entropy_with_logits(pred[:, 0], center_t[positive])
            # BCE on a soft target cannot go below the target entropy, so the
            # KL (BCE minus entropy) is what shows whether centerness learns.
            t = center_t[positive].clamp(1e-6, 1 - 1e-6)
            center_kl = (center_loss - (-(t * t.log() + (1 - t) * (1 - t).log())).mean()).detach()
            reg_loss = F.smooth_l1_loss(pred[:, 1:], reg_t.permute(0, 2, 3, 1)[positive], beta=1.0 / 9.0)
        else:
            center_loss = reg_loss = reg.sum() * 0.0
            center_kl = center_loss.detach()
        box = cls_loss + center_loss + reg_loss
        object_loss = heat_loss = offset_loss = bev.new_zeros(())
        if self.weights["object"]:
            bev_logits = outputs["bev_logits"].float()
            object_loss = masked_bce_dice(bev_logits[:, OBJECT_CLASS_INDEX], batch["bev_segmentation_target"][:, OBJECT_CLASS_INDEX],
                                          batch["bev_segmentation_valid"][:, OBJECT_CLASS_INDEX],
                                          tuple(BEV_POS_WEIGHT[i] for i in OBJECT_CLASS_INDEX))
        if self.weights["center"]:
            height, width = outputs["center_logits"].shape[-2:]
            targets_c = [center_targets(box, label, height, width)
                         for box, label in zip(batch["autobev_boxes"], batch["autobev_box_labels"])]
            heat_t = torch.stack([t[0] for t in targets_c])
            offset_t = torch.stack([t[1] for t in targets_c])
            peak_t = torch.stack([t[2] for t in targets_c])
            valid_c = batch["bev_segmentation_valid"][:, OBJECT_CLASS_INDEX] & dense[:, None, None, None]
            heat_loss = center_focal_loss(outputs["center_logits"].float(), heat_t, valid_c)
            # A masked mean keeps the offset head in the graph on steps without
            # any object, as DDP's static graph needs.
            peak = (peak_t & dense[:, None, None] & valid_c.any(1))[:, None].expand_as(offset_t).float()
            offset_error = F.l1_loss(outputs["center_offset"].float(), offset_t, reduction="none")
            offset_loss = (offset_error * peak).sum() / peak.sum().clamp_min(1.0)
        terms = {"bev": bev, "depth": depth_loss, "box": box, "pv_seg": pv_loss, "object": object_loss,
                 "center": heat_loss + offset_loss}
        total = sum(self.weights[name] * terms[name] for name in LOSS_TERMS)
        return {"total": total, "bev": bev.detach(), "bev_bce": bev_parts["bce"].detach(),
                "bev_dice": bev_parts["dice"].detach(), **per_class, "depth": depth_loss.detach(),
                "box_cls": cls_loss.detach(), "box_center": center_loss.detach(), "box_center_kl": center_kl,
                "box_reg": reg_loss.detach(), "box_positives": num_positive.detach(),
                "pv_seg": pv_loss.detach(),
                "object": object_loss.detach(), "center_heat": heat_loss.detach(), "center_offset": offset_loss.detach(),
                "pv_valid_frac": pv_valid.float().mean().detach()}
