"""Correct a localization estimate by matching the camera BEV against the HD map.

The camera BEV (AutoBEV probabilities, true ego frame) is correlated with the
map raster over a grid of crop corrections ``q = T_F0<-F`` around the
estimated crop pose. Lane boundaries constrain the lateral offset and the yaw;
crosswalks and stop lines constrain the longitudinal offset. This is semantic
map matching in the spirit of YabLoc and BEV-to-map registration (OrienterNet,
Sarlin et al., CVPR 2023; MapLocNet, arXiv:2407.08561), computed as one batched
correlation on the GPU.

The correlation is a log-likelihood over the search grid; adding the EKF prior
``N(0, covariance)`` and normalizing gives the posterior over ``q``. Its mean
and covariance are the EKF update with the match as observation (a Laplace /
grid Bayes update). An axis the map does not constrain (lane boundaries alone
say nothing about x) keeps a flat likelihood there, so the posterior follows
the prior on that axis. A sample whose best correlation is below
``MIN_PEAK_CORRELATION`` keeps its prior: that is the fallback to the EKF.
"""

from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn.functional as F

from model_components.autobev.geometry import AUTOBEV_GEOMETRY
from navigation.geometry import MapChannel
from navigation.resample import compose_poses, resample_map_context

# AutoBEV class index -> map channel, for the classes that pin the pose.
MATCH_CLASSES: Final = ((1, MapChannel.LANE_BOUNDARY), (3, MapChannel.CROSSWALK), (4, MapChannel.STOP_LINE))
# Search half-width in cells of 0.5 m: +-4 m covers the 3-10 m GNSS jumps once
# the EKF gate has rejected most of them, and the 99th percentile of nominal error.
SEARCH_CELLS: Final = 8
SEARCH_YAW_DEG: Final = (-2.0, -1.0, 0.0, 1.0, 2.0)
# Normalized correlation is in [-1, 1]; this scale turns it into a
# log-likelihood sharp enough that a clear lane match beats a 1 m prior sigma.
LOG_LIKELIHOOD_SCALE: Final = 40.0
MIN_PEAK_CORRELATION: Final = 0.15


def _correlation(prediction: torch.Tensor, padded_map: torch.Tensor) -> torch.Tensor:
    """Normalized cross-correlation of [B, C, H, W] with [B, C, H+2m, W+2m]: [B, 2m+1, 2m+1]."""
    batch, channels, height, width = prediction.shape
    prediction = prediction - prediction.mean(dim=(-2, -1), keepdim=True)
    patches_mean = F.avg_pool2d(padded_map, (height, width), stride=1)
    numerator = F.conv2d(
        padded_map.reshape(1, batch * channels, *padded_map.shape[-2:]),
        prediction.reshape(batch * channels, 1, height, width),
        groups=batch * channels,
    ).reshape(batch, channels, *patches_mean.shape[-2:])
    numerator = numerator - patches_mean * prediction.sum(dim=(-2, -1), keepdim=True)
    map_energy = F.avg_pool2d(padded_map.square(), (height, width), stride=1) - patches_mean.square()
    prediction_energy = prediction.square().sum(dim=(-2, -1), keepdim=True)
    scale = (map_energy.clamp_min(0.0) * height * width * prediction_energy).sqrt().clamp_min(1e-3)
    # Classes absent from both the prediction and the map add nothing.
    return (numerator / scale).sum(dim=1) / channels


def match_pose(
    bev_probability: torch.Tensor,
    map_context: torch.Tensor,
    prior_pose: torch.Tensor,
    prior_covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Posterior crop pose, its covariance and whether the match was used.

    ``bev_probability`` [B, 8, 220, 160] is the AutoBEV class probability,
    ``map_context`` [B, 14, h, w] the navigation raster in its frame,
    ``prior_pose`` [B, 3] the estimated crop pose and ``prior_covariance``
    [B, 3, 3] its EKF covariance (x, y in m, yaw in rad).
    """
    with torch.autocast(bev_probability.device.type, enabled=False):
        return _match_pose(bev_probability.float(), map_context.float(), prior_pose.float(),
                           prior_covariance.float())


def _match_pose(bev_probability, map_context, prior_pose, prior_covariance):
    # Full precision: the correlation sums 35,200 cells per shift.
    device = bev_probability.device
    step = AUTOBEV_GEOMETRY.meters_per_pixel
    prediction = bev_probability[:, [index for index, _ in MATCH_CLASSES]].float()
    channels = [int(channel) for _, channel in MATCH_CLASSES]
    yaws = torch.tensor([math.radians(value) for value in SEARCH_YAW_DEG], device=device)
    scores = []
    for search_yaw in yaws:
        rotation = torch.zeros_like(prior_pose)
        rotation[:, 2] = search_yaw
        padded = resample_map_context(map_context, compose_poses(prior_pose, rotation), AUTOBEV_GEOMETRY,
                                      margin_cells=SEARCH_CELLS)[:, channels].float()
        scores.append(_correlation(prediction, padded))
    score = torch.stack(scores, dim=1)  # [B, yaw, row, col]
    # Output row i reads the map shifted by (SEARCH_CELLS - i) rows: a forward
    # shift of the crop frame. Rows run front to rear, columns left to right.
    offsets = torch.arange(SEARCH_CELLS, -SEARCH_CELLS - 1, -1, device=device, dtype=torch.float32) * step
    shape = (len(yaws), len(offsets), len(offsets))
    shift_x = offsets.view(1, -1, 1).expand(shape)
    shift_y = offsets.view(1, 1, -1).expand(shape)
    yaw = yaws.view(-1, 1, 1).expand(shape)
    # The search read the map at prior * rot(yaw) * shift, which is the crop
    # correction q = (rot(yaw) shift, yaw).
    q = torch.stack([
        torch.cos(yaw) * shift_x - torch.sin(yaw) * shift_y,
        torch.sin(yaw) * shift_x + torch.cos(yaw) * shift_y,
        yaw,
    ], dim=-1)
    flat_q = q.reshape(-1, 3)  # [N, 3] in (x, y, yaw)
    information = torch.linalg.inv(prior_covariance.float())
    log_prior = -0.5 * torch.einsum("ni,bij,nj->bn", flat_q, information, flat_q)
    log_posterior = LOG_LIKELIHOOD_SCALE * score.reshape(score.shape[0], -1) + log_prior
    weight = log_posterior.softmax(dim=-1)
    mean = weight @ flat_q
    centred = flat_q[None] - mean[:, None]
    covariance = torch.einsum("bn,bni,bnj->bij", weight, centred, centred)
    # The grid quantizes q; keep at least a quarter cell (and 0.25 deg) of spread.
    floor = torch.diag(torch.tensor([(step / 4) ** 2, (step / 4) ** 2, math.radians(0.25) ** 2], device=device))
    covariance = covariance + floor
    used = score.flatten(1).amax(dim=-1) >= MIN_PEAK_CORRELATION
    pose = torch.where(used[:, None], compose_poses(prior_pose, mean), prior_pose)
    covariance = torch.where(used[:, None, None], covariance, prior_covariance.float())
    return pose, covariance, used
