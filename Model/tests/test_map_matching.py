import torch

from model_components.autobev.geometry import AUTOBEV_GEOMETRY
from navigation.geometry import MAP_CHANNEL_COUNT, MapChannel
from navigation.map_matching import match_pose
from navigation.resample import resample_map_context

LANE, CROSSWALK = 1, 3


def _map() -> torch.Tensor:
    raster = torch.zeros(1, MAP_CHANNEL_COUNT, 450, 300)
    # Lane boundaries along x at y = +-1.8 m (columns 145 and 154 at 0.4 m) and a crosswalk band at x = 25 m.
    raster[0, MapChannel.LANE_BOUNDARY, :, 144:146] = 1.0
    raster[0, MapChannel.LANE_BOUNDARY, :, 153:155] = 1.0
    raster[0, MapChannel.CROSSWALK, 236:242, 120:180] = 1.0
    return raster


def _prediction(raster: torch.Tensor, true_pose: torch.Tensor) -> torch.Tensor:
    """What a perfect camera BEV shows: the map seen from the true ego frame."""
    truth = resample_map_context(raster, true_pose, AUTOBEV_GEOMETRY)
    probability = torch.zeros(1, 8, 220, 160)
    probability[:, LANE] = truth[:, MapChannel.LANE_BOUNDARY]
    probability[:, CROSSWALK] = truth[:, MapChannel.CROSSWALK]
    return probability


def test_match_recovers_lateral_and_longitudinal_offset():
    raster = _map()
    prediction = _prediction(raster, torch.zeros(1, 3))
    prior = torch.tensor([[1.5, -1.0, 0.0]])  # the EKF believes the ego 1.5 m ahead and 1 m right
    covariance = torch.diag(torch.tensor([4.0, 4.0, 1e-3]))[None]
    pose, posterior, used = match_pose(prediction, raster, prior, covariance)
    assert bool(used[0])
    assert pose[0, :2].abs().max() < 0.5
    assert posterior[0, 0, 0] < covariance[0, 0, 0] and posterior[0, 1, 1] < covariance[0, 1, 1]


def test_lane_boundaries_alone_leave_longitudinal_offset_to_the_prior():
    raster = _map()
    raster[0, MapChannel.CROSSWALK] = 0.0
    prediction = _prediction(raster, torch.zeros(1, 3))
    prior = torch.tensor([[2.0, 1.0, 0.0]])
    covariance = torch.diag(torch.tensor([1.0, 4.0, 1e-3]))[None]
    pose, _, used = match_pose(prediction, raster, prior, covariance)
    assert bool(used[0])
    assert abs(float(pose[0, 1])) < 0.5  # lateral corrected
    assert float(pose[0, 0]) > 1.0  # longitudinal kept near the prior


def test_no_map_structure_falls_back_to_prior():
    raster = torch.zeros(1, MAP_CHANNEL_COUNT, 450, 300)
    prediction = torch.zeros(1, 8, 220, 160)
    prior = torch.tensor([[0.7, -0.3, 0.01]])
    covariance = torch.eye(3)[None]
    pose, posterior, used = match_pose(prediction, raster, prior, covariance)
    assert not bool(used[0])
    assert torch.equal(pose, prior) and torch.equal(posterior, covariance)
