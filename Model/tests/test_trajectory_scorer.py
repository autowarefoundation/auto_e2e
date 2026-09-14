"""Smoke test for TrajectoryComplianceScorer.

Runs entirely on CPU with a fake BasePlanner — no trained model,
no KITScenes data, no GPU required. Safe to run locally even when
hardware is constrained.

Usage from repo root:
    pytest Model/tests/test_trajectory_scorer.py -v
or:
    python Model/tests/test_trajectory_scorer.py
"""

import torch
import torch.nn as nn
import numpy as np
import pytest

from Model.model_components.trajectory_planning.trajectory_scorer import (
    ScorerConfig,
    TrajectoryComplianceScorer,
    decode_trajectory_to_xy,
    drivable_area_compliance,
    extract_initial_speed,
    kinematic_comfort_score,
    project_xy_to_bev_pixel,
)
from Model.navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY, MapChannel

NUM_TIMESTEPS = 4
BATCH = 2
# Real geometry (#161), not a guessed shape — see ScorerConfig's own defaults.
BEV_H = DEFAULT_NAVIGATION_GEOMETRY.height_px
BEV_W = DEFAULT_NAVIGATION_GEOMETRY.width_px
NAV_CHANNELS = 14  # Model.navigation.geometry.MAP_CHANNEL_COUNT
DEFAULT_SPEED = 5.0  # m/s, used to build realistic-looking egomotion fixtures


class FakePlanner(nn.Module):
    """Matches BasePlanner's real contract: forward() returns a single
    trajectory tensor, NOT a (trajectory, ego_hidden) tuple. An earlier
    version of this fixture returned a tuple, which let every test here
    pass while the real TrajectoryComplianceScorer.sample_and_score would
    have crashed the moment it wrapped an actual FlowMatchingPlanner —
    the fake didn't match what it was standing in for. See base.py's
    docstring: "forward() always performs inference and returns
    (trajectory) regardless of the underlying decoder."
    """

    def __init__(self, num_timesteps=4, num_signals=2, embed_dim=8):
        super().__init__()
        self.trajectory_dim = num_timesteps * num_signals
        self.embed_dim = embed_dim

    def forward(self, bev_features, visual_history, egomotion_history,
                generator=None, **kwargs):
        B = bev_features.shape[0]
        return torch.randn(B, self.trajectory_dim, generator=generator)


def _egomotion(batch=BATCH, speed=DEFAULT_SPEED):
    """(batch, 256) = 64 history timesteps x [speed, accel, yaw_rate, curvature].
    Only the most recent timestep's speed channel is read by the scorer
    (extract_initial_speed), but the full realistic shape is used here so
    a shape regression in that helper would actually be caught."""
    history = torch.zeros(batch, 64, 4)
    history[:, :, 0] = speed
    return history.reshape(batch, 256)


@pytest.fixture()
def bev_features(): return torch.randn(BATCH, 8, 6, 6)
@pytest.fixture()
def visual_history(): return torch.randn(BATCH, 16)
@pytest.fixture()
def egomotion_history(): return _egomotion()
@pytest.fixture()
def map_context():
    """[B, 14, H, W] semantic navigation raster (#161) — NOT a rendered
    RGB image any more. Only the DRIVABLE_AREA channel is set here; the
    other 13 channels are left zero since nothing under test reads them.
    A generous drivable rectangle is used so the ego origin (which the
    tests below rely on being compliant) is safely inside it — this file
    used to ship a rectangle that was exactly one row short of the ego
    row, an off-by-one that went uncaught for a while (see git history);
    generous margins here are deliberate, not laziness.
    """
    m = torch.zeros(BATCH, NAV_CHANNELS, BEV_H, BEV_W)
    m[:, MapChannel.DRIVABLE_AREA, 50:BEV_H, 50:BEV_W - 50] = 1.0
    return m
@pytest.fixture()
def planner(): return FakePlanner(num_timesteps=NUM_TIMESTEPS)
@pytest.fixture()
def scorer(planner): return TrajectoryComplianceScorer(planner, num_timesteps=NUM_TIMESTEPS)


class TestExtractInitialSpeed:
    def test_reads_last_history_row_speed_channel(self):
        eh = _egomotion(batch=3, speed=7.5)
        speed = extract_initial_speed(eh)
        assert speed.shape == (3,)
        assert torch.allclose(speed, torch.full((3,), 7.5))

    def test_wrong_last_dim_raises(self):
        with pytest.raises(ValueError, match="256"):
            extract_initial_speed(torch.randn(BATCH, 12))


class TestDecodeTrajectoryToXY:
    def test_output_shape(self):
        speed = torch.full((BATCH,), DEFAULT_SPEED)
        xy = decode_trajectory_to_xy(
            torch.zeros(BATCH, NUM_TIMESTEPS * 2), NUM_TIMESTEPS, speed)
        assert xy.shape == (BATCH, NUM_TIMESTEPS, 2)

    def test_zero_curvature_stays_straight(self):
        speed = torch.full((1,), DEFAULT_SPEED)
        xy = decode_trajectory_to_xy(
            torch.zeros(1, NUM_TIMESTEPS * 2), NUM_TIMESTEPS, speed)
        assert torch.allclose(xy[0, :, 1], torch.zeros(NUM_TIMESTEPS), atol=1e-5)
        assert (xy[0, 1:, 0] > xy[0, :-1, 0]).all()

    def test_extreme_deceleration_does_not_crash(self):
        traj = torch.full((1, NUM_TIMESTEPS * 2), -1000.0)
        speed = torch.full((1,), DEFAULT_SPEED)
        xy = decode_trajectory_to_xy(traj, NUM_TIMESTEPS, speed)
        assert not torch.isnan(xy).any() and not torch.isinf(xy).any()

    def test_different_initial_speed_gives_different_xy(self):
        traj = torch.zeros(1, NUM_TIMESTEPS * 2)
        xy_slow = decode_trajectory_to_xy(traj, NUM_TIMESTEPS, torch.full((1,), 1.0))
        xy_fast = decode_trajectory_to_xy(traj, NUM_TIMESTEPS, torch.full((1,), 20.0))
        assert not torch.allclose(xy_slow, xy_fast)

    def test_speed_row_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="initial_speed"):
            decode_trajectory_to_xy(
                torch.zeros(BATCH, NUM_TIMESTEPS * 2), NUM_TIMESTEPS,
                torch.full((BATCH + 1,), DEFAULT_SPEED),
            )


class TestProjectXYToBEVPixel:
    def test_ego_origin_maps_correctly(self):
        """Must match DEFAULT_NAVIGATION_GEOMETRY.ego_to_pixel exactly —
        this is a reimplementation of that method for batched torch use,
        not an independent formula, so cross-check against it directly."""
        cfg = ScorerConfig()
        px = project_xy_to_bev_pixel(torch.zeros(1, 2), cfg)
        expected = DEFAULT_NAVIGATION_GEOMETRY.ego_to_pixel(np.zeros((1, 2)))
        assert px[0, 0].item() == round(expected[0, 0])
        assert px[0, 1].item() == round(expected[0, 1])
        assert px[0, 0].item() == round(cfg.ego_row)
        assert px[0, 1].item() == round(cfg.ego_col)

    def test_forward_motion_reduces_row(self):
        cfg = ScorerConfig()
        px = project_xy_to_bev_pixel(torch.tensor([[10.0, 0.0]]), cfg)
        assert px[0, 0].item() < cfg.ego_row

    def test_matches_real_geometry_object_on_random_points(self):
        """Broader cross-check than the origin alone — random points
        inside the raster bounds must match
        NavigationRasterGeometry.ego_to_pixel to the pixel."""
        cfg = ScorerConfig()
        pts = np.array([[10.0, -5.0], [-20.0, 30.0], [0.0, 50.0]])
        expected = DEFAULT_NAVIGATION_GEOMETRY.ego_to_pixel(pts)
        got = project_xy_to_bev_pixel(torch.tensor(pts, dtype=torch.float64), cfg)
        for i in range(len(pts)):
            assert got[i, 0].item() == round(expected[i, 0])
            assert got[i, 1].item() == round(expected[i, 1])


class TestDrivableAreaCompliance:
    def test_oob_reduces_compliance(self, map_context):
        cfg = ScorerConfig()
        traj = torch.zeros(BATCH, 1, 2)
        traj[0, 0] = torch.tensor([10000.0, 10000.0])
        dac = drivable_area_compliance(traj, map_context, cfg)
        assert dac[0].item() < 1.0
        assert dac[1].item() == 1.0

    def test_range_zero_to_one(self, map_context):
        cfg = ScorerConfig()
        traj = torch.randn(BATCH, NUM_TIMESTEPS, 2) * 10
        dac = drivable_area_compliance(traj, map_context, cfg)
        assert ((dac >= 0.0) & (dac <= 1.0)).all()

    def test_reads_binary_channel_not_pixel_color(self, map_context):
        """The whole point of the redesign: compliance is a channel
        lookup, not a colour-distance heuristic. A trajectory sitting on
        a pixel where DRIVABLE_AREA=0 must score non-compliant even
        though every other channel (all zero in this fixture) would have
        matched an old "close to black" heuristic."""
        cfg = ScorerConfig()
        traj = torch.zeros(1, 1, 2)
        traj[0, 0] = torch.tensor([-200.0, 0.0])  # well outside the fixture's drivable rect, but in-bounds
        dac = drivable_area_compliance(traj, map_context[:1], cfg)
        assert dac[0].item() == 0.0


class TestKinematicComfortScore:
    def test_no_violations_scores_one(self):
        cfg = ScorerConfig(max_comfortable_accel=100.0, max_comfortable_lateral_accel=100.0)
        speed = torch.full((BATCH,), DEFAULT_SPEED)
        score = kinematic_comfort_score(
            torch.zeros(BATCH, NUM_TIMESTEPS * 2), NUM_TIMESTEPS, speed, cfg)
        assert torch.allclose(score, torch.ones(BATCH))

    def test_all_violations_scores_zero(self):
        cfg = ScorerConfig(max_comfortable_accel=0.0, max_comfortable_lateral_accel=0.0)
        speed = torch.full((BATCH,), DEFAULT_SPEED)
        score = kinematic_comfort_score(
            torch.ones(BATCH, NUM_TIMESTEPS * 2), NUM_TIMESTEPS, speed, cfg)
        assert torch.allclose(score, torch.zeros(BATCH))


class TestTrajectoryComplianceScorer:
    def test_output_shapes(self, scorer, bev_features, visual_history, egomotion_history, map_context):
        traj, scores = scorer.sample_and_score(
            bev_features, visual_history, egomotion_history, map_context, num_samples=5, seed=42)
        assert traj.shape == (BATCH, NUM_TIMESTEPS * 2)
        assert scores.shape == (BATCH, 5)

    def test_mean_selection(self, planner, bev_features, visual_history, egomotion_history, map_context):
        cfg = ScorerConfig(selection="mean")
        s = TrajectoryComplianceScorer(planner, NUM_TIMESTEPS, config=cfg)
        traj, _ = s.sample_and_score(bev_features, visual_history, egomotion_history, map_context, num_samples=4)
        assert traj.shape == (BATCH, NUM_TIMESTEPS * 2)

    def test_invalid_selection_raises(self, planner, bev_features, visual_history, egomotion_history, map_context):
        cfg = ScorerConfig(selection="bogus")
        s = TrajectoryComplianceScorer(planner, NUM_TIMESTEPS, config=cfg)
        with pytest.raises(ValueError, match="config.selection"):
            s.sample_and_score(bev_features, visual_history, egomotion_history, map_context, num_samples=3)

    def test_seed_reproducibility(self, scorer, bev_features, visual_history, egomotion_history, map_context):
        t1, _ = scorer.sample_and_score(bev_features, visual_history, egomotion_history, map_context, num_samples=4, seed=0)
        t2, _ = scorer.sample_and_score(bev_features, visual_history, egomotion_history, map_context, num_samples=4, seed=0)
        assert torch.allclose(t1, t2)

    def test_different_seeds_differ(self, scorer, bev_features, visual_history, egomotion_history, map_context):
        t1, _ = scorer.sample_and_score(bev_features, visual_history, egomotion_history, map_context, num_samples=4, seed=0)
        t2, _ = scorer.sample_and_score(bev_features, visual_history, egomotion_history, map_context, num_samples=4, seed=99)
        assert not torch.allclose(t1, t2)

    def test_scores_vary_across_samples(self, planner, bev_features, visual_history, egomotion_history, map_context):
        cfg = ScorerConfig(dac_weight=1.0, comfort_weight=1.0)
        s = TrajectoryComplianceScorer(planner, NUM_TIMESTEPS, config=cfg)
        _, scores = s.sample_and_score(bev_features, visual_history, egomotion_history, map_context, num_samples=8, seed=7)
        assert scores.std(dim=1).sum() > 0

    def test_different_initial_speeds_change_selection(self, planner, bev_features, visual_history, map_context):
        """Regression guard for the original bug this file's fixtures used
        to hide: a scorer wired to a fixed initial_speed can't distinguish
        a scene where the ego starts at 1 m/s from one where it starts at
        25 m/s. With the real per-row speed threaded through, the decoded
        (and thus scored) geometry must differ between the two."""
        cfg = ScorerConfig(dac_weight=1.0, comfort_weight=1.0)
        s = TrajectoryComplianceScorer(planner, NUM_TIMESTEPS, config=cfg)
        eh_slow = _egomotion(speed=1.0)
        eh_fast = _egomotion(speed=25.0)
        torch.manual_seed(0)
        traj_slow, _ = s.sample_and_score(bev_features, visual_history, eh_slow, map_context, num_samples=4, seed=3)
        torch.manual_seed(0)
        traj_fast, _ = s.sample_and_score(bev_features, visual_history, eh_fast, map_context, num_samples=4, seed=3)
        assert not torch.allclose(traj_slow, traj_fast)


if __name__ == "__main__":
    print("Running smoke test (CPU, no GPU required)...")
    bev = torch.randn(BATCH, 8, 6, 6)
    vh = torch.randn(BATCH, 16)
    eh = _egomotion()
    mp = torch.zeros(BATCH, NAV_CHANNELS, BEV_H, BEV_W)
    mp[:, MapChannel.DRIVABLE_AREA, 50:BEV_H, 50:BEV_W - 50] = 1.0
    sc = TrajectoryComplianceScorer(FakePlanner(NUM_TIMESTEPS), NUM_TIMESTEPS)
    traj, scores = sc.sample_and_score(bev, vh, eh, mp, num_samples=6, seed=42)
    print(f"  trajectory:  {tuple(traj.shape)}")
    print(f"  scores:      {tuple(scores.shape)}")
    print(f"  score range: {scores.min().item():.3f} - {scores.max().item():.3f}")
    print("PASSED.")
