"""Unit tests for SafetyReward and the AutoE2E reward framework.

Covers:
- Off-road penalty logic: inside, outside, partially off-road, multi-polygon,
  map version caching/invalidation, and coordinate transformations.
- Time-to-Collision (TTC) & dynamic agent collision logic:
  - No collision / distant agents / parallel lane traffic.
  - Immediate collision at t < 2.0s (constant penalty -5.0).
  - Delayed collision at t >= 2.0s (time-decayed penalty -5.0 / t).
  - Collision duration spanning multiple timesteps.
  - Linear kinematics projection (position + velocity * t).
  - Agent yaw orientation and custom bounding box sizes.
  - Single penalty per timestep with multiple overlapping agents.
- Combined off-road + TTC penalties.
- Input validation, torch tensor formats, and edge cases.
- RewardRegistry and auxiliary reward class interfaces.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

_ALPASIM_DRIVER_DIR = Path(__file__).resolve().parents[1] / "plugins" / "alpasim_driver"
if str(_ALPASIM_DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(_ALPASIM_DRIVER_DIR))

from alpasim_autoe2e.rewards import (  # noqa: E402
    GroundTruthDeviationReward,
    OffRoadReward,
    RewardManager,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Mock Primitives
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MockPolygonPrimitive:
    """Mock polygon primitive adhering to navigation polygon contract."""

    primitive_id: str
    points_enu_m: np.ndarray


@dataclasses.dataclass
class MockNavigationMap:
    """Mock navigation map object holding drivable area polygon primitives."""

    map_version: str
    drivable_polygons: List[MockPolygonPrimitive]


@pytest.fixture
def drivable_corridor_map() -> MockNavigationMap:
    """Drivable corridor along the x-axis: x in [-100, 100], y in [-5, 5]."""
    # Counter-clockwise rectangle: [x, y, z]
    pts = np.array(
        [
            [-100.0, -5.0, 0.0],
            [100.0, -5.0, 0.0],
            [100.0, 5.0, 0.0],
            [-100.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )
    poly = MockPolygonPrimitive(primitive_id="lane_0", points_enu_m=pts)
    return MockNavigationMap(map_version="v1.0", drivable_polygons=[poly])


@pytest.fixture
def multi_polygon_map() -> MockNavigationMap:
    """Map with two disjoint drivable polygons separated by a 10m off-road gap.

    - Polygon A: x in [-50, -5], y in [-5, 5]
    - Gap (off-road): x in (-5, 5)
    - Polygon B: x in [5, 50], y in [-5, 5]
    """
    poly_a = MockPolygonPrimitive(
        primitive_id="poly_a",
        points_enu_m=np.array(
            [[-50.0, -5.0], [-5.0, -5.0], [-5.0, 5.0], [-50.0, 5.0]],
            dtype=np.float64,
        ),
    )
    poly_b = MockPolygonPrimitive(
        primitive_id="poly_b",
        points_enu_m=np.array(
            [[5.0, -5.0], [50.0, -5.0], [50.0, 5.0], [5.0, 5.0]],
            dtype=np.float64,
        ),
    )
    return MockNavigationMap(map_version="v1.0", drivable_polygons=[poly_a, poly_b])


@pytest.fixture
def straight_trajectory_10_steps() -> tuple[torch.Tensor, torch.Tensor]:
    """10-step straight trajectory along local x-axis from x=1 to x=10 with heading 0."""
    traj = torch.stack(
        [
            torch.arange(1.0, 11.0, dtype=torch.float32),
            torch.zeros(10, dtype=torch.float32),
        ],
        dim=-1,
    )
    headings = torch.zeros(10, dtype=torch.float32)
    return traj, headings


# ---------------------------------------------------------------------------
# 1. Off-Road Penalty Tests
# ---------------------------------------------------------------------------


class TestSafetyRewardOffRoad:
    """Tests covering off-road detection and polygon intersection logic."""

    def test_trajectory_fully_inside_drivable_area(
        self,
        drivable_corridor_map: MockNavigationMap,
        straight_trajectory_10_steps: tuple[torch.Tensor, torch.Tensor],
    ):
        """When all waypoints lie inside drivable polygons, off-road penalty is 0.0."""
        traj, _ = straight_trajectory_10_steps
        reward_fn = OffRoadReward()

        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            navigation_map=drivable_corridor_map,
        )

        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_trajectory_fully_outside_drivable_area(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """When all 10 waypoints lie outside drivable polygons (y=20m, corridor y in [-5, 5]),
        each step is penalized -1.0, yielding an average penalty of -1.0.
        """
        # 10 steps along y=20.0 (corridor only extends to y=5.0)
        traj = torch.stack(
            [
                torch.arange(1.0, 11.0, dtype=torch.float32),
                torch.full((10,), 20.0, dtype=torch.float32),
            ],
            dim=-1,
        )
        reward_fn = OffRoadReward()

        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            navigation_map=drivable_corridor_map,
        )

        # 10 steps * (-1.0) / 10 steps = -1.0
        assert reward == pytest.approx(-1.0, abs=1e-6)

    def test_trajectory_partially_outside_drivable_area(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """Trajectory with 6 points inside and 4 points outside the drivable area.
        Average off-road penalty should equal -4.0 / 10 = -0.4.
        """
        # First 6 points inside corridor (y=0.0), last 4 points off-road (y=15.0)
        x_pts = torch.arange(1.0, 11.0, dtype=torch.float32)
        y_pts = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 15.0, 15.0, 15.0], dtype=torch.float32
        )
        traj = torch.stack([x_pts, y_pts], dim=-1)

        reward_fn = OffRoadReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            navigation_map=drivable_corridor_map,
        )

        assert reward == pytest.approx(-0.4, abs=1e-6)

    def test_off_road_with_ego_pose_rotation_and_translation(self):
        """Verify ego_pose (x, y, yaw) transforms ego-frame trajectory to map frame.

        Ego is at (0, 0, pi/2) facing North (+y).
        Ego-frame trajectory moving forward along local x: [1, 2, 3, 4, 5]
        transforms to global y: [1, 2, 3, 4, 5], global x: [0, 0, 0, 0, 0].
        """
        # Vertical drivable corridor along global y-axis: x in [-2, 2], y in [-10, 10]
        corridor_pts = np.array(
            [[-2.0, -10.0], [2.0, -10.0], [2.0, 10.0], [-2.0, 10.0]], dtype=np.float64
        )
        nav_map = MockNavigationMap(
            map_version="v1.0",
            drivable_polygons=[MockPolygonPrimitive("north_lane", corridor_pts)],
        )

        # Local trajectory going forward in ego x
        traj_local = torch.stack([torch.arange(1.0, 6.0), torch.zeros(5)], dim=-1)

        reward_fn = OffRoadReward()

        # Case A: Facing North (yaw = pi/2), local forward moves into global y -> inside corridor
        reward_inside = reward_fn.compute(
            ego_pose=(0.0, 0.0, np.pi / 2),
            trajectory_xy=traj_local,
            navigation_map=nav_map,
        )
        assert reward_inside == pytest.approx(0.0, abs=1e-6)

        # Case B: Facing East (yaw = 0), local forward moves into global x -> outside corridor for x > 2.0
        # For points x=1,2,3,4,5: x=1 is inside (<=2), x=2 is on boundary/outside, x=3,4,5 are outside
        reward_outside = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj_local,
            navigation_map=nav_map,
        )
        assert reward_outside < 0.0

    def test_multiple_drivable_polygons_and_gap(
        self, multi_polygon_map: MockNavigationMap
    ):
        """Trajectory traversing from polygon A, across an off-road gap, into polygon B.

        Poly A: x in [-50, -5]
        Gap: x in (-5, 5) -> off-road
        Poly B: x in [5, 50]
        """
        # 5 points at x = [-10.0, -7.0, 0.0, 7.0, 10.0], y = 0.0
        # -10 and -7 are in Poly A
        # 0 is in the gap (off-road -> penalty -1.0)
        # 7 and 10 are in Poly B
        traj = torch.tensor(
            [[-10.0, 0.0], [-7.0, 0.0], [0.0, 0.0], [7.0, 0.0], [10.0, 0.0]],
            dtype=torch.float32,
        )

        reward_fn = OffRoadReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            navigation_map=multi_polygon_map,
        )

        # 1 step off-road out of 5 steps = -1.0 / 5 = -0.2
        assert reward == pytest.approx(-0.2, abs=1e-6)

    def test_2d_and_3d_polygon_coordinates(self):
        """Navigation maps with 2D [N, 2] or 3D [N, 3] points_enu_m are handled correctly."""
        pts_3d = np.array(
            [[-20.0, -5.0, 1.5], [20.0, -5.0, 1.5], [20.0, 5.0, 2.0], [-20.0, 5.0, 2.0]]
        )
        nav_map = MockNavigationMap(
            map_version="v3d",
            drivable_polygons=[MockPolygonPrimitive("poly_3d", pts_3d)],
        )
        traj = torch.tensor([[0.0, 0.0], [5.0, 0.0]], dtype=torch.float32)

        reward_fn = OffRoadReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            navigation_map=nav_map,
        )
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_polygons_with_fewer_than_3_points_ignored(self):
        """Polygons with fewer than 3 vertices are skipped, while valid ones are indexed."""
        invalid_poly = MockPolygonPrimitive(
            "line_primitive", np.array([[0.0, 0.0], [1.0, 1.0]])
        )
        valid_poly = MockPolygonPrimitive(
            "triangle_primitive",
            np.array([[-10.0, -10.0], [10.0, -10.0], [0.0, 10.0]]),
        )
        nav_map = MockNavigationMap(
            map_version="v_mixed",
            drivable_polygons=[invalid_poly, valid_poly],
        )

        reward_fn = OffRoadReward()
        # Point (0, 0) is strictly inside triangle (-10,-10), (10,-10), (0,10)
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=torch.tensor([[0.0, 0.0]]),
            navigation_map=nav_map,
        )
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_spatial_index_caching_and_invalidation(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """STRtree is cached across compute calls with the same map_version and rebuilt on version change."""
        reward_fn = OffRoadReward()
        traj = torch.tensor([[1.0, 0.0]])

        assert reward_fn._drivable_tree is None
        assert reward_fn._cached_map_version is None

        # First compute builds the STRtree
        reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            navigation_map=drivable_corridor_map,
        )
        tree_v1 = reward_fn._drivable_tree
        assert tree_v1 is not None
        assert reward_fn._cached_map_version == "v1.0"

        # Second compute with same map_version reuses the exact same tree instance
        reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            navigation_map=drivable_corridor_map,
        )
        assert reward_fn._drivable_tree is tree_v1

        # Third compute with new map_version rebuilds the tree
        updated_map = MockNavigationMap(
            map_version="v2.0",
            drivable_polygons=drivable_corridor_map.drivable_polygons,
        )
        reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            navigation_map=updated_map,
        )
        assert reward_fn._cached_map_version == "v2.0"
        assert reward_fn._drivable_tree is not tree_v1


# ---------------------------------------------------------------------------
# 2. Ground-Truth Deviation Reward Tests
# ---------------------------------------------------------------------------


class TestGroundTruthDeviationReward:
    """Tests covering trajectory tracking against ground truth and 3DGS boundary gating."""

    def test_exact_match_zero_penalty(self):
        """When predicted trajectory perfectly matches ground truth, tracking penalty is 0.0."""
        traj = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
        reward_fn = GroundTruthDeviationReward(ade_weight=1.0, fde_weight=0.5)

        reward = reward_fn.compute(
            trajectory_xy=traj,
            gt_trajectory=traj,
        )
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_constant_lateral_offset_penalty(self):
        """Uniform 1.0m lateral offset yields ADE=1.0, FDE=1.0, and proportional negative penalty."""
        traj_gt = np.zeros((10, 2), dtype=np.float32)
        traj_gt[:, 0] = np.linspace(1.0, 10.0, 10)  # x in [1, 10], y = 0

        traj_pred = traj_gt.copy()
        traj_pred[:, 1] = 1.0  # 1.0m lateral offset

        reward_fn = GroundTruthDeviationReward(ade_weight=1.0, fde_weight=0.5)
        # ADE = 1.0, FDE = 1.0 -> penalty = -(1.0 * 1.0 + 0.5 * 1.0) = -1.5
        reward = reward_fn.compute(
            trajectory_xy=traj_pred,
            gt_trajectory=traj_gt,
        )
        assert reward == pytest.approx(-1.5, abs=1e-6)

    def test_fde_weighting(self):
        """Diverging trajectory with larger final displacement reflects in FDE penalty."""
        gt = np.zeros((4, 2), dtype=np.float32)
        pred = np.array(
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=np.float32
        )

        # ADE = (0 + 1 + 2 + 3) / 4 = 1.5, FDE = 3.0
        # penalty = -(2.0 * 1.5 + 1.0 * 3.0) = -6.0
        reward_fn = GroundTruthDeviationReward(ade_weight=2.0, fde_weight=1.0)
        reward = reward_fn.compute(
            trajectory_xy=pred,
            gt_trajectory=gt,
        )
        assert reward == pytest.approx(-6.0, abs=1e-6)

    def test_3dgs_boundary_threshold_exceeded(self):
        """When max deviation exceeds 3.0m, applies terminal penalty."""
        gt = np.zeros((5, 2), dtype=np.float32)
        pred = np.zeros((5, 2), dtype=np.float32)
        pred[-1, 1] = 3.5  # > 3.0m threshold

        reward_fn = GroundTruthDeviationReward(
            ade_weight=1.0,
            fde_weight=0.0,
            max_deviation_threshold=3.0,
            terminal_penalty=10.0,
        )
        # ADE = 3.5 / 5 = 0.7. Terminal penalty = -10.0. Total = -10.7
        reward = reward_fn.compute(
            trajectory_xy=pred,
            gt_trajectory=gt,
        )
        assert reward == pytest.approx(-10.7, abs=1e-6)

    def test_boundary_not_exceeded_within_threshold(self):
        """When max deviation is within 3.0m threshold, no terminal penalty is applied."""
        gt = np.zeros((5, 2), dtype=np.float32)
        pred = np.zeros((5, 2), dtype=np.float32)
        pred[-1, 1] = 2.9  # <= 3.0m

        reward_fn = GroundTruthDeviationReward(
            ade_weight=1.0,
            fde_weight=0.0,
            max_deviation_threshold=3.0,
            terminal_penalty=10.0,
        )
        reward = reward_fn.compute(
            trajectory_xy=pred,
            gt_trajectory=gt,
        )
        # ADE = 2.9 / 5 = 0.58. No terminal penalty.
        assert reward == pytest.approx(-0.58, abs=1e-6)

    def test_truncation_signal_on_boundary_violation(self):
        """When max deviation exceeds threshold, truncated=True is returned."""
        gt = np.zeros((5, 2), dtype=np.float32)
        pred = np.zeros((5, 2), dtype=np.float32)
        pred[-1, 1] = 3.5  # > 3.0m threshold

        reward_fn = GroundTruthDeviationReward(
            ade_weight=1.0,
            fde_weight=0.0,
            max_deviation_threshold=3.0,
            terminal_penalty=10.0,
        )
        assert reward_fn.is_out_of_bounds(pred, gt) is True

        reward, truncated = reward_fn.compute_with_truncation(pred, gt)
        assert truncated is True
        assert reward == pytest.approx(-10.7, abs=1e-6)

    def test_truncation_signal_within_boundary(self):
        """When max deviation is within threshold, truncated=False is returned."""
        gt = np.zeros((5, 2), dtype=np.float32)
        pred = np.zeros((5, 2), dtype=np.float32)
        pred[-1, 1] = 2.9  # <= 3.0m

        reward_fn = GroundTruthDeviationReward(
            ade_weight=1.0,
            fde_weight=0.0,
            max_deviation_threshold=3.0,
            terminal_penalty=10.0,
        )
        assert reward_fn.is_out_of_bounds(pred, gt) is False

        reward, truncated = reward_fn.compute_with_truncation(pred, gt)
        assert truncated is False
        assert reward == pytest.approx(-0.58, abs=1e-6)

    def test_torch_tensor_and_numpy_parity(self):
        """Parity between PyTorch Tensors (with grad) and NumPy arrays."""
        gt_np = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]], dtype=np.float32)
        pred_np = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)

        gt_torch = torch.tensor(gt_np, requires_grad=False)
        pred_torch = torch.tensor(pred_np, requires_grad=True)

        reward_fn = GroundTruthDeviationReward()
        r_np = reward_fn.compute(trajectory_xy=pred_np, gt_trajectory=gt_np)
        r_torch = reward_fn.compute(
            trajectory_xy=pred_torch.detach().numpy(),
            gt_trajectory=gt_torch.numpy(),
        )

        assert isinstance(r_torch, float)
        assert r_torch == pytest.approx(r_np, abs=1e-6)

    def test_missing_gt_trajectory_raises_error(self):
        """Raises TypeError when gt_trajectory is not provided."""
        reward_fn = GroundTruthDeviationReward()
        with pytest.raises(TypeError):
            reward_fn.compute(trajectory_xy=np.zeros((5, 2)))  # type: ignore

    def test_empty_trajectory_returns_zero(self):
        """Empty trajectory returns 0.0 cleanly without exceptions."""
        reward_fn = GroundTruthDeviationReward()
        assert (
            reward_fn.compute(
                trajectory_xy=np.zeros((0, 2)), gt_trajectory=np.zeros((0, 2))
            )
            == 0.0
        )

    def test_length_mismatch_truncation(self):
        """Unequal trajectory lengths are aligned to the shorter prefix."""
        gt = np.zeros((10, 2), dtype=np.float32)
        pred = np.zeros((5, 2), dtype=np.float32)

        reward_fn = GroundTruthDeviationReward()
        reward = reward_fn.compute(trajectory_xy=pred, gt_trajectory=gt)
        assert reward == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Off-Road Edge Cases & Input Validation
# ---------------------------------------------------------------------------


class TestSafetyRewardCombinedAndEdgeCases:
    """Tests for tensor types, missing parameters, and edge conditions."""

    def test_torch_tensor_with_grad_and_numpy_inputs(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """Handles torch.Tensor (with requires_grad), numpy.ndarray, and list inputs seamlessly."""
        reward_fn = OffRoadReward()

        # 1. PyTorch Tensor
        traj_torch = torch.tensor([[1.0, 0.0], [2.0, 0.0]], requires_grad=True)

        r1 = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj_torch.detach().numpy(),
            navigation_map=drivable_corridor_map,
        )
        assert isinstance(r1, float)
        assert r1 == pytest.approx(0.0, abs=1e-6)

        # 2. Numpy ndarray
        traj_np = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        r2 = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj_np,
            navigation_map=drivable_corridor_map,
        )
        assert r2 == pytest.approx(0.0, abs=1e-6)

        # 3. Python lists
        r3 = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=[[1.0, 0.0], [2.0, 0.0]],
            navigation_map=drivable_corridor_map,
        )
        assert r3 == pytest.approx(0.0, abs=1e-6)

    def test_empty_trajectory_returns_zero(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """Zero-step trajectory returns 0.0 scalar without division by zero errors."""
        reward_fn = OffRoadReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=torch.zeros((0, 2)),
            navigation_map=drivable_corridor_map,
        )
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_missing_required_kwargs_raises_error(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """Missing ego_pose, trajectory_xy, or navigation_map raises TypeError."""
        reward_fn = OffRoadReward()
        traj = torch.tensor([[1.0, 0.0]])

        with pytest.raises(TypeError):
            reward_fn.compute(
                trajectory_xy=traj,
                navigation_map=drivable_corridor_map,
            )  # type: ignore

        with pytest.raises(TypeError):
            reward_fn.compute(
                ego_pose=(0.0, 0.0, 0.0),
                navigation_map=drivable_corridor_map,
            )  # type: ignore

        with pytest.raises(TypeError):
            reward_fn.compute(
                ego_pose=(0.0, 0.0, 0.0),
                trajectory_xy=traj,
            )  # type: ignore

    def test_empty_drivable_area_raises_value_error(self):
        """Navigation map with empty drivable polygons raises ValueError."""
        empty_map = MockNavigationMap(map_version="empty", drivable_polygons=[])
        reward_fn = OffRoadReward()

        with pytest.raises(ValueError, match="No drivable area defined"):
            reward_fn.compute(
                ego_pose=(0.0, 0.0, 0.0),
                trajectory_xy=torch.tensor([[1.0, 0.0]]),
                navigation_map=empty_map,
            )


# ---------------------------------------------------------------------------
# 4. RewardManager & Auxiliary Rewards
# ---------------------------------------------------------------------------


class TestRewardManagerAndFramework:
    """Tests covering RewardManager, weight configurations, and active reward interfaces."""

    def test_reward_manager_initialization_and_computation(
        self,
        drivable_corridor_map: MockNavigationMap,
        straight_trajectory_10_steps: tuple[torch.Tensor, torch.Tensor],
    ):
        """RewardManager initializes active rewards based on weights and computes total reward."""
        traj, _ = straight_trajectory_10_steps
        traj_np = traj.cpu().numpy()

        manager = RewardManager(w_gt_dev=2.0, w_offroad=1.0)

        assert isinstance(manager.gt_reward, GroundTruthDeviationReward)
        assert isinstance(manager.offroad_reward, OffRoadReward)

        total_reward = manager.compute(
            trajectory_xy=traj_np,
            gt_trajectory=traj_np,
            ego_pose=(0.0, 0.0, 0.0),
            navigation_map=drivable_corridor_map,
        )

        assert total_reward == pytest.approx(0.0, abs=1e-6)

    def test_reward_manager_custom_rewards_and_validation(self):
        """RewardManager accepts custom rewards and custom weights."""
        gt_rew = GroundTruthDeviationReward(ade_weight=2.0)
        offroad_rew = OffRoadReward()
        manager = RewardManager(
            w_gt_dev=1.5, w_offroad=0.5, gt_reward=gt_rew, offroad_reward=offroad_rew
        )
        assert manager.gt_reward is gt_rew
        assert manager.offroad_reward is offroad_rew
        assert manager.w_gt_dev == 1.5
        assert manager.w_offroad == 0.5

    def test_reward_manager_weight_scaling(
        self,
        drivable_corridor_map: MockNavigationMap,
    ):
        """Total reward scales linearly according to configured component weights."""
        traj = np.zeros((4, 2), dtype=np.float32)
        gt = np.ones(
            (4, 2), dtype=np.float32
        )  # error = sqrt(1+1) = sqrt(2) approx 1.4142

        # w_offroad=0.0 to focus purely on gt deviation scaling
        manager = RewardManager(w_gt_dev=3.0, w_offroad=0.0)
        total = manager.compute(
            trajectory_xy=traj,
            gt_trajectory=gt,
            ego_pose=(0.0, 0.0, 0.0),
            navigation_map=drivable_corridor_map,
        )
        expected_penalty = -(1.0 * np.sqrt(2) + 0.5 * np.sqrt(2))
        assert total == pytest.approx(3.0 * expected_penalty, abs=1e-5)

    def test_reward_manager_truncation_signal(
        self,
        drivable_corridor_map: MockNavigationMap,
    ):
        """RewardManager.compute_with_truncation returns truncated=True on 3DGS boundary breach."""
        gt = np.zeros((5, 2), dtype=np.float32)
        pred = np.zeros((5, 2), dtype=np.float32)
        pred[-1, 1] = 3.5

        manager = RewardManager(w_gt_dev=1.0, w_offroad=0.0)
        total_reward, truncated = manager.compute_with_truncation(
            trajectory_xy=pred,
            gt_trajectory=gt,
            ego_pose=(0.0, 0.0, 0.0),
            navigation_map=drivable_corridor_map,
        )
        assert truncated is True
        assert total_reward == pytest.approx(-12.45, abs=1e-6)
