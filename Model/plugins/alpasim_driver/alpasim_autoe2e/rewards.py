from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree


class GroundTruthDeviationReward:
    """Evaluates trajectory tracking deviation against expert ground-truth demonstration.

    Computes displacement errors (ADE, FDE) between predicted trajectory and ground truth,
    applying a continuous tracking penalty and enforcing a hard threshold for 3DGS
    visual degradation (e.g. max deviation > 3.0m).
    """

    def __init__(
        self,
        ade_weight: float = 1.0,
        fde_weight: float = 0.5,
        max_deviation_threshold: float = 3.0,
        terminal_penalty: float = 10.0,
    ) -> None:
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.max_deviation_threshold = max_deviation_threshold
        self.terminal_penalty = terminal_penalty

    def is_out_of_bounds(
        self,
        trajectory_xy: np.ndarray,
        gt_trajectory: np.ndarray,
    ) -> bool:
        """Check whether max displacement between prediction and ground truth exceeds 3DGS threshold."""
        if len(trajectory_xy) == 0 or len(gt_trajectory) == 0:
            return False
        n = min(len(trajectory_xy), len(gt_trajectory))
        pred_coords = trajectory_xy[:n, :2]
        gt_coords = gt_trajectory[:n, :2]
        distances = np.linalg.norm(pred_coords - gt_coords, axis=-1)
        return bool(np.max(distances) > self.max_deviation_threshold)

    def compute_with_truncation(
        self,
        trajectory_xy: np.ndarray,
        gt_trajectory: np.ndarray,
    ) -> tuple[float, bool]:
        """Compute deviation reward and return episode truncation flag for 3DGS boundary violation."""
        if len(trajectory_xy) == 0 or len(gt_trajectory) == 0:
            return 0.0, False

        n = min(len(trajectory_xy), len(gt_trajectory))
        pred_coords = trajectory_xy[:n, :2]
        gt_coords = gt_trajectory[:n, :2]

        diffs = pred_coords - gt_coords
        distances = np.linalg.norm(diffs, axis=-1)

        ade = float(np.mean(distances))
        fde = float(distances[-1])
        max_dev = float(np.max(distances))

        tracking_penalty = -(self.ade_weight * ade + self.fde_weight * fde)
        truncated = max_dev > self.max_deviation_threshold
        bound_penalty = -self.terminal_penalty if truncated else 0.0

        return tracking_penalty + bound_penalty, truncated

    def compute(
        self,
        trajectory_xy: np.ndarray,
        gt_trajectory: np.ndarray,
    ) -> float:
        reward, _ = self.compute_with_truncation(trajectory_xy, gt_trajectory)
        return reward


class OffRoadReward:
    """Handcrafted penalty for off-road driving violations (R_offroad)."""

    def __init__(self) -> None:
        self._cached_map_version = None
        self._drivable_tree = None

    def compute(
        self,
        ego_pose: tuple[float, float, float],
        trajectory_xy: np.ndarray,
        navigation_map: Any,
    ) -> float:
        # 1. Build or retrieve the spatial index for the drivable area polygons
        if (
            self._cached_map_version != navigation_map.map_version
            or self._drivable_tree is None
        ):
            polygons = []
            for poly_primitive in navigation_map.drivable_polygons:
                pts = poly_primitive.points_enu_m[:, :2]  # Take (X, Y)
                if len(pts) >= 3:
                    polygons.append(Polygon(pts))
            self._drivable_tree = STRtree(polygons) if polygons else None
            self._cached_map_version = navigation_map.map_version

        if self._drivable_tree is None:
            raise ValueError("No drivable area defined")

        # 2. Transform trajectory from ego-centric to map frame (ENU)
        if len(trajectory_xy) == 0:
            return 0.0

        c, s = np.cos(ego_pose[2]), np.sin(ego_pose[2])
        rot_mat = np.array([[c, -s], [s, c]])
        traj_global = (trajectory_xy @ rot_mat.T) + np.array([ego_pose[0], ego_pose[1]])

        off_road_penalty = 0.0

        for x, y in traj_global:
            pt = Point(x, y)

            # --- Off-road check ---
            possible_matches = self._drivable_tree.query(pt)
            if not possible_matches.size:
                off_road_penalty -= 1.0
            else:
                is_on_road = False
                for idx in possible_matches:
                    if self._drivable_tree.geometries[idx].covers(pt):
                        is_on_road = True
                        break
                if not is_on_road:
                    off_road_penalty -= 1.0

        num_steps = len(traj_global)
        if num_steps > 0:
            return off_road_penalty / num_steps
        return 0.0


class RewardManager:
    """Computes total reward for the AutoE2E RL loop."""

    def __init__(
        self,
        w_gt_dev: float = 1.0,
        w_offroad: float = 0.5,
        gt_reward: GroundTruthDeviationReward | None = None,
        offroad_reward: OffRoadReward | None = None,
    ) -> None:
        self.w_gt_dev = w_gt_dev
        self.w_offroad = w_offroad
        self.gt_reward = gt_reward or GroundTruthDeviationReward()
        self.offroad_reward = offroad_reward or OffRoadReward()

        if self.gt_reward is None and self.offroad_reward is None:
            raise ValueError("At least one reward should be passed")

    def compute_with_truncation(
        self,
        trajectory_xy: np.ndarray,
        gt_trajectory: np.ndarray,
        ego_pose: tuple[float, float, float],
        navigation_map: Any,
    ) -> tuple[float, bool]:
        """Compute total reward and return 3DGS boundary truncation flag."""
        r_gt, truncated = self.gt_reward.compute_with_truncation(
            trajectory_xy, gt_trajectory
        )
        r_offroad = self.offroad_reward.compute(ego_pose, trajectory_xy, navigation_map)
        total_reward = self.w_gt_dev * r_gt + self.w_offroad * r_offroad
        return total_reward, truncated

    def compute(
        self,
        trajectory_xy: np.ndarray,
        gt_trajectory: np.ndarray,
        ego_pose: tuple[float, float, float],
        navigation_map: Any,
    ) -> float:
        total_reward, _ = self.compute_with_truncation(
            trajectory_xy, gt_trajectory, ego_pose, navigation_map
        )
        return total_reward


__all__ = [
    "GroundTruthDeviationReward",
    "OffRoadReward",
    "RewardManager",
]
