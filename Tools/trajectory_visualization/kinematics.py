import torch
import numpy as np
import math
from dataclasses import dataclass

@dataclass
class ModelOutputContract:
    num_timesteps: int
    num_signals: int
    sampling_interval_dt: float
    acceleration_unit: str
    curvature_unit: str
    speed_unit: str
    coordinate_handedness: str
    
    def __post_init__(self):
        if self.sampling_interval_dt <= 0:
            raise ValueError(f"Invalid sampling interval (dt): {self.sampling_interval_dt}. Must be > 0.")
        if self.num_timesteps <= 0:
            raise ValueError(f"Invalid num_timesteps: {self.num_timesteps}. Must be > 0.")
    
    @classmethod
    def from_config_and_manifest(cls, model_config: dict, dataset_manifest: dict) -> 'ModelOutputContract':
        try:
            return cls(
                num_timesteps=model_config["num_timesteps"],
                num_signals=model_config["num_signals"],
                sampling_interval_dt=dataset_manifest["dt"],
                acceleration_unit=dataset_manifest["acceleration_unit"],
                curvature_unit=dataset_manifest["curvature_unit"],
                speed_unit=dataset_manifest["speed_unit"],
                coordinate_handedness=dataset_manifest["coordinate_handedness"]
            )
        except KeyError as e:
            raise KeyError(f"Model output contract validation failed. Missing required property: {e}")

def controls_to_metric_trajectory(
        controls: torch.Tensor,
        initial_speed: float,
        contract: ModelOutputContract,
        initial_heading: float = 0.0
) -> torch.Tensor:
    """
    Converts an action sequence of vehicle controls [acceleration, curvature] into a 2D trajectory in meters.

    Args:
        controls (torch.Tensor): Flattened or (T, 2) tensor of controls.
        initial_speed (float): Initial speed of the vehicle in m/s.
        contract (ModelOutputContract): Explicit data contract detailing intervals and units.
        initial_heading (float, optional): Initial heading angle in radians. Defaults to 0.0.

    Returns:
        torch.Tensor: A tensor of shape (T + 1, 2) containing [x, y] coordinates in meters.
    """
    controls = controls.view(-1, 2)
    
    # Contract validation
    if controls.shape[0] != contract.num_timesteps:
        raise ValueError(f"Unsupported planner output: Expected {contract.num_timesteps} timesteps but got {controls.shape[0]}")
    if controls.shape[1] != contract.num_signals:
        raise ValueError(f"Unsupported planner output: Expected {contract.num_signals} signals but got {controls.shape[1]}")
    
    future_timesteps = controls.shape[0]
    dt = contract.sampling_interval_dt
    
    trajectory_m = torch.zeros((future_timesteps + 1, 2))
    trajectory_m[0, :] = 0

    v = initial_speed
    yaw = initial_heading

    for i in range(future_timesteps):
        accel = controls[i, 0].item()
        curv = controls[i, 1].item()

        v = v + (accel * dt)
        yaw = yaw + (v * curv * dt)

        # Sign convention for yaw is + = CCW
        trajectory_m[i + 1, 0] = trajectory_m[i, 0] - (v * math.sin(yaw) * dt)
        trajectory_m[i + 1, 1] = trajectory_m[i, 1] + (v * math.cos(yaw) * dt)

    return trajectory_m

def get_cumulative_distances(trajectory_m: torch.Tensor) -> np.ndarray:
    """
    Calculates the cumulative path distance along a 2D trajectory.

    Args:
        trajectory_m (torch.Tensor): Trajectory points in meters.

    Returns:
        np.ndarray: 1D array of cumulative distances from the start of the trajectory.
    """
    pts = trajectory_m.numpy()
    diffs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    dists = np.zeros(pts.shape[0], dtype=np.float32)
    dists[1:] = np.cumsum(diffs)
    return dists

def get_trajectory_boundaries_3d(trajectory_m: torch.Tensor, width_m: float = 1.8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the left and right 3D boundary lines of a trajectory based on a fixed vehicle width.

    Args:
        trajectory_m (torch.Tensor): Centerline trajectory in meters.
        width_m (float, optional): Total width of the trajectory path in meters. Defaults to 1.8.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Left and right boundary trajectories in meters.
    """
    pts = trajectory_m.numpy()
    N = pts.shape[0]
    
    left_bound = np.zeros((N, 2), dtype=np.float32)
    right_bound = np.zeros((N, 2), dtype=np.float32)
    
    for i in range(N):
        if i < N - 1:
            d = pts[i+1] - pts[i]
        else:
            d = pts[i] - pts[i-1]
            
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            d = np.array([0, 1])
        else:
            d = d / norm
            
        n = np.array([-d[1], d[0]])
        
        left_bound[i] = pts[i] + n * (width_m / 2.0)
        right_bound[i] = pts[i] - n * (width_m / 2.0)
        
    return torch.from_numpy(left_bound), torch.from_numpy(right_bound)
