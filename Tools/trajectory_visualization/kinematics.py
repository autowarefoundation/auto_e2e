import torch
import numpy as np
import math

def accel_and_curv_to_meters_trajectory(
        action_sequence: torch.Tensor,
        current_speed: float,
        future_timesteps: int,
        initial_heading: float = 0.0,
        dt: float = 0.1
) -> torch.Tensor:
    """
    Converts an action sequence of acceleration and curvature into a 2D trajectory in meters.

    Args:
        action_sequence (torch.Tensor): Flattened tensor of [acceleration, curvature] actions.
        current_speed (float): Initial speed of the vehicle in m/s.
        future_timesteps (int): Number of timesteps to predict.
        initial_heading (float, optional): Initial heading angle in radians. Defaults to 0.0.
        dt (float, optional): Time delta per step. Defaults to 0.1.

    Returns:
        torch.Tensor: A tensor of shape (future_timesteps + 1, 2) containing [x, y] coordinates in meters.
    """
    action_sequence = torch.reshape(action_sequence, (future_timesteps, 2))
    trajectory_m = torch.zeros((future_timesteps + 1, 2))
    trajectory_m[0, :] = 0

    v = current_speed
    yaw = initial_heading

    for i in range(future_timesteps):
        accel = action_sequence[i, 0].item()
        curv = action_sequence[i, 1].item()

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
