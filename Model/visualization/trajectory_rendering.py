import torch
import cv2
import numpy as np
import math

_DT = 0.1  # 10 Hz
_FUTURE_TIMESTEPS = 64
MAP_W = 640
MAP_H = 360

class Visualization:

    @staticmethod
    def accel_and_curv_to_meters_trajectory(
            action_sequence: torch.Tensor,
            current_speed: float,
            future_timesteps: int,
            initial_heading: float = 0.0,
            radius_m: float = 800.0
    ) -> torch.Tensor:

        # change the trajectory format
        action_sequence = torch.reshape(action_sequence, (future_timesteps, 2))

        # 1. Convert trajectory to [x y] in meters

        trajectory_m = torch.zeros((future_timesteps + 1, 2))
        trajectory_m[0, :] = 0

        # 1.1 velocity is needed for integration
        v = float(current_speed)

        # 1.2 Yaw angle is needed to derive 2D acceleration
        yaw = float(initial_heading)

        for i in range(future_timesteps):
            accel = action_sequence[i, 0].item()
            curv = action_sequence[i, 1].item()

            v = v + (accel * _DT)
            yaw = yaw + (v * curv * _DT)

            # The format is [X Y]. Sign convention for yaw is + = CCW
            trajectory_m[i + 1, 0] = trajectory_m[i, 0] - (v * math.sin(yaw) * _DT)
            trajectory_m[i + 1, 1] = trajectory_m[i, 1] + (v * math.cos(yaw) * _DT)

        return trajectory_m

    @staticmethod
    def meters_to_pixels_trajectory(trajectory_m: torch.Tensor, radius_m: float, map_image: np.ndarray) -> torch.Tensor:
        h, w = map_image.shape[:2]

        trajectory_px = torch.zeros_like(trajectory_m)
        trajectory_px[:, 0] = ((trajectory_m[:, 0] + radius_m) / (2 * radius_m)) * w
        trajectory_px[:, 1] = ((radius_m - trajectory_m[:, 1]) / (2 * radius_m)) * h

        return trajectory_px

    @staticmethod
    def overlay_the_trajectory_with_map(
            trajectory_px: torch.Tensor,
            map_image: np.ndarray,
            color: tuple = (0, 255, 0),
            initial_heading: float = 0.0,
            radius_m: float = 800.0
    ) -> np.ndarray:
        bgr_color = color
        black_color = (0, 0, 0)

        map_with_trajectory = map_image.copy()

        # Convert PyTorch tensor points to float first to avoid quantization errors in angle math
        pixel_points_float = [(x.item(), y.item()) for x, y in trajectory_px]
        pixel_points = np.array(pixel_points_float, np.int32)
        pts = pixel_points.reshape((-1, 1, 2))

        # Scaling based on zoom level (assuming base radius is 800.0)
        zoom_scale = 800.0 / radius_m

        linewidth = max(1, int(1 * zoom_scale))
        outline_width = max(1, int(1 * zoom_scale))

        # Draw trajectory line with OpenCV (AA = Anti-Aliased for smooth edges)
        cv2.polylines(map_with_trajectory, [pts], isClosed=False, color=black_color, thickness=linewidth + outline_width * 2, lineType=cv2.LINE_AA)
        cv2.polylines(map_with_trajectory, [pts], isClosed=False, color=bgr_color, thickness=linewidth, lineType=cv2.LINE_AA)

        # Agent marker: sleek arrowhead pointing in the initial heading
        dx = -math.sin(initial_heading)
        dy = -math.cos(initial_heading)
        rx = math.cos(initial_heading)
        ry = -math.sin(initial_heading)

        x0, y0 = pixel_points[0]
        L = 8.0 * zoom_scale
        W = 4.0 * zoom_scale

        tip = (int(x0 + L * dx), int(y0 + L * dy))
        left_back = (int(x0 - L * dx + W * rx), int(y0 - L * dy + W * ry))
        right_back = (int(x0 - L * dx - W * rx), int(y0 - L * dy - W * ry))

        poly_points = np.array([tip, right_back, left_back], np.int32).reshape((-1, 1, 2))
        
        # Draw thick black outline then filled color inside for the agent marker
        agent_color = (126, 27, 232) #purple
        cv2.fillPoly(map_with_trajectory, [poly_points], agent_color, cv2.LINE_8)
        cv2.polylines(map_with_trajectory, [poly_points], isClosed=True, color=black_color, thickness=outline_width, lineType=cv2.LINE_8)

        return map_with_trajectory

    @staticmethod
    def render_trajectory_map_tile(
        action_sequence: torch.Tensor,
        current_speed: float,
        map_image: np.ndarray,
        radius_m: float,
        color: tuple = (0, 255, 0),
        initial_heading: float = 0.0
    ) -> np.ndarray:
        """
        Integrates predicted trajectory into metric coordinates and
        draws them onto the raw BEV map tile.

        Args:
            action_sequence: (128, ) flattened (64, 2) tensor of predicted [acceleration, curvature].
            current_speed: Scalar float from the egomotion history.
            map_image: A map tile, not normalized (BGR numpy array).
            radius_m: The metric boundary of the map image.

        Returns:
            A new Numpy array with the trajectory drawn on it.
        """

        # 1. Convert trajectory to [x y] in meters

        trajectory_m = Visualization.accel_and_curv_to_meters_trajectory(
            action_sequence, current_speed, _FUTURE_TIMESTEPS, initial_heading
        )

        # 2. Convert meters to pixels

        trajectory_px = Visualization.meters_to_pixels_trajectory(trajectory_m, radius_m, map_image)

        # 3. Overlay the trajectory onto the map tile

        map_with_trajectory = Visualization.overlay_the_trajectory_with_map(trajectory_px, map_image, color, initial_heading, radius_m)
        
        return map_with_trajectory

    @staticmethod
    def render_trajectory_on_a_grid(
        action_sequence: torch.Tensor,
        current_speed: float
    ) -> np.ndarray:

        # 1. Convert trajectory to [x y] in meters

        trajectory_m = Visualization.accel_and_curv_to_meters_trajectory(
            action_sequence, current_speed, _FUTURE_TIMESTEPS, initial_heading=0.0
        )

        

        return np.array([0, 1])