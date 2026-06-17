import torch
from PIL import Image, ImageDraw
import math

_DT = 0.1  # 10 Hz
_FUTURE_TIMESTEPS = 64

class Visualization:

    @staticmethod
    def render_trajectory(
            action_sequence: torch.Tensor,
            current_speed: float,
            map_image: Image.Image,
            radius_m: float,
    ) -> Image.Image:
        """
        Integrates predicted trajectory into metric coordinates and
        draws them onto the raw BEV map tile.

        Args:
            action_sequence: (128, ) flattened (64, 2) tensor of predicted [acceleration, curvature].
            current_speed: Scalar float from the egomotion history.
            map_image: A map tile, not normalized.
            radius_m: The metric boundary of the map image.

        Returns:
            A new PIL Image with the trajectory drawn on it.
        """

        # change the trajectory format
        action_sequence = torch.reshape(action_sequence, (_FUTURE_TIMESTEPS, 2))

        # 1. Convert trajectory to [x y] in meters

        trajectory_m = torch.zeros((_FUTURE_TIMESTEPS + 1, 2))
        trajectory_m[0, :] = 0

        # 1.1 velocity is needed for integration
        v = float(current_speed)

        # 1.2 Yaw angle is needed to derive 2D acceleration
        yaw = 0.0

        for i in range(_FUTURE_TIMESTEPS):
            accel = action_sequence[i, 0].item()
            curv = action_sequence[i, 1].item()

            v = v + (accel * _DT)
            yaw = yaw + (v * curv * _DT)

            # The format is [X Y]. Sign convention for yaw is + = CCW
            trajectory_m[i + 1, 0] = trajectory_m[i, 0] - (v * math.sin(yaw) * _DT)
            trajectory_m[i + 1, 1] = trajectory_m[i, 1] + (v * math.cos(yaw) * _DT)

        # 2. Convert meters to pixels

        W, H = map_image.size

        trajectory_px = torch.zeros_like(trajectory_m)
        trajectory_px[:, 0] = ((trajectory_m[:, 0] + radius_m) / (2 * radius_m)) * W
        trajectory_px[:, 1] = ((radius_m - trajectory_m[:, 1]) / (2 * radius_m)) * H

        # 3. Overlay the trajectory onto the map tile

        pixel_points = [(x.item(), y.item()) for x, y in trajectory_px]

        map_with_trajectory = map_image.copy()

        draw = ImageDraw.Draw(map_with_trajectory)
        draw.line(pixel_points, fill="#33FF33", width=3)
        draw.circle(pixel_points[0], radius=5, fill='red')

        return map_with_trajectory