import torch
from PIL import Image, ImageDraw
import math

_DT = 0.1  # 10 Hz
_FUTURE_TIMESTEPS = 64

class Visualization:

    @staticmethod
    def accel_and_curv_to_meters_trajectory(
            action_sequence: torch.Tensor,
            current_speed: float,
            future_timesteps: int,
            initial_heading: float = 0.0
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
    def meters_to_pixels_trajectory(trajectory_m: torch.Tensor, radius_m: float, map_image: Image.Image) -> torch.Tensor:
        w, h = map_image.size

        trajectory_px = torch.zeros_like(trajectory_m)
        trajectory_px[:, 0] = ((trajectory_m[:, 0] + radius_m) / (2 * radius_m)) * w
        trajectory_px[:, 1] = ((radius_m - trajectory_m[:, 1]) / (2 * radius_m)) * h

        return trajectory_px

    @staticmethod
    def overlay_the_trajectory_with_map(trajectory_px: torch.Tensor, map_image: Image.Image, color: str = "#33FF33") -> Image.Image:

        pixel_points = [(x.item(), y.item()) for x, y in trajectory_px]

        map_with_trajectory = map_image.copy()

        draw = ImageDraw.Draw(map_with_trajectory)
        draw.line(pixel_points, fill=color, width=3)
        x0 = pixel_points[0][0]
        y0 = pixel_points[0][1]
        r = 5.0
        draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill='red')

        return map_with_trajectory

    @staticmethod
    def render_trajectory_map_tile(
            action_sequence: torch.Tensor,
            current_speed: float,
            map_image: Image.Image,
            radius_m: float,
            color: str = "#33FF33",
            initial_heading: float = 0.0
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

        # 1. Convert trajectory to [x y] in meters

        trajectory_m = Visualization.accel_and_curv_to_meters_trajectory(
            action_sequence, current_speed, _FUTURE_TIMESTEPS, initial_heading
        )

        # 2. Convert meters to pixels

        trajectory_px = Visualization.meters_to_pixels_trajectory(trajectory_m, radius_m, map_image)

        # 3. Overlay the trajectory onto the map tile

        map_with_trajectory = Visualization.overlay_the_trajectory_with_map(trajectory_px, map_image, color)

        return map_with_trajectory