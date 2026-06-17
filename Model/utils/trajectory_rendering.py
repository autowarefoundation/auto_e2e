import torch
import numpy as np
from PIL import Image, ImageDraw

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

        # 1. Convert trajectory to [x y] in meters

        # 2. Convert meters to pixels

        # 3. Overlay the trajectory onto the map tile