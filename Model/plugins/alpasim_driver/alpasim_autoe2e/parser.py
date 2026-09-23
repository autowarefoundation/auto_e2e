import io
import os
import time
from pathlib import Path
from typing import Any

import data_parsing.kit_scenes.navigation as kit_nav
import navigation.rasterizer as nav_rasterizer
import numpy as np
import torch
from model_components.view_fusion import PinholeProjection
from navigation.rasterizer import EgoPose
from PIL import Image

from .config import get_image_transform, load_projection_matrices

_HISTORY_STEPS = 64
_HISTORY_SIGNALS = 4
_VISUAL_HISTORY_DIM = 896


class AlpasimStreamParser:
    """Parses live AlpaSim frames into the exact tensor format produced by pre_extracted.py."""

    def __init__(
        self,
        camera_names: list[str],
        scene_id: str | None = None,
    ) -> None:
        self.camera_names = camera_names
        self.transform = get_image_transform()
        self._egomotion_buffer = np.zeros(
            (_HISTORY_STEPS, _HISTORY_SIGNALS), dtype=np.float32
        )
        self.visual_history = torch.zeros(1, _VISUAL_HISTORY_DIM, dtype=torch.float32)

        calib_matrices = load_projection_matrices()
        matrices = [calib_matrices[name] for name in self.camera_names]
        self.camera_params = torch.tensor(matrices, dtype=torch.float32).unsqueeze(0)
        self.projection = PinholeProjection(self.camera_params)

        self.navigation_map = None
        self.route = None
        self.rasterizer = None
        self.scene_path = None

        if scene_id:
            kitscenes_root = os.environ.get("KITSCENES_ROOT")
            if not kitscenes_root:
                local_kit = Path.cwd() / ".KITdata"
                if local_kit.exists():
                    kitscenes_root = str(local_kit)

            if kitscenes_root:
                scene_path = Path(kitscenes_root) / "data" / "val" / scene_id
                if not scene_path.exists():
                    scene_path = Path(kitscenes_root) / "data" / "train" / scene_id

                if scene_path.exists():
                    self.scene_path = scene_path
                    poses_file = scene_path / "poses.txt"
                    if poses_file.exists():
                        data = np.loadtxt(poses_file)
                        timestamps_ns = (data[:, 0] * 1e9).astype(np.int64)
                        positions_enu_m = data[:, 1:4]
                        qx, qy, qz, qw = data[:, 4], data[:, 5], data[:, 6], data[:, 7]
                        yaws_rad = np.arctan2(
                            2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
                        )

                        self.rasterizer = nav_rasterizer.NativeNavigationRasterizer()
                        nav = kit_nav.build_scene_navigation(
                            scene_id=scene_id,
                            scene_path=scene_path,
                            positions_enu_m=positions_enu_m,
                            yaws_rad=yaws_rad,
                            timestamps_ns=timestamps_ns,
                            source_revision="alpasim",
                            rasterizer=self.rasterizer,
                        )
                        self.navigation_map = nav.navigation_map
                        self.route = nav.route

    def _decode_image(self, image: bytes | np.ndarray | Image.Image) -> torch.Tensor:
        """Normalize camera frame array, bytes, or PIL Image into a [3, 256, 256] tensor."""
        if isinstance(image, bytes):
            img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image.convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256), resample=Image.Resampling.BILINEAR)
        return self.transform(img)

    def parse_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Convert a live observation dictionary into pipeline batch tensors.

        Returns:
            Dict containing camera_tiles, egomotion_history, visual_history,
            map_context, route_mask, map_valid, route_valid, projection, geometry_type.
        """
        frames = []
        for cam_name in self.camera_names:
            frame_data = observation["cameras"].get(cam_name)
            if frame_data is None:
                raise ValueError(f"Missing camera frame for {cam_name}")
            frames.append(self._decode_image(frame_data))
        visual_tiles = torch.stack(frames).unsqueeze(0)

        self._egomotion_buffer[:-1] = self._egomotion_buffer[1:]
        self._egomotion_buffer[-1] = [
            observation["speed"],
            observation["acceleration"],
            observation.get("yaw_rate", 0.0),
            observation.get("curvature", 0.0),
        ]
        egomotion_history = torch.from_numpy(
            self._egomotion_buffer.reshape(1, -1).copy()
        )

        map_context = torch.zeros(1, 14, 256, 256, dtype=torch.float32)
        route_mask = torch.zeros(1, 2, 256, 256, dtype=torch.float32)
        route_valid_flag = False

        ego_pose_tuple = observation.get("ego_pose")
        if self.rasterizer and self.route:
            if ego_pose_tuple is None:
                raise ValueError(
                    "Ego pose is missing from the observation, cannot render route mask."
                )

            x, y, yaw = ego_pose_tuple
            live_pose = EgoPose(
                timestamp_ns=time.time_ns(),
                x_enu_m=x,
                y_enu_m=y,
                yaw_rad=yaw,
            )
            raster = self.rasterizer.render(self.navigation_map, self.route, live_pose)
            route_mask = torch.from_numpy(raster.route_mask).float().unsqueeze(0)
            map_context = torch.from_numpy(raster.map_context).float().unsqueeze(0)
            route_valid_flag = raster.route_valid
        else:
            raise ImportError(
                "The rasterizer and/or route are missing, cannot render route mask."
            )

        map_valid = torch.tensor([self.navigation_map is not None], dtype=torch.bool)
        route_valid = torch.tensor([route_valid_flag], dtype=torch.bool)

        return {
            "camera_tiles": visual_tiles,
            "egomotion_history": egomotion_history,
            "visual_history": self.visual_history,
            "map_context": map_context,
            "route_mask": route_mask,
            "map_valid": map_valid,
            "route_valid": route_valid,
            "projection": self.projection,
            "geometry_type": "pinhole",
        }
