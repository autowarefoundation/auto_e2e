import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    ModelPrediction,
    PredictionInput,
)

from evaluation.metrics import integrate_trajectory

from .config import DEFAULT_CAMERA_NAMES
from .parser import AlpasimStreamParser

_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_yaw(quat: Any) -> float:
    """Extract yaw heading angle from quaternion."""
    return math.atan2(
        2.0 * (quat.w * quat.z + quat.x * quat.y),
        1.0 - 2.0 * (quat.y**2 + quat.z**2),
    )


def _unroll_unicycle_controls(
    controls: np.ndarray, v_init: float, dt: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate (acceleration, curvature) controls into (x, y) waypoints and headings."""
    points, headings = integrate_trajectory(
        accel=controls[:, 0],
        curvature=controls[:, 1],
        v0=v_init,
        dt=dt,
        return_headings=True,
    )
    return points.astype(np.float32), headings.astype(np.float32)


class AutoE2EDriver(BaseTrajectoryModel):
    """AutoE2E driver plugin for AlpaSim."""

    def __init__(
        self,
        model_checkpoint: str = "dummy_random.ckpt",
        allow_mock: bool = False,
        allow_untrained_model: bool = False,
        camera_ids: list[str] | None = None,
        scene_id: str | None = None,
    ) -> None:
        super().__init__()
        self.allow_mock = allow_mock
        self.allow_untrained_model = allow_untrained_model
        self.model_checkpoint = model_checkpoint
        self._camera_ids = camera_ids or DEFAULT_CAMERA_NAMES

        self.parser = AlpasimStreamParser(
            camera_names=self._camera_ids,
            scene_id=scene_id,
        )
        self.device = _DEFAULT_DEVICE
        self.model = None

        if model_checkpoint and Path(model_checkpoint).exists():
            checkpoint = torch.load(model_checkpoint, map_location=self.device)
            from model_components.auto_e2e import AutoE2E

            config = dict(checkpoint["config"])
            valid = set(inspect.signature(AutoE2E.__init__).parameters) - {"self"}
            kwargs = {k: v for k, v in config.items() if k in valid}
            kwargs["is_pretrained"] = False

            self.model = AutoE2E(**kwargs).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        elif self.allow_untrained_model:
            from model_components.auto_e2e import AutoE2E

            self.model = AutoE2E(
                num_views=len(self._camera_ids), is_pretrained=False
            ).to(self.device)
            self.model.eval()
        elif not self.allow_mock:
            raise FileNotFoundError(
                f"Model checkpoint '{model_checkpoint}' not found and allow_mock=False."
            )

    @classmethod
    def from_config(
        cls,
        model_cfg: Any = None,
        device: torch.device = _DEFAULT_DEVICE,
        camera_ids: list[str] | None = None,
        context_length: int | None = None,
        output_frequency_hz: int = 10,
    ) -> "AutoE2EDriver":
        checkpoint_path = "MOCK"
        allow_mock = False
        allow_untrained_model = False
        scene_id = None

        if model_cfg is not None:
            if isinstance(model_cfg, dict):
                checkpoint_path = model_cfg.get("checkpoint_path", checkpoint_path)
                scene_id = model_cfg.get("scene_id")
                allow_mock = model_cfg.get("allow_mock", False)
                allow_untrained_model = model_cfg.get("allow_untrained_model", False)
            else:
                checkpoint_path = getattr(model_cfg, "checkpoint_path", checkpoint_path)
                scene_id = getattr(model_cfg, "scene_id", None)
                allow_mock = getattr(model_cfg, "allow_mock", False)
                allow_untrained_model = getattr(
                    model_cfg, "allow_untrained_model", False
                )

        driver = cls(
            model_checkpoint=checkpoint_path,
            allow_mock=allow_mock or checkpoint_path == "MOCK" or not checkpoint_path,
            allow_untrained_model=allow_untrained_model
            or checkpoint_path == "UNTRAINED",
            camera_ids=camera_ids,
            scene_id=scene_id,
        )
        driver.device = device
        if driver.model is not None:
            driver.model.to(device)
        return driver

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return 1

    @property
    def output_frequency_hz(self) -> int:
        return 10

    def _encode_command(self, command: Any) -> None:
        """AutoE2E predicts trajectories end-to-end without discrete driving commands."""
        return

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        """Process real-time PredictionInput to ModelPrediction.

        Returns:
            ModelPrediction with trajectory_xy [64, 2] and headings [64].
        """
        cameras_dict = {}
        for cam_name, val in prediction_input.camera_images.items():
            frame = val[-1] if isinstance(val, (list, tuple)) else val
            cameras_dict[cam_name] = getattr(frame, "image", frame)

        speed = prediction_input.speed
        acceleration = prediction_input.acceleration

        yaw_rate = 0.0
        curvature = 0.0
        ego_pose = None
        ego_pose_history = prediction_input.ego_pose_history
        if ego_pose_history and len(ego_pose_history) >= 1:
            curr = ego_pose_history[-1]
            curr_yaw = _extract_yaw(curr.pose.quat)
            ego_pose = (curr.pose.x, curr.pose.y, curr_yaw)

            if len(ego_pose_history) >= 2:
                prev = ego_pose_history[-2]
                dt = (curr.timestamp_us - prev.timestamp_us) / 1_000_000.0
                if dt > 0:
                    prev_yaw = _extract_yaw(prev.pose.quat)
                    diff = math.atan2(
                        math.sin(curr_yaw - prev_yaw), math.cos(curr_yaw - prev_yaw)
                    )
                    yaw_rate = diff / dt
                    curvature = yaw_rate / max(speed, 0.1)

        observation = {
            "cameras": cameras_dict,
            "speed": speed,
            "acceleration": acceleration,
            "yaw_rate": yaw_rate,
            "curvature": curvature,
            "ego_pose": ego_pose,
        }

        if self.model is not None:
            parsed = self.parser.parse_observation(observation)
            tensors = {
                k: v.to(self.device) if hasattr(v, "to") else v for k, v in parsed.items()
            }
            with torch.no_grad():
                controls = self.model(**tensors, mode="inference")
                points, headings = _unroll_unicycle_controls(
                    controls[0].cpu().numpy().reshape(64, 2), speed
                )
        else:
            if not self.allow_mock:
                raise RuntimeError(
                    f"Model checkpoint '{self.model_checkpoint}' failed to load and allow_mock=False. "
                    "Cannot execute live inference without a loaded model."
                )
            x = np.linspace(0.0, max(speed, 1.0) * 6.4, 64, dtype=np.float32)
            points = np.stack([x, np.zeros(64, dtype=np.float32)], axis=1)
            headings = np.zeros(64, dtype=np.float32)

        return ModelPrediction(
            trajectory_xy=points.astype(np.float32),
            headings=headings.astype(np.float32),
        )


__all__ = [
    "AutoE2EDriver",
    "ModelPrediction",
    "PredictionInput",
]
