from typing import Any, Dict, Optional, List, cast
import os
import sys
import torch
import numpy as np
import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Resolve ALPASIM_ROOT from environment variable or check .alpasim / scratch/alpasim in repo root
_ALPASIM_ROOT = os.environ.get("ALPASIM_ROOT", os.path.join(_REPO_ROOT, ".alpasim"))

if os.path.exists(_ALPASIM_ROOT):
    _alpasim_src = os.path.join(_ALPASIM_ROOT, "src")
    for sub in ["driver", "plugins", "grpc", "utils", "controller", "physics", "runtime"]:
        for p in [os.path.join(_alpasim_src, sub, "src"), os.path.join(_alpasim_src, sub)]:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)


IS_MOCK_MODE = False

try:
    from alpasim_driver.models.base import (
        BaseTrajectoryModel,
        PredictionInput,
        ModelPrediction,
        DriveCommand,
    )
except ImportError:
    IS_MOCK_MODE = True

    class _MockDriveCommand(IntEnum):
        LEFT = 0
        STRAIGHT = 1
        RIGHT = 2
        UNKNOWN = 3

    @dataclass
    class _MockPredictionInput:
        camera_images: Dict[str, Any] = field(default_factory=dict)
        command: Any = _MockDriveCommand.STRAIGHT
        speed: float = 0.0
        acceleration: float = 0.0
        ego_pose_history: Optional[List[Any]] = None
        inference_seed: int = 0
        cameras: Optional[Dict[str, Any]] = None

        def __post_init__(self) -> None:
            if self.cameras is not None and not self.camera_images:
                self.camera_images = self.cameras
            elif self.camera_images and self.cameras is None:
                self.cameras = self.camera_images

    @dataclass
    class _MockModelPrediction:
        trajectory_xy: np.ndarray
        headings: np.ndarray
        reasoning_text: Optional[str] = None
        trajectory_points: Optional[np.ndarray] = None

        def __post_init__(self) -> None:
            if self.trajectory_points is not None and self.trajectory_xy is None:
                self.trajectory_xy = self.trajectory_points
            elif self.trajectory_xy is not None and self.trajectory_points is None:
                self.trajectory_points = self.trajectory_xy

    class _MockBaseTrajectoryModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def predict(self, input_data: Any) -> Any:
            raise NotImplementedError

    PredictionInput = _MockPredictionInput  # type: ignore
    ModelPrediction = _MockModelPrediction  # type: ignore
    BaseTrajectoryModel = _MockBaseTrajectoryModel  # type: ignore
    DriveCommand = _MockDriveCommand  # type: ignore

from .parser import AlpasimStreamParser, PredictionInput as ParserPredictionInput  # noqa: E402

logger = logging.getLogger(__name__)


class AutoE2EDriver(BaseTrajectoryModel):
    """AutoE2E driver plugin for AlpaSim."""

    def __init__(
        self,
        model_checkpoint: str = "dummy_random.ckpt",
        allow_mock: bool = False,
        allow_untrained_model: bool = False,
        camera_ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.allow_mock = allow_mock
        self.allow_untrained_model = allow_untrained_model

        if IS_MOCK_MODE and not self.allow_mock:
            raise ImportError(
                "alpasim_driver package is not installed and allow_mock=False. "
                "Pass allow_mock=True when initializing AutoE2EDriver(allow_mock=True) to enable mock dependencies."
            )

        self.model_checkpoint = model_checkpoint
        
        if camera_ids is None:
            from .config import AutoE2EAlpaSimConfig
            camera_ids = AutoE2EAlpaSimConfig(checkpoint_path=self.model_checkpoint).camera_names
        self._camera_ids = camera_ids
        
        self.parser = AlpasimStreamParser(camera_names=self._camera_ids)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        if model_checkpoint and os.path.exists(model_checkpoint):
            checkpoint = torch.load(model_checkpoint, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                try:
                    from model_components.auto_e2e import AutoE2E
                    self.model = AutoE2E(num_views=len(self._camera_ids), is_pretrained=False).to(self.device)
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                except Exception as e:
                    logger.error("Failed to load AutoE2E model from state_dict: %s", e)
            else:
                self.model = checkpoint
            
            if self.model is not None:
                self.model.eval()
        elif self.allow_untrained_model:
            try:
                from model_components.auto_e2e import AutoE2E
                logger.info("Checkpoint path '%s' not found. Initializing untrained AutoE2E model (allow_untrained_model=True).", model_checkpoint)
                self.model = AutoE2E(num_views=len(self._camera_ids), is_pretrained=False).to(self.device)
                self.model.eval()
            except Exception as e:
                logger.error("Failed to initialize untrained AutoE2E model: %s", e)
        else:
            if not self.allow_mock:
                logger.warning(
                    "Checkpoint path '%s' not found, allow_mock=False, and allow_untrained_model=False. "
                    "Driver will fail on predict() unless a model checkpoint is provided.", model_checkpoint
                )
            else:
                logger.warning("Checkpoint path '%s' not found. AutoE2EDriver will use mock trajectory outputs.", model_checkpoint)

    @classmethod
    def from_config(
        cls,
        model_cfg: Any,
        device: torch.device,
        camera_ids: List[str],
        context_length: Optional[int],
        output_frequency_hz: int,
        allow_mock: bool = False,
        allow_untrained_model: bool = False,
    ) -> "AutoE2EDriver":
        checkpoint_path = getattr(model_cfg, "checkpoint_path", "dummy_random.ckpt")
        
        allow_mock_cfg = getattr(model_cfg, "allow_mock", allow_mock)
        if checkpoint_path == "MOCK":
            allow_mock_cfg = True
            
        allow_untrained_cfg = getattr(model_cfg, "allow_untrained_model", allow_untrained_model)
        if checkpoint_path == "UNTRAINED":
            allow_untrained_cfg = True
            
        driver = cls(
            model_checkpoint=checkpoint_path,
            allow_mock=allow_mock_cfg,
            allow_untrained_model=allow_untrained_cfg,
            camera_ids=camera_ids,
        )
        driver.device = device
        return driver

    @property
    def camera_ids(self) -> List[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return 1

    @property
    def output_frequency_hz(self) -> int:
        return 10

    def _encode_command(self, command: Any) -> int:
        if isinstance(command, int):
            return command
        elif hasattr(command, "value"):
            return int(command.value)
        return 1

    def predict(self, input_data: Any) -> ModelPrediction:
        """Process real-time PredictionInput to ModelPrediction.
        
        Returns:
            ModelPrediction with trajectory_points / trajectory_xy [64, 2] and headings [64].
        """
        # Extract cameras dict
        cameras_dict: Dict[str, Any] = {}
        if hasattr(input_data, "camera_images") and input_data.camera_images:
            for cam_name, frames in input_data.camera_images.items():
                if not frames:
                    cameras_dict[cam_name] = None
                    continue
                
                if isinstance(frames, list):
                    frame = frames[-1]
                else:
                    frame = frames
                    
                if hasattr(frame, "image"):
                    cameras_dict[cam_name] = frame.image
                elif isinstance(frame, tuple):
                    if len(frame) == 2:
                        cameras_dict[cam_name] = frame[1]
                    else:
                        cameras_dict[cam_name] = frame[-1]
                else:
                    cameras_dict[cam_name] = frame
        elif hasattr(input_data, "cameras"):
            cameras_dict = input_data.cameras

        speed = float(getattr(input_data, "speed", 0.0))
        acceleration = float(getattr(input_data, "acceleration", 0.0))
        raw_cmd = getattr(input_data, "command", 1)
        command = self._encode_command(raw_cmd)
        
        yaw_rate = 0.0
        curvature = 0.0
        ego_pose_history = getattr(input_data, "ego_pose_history", None)
        if ego_pose_history and len(ego_pose_history) >= 2:
            prev = ego_pose_history[-2]
            curr = ego_pose_history[-1]
            dt = (curr.timestamp_us - prev.timestamp_us) / 1_000_000.0
            if dt > 0:
                def extract_yaw(quat: Any) -> float:
                    return math.atan2(
                        2.0 * (quat.w * quat.z + quat.x * quat.y),
                        1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
                    )
                try:
                    prev_yaw = extract_yaw(prev.pose.quat)
                    curr_yaw = extract_yaw(curr.pose.quat)
                    diff = curr_yaw - prev_yaw
                    diff = math.atan2(math.sin(diff), math.cos(diff))
                    yaw_rate = diff / dt
                    curvature = yaw_rate / max(speed, 0.1)
                except AttributeError:
                    pass  # Gracefully fallback to 0.0 if pose shape is mocked or unrecognized

        input_dict = cast(ParserPredictionInput, {
            "cameras": cameras_dict,
            "speed": speed,
            "acceleration": acceleration,
            "command": command,
            "yaw_rate": yaw_rate,
            "curvature": curvature,
        })
        
        tensors = self.parser.parse_observation(input_dict)
        tensors = {k: v.to(self.device) for k, v in tensors.items()}
        
        if self.model is not None:
            with torch.no_grad():
                outputs = self.model(**tensors, mode="inference")

                if isinstance(outputs, dict):
                    points = outputs["trajectory_points"][0].cpu().numpy() if isinstance(outputs.get("trajectory_points"), torch.Tensor) else outputs["trajectory_points"][0]
                    headings = outputs["headings"][0].cpu().numpy() if isinstance(outputs.get("headings"), torch.Tensor) else outputs["headings"][0]
                elif isinstance(outputs, torch.Tensor):
                    pts_tensor = outputs[0].cpu().numpy()
                    if pts_tensor.ndim == 1 and pts_tensor.shape[0] == 128:
                        controls = pts_tensor.reshape(64, 2)
                        points = np.zeros((64, 2), dtype=np.float32)
                        headings = np.zeros(64, dtype=np.float32)
                        
                        dt = 0.1  # 10Hz planning rate
                        v = float(input_dict["speed"])
                        x, y, theta = 0.0, 0.0, 0.0
                        
                        for i in range(64):
                            a, k = controls[i, 0], controls[i, 1]
                            
                            # Kinematic unicycle update
                            x += v * np.cos(theta) * dt
                            y += v * np.sin(theta) * dt
                            theta += v * k * dt
                            v += a * dt
                            
                            points[i, 0] = x
                            points[i, 1] = y
                            headings[i] = theta
                    else:
                        raise ValueError(f"Unexpected tensor shape from AutoE2E: {pts_tensor.shape}")
                else:
                    raise TypeError(f"Unexpected model output type: {type(outputs)}")
        else:
            if not self.allow_mock:
                raise RuntimeError(
                    f"Model checkpoint '{self.model_checkpoint}' failed to load and allow_mock=False. "
                    "Cannot execute live inference without a loaded model."
                )
            # Fallback mock output if model file is missing and allow_mock is True
            t = np.linspace(0, 20, 64)
            points = np.stack([t, 0.5 * t ** 2], axis=1)
            headings = np.arctan2(t, np.ones_like(t))

        try:
            return ModelPrediction(
                trajectory_xy=points,
                headings=headings
            )
        except TypeError:
            return ModelPrediction(
                trajectory_points=points,
                headings=headings
            )


AutoE2EAlpaSimModel = AutoE2EDriver


