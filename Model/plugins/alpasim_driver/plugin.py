from typing import Any, Dict, Optional, List, cast
import os
import sys
import torch
import numpy as np
import logging
from dataclasses import dataclass, field
from enum import IntEnum

# Add alpasim core driver path to sys.path first to avoid package shadowing
_ALPASIM_DRIVER_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scratch", "alpasim", "src", "driver", "src"))
if os.path.exists(_ALPASIM_DRIVER_SRC) and _ALPASIM_DRIVER_SRC not in sys.path:
    sys.path.insert(0, _ALPASIM_DRIVER_SRC)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


try:
    from alpasim_driver.models.base import (
        BaseTrajectoryModel,
        PredictionInput,
        ModelPrediction,
        DriveCommand,
    )
except ImportError:
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

from data_parsing.alpasim_stream.parser import AlpasimStreamParser, PredictionInput as ParserPredictionInput  # noqa: E402

logger = logging.getLogger(__name__)


class AutoE2EDriver(BaseTrajectoryModel):
    """AutoE2E driver plugin for AlpaSim."""

    def __init__(self, model_checkpoint: str = "dummy_random.ckpt", **kwargs: Any) -> None:
        super().__init__()
        if not model_checkpoint or not os.path.exists(model_checkpoint):
            # If default checkpoint doesn't exist yet, we will log a warning or defer loading
            logger.warning("Checkpoint path %s not found. AutoE2EDriver will expect model created later.", model_checkpoint)

        self.model_checkpoint = model_checkpoint
        self.parser = AlpasimStreamParser()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        if os.path.exists(model_checkpoint):
            self.model = torch.load(model_checkpoint, map_location=self.device)
            self.model.eval()

    @classmethod
    def from_config(
        cls,
        model_cfg: Any,
        device: torch.device,
        camera_ids: List[str],
        context_length: Optional[int],
        output_frequency_hz: int,
    ) -> "AutoE2EDriver":
        checkpoint_path = getattr(model_cfg, "checkpoint_path", "dummy_random.ckpt")
        driver = cls(model_checkpoint=checkpoint_path)
        driver.device = device
        return driver

    @property
    def camera_ids(self) -> List[str]:
        return [
            "camera_base_front_center",
            "camera_ring_front",
            "camera_ring_front_left",
            "camera_ring_front_right",
            "camera_ring_rear",
            "camera_ring_rear_left",
            "camera_ring_rear_right",
        ]

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
        cameras_dict = {}
        if hasattr(input_data, "camera_images") and input_data.camera_images:
            for cam_name, frames in input_data.camera_images.items():
                if isinstance(frames, (list, tuple)):
                    if len(frames) > 0:
                        frame = frames[-1]
                        cameras_dict[cam_name] = getattr(frame, "image", frame)
                    else:
                        cameras_dict[cam_name] = None
                else:
                    cameras_dict[cam_name] = getattr(frames, "image", frames)
        elif hasattr(input_data, "cameras"):
            cameras_dict = input_data.cameras

        speed = float(getattr(input_data, "speed", 0.0))
        acceleration = float(getattr(input_data, "acceleration", 0.0))
        raw_cmd = getattr(input_data, "command", 1)
        command = self._encode_command(raw_cmd)

        input_dict = cast(ParserPredictionInput, {
            "cameras": cameras_dict,
            "speed": speed,
            "acceleration": acceleration,
            "command": command,
        })
        
        tensors = self.parser.parse_observation(input_dict)
        tensors = {k: v.to(self.device) for k, v in tensors.items()}
        
        if self.model is not None:
            with torch.no_grad():
                outputs = self.model(tensors)
                points = outputs["trajectory_points"][0].cpu().numpy()
                headings = outputs["headings"][0].cpu().numpy()
        else:
            # Fallback mock output if model file is missing
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


