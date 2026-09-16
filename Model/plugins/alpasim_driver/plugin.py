from typing import Any, Dict, Optional, List, cast
import os
import sys
import torch
import numpy as np
import logging
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

from data_parsing.alpasim_stream.parser import AlpasimStreamParser, PredictionInput as ParserPredictionInput  # noqa: E402

logger = logging.getLogger(__name__)


class AutoE2EDriver(BaseTrajectoryModel):
    """AutoE2E driver plugin for AlpaSim."""

    def __init__(
        self,
        model_checkpoint: str = "dummy_random.ckpt",
        allow_mock: bool = False,
        allow_untrained_model: bool = False,
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
        self.parser = AlpasimStreamParser()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        if model_checkpoint and os.path.exists(model_checkpoint):
            self.model = torch.load(model_checkpoint, map_location=self.device)
            self.model.eval()
        elif self.allow_untrained_model:
            try:
                from model_components.auto_e2e import AutoE2E
                logger.info("Checkpoint path '%s' not found. Initializing untrained AutoE2E model (allow_untrained_model=True).", model_checkpoint)
                self.model = AutoE2E(num_views=7, is_pretrained=False).to(self.device)
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
        allow_untrained_cfg = getattr(model_cfg, "allow_untrained_model", allow_untrained_model)
        driver = cls(
            model_checkpoint=checkpoint_path,
            allow_mock=allow_mock_cfg,
            allow_untrained_model=allow_untrained_cfg,
        )
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
                try:
                    outputs = self.model(tensors)
                except TypeError:
                    outputs = self.model(**tensors)

                if isinstance(outputs, dict):
                    points = outputs["trajectory_points"][0].cpu().numpy() if isinstance(outputs.get("trajectory_points"), torch.Tensor) else outputs["trajectory_points"][0]
                    headings = outputs["headings"][0].cpu().numpy() if isinstance(outputs.get("headings"), torch.Tensor) else outputs["headings"][0]
                elif isinstance(outputs, torch.Tensor):
                    pts_tensor = outputs[0].cpu().numpy()
                    if pts_tensor.ndim == 1 and pts_tensor.shape[0] == 128:
                        points = pts_tensor.reshape(64, 2)
                    elif pts_tensor.ndim == 2:
                        points = pts_tensor[:, :2]
                    else:
                        points = pts_tensor
                    dx = np.gradient(points[:, 0])
                    dy = np.gradient(points[:, 1])
                    headings = np.arctan2(dy, dx)
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


