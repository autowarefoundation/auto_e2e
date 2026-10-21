"""AutoE2E driver plugin for AlpaSim.

AlpaSim builds the driver through ``from_config`` with the cameras listed in
``driver.inference.use_cameras``. The plugin picks the calibration profile of
that camera rig, rebuilds AutoE2E from the configuration saved in the
checkpoint, and reads the model's input contract (camera slots, tile sizes)
from that same configuration.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS
from model_components.view_fusion.projection import PinholeProjection
from reactive_training_contracts import (
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
)
from training.losses.control_rollout import integrate_controls_torch

from .config import find_rig_profile
from .parser import CONTEXT_FRAMES, AlpasimStreamParser, ParsedObservation

logger = logging.getLogger(__name__)

MOCK_CHECKPOINT = "MOCK"
"""``checkpoint_path`` that runs without a model and drives straight ahead."""
UNTRAINED_CHECKPOINT = "UNTRAINED"
"""``checkpoint_path`` that builds the release architecture with random weights."""

OUTPUT_FREQUENCY_HZ = 10
_WAYPOINT_STEP_US = 1_000_000 // OUTPUT_FREQUENCY_HZ
_SUPPORTED_ARCHITECTURE = "bevformer_v2_t8"


def load_checkpoint_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild AutoE2E from the configuration saved in its checkpoint.

    Same reconstruction as ``load_policy`` in ``Platform/pipelines/inference.py``.
    The checkpoint holds the complete backbone, so no pretrained weights are
    downloaded first.
    """
    from model_components.auto_e2e import AutoE2E

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint["config"])
    accepted = set(inspect.signature(AutoE2E.__init__).parameters) - {"self"}
    kwargs = {key: value for key, value in config.items() if key in accepted}
    kwargs["is_pretrained"] = False
    model = AutoE2E(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), config


def build_untrained_model(
    num_views: int, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the release architecture with random weights, for dry runs."""
    from model_components.auto_e2e import AutoE2E
    from training.reactive_multitask import (
        ReactiveTrainingStage,
        reactive_model_kwargs,
    )

    config = reactive_model_kwargs(ReactiveTrainingStage.NUPLAN_FULL, num_views=num_views)
    # The released checkpoints use the reactive training default backbone
    # (Platform/pipelines/distributed_training.py).
    config["backbone"] = "res_net_50"
    config["is_pretrained"] = False
    return AutoE2E(**config).to(device).eval(), config


class AutoE2EDriver(BaseTrajectoryModel):
    """AlpaSim ``BaseTrajectoryModel`` running the AutoE2E reactive policy."""

    def __init__(
        self,
        checkpoint_path: str,
        camera_ids: list[str],
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.profile = find_rig_profile(camera_ids)

        self.model: torch.nn.Module | None = None
        config: dict[str, Any] = {}
        if checkpoint_path == MOCK_CHECKPOINT:
            logger.warning("AutoE2E driver runs in mock mode: no model is loaded")
        elif checkpoint_path == UNTRAINED_CHECKPOINT:
            logger.warning("AutoE2E driver runs an untrained model")
            self.model, config = build_untrained_model(len(camera_ids), self.device)
        elif Path(checkpoint_path).is_file():
            self.model, config = load_checkpoint_model(checkpoint_path, self.device)
        else:
            raise FileNotFoundError(f"AutoE2E checkpoint {checkpoint_path!r} not found")

        view_fusion = dict(config.get("view_fusion_kwargs") or {})
        if self.model is not None and view_fusion.get("architecture") != _SUPPORTED_ARCHITECTURE:
            raise ValueError(
                "The AutoE2E driver supports the reactive BEVFormer T8 model "
                f"('{_SUPPORTED_ARCHITECTURE}'); the checkpoint uses "
                f"{view_fusion.get('architecture')!r}"
            )
        camera_slots = list(config.get("camera_slots") or CANONICAL_SIX_CAMERA_SLOTS)
        if self.model is not None and config.get("num_views") != len(camera_slots):
            raise ValueError(
                f"Checkpoint num_views={config.get('num_views')} does not match "
                f"its camera_slots {camera_slots}"
            )
        self.parser = AlpasimStreamParser(
            self.profile,
            camera_slots,
            image_size=int(view_fusion.get("image_size", REACTIVE_CAMERA_IMAGE_SIZE)),
            front_image_size=int(
                view_fusion.get("front_image_size", REACTIVE_FRONT_CAMERA_IMAGE_SIZE)
            ),
        )
        logger.info(
            "AutoE2E driver: rig profile %s, cameras %s",
            self.profile.name,
            self.parser.camera_ids,
        )

    @classmethod
    def from_config(
        cls,
        model_cfg: Any,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> AutoE2EDriver:
        if context_length not in (None, CONTEXT_FRAMES):
            raise ValueError(
                f"AutoE2E consumes {CONTEXT_FRAMES} frames per camera; "
                f"got inference.context_length={context_length}"
            )
        if output_frequency_hz != OUTPUT_FREQUENCY_HZ:
            raise ValueError(
                f"AutoE2E outputs waypoints at {OUTPUT_FREQUENCY_HZ} Hz; "
                f"got inference.output_frequency_hz={output_frequency_hz}"
            )
        return cls(
            checkpoint_path=str(model_cfg.checkpoint_path),
            camera_ids=list(camera_ids),
            device=device,
        )

    @property
    def camera_ids(self) -> list[str]:
        return self.parser.camera_ids

    @property
    def context_length(self) -> int:
        return CONTEXT_FRAMES

    @property
    def output_frequency_hz(self) -> int:
        return OUTPUT_FREQUENCY_HZ

    def _encode_command(self, command: DriveCommand) -> None:
        """AutoE2E is conditioned on a route raster, not on discrete commands."""
        return None

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        if self.model is None:
            return self._mock_prediction(prediction_input.speed)

        parsed = self.parser.parse(prediction_input)
        controls = self._run_model(parsed)
        positions, headings, _ = integrate_controls_torch(
            controls.reshape(1, -1, 2).float().cpu(),
            torch.tensor([parsed.initial_speed_mps], dtype=torch.float32),
        )
        prediction = ModelPrediction.from_planar(
            positions[0].numpy().astype(np.float64),
            headings[0].numpy().astype(np.float64),
        )
        num_waypoints = positions.shape[1]
        waypoint_timestamps_us = parsed.t0_us + _WAYPOINT_STEP_US * np.arange(
            1, num_waypoints + 1, dtype=np.uint64
        )
        return dataclasses.replace(
            prediction,
            model_t0_us=parsed.t0_us,
            pose_local_to_rig_t0=parsed.pose_local_to_rig_t0,
            waypoint_timestamps_us=waypoint_timestamps_us,
        )

    def _run_model(self, parsed: ParsedObservation) -> torch.Tensor:
        """Call AutoE2E the way ``predict_control`` in Platform/pipelines does."""
        assert self.model is not None
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in parsed.model_inputs.items()
        }
        projection = PinholeProjection(
            inputs.pop("camera_projection_matrix"), geometry_type="pinhole"
        )
        history_matrix = inputs.pop("camera_history_projection_matrix")
        history_projections = tuple(
            PinholeProjection(history_matrix[:, index], geometry_type="pinhole")
            for index in range(history_matrix.shape[1])
        )
        front_matrix = inputs.pop("front_camera_projection_matrix", None)
        front_projection = (
            PinholeProjection(front_matrix, geometry_type="pinhole")
            if front_matrix is not None
            else None
        )
        reset_visual_history = getattr(self.model, "reset_visual_history", None)
        if callable(reset_visual_history):
            reset_visual_history()
        with torch.no_grad():
            output = self.model(
                inputs["camera_tiles"],
                inputs["map_context"],
                inputs["visual_history"],
                inputs["egomotion_history"],
                route_mask=inputs["route_mask"],
                map_valid=inputs["map_valid"],
                route_valid=inputs["route_valid"],
                projection=projection,
                geometry_type="pinhole",
                camera_history_tiles=inputs["camera_history_tiles"],
                history_projections=history_projections,
                front_camera_tile=inputs.get("front_camera_tile"),
                front_projection=front_projection,
                mode="infer",
            )
        if isinstance(output, tuple):
            output = output[0]
        return output

    @staticmethod
    def _mock_prediction(speed: float) -> ModelPrediction:
        """Drive straight ahead at the current speed."""
        steps = np.arange(1, 65, dtype=np.float64) / OUTPUT_FREQUENCY_HZ
        positions = np.stack([max(speed, 1.0) * steps, np.zeros_like(steps)], axis=1)
        return ModelPrediction.from_planar(positions, np.zeros(len(steps)))


__all__ = ["AutoE2EDriver", "MOCK_CHECKPOINT", "UNTRAINED_CHECKPOINT"]
