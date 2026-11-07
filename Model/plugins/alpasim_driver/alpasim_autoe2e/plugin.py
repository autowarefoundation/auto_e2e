"""AutoE2E driver plugin for AlpaSim.

AlpaSim builds the driver through ``from_config`` with the cameras listed in
``driver.inference.use_cameras``. The plugin picks the calibration profile of
that camera rig, loads a stage 2 AutoE2E checkpoint written by
``training.trainer``, and feeds it the six cameras in the canonical slot order.
"""

from __future__ import annotations

import dataclasses
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
from training.losses.control_rollout import integrate_controls_torch

from .config import find_rig_profile
from .parser import CONTEXT_FRAMES, AlpasimStreamParser, ParsedObservation

logger = logging.getLogger(__name__)

MOCK_CHECKPOINT = "MOCK"
"""``checkpoint_path`` that runs without a model and drives straight ahead."""
UNTRAINED_CHECKPOINT = "UNTRAINED"
"""``checkpoint_path`` that builds AutoE2E with random weights."""

OUTPUT_FREQUENCY_HZ = 10
_WAYPOINT_STEP_US = 1_000_000 // OUTPUT_FREQUENCY_HZ


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    """Load a stage 2 AutoE2E checkpoint; it holds every weight, so nothing is downloaded."""
    from model_components import AutoE2E
    from model_components.autobev import check_autobev_contract

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = state["metadata"]
    check_autobev_contract(metadata.get("autobev_contract"), source=str(checkpoint_path))
    if metadata["stage"] != 2:
        raise ValueError(
            f"{checkpoint_path} is a stage {metadata['stage']} checkpoint; "
            "the driver needs a stage 2 AutoE2E checkpoint"
        )
    model = AutoE2E(pretrained_backbone=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def build_untrained_model(device: torch.device) -> torch.nn.Module:
    """Build AutoE2E with random weights, for dry runs."""
    from model_components import AutoE2E

    return AutoE2E(pretrained_backbone=False).to(device).eval()


class AutoE2EDriver(BaseTrajectoryModel):
    """AlpaSim ``BaseTrajectoryModel`` running the AutoE2E policy."""

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
        if checkpoint_path == MOCK_CHECKPOINT:
            logger.warning("AutoE2E driver runs in mock mode: no model is loaded")
        elif checkpoint_path == UNTRAINED_CHECKPOINT:
            logger.warning("AutoE2E driver runs an untrained model")
            self.model = build_untrained_model(self.device)
        elif Path(checkpoint_path).is_file():
            self.model = load_checkpoint_model(checkpoint_path, self.device)
        else:
            raise FileNotFoundError(f"AutoE2E checkpoint {checkpoint_path!r} not found")

        self.parser = AlpasimStreamParser(self.profile, CANONICAL_SIX_CAMERA_SLOTS)
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
        """Run ``AutoE2E.forward`` on the parsed inputs and return its ``[1, 128]`` controls."""
        assert self.model is not None
        inputs = {key: value.to(self.device) for key, value in parsed.model_inputs.items()}
        with torch.no_grad():
            controls, _ = self.model(**inputs)
        return controls

    @staticmethod
    def _mock_prediction(speed: float) -> ModelPrediction:
        """Drive straight ahead at the current speed."""
        steps = np.arange(1, 65, dtype=np.float64) / OUTPUT_FREQUENCY_HZ
        positions = np.stack([max(speed, 1.0) * steps, np.zeros_like(steps)], axis=1)
        return ModelPrediction.from_planar(positions, np.zeros(len(steps)))


__all__ = ["AutoE2EDriver", "MOCK_CHECKPOINT", "UNTRAINED_CHECKPOINT"]
