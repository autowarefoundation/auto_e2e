"""AutoE2E as a ``carla_driver_interface`` policy.

Serve it for autoware_carla_scenario with::

    carla-driver-interface serve --policy acs_autoe2e.policy:AutoE2EPolicy \\
        --policy-arg checkpoint_path=/path/to/checkpoint.pt --port 50051

and run a scenario with the AutoE2E driver preset (six nuPlan cameras), see the
package README. ``checkpoint_path=MOCK`` drives straight ahead without a model,
to test the plumbing; ``UNTRAINED`` runs the release architecture with random
weights.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from carla_driver_interface.driver import BaseDriver, DriveContext, DriveResult, SessionState
from carla_driver_interface.geometry import Pose, Trajectory
from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS
from model_components.view_fusion.projection import PinholeProjection
from reactive_training_contracts import (
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
)
from training.losses.control_rollout import integrate_controls_torch

from .observation import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    HistoryNotReady,
    Observation,
    ObservationBuilder,
)
from .rig import dataset_profile, profile_from_session

logger = logging.getLogger(__name__)

MOCK_CHECKPOINT = "MOCK"
UNTRAINED_CHECKPOINT = "UNTRAINED"
OUTPUT_DT_S = 0.1
"""The model's 64 controls are 100 ms apart (6.4 s)."""
_SUPPORTED_ARCHITECTURE = "bevformer_v2_t8"
# Frames kept per camera: 3.5 s of history at up to 20 Hz, plus margin.
_FRAME_HISTORY = 80


def load_model(checkpoint_path: str, num_views: int, device: torch.device
               ) -> tuple[torch.nn.Module | None, dict[str, Any]]:
    """The model and its config, the way the AlpaSim plugin rebuilds them."""
    if checkpoint_path == MOCK_CHECKPOINT:
        return None, {}
    import inspect

    from model_components.auto_e2e import AutoE2E

    if checkpoint_path == UNTRAINED_CHECKPOINT:
        from training.reactive_multitask import ReactiveTrainingStage, reactive_model_kwargs

        config = reactive_model_kwargs(ReactiveTrainingStage.NUPLAN_FULL, num_views=num_views)
        config["backbone"] = "res_net_50"
        config["is_pretrained"] = False
        return AutoE2E(**config).to(device).eval(), config
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"AutoE2E checkpoint {checkpoint_path!r} not found")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint["config"])
    accepted = set(inspect.signature(AutoE2E.__init__).parameters) - {"self"}
    kwargs = {key: value for key, value in config.items() if key in accepted}
    kwargs["is_pretrained"] = False
    model = AutoE2E(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    view_fusion = dict(config.get("view_fusion_kwargs") or {})
    if view_fusion.get("architecture") != _SUPPORTED_ARCHITECTURE:
        raise ValueError(
            f"AutoE2EPolicy supports the reactive BEVFormer T8 model "
            f"({_SUPPORTED_ARCHITECTURE!r}); the checkpoint uses "
            f"{view_fusion.get('architecture')!r}"
        )
    return model.to(device).eval(), config


def straight_plan(speed_mps: float, steps: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Positions and headings of a plan straight ahead at ``speed_mps``."""
    t = np.arange(1, steps + 1, dtype=np.float64) * OUTPUT_DT_S
    return np.stack([speed_mps * t, np.zeros_like(t)], axis=1), np.zeros(steps)


class AutoE2EPolicy(BaseDriver):
    """Runs the AutoE2E reactive model on the cameras of an ACS scenario.

    Args:
        checkpoint_path: a training checkpoint, ``MOCK`` or ``UNTRAINED``.
        rig: the dataset rig whose camera slots and egomotion rule apply
            (``nuplan`` or ``kit_scenes``, or a profile file). The cameras
            themselves are taken from what the scenario declares.
        device: ``cuda`` when available, else ``cpu``.
        mock_speed_mps: speed of the ``MOCK`` plan.
    """

    name = "auto_e2e"
    frame_history_length = _FRAME_HISTORY

    def __init__(self, checkpoint_path: str = MOCK_CHECKPOINT, rig: str = "nuplan",
                 device: str | None = None, mock_speed_mps: float = 5.0) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dataset = dataset_profile(rig)
        self.model, config = load_model(str(checkpoint_path), len(self.dataset.cameras),
                                        self.device)
        self.camera_slots = list(config.get("camera_slots") or CANONICAL_SIX_CAMERA_SLOTS)
        view_fusion = dict(config.get("view_fusion_kwargs") or {})
        self.image_size = int(view_fusion.get("image_size", REACTIVE_CAMERA_IMAGE_SIZE))
        self.front_image_size = int(
            view_fusion.get("front_image_size", REACTIVE_FRONT_CAMERA_IMAGE_SIZE)
        )
        self.mock_speed_mps = float(mock_speed_mps)
        #: AUTOE2E_ACS_DEBUG_DIR: every AUTOE2E_ACS_DEBUG_EVERY-th prediction writes
        #: its camera tiles, egomotion and plan there.
        debug_dir = os.environ.get("AUTOE2E_ACS_DEBUG_DIR")
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.debug_every = int(os.environ.get("AUTOE2E_ACS_DEBUG_EVERY", "10"))
        self._builders: dict[str, ObservationBuilder] = {}
        logger.info("AutoE2EPolicy: checkpoint %s, %s rig, device %s",
                    checkpoint_path, self.dataset.name, self.device)

    # -- session ------------------------------------------------------------

    def on_session_start(self, session: SessionState) -> None:
        tile_hw = (self.image_size, self.image_size)
        profile, views = profile_from_session(session.cameras, self.dataset, tile_hw)
        builder = ObservationBuilder(
            profile, self.camera_slots, self.image_size, self.front_image_size, views
        )
        for logical_id, view in views.items():
            _, coverage = view.grids(tile_hw)
            logger.info("%s: drawn from F-theta %s (%.0f%% of the view)",
                        logical_id, " + ".join(view.source_ids), 100 * coverage)
        self._builders[session.uuid] = builder

    def on_session_close(self, session: SessionState) -> None:
        self._builders.pop(session.uuid, None)

    # -- planning -----------------------------------------------------------

    def drive(self, ctx: DriveContext) -> DriveResult:
        session = ctx.session
        if self.model is None:
            positions, headings = straight_plan(max(session.speed_mps(), self.mock_speed_mps))
            return DriveResult(
                trajectory_in_rig=self._plan(positions, headings, ctx.time_now_us),
                debug_scalars={"mock": 1.0},
            )
        try:
            observation = self._builders[session.uuid].build(session)
        except HistoryNotReady as reason:
            # Still warming up: hold still rather than guess (ACS's driver.warmup_s
            # lets the history fill before the scenario's clock starts).
            logger.info("drive %d: history not ready (%s)", session.drive_count, reason)
            positions, headings = straight_plan(0.0)
            return DriveResult(
                trajectory_in_rig=self._plan(positions, headings, ctx.time_now_us),
                debug_scalars={"history_ready": 0.0},
            )
        started = time.perf_counter()
        controls = self._run_model(observation)
        inference_ms = 1000.0 * (time.perf_counter() - started)
        positions_t, headings_t, _ = integrate_controls_torch(
            controls.reshape(1, -1, 2).float().cpu(),
            torch.tensor([observation.initial_speed_mps], dtype=torch.float32),
            dt=OUTPUT_DT_S,
        )
        positions = positions_t[0].numpy().astype(np.float64)
        plan_t0 = self._plan(positions, headings_t[0].numpy().astype(np.float64),
                             observation.t0_us)
        logger.info(
            "drive %d: t0 lag %.0f ms, v0 %.2f m/s, plan end (%.1f, %.1f) m at 6.4 s, "
            "(%.1f, %.1f) m at 3 s, inference %.0f ms",
            session.drive_count, (ctx.time_now_us - observation.t0_us) / 1000.0,
            observation.initial_speed_mps, positions[-1, 0], positions[-1, 1],
            positions[29, 0], positions[29, 1], inference_ms,
        )
        if self.debug_dir is not None and session.drive_count % self.debug_every == 0:
            self._dump(session, observation, positions)
        return DriveResult(
            trajectory_in_rig=self._to_rig_at(plan_t0, observation, session, ctx.time_now_us),
            debug_scalars={
                "history_ready": 1.0,
                "t0_lag_ms": (ctx.time_now_us - observation.t0_us) / 1000.0,
                "initial_speed_mps": observation.initial_speed_mps,
            },
        )

    def _dump(self, session: SessionState, observation: Observation,
              positions: np.ndarray) -> None:
        """Write one prediction's inputs and plan to ``debug_dir``."""
        from PIL import Image

        assert self.debug_dir is not None
        out = self.debug_dir / f"{session.uuid[:8]}_{session.drive_count:04d}"
        out.mkdir(parents=True, exist_ok=True)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tiles = observation.model_inputs["camera_tiles"][0]
        row = torch.cat([((tile * std + mean).clamp(0, 1)) for tile in tiles], dim=2)
        Image.fromarray((row.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
            out / "tiles.jpg", quality=85)
        # the frames as the scenario sent them (the fisheye frames for Cosmos3)
        with session.lock:
            latest = {cid: frames[-1] for cid, frames in session.frame_history.items() if frames}
        for cid, frame in latest.items():
            image_bytes = getattr(frame, "image_bytes", None)
            if image_bytes is None:  # a LiDAR sweep
                continue
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.convert("RGB").save(out / f"raw_{cid}.jpg", quality=90)
        np.savez(out / "inputs.npz",
                 egomotion=observation.model_inputs["egomotion_history"].numpy(),
                 projection=observation.model_inputs["camera_projection_matrix"].numpy(),
                 plan=positions, v0=observation.initial_speed_mps)

    @staticmethod
    def _plan(positions: np.ndarray, headings: np.ndarray, t0_us: int) -> Trajectory:
        plan = Trajectory.empty()
        step_us = int(OUTPUT_DT_S * 1_000_000)
        for index, ((x, y), yaw) in enumerate(zip(positions, headings), start=1):
            plan.append(t0_us + index * step_us, Pose.from_xyz_yaw(float(x), float(y), 0.0,
                                                                   float(yaw)))
        return plan

    @staticmethod
    def _to_rig_at(plan_t0: Trajectory, observation: Observation, session: SessionState,
                   time_now_us: int) -> Trajectory:
        """Re-express a plan made in the rig frame at ``t0`` in the rig frame now."""
        if time_now_us == observation.t0_us:
            return plan_t0
        with session.lock:
            local_from_rig_now = session.ego_trajectory.interpolate(time_now_us)
        if local_from_rig_now is None:
            return plan_t0
        delta = local_from_rig_now.inverse() @ observation.local_from_rig_t0
        return plan_t0.transform(delta)

    def _run_model(self, observation: Observation) -> torch.Tensor:
        """Call AutoE2E the way ``predict_control`` in Platform/pipelines does."""
        assert self.model is not None
        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value
                  for key, value in observation.model_inputs.items()}
        projection = PinholeProjection(inputs.pop("camera_projection_matrix"),
                                       geometry_type="pinhole")
        history_matrix = inputs.pop("camera_history_projection_matrix")
        history_projections = tuple(
            PinholeProjection(history_matrix[:, index], geometry_type="pinhole")
            for index in range(history_matrix.shape[1])
        )
        front_matrix = inputs.pop("front_camera_projection_matrix", None)
        front_projection = (PinholeProjection(front_matrix, geometry_type="pinhole")
                            if front_matrix is not None else None)
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


__all__ = ["AutoE2EPolicy", "MOCK_CHECKPOINT", "UNTRAINED_CHECKPOINT",
           "REACTIVE_BEVFORMER_FRAME_INTERVAL_US"]
