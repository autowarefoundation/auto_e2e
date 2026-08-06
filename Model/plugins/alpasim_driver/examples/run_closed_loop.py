"""Standalone Closed-Loop Simulation Example with AlpaSim & AutoE2EDriver.

Demonstrates AlpaSim entry-point discovery, 7-camera observation stream ingestion,
and closed-loop kinematic simulation over 50 steps at 10 Hz.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
import tempfile
import torch
import numpy as np
from PIL import Image

# Dynamically resolve repository root and add to sys.path (no hardcoded absolute paths)
_EXAMPLES_DIR = Path(__file__).resolve().parent
_DRIVER_DIR = _EXAMPLES_DIR.parent
_PLUGINS_DIR = _DRIVER_DIR.parent
_MODEL_DIR = _PLUGINS_DIR.parent
_REPO_ROOT = _MODEL_DIR.parent

for path in [_REPO_ROOT, _MODEL_DIR, _DRIVER_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Resolve ALPASIM_ROOT from environment variable or check .alpasim / scratch/alpasim in repo root
alpasim_root_env = os.environ.get("ALPASIM_ROOT", "")
alpasim_root = Path(alpasim_root_env) if alpasim_root_env else _REPO_ROOT / ".alpasim"

alpasim_src = alpasim_root / "src"
if alpasim_src.exists():
    for sub in ["driver", "plugins", "grpc", "utils", "controller", "physics", "runtime"]:
        p = alpasim_src / sub / "src"
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

try:
    from alpasim_driver.plugin import AutoE2EDriver
    from alpasim_driver.config import AutoE2EAlpaSimConfig
except ImportError:
    from Model.plugins.alpasim_driver.plugin import AutoE2EDriver  # type: ignore[no-redef]
    from Model.plugins.alpasim_driver.config import AutoE2EAlpaSimConfig  # type: ignore[no-redef]

try:
    from alpasim_driver.models.base import PredictionInput, DriveCommand
except ImportError:
    from dataclasses import dataclass
    from typing import Any, Dict

    @dataclass
    class PredictionInput:  # type: ignore
        camera_images: Dict[str, Any]
        command: int
        speed: float
        acceleration: float
        ego_pose_history: list
        inference_seed: int

    class DriveCommand:  # type: ignore
        STRAIGHT = 1

try:
    import alpasim_plugins.plugins as alpasim_plugins
except ImportError:
    alpasim_plugins = None  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AlpaSimClosedLoopExample")


from model_components.auto_e2e import AutoE2E  # noqa: E402


def create_checkpoint(ckpt_path: str) -> None:
    model = AutoE2E(num_views=7, is_pretrained=False)
    torch.save(model, ckpt_path)
    logger.info("Created AutoE2E model checkpoint: %s", ckpt_path)


def generate_camera_observation(step: int) -> dict[str, Image.Image]:
    """Generate 7 camera frames matching KitScenes sensor topology."""
    camera_names = [
        "camera_base_front_center",
        "camera_ring_front",
        "camera_ring_front_left",
        "camera_ring_front_right",
        "camera_ring_rear",
        "camera_ring_rear_left",
        "camera_ring_rear_right",
    ]
    cameras = {}
    for i, name in enumerate(camera_names):
        r = (30 + step * 3 + i * 20) % 256
        g = (60 + step * 2 + i * 15) % 256
        b = (100 + step * 5 + i * 10) % 256
        cameras[name] = Image.new("RGB", (256, 256), color=(r, g, b))
    return cameras


def main() -> None:
    logger.info("=" * 60)
    logger.info("Starting Closed-Loop Simulation Example")
    logger.info("=" * 60)

    if alpasim_plugins is not None:
        try:
            models_registry = alpasim_plugins.PluginRegistry("alpasim.models")
            configs_registry = alpasim_plugins.PluginRegistry("alpasim.configs")
            logger.info("AlpaSim Registered Models: %s", models_registry.get_names())
            logger.info("AlpaSim Registered Configs: %s", configs_registry.get_names())
        except Exception as e:
            logger.warning("PluginRegistry query failed: %s", e)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "autoe2e_model.ckpt")
        create_checkpoint(ckpt_path)

        cfg = AutoE2EAlpaSimConfig(checkpoint_path=ckpt_path, allow_mock=False)
        driver = AutoE2EDriver.from_config(
            cfg,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            camera_ids=[],
            context_length=1,
            output_frequency_hz=10,
        )
        logger.info("Instantiated driver plugin: %s", driver.__class__.__name__)
        logger.info("Camera topology (%d cameras): %s", len(driver.camera_ids), driver.camera_ids)

        n_steps = 50  # 5.0 seconds at 10 Hz
        dt = 0.1

        state = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "v": 10.0,
            "a": 0.0,
        }

        logger.info("-" * 60)
        logger.info("Executing 50-step closed-loop simulation loop...")
        logger.info("-" * 60)

        for step in range(n_steps):
            t_sim = step * dt
            cameras = generate_camera_observation(step)

            obs = PredictionInput(
                camera_images={cam: [type("CameraFrame", (), {"image": img})()] for cam, img in cameras.items()},
                speed=state["v"],
                acceleration=state["a"],
                command=DriveCommand.STRAIGHT,
                ego_pose_history=[],
                inference_seed=42 + step,
            )

            step_start_t = time.perf_counter()
            prediction = driver.predict(obs)
            inference_ms = (time.perf_counter() - step_start_t) * 1000.0

            traj_pts = prediction.trajectory_xy
            headings = prediction.headings

            dx_local = float(traj_pts[1, 0])
            dy_local = float(traj_pts[1, 1])
            target_heading = float(headings[1])

            cos_yaw = np.cos(state["yaw"])
            sin_yaw = np.sin(state["yaw"])
            dx_global = dx_local * cos_yaw - dy_local * sin_yaw
            dy_global = dx_local * sin_yaw + dy_local * cos_yaw

            state["x"] += dx_global
            state["y"] += dy_global
            state["yaw"] += target_heading * dt

            new_v = np.hypot(dx_local, dy_local) / dt
            state["a"] = (new_v - state["v"]) / dt
            state["v"] = new_v

            if step % 10 == 0 or step == n_steps - 1:
                logger.info(
                    f"[Step {step:02d}/{n_steps}] t={t_sim:4.1f}s | "
                    f"Ego Pos: ({state['x']:6.2f}m, {state['y']:6.2f}m) | "
                    f"Speed: {state['v']:5.2f} m/s | "
                    f"Heading: {np.degrees(state['yaw']):5.2f}° | "
                    f"Inference: {inference_ms:6.2f} ms"
                )

        logger.info("=" * 60)
        logger.info("Closed-Loop Simulation completed successfully!")
        logger.info(f"Final Ego Position: ({state['x']:.2f}m, {state['y']:.2f}m), Total Distance: {state['x']:.2f}m")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
