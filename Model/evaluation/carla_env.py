"""Gymnasium environment wrapping CARLA for RL training of AutoE2E.

Observation: same dict as training DataLoader (visual_tiles, egomotion_history, visual_history)
Action: [acceleration, curvature] continuous
Reward: progress + safety penalties

Usage:
    env = CarlaEnv(carla_host="carla-server", carla_port=2000)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CarlaEnv(gym.Env):
    """Gym wrapper for CARLA closed-loop driving."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        carla_host: str = "carla-server",
        carla_port: int = 2000,
        town: str = "Town01",
        max_steps: int = 600,  # 60s at 10Hz
        num_cameras: int = 7,
    ):
        super().__init__()
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.town = town
        self.max_steps = max_steps
        self.num_cameras = num_cameras

        # Action: [acceleration (-5 to +3 m/s²), curvature (-0.1 to +0.1 1/m)]
        self.action_space = spaces.Box(
            low=np.array([-5.0, -0.1], dtype=np.float32),
            high=np.array([3.0, 0.1], dtype=np.float32),
        )

        # Observation: flat dict matching model input
        self.observation_space = spaces.Dict({
            "visual_tiles": spaces.Box(0, 1, shape=(num_cameras, 3, 256, 256), dtype=np.float32),
            "egomotion_history": spaces.Box(-np.inf, np.inf, shape=(256,), dtype=np.float32),
            "visual_history": spaces.Box(-np.inf, np.inf, shape=(896,), dtype=np.float32),
        })

        self._client = None
        self._world = None
        self._ego = None
        self._cameras = []
        self._collision_sensor = None
        self._step_count = 0
        self._collisions = 0
        self._total_distance = 0.0
        self._prev_location = None
        self._ego_history = np.zeros((64, 4), dtype=np.float32)
        self._camera_data = [None] * num_cameras

    def _connect(self):
        import carla
        self._client = carla.Client(self.carla_host, self.carla_port)
        self._client.set_timeout(30.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        import carla

        if self._client is None:
            self._connect()

        # Load world
        self._world = self._client.load_world(self.town)
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self._world.apply_settings(settings)

        # Spawn ego
        bp_lib = self._world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_points = self._world.get_map().get_spawn_points()
        spawn = self.np_random.choice(spawn_points)
        self._ego = self._world.spawn_actor(vehicle_bp, spawn)

        # Cameras (front only for now, replicated to 7)
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "256")
        cam_bp.set_attribute("image_size_y", "256")
        cam_bp.set_attribute("fov", "120")
        cam = self._world.spawn_actor(
            cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=self._ego
        )
        cam.listen(lambda img: self._camera_data.__setitem__(0, img))
        self._cameras = [cam]

        # Collision sensor
        col_bp = bp_lib.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self._ego
        )
        self._collisions = 0
        self._collision_sensor.listen(lambda _: setattr(self, '_collisions', self._collisions + 1))

        # Reset state
        self._step_count = 0
        self._total_distance = 0.0
        self._prev_location = self._ego.get_location()
        self._ego_history = np.zeros((64, 4), dtype=np.float32)

        self._world.tick()
        return self._get_obs(), {}

    def step(self, action):
        import carla

        accel, curvature = float(action[0]), float(action[1])

        # Apply control
        if accel >= 0:
            throttle, brake = min(accel / 3.0, 1.0), 0.0
        else:
            throttle, brake = 0.0, min(-accel / 5.0, 1.0)

        vel = self._ego.get_velocity()
        speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        steer = float(np.clip(np.arctan(curvature * 2.9) / (np.pi / 4), -1, 1))

        self._ego.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=steer
        ))
        self._world.tick()
        self._step_count += 1

        # Update state
        vel = self._ego.get_velocity()
        speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        acc = self._ego.get_acceleration()
        yaw_rate = self._ego.get_angular_velocity().z * np.pi / 180
        curv = yaw_rate / max(speed, 0.1)

        self._ego_history = np.roll(self._ego_history, -1, axis=0)
        self._ego_history[-1] = [speed, acc.x, yaw_rate, curv]

        loc = self._ego.get_location()
        step_dist = loc.distance(self._prev_location)
        self._total_distance += step_dist
        self._prev_location = loc

        # Reward
        reward = self._compute_reward(step_dist, speed, acc.x)

        # Termination
        terminated = self._collisions > 0
        truncated = self._step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "distance": self._total_distance,
            "collisions": self._collisions,
            "speed": speed,
        }

    def _compute_reward(self, step_dist: float, speed: float, accel: float) -> float:
        reward = step_dist * 1.0  # progress
        reward -= self._collisions * 10.0  # collision penalty
        # Offroad check (simplified: speed = 0 for many steps → stuck)
        if speed < 0.5 and self._step_count > 50:
            reward -= 0.5
        # Comfort: penalize high jerk
        if len(self._ego_history) >= 2:
            jerk = abs(self._ego_history[-1, 1] - self._ego_history[-2, 1]) / 0.1
            if jerk > 2.5:
                reward -= 0.1
        return reward

    def _get_obs(self) -> dict:
        # Camera → tensor (replicate single camera to 7 for now)
        if self._camera_data[0] is not None:
            raw = np.frombuffer(self._camera_data[0].raw_data, dtype=np.uint8)
            img = raw.reshape(256, 256, 4)[:, :, :3].astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # HWC → CHW
        else:
            img = np.zeros((3, 256, 256), dtype=np.float32)

        visual_tiles = np.stack([img] * self.num_cameras)

        return {
            "visual_tiles": visual_tiles,
            "egomotion_history": self._ego_history.flatten(),
            "visual_history": np.zeros(896, dtype=np.float32),
        }

    def close(self):
        if self._ego:
            for cam in self._cameras:
                cam.stop()
                cam.destroy()
            if self._collision_sensor:
                self._collision_sensor.stop()
                self._collision_sensor.destroy()
            self._ego.destroy()
        if self._world:
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            self._world.apply_settings(settings)
