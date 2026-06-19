"""Closed-loop CARLA evaluation runner for AutoE2E.

Connects to a CARLA server, loads a model checkpoint, runs scenarios,
and collects driving metrics (route completion, collisions, comfort).

Usage:
    python -m evaluation.closed_loop_runner \
        --carla-host carla-server \
        --checkpoint /tmp/ckpt/epoch_19.pt \
        --scenarios town01_straight,town03_intersection \
        --output /tmp/results.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class ScenarioResult:
    scenario_id: str
    route_completion: float = 0.0
    collisions: int = 0
    red_light_violations: int = 0
    distance_km: float = 0.0
    duration_s: float = 0.0
    max_jerk: float = 0.0
    max_lat_accel: float = 0.0
    timed_out: bool = False
    success: bool = False


@dataclass
class ScenarioConfig:
    id: str
    town: str = "Town01"
    spawn_index: int = 0
    destination_index: int = 50
    max_duration_s: float = 60.0
    traffic_vehicles: int = 0
    traffic_pedestrians: int = 0


# Default scenario suite
DEFAULT_SCENARIOS = [
    ScenarioConfig(id="S01", town="Town01", spawn_index=0, destination_index=50),
    ScenarioConfig(id="S02", town="Town01", spawn_index=10, destination_index=60, traffic_vehicles=5),
    ScenarioConfig(id="S03", town="Town03", spawn_index=0, destination_index=30, traffic_vehicles=10),
]


def load_model(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load AutoE2E model from checkpoint."""
    import sys
    sys.path.insert(0, "/workspace/Model")
    from model_components.auto_e2e import AutoE2E

    model = AutoE2E(
        backbone="swin_v2_tiny", num_views=7, embed_dim=256,
        fusion_mode="concat", is_pretrained=False,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model


def accel_to_throttle_brake(accel: float) -> tuple[float, float]:
    """Convert acceleration signal to CARLA throttle/brake."""
    if accel >= 0:
        return min(accel / 3.0, 1.0), 0.0
    else:
        return 0.0, min(-accel / 5.0, 1.0)


def curvature_to_steer(curvature: float, speed: float, wheelbase: float = 2.9) -> float:
    """Convert curvature to CARLA steering [-1, 1]."""
    # steer = atan(curvature * wheelbase) normalized to [-1, 1]
    steer = np.arctan(curvature * wheelbase) / (np.pi / 4)
    return float(np.clip(steer, -1.0, 1.0))


def run_scenario(
    carla_host: str,
    carla_port: int,
    model: torch.nn.Module,
    config: ScenarioConfig,
    device: str = "cpu",
) -> ScenarioResult:
    """Run a single CARLA scenario. Returns metrics."""
    import carla
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Connect
    client = carla.Client(carla_host, carla_port)
    client.set_timeout(30.0)
    world = client.load_world(config.town)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # 10Hz
    world.apply_settings(settings)

    # Spawn ego vehicle
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    ego = world.spawn_actor(vehicle_bp, spawn_points[config.spawn_index])

    # Attach cameras (simplified: 1 front camera for now)
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "256")
    cam_bp.set_attribute("image_size_y", "256")
    cam_bp.set_attribute("fov", "120")
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)

    # Camera data buffer
    camera_data = [None]
    camera.listen(lambda img: camera_data.__setitem__(0, img))

    # Collision sensor
    collision_bp = bp_lib.find("sensor.other.collision")
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
    collisions = []
    collision_sensor.listen(lambda event: collisions.append(event))

    # Run loop
    ego_history = np.zeros((64, 4), dtype=np.float32)
    result = ScenarioResult(scenario_id=config.id)
    start_time = time.time()
    total_distance = 0.0
    prev_location = None
    prev_speed = 0.0

    try:
        for step in range(int(config.max_duration_s / 0.1)):
            world.tick()

            # Get ego state
            vel = ego.get_velocity()
            speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            accel_vec = ego.get_acceleration()
            accel_x = accel_vec.x
            yaw_rate = ego.get_angular_velocity().z * np.pi / 180
            curvature = yaw_rate / max(speed, 0.1)

            # Update ego history (rolling buffer)
            ego_history = np.roll(ego_history, -1, axis=0)
            ego_history[-1] = [speed, accel_x, yaw_rate, curvature]

            # Distance
            loc = ego.get_location()
            if prev_location:
                total_distance += loc.distance(prev_location)
            prev_location = loc

            # Camera frame → model input
            if camera_data[0] is not None:
                raw = np.frombuffer(camera_data[0].raw_data, dtype=np.uint8)
                img = raw.reshape(256, 256, 4)[:, :, :3]
                frame = transform(Image.fromarray(img))
                # Replicate to 7 views (simplified — real impl uses 7 cameras)
                visual_tiles = frame.unsqueeze(0).repeat(1, 7, 1, 1, 1).to(device)
            else:
                visual_tiles = torch.zeros(1, 7, 3, 256, 256, device=device)

            ego_hist_tensor = torch.from_numpy(ego_history.flatten()).unsqueeze(0).to(device)
            vis_hist = torch.zeros(1, 896, device=device)

            # Inference
            with torch.no_grad():
                trajectory, _, _ = model(visual_tiles, vis_hist, ego_hist_tensor, mode="eval")

            # Extract next-step control
            pred = trajectory[0].cpu().numpy().reshape(64, 2)
            next_accel = float(pred[0, 0])
            next_curv = float(pred[0, 1])

            throttle, brake = accel_to_throttle_brake(next_accel)
            steer = curvature_to_steer(next_curv, speed)
            ego.apply_control(carla.VehicleControl(
                throttle=throttle, brake=brake, steer=steer
            ))

            # Jerk
            jerk = abs(accel_x - (ego_history[-2, 1] if step > 0 else 0)) / 0.1
            result.max_jerk = max(result.max_jerk, jerk)
            result.max_lat_accel = max(result.max_lat_accel, abs(curvature * speed**2))

            prev_speed = speed

    except Exception as e:
        print(f"Scenario {config.id} error: {e}")
    finally:
        elapsed = time.time() - start_time
        result.duration_s = elapsed
        result.distance_km = total_distance / 1000
        result.collisions = len(collisions)
        result.route_completion = min(total_distance / 200.0, 1.0)  # approximate
        result.success = result.collisions == 0 and result.route_completion >= 0.9

        # Cleanup
        camera.stop()
        collision_sensor.stop()
        ego.destroy()
        camera.destroy()
        collision_sensor.destroy()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenarios", default="S01")
    parser.add_argument("--output", default="/tmp/closed_loop_results.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    scenario_ids = args.scenarios.split(",")

    results = []
    for config in DEFAULT_SCENARIOS:
        if config.id in scenario_ids:
            print(f"Running scenario {config.id} ({config.town})...")
            result = run_scenario(args.carla_host, args.carla_port, model, config, args.device)
            print(f"  Done: completion={result.route_completion:.0%} collisions={result.collisions}")
            results.append(result.__dict__)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
