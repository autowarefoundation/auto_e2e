"""Write the ACS driver preset that renders a dataset rig in CARLA.

ACS mounts each camera with a ``carla.Transform`` relative to the vehicle
actor (x forward, y right, z up; roll, pitch, yaw in degrees, yaw to the
right) and renders an ideal pinhole with square pixels and a centred
principal point. A rig profile (``alpasim_autoe2e/configs/calibration``) gives
camera-to-rig extrinsics in the rig frame (rear axle, x forward, y left,
z up; the camera in its optical frame) and native intrinsics.

For each camera the preset keeps the mount and the angular field of view:

* position: the rig position with y mirrored, shifted by the rear-axle offset
  to the actor origin (``--rear-axle-x``, also written as
  ``driver.rear_axle_offset_m`` so ACS reports poses for the same rig origin);
* orientation: yaw, pitch and roll of the camera body;
* image: the native width and ``fov = 2 atan(w / 2 fx)``, and a height that
  keeps the native vertical field of view, ``h = 2 fx tan(atan(h0 / 2 fy))``.
  nuPlan's rectified pixels are not square (fy > fx), so the CARLA image is
  shorter than 1080; squashed to the model's square tiles, both give the same
  angle per tile pixel.

With ``--cosmos-lenses`` the cameras are instead the Hyperion 8 cameras the
Cosmos3 world model generates (cosmos3-canvas), each declared with its F-theta
lens as ``python -m cosmos3_canvas.camera <camera> --format json`` prints it
(one JSON object per line in the file). Each is mounted at the dataset camera
it stands in for (``HYPERION_MOUNTS``); the policy draws the dataset's pinhole
views from them, the rear one from both rear cameras and the rear tele.

Usage::

    python tools/acs_preset.py nuplan > autoe2e_nuplan.yaml
    python tools/acs_preset.py nuplan --cosmos-lenses lenses.jsonl > autoe2e_nuplan_cosmos.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acs_autoe2e  # noqa: E402,F401  (puts Model/ and the AlpaSim plugin on sys.path)
from acs_autoe2e.rig import dataset_profile  # noqa: E402

# nuPlan's vehicle is a Lincoln MKZ; CARLA 0.10 has its model.
DEFAULT_VEHICLE = "vehicle.lincoln.mkz"
DEFAULT_WHEELBASE_M = 2.85

#: The nuPlan camera each Hyperion 8 camera is mounted at, for the Cosmos3 rig.
HYPERION_MOUNTS = {
    "camera_front_wide_120fov": "CAM_F0",
    "camera_front_tele_30fov": "CAM_F0",
    "camera_cross_left_120fov": "CAM_L0",
    "camera_cross_right_120fov": "CAM_R0",
    "camera_rear_left_70fov": "CAM_L2",
    "camera_rear_right_70fov": "CAM_R2",
    "camera_rear_tele_30fov": "CAM_B0",
}


def carla_mount(camera_to_reference: np.ndarray, rear_axle_x: float) -> dict[str, float]:
    """``position_*``, ``roll``, ``pitch``, ``yaw`` of one camera for ACS."""
    optical = camera_to_reference[:3, :3]
    # camera body (x along the optical axis, z up) in the rig frame
    forward, up = optical[:, 2], -optical[:, 1]
    yaw_left = math.atan2(forward[1], forward[0])
    pitch = math.asin(max(-1.0, min(1.0, forward[2])))
    # roll: the body's up axis turned about the optical axis, from the level up axis
    level_left = np.array([-math.sin(yaw_left), math.cos(yaw_left), 0.0])
    level_up = np.cross(forward, level_left)
    roll_left = math.atan2(float(np.dot(np.cross(level_up, up), forward)), float(np.dot(level_up, up)))
    x, y, z = camera_to_reference[:3, 3]
    return {
        "position_x": round(float(x) + rear_axle_x, 4),
        "position_y": round(-float(y), 4),
        "position_z": round(float(z), 4),
        # CARLA: yaw positive to the right, pitch positive up, roll positive with the
        # right side down -- which is the right-handed roll about x unchanged (ACS's
        # camera_extrinsics_to_rig keeps the roll and negates yaw and pitch)
        "roll": round(math.degrees(roll_left), 3),
        "pitch": round(math.degrees(pitch), 3),
        "yaw": round(-math.degrees(yaw_left), 3),
    }


def carla_image(intrinsic: np.ndarray, size_wh: tuple[int, int]) -> dict[str, float]:
    width, height = size_wh
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    vertical_half = math.atan(height / (2.0 * fy))
    carla_height = int(round(2.0 * fx * math.tan(vertical_half)))
    return {
        "image_width": int(width),
        "image_height": carla_height + (carla_height % 2),
        "fov": round(math.degrees(2.0 * math.atan(width / (2.0 * fx))), 4),
    }


def _yaml_value(value) -> str:
    return json.dumps(value) if isinstance(value, (list, str)) else str(value)


def preset_yaml(rig: str, vehicle: str, rear_axle_x: float, wheelbase: float,
                fps: float, warmup_s: float, cosmos_lenses: list[dict] | None = None) -> str:
    profile = dataset_profile(rig)
    name = f"autoe2e_{profile.name}" + ("_cosmos" if cosmos_lenses else "")
    lines = [
        "# @package _global_",
        "#",
        f"# AutoE2E driving the ego with the {profile.name} rig, generated by",
        "# Model/plugins/carla_driver/tools/acs_preset.py in autowarefoundation/auto_e2e:",
        "#",
        "#   carla-driver-interface serve --policy acs_autoe2e.policy:AutoE2EPolicy \\",
        "#       --policy-arg checkpoint_path=<checkpoint> --port 50051",
        f"#   scenario driver={name} driver.address=localhost:50051"
        + (" canvas.config=<cosmos3 canvas.yaml>" if cosmos_lenses else ""),
        "#",
        ("# The Hyperion 8 cameras Cosmos3 generates, at the dataset cameras' mounts; see the tool."
         if cosmos_lenses else
         "# Each camera keeps the dataset mount and angular field of view; see the tool."),
        "defaults:",
        "  - default",
        "  - _self_",
        "",
        "ego:",
        f"  vehicle_type: {vehicle}",
        "  entity: carla_driver",
        "",
        "driver:",
        "  # AutoE2E plans at 10 Hz from frames 500 ms apart.",
        "  policy_timestep_s: 0.1",
        "  # 6.4 s of egomotion and 3.5 s of camera history before it drives.",
        f"  warmup_s: {warmup_s}",
        f"  rear_axle_offset_m: {rear_axle_x}",
        "  cameras:",
    ]
    by_id = {camera.logical_id: camera for camera in profile.cameras}
    if cosmos_lenses:
        entries = []
        for lens_entry in cosmos_lenses:
            key = lens_entry["logical_id"]
            mount = by_id[HYPERION_MOUNTS[key]].camera_to_reference
            entries.append({**{k: v for k, v in lens_entry.items() if k != "lens"}, "fps": fps,
                            **carla_mount(mount, rear_axle_x), "lens": lens_entry["lens"]})
    else:
        entries = [{"logical_id": camera.logical_id,
                    **carla_image(camera.intrinsic, camera.image_size_wh),
                    "fps": fps,
                    **carla_mount(camera.camera_to_reference, rear_axle_x)}
                   for camera in profile.cameras]
    for entry in entries:
        lines.append(f"    - logical_id: {entry.pop('logical_id')}")
        for key, value in entry.items():
            if isinstance(value, dict):
                lines.append(f"      {key}:")
                lines += [f"        {k}: {_yaml_value(v)}" for k, v in value.items()]
            else:
                lines.append(f"      {key}: {_yaml_value(value)}")
    lines += ["  control:", f"    wheelbase_m: {wheelbase}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("rig", nargs="?", default="nuplan",
                        help="shipped rig profile name or a profile JSON (default: nuplan)")
    parser.add_argument("--vehicle", default=DEFAULT_VEHICLE)
    parser.add_argument("--wheelbase", type=float, default=DEFAULT_WHEELBASE_M)
    parser.add_argument("--rear-axle-x", type=float, default=None,
                        help="rear axle x in the actor frame (default: -wheelbase / 2)")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--warmup-s", type=float, default=7.0)
    parser.add_argument("--cosmos-lenses", type=Path, default=None,
                        help="Hyperion camera entries from cosmos3_canvas.camera, one JSON per line")
    args = parser.parse_args()
    lenses = None
    if args.cosmos_lenses is not None:
        lenses = [json.loads(line) for line in args.cosmos_lenses.read_text().splitlines()
                  if line.strip().startswith("{")]
    rear_axle_x = args.rear_axle_x if args.rear_axle_x is not None else -args.wheelbase / 2.0
    sys.stdout.write(preset_yaml(args.rig, args.vehicle, rear_axle_x, args.wheelbase,
                                 args.fps, args.warmup_s, lenses))


if __name__ == "__main__":
    main()
