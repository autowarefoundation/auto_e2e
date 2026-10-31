"""Export the KITScenes camera rig into the plugin's calibration profile.

The profile stores native intrinsics and camera-to-reference extrinsics, so the
plugin can build the same projections as ``data_parsing/kit_scenes/camera.py``
for any tile size and for the T8 history.

Usage (with the ``kitscenes`` SDK installed):
    python Model/plugins/alpasim_driver/tools/export_kitscenes_calibration.py \
        /path/to/KITScenes/data/val/<scene_id> \
        Model/plugins/alpasim_driver/alpasim_autoe2e/configs/calibration/kit_scenes.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from kitscenes.sensors import SensorDataLoader

# Slot order of Model/data_parsing/kit_scenes/camera.py (CAMERA_NAMES).
CAMERAS = (
    ("camera_base_front_center", "front"),
    ("camera_ring_front_left", "front_left"),
    ("camera_ring_front_right", "front_right"),
    ("camera_ring_rear_left", "rear_left"),
    ("camera_ring_rear", "rear"),
    ("camera_ring_rear_right", "rear_right"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_path", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    loader = SensorDataLoader(args.scene_path)
    cameras = []
    for logical_id, slot in CAMERAS:
        calibration = loader.get_camera_calibration(logical_id)
        width, height = calibration.image_size
        cameras.append(
            {
                "logical_id": logical_id,
                "slot": slot,
                "image_size_wh": [int(width), int(height)],
                "intrinsic": np.asarray(calibration.intrinsic, dtype=float).tolist(),
                "camera_to_reference": np.asarray(
                    calibration.extrinsic, dtype=float
                ).tolist(),
            }
        )

    profile = {
        "dataset_name": "KIT-MRT/KITScenes-Multimodal",
        "source": f"kitscenes SensorDataLoader calibration of scene {args.scene_path.name}",
        "reference_frame": (
            "KITScenes ego reference frame (the frame of poses.txt). The AlpaSim "
            "rig frame of the rendered world must match it."
        ),
        # KITScenes shards pack the front camera at 1024 px (front_camera_tile).
        "native_front_camera_tile": True,
        # data_parsing/kit_scenes/egomotion.py: yaw_rate / max(speed, 0.1).
        "egomotion": {
            "curvature_speed_floor_mps": 0.1,
            "curvature_zero_below_mps": None,
            "curvature_limit": None,
        },
        "cameras": cameras,
    }
    text = json.dumps(profile, indent=2)
    # One line per matrix row or size pair keeps the matrices readable.
    text = re.sub(
        r"\[\s+([-0-9.e, \n]+?)\s+\]",
        lambda match: "[" + ", ".join(match.group(1).replace(",", " ").split()) + "]",
        text,
    )
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
