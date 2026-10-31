"""Turn a rig profile into AlpaSim ``runtime.extra_cameras`` entries.

AlpaSim's ``extra_cameras`` override the pose (``rig_to_camera``: camera pose in
the rig frame, OpenCV camera axes), intrinsics and resolution of cameras the
scene already has. Rendering a profile's cameras this way keeps the rendered
geometry identical to the projections the plugin feeds the model. ``--rename``
puts the profile's cameras on scene camera IDs (AlpaSim's Transfuser plugin
renders its nuPlan-style rig the same way) and writes the matching profile:

    python Model/plugins/alpasim_driver/tools/profile_to_extra_cameras.py \
        Model/plugins/alpasim_driver/alpasim_autoe2e/configs/calibration/nuplan.json \
        --rename CAM_F0=camera_front_wide_120fov,CAM_L0=camera_cross_left_120fov,... \
        --profile-out Model/plugins/alpasim_driver/alpasim_autoe2e/configs/calibration/nuplan_nurec.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def extra_cameras(profile: dict) -> list[dict]:
    entries = []
    for camera in profile["cameras"]:
        camera_to_rig = np.asarray(camera["camera_to_reference"], dtype=np.float64)
        intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64)
        width, height = camera["image_size_wh"]
        entries.append(
            {
                "logical_id": camera["logical_id"],
                "rig_to_camera": {
                    "translation_m": camera_to_rig[:3, 3].round(6).tolist(),
                    "rotation_xyzw": Rotation.from_matrix(camera_to_rig[:3, :3])
                    .as_quat()
                    .round(6)
                    .tolist(),
                },
                "intrinsics": {
                    "model": "opencv_pinhole",
                    "opencv_pinhole": {
                        "focal_length": [round(intrinsic[0, 0], 4), round(intrinsic[1, 1], 4)],
                        "principal_point": [round(intrinsic[0, 2], 4), round(intrinsic[1, 2], 4)],
                        "radial": [0.0] * 6,
                        "tangential": [0.0, 0.0],
                        "thin_prism": [0.0] * 4,
                    },
                },
                "resolution_hw": [height, width],
                "shutter_type": "ROLLING_TOP_TO_BOTTOM",
            }
        )
    return entries


def renamed(profile: dict, mapping: dict[str, str]) -> dict:
    """The same rig with its cameras on other logical IDs."""
    cameras = []
    for camera in profile["cameras"]:
        cameras.append(
            {
                **camera,
                "logical_id": mapping[camera["logical_id"]],
                "source_camera": camera["logical_id"],
            }
        )
    return {**profile, "cameras": cameras}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--rename", default="", help="SRC=DST,... camera ID mapping")
    parser.add_argument("--profile-out", type=Path, help="where to write the renamed profile")
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if args.rename:
        mapping = dict(pair.split("=") for pair in args.rename.split(","))
        profile = renamed(profile, mapping)
        if args.profile_out:
            text = json.dumps(profile, indent=2)
            # One line per matrix row or size pair keeps the matrices readable.
            text = re.sub(
                r"\[\s+([-0-9.e, \n]+?)\s+\]",
                lambda match: "[" + ", ".join(match.group(1).replace(",", " ").split()) + "]",
                text,
            )
            args.profile_out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"extra_cameras": extra_cameras(profile)}, indent=2))


if __name__ == "__main__":
    main()
