"""KITScenes parser exports, loaded lazily for workflow registration."""

from __future__ import annotations

from typing import Any

__all__ = [
    "KitScenesDataset",
    "load_camera_frame",
    "CAMERA_NAMES",
    "VIEW_NAMES",
    "TELE_VIEW_NAME",
    "load_egomotion",
    "poses_to_arrays",
    "generate_bev_map_tile",
    "NUM_VIEWS",
    "EGOMOTION_DIM",
    "TRAJECTORY_DIM",
]


def __getattr__(name: str) -> Any:
    if name == "KitScenesDataset":
        from .dataset import KitScenesDataset
        return KitScenesDataset
    if name in {"load_camera_frame", "CAMERA_NAMES", "NUM_VIEWS",
                "VIEW_NAMES", "TELE_VIEW_NAME"}:
        from .camera import (
            CAMERA_NAMES, NUM_VIEWS, TELE_VIEW_NAME, VIEW_NAMES, load_camera_frame,
        )
        return {
            "load_camera_frame": load_camera_frame,
            "CAMERA_NAMES": CAMERA_NAMES,
            "VIEW_NAMES": VIEW_NAMES,
            "TELE_VIEW_NAME": TELE_VIEW_NAME,
            "NUM_VIEWS": NUM_VIEWS,
        }[name]
    if name in {
        "load_egomotion",
        "poses_to_arrays",
        "EGOMOTION_DIM",
        "TRAJECTORY_DIM",
    }:
        from .egomotion import (
            EGOMOTION_DIM,
            TRAJECTORY_DIM,
            load_egomotion,
            poses_to_arrays,
        )
        return {
            "load_egomotion": load_egomotion,
            "poses_to_arrays": poses_to_arrays,
            "EGOMOTION_DIM": EGOMOTION_DIM,
            "TRAJECTORY_DIM": TRAJECTORY_DIM,
        }[name]
    if name == "generate_bev_map_tile":
        from .map import generate_bev_map_tile
        return generate_bev_map_tile
    raise AttributeError(name)
