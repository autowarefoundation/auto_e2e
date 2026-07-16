"""Flyte wrapper for CPU-only canonical trajectory report exports."""

from __future__ import annotations

import os
from typing import List, Optional

from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile


DATA_PREP_IMAGE = os.environ.get(
    "AUTO_E2E_DATA_PREP_IMAGE",
    "auto-e2e/data-prep:latest",
)


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="2", mem="4Gi"),
    cache=True,
    cache_version="trajectory-report-v2",
    cache_serialize=True,
)
def export_trajectory_report(
    shard: FlyteFile,
    overlay: FlyteFile,
    dataset_manifest: FlyteFile,
    overlay_manifest: FlyteFile,
    selection_manifest: Optional[FlyteFile] = None,
    scene_uids: List[str] = [],
    seed_index: int = 0,
    camera_index: int = 0,
    max_frames_per_scene: int = 300,
    fps: float = 10.0,
) -> FlyteDirectory:
    """Materialize immutable inputs and return one MP4 report directory."""
    import tempfile
    from pathlib import Path

    from Tools.trajectory_visualization.report import (
        generate_report,
        load_scene_selections,
    )

    output = Path(tempfile.mkdtemp(prefix="trajectory-export-")) / "report"
    scene_selections = (
        load_scene_selections(selection_manifest.download())
        if selection_manifest is not None
        else None
    )
    manifest = generate_report(
        shard_path=shard.download(),
        overlay_path=overlay.download(),
        output_dir=output,
        dataset_manifest_path=dataset_manifest.download(),
        overlay_manifest_path=overlay_manifest.download(),
        scene_uids=scene_uids or None,
        scene_selections=scene_selections,
        seed_index=seed_index,
        camera_index=camera_index,
        max_frames_per_scene=max_frames_per_scene,
        fps=fps,
    )
    print(
        "trajectory report ready: "
        f"{manifest['scene_count']} scenes, {manifest['frame_count']} frames"
    )
    return FlyteDirectory(str(output))
