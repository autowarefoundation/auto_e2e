"""Generate self-contained MP4 reports from one shard and canonical AOVL."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from Tools.trajectory_visualization.artifacts import (
    OverlayReader,
    ShardSample,
    load_overlay,
    read_shard_samples,
)
from Tools.trajectory_visualization.rendering import (
    curvature_sign_for_dataset,
    integrate_controls,
    render_frame,
    trajectory_extent,
)


REPORT_SCHEMA_VERSION = 1
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")
VideoWriter = Callable[[Path, Iterable[Image.Image], float], None]


@dataclass(frozen=True)
class PreparedFrame:
    sample: ShardSample
    prediction: np.ndarray
    target: np.ndarray
    v0: float


def _sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_mp4(
    path: Path,
    frames: Iterable[Image.Image],
    fps: float,
) -> None:
    """Encode incrementally; imageio/ffmpeg is supplied by data-prep image."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "MP4 export requires imageio[ffmpeg]; use the data-prep image"
        ) from exc

    with imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))


def _prepare_frames(
    samples: Sequence[ShardSample],
    overlay: OverlayReader,
    *,
    seed_index: int,
) -> list[PreparedFrame]:
    prepared = []
    for sample in samples:
        prediction_controls, v0 = overlay.sample(
            sample.sample_uid,
            seed_index,
        )
        sign = curvature_sign_for_dataset(sample.dataset)
        prepared.append(PreparedFrame(
            sample=sample,
            prediction=integrate_controls(
                prediction_controls,
                v0,
                curvature_sign=sign,
            ),
            target=integrate_controls(
                sample.target_controls,
                v0,
                curvature_sign=sign,
            ),
            v0=v0,
        ))
    return prepared


def _scene_metrics(frames: Sequence[PreparedFrame]) -> dict[str, float]:
    errors = [
        np.linalg.norm(frame.prediction - frame.target, axis=1)
        for frame in frames
    ]
    return {
        "ade_m": float(np.mean([values.mean() for values in errors])),
        "fde_m": float(np.mean([values[-1] for values in errors])),
        "max_error_m": float(max(values.max() for values in errors)),
    }


def _rendered_frames(
    frames: Sequence[PreparedFrame],
    *,
    extent: float,
    base_seed: int,
    camera_index: int,
) -> Iterable[Image.Image]:
    for frame in frames:
        yield render_frame(
            frame.sample,
            prediction=frame.prediction,
            target=frame.target,
            v0=frame.v0,
            base_seed=base_seed,
            extent=extent,
            camera_index=camera_index,
        )


def _safe_scene_segment(scene_uid: str) -> str:
    if _SAFE_SEGMENT.fullmatch(scene_uid):
        return scene_uid
    digest = hashlib.sha256(scene_uid.encode()).hexdigest()[:16]
    return f"scene-{digest}"


def generate_report(
    *,
    shard_path: str | Path,
    overlay_path: str | Path,
    output_dir: str | Path,
    seed_index: int = 0,
    camera_index: int = 0,
    scene_uids: Sequence[str] | None = None,
    max_frames_per_scene: int = 300,
    fps: float = 10.0,
    video_writer: VideoWriter | None = None,
) -> dict[str, Any]:
    """Write one MP4 per scene/clip and return the manifest document."""
    if max_frames_per_scene < 1:
        raise ValueError("max_frames_per_scene must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    requested_scenes = set(scene_uids or ())
    if len(requested_scenes) != len(scene_uids or ()):
        raise ValueError("scene_uids must be unique")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(
            f"report output directory must be empty: {destination}"
        )

    samples = read_shard_samples(shard_path, camera_index=camera_index)
    overlay = load_overlay(overlay_path)
    if requested_scenes:
        samples = [
            sample
            for sample in samples
            if sample.scene_uid in requested_scenes
        ]
        missing_scenes = requested_scenes.difference(
            sample.scene_uid for sample in samples
        )
        if missing_scenes:
            raise KeyError(
                "requested scenes are absent from shard: "
                + ", ".join(sorted(missing_scenes))
            )
    if not samples:
        raise ValueError("no samples remain after scene filtering")

    datasets = {sample.dataset for sample in samples}
    if len(datasets) != 1:
        raise ValueError("one report cannot mix dataset coordinate contracts")
    if seed_index < 0 or seed_index >= len(overlay.base_seeds):
        raise IndexError(
            f"seed_index {seed_index} is outside "
            f"[0, {len(overlay.base_seeds)})"
        )
    base_seed = overlay.base_seeds[seed_index]
    writer = video_writer or _write_mp4

    grouped: dict[str, list[ShardSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.scene_uid].append(sample)

    scene_entries = []
    total_frames = 0
    for scene_uid in sorted(grouped):
        scene_samples = sorted(
            grouped[scene_uid],
            key=lambda sample: (sample.frame_idx, sample.sample_uid),
        )[:max_frames_per_scene]
        prepared = _prepare_frames(
            scene_samples,
            overlay,
            seed_index=seed_index,
        )
        extent = trajectory_extent(
            trajectory
            for frame in prepared
            for trajectory in (frame.prediction, frame.target)
        )
        scene_dir = destination / "scenes" / _safe_scene_segment(scene_uid)
        scene_dir.mkdir(parents=True)
        video_path = scene_dir / "video.mp4"
        thumbnail_path = scene_dir / "thumbnail.jpg"

        first_frame = next(iter(_rendered_frames(
            prepared[:1],
            extent=extent,
            base_seed=base_seed,
            camera_index=camera_index,
        )))
        first_frame.save(
            thumbnail_path,
            format="JPEG",
            quality=90,
            optimize=True,
        )
        writer(
            video_path,
            _rendered_frames(
                prepared,
                extent=extent,
                base_seed=base_seed,
                camera_index=camera_index,
            ),
            fps,
        )
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise RuntimeError(f"video writer produced no output: {video_path}")

        frame_count = len(prepared)
        total_frames += frame_count
        scene_entries.append({
            "scene_uid": scene_uid,
            "start_frame": prepared[0].sample.frame_idx,
            "end_frame": prepared[-1].sample.frame_idx,
            "frame_count": frame_count,
            "sample_uids": [
                frame.sample.sample_uid for frame in prepared
            ],
            "video": str(video_path.relative_to(destination)),
            "thumbnail": str(thumbnail_path.relative_to(destination)),
            "metrics": _scene_metrics(prepared),
            "bev_extent_m": extent,
        })

    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": (
            datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        "dataset": next(iter(datasets)),
        "source": {
            "shard_name": Path(shard_path).name,
            "shard_sha256": _sha256_file(shard_path),
            "overlay_name": Path(overlay_path).name,
            "overlay_sha256": overlay.sha256,
        },
        "render": {
            "camera_index": camera_index,
            "fps": fps,
            "seed_index": seed_index,
            "base_seed": base_seed,
            "coordinate_frame": "x_forward_y_left",
            "curvature_sign": curvature_sign_for_dataset(
                next(iter(datasets))
            ),
        },
        "scene_count": len(scene_entries),
        "frame_count": total_frames,
        "scenes": scene_entries,
    }
    manifest_path = destination / "manifest.json"
    temporary = destination / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(manifest_path)
    return manifest
