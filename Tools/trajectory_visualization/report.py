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
from Tools.trajectory_visualization.kinematics import (
    AOVL_V1_CONTROL_CONTRACT,
    curvature_sign_for_dataset,
    integrate_controls,
)
from Tools.trajectory_visualization.rendering import (
    camera_projection_status,
    render_frame,
    trajectory_extent,
)


REPORT_SCHEMA_VERSION = 1
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_SELECTION_BYTES = 1024 * 1024
VideoWriter = Callable[[Path, Iterable[Image.Image], float], None]


@dataclass(frozen=True)
class PreparedFrame:
    sample: ShardSample
    prediction: np.ndarray
    target: np.ndarray
    v0: float


@dataclass(frozen=True)
class SceneSelection:
    scene_uid: str
    start_frame: int | None = None
    end_frame: int | None = None

    def __post_init__(self) -> None:
        if not self.scene_uid:
            raise ValueError("scene_uid must not be empty")
        for name, value in (
            ("start_frame", self.start_frame),
            ("end_frame", self.end_frame),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.start_frame > self.end_frame
        ):
            raise ValueError("start_frame must not exceed end_frame")

    def contains(self, frame_idx: int) -> bool:
        return (
            (self.start_frame is None or frame_idx >= self.start_frame)
            and (self.end_frame is None or frame_idx <= self.end_frame)
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "scene_uid": self.scene_uid,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


def load_scene_selections(path: str | Path) -> tuple[SceneSelection, ...]:
    """Load the strict, deterministic scene/frame selection contract."""
    source = Path(path)
    if source.stat().st_size > _MAX_SELECTION_BYTES:
        raise ValueError("selection manifest exceeds the 1 MiB limit")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("selection manifest is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("selection manifest must use schema_version 1")
    entries = document.get("scenes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("selection manifest scenes must be a non-empty list")

    selections = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"selection scenes[{index}] must be an object")
        unknown = set(entry).difference({
            "scene_uid",
            "start_frame",
            "end_frame",
        })
        if unknown:
            raise ValueError(
                f"selection scenes[{index}] has unknown fields: "
                + ", ".join(sorted(unknown))
            )
        scene_uid = entry.get("scene_uid")
        if not isinstance(scene_uid, str):
            raise ValueError(
                f"selection scenes[{index}].scene_uid must be a string"
            )
        selections.append(SceneSelection(
            scene_uid=scene_uid,
            start_frame=entry.get("start_frame"),
            end_frame=entry.get("end_frame"),
        ))
    scene_uids = [selection.scene_uid for selection in selections]
    if len(set(scene_uids)) != len(scene_uids):
        raise ValueError("selection manifest scene_uid values must be unique")
    return tuple(selections)


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
        if not np.isclose(
            v0,
            sample.initial_speed,
            rtol=1e-6,
            atol=1e-5,
        ):
            raise ValueError(
                f"overlay v0 {v0} disagrees with shard history speed "
                f"{sample.initial_speed} for {sample.sample_uid!r}"
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
    scene_selections: Sequence[SceneSelection] | None = None,
    max_frames_per_scene: int = 300,
    fps: float = 10.0,
    video_writer: VideoWriter | None = None,
) -> dict[str, Any]:
    """Write one MP4 per scene/clip and return the manifest document."""
    if max_frames_per_scene < 1:
        raise ValueError("max_frames_per_scene must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if scene_uids and scene_selections:
        raise ValueError("scene_uids and scene_selections are mutually exclusive")
    selections = tuple(scene_selections or ())
    selected_uids = (
        [selection.scene_uid for selection in selections]
        if selections
        else list(scene_uids or ())
    )
    requested_scenes = set(selected_uids)
    if len(requested_scenes) != len(selected_uids):
        raise ValueError("scene_uids must be unique")
    selection_by_scene = {
        selection.scene_uid: selection for selection in selections
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(
            f"report output directory must be empty: {destination}"
        )

    samples = read_shard_samples(shard_path, camera_index=camera_index)
    overlay = load_overlay(overlay_path)
    overlay.validate_sample_uids([
        sample.sample_uid for sample in samples
    ])
    if requested_scenes:
        available_scenes = {sample.scene_uid for sample in samples}
        missing_scenes = requested_scenes.difference(available_scenes)
        if missing_scenes:
            raise KeyError(
                "requested scenes are absent from shard: "
                + ", ".join(sorted(missing_scenes))
            )
        samples = [
            sample
            for sample in samples
            if sample.scene_uid in requested_scenes
            and (
                not selection_by_scene
                or selection_by_scene[sample.scene_uid].contains(
                    sample.frame_idx
                )
            )
        ]
        empty_ranges = requested_scenes.difference(
            sample.scene_uid for sample in samples
        )
        if empty_ranges:
            raise KeyError(
                "requested frame ranges contain no samples: "
                + ", ".join(sorted(empty_ranges))
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
    projection_statuses = {
        camera_projection_status(
            sample.calibration,
            camera_index=camera_index,
        )
        for sample in samples
    }
    projection_status = (
        next(iter(projection_statuses))
        if len(projection_statuses) == 1
        else "mixed"
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
        )
        if not selections:
            scene_samples = scene_samples[:max_frames_per_scene]
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
            "control_contract": AOVL_V1_CONTROL_CONTRACT.manifest(),
            "v0_source": "overlay_verified_against_shard_history",
            "curvature_sign": curvature_sign_for_dataset(
                next(iter(datasets))
            ),
            "panel_order": ["camera", "metric_bev"],
            "camera_projection_status": projection_status,
            "scene_selection": (
                [selection.manifest() for selection in selections]
                if selections
                else [
                    SceneSelection(scene_uid=scene_uid).manifest()
                    for scene_uid in selected_uids
                ]
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
