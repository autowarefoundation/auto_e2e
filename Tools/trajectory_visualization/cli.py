"""Command-line interface for offline canonical trajectory reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from Tools.trajectory_visualization.report import generate_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render one canonical AOVL body and matching WebDataset shard "
            "as per-scene MP4 reports."
        )
    )
    parser.add_argument("--shard", required=True, help="Local WebDataset tar")
    parser.add_argument(
        "--overlay",
        required=True,
        help="Local canonical overlay.bin.gz",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Empty destination directory",
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help="Render only this scene_uid; may be repeated",
    )
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-frames-per-scene", type=int, default=300)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_report(
        shard_path=args.shard,
        overlay_path=args.overlay,
        output_dir=args.output_dir,
        seed_index=args.seed_index,
        camera_index=args.camera_index,
        scene_uids=args.scenes,
        max_frames_per_scene=args.max_frames_per_scene,
        fps=args.fps,
    )
    print(json.dumps({
        "frame_count": manifest["frame_count"],
        "scene_count": manifest["scene_count"],
        "output_dir": args.output_dir,
    }, sort_keys=True))
    return 0
