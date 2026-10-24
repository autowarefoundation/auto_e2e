"""Evaluate official Drive-JEPA weights; defaults to perception-based v2 / 20 m."""

import argparse
import json
from pathlib import Path

from .external_kitscenes import evaluate, file_sha256, git_revision
from .paths import (
    CHECKPOINTS,
    DRIVE_JEPA_ROOT,
    DRIVE_JEPA_REVISION,
    SDK_ROOT,
    WORKSPACE,
)
from .prepare_inputs import prepare
from .source import KITSCENES_DATA_REVISION


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=CHECKPOINTS, default="perception-based-v2")
    parser.add_argument("--commands", choices=("20m", "100m", "missing"), default="20m")
    parser.add_argument("--drive-jepa-root", type=Path, default=DRIVE_JEPA_ROOT)
    parser.add_argument("--sdk-root", type=Path, default=SDK_ROOT)
    parser.add_argument(
        "--checkpoint", type=Path, help="Same official checkpoint, at a different path"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Existing SDK data/val/<uuid> layout; otherwise stream scenes",
    )
    parser.add_argument(
        "--scene", action="append", help="One val UUID; repeat for a labelled subset"
    )
    parser.add_argument(
        "--command-file",
        type=Path,
        help="Explicit regenerated route commands plus provenance sidecar",
    )
    parser.add_argument(
        "--episode-ids",
        type=Path,
        help="Optional 117-val or 140-val-plus-overlap audit",
    )
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--staging-root", type=Path, default=WORKSPACE / "staging")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"))
    args = parser.parse_args(argv)
    if args.commands == "missing" and args.command_file:
        parser.error("--command-file conflicts with --commands missing")
    if args.variant != "perception-free-v1" and args.precision not in (None, "fp32"):
        parser.error("Perception-based CUDA attention requires --precision fp32")
    filename, args.expected_checkpoint_sha256 = CHECKPOINTS[args.variant]
    args.checkpoint = args.checkpoint or WORKSPACE / "weights" / filename
    # The upstream v1/v2 source is selected by this canonical filename.
    if args.checkpoint.name != filename:
        parser.error(f"Checkpoint basename must be {filename}")
    args.output = (
        args.output
        or WORKSPACE / "results" / f"drive_jepa_{args.variant}_{args.commands}"
    )
    args.precision = args.precision or (
        "bf16" if args.variant == "perception-free-v1" else "fp32"
    )
    args.adapter = (
        "accuracy_benchmark.drive_jepa.drive_jepa_adapter:create_adapter"
        if args.variant == "perception-free-v1"
        else "accuracy_benchmark.drive_jepa.drive_jepa_perception_adapter:create_adapter"
    )
    args.stream = args.raw_root is None
    args.dataset_revision = KITSCENES_DATA_REVISION
    args.camera = "camera_base_front_center"
    args.command = [0.0] * 4
    return args


def main(argv=None):
    args = parse_args(argv)
    if git_revision(args.drive_jepa_root) != DRIVE_JEPA_REVISION:
        raise ValueError(f"Drive-JEPA checkout must be {DRIVE_JEPA_REVISION}")
    if file_sha256(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 differs from the official pinned artifact")
    if args.commands != "missing" and args.command_file is None:
        prepare()
        directory = "route_inputs_20m" if args.commands == "20m" else "route_inputs"
        args.command_file = WORKSPACE / directory / "commands.json"
    if args.command_file:
        provenance = args.command_file.with_suffix(".provenance.json")
        if not provenance.is_file():
            raise ValueError("Route commands require their .provenance.json sidecar")
        policy = json.loads(provenance.read_text())["command_policy"]
        expected = {
            "20m": "route_centerline_20m_lateral_2m_v1",
            "100m": "autoe2e_published_route_maneuver_100m_v1",
        }[args.commands]
        if policy != expected:
            raise ValueError("Command policy differs from --commands")
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
