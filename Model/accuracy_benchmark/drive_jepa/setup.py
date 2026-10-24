"""Clone pinned sources and download one official checkpoint in an isolated environment."""

import argparse
from pathlib import Path
import subprocess

from .external_kitscenes import file_sha256, git_revision
from .paths import (
    CHECKPOINTS,
    DRIVE_JEPA_ROOT,
    DRIVE_JEPA_REVISION,
    NUPLAN_REVISION,
    SDK_ROOT,
    WEIGHTS_REVISION,
    WORKSPACE,
)
from .source import KITSCENES_SDK_REVISION


def clone(url, destination, revision):
    if destination.exists():
        if git_revision(destination) != revision:
            raise ValueError(
                f"Existing checkout differs; preserve it and use a fresh workspace: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", url, str(destination)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "-c",
            "core.autocrlf=false",
            "checkout",
            "--detach",
            revision,
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=CHECKPOINTS, default="perception-based-v2")
    parser.add_argument(
        "--navigation",
        action="store_true",
        help="Also clone the original navigation implementation",
    )
    args = parser.parse_args()
    clone("https://github.com/KIT-MRT/kitscenes.git", SDK_ROOT, KITSCENES_SDK_REVISION)
    clone(
        "https://github.com/linhanwang/Drive-JEPA.git",
        DRIVE_JEPA_ROOT,
        DRIVE_JEPA_REVISION,
    )
    if args.variant != "perception-free-v1":
        clone(
            "https://github.com/motional/nuplan-devkit.git",
            WORKSPACE / "vendor/nuplan",
            NUPLAN_REVISION,
        )
    if args.navigation:
        clone(
            "https://github.com/autowarefoundation/auto_e2e.git",
            WORKSPACE / "vendor/auto_e2e_navigation",
            "40a75cf8a26b60f6b094125e3924fe0bdb427b02",
        )
    from huggingface_hub import hf_hub_download

    filename, digest = CHECKPOINTS[args.variant]
    checkpoint = Path(
        hf_hub_download(
            "LinhanWang/Drive-JEPA",
            filename,
            repo_type="dataset",
            revision=WEIGHTS_REVISION,
            local_dir=WORKSPACE / "weights",
        )
    )
    if file_sha256(checkpoint) != digest:
        raise ValueError("Official checkpoint checksum differs")
    print(f"Verified {args.variant}: {checkpoint}")


if __name__ == "__main__":
    main()
