"""Build the dependency-free navigation rasterizer shared library."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
from pathlib import Path


def default_output(directory: Path) -> Path:
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return directory / f"libnavigation_rasterizer{suffix}"


def build(output: Path, *, compiler: str | None = None) -> Path:
    directory = Path(__file__).resolve().parent
    source = directory / "navigation_rasterizer.cpp"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler or os.environ.get("CXX", "c++"),
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-fPIC",
        "-fvisibility=hidden",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]
    command.append("-dynamiclib" if platform.system() == "Darwin" else "-shared")
    command.extend([str(source), "-o", str(output)])
    subprocess.run(command, check=True)
    return output


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default_output(directory))
    parser.add_argument("--compiler")
    args = parser.parse_args()
    print(build(args.output, compiler=args.compiler))


if __name__ == "__main__":
    main()
