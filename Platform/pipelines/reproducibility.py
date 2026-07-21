"""Narrow, reproducible identities for trajectory-overlay inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path
from typing import Sequence


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_hex(value: str, name: str) -> str:
    normalized = value.removeprefix("sha256:")
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def preprocessing_contract_digest(root: str | Path | None = None) -> str:
    """Hash the cache-invalidating data contract versions."""
    root = Path(root) if root is not None else repository_root()
    namespace = runpy.run_path(
        str(root / "Model/data_processing/contract_versions.py")
    )
    versions = namespace["contract_versions"]()
    payload = json.dumps(
        versions, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def inference_contract_files(
    root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return only Python sources that can alter stored model controls."""
    root = Path(root) if root is not None else repository_root()
    files = [
        root / "Model/data_parsing/pre_extracted.py",
        root / "Platform/pipelines/inference.py",
        root / "Platform/pipelines/overlay_precompute.py",
    ]
    files.extend(sorted((root / "Model/model_components").rglob("*.py")))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "inference contract source is missing: " + ", ".join(missing)
        )
    return tuple(files)


def sha256_files(
    files: Sequence[str | Path],
    *,
    root: str | Path,
) -> str:
    """Hash relative paths and bytes so renames also change the identity."""
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for candidate in sorted(Path(path).resolve() for path in files):
        relative = candidate.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(candidate.stat().st_size.to_bytes(8, "little"))
        with candidate.open("rb") as stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def model_inference_code_digest(root: str | Path | None = None) -> str:
    root = Path(root) if root is not None else repository_root()
    return sha256_files(inference_contract_files(root), root=root)


def container_digest_from_image(image: str) -> str:
    """Extract the canonical digest from an immutable OCI image reference."""
    marker = "@sha256:"
    if marker not in image:
        raise ValueError("overlay task image must be pinned by digest")
    digest = image.rsplit(marker, 1)[1]
    return "sha256:" + _sha256_hex(digest, "container image")


def validate_runtime_contract(
    *,
    preprocessing_digest: str,
    inference_code_digest: str,
    container_image_digest: str,
    task_image: str,
    root: str | Path | None = None,
) -> str:
    """Reject launcher identities that do not describe the running task."""
    expected = {
        "preprocessing_contract_digest": preprocessing_contract_digest(root),
        "model_inference_code_digest": model_inference_code_digest(root),
        "container_image_digest": container_digest_from_image(task_image),
    }
    actual = {
        "preprocessing_contract_digest": _sha256_hex(
            preprocessing_digest, "preprocessing_contract_digest"
        ),
        "model_inference_code_digest": _sha256_hex(
            inference_code_digest, "model_inference_code_digest"
        ),
        "container_image_digest": (
            "sha256:"
            + _sha256_hex(
                container_image_digest, "container_image_digest"
            )
        ),
    }
    mismatches = [
        name for name in expected if actual[name] != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "overlay runtime contract differs in "
            + ", ".join(sorted(mismatches))
        )
    return expected["container_image_digest"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "identity",
        choices=("preprocessing", "inference"),
    )
    parser.add_argument("--root", default=str(repository_root()))
    args = parser.parse_args()
    if args.identity == "preprocessing":
        print(preprocessing_contract_digest(args.root))
    else:
        print(model_inference_code_digest(args.root))


if __name__ == "__main__":
    main()
