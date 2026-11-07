"""Read and write run files on S3 or a local disk through one filesystem interface.

Ray Train stores checkpoints through ``pyarrow.fs``; the trainer, the data
staging and the checkpoint validation use the same interface, so an ``s3://``
run and a local test run take the same code path.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pyarrow import fs


def _resolve(uri: str) -> tuple[fs.FileSystem, str]:
    return fs.FileSystem.from_uri(uri if "://" in uri else str(Path(uri).resolve()))


def read_bytes(uri: str) -> bytes:
    filesystem, path = _resolve(uri)
    with filesystem.open_input_stream(path) as stream:
        return stream.read()


def write_bytes(uri: str, payload: bytes) -> None:
    filesystem, path = _resolve(uri)
    filesystem.create_dir(path.rsplit("/", 1)[0], recursive=True)
    with filesystem.open_output_stream(path) as stream:
        stream.write(payload)


def exists(uri: str) -> bool:
    filesystem, path = _resolve(uri)
    return filesystem.get_file_info(path).type != fs.FileType.NotFound


def download(uri: str, target: Path) -> Path:
    filesystem, path = _resolve(uri)
    with filesystem.open_input_stream(path) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=16 * 1024**2)
    return target


def child_directories(uri: str) -> list[str]:
    """Sorted URIs of the directories directly under ``uri``."""
    filesystem, path = _resolve(uri)
    infos = filesystem.get_file_info(fs.FileSelector(path, allow_not_found=True))
    base = uri.rstrip("/")
    return sorted(f"{base}/{info.base_name}" for info in infos if info.type == fs.FileType.Directory)


def sha256(uri: str) -> str:
    import hashlib

    filesystem, path = _resolve(uri)
    digest = hashlib.sha256()
    with filesystem.open_input_stream(path) as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()
