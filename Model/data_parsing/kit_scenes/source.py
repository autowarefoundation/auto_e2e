"""Pinned KITScenes source inventory and scene downloader."""

from __future__ import annotations

import csv
import logging
import shutil
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Mapping, Sequence

KITSCENES_REPO_ID = "KIT-MRT/KITScenes-Multimodal"
KITSCENES_DATA_REVISION = "6fde0034446669e2ed7235e4c7fe323cd23d599d"
KITSCENES_SDK_REVISION = "7765cdec5490894266070ab46e23724b58b3da42"
KITSCENES_MANIFEST_PATH = "data/sequence_archives.csv"

_SPLIT_FILES = {
    "train": "train.txt",
    "val": "validation.txt",
    "test": "test.txt",
    "test_e2e": "test-e2e.txt",
    "overlap_train_val": "overlap_train_val.txt",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneArchive:
    scene_id: str
    split: str
    filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class InventoryResolution:
    split: str
    expected_scene_ids: tuple[str, ...]
    selected_scene_ids: tuple[str, ...]
    missing_scene_ids: tuple[str, ...]
    total_size_bytes: int
    source_revision: str

    def metadata(self) -> dict[str, object]:
        return {
            "split": self.split,
            "expected_scene_count": len(self.expected_scene_ids),
            "selected_scene_count": len(self.selected_scene_ids),
            "missing_scene_ids": list(self.missing_scene_ids),
            "total_size_bytes": self.total_size_bytes,
            "source_revision": self.source_revision,
        }


def sdk_split_scene_ids(split: str) -> list[str]:
    """Read one official scene split bundled with the pinned SDK."""
    try:
        filename = _SPLIT_FILES[split]
    except KeyError as exc:
        raise ValueError(
            f"unknown KITScenes split {split!r}; expected {sorted(_SPLIT_FILES)}"
        ) from exc
    split_file = (
        resources.files("kitscenes")
        .joinpath("split")
        .joinpath("generated_splits")
        .joinpath("default_geo_split_v1_0")
        .joinpath(filename)
    )
    scene_ids = [
        line.strip() for line in split_file.read_text().splitlines()
        if line.strip()
    ]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError(f"KITScenes SDK split {split!r} contains duplicates")
    return scene_ids


def parse_archive_manifest(path: str | Path) -> dict[str, SceneArchive]:
    """Parse the HF archive manifest and reject duplicate scene rows."""
    archives: dict[str, SceneArchive] = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            scene_id = row["sequence_id"]
            if scene_id in archives:
                raise ValueError(
                    f"KITScenes archive manifest has duplicate scene {scene_id}"
                )
            archive = SceneArchive(
                scene_id=scene_id,
                split=row["split"],
                filename=row["archive_path"],
                sha256=row["archive_sha256"],
                size_bytes=int(row["archive_size_bytes"]),
            )
            if archive.size_bytes <= 0:
                raise ValueError(
                    f"KITScenes archive {scene_id} has invalid size "
                    f"{archive.size_bytes}"
                )
            archives[scene_id] = archive
    if not archives:
        raise ValueError("KITScenes archive manifest is empty")
    return archives


def fetch_archive_manifest(
    output_dir: str | Path,
    *,
    repo_id: str = KITSCENES_REPO_ID,
    revision: str = KITSCENES_DATA_REVISION,
    token: str | bool | None = None,
) -> dict[str, SceneArchive]:
    """Fetch the manifest at an explicit immutable dataset revision."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=KITSCENES_MANIFEST_PATH,
        local_dir=str(output_dir),
        token=token,
    )
    return parse_archive_manifest(path)


def resolve_inventory(
    archives: Mapping[str, SceneArchive],
    *,
    split: str,
    source_revision: str,
    max_missing_scenes: int,
    requested_scene_ids: Sequence[str] | None = None,
) -> InventoryResolution:
    """Resolve the exact selected scene set under a bounded missing policy."""
    if max_missing_scenes < 0:
        raise ValueError("max_missing_scenes must be non-negative")

    expected = sdk_split_scene_ids(split)
    expected_set = set(expected)
    wrong_split = sorted(
        scene_id
        for scene_id in expected
        if scene_id in archives and archives[scene_id].split != split
    )
    available_set = {
        scene_id
        for scene_id, archive in archives.items()
        if archive.split == split
    }
    unexpected = sorted(available_set - expected_set)
    missing = [scene_id for scene_id in expected if scene_id not in available_set]

    if wrong_split:
        raise ValueError(
            f"KITScenes scenes stored under the wrong split: {wrong_split}"
        )
    if unexpected:
        raise ValueError(
            f"KITScenes manifest has unexpected {split} scenes: {unexpected}"
        )
    if len(missing) > max_missing_scenes:
        raise ValueError(
            f"KITScenes {split} inventory is missing {len(missing)} scenes "
            f"(max_missing_scenes={max_missing_scenes}): {missing}"
        )
    if missing:
        logger.warning(
            "KITScenes %s inventory: proceeding with %d/%d scenes; missing=%s",
            split,
            len(expected) - len(missing),
            len(expected),
            missing,
        )

    available_in_official_order = [
        scene_id for scene_id in expected if scene_id in available_set
    ]
    if requested_scene_ids is None:
        selected = available_in_official_order
    else:
        requested = list(requested_scene_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("requested_scene_ids contains duplicates")
        unavailable = sorted(set(requested) - set(available_in_official_order))
        if unavailable:
            raise ValueError(
                f"requested KITScenes scenes are unavailable: {unavailable}"
            )
        requested_set = set(requested)
        selected = [
            scene_id
            for scene_id in available_in_official_order
            if scene_id in requested_set
        ]

    if not selected:
        raise ValueError("KITScenes inventory selected no scenes")
    total_size = sum(archives[scene_id].size_bytes for scene_id in selected)
    return InventoryResolution(
        split=split,
        expected_scene_ids=tuple(expected),
        selected_scene_ids=tuple(selected),
        missing_scene_ids=tuple(missing),
        total_size_bytes=total_size,
        source_revision=source_revision,
    )


class PinnedKITScenesDownloader:
    """Download, checksum, and extract exact scenes from one dataset revision."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        repo_id: str = KITSCENES_REPO_ID,
        revision: str = KITSCENES_DATA_REVISION,
        token: str | bool | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.revision = revision
        self.token = token
        self.archives = fetch_archive_manifest(
            self.output_dir,
            repo_id=repo_id,
            revision=revision,
            token=token,
        )

    def download(self, scene_ids: Sequence[str], *, expected_split: str) -> None:
        """Download each requested scene and fail on the first discrepancy."""
        from huggingface_hub import hf_hub_download
        from kitscenes.download.downloader import _extract_tar_file

        requested = list(scene_ids)
        if not requested:
            raise ValueError("KITScenes download requires at least one scene")
        if len(requested) != len(set(requested)):
            raise ValueError("KITScenes download scene_ids contains duplicates")

        staging_root = self.output_dir / ".kitscenes_staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        for scene_id in requested:
            try:
                archive = self.archives[scene_id]
            except KeyError as exc:
                raise ValueError(
                    f"KITScenes scene {scene_id} is absent from pinned manifest"
                ) from exc
            if archive.split != expected_split:
                raise ValueError(
                    f"KITScenes scene {scene_id} belongs to {archive.split!r}, "
                    f"not {expected_split!r}"
                )

            scene_dir = (
                self.output_dir / "data" / archive.split / archive.scene_id
            )
            if scene_dir.exists():
                raise FileExistsError(
                    f"refusing to reuse pre-existing KITScenes scene {scene_dir}"
                )

            with tempfile.TemporaryDirectory(
                prefix="kitscenes_dl_", dir=staging_root
            ) as temporary:
                downloaded = hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    filename=archive.filename,
                    local_dir=temporary,
                    token=self.token,
                )
                tar_path = Path(downloaded)
                if not tar_path.is_file():
                    raise FileNotFoundError(
                        f"downloaded KITScenes archive not found: {tar_path}"
                    )
                actual_size = tar_path.stat().st_size
                if actual_size != archive.size_bytes:
                    raise ValueError(
                        f"KITScenes archive size mismatch for {scene_id}: "
                        f"expected {archive.size_bytes}, got {actual_size}"
                    )
                _extract_tar_file(
                    tar_path,
                    self.output_dir / "data" / archive.split,
                    expected_sha256=archive.sha256,
                    size_bytes=archive.size_bytes,
                    scene_id=scene_id,
                )
            if not scene_dir.is_dir():
                raise RuntimeError(
                    f"KITScenes archive {scene_id} did not extract to {scene_dir}"
                )

        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(self.output_dir / ".cache", ignore_errors=True)
