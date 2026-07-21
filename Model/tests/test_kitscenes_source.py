from __future__ import annotations

import csv
import hashlib
import sys
import types
from pathlib import Path

import pytest

from data_parsing.kit_scenes import source
from data_parsing.kit_scenes.source import (
    KITSCENES_DATA_REVISION,
    PinnedKITScenesDownloader,
    SceneArchive,
    parse_archive_manifest,
    resolve_inventory,
)


def _archive(scene_id: str, split: str = "train") -> SceneArchive:
    return SceneArchive(
        scene_id=scene_id,
        split=split,
        filename=f"data/{split}/{scene_id}.tar",
        sha256="a" * 64,
        size_bytes=10,
    )


def test_inventory_allows_only_bounded_missing_scenes(monkeypatch):
    expected = ["scene-a", "scene-b", "scene-c"]
    monkeypatch.setattr(source, "sdk_split_scene_ids", lambda split: expected)
    archives = {"scene-a": _archive("scene-a"), "scene-c": _archive("scene-c")}

    resolved = resolve_inventory(
        archives,
        split="train",
        source_revision="revision-a",
        max_missing_scenes=1,
    )

    assert resolved.selected_scene_ids == ("scene-a", "scene-c")
    assert resolved.missing_scene_ids == ("scene-b",)
    assert resolved.metadata()["expected_scene_count"] == 3
    assert resolved.metadata()["selected_scene_count"] == 2
    with pytest.raises(ValueError, match="max_missing_scenes=0"):
        resolve_inventory(
            archives,
            split="train",
            source_revision="revision-a",
            max_missing_scenes=0,
        )


def test_inventory_rejects_second_missing_wrong_split_and_extra(monkeypatch):
    expected = ["scene-a", "scene-b", "scene-c"]
    monkeypatch.setattr(source, "sdk_split_scene_ids", lambda split: expected)

    with pytest.raises(ValueError, match="missing 2 scenes"):
        resolve_inventory(
            {"scene-a": _archive("scene-a")},
            split="train",
            source_revision="revision-a",
            max_missing_scenes=1,
        )
    with pytest.raises(ValueError, match="wrong split"):
        resolve_inventory(
            {
                "scene-a": _archive("scene-a", split="val"),
                "scene-b": _archive("scene-b"),
                "scene-c": _archive("scene-c"),
            },
            split="train",
            source_revision="revision-a",
            max_missing_scenes=1,
        )
    with pytest.raises(ValueError, match="unexpected"):
        resolve_inventory(
            {
                "scene-a": _archive("scene-a"),
                "scene-b": _archive("scene-b"),
                "scene-c": _archive("scene-c"),
                "scene-extra": _archive("scene-extra"),
            },
            split="train",
            source_revision="revision-a",
            max_missing_scenes=1,
        )


def test_manifest_parser_rejects_duplicate_scene_rows(tmp_path):
    path = tmp_path / "sequence_archives.csv"
    fields = [
        "sequence_id",
        "split",
        "archive_path",
        "archive_sha256",
        "archive_size_bytes",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for _ in range(2):
            writer.writerow({
                "sequence_id": "scene-a",
                "split": "train",
                "archive_path": "data/train/scene-a.tar",
                "archive_sha256": "a" * 64,
                "archive_size_bytes": "10",
            })
    with pytest.raises(ValueError, match="duplicate scene"):
        parse_archive_manifest(path)


def test_downloader_pins_manifest_and_archive_revision(monkeypatch, tmp_path):
    scene_id = "scene-a"
    archive_bytes = b"archive-bytes"
    digest = hashlib.sha256(archive_bytes).hexdigest()
    calls: list[dict] = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if kwargs["filename"] == source.KITSCENES_MANIFEST_PATH:
            with target.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "sequence_id",
                        "split",
                        "archive_path",
                        "archive_sha256",
                        "archive_size_bytes",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "sequence_id": scene_id,
                    "split": "train",
                    "archive_path": f"data/train/{scene_id}.tar",
                    "archive_sha256": digest,
                    "archive_size_bytes": str(len(archive_bytes)),
                })
        else:
            target.write_bytes(archive_bytes)
        return str(target)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    downloader_module = types.ModuleType("kitscenes.download.downloader")

    def fake_extract(_tar_path, split_dir, **kwargs):
        (Path(split_dir) / kwargs["scene_id"]).mkdir(parents=True)

    downloader_module._extract_tar_file = fake_extract
    download_module = types.ModuleType("kitscenes.download")
    kitscenes_module = types.ModuleType("kitscenes")
    monkeypatch.setitem(sys.modules, "kitscenes", kitscenes_module)
    monkeypatch.setitem(sys.modules, "kitscenes.download", download_module)
    monkeypatch.setitem(
        sys.modules, "kitscenes.download.downloader", downloader_module
    )

    downloader = PinnedKITScenesDownloader(tmp_path)
    downloader.download([scene_id], expected_split="train")

    assert [call["filename"] for call in calls] == [
        source.KITSCENES_MANIFEST_PATH,
        f"data/train/{scene_id}.tar",
    ]
    assert all(call["revision"] == KITSCENES_DATA_REVISION for call in calls)
    assert (tmp_path / "data" / "train" / scene_id).is_dir()
