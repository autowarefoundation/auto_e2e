"""Reuse AutoE2E's pinned downloader with checkout-independent split hashing."""

import hashlib
from pathlib import Path

from data_parsing.kit_scenes import source as core
from data_parsing.kit_scenes.source import (
    KITSCENES_DATA_REVISION as KITSCENES_DATA_REVISION,
    KITSCENES_REPO_ID as KITSCENES_REPO_ID,
    KITSCENES_SDK_REVISION as KITSCENES_SDK_REVISION,
    KITSCENES_STANDARD_VAL_SCENE_UID_SHA256 as KITSCENES_STANDARD_VAL_SCENE_UID_SHA256,
    PinnedKITScenesDownloader as PinnedKITScenesDownloader,
)


def sdk_split_scene_ids(split):
    if split not in core._PINNED_SPLIT_FILE_SHA256:
        return core.sdk_split_scene_ids(split)
    path = (
        Path(core.__file__).with_name("pinned_splits")
        / "default_geo_split_v1_0"
        / core._SPLIT_FILES[split]
    )
    # Git CRLF conversion is a checkout property, not a different dataset split.
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(payload).hexdigest() != core._PINNED_SPLIT_FILE_SHA256[split]:
        raise ValueError("Pinned split content differs")
    scenes = payload.decode("ascii").split()
    if len(scenes) != len(set(scenes)):
        raise ValueError("Duplicate scenes in split")
    return scenes
