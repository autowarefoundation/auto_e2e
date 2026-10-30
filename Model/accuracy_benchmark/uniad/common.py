# SPDX-License-Identifier: Apache-2.0
"""Pinned public inputs and small provenance/coverage utilities."""

import hashlib
import json
from pathlib import Path, PurePosixPath

RAW_REPO = "KIT-MRT/KITScenes-Multimodal"
RAW_REVISION = "6fde0034446669e2ed7235e4c7fe323cd23d599d"
MIRROR_REPO = "AutowareFoundation/auto_e2e"
MIRROR_REVISION = "7262c151599c1b5e4a99f0da9c3196ff9ff615fe"
MIRROR_PREFIX = "datasets/kitscenes-val/v3.5/"
UNIAD_COMMIT = "609ee083ea51c3521c323f1279dfc4cee0e60467"
ASSET_REPO = "OpenDriveLab/UniAD2.0_R101_nuScenes"
ASSET_REVISION = "ea565f376a6998bade2bfc528d6b8d71590e73df"
ASSETS = {
    "ckpts/uniad_base_e2e.pth": "4dcdc34c09f964e377f608dbb764f003844a9ccc62b4a0469f94375302ed76d0",
    "data/motion_anchor_infos_mode6.pkl": "05c84020204841eb45a1599228e4452b021213a83d8f7a63b40b47531f0cac6b",
}
CONFIG_SHA256 = "2d45be9b330f8e82456203e2d6374053bd7b80171c5316412dfa88b32e187b8c"
DENSE_UID_SHA256 = "28c03331e86d6296d2a1c4da29bc5b28a06dc86a38bd3dc282224a8f06f03146"
EXPECTED = {
    "dense": {"samples": 11035, "scenes": 117, "streams": 585, "model_frames": 15130},
    "keyframes": {"samples": 202, "scenes": 117, "streams": 202, "model_frames": 1818},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Atomically update our status file; callers create a fresh output directory."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def safe_child(root, name):
    relative = PurePosixPath(name)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or ":" in name
    ):
        raise ValueError("Unsafe relative path: " + name)
    target = Path(root).joinpath(*relative.parts)
    if Path(root).resolve() not in target.resolve().parents or target.is_symlink():
        raise ValueError("Path escapes its root: " + name)
    return target


def membership(actual, expected):
    actual, expected = list(actual), list(expected)
    if (
        len(actual) != len(set(actual))
        or len(expected) != len(set(expected))
        or set(actual) != set(expected)
    ):
        raise ValueError("Missing, duplicate or unexpected sample IDs")


def uid_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def coverage(streams):
    ids = [f["sample_uid"] for s in streams for f in s["frames"] if f["scored"]]
    membership(ids, ids)
    return dict(
        samples=len(ids),
        scenes=len({s["physical_scene_id"] for s in streams}),
        streams=len(streams),
        model_frames=sum(len(s["frames"]) for s in streams),
    )


def verified_download(repo, revision, name, root, repo_type="model"):
    """Download public/gated assets using the caller's HF login; verify pinned bytes."""
    from huggingface_hub import HfApi, hf_hub_download

    safe_child(root, name)
    records = HfApi().get_paths_info(
        repo, [name], revision=revision, repo_type=repo_type
    )
    if len(records) != 1 or records[0].path != name:
        raise ValueError("Missing pinned artifact: " + name)
    record = records[0]
    path = Path(
        hf_hub_download(
            repo, name, revision=revision, repo_type=repo_type, local_dir=root
        )
    )
    if (
        path.resolve() != safe_child(root, name).resolve()
        or path.stat().st_size != record.size
    ):
        raise ValueError("Downloaded artifact location/size changed")
    if record.lfs:
        expected = (
            record.lfs["sha256"] if isinstance(record.lfs, dict) else record.lfs.sha256
        )
        if sha256(path) != expected:
            raise ValueError("Downloaded LFS checksum mismatch: " + name)
    else:
        digest = hashlib.sha1(f"blob {record.size}\0".encode())
        digest.update(path.read_bytes())
        if digest.hexdigest() != record.blob_id:
            raise ValueError("Downloaded Git blob mismatch: " + name)
    return path
