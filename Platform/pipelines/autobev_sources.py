"""Upstream sources of the AutoBEV-only datasets, used only by wf_acquire_raw_snapshot.

nuScenes and Argoverse 2 come from public S3 buckets without credentials;
ONCE and SimBEV from Google Drive. Drive cuts large downloads when a file's
download quota runs out, so every Drive read is a range request that is
retried with backoff, and whole-file downloads resume where they stopped.

Every unit becomes a directory of upstream files, byte for byte (archives
are extracted, as the KITScenes downloader does), which the acquisition tars
and indexes like any other unit:

| Dataset | Units | Bytes |
|---|---|---|
| nuScenes | trainval metadata, map expansion v1.3, the ten keyframe archives | about 46 GB |
| Argoverse 2 | one per non-Pittsburgh train or val log, only the files its samples read | about 150 MB per log |
| ONCE | train infos, roof LiDAR, the six used cameras, ONCE-3DLanes | about 20 GB |
| SimBEV | infos, and of each sensor and ground-truth zip only the members a sample reads | about 25 GB of 1.5 TB |

All four licenses forbid commercial use; Autoware uses them non-commercially.
"""

from __future__ import annotations

import io
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final

from Platform.pipelines.autoe2e_dataset import AutoE2EDataset

LICENSES: Final = {
    AutoE2EDataset.NUSCENES: {
        "name": "nuScenes Terms of Use (CC BY-NC-SA 4.0 for non-commercial use)",
        "url": "https://www.nuscenes.org/terms-of-use",
        "commercial_use": False,
    },
    AutoE2EDataset.ARGOVERSE2: {
        "name": "Argoverse Terms of Use (CC BY-NC-SA 4.0)",
        "url": "https://www.argoverse.org/about.html#terms-of-use",
        "commercial_use": False,
    },
    AutoE2EDataset.ONCE: {
        "name": "ONCE Terms of Use (CC BY-NC-SA 4.0); ONCE-3DLanes states no own license and is derived from ONCE",
        "url": "https://once-for-auto-driving.github.io/terms_of_use.html",
        "commercial_use": False,
    },
    AutoE2EDataset.SIMBEV: {
        "name": "SimBEV dataset (CC BY-NC-SA 4.0)",
        "url": "https://github.com/GoodarzMehr/SimBEV",
        "commercial_use": False,
    },
}

NUSCENES_BUCKET: Final = "motional-nuscenes"
NUSCENES_PREFIX: Final = "public/v1.0"
NUSCENES_CHECKSUMS: Final = "md5.checksum"
# Unit id -> (archive, directory of the unit tree it extracts into).
NUSCENES_ARCHIVES: Final = {
    "meta": ("v1.0-trainval_meta.tgz", ""),
    # The devkit layout keeps the expansion under maps/.
    "map-expansion": ("nuScenes-map-expansion-v1.3.zip", "maps"),
    **{f"keyframes{index:02d}": (f"v1.0-trainval{index:02d}_keyframes.tgz", "") for index in range(1, 11)},
}

# ONCE train archives on Google Drive (download page folders), extracted whole.
# cam03, the telephoto twin of cam01, is not used by the packer and not fetched.
ONCE_ARCHIVES: Final = {
    "infos": "1Qa6JYbp7SEZtq7wI0Sw7bvhLc-CZdX8I",
    "lidar": "1fy_R1TOKOyvMtCgwR5YwpZtiih0c_oD5",
    "cam01": "1bOVcrja8AZi8fmeZwYS98Y9mfDHU7wYl",
    "cam05": "1P2N3uqbuFQ-ACJJv9bt8Pa15jnncrWyc",
    "cam06": "14ARNQu7-Rscr1vta-pv35z9kfYnPeV0X",
    "cam07": "1n9t3nmy84bTJHnUOF2KUyfFOoWnyGJCi",
    "cam08": "1rz239PKqkFmt9m0r95AK7upjgKm1jVrP",
    "cam09": "10CajrJ02H4YEwWM87N1O1mWzYjdV18P3",
    "lanes": "1-blzGV6Q0R-6aa0dsRHjR9kKiA5o5bqv",
}

# SimBEV 1.0 zips in Drive folder 14MytQeGmW80Btg_AGPNrE18ZLdLzyGx5; members are read by range.
SIMBEV_ARCHIVES: Final = {
    "infos": "1E44LJVwGj1tV5t1cCBacBCXkn4jKHKAD",
    "det": "1dIQvIB1GtX39Cj5tvfK8l6gQCuglutq8",
    "seg": "1TdUGqA6fa0VxiSFHLb9fL8JKbuJJE9aI",
    "RGB-CAM_FRONT": "1j8QrQa7V_s-L1WHwbrce4HQfYTdk0mda",
    "RGB-CAM_FRONT_LEFT": "18LzUDAJ4e-AXUPQ9OfmgX-BjBbV98_QE",
    "RGB-CAM_FRONT_RIGHT": "1ZBloj2vwXL_M0NOVVTv5asvnzWApsMZi",
    "RGB-CAM_BACK_LEFT": "1yrwxsY1Q20BSE4vwfvEMKlOacQFNsscC",
    "RGB-CAM_BACK": "19eSb17NDChoNi2V2Ll6jk7qwa7TUQeli",
    "RGB-CAM_BACK_RIGHT": "18nSgohVRcCEEcDFsbMR54vTi56DfJQr6",
    "LIDAR": "1LIp27EaYQyXIJf86RR0fN4zb7rRzxLoG",
}

# Drive quota refusals last minutes to hours: back off 1, 2, 4 ... up to 30 minutes, ten times.
DRIVE_ATTEMPTS: Final = 10
DRIVE_BACKOFF_S: Final = 60
DRIVE_MAX_BACKOFF_S: Final = 1800
# A resumed whole-file download asks for this many bytes per request.
DRIVE_CHUNK: Final = 256 * 1024**2
# Zip directory reads are cached in blocks of this size.
RANGE_BLOCK: Final = 8 * 1024**2
MEMBER_THREADS: Final = 16


def unsigned_s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED, max_pool_connections=32))


def list_keys(s3, bucket: str, prefix: str) -> list[dict]:
    items = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        items += page.get("Contents", [])
    return items


def list_prefixes(s3, bucket: str, prefix: str) -> list[str]:
    names = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        names += [item["Prefix"][len(prefix):].rstrip("/") for item in page.get("CommonPrefixes", [])]
    return names


def _download_s3(s3, bucket: str, key: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(target))


def _md5(path: Path) -> str:
    import hashlib

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive: Path, into: Path) -> None:
    """Extract a .tgz, .tar or .zip of regular files and directories under ``into``."""
    import tarfile

    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as source:
            for name in source.namelist():
                if name.startswith("/") or ".." in name.split("/"):
                    raise ValueError(f"{archive.name}: unsafe member {name!r}")
            source.extractall(into)
        return
    with tarfile.open(archive) as source:
        # The "data" filter refuses links, devices and paths outside ``into``.
        source.extractall(into, filter="data")


# --- planning -------------------------------------------------------------


def plan_units(dataset: AutoE2EDataset) -> tuple[list[str], dict[str, Any]]:
    """Unit ids and the snapshot provenance: what each unit task fetches, and the license."""
    units, details = _PLANNERS[dataset]()
    return units, {"method": "upstream", "split": "train", "license": dict(LICENSES[dataset]), **details}


def _plan_nuscenes() -> tuple[list[str], dict[str, Any]]:
    s3 = unsigned_s3()
    body = s3.get_object(Bucket=NUSCENES_BUCKET, Key=f"{NUSCENES_PREFIX}/{NUSCENES_CHECKSUMS}")["Body"].read()
    md5 = {name: digest for digest, name in (line.split() for line in body.decode().splitlines() if line.strip())}
    archives = {
        unit: {"key": f"{NUSCENES_PREFIX}/{name}", "into": into, "md5": md5.get(name)}
        for unit, (name, into) in NUSCENES_ARCHIVES.items()
    }
    return sorted(archives), {"bucket": NUSCENES_BUCKET, "archives": archives}


def _plan_argoverse2() -> tuple[list[str], dict[str, Any]]:
    from data_parsing.argoverse2.pack import (
        HELD_OUT_CITY,
        SPLITS,
        UPSTREAM_BUCKET,
        UPSTREAM_PREFIX,
        is_map_archive,
        log_city,
        training_logs,
    )

    s3 = unsigned_s3()
    roots = [(split, log) for split in SPLITS for log in list_prefixes(s3, UPSTREAM_BUCKET, f"{UPSTREAM_PREFIX}/{split}/")]

    def map_name(root):
        split, log = root
        names = [item["Key"] for item in list_keys(s3, UPSTREAM_BUCKET, f"{UPSTREAM_PREFIX}/{split}/{log}/map/")]
        (name,) = [name for name in names if is_map_archive(name)]
        return f"{split}-{log}", name.rsplit("/", 1)[-1]

    with ThreadPoolExecutor(32) as pool:
        maps = dict(pool.map(map_name, roots))
    units = training_logs(maps)
    logs = {unit: {"split": unit.split("-", 1)[0], "log": unit.split("-", 1)[1], "map": maps[unit]} for unit in units}
    held_out = sorted(unit for unit in maps if log_city(maps[unit]) == HELD_OUT_CITY)
    return units, {"bucket": UPSTREAM_BUCKET, "logs": logs, "held_out_city": HELD_OUT_CITY, "held_out_logs": held_out}


def _plan_once() -> tuple[list[str], dict[str, Any]]:
    return sorted(ONCE_ARCHIVES), {"drive_files": dict(ONCE_ARCHIVES)}


def _plan_simbev() -> tuple[list[str], dict[str, Any]]:
    return sorted(SIMBEV_ARCHIVES), {"drive_files": dict(SIMBEV_ARCHIVES)}


_PLANNERS: Final[dict[AutoE2EDataset, Callable[[], tuple[list[str], dict[str, Any]]]]] = {
    AutoE2EDataset.NUSCENES: _plan_nuscenes,
    AutoE2EDataset.ARGOVERSE2: _plan_argoverse2,
    AutoE2EDataset.ONCE: _plan_once,
    AutoE2EDataset.SIMBEV: _plan_simbev,
}


# --- fetching -------------------------------------------------------------


def fetch_unit(dataset: AutoE2EDataset, unit_id: str, provenance: dict[str, Any], root: Path, work: Path) -> None:
    """Write one unit's upstream files under ``root``; ``work`` holds downloads meanwhile."""
    _FETCHERS[dataset](unit_id, provenance, root, work)


def _fetch_nuscenes(unit_id: str, provenance: dict[str, Any], root: Path, work: Path) -> None:
    item = provenance["archives"][unit_id]
    archive = work / item["key"].rsplit("/", 1)[-1]
    _download_s3(unsigned_s3(), provenance["bucket"], item["key"], archive)
    if item["md5"] is not None and _md5(archive) != item["md5"]:
        raise IOError(f"{archive.name}: MD5 differs from the upstream md5.checksum")
    extract_archive(archive, root / item["into"])
    archive.unlink()


def _fetch_argoverse2(unit_id: str, provenance: dict[str, Any], root: Path, work: Path) -> None:
    import pandas as pd

    from data_parsing.argoverse2.pack import METADATA_FILES, UPSTREAM_PREFIX, needed_files, plan_samples

    s3, bucket, log = unsigned_s3(), provenance["bucket"], provenance["logs"][unit_id]
    base = f"{UPSTREAM_PREFIX}/{log['split']}/{log['log']}"
    tree = root / log["split"] / log["log"]
    for name in (*METADATA_FILES, f"map/{log['map']}"):
        _download_s3(s3, bucket, f"{base}/{name}", tree / name)
    camera_times: dict[str, list[int]] = {}
    for item in list_keys(s3, bucket, f"{base}/sensors/cameras/"):
        camera, stamp = item["Key"][len(f"{base}/sensors/cameras/"):].removesuffix(".jpg").split("/")
        camera_times.setdefault(camera, []).append(int(stamp))
    annotated = pd.read_feather(tree / "annotations.feather").timestamp_ns.unique().tolist()
    names = needed_files(plan_samples(log["log"], annotated, camera_times))
    with ThreadPoolExecutor(MEMBER_THREADS) as pool:
        list(pool.map(lambda name: _download_s3(s3, bucket, f"{base}/{name}", tree / name), names))


def _fetch_once(unit_id: str, provenance: dict[str, Any], root: Path, work: Path) -> None:
    archive = work / f"{unit_id}.tar"
    drive_download(provenance["drive_files"][unit_id], archive)
    extract_archive(archive, root)
    archive.unlink()


def _fetch_simbev(unit_id: str, provenance: dict[str, Any], root: Path, work: Path) -> None:
    from data_parsing.simbev.pack import needed_member

    url = drive_url(provenance["drive_files"][unit_id])
    with zipfile.ZipFile(HTTPRangeFile(url, http_size(url))) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
    wanted = [info for info in infos if needed_member(info.filename)]
    if not wanted:
        raise ValueError(f"SimBEV {unit_id}: no member of the archive is needed")

    def fetch(info: zipfile.ZipInfo) -> None:
        if info.filename.startswith("/") or ".." in info.filename.split("/"):
            raise ValueError(f"SimBEV {unit_id}: unsafe member {info.filename!r}")
        target = root / info.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zip_member(url, info))

    with ThreadPoolExecutor(MEMBER_THREADS) as pool:
        list(pool.map(fetch, wanted))


_FETCHERS: Final[dict[AutoE2EDataset, Callable[[str, dict[str, Any], Path, Path], None]]] = {
    AutoE2EDataset.NUSCENES: _fetch_nuscenes,
    AutoE2EDataset.ARGOVERSE2: _fetch_argoverse2,
    AutoE2EDataset.ONCE: _fetch_once,
    AutoE2EDataset.SIMBEV: _fetch_simbev,
}


# --- Google Drive ---------------------------------------------------------


def drive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _retry(read: Callable[[], bytes], what: str, sleep: Callable[[float], None] = time.sleep) -> bytes:
    """Call ``read`` until it succeeds; Drive answers a used-up quota with an HTML error page."""
    import http.client

    for attempt in range(DRIVE_ATTEMPTS):
        try:
            return read()
        except (urllib.error.URLError, IOError, TimeoutError, http.client.HTTPException) as error:
            if attempt == DRIVE_ATTEMPTS - 1:
                raise IOError(f"{what}: gave up after {DRIVE_ATTEMPTS} attempts") from error
            sleep(min(DRIVE_BACKOFF_S * 2**attempt, DRIVE_MAX_BACKOFF_S))
    raise AssertionError("unreachable")


def http_range(url: str, start: int, end: int) -> bytes:
    """Bytes ``[start, end]`` of ``url``, retried; anything but a 206 of that length is an error."""

    def read() -> bytes:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise IOError(f"range request answered {response.status}")
            payload = response.read()
        if len(payload) != end - start + 1:
            raise IOError(f"short range read {len(payload)} of {end - start + 1}")
        return payload

    return _retry(read, f"{url} bytes {start}-{end}")


def http_size(url: str) -> int:
    def read() -> bytes:
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or "/" not in content_range:
            raise IOError(f"no Content-Range in the answer ({response.status})")
        return content_range.rsplit("/", 1)[1].encode()

    return int(_retry(read, f"size of {url}"))


def drive_download(file_id: str, target: Path) -> None:
    """Download a whole Drive file, resuming from the partial file a quota cut left behind."""
    url = drive_url(file_id)
    size = http_size(url)
    partial = target.with_name(f"{target.name}.partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("ab") as out:
        while (done := out.tell()) < size:
            out.write(http_range(url, done, min(done + DRIVE_CHUNK, size) - 1))
    partial.replace(target)


class HTTPRangeFile(io.RawIOBase):
    """A seekable read-only file over range requests, cached in blocks, for reading zip directories."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0
        self._blocks: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.position = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence] + offset
        return self.position

    def _block(self, index: int) -> bytes:
        if index not in self._blocks:
            if len(self._blocks) > 8:
                self._blocks.pop(next(iter(self._blocks)))
            start = index * RANGE_BLOCK
            self._blocks[index] = http_range(self.url, start, min(start + RANGE_BLOCK, self.size) - 1)
        return self._blocks[index]

    def read(self, size: int = -1) -> bytes:
        end = self.size if size < 0 else min(self.size, self.position + size)
        parts = []
        while self.position < end:
            index, offset = divmod(self.position, RANGE_BLOCK)
            chunk = self._block(index)[offset: offset + end - self.position]
            parts.append(chunk)
            self.position += len(chunk)
        return b"".join(parts)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


# Room for the local header's name and extra field in the first range request.
_LOCAL_HEADER_SLACK: Final = 256


def zip_member(url: str, info: zipfile.ZipInfo, get: Callable[[str, int, int], bytes] = http_range) -> bytes:
    """One zip member with one range request (two if its local extra field is long), CRC-checked."""
    start = info.header_offset
    blob = get(url, start, start + 30 + len(info.filename.encode()) + _LOCAL_HEADER_SLACK + info.compress_size - 1)
    if blob[:4] != b"PK\x03\x04":
        raise IOError(f"bad local header for {info.filename}")
    name_length, extra_length = struct.unpack("<HH", blob[26:30])
    data_start = 30 + name_length + extra_length
    if data_start + info.compress_size > len(blob):
        blob = get(url, start, start + data_start + info.compress_size - 1)
    data = blob[data_start: data_start + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise IOError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(data) != info.CRC:
        raise IOError(f"CRC mismatch for {info.filename}")
    return data
