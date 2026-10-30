"""Sparse, authenticated access to pinned KITScenes scene archives.

Each val scene is one ~1.8 GB TAR on Hugging Face. The benchmark needs about a
dozen JPEGs and three small text files per scene, so this module:

1. reads a small prefix of the archive with an HTTP ``Range`` request,
2. walks the TAR header chain, fetching only headers that are not buffered,
3. caches the member table per archive, and
4. fetches member payloads on demand, each with a SHA-256 sidecar.

Cache entries are namespaced by ``<repo>/<revision>/`` and bound to the archive
digest, so bytes for one dataset revision or archive are never served for
another.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import quote

import numpy as np

from data_parsing.kit_scenes.source import SceneArchive

DEFAULT_HF_ENDPOINT = "https://huggingface.co"
TAR_BLOCK = 512
PREFETCH_BYTES = 24 * 1024 * 1024
WINDOW_BYTES = 64 * 1024
INDEX_SCHEMA_VERSION = "simforge_kitscenes_sparse_index_v1"

TUM_POSES_FILENAME = "poses.txt"
REFERENCE_TIMESTAMP_FILENAME = "timestamp.reference.txt"
CALIBRATION_FILENAME = "calibration/calib.json"
FRAME_INDEX_WIDTH = 10
CAMERA_SUFFIX = "_pinhole"


# ---------------------------------------------------------------------------
# Range sources
# ---------------------------------------------------------------------------


class RangeSource(Protocol):
    size: int | None
    requests: int
    bytes_read: int

    def read(self, offset: int, length: int) -> bytes: ...


class HttpRangeSource:
    """HTTP ``Range`` reader that resolves the authenticated URL once.

    The first request goes to the authenticated ``resolve`` URL and follows the
    redirect to a signed CDN URL, which is reused until it expires (401/403).
    The token is sent only to the first host, never to the CDN.
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        expected_size: int | None = None,
        timeout_s: float = 60.0,
        attempts: int = 6,
        session: Any | None = None,
    ) -> None:
        self._url = url
        self._token = token
        self.size = expected_size
        self._timeout = timeout_s
        self._attempts = attempts
        self._session = session
        self._resolved: str | None = None
        self.requests = 0
        self.bytes_read = 0

    def _get_session(self) -> Any:
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def read(self, offset: int, length: int) -> bytes:
        import requests

        if offset < 0 or length < 0:
            raise ValueError("offset and length must be non-negative")
        if length == 0:
            return b""
        if self.size is not None:
            if offset >= self.size:
                return b""
            length = min(length, self.size - offset)
        end = offset + length - 1
        last_error: Exception | None = None
        for attempt in range(self._attempts):
            resolved = self._resolved is not None
            target = self._resolved if resolved else self._url
            headers = {"Range": f"bytes={offset}-{end}"}
            if not resolved and self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            try:
                response = self._get_session().get(
                    target, headers=headers, allow_redirects=True, timeout=self._timeout
                )
            except (requests.ConnectionError, requests.Timeout) as error:
                last_error = error
                time.sleep(2**attempt)
                continue
            self.requests += 1 + len(response.history)
            status = response.status_code
            if status == 206:
                start, total = _parse_content_range(response.headers.get("Content-Range"))
                if start != offset:
                    raise RuntimeError(f"server returned bytes from {start}, expected {offset}")
                if total is not None:
                    if self.size is not None and total != self.size:
                        raise RuntimeError(f"archive size {total} differs from the pinned {self.size}")
                    self.size = total
                if not resolved:
                    self._resolved = response.url
                payload = bytes(response.content[:length])
                self.bytes_read += len(payload)
                return payload
            if status in (401, 403):
                if resolved:
                    self._resolved = None  # signed CDN URL expired: resolve again
                    continue
                raise PermissionError(
                    f"HTTP {status} for {self._url}: accept the KITScenes terms and check the Hugging Face token"
                )
            if status == 404:
                raise FileNotFoundError(f"HTTP 404 for {self._url}")
            if status == 429 or status >= 500:
                last_error = RuntimeError(f"HTTP {status}")
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"unexpected HTTP {status} for {target}")
        raise RuntimeError(f"range request failed after {self._attempts} attempts: {last_error}")


def _parse_content_range(value: str | None) -> tuple[int, int | None]:
    if not value or not value.startswith("bytes "):
        raise RuntimeError(f"missing or invalid Content-Range: {value!r}")
    span, _, total = value[len("bytes ") :].partition("/")
    start = int(span.partition("-")[0])
    return start, (None if total.strip() in {"", "*"} else int(total))


def hf_dataset_file_url(repo_id: str, filename: str, *, revision: str, endpoint: str | None = None) -> str:
    """Immutable ``resolve`` URL of one dataset file at a pinned revision."""
    base = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT).rstrip("/")
    return f"{base}/datasets/{repo_id}/resolve/{quote(revision, safe='')}/{quote(filename)}"


def resolve_hf_token() -> str | None:
    """Token from the Hugging Face credential store or ``HF_TOKEN``; never from the command line."""
    from huggingface_hub import get_token

    token = get_token()
    return str(token) if token else (os.environ.get("HF_TOKEN") or None)


# ---------------------------------------------------------------------------
# TAR index
# ---------------------------------------------------------------------------


def _octal_or_base256(raw: bytes) -> int:
    if raw and raw[0] & 0x80:
        return int.from_bytes(raw[1:], "big")
    text = raw.split(b"\x00", 1)[0].strip()
    return int(text, 8) if text else 0


def _name(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="surrogateescape")


def _parse_pax(payload: bytes) -> dict[str, str]:
    records: dict[str, str] = {}
    position = 0
    while position < len(payload):
        space = payload.index(b" ", position)
        length = int(payload[position:space])
        key, _, value = payload[space + 1 : position + length - 1].partition(b"=")
        records[key.decode("utf-8")] = value.decode("utf-8")
        position += length
    return records


class UnsupportedTarFeature(ValueError):
    """The archive uses a TAR feature this reader does not implement."""


@dataclass(frozen=True)
class TarEntry:
    name: str
    offset: int
    size: int
    is_file: bool


def build_tar_index(
    source: RangeSource,
    *,
    prefetch_bytes: int = PREFETCH_BYTES,
    window_bytes: int = WINDOW_BYTES,
) -> tuple[dict[str, TarEntry], int]:
    """Walk the header chain through ``source``; returns (entries by name, archive size)."""
    buffer = source.read(0, prefetch_bytes)
    buffer_start = 0

    def ensure(offset: int, length: int) -> bytes:
        nonlocal buffer, buffer_start
        if offset < buffer_start or offset + length > buffer_start + len(buffer):
            buffer = source.read(offset, max(window_bytes, length))
            buffer_start = offset
        chunk = buffer[offset - buffer_start : offset - buffer_start + length]
        if len(chunk) < length:
            raise EOFError(f"archive ends inside a header at offset {offset}")
        return chunk

    entries: dict[str, TarEntry] = {}
    position = 0
    pending_name: str | None = None
    pending_pax: dict[str, str] = {}
    while source.size is None or position < source.size:
        try:
            block = ensure(position, TAR_BLOCK)
        except EOFError:
            break
        if block == bytes(TAR_BLOCK):
            break
        raw_name = _name(block[0:100])
        prefix = _name(block[345:500]) if block[257:262] == b"ustar" else ""
        name = f"{prefix}/{raw_name}" if prefix else raw_name
        size = _octal_or_base256(block[124:136])
        flag = block[156:157]
        data_offset = position + TAR_BLOCK
        padded = math.ceil(size / TAR_BLOCK) * TAR_BLOCK
        if flag == b"g":
            # Global PAX headers change the meaning of every later member. The official
            # KITScenes archives use only per-member (local) PAX headers, so a global one
            # is refused rather than silently ignored.
            raise UnsupportedTarFeature(f"global PAX header at offset {position} is not supported")
        if flag in (b"L", b"x"):
            payload = ensure(data_offset, size) if size else b""
            if flag == b"L":
                pending_name = _name(payload)
            else:
                pending_pax = _parse_pax(payload)
            position = data_offset + padded
            continue
        name = pending_name or pending_pax.get("path") or name
        if flag == b"5":
            name = name.rstrip("/")
        size = int(pending_pax["size"]) if "size" in pending_pax else size
        pending_name, pending_pax = None, {}
        if flag in (b"0", b"\x00", b"7", b"5"):
            if name in entries:
                raise ValueError(f"duplicate archive member {name!r}")
            entries[name] = TarEntry(name, data_offset, size, is_file=flag in (b"0", b"\x00", b"7"))
        position = data_offset + math.ceil(size / TAR_BLOCK) * TAR_BLOCK
    archive_size = source.size if source.size is not None else position
    return entries, int(archive_size)


# ---------------------------------------------------------------------------
# Scene access with an identity-bound cache
# ---------------------------------------------------------------------------


class CacheIdentityError(RuntimeError):
    """A cache entry does not belong to the archive it is being used for."""


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


class KitScene:
    """One scene: member table from the sparse index plus cached payload reads."""

    def __init__(self, scene_id: str, files: Mapping[str, TarEntry], source: RangeSource, cache_dir: Path) -> None:
        self.scene_id = scene_id
        self._files = dict(files)
        self._source = source
        self._cache = cache_dir

    def relative_files(self) -> list[str]:
        return sorted(self._files)

    def read(self, relative: str) -> bytes:
        entry = self._files.get(relative)
        if entry is None:
            raise FileNotFoundError(f"{relative} is not in scene {self.scene_id}")
        path = self._cache / relative
        sidecar = path.with_name(path.name + ".sha256")
        if path.is_file() and sidecar.is_file():
            recorded_digest, recorded_size = sidecar.read_text().split()
            payload = path.read_bytes()
            if int(recorded_size) == entry.size == len(payload) and hashlib.sha256(payload).hexdigest() == recorded_digest:
                return payload
        payload = self._source.read(entry.offset, entry.size)
        if len(payload) != entry.size:
            raise RuntimeError(f"short read for {relative}: {len(payload)} of {entry.size} bytes")
        _atomic_write(path, payload)
        _atomic_write(sidecar, f"{hashlib.sha256(payload).hexdigest()} {entry.size}\n".encode("ascii"))
        return payload

    def text(self, relative: str) -> str:
        return self.read(relative).decode("utf-8")

    # -- full-scene accessors: evaluation only (ground truth, population checks) --

    def pose_track(self) -> PoseTrack:
        return parse_tum_poses(self.text(TUM_POSES_FILENAME))

    def reference_timestamps_ns(self) -> np.ndarray:
        return parse_timestamp_file(self.text(REFERENCE_TIMESTAMP_FILENAME))

    def camera_timestamps_ns(self, camera: str) -> np.ndarray | None:
        path = f"timestamp.{camera}.txt"
        return parse_timestamp_file(self.text(path)) if path in self._files else None

    # -- bounded accessors: the only scene data the model-input builder may use --
    #
    # ``poses.txt`` and the timestamp files are single archive members, so their bytes
    # arrive as one blob. They are then decoded and parsed line by line only until the
    # rows 0..frame_index have been read; no later row is decoded, parsed, counted or
    # validated (see ``parse_pose_prefix`` / ``parse_timestamp_prefix``). Camera frames
    # are fetched one member at a time and refused after t0.

    def pose_rows_until(self, frame_index: int) -> PoseTrack:
        return parse_pose_prefix(self.read(TUM_POSES_FILENAME), frame_index + 1)

    def reference_timestamps_until(self, frame_index: int) -> np.ndarray:
        return parse_timestamp_prefix(self.read(REFERENCE_TIMESTAMP_FILENAME), frame_index + 1)

    def camera_timestamps_until(self, camera: str, frame_index: int) -> np.ndarray | None:
        """Camera timestamps 0..frame_index, or ``None`` when the file is absent or shorter."""
        path = f"timestamp.{camera}.txt"
        if path not in self._files:
            return None
        try:
            return parse_timestamp_prefix(self.read(path), frame_index + 1)
        except PrefixTooShort:
            return None

    def camera_image_until(self, camera: str, frame: int, frame_index: int) -> bytes:
        if frame > frame_index:
            raise ValueError(f"{self.scene_id}: frame {frame} of {camera} lies after t0 (frame {frame_index})")
        return self.camera_image(camera, frame)

    def calibration(self) -> dict[str, Any]:
        payload = json.loads(self.text(CALIBRATION_FILENAME))
        if not isinstance(payload, dict):
            raise ValueError("calib.json must contain an object")
        return payload

    def camera_frame_indices(self, camera: str) -> list[int]:
        prefix = f"{camera}/"
        indices = []
        for path in self._files:
            if path.startswith(prefix) and path.endswith(".jpg"):
                stem = path[len(prefix) : -4]
                if len(stem) == FRAME_INDEX_WIDTH and stem.isdigit():
                    indices.append(int(stem))
        return sorted(indices)

    def camera_image(self, camera: str, frame: int) -> bytes:
        return self.read(f"{camera}/{frame:0{FRAME_INDEX_WIDTH}d}.jpg")


class KitSceneStore:
    """Opens scenes from the pinned archive manifest with cached sparse indices."""

    def __init__(
        self,
        *,
        archives: Mapping[str, SceneArchive],
        cache_root: Path,
        repo_id: str,
        revision: str,
        token: str | None,
        endpoint: str | None = None,
        source_factory: Any = None,
    ) -> None:
        self._archives = dict(archives)
        self._root = Path(cache_root) / repo_id.replace("/", "__") / revision
        self._repo_id = repo_id
        self._revision = revision
        self._token = token
        self._endpoint = endpoint
        self._factory = source_factory
        self.open_requests = 0

    def _source(self, archive: SceneArchive) -> RangeSource:
        if self._factory is not None:
            return self._factory(archive)
        url = hf_dataset_file_url(self._repo_id, archive.filename, revision=self._revision, endpoint=self._endpoint)
        return HttpRangeSource(url, token=self._token, expected_size=archive.size_bytes)

    def open(self, scene_id: str) -> KitScene:
        archive = self._archives.get(scene_id)
        if archive is None:
            raise ValueError(f"scene {scene_id} is not in the archive manifest")
        identity = {
            "dataset_repo_id": self._repo_id,
            "dataset_revision": self._revision,
            "scene_id": scene_id,
            "archive_path": archive.filename,
            "archive_sha256": archive.sha256,
            "archive_size": archive.size_bytes,
            "schema_version": INDEX_SCHEMA_VERSION,
        }
        index_path = self._root / "index" / f"{scene_id}.{archive.sha256}.json"
        source = self._source(archive)
        if index_path.is_file():
            stored = json.loads(index_path.read_text())
            mismatched = [key for key, value in identity.items() if stored.get(key) != value]
            if mismatched:
                raise CacheIdentityError(f"sparse index {index_path} does not match the archive: {mismatched}")
            entries = {
                name: TarEntry(name, int(offset), int(size), bool(is_file))
                for name, offset, size, is_file in stored["members"]
            }
        else:
            entries, archive_size = build_tar_index(source)
            if archive_size != archive.size_bytes:
                raise RuntimeError(f"archive size {archive_size} differs from the manifest {archive.size_bytes}")
            payload = {
                **identity,
                "members": [[e.name, e.offset, e.size, e.is_file] for e in entries.values()],
            }
            _atomic_write(index_path, (json.dumps(payload) + "\n").encode("utf-8"))
        self.open_requests += getattr(source, "requests", 0)
        prefix = f"{scene_id}/"
        files = {
            (name[len(prefix) :] if name.startswith(prefix) else name): entry
            for name, entry in entries.items()
            if entry.is_file
        }
        return KitScene(scene_id, files, source, self._root / "members" / archive.sha256)


# ---------------------------------------------------------------------------
# Byte-level parsers (KITScenes SDK semantics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseTrack:
    timestamps_ns: np.ndarray  # (T,) int64
    xyz: np.ndarray  # (T, 3) world (UTM) frame
    quat_xyzw: np.ndarray  # (T, 4)

    def __len__(self) -> int:
        return int(len(self.timestamps_ns))

    def rotation_matrices(self) -> np.ndarray:
        x, y, z, w = np.moveaxis(self.quat_xyzw / np.linalg.norm(self.quat_xyzw, axis=-1, keepdims=True), -1, 0)
        matrix = np.empty((len(self), 3, 3), dtype=np.float64)
        matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
        matrix[:, 0, 1] = 2 * (x * y - z * w)
        matrix[:, 0, 2] = 2 * (x * z + y * w)
        matrix[:, 1, 0] = 2 * (x * y + z * w)
        matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
        matrix[:, 1, 2] = 2 * (y * z - x * w)
        matrix[:, 2, 0] = 2 * (x * z - y * w)
        matrix[:, 2, 1] = 2 * (y * z + x * w)
        matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return matrix


def parse_tum_poses(text: str) -> PoseTrack:
    """TUM lines ``t tx ty tz qx qy qz qw``; timestamps converted like the SDK (``int(t * 1e9)``)."""
    stamps: list[int] = []
    xyz: list[list[float]] = []
    quat: list[list[float]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"malformed TUM pose on line {number}")
        values = [float(part) for part in parts]
        stamps.append(int(values[0] * 1e9))
        xyz.append(values[1:4])
        quat.append(values[4:8])
    if not stamps:
        raise ValueError("poses.txt contains no poses")
    timestamps = np.asarray(stamps, dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("poses.txt timestamps must be strictly increasing")
    return PoseTrack(timestamps, np.asarray(xyz, dtype=np.float64), np.asarray(quat, dtype=np.float64))


class PrefixTooShort(ValueError):
    """A file ends before the requested number of data rows."""


def _prefix_rows(payload: bytes, rows: int, *, skip_comments: bool) -> list[tuple[int, str]]:
    """The first ``rows`` data rows as (line number, text); later lines are never decoded."""
    if rows < 1:
        raise ValueError("rows must be positive")
    found: list[tuple[int, str]] = []
    for number, raw in enumerate(io.BytesIO(payload), start=1):  # lazy: one line at a time
        line = raw.decode("utf-8").strip()
        if not line or (skip_comments and line.startswith("#")):
            continue
        found.append((number, line))
        if len(found) == rows:
            return found
    raise PrefixTooShort(f"file has {len(found)} data rows, {rows} required")


def parse_pose_prefix(payload: bytes, rows: int) -> PoseTrack:
    """The first ``rows`` poses of ``poses.txt``, parsed and validated like ``parse_tum_poses``."""
    stamps: list[int] = []
    xyz: list[list[float]] = []
    quat: list[list[float]] = []
    for number, line in _prefix_rows(payload, rows, skip_comments=True):
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"malformed TUM pose on line {number}")
        values = [float(part) for part in parts]
        stamps.append(int(values[0] * 1e9))
        xyz.append(values[1:4])
        quat.append(values[4:8])
    timestamps = np.asarray(stamps, dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("poses.txt timestamps must be strictly increasing")
    return PoseTrack(timestamps, np.asarray(xyz, dtype=np.float64), np.asarray(quat, dtype=np.float64))


def parse_timestamp_prefix(payload: bytes, rows: int) -> np.ndarray:
    """The first ``rows`` timestamps, converted like ``parse_timestamp_file``."""
    values = []
    for _, line in _prefix_rows(payload, rows, skip_comments=False):
        value = float(line)
        values.append(int(value) if value > 1e12 else int(value * 1e9))
    return np.asarray(values, dtype=np.int64)


def parse_timestamp_file(text: str) -> np.ndarray:
    """One timestamp per line; values above 1e12 are already nanoseconds."""
    values = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if line:
            value = float(line)
            values.append(int(value) if value > 1e12 else int(value * 1e9))
    return np.asarray(values, dtype=np.int64)


@dataclass(frozen=True)
class Calibration:
    camera: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def hfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan(self.width / (2.0 * self.fx)))

    @property
    def vfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan(self.height / (2.0 * self.fy)))


def parse_calibration(payload: Mapping[str, Any], camera: str) -> Calibration:
    key = camera if camera in payload else camera + CAMERA_SUFFIX
    if key not in payload:
        raise KeyError(f"camera {camera} is not in calib.json")
    entry = payload[key]
    intrinsics = entry["intrinsics"]
    resolution = entry.get("resolution")
    if resolution is None:
        raise ValueError(f"{camera}: calibration has no resolution")
    return Calibration(
        camera=camera,
        fx=float(intrinsics["focal_length"]),
        fy=float(intrinsics["focal_length"]),
        cx=float(intrinsics["principal_point_u"]),
        cy=float(intrinsics["principal_point_v"]),
        width=int(resolution["width"]),
        height=int(resolution["height"]),
    )


@dataclass(frozen=True)
class VirtualCrop:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in source pixels
    source: Calibration
    target_hfov_deg: float
    target_aspect_ratio: float

    @property
    def resulting_hfov_deg(self) -> float:
        width = self.box[2] - self.box[0]
        return 2.0 * math.degrees(math.atan(width / (2.0 * self.source.fx)))

    def to_metadata(self) -> dict[str, Any]:
        x0, y0, x1, y1 = self.box
        return {
            "source_camera": self.source.camera,
            "source_resolution_wh": [self.source.width, self.source.height],
            "source_focal_length_px": [self.source.fx, self.source.fy],
            "source_principal_point_px": [self.source.cx, self.source.cy],
            "source_hfov_deg": round(self.source.hfov_deg, 3),
            "target_hfov_deg": self.target_hfov_deg,
            "target_aspect_ratio": self.target_aspect_ratio,
            "crop_box_x0_y0_x1_y1": [x0, y0, x1, y1],
            "crop_size_wh": [x1 - x0, y1 - y0],
            "resulting_hfov_deg": round(self.resulting_hfov_deg, 3),
            "resampling": "none (pixel-exact centre crop)",
        }


def virtual_telephoto_crop(calibration: Calibration, *, hfov_deg: float, aspect_ratio: float) -> VirtualCrop:
    """Centre crop on the calibrated principal point with width ``2 f tan(hfov/2)``.

    Raises when the crop would leave the source image; the centre is never moved.
    """
    if not 0.0 < hfov_deg < calibration.hfov_deg:
        raise ValueError(f"{calibration.camera}: target HFOV must be narrower than the source")
    half_w = calibration.fx * math.tan(math.radians(hfov_deg) / 2.0)
    half_h = half_w / aspect_ratio
    x0, x1 = int(round(calibration.cx - half_w)), int(round(calibration.cx + half_w))
    y0, y1 = int(round(calibration.cy - half_h)), int(round(calibration.cy + half_h))
    if x0 < 0 or y0 < 0 or x1 > calibration.width or y1 > calibration.height:
        raise ValueError(f"{calibration.camera}: the centred crop leaves the source image")
    return VirtualCrop((x0, y0, x1, y1), calibration, float(hfov_deg), float(aspect_ratio))


def crop_to_png(jpeg: bytes, box: tuple[int, int, int, int]) -> bytes:
    """Decode, crop without resampling, and re-encode losslessly."""
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as image:
        cropped = image.convert("RGB").crop(box)
    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()
