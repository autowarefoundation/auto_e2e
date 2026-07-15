"""PreExtractedDataset: WebDataset-backed DataLoader for training.

Reads from local EBS shard cache (init container syncs from S3).
No video decode, no lerobot dependency. Sequential tar reads at full
disk bandwidth.

Usage:
    from data_parsing.pre_extracted import make_pre_extracted_loader

    loader = make_pre_extracted_loader("/data/shards", batch_size=8)
    for batch in loader:
        # batch["visual_tiles"]       (B, V, 3, 256, 256)  V real cameras
        # batch["map_input"]          (B, 3, 256, 256)     nav-map (map branch)
        # batch["egomotion_history"]  (B, 256)
        # batch["visual_history"]     (B, 896)
        # batch["trajectory_target"]  (B, 128)
        # batch["camera_params"]      (B, V, 3, 4)         if the manifest has calib
"""

from __future__ import annotations

import functools
import io
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torchvision import transforms

_HISTORY_STEPS = 64
_FUTURE_STEPS = 64
_HISTORY_SIGNALS = 4
_TARGET_SIGNALS = 2
_VISUAL_HISTORY_DIM = 896

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Camera frames are keyed "cam_<i>.jpg"; the nav-map is "map.jpg". The map MUST
# NOT be picked up as a camera view — matching cam_ explicitly (not any ".jpg")
# keeps V correct and stops the map being double-counted in the BEV projection.
_CAM_KEY_RE = re.compile(r"^cam_\d+\.jpg$")
# World-Model window frames: hist_<t>_cam_<v>.jpg / fut_<f>_cam_<v>.jpg (#13).
_HIST_KEY_RE = re.compile(r"^hist_(\d+)_cam_(\d+)\.jpg$")
_FUT_KEY_RE = re.compile(r"^fut_(\d+)_cam_(\d+)\.jpg$")


def _decode_image(data) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)) if isinstance(data, bytes) else data
    return _TRANSFORM(img)


class _PoolAccessor:
    """Path-based frame-pool reader (#121 §3.4d): ``frame_id -> jpeg bytes``.

    Reads ``<pool_dir>/<frame_id>.jpg`` on demand. Path-based (not an open handle)
    so it pickles cleanly to spawn DataLoader workers; the OS page cache shares the
    bytes across workers/epochs. Returns None if the pool dir is absent (a shard
    with no deduped windows), so the loader falls back to the legacy layout.
    """

    def __init__(self, pool_dir: str):
        self.pool_dir = pool_dir

    def __call__(self, frame_id: str) -> bytes:
        with open(f"{self.pool_dir}/{frame_id}.jpg", "rb") as f:
            return f.read()


def _make_pool_accessor(shard_dir: str):
    """Return a ``_PoolAccessor`` if ``<shard_dir>/pool/`` exists, else None."""
    pool_dir = Path(shard_dir) / "pool"
    return _PoolAccessor(str(pool_dir)) if pool_dir.is_dir() else None


def _decode_sample(sample: dict, pool=None) -> dict:
    """Decode a WebDataset sample into training tensors (geometry-free).

    Calibration is a per-dataset rig constant, not per-sample, so it is NOT
    decoded here — it is reconstructed once by ``make_pre_extracted_loader`` and
    exposed on the loader as ``.projection`` / ``.geometry_type``.

    ``pool`` is a frame-pool accessor (``frame_id -> jpeg bytes``) for shards packed
    with the deduped WM window (#121 §3.4d): the sample carries a
    ``window_index.json`` mapping (step,view)→frame_id and the pixels live in a
    sibling ``pool/`` dir. None on shards without a pool (imitation-only / legacy).
    """
    # Keys: "cam_0.jpg" ... "cam_{V-1}.jpg", optional "map.jpg",
    # "ego.npy", "meta.json", "__key__".
    cam_keys = sorted(
        (k for k in sample if _CAM_KEY_RE.match(k)),
        key=lambda k: int(k[len("cam_"):-len(".jpg")]),
    )
    frames = [_decode_image(sample[k]) for k in cam_keys]

    # Map view -> map branch. Absent (legacy shards / NVIDIA zeros) -> zeros.
    if "map.jpg" in sample:
        map_input = _decode_image(sample["map.jpg"])
    else:
        ref = frames[0] if frames else torch.zeros(3, 256, 256)
        map_input = torch.zeros_like(ref)

    # Ego: raw bytes → numpy → split into history and future
    ego_bytes = sample.get("ego.npy", b"")
    if isinstance(ego_bytes, bytes) and len(ego_bytes) > 0:
        ego = np.frombuffer(ego_bytes, dtype=np.float32).copy()
    else:
        ego = np.zeros(384, dtype=np.float32)

    # History: (64, 4) flattened = 256; Future: (64, 2) flattened = 128
    history_size = _HISTORY_STEPS * _HISTORY_SIGNALS
    ego_history = torch.from_numpy(ego[:history_size])
    ego_future = torch.from_numpy(ego[history_size:])

    out = {
        # Overlay inference derives noise from this stable identity. Keep it in
        # every batch so predictions do not depend on batch position or size.
        "sample_uid": sample.get("__key__", ""),
        "visual_tiles": torch.stack(frames),
        "map_input": map_input,
        "egomotion_history": ego_history,
        "visual_history": torch.zeros(_VISUAL_HISTORY_DIM),
        "trajectory_target": ego_future,
    }

    # Optional World-Model windows (#13/#3.4d): the sample carries window_index.json
    # (a (step,view)→frame_id map); the frames themselves are in the sibling pool/.
    # Rebuild history_frames [T, V, 3, H, W] and future_frames [F, V, 3, H, W]
    # (oldest→newest) — IDENTICAL tensors to the old per-sample hist_/fut_ layout,
    # just deduped in storage. Present only on WM shards; absent → no JEPA loss.
    # (Legacy hist_/fut_ shards still decode via _decode_window_legacy for back-compat.)
    windows = _decode_windows_from_pool(sample, pool)
    if windows is not None:
        out["history_frames"], out["future_frames"] = windows

    # Optional reasoning labels (#98): a per-sample "reasoning.json" member holds
    # a serialized ReasoningLabelRecord (same shard key → auto-aligned with this
    # sample's frames, no sample_id join). Decode it to per-sample target tensors
    # for HorizonReasoningLoss, flattened to top-level "reasoning__*" keys so
    # WebDataset's per-key default collation stacks them into [B, ...] batches.
    # Absent on shards packed without a teacher — the loader stays
    # reasoning-agnostic and training skips the reasoning loss.
    # ALWAYS emit reasoning__* keys so a batch that mixes labeled + unlabeled
    # samples collates (default_collate needs identical keys across a batch). An
    # unlabeled sample gets a fully-MASKED target (abstained record → IGNORE_INDEX
    # / zero source_weight), so it contributes nothing to the reasoning loss —
    # never a false-negative all-zero row. Shards packed with a teacher carry
    # reasoning.json; imitation-only samples don't, and both must batch together.
    reasoning_data = sample.get("reasoning.json")
    for key, tensor in _decode_reasoning_targets(reasoning_data).items():
        out[f"reasoning__{key}"] = tensor

    return out


def _decode_reasoning_targets(data) -> dict:
    """Decode the reasoning.json member into per-sample target tensors (#98).

    Lazy imports the data_processing tensorizer so importing this loader never
    pulls the label package unless training touches reasoning. When ``data`` is
    None (sample has no reasoning.json), return the tensors of an ABSTAINED
    record — all IGNORE_INDEX / zero source_weight — so the sample batches with
    labeled ones and is fully masked out of the reasoning loss (R9).
    """
    from data_processing.reasoning_label_generation.schema import ReasoningLabelRecord
    from data_processing.reasoning_label_generation.targets import (
        record_from_json,
        record_to_target_tensors,
    )

    if data is None:
        record = ReasoningLabelRecord.abstain(
            sample_id="", dataset_name="", teacher_provider="none",
            teacher_model="none", prompt_version="none",
            request_mode="clip_horizons", teacher_error="no reasoning.json")
    else:
        payload = json.loads(data.decode() if isinstance(data, (bytes, bytearray)) else data)
        record = record_from_json(payload)
    return record_to_target_tensors(record)


def _decode_window_from_index(index_steps, pool) -> torch.Tensor:
    """Decode one window (history or future) from a ``[[frame_id/view] /step]`` index.

    Looks each frame_id up in the pool accessor, decodes, and stacks into
    ``[steps, V, 3, H, W]`` (oldest→newest) — the exact shape/order the model
    consumes. Byte-identical to the old per-sample layout because the pool holds the
    same JPEG bytes the packer produced.
    """
    frame_steps = []
    for view_ids in index_steps:                       # one list of frame_ids per step
        view_frames = [_decode_image(pool(fid)) for fid in view_ids]
        frame_steps.append(torch.stack(view_frames))   # [V, 3, H, W]
    return torch.stack(frame_steps)                     # [steps, V, 3, H, W]


def _decode_windows_from_pool(sample: dict, pool):
    """Rebuild (history_frames, future_frames) from window_index.json + the pool.

    Returns None when the sample has no window_index.json (imitation-only). Falls
    back to the LEGACY per-sample hist_/fut_ member layout when a shard predates the
    frame pool (so old shards still train). Requires a pool accessor when a
    window_index.json is present.
    """
    idx_blob = sample.get("window_index.json")
    if idx_blob is None:
        return _decode_windows_legacy(sample)          # old hist_/fut_ layout or None
    if pool is None:
        raise ValueError(
            "sample has window_index.json but the loader has no frame pool accessor; "
            "the sibling pool/ dir must exist next to the shards (#121 §3.4d).")
    index = json.loads(idx_blob.decode() if isinstance(idx_blob, (bytes, bytearray)) else idx_blob)
    hist = _decode_window_from_index(index["history"], pool)
    fut = _decode_window_from_index(index["future"], pool)
    return hist, fut


def _decode_windows_legacy(sample: dict):
    """Legacy path: decode hist_<t>_cam_<v>.jpg / fut_<f>_cam_<v>.jpg members
    (pre-#3.4d shards). Returns (history, future) or None if absent."""
    def _one(key_re):
        matches = [(m, k) for k in sample if (m := key_re.match(k))]
        if not matches:
            return None
        steps = max(int(m.group(1)) for m, _ in matches) + 1
        frame_steps = []
        for t in range(steps):
            view_frames = [
                _decode_image(sample[k])
                for m, k in sorted(matches, key=lambda mk: int(mk[0].group(2)))
                if int(m.group(1)) == t
            ]
            frame_steps.append(torch.stack(view_frames))
        return torch.stack(frame_steps)
    hist = _one(_HIST_KEY_RE)
    fut = _one(_FUT_KEY_RE)
    if hist is None or fut is None:
        return None
    return hist, fut


def load_projection_from_manifest(shard_dir: str):
    """Reconstruct the per-dataset projection operator from manifest.json.

    Returns ``(projection, geometry_type)``. A dataset with real calibration
    stores an operator spec under ``projection`` in its manifest:

        {"geometry_type": "pinhole",
         "projection": {"type": "pinhole", "matrix": [[...]]}}   # [V,3,4]
        {"geometry_type": "ftheta",
         "projection": {"type": "ftheta", "t_camera_ego": [...],  # [V,4,4]
                        "fw_poly": [...], "cx": [...], "cy": [...],
                        "image_wh": [...], "max_theta": ...}}  # native (W,H), FOV

    A dataset without calibration (pseudo geometry, e.g. L2D) returns
    ``(None, "pseudo")`` and the caller runs the explicit pseudo path. This is
    the single geometry-reconstruction point, keeping the pinhole/f-theta split
    out of the training loop.
    """
    mpath = Path(shard_dir) / "manifest.json"
    # Missing manifest -> pseudo (a legacy shard has no geometry). But a manifest
    # that EXISTS and cannot be read must RAISE: silently degrading a calibrated
    # run to pseudo geometry would corrupt experiments. Corrupt/unreadable is a
    # hard error, not a fallback.
    if not mpath.exists():
        return None, "pseudo"
    try:
        manifest = json.loads(mpath.read_text())
    except (ValueError, OSError) as e:
        raise ValueError(
            f"manifest.json at {mpath} exists but could not be parsed ({e}); "
            f"refusing to silently fall back to pseudo geometry."
        ) from e

    spec = manifest.get("projection")
    if spec is None:
        return None, manifest.get("geometry_type", "pseudo")
    return projection_from_spec(spec)


def projection_from_spec(spec):
    """Reconstruct ``(projection, geometry_type)`` from a serialized spec dict.

    Shared by the single-dataset manifest path and the per-sample calib.json
    path (merged loader). ``spec`` is what ``CameraProjectionModel.to_spec()``
    produced; ``None`` returns the pseudo path.
    """
    from model_components.view_fusion.projection import (
        FThetaProjection,
        PinholeProjection,
    )

    if spec is None:
        return None, "pseudo"
    kind = spec.get("type")
    if kind in ("pinhole", "rectified_pinhole"):
        matrix = torch.tensor(spec["matrix"], dtype=torch.float32).unsqueeze(0)  # [1,V,3,4]
        return PinholeProjection(matrix, geometry_type=kind), kind
    if kind == "ftheta":
        def _t(key):
            return torch.tensor(spec[key], dtype=torch.float32).unsqueeze(0)
        # fw_poly may be serialized as a shared [K] (flat list) or per-view [V,K]
        # (nested list) — to_spec keeps a shared vector whole. Reconstruct the
        # matching shape so to_spec/load round-trip is exact: shared -> [K],
        # per-view -> [1,V,K].
        fw = spec["fw_poly"]
        if fw and isinstance(fw[0], (list, tuple)):
            fw_poly = torch.tensor(fw, dtype=torch.float32).unsqueeze(0)  # [1,V,K]
        else:
            fw_poly = torch.tensor(fw, dtype=torch.float32)               # [K] shared
        max_theta = spec.get("max_theta")
        if isinstance(max_theta, (list, tuple)):
            max_theta = torch.tensor(max_theta, dtype=torch.float32)      # per-view
        return (
            FThetaProjection(
                t_camera_ego=_t("t_camera_ego"),   # [1,V,4,4]
                fw_poly=fw_poly,
                cx=_t("cx"), cy=_t("cy"),          # [1,V]
                image_wh=_t("image_wh"),           # [1,V,2] native (W,H)
                max_theta=max_theta,
            ),
            "ftheta",
        )
    raise ValueError(f"Unknown projection type in spec: {kind!r}")


def _split_bucket(key: str, buckets: int = 10) -> int:
    """Deterministic bucket in [0, buckets) from a stable string.

    Uses a fixed hash (blake2b) — NOT Python's ``hash()``, which is salted per
    process, so train and eval workers (and reruns) would disagree on the split.
    Reproducible across the train task and the (separate) eval task.
    """
    from data_processing.dataset_snapshot import split_bucket
    return split_bucket(key, buckets)


def _split_group_of(sample) -> str:
    """The train/val SPLIT key for a raw shard sample (#121 §3.1).

    Hash the ``split_group_uid`` from the sample's ``meta.json`` (episode/clip
    granularity) — NOT the per-frame ``__key__`` — so all frames of one episode
    fall in the SAME bucket and never straddle train/val (adjacent frames are
    strongly correlated → a per-frame split leaks). Falls back to ``__key__`` for
    legacy shards whose meta.json predates split_group_uid.
    """
    import json
    meta = sample.get("meta.json")
    if meta is not None:
        try:
            g = json.loads(meta.decode() if isinstance(meta, (bytes, bytearray)) else meta)
            grp = g.get("split_group_uid")
            if grp:
                return grp
        except Exception:
            pass
    return sample.get("__key__", "")


def _split_keep(split: str, val_fraction: float):
    """Return a predicate ``sample -> bool`` selecting the requested split.

    ``split="all"`` (default) keeps everything (backward-compatible, single-set
    behaviour). ``"train"`` / ``"val"`` partition by a stable hash of the sample's
    ``split_group_uid`` (episode/clip) into disjoint sets: ``val`` is the first
    ``round(val_fraction*10)`` of 10 buckets, ``train`` is the rest. Splitting by
    GROUP (not per-frame) keeps a whole episode/clip on one side, so eval-on-``val``
    measures generalization to UNSEEN episodes, not memorization of neighbours.
    """
    if split == "all" or val_fraction <= 0.0:
        return lambda sample: True
    buckets = 10
    val_buckets = max(1, min(buckets - 1, round(val_fraction * buckets)))

    def keep(sample):
        b = _split_bucket(_split_group_of(sample), buckets)
        in_val = b < val_buckets
        return in_val if split == "val" else (not in_val)

    return keep


def make_pre_extracted_loader(
    shard_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    split: str = "all",
    val_fraction: float = 0.0,
    shuffle: int = 1000,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    shard_files: Sequence[str | Path] | None = None,
) -> wds.WebLoader:
    """Create a WebDataset DataLoader reading from local EBS shard cache.

    Args:
        shard_dir: Path to directory containing .tar shard files.
        batch_size: Batch size.
        num_workers: DataLoader workers. >0 decodes JPEGs in parallel worker
            processes (the per-sample WM window is ~55 decodes; at num_workers=0
            this is fully serial and the GPU stalls — #121 P0). Workers are
            sharded over the .tar files via ``split_by_worker``, so effective
            parallelism is capped by shard count — pack more, smaller shards to use
            more workers.
        split: ``"all"`` (default, every sample), ``"train"``, or ``"val"``. With
            ``val_fraction`` > 0, ``train``/``val`` are a disjoint per-sample hash
            split (see ``_split_keep``) so eval-on-``val`` measures generalization
            rather than training-set memorization.
        val_fraction: fraction of samples held out for ``val`` (0 disables the
            split → ``"all"`` behaviour regardless of ``split``).
        shuffle: Shuffle buffer size (0 to disable).
        pin_memory: pin host buffers for faster H2D copy (set True on GPU).
        prefetch_factor: batches prefetched per worker (only used when
            num_workers>0); overlaps decode with the GPU step.
        shard_files: optional explicit subset of tar files. Overlay precompute
            uses one file at a time so each output body is canonical per shard.

    The returned loader carries two extra attributes describing the dataset's
    geometry (a rig constant, so it lives on the loader, not per batch):
      - ``.projection``: a CameraProjectionModel operator, or None (pseudo).
      - ``.geometry_type``: "pinhole" / "rectified_pinhole" / "ftheta" / "pseudo".
    Pass these to the model's forward alongside each batch.
    """
    tarfiles = (
        sorted(Path(path) for path in shard_files)
        if shard_files is not None
        else sorted(Path(shard_dir).glob("*.tar"))
    )
    if not tarfiles:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")
    shard_root = Path(shard_dir).resolve()
    for path in tarfiles:
        resolved = path.resolve()
        if not resolved.is_file() or resolved.parent != shard_root:
            raise ValueError(
                f"shard file must be a direct .tar child of {shard_root}: {path}"
            )
        if resolved.suffix != ".tar":
            raise ValueError(f"shard file must use .tar suffix: {path}")

    urls = [str(p) for p in tarfiles]

    # CRITICAL (webdataset 1.0.2): WebDataset has BOTH `nodesplitter` and
    # `workersplitter`, and `workersplitter` DEFAULTS to split_by_worker. Passing
    # nodesplitter=split_by_worker applies the worker split TWICE, so with
    # num_workers=N each worker sees only 1/N of the shards → the loader silently
    # drops (N-1)/N of the data (verified: 48 samples → 24 at nw=2, 12 at nw=4).
    # Use single_node_only for the NODE slot (correct until multi-node DDP, which
    # would set split_by_node here) and let the default workersplitter do the
    # per-worker shard split exactly once.
    dataset = wds.WebDataset(urls, shardshuffle=False, empty_check=False,
                             nodesplitter=wds.single_node_only)
    # Split BEFORE decode (cheap: filters on __key__ only, skips image decode for
    # dropped samples). Keeps train/val disjoint at the sample level.
    keep = _split_keep(split, val_fraction)
    if split != "all" and val_fraction > 0.0:
        dataset = dataset.select(keep)
    if shuffle > 0:
        dataset = dataset.shuffle(shuffle)
    # Frame-pool accessor for deduped WM windows (#121 §3.4d): a sibling pool/ dir
    # next to the .tar shards, NOT part of `urls`, so split_by_worker never shards
    # it away — every worker reaches any frame_id by path. Path-based + lazily read,
    # so it pickles cleanly to spawn workers (no open handle crosses the boundary).
    pool = _make_pool_accessor(shard_dir)
    # functools.partial (NOT a lambda) so the map fn pickles to spawn workers.
    dataset = dataset.map(functools.partial(_decode_sample, pool=pool))

    # split_by_worker shards the .tar list across workers, so more workers than
    # shards is wasted; cap accordingly. Partition-scoped loaders are retired as
    # soon as that partition is exhausted, so workers MUST NOT persist beyond the
    # iterator lifetime. prefetch_factor overlaps decode with the GPU step.
    eff_workers = min(num_workers, len(tarfiles)) if num_workers > 0 else 0
    loader_kwargs: dict = {"batch_size": batch_size, "num_workers": eff_workers}
    if eff_workers > 0:
        loader_kwargs.update(
            persistent_workers=False,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )
    loader = wds.WebLoader(dataset, **loader_kwargs)

    # Per-dataset geometry, reconstructed once from the manifest.
    projection, geometry_type = load_projection_from_manifest(shard_dir)
    loader.projection = projection
    loader.geometry_type = geometry_type
    return loader


@dataclass
class _ActiveLoader:
    loader: object
    iterator: object
    owned: bool
    closed: bool = False

    def close(self):
        """Release an active child iterator and its owned loader exactly once."""
        if self.closed:
            return
        self.closed = True
        iterator_close = getattr(self.iterator, "close", None)
        try:
            if iterator_close is not None:
                iterator_close()
        finally:
            if self.owned:
                loader_close = getattr(self.loader, "close", None)
                if loader_close is not None:
                    loader_close()


class MergedDatasetLoader:
    """Bounded round-robin over multiple single-dataset loaders.

    Different datasets have different camera counts (L2D 6, NVIDIA 7) and
    geometries (pseudo vs f-theta), which cannot be stacked into one batch. So
    every batch remains same-dataset (uniform num_views/geometry) and carries that
    dataset's projection.

    Only ``max_active_loaders`` child iterators exist at once. Within that window
    batches retain the original round-robin ordering; when a child is exhausted
    it is closed before the next pending child is opened. Loader factories are
    invoked lazily and recreated per epoch, which bounds worker, prefetch, shuffle,
    and pin-memory state even when the input contains hundreds of partitions.

    Each yielded item is ``(batch, projection, geometry_type)`` so the training
    loop applies the right geometry to each (same-dataset) batch.
    """

    def __init__(self, loaders=None, *, loader_factories=None, max_active_loaders: int = 4):
        if loaders is not None and loader_factories is not None:
            raise ValueError("pass loaders or loader_factories, not both")
        if loaders is None and loader_factories is None:
            raise ValueError("MergedDatasetLoader needs at least one loader.")
        if max_active_loaders <= 0:
            raise ValueError("max_active_loaders must be positive")

        if loader_factories is not None:
            sources = list(loader_factories)
            owned = True
            self.loaders = []
        else:
            sources = list(loaders)
            owned = False
            self.loaders = sources
        if not sources:
            raise ValueError("MergedDatasetLoader needs at least one loader.")
        self._sources = [(source, owned) for source in sources]
        self.max_active_loaders = min(max_active_loaders, len(sources))

    @staticmethod
    def _open(source, owned: bool) -> _ActiveLoader:
        loader = source() if owned else source
        try:
            iterator = iter(loader)
        except BaseException:
            if owned:
                loader_close = getattr(loader, "close", None)
                if loader_close is not None:
                    loader_close()
            raise
        return _ActiveLoader(loader=loader, iterator=iterator, owned=owned)

    def __iter__(self):
        pending = iter(self._sources)
        active: deque[_ActiveLoader] = deque()

        def fill_active():
            while len(active) < self.max_active_loaders:
                try:
                    source, owned = next(pending)
                except StopIteration:
                    return
                active.append(self._open(source, owned))

        try:
            fill_active()
            while active:
                child = active.popleft()
                try:
                    batch = next(child.iterator)
                except StopIteration:
                    child.close()
                    fill_active()
                    continue
                except BaseException:
                    child.close()
                    raise

                # Requeue before yielding so generator.close() also reaches this
                # child when the consumer stops after the current batch.
                active.append(child)
                yield (
                    batch,
                    getattr(child.loader, "projection", None),
                    getattr(child.loader, "geometry_type", "pseudo"),
                )
        finally:
            close_error = None
            while active:
                try:
                    active.popleft().close()
                except BaseException as error:
                    if close_error is None:
                        close_error = error
            if close_error is not None:
                raise close_error


def make_multi_dataset_loader(
    shard_dirs,
    batch_size: int = 8,
    num_workers: int = 4,
    split: str = "all",
    val_fraction: float = 0.0,
    shuffle: int = 1000,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    max_active_loaders: int | None = None,
) -> MergedDatasetLoader:
    """Build a :class:`MergedDatasetLoader` over several shard directories.

    Each directory is one dataset (its own manifest + geometry). Datasets are
    merged through a bounded active window (see MergedDatasetLoader). A single
    directory degrades to a one-loader merge (identical to the single dataset
    path, but yielding the ``(batch, projection, geometry_type)`` tuple).

    ``split`` / ``val_fraction`` select a disjoint per-sample train/val split
    applied per dataset (see make_pre_extracted_loader). ``num_workers`` is a
    GLOBAL worker budget: each active partition gets one worker, and no more than
    four partition loaders are active. Evaluation can use
    ``max_active_loaders=1`` together with a small
    ``prefetch_factor`` to bound its larger batches.
    """
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if max_active_loaders is not None and max_active_loaders <= 0:
        raise ValueError("max_active_loaders must be positive")

    shard_dirs = list(shard_dirs)
    child_workers = 1 if num_workers > 0 else 0
    default_active = min(4, num_workers) if num_workers > 0 else 1
    active_limit = default_active if max_active_loaders is None else max_active_loaders
    if num_workers > 0:
        active_limit = min(active_limit, num_workers, 4)
    else:
        active_limit = 1

    factories = [
        functools.partial(
            make_pre_extracted_loader,
            d,
            batch_size=batch_size,
            num_workers=child_workers,
            split=split,
            val_fraction=val_fraction,
            shuffle=shuffle,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
        for d in shard_dirs
    ]
    merged = MergedDatasetLoader(
        loader_factories=factories,
        max_active_loaders=active_limit,
    )
    merged.num_workers = num_workers
    return merged
