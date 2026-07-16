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

import io
import json
import re
from pathlib import Path

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


def _decode_sample(sample: dict) -> dict:
    """Decode a WebDataset sample into training tensors (geometry-free).

    Calibration is a per-dataset rig constant, not per-sample, so it is NOT
    decoded here — it is reconstructed once by ``make_pre_extracted_loader`` and
    exposed on the loader as ``.projection`` / ``.geometry_type``.
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
        "visual_tiles": torch.stack(frames),
        "map_input": map_input,
        "egomotion_history": ego_history,
        "visual_history": torch.zeros(_VISUAL_HISTORY_DIM),
        "trajectory_target": ego_future,
    }

    # Optional World-Model windows (#13): hist_<t>_cam_<v>.jpg / fut_<f>_cam_<v>.jpg
    # decode to history_frames [T, V, 3, H, W] and future_frames [F, V, 3, H, W]
    # (oldest→newest). Present only on shards packed with world_model=True; when
    # absent, training runs without the JEPA loss.
    hist = _decode_window(sample, _HIST_KEY_RE)
    if hist is not None:
        out["history_frames"] = hist
    fut = _decode_window(sample, _FUT_KEY_RE)
    if fut is not None:
        out["future_frames"] = fut

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


def _decode_window(sample: dict, key_re) -> "torch.Tensor | None":
    """Decode a World-Model window into ``[steps, V, 3, H, W]`` (oldest→newest).

    Matches ``key_re`` (hist_/fut_) against the sample keys, groups by step and
    view index, and stacks. Returns None if the window is absent (#13).
    """
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
        frame_steps.append(torch.stack(view_frames))  # [V, 3, H, W]
    return torch.stack(frame_steps)                    # [steps, V, 3, H, W]


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
    """Deterministic bucket in [0, buckets) from a sample's stable ``__key__``.

    Uses a fixed hash (blake2b) — NOT Python's ``hash()``, which is salted per
    process, so train and eval workers (and reruns) would disagree on the split.
    A per-sample hash split keeps train/val disjoint at the SAMPLE level and is
    reproducible across the train task and the (separate) eval task.
    """
    import hashlib
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % buckets


def _split_keep(split: str, val_fraction: float):
    """Return a predicate ``sample -> bool`` selecting the requested split.

    ``split="all"`` (default) keeps everything (backward-compatible, single-set
    behaviour). ``"train"`` / ``"val"`` partition by a stable per-sample hash of
    ``__key__`` into disjoint sets: ``val`` is the first ``round(val_fraction*10)``
    of 10 buckets, ``train`` is the rest. So train and val NEVER share a sample,
    and eval-on-val measures generalization, not memorization.
    """
    if split == "all" or val_fraction <= 0.0:
        return lambda sample: True
    buckets = 10
    val_buckets = max(1, min(buckets - 1, round(val_fraction * buckets)))

    def keep(sample):
        b = _split_bucket(sample.get("__key__", ""), buckets)
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
) -> wds.WebLoader:
    """Create a WebDataset DataLoader reading from local EBS shard cache.

    Args:
        shard_dir: Path to directory containing .tar shard files.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        split: ``"all"`` (default, every sample), ``"train"``, or ``"val"``. With
            ``val_fraction`` > 0, ``train``/``val`` are a disjoint per-sample hash
            split (see ``_split_keep``) so eval-on-``val`` measures generalization
            rather than training-set memorization.
        val_fraction: fraction of samples held out for ``val`` (0 disables the
            split → ``"all"`` behaviour regardless of ``split``).
        shuffle: Shuffle buffer size (0 to disable).

    The returned loader carries two extra attributes describing the dataset's
    geometry (a rig constant, so it lives on the loader, not per batch):
      - ``.projection``: a CameraProjectionModel operator, or None (pseudo).
      - ``.geometry_type``: "pinhole" / "rectified_pinhole" / "ftheta" / "pseudo".
    Pass these to the model's forward alongside each batch.
    """
    tarfiles = sorted(Path(shard_dir).glob("*.tar"))
    if not tarfiles:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")

    urls = [str(p) for p in tarfiles]

    dataset = wds.WebDataset(urls, shardshuffle=False, empty_check=False, nodesplitter=wds.split_by_worker)
    # Split BEFORE decode (cheap: filters on __key__ only, skips image decode for
    # dropped samples). Keeps train/val disjoint at the sample level.
    keep = _split_keep(split, val_fraction)
    if split != "all" and val_fraction > 0.0:
        dataset = dataset.select(keep)
    if shuffle > 0:
        dataset = dataset.shuffle(shuffle)
    dataset = dataset.map(_decode_sample)

    loader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=min(num_workers, len(tarfiles)))

    # Per-dataset geometry, reconstructed once from the manifest.
    projection, geometry_type = load_projection_from_manifest(shard_dir)
    loader.projection = projection
    loader.geometry_type = geometry_type
    return loader


class MergedDatasetLoader:
    """Round-robin over multiple single-dataset loaders (merged training).

    Different datasets have different camera counts (L2D 6, NVIDIA 7) and
    geometries (pseudo vs f-theta), which cannot be stacked into one batch. So
    each dataset keeps its own WebDataset loader and we interleave BATCHES: every
    batch is same-dataset (uniform num_views/geometry) and carries that dataset's
    projection, while an epoch mixes all datasets. This is the merge point — one
    ready-to-train stream over many datasets, per-sample/per-dataset geometry
    preserved (self-describing calib.json in the shards, manifest per dir).

    Each yielded item is ``(batch, projection, geometry_type)`` so the training
    loop applies the right geometry to each (same-dataset) batch.
    """

    def __init__(self, loaders):
        if not loaders:
            raise ValueError("MergedDatasetLoader needs at least one loader.")
        self.loaders = list(loaders)

    def __iter__(self):
        iterators = [iter(dl) for dl in self.loaders]
        active = list(range(len(iterators)))
        # Round-robin: pull one batch from each live loader in turn until all
        # are exhausted, so datasets are interleaved rather than concatenated.
        while active:
            still: list[int] = []
            for i in active:
                try:
                    batch = next(iterators[i])
                except StopIteration:
                    continue
                dl = self.loaders[i]
                yield batch, getattr(dl, "projection", None), getattr(dl, "geometry_type", "pseudo")
                still.append(i)
            active = still


def make_multi_dataset_loader(
    shard_dirs,
    batch_size: int = 8,
    num_workers: int = 4,
    split: str = "all",
    val_fraction: float = 0.0,
    shuffle: int = 1000,
) -> MergedDatasetLoader:
    """Build a :class:`MergedDatasetLoader` over several shard directories.

    Each directory is one dataset (its own manifest + geometry). Datasets are
    merged by interleaving same-dataset batches (see MergedDatasetLoader). A
    single directory degrades to a one-loader merge (identical to the single
    dataset path, but yielding the ``(batch, projection, geometry_type)`` tuple).

    ``split`` / ``val_fraction`` select a disjoint per-sample train/val split
    applied per dataset (see make_pre_extracted_loader).
    """
    loaders = [
        make_pre_extracted_loader(d, batch_size=batch_size, num_workers=num_workers,
                                  split=split, val_fraction=val_fraction, shuffle=shuffle)
        for d in shard_dirs
    ]
    return MergedDatasetLoader(loaders)
