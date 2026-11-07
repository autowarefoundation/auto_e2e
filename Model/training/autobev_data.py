"""Stage published AutoBEV partitions onto a rank's disk and stream training and validation batches.

Each dataset's partitions (S3 prefixes or local directories) are merged into
one local directory per dataset and rank: the rank takes every ``world``-th
shard of the dataset and a merged ``manifest.json`` lists them, so a loader
reads one directory per dataset with as many workers as it has shards.

Splits: ``heldout-`` split groups (nuPlan Pittsburgh) are always validation;
``CALIBRATION_FRACTION`` of the other groups form the calibration split that
sets the BEV thresholds and never trains; the rest trains.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from training import storage

CALIBRATION_FRACTION: Final = 0.02
LOADER_SHUFFLE: Final = 256
IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)
HELDOUT_PREFIX: Final = "heldout-"
VALIDATION_CACHE: Final = Path(tempfile.gettempdir()) / "autobev-validation-shards"


def _fetch(uri: str, name: str, target: Path) -> None:
    storage.download(f"{uri.rstrip('/')}/{name}", target)


def _manifest(uri: str) -> dict[str, Any]:
    return json.loads(storage.read_bytes(f"{uri.rstrip('/')}/manifest.json"))


def stage_rank_shards(partition_uris: Sequence[str], *, rank: int, world: int, root: Path) -> list[str]:
    """Copy this rank's shards of every dataset into one directory per dataset; returns those directories."""
    by_dataset: dict[str, list[tuple[str, dict]]] = {}
    for uri in partition_uris:
        manifest = _manifest(uri)
        by_dataset.setdefault(str(manifest.get("dataset", "nuplan")), []).append((uri, manifest))
    directories = []
    for dataset, partitions in sorted(by_dataset.items()):
        shards = [(uri, name, manifest) for uri, manifest in partitions for name in manifest["shard_names"]]
        mine = shards[rank::world]
        if not mine:
            continue
        target = root / dataset
        target.mkdir(parents=True, exist_ok=True)
        names, counts = [], {}
        for index, (uri, name, manifest) in enumerate(mine):
            local = f"{index:06d}-{name}"
            _fetch(uri, name, target / local)
            names.append(local)
            counts[local] = int(manifest["shard_sample_counts"][name])
        first = partitions[0][1]
        merged = {key: value for key, value in first.items()
                  if key not in {"shard_names", "shard_sample_counts", "shard_sha256", "total_samples", "shards"}}
        merged.update(shard_names=names, shard_sample_counts=counts, total_samples=sum(counts.values()))
        (target / "manifest.json").write_text(json.dumps(merged))
        directories.append(str(target))
    return directories


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a batch to ``device`` and normalize its uint8 images there."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            out[key] = [item.to(device) for item in value]
        else:
            out[key] = value
    for key in ("visual_tiles", "camera_history_tiles"):
        out[key] = (out[key].float() / 255.0 - mean) / std
    return out


def _forever(make_loader) -> Iterator[dict[str, Any]]:
    epoch = 0
    while True:
        for batch, _, _ in make_loader(epoch):
            yield batch
        epoch += 1


def autobev_training_stream(
    *, stage: int, primary: Sequence[str], mix: Sequence[str], early: Sequence[str], early_until_step: int,
    batch_size: int, seed: int, rank: int, world: int, start_step: int, workers: int, root: Path,
) -> Iterator[dict[str, Any]]:
    """Endless training batches of ``stage`` for one rank, reshuffled per epoch.

    Stage 1 mixes every dataset by sample count, with ``early`` (SimBEV) only
    before ``early_until_step``. Stage 2 draws the BEV-only share from ``mix``.
    A resumed run reseeds by its start step instead of replaying the order.
    """
    from data_parsing.pre_extracted import make_autobev_bev_share_loader, make_autobev_sample_count_loader

    primary_dirs = stage_rank_shards(primary, rank=rank, world=world, root=root / "primary")
    mix_dirs = stage_rank_shards(mix, rank=rank, world=world, root=root / "mix")
    early_dirs = stage_rank_shards(early, rank=rank, world=world, root=root / "early")
    common = {"batch_size": batch_size, "num_workers": workers, "split": "train",
              "val_fraction": CALIBRATION_FRACTION, "shuffle": LOADER_SHUFFLE, "drop_last": True,
              "pin_memory": True}
    seed = seed * 1_000_003 + rank * 1009 + start_step
    if stage == 2:
        yield from _forever(lambda epoch: make_autobev_bev_share_loader(
            primary_dirs, mix_dirs, shuffle_seed=seed + epoch, **common))
        return
    # SimBEV leaves the stage 1 mix halfway through.
    with_early = _forever(lambda epoch: make_autobev_sample_count_loader(
        primary_dirs + mix_dirs + early_dirs, shuffle_seed=seed + epoch, **common))
    yield from itertools.islice(with_early, max(0, early_until_step - start_step))
    yield from _forever(lambda epoch: make_autobev_sample_count_loader(
        primary_dirs + mix_dirs, shuffle_seed=seed + 7919 + epoch, **common))


def autobev_validation_batches(partition_uris: Sequence[str], *, split: str, limit: int) -> Iterator[dict[str, Any]]:
    """Single-sample batches of the ``calibration`` or ``validation`` split, read shard by shard.

    Validation is the held-out Pittsburgh groups; calibration the
    ``CALIBRATION_FRACTION`` of the other groups. ``limit`` 0 reads them all.
    """
    from data_parsing.pre_extracted import make_pre_extracted_loader

    held_out = split == "validation"
    count = 0
    for uri in partition_uris:
        manifest = _manifest(uri)
        for name in manifest["shard_names"]:
            # Kept between passes: checkpoint validation reads the same shards every time.
            local = VALIDATION_CACHE / hashlib.sha256(f"{uri}/{name}".encode()).hexdigest()[:24]
            if not (local / "manifest.json").is_file():
                local.mkdir(parents=True, exist_ok=True)
                _fetch(uri, name, local / name)
                single = {**manifest, "shard_names": [name],
                          "shard_sample_counts": {name: manifest["shard_sample_counts"][name]}}
                (local / "manifest.json").write_text(json.dumps(single))
            for batch in make_pre_extracted_loader(str(local), batch_size=1, num_workers=2, split="val",
                                                   val_fraction=CALIBRATION_FRACTION, shuffle=0):
                if batch["split_group_uid"][0].startswith(HELDOUT_PREFIX) != held_out:
                    continue
                yield batch
                count += 1
                if count == limit:
                    return
