"""End-to-end smoke of both stages through Ray Train, with checkpoint validation, on one GPU.

Tiny synthetic packs (a planner pack with a held-out Pittsburgh group and a
calibration-share group, and a BEV-only pack) go through the real staging,
loaders, Ray Train loop, S3 checkpoints, stage 1 to stage 2 hand-over, and the
checkpoint validation watcher. ``AUTOE2E_SMOKE_STORAGE`` names the run storage
(an S3 prefix or a local directory); the test is skipped without it and
without a GPU.
"""

import os
import uuid

import pytest
import torch

from tests.test_autobev_data import _bev_members, _meta, _planner_members, _write_pack

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not os.environ.get("AUTOE2E_SMOKE_STORAGE"), reason="needs AUTOE2E_SMOKE_STORAGE"),
]


def _group_in_split(validation: bool) -> str:
    """A split group that lands in the calibration share (or in training)."""
    from data_parsing.pre_extracted import SPLIT_BUCKETS, _split_bucket
    from training.autobev_data import CALIBRATION_FRACTION

    limit = round(CALIBRATION_FRACTION * SPLIT_BUCKETS)
    for index in range(10_000):
        group = f"nuplan-log-{index}"
        if (_split_bucket(group) < limit) == validation:
            return group
    raise AssertionError("no group found")


def test_two_stage_training_and_validation(tmp_path):
    import ray

    from evaluation.checkpoint_validation import list_checkpoints, watch
    from training.autobev_data import autobev_validation_batches
    from training.trainer import TrainRun, run_training

    train_group, calibration_group = _group_in_split(False), _group_in_split(True)
    samples = {}
    for index in range(6):
        samples[f"train-{index}"] = {**_planner_members(), **_meta(f"train-{index}", train_group)}
    samples["cal-0"] = {**_planner_members(), **_meta("cal-0", calibration_group)}
    samples["pit-0"] = {**_planner_members(), **_meta("pit-0", "heldout-nuplan-log-pittsburgh")}
    planner = str(_write_pack(tmp_path / "nuplan", samples, sample_kind="planner", dataset="nuplan"))
    bev_only = str(_write_pack(
        tmp_path / "nuscenes", {f"nusc-{i}": {**_bev_members(), **_meta(f"nusc-{i}", f"scene-{i}")} for i in range(4)},
        sample_kind="autobev_only", dataset="nuscenes"))
    storage = f"{os.environ['AUTOE2E_SMOKE_STORAGE'].rstrip('/')}/{uuid.uuid4().hex[:8]}"
    ray.init(num_cpus=6, num_gpus=1, ignore_reinit_error=True)
    try:
        stage1 = TrainRun(stage=1, run_name="stage1", storage_uri=storage, primary_shards=[planner],
                          mix_shards=[bev_only], steps=4, checkpoint_interval_steps=2, world_size=1)
        run_training(stage1)
        stage1_checkpoints = list_checkpoints(f"{storage}/stage1")
        assert len(stage1_checkpoints) == 2
        stage2 = TrainRun(stage=2, run_name="stage2", storage_uri=storage, primary_shards=[planner],
                          mix_shards=[bev_only], steps=4, checkpoint_interval_steps=2, world_size=1,
                          parent_checkpoint_uri=stage1_checkpoints[-1])
        run_training(stage2)
    finally:
        ray.shutdown()
    results = watch(
        stage2,
        calibration=lambda: autobev_validation_batches([planner], split="calibration", limit=1),
        validation=lambda: autobev_validation_batches([planner], split="validation", limit=1),
    )
    assert [result["step"] for result in results] == [2, 4]
    assert all("ade/6.4s" in result["metrics"] for result in results)
    assert all(result["sample_uids"] == ["pit-0"] for result in results)
