"""Flyte wiring tests for the KITScenes scene fan-out."""

from __future__ import annotations

import functools

import pytest

pytest.importorskip("flytekit")

from flytekit import map_task

from Platform.pipelines import workflows
from data_parsing.kit_scenes.source import InventoryResolution, SceneArchive


def test_inventory_preflight_emits_one_scene_per_partition(monkeypatch):
    scene_ids = ("scene-a", "scene-c")
    inventory = InventoryResolution(
        split="train",
        expected_scene_ids=("scene-a", "scene-b", "scene-c"),
        selected_scene_ids=scene_ids,
        missing_scene_ids=("scene-b",),
        total_size_bytes=20,
        source_revision=workflows.KITSCENES_SOURCE_REVISION,
    )
    archives = {
        scene_id: SceneArchive(
            scene_id=scene_id,
            split="train",
            filename=f"data/train/{scene_id}.tar",
            sha256="a" * 64,
            size_bytes=10,
        )
        for scene_id in scene_ids
    }
    monkeypatch.setattr(
        "data_parsing.kit_scenes.source.fetch_archive_manifest",
        lambda *args, **kwargs: archives,
    )
    monkeypatch.setattr(
        "data_parsing.kit_scenes.source.resolve_inventory",
        lambda *args, **kwargs: inventory,
    )

    partitions = workflows.plan_fanout_partitions.task_function(
        dataset=workflows.Dataset.KITSCENES,
        source_revision=workflows.KITSCENES_SOURCE_REVISION,
        episodes=0,
        start_ep=-1,
        end_ep=-1,
        partition_size=1,
        max_partitions=600,
        max_missing_scenes=1,
        split="train",
    )

    assert partitions == [["scene-a"], ["scene-c"]]


def test_ingest_map_binds_scalars_and_maps_only_group_ids():
    mapped = map_task(
        functools.partial(
            workflows.data_ingest,
            dataset=workflows.Dataset.KITSCENES,
            source_revision=workflows.KITSCENES_SOURCE_REVISION,
            episodes=0,
        ),
        concurrency=60,
    )

    assert mapped.bound_inputs == {"dataset", "source_revision", "episodes"}
    assert mapped.concurrency == 60
    assert set(mapped.python_interface.inputs) == {
        "dataset",
        "source_revision",
        "episodes",
        "group_ids",
    }
