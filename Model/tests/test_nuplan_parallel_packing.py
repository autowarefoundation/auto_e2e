"""Deterministic parallel packing contracts for full nuPlan snapshots."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS
from data_parsing.nuplan import packing as nuplan_packing
from data_parsing.nuplan.packing import (
    NUPLAN_CAMERA_CHANNELS,
    NUPLAN_CAMERA_SLOTS,
    NUPLAN_CAMERA_SYNC_TOLERANCE_US,
    NUPLAN_CAMERA_VISIBILITY_HEIGHTS_M,
    NUPLAN_HISTORY_CAMERA_SPREAD_MAX_US,
    NUPLAN_HISTORY_FALLBACK_MAX_OFFSET_US,
    NUPLAN_PACK_MANIFEST_VERSION,
    _NUPLAN_MANIFEST_INVARIANT_KEYS,
    _NuPlanNoScenariosError,
    _NuPlanPackPartition,
    _NuPlanPackWorkerConfig,
    _initialize_nuplan_pack_worker,
    _merge_nuplan_pack_partitions,
    _nuplan_split_group_uid,
    _nuplan_db_scenario_count,
    _pack_nuplan_partition,
    _partition_weighted_nuplan_db_files,
    _sample_camera_timing_metrics,
    pack_nuplan_local_dataset,
    pack_nuplan_reactive_scenarios,
)
from navigation.contracts import canonical_json_bytes
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_FRAME_OFFSETS,
    REACTIVE_BEVFORMER_HISTORY_FRAMES,
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_INDEX,
)


def _scenario(
    sample_uid: str,
    split_group_uid: str,
    *,
    rejected: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        log_name=split_group_uid,
        rejected=rejected,
        sample_uid=sample_uid,
        split_group_uid=split_group_uid,
        token=f"token-{sample_uid}",
    )


def _sample_builder(
    scenario: SimpleNamespace,
    **_kwargs: object,
) -> tuple[str, str, dict[str, bytes]]:
    if scenario.rejected:
        raise ValueError("synthetic rejection")
    return (
        scenario.sample_uid,
        scenario.split_group_uid,
        {
            "calib.json": json.dumps({
                "cameras": [
                    {
                        "channel": channel,
                        "image_time_offset_us": (
                            NUPLAN_CAMERA_SYNC_TOLERANCE_US
                            if index == 0
                            else 0
                        ),
                    }
                    for index, channel in enumerate(NUPLAN_CAMERA_CHANNELS)
                ],
                "reference_lidar_timestamp_us": 4_000_000,
                "temporal_camera_timestamps_us": [
                    [
                        4_000_000
                        + offset * REACTIVE_BEVFORMER_FRAME_INTERVAL_US
                        for _ in NUPLAN_CAMERA_CHANNELS
                    ]
                    for offset in REACTIVE_BEVFORMER_FRAME_OFFSETS[:-1]
                ],
                "temporal_frame_interval_us": (
                    REACTIVE_BEVFORMER_FRAME_INTERVAL_US
                ),
                "temporal_frame_offsets": list(
                    REACTIVE_BEVFORMER_FRAME_OFFSETS
                ),
            }).encode("ascii"),
            "meta.json": b"{}",
        },
    )


def _worker_manifest(
    worker_directory: Path,
    scenarios: list[SimpleNamespace],
) -> dict[str, object]:
    return pack_nuplan_reactive_scenarios(
        scenarios,
        worker_directory,
        source_revision="nuplan-v1.1-mini",
        map_version="nuplan-maps-v1.0",
        samples_per_shard=2,
        max_rejection_fraction=1.0,
        require_accepted=False,
        sample_builder=_sample_builder,
    )


def test_nuplan_camera_rig_uses_canonical_six_slots_without_lateral_views():
    assert NUPLAN_CAMERA_CHANNELS == (
        "CAM_F0",
        "CAM_L0",
        "CAM_R0",
        "CAM_L2",
        "CAM_B0",
        "CAM_R2",
    )
    assert NUPLAN_CAMERA_SLOTS == CANONICAL_SIX_CAMERA_SLOTS
    assert {"CAM_L1", "CAM_R1"}.isdisjoint(NUPLAN_CAMERA_CHANNELS)


def test_nuplan_history_timing_metrics_reject_degenerate_frames():
    _, _, members = _sample_builder(
        _scenario("timing", "group-timing")
    )

    assert _sample_camera_timing_metrics(members) == (
        NUPLAN_CAMERA_SYNC_TOLERANCE_US,
        0,
        REACTIVE_BEVFORMER_HISTORY_FRAMES,
    )

    calibration = json.loads(members["calib.json"])
    calibration["temporal_camera_timestamps_us"][-1] = list(
        calibration["temporal_camera_timestamps_us"][-2]
    )
    repeated_members = {
        **members,
        "calib.json": json.dumps(calibration).encode("ascii"),
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        _sample_camera_timing_metrics(repeated_members)

    calibration = json.loads(members["calib.json"])
    calibration["temporal_camera_timestamps_us"][0][0] += (
        NUPLAN_HISTORY_FALLBACK_MAX_OFFSET_US + 1
    )
    stale_members = {
        **members,
        "calib.json": json.dumps(calibration).encode("ascii"),
    }
    with pytest.raises(ValueError, match="history camera time offset"):
        _sample_camera_timing_metrics(stale_members)

    calibration = json.loads(members["calib.json"])
    calibration["temporal_camera_timestamps_us"][0][0] += (
        NUPLAN_HISTORY_CAMERA_SPREAD_MAX_US + 1
    )
    unsynchronized_members = {
        **members,
        "calib.json": json.dumps(calibration).encode("ascii"),
    }
    with pytest.raises(ValueError, match="camera timestamp spread"):
        _sample_camera_timing_metrics(unsynchronized_members)


def _partition_layout(
    tmp_path: Path,
    weighted: list[tuple[str, int]],
) -> tuple[Path, Path, list[_NuPlanPackPartition]]:
    output = tmp_path / "output"
    partition_root = output / ".partitions"
    output.mkdir()
    partition_root.mkdir()
    partitions = _partition_weighted_nuplan_db_files(
        [(tmp_path / name, weight) for name, weight in weighted],
        len(weighted),
    )
    return output, partition_root, partitions


def test_nuplan_db_partitioning_is_deterministic_and_balanced(
    tmp_path: Path,
):
    weighted = [
        (tmp_path / "a.db", 100),
        (tmp_path / "b.db", 80),
        (tmp_path / "c.db", 60),
        (tmp_path / "d.db", 40),
        (tmp_path / "empty.db", 0),
    ]

    first = _partition_weighted_nuplan_db_files(weighted, 2)
    second = _partition_weighted_nuplan_db_files(
        list(reversed(weighted)),
        2,
    )

    assert first == second
    assert [item.scenario_estimate for item in first] == [140, 140]
    assert [
        [Path(path).name for path in partition.db_files]
        for partition in first
    ] == [
        ["a.db", "d.db", "empty.db"],
        ["b.db", "c.db"],
    ]


def test_nuplan_limited_partitioning_allocates_exact_weighted_quota(
    tmp_path: Path,
):
    weighted = [
        (tmp_path / "a.db", 100),
        (tmp_path / "b.db", 80),
        (tmp_path / "c.db", 60),
        (tmp_path / "d.db", 40),
    ]

    partitions = _partition_weighted_nuplan_db_files(
        weighted,
        worker_count=3,
        scenario_limit=17,
    )

    assert len(partitions) == 3
    assert sum(item.scenario_limit for item in partitions) == 17
    assert all(item.scenario_limit > 0 for item in partitions)
    assert all(
        item.scenario_limit >= item.positive_db_file_count
        for item in partitions
    )
    assert partitions == _partition_weighted_nuplan_db_files(
        list(reversed(weighted)),
        worker_count=3,
        scenario_limit=17,
    )


def test_nuplan_limited_partitioning_caps_quota_at_candidates(
    tmp_path: Path,
):
    partitions = _partition_weighted_nuplan_db_files(
        [
            (tmp_path / "small.db", 1),
            (tmp_path / "large.db", 100),
        ],
        worker_count=2,
        scenario_limit=50,
    )

    assert sum(item.scenario_limit for item in partitions) == 50
    assert all(
        item.scenario_limit <= item.scenario_estimate
        for item in partitions
    )
    small = next(
        item
        for item in partitions
        if item.scenario_estimate == 1
    )
    assert small.scenario_limit == 1


def test_nuplan_limited_partitioning_rejects_unfillable_limit(
    tmp_path: Path,
):
    with pytest.raises(
        ValueError,
        match="limit exceeds available candidates",
    ):
        _partition_weighted_nuplan_db_files(
            [
                (tmp_path / "a.db", 2),
                (tmp_path / "b.db", 3),
            ],
            worker_count=2,
            scenario_limit=6,
        )


def test_nuplan_db_partitioning_rejects_duplicate_paths(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="contains duplicates"):
        _partition_weighted_nuplan_db_files(
            [(tmp_path / "same.db", 2), (tmp_path / "same.db", 1)],
            2,
        )


def test_nuplan_db_scenario_count_matches_builder_filters(tmp_path: Path):
    db_path = tmp_path / "log?special.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE scene (
                token BLOB,
                name TEXT,
                goal_ego_pose_token BLOB
            );
            CREATE TABLE ego_pose (token BLOB);
            CREATE TABLE lidar_pc (
                token BLOB,
                timestamp INTEGER,
                lidar_token BLOB,
                scene_token BLOB,
                ego_pose_token BLOB
            );
            CREATE TABLE image (ego_pose_token BLOB);
            CREATE TABLE scenario_tag (
                lidar_pc_token BLOB,
                type TEXT
            );
            CREATE TABLE lidar (token BLOB, log_token BLOB);
            CREATE TABLE log (token BLOB, map_version TEXT);
            """
        )
        log_token = b"log"
        lidar_token = b"lidar"
        connection.execute(
            "INSERT INTO log VALUES (?, ?)",
            (log_token, "us-nv-las-vegas-strip"),
        )
        connection.execute(
            "INSERT INTO lidar VALUES (?, ?)",
            (lidar_token, log_token),
        )
        scenes = []
        for index in range(6):
            scene_token = f"scene-{index}".encode()
            goal_token = f"goal-{index}".encode()
            scenes.append(scene_token)
            connection.execute(
                "INSERT INTO ego_pose VALUES (?)",
                (goal_token,),
            )
            connection.execute(
                "INSERT INTO scene VALUES (?, ?, ?)",
                (scene_token, f"{index:02d}", goal_token),
            )
        for index in range(3):
            token = f"candidate-{index}".encode()
            pose_token = f"pose-{index}".encode()
            connection.execute(
                "INSERT INTO lidar_pc VALUES (?, ?, ?, ?, ?)",
                (token, index, lidar_token, scenes[2], pose_token),
            )
            connection.executemany(
                "INSERT INTO image VALUES (?)",
                [(pose_token,), (pose_token,)],
            )
            connection.executemany(
                "INSERT INTO scenario_tag VALUES (?, ?)",
                [(token, "a"), (token, "b")],
            )
        connection.execute(
            "INSERT INTO lidar_pc VALUES (?, ?, ?, ?, ?)",
            (
                b"tagless-valid",
                3,
                lidar_token,
                scenes[2],
                b"pose-tagless-valid",
            ),
        )
        connection.execute(
            "INSERT INTO image VALUES (?)",
            (b"pose-tagless-valid",),
        )
        connection.execute(
            "INSERT INTO lidar_pc VALUES (?, ?, ?, ?, ?)",
            (
                b"outside-valid-scenes",
                4,
                lidar_token,
                scenes[1],
                b"pose-outside-valid-scenes",
            ),
        )
        connection.execute(
            "INSERT INTO image VALUES (?)",
            (b"pose-outside-valid-scenes",),
        )
        connection.execute(
            "INSERT INTO scenario_tag VALUES (?, ?)",
            (b"outside-valid-scenes", "a"),
        )
        connection.execute(
            "INSERT INTO lidar_pc VALUES (?, ?, ?, ?, ?)",
            (b"no-image", 4, lidar_token, scenes[2], b"pose-no-image"),
        )
        connection.execute(
            "UPDATE scene SET goal_ego_pose_token = NULL WHERE token = ?",
            (scenes[3],),
        )
        connection.execute(
            "INSERT INTO lidar_pc VALUES (?, ?, ?, ?, ?)",
            (b"no-goal", 5, lidar_token, scenes[3], b"pose-no-goal"),
        )
        connection.execute(
            "INSERT INTO image VALUES (?)",
            (b"pose-no-goal",),
        )
        connection.commit()
    finally:
        connection.close()

    assert _nuplan_db_scenario_count(db_path) == 4


def test_nuplan_partition_merge_preserves_current_manifest(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("a.db", 10), ("b.db", 9)],
    )
    manifests = []
    expected_uids = []
    for partition in partitions:
        sample_uids = [
            f"nuplan-worker-{partition.index}-0",
            f"nuplan-worker-{partition.index}-1",
        ]
        expected_uids.extend(sample_uids)
        manifests.append(_worker_manifest(
            partition_root / f"worker-{partition.index:03d}",
            [
                _scenario(
                    sample_uid,
                    f"group-{partition.index}",
                )
                for sample_uid in sample_uids
            ],
        ))

    merged = _merge_nuplan_pack_partitions(
        output=output,
        partition_root=partition_root,
        partitions=partitions,
        manifests=manifests,
        max_rejection_fraction=0.0,
    )

    assert set(manifests[0]).issubset(merged)
    assert all(
        merged[key] == manifests[0][key]
        for key in _NUPLAN_MANIFEST_INVARIANT_KEYS
    )
    assert merged["schema_version"] == NUPLAN_PACK_MANIFEST_VERSION
    assert merged["image_size"] == REACTIVE_CAMERA_IMAGE_SIZE
    assert merged["camera_visibility_heights_m"] == list(
        NUPLAN_CAMERA_VISIBILITY_HEIGHTS_M
    )
    assert merged["front_camera_index"] == REACTIVE_FRONT_CAMERA_INDEX
    assert merged["max_camera_time_offset_us"] == (
        NUPLAN_CAMERA_SYNC_TOLERANCE_US
    )
    assert merged["max_history_camera_time_offset_us"] == 0
    assert (
        merged["history_camera_spread_max_us"]
        == NUPLAN_HISTORY_CAMERA_SPREAD_MAX_US
    )
    assert (
        merged["distinct_history_frame_count"]
        == REACTIVE_BEVFORMER_HISTORY_FRAMES
    )
    assert (
        merged["front_camera_image_size"]
        == REACTIVE_FRONT_CAMERA_IMAGE_SIZE
    )
    assert merged["packing_workers"] == 2
    assert merged["total_samples"] == 4
    assert merged["bev_segmentation_count"] == 4
    assert merged["bev_statistics_count"] == 4
    assert merged["split_group_count"] == 2
    assert merged["split_group_uids"] == ["group-0", "group-1"]
    assert merged["shard_names"] == [
        "nuplan-000000.tar",
        "nuplan-000001.tar",
    ]
    assert merged["sample_uid_digest"] == hashlib.sha256(
        "\n".join(sorted(expected_uids)).encode("ascii")
    ).hexdigest()
    assert not partition_root.exists()
    shard_hashes = merged["shard_sha256"]
    assert isinstance(shard_hashes, dict)
    for shard_name in merged["shard_names"]:
        shard_path = output / shard_name
        assert shard_hashes[shard_name] == hashlib.sha256(
            shard_path.read_bytes()
        ).hexdigest()


def test_nuplan_partition_merge_skips_filtered_empty_worker(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("empty-after-filter.db", 10), ("accepted.db", 9)],
    )
    empty_partition, accepted_partition = partitions
    (
        partition_root / f"worker-{empty_partition.index:03d}"
    ).mkdir()
    accepted_manifest = _worker_manifest(
        partition_root / f"worker-{accepted_partition.index:03d}",
        [_scenario("nuplan-accepted-0", "group-accepted")],
    )

    empty_prefilter = {
        "log_name": "empty-after-filter",
        "reason": "local lidar asset is missing",
        "scenario_token": "filtered-token",
    }
    merged = _merge_nuplan_pack_partitions(
        output=output,
        partition_root=partition_root,
        partitions=partitions,
        manifests=[
            {
                "empty_partition": True,
                "empty_reason": "all candidates were prefiltered",
                "prefiltered_count": 1,
                "prefiltered_samples": [empty_prefilter],
                "rejected_count": 0,
            },
            accepted_manifest,
        ],
        max_rejection_fraction=0.0,
    )

    assert merged["total_samples"] == 1
    assert merged["bev_statistics_count"] == 1
    assert merged["packing_workers"] == 2
    assert merged["packing_nonempty_workers"] == 1
    assert merged["packing_partitions"] == [
        {
            "accepted_count": 0,
            "db_file_count": 1,
            "empty_reason": "all candidates were prefiltered",
            "is_empty": True,
            "prefiltered_count": 1,
            "rejected_count": 0,
            "scenario_estimate": 10,
            "scenario_limit": 0,
        },
        {
            "accepted_count": 1,
            "db_file_count": 1,
            "is_empty": False,
            "prefiltered_count": 0,
            "rejected_count": 0,
            "scenario_estimate": 9,
            "scenario_limit": 0,
        },
    ]
    assert merged["prefiltered_count"] == 1
    assert merged["prefiltered_samples"] == [empty_prefilter]
    assert not partition_root.exists()


def test_nuplan_partition_merge_accounts_for_all_rejected_worker(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("rejected.db", 10), ("accepted.db", 9)],
    )
    rejected_manifest = _worker_manifest(
        partition_root / "worker-000",
        [_scenario("rejected", "group-rejected", rejected=True)],
    )
    accepted_manifest = _worker_manifest(
        partition_root / "worker-001",
        [_scenario("accepted", "group-accepted")],
    )

    merged = _merge_nuplan_pack_partitions(
        output=output,
        partition_root=partition_root,
        partitions=partitions,
        manifests=[rejected_manifest, accepted_manifest],
        max_rejection_fraction=0.5,
    )

    assert rejected_manifest["total_samples"] == 0
    assert rejected_manifest["rejection_count"] == 1
    assert rejected_manifest["shard_names"] == []
    assert merged["total_samples"] == 1
    assert merged["rejection_count"] == 1
    assert merged["rejection_fraction"] == 0.5
    packing_partitions = cast(
        list[dict[str, object]],
        merged["packing_partitions"],
    )
    assert packing_partitions[0]["rejected_count"] == 1


def test_nuplan_bounded_merge_allows_rejections_after_filling_quota(
    tmp_path: Path,
):
    output, partition_root, unbounded_partitions = _partition_layout(
        tmp_path,
        [("a.db", 3), ("b.db", 2)],
    )
    partitions = [
        dataclasses.replace(partition, scenario_limit=1)
        for partition in unbounded_partitions
    ]
    manifests = [
        _worker_manifest(
            partition_root / f"worker-{partition.index:03d}",
            [
                _scenario(
                    f"rejected-{partition.index}",
                    f"group-{partition.index}",
                    rejected=True,
                ),
                _scenario(
                    f"accepted-{partition.index}",
                    f"group-{partition.index}",
                ),
            ],
        )
        for partition in partitions
    ]

    merged = _merge_nuplan_pack_partitions(
        output=output,
        partition_root=partition_root,
        partitions=partitions,
        manifests=manifests,
        max_rejection_fraction=0.5,
    )

    assert merged["total_samples"] == 2
    assert merged["rejection_count"] == 2
    assert merged["packing_scenario_limit"] == 2


def test_nuplan_bounded_merge_rejects_unfilled_quota(
    tmp_path: Path,
):
    output, partition_root, unbounded_partitions = _partition_layout(
        tmp_path,
        [("a.db", 3), ("b.db", 2)],
    )
    partitions = [
        dataclasses.replace(partition, scenario_limit=1)
        for partition in unbounded_partitions
    ]
    manifest = _worker_manifest(
        partition_root / "worker-000",
        [_scenario("accepted", "group-a")],
    )
    (partition_root / "worker-001").mkdir()

    with pytest.raises(ValueError, match="did not fill its scenario limit"):
        _merge_nuplan_pack_partitions(
            output=output,
            partition_root=partition_root,
            partitions=partitions,
            manifests=[manifest, None],
            max_rejection_fraction=1.0,
        )


def test_nuplan_merge_rejects_missing_log_before_writing_manifest(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("a.db", 3)],
    )
    manifest = _worker_manifest(
        partition_root / "worker-000",
        [_scenario("accepted", "group-a")],
    )

    with pytest.raises(ValueError, match="coverage mismatch"):
        _merge_nuplan_pack_partitions(
            output=output,
            partition_root=partition_root,
            partitions=partitions,
            manifests=[manifest],
            max_rejection_fraction=0.0,
            expected_split_group_uids={"group-a", "group-b"},
            require_full_log_coverage=True,
        )

    assert not (output / "manifest.json").exists()
    assert not list(output.glob("*.tar"))
    assert list(partition_root.rglob("*.tar"))


def test_nuplan_merge_reports_unexpected_source_log(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("a.db", 3)],
    )
    manifest = _worker_manifest(
        partition_root / "worker-000",
        [_scenario("accepted", "group-extra")],
    )

    with pytest.raises(ValueError, match=r"unexpected=\['group-extra'\]"):
        _merge_nuplan_pack_partitions(
            output=output,
            partition_root=partition_root,
            partitions=partitions,
            manifests=[manifest],
            max_rejection_fraction=0.0,
            expected_split_group_uids={"group-a"},
            require_full_log_coverage=False,
        )


def test_nuplan_sensor_prefilter_backfills_without_hiding_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scenarios = [
        _scenario("missing", "group-a"),
        _scenario("accepted-0", "group-a"),
        _scenario("accepted-1", "group-a"),
    ]
    monkeypatch.setattr(
        nuplan_packing,
        "_nuplan_local_sensor_asset_issue",
        lambda scenario: (
            "local lidar asset is missing"
            if scenario.sample_uid == "missing"
            else None
        ),
    )

    manifest = pack_nuplan_reactive_scenarios(
        scenarios,
        tmp_path / "output",
        source_revision="nuplan-v1.1-mini",
        map_version="nuplan-maps-v1.0",
        samples_per_shard=2,
        prefilter_local_sensor_assets=True,
        target_accepted_count=2,
        sample_builder=_sample_builder,
    )

    assert manifest["total_samples"] == 2
    assert manifest["prefiltered_count"] == 1
    assert manifest["rejection_count"] == 0
    assert manifest["prefiltered_samples"] == [{
        "log_name": "group-a",
        "reason": "local lidar asset is missing",
        "scenario_token": "token-missing",
    }]
    assert (
        tmp_path / "output" / "manifest.json"
    ).read_bytes() == canonical_json_bytes(manifest)
    assert not (
        tmp_path / "output" / ".manifest.json.tmp"
    ).exists()


def test_nuplan_bounded_pack_reports_top_failure_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scenarios = [
        _scenario("missing", "group-a"),
        _scenario("rejected-0", "group-a", rejected=True),
        _scenario("rejected-1", "group-a", rejected=True),
    ]
    monkeypatch.setattr(
        nuplan_packing,
        "_nuplan_local_sensor_asset_issue",
        lambda scenario: (
            "local lidar asset is missing"
            if scenario.sample_uid == "missing"
            else None
        ),
    )

    with pytest.raises(ValueError) as captured:
        pack_nuplan_reactive_scenarios(
            scenarios,
            tmp_path / "output",
            source_revision="nuplan-v1.1-mini",
            map_version="nuplan-maps-v1.0",
            prefilter_local_sensor_assets=True,
            target_accepted_count=1,
            sample_builder=_sample_builder,
        )

    message = str(captured.value)
    assert "top_prefilter_reasons=" in message
    assert "('local lidar asset is missing', 1)" in message
    assert "top_rejection_reasons=" in message
    assert "('ValueError: synthetic rejection', 2)" in message


def test_nuplan_bounded_pack_prioritizes_log_coverage(
    tmp_path: Path,
):
    scenarios = [
        _scenario("b-accepted-0", "group-b"),
        _scenario("b-accepted-1", "group-b"),
        _scenario("a-rejected", "group-a", rejected=True),
        _scenario("a-accepted", "group-a"),
    ]

    manifest = pack_nuplan_reactive_scenarios(
        scenarios,
        tmp_path / "output",
        source_revision="nuplan-v1.1-mini",
        map_version="nuplan-maps-v1.0",
        samples_per_shard=2,
        max_rejection_fraction=1.0,
        target_accepted_count=2,
        sample_builder=_sample_builder,
    )

    assert manifest["total_samples"] == 2
    assert manifest["split_group_uids"] == ["group-a", "group-b"]


def test_nuplan_partition_merge_enforces_global_rejection_before_move(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("a.db", 10), ("b.db", 9)],
    )
    manifests = [
        _worker_manifest(
            partition_root / f"worker-{index:03d}",
            [
                _scenario(f"accepted-{index}", f"group-{index}"),
                _scenario(
                    f"rejected-{index}",
                    f"group-{index}",
                    rejected=True,
                ),
            ],
        )
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="rejection policy failed"):
        _merge_nuplan_pack_partitions(
            output=output,
            partition_root=partition_root,
            partitions=partitions,
            manifests=manifests,
            max_rejection_fraction=0.49,
        )

    assert not list(output.glob("*.tar"))
    assert len(list(partition_root.rglob("*.tar"))) == 2


def test_nuplan_partition_merge_rejects_duplicate_sample_uid(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("a.db", 10), ("b.db", 9)],
    )
    manifests = [
        _worker_manifest(
            partition_root / f"worker-{index:03d}",
            [_scenario("duplicate", f"group-{index}")],
        )
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="duplicate sample UIDs"):
        _merge_nuplan_pack_partitions(
            output=output,
            partition_root=partition_root,
            partitions=partitions,
            manifests=manifests,
            max_rejection_fraction=0.0,
        )

    assert not list(output.glob("*.tar"))


def test_nuplan_partition_merge_rejects_corrupted_shard(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("a.db", 10)],
    )
    manifest = _worker_manifest(
        partition_root / "worker-000",
        [_scenario("accepted", "group-accepted")],
    )
    shard_names = cast(list[str], manifest["shard_names"])
    shard_path = partition_root / "worker-000" / shard_names[0]
    with shard_path.open("ab") as shard:
        shard.write(b"corruption")

    with pytest.raises(ValueError, match="checksum mismatch"):
        _merge_nuplan_pack_partitions(
            output=output,
            partition_root=partition_root,
            partitions=partitions,
            manifests=[manifest],
            max_rejection_fraction=0.0,
        )


def test_nuplan_partition_merge_rejects_unreported_shard(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("a.db", 10)],
    )
    worker_directory = partition_root / "worker-000"
    manifest = _worker_manifest(
        worker_directory,
        [_scenario("accepted", "group-accepted")],
    )
    shard_names = cast(list[str], manifest["shard_names"])
    shutil.copyfile(
        worker_directory / shard_names[0],
        worker_directory / "nuplan-unreported.tar",
    )

    with pytest.raises(ValueError, match="account for every shard"):
        _merge_nuplan_pack_partitions(
            output=output,
            partition_root=partition_root,
            partitions=partitions,
            manifests=[manifest],
            max_rejection_fraction=0.0,
        )


def _worker_config(tmp_path: Path) -> _NuPlanPackWorkerConfig:
    return _NuPlanPackWorkerConfig(
        partition=_NuPlanPackPartition(
            index=0,
            db_files=(str(tmp_path / "empty.db"),),
            scenario_estimate=10,
        ),
        data_root=str(tmp_path / "data"),
        map_root=str(tmp_path / "maps"),
        sensor_root=str(tmp_path / "sensors"),
        output_directory=str(tmp_path / "output"),
        source_revision="nuplan-v1.1-complete",
        map_version="nuplan-maps-v1.0",
        image_size=REACTIVE_CAMERA_IMAGE_SIZE,
        samples_per_shard=100,
    )


def test_nuplan_partition_worker_allows_all_rejected_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def all_rejected(**kwargs):
        captured.update(kwargs)
        return {"rejection_count": 2, "total_samples": 0}

    monkeypatch.setattr(
        nuplan_packing,
        "pack_nuplan_local_dataset",
        all_rejected,
    )

    assert _pack_nuplan_partition(_worker_config(tmp_path)) == {
        "rejection_count": 2,
        "total_samples": 0,
    }
    assert captured["require_accepted"] is False
    assert captured["limit_total_scenarios"] == 0


def test_nuplan_partition_worker_reports_filtered_empty_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prefiltered = [
        {
            "log_name": f"log-{index}",
            "reason": "missing sensor",
            "scenario_token": f"token-{index}",
        }
        for index in range(7)
    ]

    def filtered_empty(**_kwargs):
        raise _NuPlanNoScenariosError(
            "nuPlan scenario builder returned no scenarios",
            prefiltered_samples=prefiltered,
        )

    monkeypatch.setattr(
        nuplan_packing,
        "pack_nuplan_local_dataset",
        filtered_empty,
    )

    assert _pack_nuplan_partition(_worker_config(tmp_path)) == {
        "empty_partition": True,
        "empty_reason": "nuPlan scenario builder returned no scenarios",
        "prefiltered_count": 7,
        "prefiltered_samples": prefiltered,
        "rejected_count": 0,
    }


def test_nuplan_partition_worker_preserves_other_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def corrupted(**_kwargs):
        raise ValueError("corrupt nuPlan partition")

    monkeypatch.setattr(
        nuplan_packing,
        "pack_nuplan_local_dataset",
        corrupted,
    )

    with pytest.raises(ValueError, match="corrupt nuPlan partition"):
        _pack_nuplan_partition(_worker_config(tmp_path))


def _create_tagged_db(path: Path, scenario_count: int) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE scenario_tag (token TEXT)")
        connection.executemany(
            "INSERT INTO scenario_tag VALUES (?)",
            [(f"token-{index}",) for index in range(scenario_count)],
        )
        connection.commit()
    finally:
        connection.close()


def test_nuplan_local_parallel_path_builds_spawned_worker_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_root = tmp_path / "data"
    map_root = tmp_path / "maps"
    sensor_root = tmp_path / "sensors"
    data_root.mkdir()
    map_root.mkdir()
    sensor_root.mkdir()
    db_files = [data_root / "a.db", data_root / "b.db"]
    for db_file in db_files:
        _create_tagged_db(db_file, 2)
    executor_state: dict[str, object] = {}
    context = object()

    class InlineExecutor:
        def __init__(self, **kwargs: object):
            executor_state.update(kwargs)

        def __enter__(self) -> InlineExecutor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def map(self, function, configs):
            executor_state["function"] = function
            executor_state["configs"] = list(configs)
            return [
                function(config)
                for config in executor_state["configs"]
            ]

    def fake_partition(
        config: _NuPlanPackWorkerConfig,
    ) -> dict[str, object]:
        log_name = Path(config.partition.db_files[0]).stem
        return _worker_manifest(
            Path(config.output_directory),
            [
                _scenario(
                    f"sample-{config.partition.index}-{sample_index}",
                    _nuplan_split_group_uid(log_name),
                )
                for sample_index in range(
                    config.partition.scenario_limit
                )
            ],
        )

    monkeypatch.setattr(
        nuplan_packing.multiprocessing,
        "get_context",
        lambda method: (
            executor_state.update(context_method=method) or context
        ),
    )
    monkeypatch.setattr(
        nuplan_packing,
        "ProcessPoolExecutor",
        InlineExecutor,
    )
    monkeypatch.setattr(
        nuplan_packing,
        "_pack_nuplan_partition",
        fake_partition,
    )
    monkeypatch.setattr(
        nuplan_packing,
        "_nuplan_db_scenario_count",
        lambda _path: 2,
    )
    for variable in (
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        monkeypatch.setenv(variable, "8")
    monkeypatch.setenv("NUM_NODES", "8")

    merged = pack_nuplan_local_dataset(
        data_root=data_root,
        map_root=map_root,
        sensor_root=sensor_root,
        db_files=db_files,
        output_directory=tmp_path / "output",
        source_revision="nuplan-v1.1-mini",
        map_version="nuplan-maps-v1.0",
        limit_total_scenarios=3,
        pack_workers=2,
    )

    configs = cast(
        list[_NuPlanPackWorkerConfig],
        executor_state["configs"],
    )
    assert executor_state["context_method"] == "spawn"
    assert executor_state["mp_context"] is context
    assert executor_state["max_workers"] == 2
    assert executor_state["initializer"] is _initialize_nuplan_pack_worker
    assert len(configs) == 2
    assert sum(
        config.partition.scenario_limit for config in configs
    ) == 3
    assert all(
        config.partition.scenario_limit > 0 for config in configs
    )
    assert merged["packing_workers"] == 2
    assert merged["packing_scenario_limit"] == 3
    assert merged["packing_source_log_count"] == 2
    assert merged["packing_covered_log_count"] == 2
    assert merged["total_samples"] == 3
    assert all(
        os.environ[variable] == "1"
        for variable in (
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
    )
    assert os.environ["NUM_NODES"] == "1"


def test_nuplan_local_parallel_path_validates_limits_before_db_work(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="packing limits"):
        pack_nuplan_local_dataset(
            data_root=tmp_path / "missing-data",
            map_root=tmp_path / "missing-maps",
            sensor_root=tmp_path / "missing-sensors",
            db_files=[tmp_path / "missing.db"],
            output_directory=tmp_path / "output",
            source_revision="nuplan-v1.1-mini",
            map_version="nuplan-maps-v1.0",
            max_rejection_fraction=1.1,
            pack_workers=2,
        )


def test_nuplan_local_parallel_path_rejects_duplicate_log_names(
    tmp_path: Path,
):
    data_root = tmp_path / "data"
    map_root = tmp_path / "maps"
    sensor_root = tmp_path / "sensors"
    first_root = data_root / "first"
    second_root = data_root / "second"
    first_root.mkdir(parents=True)
    second_root.mkdir()
    map_root.mkdir()
    sensor_root.mkdir()
    first = first_root / "same.db"
    second = second_root / "same.db"
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="duplicate log names"):
        pack_nuplan_local_dataset(
            data_root=data_root,
            map_root=map_root,
            sensor_root=sensor_root,
            db_files=[first, second],
            output_directory=tmp_path / "output",
            source_revision="nuplan-v1.1-mini",
            map_version="nuplan-maps-v1.0",
            pack_workers=2,
        )


def test_nuplan_pack_worker_initializer_pins_native_threads(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(setNumThreads=calls.append),
    )
    for variable in (
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        monkeypatch.setenv(variable, "8")

    _initialize_nuplan_pack_worker()

    assert calls == [1]
    assert all(
        os.environ[variable] == "1"
        for variable in (
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
    )
