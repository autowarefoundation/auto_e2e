"""Contiguous nuPlan scene partitions for held-out Console playback."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from data_parsing.nuplan import packing as nuplan_packing
from data_parsing.nuplan.packing import (
    NUPLAN_CAMERA_CHANNELS,
    NUPLAN_CAMERA_SLOTS,
    NUPLAN_DATASET_ID,
    NUPLAN_EGO_GROUND_Z_M,
    NUPLAN_PACK_MANIFEST_VERSION,
    NUPLAN_RIG_REFERENCE_FRAME,
    NUPLAN_SCENE_FRAME_STRIDE,
    _NuPlanSceneWorkerConfig,
    _initialize_nuplan_pack_worker,
    _sample_identity,
    nuplan_scene_episode_id,
    nuplan_scene_partition_id,
    nuplan_scene_sample_uid,
    nuplan_static_rig_projection,
    pack_nuplan_local_scenes,
    pack_nuplan_reactive_scene_partitions,
)
from data_parsing.pre_extracted import packed_sample_tar_paths
from navigation.contracts import canonical_json_bytes
from Platform.pipelines import nuplan_dataset
from Platform.pipelines.dataset_publication import (
    PUBLICATION_SCHEMA,
    merge_partition_results,
)
from Platform.pipelines.dataset_publication_tasks import (
    _plan_partition_artifact_copies,
)
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_FRAME_OFFSETS,
    REACTIVE_CAMERA_IMAGE_SIZE,
)


def _camera_calibration(yaw_offset: float = 0.0) -> list[dict[str, object]]:
    cameras = []
    for index, channel in enumerate(NUPLAN_CAMERA_CHANNELS):
        yaw = index * np.pi / 3 + yaw_offset
        # Optical axis along +X ego, image x to the right, image y down.
        rotation = np.array([
            [np.sin(yaw), 0.0, np.cos(yaw)],
            [-np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, -1.0, 0.0],
        ])
        ego_from_camera = np.eye(4)
        ego_from_camera[:3, :3] = rotation
        ego_from_camera[:3, 3] = [1.5, 0.1 * index, 1.5]
        cameras.append({
            "base_scaled_rectified_intrinsic": [
                [350.0 + index, 0.0, 256.0],
                [0.0, 700.0, 260.0],
                [0.0, 0.0, 1.0],
            ],
            "channel": channel,
            "image_time_offset_us": 0,
            "sensor_to_ego": ego_from_camera.tolist(),
        })
    return cameras


def _calib(yaw_offset: float = 0.0) -> dict[str, object]:
    return {
        "camera_order": list(NUPLAN_CAMERA_CHANNELS),
        "cameras": _camera_calibration(yaw_offset),
        "image_size": REACTIVE_CAMERA_IMAGE_SIZE,
        "rectification_policy": "nuplan_rectified_pinhole_v1",
        "reference_lidar_timestamp_us": 4_000_000,
        "temporal_camera_timestamps_us": [
            [
                4_000_000 + offset * REACTIVE_BEVFORMER_FRAME_INTERVAL_US
                for _ in NUPLAN_CAMERA_CHANNELS
            ]
            for offset in REACTIVE_BEVFORMER_FRAME_OFFSETS[:-1]
        ],
        "temporal_frame_interval_us": REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
        "temporal_frame_offsets": list(REACTIVE_BEVFORMER_FRAME_OFFSETS),
    }


def _scene_scenario(
    log_name: str,
    token: str,
    *,
    iterations: int = 400,
    failing_iteration: int | None = None,
    calibration_change_iteration: int | None = None,
    scenario_type: str = "traversing_intersection",
) -> SimpleNamespace:
    return SimpleNamespace(
        calibration_change_iteration=calibration_change_iteration,
        database_interval=0.05,
        failing_iteration=failing_iteration,
        get_number_of_iterations=lambda: iterations,
        log_name=log_name,
        scenario_type=scenario_type,
        token=token,
    )


def _scene_sample_builder(
    scenario: SimpleNamespace,
    *,
    iteration: int,
    **_kwargs: object,
) -> tuple[str, str, dict[str, bytes]]:
    if iteration == scenario.failing_iteration:
        raise ValueError("synthetic camera gap")
    sample_uid, split_group_uid = _sample_identity(
        scenario,
        iteration=iteration,
    )
    change = scenario.calibration_change_iteration
    calib = _calib(
        0.01 if change is not None and iteration >= change else 0.0
    )
    return sample_uid, split_group_uid, {
        "calib.json": canonical_json_bytes(calib),
        "meta.json": canonical_json_bytes({
            "frame_idx": iteration,
            "sample_uid": sample_uid,
        }),
    }


def _pack(
    scenarios: list[SimpleNamespace],
    output: Path,
    *,
    scene_count: int,
    frames_per_scene: int = 4,
    **kwargs: Any,
) -> dict[str, object]:
    return pack_nuplan_reactive_scene_partitions(
        scenarios,
        output,
        source_revision="nuplan-v1.1-complete",
        map_version="nuplan-maps-v1.0",
        publication_version="v1.0",
        scene_count=scene_count,
        frames_per_scene=frames_per_scene,
        sample_builder=_scene_sample_builder,
        **kwargs,
    )


def test_scene_iterations_extend_training_sample_identity():
    scenario = _scene_scenario("log-a", "00ff")
    legacy_digest = hashlib.sha256(b"log-a:00ff").hexdigest()[:24]

    first_uid, split_group_uid = _sample_identity(scenario)
    later_uid, later_group = _sample_identity(scenario, iteration=2)

    assert first_uid == f"nuplan-{legacy_digest}"
    assert _sample_identity(scenario, iteration=0)[0] == first_uid
    assert later_uid != first_uid
    assert later_group == split_group_uid
    with pytest.raises(ValueError, match="non-negative"):
        _sample_identity(scenario, iteration=-1)


def test_static_rig_matches_motion_free_sample_projection():
    calib = _calib()
    rig = nuplan_static_rig_projection(calib)
    projection = cast(dict[str, Any], rig["projection"])
    matrices = np.asarray(projection["matrix"])

    assert rig["dataset"] == NUPLAN_DATASET_ID
    assert rig["geometry_type"] == "rectified_pinhole"
    assert projection["camera_order"] == list(NUPLAN_CAMERA_CHANNELS)
    assert projection["camera_slots"] == list(NUPLAN_CAMERA_SLOTS)
    assert projection["reference_frame"] == NUPLAN_RIG_REFERENCE_FRAME
    assert projection["ground_z_m"] == NUPLAN_EGO_GROUND_Z_M
    assert matrices.shape == (6, 3, 4)
    for camera, matrix in zip(
        cast(list[dict[str, Any]], calib["cameras"]),
        matrices,
        strict=True,
    ):
        ego_from_camera = np.asarray(camera["sensor_to_ego"])
        centre = np.append(ego_from_camera[:3, 3], 1.0)
        # The camera centre is the null space of its projection matrix.
        np.testing.assert_allclose(matrix @ centre, 0.0, atol=1e-3)
        ahead = ego_from_camera @ np.array([0.0, 0.0, 10.0, 1.0])
        pixel = matrix @ ahead
        intrinsic = np.asarray(camera["base_scaled_rectified_intrinsic"])
        np.testing.assert_allclose(
            pixel[:2] / pixel[2],
            intrinsic[:2, 2],
            atol=1e-3,
        )

    reordered = dict(calib)
    reordered["cameras"] = list(reversed(_camera_calibration()))
    with pytest.raises(ValueError, match="camera order"):
        nuplan_static_rig_projection(reordered)


def test_scene_partitions_are_contiguous_and_publishable(tmp_path: Path):
    scenarios = [
        _scene_scenario("log-b", "0b01"),
        _scene_scenario("log-a", "0a01"),
        _scene_scenario("log-a", "0a02"),
    ]

    summary = _pack(scenarios, tmp_path / "output", scene_count=2)

    expected_ids = [
        nuplan_scene_partition_id(scenarios[1]),
        nuplan_scene_partition_id(scenarios[0]),
    ]
    assert summary["partition_ids"] == expected_ids
    assert summary["scene_count"] == 2
    assert summary["total_samples"] == 8
    assert json.loads(
        (tmp_path / "output" / "manifest.json").read_bytes()
    ) == summary

    results = []
    for partition_id in expected_ids:
        directory = tmp_path / "output" / "partitions" / partition_id
        manifest = json.loads((directory / "manifest.json").read_bytes())
        rig = json.loads(
            (directory / "rig" / "projection.json").read_bytes()
        )
        shard_name = f"{partition_id}-000000.tar"
        assert manifest["schema_version"] == NUPLAN_PACK_MANIFEST_VERSION
        assert manifest["dataset"] == NUPLAN_DATASET_ID
        assert manifest["dataset_version"] == "v1.0"
        assert manifest["partition_id"] == partition_id
        assert manifest["shard_names"] == [shard_name]
        assert manifest["shards"] == 1
        assert manifest["hz"] == 10
        assert manifest["frame_stride"] == NUPLAN_SCENE_FRAME_STRIDE
        assert manifest["total_samples"] == 4
        assert manifest["episodes"] == 1
        assert manifest["has_map"] is True
        assert manifest["projection_scope"] == "per_sample"
        assert "projection" not in manifest
        assert rig == nuplan_static_rig_projection(_calib())
        assert packed_sample_tar_paths(directory) == (
            directory / shard_name,
        )
        metas = []
        with tarfile.open(directory / shard_name) as archive:
            for member in archive.getmembers():
                if not member.name.endswith(".meta.json"):
                    continue
                stream = archive.extractfile(member)
                assert stream is not None
                meta = json.loads(stream.read())
                assert member.name == f"{meta['sample_uid']}.meta.json"
                metas.append(meta)
        episode_id = nuplan_scene_episode_id(partition_id)
        assert [meta["frame_idx"] for meta in metas] == [0, 1, 2, 3]
        assert [meta["source_iteration"] for meta in metas] == [0, 2, 4, 6]
        assert [meta["sample_uid"] for meta in metas] == [
            f"nuplan-v1-{episode_id}-f{frame:06d}" for frame in range(4)
        ]
        assert {meta["episode_id"] for meta in metas} == {episode_id}
        assert metas[0]["source_sample_uid"] == _sample_identity(
            next(
                scenario
                for scenario in scenarios
                if nuplan_scene_partition_id(scenario) == partition_id
            )
        )[0]
        assert manifest["has_gps"] is False

        objects = [
            {
                "relative": relative,
                "key": f"source/{relative}",
                "size": (directory / relative).stat().st_size,
                "content_identity": f"identity-{relative}",
            }
            for relative in (
                "manifest.json",
                "rig/projection.json",
                shard_name,
            )
        ]
        copies, shards, pools = _plan_partition_artifact_copies(
            objects,
            manifest,
            published_dataset="nuplan-test",
            dataset_version="v1.0",
        )
        assert [copy["relative"] for copy in copies] == [shard_name]
        assert pools == []
        results.append({
            "schema_version": PUBLICATION_SCHEMA,
            "source_uri": f"s3://artifacts/{partition_id}",
            "source_manifest_sha256": "0" * 64,
            "dataset_version": "v1.0",
            "manifest": manifest,
            "rig": rig,
            "shards": [{**shard, "etag": '"etag"'} for shard in shards],
            "pool": {"object_count": 0, "byte_size": 0, "digest": "0"},
            "geo": None,
        })

    published, rigs, heatmap = merge_partition_results(
        results,
        dataset="nuplan-test",
        version="v1.0",
    )
    assert published["total_samples"] == 8
    assert published["shard_count"] == 2
    assert published["rig_count"] == 1
    assert published["hz"] == 10
    assert published["num_views"] == 6
    assert published["episodes"] == 2
    assert published["has_map"] is True
    assert len(rigs) == 1
    assert heatmap is None


def test_scene_with_rejected_frame_falls_back_within_the_log(
    tmp_path: Path,
):
    scenarios = [
        _scene_scenario("log-a", "0a01", failing_iteration=4),
        _scene_scenario("log-a", "0a02"),
    ]

    summary = _pack(scenarios, tmp_path / "output", scene_count=1)

    assert summary["partition_ids"] == [
        nuplan_scene_partition_id(scenarios[1])
    ]
    rejected = cast(list[dict[str, str]], summary["rejected_scenes"])
    assert [item["scenario_token"] for item in rejected] == ["0a01"]
    assert "synthetic camera gap" in rejected[0]["error"]
    partitions = tmp_path / "output" / "partitions"
    assert sorted(path.name for path in partitions.iterdir()) == [
        nuplan_scene_partition_id(scenarios[1])
    ]


def test_scene_rejects_short_scenarios_and_calibration_changes(
    tmp_path: Path,
):
    scenarios = [
        _scene_scenario("log-a", "0a01", iterations=6),
        _scene_scenario("log-a", "0a02", calibration_change_iteration=4),
        _scene_scenario("log-b", "0b01"),
    ]

    summary = _pack(scenarios, tmp_path / "output", scene_count=1)

    assert summary["partition_ids"] == [
        nuplan_scene_partition_id(scenarios[2])
    ]
    errors = [
        item["error"]
        for item in cast(list[dict[str, str]], summary["rejected_scenes"])
    ]
    assert "shorter than the scene" in errors[0]
    assert "calibration changed" in errors[1]


def test_scene_packing_reports_an_unfilled_target(tmp_path: Path):
    scenarios = [
        _scene_scenario("log-a", "0a01", failing_iteration=0),
        _scene_scenario("log-b", "0b01"),
    ]

    with pytest.raises(ValueError, match="accepted=1 rejected=1"):
        _pack(scenarios, tmp_path / "output", scene_count=2)


def test_scene_candidate_issue_skips_without_packing(tmp_path: Path):
    scenarios = [
        _scene_scenario("log-a", "0a01"),
        _scene_scenario("log-a", "0a02"),
    ]

    summary = _pack(
        scenarios,
        tmp_path / "output",
        scene_count=1,
        candidate_issue=lambda scenario: (
            "missing camera" if scenario.token == "0a01" else None
        ),
    )

    assert summary["partition_ids"] == [
        nuplan_scene_partition_id(scenarios[1])
    ]
    assert cast(list[dict[str, str]], summary["rejected_scenes"])[0][
        "error"
    ] == "prefiltered: missing camera"


def test_scene_stride_must_produce_ten_hertz(tmp_path: Path):
    scenarios = [_scene_scenario("log-a", "0a01")]

    with pytest.raises(ValueError, match="does not produce 10 Hz frames"):
        _pack(scenarios, tmp_path / "output", scene_count=1, frame_stride=1)


def test_parallel_scenes_replace_logs_without_valid_scenes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_root = tmp_path / "data"
    map_root = tmp_path / "maps"
    sensor_root = tmp_path / "sensors"
    for directory in (data_root, map_root, sensor_root):
        directory.mkdir()
    db_files = [data_root / f"log-{name}.db" for name in "dcba"]
    for db_file in db_files:
        db_file.write_bytes(b"")
    executor_state: dict[str, Any] = {"batches": []}

    class InlineExecutor:
        def __init__(self, **kwargs: object):
            executor_state.update(kwargs)

        def __enter__(self) -> InlineExecutor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def map(self, function, configs):
            configs = list(configs)
            executor_state["batches"].append(
                [Path(config.db_file).stem for config in configs]
            )
            return [function(config) for config in configs]

    def fake_scene_log(
        config: _NuPlanSceneWorkerConfig,
    ) -> dict[str, object]:
        log_name = Path(config.db_file).stem
        if log_name == "log-b":
            return {
                "empty_log": True,
                "error": "ValueError: no scene",
                "log_name": log_name,
            }
        return _pack(
            [_scene_scenario(log_name, f"{log_name}-token")],
            Path(config.output_directory),
            scene_count=1,
            frames_per_scene=config.frames_per_scene,
        )

    monkeypatch.setattr(
        nuplan_packing.multiprocessing,
        "get_context",
        lambda method: executor_state.update(method=method) or "context",
    )
    monkeypatch.setattr(nuplan_packing, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(nuplan_packing, "_pack_nuplan_scene_log", fake_scene_log)

    summary = pack_nuplan_local_scenes(
        data_root=data_root,
        map_root=map_root,
        sensor_root=sensor_root,
        db_files=db_files,
        output_directory=tmp_path / "output",
        source_revision="nuplan-v1.1-complete",
        map_version="nuplan-maps-v1.0",
        publication_version="v1.0",
        scene_count=2,
        frames_per_scene=3,
        scene_workers=4,
    )

    assert executor_state["method"] == "spawn"
    assert executor_state["max_workers"] == 2
    assert executor_state["initializer"] is _initialize_nuplan_pack_worker
    assert executor_state["batches"] == [["log-a", "log-b"], ["log-c"]]
    assert summary["scene_count"] == 2
    assert summary["total_samples"] == 6
    scenes = cast(list[dict[str, object]], summary["scenes"])
    assert [scene["log_name"] for scene in scenes] == [
        "log-a",
        "log-c",
    ]
    assert cast(list[dict[str, object]], summary["empty_logs"])[0][
        "log_name"
    ] == "log-b"
    partitions = tmp_path / "output" / "partitions"
    assert sorted(path.name for path in partitions.iterdir()) == sorted(
        cast(list[str], summary["partition_ids"])
    )
    assert not (tmp_path / "output" / ".workers").exists()


def _test_split_archives() -> list[dict[str, str]]:
    archives = [
        {
            "archive_id": "maps-v1.0",
            "component": "maps",
            "filename": "nuplan-maps-v1.0.zip",
        },
        {
            "archive_id": "db-test",
            "component": "database",
            "filename": "nuplan-v1.1_test.zip",
        },
        {
            "archive_id": "db-train_vegas_1",
            "component": "database",
            "filename": "train.zip",
        },
    ]
    for group_index in range(nuplan_dataset.NUPLAN_TEST_GROUP_COUNT):
        for modality in ("camera", "lidar"):
            archives.append({
                "archive_id": f"sensor-test-test_{modality}_{group_index}",
                "component": "sensor_blobs",
                "filename": f"{modality}_{group_index}.zip",
            })
    return archives


def test_test_split_inventory_and_archive_sets_share_train_rules():
    payload = "\n".join(
        f"File group: {index}\nlog-{index}-0\nlog-{index}-1"
        for index in range(nuplan_dataset.NUPLAN_TEST_GROUP_COUNT)
    )
    sensor_groups = nuplan_dataset._parse_nuplan_sensor_inventory(
        payload,
        split="test",
        group_count=nuplan_dataset.NUPLAN_TEST_GROUP_COUNT,
    )
    database_logs = {
        "db-test": tuple(
            log_name
            for log_names in sensor_groups.values()
            for log_name in log_names
        ),
    }

    archive_sets, log_counts = nuplan_dataset._nuplan_split_archive_sets(
        {"archives": _test_split_archives(), "map_version": "nuplan-maps-v1.0"},
        sensor_groups,
        database_logs,
        split="test",
        group_count=nuplan_dataset.NUPLAN_TEST_GROUP_COUNT,
    )

    assert archive_sets[6] == [
        "maps-v1.0",
        "db-test",
        "sensor-test-test_camera_6",
        "sensor-test-test_lidar_6",
    ]
    assert log_counts == [2] * nuplan_dataset.NUPLAN_TEST_GROUP_COUNT
    with pytest.raises(ValueError, match="nuPlan test sensor inventory"):
        nuplan_dataset._parse_nuplan_sensor_inventory(
            payload,
            split="test",
            group_count=nuplan_dataset.NUPLAN_FULL_TRAIN_GROUP_COUNT,
        )
    with pytest.raises(ValueError, match="test DB inventory"):
        nuplan_dataset._nuplan_split_archive_sets(
            {
                "archives": _test_split_archives(),
                "map_version": "nuplan-maps-v1.0",
            },
            sensor_groups,
            {**database_logs, "db-train_vegas_1": ("other",)},
            split="test",
            group_count=nuplan_dataset.NUPLAN_TEST_GROUP_COUNT,
        )


def test_scene_workflow_binds_the_shared_scene_pack_task():
    node, = nuplan_dataset.wf_pack_nuplan_snapshot_scene_dataset.nodes
    bindings = {binding.var: binding.binding for binding in node.bindings}
    task = nuplan_dataset.pack_nuplan_snapshot_scene_partitions

    assert node.flyte_entity.name.endswith(
        "_pack_nuplan_snapshot_scene_dataset"
    )
    assert bindings["group_indices"].promise.var == "group_indices"
    assert bindings["split"].promise.var == "split"
    assert (
        bindings["require_mission_goal"].promise.var
        == "require_mission_goal"
    )
    assert task.metadata.cache_version == (
        "nuplan-scene-pack-v3-manifest-v11"
    )
    assert task.metadata.retries == 1
    assert task.python_interface.inputs["archive_ids"] is not None
    assert (
        nuplan_dataset.NUPLAN_SPLIT_GROUP_COUNTS["test"]
        == nuplan_dataset.NUPLAN_TEST_GROUP_COUNT
    )


def test_scene_candidates_prefer_goals_then_preferred_types():
    def scenario(token: str, scenario_type: str, has_goal: bool):
        return SimpleNamespace(
            token=token,
            scenario_type=scenario_type,
            get_mission_goal=lambda: object() if has_goal else None,
        )

    scenarios = [
        scenario("no-goal-turn", "starting_left_turn", False),
        scenario("goal-other", "stationary", True),
        scenario("goal-turn", "starting_left_turn", True),
        scenario("no-goal-other", "stationary", False),
    ]

    goal_first = nuplan_packing._order_nuplan_scene_candidates(
        scenarios,
        preferred_scenario_types=("starting_left_turn",),
        prefer_mission_goal=True,
    )
    type_only = nuplan_packing._order_nuplan_scene_candidates(
        scenarios,
        preferred_scenario_types=("starting_left_turn",),
        prefer_mission_goal=False,
    )

    assert [item.token for item in goal_first] == [
        "goal-turn",
        "goal-other",
        "no-goal-turn",
        "no-goal-other",
    ]
    assert [item.token for item in type_only] == [
        "no-goal-turn",
        "goal-turn",
        "goal-other",
        "no-goal-other",
    ]


def test_parallel_scene_workers_receive_the_goal_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_root = tmp_path / "data"
    for name in ("data", "maps", "sensors"):
        (tmp_path / name).mkdir()
    db_file = data_root / "log-a.db"
    db_file.write_bytes(b"")
    seen: list[bool] = []

    class InlineExecutor:
        def __init__(self, **_kwargs: object):
            pass

        def __enter__(self) -> InlineExecutor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def map(self, function, configs):
            return [function(config) for config in configs]

    def fake_scene_log(config: _NuPlanSceneWorkerConfig) -> dict[str, object]:
        seen.append(config.require_mission_goal)
        return _pack(
            [_scene_scenario("log-a", "log-a-token")],
            Path(config.output_directory),
            scene_count=1,
            frames_per_scene=config.frames_per_scene,
        )

    monkeypatch.setattr(
        nuplan_packing.multiprocessing,
        "get_context",
        lambda _method: "context",
    )
    monkeypatch.setattr(nuplan_packing, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(nuplan_packing, "_pack_nuplan_scene_log", fake_scene_log)

    pack_nuplan_local_scenes(
        data_root=data_root,
        map_root=tmp_path / "maps",
        sensor_root=tmp_path / "sensors",
        db_files=[db_file],
        output_directory=tmp_path / "output",
        source_revision="nuplan-v1.1-complete",
        map_version="nuplan-maps-v1.0",
        publication_version="v1.0",
        scene_count=1,
        frames_per_scene=2,
        scene_workers=2,
        require_mission_goal=False,
    )

    assert seen == [False]


def test_scene_sample_uids_follow_the_console_episode_frame_layout():
    assert nuplan_scene_sample_uid("nuplan-398cd2903ae3a6e9", 12) == (
        "nuplan-v1-398cd2903ae3a6e9-f000012"
    )
    assert nuplan_scene_episode_id("nuplan-398cd2903ae3a6e9") == (
        "398cd2903ae3a6e9"
    )
    with pytest.raises(ValueError, match="partition id"):
        nuplan_scene_episode_id("kitscenes-0001")
    with pytest.raises(ValueError, match="non-negative"):
        nuplan_scene_sample_uid("nuplan-398cd2903ae3a6e9", -1)


def _fake_geospatial(scenario: SimpleNamespace, iteration: int):
    lat = 42.35 + 1e-5 * iteration
    pose = {
        "latitude_deg": lat,
        "longitude_deg": -71.05,
        "heading_deg_cw_from_north": 90.0,
        "timestamp_ns": 1_000_000_000 + iteration * 50_000_000,
        "gps_accuracy_m": float("nan"),
    }
    gps_future = np.tile([[lat, -71.05]], (65, 1))
    return pose, gps_future


def test_scene_partitions_publish_wgs84_poses_and_episode_paths(
    tmp_path: Path,
):
    scenarios = [
        _scene_scenario(f"log-{index}", f"0c{index:02d}")
        for index in range(5)
    ]

    summary = _pack(
        scenarios,
        tmp_path / "output",
        scene_count=5,
        frames_per_scene=24,
        frame_geospatial=_fake_geospatial,
    )

    results = []
    for scenario in scenarios:
        partition_id = nuplan_scene_partition_id(scenario)
        episode_id = nuplan_scene_episode_id(partition_id)
        directory = tmp_path / "output" / "partitions" / partition_id
        manifest = json.loads((directory / "manifest.json").read_bytes())
        assert manifest["has_gps"] is True
        assert manifest["geospatial"]["summary"]["sample_pose_count"] == 24
        path = np.frombuffer(
            (directory / "geo" / "episode_paths" / f"{episode_id}.f64")
            .read_bytes(),
            dtype="<f8",
        ).reshape(-1, 4)
        # The first and last ten frames are excluded from published paths.
        assert path.shape == (4, 4)
        np.testing.assert_allclose(path[0, 0], 42.35 + 1e-5 * 20)
        shard_name = manifest["shard_names"][0]
        with tarfile.open(directory / shard_name) as archive:
            names = set(archive.getnames())
        uid = nuplan_scene_sample_uid(partition_id, 3)
        assert {f"{uid}.pose.npy", f"{uid}.gps.npy"} <= names

        objects = [
            {
                "relative": str(file.relative_to(directory)),
                "key": f"source/{file.relative_to(directory)}",
                "size": file.stat().st_size,
                "content_identity": f"identity-{file.name}",
            }
            for file in sorted(directory.rglob("*"))
            if file.is_file()
        ]
        copies, shards, _ = _plan_partition_artifact_copies(
            objects,
            manifest,
            published_dataset="nuplan-test",
            dataset_version="v1.0",
        )
        assert f"geo/episode_paths/{episode_id}.f64" in {
            copy["relative"] for copy in copies
        }
        results.append({
            "schema_version": PUBLICATION_SCHEMA,
            "source_uri": f"s3://artifacts/{partition_id}",
            "source_manifest_sha256": "0" * 64,
            "dataset_version": "v1.0",
            "manifest": manifest,
            "rig": json.loads(
                (directory / "rig" / "projection.json").read_bytes()
            ),
            "shards": [{**shard, "etag": '"etag"'} for shard in shards],
            "pool": {"object_count": 0, "byte_size": 0, "digest": "0"},
            "geo": {
                "privacy": manifest["geospatial"]["summary"]["privacy"],
                "source_coordinate_dtype": "float32",
                "stored_coordinate_dtype": "float64",
                "timestamp_dtype": "int64_ns",
                "gps_accuracy_available": False,
                "sample_pose_count": 24,
                "sample_pose_source": None,
                "episode_paths": [{
                    "filename": f"{episode_id}.f64",
                    "point_count": 4,
                }],
                "cells": [{
                    "lat_cell": 4235,
                    "lon_cell": -7105,
                    "sample_count": 4,
                    "episode_count": 1,
                }],
            },
        })

    published, _, heatmap = merge_partition_results(
        results,
        dataset="nuplan-test",
        version="v1.0",
    )
    assert summary["scene_count"] == 5
    assert published["has_gps"] is True
    assert published["episodes"] == 5
    assert published["geo"]["sample_pose_count"] == 120
    assert heatmap is not None
    assert len(heatmap["features"]) == 1
