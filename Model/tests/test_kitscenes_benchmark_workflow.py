"""Flyte graph contracts for retrospective KITScenes benchmark scoring."""

from __future__ import annotations

import ast
import inspect

import pytest

pytest.importorskip("flytekit")

from Platform.pipelines import workflows  # noqa: E402


def _task_tree():
    return ast.parse(inspect.getsource(
        workflows.evaluate_kitscenes_benchmark_checkpoint.task_function
    ))


def test_benchmark_workflow_is_one_independent_evaluation_node():
    assert sorted(
        workflows.wf_evaluate_kitscenes_benchmark.python_interface.inputs
    ) == [
        "batch_size",
        "benchmark_manifest",
        "benchmark_shards",
        "checkpoint",
        "expected_manifest_sha256",
        "mlflow_run_id",
    ]
    assert [
        node.flyte_entity.name
        for node in workflows.wf_evaluate_kitscenes_benchmark.nodes
    ] == [
        workflows.evaluate_kitscenes_benchmark_checkpoint.name
    ]


def test_training_workflows_do_not_invoke_benchmark_task():
    benchmark_name = (
        workflows.evaluate_kitscenes_benchmark_checkpoint.name
    )
    for training_workflow in (
        workflows.wf_sharded_full_run,
        workflows.wf_recovered_kitscenes_full_run,
        workflows.wf_train_il,
    ):
        assert benchmark_name not in {
            node.flyte_entity.name for node in training_workflow.nodes
        }


def test_benchmark_task_records_its_resolved_runtime_image():
    task = workflows.evaluate_kitscenes_benchmark_checkpoint

    assert task.container_image == workflows.EVAL_IMAGE
    assert task.environment == {
        "MLFLOW_TRACKING_URI": workflows.MLFLOW_URI,
        "AUTO_E2E_EVAL_IMAGE": workflows.EVAL_IMAGE,
    }


def test_benchmark_task_filters_uids_and_never_decodes_future_images():
    source = inspect.getsource(
        workflows.evaluate_kitscenes_benchmark_checkpoint.task_function
    )

    assert "sample_uids=manifest.sample_uids" in source
    assert "decode_future_frames=False" in source
    assert "limit_egomotion_history(" in source
    assert 'batch.get("future_frames")' in source
    assert 'packed_manifest.get("has_map", False)' in source
    assert 'packed_manifest.get("has_gps", False)' in source
    assert 'batch.get("pose_current")' in source
    assert 'batch.get("gps_future")' in source
    assert "wgs84_trajectory_to_ego_xy(" in source
    assert '"target_trajectory": "packed_gps_to_utm32_ego_frame"' in source
    assert '"map" not in manifest.input_track.lower()' in source

    tree = _task_tree()
    model_call = next(
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "model"
        )
    )
    model_keywords = {keyword.arg for keyword in model_call.keywords}
    assert "history_frames" in model_keywords
    assert "future_frames" not in model_keywords


def test_benchmark_metrics_use_checkpoint_epoch_and_existing_run():
    tree = _task_tree()
    log_metrics_call = next(
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "log_metrics"
        )
    )
    log_keywords = {
        keyword.arg: keyword.value
        for keyword in log_metrics_call.keywords
    }
    assert isinstance(log_keywords["step"], ast.Name)
    assert log_keywords["step"].id == "epoch"

    start_run_call = next(
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "start_run"
        )
    )
    run_keywords = {
        keyword.arg: keyword.value
        for keyword in start_run_call.keywords
    }
    assert isinstance(run_keywords["run_id"], ast.Name)
    assert run_keywords["run_id"].id == "run_id"


def test_benchmark_task_emits_only_released_displacement_metrics():
    source = inspect.getsource(
        workflows.evaluate_kitscenes_benchmark_checkpoint.task_function
    )

    for metric in ("ade_3s", "fde_3s", "ade_5s", "fde_5s"):
        assert f'metrics["{metric}"]' in source
    for unavailable in (
        "centerline_distance",
        "collision_free_rate",
        "drivable_surface_survival",
        "mms",
    ):
        assert (
            f'"{unavailable}": "authority_assets_required"'
            in source
        )
