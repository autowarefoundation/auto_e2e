from types import SimpleNamespace

import pytest

from Platform.scripts.extract_navigation_comparison_inputs import (
    build_comparison_inputs,
)


CONDITIONED_EXECUTION = "a8xz87tksvcp9zwwfpzk"
BASELINE_EXECUTION = "asm7xnll2svdjgq6q7sq"
ARTIFACT_SET = "e" * 64
RECOVERY_MANIFEST = (
    "s3://checkpoints/recovery-manifests/"
    f"{ARTIFACT_SET}.json"
)


def _inputs(route_conditioning, **overrides):
    values = {
        "recovery_manifest": RECOVERY_MANIFEST,
        "artifact_set_sha256": ARTIFACT_SET,
        "dataset_version": "v3.0",
        "image_size": 256,
        "pack_concurrency": 60,
        "backbone": "swin_v2_tiny",
        "epochs": 10,
        "batch_size": 1,
        "grad_accum_steps": 4,
        "lr": 1e-4,
        "training_seed": 149,
        "reasoning_mode": "pooled_latent",
        "val_fraction": 0.1,
        "num_workers": 4,
        "resume_from": None,
        "early_stopping_patience": 3,
        "enable_route_conditioning": route_conditioning,
    }
    values.update(overrides)
    return values


def _literal(uri):
    return SimpleNamespace(
        scalar=SimpleNamespace(
            blob=SimpleNamespace(uri=uri),
        )
    )


def _execution(execution_id, route_conditioning, **overrides):
    workflow = SimpleNamespace(
        id=SimpleNamespace(
            name=(
                "Platform.pipelines.workflows."
                "wf_recovered_kitscenes_full_run"
            )
        ),
        flyte_nodes=[
            SimpleNamespace(
                id="n2",
                flyte_entity=SimpleNamespace(
                    name="Platform.pipelines.workflows.train_il"
                ),
                metadata=SimpleNamespace(name="train_il"),
            )
        ],
    )
    return SimpleNamespace(
        id=execution_id,
        closure=SimpleNamespace(phase=4),
        flyte_workflow=workflow,
        inputs=_inputs(route_conditioning, **overrides),
    )


class _Client:
    def __init__(self, outputs):
        self.outputs = outputs

    def list_node_executions(self, execution_id, limit, token):
        assert limit == 100
        assert token is None
        node_id = SimpleNamespace(
            execution_id=execution_id,
            node_id="n2",
        )
        return [
            SimpleNamespace(
                id=node_id,
                closure=SimpleNamespace(phase=3),
            )
        ], None

    def get_node_execution_data(self, node_id):
        return self.outputs[node_id.execution_id]


class _Remote:
    def __init__(self, conditioned, baseline):
        self.executions = {
            CONDITIONED_EXECUTION: conditioned,
            BASELINE_EXECUTION: baseline,
        }
        self.client = _Client({
            CONDITIONED_EXECUTION: SimpleNamespace(literals={
                "checkpoint": _literal("s3://checkpoints/conditioned.pt"),
                "metadata": _literal("s3://artifacts/conditioned.json"),
            }),
            BASELINE_EXECUTION: SimpleNamespace(literals={
                "checkpoint": _literal("s3://checkpoints/baseline.pt"),
                "metadata": _literal("s3://artifacts/baseline.json"),
            }),
        })

    def fetch_execution(self, name):
        return self.executions[name]

    def _get_output_literal_map(self, data):
        return data


def _remote(**baseline_overrides):
    return _Remote(
        _execution(CONDITIONED_EXECUTION, True),
        _execution(
            BASELINE_EXECUTION,
            False,
            **baseline_overrides,
        ),
    )


def test_build_comparison_inputs_resolves_controlled_pair():
    payload = build_comparison_inputs(
        _remote(),
        conditioned_execution_id=CONDITIONED_EXECUTION,
        baseline_execution_id=BASELINE_EXECUTION,
    )

    assert payload == {
        "recovery_manifest": RECOVERY_MANIFEST,
        "artifact_set_sha256": ARTIFACT_SET,
        "conditioned_checkpoint": "s3://checkpoints/conditioned.pt",
        "conditioned_train_metadata": "s3://artifacts/conditioned.json",
        "baseline_checkpoint": "s3://checkpoints/baseline.pt",
        "baseline_train_metadata": "s3://artifacts/baseline.json",
        "dataset_version": "v3.0",
        "image_size": 256,
        "pack_concurrency": 60,
    }


def test_build_comparison_inputs_rejects_seed_drift():
    with pytest.raises(
        ValueError,
        match=r"comparison inputs differ: \['training_seed'\]",
    ):
        build_comparison_inputs(
            _remote(training_seed=150),
            conditioned_execution_id=CONDITIONED_EXECUTION,
            baseline_execution_id=BASELINE_EXECUTION,
        )


def test_build_comparison_inputs_rejects_wrong_route_role():
    remote = _Remote(
        _execution(CONDITIONED_EXECUTION, False),
        _execution(BASELINE_EXECUTION, True),
    )

    with pytest.raises(ValueError, match="route mode is False"):
        build_comparison_inputs(
            remote,
            conditioned_execution_id=CONDITIONED_EXECUTION,
            baseline_execution_id=BASELINE_EXECUTION,
        )


def test_build_comparison_inputs_requires_completed_workflows():
    remote = _remote()
    remote.executions[BASELINE_EXECUTION].closure.phase = 2

    with pytest.raises(ValueError, match="is not SUCCEEDED"):
        build_comparison_inputs(
            remote,
            conditioned_execution_id=CONDITIONED_EXECUTION,
            baseline_execution_id=BASELINE_EXECUTION,
        )
