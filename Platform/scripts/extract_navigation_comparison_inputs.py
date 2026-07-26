#!/usr/bin/env python3
"""Resolve paired KITScenes navigation checkpoints from completed Flyte runs."""

from __future__ import annotations

import argparse
import enum
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


_EXECUTION_ID_RE = re.compile(r"^a[a-z0-9]{19}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECOVERY_WORKFLOWS = {
    "pipelines.workflows.wf_recovered_kitscenes_full_run",
    "Platform.pipelines.workflows.wf_recovered_kitscenes_full_run",
}
_WORKFLOW_SUCCEEDED = 4
_NODE_SUCCEEDED = 3
_COMPARABLE_INPUTS = (
    "recovery_manifest",
    "artifact_set_sha256",
    "dataset_version",
    "image_size",
    "pack_concurrency",
    "backbone",
    "epochs",
    "batch_size",
    "grad_accum_steps",
    "lr",
    "training_seed",
    "reasoning_mode",
    "val_fraction",
    "num_workers",
    "resume_from",
    "early_stopping_patience",
)


def _plain_value(value: Any) -> Any:
    remote_source = getattr(value, "remote_source", None)
    if remote_source:
        return remote_source
    value = getattr(value, "value", value)
    if isinstance(value, enum.Enum):
        return value.value
    return value


def _required_input(inputs: Mapping[str, Any], name: str) -> Any:
    if name not in inputs:
        raise ValueError(f"recovery run has no {name!r} input")
    return _plain_value(inputs[name])


def _workflow_name(execution: Any) -> str:
    return str(
        getattr(
            getattr(execution.flyte_workflow, "id", None),
            "name",
            "",
        )
    )


def _validate_execution(
    execution: Any,
    *,
    execution_id: str,
    expected_route_conditioning: bool,
    expected_dataset_version: str,
) -> dict[str, Any]:
    if int(execution.closure.phase) != _WORKFLOW_SUCCEEDED:
        raise ValueError(
            f"recovery workflow {execution_id} is not SUCCEEDED "
            f"(phase={execution.closure.phase})"
        )
    workflow_name = _workflow_name(execution)
    if workflow_name not in _RECOVERY_WORKFLOWS:
        raise ValueError(
            f"execution {execution_id} has unsupported workflow "
            f"{workflow_name!r}"
        )
    inputs = {
        name: _required_input(execution.inputs, name)
        for name in _COMPARABLE_INPUTS
    }
    if inputs["dataset_version"] != expected_dataset_version:
        raise ValueError(
            f"execution {execution_id} dataset_version is "
            f"{inputs['dataset_version']!r}, expected "
            f"{expected_dataset_version!r}"
        )
    artifact_set_sha256 = str(inputs["artifact_set_sha256"])
    if not _SHA256_RE.fullmatch(artifact_set_sha256):
        raise ValueError(
            f"execution {execution_id} has an invalid artifact set digest"
        )
    if not str(inputs["recovery_manifest"]).startswith("s3://"):
        raise ValueError(
            f"execution {execution_id} recovery manifest is not on S3"
        )
    actual_route_conditioning = bool(
        _required_input(execution.inputs, "enable_route_conditioning")
    )
    if actual_route_conditioning != expected_route_conditioning:
        raise ValueError(
            f"execution {execution_id} route mode is "
            f"{actual_route_conditioning}, expected "
            f"{expected_route_conditioning}"
        )
    return inputs


def _train_node_id(execution: Any) -> str:
    matches = []
    for node in execution.flyte_workflow.flyte_nodes:
        entity_name = str(
            getattr(getattr(node, "flyte_entity", None), "name", "")
        )
        metadata_name = str(
            getattr(getattr(node, "metadata", None), "name", "")
        )
        if entity_name.endswith(".train_il") or metadata_name == "train_il":
            matches.append(str(node.id))
    if len(matches) != 1:
        raise ValueError(
            "recovery workflow must contain exactly one train_il node, "
            f"found {matches}"
        )
    return matches[0]


def _iter_node_executions(
    remote: Any,
    execution: Any,
) -> Iterable[Any]:
    token = None
    while True:
        nodes, token = remote.client.list_node_executions(
            execution.id,
            limit=100,
            token=token,
        )
        yield from nodes
        if not token:
            return


def _artifact_uri(literal_map: Any, *names: str) -> str:
    for name in names:
        literal = literal_map.literals.get(name)
        scalar = getattr(literal, "scalar", None)
        blob = getattr(scalar, "blob", None)
        uri = str(getattr(blob, "uri", ""))
        if uri:
            if not uri.startswith("s3://"):
                raise ValueError(f"train output {name!r} is not an S3 file")
            return uri
    raise ValueError(f"train output has none of {names!r}")


def _train_artifacts(
    remote: Any,
    execution: Any,
) -> dict[str, str]:
    wanted_node_id = _train_node_id(execution)
    matches = [
        node
        for node in _iter_node_executions(remote, execution)
        if str(node.id.node_id) == wanted_node_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"train node {wanted_node_id!r} has "
            f"{len(matches)} executions"
        )
    node = matches[0]
    if int(node.closure.phase) != _NODE_SUCCEEDED:
        raise ValueError(
            f"train node {wanted_node_id!r} is not SUCCEEDED "
            f"(phase={node.closure.phase})"
        )
    data = remote.client.get_node_execution_data(node.id)
    outputs = remote._get_output_literal_map(data)
    return {
        "checkpoint": _artifact_uri(outputs, "checkpoint", "o0"),
        "metadata": _artifact_uri(outputs, "metadata", "o1"),
    }


def build_comparison_inputs(
    remote: Any,
    *,
    conditioned_execution_id: str,
    baseline_execution_id: str,
    expected_dataset_version: str = "v3.0",
) -> dict[str, Any]:
    for execution_id in (
        conditioned_execution_id,
        baseline_execution_id,
    ):
        if not _EXECUTION_ID_RE.fullmatch(execution_id):
            raise ValueError(f"invalid Flyte execution ID {execution_id!r}")
    if conditioned_execution_id == baseline_execution_id:
        raise ValueError("conditioned and baseline executions must differ")

    conditioned_execution = remote.fetch_execution(
        name=conditioned_execution_id
    )
    baseline_execution = remote.fetch_execution(
        name=baseline_execution_id
    )
    conditioned_inputs = _validate_execution(
        conditioned_execution,
        execution_id=conditioned_execution_id,
        expected_route_conditioning=True,
        expected_dataset_version=expected_dataset_version,
    )
    baseline_inputs = _validate_execution(
        baseline_execution,
        execution_id=baseline_execution_id,
        expected_route_conditioning=False,
        expected_dataset_version=expected_dataset_version,
    )
    mismatches = [
        name
        for name in _COMPARABLE_INPUTS
        if conditioned_inputs[name] != baseline_inputs[name]
    ]
    if mismatches:
        raise ValueError(
            f"navigation comparison inputs differ: {mismatches}"
        )

    conditioned_artifacts = _train_artifacts(
        remote,
        conditioned_execution,
    )
    baseline_artifacts = _train_artifacts(
        remote,
        baseline_execution,
    )
    return {
        "recovery_manifest": conditioned_inputs["recovery_manifest"],
        "artifact_set_sha256": conditioned_inputs[
            "artifact_set_sha256"
        ],
        "conditioned_checkpoint": conditioned_artifacts["checkpoint"],
        "conditioned_train_metadata": conditioned_artifacts["metadata"],
        "baseline_checkpoint": baseline_artifacts["checkpoint"],
        "baseline_train_metadata": baseline_artifacts["metadata"],
        "dataset_version": conditioned_inputs["dataset_version"],
        "image_size": conditioned_inputs["image_size"],
        "pack_concurrency": conditioned_inputs["pack_concurrency"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioned-execution-id", required=True)
    parser.add_argument("--baseline-execution-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--project", default="auto-e2e")
    parser.add_argument("--domain", default="development")
    parser.add_argument("--expected-dataset-version", default="v3.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from flytekit.configuration import Config
    from flytekit.remote import FlyteRemote

    remote = FlyteRemote(
        config=Config.auto(config_file=args.config),
        default_project=args.project,
        default_domain=args.domain,
    )
    payload = build_comparison_inputs(
        remote,
        conditioned_execution_id=args.conditioned_execution_id,
        baseline_execution_id=args.baseline_execution_id,
        expected_dataset_version=args.expected_dataset_version,
    )
    output = Path(args.output)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote paired navigation inputs to {output}")


if __name__ == "__main__":
    main()
