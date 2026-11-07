#!/usr/bin/env python3
"""Launch wf_autoe2e or wf_acquire_raw_snapshot from a JSON run file.

The run file holds account-specific values (buckets, manifest URIs), so it is
kept out of git. Every input is explicit, so an execution page shows exactly
what the run used.

    python Platform/scripts/launch_autoe2e.py --endpoint localhost:8089 \
        --version <flyte version> --run runs/smoke.json [--execute]
"""

from __future__ import annotations

import argparse
import json
import sys


def _dataset_spec(payload: dict):
    from Platform.pipelines.autoe2e_dataset import AutoE2EDataset, DatasetSpec, ManifestRef, ReasoningSpec

    reasoning = payload.get("reasoning")
    return DatasetSpec(
        dataset=AutoE2EDataset(payload["dataset"]),
        source_manifest=ManifestRef(**payload["source_manifest"]),
        pack_version=payload["pack_version"],
        image_size=int(payload["image_size"]),
        unit_limit=int(payload.get("unit_limit", 0)),
        pack_params=dict(payload.get("pack_params", {})),
        reasoning=None if reasoning is None else ReasoningSpec(**reasoning),
    )


def build_inputs(run: dict) -> tuple[str, dict]:
    if run["workflow"] == "wf_acquire_raw_snapshot":
        from Platform.pipelines.autoe2e_dataset import AutoE2EDataset, ManifestRef, ReasoningSpec

        inputs = dict(run["inputs"])
        inputs["dataset"] = AutoE2EDataset(inputs["dataset"])
        if inputs.get("recovery_manifest"):
            inputs["recovery_manifest"] = ManifestRef(**inputs["recovery_manifest"])
        if inputs.get("adopt_labels"):
            inputs["adopt_labels"] = ReasoningSpec(**inputs["adopt_labels"])
        return "Platform.pipelines.autoe2e_acquire.wf_acquire_raw_snapshot", inputs
    if run["workflow"] != "wf_autoe2e":
        raise ValueError(f"unknown workflow {run['workflow']!r}")
    from Platform.pipelines.autoe2e import EvalSuite, EvalTarget, EvaluationSpec, ModelRef, RuntimeOptions
    from Platform.pipelines.autoe2e_training import GpuProfile, TrainingSpec

    training = dict(run["training"])
    training["gpu_profile"] = GpuProfile(training.get("gpu_profile", GpuProfile.RAY_8.value))
    inputs = {
        "dataset": _dataset_spec(run["dataset"]),
        "mix": [_dataset_spec(spec) for spec in run.get("mix", [])],
        "early_mix": [_dataset_spec(spec) for spec in run.get("early_mix", [])],
        "training": TrainingSpec(**training),
        "evaluation": EvaluationSpec(targets=[EvalTarget(EvalSuite(suite)) for suite in run.get("evaluation", [])]),
        "runtime": RuntimeOptions(**run["runtime"]),
    }
    if run.get("parent"):
        inputs["parent"] = ModelRef(**run["parent"])
    return "Platform.pipelines.autoe2e.wf_autoe2e", inputs


def _type_hints(name: str) -> dict:
    import importlib
    import typing

    module_name, _, function = name.rpartition(".")
    workflow = getattr(importlib.import_module(module_name), function)
    return typing.get_type_hints(workflow._workflow_function)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="flyteadmin gRPC endpoint, e.g. localhost:8089")
    parser.add_argument("--version", required=True)
    parser.add_argument("--run", required=True, help="JSON run file")
    parser.add_argument("--project", default="auto-e2e")
    parser.add_argument("--domain", default="development")
    parser.add_argument("--name", default=None, help="execution name")
    parser.add_argument("--execute", action="store_true", help="launch; default prints inputs only")
    args = parser.parse_args(argv)
    with open(args.run, encoding="utf-8") as handle:
        run = json.load(handle)
    name, inputs = build_inputs(run)
    print(json.dumps({"workflow": name, "inputs": {k: repr(v) for k, v in inputs.items()}}, indent=1))
    if not args.execute:
        return 0
    from flytekit.configuration import Config, PlatformConfig
    from flytekit.remote import FlyteRemote

    remote = FlyteRemote(
        config=Config(platform=PlatformConfig(endpoint=args.endpoint, insecure=True)),
        default_project=args.project, default_domain=args.domain,
    )
    workflow = remote.fetch_workflow(name=name, version=args.version)
    # The fetched workflow only knows guessed types; hand it the real ones.
    type_hints = _type_hints(name)
    execution = remote.execute(
        workflow, inputs=inputs, execution_name=args.name, wait=False,
        type_hints={key: type_hints[key] for key in inputs},
    )
    print(json.dumps({"execution": execution.id.name}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
