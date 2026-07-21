#!/usr/bin/env python3
"""Collect wf_create_dataset_sharded shard-dir outputs across 17 batch executions.

Reads each batch's `state_dir/batch_N.exec` file (recorded by
launch_17batch_fullrun.sh) which holds the Flyte execution ID. Uses
FlyteRemote to fetch each execution's output — a `List[FlyteDirectory]` —
and flattens all of them into a single input JSON suitable for
`pyflyte run --inputs-file` on `wf_train_il`.

Output JSON schema (matches wf_train_il inputs):
    { "shards": [<s3-uri>, <s3-uri>, ...] }
"""

import argparse
import json
import pathlib
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True,
                    help="Directory holding batch_*.exec files")
    ap.add_argument("--project", default="auto-e2e")
    ap.add_argument("--domain", default="development")
    ap.add_argument("--output", required=True,
                    help="Path to write the merged shard-list JSON")
    args = ap.parse_args()

    from flytekit.remote import FlyteRemote
    from flytekit.configuration import Config

    remote = FlyteRemote(
        config=Config.auto(),
        default_project=args.project,
        default_domain=args.domain,
    )

    state_dir = pathlib.Path(args.state_dir)
    exec_files = sorted(state_dir.glob("batch_*.exec"))
    if not exec_files:
        raise SystemExit(f"No batch_*.exec files in {state_dir}")

    all_shards = []
    for ef in exec_files:
        exec_id = ef.read_text().strip()
        if not exec_id:
            print(f"WARN: {ef.name} is empty — skipping", file=sys.stderr)
            continue
        print(f"Fetching outputs of {exec_id} ({ef.name}) ...")
        execution = remote.fetch_execution(name=exec_id)
        if execution.closure.phase != 4:  # 4 = SUCCEEDED
            raise SystemExit(
                f"{exec_id} not in SUCCEEDED phase ({execution.closure.phase}). "
                f"Refuse to build training input from an incomplete batch.")
        outputs = execution.outputs
        # wf_create_dataset_sharded returns a bare List[FlyteDirectory] with
        # the default output key "o0" when invoked as a workflow.
        shard_dirs = outputs.get("o0", outputs)
        # shard_dirs is a list of FlyteDirectory-like objects; extract their
        # remote URIs. flytekit's FlyteDirectory has a `.remote_source`.
        for sd in shard_dirs:
            uri = getattr(sd, "remote_source", None) or getattr(sd, "path", None) or str(sd)
            all_shards.append(uri)
        print(f"  → {len(shard_dirs)} shard dirs from batch {ef.stem}")

    print(f"Total shard dirs across {len(exec_files)} batches: {len(all_shards)}")

    payload = {"shards": all_shards}
    pathlib.Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"Wrote merged shard list to {args.output}")


if __name__ == "__main__":
    main()
