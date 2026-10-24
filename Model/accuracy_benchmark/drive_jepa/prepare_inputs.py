"""Verify and unpack audited v3.5 commands without running a baseline model."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .external_kitscenes import EXPECTED_VAL_SAMPLE_UID_SHA256, published_digest_ids
from .paths import WORKSPACE


def prepare(workspace=WORKSPACE):
    assets = Path(__file__).with_name("inputs")
    population = json.loads(
        gzip.decompress((assets / "population.json.gz").read_bytes())
    )
    timestamps = {
        uid: timestamp
        for scene in population.values()
        for uid, timestamp in zip(
            scene["sample_uid"], scene["timestamp_ns"], strict=True
        )
    }
    if len(population) != 117 or len(timestamps) != 11035:
        raise ValueError("Bundled population is incomplete")
    if published_digest_ids(timestamps) != EXPECTED_VAL_SAMPLE_UID_SHA256:
        raise ValueError("Bundled population UID digest differs")
    baseline = workspace / "population"
    baseline.mkdir(parents=True, exist_ok=True)
    for scene, data in population.items():
        # Identity records only: no images, future poses, targets or predictions.
        np.savez_compressed(
            baseline / f"{scene}.npz",
            sample_uid=np.array(data["sample_uid"]),
            timestamp_ns=np.array(data["timestamp_ns"], dtype=np.int64),
        )
    for policy, directory in (("20m", "route_inputs_20m"), ("100m", "route_inputs")):
        payload = gzip.decompress((assets / f"commands_{policy}.json.gz").read_bytes())
        provenance = json.loads(
            (assets / f"commands_{policy}.provenance.json").read_text()
        )
        if hashlib.sha256(payload).hexdigest() != provenance["commands_sha256"]:
            raise ValueError(f"{policy} command checksum differs")
        commands = json.loads(payload)
        if set(commands) != set(timestamps) or any(
            len(v) != 4 or set(v) - {0.0, 1.0} or sum(v) != 1 for v in commands.values()
        ):
            raise ValueError(f"Invalid {policy} command population or encoding")
        output = workspace / directory
        output.mkdir(parents=True, exist_ok=True)
        # Never replace a user-provided or newly regenerated command file.
        for path, content in (
            (output / "commands.json", payload),
            (
                output / "commands.provenance.json",
                (assets / f"commands_{policy}.provenance.json").read_bytes(),
            ),
        ):
            if path.exists() and path.read_bytes() != content:
                raise ValueError(
                    f"Existing file differs; choose a new workspace: {path}"
                )
            path.write_bytes(content)
    print(f"Verified 117 scenes / 11035 samples; inputs in {workspace}")
    return baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    prepare(parser.parse_args().workspace)


if __name__ == "__main__":
    main()
