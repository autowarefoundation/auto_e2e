# SPDX-License-Identifier: Apache-2.0
"""Fetch pinned official model assets and the original shared metric source."""

import argparse
from pathlib import Path
from urllib.request import urlopen

from .common import ASSET_REPO, ASSET_REVISION, ASSETS, sha256, verified_download
from .shared_metrics import SOURCE_COMMIT, SOURCE_SHA256, load_metric_code, self_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        path = verified_download(ASSET_REPO, ASSET_REVISION, name, args.out)
        if sha256(path) != expected:
            raise ValueError("Official asset hash mismatch: " + name)
        print("Verified", name, flush=True)
    path = args.out / "reactive_stage_runner.py"
    url = (
        "https://raw.githubusercontent.com/autowarefoundation/auto_e2e/"
        + SOURCE_COMMIT
        + "/Model/training/reactive_stage_runner.py"
    )
    if not path.exists():
        import hashlib

        with urlopen(url, timeout=60) as response:
            source = response.read(8 * 2**20 + 1)
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
            raise ValueError("Shared evaluator hash mismatch")
        with path.open("xb") as handle:
            handle.write(source)
    self_test(load_metric_code(path))
    print("Pinned shared metric conformance passed")


if __name__ == "__main__":
    main()
