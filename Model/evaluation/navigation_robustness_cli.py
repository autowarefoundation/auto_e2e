"""CLI: navigation-input robustness matrix (#157) on driving-scene rasters.

Default builds left/straight/right corridor scenes (KITScenes-like map/route
layout) and reports ADE/FDE under the issue's ablation matrix.

Packed shards::

    python -m evaluation.navigation_robustness_cli --shard-dir /path/to/partition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.navigation_robustness import (
    DEFAULT_MODES,
    build_corridor_scenes,
    route_follow_predict,
    run_navigation_robustness,
)


def _from_shard(shard_dir: Path):
    from data_parsing.pre_extracted import make_pre_extracted_loader
    from evaluation.metrics import integrate_trajectory

    loader = make_pre_extracted_loader(str(shard_dir), batch_size=4, num_workers=0, shuffle=0)
    raw = next(iter(loader))
    map_context = raw["map_context"].numpy()
    route_mask = raw["route_mask"].numpy()
    tgt = raw["trajectory_target"].numpy()
    b = tgt.shape[0]
    paired = tgt.reshape(b, -1, 2)
    gt = np.stack(
        [integrate_trajectory(paired[i, :, 0], paired[i, :, 1], 5.0) for i in range(b)],
        axis=0,
    )
    return map_context, route_mask, gt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", action="store_true", default=True,
                        help="Corridor driving scenes (default)")
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=157)
    parser.add_argument("--modes", nargs="*", default=list(DEFAULT_MODES))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.shard_dir is not None:
        map_context, route_mask, gt = _from_shard(args.shard_dir)
        source = "shard"
    else:
        map_context, route_mask, gt = build_corridor_scenes()
        source = "corridor_scenes"

    def predict(m, r):
        return route_follow_predict(m, r, gt)

    report = run_navigation_robustness(
        map_context, route_mask, gt, predict, modes=tuple(args.modes), seed=args.seed
    )
    payload = report.to_dict()
    payload["source"] = source
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
