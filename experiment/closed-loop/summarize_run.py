"""Per-scene summary of one closed-loop run.

AlpaSim's aggregate table averages over scenes and trims timesteps after the
first incident; this prints one row per rollout from the untrimmed
``metrics.parquet`` so failures can be traced to a scene and a time.

Run with AlpaSim's environment, which has polars:
    cd $ALPASIM_ROOT && uv run --no-sync python \
        <repo>/experiment/closed-loop/summarize_run.py ~/closed-loop-runs/<run>
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

# Metric name -> how one rollout's time series is reduced.
REDUCTIONS = {
    "collision_any": "max",
    "offroad": "max",
    "wrong_lane": "max",
    "dist_to_gt_trajectory": "max",
    "dist_traveled_m": "last",
    "gt_dist_traveled_m": "last",
    "progress_rel_to_total": "last",
}
# Incidents whose first occurrence is reported, in seconds after the driver
# takes over (force_gt_duration_us in config.env's driver config).
FIRST_EVENTS = ("collision_any", "offroad", "wrong_lane")
FORCE_GT_S = 7.0


def summarize(path: Path) -> dict[str, object]:
    df = pl.read_parquet(path).filter(pl.col("valid"))
    t0 = df["timestamps_us"].min()
    row: dict[str, object] = {"scene": path.parents[1].name.removeprefix("clipgt-")[:8]}
    for name, how in REDUCTIONS.items():
        series = df.filter(pl.col("name") == name).sort("timestamps_us")["values"]
        if series.is_empty():
            row[name] = None
        else:
            row[name] = round(float(series.max() if how == "max" else series[-1]), 2)
    for name in FIRST_EVENTS:
        hits = df.filter((pl.col("name") == name) & (pl.col("values") > 0))
        first = hits["timestamps_us"].min()
        row[f"first_{name}_s"] = (
            None if first is None else round((first - t0) / 1e6 - FORCE_GT_S, 1)
        )
    return row


def main() -> None:
    run_dir = Path(sys.argv[1])
    rows = [summarize(p) for p in sorted(run_dir.glob("rollouts/*/*/metrics.parquet"))]
    if not rows:
        sys.exit(f"no metrics.parquet under {run_dir}/rollouts")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, tbl_hide_dataframe_shape=True):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
