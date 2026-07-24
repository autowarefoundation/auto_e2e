"""Latency and resident-memory benchmark for the native navigation renderer."""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np

from .artifacts import decode_scene_navigation
from .contracts import NavigationMap, NavigationRoute
from .rasterizer import EgoPose, NativeNavigationRasterizer


DEFAULT_RENDER_P95_BUDGET_MS = 20.0
DEFAULT_RESIDENT_MEMORY_BUDGET_MB = 100.0


def _resident_memory_mb() -> float:
    maximum_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0
    return maximum_rss / divisor


def _latency_summary(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(values.max()),
        "mean_ms": float(values.mean()),
    }


@dataclasses.dataclass(frozen=True)
class NavigationBenchmarkResult:
    iterations: int
    warmup_iterations: int
    render: dict[str, float]
    warp: dict[str, float]
    resident_memory_increase_mb: float
    render_p95_budget_ms: float
    resident_memory_budget_mb: float
    passed: bool
    renderer_version: str
    geometry_id: str

    def to_json(self) -> str:
        return json.dumps(
            dataclasses.asdict(self),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )


def benchmark_navigation_renderer(
    rasterizer: NativeNavigationRasterizer,
    navigation_map: NavigationMap,
    route: NavigationRoute,
    *,
    iterations: int = 200,
    warmup_iterations: int = 20,
    render_p95_budget_ms: float = DEFAULT_RENDER_P95_BUDGET_MS,
    resident_memory_budget_mb: float = DEFAULT_RESIDENT_MEMORY_BUDGET_MB,
) -> NavigationBenchmarkResult:
    if iterations <= 0 or warmup_iterations < 0:
        raise ValueError("benchmark iteration counts are invalid")
    if render_p95_budget_ms <= 0.0 or resident_memory_budget_mb <= 0.0:
        raise ValueError("benchmark budgets must be positive")

    render_pose = EgoPose(0.0, 0.0, 0.0, 0)
    sample_pose = EgoPose(1.0, 0.2, 0.01, 100_000_000)
    for _ in range(warmup_iterations):
        anchor = rasterizer.render(navigation_map, route, render_pose)
        rasterizer.warp(anchor, sample_pose)

    memory_before = _resident_memory_mb()
    render_ms: list[float] = []
    warp_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        anchor = rasterizer.render(navigation_map, route, render_pose)
        render_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

        started = time.perf_counter_ns()
        rasterizer.warp(anchor, sample_pose)
        warp_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    memory_increase = max(0.0, _resident_memory_mb() - memory_before)

    render = _latency_summary(render_ms)
    warp = _latency_summary(warp_ms)
    passed = (
        render["p95_ms"] <= render_p95_budget_ms
        and memory_increase <= resident_memory_budget_mb
    )
    return NavigationBenchmarkResult(
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        render=render,
        warp=warp,
        resident_memory_increase_mb=memory_increase,
        render_p95_budget_ms=render_p95_budget_ms,
        resident_memory_budget_mb=resident_memory_budget_mb,
        passed=passed,
        renderer_version=rasterizer.renderer_version,
        geometry_id=rasterizer.geometry.geometry_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-navigation",
        type=Path,
        required=True,
        help="Path to scene_navigation.json",
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument(
        "--render-p95-budget-ms",
        type=float,
        default=DEFAULT_RENDER_P95_BUDGET_MS,
    )
    parser.add_argument(
        "--resident-memory-budget-mb",
        type=float,
        default=DEFAULT_RESIDENT_MEMORY_BUDGET_MB,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    navigation_map, route = decode_scene_navigation(
        args.scene_navigation.read_bytes()
    )
    result = benchmark_navigation_renderer(
        NativeNavigationRasterizer(library_path=args.library),
        navigation_map,
        route,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        render_p95_budget_ms=args.render_p95_budget_ms,
        resident_memory_budget_mb=args.resident_memory_budget_mb,
    )
    payload = result.to_json()
    if args.output is not None:
        args.output.write_text(payload + "\n")
    print(payload)
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
