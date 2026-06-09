#!/usr/bin/env python3
"""Read benchmark result JSON files and generate a README markdown table."""

import json
import sys
from pathlib import Path


def load_results(results_dir):
    """Load all JSON result files, grouped by GPU name."""
    results_dir = Path(results_dir)
    gpu_groups = {}

    for filepath in sorted(results_dir.glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)

        gpu_name = data.get("gpu_name", "Unknown GPU")
        if gpu_name not in gpu_groups:
            gpu_groups[gpu_name] = []
        gpu_groups[gpu_name].append(data)

    return gpu_groups


def format_metadata_line(data):
    """Format environment metadata as a single line."""
    parts = [
        f"CUDA {data.get('cuda_version', 'N/A')}",
        f"Driver {data.get('driver_version', 'N/A')}",
        f"PyTorch {data.get('pytorch_version', 'N/A')}",
        f"Commit `{data.get('commit_sha', 'unknown')}`",
        f"Resolution {data.get('input_resolution', [256, 256])}",
    ]
    return " | ".join(parts)


def generate_markdown(gpu_groups):
    """Generate markdown benchmark section."""
    lines = []
    lines.append("## Benchmark Results\n")

    for gpu_name, runs in gpu_groups.items():
        lines.append(f"### {gpu_name}\n")

        latest = runs[-1]
        lines.append(f"> {format_metadata_line(latest)}\n")

        lines.append(
            "| Backbone | Fusion Mode | Batch | FPS | Latency (ms) "
            "| p99 (ms) | VRAM (MB) | Params |"
        )
        lines.append(
            "|----------|-------------|-------|-----|----------"
            "----|----------|-----------|--------|"
        )

        for r in latest["results"]:
            params_m = r["total_params"] / 1_000_000
            lines.append(
                f"| {r['backbone']} | {r['fusion_mode']} | {r['batch_size']} | "
                f"{r['avg_fps']:.1f} | {r['avg_latency_ms']:.1f} | "
                f"{r['p99_latency_ms']:.1f} | "
                f"{r['peak_vram_allocated_mb']:.0f} | {params_m:.1f}M |"
            )

        lines.append("")

    return "\n".join(lines)


def main():
    results_dir = Path(__file__).parent / "results"

    if not results_dir.exists():
        print("Error: results/ directory not found.", file=sys.stderr)
        sys.exit(1)

    gpu_groups = load_results(results_dir)

    if not gpu_groups:
        print("Error: no JSON result files found in results/.", file=sys.stderr)
        sys.exit(1)

    # Only the latest run per GPU is displayed in the generated table.
    markdown = generate_markdown(gpu_groups)
    print(markdown)


if __name__ == "__main__":
    main()
