import argparse
import subprocess
import torch
import time
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
sys.path.append('..')
from model_components.auto_e2e import AutoE2E
from model_components.view_fusion import PinholeProjection


def run_speed_benchmark(backbone, device, batch_size=1, num_views=7,
                        reasoning="off"):

    print(f"{'='*80}")
    print(f"  backbone = '{backbone}' | fusion = 'bev' | batch={batch_size} "
          f"| views={num_views} | reasoning = '{reasoning}'")
    print(f"{'='*80}\n")

    # Instantiate model. Fusion is always BEV (concat / cross_attn and the
    # fusion_mode knob were removed); nav-map is a separate map_input branch, not
    # a camera view. Small BEV grid keeps the benchmark fast.
    #
    # ``reasoning`` optionally enables the Reasoning band (#98) so its inference
    # cost is measured explicitly.  The band runs at 1 Hz in deployment, so the
    # per-forward cost here is an upper bound on its real-time budget.
    extra_kwargs = {}
    if reasoning != "off":
        # ReasoningBand takes no "backbone" argument (only visual_history_dim,
        # hidden_dim, num_future_horizons, taxonomy); enable the band with its
        # defaults. ``reasoning`` is kept purely as the result label/column.
        extra_kwargs = dict(enable_reasoning_band=True)
    model = AutoE2E(backbone=backbone, num_views=num_views,
                    view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
                    **extra_kwargs)
    model = model.to(device)
    model.eval()

    # Visual Scene Input: [batch, num_views, channels, height, width]
    visual_tiles = torch.randn(batch_size, num_views, 3, 256, 256).to(device)
    # Map Input: [batch, channels, height, width]
    map_input = torch.randn(batch_size, 3, 256, 256).to(device)
    # Visual History Input: [batch, 896]
    visual_history = torch.randn(batch_size, 896).to(device)
    # Egomotion History Input: [batch, 256]
    egomotion_history = torch.randn(batch_size, 256).to(device)

    # Geometry: a pinhole projection operator (the real-calibration path; there
    # is no camera_params matrix argument on forward).
    projection = PinholeProjection(torch.randn(batch_size, num_views, 3, 4).to(device))

    def _forward():
        return model(visual_tiles, map_input, visual_history, egomotion_history,
                     projection=projection, geometry_type="pinhole", mode="infer")

    # 1. Warm-up Phase (GPU kernel compilation and cache warming)
    num_warmup = 30 if device.type == 'cuda' else 5
    print(f"Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = _forward()

    # 2. Benchmark Phase
    num_iters = 100 if device.type == 'cuda' else 10
    print(f"Benchmarking ({num_iters} iterations)...")

    latencies = []

    with torch.no_grad():
        for _ in range(num_iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            _ = _forward()

            if device.type == 'cuda':
                torch.cuda.synchronize()

            latencies.append((time.perf_counter() - start_time) * 1000)

    latencies = np.array(latencies)

    # 3. Calculate and Print Metrics
    avg_fps = 1000 / np.mean(latencies)
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p99_latency = np.percentile(latencies, 99)
    jitter = p99_latency - p50_latency

    if device.type == 'cuda':
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    else:
        peak_allocated = 0.0
        peak_reserved = 0.0

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        "backbone": backbone,
        "fusion_mode": "bev",
        "reasoning_band": reasoning,
        "batch_size": batch_size,
        "num_views": num_views,
        "avg_fps": round(avg_fps, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p99_latency_ms": round(p99_latency, 2),
        "jitter_ms": round(jitter, 2),
        "peak_vram_allocated_mb": round(peak_allocated, 2),
        "peak_vram_reserved_mb": round(peak_reserved, 2),
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    print("======================")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Worst-Case Latency (p99): {p99_latency:.2f} ms")
    print(f"Latency Jitter (p99 - p50): {jitter:.2f} ms")
    print("----------------------")
    print(f"Peak VRAM Allocated: {peak_allocated:.2f} MB")
    print(f"Peak VRAM Reserved: {peak_reserved:.2f} MB")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    return results


def get_commit_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_driver_version():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return output.split("\n")[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"


def save_results_json(all_results, device, input_resolution=(256, 256)):
    """Save benchmark results to a JSON file with hardware metadata."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "driver_version": get_driver_version(),
        "pytorch_version": torch.__version__,
        "commit_sha": get_commit_sha(),
        "input_resolution": list(input_resolution),
        "results": all_results,
    }
    gpu_slug = output["gpu_name"].replace(" ", "_").replace("/", "-").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    filepath = results_dir / f"{gpu_slug}_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {filepath}")


def print_markdown_table(all_results):
    """Print results as a Markdown table for easy pasting into README."""
    print("\n## Benchmark Results\n")
    print("| Backbone | Fusion Mode | Reasoning | Batch | FPS | Latency (ms) | p99 (ms) | VRAM (MB) | Params |")
    print("|----------|-------------|-----------|-------|-----|--------------|----------|-----------|--------|")
    for r in all_results:
        params_m = r["total_params"] / 1_000_000
        print(f"| {r['backbone']} | {r['fusion_mode']} | {r.get('reasoning_band', 'off')} | {r['batch_size']} | "
              f"{r['avg_fps']:.1f} | {r['avg_latency_ms']:.1f} | {r['p99_latency_ms']:.1f} | "
              f"{r['peak_vram_allocated_mb']:.0f} | {params_m:.1f}M |")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoE2E speed benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference\n')

    all_results = []

    # Test all registered backbones (BEV is the only fusion after the refactor).
    backbones = ["swin_v2_tiny", "conv_next_v2_tiny"]
    batch_sizes = [1, 2, 4]

    for backbone in backbones:
        for batch_size in batch_sizes:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            result = run_speed_benchmark(backbone, device, batch_size=batch_size)
            all_results.append(result)
            print()

    # Reasoning band (#98): measure the added cost of the branch explicitly,
    # on the default backbone at batch 1 (its deployment operating point —
    # the band runs at 1 Hz).
    reasoning_variants = ["head_on_896"]
    for reasoning in reasoning_variants:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = run_speed_benchmark(backbones[0], device, batch_size=1,
                                     reasoning=reasoning)
        all_results.append(result)
        print()

    # Save structured results
    save_results_json(all_results, device)

    # Print Markdown table for README
    print_markdown_table(all_results)


if __name__ == "__main__":
    main()
