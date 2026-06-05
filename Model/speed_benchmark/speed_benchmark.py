import torch
import time
import sys
import json
import numpy as np
from datetime import datetime
sys.path.append('..')
from model_components.auto_e2e import AutoE2E


def run_speed_benchmark(backbone, fusion_mode, device, batch_size=1, num_views=8):
    
    print(f"{'='*80}")
    print(f"  backbone = '{backbone}' | fusion_mode = '{fusion_mode}' | batch={batch_size} | views={num_views}")
    print(f"{'='*80}\n")

    # Instantiate model
    model = AutoE2E(backbone=backbone, num_views=num_views, fusion_mode=fusion_mode)
    model = model.to(device)
    model.eval()

    # Visual Scene Input: [batch, num_views, channels, height, width]
    visual_tiles = torch.randn(batch_size, num_views, 3, 256, 256).to(device)

    # Egomotion History Input: [batch, 256]
    egomotion_history = torch.randn(batch_size, 256).to(device)

    # Visual Scene History: [batch, 896]
    visual_history = torch.randn(batch_size, 896).to(device)

    # Camera parameters: [batch, num_views, 3, 4] projection matrices
    # Only used by BEV fusion; None triggers learnable pseudo-projection
    camera_params = None
    if fusion_mode == "bev":
        camera_params = torch.randn(batch_size, num_views, 3, 4).to(device)

    # 1. Warm-up Phase (GPU kernel compilation and cache warming)
    num_warmup = 30 if device.type == 'cuda' else 5
    print(f"Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(visual_tiles, visual_history, egomotion_history,
                       backbone=backbone, camera_params=camera_params, mode="infer")

    # 2. Benchmark Phase
    num_iters = 100 if device.type == 'cuda' else 10
    print(f"Benchmarking ({num_iters} iterations)...")

    latencies = []

    with torch.no_grad():
        for _ in range(num_iters):

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            _ = model(visual_tiles, visual_history, egomotion_history, 
                      backbone=backbone, camera_params=camera_params, mode="infer") # we discard the output

            torch.cuda.synchronize()
            # Record individual frame processing times in milliseconds
            latencies.append((time.perf_counter() - start_time) * 1000)

    latencies = np.array(latencies)

    # 3. Calculate and Print Metrics
    avg_fps = 1000 / np.mean(latencies)
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p99_latency = np.percentile(latencies, 99)
    jitter = p99_latency - p50_latency

    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        "backbone": backbone,
        "fusion_mode": fusion_mode,
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


def save_results_json(all_results, device):
    """Save benchmark results to a JSON file with hardware metadata."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "results": all_results,
    }
    filepath = "benchmark_results.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {filepath}")


def print_markdown_table(all_results):
    """Print results as a Markdown table for easy pasting into README."""
    print("\n## Benchmark Results\n")
    print("| Backbone | Fusion Mode | Batch | FPS | Latency (ms) | p99 (ms) | VRAM (MB) | Params |")
    print("|----------|-------------|-------|-----|--------------|----------|-----------|--------|")
    for r in all_results:
        params_m = r["total_params"] / 1_000_000
        print(f"| {r['backbone']} | {r['fusion_mode']} | {r['batch_size']} | "
              f"{r['avg_fps']:.1f} | {r['avg_latency_ms']:.1f} | {r['p99_latency_ms']:.1f} | "
              f"{r['peak_vram_allocated_mb']:.0f} | {params_m:.1f}M |")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference\n')

    all_results = []

    # Test all registered backbones and fusion modes
    backbones = ["swin_v2_tiny", "conv_next_v2_tiny"]
    fusion_modes = ["concat", "cross_attn", "bev"]
    batch_sizes = [1, 2, 4]

    for backbone in backbones:
        for fusion_mode in fusion_modes:
            for batch_size in batch_sizes:
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                result = run_speed_benchmark(backbone, fusion_mode, device, batch_size=batch_size)
                all_results.append(result)
                print()

    # Save structured results
    save_results_json(all_results, device)

    # Print Markdown table for README
    print_markdown_table(all_results)


if __name__ == "__main__":
    main()
