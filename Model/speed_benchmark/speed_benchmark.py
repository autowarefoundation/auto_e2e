import torch
import time
import sys
import numpy as np
sys.path.append('..')
from model_components.auto_e2e import AutoE2E


def run_speed_benchmark(fusion_mode, device, batch_size=1, num_views=8):
    
    print(f"{'='*60}")
    print(f"  fusion_mode = '{fusion_mode}' | batch={batch_size} | views={num_views}")
    print(f"{'='*60}\n")

    # Instantiate model
    model = AutoE2E(num_views=num_views, fusion_mode=fusion_mode)
    model = model.to(device)
    model.eval()

    # Visual Scene Input: [batch, num_views, channels, height, width]
    visual_tiles = torch.randn(batch_size, num_views, 3, 224, 224).to(device)

    # Egomotion History Input: [batch, 256]
    egomotion_history = torch.randn(batch_size, 256).to(device)

    # Visual Scene History: [batch, 896]
    visual_history = torch.randn(batch_size, 896).to(device)

    # Camera parameters: [batch, num_views, 3, 4] projection matrices
    # Only used by BEV fusion; None triggers learnable pseudo-projection
    camera_params = None
    if fusion_mode == "bev":
        camera_params = torch.randn(batch_size, num_views, 3, 4).to(device)

    # 1. Warm-up Phase
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(30):
            _ = model(visual_tiles, visual_history, egomotion_history, 
                      camera_params=camera_params) # we discard the output

    # 2. Benchmark Phase
    print("Benchmarking now ...")
    num_iters = 100

    latencies = []

    with torch.no_grad():
        for _ in range(num_iters):

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            _ = model(visual_tiles, visual_history, egomotion_history, 
                      camera_params=camera_params) # we discard the output

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

    print("======================")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average Latency: {avg_latency:.2f}")
    print(f"Worst-Case Latency (p99): {p99_latency:.2f} ms")
    print(f"Latency Jitter (p99 - p50): {jitter:.2f} ms")
    print("----------------------")
    print(f"Peak VRAM Allocated: {peak_allocated:.2f} MB")
    print(f"Peak VRAM Reserved: {peak_reserved:.2f} MB")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference\n')

    # Test all registered fusion modes
    run_speed_benchmark("concat", device)
    run_speed_benchmark("cross_attn", device)
    run_speed_benchmark("bev", device)


if __name__ == "__main__":
    main()
