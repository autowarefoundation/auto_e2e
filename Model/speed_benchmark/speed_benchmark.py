import torch
import time
import sys
sys.path.append('..')
from model_components.auto_e2e import AutoE2E
import numpy as np

def main():
    # Device for benchmarking
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for benchmarking \n')

    # Instantiate model
    model = AutoE2E().to(device)

    # Dummy Visual Scene Input
    # 7 cameras + 1 map tile - in batch dimension
    # giving 8 effective visual inputs assuming batch
    # size of 1
    visual_tiles = torch.randn(8, 3, 224, 224).to(device)

    # Egomotion History Input
    # Speed, Acceleration, Yaw Angle, Yaw Rate for
    # 6.4s past history giving 64 x 4 samples at 10Hz
    egomotion_history = torch.randn(256).to(device)

    # Dummy Visual Scene History
    # Length 14 compressed visual feature vector at 10Hz
    # for 6.4s past horizon giving 64 x 14 samples
    visual_history = torch.randn(896).to(device)

    # 1. Warm-up Phase
    print("Warming up GPU...")
    for _ in range(30):
        _ = model(visual_tiles, visual_history, egomotion_history) # we discard the output

    # 2. Benchmark Phase
    print("Benchmarking now ...")
    num_iters = 100

    latencies = []

    for _ in range(num_iters):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        _ = model(visual_tiles, visual_history, egomotion_history) # we discard the output

        torch.cuda.synchronize()
        # Record individual frame processing times in milliseconds
        latencies.append((time.perf_counter() - start_time) * 1000)

    latencies = np.array(latencies)

    # 3. Calculate and Print Metrics
    avg_fps = 1000 / np.mean(latencies)
    avg_latency = np.mean(latencies)
    p1_latency = np.percentile(latencies, 1)
    p99_latency = np.percentile(latencies, 99)
    jitter = p99_latency - p1_latency

    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    print("======================")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average Latency: {avg_latency:.2f}")
    print(f"Worst-Case Latency (p99): {p99_latency:.2f} ms")
    print(f"Latency Jitter (p99 - p1): {jitter:.2f} ms")
    print("----------------------")
    print(f"Peak VRAM Allocated: {peak_allocated:.2f} MB")
    print(f"Peak VRAM Reserved: {peak_reserved:.2f} MB")

if __name__ == "__main__":
    main()