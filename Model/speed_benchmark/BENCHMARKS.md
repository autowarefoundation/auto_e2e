# Inference Speed Benchmarks

Per-GPU inference benchmarks for AutoE2E. To add results for your own GPU, run the
[benchmarking script](./) in this folder — it documents the meaning of each benchmark
parameter.

## NVIDIA GeForce RTX 3060 Laptop GPU

<details open>
  <summary>Toggle view</summary>


| Model | Backbone | Fusion Method | FPS | Average Latency [ms] | Worst-Case Latency [ms] | Latency Jitter [ms] | Peak VRAM Allocated [MB] | Peak VRAM Reserved [MB] |
|-------| -------- | ------------- | --- | --------------- | ------------------ | -------------- | ------------------- | ------------------ |
| Reactive | SwinV2 Tiny | Feature Concat | 24.99 | 40.01 | 40.68 | 0.71 | 1067.52 | 1216.00 |
| Reactive | SwinV2 Tiny | Spatial Attention | 24.48 | 44.49 | 47.23 | 2.75 | 1069.18 | 1218.00 |
| Reactive | SwinV2 Tiny | BEV Fusion | 22.02 | 45.42 | 67.72 | 23.87 | 1069.18 | 1220.00 |
| Reactive | ConvNextV2 Tiny | Feature Concat | 22.99 | 43.49 | 49.23 | 7.26 | 1092.58 | 1268.00 |
| Reactive | ConvNextV2 Tiny | Spatial Attention | 18.60 | 53.75 | 54.15 | 0.36 | 1092.58 | 1268.00 |
| Reactive | ConvNextV2 Tiny | BEV Fusion | 18.63 | 53.69 | 54.37 | 0.67 | 1092.58 | 1268.00 |

</details>

## NVIDIA GeForce RTX 4050 Laptop GPU

<details open>
  <summary>Toggle view</summary>

> CUDA 12.6 | Driver 610.62 | PyTorch 2.7.1+cu126 | Commit `82cc8ae` | Resolution [256, 256]

| Model | Backbone | Fusion Mode | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|----------|----------|-------------|-------|-----|--------------|----------|-------------|-----------|--------|
| Reactive | swin_v2_tiny | bev | 1 | 20.0 | 50.0 | 72.4 | 25.2 | 417 | 62.8M |
| Combined | swin_v2_tiny | bev | 1 | 12.2 | 82.0 | 86.4 | 4.7 | 556 | 98.3M |
| Reactive | swin_v2_tiny | bev | 2 | 11.3 | 88.1 | 91.7 | 4.0 | 580 | 62.8M |
| Combined | swin_v2_tiny | bev | 2 | 6.2 | 161.8 | 167.6 | 6.0 | 719 | 98.3M |
| Reactive | swin_v2_tiny | bev | 4 | 5.6 | 178.0 | 181.8 | 4.0 | 905 | 62.8M |
| Combined | swin_v2_tiny | bev | 4 | 3.0 | 331.4 | 335.6 | 4.2 | 1045 | 98.3M |
| Reactive | conv_next_v2_tiny | bev | 1 | 19.6 | 50.9 | 68.7 | 19.8 | 443 | 63.1M |
| Combined | conv_next_v2_tiny | bev | 1 | 11.3 | 88.8 | 94.1 | 6.3 | 579 | 98.8M |
| Reactive | conv_next_v2_tiny | bev | 2 | 10.4 | 96.0 | 104.6 | 9.1 | 630 | 63.1M |
| Combined | conv_next_v2_tiny | bev | 2 | 5.7 | 174.6 | 179.9 | 5.6 | 767 | 98.8M |
| Reactive | conv_next_v2_tiny | bev | 4 | 5.3 | 188.7 | 192.0 | 3.5 | 1003 | 63.1M |
| Combined | conv_next_v2_tiny | bev | 4 | 2.8 | 353.2 | 364.9 | 12.1 | 1140 | 98.8M |

</details>

## NVIDIA GeForce RTX 5080 GPU

<details open>
  <summary>Toggle view</summary>

> CUDA 12.8 | Driver 595.71.05 | PyTorch 2.11.0+cu128 | Commit `9015914` | Resolution [256, 256]

| Model | Backbone | Fusion Mode | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|-------|----------|-------------|-------|-----|--------------|----------|-------------|-----------|--------|
| Reactive | swin_v2_tiny | concat | 1 | 76.5 | 13.1 | 13.6 | 0.6 | 308 | 35.3M |
| Reactive | swin_v2_tiny | concat | 2 | 46.8 | 21.4 | 21.7 | 0.3 | 473 | 35.3M |
| Reactive | swin_v2_tiny | concat | 4 | 25.2 | 39.7 | 40.4 | 0.6 | 797 | 35.3M |
| Reactive | swin_v2_tiny | cross_attn | 1 | 75.6 | 13.2 | 13.7 | 0.5 | 311 | 35.3M |
| Reactive | swin_v2_tiny | cross_attn | 2 | 46.6 | 21.5 | 21.9 | 0.4 | 473 | 35.3M |
| Reactive | swin_v2_tiny | cross_attn | 4 | 25.1 | 39.9 | 40.5 | 0.6 | 797 | 35.3M |
| Reactive | swin_v2_tiny | bev | 1 | 16.3 | 61.5 | 62.0 | 0.5 | 1820 | 69.7M |
| Reactive | swin_v2_tiny | bev | 2 | 8.2 | 121.6 | 122.4 | 0.8 | 3354 | 69.7M |
| Reactive | swin_v2_tiny | bev | 4 | 4.2 | 239.6 | 240.8 | 1.1 | 6421 | 69.7M |
| Reactive | conv_next_v2_tiny | concat | 1 | 74.2 | 13.5 | 14.2 | 0.7 | 334 | 35.6M |
| Reactive | conv_next_v2_tiny | concat | 2 | 42.7 | 23.4 | 23.9 | 0.5 | 520 | 35.6M |
| Reactive | conv_next_v2_tiny | concat | 4 | 22.7 | 44.0 | 44.7 | 0.6 | 892 | 35.6M |
| Reactive | conv_next_v2_tiny | cross_attn | 1 | 73.8 | 13.6 | 14.1 | 0.6 | 333 | 35.6M |
| Reactive | conv_next_v2_tiny | cross_attn | 2 | 42.3 | 23.6 | 24.9 | 1.3 | 519 | 35.6M |
| Reactive | conv_next_v2_tiny | cross_attn | 4 | 22.6 | 44.1 | 44.7 | 0.6 | 891 | 35.6M |
| Reactive | conv_next_v2_tiny | bev | 1 | 16.2 | 61.9 | 62.5 | 0.7 | 1820 | 70.0M |
| Reactive | conv_next_v2_tiny | bev | 2 | 8.1 | 122.8 | 123.8 | 0.9 | 3351 | 70.0M |
| Reactive | conv_next_v2_tiny | bev | 4 | 4.1 | 243.1 | 244.0 | 0.9 | 6419 | 70.0M |

</details>

## NVIDIA RTX A6000 GPU

<details open>
  <summary>Toggle view</summary>

> CUDA 11.8 | Driver 580.159.03 | PyTorch 2.4.1+cu118 | Commit `9015914` | Resolution [256, 256]

| Model | Backbone | Fusion Mode | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|-------|----------|-------------|-------|-----|--------------|----------|-------------|-----------|--------|
| Reactive | swin_v2_tiny | concat | 1 | 28.2 | 35.4 | 35.9 | 0.6 | 307 | 35.3M |
| Reactive | swin_v2_tiny | concat | 2 | 27.4 | 36.5 | 37.7 | 1.3 | 472 | 35.3M |
| Reactive | swin_v2_tiny | concat | 4 | 15.3 | 65.2 | 66.4 | 1.2 | 796 | 35.3M |
| Reactive | swin_v2_tiny | cross_attn | 1 | 27.9 | 35.8 | 36.5 | 0.7 | 310 | 35.3M |
| Reactive | swin_v2_tiny | cross_attn | 2 | 27.4 | 36.5 | 37.9 | 1.4 | 472 | 35.3M |
| Reactive | swin_v2_tiny | cross_attn | 4 | 15.2 | 65.9 | 71.6 | 6.1 | 796 | 35.3M |
| Reactive | swin_v2_tiny | bev | 1 | 10.6 | 94.1 | 95.4 | 1.4 | 1819 | 69.7M |
| Reactive | swin_v2_tiny | bev | 2 | 5.4 | 184.5 | 188.4 | 4.3 | 3353 | 69.7M |
| Reactive | swin_v2_tiny | bev | 4 | 2.8 | 360.2 | 380.2 | 21.3 | 6420 | 69.7M |
| Reactive | conv_next_v2_tiny | concat | 1 | 32.0 | 31.2 | 36.7 | 5.7 | 333 | 35.6M |
| Reactive | conv_next_v2_tiny | concat | 2 | 27.9 | 35.8 | 37.9 | 2.3 | 519 | 35.6M |
| Reactive | conv_next_v2_tiny | concat | 4 | 15.6 | 64.2 | 67.0 | 2.8 | 891 | 35.6M |
| Reactive | conv_next_v2_tiny | cross_attn | 1 | 31.6 | 31.6 | 33.4 | 2.0 | 332 | 35.6M |
| Reactive | conv_next_v2_tiny | cross_attn | 2 | 27.8 | 35.9 | 37.6 | 1.9 | 518 | 35.6M |
| Reactive | conv_next_v2_tiny | cross_attn | 4 | 15.5 | 64.5 | 67.2 | 2.6 | 890 | 35.6M |
| Reactive | conv_next_v2_tiny | bev | 1 | 10.7 | 93.9 | 94.2 | 0.3 | 1819 | 70.0M |
| Reactive | conv_next_v2_tiny | bev | 2 | 5.5 | 182.1 | 183.2 | 1.2 | 3350 | 70.0M |
| Reactive | conv_next_v2_tiny | bev | 4 | 2.8 | 355.7 | 356.8 | 1.1 | 6418 | 70.0M |

### NVIDIA GeForce RTX 4070

> CUDA 11.8 | Driver 575.57.08 | PyTorch 2.4.1+cu118 | Commit `f5647a2` | Resolution [256, 256]

| Backbone | Fusion Mode | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|----------|-------------|-------|-----|--------------|----------|-------------|-----------|--------|
| swin_v2_tiny | bev | 1 | 54.1 | 18.5 | 18.8 | 0.2 | 417 | 62.8M |
| swin_v2_tiny | bev | 2 | 27.2 | 36.8 | 37.1 | 0.4 | 580 | 62.8M |
| swin_v2_tiny | bev | 4 | 13.6 | 73.6 | 73.8 | 0.2 | 905 | 62.8M |
| conv_next_v2_tiny | bev | 1 | 53.1 | 18.8 | 18.9 | 0.1 | 443 | 63.1M |
| conv_next_v2_tiny | bev | 2 | 26.3 | 38.0 | 38.1 | 0.2 | 630 | 63.1M |
| conv_next_v2_tiny | bev | 4 | 13.1 | 76.4 | 76.6 | 0.2 | 1003 | 63.1M |

</details>

## NVIDIA A40
<details open>
  <summary>Toggle view</summary>

> CUDA 12.8 | Driver 610.43.02 | PyTorch 2.7.1+cu128 | Commit `ead2171` | Resolution [256, 256]

| Backbone | Fusion Mode | Reasoning | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|----------|-------------|-----------|-------|-----|--------------|----------|-------------|-----------|--------|
| swin_v2_tiny | bev | off | 1 | 32.9 | 30.4 | 30.9 | 0.5 | 375 | 56.8M |
| swin_v2_tiny | bev | off | 2 | 22.9 | 43.7 | 46.3 | 2.6 | 520 | 56.8M |
| swin_v2_tiny | bev | off | 4 | 14.1 | 70.8 | 71.3 | 0.7 | 803 | 56.8M |
| conv_next_v2_tiny | bev | off | 1 | 33.1 | 30.2 | 31.2 | 1.2 | 396 | 57.1M |
| conv_next_v2_tiny | bev | off | 2 | 23.0 | 43.6 | 45.1 | 1.7 | 561 | 57.1M |
| conv_next_v2_tiny | bev | off | 4 | 14.3 | 69.7 | 72.8 | 3.2 | 887 | 57.1M |
| swin_v2_tiny | bev | pooled_latent | 1 | 30.3 | 33.0 | 38.1 | 5.4 | 386 | 59.4M |
| swin_v2_tiny | bev | horizon_cross_attention | 1 | 30.3 | 33.0 | 34.0 | 1.1 | 388 | 59.6M |

</details>

## NVIDIA GeForce RTX 5080 GPU (PyTorch 2.7.1 re-run)
<details open>
  <summary>Toggle view</summary>

> CUDA 12.8 | Driver 580.95.05 | PyTorch 2.7.1+cu128 | Commit `ead2171` | Resolution [256, 256]

| Backbone | Fusion Mode | Reasoning | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|----------|-------------|-----------|-------|-----|--------------|----------|-------------|-----------|--------|
| swin_v2_tiny | bev | off | 1 | 55.9 | 17.9 | 18.9 | 0.9 | 375 | 56.8M |
| swin_v2_tiny | bev | off | 2 | 30.6 | 32.7 | 34.7 | 1.4 | 520 | 56.8M |
| swin_v2_tiny | bev | off | 4 | 15.2 | 66.0 | 68.8 | 1.2 | 803 | 56.8M |
| conv_next_v2_tiny | bev | off | 1 | 57.4 | 17.4 | 18.4 | 0.8 | 396 | 57.1M |
| conv_next_v2_tiny | bev | off | 2 | 29.8 | 33.5 | 35.0 | 0.8 | 561 | 57.1M |
| conv_next_v2_tiny | bev | off | 4 | 14.7 | 67.9 | 70.5 | 1.0 | 887 | 57.1M |
| swin_v2_tiny | bev | pooled_latent | 1 | 53.6 | 18.7 | 19.7 | 0.8 | 386 | 59.4M |
| swin_v2_tiny | bev | horizon_cross_attention | 1 | 53.1 | 18.8 | 20.7 | 1.7 | 388 | 59.6M |

</details>

## NVIDIA GeForce RTX 5090
<details open>
  <summary>Toggle view</summary>

> CUDA 12.8 | Driver 590.48.01 | PyTorch 2.7.1+cu128 | Commit `ead2171` | Resolution [256, 256]

| Backbone | Fusion Mode | Reasoning | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|----------|-------------|-----------|-------|-----|--------------|----------|-------------|-----------|--------|
| swin_v2_tiny | bev | off | 1 | 88.2 | 11.3 | 12.9 | 1.7 | 375 | 56.8M |
| swin_v2_tiny | bev | off | 2 | 58.4 | 17.1 | 18.2 | 1.2 | 520 | 56.8M |
| swin_v2_tiny | bev | off | 4 | 30.4 | 32.9 | 33.7 | 0.9 | 803 | 56.8M |
| conv_next_v2_tiny | bev | off | 1 | 95.6 | 10.5 | 11.8 | 1.4 | 396 | 57.1M |
| conv_next_v2_tiny | bev | off | 2 | 58.7 | 17.0 | 17.9 | 1.0 | 561 | 57.1M |
| conv_next_v2_tiny | bev | off | 4 | 30.0 | 33.4 | 34.0 | 0.8 | 887 | 57.1M |
| swin_v2_tiny | bev | pooled_latent | 1 | 84.2 | 11.9 | 12.8 | 1.0 | 386 | 59.4M |
| swin_v2_tiny | bev | horizon_cross_attention | 1 | 83.5 | 12.0 | 12.8 | 0.9 | 388 | 59.6M |

</details>

## NVIDIA GeForce RTX 4070 Laptop GPU
<details open>
  <summary>Toggle view</summary>

> CUDA 12.8 | Driver 595.71.05 | PyTorch 2.7.1+cu128 | Commit `ead2171` | Resolution [256, 256]

| Backbone | Fusion Mode | Reasoning | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|----------|-------------|-----------|-------|-----|--------------|----------|-------------|-----------|--------|
| swin_v2_tiny | bev | off | 1 | 24.6 | 40.6 | 43.5 | 3.0 | 375 | 56.8M |
| swin_v2_tiny | bev | off | 2 | 13.0 | 76.8 | 80.5 | 4.0 | 520 | 56.8M |
| swin_v2_tiny | bev | off | 4 | 6.3 | 157.8 | 160.0 | 2.1 | 803 | 56.8M |
| conv_next_v2_tiny | bev | off | 1 | 25.8 | 38.8 | 41.9 | 3.1 | 396 | 57.1M |
| conv_next_v2_tiny | bev | off | 2 | 13.0 | 77.1 | 80.3 | 3.3 | 561 | 57.1M |
| conv_next_v2_tiny | bev | off | 4 | 6.4 | 157.0 | 160.3 | 3.2 | 887 | 57.1M |
| swin_v2_tiny | bev | pooled_latent | 1 | 24.6 | 40.7 | 43.4 | 3.0 | 386 | 59.4M |
| swin_v2_tiny | bev | horizon_cross_attention | 1 | 24.6 | 40.7 | 43.8 | 3.5 | 388 | 59.6M |

</details>

## Add benchmarks for your own GPU

To obtain benchmarks for your GPU, simply run the
[benchmarking script](https://github.com/autowarefoundation/auto_e2e/tree/main/Model/speed_benchmark).
There, you can also read more about the meaning of benchmark parameters.
