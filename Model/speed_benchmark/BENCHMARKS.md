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

> CUDA 11.8 | Driver 610.62 | PyTorch 2.7.1+cu118 | Commit `d01fb8b` | Resolution [256, 256]

| Model | Backbone | Fusion Mode | Reasoning | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|-------|----------|-------------|-----------|-------|-----|--------------|----------|-------------|-----------|--------|
| Reactive | swin_v2_tiny | bev | off | 1 | 18.6 | 53.9 | 80.1 | 29.2 | 375 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 1 | 12.5 | 79.9 | 99.4 | 21.1 | 525 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 2 | 12.0 | 83.4 | 100.5 | 20.1 | 521 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 2 | 6.8 | 147.8 | 171.8 | 25.7 | 668 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 4 | 6.3 | 159.0 | 168.1 | 9.8 | 803 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 4 | 3.4 | 294.2 | 304.6 | 10.9 | 952 | 94.7M |
| Reactive | conv_next_v2_tiny | bev | off | 1 | 19.0 | 52.5 | 63.1 | 10.8 | 396 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 1 | 11.4 | 87.9 | 111.3 | 26.2 | 543 | 95.2M |
| Reactive | conv_next_v2_tiny | bev | off | 2 | 10.9 | 91.4 | 115.7 | 27.4 | 562 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 2 | 6.3 | 158.3 | 173.5 | 16.1 | 707 | 95.2M |
| Reactive | conv_next_v2_tiny | bev | off | 4 | 5.9 | 170.2 | 185.2 | 16.5 | 887 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 4 | 3.2 | 313.4 | 327.6 | 15.1 | 1034 | 95.2M |
| Reactive | swin_v2_tiny | bev | pooled_latent | 1 | 16.3 | 61.4 | 131.4 | 75.9 | 388 | 59.4M |
| Reactive | swin_v2_tiny | bev | horizon_cross_attention | 1 | 18.1 | 55.4 | 80.3 | 26.1 | 389 | 59.6M |

</details>

## NVIDIA GeForce RTX 4080

<details open>
  <summary>Toggle view</summary>

> CUDA 12.8 | Driver 580.65.06 | PyTorch 2.7.1+cu128 | Commit `9204344` | Resolution [256, 256]

| Model | Backbone | Fusion Mode | Reasoning | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|-------|----------|-------------|-----------|-------|-----|--------------|----------|-------------|-----------|--------|
| Reactive | swin_v2_tiny | bev | off | 1 | 69.6 | 14.4 | 15.5 | 1.1 | 375 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 1 | 45.6 | 21.9 | 22.6 | 0.7 | 525 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 2 | 45.9 | 21.8 | 22.5 | 0.8 | 521 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 2 | 27.2 | 36.8 | 37.9 | 1.2 | 668 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 4 | 25.0 | 40.0 | 40.8 | 0.9 | 803 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 4 | 13.9 | 72.2 | 73.2 | 1.1 | 952 | 94.7M |
| Reactive | conv_next_v2_tiny | bev | off | 1 | 71.0 | 14.1 | 15.6 | 1.6 | 396 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 1 | 46.4 | 21.6 | 22.1 | 0.6 | 543 | 95.3M |
| Reactive | conv_next_v2_tiny | bev | off | 2 | 43.8 | 22.8 | 23.5 | 0.7 | 562 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 2 | 25.8 | 38.8 | 39.6 | 0.9 | 707 | 95.3M |
| Reactive | conv_next_v2_tiny | bev | off | 4 | 23.3 | 42.9 | 43.8 | 1.0 | 887 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 4 | 12.8 | 78.2 | 78.8 | 0.7 | 1034 | 95.3M |
| Reactive | swin_v2_tiny | bev | pooled_latent | 1 | 64.8 | 15.4 | 16.0 | 0.7 | 388 | 59.4M |
| Reactive | swin_v2_tiny | bev | horizon_cross_attention | 1 | 64.4 | 15.5 | 16.1 | 0.7 | 389 | 59.6M |

</details>


## NVIDIA GeForce RTX 5080

<details open>
  <summary>Toggle view</summary>

> CUDA 12.8 | Driver 580.95.05 | PyTorch 2.7.1+cu128 | Commit `9204344` | Resolution [256, 256]

| Model | Backbone | Fusion Mode | Reasoning | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|-------|----------|-------------|-----------|-------|-----|--------------|----------|-------------|-----------|--------|
| Reactive | swin_v2_tiny | bev | off | 1 | 56.9 | 17.6 | 18.0 | 0.5 | 375 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 1 | 33.1 | 30.2 | 31.1 | 0.9 | 525 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 2 | 30.4 | 32.9 | 33.4 | 0.5 | 521 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 2 | 16.8 | 59.4 | 60.2 | 1.0 | 668 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 4 | 15.2 | 65.5 | 66.2 | 0.7 | 803 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 4 | 8.2 | 122.2 | 123.2 | 1.1 | 952 | 94.7M |
| Reactive | conv_next_v2_tiny | bev | off | 1 | 58.2 | 17.2 | 17.7 | 0.5 | 396 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 1 | 34.4 | 29.1 | 29.6 | 0.5 | 543 | 95.3M |
| Reactive | conv_next_v2_tiny | bev | off | 2 | 30.0 | 33.3 | 33.6 | 0.3 | 562 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 2 | 16.6 | 60.2 | 60.5 | 0.3 | 707 | 95.3M |
| Reactive | conv_next_v2_tiny | bev | off | 4 | 14.8 | 67.7 | 68.0 | 0.4 | 887 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 4 | 8.0 | 125.0 | 126.0 | 1.3 | 1034 | 95.3M |
| Reactive | swin_v2_tiny | bev | pooled_latent | 1 | 54.1 | 18.5 | 19.1 | 0.6 | 388 | 59.4M |
| Reactive | swin_v2_tiny | bev | horizon_cross_attention | 1 | 54.2 | 18.4 | 19.0 | 0.6 | 389 | 59.6M |

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

> CUDA 12.8 | Driver 610.43.02 | PyTorch 2.7.1+cu128 | Commit `92043448` | Resolution [256, 256]

| Model | Backbone | Fusion Mode | Reasoning | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|-------|----------|-------------|-----------|-------|-----|--------------|----------|-------------|-----------|--------|
| Reactive | swin_v2_tiny | bev | off | 1 | 27.7 | 36.1 | 49.3 | 14.3 | 375 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 1 | 20.6 | 48.7 | 84.7 | 38.0 | 525 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 2 | 18.3 | 54.7 | 81.0 | 35.8 | 521 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 2 | 13.0 | 76.7 | 142.2 | 67.6 | 668 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 4 | 12.9 | 77.6 | 146.3 | 73.6 | 803 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 4 | 7.5 | 134.2 | 219.4 | 92.3 | 952 | 94.7M |
| Reactive | conv_next_v2_tiny | bev | off | 1 | 30.6 | 32.7 | 52.7 | 21.3 | 396 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 1 | 17.3 | 57.9 | 84.4 | 38.5 | 543 | 95.3M |
| Reactive | conv_next_v2_tiny | bev | off | 2 | 22.0 | 45.4 | 49.7 | 4.6 | 562 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 2 | 13.0 | 76.7 | 139.2 | 66.8 | 707 | 95.3M |
| Reactive | conv_next_v2_tiny | bev | off | 4 | 13.5 | 74.0 | 133.3 | 61.9 | 887 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 4 | 7.2 | 139.1 | 263.2 | 138.4 | 1034 | 95.3M |
| Reactive | swin_v2_tiny | bev | pooled_latent | 1 | 27.8 | 36.0 | 55.6 | 20.8 | 388 | 59.4M |
| Reactive | swin_v2_tiny | bev | horizon_cross_attention | 1 | 28.3 | 35.3 | 35.8 | 0.6 | 389 | 59.6M |

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

> CUDA 12.8 | Driver 590.48.01 | PyTorch 2.7.1+cu128 | Commit `92043448` | Resolution [256, 256]

| Model | Backbone | Fusion Mode | Reasoning | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|-------|----------|-------------|-----------|-------|-----|--------------|----------|-------------|-----------|--------|
| Reactive | swin_v2_tiny | bev | off | 1 | 87.3 | 11.4 | 13.5 | 2.1 | 375 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 1 | 55.4 | 18.1 | 18.9 | 0.9 | 525 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 2 | 58.2 | 17.2 | 18.0 | 0.8 | 521 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 2 | 34.0 | 29.4 | 30.3 | 0.9 | 668 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 4 | 30.4 | 32.9 | 33.7 | 0.8 | 803 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 4 | 16.7 | 59.8 | 60.7 | 1.0 | 952 | 94.7M |
| Reactive | conv_next_v2_tiny | bev | off | 1 | 94.7 | 10.6 | 11.2 | 0.6 | 396 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 1 | 60.9 | 16.4 | 17.1 | 0.7 | 543 | 95.3M |
| Reactive | conv_next_v2_tiny | bev | off | 2 | 58.8 | 17.0 | 17.6 | 0.6 | 562 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 2 | 34.4 | 29.1 | 30.7 | 1.7 | 707 | 95.3M |
| Reactive | conv_next_v2_tiny | bev | off | 4 | 29.9 | 33.4 | 34.1 | 0.9 | 887 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 4 | 16.5 | 60.6 | 61.4 | 0.9 | 1034 | 95.3M |
| Reactive | swin_v2_tiny | bev | pooled_latent | 1 | 83.5 | 12.0 | 12.5 | 0.6 | 388 | 59.4M |
| Reactive | swin_v2_tiny | bev | horizon_cross_attention | 1 | 82.9 | 12.1 | 12.5 | 0.5 | 389 | 59.6M |

</details>


## NVIDIA GeForce RTX 4070 Laptop GPU

<details open>
  <summary>Toggle view</summary>

> CUDA 12.8 | Driver 595.84 | PyTorch 2.7.1+cu128 | Commit `92043448` | Resolution [256, 256]

| Model | Backbone | Fusion Mode | Reasoning | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|-------|----------|-------------|-----------|-------|-----|--------------|----------|-------------|-----------|--------|
| Reactive | swin_v2_tiny | bev | off | 1 | 24.6 | 40.7 | 43.9 | 3.5 | 375 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 1 | 13.7 | 73.3 | 78.8 | 4.9 | 525 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 2 | 13.0 | 76.8 | 79.7 | 3.0 | 521 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 2 | 7.0 | 142.9 | 146.3 | 3.4 | 668 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 4 | 6.3 | 158.1 | 160.9 | 2.6 | 803 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 4 | 3.4 | 293.4 | 296.6 | 3.4 | 952 | 94.7M |
| Reactive | conv_next_v2_tiny | bev | off | 1 | 25.5 | 39.2 | 41.1 | 1.4 | 396 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 1 | 14.2 | 70.3 | 72.4 | 2.2 | 543 | 95.3M |
| Reactive | conv_next_v2_tiny | bev | off | 2 | 12.9 | 77.5 | 80.1 | 2.6 | 562 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 2 | 7.0 | 142.9 | 147.0 | 4.0 | 707 | 95.3M |
| Reactive | conv_next_v2_tiny | bev | off | 4 | 6.4 | 157.0 | 160.4 | 3.2 | 887 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 4 | 3.4 | 291.5 | 295.5 | 4.1 | 1034 | 95.3M |
| Reactive | swin_v2_tiny | bev | pooled_latent | 1 | 24.6 | 40.6 | 45.0 | 4.7 | 388 | 59.4M |
| Reactive | swin_v2_tiny | bev | horizon_cross_attention | 1 | 24.4 | 40.9 | 44.6 | 4.1 | 389 | 59.6M |

</details>


## NVIDIA Quadro RTX 5000

<details open>
  <summary>Toggle view</summary>

> CUDA 11.8 | Driver 590.48.01 | PyTorch 2.7.1+cu118 | Commit `47c1e50` | Resolution [256, 256]

| Model | Backbone | Fusion Mode | Reasoning | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|-------|----------|-------------|-----------|-------|-----|--------------|----------|-------------|-----------|--------|
| Reactive | swin_v2_tiny | bev | off | 1 | 30.1 | 33.3 | 37.8 | 5.0 | 375 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 1 | 17.7 | 56.6 | 61.3 | 5.1 | 525 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 2 | 18.1 | 55.3 | 60.2 | 5.3 | 521 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 2 | 10.1 | 99.0 | 102.5 | 4.3 | 668 | 94.7M |
| Reactive | swin_v2_tiny | bev | off | 4 | 9.7 | 102.7 | 104.3 | 1.6 | 803 | 56.8M |
| Combined | swin_v2_tiny | bev | off | 4 | 5.3 | 188.3 | 192.2 | 4.2 | 952 | 94.7M |
| Reactive | conv_next_v2_tiny | bev | off | 1 | 28.3 | 35.4 | 37.6 | 2.4 | 396 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 1 | 16.4 | 60.9 | 65.5 | 4.8 | 543 | 95.3M |
| Reactive | conv_next_v2_tiny | bev | off | 2 | 16.9 | 59.4 | 63.6 | 4.7 | 562 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 2 | 9.4 | 106.7 | 109.6 | 3.2 | 707 | 95.3M |
| Reactive | conv_next_v2_tiny | bev | off | 4 | 9.0 | 111.6 | 113.7 | 2.3 | 887 | 57.1M |
| Combined | conv_next_v2_tiny | bev | off | 4 | 4.9 | 203.5 | 205.7 | 2.3 | 1034 | 95.3M |
| Reactive | swin_v2_tiny | bev | pooled_latent | 1 | 28.2 | 35.5 | 38.9 | 3.7 | 388 | 59.4M |
| Reactive | swin_v2_tiny | bev | horizon_cross_attention | 1 | 27.9 | 35.8 | 39.0 | 3.5 | 389 | 59.6M |

</details>

## Add benchmarks for your own GPU

To obtain benchmarks for your GPU, simply run the
[benchmarking script](https://github.com/autowarefoundation/auto_e2e/tree/main/Model/speed_benchmark).
There, you can also read more about the meaning of benchmark parameters.
