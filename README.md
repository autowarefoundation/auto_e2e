# AutoE2E - End-to-End AI for Self Driving

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)">
        <img src="./Media/auto_e2e_logo.jpg" alt="VisionPilot" width="100%">
    </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/discord/953808765935816715?label=Autoware%20Discord)](https://discord.com/invite/Q94UsPvReQ)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/autowarefoundation/auto_e2e)
![GitHub Repo stars](https://img.shields.io/github/stars/autowarefoundation/auto_e2e)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/the-autoware-foundation)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@autowarefoundation)
[![Website](https://img.shields.io/badge/website-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://autoware.org/)
</div>

<div align="center">

⭐ Star us on GitHub — your support motivates us a lot!

</div>

## Free and fully open-source End-to-End AI model
**AutoE2E is an open-source End-to-End AI model** which enables autonomous driving across highways, arterial roads and city streets using cameras-only, and without reliance on HD-maps. 

AutoE2E outputs can be fused with Physics-based sensors such as LIDAR/RADAR to power **fully driverless Robotaxi applications**, and the basline camera-only model can be used to enable **L2++ automotive ADAS** applications for point-to-point hands-free navigation.

To learn more about how to participate in this project, please read the [onboarding guide](/ONBOARDING.md)

## Getting started
- Install the dependencies with `make setup` (CPU torch wheels) or `make setup TORCH_CHANNEL=cu118` for a CUDA build; plain `pip install -r requirements.txt` also works and uses the default PyPI torch wheels
- Visit the [Model](./Model/) folder to view the model components, run training and perform inference
- See the [Trial Guide](./TRIAL.md) for step-by-step instructions on running the inference test on AWS EC2

## Inference Speed Benchmarks

### NVIDIA GeForce RTX 3060 Laptop GPU 
<details open>
  <summary>Toggle view</summary>


| Backbone | Fusion Method | FPS | Average Latency [ms] | Worst-Case Latency [ms] | Latency Jitter [ms] | Peak VRAM Allocated [MB] | Peak VRAM Reserved [MB] |
| -------- | ------------- | --- | --------------- | ------------------ | -------------- | ------------------- | ------------------ |
| SwinV2 Tiny | Feature Concat | 24.99 | 40.01 | 40.68 | 0.71 | 1067.52 | 1216.00 |
| SwinV2 Tiny | Spatial Attention | 24.48 | 44.49 | 47.23 | 2.75 | 1069.18 | 1218.00 |
| SwinV2 Tiny | BEV Fusion | 22.02 | 45.42 | 67.72 | 23.87 | 1069.18 | 1220.00 |
| ConvNextV2 Tiny | Feature Concat | 22.99 | 43.49 | 49.23 | 7.26 | 1092.58 | 1268.00 |
| ConvNextV2 Tiny | Spatial Attention | 18.60 | 53.75 | 54.15 | 0.36 | 1092.58 | 1268.00 |
| ConvNextV2 Tiny | BEV Fusion | 18.63 | 53.69 | 54.37 | 0.67 | 1092.58 | 1268.00 |

</details>

### NVIDIA GeForce RTX 4050 Laptop GPU 
<details open>
  <summary>Toggle view</summary>


| Backbone | Fusion Method | FPS | Average Latency [ms] | Worst-Case Latency [ms] | Latency Jitter [ms] | Peak VRAM Allocated [MB] | Peak VRAM Reserved [MB] |
| -------- | ------------- | --- | --------------- | ------------------ | -------------- | ------------------- | ------------------ |
| SwinV2 Tiny | Feature Concat | 25.76 | 38.81 | 40.60 | 1.80 | 1067.52 | 1216.00 |
| SwinV2 Tiny | Spatial Attention | 24.85 | 40.24 | 41.32 | 1.04 | 1069.18 | 1218.00 |
| SwinV2 Tiny | BEV Fusion | 25.47 | 39.27 | 41.36 | 2.36 | 1069.18 | 1220.00 |
| ConvNextV2 Tiny | Feature Concat | 25.92 | 38.58 | 39.27 | 0.74 | 1092.58 | 1268.00 |
| ConvNextV2 Tiny | Spatial Attention | 23.06 | 43.37 | 52.16 | 9.03 | 1092.58 | 1268.00 |
| ConvNextV2 Tiny | BEV Fusion | 21.70 | 46.09 | 77.30 | 33.68 | 1092.58 | 1268.00 |
  
</details>

### NVIDIA GeForce RTX 5080 GPU 
<details open>
  <summary>Toggle view</summary>
  
> CUDA 12.8 | Driver 595.71.05 | PyTorch 2.11.0+cu128 | Commit `9015914` | Resolution [256, 256]

| Backbone | Fusion Mode | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|----------|-------------|-------|-----|--------------|----------|-------------|-----------|--------|
| swin_v2_tiny | concat | 1 | 76.5 | 13.1 | 13.6 | 0.6 | 308 | 35.3M |
| swin_v2_tiny | concat | 2 | 46.8 | 21.4 | 21.7 | 0.3 | 473 | 35.3M |
| swin_v2_tiny | concat | 4 | 25.2 | 39.7 | 40.4 | 0.6 | 797 | 35.3M |
| swin_v2_tiny | cross_attn | 1 | 75.6 | 13.2 | 13.7 | 0.5 | 311 | 35.3M |
| swin_v2_tiny | cross_attn | 2 | 46.6 | 21.5 | 21.9 | 0.4 | 473 | 35.3M |
| swin_v2_tiny | cross_attn | 4 | 25.1 | 39.9 | 40.5 | 0.6 | 797 | 35.3M |
| swin_v2_tiny | bev | 1 | 16.3 | 61.5 | 62.0 | 0.5 | 1820 | 69.7M |
| swin_v2_tiny | bev | 2 | 8.2 | 121.6 | 122.4 | 0.8 | 3354 | 69.7M |
| swin_v2_tiny | bev | 4 | 4.2 | 239.6 | 240.8 | 1.1 | 6421 | 69.7M |
| conv_next_v2_tiny | concat | 1 | 74.2 | 13.5 | 14.2 | 0.7 | 334 | 35.6M |
| conv_next_v2_tiny | concat | 2 | 42.7 | 23.4 | 23.9 | 0.5 | 520 | 35.6M |
| conv_next_v2_tiny | concat | 4 | 22.7 | 44.0 | 44.7 | 0.6 | 892 | 35.6M |
| conv_next_v2_tiny | cross_attn | 1 | 73.8 | 13.6 | 14.1 | 0.6 | 333 | 35.6M |
| conv_next_v2_tiny | cross_attn | 2 | 42.3 | 23.6 | 24.9 | 1.3 | 519 | 35.6M |
| conv_next_v2_tiny | cross_attn | 4 | 22.6 | 44.1 | 44.7 | 0.6 | 891 | 35.6M |
| conv_next_v2_tiny | bev | 1 | 16.2 | 61.9 | 62.5 | 0.7 | 1820 | 70.0M |
| conv_next_v2_tiny | bev | 2 | 8.1 | 122.8 | 123.8 | 0.9 | 3351 | 70.0M |
| conv_next_v2_tiny | bev | 4 | 4.1 | 243.1 | 244.0 | 0.9 | 6419 | 70.0M |

</details>

### NVIDIA RTX A6000 GPU
<details open>
  <summary>Toggle view</summary>

> CUDA 11.8 | Driver 580.159.03 | PyTorch 2.4.1+cu118 | Commit `9015914` | Resolution [256, 256]

| Backbone | Fusion Mode | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|----------|-------------|-------|-----|--------------|----------|-------------|-----------|--------|
| swin_v2_tiny | concat | 1 | 28.2 | 35.4 | 35.9 | 0.6 | 307 | 35.3M |
| swin_v2_tiny | concat | 2 | 27.4 | 36.5 | 37.7 | 1.3 | 472 | 35.3M |
| swin_v2_tiny | concat | 4 | 15.3 | 65.2 | 66.4 | 1.2 | 796 | 35.3M |
| swin_v2_tiny | cross_attn | 1 | 27.9 | 35.8 | 36.5 | 0.7 | 310 | 35.3M |
| swin_v2_tiny | cross_attn | 2 | 27.4 | 36.5 | 37.9 | 1.4 | 472 | 35.3M |
| swin_v2_tiny | cross_attn | 4 | 15.2 | 65.9 | 71.6 | 6.1 | 796 | 35.3M |
| swin_v2_tiny | bev | 1 | 10.6 | 94.1 | 95.4 | 1.4 | 1819 | 69.7M |
| swin_v2_tiny | bev | 2 | 5.4 | 184.5 | 188.4 | 4.3 | 3353 | 69.7M |
| swin_v2_tiny | bev | 4 | 2.8 | 360.2 | 380.2 | 21.3 | 6420 | 69.7M |
| conv_next_v2_tiny | concat | 1 | 32.0 | 31.2 | 36.7 | 5.7 | 333 | 35.6M |
| conv_next_v2_tiny | concat | 2 | 27.9 | 35.8 | 37.9 | 2.3 | 519 | 35.6M |
| conv_next_v2_tiny | concat | 4 | 15.6 | 64.2 | 67.0 | 2.8 | 891 | 35.6M |
| conv_next_v2_tiny | cross_attn | 1 | 31.6 | 31.6 | 33.4 | 2.0 | 332 | 35.6M |
| conv_next_v2_tiny | cross_attn | 2 | 27.8 | 35.9 | 37.6 | 1.9 | 518 | 35.6M |
| conv_next_v2_tiny | cross_attn | 4 | 15.5 | 64.5 | 67.2 | 2.6 | 890 | 35.6M |
| conv_next_v2_tiny | bev | 1 | 10.7 | 93.9 | 94.2 | 0.3 | 1819 | 70.0M |
| conv_next_v2_tiny | bev | 2 | 5.5 | 182.1 | 183.2 | 1.2 | 3350 | 70.0M |
| conv_next_v2_tiny | bev | 4 | 2.8 | 355.7 | 356.8 | 1.1 | 6418 | 70.0M |

</details>

### Add benchmarks for your own GPU .... 

To obtain benchmarks for your GPU, simply run the [benchmarking script](https://github.com/autowarefoundation/auto_e2e/tree/main/Model/speed_benchmark). There, you can also read more about the meaning of benchmark parameters.
