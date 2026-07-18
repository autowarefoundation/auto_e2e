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

Requires **Python 3.12** (the pinned PyTorch build has no wheels for 3.13+).

### Using `make` tool ###
<details open>
  <summary>Toggle view</summary>

1. **Clone and install dependencies**

   ```bash
   git clone https://github.com/autowarefoundation/auto_e2e.git
   cd auto_e2e
   make setup                      # CPU torch wheels
   make setup TORCH_CHANNEL=cu118  # or a CUDA build (cu121, ... work too)
   ```

2. **Verify the install** (optional)

   ```bash
   make test
   ```
</details>

### Using plain pip ###
<details open>
  <summary>Toggle view</summary>

**Clone and install dependencies**

```bash
git clone https://github.com/autowarefoundation/auto_e2e.git
cd auto_e2e
pip install -r requirements.txt                      # CPU torch wheels
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118  # or a CUDA build (cu121, ... work too)
```

Without a `make` tool, you unfortunately cannot verify the install 
using a `test` from the Makefile. It is highly recommended to install 
the tool through a [package manager](https://chocolatey.org/).

</details>

### Documentation

Review our academic paper, access our knowledge base and read through our work on safety verification in our documentation pages, alongside more information about the AutoE2E model at [https://autowarefoundation.github.io/auto_e2e/](https://autowarefoundation.github.io/auto_e2e/)

### Next steps
- Explore the [Model](./Model/) folder for the model components, training and inference.
- Follow the [Trial Guide](./TRIAL.md) to run the inference test on AWS EC2.

## Architecture at a glance

<img src="./Media/auto_e2e_architecture.jpg" width="100%">

AutoE2E takes **7 surround and telephoto cameras plus a rendered map tile**, along with
egomotion and visual history, and predicts a **6.4s future driving trajectory**
(acceleration and curvature at 10Hz). See the [Model architecture guide](./Model/) for the
full inputs, outputs and forward signature.

## Performance

Up to **~76 FPS** (SwinV2-Tiny, feature-concat fusion, RTX 5080, batch 1). Full per-GPU
inference benchmarks covering latency, jitter and VRAM across backbones, fusion modes and
batch sizes live in [BENCHMARKS.md](./Model/speed_benchmark/BENCHMARKS.md). Run the
[benchmarking script](./Model/speed_benchmark) to add results for your own GPU.
### NVIDIA GeForce RTX 5080

> CUDA 12.8 | Driver 580.95.05 | PyTorch 2.7.1+cu128 | Commit `ead2171` | Resolution [256, 256]

| Backbone | Fusion Mode | Batch | FPS | Latency (ms) | p99 (ms) | Jitter (ms) | VRAM (MB) | Params |
|----------|-------------|-------|-----|--------------|----------|-------------|-----------|--------|
| swin_v2_tiny | bev | 1 | 55.9 | 17.9 | 18.9 | 0.9 | 375 | 56.8M |
| swin_v2_tiny | bev | 2 | 30.6 | 32.7 | 34.7 | 1.4 | 520 | 56.8M |
| swin_v2_tiny | bev | 4 | 15.2 | 66.0 | 68.8 | 1.2 | 803 | 56.8M |
| conv_next_v2_tiny | bev | 1 | 57.4 | 17.4 | 18.4 | 0.8 | 396 | 57.1M |
| conv_next_v2_tiny | bev | 2 | 29.8 | 33.5 | 35.0 | 0.8 | 561 | 57.1M |
| conv_next_v2_tiny | bev | 4 | 14.7 | 67.9 | 70.5 | 1.0 | 887 | 57.1M |
| swin_v2_tiny | bev | 1 | 53.6 | 18.7 | 19.7 | 0.8 | 386 | 59.4M |
| swin_v2_tiny | bev | 1 | 53.1 | 18.8 | 20.7 | 1.7 | 388 | 59.6M |

### NVIDIA RTX A6000

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