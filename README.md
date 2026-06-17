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

1. **Clone and install dependencies**

   ```bash
   git clone https://github.com/autowarefoundation/auto_e2e.git
   cd auto_e2e
   make setup                      # CPU torch wheels
   make setup TORCH_CHANNEL=cu118  # or a CUDA build (cu121, ... work too)
   ```

   Plain `pip install -r requirements.txt` also works and uses the default PyPI torch wheels.

2. **Verify the install** (optional)

   ```bash
   make test
   ```

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
