# AutoE2E - End-to-End AI for Self Driving

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)">
        <img src="./Media/auto_e2e_logo.jpg" alt="AutoE2E" width="100%">
    </picture>
</p>

<p align="center">
  <a href="https://huggingface.co/AutowareFoundation/auto_e2e">
    <img src="./Media/readme/huggingface-cta.svg" alt="AutoE2E v1.0 checkpoints on Hugging Face" width="100%">
  </a>
</p>

<p align="center">
  <b>An open-source, camera-first End-to-End driving model for highways, arterial roads and city streets.</b>
</p>

<p align="center">
  <a href="https://huggingface.co/AutowareFoundation/auto_e2e"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FAutowareFoundation%2Fauto_e2e&query=%24.downloads&label=HF%20downloads%2Fmonth&logo=huggingface&color=FFD21E&labelColor=1b1400&style=for-the-badge" alt="Hugging Face downloads"></a>
  <a href="https://github.com/autowarefoundation/auto_e2e/stargazers"><img src="https://img.shields.io/github/stars/autowarefoundation/auto_e2e?style=for-the-badge&logo=github&color=a24bf5&labelColor=140f2a" alt="GitHub stars"></a>
  <a href="https://discord.com/invite/Q94UsPvReQ"><img src="https://img.shields.io/discord/953808765935816715?style=for-the-badge&logo=discord&logoColor=white&label=Discord&color=5865F2&labelColor=140f2a" alt="Autoware Discord"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-5ee0d6?style=for-the-badge&labelColor=0a2027" alt="Apache 2.0 license"></a>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
</p>

<br>

<p align="center">
  <img src="./Media/readme/pillars.svg" alt="Camera-first, no HD maps, fully open" width="100%">
</p>

AutoE2E plans the vehicle's trajectory directly from surround cameras. Fuse its output with LiDAR and radar safety layers for **driverless robotaxis**, or run it camera-only for **L2++ hands-free ADAS**.

<p align="center">
  <img src="./Media/readme/stats.svg" alt="79.9 M parameters, 6 cameras, 6.4 s horizon, 10 Hz" width="100%">
</p>

<img src="./Media/readme/section-architecture.svg" alt="Architecture" width="100%">

<p align="center">
  <img src="./Media/readme/architecture.svg" alt="AutoE2E architecture: World Action, Reasoning and Reactive models" width="100%">
</p>

AutoE2E is three cooperating models. The **Reactive model** runs at 10 Hz and drives: it lifts six surround cameras into a bird's-eye-view grid, optionally fuses a map raster, and lets a GRU planner turn that scene plus ego-motion history into the next 6.4 s of acceleration and curvature. The 1 Hz **World Action** and **Reasoning** models are in development. The [Model guide](./Model/) has the full inputs, outputs and forward signature.

<details>
  <summary>Original design diagram</summary>
  <br>
  <img src="./Media/auto_e2e_architecture.jpg" alt="Original AutoE2E design diagram" width="100%">
</details>

<img src="./Media/readme/section-see-it-drive.svg" alt="See it drive" width="100%">

<p align="center">
  <img src="https://huggingface.co/AutowareFoundation/auto_e2e/resolve/main/assets/prediction_kitscenes_epoch5_val_roundabout.jpg" alt="AutoE2E entering a roundabout in Karlsruhe" width="100%">
  <br>
  <sub>KITScenes Epoch 5 entering a roundabout in Karlsruhe. <b>Purple</b> is the logged path, <b>green</b> is the AutoE2E prediction. Camera images: <a href="https://kitscenes.com/">KITScenes-Multimodal</a>, CC BY-NC 4.0.</sub>
</p>

<p align="center">
  <a href="https://d2itskdqq39tx1.cloudfront.net/"><img src="https://img.shields.io/badge/Play_every_frame-DataModelConsole_dashboard-0ea5e9?style=for-the-badge&labelColor=0a2027" alt="Open the DataModelConsole dashboard"></a>
</p>

The read-only [DataModelConsole dashboard](https://d2itskdqq39tx1.cloudfront.net/) replays synchronized camera scenes with ground-truth and predicted trajectories in camera and bird's-eye views, next to dataset versions, MLflow models and Flyte executions.

<img src="./Media/readme/section-benchmarks.svg" alt="Benchmarks" width="100%">

<p align="center">
  <img src="./Media/readme/benchmark-kitscenes-3s.svg" alt="ADE and FDE at 3 s on KITScenes-Multimodal validation for eight open-weights driving models. The fine-tuned AutoE2E checkpoint is second, behind Qwen Drive 1.0." width="100%">
</p>

The fine-tuned AutoE2E checkpoint has the second-lowest error of the eight open-weights models compared here on KITScenes-Multimodal validation. Only Qwen Drive 1.0 is ahead, and AutoE2E beats SimForge, Alpamayo 1.5 and METEOR with 79.9 M parameters. All models are scored open-loop: each one predicts the next 3 s from logged camera frames, and the prediction is compared with the path the human driver took. ADE is the average position error over those 3 s and FDE is the error at the 3 s point.

| Model | Checkpoint / setting | Evaluated on | ADE @ 3 s (m) | FDE @ 3 s (m) |
|---|---|---|---:|---:|
| Qwen Drive 1.0 | first of 6 trajectories | 202 keyframes | 0.565 | 1.397 |
| **AutoE2E fine-tuned** | [`kitscenes-epoch-5`](https://huggingface.co/AutowareFoundation/auto_e2e/tree/main/models/kitscenes-epoch-5) | 202 keyframes | **0.744** | **1.980** |
| SimForge D1.5 | 7 camera slots (2 virtual telephoto), first of 6 trajectories | 202 keyframes | 0.831 | 2.209 |
| **AutoE2E** | [`nuplan-epoch-5`](https://huggingface.co/AutowareFoundation/auto_e2e/tree/main/models/nuplan-epoch-5), zero-shot | 11,035 samples | **0.915** | **2.423** |
| Alpamayo 1.5 | first of 6 trajectories | 202 keyframes | 1.020 | 2.820 |
| VAD | Base stage-2, oracle driving command | 11,035 samples | 1.242 | 2.821 |
| METEOR | top-1 of 3 paths | 202 keyframes | 1.240 | 3.282 |
| Drive-JEPA | perception-based v2, 20 m route command | 11,035 samples | 2.266 | 4.774 |

The 11,035 samples are every frame of the 117 validation scenes, and the 202 keyframes are non-overlapping anchors taken from the same scenes. Inputs differ by model: AutoE2E also gets an HD map and the route, and VAD and Drive-JEPA get a driving command. [Issue #210](https://github.com/autowarefoundation/auto_e2e/issues/210) has each model's adapter, ground truth and results at the other horizons.

<img src="./Media/readme/section-get-started.svg" alt="Get started" width="100%">

**1. Install** with Python 3.12 (the pinned PyTorch build has no wheels for 3.13+).

```bash
git clone https://github.com/autowarefoundation/auto_e2e.git
cd auto_e2e
make setup                      # CPU torch wheels
make setup TORCH_CHANNEL=cu118  # or a CUDA build (cu126, cu128, ... work too)
make test                       # optional: verify the install
```

<details>
  <summary>No <code>make</code>? Use plain pip</summary>

```bash
pip install -r requirements.txt                      # CPU torch wheels
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118  # or a CUDA build
```

The install check needs `make`, which you can get from a [package manager](https://chocolatey.org/).

</details>

**2. Pick a checkpoint** from the [Hugging Face release](https://huggingface.co/AutowareFoundation/auto_e2e). The model card has the loading example, the evaluation reports and the KITScenes validation shards.

| Checkpoint | Trained on | Use it for |
|---|---|---|
| [`nuplan-epoch-5`](https://huggingface.co/AutowareFoundation/auto_e2e/tree/main/models/nuplan-epoch-5) | nuPlan v1.1 | Zero-shot driving in new cities and fine-tuning on your data |
| [`kitscenes-epoch-5`](https://huggingface.co/AutowareFoundation/auto_e2e/tree/main/models/kitscenes-epoch-5) | nuPlan, then KITScenes-Multimodal | Karlsruhe roads and the KITScenes camera rig |

The checkpoints are for non-commercial use under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), because they are trained on nuPlan (CC BY-NC-SA 4.0) and KITScenes-Multimodal (CC BY-NC 4.0) and start from BEVFormer V2 weights trained on nuScenes (CC BY-NC-SA 4.0). The source code and the assets in this repository remain under Apache-2.0. The [model card](https://huggingface.co/AutowareFoundation/auto_e2e#license) lists the license of every release file.

These checkpoints run with the code tagged [`v1.0`](https://github.com/autowarefoundation/auto_e2e/tree/v1.0). The current code trains the AutoBEV model ([`Design/autobev_integration.md`](./Design/autobev_integration.md)) through `wf_autoe2e`; [`Platform/HowToUseFlyte.md`](./Platform/HowToUseFlyte.md) shows how to launch training and evaluation.

**3. Go further.** Read the [documentation and academic paper](https://autowarefoundation.github.io/auto_e2e/), explore the [Model](./Model/) folder, or follow the [Trial Guide](./TRIAL.md) to run the test suite on AWS EC2.

<img src="./Media/readme/section-community.svg" alt="Join the community" width="100%">

AutoE2E is developed in the Autoware Robotaxi working group, and new contributors are welcome. The [onboarding guide](./ONBOARDING.md) explains how to join the weekly meetings, and the community benchmark against other end-to-end models runs in [issue #210](https://github.com/autowarefoundation/auto_e2e/issues/210). If AutoE2E is useful to you, a ⭐ on GitHub helps the project grow.

<p align="center">
  <a href="https://discord.com/invite/Q94UsPvReQ"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://huggingface.co/AutowareFoundation/auto_e2e"><img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="https://www.linkedin.com/company/the-autoware-foundation"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  <a href="https://www.youtube.com/@autowarefoundation"><img src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube"></a>
  <a href="https://autoware.org/"><img src="https://img.shields.io/badge/autoware.org-000000?style=for-the-badge&logo=About.me&logoColor=white" alt="Autoware website"></a>
</p>

<p align="center">
  <img src="./Media/readme/footer.svg" alt="Built in the open by the Autoware Foundation" width="100%">
</p>
