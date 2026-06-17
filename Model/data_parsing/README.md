# Data Parsing

Dataset loaders and utilities for AutoE2E training data.

## Modules

- **`l2d/`** — [L2D dataset](https://huggingface.co/datasets/yaak-ai/L2D) loader with camera and egomotion utilities
- **`nvidia_physical_ai/`** — [NVIDIA Autonomous Vehicle dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) loader
- **`map_rendering/`** — Map tile rendering and GPS-to-map conversions
- **`kit_scenes/`** — [KITScenes](https://kitscenes.com/multimodal/) data utilities

Each module provides dataset classes (`*Dataset`) and helper functions for loading camera frames, extracting egomotion, and handling map data.
