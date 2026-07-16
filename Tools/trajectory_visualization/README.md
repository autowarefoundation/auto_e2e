# Trajectory Visualization Tool

This standalone tool visualizes the behavior of a trained `AutoE2E` model by consuming a saved checkpoint and processed evaluation dataset artifacts. It generates MP4 videos, thumbnails, and a JSON manifest without modifying the normal training dataset output schema.

## Architecture

The tool is modularized into several components:
- `cli.py`: The entry point for the tool.
- `checkpoint_loader.py`: Reconstructs the `AutoE2E` model from its configuration and state dictionary.
- `dataset_reader.py`: Wraps WebDataset to iterate over shards sequentially.
- `kinematics.py`: Math and coordinate transformations.
- `rendering.py`: Routines for plotting trajectories onto a grid and overlaying them on camera views.
- `runner.py`: The core loop orchestrating the model inference and rendering processes.
- `manifest.py`: Logs run artifacts into `manifest.json`.

## Usage

### 1. WebDataset E2E Pipeline (cli.py)

You can run the core tool directly from the command line against WebDataset `.tar` shards output by the data parsing stage:

```bash
PYTHONPATH=. python Tools/trajectory_visualization/cli.py \
    --checkpoint /path/to/model_weights.pt \
    --dataset-dir /path/to/dataset/ \
    --output-dir /path/to/output/ \
    --max-frames-per-episode 300
```

Other available options:
- `--episodes 1 2 3`: List of specific episode indices to process.
- `--selection-manifest eval-selection.json`: Path to a JSON file explicitly specifying episodes and frame ranges for scene selection.

### 2. Raw KIT Scenes Visualization

If you want to test visualization of predictions directly on raw KIT Scenes validation data without extracting it to WebDataset format, you can use the specialized `generate_kitscenes_video.py` script:

```bash
PYTHONPATH=. python Tools/trajectory_visualization/Kit_Scenes_visualization/generate_kitscenes_video.py \
    --scene_ids c34c778f-ad8c-0aa9-7e1a-c86a73f887c7 \
    --dataset_root /path/to/Kit_Scenes_visualization/data \
    --num_frames 30
```
Also, you can refer to the script as an example for the usage of the trajectory visualization tools.

*Note: This script requires `ffmpeg` with `libx264` codec to be available in your system path.*

### 3. Raw L2D Visualization

> LeRobot is incompatible with the dependencies of the project, so unexpected errors might occur

If you want to test visualization of predictions directly on raw L2D data (e.g. from HuggingFace `yaak-ai/L2D`), you can use the specialized `run_l2d_visualization.py` script:

```bash
PYTHONPATH=. python Tools/trajectory_visualization/L2D_visualization/run_l2d_visualization.py \
    --live \
    --episodes 0 \
    --frame 0
```
This script will output individual frame images (`visualization_result_map.png`, `_grid.png`, `_cam.png`) inside the `generated_images/` subfolder.

*Note: This script requires `lerobot` dependencies to fetch and parse the dataset.*

### Flyte Integration

The tool is designed so that a later Flyte task can call `cli.py` directly with downloaded `FlyteFile` and `FlyteDirectory` inputs. The generated output directory can then be logged through MLflow:

```python
mlflow.log_artifacts(output_dir, artifact_path="trajectory_visualization")
```