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

You can run the tool directly from the command line:

```bash
python Tools/trajectory_visualization/cli.py \
    --checkpoint_dir /path/to/checkpoint/ \
    --dataset_dir /path/to/dataset/ \
    --output_dir /path/to/output/ \
    --num_frames 100
```

### Flyte Integration

The tool is designed so that a later Flyte task can call it directly with downloaded `FlyteFile` and `FlyteDirectory` inputs. The generated output directory can then be logged through MLflow:

```python
mlflow.log_artifacts(output_dir, artifact_path="trajectory_visualization")
```