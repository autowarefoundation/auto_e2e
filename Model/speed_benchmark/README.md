# Speed Benchmark

`speed_benchmark.py` loads dummy data, warms up the GPU, and performs inference on 100 samples to calculate inference speed benchmarks of the model.

## Tracked Parameters

The script outputs:
* Average FPS — Frames per second that the model can process.
* Average Latency [ms] — The typical delay for a single forward pass.
* Worst-Case Latency [ms] — The 99th percentile latency.
* Latency Jitter [ms] — Variance in processing time (p99 - p50).
* Peak VRAM Allocated [MB] — Minimum theoretical GPU memory footprint.
* Peak VRAM Reserved [MB] — Realistic memory footprint seen by the OS.

## JSON Schema

Each result file in `results/` includes the following metadata:
* `timestamp` — ISO 8601 timestamp of the run
* `device` — PyTorch device string (e.g. `cuda`)
* `gpu_name` — GPU model name
* `cuda_version` — CUDA toolkit version
* `driver_version` — NVIDIA driver version
* `pytorch_version` — PyTorch version
* `commit_sha` — Git short SHA at time of benchmark
* `input_resolution` — [height, width] of input tiles

## Workflow

### 1. Run the benchmark

```bash
cd Model/speed_benchmark
python speed_benchmark.py --seed 42
```

Results are saved to `results/<gpu_name>_<timestamp>.json`.

### 2. Commit the JSON result

```bash
git add results/*.json
git commit --signoff -m "bench: add <GPU> results"
```

### 3. Generate the README table

```bash
python generate_readme_table.py
```

Copy the output into the main project README under the benchmark section.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed for reproducibility |

## Output Files

* `results/*.json` — Benchmark results with full hardware and environment metadata
