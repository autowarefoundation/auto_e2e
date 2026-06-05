# Speed Benchmark

`speed_benchmarking.py` is a script to load dummy data, warm up the GPU, and perform inference on 100 samples to calculate inference speed benchmarks of the model.

## Tracked parameters

The script outputs:
* Average FPS — Frames per second that the model can process.
* Average Latency [ms] — The typical delay it takes the GPU to complete a single forward pass of the model from start to finish.
* Worst-Case Latency [ms] — The 99th percentile latency. It means 99% of your frames were processed faster than this number.
* Latency Jitter [ms] — This measures the predictability and stability of the inference. By taking your slowest typical frame (p99) and subtracting your median typical frame (p50), you get the variance in your processing time.
* Peak VRAM Allocated [MB] — The minimum theoretical footprint your model needs to exist. This is the maximum amount of GPU memory that actually held the forward-pass-related data.
* Peak VRAM Reserved [MB] — The realistic memory footprint your system feels. This is the maximum amount of GPU memory walled off from the computer's operating system.

## How to Run

```bash
make benchmark
```

The script will:
1. Run all combinations of backbones × fusion modes × batch sizes
2. Print results to stdout
3. Save structured results to `benchmark_results.json`
4. Print a Markdown table ready to paste into the main README

## Output Files

* `benchmark_results.json` — Full results with hardware metadata (GPU name, CUDA version, PyTorch version, timestamp)

After running the script, please kindly add the results to the [main README](https://github.com/autowarefoundation/auto_e2e/blob/main/README.md)
