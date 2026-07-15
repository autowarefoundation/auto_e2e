# Trajectory visualization

This tool turns one canonical trajectory overlay (`overlay.bin.gz`) and its
matching v2.1 WebDataset shard into a self-contained report:

```text
report/
├── manifest.json
└── scenes/
    └── <scene_uid>/
        ├── thumbnail.jpg
        └── video.mp4
```

Each video follows packed `frame_idx` order and shows the selected camera next
to a metric BEV. Prediction and recorded-future controls use the same
`v0`, coordinate convention, and integrator as the Console. The manifest pins
the shard and overlay SHA-256 digests, AOVL seed, sample UIDs, and ADE/FDE.

The implementation incorporates the standalone report boundary proposed in
PR #74. Its old checkpoint inference and dataset-specific live loaders are not
used: Flyte's canonical AOVL is the only prediction source, so an export cannot
silently diverge from the Console.

## Local usage

The input paths must already be local. MP4 encoding requires
`imageio[ffmpeg]`, which is installed in the Platform data-prep image.

```bash
PYTHONPATH=Model:. python -m Tools.trajectory_visualization \
  --shard /input/part-train-000000.tar \
  --overlay /input/overlay.bin.gz \
  --output-dir /output/trajectory-report \
  --seed-index 0 \
  --camera-index 0 \
  --max-frames-per-scene 300
```

Use repeated `--scene <scene_uid>` arguments to export selected scenes only.
The output directory must be empty to prevent reports from different immutable
inputs being mixed.

## Flyte usage

`wf_export_trajectory_report` accepts the shard and overlay as `FlyteFile`
inputs. Flyte materializes S3 objects locally and returns the report as a
`FlyteDirectory`; the core tool has no S3, MLflow, or Kubernetes dependency.
