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

Each video follows packed `frame_idx` order. The left panel is the selected
camera and the right panel is a synthetic metric BEV; the BEV is not a camera
frame or a geographic map tile. `manifest.json` records this fixed
`panel_order` as `["camera", "metric_bev"]`.

Prediction and recorded-future controls use the same `v0`, coordinate
convention, and integrator as the Console. The manifest pins the shard and
overlay SHA-256 digests, AOVL seed, sample UIDs, ADE/FDE, and the explicit AOVL
control contract (64 acceleration/curvature steps at 10 Hz). It also verifies
and records the immutable dataset and overlay-set manifests, including model
artifact, dataset version, request identity, and cache identity.

Calibrated camera trajectories are rendered only when the shard contains a
supported projection contract. Pseudo or missing geometry is reported as
unsupported in both the frame label and `manifest.json`; the tool does not
invent synthetic intrinsics.

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
  --dataset-manifest /input/dataset-manifest.json \
  --overlay-manifest /input/overlay-manifest.json \
  --output-dir /output/trajectory-report \
  --seed-index 0 \
  --camera-index 0 \
  --max-frames-per-scene 300
```

Use repeated `--scene <scene_uid>` arguments to export selected scenes only.
For fixed frame ranges, pass `--selection-manifest` instead:

```json
{
  "schema_version": 1,
  "scenes": [
    {
      "scene_uid": "kitscenes-v1-c0123456789abcdef",
      "start_frame": 120,
      "end_frame": 240
    }
  ]
}
```

`--scene` and `--selection-manifest` are mutually exclusive. Selection uses
canonical `scene_uid` and inclusive frame bounds; legacy `episode_id` fields
are rejected.

The output directory must be empty to prevent reports from different immutable
inputs being mixed.

## Flyte usage

`wf_export_trajectory_report` accepts the shard, overlay, dataset manifest, and
overlay-set manifest as `FlyteFile` inputs. An optional selection manifest is
also a `FlyteFile`. Flyte materializes S3 objects locally and returns the report
as a `FlyteDirectory`; the core tool has no S3, MLflow, or Kubernetes
dependency.
