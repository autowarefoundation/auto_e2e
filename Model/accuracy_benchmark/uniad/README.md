# UniAD evaluation on KITScenes

Run the official nuScenes-pretrained [UniAD](https://github.com/OpenDriveLab/UniAD)
R101 stage-2 model, without fine-tuning, on the 117 KITScenes validation scenes.
This packages the original-image, wrapped-heading evaluation used for the
reported results. It is a zero-shot UniAD run. Related benchmark discussion: [#210](https://github.com/autowarefoundation/auto_e2e/issues/210).

## Evaluation contract

| Cohort | Anchors | Temporal input | Primary model frames |
|---|---:|---|---:|
| Dense Val v3.5 | 11,035 | Five independent 2 Hz phase streams per scene; seven past frames (3.5 s) before the first scored anchor; state continues within each stream | 15,130 |
| Complete keyframe comparison | 202 | Frame 40, then 40 + 90k, with 30 future frames available; reset at each anchor and replay nine 2 Hz frames spanning 4 s | 1,818 |

Both cohorts use all 117 physical scenes.

The primary navigation command is always **forward (2)**. The optional dense
`--oracle-diagnostic` reruns only the planner with a turn command derived from
the future ground-truth endpoint. It is labeled `oracle_turn`, uses privileged
future information and must not replace the fixed-forward result in comparisons.
It does not change the primary image/temporal replay. The historical run included
this diagnostic and the reproduction commands below include it.

### Images, cameras and state

Inputs are the original, unresized KITScenes JPEGs, in this UniAD slot order:

1. `camera_base_front_center`
2. `camera_ring_front_right`
3. `camera_ring_front_left`
4. `camera_ring_rear`
5. `camera_ring_rear_left`
6. `camera_ring_rear_right`

Decode RGB, reverse to BGR, subtract `[103.530, 116.280, 123.675]`, and use unit
standard deviations. Place each camera in a common canvas, with zero padding
after normalization, then pad bottom/right to a multiple of 32. Calibration uses
the original pixel dimensions; there is no crop, resize or replacement camera.
The evaluated metadata uses the common padded `img_shape` for all six cameras
and retains each camera's native `ori_shape`.

Projection is `K @ inverse(T_camera_to_base) @ Q`, where
`Q = [[0,1,0],[-1,0,0],[0,0,1]]` maps UniAD coordinates into KIT's forward/left/up
frame. Preparation checks the raw calibration against the mirror projections.
CAN position is relative to each stream's first pose; orientation, velocity,
acceleration and angular velocity come from current/past poses. Absolute heading
is wrapped into `[0,2π)` radians and `[0,360)` degrees. The evaluated row-vector
rotation contract is `(R @ Q).T`. All tracking/BEV history resets between streams.
The runner creates a fresh model process and seed 0 for each scene and cohort.

The official checkpoint, config and collision optimization remain enabled.
Auxiliary occupancy/map evaluation receives zero placeholders; no map/future
ground truth is a primary model input. Their collision and IoU scores are not
valid benchmark measurements and are not reported.

### Targets and metrics

- **Canonical:** original `poses.txt`, future translations transformed by the
  current yaw only. Predictions are converted from body FLU into that same frame.
- **Legacy:** unchanged v3.5 `trajectory_xy_m`, using the publisher's WGS84
  convention and matching current-pose conversion. Report it separately.
- The 202 comparison is reported against **canonical poses only**.
- Six native predictions at 0.5–3.0 s are linearly interpolated with the origin
  to 30 points at 10 Hz. ADE is the mean Euclidean error through the horizon;
  FDE is the error at that horizon. Report 1, 2 and 3 s; **4/5 s are N/A**.
- `native_2hz` evaluates only the six native timestamps. This is a scoring grid,
  not a different image replay. `interpolated_10hz` is the published primary grid;
  it does not mean UniAD was run at 10 Hz.
- Constant velocity uses the current and previous poses and is scored against
  the same target convention and cohort. Raw predictions are independently
  rescored in NumPy and checked against the hash-pinned original AutoE2E metric
  statements, including per-horizon sample counts.


## Setup

Run commands from the **AutoE2E repository root** on Linux. Use a separate Python
3.9 environment and CUDA 11.8 toolkit for this older OpenMMLab stack. Do not merge
these dependencies with AutoE2E's Python 3.12 / PyTorch 2.7 CI environment.
The validated stack used PyTorch 2.0.1+cu118, MMCV 1.6.1, MMDetection 2.26.0 and
MMDetection3D 1.0.0rc6. CUDA extensions may need compilation for your GPU.

```bash
BENCH="/path/to/large-volume/uniad-benchmark"  # choose your data/output location
mkdir -p "$BENCH"
python3.9 -m venv "$BENCH/venv"
source "$BENCH/venv/bin/activate"
pip install pip==25.3 setuptools==69.5.1 wheel==0.45.1 Cython==0.29.37 ninja
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
pip install numpy==1.22.4
pip install --no-build-isolation mmcv-full==1.6.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install -r Model/accuracy_benchmark/uniad/requirements.txt
pip install --no-deps --no-build-isolation mmdet3d==1.0.0rc6
git clone https://github.com/OpenDriveLab/UniAD.git "$BENCH/UniAD"
git -C "$BENCH/UniAD" checkout --detach 609ee083ea51c3521c323f1279dfc4cee0e60467
python -m Model.accuracy_benchmark.uniad.download_assets --out "$BENCH/assets"
```

The explicit `--no-deps` install for MMDetection3D preserves UniAD's NetworkX 2.5;
MMDetection3D's older metadata asks for `<2.3`. This is the documented dependency
metadata override used by the evaluated environment. Keep the tracked UniAD
checkout unmodified. The runner checks its revision, config and both asset hashes,
then loads the checkpoint with `strict=True`. Only the motion-anchor file path
in the resolved config is relocated. Model and dataset licenses/access terms
continue to apply; neither weights nor dataset contents are included here.

### Pinned inputs

| Input | Revision / identity |
|---|---|
| Original KITScenes-Multimodal | `6fde0034446669e2ed7235e4c7fe323cd23d599d` |
| AutoE2E model-repository v3.5 mirror | `7262c151599c1b5e4a99f0da9c3196ff9ff615fe` |
| UniAD source | `609ee083ea51c3521c323f1279dfc4cee0e60467` |
| Official HF asset repository `OpenDriveLab/UniAD2.0_R101_nuScenes` | `ea565f376a6998bade2bfc528d6b8d71590e73df` |
| Shared AutoE2E metric source | `40a75cf8a26b60f6b094125e3924fe0bdb427b02` |
| Dense sorted sample-ID SHA-256 | `28c03331e86d6296d2a1c4da29bc5b28a06dc86a38bd3dc282224a8f06f03146` |

Full checkpoint, motion-anchor, config and metric-source SHA-256 values are in
[`common.py`](common.py) and [`shared_metrics.py`](shared_metrics.py).
The metric loader verifies the source hash before extracting only its numerical
reduction statements; it does not import the training pipeline.

## Prepare the data

Accept the [KITScenes dataset terms](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal)
and authenticate with your own account using `hf auth login`.

```bash
python -m Model.accuracy_benchmark.uniad.prepare_kitscenes \
  --mirror-dir "$BENCH/mirror" --raw-dir "$BENCH/raw" \
  --out "$BENCH/prepared" --download
```

This verifies pinned downloads, selects the Val mirror shards, and extracts the
six original cameras needed for both cohorts plus poses/calibration. Mirror
images are never used for inference. The raw validation archives are about
508 GB, **plus** extracted images and mirror shards; provision space accordingly.
This portable preparation keeps downloaded archives and extracted data and does
not automatically delete anything. It does not reproduce the original rolling
cache's storage or runtime characteristics.

If the pinned data is already available, omit `--download`. Supply the mirror
snapshot root containing `datasets/kitscenes-val/v3.5/shards/manifest.json` and its
selected shard tars, and a raw directory containing `<scene-id>/poses.txt`,
`timestamp.reference.txt`, `calibration/calib.json` and camera subdirectories.
Verify local source files against the publisher's pinned checksums first.
Preparation records metadata/image hashes, checks mirror/raw identity and
enforces the frozen dense ID set. No previous experiment outputs are needed.

For a small input check, add `--scenes <scene-id> --smoke` and use a different
output directory. It prepares two dense anchors/nine model frames and one
comparison anchor/nine frames from the selected scene. This still needs the
mirror metadata shards and the selected original scene; it is not a zero-download
test. Subsets are explicitly labeled and cannot be reported as full validation.

## Run and audit

Select an available CUDA device using your normal environment configuration.
Run the smoke subset before a complete evaluation, using the same command with
its prepared-input/output paths. The complete run is:

```bash
python -m Model.accuracy_benchmark.uniad.eval_kitscenes \
  --prepared "$BENCH/prepared" --uniad-dir "$BENCH/UniAD" \
  --assets-dir "$BENCH/assets" --out "$BENCH/results" --oracle-diagnostic

python -m Model.accuracy_benchmark.uniad.audit_results rescore \
  --prepared "$BENCH/prepared" --run "$BENCH/results" \
  --metric-source "$BENCH/assets/reactive_stage_runner.py" \
  --out "$BENCH/results/independent_rescore.json"
```

Use `--track dense` or `--track keyframes` to run a single cohort; the default is
both. Every output directory must be new. Failures preserve completed and partial
outputs, with a concrete error in `status.json` or the scene's `run.json`; there
is no implicit restart or destructive cleanup. A successful `summary.json` requires
coverage, input and independent/shared metric checks to succeed; failed audits
save their error with `status: failed`. Keep the
prepared directory with its source metadata for later independent rescoring.

Each scene has raw `samples.jsonl`, strict-load/hardware/timing `run.json`, adapter
source hashes, the
resolved config and `planner_observations.jsonl` recording unmodified pre/post
optimizer trajectories and occupancy hashes. These observations do not change
the optimization or claim valid collision metrics.

For an archived smoke prediction file from the same inputs, compare without
loosening the original 0.05 m bound:

```bash
python -m Model.accuracy_benchmark.uniad.audit_results compare \
  --actual "$BENCH/smoke-results/dense/<scene-id>/samples.jsonl" \
  --reference /path/to/archived/samples.jsonl --out "$BENCH/smoke-comparison.json"
```

The comparator requires equal unique IDs and the same coordinate frame. An
archived prediction file is optional validation evidence, not a prerequisite for
preparing or evaluating the public model/dataset.

## Reference measurements and limits

These are **archived measurements**, not a new GPU execution of this portable
package. Values below use independent raw rescoring, interpolated 10 Hz,
fixed-forward commands; ADE / FDE are in metres:

| Cohort / target | 1 s | 2 s | 3 s |
|---|---|---|---|
| Dense, canonical | 1.851855 / 3.327465 | 3.475972 / 6.528387 | 5.096484 / 9.868293 |
| Dense, legacy WGS84 | 1.852337 / 3.328318 | 3.476806 / 6.529825 | 5.097617 / 9.870252 |
| 202 keyframes, canonical | 1.905734 / 3.428424 | 3.579685 / 6.734727 | 5.245813 / 10.141134 |
| Dense constant velocity, canonical | 0.165262 / 0.396435 | 0.540287 / 1.431192 | 1.116821 / 3.035694 |
| 202 constant velocity, canonical | 0.167962 / 0.406137 | 0.557045 / 1.484130 | 1.162181 / 3.194996 |

All 4/5 s results are N/A. Exact values, counts and qualification are in
[`reference_results.json`](reference_results.json). Independent/shared agreement
was within 6.53e-7 m for dense and 1.20e-6 m for keyframes. The combined original
run used one NVIDIA H100 80GB HBM3 and took **17 h 11 min 24 s**, including data
staging and verification; that is not an ETA for this portable implementation.

Constant velocity substantially outperformed this zero-shot UniAD transfer.
No fine-tuning, speed scaling or label-based exclusions were used.

## Development checks

In the repository's normal **CPU development environment**, run:

```bash
python -m pytest Model/tests/test_uniad_benchmark.py -v
make ci
```

The CPU tests cover camera/padding conventions, temporal resets, wrapped CAN,
coordinate transforms, metric horizons, exact membership and failure handling.
They do not download a dataset/checkpoint or run UniAD inference. A CUDA replay
is a separate numerical validation step; CPU checks cannot establish GPU parity.
