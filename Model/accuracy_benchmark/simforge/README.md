# SIMFORGE-D1.5 evaluation

Evaluate the released [SIMFORGE-D1.5](https://huggingface.co/SIMFORGE-AI/SIMFORGE-D1.5) model
(a distilled Alpamayo 1.5) on the 117 validation scenes of KITScenes-Multimodal, with ADE/FDE
at 1 to 5 s. This is the SIMFORGE-D1.5 part of [issue #210](https://github.com/autowarefoundation/auto_e2e/issues/210).

The model gets camera images, 1.5 s of ego motion and no map, route or navigation text.

## Benchmark protocol

| Item | Value |
|---|---|
| Model | `SIMFORGE-AI/SIMFORGE-D1.5` at revision `ec31596b99d05f0a63904ce624821652d1f9e047` (all 16 files checked by SHA-256) |
| Reference code | [`NVlabs/alpamayo1.5`](https://github.com/NVlabs/alpamayo1.5) at commit `36aeb4c5938cbc2eb2aed33b22434773da4ab639` |
| Dataset | `KIT-MRT/KITScenes-Multimodal` at revision `6fde0034446669e2ed7235e4c7fe323cd23d599d` (SDK reference `7765cdec`) |
| Population | Official `val` split, 117 scenes. Anchors are frames `40 + 90k` with at least 30 future frames: **202 anchors**, scored at 1, 2, 3, 4 and 5 s with 202 / 202 / 202 / 200 / 186 anchors. The list is in `population.json`. |
| Ground truth | Recomputed from the official `poses.txt` with the direct t0 transform. The AutoE2E v3.5 `trajectory_xy_m` is not used. |
| Runtime | bf16. VLM FlashAttention 2, expert SDPA. 6 trajectories, 1 set, temperature 0.6, top-p 0.98, top-k off, seed 42, 256 generated tokens. |

Every anchor is processed in this order:

1. The model input is built from the scene and frame alone. It sees only pose, reference-timestamp
   and camera rows up to the anchor frame t0. `poses.txt` and the timestamp files are single archive
   members, so their bytes are fetched whole, but they are parsed as a prefix: parsing and
   validation stop at row t0, and later rows are never decoded, counted or checked. Camera images
   after t0 are never fetched.
2. The prediction is generated and saved.
3. Only then is the scene checked against the population's recorded future length and the
   canonical ground truth built. A mismatch with the population stops the run.

The ground truth and the future length are never visible to the model. Unit tests run this
order with a scene whose archive contains no frames after t0, and with a scene whose pose and
timestamp rows after t0 are corrupt: the prediction is still made and saved, and only the
evaluation step fails.

## Camera configurations

### 3Cam (common-comparison benchmark)

Front-left, front and front-right ring cameras go to SimForge slots 0, 1 and 2. Each camera
supplies 4 frames at t0 - 0.3, -0.2, -0.1 and 0 s, as raw JPEGs. No rectification, resizing or
cropping is applied; the model's Qwen3-VL processor resizes each image.

### 7Cam-VT (experimental adapted-camera ablation)

Slots 0 to 3 and 5 are the ring cameras front-left, front, front-right, rear-left and
rear-right. Slot 4 is a 30 degree virtual telephoto crop of `camera_ring_rear`, and slot 6 is a
30 degree crop of `camera_base_front_center`. The crops are centred on the calibrated principal
point, cut without resampling and stored losslessly.

This is an **adapted-camera ablation**. It is not an exact reproduction of SimForge's physical
telephoto sensors, and its results are labelled as experimental.

## Setup

Run these commands from the repository root. They assume Linux, Python 3.12 and an NVIDIA GPU
with CUDA 12.8 drivers (tested on an A40).

```bash
python3.12 -m venv .venv-simforge
source .venv-simforge/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install -r Model/accuracy_benchmark/simforge/requirements.txt
```

`torch` and `flash-attn` are installed separately because their wheels depend on the CUDA
driver and the Python ABI. `requirements.txt` pins the rest.

Access and downloads:

1. Accept the [KITScenes-Multimodal access terms](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal)
   and the [SIMFORGE-D1.5 license](https://huggingface.co/SIMFORGE-AI/SIMFORGE-D1.5) on Hugging Face.
2. Log in with `hf auth login`. The benchmark reads the credential from the Hugging Face
   store or from `HF_TOKEN`. Tokens are never passed on the command line or written to results.
3. Run the setup step. It fetches the pinned Alpamayo commit into `.benchmark/simforge/`, the
   KITScenes archive manifest, and the 14.4 GB model snapshot. Every file is checked against
   its pinned size and SHA-256.

```bash
export PYTHONPATH=Model
python -m accuracy_benchmark.simforge.run_kitscenes setup
```

The benchmark works in `.benchmark/simforge/` by default (git-ignored). Pass `--workspace` to use
another directory. Use `--model-snapshot-dir` to point at a snapshot you already downloaded.

## Run

The population check needs no network or GPU:

```bash
python -m accuracy_benchmark.simforge.run_kitscenes population
```

Index the scenes once, in parallel. Each scene index is about 1,900 range requests, and the index is cached:

```bash
python -m accuracy_benchmark.simforge.run_kitscenes index --camera-config 3cam --workers 8
```

Full 3Cam benchmark, then its scores and plots:

```bash
python -m accuracy_benchmark.simforge.run_kitscenes run --camera-config 3cam --device cuda
python -m accuracy_benchmark.simforge.run_kitscenes score --camera-config 3cam --plots
```

Full 7Cam-VT ablation:

```bash
python -m accuracy_benchmark.simforge.run_kitscenes run --camera-config 7cam-vt --device cuda
python -m accuracy_benchmark.simforge.run_kitscenes score --camera-config 7cam-vt --plots
```

Smoke test on one scene, or on exact anchors. `--scene`, `--limit` and `--sample-uid` are
diagnostic tools: a report is labelled as the full issue #210 benchmark only when the scored
anchors are exactly the 202 population anchors. Any other set, including an unfinished run, is
labelled `PARTIAL / DIAGNOSTIC RUN ... NOT THE FULL ISSUE #210 BENCHMARK`.

```bash
python -m accuracy_benchmark.simforge.run_kitscenes run --camera-config 3cam --scene 02be7d74-6f74-9167-c8a2-84fc961c0fbd --limit 2 --device cuda
python -m accuracy_benchmark.simforge.run_kitscenes run --camera-config 3cam --sample-uid kitscenes-v1-02be7d74-6f74-9167-c8a2-84fc961c0fbd-f000040 --device cuda
```

Resume: run the same command again. Anchors that already have a valid prediction are not
regenerated, and the model is not loaded if nothing is pending.

A results directory belongs to one run. Its `run_manifest.json` records the camera
configuration, model and generation settings, dataset revision, population and Alpamayo commit,
and a `run_id` digest of them that every prediction and ground-truth record carries. `run` and
`score` refuse a directory created for another configuration or revision. `score` also refuses
records from another run, prediction or ground-truth files that fail their recorded SHA-256,
ground-truth records that name another sample, orphan files, and anchors outside the population. The report takes its configuration from the manifest, not from
the command line.

The full 3Cam run reads about 30 GB of image data through HTTP range requests. The full 7Cam-VT run reads
somewhat more.

## Outputs

Default results directory: `.benchmark/simforge/results/<camera-config>/`

| Path | Contents |
|---|---|
| `run_manifest.json` | Configuration, population, model and dataset pins, label, code revision |
| `model_provenance.json` | Attention implementation, dtypes and image-processor settings of the loaded model |
| `predictions/<sample_uid>.npz` | Six trajectories (`pred_xyz`, `pred_rot`) and the ego history used |
| `predictions/<sample_uid>.json` | Input provenance: archive member, SHA-256, timestamp and skew of every image, crop geometry, token counts, generated text, timing |
| `ground_truth/<sample_uid>.npz` | Canonical 64-step future in the t0 frame, with validity mask |
| `ground_truth/<sample_uid>.json` | Sample UID, run ID, SHA-256 of the ground-truth file, and the population check |
| `scores/per_anchor.csv` | ADE/FDE per anchor and horizon |
| `scores/summary.json` and `scores/summary.md` | Aggregate table (see below), labelled full benchmark or partial run |
| `plots/<sample_uid>.png` | Ground truth against all six predicted trajectories |

## Expected results

ADE/FDE in metres, averaged over the anchors that have genuine ground truth at each horizon.

`ADE First` and `FDE First` use the first trajectory. `ADE Mean` and `FDE Mean` use the
point-wise mean of the six trajectories. **`minADE` and `minFDE` are oracle diagnostics**: they
take the best of the six trajectories for each anchor, which requires the ground truth. They are
not a fair metric for a model that must commit to one prediction.

**3Cam** (common-comparison benchmark)

| Horizon | n | ADE First | FDE First | ADE Mean | FDE Mean | minADE (oracle) | minFDE (oracle) |
|---|---|---|---|---|---|---|---|
| 1s | 202 | 0.168 | 0.367 | 0.162 | 0.346 | 0.125 | 0.246 |
| 2s | 202 | 0.503 | 1.336 | 0.454 | 1.151 | 0.314 | 0.742 |
| 3s | 202 | 1.083 | 3.119 | 0.920 | 2.527 | 0.619 | 1.596 |
| 4s | 200 | 1.912 | 5.594 | 1.579 | 4.488 | 1.048 | 2.843 |
| 5s | 186 | 2.979 | 8.616 | 2.402 | 6.806 | 1.599 | 4.319 |

**7Cam-VT** (experimental adapted-camera ablation)

| Horizon | n | ADE First | FDE First | ADE Mean | FDE Mean | minADE (oracle) | minFDE (oracle) |
|---|---|---|---|---|---|---|---|
| 1s | 202 | 0.159 | 0.336 | 0.153 | 0.317 | 0.115 | 0.219 |
| 2s | 202 | 0.427 | 1.047 | 0.388 | 0.918 | 0.261 | 0.569 |
| 3s | 202 | 0.831 | 2.209 | 0.725 | 1.853 | 0.464 | 1.089 |
| 4s | 200 | 1.388 | 3.829 | 1.175 | 3.145 | 0.727 | 1.810 |
| 5s | 186 | 2.077 | 5.756 | 1.729 | 4.697 | 1.061 | 2.699 |
