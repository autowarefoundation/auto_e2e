# Drive-JEPA on KITScenes validation

Evaluate the authors' official **ViT-L perception-based NAVSIM v2 checkpoint
with 20 m route commands**. This is the default. Perception-based v1,
perception-free v1, published 100 m commands, and missing-command ablations
are available through the same runner. No fine-tuning is performed.

This package follows the `Model/accuracy_benchmark/<model>/` layout. It does
not depend on PR #220 or change AutoE2E training. It imports the existing
AutoE2E dataset downloader and navigation contracts, and loads the trained
model computation from a separate, pinned
[Drive-JEPA checkout](https://github.com/linhanwang/Drive-JEPA).

## Evaluation population and inputs

- **117 scenes / 11,035 samples**, matching AutoE2E v3.5's published validation
  population and its [epoch-5 identity report][reference]. The uploaded
  140-scene mirror also includes 23 `overlap_train_val` scenes; these are
  excluded here. This is not the KITScenes paper's separate 200-window E2E
  protocol.
- Original images and recorded ego poses come from
  [KIT-MRT/KITScenes-Multimodal][dataset], loaded through the official SDK.
  Mirror images resized to 512x512 are **not** used for inference.
- Current front RGB and the frame 0.5 seconds earlier use the released
  preprocessing: crop 28 pixels at the top/bottom, resize to **512x256**,
  then ImageNet normalization. Ego velocity/acceleration use only past/current
  poses. Ground truth is the future recorded ego XY in the current ego frame.
- Perception-based checkpoints predict **32 candidates**, then select one
  using learned scores; perception-free predicts **one trajectory**. Each
  contains 8 poses at 0.5-second intervals, ending at 4 seconds.
- ADE/FDE are evaluated at 1, 2, 3 and 4 seconds on a 10 Hz grid, using linear
  interpolation including the known origin at time zero. Samples have equal
  weight. **5-second metrics are unavailable**, not extrapolated.

Full runs must match sorted UID SHA256 (including a terminal LF):
`00f824202c97b47ec74561d247e8e30686e5505eef2ad2bc0acba97e41ce9925`.
Subset runs are explicitly labelled and cannot claim full-val verification.

## Setup

The tested environment is **Windows, Python 3.11, RTX 3070 8 GB, PyTorch
2.0.1/cu117, torchvision 0.15.2, MMCV 1.7.2 and mmdet 2.28.2**. Use a separate
environment from AutoE2E training: the released perception-based model uses
legacy MMCV CUDA operators. A CPU-only PyTorch install cannot run inference.
CUDA driver support is required; a local CUDA compiler is not needed when
using the matching prebuilt wheel.

From the AutoE2E repository root in PowerShell:

```powershell
py -3.11 -m venv .venv-drive-jepa
.\.venv-drive-jepa\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r Model/accuracy_benchmark/drive_jepa/requirements-benchmark.txt
python -m pip install -r Model/accuracy_benchmark/drive_jepa/requirements-perception.txt
$env:PYTHONPATH = "$PWD\Model;$PWD\.benchmark\vendor\kitscenes"+$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python -m accuracy_benchmark.drive_jepa.setup
python -m accuracy_benchmark.drive_jepa.prepare_inputs
```

`setup` clones pinned SDK/Drive-JEPA/nuPlan sources under `.benchmark/vendor/`
and downloads only the selected planning checkpoint under `.benchmark/weights/`.
The complete checkpoint includes the encoder; the separate V-JEPA pretraining
weights are unnecessary. Existing source checkouts with a different revision
are rejected rather than reset. `setup --variant <name>` selects another
checkpoint from the options below. It does not install packages or modify the
upstream source. Run each model version in a fresh Python process.

`prepare_inputs` verifies and unpacks the audited 20 m and 100 m commands,
their provenance, and identity-only population records. The compressed files
in `inputs/` contain command labels, sample IDs and timestamps; they contain
no camera images, future trajectories, predictions or weights. An evaluation
automatically prepares these inputs when no custom command file is supplied.

Accept the dataset's access terms on [Hugging Face][dataset], then authenticate:

```powershell
hf auth login
```

For Linux, use Python 3.11 with the same torch/torchvision versions and install
the **Linux** MMCV-full 1.7.2 wheel matching cu117/torch2.0 from
[OpenMMLab's wheel index](https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html),
then the other packages in `requirements-perception.txt`, skipping its Windows
wheel URL. The Linux GPU evaluation path has not been validated in this PR.
Docker is needed only for optional route reconstruction below.

### Pinned identities

| Component | Revision |
|---|---|
| Drive-JEPA source | `548bb8215e3aae18e162a0f12f1ba83b4d3eb57e` |
| Official HF weights dataset `LinhanWang/Drive-JEPA` | `65e0d7284f69bf29d1a4864affcfd84ca4e97a2e` |
| KITScenes SDK | `7765cdec5490894266070ab46e23724b58b3da42` |
| Raw KITScenes dataset | `6fde0034446669e2ed7235e4c7fe323cd23d599d` |
| AutoE2E v3.5 mirror | `7262c151599c1b5e4a99f0da9c3196ff9ff615fe` |
| nuPlan v1.2 source | `ce3c323af01c0d7ec5672f7832ef53f9c679aab0` |

Checkpoint filenames and mandatory SHA256 values are in [paths.py](paths.py).
nuPlan is imported from source because its Windows package build encounters
a `BUILD`/`build` filename collision. A full NAVSIM simulator installation and
its metric cache are not required for this independent-sample evaluation.

## Run the default evaluation

Keep the environment and `PYTHONPATH` above active, and run from the repository root:

```powershell
python -m accuracy_benchmark.drive_jepa.eval_kitscenes
```

This runs **perception-based v2 + 20 m**, in fp32, streaming one raw scene at
a time. Default output:
`.benchmark/results/drive_jepa_perception-based-v2_20m/`.
For a real subset smoke test first:

```powershell
python -m accuracy_benchmark.drive_jepa.eval_kitscenes --scene 00fc1b9b-005e-554c-7733-c4c0adeca015 --output .benchmark/results/drive_jepa_smoke
```

The complete 117-scene download transfers **about 473 GiB** even though only
the front camera is consumed. Archives include other sensors. Streaming avoids
holding the complete dataset locally; allow roughly **35 GiB free staging
space**, plus approximately 3.5 GiB per checkpoint and any HF cache. The runner
checks space against twice each archive size plus 5 GiB before downloading.
Only temporary scene directories created by this runner are removed.

For an existing extracted SDK dataset, supply `--raw-root D:/data/kitscenes`
with `data/val/<scene_uuid>/` beneath it. Those files are never removed. Their
archive origin is recorded as user-supplied/unverified; sample identities are
still checked. Set `HF_XET_CACHE` if necessary to control additional HF cache
storage. `--staging-root` controls temporary scene storage.

### Other checkpoint and command options

| `--variant` | Planning checkpoint | Candidates | Default precision |
|---|---|---:|---|
| `perception-based-v2` (default) | `drive_jepa_perception_based_agent_vitl_v2.ckpt` | 32 | fp32 |
| `perception-based-v1` | `drive_jepa_perception_based_agent_vitl.ckpt` | 32 | fp32 |
| `perception-free-v1` | `drive_jepa_perception_free_agent_vitl.ckpt` | 1 | bf16 |

`--commands` accepts `20m` (default), `100m`, or `missing`. For example:

```powershell
python -m accuracy_benchmark.drive_jepa.eval_kitscenes --commands 100m
python -m accuracy_benchmark.drive_jepa.setup --variant perception-based-v1
python -m accuracy_benchmark.drive_jepa.eval_kitscenes --variant perception-based-v1 --commands 20m
python -m accuracy_benchmark.drive_jepa.setup --variant perception-free-v1
python -m accuracy_benchmark.drive_jepa.eval_kitscenes --variant perception-free-v1 --commands missing
```

Each condition gets a separate output directory. `missing` supplies an all-zero
four-component command vector. This differs from the **unknown one-hot**
category used for invalid supplied routes. Custom route inputs require
`--command-file path/to/commands.json` and its matching
`commands.provenance.json`; the policy and checksum must match the requested
command condition. `--checkpoint` and `--drive-jepa-root` allow existing pinned
artifacts at other paths. `--reference-report` and `--episode-ids` optionally
audit a local AutoE2E report and scene list.

## Outputs, resume and proposal audits

- `run.json`: dataset, SDK, upstream source, checkpoint, adapter, precision,
  command and preprocessing identities.
- `<scene_uuid>.npz`: IDs, timestamps, selected XY predictions and ground truth.
  Perception-based runs also retain 32 native proposals and their scores.
- `report.json`: progress, population verification and selected ADE/FDE.

Run the same command again to resume completed scenes. Interrupted scenes are
rerun. Changed source/checkpoint/command identities require a new output
directory; existing results are not silently reused for another condition.

After a **complete perception-based run**, independently audit it against
another complete run to verify exact IDs, timestamps and target arrays, score
argmax selection, selected metrics, and candidate-first/mean/oracle-min metrics:

```powershell
python -m accuracy_benchmark.drive_jepa.inspect_external_proposals --results .benchmark/results/drive_jepa_perception-based-v2_20m --baseline .benchmark/results/drive_jepa_perception-based-v2_20m
```

Using the same directory verifies selection and metrics; pass a *different*
condition's output as `--baseline` to independently check cross-run target
equality. Oracle ADE/FDE choose candidates independently per sample/horizon
using ground truth **offline only**. They are not the deployed selected result.

## Optional: regenerate 100 m labels and 20 m commands

The default uses the bundled, audited commands; reconstruction is optional and
expensive. The mirror publishes per-sample 100 m labels, rasters and scene
navigation hashes, but not the corresponding vector `scene_navigation.json`.
This workflow reconstructs it with the **original AutoE2E implementation**,
demands an exact published SHA256 for every scene, and derives 20 m commands.
It never runs a preliminary model baseline: identity-only population records
from `prepare_inputs` are sufficient.

```powershell
python -m accuracy_benchmark.drive_jepa.setup --navigation
python -m accuracy_benchmark.drive_jepa.prepare_inputs
python -m accuracy_benchmark.drive_jepa.kitscenes_route_commands --workers 2
docker build --build-context sdk=.benchmark/vendor/kitscenes -f Model/accuracy_benchmark/drive_jepa/Dockerfile.navigation -t autoe2e-kitscenes-navigation:local .
python -m accuracy_benchmark.drive_jepa.regenerate_kitscenes_routes --output .benchmark/route_inputs_20m_regenerated
python -m accuracy_benchmark.drive_jepa.eval_kitscenes --command-file .benchmark/route_inputs_20m_regenerated/commands.json --output .benchmark/results/drive_jepa_v2_regenerated20m
```

Docker Desktop must be using Linux containers with workspace bind-mount access.
The image uses Python 3.12 and the SDK's bundled Lanelet2 1.2.2 Linux wheel.
The original AutoE2E source is pinned to
`40a75cf8a26b60f6b094125e3924fe0bdb427b02` under
`.benchmark/vendor/auto_e2e_navigation/`. No upstream model or navigation source
is rewritten. Raw scenes are streamed one at a time; `--raw-root` can instead
reuse extracted data. Re-running resumes verified scene artifacts. Use
`--scene <uuid>` for an explicitly labelled reconstruction subset.

Only published sample-pose and quality/confidence scalar roundoff within
absolute `1e-12` is reconciled across platforms. Raw traces and reconciliation
records are saved; route geometry is not edited, and final full scene hashes
must still match. The 20 m rule follows centreline arc length with +/-2 m
lateral thresholds. Invalid, disconnected, or too-short routes become unknown.

## Results and interpretation

All **nine conditions** completed on the same 117 scenes / 11,035 samples.
Exact IDs, timestamps, target arrays and the published UID digest were checked;
selected metrics were recomputed from retained predictions. See
[RESULTS.md](RESULTS.md) for every horizon and candidate metric, and
[reference_results.json](reference_results.json) for exact values and hashes.
These are the existing completed runs, not a new evaluation performed for this PR.

Selected **ADE / FDE at 4 s**, metres:

| Checkpoint | Missing | 20 m | 100 m |
|---|---:|---:|---:|
| Perception-free v1 | 1.482 / 3.829 | 1.511 / 3.864 | 1.512 / 3.868 |
| Perception-based v1 | 2.243 / 5.340 | 2.276 / 5.386 | 2.268 / 5.367 |
| Perception-based v2 | 3.067 / 6.770 | **3.177 / 6.880** | 3.182 / 6.898 |

The bold value identifies the default reproducibility condition, not the best
result. Missing commands yielded the lowest selected 4 s ADE/FDE for each
checkpoint on this population.

Interpretation limits:

- Route provenance is **offline whole-scene map matching**. Commands do not
  directly inspect future evaluation targets, but routes are reconstructed
  from complete recorded traces, not independently supplied online plans.
- The 100 m labels are AutoE2E's published maneuver labels. The 20 m rule is a
  NAVSIM-style adaptation using the KITScenes recorded ego origin, without an
  invented NAVSIM rear-axle offset. Counts are 827 left / 844 right / 5,914
  straight / 3,450 unknown at 20 m; 920 / 800 / 6,335 / 2,980 at 100 m.
- Perception-based inputs use actual front calibration with the upstream
  0.4-scale/32-pixel-pad projection convention. Unequal KITScenes camera sizes
  require a front-only metadata adaptation. KITScenes ego origin and upstream
  trained vehicle geometry are retained. These are cross-dataset adaptations.
- NAVSIM v2 uses learned-score selection without its optional sequential
  simulator comfort calibration. These are independent-sample KITScenes
  ADE/FDE results, **not NAVSIM PDMS/EPDMS or closed-loop results**.
- Trained model computation and strict full checkpoint loading are preserved.
  Only redundant encoder pretraining initialization and unused simulator
  imports/global initialization are bridged. PF used bf16; PB used fp32.

## Tests

```powershell
python -m pytest Model/accuracy_benchmark/drive_jepa/tests -q
```

Tests cover interpolation support, causal ego features, population/digest
identity, masked scoring, command boundaries and invalid routes, calibration,
resume, input selection, command assets and serialization reconciliation.
They do not download weights or require CUDA. Actual official-checkpoint GPU
inference and the complete benchmark matrix were verified separately.

[dataset]: https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal
[reference]: https://huggingface.co/AutowareFoundation/auto_e2e/blob/7262c151599c1b5e4a99f0da9c3196ff9ff615fe/evaluations/overlay-replay-v1/kitscenes-val/kitscenes-epoch-5/report.json
