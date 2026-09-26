# Training the Reactive branch on your own machine

A practical guide for running KITScenes training outside the cluster, so that the
results are comparable with everyone else's.

The pipeline assumes a cluster in a few places. None of those assumptions is a bug, but
together they are what stands between a new contributor and a first epoch using only the
Reactive branch. This page collects them, along with the corpus handling that makes a
local run possible at all.

**If you read one section, read the last one.**
[Reporting results so they can be compared](#reporting-results-so-they-can-be-compared)
is where an ADE turns into a result: the three lines from your own log that say which
scenes, which training policy and which metric contract produced it, and the score the
same validation set gets from a model with no cameras at all. Everything before it
exists so that those four things are obtainable — the corpus tooling included. A number
without them cannot be compared with anyone else's, which is the problem this page was
written for.

Every step below has been run on a laptop that started with nothing installed and no
data: environment, dataset access, the full 533-scene corpus, verification against the
frozen split, the navigation audit, local S3 and MLflow, and a training epoch whose
checkpoint uploaded. Every wall described here was hit on that machine, and every fix
was applied there — including the training epoch, which needed the BEV grid edit under
[GPU memory](#gpu-memory) because the card was a 6 GB one.

**If your GPU has less than 12 GB, read [GPU memory](#gpu-memory) before you start.**
There is a code edit you want to make before step 3, not after.

---

## The commands, in order

Copy-paste sequence. Each step names the section that explains it; read that one if
the step fails or if you want to know why it is there. Replace `/data` and `<repo>`
with your own paths.

<!-- MAINTAINERS: this block repeats commands that also appear, with their reasoning,
     in the sections below. That is deliberate — it has to be runnable on its own —
     but it means every command here exists twice. Change one, change both. -->


```bash
# 1. Environment. The SDK is not on PyPI, needs numpy<2.0, and the pipelines do
#    not import without kubernetes.                              -> "Prerequisites"
#    Build it OUTSIDE the repo: neither the venv nor the SDK clone is gitignored,
#    and a 7 GB .venv/ in `git status` is a mistake waiting to be committed.
mkdir -p /data/env && cd /data/env
python3 -m venv venv && source venv/bin/activate   # python3: `python` may not exist yet
python -m pip install --upgrade pip   # the bundled pip cannot parse the SDK's extra
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/KIT-MRT/kitscenes.git
git -C kitscenes checkout 7765cdec5490894266070ab46e23724b58b3da42
git -C kitscenes lfs pull
cd kitscenes && pip install ".[map]" && cd ..   # from INSIDE: the wheels are relative
#    torchvision and mlflow are on the training path but NOT in requirements.txt,
#    and requirements.txt pins numpy==2.2.6 which breaks the SDK. Do not apply it.
pip install torch torchvision timm webdataset boto3 pyproj mlflow \
            "flytekit==1.14.9" kubernetes "opencv-python-headless<4.10"
pip install "numpy<2"
#    Check the environment before spending days on a corpus: NumPy must be 1.x, and
#    torch must actually see the GPU -- `torch` is unpinned, so pip picks a CUDA build
#    that your driver may be too old for, and a CPU-only install trains silently.
env -u PYTHONPATH python -c "import numpy, torch, kitscenes, flytekit, kubernetes, \
webdataset, boto3, pyproj, mlflow; print(numpy.__version__, torch.__version__, \
torch.cuda.is_available())"

# 2. Dataset access. Accept the terms on the dataset page first, then log in.
#    Check the second line: an expired token reports a permissions error. -> "Prerequisites"
hf auth login
python -c "from huggingface_hub import whoami; print(whoami()['name'])"

#    On a card under 12 GB, make the BEV grid edit NOW -> "GPU memory". Packing
#    re-imports workflows.py in worker processes, so editing it mid-pack can kill a
#    multi-day run; before it starts or after it ends are the safe moments.

# 3. Corpus. Downloads, packs and deletes one scene at a time: peak disk is about
#    8 GB, and the packed result is 10.2 GB. The full split took 63 h on a laptop,
#    so budget two and a half days. Run it detached, give it its own checkout, and
#    do not train on this machine meanwhile. Resumable.         -> "Getting the corpus"
#    The dataset version is read from the frozen manifest; do not pass one.
cd <repo>
export PYTHONPATH=<repo>/Model:<repo>
setsid nohup python Platform/scripts/pack_kitscenes_corpus.py \
    --fetch --work-root /data/_staging --out-root /data/_shards \
    > /data/pack.log 2>&1 < /dev/null &
tail -f /data/pack.log          # the header must print: #   version: <the frozen one>

# 4. Check the corpus BEFORE training, not during. Eight checks; exit 0 means your
#    corpus matches the frozen split and your numbers will compare. -> "Check the corpus"
python Platform/scripts/verify_kitscenes_corpus.py --shards-root /data/_shards

# 5. Navigation audit. Mandatory: train_il refuses to run on KITScenes without it,
#    and it must cover exactly the shards you train on.          -> "Running the training"
#    shards_index_verified.json is written by step 4, and only when all eight
#    checks pass -- so if it is missing, go back to step 4.
SHARDS=$(python -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['packed']))" \
         /data/_shards/shards_index_verified.json)
pyflyte run Platform/pipelines/workflows.py \
    audit_kitscenes_navigation_quality --shards "$SHARDS"
cp <the file it printed> /data/_shards/audit.json

# 6. Local services. Checkpoint upload assumes S3 and MLflow needs a real backend;
#    a file:// tracking URI is rejected by MLflow 3.x.           -> "Cluster assumptions"
export AWS_ENDPOINT_URL=http://localhost:9000
export AUTO_E2E_CHECKPOINT_BUCKET=<your-bucket>
export MLFLOW_TRACKING_URI=sqlite:///$HOME/mlflow.db

# 7. Train. One epoch first: it is 2 h 20 min on a 6 GB card, and it is where a
#    wrong audit or a missing bucket shows up. Raise --epochs once it has run
#    through -- at that rate ten is about a day.                -> "Running the training"
pyflyte run Platform/pipelines/workflows.py wf_train_il \
    --shards "$SHARDS" \
    --navigation_quality_audit /data/_shards/audit.json \
    --dataset "KIT-MRT/KITScenes-Multimodal" \
    --validation_scope full \
    --backbone swin_v2_tiny --epochs 1 \
    --batch_size 1 --grad_accum_steps 4 --lr 1e-4 \
    --val_fraction 0.1 --num_workers 4 --training_seed 149

# 8. Report your numbers together with the three lines that make them comparable:
#    the split digest, the training policy and the epoch result. Do not stop here --
#    the section explains what each line rules out, and how to find out what your
#    validation set scores with no perception at all. -> "Reporting results"
grep -E "group_digest=|Dataset training policy:|Epoch [0-9]+/" /data/train.log
```

The training step is where a small GPU stops — see [GPU memory](#gpu-memory).

---

## GPU memory

This is the one thing the page does not solve. The KITScenes geometry fixes the camera
BEV grid at 256x256 — 65,536 queries — and `train_il` exposes no way to change it.
Measured on a 6 GB card: out of memory at `batch_size 1`, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` does not close the gap. A 12 GB card
is reported to fit with that flag; the threshold in between is not measured here. CPU
completes a smoke epoch and is far too slow for a real run.

If your card is too small, the workaround is a code change. It is an *addition*, not a
value to find and replace: `train_il` does not spell the grid out, it asks the
geometry for it. Find where `view_fusion_kwargs` is built for KITScenes in
`Platform/pipelines/workflows.py` and override the two keys straight after:

```python
view_fusion_kwargs = (
    DEFAULT_NAVIGATION_GEOMETRY.camera_bev_kwargs()
)
view_fusion_kwargs["bev_h"] = 64
view_fusion_kwargs["bev_w"] = 64
```

Leave `pc_range` alone, so the BEV still covers the same ground with coarser cells
rather than a smaller patch of it. Those two lines are the whole change. (Searching
the repository instead leads to `matching_bev_h`/`matching_bev_w` in
`Model/navigation/geometry.py`, which is the wrong place: the geometry is validated
against the map raster and rejects a smaller grid on construction.)

At 64 the grid fits, but not comfortably: a full-corpus epoch on that 6 GB card peaked
at 5.7 GB of the 6.1 available, and the allocator logged recoverable
`memory allocation failed with OOM` warnings on the way. They are warnings, not a
crash — it frees cache and retries — but there is little headroom left.

**Make this edit before step 3, not between steps.** Packing re-imports
`Platform/pipelines/workflows.py` in worker processes, so editing that file while a
multi-day pack is running can kill it. Before the pack or after it, never during.

That edit is outside the scope of this page, which documents the pipeline as it
stands; making the grid a parameter is a separate change.

---

## Prerequisites

Everything below was hit in order on a fresh machine. None of it is written down
anywhere a contributor would look, so it is collected here first.

**Put the environment outside the repository.** Neither `.venv/` nor a `kitscenes/`
clone is in `.gitignore`, so building them in the checkout puts several gigabytes into
`git status` where they can be committed by accident. Everything below assumes a
directory of your own next to the data, not inside the repo.

Create it with `python3`, not `python`: on Debian and Ubuntu there is no bare `python`
unless `python-is-python3` is installed, so the very first command fails on a machine
that has nothing set up — which is the machine this page is for. Every later `python`
in this page is the one the activated venv provides, and those are fine.

**The KITScenes SDK is not in `requirements.txt`, and not on PyPI.** It is a git
repository with LFS assets, pinned in `Platform/docker/data-prep/Dockerfile`:

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/KIT-MRT/kitscenes.git
git -C kitscenes checkout 7765cdec5490894266070ab46e23724b58b3da42
git -C kitscenes lfs pull
cd kitscenes && pip install ".[map]" && cd ..
```

The `map` extra ships prebuilt Lanelet2 wheels for CPython 3.8 to 3.12, so pick an
interpreter in that range.

**The `map` extra declares its wheels as *relative* file URLs**, and that one fact
breaks the install twice, in this order.

*First, old pip cannot parse them.* The pip that `python3 -m venv` bundles on Ubuntu
22.04 (22.0.2) rejects `lanelet2 @ file:res/ml_converter_wheels/...whl` during
dependency resolution, with a traceback ending in:

```
pip._vendor.pkg_resources.RequirementParseError: Invalid URL given
```

It names neither pip nor the SDK, and reads like a corrupt package. Verified on the
same machine: pip 22.0.2 rejects that requirement string and pip 26.2.1 accepts it.
`python -m pip install --upgrade pip` first is the whole fix.

*Then, with a pip that parses them, the path is resolved from the wrong place.* Being
relative, those URLs resolve against the current working directory rather than against
the package being built, so `pip install "./kitscenes[map]"` gets as far as the wheel
and stops:

```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or
directory: 'res/ml_converter_wheels/lanelet2-1.2.2-cp310-...whl'
```

The file is there; pip is looking one directory too high. Install from inside the
clone — `Platform/docker/data-prep/Dockerfile` does
`cd /tmp/kitscenes && pip install ".[map]"` for exactly this reason.

**Use a dedicated virtual environment, and do not simply apply `requirements.txt` on
top.** The SDK requires `numpy<2.0`; `requirements.txt` pins `numpy==2.2.6`. The two
cannot both be satisfied, and installing the requirements after the SDK silently
upgrades NumPy and breaks it — with an error that points at the SDK rather than at
the version. Install the runtime dependencies explicitly instead, and pin NumPy last:

```bash
pip install torch torchvision timm webdataset boto3 pyproj mlflow \
            "flytekit==1.14.9" kubernetes "opencv-python-headless<4.10"
pip install "numpy<2"
```

`torchvision` and `mlflow` are imported by the training path but are not in
`requirements.txt`, so neither an editable install nor the requirements file gets you
there.

OpenCV is not in `requirements.txt` either, and packing does not survive without it:
`Model/data_parsing/kit_scenes/map.py` imports `cv2`, so every partition fails with a
bare `ModuleNotFoundError` — after its 3 GB archive has already been downloaded — and
three consecutive failures abort the run. The `<4.10` bound is the one
`Platform/docker/data-prep/Dockerfile` uses, and it is what keeps OpenCV from pulling
NumPy 2 back in on top of the SDK.

**If you have ROS sourced, unset `PYTHONPATH` first.** ROS puts its own
`site-packages` on `PYTHONPATH`, and that leaks into a virtual environment and can
shadow packages — NumPy in particular. Either `unset PYTHONPATH` or prefix commands
with `env -u PYTHONPATH`. This is easy to miss precisely because the environment
looks activated.

**`kubernetes` is what makes the import fail first**, if you wonder why it is in that
list: `overlay_tasks` builds a pod template at import time, so
`Platform.pipelines.workflows` will not import without it even for a purely local run.

**Dataset access.** `KIT-MRT/KITScenes-Multimodal` is gated: the file listing is
public but downloads are not. Accept the terms on the dataset page, then
authenticate:

```bash
hf auth login
python -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

Worth checking that second line, because an **expired** token reports
`Access denied. This repository requires approval` on download — which points at the
terms rather than at the token, and sends you to the wrong place.

## Getting the archives

You need the KITScenes archives on disk first. `data_ingest` in
`Platform/pipelines/workflows.py` is the pipeline's own ingest path, but note that
`PinnedKITScenesDownloader.download()` downloads **and extracts** each scene, so
running it over the whole train split needs the same 2 TB-plus this page exists to
avoid. On a workstation, fetch the archives without extracting them:

```bash
hf download KIT-MRT/KITScenes-Multimodal \
    --repo-type dataset --revision 6fde0034446669e2ed7235e4c7fe323cd23d599d \
    --include "data/train/*.tar" --local-dir /data/KITScenes-Multimodal
```

(`huggingface-cli download` was the older spelling and no longer works on current
`huggingface_hub`.)

That revision is the one `data_processing` pins; packing against a different one is
rejected. The archives are around 3 GB each and roughly 1.6 TB in total, and they
stay compressed — only one is extracted at a time by the next step.

## The split the code enforces today

`KITSCENES_TRAINING_POLICY.validation_manifest` in `Model/training/dataset_policy.py`
names the frozen train/dev split, and `train_il` checks the packed corpus against it.
This section describes what that manifest contains and what happens if a corpus does
not match it; which split the working group ultimately adopts is a separate
discussion, in #168.

Read the filename from the policy rather than memorising it. The snapshot is re-cut
as the packed data changes, and each re-cut carries a new dataset version and a new
contract digest — a corpus packed against the previous one no longer validates.

The manifest declares:

| Field | Value | Stable across re-cuts? |
|---|---|---|
| `official_split` | `train` | yes |
| `available_scene_count` | 533 | yes |
| `excluded_empty_scene_count` | 129 | yes |
| `eligible_group_count` | 404 | yes |
| `validation_fraction` | 0.1 | yes |
| `validation_group_count` | 40 | yes |
| `validation_sample_count` | 3820 | yes |
| `dataset_version` | changes | **no** |
| `packed_contract_digest` | changes | **no** |

The counts and the holdout have survived every re-cut so far; the version and the
contract digest are exactly what a re-cut changes. Print the current ones rather than
copying them from here:

```bash
python - <<'EOF'
import json, pathlib
from training.dataset_policy import KITSCENES_TRAINING_POLICY as policy
path = pathlib.Path("Model/training") / policy.validation_manifest
frozen = json.loads(path.read_text())
print(path.name, frozen["dataset_version"], frozen["packed_contract_digest"][:16])
EOF
```

With `--validation_scope full`, `train_il` compares your packed corpus against that
manifest — partition counts, sample counts and two SHA-256 digests — and aborts on
any mismatch. In practice that means packing the 533 `data/train/` archives and
letting the code select the holdout; anything else stops the run rather than
silently training on a different corpus.

Use `--validation_scope subset` only for a bring-up run on a partial corpus. It
relaxes the *counts*, accepting a proper subset of the 533 scenes, but still requires
the provenance triple (source revision, dataset version, packed contract digest) to
match. Metrics from a subset run are **not** comparable with anyone else's.

## Getting the corpus onto a workstation

**If your card is under 12 GB, make the BEV grid edit from [GPU memory](#gpu-memory)
before you start this step.** Nothing in the corpus depends on it — it is only step 7
that cares — but packing re-imports `Platform/pipelines/workflows.py` in worker
processes, so editing that file mid-run can kill a multi-day job. Before the pack or
after it, never during.

The archives are around 3 GB each, so extracting the whole train split at once needs
well over 2 TB. Streaming the download does not help, because each archive still
lands on disk before it can be packed.

Stream the *packing* instead — extract one archive, pack it, delete the extracted
copy, move on:

```bash
export PYTHONPATH=<repo>/Model:<repo>
python Platform/scripts/pack_kitscenes_corpus.py \
    --tar-src  /data/KITScenes-Multimodal/data/train \
    --work-root /data/_staging \
    --out-root  /data/_shards
```

Peak disk stays at roughly one scene instead of the whole corpus, because the packed
output is a small fraction of the raw archives. Measured over the complete train
split — 533 partitions, 404 of them non-empty, 42,667 samples — it comes to **252 KB
per sample and 10.2 GB in total**.

Do not size a disk from a handful of scenes: per-scene cost ranges from 114 to 553 KB
per sample, so a small subset can be off by a factor of two in either direction.

The run is resumable — a partition that already has a `manifest.json` is skipped — so
it can be interrupted and restarted.

**If the archives themselves do not fit either, let it fetch them one at a time:**

```bash
python Platform/scripts/pack_kitscenes_corpus.py \
    --fetch --work-root /data/_staging --out-root /data/_shards
```

Each archive is downloaded, unpacked, deleted, and the extracted copy deleted after
packing, so peak usage is one archive plus one extraction — roughly 8 GB for the
whole 533-scene corpus rather than the 1.6 TB the archives would occupy together.

Budget time generously. The full 533-scene split took **63 hours** on a laptop over a
USB disk, fetch to packed: 7.1 minutes per scene on average, a median of 5.1, and a
spread of 1.3 to 40 minutes that tracks how long each scene is. Plan for two and a
half days, unattended.

Do not extrapolate from a handful. The first two scenes we timed averaged 3.2 minutes
and would have predicted one day; sixteen consecutive scenes gave a 6.2-minute median
and predicted three. Only the whole run settles it.

**Run it detached, because two and a half days is longer than a terminal session:**

```bash
setsid nohup python Platform/scripts/pack_kitscenes_corpus.py \
    --fetch --work-root /data/_staging --out-root /data/_shards \
    > /data/pack.log 2>&1 < /dev/null &
```

Write the log outside the checkout — `pack.log` is not in `.gitignore` either.
`setsid` and `< /dev/null` are what make it survive the shell exiting or an SSH drop;
`nohup` alone is not always enough. Follow it with `tail -f /data/pack.log`. To stop it, take
the PID from `pgrep -f "[p]ack_kitscenes_corpus"` and `kill` that — a bare
`pkill -f pack_kitscenes_corpus` also matches the shell you typed it in.

**Do not train on the same machine while it packs.** Packing is CPU and IO heavy and
holds a scene in memory; a training run alongside it exhausted RAM on a 14 GB laptop
and killed both. The pack resumes, but the hours in flight are lost.

**Do not switch branches in the checkout it is running from.** Packing spawns worker
processes, and each one re-imports the script by its path; if the file has moved or
disappeared in the meantime, every worker dies with `BrokenProcessPool` and the run
aborts after three consecutive failures. The traceback names `multiprocessing` and
reads like a resource problem, but the cause is that the script is no longer on disk.
Give a multi-day pack its own checkout — `git worktree add` is enough — and leave that
tree alone until it finishes.

Note that `shards_index.json` is written when the run finishes, so after an
interruption it lags behind what is on disk until you resume and let it complete.

Clear the staging root after killing a run, **and only while nothing is running.**
Cleanup happens per scene in a `finally`, which a killed process never reaches, so the
scene that was in flight stays extracted — a few gigabytes that no later run will
remove, since each one only cleans up after itself.

Deleting it later, with a pack in progress, is how you get a corpus that fails
verification for no visible reason. A resumed run walks the scenes in the same order,
so it reaches that same directory again and starts extracting into it; a concurrent
`rm -rf` removes files bottom-up, takes out a camera directory, and fails on the
parent with *"directory not empty"* — which reads like the delete did nothing. The
scene then packs as `missing cameras [...]. Skipping.` and lands as an empty
partition, one line among hundreds. Nothing else is affected, and the counts come up
one short at the end of the run. Wait for the run to end.

Two things the script encodes, because both are silent when wrong:

- **One scene per partition.** Calibration and map state are scene-scoped, so
  `data_processing` raises *"KITScenes partition contains scenes with different
  calibration; pack one scene per partition"* when a partition mixes them.
  `train_il` takes the resulting list of shard directories directly.
- **Pack with the KITScenes dataset version.** `data_processing` defaults to
  `DATASET_PACK_VERSION`, but the KITScenes navigation path uses
  `KITSCENES_NAVIGATION_DATASET_VERSION`, and that is what the frozen manifest
  carries. Packing with the default and then training fails with *"validation
  manifest dataset version does not match packed shards"*. The script reads the
  version out of the frozen manifest and prints it in its header, so leave
  `--dataset-version` alone unless you are deliberately packing against an older
  snapshot.

## Check the corpus before you train, not during

`train_il` validates the corpus at start-up, after scanning every shard. On a corpus
that took hours to pack, that is a late place to discover a wrong dataset version.
Run the same checks first:

```bash
python Platform/scripts/verify_kitscenes_corpus.py --shards-root /data/_shards
```

It reports the eight comparisons individually and exits non-zero if any fails. On
success it writes `shards_index_verified.json` next to the shards — that file is the
input to the next two steps, so a missing one means the corpus did not pass.

### When the frozen snapshot is re-cut

It will be, and it does not announce itself. What you see is a corpus that passed
last week failing two checks and only two:

```
  [FAIL] dataset version                          v3.0  expected v3.3
  [FAIL] contract digest           c81a5746a365246f...  expected 6fb9d857d877e570...
```

Counts and group digests still pass, which is the confusing part: the data is fine,
the contract it was packed under is not. `data_processing` stamps both at pack time,
so there is nothing to edit — **a re-cut corpus has to be repacked, not re-checked.**

Pull first and check before you spend days on it. If a pack is already running when
the re-cut lands, let it finish and repack afterwards rather than resuming into it:
resuming skips partitions that already have a `manifest.json`, so the old ones keep
the old version and you end up with a corpus that can never pass. Packing into a
fresh `--out-root` keeps the previous one intact while you decide.

## What happens to your scenes

Two filters remove data, and they are easy to confuse.

**Packing drops scenes that are too short.** A sample needs 64 history steps plus 64
future steps plus one, so a scene needs at least 129 usable frames — 12.9 s at 10 Hz
(`MIN_ROWS` in `Model/data_parsing/kit_scenes/egomotion.py`). "Usable" is the minimum
across poses, reference timestamps and the *contiguous prefix* of each camera, so one
early camera gap truncates a scene that otherwise looks long enough. This is where
`excluded_empty_scene_count: 129` comes from.

**The navigation quality audit trims further, but only for the optimizer.** It is
mandatory for KITScenes — `train_il` refuses to run without one — and removes
partitions from training. The validation groups are built from the unfiltered
non-empty partitions, so two contributors whose audits accept different scenes still
validate on the same samples.

One line in the training log gives you all three numbers:

```
Selected N/M non-empty partition(s) for the optimizer (skipped_empty=K, ...)
```

`M` is what survived packing, `K` what packing dropped, `N` what the audit kept.

## Running the training

`train_il` refuses to run on KITScenes without a navigation quality audit, and the
audit's coverage must match the shards you pass **exactly** — reusing one generated
over a different set of partitions fails with "navigation quality audit partition
coverage differs from shards". So generate it over the corpus you are about to train
on:

```bash
SHARDS=$(python -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['packed']))" \
         /data/_shards/shards_index_verified.json)

pyflyte run Platform/pipelines/workflows.py \
    audit_kitscenes_navigation_quality --shards "$SHARDS"
# copy the resulting file next to the shards, e.g. /data/_shards/audit.json
```

Over 400 partitions this takes a while and says nothing until it is done, then prints
one line and the report path:

```
KITScenes navigation quality: accepted=329 excluded=75 report=/tmp/navigation-quality-.../navigation_quality_audit.json
```

**Copy it out of `/tmp` before you reboot.** That is the only copy, and regenerating it
means running the audit over every partition again.

Keep `$SHARDS` in the same shell you train from, or rebuild it there: it is a JSON
array of a few hundred absolute paths, and the training command needs the same one the
audit was generated over.

Then train:

```bash
pyflyte run Platform/pipelines/workflows.py wf_train_il \
    --shards "$SHARDS" \
    --navigation_quality_audit /data/_shards/audit.json \
    --dataset "KIT-MRT/KITScenes-Multimodal" \
    --validation_scope full \
    --backbone swin_v2_tiny --epochs 1 \
    --batch_size 1 --grad_accum_steps 4 --lr 1e-4 \
    --val_fraction 0.1 \
    --num_workers 4 --training_seed 149
```

Without `--remote`, `pyflyte` runs locally. The CI uses `--remote` for the cluster.

`pyflyte` opens with a screenful of `UserWarning: The parameter --x is used more than
once`, once per boolean flag, before anything of yours runs. It is click reporting on
the generated CLI, not on your command. Ignore it; the first line that is about your
run is `Running Execution on local.`

**Nothing is printed between epochs.** There is no per-step counter, and the MLflow
metrics are written when an epoch ends, so on a long epoch the run is silent for hours
and an interrupted one leaves no trace of how far it got. To tell a working run from a
hung one, watch the GPU rather than the log:

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Sustained non-zero utilisation means it is training. A rate well below 100% is normal
here: at `batch_size 1` the JPEG decode in the dataloader, not the GPU, is usually the
limit.

**What an epoch costs.** On an RTX 3060 Laptop (6 GB) with the BEV grid at 64 and
`num_workers 4`, over the 329 partitions a navigation audit accepted out of 404: one
epoch took **2 h 20 min**, at 3.49 samples per second and 0.87 optimizer steps per
second. Ten epochs is therefore about a day on that machine, and the run reports both
rates itself, so you can extrapolate from your own first epoch rather than from this
one.

When the epoch ends it prints a single line with everything in it:

```
Epoch 1/1 loss=0.2037 traj=0.2037 route=0.0000 jepa=0.0000 reason=0.0000
val_ADE=1.5738 val_FDE=4.3911 score_improved=True trajectory_improved=True
samples_per_second=3.492 optimizer_steps_per_second=0.873 bad_epochs=0
checkpoint=s3://<bucket>/imitation-learning/<run>/epoch-0001.pt
```

`val_ADE` and `val_FDE` are **at 3 seconds**, not over the full prediction horizon —
see [Reporting results](#reporting-results-so-they-can-be-compared) before comparing
them with anything.

**On GPU memory**, see [GPU memory](#gpu-memory): the 256x256 grid does not fit in
6 GB, and the workaround is the `bev_h`/`bev_w` edit described there. A deformable
map fusion, which removes the quadratic term in the map-to-BEV attention at
production resolution, is a separate lever and is already available as
`map_fusion_mode`.

One caveat if you have more than one GPU: `train_il` writes its local epoch
checkpoints under a shared directory, so two trainings on the same machine can
delete each other's files and one dies mid-run with a `FileNotFoundError`. Run them
one at a time unless you are on a revision that qualifies that path per run.

## Cluster assumptions to work around

**Checkpoint storage assumes AWS, and needs to be disk-backed.** `train_il` resolves
the bucket through STS `get_caller_identity` and uploads with `put_object`, so
without AWS credentials it stops before the first epoch. Two variables avoid it:

```bash
export AUTO_E2E_CHECKPOINT_BUCKET=<bucket>   # skips the STS lookup
export AWS_ENDPOINT_URL=http://localhost:9000
```

The upload path is plain `put_object`, so any S3-compatible server works — but an
**in-memory** mock is not enough. An epoch checkpoint is 650 to 850 MB depending on
the BEV grid — 663 MB at 64, where the BEV query embedding is small — and an
in-memory server starts returning 500s after a few of them. A disk-backed server
runs a full training without trouble: verified here with MinIO writing to a USB disk,
where an epoch over the full corpus uploaded and the run went on to register the
checkpoint under its `best`, `best_trajectory` and `final` roles.

**`MLFLOW_TRACKING_URI` has no default.** It is read as `os.environ[...]`, so an
unset variable is a bare `KeyError`. Point it at a SQLite file rather than a
directory:

```bash
export MLFLOW_TRACKING_URI=sqlite:///$HOME/mlflow.db
```

A `file://` URI used to work and no longer does: MLflow 3.x refuses the filesystem
tracking backend outright — *"in maintenance mode and will not receive further
updates... migrate to a database backend"* — unless `MLFLOW_ALLOW_FILE_STORE=true`
is set. SQLite needs no server and is what MLflow recommends.

The SQLite URI covers the tracking database, not the artifacts: MLflow still writes an
`mlruns/` tree into the working directory, and that directory is not gitignored
either. Training from the repository root therefore leaves it in `git status`. Run
from elsewhere, or delete it afterwards.

**`num_workers` changes results, not just speed.** It defaults to 0. Running the same
configuration three times with each value — identical data, identical validation
groups, only that parameter differing — the runs with 0 converged worse and far more
erratically: the spread across seeds was several times larger and the training loss
stayed higher. Treat `num_workers > 0` as a correctness setting rather than a
performance one.

## Reporting results so they can be compared

**Paste your `group_digest`.** When `--validation_scope full` succeeds, training
prints:

```
Validation split: ... groups=40 group_digest=<digest>
```

An ADE on its own is not a result. What makes it one is the three lines below it that
say which scenes it was measured on, under which training policy, and under which
metric contract — because all three have changed at least once, and a number produced
before a change looks exactly like one produced after.

### Pull the report out of your own log

```bash
grep -E "group_digest=|Dataset training policy:|Epoch [0-9]+/" /data/train.log
```

That is the whole report. On the run this page was written from it prints:

```
Validation split: strategy=exact_group_fraction split_id=kitscenes_train_dev_v1
    groups=40 group_digest=903fec7dd3ca779875ed634ea4fae3c0bddd0bee58c39b3fc2573eb2ca1c1685
Dataset training policy: auto_e2e_timesteps=64 temporal_decay=0.99
    temporal_weight_normalization=mean_one acceleration_scale=0.778 curvature_scale=0.035
Epoch 1/1 loss=0.2037 val_ADE=1.5738 val_FDE=4.3911 samples_per_second=3.492 ...
```

Post those three lines together and anyone can check your numbers are commensurable
with theirs instead of assuming it.

- **`group_digest`** — same digest, same 40 scenes and 3,820 samples. A different one
  means a different exam.
- **The policy line** — not yours to choose, it comes from `KITSCENES_TRAINING_POLICY`,
  but it is revised from time to time and it changes what the loss weights. Two runs
  on the same scenes under different policies are not the same experiment.
- **`val_ADE`/`val_FDE`** — the mean and the last-step error **over 3 seconds**, not
  over the full prediction horizon. The metric contract that produced them travels
  with the validation result:

  ```
  "metric_contract": {"version": "control_rollout_validation_v2",
                      "horizon_seconds": 3.0, "horizon_steps": 30,
                      "aggregation": "sample_mean"}
  ```

  `prediction_steps` stays at 64, so the head still predicts 6.4 s; only the scored
  window is shorter. Earlier numbers in the thread quote 6.4 s aggregates that are no
  longer computed, and nothing warns you — both print as `val_ADE`. The per-horizon
  breakdown is reported alongside under `horizons` if you want to know whether an
  error invisible at 3 s grows later; the head emits (acceleration, curvature) and the
  metric integrates them twice, so it can.

### Two things the log will not tell you

**How much of your number is seed noise.** Repeating one fixed configuration across
seeds produces a spread wide enough that a single run per variant cannot separate a
real effect from initialisation luck. Run two or three seeds and report the range, not
a point.

**What the same validation set scores with no perception at all.** The metric
integrates ego dynamics, so much of it is predictable from the egomotion history
alone: holding the last observed acceleration and curvature already produces a
trajectory. Until you know that number, an ADE cannot be read as good or bad — and it
is not small. The repository does not compute it, but the packed corpus has everything
needed, and the recipe is short enough to state in full:

Run this from the repository root, with `PYTHONPATH` set as in step 3. It reads only
the packed corpus you already have — no GPU, no SDK, no original archives — and takes
a few minutes:

```python
import glob
import json
import tarfile
from pathlib import Path

import numpy as np

from evaluation.metrics import integrate_trajectory
from training.dataset_policy import KITSCENES_TRAINING_POLICY as policy

SHARDS = "/data/_shards"

frozen = json.loads(
    (Path("Model/training") / policy.validation_manifest).read_text()
)
validation = set(frozen["validation_group_uids"])

ade, fde = [], []
for tar_path in sorted(glob.glob(f"{SHARDS}/*/*.tar")):
    with tarfile.open(tar_path) as archive:
        egos, groups = {}, {}
        for member in archive:
            # Despite the name, .ego.npy is a raw buffer, not a .npy container:
            # np.load rejects it. float32[384] = history(64,4) + future(64,2),
            # history columns 0 speed, 1 acceleration, 3 curvature.
            if member.name.endswith(".ego.npy"):
                uid = member.name[: -len(".ego.npy")]
                egos[uid] = np.frombuffer(
                    archive.extractfile(member).read(), dtype=np.float32
                )
            elif member.name.endswith(".meta.json"):
                uid = member.name[: -len(".meta.json")]
                meta = json.loads(archive.extractfile(member).read())
                groups[uid] = meta.get("split_group_uid")
    for uid, ego in egos.items():
        if groups.get(uid) not in validation:   # score the holdout only
            continue
        hist = ego[: 64 * 4].reshape(64, 4)
        fut = ego[64 * 4:].reshape(64, 2)       # the logged target signals
        v0 = float(hist[-1, 0])                 # the v0 the evaluator uses
        gt = integrate_trajectory(fut[:, 0], fut[:, 1], v0)
        hold = integrate_trajectory(np.full(64, hist[-1, 1]),
                                    np.full(64, hist[-1, 3]), v0)
        err = np.linalg.norm(hold - gt, axis=1)
        ade.append(err[:30].mean())             # the contract's window
        fde.append(err[29])

print(f"samples {len(ade)}   ADE@3s {np.mean(ade):.3f}   FDE@3s {np.mean(fde):.3f}")
```

On the frozen split at the time of writing it prints:

```
samples 3820   ADE@3s 0.843   FDE@3s 2.516
```

That is the bar. A model that scores above it on the same 40 scenes is doing worse
than ignoring its cameras and holding the last observed control, which is worth
knowing before reporting the number as an improvement.
