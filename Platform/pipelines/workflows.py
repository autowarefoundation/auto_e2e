"""AutoE2E Flyte-native workflows — Real Training Pipeline.

Architecture:
  data_ingest → data_processing → train_il → evaluate
                                      ↓
                              train_offline_rl → evaluate

MLflow: Only evaluate task logs. 2 experiments: imitation-learning, offline-rl.
"""
import enum
from flytekit import task, workflow, Resources, Secret
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from typing import NamedTuple, List, Optional

import os as _os

ECR_PREFIX = _os.environ.get("ECR_PREFIX", "381491877296.dkr.ecr.us-west-2.amazonaws.com")
TRAINING_IMAGE = f"{ECR_PREFIX}/auto-e2e/training:latest"
EVAL_IMAGE = f"{ECR_PREFIX}/auto-e2e/eval:latest"
OFFLINE_RL_IMAGE = f"{ECR_PREFIX}/auto-e2e/offline-rl:latest"
DATA_PREP_IMAGE = f"{ECR_PREFIX}/auto-e2e/data-prep:latest"

MLFLOW_URI = "http://mlflow.mlflow.svc.cluster.local:5000"

# S3 bucket for the persistent, sample_id-keyed reasoning-label cache (#98/#117).
# The teacher (Cosmos) is called at most ONCE per (dataset, teacher, prompt, sample)
# over the dataset's lifetime; re-packing hits the cache and never re-bills the
# endpoint. Empty → caching disabled (always recompute). Externalized so no
# account-specific bucket name is committed; set in .env / the task env.
REASONING_LABELS_CACHE_BUCKET = _os.environ.get(
    "REASONING_LABELS_CACHE_BUCKET", "auto-e2e-platform-datasets-381491877296")


# --- Enums ---
class Dataset(enum.Enum):
    L2D = "yaak-ai/L2D"
    NVIDIA_PHYSICAL_AI = "nvidia/PhysicalAI-Autonomous-Vehicles"


class Backbone(enum.Enum):
    SWIN_V2_TINY = "swin_v2_tiny"
    CONVNEXT_V2_TINY = "conv_next_v2_tiny"
    RESNET_50 = "res_net_50"


# NOTE: view fusion is no longer selectable. The reactive-refactor (PR #94)
# removed concat/cross_attn and hardcoded BEV fusion inside ReactiveE2E, and
# dropped the `fusion_mode` argument from AutoE2E.__init__. We keep the string
# "bev" only as a metadata label so MLflow runs stay comparable with old runs.
FUSION_LABEL = "bev"

TrainOutput = NamedTuple("TrainOutput", checkpoint=FlyteFile, metadata=FlyteFile)
EvalMetrics = NamedTuple("EvalMetrics", ade=float, fde=float, gate_pass=bool)
# wf_create_dataset returns just the ready-to-train WebDataset shards (train_il
# reads reasoning supervision from in-shard reasoning.json members). The
# versioned reasoning-label artifact persists independently in S3 (the
# generate_reasoning_labels task output + the sample_id-keyed cache), so it is
# not a workflow return value.


def _model_kwargs(config: dict) -> dict:
    """Filter a saved checkpoint `config` down to kwargs the current AutoE2E
    accepts. The reactive refactor (PR #94) removed `fusion_mode`, but old
    checkpoints (and our own metadata) may still carry it, which would make
    `AutoE2E(**config)` raise. Drop any keys the constructor no longer takes.
    """
    import inspect
    from model_components.auto_e2e import AutoE2E
    valid = set(inspect.signature(AutoE2E.__init__).parameters) - {"self"}
    return {k: v for k, v in config.items() if k in valid}


def _select_shard_dir(shards, dataset) -> str:
    """Download all shard FlyteDirectories and return the local path of the one
    whose manifest matches `dataset`.

    All datasets are passed in (each a separately-packed WebDataset), but only
    the selected dataset is used for this run. Multi-dataset training of a single
    model is tracked in issue #77 (requires dynamic-num_views BEV fusion).
    """
    import os
    import json
    target = dataset.value
    fallback = None
    for sh in shards:
        d = sh.download()
        fallback = fallback or d
        mpath = os.path.join(str(d), "manifest.json")
        if os.path.exists(mpath):
            try:
                if json.load(open(mpath)).get("dataset") == target:
                    print(f"Selected shards for dataset={target}: {d}")
                    return d
            except Exception:
                pass
    print(f"WARN: no shards matched dataset={target}; using first ({fallback})")
    return fallback


def _loader_download_dir(shard) -> str:
    """Download one shard FlyteDirectory and return its local path (merged path)."""
    return str(shard.download())


def _loader_projection(loader, device):
    """Return the loader's per-dataset projection operator on ``device``.

    Geometry is a rig constant exposed on the loader (``.projection`` /
    ``.geometry_type``) by make_pre_extracted_loader, not per batch. Datasets
    without calibration expose ``projection=None`` + ``geometry_type='pseudo'``,
    so we run the explicit pseudo path — never a silent real-geometry claim.
    """
    projection = getattr(loader, "projection", None)
    geometry_type = getattr(loader, "geometry_type", "pseudo")
    if projection is not None:
        projection = projection.to(device)
    return projection, geometry_type


# ============================================================
# Task: Data Ingest (download raw from HuggingFace)
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    # Raw video is large: L2D is ~5-7 GB/episode of multi-cam mp4, so 10 episodes
    # blew past the old 50Gi ephemeral limit (pod evicted mid-download). Size for
    # tens of episodes; the request+limit both scale so the node reserves enough.
    requests=Resources(cpu="2", mem="24Gi", ephemeral_storage="400Gi"),
    limits=Resources(ephemeral_storage="450Gi"),
    secret_requests=[Secret(group="hf-token", key="HF_TOKEN",
                            mount_requirement=Secret.MountType.ENV_VAR)],
)
def data_ingest(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
) -> FlyteDirectory:
    """Download raw dataset from HuggingFace (lerobot for L2D, physical_ai_av for NVIDIA).

    HF token comes from the `hf-token` K8s Secret (injected as env var by Flyte),
    never from a workflow input — so it is not visible in the Flyte/MLflow UI.
    """
    import os
    import shutil
    from huggingface_hub import login
    from flytekit import current_context

    token = ""
    try:
        token = current_context().secrets.get("hf-token", "HF_TOKEN")
    except Exception:
        token = os.environ.get("HF_TOKEN", "")
    if token:
        login(token=token)
        os.environ["HF_TOKEN"] = token

    out_dir = "/tmp/raw_data"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if dataset == Dataset.NVIDIA_PHYSICAL_AI:
        # NVIDIA PhysicalAI-AV: download via physical_ai_av SDK + unpack into the
        # parser layout (camera/<cam>/, labels/egomotion/) that NvidiaAVDataset reads.
        import pathlib
        from physical_ai_av import PhysicalAIAVDatasetInterface
        from data_parsing.nvidia_physical_ai.download_dataset import (
            CAMERAS, unpack_camera_zip, unpack_egomotion_zip,
        )
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ds = PhysicalAIAVDatasetInterface(
            local_dir=str(out / ".hf_cache"),
            confirm_download_threshold_gb=float("inf"),
        )
        clip_ids = ds.clip_index.index.tolist()[:episodes]
        feats = CAMERAS + ["egomotion"]
        # Real calibration: native f-theta intrinsics + sensor extrinsics. Enables
        # geometrically-meaningful BEV projection (#77). The rig is shared across
        # the subset, so we save calibration from the first clip that has it and
        # fall back to pseudo geometry downstream if none does.
        calib_saved = False
        for clip_id in clip_ids:
            ds.download_clip_features(clip_id, features=feats)
            for cam in CAMERAS:
                cf = ds.features.get_chunk_feature_filename(ds.get_clip_chunk(clip_id), cam)
                with ds.open_file(cf, maybe_stream=True) as f:
                    unpack_camera_zip(f.read(), clip_id, cam, out)
            cf = ds.features.get_chunk_feature_filename(ds.get_clip_chunk(clip_id), "egomotion")
            with ds.open_file(cf, maybe_stream=True) as f:
                unpack_egomotion_zip(f.read(), clip_id, out)
            if not calib_saved:
                try:
                    import pickle
                    ds.download_clip_features(
                        clip_id, features=["camera_intrinsics", "sensor_extrinsics"])
                    intr = ds.get_clip_feature(clip_id, "camera_intrinsics")
                    extr = ds.get_clip_feature(clip_id, "sensor_extrinsics")
                    calib_dir = out / "calibration"
                    calib_dir.mkdir(parents=True, exist_ok=True)
                    with open(calib_dir / "intrinsics.pkl", "wb") as f:
                        pickle.dump(intr, f)
                    with open(calib_dir / "extrinsics.pkl", "wb") as f:
                        pickle.dump(extr, f)
                    calib_saved = True
                    print(f"Saved NVIDIA calibration from clip {clip_id}")
                except Exception as e:
                    print(f"WARN: no calibration for clip {clip_id}: {e}")
        print(f"Ingested {dataset.value}: {len(clip_ids)} clips → {out_dir}")
        return FlyteDirectory(out_dir)

    # L2D: lerobot
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from ledataset.datasets.lerobot_dataset import LeRobotDataset

    ep_list = list(range(episodes)) if episodes > 0 else None
    ds = LeRobotDataset(repo_id=dataset.value, episodes=ep_list)
    cache_dir = ds.root
    shutil.copytree(str(cache_dir), out_dir)

    print(f"Ingested {dataset.value}: {len(ds)} frames, {episodes} episodes → {out_dir}")
    return FlyteDirectory(out_dir)


# ============================================================
# Task: Data Processing (Issue #30: pre-extract frames)
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    # 16 vCPU for the process-parallel pack workers (decode+JPEG across cores;
    # WM windows = N history + N future x V cams dominate). Memory <= the Flyte
    # task cap (32Gi); large ephemeral storage holds tens of episodes of raw
    # video + decoded windows. Karpenter provisions a fitting node.
    requests=Resources(cpu="16", mem="30Gi", ephemeral_storage="400Gi"),
    limits=Resources(cpu="16", mem="32Gi", ephemeral_storage="450Gi"),
)
def data_processing(
    raw_data: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    hz: int = 10,
    image_size: int = 256,
    episodes: int = 3,
    world_model: bool = False,
    reasoning_labels: Optional[FlyteDirectory] = None,
) -> FlyteDirectory:
    """Pre-extract aligned frames + egomotion → WebDataset shards.

    Solves Issue #30: no video decode at training time.

    Pure deterministic packing: this task calls NO external teacher. When
    ``reasoning_labels`` is provided (the artifact from
    ``generate_reasoning_labels``), each sample's frozen label is JOINed in by
    ``sample_id`` and embedded as a per-sample ``reasoning.json`` member (#98),
    the single source of truth train_il reads. Labels are generated (and
    S3-cached) once, upstream; re-packing never re-bills the teacher (#117).

    When ``world_model`` is set (L2D only for now), each sample also gets the 1 Hz
    past/future multi-view windows for the JEPA loss (#13): members
    ``hist_{t}_cam_{v}.jpg`` (oldest→newest, current last) and
    ``fut_{f}_cam_{v}.jpg`` (the frozen JEPA targets). The window config
    (num_frames/stride) matches the online dataset so shards and on-the-fly
    windows are identical.
    """
    import os
    import io
    import json
    import tarfile
    import tempfile

    raw_path = raw_data.download()
    print(f"Processing raw data from: {raw_path} (dataset={dataset.value})")

    # Enforce the sample_id JOIN invariant in CODE (not caller discipline):
    # reasoning labels are generated over the World-Model enumeration
    # (include_world_model_windows=True), and len(L2DDataset)/ordering depend on
    # the WM flag. If reasoning labels are present we MUST enumerate the same
    # sample set here, or every label would attach to the wrong frame. Today the
    # WM margins happen to be ≤ the egomotion margins so the sets coincide even
    # with WM off, but that is fragile arithmetic — force it on so a future window
    # change can't silently mislabel. (NVIDIA has no WM windows and no labels.)
    if reasoning_labels is not None and dataset != Dataset.NVIDIA_PHYSICAL_AI and not world_model:
        print("reasoning_labels present → forcing world_model=True so packing "
              "enumerates the same sample_ids the labels were generated over.")
        world_model = True

    # Build the appropriate Dataset. Both are RAW pre-extraction sources: they
    # emit unmodified frames (no backbone resize/crop/normalize). The shard packer
    # below owns the single, explicit, geometry-aware resize; the pre-extracted
    # loader owns the single ToTensor+Normalize. This avoids any double-normalize /
    # center-crop and keeps the projection ABI targeting a known (plain-resized)
    # frame. Sample schema: visual_tiles (V,3,H,W), map_tile (3,H,W),
    # egomotion_history (256), trajectory_target (128). See #77.
    ep_list = list(range(episodes)) if episodes > 0 else None
    if dataset == Dataset.NVIDIA_PHYSICAL_AI:
        from data_parsing.nvidia_physical_ai.dataset import NvidiaAVDataset
        ds = NvidiaAVDataset(data_root=raw_path)
        if world_model:
            print("world_model requested but NVIDIA has no window support yet; "
                  "packing without JEPA windows.")
        n_samples = len(ds)
        idx_iter = range(n_samples)
    else:
        from data_parsing.l2d import L2DDataset
        # World-Model windows (#16/#13) are only produced when requested, so the
        # imitation-only path stays cheap (no extra frame decode).
        ds = L2DDataset(repo_id=dataset.value, episodes=ep_list,
                        include_world_model_windows=world_model)
        n_samples = len(ds)
        idx_iter = range(n_samples)

    # Reasoning labels (#98): JOINed in from the generate_reasoning_labels
    # artifact by sample_id — NO teacher call here (this task is pure packing).
    # None → shards carry no reasoning.json (training runs imitation-only). The
    # artifact's whole-record records.jsonl is read into a {sample_id: record}
    # map; each matching sample gets a frozen reasoning.json member.
    labels_by_id = {}
    _record_to_json = None
    if reasoning_labels is not None:
        from pathlib import Path
        from data_processing.reasoning_label_generation.targets import (
            load_records_by_sample_id, record_to_json,
        )
        labels_dir = reasoning_labels.download()
        records_files = sorted(Path(labels_dir).rglob("records.jsonl"))
        if records_files:
            for rf in records_files:
                labels_by_id.update(load_records_by_sample_id(str(rf)))
            _record_to_json = record_to_json
            print(f"Reasoning labels JOIN: {len(labels_by_id)} records from "
                  f"{[str(p) for p in records_files]}")
        else:
            print(f"WARN: reasoning_labels dir {labels_dir} has no records.jsonl; "
                  "packing without reasoning.json (imitation-only).")

    # Geometry is a per-dataset rig constant, computed once. It is written into
    # EACH sample's calib.json (self-describing shards) so datasets can later be
    # merged — a merged loader resolves geometry per sample/dataset rather than
    # from a single manifest. geometry_type "pseudo" when no calibration exists.
    projection_spec = None
    build_spec = getattr(ds, "projection_spec", None)
    if callable(build_spec) and n_samples:
        projection_spec = build_spec(image_size)
    sample_geometry_type = (projection_spec or {}).get("type", "pseudo")
    calib_bytes = json.dumps(
        {"dataset": dataset.value, "geometry_type": sample_geometry_type,
         "projection": projection_spec}
    ).encode()

    out_dir = tempfile.mkdtemp()
    shard_idx = 0
    sample_count = 0
    samples_per_shard = 1000
    current_tar = None

    def open_new_shard():
        nonlocal current_tar, shard_idx
        if current_tar:
            current_tar.close()
        current_tar = tarfile.open(os.path.join(out_dir, f"train-{shard_idx:06d}.tar"), "w")
        shard_idx += 1

    open_new_shard()

    # Decode+JPEG-encode happens in the pack workers (parallel_pack); the parent
    # only appends the returned byte blobs to the current tar (single-threaded).
    def _add_member(sample_key, suffix, blob):
        ti = tarfile.TarInfo(name=f"{sample_key}.{suffix}")
        ti.size = len(blob)
        current_tar.addfile(ti, io.BytesIO(blob))

    # Process-parallel decode+encode: workers own their own dataset/reader and
    # return per-sample JPEG/npy bytes; the parent (here) appends them to the tar
    # in sample order. Decode is the bottleneck (esp. WM windows = N history + N
    # future x V cams), so parallelizing it across CPU cores turns a ~1-hour
    # single-core pack into minutes. tar writing stays single-threaded so the
    # archive is valid. ProcessPoolExecutor.map preserves input order, so shard
    # boundaries and sample_ids are identical to the serial packer.
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from data_processing.reasoning_label_generation import parallel_pack

    idx_list = list(idx_iter)
    # Worker count is MEMORY-bound for WM: even with the single delta_timestamps
    # read, each worker holds a full 1 Hz window ([~8, V, 3, 1080, 1920] ~300MB)
    # plus lerobot/torch buffers, and 16 concurrent OOM-killed the 32Gi task. Cap
    # WM packing at 6 (measured safe); imitation-only samples are light -> 16.
    max_workers_cap = 6 if world_model else 16
    pack_workers = max(1, min(max_workers_cap, len(idx_list)))
    print(f"Packing {len(idx_list)} samples with {pack_workers} parallel processes "
          f"(world_model={world_model})...")
    ctx = mp.get_context("spawn")
    pack_init = (dataset.value, ep_list, raw_path, image_size, world_model, calib_bytes)
    del ds  # workers build their own; free the parent's handle
    num_views = 0
    has_map = False
    has_wm = False
    with ProcessPoolExecutor(max_workers=pack_workers, mp_context=ctx,
                             initializer=parallel_pack.init_pack_worker,
                             initargs=pack_init) as pool:
        for si, nviews, members in pool.map(parallel_pack.pack_sample, idx_list):
            sample_key = f"s{si:08d}"
            for suffix, blob in members.items():
                _add_member(sample_key, suffix, blob)
            num_views = nviews
            has_map = has_map or ("map.jpg" in members)
            has_wm = has_wm or any(k.startswith("hist_") for k in members)
            # Per-sample reasoning label (#98): JOIN the frozen record by
            # sample_id (produced + S3-cached upstream by
            # generate_reasoning_labels). A sample with no matching label is
            # packed without reasoning.json (imitation-only for that sample).
            if _record_to_json is not None:
                record = labels_by_id.get(sample_key)
                if record is not None:
                    _add_member(sample_key, "reasoning.json",
                                json.dumps(_record_to_json(record)).encode())
            sample_count += 1
            if sample_count % samples_per_shard == 0:
                open_new_shard()

    if current_tar:
        current_tar.close()

    manifest = {"total_samples": sample_count, "shards": shard_idx,
                "hz": hz, "image_size": image_size, "dataset": dataset.value,
                "episodes": episodes,
                # num_views = real cameras only; the map view is stored under a
                # separate map.jpg key and is NOT counted here (#77).
                "num_views": num_views if sample_count else 0,
                "has_map": bool(sample_count) and has_map,
                # World-Model windows present when packed (enables JEPA training).
                "has_world_model": bool(sample_count) and has_wm}

    # Manifest also carries the projection spec (computed once above) for the
    # single-dataset loader path; the merged loader uses per-sample calib.json.
    if projection_spec is not None:
        manifest["projection"] = projection_spec
        manifest["geometry_type"] = projection_spec.get("type", "pinhole")
    else:
        manifest["geometry_type"] = "pseudo"

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    print(f"Processed {dataset.value}: {sample_count} samples → {shard_idx} shards")
    return FlyteDirectory(out_dir)


# ============================================================
# Task: Reasoning label generation (offline teacher → versioned S3 artifact)
#
# This is the SINGLE place the teacher (Cosmos) is ever called. It enumerates
# samples straight from the raw dataset (same parser / episodes / order as
# data_processing, so sample_id = s{si:08d} matches), asks the teacher for a
# label ONLY on a cache miss (LabelCache, sample_id-keyed in S3), and writes a
# versioned label artifact. data_processing later JOINs this artifact into the
# shards by sample_id — it does NOT call the teacher (#98/#117).
# ============================================================
@task(
    container_image=DATA_PREP_IMAGE,
    # 16 vCPU so the process-parallel labeler decodes WM windows across all cores
    # (decode is the bottleneck; more cores → more concurrent teacher calls to the
    # scaled-out vLLM replicas). EKS Auto Mode / Karpenter provisions a fitting
    # c/m/r node on demand. Memory kept ≤ the Flyte platform task cap (32Gi);
    # decode frames are transient so 30Gi across 16 workers is ample. Large
    # ephemeral storage holds tens of episodes of raw video + decoded windows.
    requests=Resources(cpu="16", mem="30Gi", ephemeral_storage="400Gi"),
    limits=Resources(cpu="16", mem="32Gi", ephemeral_storage="450Gi"),
    # The openai_compatible teacher endpoint (e.g. the Cosmos3-Nano vLLM ALB) is
    # injected from a K8s Secret so no concrete URL / account value is committed
    # to git or shown in the Flyte UI. Optional: only consumed when
    # teacher="openai_compatible" (mock/cached ignore it).
    secret_requests=[
        Secret(group="cosmos-teacher", key="COSMOS_TEACHER_BASE_URL",
               mount_requirement=Secret.MountType.ENV_VAR),
        Secret(group="cosmos-teacher", key="COSMOS_TEACHER_MODEL",
               mount_requirement=Secret.MountType.ENV_VAR),
    ],
)
def generate_reasoning_labels(
    raw_data: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
    split: str = "train",
    teacher: str = "openai_compatible",
    prompt_version: str = "action_relevant_reasoning_v3_temporal_front256",
    cache_bucket: str = REASONING_LABELS_CACHE_BUCKET,
    # Process-parallel worker count. Front-clip mode decodes only 5 front frames
    # per sample (not the ~48-frame WM window), so memory per worker is small and
    # we can run many: 24 workers overlap the ~12s teacher HTTP wait and keep the
    # 10 scaled-out vLLM replicas busy. (Was capped at 8 under the old WM-window
    # decode which OOM-killed the 32Gi task at 24.)
    label_workers: int = 24,
) -> FlyteDirectory:
    """Label each 1 Hz World-Model sample with a TEMPORAL front-camera clip, then
    write a versioned label artifact for the data_processing JOIN.

    Reasoning is a 1 Hz, temporal concern: the teacher is shown one FRONT-camera
    frame per horizon (0 s current + 1/2/3/4 s future) so it can reason about how
    the scene evolves (cut-ins, stops, yields) instead of guessing from a single
    instant with many cameras. Both datasets expose ``get_front_clip(idx)`` in a
    light-weight ``reasoning_clip_only`` mode that decodes ONLY those 5 front
    frames (L2D via lerobot delta_timestamps; NVIDIA via a sparse front-camera
    PyAV decode) — far cheaper than the full multi-view World-Model window.

    Sample enumeration matches ``data_processing`` exactly (same parser, same
    ``episodes``/``raw_path``, same order) so ``sample_id = s{si:08d}`` is the
    shared JOIN key. Both L2D and NVIDIA are labelled (NVIDIA is no longer
    skipped): ``reasoning_clip_only`` does not change either dataset's sample set.

    The teacher is called at most ONCE per (dataset, teacher, prompt_version,
    sample): :class:`LabelCache` keys each sample in
    ``s3://<cache_bucket>/reasoning_labels_cache/dataset=/teacher=/prompt_version=/{sample_id}.json``.
    Changing teacher or prompt_version yields a fresh cache prefix (never a stale
    mix) — so the temporal-clip / front-only / 256px change ships under a new
    prompt_version to avoid reusing the old multi-cam single-frame labels.

    Returns:
        FlyteDirectory with a whole-record ``records.jsonl`` (the JOIN
        interchange data_processing reads), the flattened
        ``reasoning_labels_v2.{parquet,jsonl}`` analytics export, and a
        provenance ``meta.json``.
    """
    import json
    import os
    import tempfile

    # Parent only needs the artifact writers; the teacher/dataset/cache/clip live
    # in the per-process workers (see parallel_label). teacher_kwargs is still
    # assembled here (from the Flyte secret context) and passed to the workers.
    from data_processing.reasoning_label_generation.parquet_writer import (
        write_jsonl, write_parquet,
    )
    from data_processing.reasoning_label_generation.targets import write_records_jsonl

    raw_path = raw_data.download()
    print(f"Generating reasoning labels: dataset={dataset.value} split={split} "
          f"teacher={teacher} prompt={prompt_version} raw={raw_path}")

    # Sample count: build the dataset once (front-clip mode) just to get len().
    # Enumeration matches data_processing (WM-window sample set) so sample_ids
    # JOIN; workers rebuild their own front-clip dataset in init_worker.
    ep_list = list(range(episodes)) if episodes > 0 else None
    if dataset == Dataset.NVIDIA_PHYSICAL_AI:
        from data_parsing.nvidia_physical_ai.dataset import NvidiaAVDataset
        ds = NvidiaAVDataset(data_root=raw_path, reasoning_clip_only=True)
    else:
        from data_parsing.l2d import L2DDataset
        ds = L2DDataset(repo_id=dataset.value, episodes=ep_list,
                        reasoning_clip_only=True)
    n_samples = len(ds)

    # openai_compatible resolves base_url/model/api_key from the Secret (env
    # fallback); mock/cached need none of these.
    teacher_kwargs = {}
    if teacher == "openai_compatible":
        from flytekit import current_context

        def _secret(key, default=None):
            try:
                return current_context().secrets.get("cosmos-teacher", key)
            except Exception:
                return os.environ.get(key, default)

        base_url = _secret("COSMOS_TEACHER_BASE_URL")
        if not base_url:
            raise ValueError(
                "teacher='openai_compatible' needs COSMOS_TEACHER_BASE_URL "
                "(cosmos-teacher K8s Secret / env); none found."
            )
        teacher_kwargs = {
            "base_url": base_url,
            "model": _secret("COSMOS_TEACHER_MODEL", "nvidia/Cosmos3-Nano"),
        }
        api_key = _secret("COSMOS_TEACHER_API_KEY")
        if api_key:
            teacher_kwargs["api_key"] = api_key
    teacher_kwargs["prompt_version"] = prompt_version
    # strict=False for bulk offline labeling: a single sample whose response the
    # model returns malformed / with <5 horizons becomes an ABSTAINED record
    # (masked out of the reasoning loss, R9) instead of raising and killing the
    # whole 1000+-sample run. meta.json reports num_abstained so a systematically
    # high rate (bad prompt/model) is still visible.
    teacher_kwargs["strict"] = False

    # Free the parent's dataset handle: each worker process builds its own.
    del ds

    # Process-parallel labeling (NOT threads): decode dominates and lerobot's
    # reader is not thread-safe, so a ThreadPool had to serialize decode under a
    # lock, leaving the scaled-out vLLM replicas idle. With processes, each worker
    # owns an independent dataset + reader, so decode runs truly in parallel across
    # CPU cores and the teacher calls overlap — finally using the extra GPUs. Only
    # the sample index crosses the process boundary; frames never do. Spawn context
    # (torch is imported) re-imports the worker module cleanly.
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from data_processing.reasoning_label_generation import parallel_label
    from data_processing.reasoning_label_generation.targets import record_from_json

    workers = max(1, min(label_workers, n_samples))
    print(f"Labeling {n_samples} samples with {workers} parallel PROCESSES "
          f"(teacher={teacher}, cache_bucket={cache_bucket or '(disabled)'})...")
    ctx = mp.get_context("spawn")
    # Order MUST match parallel_label.init_worker(repo_id, episodes, dataset_name,
    # teacher, teacher_kwargs, cache_bucket, prompt_version, raw_path).
    init_args = (dataset.value, ep_list, dataset.value, teacher, teacher_kwargs,
                 cache_bucket or None, prompt_version, raw_path)
    results = [None] * n_samples
    n_hit = n_computed = n_abstain = 0
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx,
        initializer=parallel_label.init_worker, initargs=init_args,
    ) as pool:
        for si, rec_json, status in pool.map(parallel_label.label_sample,
                                             range(n_samples)):
            results[si] = record_from_json(rec_json)
            if status == "hit":
                n_hit += 1
            elif status == "abstained":
                n_abstain += 1
            else:
                n_computed += 1
    records = results
    print(f"Labeled {len(records)} samples "
          f"(cache hits={n_hit}, computed={n_computed}, abstained={n_abstain})")
    # A few abstentions (malformed teacher JSON) are fine — they are masked out of
    # the reasoning loss. A HIGH rate means a systemic prompt/model problem, so
    # fail loudly rather than silently shipping a mostly-unlabeled dataset.
    if records and n_abstain > 0.5 * len(records):
        raise RuntimeError(
            f"{n_abstain}/{len(records)} samples abstained (>50%) — the teacher "
            f"is failing systematically (prompt/model/endpoint), not just on a few "
            f"hard frames. Aborting so the problem is fixed rather than masked.")

    out_dir = tempfile.mkdtemp()
    layout = os.path.join(
        out_dir, f"dataset={dataset.value}", f"split={split}",
        "schema_version=reasoning_label_v2", f"teacher={teacher}",
    )
    os.makedirs(layout, exist_ok=True)
    # records.jsonl = whole-record JOIN interchange data_processing reads back.
    write_records_jsonl(records, os.path.join(layout, "records.jsonl"))
    # Flattened analytics export (per-horizon rows) for querying/diffing.
    write_jsonl(records, os.path.join(layout, "reasoning_labels_v2.jsonl"))
    write_parquet(records, os.path.join(layout, "reasoning_labels_v2.parquet"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"dataset": dataset.value, "split": split, "teacher": teacher,
                   "prompt_version": prompt_version, "num_records": len(records),
                   "cache_hits": n_hit, "computed": n_computed,
                   "num_abstained": n_abstain,
                   "source": "offline teacher (generate_reasoning_labels), S3-cached"}, f)
    print(f"Wrote reasoning label artifact → {layout}")
    return FlyteDirectory(out_dir)


# ============================================================
# Task: IL Training (real AutoE2E)
# ============================================================
@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(gpu="1"),
)
def train_il(
    shards: List[FlyteDirectory],
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs: int = 3,
    batch_size: int = 4,
    # Effective batch size = batch_size * grad_accum_steps. The World-Model
    # windows (T history + F future frames x V cams) blow up activation memory,
    # forcing batch_size=1 on the L40S; but the trajectory loss needs a larger
    # effective batch to descend past ~0.84 (the bs=1 per-sample SmoothL1 gradient
    # is too noisy — the bs=4 imitation run reached 0.36). Accumulating grads over
    # N micro-batches recovers the bs=4 signal at bs=1 memory: zero_grad at the
    # window start, step once at the window end. Default 1 = plain per-batch step.
    grad_accum_steps: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    # AMP off by default: with fp16 autocast the GradScaler detected inf/nan grads
    # every step (fp16 overflow somewhere in the BEV projection / Bezier basis /
    # backbone path) and skipped optimizer.step() FOREVER — weights never updated,
    # so the trajectory loss sat perfectly flat (~2.95) while fp32 learns in one
    # step (verified: control_head grad norm ~6.6, loss 6.30->5.00). Keep fp32
    # until the specific overflow op is isolated and kept in fp32 explicitly.
    amp: bool = False,
    enable_reasoning: bool = False,
    reasoning_mode: str = "pooled_latent",
    # Small default: the reasoning branch is zero-init coupled (alpha=0), so it
    # does not move the trajectory yet, and its structured-CE term sits at a
    # large near-constant floor (~ln(num_classes) per group) until real (non-mock)
    # labels + a non-zero visual history are available. A large weight only adds
    # a constant that masks the trajectory loss in the logged total. Keep it small
    # until the reasoning branch is actually learnable.
    reasoning_loss_weight: float = 0.05,
    enable_world_model: bool = False,
    jepa_loss_weight: float = 1.0,
    # Held-out split: train on the (1 - val_fraction) majority of samples, so the
    # separate eval task can score the disjoint val split and measure
    # GENERALIZATION rather than training-set memorization (which structurally
    # favours the lower-capacity imitation model). 0.0 = train on everything
    # (legacy in-sample behaviour). The split is a stable per-sample hash of
    # __key__, so train and eval never share a sample and both tasks agree.
    val_fraction: float = 0.0,
    # Parallel JPEG decode (#121 P0). num_workers=0 decodes every sample (~55
    # JPEGs/sample with WM windows) serially on the training process, stalling the
    # GPU — the dominant per-epoch cost at scale. >0 spreads decode across worker
    # processes (sharded over shards by split_by_worker), overlapping it with the
    # GPU step. Effective parallelism is capped by shard count, so scale needs more
    # (smaller) shards too.
    num_workers: int = 0,
) -> TrainOutput:
    """Train AutoE2E model on pre-extracted WebDataset shards.

    All datasets' shards are passed in; the one matching `dataset` is selected
    (single-dataset training; multi-dataset tracked in #77).

    When ``enable_reasoning`` is set, the horizon-aware reasoning branch (#98) is
    built with the given ``reasoning_mode`` (pooled_latent /
    horizon_cross_attention) and, if the shards carry per-sample reasoning labels
    (a ``reasoning.json`` member), its HorizonReasoningLoss is added to the
    imitation loss. If reasoning is on but a batch has no labels, only the
    trajectory loss is used for that batch (the branch still runs, zero-init so
    it does not perturb the trajectory until trained).

    When ``enable_world_model`` is set and the shards carry World-Model windows
    (packed via data_processing(world_model=True)), the JEPA future-feature
    reconstruction loss (#13) is added: the model runs the stateless windowed
    path (encode_history → aggregate → predict_future), and jepa_loss compares
    the prediction against the frozen target on the real future frames. The WM
    also supplies the Encoded Visual History to the planner and reasoning branch
    (otherwise visual_history is zeros).
    """
    import os
    import json
    import torch
    import numpy as np
    from flytekit import current_context

    # DataLoader workers (num_workers>0) transport batches to the parent via shared
    # memory (/dev/shm) by default; the Flyte pod's /dev/shm is tiny (~64MB), so
    # WM-window batches overflow it → "Bus error / worker killed by signal"
    # (#121 P0, documented in Platform/HowToUseFlyte.md). Switch torch's tensor
    # sharing to the file_system strategy, which passes tensors via mmap'd temp
    # files instead of /dev/shm — the standard fix for constrained-shm containers.
    # No-op when num_workers=0.
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")

    from model_components.auto_e2e import AutoE2E
    from model_components.losses import TrajectoryImitationLoss
    from data_parsing.pre_extracted import make_multi_dataset_loader
    # _loader_download_dir is a module-level helper in THIS file, not in
    # pre_extracted — call it directly (importing it from there is an ImportError).

    ctx = current_context()
    bb, fm = backbone.value, FUSION_LABEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training: backbone={bb} fusion={fm} epochs={epochs} bs={batch_size} device={device}")

    # MERGED DataLoader over ALL provided shard dirs. Each dataset keeps its own
    # geometry/num_views; batches are same-dataset (uniform), interleaved across
    # datasets, each carrying its projection — so L2D (6cam pseudo) and NVIDIA
    # (7cam f-theta) train together. The model is runtime-V-dynamic (projection
    # ABI, #77), so a single model consumes both. num_views only sizes defaults.
    shard_dirs = [_loader_download_dir(s) for s in shards]
    # Train on the "train" split when a held-out fraction is requested (eval scores
    # the disjoint "val" split); val_fraction=0 keeps the legacy all-samples path.
    _split = "train" if val_fraction > 0.0 else "all"
    merged = make_multi_dataset_loader(shard_dirs, batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=(device.type == "cuda"),
                                       split=_split, val_fraction=val_fraction)
    print(f"Merged {len(shard_dirs)} dataset(s) into one training stream "
          f"(split={_split}, val_fraction={val_fraction}, num_workers={num_workers}).")

    # Peek the first batch to size num_views defaults.
    _peek, _peek_proj, _peek_geom = next(iter(merged))
    num_views = int(_peek["visual_tiles"].shape[1])
    print(f"Detected num_views={num_views} (first dataset); geometry={_peek_geom}")

    # Consistency guard (packing ↔ training) across EVERY merged dataset. A batch
    # missing an enabled branch's data trains that branch unsupervised (the loss
    # is silently skipped per-batch), so verify EACH shard dir carries what the
    # enabled branches need — not just the first peeked batch. World-Model
    # windows are recorded in each manifest (has_world_model); reasoning labels
    # are per-sample, so peek one batch per dataset loader.
    for d in shard_dirs:
        mpath = os.path.join(d, "manifest.json")
        manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
        dname = manifest.get("dataset", d)
        if enable_world_model and not manifest.get("has_world_model", False):
            raise ValueError(
                f"enable_world_model=True but dataset '{dname}' ({d}) has no "
                f"World-Model windows. Re-pack that dataset with world_model=True "
                f"(NVIDIA has no window support yet — exclude it or disable WM)."
            )
        if enable_reasoning:
            probe = make_multi_dataset_loader([d], batch_size=1, num_workers=0)
            first, _, _ = next(iter(probe))
            if "reasoning__source_weight" not in first:
                raise ValueError(
                    f"enable_reasoning=True but dataset '{dname}' ({d}) carries no "
                    f"reasoning labels. Re-pack it with reasoning_teacher set."
                )

    # Model. fusion_mode is gone (BEV hardcoded inside ReactiveE2E); the model
    # now also owns the map branch, so its forward requires a map_input tensor.
    model = AutoE2E(
        backbone=bb, num_views=num_views, embed_dim=256,
        is_pretrained=True,
        enable_reasoning=enable_reasoning, reasoning_mode=reasoning_mode,
        enable_world_model=enable_world_model,
    ).to(device)
    print(f"Reasoning: {'on' if enable_reasoning else 'off'}"
          + (f" (mode={reasoning_mode})" if enable_reasoning else ""))
    print(f"World Model: {'on' if enable_world_model else 'off'}")

    # Optimizer + Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = TrajectoryImitationLoss(loss_type="smooth_l1")
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)

    # Reasoning loss (#98): computed outside the model on the aux reasoning_pred
    # against the shard's per-sample labels. Built only when reasoning is on.
    reasoning_loss_fn = None
    target_batch_from_loader = None
    if enable_reasoning:
        from training.losses.horizon_reasoning_loss import HorizonReasoningLoss
        from data_processing.reasoning_label_generation.targets import (
            target_batch_from_loader as _tb_from_loader,
        )
        reasoning_loss_fn = HorizonReasoningLoss()
        target_batch_from_loader = _tb_from_loader

    # Training loop
    model.train()
    losses_per_epoch = []
    scaler = torch.amp.GradScaler(enabled=amp)

    _proj_cache = {}
    _first_step = True  # gate the one-time gradient-flow probe below
    accum = max(1, int(grad_accum_steps))
    if accum > 1:
        print(f"Gradient accumulation: {accum} micro-batches "
              f"(effective batch size = {batch_size * accum})")
    for epoch in range(epochs):
        epoch_losses = []
        traj_losses = []
        jepa_vals = []
        reason_vals = []
        micro_idx = 0  # position within the current accumulation window
        # Merged loader yields (batch, projection, geometry_type): each batch is
        # same-dataset (uniform num_views/geometry) but datasets are interleaved,
        # so the per-batch projection is applied to the batch it belongs to.
        for batch, batch_proj, batch_geom in merged:
            visual = batch["visual_tiles"].to(device)        # (B, V, 3, H, W)
            ego_hist = batch["egomotion_history"].to(device)  # (B, 256)
            vis_hist = batch["visual_history"].to(device)     # (B, 896)
            target = batch["trajectory_target"].to(device)    # (B, 128)
            map_input = batch["map_input"].to(device)

            # Per-batch geometry, moved to device once per operator (cached).
            if batch_proj is not None:
                key = id(batch_proj)
                if key not in _proj_cache:
                    _proj_cache[key] = batch_proj.to(device)
                proj_dev = _proj_cache[key]
            else:
                proj_dev = None

            # World-Model windows (#13): present only on world_model shards. The
            # windowed path makes JEPA loss differentiable and also supplies the
            # Encoded Visual History to the planner + reasoning branch.
            history_frames = batch.get("history_frames")
            future_frames = batch.get("future_frames")
            if history_frames is not None:
                history_frames = history_frames.to(device)
            if future_frames is not None:
                future_frames = future_frames.to(device)

            # Accumulation window: zero grads only at its start, step at its end.
            if micro_idx == 0:
                optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(visual, map_input, vis_hist, ego_hist,
                            projection=proj_dev, geometry_type=batch_geom,
                            mode="train", trajectory_target=target,
                            history_frames=history_frames, future_frames=future_frames)
                # Train mode returns (trajectory, aux) when a branch (reasoning /
                # world model) is on; otherwise just the trajectory tensor.
                trajectory, aux = out if isinstance(out, tuple) else (out, {})
                traj_loss = loss_fn(trajectory, target)
                loss = traj_loss

                # JEPA loss (#13): future-feature reconstruction, added when the
                # WM ran the windowed path AND this batch carries future frames.
                jepa_val = 0.0
                future_state_pred = aux.get("future_state_pred")
                if (enable_world_model and future_state_pred is not None
                        and future_frames is not None):
                    jepa = model.World_Action_Model_E2E.jepa_loss(
                        future_state_pred, future_frames)
                    loss = loss + jepa_loss_weight * jepa
                    jepa_val = float(jepa.item())

                # Add the reasoning loss when the branch is on AND this batch
                # carries labels (shards packed with a teacher). The branch is
                # zero-init, so with no labels the trajectory is unaffected.
                reason_val = 0.0
                reasoning_pred = aux.get("reasoning_pred")
                if reasoning_loss_fn is not None and reasoning_pred is not None:
                    tb = target_batch_from_loader(batch)
                    if tb is not None:
                        terms = reasoning_loss_fn(
                            reasoning_pred,
                            {g: t.to(device) for g, t in tb.targets.items()},
                            source_weights=tb.source_weights.to(device),
                            confidence_targets=tb.confidence_targets.to(device),
                        )
                        loss = loss + reasoning_loss_weight * terms["total"]
                        reason_val = float(terms["total"].item())

            # Divide by accum so summed micro-batch grads equal the MEAN gradient
            # of an effective batch of (batch_size * accum) — same scale as a plain
            # step, so lr/grad_clip keep their meaning. Log the unscaled loss.
            scaler.scale(loss / accum).backward()

            epoch_losses.append(loss.item())
            traj_losses.append(traj_loss.item())
            jepa_vals.append(jepa_val)
            reason_vals.append(reason_val)

            # Step only at the end of an accumulation window (or plain step when
            # accum==1). Grads persist across micro-batches until then.
            micro_idx += 1
            if micro_idx < accum:
                continue
            micro_idx = 0

            scaler.unscale_(optimizer)
            # One-time gradient-flow probe (very first optimizer step): prove each
            # enabled branch actually receives gradient (not just the trajectory
            # head). We report the grad-norm of a parameter unique to each branch —
            # a zero/None here means that branch is not training even though its
            # loss is being added.
            if _first_step:
                def _branch_gn(substr):
                    tot, n = 0.0, 0
                    for nm, p in model.named_parameters():
                        if substr in nm and p.grad is not None:
                            tot += float(p.grad.norm().item()) ** 2
                            n += 1
                    return (tot ** 0.5, n)
                planner_gn = _branch_gn("TrajectoryPlanner")
                probe = f"grad-flow probe: planner={planner_gn}"
                if enable_world_model:
                    probe += f" world_model={_branch_gn('World_Action_Model')}"
                if enable_reasoning:
                    probe += f" reasoning={_branch_gn('Reasoning')}"
                print(probe)
                _first_step = False
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        # Flush a trailing partial accumulation window (epoch batch count not a
        # multiple of accum) so its grads aren't silently dropped at epoch end.
        if micro_idx > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            micro_idx = 0

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_traj = np.mean(traj_losses) if traj_losses else 0.0
        avg_jepa = np.mean(jepa_vals) if jepa_vals else 0.0
        avg_reason = np.mean(reason_vals) if reason_vals else 0.0
        losses_per_epoch.append(float(avg_loss))
        # Log each branch's sub-loss separately: the total carries traj + JEPA
        # (world model) + reasoning aux terms, so per-branch values show whether
        # EACH branch is actually being optimized (not just the trajectory head).
        print(f"  Epoch {epoch+1}/{epochs} loss={avg_loss:.4f} "
              f"traj_loss={avg_traj:.4f} jepa={avg_jepa:.4f} reason={avg_reason:.4f}")

    # Save checkpoint
    os.makedirs("/tmp/train", exist_ok=True)
    ckpt_path = "/tmp/train/best.pt"
    # `config` must be reconstruction kwargs for AutoE2E(**config); fusion_mode
    # is no longer a constructor arg, so it lives only in metadata below. The
    # branch flags MUST be recorded so a later stage (offline RL / eval) rebuilds
    # the SAME architecture — otherwise load_state_dict fails on missing keys.
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "backbone": bb, "embed_dim": 256, "num_views": num_views,
            "enable_reasoning": enable_reasoning, "reasoning_mode": reasoning_mode,
            "enable_world_model": enable_world_model,
        },
        "epoch": epochs,
    }, ckpt_path)

    # Metadata
    meta = {
        "data": {"dataset": dataset.value, "shard_dirs": shard_dirs,
                 "merged_datasets": len(shard_dirs)},
        "model": {"backbone": bb, "fusion_mode": fm, "embed_dim": 256, "num_views": num_views},
        "training": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "grad_clip": grad_clip, "amp": amp,
            "optimizer": "AdamW", "final_loss": losses_per_epoch[-1] if losses_per_epoch else 0,
            "losses_per_epoch": losses_per_epoch,
            # Recorded so the (separate) eval task scores the SAME held-out split
            # this run trained around — eval reads this to pick split="val".
            "val_fraction": val_fraction,
        },
        "context": {
            "flyte_execution_id": ctx.execution_id.name if ctx.execution_id else "local",
            "docker_image": TRAINING_IMAGE,
        },
    }
    meta_path = "/tmp/train/metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return TrainOutput(checkpoint=FlyteFile(ckpt_path), metadata=FlyteFile(meta_path))


# ============================================================
# Task: Offline RL
# ============================================================
@task(
    container_image=OFFLINE_RL_IMAGE,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(gpu="1"),
)
def train_offline_rl(
    pretrained: FlyteFile,
    shards: List[FlyteDirectory],
    il_metadata: FlyteFile,
    dataset: Dataset = Dataset.L2D,
    epochs: int = 3,
    tau: float = 0.7,
    beta: float = 3.0,
) -> TrainOutput:
    """Offline RL refinement of the IL checkpoint via advantage-weighted regression
    against a frozen IL prior (AWR — not full IQL; no learned value network)."""
    import os
    import json
    import torch
    import numpy as np
    from flytekit import current_context

    ckpt_path = pretrained.download()
    shard_dir = _select_shard_dir(shards, dataset)
    il_meta = json.load(open(il_metadata.download()))
    ctx = current_context()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Offline RL (AWR, frozen prior): epochs={epochs} beta={beta}")

    # Load IL model
    from model_components.auto_e2e import AutoE2E
    from data_parsing.pre_extracted import make_pre_extracted_loader

    import copy

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = AutoE2E(**_model_kwargs(config)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # FROZEN behavior prior = the IL checkpoint at t=0, kept fixed. The advantage
    # must be measured against a policy that does NOT move with the one being
    # trained; using the LIVE model for both terms makes advantage identically 0
    # (a no-op that silently reduces to plain BC). This frozen prior gives a real
    # signal: "does the fine-tuned policy beat the IL prior on this sample?".
    baseline_model = copy.deepcopy(model).to(device).eval()
    for p in baseline_model.parameters():
        p.requires_grad_(False)

    loader = make_pre_extracted_loader(shard_dir, batch_size=4, num_workers=0)
    projection, geometry_type = _loader_projection(loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-3)

    # Advantage-weighted regression (AWR) against the frozen IL prior.
    model.train()
    losses_per_epoch = []
    for epoch in range(epochs):
        epoch_losses = []
        for batch in loader:
            # Reset the WM per-sequence rolling buffer per batch (see eval note):
            # avoids cross-batch history leakage and ragged-batch cat crashes.
            if hasattr(model, "reset_visual_history"):
                model.reset_visual_history()
            if hasattr(baseline_model, "reset_visual_history"):
                baseline_model.reset_visual_history()
            visual = batch["visual_tiles"].to(device)
            ego_hist = batch["egomotion_history"].to(device)
            vis_hist = batch["visual_history"].to(device)
            target = batch["trajectory_target"].to(device)
            map_input = batch["map_input"].to(device)

            optimizer.zero_grad()
            # Offline RL regresses only the trajectory; run mode="infer" so the
            # forward returns a bare trajectory tensor even when the checkpoint
            # was trained with reasoning / world-model branches on (mode="train"
            # would return a (trajectory, aux) tuple and break the arithmetic).
            # The inference forward is still differentiable for the policy grad.
            pred = model(visual, map_input, vis_hist, ego_hist,
                         projection=projection, geometry_type=geometry_type,
                         mode="infer")
            # Advantage-weighted regression against the FROZEN IL prior. advantage
            # > 0 where the trained policy is already closer to the logged action
            # than the prior; exp(beta*advantage) up-weights those samples. Using
            # the frozen prior (not the live model) makes the advantage real and
            # non-zero, and makes beta actually do something.
            with torch.no_grad():
                baseline_pred = baseline_model(visual, map_input, vis_hist, ego_hist,
                                               projection=projection, geometry_type=geometry_type,
                                               mode="infer")
            advantage = -(pred.detach() - target).pow(2).mean(dim=-1) \
                + (baseline_pred - target).pow(2).mean(dim=-1)
            weights = torch.exp(beta * advantage).clamp(max=100.0)
            loss = (weights * (pred - target).pow(2).mean(dim=-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        losses_per_epoch.append(float(avg_loss))
        print(f"  Epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")

    os.makedirs("/tmp/rl", exist_ok=True)
    out_path = "/tmp/rl/policy_rl.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epochs}, out_path)

    meta = {
        "base_model": {"il_metadata": il_meta, "il_checkpoint": str(ckpt_path)},
        # AWR against a frozen IL prior — NOT full IQL: there is no learned value
        # / expectile network, so tau is not used (recorded as null for honesty;
        # a true IQL value head is future work).
        "rl": {"method": "awr_frozen_prior", "epochs": epochs, "tau": None, "beta": beta,
                "losses_per_epoch": losses_per_epoch},
        "context": {
            "flyte_execution_id": ctx.execution_id.name if ctx.execution_id else "local",
            "docker_image": OFFLINE_RL_IMAGE,
        },
    }
    meta_path = "/tmp/rl/metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return TrainOutput(checkpoint=FlyteFile(out_path), metadata=FlyteFile(meta_path))


# ============================================================
# Task: Evaluate (THE ONLY MLflow logging point)
# ============================================================
def _run_evaluation(checkpoint, shards, train_metadata, dataset, experiment_name):
    """Shared open-loop evaluation + MLflow logging logic.

    Called by both evaluate_il_policy and evaluate_rl_policy. Kept as a plain
    module-level function (not a @task) so the two evaluation tasks share one
    implementation while appearing as distinct nodes in the Flyte UI.
    """
    import os
    import json
    import yaml
    import torch
    import numpy as np
    import mlflow
    from flytekit import current_context

    from model_components.auto_e2e import AutoE2E
    from data_parsing.pre_extracted import make_pre_extracted_loader
    from evaluation.metrics import integrate_trajectory

    # Eval uses num_workers=4; use the file_system sharing strategy so the small
    # pod /dev/shm doesn't bus-error on WM-window batches (same as train_il, #121).
    torch.multiprocessing.set_sharing_strategy("file_system")

    ckpt_path = checkpoint.download()
    shard_dir = _select_shard_dir(shards, dataset)
    meta = json.load(open(train_metadata.download()))
    ctx = current_context()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = AutoE2E(**_model_kwargs(config)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate on the HELD-OUT split this checkpoint trained around (recorded in
    # the train metadata), so ADE/FDE measure generalization, not training-set
    # memorization. val_fraction=0 (legacy) → score all samples (in-sample).
    val_fraction = float(meta.get("training", {}).get("val_fraction", 0.0) or 0.0)
    eval_split = "val" if val_fraction > 0.0 else "all"
    loader = make_pre_extracted_loader(shard_dir, batch_size=8, num_workers=4, shuffle=0,
                                       pin_memory=(device.type == "cuda"),
                                       split=eval_split, val_fraction=val_fraction)
    print(f"Eval split={eval_split} (val_fraction={val_fraction}) — "
          f"{'held-out generalization' if eval_split == 'val' else 'in-sample'}")
    projection, geometry_type = _loader_projection(loader, device)
    all_ade, all_fde = [], []

    with torch.no_grad():
        for batch in loader:
            # WM rolling buffer is per-sequence state; reset per batch so batch N's
            # planner history is not built from unrelated prior batches (leakage),
            # and a ragged final batch cannot crash torch.cat over mixed batch dims.
            if hasattr(model, "reset_visual_history"):
                model.reset_visual_history()
            visual = batch["visual_tiles"].to(device)
            ego_hist = batch["egomotion_history"].to(device)
            vis_hist = batch["visual_history"].to(device)
            target = batch["trajectory_target"]  # (B, 128) on CPU
            map_input = batch["map_input"].to(device)

            # Train/eval consistency (#13): if the shard carries World-Model
            # windows, feed them so eval runs the SAME windowed path the model
            # trained on — the planner sees the DENSE WM-derived visual_history it
            # learned to use, not the mostly-zero rolling-buffer vector. Without
            # this, a WM-trained model is evaluated out-of-distribution (the
            # planner's visual_history is 3/4 zeros) and ADE inflates. On shards
            # without WM windows (imitation-only) batch.get returns None and the
            # model takes its normal path — identical to before. Future prediction
            # is gated on mode=="train", so mode="infer" safely skips it.
            history_frames = batch.get("history_frames")
            future_frames = batch.get("future_frames")
            if history_frames is not None:
                history_frames = history_frames.to(device)
            if future_frames is not None:
                future_frames = future_frames.to(device)

            pred = model(visual, map_input, vis_hist, ego_hist,
                         projection=projection, geometry_type=geometry_type,
                         history_frames=history_frames, future_frames=future_frames,
                         mode="infer")
            pred = pred.cpu().numpy()  # (B, 128)
            target_np = target.numpy()

            for i in range(pred.shape[0]):
                # Reshape: (64, 2) = [accel_x, curvature]
                pred_signals = pred[i].reshape(64, 2)
                gt_signals = target_np[i].reshape(64, 2)
                # Get initial speed from egomotion history (first signal)
                ego_np = batch["egomotion_history"][i].numpy()
                v0 = float(ego_np[-4])  # last speed value in history

                pred_traj = integrate_trajectory(pred_signals[:, 0], pred_signals[:, 1], v0)
                gt_traj = integrate_trajectory(gt_signals[:, 0], gt_signals[:, 1], v0)

                ade = float(np.mean(np.linalg.norm(pred_traj - gt_traj, axis=1)))
                fde = float(np.linalg.norm(pred_traj[-1] - gt_traj[-1]))
                all_ade.append(ade)
                all_fde.append(fde)

    avg_ade = float(np.mean(all_ade)) if all_ade else 99.0
    avg_fde = float(np.mean(all_fde)) if all_fde else 99.0
    passed = avg_ade < 2.0 and avg_fde < 4.0

    # --- MLflow logging ---
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)

    model_info = meta.get("model", meta.get("base_model", {}).get("il_metadata", {}).get("model", {}))
    bb = model_info.get("backbone", "?")
    fm = model_info.get("fusion_mode", "?")
    training = meta.get("training", meta.get("base_model", {}).get("il_metadata", {}).get("training", {}))
    run_name = f"{bb}-{fm}-e{training.get('epochs','?')}"

    with mlflow.start_run(run_name=run_name):
        # Flatten params
        params = {}
        data = meta.get("data", meta.get("base_model", {}).get("il_metadata", {}).get("data", {}))
        params["data/dataset"] = data.get("dataset", "?")
        params["model/backbone"] = bb
        params["model/fusion_mode"] = fm
        params["train/epochs"] = training.get("epochs", "?")
        params["train/batch_size"] = training.get("batch_size", "?")
        params["train/lr"] = training.get("lr", "?")
        params["train/weight_decay"] = training.get("weight_decay", "?")
        params["train/amp"] = training.get("amp", "?")
        params["train/final_loss"] = training.get("final_loss", "?")

        # RL params
        if "rl" in meta:
            rl = meta["rl"]
            params["rl/method"] = rl.get("method", "?")
            params["rl/tau"] = rl.get("tau", "?")
            params["rl/beta"] = rl.get("beta", "?")
            params["rl/epochs"] = rl.get("epochs", "?")

        # Context
        train_ctx = meta.get("context", {})
        params["ctx/train_execution_id"] = train_ctx.get("flyte_execution_id", "?")
        params["ctx/train_docker_image"] = train_ctx.get("docker_image", "?")
        params["ctx/eval_execution_id"] = ctx.execution_id.name if ctx.execution_id else "local"
        params["ctx/eval_docker_image"] = EVAL_IMAGE

        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.set_tags({"pipeline": experiment_name, "backbone": bb, "fusion": fm})

        # Training loss curve
        for i, loss_val in enumerate(training.get("losses_per_epoch", [])):
            mlflow.log_metric("train/loss", loss_val, step=i)

        # Eval metrics
        mlflow.log_metrics({"eval/ade": avg_ade, "eval/fde": avg_fde, "eval/gate_pass": 1.0 if passed else 0.0})

        # Artifacts
        os.makedirs("/tmp/eval-artifacts", exist_ok=True)
        with open("/tmp/eval-artifacts/config.yaml", "w") as f:
            yaml.dump(meta, f)
        mlflow.log_artifact("/tmp/eval-artifacts/config.yaml")
        mlflow.log_artifact(ckpt_path, artifact_path="model")

        # Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        try:
            mlflow.register_model(model_uri, "auto-e2e-driving-policy")
        except Exception as e:
            print(f"Registry: {e}")

    print(f"Eval: ADE={avg_ade:.3f} FDE={avg_fde:.3f} Gate={'PASS' if passed else 'FAIL'}")
    return EvalMetrics(ade=avg_ade, fde=avg_fde, gate_pass=passed)


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="2", mem="8Gi", gpu="1"),
    limits=Resources(gpu="1"),
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
)
def evaluate_il_policy(
    checkpoint: FlyteFile,
    shards: List[FlyteDirectory],
    train_metadata: FlyteFile,
    dataset: Dataset = Dataset.L2D,
) -> EvalMetrics:
    """Open-loop evaluation of the Imitation-Learning policy.

    Logs ADE/FDE, params, artifacts to the MLflow `imitation-learning` experiment
    and registers the checkpoint in the `auto-e2e-driving-policy` model registry.
    """
    return _run_evaluation(checkpoint, shards, train_metadata, dataset, "imitation-learning")


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="2", mem="8Gi", gpu="1"),
    limits=Resources(gpu="1"),
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
)
def evaluate_rl_policy(
    checkpoint: FlyteFile,
    shards: List[FlyteDirectory],
    train_metadata: FlyteFile,
    dataset: Dataset = Dataset.L2D,
) -> EvalMetrics:
    """Open-loop evaluation of the Offline-RL refined policy.

    Logs ADE/FDE, params (incl. rl/*), artifacts to the MLflow `offline-rl`
    experiment and registers the refined checkpoint in the model registry.
    """
    return _run_evaluation(checkpoint, shards, train_metadata, dataset, "offline-rl")



# ============================================================
# Workflows
# ============================================================
@workflow
def wf_data_ingest(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
) -> FlyteDirectory:
    """Download raw dataset from HuggingFace."""
    return data_ingest(dataset=dataset, episodes=episodes)


@workflow
def wf_data_processing(
    raw_data: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    hz: int = 10,
    image_size: int = 256,
    episodes: int = 3,
    world_model: bool = False,
    reasoning_labels: Optional[FlyteDirectory] = None,
) -> FlyteDirectory:
    """Pre-process raw data → WebDataset shards.

    ``world_model`` packs the JEPA per-sample windows (#13). ``reasoning_labels``
    (the generate_reasoning_labels artifact) is JOINed into reasoning.json (#98).
    Both MUST match the branch flags used at ``train_il`` time or that branch
    trains unsupervised.
    """
    return data_processing(raw_data=raw_data, dataset=dataset,
                           hz=hz, image_size=image_size, episodes=episodes,
                           world_model=world_model, reasoning_labels=reasoning_labels)


@workflow
def wf_generate_reasoning_labels(
    raw_data: FlyteDirectory,
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
    split: str = "train",
    teacher: str = "openai_compatible",
    prompt_version: str = "action_relevant_reasoning_v3_temporal_front256",
) -> FlyteDirectory:
    """Label raw samples with the offline teacher (S3-cached) → versioned artifact."""
    return generate_reasoning_labels(
        raw_data=raw_data, dataset=dataset, episodes=episodes, split=split,
        teacher=teacher, prompt_version=prompt_version)


@workflow
def _pack_with_labels(
    raw: FlyteDirectory,
    dataset: Dataset,
    episodes: int,
    image_size: int,
    world_model: bool,
    teacher: str,
    prompt_version: str,
) -> FlyteDirectory:
    """The 'with reasoning labels' branch of wf_create_dataset: label from raw
    (teacher, S3-cached) → pack shards with the labels JOINed in.

    A Flyte conditional branch is a single node, so the two-task label→pack chain
    lives in this sub-workflow.

    Reasoning labels are built from the 1 Hz World-Model window (temporal front
    clip), and ``len(L2DDataset)`` / sample ordering depend on
    ``include_world_model_windows``. So both generate and data_processing MUST run
    with world_model=True for the ``sample_id`` JOIN to align — we force it on
    here (the ``world_model`` arg is ignored on the labelled branch). Training can
    still ignore the JEPA windows if enable_world_model is off.
    """
    labels = generate_reasoning_labels(
        raw_data=raw, dataset=dataset, episodes=episodes, split="train",
        teacher=teacher, prompt_version=prompt_version)
    return data_processing(
        raw_data=raw, dataset=dataset, episodes=episodes, image_size=image_size,
        world_model=True, reasoning_labels=labels)


@workflow
def wf_create_dataset(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
    image_size: int = 256,
    world_model: bool = False,
    reasoning_teacher: str = "none",
    prompt_version: str = "action_relevant_reasoning_v3_temporal_front256",
) -> FlyteDirectory:
    """CreateDataset: raw → ready-to-train WebDataset shards.

    "Dataset" means data already in a form training consumes DIRECTLY: the
    WebDataset shards (frames + ego + optional WM windows + per-sample
    reasoning.json when a teacher is set). train_il reads its reasoning
    supervision from those in-shard members — the shards ARE the dataset.

    Reasoning labels are generated once by ``generate_reasoning_labels`` (the only
    place the teacher is called; each sample S3-cached so re-packing never
    re-bills it, #117) and JOINed into the shards by ``data_processing``. The
    versioned label artifact persists independently in S3 (task output + cache),
    so it need not be a workflow return value — the shards are the single output.

    Chains: data_ingest → [teacher != none] generate_reasoning_labels →
    data_processing (JOIN labels). With reasoning_teacher="none", no labels are
    generated and the shards carry no reasoning.json (imitation-only).
    """
    from flytekit import conditional

    raw = data_ingest(dataset=dataset, episodes=episodes)
    return (
        conditional("reasoning_labels")
        .if_(reasoning_teacher != "none")
        .then(_pack_with_labels(
            raw=raw, dataset=dataset, episodes=episodes, image_size=image_size,
            world_model=world_model, teacher=reasoning_teacher,
            prompt_version=prompt_version))
        .else_()
        .then(data_processing(
            raw_data=raw, dataset=dataset, episodes=episodes,
            image_size=image_size, world_model=world_model))
    )


@workflow
def wf_train_il(
    shards: List[FlyteDirectory],
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    lr: float = 1e-4,
    amp: bool = False,
    enable_reasoning: bool = False,
    reasoning_mode: str = "pooled_latent",
    enable_world_model: bool = False,
    val_fraction: float = 0.0,
    num_workers: int = 0,
) -> EvalMetrics:
    """IL Train → Evaluate. All datasets' shards passed in; `dataset` selects one.

    The branch flags must match how the shards were packed (see
    ``wf_data_processing``); train_il fails loudly if a branch is enabled but its
    shard data is missing rather than training it unsupervised. ``amp`` defaults
    off: fp16 autocast made the GradScaler skip every step (see train_il).
    ``grad_accum_steps`` recovers a larger effective batch when the World-Model
    windows force batch_size=1 (effective batch = batch_size * grad_accum_steps).
    ``val_fraction`` > 0 trains on a per-sample train split and evaluates on the
    disjoint held-out val split (generalization, not in-sample memorization).
    ``num_workers`` > 0 parallelizes JPEG decode across worker processes (#121 P0)
    — the dominant per-epoch cost once episodes scale up.
    """
    out = train_il(shards=shards, dataset=dataset, backbone=backbone,
                   epochs=epochs, batch_size=batch_size,
                   grad_accum_steps=grad_accum_steps, lr=lr, amp=amp,
                   enable_reasoning=enable_reasoning, reasoning_mode=reasoning_mode,
                   enable_world_model=enable_world_model, val_fraction=val_fraction,
                   num_workers=num_workers)
    return evaluate_il_policy(checkpoint=out.checkpoint, shards=shards, dataset=dataset,
                              train_metadata=out.metadata)


@workflow
def wf_train_offline_rl(
    pretrained: FlyteFile,
    shards: List[FlyteDirectory],
    il_metadata: FlyteFile,
    dataset: Dataset = Dataset.L2D,
    epochs: int = 3,
    tau: float = 0.7,
    beta: float = 3.0,
) -> EvalMetrics:
    """Offline RL → Evaluate. All datasets' shards passed in; `dataset` selects one."""
    out = train_offline_rl(pretrained=pretrained, shards=shards, dataset=dataset,
                           il_metadata=il_metadata, epochs=epochs, tau=tau, beta=beta)
    return evaluate_rl_policy(checkpoint=out.checkpoint, shards=shards, dataset=dataset,
                              train_metadata=out.metadata)


@workflow
def wf_full_pipeline(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs_il: int = 3,
    epochs_rl: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    tau: float = 0.7,
    beta: float = 3.0,
) -> EvalMetrics:
    """Full: Ingest+Process ALL datasets (separately packed) → IL Train+Eval → RL Train+Eval.

    Every dataset is ingested and processed into its own WebDataset shard dir, and
    all shard dirs are passed to the train/eval tasks. The `dataset` argument selects
    which one is actually used for this run (single-dataset training; multi-dataset
    on one model tracked in #77).
    """
    # Ingest + process every dataset into separate WebDataset shard dirs
    raw_l2d = data_ingest(dataset=Dataset.L2D, episodes=episodes)
    shards_l2d = data_processing(raw_data=raw_l2d, dataset=Dataset.L2D, episodes=episodes)

    raw_nv = data_ingest(dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)
    shards_nv = data_processing(raw_data=raw_nv, dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)

    all_shards = [shards_l2d, shards_nv]

    il_out = train_il(shards=all_shards, dataset=dataset, backbone=backbone,
                      epochs=epochs_il, batch_size=batch_size, lr=lr)
    evaluate_il_policy(checkpoint=il_out.checkpoint, shards=all_shards, dataset=dataset,
                       train_metadata=il_out.metadata)
    rl_out = train_offline_rl(pretrained=il_out.checkpoint, shards=all_shards, dataset=dataset,
                              il_metadata=il_out.metadata, epochs=epochs_rl, tau=tau, beta=beta)
    return evaluate_rl_policy(checkpoint=rl_out.checkpoint, shards=all_shards, dataset=dataset,
                              train_metadata=rl_out.metadata)


@workflow
def wf_ingest_train_eval(
    dataset: Dataset = Dataset.L2D,
    episodes: int = 3,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    epochs_il: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
) -> EvalMetrics:
    """Ingest+Process ALL datasets → IL Train → IL Eval (no offline RL).

    Same as wf_full_pipeline but stops after IL evaluation. Useful when you only
    want a supervised checkpoint + open-loop metrics, or when the offline-RL step
    is too memory-hungry to co-run at the current BEV resolution (#77).
    """
    raw_l2d = data_ingest(dataset=Dataset.L2D, episodes=episodes)
    shards_l2d = data_processing(raw_data=raw_l2d, dataset=Dataset.L2D, episodes=episodes)

    raw_nv = data_ingest(dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)
    shards_nv = data_processing(raw_data=raw_nv, dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)

    all_shards = [shards_l2d, shards_nv]

    il_out = train_il(shards=all_shards, dataset=dataset, backbone=backbone,
                      epochs=epochs_il, batch_size=batch_size, lr=lr)
    return evaluate_il_policy(checkpoint=il_out.checkpoint, shards=all_shards, dataset=dataset,
                              train_metadata=il_out.metadata)
