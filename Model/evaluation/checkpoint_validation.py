"""Validate training checkpoints as they appear, on one GPU next to the training task.

The training loop does not evaluate. This watcher lists the run's Ray
checkpoints under ``run_uri``, and for each new one:

1. calibrates the per-class BEV thresholds on ``CALIBRATION_SAMPLES`` held-out
   training samples;
2. measures BEV IoU by band on ``VALIDATION_SAMPLES`` of the geographic
   validation split (nuPlan Pittsburgh) and, in stage 2, ADE/FDE with the map
   cropped at a simulated onboard localization estimate refined by map
   matching, as at inference;
3. writes ``validation.json`` into the checkpoint directory.

These numbers choose which checkpoints go to the formal evaluation and how
long to train; they are not recorded on any MLflow model version.

In stage 2 the first result is the parent stage 1 checkpoint. When a
checkpoint's Pittsburgh drivable-area IoU (mean of the forward bands) falls
more than ``BACKOFF_DROP`` below it, the watcher adds one to
``autobev_backoffs`` in the run's feedback file and the trainer halves the
AutoBEV learning rates at its next checkpoint.

The watcher ends after the checkpoint of the last training step.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import torch

from evaluation.autoe2e_metrics import bev_counts, calibrate_thresholds, summarize_bev, trajectory_errors
from training import storage
from training.trainer import CHECKPOINT_FILE, FEEDBACK_FILE, METADATA_FILE, TrainRun, model_inputs

VALIDATION_SAMPLES: Final = 512
CALIBRATION_SAMPLES: Final = 512
POLL_SECONDS: Final = 300
BACKOFF_DROP: Final = 0.10
RESULT_FILE: Final = "validation.json"
SEED: Final = 0


def _sum(totals: dict[str, torch.Tensor], counts: Mapping[str, torch.Tensor]) -> None:
    for key, value in counts.items():
        totals[key] = totals.get(key, 0) + value.sum(0).cpu()


@torch.no_grad()
def _bev_and_trajectory(model, stage: int, batches: Iterable[dict[str, Any]], device):
    from model_components.losses import TrajectoryXYImitationLoss
    from training.autobev_data import to_device
    from training.localization_noise import simulate_localization

    totals: dict[str, torch.Tensor] = {}
    errors: dict[str, list[torch.Tensor]] = {}
    sample_ids: list[str] = []
    generator = torch.Generator().manual_seed(SEED)
    integrate = TrajectoryXYImitationLoss().predicted_xy
    for raw in batches:
        batch = to_device(raw, device)
        sample_ids += list(batch["sample_uid"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if stage == 1:
                outputs = model(batch["visual_tiles"], batch["camera_history_tiles"],
                                batch["camera_projection_matrix"], batch["camera_history_projection_matrix"])
            else:
                localization = simulate_localization(batch["egomotion_history"], generator)
                localization = type(localization)(localization.pose, localization.covariance,
                                                  torch.ones_like(localization.map_matching))
                controls, outputs = model(**model_inputs(batch, localization))
                predicted = integrate(controls.float(), batch["initial_speed_mps"])
                supervised = batch["planner_supervised"].bool()
                for key, value in trajectory_errors(predicted, batch["trajectory_xy_m"],
                                                    batch["trajectory_valid"].bool() & supervised[:, None]).items():
                    errors.setdefault(key, []).append(value.cpu())
        _sum(totals, bev_counts(outputs["bev_logits"], batch["bev_segmentation_target"],
                                batch["bev_segmentation_valid"]))
    trajectory = {key: torch.cat(values) for key, values in errors.items()}
    return totals, trajectory, sample_ids


def validate(model, stage: int, calibration: Iterable, validation: Iterable, device) -> dict[str, Any]:
    """Calibrate thresholds, then report BEV IoU (and ADE/FDE in stage 2) with per-sample trajectory errors."""
    model.eval()
    calibration_totals, _, _ = _bev_and_trajectory(model, stage, calibration, device)
    thresholds = calibrate_thresholds(calibration_totals)
    totals, trajectory, sample_ids = _bev_and_trajectory(model, stage, validation, device)
    metrics = summarize_bev(totals, thresholds)
    for key, values in trajectory.items():
        finite = values[torch.isfinite(values)]
        if finite.numel():
            metrics[key] = float(finite.mean())
    return {
        "metrics": metrics,
        "thresholds": thresholds,
        "sample_uids": sample_ids,
        "per_sample": {key: [float(v) for v in values] for key, values in trajectory.items()},
    }


def drivable_score(metrics: Mapping[str, float]) -> float:
    bands = [metrics.get(f"bev/{band}/drivable_area", 0.0) for band in ("0-30", "30-60", "60-80")]
    return sum(bands) / len(bands)


def list_checkpoints(run_uri: str) -> list[str]:
    """Ray checkpoint directories of the run, oldest first (their names sort by time)."""
    return [uri for uri in storage.child_directories(run_uri) if uri.rsplit("/", 1)[-1].startswith("checkpoint_")]


def read_json(uri: str) -> dict:
    return json.loads(storage.read_bytes(uri))


def write_json(uri: str, payload: Mapping) -> None:
    storage.write_bytes(uri, json.dumps(payload, indent=1, sort_keys=True).encode())


def _load(model, checkpoint_dir: str) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        local = storage.download(f"{checkpoint_dir}/{CHECKPOINT_FILE}", Path(directory) / CHECKPOINT_FILE)
        state = torch.load(local, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return state["metadata"]


def watch(run: TrainRun, calibration: Callable[[], Iterable], validation: Callable[[], Iterable]) -> list[dict]:
    """Validate every checkpoint of ``run`` until its last step; returns the results in order.

    ``calibration`` and ``validation`` make a fresh pass over their split each
    time: holding 512 full T8 samples in memory would take about 75 GB.
    """
    from model_components import AutoE2E
    from model_components.autobev import AutoBEV

    device = torch.device("cuda")
    run_uri = f"{run.storage_uri.rstrip('/')}/{run.run_name}"
    model = (AutoBEV(pretrained_backbone=False) if run.stage == 1 else AutoE2E(pretrained_backbone=False)).to(device)
    baseline = None
    if run.stage == 2:
        _load(model.Reactive_E2E.AutoBEV, run.parent_checkpoint_uri.rstrip("/"))
        baseline = validate(model, run.stage, calibration(), validation(), device)
        write_json(f"{run_uri}/parent_{RESULT_FILE}", baseline)
    results, backoffs = [], 0
    while True:
        # A checkpoint counts once both of its files are uploaded.
        pending = [c for c in list_checkpoints(run_uri)
                   if storage.exists(f"{c}/{METADATA_FILE}") and storage.exists(f"{c}/{CHECKPOINT_FILE}")
                   and not storage.exists(f"{c}/{RESULT_FILE}")]
        if not pending:
            time.sleep(POLL_SECONDS)
            continue
        for checkpoint in pending:
            metadata = _load(model, checkpoint)
            result = {"step": metadata["step"], "checkpoint": checkpoint,
                      **validate(model, run.stage, calibration(), validation(), device)}
            write_json(f"{checkpoint}/{RESULT_FILE}", result)
            results.append(result)
            if baseline is not None and drivable_score(result["metrics"]) < (1 - BACKOFF_DROP) * drivable_score(
                    baseline["metrics"]):
                backoffs += 1
                write_json(f"{run_uri}/{FEEDBACK_FILE}", {"autobev_backoffs": backoffs, "step": metadata["step"]})
            if metadata["step"] >= run.steps:
                return results
