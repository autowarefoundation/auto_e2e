"""Per-sample metrics of AutoE2E: BEV IoU by distance band and trajectory displacement.

Both the asynchronous checkpoint validation and the formal evaluation use these
functions, so a checkpoint is judged the same way in both. Every function keeps
per-sample values; sums over a set are taken by the caller, which is what lets
a formal evaluation explain which samples moved a metric.

BEV IoU follows the AutoBEV experiment: forward distance bands
(``BANDS``), a per-class binarization threshold chosen on held-out training
samples (``calibrate_thresholds``, never on the evaluation set), and a
one-cell tolerant variant that forgives offsets of one 0.5 m cell on thin
lines (as boundary-tolerant map metrics do).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import torch
import torch.nn.functional as F

from data_processing.reactive_training_artifacts import BEV_SEGMENTATION_CLASSES
from model_components.autobev.geometry import PC_RANGE

BANDS: Final = (("0-30", 0.0, 30.0), ("30-60", 30.0, 60.0), ("60-80", 60.0, 80.0 + 1e-6),
                ("rear", -30.0 - 1e-6, 0.0))
FORWARD_BANDS: Final = ("0-30", "30-60", "60-80")
IOU_THRESHOLDS: Final = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99)
# Displacement horizons in seconds of the 10 Hz, 6.4 s trajectory.
HORIZONS_S: Final = (1.0, 2.0, 3.0, 6.4)
STEP_S: Final = 0.1


def _band_masks(rows: int, device) -> dict[str, torch.Tensor]:
    mpp = (PC_RANGE[3] - PC_RANGE[0]) / rows
    x = PC_RANGE[3] - (torch.arange(rows, device=device) + 0.5) * mpp
    return {name: ((x >= low) & (x < high)).view(1, 1, rows, 1) for name, low, high in BANDS}


def bev_counts(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-sample [B, C] intersection and union per band and threshold, and tolerant hit counts at each threshold.

    Keys: ``{band}/{threshold}/inter``, ``.../union``, ``.../pred``,
    ``.../pred_hit``, ``.../true``, ``.../true_hit``. A tolerant hit is a
    predicted cell within one cell of a true one (and vice versa), counted only
    inside the same band and over valid cells.
    """
    probability = logits.float().sigmoid()
    truth = target > 0.5
    counts: dict[str, torch.Tensor] = {}

    def dilate(mask: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(mask.float(), 3, stride=1, padding=1) > 0

    for name, band in _band_masks(logits.shape[-2], logits.device).items():
        inside = band & valid.bool()
        counts[f"{name}/valid"] = inside.sum((2, 3)).float()
        near_truth = dilate(truth & inside)
        for threshold in IOU_THRESHOLDS:
            predicted = (probability > threshold) & inside
            counts[f"{name}/{threshold}/inter"] = (predicted & truth).sum((2, 3)).float()
            counts[f"{name}/{threshold}/union"] = (predicted | (truth & inside)).sum((2, 3)).float()
            counts[f"{name}/{threshold}/pred"] = predicted.sum((2, 3)).float()
            counts[f"{name}/{threshold}/pred_hit"] = (predicted & near_truth).sum((2, 3)).float()
            counts[f"{name}/{threshold}/true"] = (truth & inside).sum((2, 3)).float()
            counts[f"{name}/{threshold}/true_hit"] = (truth & inside & dilate(predicted)).sum((2, 3)).float()
    return counts


def calibrate_thresholds(totals: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Per class, the threshold with the best IoU summed over the forward bands of the calibration counts."""
    thresholds = {}
    for index, name in enumerate(BEV_SEGMENTATION_CLASSES):
        def score(threshold: float) -> float:
            inter = sum(float(totals[f"{band}/{threshold}/inter"][index]) for band in FORWARD_BANDS)
            union = sum(float(totals[f"{band}/{threshold}/union"][index]) for band in FORWARD_BANDS)
            return inter / union if union > 0 else -1.0
        thresholds[name] = max(IOU_THRESHOLDS, key=score)
    return thresholds


def summarize_bev(totals: Mapping[str, torch.Tensor], thresholds: Mapping[str, float]) -> dict[str, float]:
    """IoU and one-cell tolerant IoU per band and class at each class's calibrated threshold."""
    report = {}
    for index, name in enumerate(BEV_SEGMENTATION_CLASSES):
        threshold = thresholds[name]
        report[f"bev_threshold/{name}"] = float(threshold)
        for band, _, _ in BANDS:
            # Predicting every valid cell scores the positive rate; an IoU near
            # it means the class has not been learned.
            cells = float(totals[f"{band}/valid"][index])
            if cells > 0:
                report[f"bev_positive_rate/{band}/{name}"] = float(totals[f"{band}/{threshold}/true"][index]) / cells
            union = float(totals[f"{band}/{threshold}/union"][index])
            if union > 0:
                report[f"bev/{band}/{name}"] = float(totals[f"{band}/{threshold}/inter"][index]) / union
            pred, true = (float(totals[f"{band}/{threshold}/{part}"][index]) for part in ("pred", "true"))
            if pred > 0 and true > 0:
                precision = float(totals[f"{band}/{threshold}/pred_hit"][index]) / pred
                recall = float(totals[f"{band}/{threshold}/true_hit"][index]) / true
                if precision > 0 and recall > 0:
                    report[f"bev_tol/{band}/{name}"] = 1.0 / (1.0 / precision + 1.0 / recall - 1.0)
    return report


def trajectory_errors(predicted_xy: torch.Tensor, target_xy: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-sample ADE and FDE [B] up to each horizon; NaN where the horizon's last point is invalid."""
    distance = (predicted_xy.float() - target_xy.float()).norm(dim=-1)
    valid = valid.bool()
    errors = {}
    for horizon in HORIZONS_S:
        steps = round(horizon / STEP_S)
        window = valid[:, :steps]
        complete = window.all(dim=1)
        ade = (distance[:, :steps] * window).sum(1) / window.sum(1).clamp_min(1)
        nan = torch.full_like(ade, float("nan"))
        errors[f"ade/{horizon}s"] = torch.where(complete, ade, nan)
        errors[f"fde/{horizon}s"] = torch.where(complete, distance[:, steps - 1], nan)
    return errors
