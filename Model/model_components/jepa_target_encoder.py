"""Frozen / EMA target encoder for the JEPA feature-reconstruction objective.

Background
----------
Feature B (merged, #80) added the *prediction* side of the JEPA objective:

* ``FutureState`` predicts future BEV feature maps — a list of
  ``num_future_steps`` tensors, each ``[B, C, H, W]``.
* ``losses.FeatureReconstructionLoss`` scores those predictions against
  *target* feature maps of identical shape. Its own docstring states the
  targets are "extracted by a frozen copy of the image backbone (no gradient)
  applied to the future frames at +1.6s, +3.2s, +4.8s and +6.4s".

That **target encoder** was the missing piece. In Joint-Embedding Predictive
Architectures (I-JEPA / V-JEPA) the targets come from an encoder that is *not*
trained by backprop from the loss — it is either a **frozen** copy of the
online encoder or an **exponential-moving-average (EMA)** of it, and its output
is detached (stop-gradient). This is what prevents representational collapse
(the predictor cannot win by driving every feature to zero, because the target
encoder is not pulled along).

This module implements exactly that, supporting BOTH modes @RyotaYamada listed
in #56 / #13 ("frozen or EMA") so the choice stays a *configuration* rather than
a hard-coded decision.

Deliberately left open (owned by Zain — 17-06 action item, #13)
---------------------------------------------------------------
This module does NOT decide: the input frequency (1 Hz / TBD), the predictor
space (BEV vs feature), the exact future horizons, the JEPA-vs-imitation loss
weighting, or the data pipeline that supplies the future frames. It only
provides the collapse-safe target generator those decisions will plug into.
"""

import copy

import torch
import torch.nn as nn


class JepaTargetEncoder(nn.Module):
    """Produce stop-gradient target features for the JEPA loss.

    Wraps a *copy* of an online encoder (e.g. the image backbone, or the
    backbone+fusion path) and runs it without gradients to generate the target
    feature maps that ``FeatureReconstructionLoss`` compares ``FutureState``'s
    predictions against.

    Args:
        encoder: the online encoder to mirror. A deep copy is taken at
            construction; the original is never modified by this module.
        mode: ``"frozen"`` (a fixed copy, never updated) or ``"ema"`` (updated
            from the online encoder via :meth:`update`). Default ``"ema"``.
        ema_decay: EMA momentum in ``[0, 1]``; only used in ``"ema"`` mode.
            ``target ← decay * target + (1 - decay) * online``.

    The wrapped encoder's parameters always have ``requires_grad=False`` and
    :meth:`forward` returns **detached** outputs, so no gradient ever reaches
    the target branch regardless of how the caller composes the loss.
    """

    def __init__(self, encoder: nn.Module, mode: str = "ema",
                 ema_decay: float = 0.999):
        super().__init__()
        if mode not in ("frozen", "ema"):
            raise ValueError(f"mode must be 'frozen' or 'ema', got {mode!r}")
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in [0, 1], got {ema_decay}")
        self.mode = mode
        self.ema_decay = ema_decay

        # A detached, non-trainable mirror of the online encoder.
        self.encoder = copy.deepcopy(encoder)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        """EMA-update the target weights toward ``online_encoder``.

        No-op in ``"frozen"`` mode. Buffers (e.g. BatchNorm running stats) are
        copied directly rather than averaged. Call once per optimizer step,
        after the online encoder has been updated.
        """
        if self.mode != "ema":
            return
        d = self.ema_decay
        for t_p, o_p in zip(self.encoder.parameters(),
                            online_encoder.parameters()):
            t_p.mul_(d).add_(o_p.detach(), alpha=1.0 - d)
        for t_b, o_b in zip(self.encoder.buffers(), online_encoder.buffers()):
            t_b.copy_(o_b)

    @torch.no_grad()
    def forward(self, future_observations):
        """Encode each future observation into a detached target feature map.

        Args:
            future_observations: an iterable of ``num_future_steps`` tensors,
                each a valid input to the wrapped encoder (e.g. future camera
                frames or future fused features), ordered by horizon.

        Returns:
            list of ``num_future_steps`` detached feature maps ``[B, C, H, W]``,
            ready to pass as ``target_features`` to
            :class:`FeatureReconstructionLoss`.
        """
        self.encoder.eval()
        return [self.encoder(obs).detach() for obs in future_observations]


def compute_jepa_loss(predicted_features, future_observations, target_encoder,
                      loss_fn, weight: float = 1.0):
    """Glue helper: targets ← target_encoder, then weighted JEPA loss.

    This is the single call a training loop adds to fold the JEPA objective in
    alongside the trajectory imitation loss, e.g.::

        total = imitation_loss + compute_jepa_loss(
            future, future_frames, target_encoder, recon_loss, weight=0.1)

    Args:
        predicted_features: ``FutureState`` output (list of ``[B, C, H, W]``).
        future_observations: inputs for ``target_encoder`` (one per horizon).
        target_encoder: a :class:`JepaTargetEncoder`.
        loss_fn: a :class:`FeatureReconstructionLoss` (or compatible callable).
        weight: scalar coefficient for the JEPA term (``lambda``).

    Returns:
        ``weight * loss_fn(predicted_features, targets)`` — a scalar tensor
        differentiable w.r.t. ``predicted_features`` only (targets are detached).
    """
    targets = target_encoder(future_observations)
    return weight * loss_fn(predicted_features, targets)
