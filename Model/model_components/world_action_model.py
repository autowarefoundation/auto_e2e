"""World Action Model — slow World-Model branch (~1 Hz). Agreed in WG 2026-06-24.

Encodes the recent multi-camera history into a rolling **visual-history** vector
and predicts the **future** visual features (JEPA, self-supervised). Decoded from
Zain's answers to the 5 interface questions (24/06 transcript + miro):

1. **Backbone:** one SHARED image backbone; the JEPA target is a **FROZEN copy**
   of it (not EMA) — `JepaTargetEncoder(mode="frozen")`.
2. **Horizons:** `N_past = N_future`, default **4**, sampled at **1 Hz**.
3. **Feature level:** a **per-frame embedding** (default **224**); the
   reconstruction lives in that feature space.
4. **Visual history:** a rolling **FIFO buffer** of the last `history_len`
   embeddings → `history_len * frame_embed_dim = 4 * 224 = 896` (= the existing
   `visual_history_dim`), fed to the reactive planner.
5. **Training:** in `train_il` with equal loss weight; **L1** in feature space.
   Future frames come from the 1 Hz stream of the dataloader.

Forward returns the visual-history vector always; **in training** it also returns
the future-feature prediction and the JEPA loss (so `AutoE2E.forward` can return
`(trajectory, future_states, ego_hidden)` when training and just `trajectory`
otherwise). Reuses the merged/queued building blocks (JepaTargetEncoder,
FeatureReconstructionLoss).
"""

import torch
import torch.nn as nn

from .jepa_target_encoder import JepaTargetEncoder, compute_jepa_loss
from .losses.feature_reconstruction_loss import FeatureReconstructionLoss


class FrameEncoder(nn.Module):
    """One multi-camera frame ``[B, 3, H, W]`` -> embedding ``[B, frame_embed_dim]``.

    backbone -> last feature map -> global average pool -> linear projection.
    """

    def __init__(self, backbone: nn.Module, feature_channels: int = 768,
                 frame_embed_dim: int = 224):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(feature_channels, frame_embed_dim)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(frame)
        x = feats[-1] if isinstance(feats, (list, tuple)) else feats  # [B, C, h, w]
        x = x.mean(dim=(2, 3))                                        # GAP -> [B, C]
        return self.proj(x)                                          # [B, embed]


class RollingHistoryBuffer:
    """Inference-time FIFO buffer of the last ``history_len`` frame embeddings.

    Push one embedding per tick; ``visual_history()`` concatenates them
    (oldest -> newest) into ``[B, history_len * frame_embed_dim]``, left-padding
    with zeros until the buffer fills. Mirrors the windowed encoding used in
    training so train/inference share a representation.
    """

    def __init__(self, history_len: int = 4):
        self.history_len = history_len
        self._buf: list[torch.Tensor] = []

    def push(self, embedding: torch.Tensor) -> None:
        self._buf.append(embedding)
        if len(self._buf) > self.history_len:
            self._buf.pop(0)  # first-in, first-out

    def visual_history(self) -> torch.Tensor | None:
        if not self._buf:
            return None
        pad = [torch.zeros_like(self._buf[0])] * (self.history_len - len(self._buf))
        return torch.cat(pad + self._buf, dim=1)


class WorldActionModel(nn.Module):
    """Slow world-model branch: history -> visual_history (+ future JEPA in train).

    Args:
        backbone: shared image backbone (e.g. ``Backbone``); a frozen copy is
            used as the JEPA target.
        feature_channels: channels of the backbone's last feature map.
        frame_embed_dim: per-frame embedding size (default 224).
        history_len: number of past frames in the FIFO buffer (default 4).
        num_future_steps: future horizons to predict (default = history_len).
        loss_type: feature-space distance for the JEPA loss (``"l1"`` default).
    """

    def __init__(self, backbone: nn.Module, feature_channels: int = 768,
                 frame_embed_dim: int = 224, history_len: int = 4,
                 num_future_steps: int = 4, loss_type: str = "l1"):
        super().__init__()
        self.history_len = history_len
        self.num_future_steps = num_future_steps
        self.frame_embed_dim = frame_embed_dim
        self.visual_history_dim = history_len * frame_embed_dim  # 4*224 = 896

        # Online per-frame encoder (shared backbone, trainable).
        self.encoder = FrameEncoder(backbone, feature_channels, frame_embed_dim)
        # JEPA target: a FROZEN, stop-gradient copy of the encoder (#1).
        self.target = JepaTargetEncoder(self.encoder, mode="frozen")
        # Future feature predictor: visual_history -> num_future_steps embeddings.
        self.future_predictor = nn.Sequential(
            nn.Linear(self.visual_history_dim, self.visual_history_dim),
            nn.GELU(),
            nn.Linear(self.visual_history_dim, num_future_steps * frame_embed_dim),
        )
        self.recon_loss = FeatureReconstructionLoss(
            num_future_steps=num_future_steps, loss_type=loss_type)

    def encode_history(self, history_frames: torch.Tensor) -> torch.Tensor:
        """``[B, history_len, 3, H, W]`` -> visual_history ``[B, 896]`` (FIFO order)."""
        T = history_frames.shape[1]
        embs = [self.encoder(history_frames[:, t]) for t in range(T)]
        return torch.cat(embs, dim=1)

    def forward(self, history_frames: torch.Tensor,
                future_frames: torch.Tensor | None = None):
        """Encode the history; in training also predict the future + JEPA loss.

        Args:
            history_frames: ``[B, history_len, 3, H, W]`` past frames at 1 Hz.
            future_frames: ``[B, num_future_steps, 3, H, W]`` future frames
                (training only).

        Returns:
            inference: ``visual_history`` ``[B, 896]``.
            training:  ``(visual_history, predicted_future, jepa_loss)`` where
                ``predicted_future`` is a list of ``num_future_steps`` tensors
                ``[B, frame_embed_dim]`` and ``jepa_loss`` is a scalar.
        """
        visual_history = self.encode_history(history_frames)
        if future_frames is None:
            return visual_history

        predicted = list(torch.chunk(
            self.future_predictor(visual_history), self.num_future_steps, dim=1))
        future_obs = [future_frames[:, k] for k in range(self.num_future_steps)]
        loss = compute_jepa_loss(predicted, future_obs, self.target,
                                 self.recon_loss, weight=1.0)
        return visual_history, predicted, loss
