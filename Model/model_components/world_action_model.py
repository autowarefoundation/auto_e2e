"""World Action Model — slow World-Model branch (~1 Hz). Agreed in WG 2026-06-24.

Encodes the recent multi-camera history into a rolling **visual-history** vector
and predicts the **future** visual features (JEPA, self-supervised). Decoded from
Zain's answers to the 5 interface questions (24/06 transcript + miro):

1. **Backbone:** one SHARED image backbone; the JEPA target is a **FROZEN copy**
   of it (not EMA) — `JepaTargetEncoder(mode="frozen")`.
2. **Horizons:** `N_past = N_future`, default **4**, sampled at **1 Hz**.
3. **Feature level:** the history uses a per-frame **224** embedding; the JEPA
   **future prediction reconstructs the backbone feature maps** ``[B, C, 8, 8]``
   (per @m-zain-khawaja's #85 review, 30/06) — not a pooled vector.
4. **Visual history:** a rolling **FIFO buffer** of the last `history_len`
   embeddings → `history_len * frame_embed_dim = 4 * 224 = 896` (= the existing
   `visual_history_dim`), fed to the reactive planner.
5. **Training:** in `train_il` with equal loss weight; **L1** in feature space.
   Future frames come from the 1 Hz stream of the dataloader.

The per-tick ``forward`` returns ``(visual_embedding, future_state_pred)``: the
embedding is pushed to the rolling history (→ reactive planner), and
``future_state_pred`` (the predicted future feature maps, training only) feeds the
JEPA loss, which is computed separately via :meth:`jepa_loss` in the training
loop. Reuses the merged building blocks (JepaTargetEncoder,
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
        """``[B, 3, H, W]`` or multi-view ``[B, V, 3, H, W]`` -> ``[B, embed]``.

        For multi-view input each camera is encoded and the per-view embeddings
        are mean-pooled into a single per-frame embedding.
        """
        if frame.dim() == 5:                       # [B, V, 3, H, W]
            B, V = frame.shape[:2]
            feats = self.backbone(frame.reshape(B * V, *frame.shape[2:]))
            m = feats[-1] if isinstance(feats, (list, tuple)) else feats
            m = m.mean(dim=(2, 3)).reshape(B, V, -1).mean(dim=1)  # GAP + pool views
            return self.proj(m)                                  # [B, embed]
        feats = self.backbone(frame)               # [B, 3, H, W]
        m = feats[-1] if isinstance(feats, (list, tuple)) else feats
        return self.proj(m.mean(dim=(2, 3)))                     # [B, embed]


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


class HistoryAttentionPool(nn.Module):
    """Opt-in learnable temporal aggregator for the World Model history — the
    "temporal attention-pool" alternative to the equal-weight concat-FIFO.

    Adapted from the BEV-queue temporal self-attention (T3 / #87): a single
    learnable query attends over the ``history_len`` frame embeddings plus
    learned temporal positional embeddings (recency-aware), so frames are
    weighted by relevance instead of equally. The pooled per-frame vector is
    projected back to ``history_len * frame_embed_dim`` so it is a drop-in for
    the concat-FIFO interface (same ``visual_history_dim`` output, e.g. 896).
    """

    def __init__(self, frame_embed_dim: int = 224, history_len: int = 4,
                 num_heads: int = 4):
        super().__init__()
        self.history_len = history_len
        self.frame_embed_dim = frame_embed_dim
        self.temporal_pos = nn.Parameter(
            torch.randn(1, history_len, frame_embed_dim) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, frame_embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            frame_embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(frame_embed_dim)
        self.out = nn.Linear(frame_embed_dim, history_len * frame_embed_dim)

    def forward(self, history_concat: torch.Tensor) -> torch.Tensor:
        """``[B, history_len*frame_embed_dim]`` -> ``[B, history_len*frame_embed_dim]``."""
        B = history_concat.shape[0]
        tokens = history_concat.reshape(B, self.history_len, self.frame_embed_dim)
        tokens = tokens + self.temporal_pos                 # recency-aware
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, tokens, tokens)            # [B, 1, embed]
        pooled = self.norm(pooled.squeeze(1))               # [B, embed]
        return self.out(pooled)                             # [B, history_len*embed]


class BackboneFeatureMap(nn.Module):
    """One multi-camera frame -> the backbone's last **feature map** ``[B, C, hw, hw]``.

    Unlike :class:`FrameEncoder` (which pools to a vector for the history), this
    keeps the spatial feature map — it is the JEPA target the future predictor
    reconstructs (per Zain's #85 review: reconstruct ``[B, backbone_channels, 8, 8]``).
    Multi-view frames are mean-pooled over views; the map is adaptive-avg-pooled
    to ``hw x hw`` so the target grid is fixed regardless of the backbone's native
    resolution.
    """

    def __init__(self, backbone: nn.Module, feature_hw: int = 8):
        super().__init__()
        self.backbone = backbone
        self.feature_hw = feature_hw

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dim() == 5:                                   # [B, V, 3, H, W]
            B, V = frame.shape[:2]
            feats = self.backbone(frame.reshape(B * V, *frame.shape[2:]))
            m = feats[-1] if isinstance(feats, (list, tuple)) else feats
            m = m.reshape(B, V, *m.shape[1:]).mean(dim=1)      # [B, C, h, w]
        else:                                                  # [B, 3, H, W]
            feats = self.backbone(frame)
            m = feats[-1] if isinstance(feats, (list, tuple)) else feats
        if m.shape[-2:] != (self.feature_hw, self.feature_hw):
            m = nn.functional.adaptive_avg_pool2d(m, (self.feature_hw, self.feature_hw))
        return m                                               # [B, C, hw, hw]


class FutureFeatureMapPredictor(nn.Module):
    """Predict ``num_future_steps`` future backbone feature maps ``[B, C, hw, hw]``
    from the Encoded Visual History ``[B, visual_history_dim]``.

    Lightweight decoder: a shared linear seed -> ``[B, mid, hw, hw]`` followed by
    one 1x1 conv head per future step (keeps params modest vs a direct
    ``history_dim -> C*hw*hw`` linear).
    """

    def __init__(self, in_dim: int, channels: int, feature_hw: int,
                 num_future_steps: int, mid: int = 128):
        super().__init__()
        self.channels, self.feature_hw, self.mid = channels, feature_hw, mid
        self.num_future_steps = num_future_steps
        self.seed = nn.Linear(in_dim, mid * feature_hw * feature_hw)
        self.act = nn.GELU()
        self.heads = nn.ModuleList(
            [nn.Conv2d(mid, channels, kernel_size=1) for _ in range(num_future_steps)])

    def forward(self, visual_history: torch.Tensor) -> list:
        B = visual_history.shape[0]
        seed = self.act(self.seed(visual_history))
        seed = seed.reshape(B, self.mid, self.feature_hw, self.feature_hw)
        return [head(seed) for head in self.heads]   # N x [B, C, hw, hw]


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
                 num_future_steps: int = 4, loss_type: str = "l1",
                 history_aggregator: str = "concat", feature_hw: int = 8):
        super().__init__()
        if history_aggregator not in ("concat", "attention"):
            raise ValueError(
                f"Unknown history_aggregator {history_aggregator!r}. "
                "Available: 'concat' (default), 'attention'.")
        self.history_len = history_len
        self.num_future_steps = num_future_steps
        self.frame_embed_dim = frame_embed_dim
        self.feature_channels = feature_channels
        self.feature_hw = feature_hw
        self.visual_history_dim = history_len * frame_embed_dim  # 4*224 = 896

        # History aggregator: how the FIFO of frame embeddings becomes the
        # visual_history fed to the planner. "concat" = equal-weight FIFO
        # (default, byte-identical to before); "attention" = the opt-in
        # learnable temporal attention-pool (recency-aware), reused from T3.
        self.history_pool = (
            HistoryAttentionPool(frame_embed_dim, history_len)
            if history_aggregator == "attention" else None)

        # Online per-frame encoder (shared backbone, trainable) -> 224 history embedding.
        self.encoder = FrameEncoder(backbone, feature_channels, frame_embed_dim)
        # JEPA target: a FROZEN, stop-gradient copy of the backbone FEATURE MAP
        # extractor (#1). Per Zain's #85 review (30/06) the JEPA reconstructs the
        # future backbone feature maps [B, C, hw, hw], not a pooled vector.
        self.target = JepaTargetEncoder(
            BackboneFeatureMap(backbone, feature_hw), mode="frozen")
        # Future feature-map predictor: visual_history -> N x [B, C, hw, hw].
        self.future_predictor = FutureFeatureMapPredictor(
            self.visual_history_dim, feature_channels, feature_hw, num_future_steps)
        self.recon_loss = FeatureReconstructionLoss(
            num_future_steps=num_future_steps, loss_type=loss_type)

    def encode_history(self, history_frames: torch.Tensor) -> torch.Tensor:
        """``[B, history_len, 3, H, W]`` -> visual_history ``[B, 896]`` (FIFO order)."""
        T = history_frames.shape[1]
        embs = [self.encoder(history_frames[:, t]) for t in range(T)]
        return torch.cat(embs, dim=1)

    def aggregate_history(self, history_concat):
        """Turn the FIFO of frame embeddings into the visual_history vector.

        ``concat`` mode returns the concatenation unchanged; ``attention`` mode
        runs the learnable temporal attention-pool. Same ``[B, visual_history_dim]``
        output either way (drop-in). Passes ``None`` through (empty buffer).
        """
        if history_concat is None or self.history_pool is None:
            return history_concat
        return self.history_pool(history_concat)

    def predict_future(self, visual_history: torch.Tensor) -> list:
        """``visual_history [B, visual_history_dim]`` -> list of ``num_future_steps``
        predicted future backbone **feature maps** ``[B, feature_channels, hw, hw]``."""
        return self.future_predictor(visual_history)

    def forward(self, frame: torch.Tensor,
                visual_history: torch.Tensor | None = None):
        """Per-tick (online) call, matching the AutoE2E wiring agreed 24/06:

            visual_embedding, future_state_pred = WorldActionModel(frame, visual_history)

        Args:
            frame: ``[B, 3, H, W]`` (or ``[B, V, 3, H, W]`` collapsed) current
                1 Hz multi-camera frame.
            visual_history: ``[B, history_len*frame_embed_dim]`` current circular
                buffer (the Encoded Visual History) used to predict the future;
                ``None`` at the very first ticks / pure inference.

        Returns:
            ``(visual_embedding, future_state_pred)`` where
            * ``visual_embedding`` ``[B, frame_embed_dim]`` is pushed to the
              external :class:`RollingHistoryBuffer` (FIFO, size N) which forms
              the Encoded Visual History fed to the reactive planner;
            * ``future_state_pred`` is a list of ``num_future_steps`` future
              backbone **feature maps** ``[B, feature_channels, hw, hw]`` (only
              needed in training; ``None`` if no ``visual_history`` is given). The
              JEPA loss is computed separately via :meth:`jepa_loss` (kept out of
              the model, in the training loop).
        """
        visual_embedding = self.encoder(frame)
        future_state_pred = (self.predict_future(visual_history)
                             if visual_history is not None else None)
        return visual_embedding, future_state_pred

    def jepa_loss(self, future_state_pred: list,
                  future_frames: torch.Tensor) -> torch.Tensor:
        """Future Feature Reconstruction Loss (JEPA, L1) vs the FROZEN target.

        Args:
            future_state_pred: list of ``num_future_steps`` feature maps
                ``[B, feature_channels, hw, hw]`` from :meth:`predict_future`.
            future_frames: ``[B, num_future_steps, 3, H, W]`` (or
                ``[B, num_future_steps, V, 3, H, W]``) actual future frames; the
                frozen target maps each to ``[B, feature_channels, hw, hw]``.
        """
        future_obs = [future_frames[:, k] for k in range(self.num_future_steps)]
        return compute_jepa_loss(future_state_pred, future_obs, self.target,
                                 self.recon_loss, weight=1.0)
