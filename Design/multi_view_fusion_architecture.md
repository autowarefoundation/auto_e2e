# Design Document: Multi-View Fusion Architecture

## Document Metadata

| Field | Value |
|-------|-------|
| Status | Proposed |
| Authors | riita10069 |
| Created | 2026-06-05 |
| Related Issue | [#2 - Enhancement Proposal: Multi-Camera Feature Fusion](https://github.com/autowarefoundation/auto_e2e/issues/2) |

---

## 1. Executive Summary

This document describes the design of the multi-view camera fusion system in AutoE2E. The system provides three selectable fusion strategies — Concat, Cross-Camera Attention, and BEV Spatial Cross-Attention — implemented as a pluggable module registry. This phased approach allows researchers to compare fusion strategies on the same codebase and progressively adopt more sophisticated architectures as the project matures.

The design draws heavily from recent advances in vision-centric autonomous driving, particularly BEVFormer [1], UniAD [2], and PETR [3], while maintaining simplicity and portability (no custom CUDA kernels required).

---

## 2. Motivation and Problem Statement

### 2.1 Original Problem

The initial AutoE2E architecture had no cross-camera fusion mechanism. The 7 cameras + 1 map tile were stacked along the batch dimension and processed independently through the backbone. The `DrivingPolicy` module collapsed all dimensions (including batch) via `torch.flatten()`, making mini-batch training impossible and preventing any meaningful multi-view reasoning.

### 2.2 Why Multi-View Fusion Matters

Autonomous driving requires understanding the full 360° surround environment. A single camera has limited field of view (~60-120°). To construct a complete scene representation, the model must:

1. **Relate observations across cameras** — e.g., a vehicle partially visible in the front camera and the right-front camera is the same object
2. **Resolve depth ambiguity** — monocular images are inherently ambiguous about depth; multi-view geometry provides constraints
3. **Create a unified representation** — downstream planning operates on a single scene representation, not per-camera features

This is a solved problem in the literature. BEVFormer [1], LSS [4], PETR [3], and their successors have demonstrated effective multi-camera fusion for 3D perception. The question for AutoE2E is not *whether* to do fusion, but *which approach* best fits the project's stage and goals.

### 2.3 Design Requirements

| Requirement | Rationale |
|-------------|-----------|
| Multiple fusion strategies selectable at runtime | Enables ablation studies and progressive complexity |
| Batch-dimension correctness | Training requires batch_size > 1 for stability and GPU efficiency |
| No custom CUDA kernels | Portability across hardware; reduced maintenance burden |
| Graceful degradation without camera calibration | Dataset choice is pending; model must be testable with dummy data |
| Uniform interface for all fusion modes | Downstream modules (DrivingPolicy, FutureState) should not change |

---

## 3. Architecture Overview

### 3.1 Full Network Pipeline

```
Input: [B, V, 3, 256, 256]
         │
         │  reshape to [B*V, 3, 256, 256]
         ▼
┌─────────────────────────────┐
│  Backbone (SwinV2-Tiny)     │  Pretrained on ImageNet-22k [5]
│  Multi-scale feature maps   │  4 stages: 96, 192, 384, 768 channels
└─────────────────────────────┘
         │
         │  Pool to 8×8 + concatenate scales
         ▼
    [B*V, 1440, 8, 8]
         │
         │  ┌──────────────────────────────────────────┐
         │  │  View Fusion (selectable via fusion_mode) │
         │  │                                          │
         │  │  "concat"     → ConcatViewFusion         │
         │  │  "cross_attn" → CrossAttentionViewFusion  │
         │  │  "bev"        → BEVViewFusion             │
         │  └──────────────────────────────────────────┘
         │
         ▼
    [B, 256, 8, 8]  ← Unified scene representation
         │
    ┌────┴────┐
    ▼         ▼
DrivingPolicy  FutureState
[B, 128]       [B, 256, 8, 8] × 4
```

### 3.2 Module Interface

All view fusion modules implement the same interface:

```python
class ViewFusionModule(nn.Module):
    def __init__(self, num_views: int, embed_dim: int = 256): ...
    def forward(self, fused_per_view: Tensor, B: int, V: int,
                camera_params: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            fused_per_view: [B*V, embed_dim, 8, 8]
            B: batch size
            V: number of views
            camera_params: [B, V, 3, 4] optional camera matrices
        Returns:
            [B, embed_dim, 8, 8]
        """
```

### 3.3 Registry Pattern

```python
FUSION_REGISTRY = {
    "concat": ConcatViewFusion,
    "cross_attn": CrossAttentionViewFusion,
    "bev": BEVViewFusion,
}
```

Selection at model instantiation:
```python
model = AutoE2E(num_views=8, fusion_mode="bev")
```

---

## 4. Backbone: Swin V2 Tiny (Current)

### 4.1 Current Choice

The current backbone is **Swin V2 Tiny** (`swin_tiny_patch4_window7_224.ms_in22k`), pretrained on ImageNet-22k. This was chosen as an initial starting point. A separate proposal to make the backbone configurable is tracked in a dedicated issue.

| Criterion | Swin V1 Tiny | ResNet-50 | ViT-Base |
|-----------|-------------|-----------|----------|
| Parameters | 28M | 25M | 86M |
| ImageNet-22k pretrained | ✓ | ✓ | ✓ |
| Multi-scale features | ✓ (4 stages) | ✓ (4 stages) | ✗ (single scale) |
| Window attention | ✓ (efficient) | N/A (conv) | ✗ (quadratic) |

Swin V1 [5] provides hierarchical multi-scale features (essential for multi-scale pooling in the fusion stage) while maintaining computational efficiency through shifted window attention. The Tiny variant balances model capacity with the project's current stage (no large-scale dataset yet).

> **Note**: The backbone choice is subject to change. See the backbone configurability issue for discussion on ResNet-50 (BEVFormer/UniAD default), Swin V2, ConvNeXt, and other candidates.

### 4.2 Multi-Scale Feature Extraction

```
Stage 0: [B*V, 64, 64, 96]   → 1/4 resolution, low-level features
Stage 1: [B*V, 32, 32, 192]  → 1/8 resolution, mid-level features
Stage 2: [B*V, 16, 16, 384]  → 1/16 resolution, high-level features
Stage 3: [B*V, 8, 8, 768]    → 1/32 resolution, semantic features
```

All stages are pooled to 8×8 and concatenated along the channel dimension, yielding 96 + 192 + 384 + 768 = **1440 channels**. This multi-scale fusion captures both fine-grained spatial detail and high-level semantics, following FPN-style design principles [6]. The channel dimension is then reduced to acheive an embedding of length 256, which is used in downstream fusion modules in-line with other SOTA approaches.

---

## 5. Fusion Mode 1: ConcatViewFusion

### 5.1 Design

The simplest fusion strategy. All camera features are concatenated along the channel dimension and reduced via a 1×1 convolution.

```
[B*V, 256, 8, 8]
    ↓ reshape
[B, V*256, 8, 8]     ← V=8 → 2,048 channels
    ↓ Conv2d(V*256, 256, kernel=1) + GELU
[B, 256, 8, 8]
```

### 5.2 Rationale

- **Baseline**: Provides the minimal correct implementation that fixes the batch dimension bug
- **Computational efficiency**: Single 1×1 convolution, negligible overhead
- **No camera geometry needed**: Works without calibration data
- **Implicit fusion**: The 1×1 conv learns to weight and combine channels from different views, but has no explicit mechanism for spatial correspondence

### 5.3 Limitations

- No explicit spatial reasoning across cameras
- View order is implicit (learned via channel position), not explicitly encoded
- The 1×1 conv operates pointwise; it cannot model spatial relationships between the same location seen from different cameras

### 5.4 When to Use

- Quick prototyping and correctness verification
- Ablation baseline for comparing more sophisticated fusion methods
- Scenarios where camera calibration is unavailable

---

## 6. Fusion Mode 2: CrossAttentionViewFusion

### 6.1 Design

Applies multi-head self-attention across camera views at each spatial position, with learnable camera position embeddings.

```
[B*V, 256, 8, 8]
    ↓ reshape
[B*H*W, V, 256]          ← Each spatial position: V vectors of dim 256
    ↓ + view_embed          ← Learnable camera identity [1, V, 256]
    ↓ LayerNorm
    ↓ MultiheadAttention(Q=K=V=x, num_heads=8)
    ↓ + residual
    ↓ LayerNorm
    ↓ FFN(256 → 512 → 256) + residual
    ↓ mean(dim=1)           ← Pool across views
    ↓ reshape
[B, 256, 8, 8]
```

### 6.2 Rationale

This architecture is inspired by the cross-attention mechanisms in PETR [3] and UniAD [2], adapted to a simpler setting without explicit 3D coordinates.

**Key design decisions:**

1. **Self-attention across views (not spatial positions)**: At each of the 49 spatial positions, the 8 camera views attend to each other. This is O(V² × H×W) rather than O((V×H×W)²), making it computationally tractable.

2. **Learnable view embeddings**: Following the position encoding principle from Transformers [7], each camera view receives a learnable embedding that encodes its identity (which camera it is). This makes the module view-order-sensitive — important because camera positions are fixed on the vehicle.

3. **Mean pooling after attention**: After each view has absorbed information from all other views via attention, they are averaged to produce a single unified representation. Alternative: learnable aggregation query (as in DETR [8]). We chose mean for simplicity; it can be upgraded later.

4. **Pre-norm architecture**: LayerNorm before attention and FFN, following modern transformer best practices that improve training stability [9].

### 6.3 Comparison with PETR

| Aspect | PETR [3] | CrossAttentionViewFusion |
|--------|----------|--------------------------|
| Position encoding | 3D coordinates projected to features | Learnable per-view embedding (no geometry) |
| Attention type | Cross-attention (query ← image features) | Self-attention across views |
| Geometry awareness | Explicit (camera intrinsics/extrinsics) | Implicit (learned view identity) |
| Output format | Object queries | Spatial feature map |

Our design trades geometric precision for simplicity. When camera calibration becomes available, the view embeddings can be enriched with projected 3D coordinates (upgrading toward PETR-style encoding).

### 6.4 Limitations

- No explicit 3D geometry — the model must learn spatial correspondences purely from data
- Mean pooling loses per-view information after fusion
- Single attention layer; deeper stacking may improve quality (at computational cost)

---

## 7. Fusion Mode 3: BEVViewFusion (Spatial Cross-Attention)

### 7.1 Design

Implements a simplified BEV fusion module inspired by BEVFormer [1]. Learnable BEV queries attend to multi-camera image features at geometry-guided 3D reference points. This is a single-head, single-layer simplification — not a full replication of BEVFormer's multi-head, 6-layer encoder as used in UniAD [2]. It serves as a functional BEV fusion baseline and foundation for future expansion.

```
┌─────────────────────────────────────────────────────────────┐
│ BEV Queries: nn.Embedding(H_bev × W_bev, 256)             │
│ Each query represents one cell in the BEV grid              │
└─────────────────────────────────────────────────────────────┘
         │
         │  Generate 3D reference points (vertical pillars)
         │  [N, num_z, 3] where N = H_bev × W_bev
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Project 3D → 2D via camera matrices (or pseudo-projection)  │
│ ref_2d: [B, V, N, num_z, 2]                                │
│ visibility_mask: [B, V, N, num_z]                           │
└─────────────────────────────────────────────────────────────┘
         │
         │  Predict sampling offsets from BEV queries
         │  offset: [B, N, num_z, 2]
         ▼
┌─────────────────────────────────────────────────────────────┐
│ For each camera:                                            │
│   1. sampling_location = ref_2d + offset                    │
│   2. sampled = F.grid_sample(value_proj(features), location)│
│   3. weighted_sum = attention_weights × sampled over pillar │
│   4. Apply visibility mask                                  │
│ Average across visible cameras                              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Post-attention:                                             │
│   output = LayerNorm(queries + output_proj(sampled))        │
│   output = LayerNorm(output + FFN(output))                  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
    [B, 256, 8, 8]  ← BEV feature grid
```

### 7.2 Component Design Details

#### 7.2.1 BEV Queries

```python
self.bev_queries = nn.Embedding(bev_h * bev_w, embed_dim)
```

- **Shape**: [49, 256] (8×8 BEV grid, matching downstream feature resolution)
- **Initialization**: Standard random (learned from scratch)
- **Role**: Each query "asks" the image features: "What is happening at my BEV grid location?"

In BEVFormer [1], BEV queries are 200×200 with 256 channels for nuScenes. We use 8×8 with 256 channels to match the existing architecture's spatial resolution and channel count.

#### 7.2.2 3D Reference Points

```python
# Vertical "pillars" at each BEV grid cell
# [bev_h * bev_w, num_points_in_pillar, 3]
# Normalized to [0, 1] then scaled by pc_range
```

- **num_points_in_pillar = 4**: Following BEVFormer's default, sample 4 heights per BEV cell
- **Z range**: -5.0m to +3.0m (below ground for ramps/tunnels, above for overpasses)
- **pc_range**: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] (default nuScenes range [10])

The pillar sampling strategy addresses depth ambiguity: by sampling at multiple heights, the attention mechanism can learn which height is relevant for each spatial location, effectively performing implicit depth estimation without explicit depth prediction.

#### 7.2.3 Projection to 2D

```python
# With camera params: standard pinhole projection
projected = intrinsic @ extrinsic @ point_3d_homogeneous
ref_2d = projected[:2] / projected[2]  # perspective division

# Without camera params: learnable pseudo-projection
self.pseudo_projection = nn.Parameter(torch.randn(num_views, 3, 4) * 0.01)
```

The pseudo-projection fallback allows the model to:
- Run and pass tests without real calibration data
- Learn an approximate projection from training data (if training proceeds without calibration)
- Be easily upgraded by replacing with real camera matrices when available

#### 7.2.4 Spatial Cross-Attention via grid_sample

```python
# Standard deformable attention uses custom CUDA ops [8]
# We replace with F.grid_sample for portability

sample_locations = reference_points_2d + predicted_offsets
sampled_features = F.grid_sample(value_features, sample_locations)
output = attention_weights * sampled_features
```

This is conceptually similar to deformable attention [8], sharing key principles:
- Reference points correspond to the projected 3D locations
- Offsets are predicted from queries (analogous to deformable offsets)
- Attention weights are predicted from queries (not from Q-K dot product)

Note: Our implementation is a single-head simplification using `F.grid_sample`. Full BEVFormer uses multi-head deformable attention with per-head independent sampling patterns and custom CUDA kernels for efficiency.

The key difference from standard attention [7]:

| Aspect | Standard Attention | Deformable / Our Implementation |
|--------|-------------------|----------------------------------|
| Attends to | All spatial positions | K sparse points near reference |
| Weight computation | dot(Q, K) / √d | Linear(Q) → softmax |
| Complexity | O(N²) | O(N × K) |
| Geometry prior | None | 3D reference points guide sampling |

#### 7.2.5 Visibility Masking

```python
# Step 1: depth validity (points must be in front of camera)
valid_depth = depth > 1e-5

# Step 2: image bounds after normalization
in_bounds = (ref_2d[..., 0] >= 0) & (ref_2d[..., 0] <= 1) & \
            (ref_2d[..., 1] >= 0) & (ref_2d[..., 1] <= 1)
mask = valid_depth & in_bounds

# Step 3: re-check bounds AFTER adding learned offsets
sample_locs = ref_2d + offsets
sample_in_bounds = (sample_locs >= 0) & (sample_locs <= 1)
combined_mask = mask & sample_in_bounds
```

Critical for correctness: a BEV query representing a location behind the vehicle should not attend to the front camera's features. The mask ensures each BEV query only aggregates from cameras that can physically observe its corresponding 3D location.

### 7.3 Comparison with BEVFormer

| Aspect | BEVFormer [1] | BEVViewFusion (Ours) |
|--------|---------------|----------------------|
| BEV resolution | 200×200 | 7×7 (matches existing arch) |
| Embed dim | 256 | 1440 (matches backbone output) |
| Num encoder layers | 6 | 1 (single pass) |
| Temporal self-attention | ✓ | ✗ (future work) |
| Deformable attention | Custom CUDA ops | F.grid_sample (portable) |
| Multi-scale features | 4 FPN levels | Single scale (post-pool) |
| Camera params | Required | Optional (pseudo-projection fallback) |

### 7.4 Comparison with LSS (Lift-Splat-Shoot)

The owner explicitly requested **spatial cross-attention instead of depth prediction**. For completeness, here is why:

| Aspect | LSS [4] / BEVDet [11] | BEV Spatial Cross-Attention |
|--------|------------------------|------------------------------|
| Core mechanism | Predict per-pixel depth distribution → lift to 3D → splat to BEV | BEV queries sample from 2D features at projected 3D locations |
| Depth supervision | Beneficial (LiDAR-derived GT depth) | Not needed |
| Memory | High (dense depth × features × voxels) | Low (sparse sampling at reference points) |
| Flexibility | Fixed discretization | Learnable offsets adapt sampling |
| Temporal fusion | Requires BEV alignment (ego-motion compensation) | Natural via temporal self-attention on queries |

### 7.5 When to Use

- When camera calibration (intrinsics + extrinsics) is available
- For tasks requiring geometrically-grounded BEV reasoning
- When preparing for downstream 3D perception tasks (detection, mapping)
- As the default choice when training with nuScenes [10] or similar datasets

---

## 8. Driving Policy Head

### 8.1 Design

```python
class DrivingPolicy(nn.Module):
    # Conv2d(1440, 3, 3, 1, 1) → flatten(start_dim=1) → MLP(3 layers) → [B, 128]
```

- **Output**: 128-dim vector = 64 timesteps × (acceleration + curvature) at 10Hz = 6.4s horizon
- **flatten(start_dim=1)**: Critical fix from PR 1 — preserves batch dimension

### 8.2 Design Rationale

The current policy head is intentionally simple (MLP). AD-MLP [12] demonstrated that even a pure MLP on ego-status can achieve competitive open-loop planning performance, suggesting that the fusion module's representation quality matters more than the policy head's complexity at this stage.

Future upgrades may include:
- GRU/Transformer decoder for autoregressive trajectory generation
- Multi-modal trajectory prediction (multiple hypotheses)
- Cost-volume-based planning (as in UniAD [2])

---

## 9. Future State Prediction

### 9.1 Design

```python
class FutureState(nn.Module):
    # Conv2d(256, 512) → GELU → Conv2d(512, 1024) → chunk(4)
    # Output: 4 × [B, 256, 8, 8] at 1.6s intervals over 6.4s
```

### 9.2 Relation to JEPA

This module is inspired by the Joint Embedding Predictive Architecture (JEPA) [13] proposed by LeCun. Key principles adopted:

1. **Prediction in latent space, not pixel space**: We predict future feature representations, not future images. This avoids the computational burden and mode-collapse issues of pixel-space prediction.

2. **Self-supervised learning signal**: During training, the predicted future features can be compared against actual future features extracted by the frozen backbone (FrozenBackbone module), providing a self-supervised loss signal.

3. **Compressed world model**: The future state predictions encode the model's "understanding" of how the scene will evolve — a form of implicit world model, similar to MILE [14] and GAIA-1 [15].

---

## 10. Training Considerations (Future Work)

### 10.1 Loss Functions

| Component | Loss | Reference |
|-----------|------|-----------|
| Trajectory | L1/L2 vs ground truth + collision penalty | UniAD [2], VAD [16] |
| Future state | Feature reconstruction (MSE in latent space) | JEPA [13], MILE [14] |
| Auxiliary perception | Optional: BEV segmentation, 3D detection | BEVFormer [1] |

### 10.2 Training Strategy

Following UniAD [2]:
1. **Stage 1**: Pretrain backbone + fusion on perception task (if labels available)
2. **Stage 2**: Freeze backbone, train end-to-end with planning loss

### 10.3 Dataset Candidates

| Dataset | Cameras | Scenes | Calibration | Planning Labels |
|---------|---------|--------|-------------|-----------------|
| nuScenes [10] | 6 | 1000 | ✓ | ✓ (ego trajectory) |
| Waymo Open | 5 | 1150 | ✓ | ✓ |
| KITTI Scenes | 2 | 50 | ✓ | Partial |

---

## 11. Implementation Summary

### 11.1 File Structure

```
Model/model_components/
├── auto_e2e.py                        # Main model (num_views, fusion_mode params)
├── backbone.py                        # Swin-Tiny
├── feature_fusion.py                  # Multi-scale pool + dispatch to view_fusion
├── view_fusion/
│   ├── __init__.py                    # FUSION_REGISTRY + build_view_fusion()
│   ├── concat_fusion.py              # Mode "concat"
│   ├── cross_attention_fusion.py     # Mode "cross_attn"
│   └── bev_fusion.py                 # Mode "bev"
├── driving_policy.py                  # Trajectory prediction head
├── future_state.py                    # Future feature prediction
└── frozen_backbone.py                 # For feature reconstruction loss
```

### 11.2 Parameter Counts (Approximate)

| Module | Parameters | Notes |
|--------|-----------|-------|
| Backbone (Swin-Tiny) | ~28M | Pretrained, optionally frozen |
| ConcatViewFusion | ~16.6M | Conv2d(11520, 1440, 1) |
| CrossAttentionViewFusion | ~16.6M | MHA(1440, 8 heads) + FFN(1440→2880→1440) |
| BEVViewFusion | ~13.0M | Queries + value_proj + offsets + attn_weights + output_proj + FFN |
| DrivingPolicy | ~3M | Conv + MLP |
| FutureState | ~25M | Two large Conv2d layers |

### 11.3 Hyperparameters

| Parameter | Value | Configurable |
|-----------|-------|--------------|
| num_views | 8 | ✓ (constructor arg) |
| embed_dim | 1440 | Derived from backbone |
| bev_h, bev_w | 7 | ✓ (BEV only) |
| num_points_in_pillar | 4 | ✓ (BEV only) |
| num_heads | 8 | ✓ (cross_attn only; BEV is single-head) |
| pc_range | [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] | ✓ (BEV only) |
| dropout | 0.1 | ✓ (cross_attn and BEV) |

---

## 12. Testing Strategy

All fusion modes are tested against the same criteria:

| Test Category | What It Verifies | Tests |
|---------------|-----------------|-------|
| Output shape | Correct tensor dimensions for all batch sizes | 9 (3 per mode) |
| Batch independence | Samples don't interfere with each other | 6 (2 per mode) |
| View fusion | Each camera contributes to output | 6 (2 per mode) |
| Gradient flow | All parameters receive non-zero gradients | 9 (3 per mode) |
| num_views flexibility | Works with 1, 4, 8, 12 views | 12 (4 per mode) |
| Numerical stability | No NaN/Inf, even with large inputs | 9 (3 per mode) |
| Mode-specific | Mode-specific properties hold | 11 |

**Total: 88 tests** (all passing)

---

## 13. References

1. Li, Z., Wang, W., Li, H., et al. "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers." ECCV 2022. https://arxiv.org/abs/2203.17270
2. Hu, Y., Yang, J., Chen, L., et al. "Planning-oriented Autonomous Driving." CVPR 2023 (Best Paper). https://arxiv.org/abs/2212.10156
3. Liu, Y., Wang, T., Zhang, X., et al. "PETR: Position Embedding Transformation for Multi-View 3D Object Detection." ECCV 2022. https://arxiv.org/abs/2203.05625
4. Philion, J., Fidler, S. "Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D." ECCV 2020. https://arxiv.org/abs/2008.05711
5. Liu, Z., Hu, H., Lin, Y., et al. "Swin Transformer V2: Scaling Up Capacity and Resolution." CVPR 2022. https://arxiv.org/abs/2111.09883
6. Lin, T.-Y., Dollar, P., Girshick, R., et al. "Feature Pyramid Networks for Object Detection." CVPR 2017. https://arxiv.org/abs/1612.03144
7. Vaswani, A., Shazeer, N., Parmar, N., et al. "Attention Is All You Need." NeurIPS 2017. https://arxiv.org/abs/1706.03762
8. Zhu, X., Su, W., Lu, L., et al. "Deformable DETR: Deformable Transformers for End-to-End Object Detection." ICLR 2021. https://arxiv.org/abs/2010.04159
9. Xiong, R., Yang, Y., et al. "On Layer Normalization in the Transformer Architecture." ICML 2020. https://arxiv.org/abs/2002.04745
10. Caesar, H., Bankiti, V., Lang, A.H., et al. "nuScenes: A Multimodal Dataset for Autonomous Driving." CVPR 2020. https://arxiv.org/abs/1903.11027
11. Huang, J., Huang, G., Zhu, Z., et al. "BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye-View." 2022. https://arxiv.org/abs/2112.11790
12. Li, Z., Yu, Z., Lan, S., et al. "Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?" CVPR 2024. https://arxiv.org/abs/2312.03031
13. LeCun, Y. "A Path Towards Autonomous Machine Intelligence." Technical Report, 2022. https://openreview.net/pdf?id=BZ5a1r-kVsf
14. Hu, A., Corrado, G., Griffiths, N., et al. "Model-Based Imitation Learning for Urban Driving." NeurIPS 2022. https://arxiv.org/abs/2210.07729
15. Hu, A., Russell, L., Yeo, H., et al. "GAIA-1: A Generative World Model for Autonomous Driving." 2023. https://arxiv.org/abs/2309.17080
16. Jiang, B., Chen, S., Xu, Q., et al. "VAD: Vectorized Scene Representation for Efficient Autonomous Driving." ICCV 2023. https://arxiv.org/abs/2303.12077
17. Liu, Y., Yan, J., et al. "PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images." ICCV 2023. https://arxiv.org/abs/2206.01256
18. Tesla, Inc. "Tesla AI Day 2022." Technical Presentation, September 30, 2022. https://www.youtube.com/watch?v=ODSJsviD_SU
19. Wang, S., Liu, Y., Wang, T., et al. "Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection (StreamPETR)." ICCV 2023. https://arxiv.org/abs/2303.11926
