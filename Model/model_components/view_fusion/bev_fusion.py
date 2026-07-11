import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection import (
    GEOMETRY_PSEUDO,
    VALID_GEOMETRY_TYPES,
    ImageTransform,
    PinholeProjection,
    PseudoProjection,
)


def _validate_offset_scale(offset_scale):
    if not isinstance(offset_scale, (int, float)) or isinstance(offset_scale, bool):
        raise ValueError(
            f"offset_scale must be a finite non-negative number, "
            f"got {offset_scale!r}."
        )
    if not math.isfinite(offset_scale) or offset_scale < 0:
        raise ValueError(
            f"offset_scale must be a finite non-negative number, "
            f"got {offset_scale!r}."
        )


class BEVViewFusion(nn.Module):
    """Fuse multi-view features into a BEV representation via learned sparse grid sampling.

    Simplified variant inspired by BEVFormer's spatial cross-attention:
    learnable BEV queries attend to multi-camera image features at
    geometry-guided 3D reference points projected onto each camera's
    image plane. No explicit depth prediction is needed.

    This is a simplified single-head implementation that does not fully
    replicate BEVFormer's multi-head deformable attention. It serves as
    a functional BEV fusion module suitable for experimentation and as a
    foundation for a full BEVFormer-style implementation.

    Geometry contract:
        Geometry is a projection OPERATOR (see ``projection.py``), not a fixed
        ``[B, V, 3, 4]`` matrix. ``forward`` takes a ``projection``
        (:class:`CameraProjectionModel`) that maps ego-frame BEV reference points
        (X=forward, Y=left, Z=up, metres) to normalized sampling coordinates plus
        a visibility mask on each camera's model-input image plane:

        - :class:`PinholeProjection` — the linear ``K @ T`` fast path.
        - :class:`FThetaProjection` — native fisheye (no rectification/FOV loss).
        - :class:`PseudoProjection` — learnable calibration-free prior; NOT real
          geometry. Requested explicitly via ``geometry_type="pseudo"``; the
          module never falls into it on behalf of a caller that meant to pass
          real calibration.

        There is no ``camera_params`` matrix argument on ``forward`` — a pinhole
        matrix is constructed by the caller as ``PinholeProjection(matrix)`` (or
        ``PinholeProjection.from_KT(K, T)``) and passed as the operator.

        The module is runtime-``V``-dynamic: one instance consumes batches of any
        camera count. ``num_views`` only sizes the default pseudo-prior expansion
        and is not otherwise baked into any tensor shape.

    Reference:
        - BEVFormer (Li et al., ECCV 2022): spatial cross-attention
        - UniAD (Hu et al., CVPR 2023): uses BEVFormer as BEV encoder
    """

    def __init__(self, num_views=8, embed_dim=256, bev_h=450, bev_w=300,
                 num_points_in_pillar=4, dropout=0.1,
                 pc_range=(-60.0, -60.0, -5.0, 120.0, 60.0, 3.0),
                 image_size=256, offset_scale=0.1):
        super().__init__()

        self.num_views = num_views
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.image_size = image_size
        # Model-input image frame the projection targets. A square shard/backbone
        # input is the common case; a caller can override per forward with a
        # non-square ImageTransform without changing the module.
        self.image_transform = ImageTransform.square(image_size)
        _validate_offset_scale(offset_scale)
        self.offset_scale = offset_scale

        # Learnable BEV queries: each grid cell gets its own query vector
        self.bev_queries = nn.Embedding(bev_h * bev_w, embed_dim)

        # Fallback for testing/ablation when camera calibration is unavailable.
        # This is NOT a substitute for real camera geometry — it provides a
        # fixed learned spatial prior that allows the module to run without
        # calibration data. For production use, pass a real projection operator.
        #
        # A single shared [3, 4] prior, expanded to V views at projection time
        # (see PseudoProjection), so one instance serves ANY view count. Kept as
        # the attribute name `pseudo_projection` and as a leaf Parameter so the
        # optimizer trains it and gradient-flow expectations hold.
        self.pseudo_projection = nn.Parameter(torch.randn(3, 4) * 0.01)

        # Sampling offsets predicted from BEV queries
        self.sampling_offsets = nn.Linear(embed_dim, num_points_in_pillar * 2)

        # Attention weights over pillar points only (height-level relevance),
        # shared across cameras and applied per-camera in the sampling loop.
        # There is NO per-camera axis: camera-level weighting comes from
        # visibility-based averaging (1/|V_hit|), matching BEVFormer's SCA, and a
        # fixed per-camera-slot head is ill-defined once slots vary by dataset
        # (L2D slot 3 != NVIDIA slot 3). Dropping the num_views factor is what
        # makes the layer runtime-V-dynamic.
        self.attention_weights = nn.Linear(embed_dim, num_points_in_pillar)

        # Value projection applied to image features
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection after attention
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Layer norm and FFN for post-attention processing
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

        self._init_reference_points()

    def _init_reference_points(self):
        """Pre-compute normalized 3D reference points for the BEV grid.

        Each BEV cell (x, y) gets a vertical "pillar" of points along Z.
        These represent the 3D ego/vehicle-frame locations that each BEV
        query should attend to.
        """
        xs = torch.linspace(0.5, self.bev_w - 0.5, self.bev_w) / self.bev_w
        ys = torch.linspace(0.5, self.bev_h - 0.5, self.bev_h) / self.bev_h
        zs = torch.linspace(0.5, self.num_points_in_pillar - 0.5,
                            self.num_points_in_pillar) / self.num_points_in_pillar

        grid_y, grid_x, grid_z = torch.meshgrid(ys, xs, zs, indexing='ij')
        ref_3d = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        ref_3d = ref_3d.reshape(self.bev_h * self.bev_w, self.num_points_in_pillar, 3)

        self.register_buffer('reference_points_3d', ref_3d)

    def _ego_reference_homo(self, reference_points_3d):
        """Denormalize [0,1] reference points to ego metres and homogenize.

        Returns ``[N*num_z, 4]`` ego-frame homogeneous points (X=forward,
        Y=left, Z=up), the input every projection operator expects.
        """
        pc_range = self.pc_range
        ref_world = reference_points_3d.clone()
        ref_world[..., 0] = ref_world[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_world[..., 1] = ref_world[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_world[..., 2] = ref_world[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

        ones = torch.ones(*ref_world.shape[:-1], 1,
                          device=ref_world.device, dtype=ref_world.dtype)
        ref_homo = torch.cat([ref_world, ones], dim=-1)  # [N, num_z, 4]
        return ref_homo.reshape(-1, 4)                   # [N*num_z, 4]

    def _project_to_2d(self, reference_points_3d, camera_params=None):
        """Project reference points to 2D — a test-only convenience wrapper.

        NOT part of the public ABI (the public geometry contract is a projection
        operator, see :meth:`forward`). A ``[B, V, 3, 4]`` ``camera_params``
        matrix is treated as a :class:`PinholeProjection`; ``None`` uses the
        pseudo prior. Kept because several unit tests probe the projection math
        directly with a raw matrix.

        Returns:
            ref_2d: [B, V, N, num_z, 2] normalized coords
            mask: [B, V, N, num_z] visibility mask
        """
        if camera_params is not None:
            projection = PinholeProjection(camera_params)
        else:
            projection = PseudoProjection(self.pseudo_projection, num_views=self.num_views)
        return self._project_operator(reference_points_3d, projection)

    def _project_operator(self, reference_points_3d, projection, image_transform=None):
        """Project reference points using a :class:`CameraProjectionModel`.

        ``V`` is derived from the operator (``projection.num_views``), never from
        the construction-time ``self.num_views`` — this is what makes the module
        runtime-``V``-dynamic. Uses the module's default ``image_transform`` (the
        model-input frame) unless one is supplied.

        Returns ``(ref_2d, mask)`` reshaped to ``[Bp, V, N, num_z, ...]``.
        """
        N, num_z, _ = reference_points_3d.shape
        ref_homo = self._ego_reference_homo(reference_points_3d)  # [N*num_z, 4]

        it = image_transform if image_transform is not None else self.image_transform
        result = projection.project_ego_to_image(ref_homo, it)
        Bp, V = result.uv_norm.shape[0], result.uv_norm.shape[1]

        ref_2d = result.uv_norm.reshape(Bp, V, N, num_z, 2)
        mask = result.valid_mask.reshape(Bp, V, N, num_z)
        return ref_2d, mask

    def _resolve_projection(self, projection, geometry_type, V):
        """Turn (projection | geometry_type) into a single projection operator,
        enforcing honest geometry.

        Rules:
          - a supplied operator must match the runtime view count V.
          - `geometry_type`, if given, must be a valid label and must agree with
            the operator actually supplied — you cannot label a pseudo run
            "pinhole". When no operator is supplied, only "pseudo" (or None) is
            allowed, so the calibration-free path is never entered on behalf of a
            caller that meant to pass real calibration.
        """
        if geometry_type is not None and geometry_type not in VALID_GEOMETRY_TYPES:
            raise ValueError(
                f"Unknown geometry_type {geometry_type!r}; "
                f"expected one of {VALID_GEOMETRY_TYPES}."
            )

        if projection is not None:
            if getattr(projection, "num_views", V) != V:
                raise ValueError(
                    f"projection.num_views ({projection.num_views}) != runtime V ({V})."
                )
            if geometry_type is not None and geometry_type != projection.geometry_type:
                raise ValueError(
                    f"geometry_type={geometry_type!r} contradicts the supplied "
                    f"{projection.geometry_type!r} projection operator."
                )
            return projection

        # No operator supplied → calibration-free pseudo prior, but only if the
        # caller did not claim real geometry.
        if geometry_type is not None and geometry_type != GEOMETRY_PSEUDO:
            raise ValueError(
                f"geometry_type={geometry_type!r} requires a projection operator, "
                f"but none was given."
            )
        return PseudoProjection(self.pseudo_projection, num_views=V)

    def forward(self, fused_per_view, B, V, projection=None, geometry_type=None,
                image_transform=None):
        """
        Args:
            fused_per_view: [B*V, C, H, W] multi-view image features
            B: batch size
            V: number of views (runtime; the module is V-dynamic)
            projection: a CameraProjectionModel (Pinhole / FTheta / Pseudo) — the
                general geometry ABI. Its `num_views` must equal V. This is the
                public geometry contract; there is no `[B,V,3,4]` matrix argument
                (a pinhole matrix lives inside PinholeProjection).
            geometry_type: optional explicit geometry label ("pinhole",
                "rectified_pinhole", "ftheta", "pseudo"). Must be consistent with
                the supplied operator; the module never claims real geometry
                without an operator, nor silently downgrades to the pseudo prior.
                When no `projection` is given, the pseudo prior is used (and
                `geometry_type`, if named, must be "pseudo").
            image_transform: optional ImageTransform describing the model-input
                image frame; defaults to the module's square input.

        Returns:
            bev_features: [B, C, bev_h, bev_w] BEV representation
        """
        C, H, W = fused_per_view.shape[1], fused_per_view.shape[2], fused_per_view.shape[3]
        N = self.bev_h * self.bev_w
        dtype = fused_per_view.dtype

        # --- 0. Resolve the projection operator (explicit, honest geometry) ---
        proj_op = self._resolve_projection(projection, geometry_type, V)

        # --- 1. Prepare BEV queries ---
        queries = self.bev_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N, C]

        # --- 2. Prepare image features as values ---
        feat = fused_per_view.reshape(B, V, C, H * W).permute(0, 1, 3, 2)  # [B, V, H*W, C]
        values = self.value_proj(feat)  # [B, V, H*W, C]

        # --- 3. Project 3D reference points to 2D ---
        ref_2d, mask = self._project_operator(self.reference_points_3d, proj_op, image_transform)
        # ref_2d: [Bp, V, N, num_z, 2], mask: [Bp, V, N, num_z]
        # (Bp is B for calibrated cameras, 1 for the batch-independent pseudo
        #  prior — it broadcasts across the batch in the sampling loop below.)

        # --- 4. Predict sampling offsets and attention weights from queries ---
        offsets = self.sampling_offsets(queries)  # [B, N, num_z * 2]
        offsets = offsets.reshape(B, N, self.num_points_in_pillar, 2)
        offsets = offsets * self.offset_scale  # Scale down for stability

        # Attention weights over pillar points only (height relevance), shared
        # across cameras (no per-camera axis — see __init__). Camera-level
        # weighting is visibility-based averaging (1/|V_hit|), BEVFormer's SCA.
        attn_weights = self.attention_weights(queries)  # [B, N, num_z]
        attn_weights = attn_weights.softmax(dim=-1)      # over pillar points

        # --- 5. Sample features from each camera via grid_sample ---
        values_spatial = values.permute(0, 1, 3, 2).reshape(B * V, C, H, W)

        output = torch.zeros(B, N, C, device=fused_per_view.device, dtype=dtype)
        visible_count = torch.zeros(B, N, 1, device=fused_per_view.device, dtype=dtype)

        for v_idx in range(V):
            # Per-view projection. ref_2d/mask have batch dim Bp (B for real
            # cameras, 1 for the pseudo prior); expand a size-1 batch to B so it
            # broadcasts uniformly across the batch.
            ref_2d_v = ref_2d[:, v_idx]   # [Bp, N, num_z, 2]
            ref_mask = mask[:, v_idx]     # [Bp, N, num_z]
            if ref_2d_v.shape[0] == 1 and B > 1:
                ref_2d_v = ref_2d_v.expand(B, -1, -1, -1)
                ref_mask = ref_mask.expand(B, -1, -1)

            # Sampling locations: reference + offset
            sample_locs = ref_2d_v + offsets  # [B, N, num_z, 2]

            # Recompute visibility mask AFTER adding offsets
            sample_in_bounds = (sample_locs[..., 0] >= 0) & (sample_locs[..., 0] <= 1) & \
                               (sample_locs[..., 1] >= 0) & (sample_locs[..., 1] <= 1)
            combined_mask = ref_mask & sample_in_bounds  # [B, N, num_z]

            # Convert [0, 1] to grid_sample's [-1, 1] range
            sample_grid = sample_locs * 2 - 1  # [B, N, num_z, 2]

            # Sample from value-projected features
            feat_v = values_spatial.reshape(B, V, C, H, W)[:, v_idx]  # [B, C, H, W]
            sampled = F.grid_sample(
                feat_v, sample_grid, mode='bilinear',
                padding_mode='zeros', align_corners=False
            )  # [B, C, N, num_z]

            # Apply per-point mask and re-normalize weights over valid points.
            # attn_weights is camera-independent [B, N, num_z] (shared per view).
            point_mask = combined_mask.float()  # [B, N, num_z]
            w = attn_weights * point_mask
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # Re-normalize

            # sampled: [B, C, N, num_z] → [B, N, num_z, C]
            sampled = sampled.permute(0, 2, 3, 1)
            weighted = (sampled * w.unsqueeze(-1)).sum(dim=2)  # [B, N, C]

            # Camera is visible if ANY pillar point is valid
            cam_visible = combined_mask.any(dim=-1).float().unsqueeze(-1)  # [B, N, 1]
            output = output + weighted * cam_visible
            visible_count = visible_count + cam_visible

        # Average across visible cameras (matches BEVFormer's 1/|V_hit|)
        output = output / visible_count.clamp(min=1.0)

        # --- 6. Post-attention: residual + LayerNorm + FFN ---
        output = queries + self.output_proj(output)
        output = self.norm1(output)
        output = output + self.ffn(output)
        output = self.norm2(output)

        # Zero out BEV cells with no visible camera observations.
        # Applied AFTER LayerNorm+FFN so that their biases don't override the mask.
        has_observation = (visible_count > 0).float()  # [B, N, 1]
        output = output * has_observation

        # --- 7. Reshape to spatial BEV grid ---
        bev_features = output.reshape(B, self.bev_h, self.bev_w, C).permute(0, 3, 1, 2)

        return bev_features
