import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection import (
    GEOMETRY_PSEUDO,
    VALID_GEOMETRY_TYPES,
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

    Camera Parameters Convention:
        camera_params: [B, num_views, 3, 4] matrices that project
        homogeneous 3D ego/vehicle-frame coordinates to 2D pixel coordinates.

        Expected to be: intrinsic @ extrinsic (ego-to-pixel)
        - extrinsic: [3, 4] or [4, 4] ego/vehicle-frame-to-camera transform
        - intrinsic: [3, 3] camera matrix (focal length, principal point)
        - Combined: [3, 4] = intrinsic @ extrinsic[:3, :]

        The BEV reference points are defined in ego/vehicle frame (centered
        on the vehicle, X=forward, Y=left, Z=up). camera_params must
        transform these ego-frame coordinates to pixel coordinates.

        Output of projection: [u, v, depth] in pixel coordinates where
        - u: horizontal pixel (0 to image_width)
        - v: vertical pixel (0 to image_height)
        - depth: distance along camera optical axis (positive = in front)

        The module normalizes pixel coords to [0, 1] using the provided
        image_size parameter (default: 256, matching square input resolution).

        Geometry is supplied to ``forward`` as a projection operator (see
        ``projection.py``): pass a ``[B, V, 3, 4]`` ``camera_params`` matrix
        (wrapped as a :class:`PinholeProjection`), or a pre-built ``projection``
        operator (e.g. :class:`FThetaProjection` for native fisheye), or request
        the calibration-free path explicitly with ``geometry_type="pseudo"``.
        The pseudo path uses a learnable ``pseudo_projection`` prior; it does NOT
        learn true camera geometry — it is a fixed learned spatial prior for
        sampling locations. For meaningful BEV projection, real calibration is
        required, and the fusion module never falls into the pseudo path on
        behalf of a caller that meant to pass real calibration.

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
        _validate_offset_scale(offset_scale)
        self.offset_scale = offset_scale

        # Learnable BEV queries: each grid cell gets its own query vector
        self.bev_queries = nn.Embedding(bev_h * bev_w, embed_dim)

        # Fallback for testing/ablation when camera calibration is unavailable.
        # This is NOT a substitute for real camera geometry — it provides a
        # fixed learned spatial prior that allows the module to run without
        # calibration data. For production use, pass real camera_params.
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
        """Project 3D reference points to normalized 2D coordinates on each camera.

        Backward-compatible convenience wrapper: a ``[B, V, 3, 4]``
        ``camera_params`` matrix is treated as a :class:`PinholeProjection`, and
        ``None`` uses the learnable pseudo prior. New code should prefer passing
        a projection operator via :meth:`_project_operator`.

        Args:
            reference_points_3d: [N, num_z, 3] normalized 3D points in [0, 1]
            camera_params: [B, V, 3, 4] ego-to-pixel projection matrices.
                If None, uses pseudo_projection fallback (testing/ablation only).

        Returns:
            ref_2d: [B, V, N, num_z, 2] coordinates in [0, 1] range
            mask: [B, V, N, num_z] visibility mask (depth > 0 AND 0 <= u,v <= 1)
        """
        if camera_params is not None:
            projection = PinholeProjection(camera_params)
        else:
            projection = PseudoProjection(self.pseudo_projection, num_views=self.num_views)
        return self._project_operator(reference_points_3d, projection)

    def _project_operator(self, reference_points_3d, projection):
        """Project reference points using a :class:`CameraProjectionModel`.

        ``V`` is derived from the operator (``projection.num_views``), never from
        the construction-time ``self.num_views`` — this is what makes the module
        runtime-``V``-dynamic.

        Returns ``(ref_2d, mask)`` reshaped to ``[Bp, V, N, num_z, ...]`` so the
        sampling loop can index per view.
        """
        N, num_z, _ = reference_points_3d.shape
        ref_homo = self._ego_reference_homo(reference_points_3d)  # [N*num_z, 4]

        result = projection.project(ref_homo, self.image_size)
        Bp, V = result.uv_norm.shape[0], result.uv_norm.shape[1]

        ref_2d = result.uv_norm.reshape(Bp, V, N, num_z, 2)
        mask = result.valid_mask.reshape(Bp, V, N, num_z)
        return ref_2d, mask

    def _resolve_projection(self, camera_params, projection, geometry_type, V, B):
        """Turn the (camera_params | projection | geometry_type) inputs into a
        single projection operator, enforcing honest geometry.

        Rules:
          - `camera_params` and `projection` are mutually exclusive.
          - a supplied operator/matrix must match the runtime view count V.
          - `geometry_type`, if given, must be a valid label and must agree with
            the geometry actually supplied — you cannot label a pseudo run
            "pinhole", and asking for "pseudo" while passing real calibration is
            rejected. When no geometry is supplied at all, only "pseudo" (or
            None) is allowed, so the calibration-free path is never entered on
            behalf of a caller that meant to pass calibration.
        """
        if geometry_type is not None and geometry_type not in VALID_GEOMETRY_TYPES:
            raise ValueError(
                f"Unknown geometry_type {geometry_type!r}; "
                f"expected one of {VALID_GEOMETRY_TYPES}."
            )
        if camera_params is not None and projection is not None:
            raise ValueError("Pass at most one of camera_params or projection.")

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

        if camera_params is not None:
            if camera_params.shape[1] != V:
                raise ValueError(
                    f"camera_params view dim ({camera_params.shape[1]}) != runtime V ({V})."
                )
            if camera_params.shape[0] not in (1, B):
                raise ValueError(
                    f"camera_params batch dim ({camera_params.shape[0]}) must be 1 "
                    f"(shared rig, broadcast) or B ({B})."
                )
            if geometry_type == GEOMETRY_PSEUDO:
                raise ValueError(
                    "geometry_type='pseudo' but real camera_params were supplied; "
                    "the pseudo path must not consume real calibration."
                )
            gt = geometry_type or "pinhole"
            return PinholeProjection(camera_params, geometry_type=gt)

        # No calibration supplied → calibration-free pseudo prior, but only if
        # the caller did not claim real geometry.
        if geometry_type is not None and geometry_type != GEOMETRY_PSEUDO:
            raise ValueError(
                f"geometry_type={geometry_type!r} requires calibration "
                f"(camera_params or a projection operator), but none was given."
            )
        return PseudoProjection(self.pseudo_projection, num_views=V)

    def forward(self, fused_per_view, B, V, camera_params=None,
                projection=None, geometry_type=None):
        """
        Args:
            fused_per_view: [B*V, C, H, W] multi-view image features
            B: batch size
            V: number of views (runtime; the module is V-dynamic)
            camera_params: [B, V, 3, 4] ego-to-pixel pinhole matrices (intrinsic
                @ extrinsic). Wrapped as a PinholeProjection. Mutually exclusive
                with `projection`.
            projection: a CameraProjectionModel (Pinhole / FTheta / Pseudo) to use
                directly — the general geometry ABI. Its `num_views` must equal V.
            geometry_type: optional explicit geometry label ("pinhole",
                "rectified_pinhole", "ftheta", "pseudo"). If given it must be
                consistent with the supplied geometry — the module never claims
                real geometry without calibration, nor silently downgrades a
                real-calibration request to the pseudo prior. When neither
                `camera_params` nor `projection` is given, the calibration-free
                pseudo prior is used (and `geometry_type`, if named, must be
                "pseudo").

        Returns:
            bev_features: [B, C, bev_h, bev_w] BEV representation
        """
        C, H, W = fused_per_view.shape[1], fused_per_view.shape[2], fused_per_view.shape[3]
        N = self.bev_h * self.bev_w
        dtype = fused_per_view.dtype

        # --- 0. Resolve the projection operator (explicit, honest geometry) ---
        proj_op = self._resolve_projection(camera_params, projection, geometry_type, V, B)

        # --- 1. Prepare BEV queries ---
        queries = self.bev_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N, C]

        # --- 2. Prepare image features as values ---
        feat = fused_per_view.reshape(B, V, C, H * W).permute(0, 1, 3, 2)  # [B, V, H*W, C]
        values = self.value_proj(feat)  # [B, V, H*W, C]

        # --- 3. Project 3D reference points to 2D ---
        ref_2d, mask = self._project_operator(self.reference_points_3d, proj_op)
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
