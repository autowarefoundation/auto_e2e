import torch
import torch.nn as nn
import torch.nn.functional as F


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
        image_size parameter (default: 224, matching square input resolution).

        When camera_params is None, a learnable pseudo_projection is used
        as a shape-testing and ablation fallback. This does NOT learn true
        camera geometry — it acts as a fixed learned spatial prior for
        sampling locations. For meaningful BEV projection, real camera
        calibration is required.

    Reference:
        - BEVFormer (Li et al., ECCV 2022): spatial cross-attention
        - UniAD (Hu et al., CVPR 2023): uses BEVFormer as BEV encoder
    """

    def __init__(self, num_views=8, embed_dim=1440, bev_h=7, bev_w=7,
                 num_points_in_pillar=4, dropout=0.1,
                 pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 image_size=224):
        super().__init__()

        self.num_views = num_views
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.image_size = image_size

        # Learnable BEV queries: each grid cell gets its own query vector
        self.bev_queries = nn.Embedding(bev_h * bev_w, embed_dim)

        # Fallback for testing/ablation when camera calibration is unavailable.
        # This is NOT a substitute for real camera geometry — it provides a
        # fixed learned spatial prior that allows the module to run without
        # calibration data. For production use, pass real camera_params.
        self.pseudo_projection = nn.Parameter(
            torch.randn(num_views, 3, 4) * 0.01
        )

        # Sampling offsets predicted from BEV queries
        self.sampling_offsets = nn.Linear(embed_dim, num_points_in_pillar * 2)

        # Attention weights over (num_views × num_points_in_pillar).
        # Softmax is applied per-camera over pillar points (height-level relevance).
        # Camera-level weighting uses visibility-based averaging, matching
        # BEVFormer's SCA formula: output = (1/|V_hit|) * sum_{i in V_hit}(...)
        self.attention_weights = nn.Linear(
            embed_dim, num_views * num_points_in_pillar
        )

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

    def _project_to_2d(self, reference_points_3d, camera_params=None):
        """Project 3D reference points to normalized 2D coordinates on each camera.

        Args:
            reference_points_3d: [N, num_z, 3] normalized 3D points in [0, 1]
            camera_params: [B, num_views, 3, 4] ego-to-pixel projection matrices.
                If None, uses pseudo_projection fallback (testing/ablation only).

        Returns:
            ref_2d: [B, num_views, N, num_z, 2] coordinates in [0, 1] range
                    (normalized by image_size)
            mask: [B, num_views, N, num_z] visibility mask
                  True where: depth > 0 AND 0 <= u,v <= 1
        """
        N, num_z, _ = reference_points_3d.shape

        # Scale normalized [0,1] coords to ego/vehicle coordinates using pc_range
        pc_range = self.pc_range
        ref_world = reference_points_3d.clone()
        ref_world[..., 0] = ref_world[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_world[..., 1] = ref_world[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_world[..., 2] = ref_world[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # Homogeneous coordinates: [N, num_z, 4]
        ones = torch.ones(*ref_world.shape[:-1], 1,
                          device=ref_world.device, dtype=ref_world.dtype)
        ref_homo = torch.cat([ref_world, ones], dim=-1)

        # Select projection matrices
        if camera_params is not None:
            proj = camera_params  # [B, num_views, 3, 4]
        else:
            proj = self.pseudo_projection.unsqueeze(0)  # [1, num_views, 3, 4]

        B = proj.shape[0]

        # Project: einsum 'bvij,nj->bvni'
        ref_flat = ref_homo.reshape(N * num_z, 4)
        projected = torch.einsum('bvij,nj->bvni', proj, ref_flat)  # [B, V, N*num_z, 3]

        # Separate depth before perspective division
        depth = projected[..., 2]  # [B, V, N*num_z]
        valid_depth = depth > 1e-5  # Points in front of camera

        # Perspective division (clamp only to avoid NaN, masked points are excluded)
        depth_safe = depth.clamp(min=1e-5).unsqueeze(-1)  # [B, V, N*num_z, 1]
        ref_2d = projected[..., :2] / depth_safe  # [B, V, N*num_z, 2] in pixel coords

        # Normalize pixel coordinates to [0, 1] using image size
        if camera_params is not None:
            ref_2d = ref_2d / self.image_size
        else:
            # pseudo_projection outputs are unbounded; use sigmoid as fallback
            ref_2d = ref_2d.sigmoid()

        # Reshape to [B, V, N, num_z, 2]
        ref_2d = ref_2d.reshape(B, self.num_views, N, num_z, 2)
        valid_depth = valid_depth.reshape(B, self.num_views, N, num_z)

        # Visibility mask: in front of camera AND within image bounds [0, 1]
        in_bounds = (ref_2d[..., 0] >= 0) & (ref_2d[..., 0] <= 1) & \
                    (ref_2d[..., 1] >= 0) & (ref_2d[..., 1] <= 1)
        mask = valid_depth & in_bounds

        return ref_2d, mask

    def forward(self, fused_per_view, B, V, camera_params=None):
        """
        Args:
            fused_per_view: [B*V, C, H, W] multi-view image features
            B: batch size
            V: number of views
            camera_params: [B, V, 3, 4] ego-to-pixel projection matrices.
                Combined intrinsic @ extrinsic that maps homogeneous 3D
                ego/vehicle-frame coordinates [x, y, z, 1] to pixel
                coordinates [u, v, depth].
                If None, pseudo_projection is used (testing/ablation only).

        Returns:
            bev_features: [B, C, bev_h, bev_w] BEV representation
        """
        C, H, W = fused_per_view.shape[1], fused_per_view.shape[2], fused_per_view.shape[3]
        N = self.bev_h * self.bev_w
        dtype = fused_per_view.dtype

        # --- 1. Prepare BEV queries ---
        queries = self.bev_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N, C]

        # --- 2. Prepare image features as values ---
        feat = fused_per_view.reshape(B, V, C, H * W).permute(0, 1, 3, 2)  # [B, V, H*W, C]
        values = self.value_proj(feat)  # [B, V, H*W, C]

        # --- 3. Project 3D reference points to 2D ---
        ref_2d, mask = self._project_to_2d(self.reference_points_3d, camera_params)
        # ref_2d: [B, V, N, num_z, 2], mask: [B, V, N, num_z]

        # --- 4. Predict sampling offsets and attention weights from queries ---
        offsets = self.sampling_offsets(queries)  # [B, N, num_z * 2]
        offsets = offsets.reshape(B, N, self.num_points_in_pillar, 2)
        offsets = offsets * 0.1  # Scale down for stability

        # Attention weights: per-camera softmax over pillar points (height relevance).
        # Camera-level weighting is handled by visibility-based averaging,
        # matching BEVFormer's SCA: output = (1/|V_hit|) * sum_{i in V_hit}(...)
        attn_weights = self.attention_weights(queries)  # [B, N, V * num_z]
        attn_weights = attn_weights.reshape(B, N, V, self.num_points_in_pillar)
        attn_weights = attn_weights.softmax(dim=-1)  # Per-camera over pillar points

        # --- 5. Sample features from each camera via grid_sample ---
        values_spatial = values.permute(0, 1, 3, 2).reshape(B * V, C, H, W)

        output = torch.zeros(B, N, C, device=fused_per_view.device, dtype=dtype)
        visible_count = torch.zeros(B, N, 1, device=fused_per_view.device, dtype=dtype)

        for v_idx in range(V):
            # Sampling locations: reference + offset
            sample_locs = ref_2d[:, v_idx] + offsets  # [B, N, num_z, 2]

            # Recompute visibility mask AFTER adding offsets
            sample_in_bounds = (sample_locs[..., 0] >= 0) & (sample_locs[..., 0] <= 1) & \
                               (sample_locs[..., 1] >= 0) & (sample_locs[..., 1] <= 1)
            ref_mask = mask[:, v_idx]  # [B, N, num_z] (depth + original bounds)
            combined_mask = ref_mask & sample_in_bounds  # [B, N, num_z]

            # Convert [0, 1] to grid_sample's [-1, 1] range
            sample_grid = sample_locs * 2 - 1  # [B, N, num_z, 2]

            # Sample from value-projected features
            feat_v = values_spatial.reshape(B, V, C, H, W)[:, v_idx]  # [B, C, H, W]
            sampled = F.grid_sample(
                feat_v, sample_grid, mode='bilinear',
                padding_mode='zeros', align_corners=False
            )  # [B, C, N, num_z]

            # Apply per-point mask and re-normalize weights over valid points
            point_mask = combined_mask.float()  # [B, N, num_z]
            w = attn_weights[:, :, v_idx, :]  # [B, N, num_z]
            w = w * point_mask
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
