import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVViewFusion(nn.Module):
    """Fuse multi-view features into a BEV representation via spatial cross-attention.

    Follows the BEVFormer approach: learnable BEV queries attend to multi-camera
    image features at geometry-guided 3D reference points projected onto each
    camera's image plane. No explicit depth prediction is needed.

    When camera calibration parameters are not available, a learnable pseudo-projection
    is used as fallback, allowing the model to run and train without real calibration.

    Reference:
        - BEVFormer (Li et al., ECCV 2022): spatial cross-attention with 3D reference points
        - UniAD (Hu et al., CVPR 2023): uses BEVFormer encoder as default BEV backbone
    """

    def __init__(self, num_views=8, embed_dim=1440, bev_h=7, bev_w=7,
                 num_points_in_pillar=4, num_heads=8, dropout=0.1,
                 pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)):
        super().__init__()

        self.num_views = num_views
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_points_in_pillar = num_points_in_pillar
        self.num_heads = num_heads
        self.pc_range = pc_range

        # Learnable BEV queries: each grid cell gets its own query vector
        self.bev_queries = nn.Embedding(bev_h * bev_w, embed_dim)

        # Learnable pseudo-projection matrices for when camera params are unavailable
        # Shape: [num_views, 3, 4] — maps homogeneous 3D coords to 2D image coords
        self.pseudo_projection = nn.Parameter(
            torch.randn(num_views, 3, 4) * 0.01
        )

        # Sampling offsets predicted from BEV queries
        # Each head samples num_points_in_pillar points, each with (dx, dy) offset
        num_sample_points = num_heads * num_points_in_pillar
        self.sampling_offsets = nn.Linear(embed_dim, num_sample_points * 2)

        # Attention weights predicted from BEV queries
        self.attention_weights = nn.Linear(embed_dim, num_views * num_sample_points)

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
        These represent the 3D world locations that each BEV query should attend to.
        """
        xs = torch.linspace(0.5, self.bev_w - 0.5, self.bev_w) / self.bev_w
        ys = torch.linspace(0.5, self.bev_h - 0.5, self.bev_h) / self.bev_h
        zs = torch.linspace(0.5, self.num_points_in_pillar - 0.5,
                            self.num_points_in_pillar) / self.num_points_in_pillar

        # Create meshgrid: [bev_h, bev_w, num_z, 3]
        grid_y, grid_x, grid_z = torch.meshgrid(ys, xs, zs, indexing='ij')
        ref_3d = torch.stack([grid_x, grid_y, grid_z], dim=-1)

        # Reshape to [bev_h * bev_w, num_z, 3]
        ref_3d = ref_3d.reshape(self.bev_h * self.bev_w, self.num_points_in_pillar, 3)

        # Register as buffer (not a parameter, but moves with device)
        self.register_buffer('reference_points_3d', ref_3d)

    def _project_to_2d(self, reference_points_3d, camera_params=None):
        """Project 3D reference points to 2D coordinates on each camera's image plane.

        Args:
            reference_points_3d: [N, num_z, 3] normalized 3D points
            camera_params: [B, num_views, 3, 4] projection matrices (intrinsic @ extrinsic)
                          If None, uses learnable pseudo_projection.

        Returns:
            ref_2d: [B, num_views, N, num_z, 2] normalized 2D coordinates
            mask: [B, num_views, N, num_z] visibility mask
        """
        N, num_z, _ = reference_points_3d.shape

        # Scale normalized coords to world coordinates using pc_range
        pc_range = self.pc_range
        ref_world = reference_points_3d.clone()
        ref_world[..., 0] = ref_world[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_world[..., 1] = ref_world[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_world[..., 2] = ref_world[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # Convert to homogeneous coordinates: [N, num_z, 4]
        ones = torch.ones(*ref_world.shape[:-1], 1, device=ref_world.device)
        ref_homo = torch.cat([ref_world, ones], dim=-1)

        # Get projection matrices
        if camera_params is not None:
            proj = camera_params  # [B, num_views, 3, 4]
        else:
            proj = self.pseudo_projection.unsqueeze(0)  # [1, num_views, 3, 4]

        B = proj.shape[0]

        # Project: [B, num_views, 3, 4] x [N*num_z, 4, 1] -> [B, num_views, N*num_z, 3]
        ref_flat = ref_homo.reshape(N * num_z, 4)  # [N*num_z, 4]
        # Einstein notation: b v i j, n j -> b v n i
        projected = torch.einsum('bvij,nj->bvni', proj, ref_flat)  # [B, V, N*num_z, 3]

        # Perspective division (avoid division by zero)
        depth = projected[..., 2:3].clamp(min=1e-5)
        ref_2d = projected[..., :2] / depth  # [B, V, N*num_z, 2]

        # Reshape to [B, V, N, num_z, 2]
        ref_2d = ref_2d.reshape(B, self.num_views, N, num_z, 2)

        # Normalize to [0, 1] range using sigmoid (pseudo_projection outputs are unbounded)
        ref_2d = ref_2d.sigmoid()

        # Visibility mask: points within [0, 1] image bounds
        mask = (ref_2d[..., 0] > 0.01) & (ref_2d[..., 0] < 0.99) & \
               (ref_2d[..., 1] > 0.01) & (ref_2d[..., 1] < 0.99)

        return ref_2d, mask

    def forward(self, fused_per_view, B, V, camera_params=None):
        """
        Args:
            fused_per_view: [B*V, C, H, W] multi-view image features
            B: batch size
            V: number of views
            camera_params: [B, V, 3, 4] camera projection matrices (optional)

        Returns:
            bev_features: [B, C, bev_h, bev_w] BEV representation
        """
        C, H, W = fused_per_view.shape[1], fused_per_view.shape[2], fused_per_view.shape[3]
        N = self.bev_h * self.bev_w  # number of BEV queries

        # --- 1. Prepare BEV queries ---
        queries = self.bev_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N, C]

        # --- 2. Prepare image features as values ---
        # Reshape to [B, V, H*W, C] for value projection
        feat = fused_per_view.reshape(B, V, C, H * W).permute(0, 1, 3, 2)  # [B, V, H*W, C]
        values = self.value_proj(feat)  # [B, V, H*W, C]

        # --- 3. Project 3D reference points to 2D ---
        ref_2d, mask = self._project_to_2d(self.reference_points_3d, camera_params)
        # ref_2d: [B, V, N, num_z, 2], mask: [B, V, N, num_z]

        # --- 4. Predict sampling offsets and attention weights from queries ---
        offsets = self.sampling_offsets(queries)  # [B, N, num_heads * num_z * 2]
        offsets = offsets.reshape(B, N, self.num_heads, self.num_points_in_pillar, 2)
        offsets = offsets * 0.1  # Scale down offsets for stability

        attn_weights = self.attention_weights(queries)  # [B, N, V * num_heads * num_z]
        attn_weights = attn_weights.reshape(B, N, V, self.num_heads * self.num_points_in_pillar)
        attn_weights = attn_weights.softmax(dim=-1)  # Normalize over sampling points per view

        # --- 5. Sample features from each camera via grid_sample ---
        # Combine reference points with offsets for sampling locations
        # Average offsets across heads for simplicity
        offset_mean = offsets.mean(dim=2)  # [B, N, num_z, 2]

        output = torch.zeros(B, N, C, device=fused_per_view.device)
        visible_count = torch.zeros(B, N, 1, device=fused_per_view.device)

        # Reshape value-projected features to spatial format for grid_sample
        # values: [B, V, H*W, C] -> [B*V, C, H, W]
        values_spatial = values.permute(0, 1, 3, 2).reshape(B * V, C, H, W)

        for v_idx in range(V):
            # Sampling locations for this camera: [B, N, num_z, 2]
            sample_locs = ref_2d[:, v_idx] + offset_mean  # [B, N, num_z, 2]

            # Convert to grid_sample format: [-1, 1] range
            sample_grid = sample_locs * 2 - 1  # [B, N, num_z, 2]

            # Use value-projected features (not raw features) for sampling
            feat_v = values_spatial.reshape(B, V, C, H, W)[:, v_idx]  # [B, C, H, W]

            # grid_sample expects grid of shape [B, H_out, W_out, 2]
            # We treat our sampling as [B, N, num_z, 2]
            sampled = F.grid_sample(
                feat_v, sample_grid, mode='bilinear',
                padding_mode='zeros', align_corners=False
            )  # [B, C, N, num_z]

            # Weighted sum over pillar points using attention weights
            # attn_weights for this view: [B, N, num_heads * num_z]
            w = attn_weights[:, :, v_idx, :]  # [B, N, num_heads * num_z]
            # Reshape to match pillar points
            w = w.reshape(B, N, self.num_heads, self.num_points_in_pillar)
            w = w.mean(dim=2)  # Average across heads: [B, N, num_z]

            # sampled: [B, C, N, num_z] -> weighted sum over num_z
            sampled = sampled.permute(0, 2, 3, 1)  # [B, N, num_z, C]
            weighted = (sampled * w.unsqueeze(-1)).sum(dim=2)  # [B, N, C]

            # Apply visibility mask
            cam_mask = mask[:, v_idx].any(dim=-1).float().unsqueeze(-1)  # [B, N, 1]
            output = output + weighted * cam_mask
            visible_count = visible_count + cam_mask

        # Average across visible cameras
        output = output / visible_count.clamp(min=1.0)

        # --- 6. Post-attention: residual + LayerNorm + FFN ---
        output = queries + self.output_proj(output)
        output = self.norm1(output)
        output = output + self.ffn(output)
        output = self.norm2(output)

        # --- 7. Reshape to spatial BEV grid ---
        bev_features = output.reshape(B, self.bev_h, self.bev_w, C).permute(0, 3, 1, 2)

        return bev_features
