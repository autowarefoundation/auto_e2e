"""Contracts for the t1 BEVFormer camera encoder and split navigation path."""

from __future__ import annotations

import pytest
import torch

from model_components.auxiliary_heads import (
    BEVSegmentationHead,
    RouteReconstructionHead,
)
from model_components.feature_pyramid import BEVFormerFeaturePyramid
from model_components.map_encoder import build_split_navigation_encoder
from model_components.view_fusion.bevformer_v2_t1 import (
    BEVFormerV2T1EncoderLayer,
    BEVFormerV2T1ViewFusion,
    MultiScaleSpatialCrossAttention,
)
from model_components.view_fusion.projection import PinholeProjection


def test_projection_and_sampling_grid_stay_fp32_under_bf16_autocast(
    monkeypatch,
):
    matrix = torch.tensor([[[[
        73.125,
        0.0,
        11.75,
        3.0,
    ], [
        0.0,
        71.875,
        9.25,
        -2.0,
    ], [
        0.0,
        0.0,
        1.0,
        0.0,
    ]]]])
    points = torch.tensor([
        [1.2345, -0.7654, 12.345],
        [-2.3456, 1.4567, 23.456],
    ])
    projection = PinholeProjection(matrix)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        projected = projection.project_ego_to_image(points, 512)

    assert projected.uv_norm.dtype == torch.float32

    attention = MultiScaleSpatialCrossAttention(
        embed_dim=4,
        num_heads=1,
        num_levels=1,
        num_points=1,
    )
    observed = {}
    original_grid_sample = torch.nn.functional.grid_sample

    def capture_grid_sample(value, grid, **kwargs):
        observed["value_dtype"] = value.dtype
        observed["grid_dtype"] = grid.dtype
        return original_grid_sample(value, grid, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional,
        "grid_sample",
        capture_grid_sample,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = attention._sample_level(
            torch.randn(1, 4, 2, 2, dtype=torch.bfloat16),
            torch.zeros(1, 1, 1, 2),
            torch.ones(1, 1, 1, dtype=torch.bfloat16),
        )

    assert observed == {
        "value_dtype": torch.float32,
        "grid_dtype": torch.float32,
    }
    assert output.dtype == torch.bfloat16


def test_feature_pyramid_retains_four_native_scales():
    pyramid = BEVFormerFeaturePyramid(
        (64, 256, 512, 1024, 2048),
        embed_dim=32,
    )
    features = [
        torch.randn(2, 64, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 512, 8, 8),
        torch.randn(2, 1024, 4, 4),
        torch.randn(2, 2048, 2, 2),
    ]

    outputs = pyramid(features)

    assert [tuple(output.shape) for output in outputs] == [
        (2, 32, 8, 8),
        (2, 32, 4, 4),
        (2, 32, 2, 2),
        (2, 32, 1, 1),
    ]


def test_t1_encoder_layer_is_finite_and_checkpoint_compatible():
    layer = BEVFormerV2T1EncoderLayer(
        embed_dim=32,
        num_heads=4,
        num_levels=4,
        num_points=2,
        feedforward_channels=64,
        dropout=0.0,
        query_chunk_size=5,
    )
    query = torch.randn(1, 12, 32, requires_grad=True)
    position = torch.randn_like(query)
    features = [
        torch.randn(2, 32, 8, 8, requires_grad=True),
        torch.randn(2, 32, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
        torch.randn(2, 32, 1, 1, requires_grad=True),
    ]
    references = torch.full((1, 2, 12, 2, 2), 0.5)
    mask = torch.ones(1, 2, 12, 2, dtype=torch.bool)

    output = layer(
        query,
        position,
        features,
        references,
        mask,
        bev_h=3,
        bev_w=4,
        num_views=2,
        level_embeddings=torch.randn(4, 32),
        camera_embeddings=torch.randn(2, 32),
    )
    output.square().mean().backward()

    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert query.grad is not None
    assert all(feature.grad is not None for feature in features)
    assert layer.self_attention.sampling_offsets.weight.shape == (
        4 * 2 * 4 * 2,
        32 * 2,
    )


def test_spatial_attention_checkpoints_each_view_level_chunk(monkeypatch):
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    calls = []

    def recording_checkpoint(function, *args, **kwargs):
        calls.append(function)
        return torch_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(
        "model_components.view_fusion.bevformer_v2_t1.checkpoint",
        recording_checkpoint,
    )
    attention = MultiScaleSpatialCrossAttention(
        embed_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
        dropout=0.0,
        query_chunk_size=4,
        activation_checkpointing=True,
    ).train()
    query = torch.randn(1, 6, 8, requires_grad=True)
    features = [
        torch.randn(2, 8, 4, 4, requires_grad=True),
        torch.randn(2, 8, 2, 2, requires_grad=True),
    ]
    references = torch.full((1, 2, 6, 2, 2), 0.5)
    mask = torch.ones(1, 2, 6, 2, dtype=torch.bool)

    output = attention(
        query,
        features,
        references,
        mask,
        num_views=2,
        level_embeddings=torch.randn(2, 8),
        camera_embeddings=torch.randn(2, 8),
    )
    output.square().mean().backward()

    assert len(calls) == 2 * 2 * 2
    assert query.grad is not None
    assert all(feature.grad is not None for feature in features)


def test_spatial_attention_keeps_official_unmasked_anchor_semantics():
    attention = MultiScaleSpatialCrossAttention(
        embed_dim=1,
        num_heads=1,
        num_levels=1,
        num_points=2,
        dropout=0.0,
        query_chunk_size=1,
    ).eval()
    with torch.no_grad():
        attention.sampling_offsets.weight.zero_()
        attention.sampling_offsets.bias.zero_()
        attention.attention_weights.weight.zero_()
        attention.attention_weights.bias.zero_()
        attention.value_proj.weight.fill_(1.0)
        attention.value_proj.bias.zero_()
        attention.output_proj.weight.fill_(1.0)
        attention.output_proj.bias.zero_()
    query = torch.zeros(1, 1, 1)
    features = [torch.ones(1, 1, 2, 2)]
    references = torch.full((1, 1, 1, 2, 2), 0.5)
    mask = torch.tensor([[[[True, False]]]])

    output = attention(
        query,
        features,
        references,
        mask,
        num_views=1,
        level_embeddings=torch.zeros(1, 1),
        camera_embeddings=torch.zeros(1, 1),
    )

    # Official SCA uses the mask to select the query/camera pair, then samples
    # all Z anchors. Masking the second anchor here would produce 0.5, not 1.0.
    assert torch.allclose(output, torch.ones_like(output))


def test_split_navigation_encoder_has_late_per_channel_route_gate():
    encoder = build_split_navigation_encoder(
        "semantic_raster",
        map_channels=14,
        route_channels=2,
        embed_dim=32,
        output_h=16,
        output_w=16,
        route_hidden_channels=24,
    )
    map_raster = torch.randn(1, 14, 20, 12)
    route_zero = torch.zeros(1, 2, 20, 12)
    route_one = torch.ones_like(route_zero)

    base, base_route = encoder(
        map_raster,
        route_zero,
        return_route_contribution=True,
    )
    changed, changed_route = encoder(
        map_raster,
        route_one,
        return_route_contribution=True,
    )

    assert encoder.route_gate.shape == (32,)
    assert torch.allclose(encoder.route_gate_values(), torch.full((32,), 0.5))
    assert base.shape == changed.shape == (1, 32, 16, 16)
    assert not torch.allclose(base_route, changed_route)
    assert torch.allclose(
        changed - base,
        changed_route - base_route,
        atol=1e-6,
    )


def test_auxiliary_heads_restore_supervision_resolution():
    features = torch.randn(1, 32, 16, 16)
    bev_head = BEVSegmentationHead(
        embed_dim=32,
        hidden_channels=16,
        num_classes=8,
        output_size=(45, 30),
    )
    route_head = RouteReconstructionHead(
        embed_dim=32,
        hidden_channels=16,
        route_channels=2,
        output_size=(45, 30),
    )

    assert bev_head(features).shape == (1, 8, 45, 30)
    assert route_head(features).shape == (1, 2, 45, 30)


def test_query_position_preserves_official_channel_half_order():
    fusion = BEVFormerV2T1ViewFusion(
        num_views=1,
        embed_dim=4,
        bev_h=2,
        bev_w=3,
        pc_range=(-30.0, -40.0, -0.5, 80.0, 40.0, 3.5),
        num_points_in_pillar=2,
        num_heads=2,
        num_levels=1,
        num_points=2,
        num_encoder_layers=1,
        feedforward_channels=8,
    )
    with torch.no_grad():
        fusion.row_embed.weight.copy_(
            torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        )
        fusion.col_embed.weight.copy_(
            torch.tensor([
                [100.0, 101.0],
                [200.0, 201.0],
                [300.0, 301.0],
            ])
        )

    position = fusion._query_position(1).reshape(2, 3, 4)

    # Rows are ego X and columns ego Y; their embeddings keep the first and
    # second channel halves.
    assert torch.equal(
        position[0, 0],
        torch.tensor([10.0, 11.0, 100.0, 101.0]),
    )
    assert torch.equal(
        position[1, 2],
        torch.tensor([20.0, 21.0, 300.0, 301.0]),
    )


