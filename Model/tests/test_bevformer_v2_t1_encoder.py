"""Contracts for the t1 BEVFormer camera encoder and split navigation path."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from model_components.auxiliary_heads import (
    BEVSegmentationHead,
    RouteReconstructionHead,
)
from model_components.auto_e2e import AutoE2E
from model_components.backbone import Backbone
from model_components.bevformer_v2_pretrained import (
    BEVFORMER_V2_T1_CHECKPOINT_MIRROR_KEY,
    BEVFORMER_V2_T1_CHECKPOINT_SHA256,
    _adapt_camera_embeddings,
    _resize_bev_queries,
    _resize_position_embedding,
    bevformer_v2_t1_checkpoint_mirror_uri,
    load_bevformer_v2_t1_checkpoint,
    load_bevformer_v2_t8_checkpoint,
    sha256_file,
)
from model_components.feature_fusion import FeatureFusion
from model_components.feature_pyramid import BEVFormerFeaturePyramid
from model_components.losses import (
    BEVSegmentationAuxiliaryLoss,
    RouteReconstructionLoss,
)
from model_components.map_encoder import build_split_navigation_encoder
from model_components.reactive_e2e import ReactiveE2E
from model_components.view_fusion.bevformer_v2_t1 import (
    BEVFormerV2T1EncoderLayer,
    BEVFormerV2T1ViewFusion,
    MultiScaleSpatialCrossAttention,
)
from model_components.view_fusion.bevformer_v2_t8 import (
    BEVFormerV2T8TemporalFusion,
)
from model_components.view_fusion.projection import PinholeProjection
from navigation.geometry import MAP_CHANNEL_COUNT, ROUTE_CHANNEL_COUNT
from reactive_training_contracts import (
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
)
from training.reactive_multitask import (
    ReactiveTrainingStage,
    reactive_model_kwargs,
)


class _CaptureBackbone(torch.nn.Module):
    feature_info = [{"num_chs": 3}]

    def __init__(self) -> None:
        super().__init__()
        self.observed = None

    def forward(self, image):
        self.observed = image.detach().clone()
        return [image]


class _BatchNormBackbone(torch.nn.Module):
    feature_info = [{"num_chs": 3}]

    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(3)

    def forward(self, image):
        return [self.norm(image)]


def test_bevformer_input_profile_restores_official_bgr_scale(monkeypatch):
    capture = _CaptureBackbone()
    monkeypatch.setattr(
        "model_components.backbone.build_backbone",
        lambda *_args, **_kwargs: capture,
    )
    backbone = Backbone(
        backbone="stub",
        is_pretrained=False,
        input_profile="bevformer_v2",
    )
    raw_rgb = torch.tensor([10.0, 20.0, 30.0]).view(1, 3, 1, 1)
    imagenet = (
        raw_rgb / 255.0
        - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    backbone(imagenet)

    expected = raw_rgb.flip(dims=(1,)) - torch.tensor(
        [103.53, 116.28, 123.675]
    ).view(1, 3, 1, 1)
    assert torch.allclose(capture.observed, expected, atol=1e-5)


def test_bevformer_input_profile_freezes_batch_norm_statistics(monkeypatch):
    capture = _BatchNormBackbone()
    monkeypatch.setattr(
        "model_components.backbone.build_backbone",
        lambda *_args, **_kwargs: capture,
    )
    backbone = Backbone(
        backbone="stub",
        is_pretrained=False,
        input_profile="bevformer_v2",
    ).train()
    running_mean = capture.norm.running_mean.clone()
    running_var = capture.norm.running_var.clone()
    tracked = capture.norm.num_batches_tracked.clone()

    backbone(torch.randn(7, 3, 8, 8))
    backbone(torch.randn(1, 3, 16, 16))

    assert not capture.norm.training
    assert not capture.norm.weight.requires_grad
    assert not capture.norm.bias.requires_grad
    assert torch.equal(capture.norm.running_mean, running_mean)
    assert torch.equal(capture.norm.running_var, running_var)
    assert torch.equal(capture.norm.num_batches_tracked, tracked)


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


def test_checkpoint_mirror_uri_is_account_local_and_content_addressed():
    uri = bevformer_v2_t1_checkpoint_mirror_uri("123456789012")

    assert uri == (
        "s3://auto-e2e-platform-checkpoints-123456789012/"
        f"{BEVFORMER_V2_T1_CHECKPOINT_MIRROR_KEY}"
    )
    with pytest.raises(ValueError, match="12 digits"):
        bevformer_v2_t1_checkpoint_mirror_uri("123")


def test_camera_embeddings_are_averaged_even_for_six_target_views():
    source = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    adapted = _adapt_camera_embeddings(source, num_views=6)

    expected = source.mean(dim=0, keepdim=True).expand_as(source)
    assert torch.equal(adapted, expected)
    assert not torch.equal(adapted, source)


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


def test_front_fpn_keeps_native_scales_and_preserves_base_view_pyramid():
    fusion = FeatureFusion(
        num_views=3,
        backbone_channels=(8, 16, 32),
        embed_dim=8,
        view_fusion_kwargs={
            "architecture": "bevformer_v2_t1",
            "bev_h": 2,
            "bev_w": 3,
            "image_size": 32,
            "front_camera_index": 1,
            "front_image_size": 64,
            "num_heads": 2,
            "num_levels": 4,
            "num_points": 2,
            "num_encoder_layers": 1,
            "feedforward_channels": 16,
            "dropout": 0.0,
            "query_chunk_size": 3,
        },
    ).eval()
    with torch.no_grad():
        fusion.view_fusion.pseudo_projection.zero_()
        fusion.view_fusion.pseudo_projection[2, 3] = 1.0
    pyramid_shapes = []
    pyramid_outputs = []
    merged_pyramid = []
    base_bev = []

    def capture_pyramid(_module, _inputs, outputs):
        pyramid_shapes.append([
            tuple(output.shape) for output in outputs
        ])
        pyramid_outputs.append([
            output.detach().clone() for output in outputs
        ])

    pyramid_hook = fusion.feature_pyramid.register_forward_hook(
        capture_pyramid
    )
    fusion_pre_hook = fusion.view_fusion.register_forward_pre_hook(
        lambda _module, inputs: merged_pyramid.append([
            feature.detach().clone() for feature in inputs[0]
        ])
    )
    fusion_hook = fusion.view_fusion.register_forward_hook(
        lambda _module, _inputs, output: base_bev.append(output.detach())
    )
    base_features = [
        torch.randn(3, 8, 8, 8),
        torch.randn(3, 16, 4, 4),
        torch.randn(3, 32, 2, 2),
    ]
    front_features = [
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
    ]
    try:
        output = fusion(
            base_features,
            1,
            3,
            front_features=front_features,
        )
    finally:
        pyramid_hook.remove()
        fusion_pre_hook.remove()
        fusion_hook.remove()

    assert pyramid_shapes == [
        [
            (3, 8, 8, 8),
            (3, 8, 4, 4),
            (3, 8, 2, 2),
            (3, 8, 1, 1),
        ],
        [
            (1, 8, 16, 16),
            (1, 8, 8, 8),
            (1, 8, 4, 4),
            (1, 8, 2, 2),
        ],
    ]
    assert len(merged_pyramid) == 1
    assert all(
        torch.equal(merged, expected)
        for merged, expected in zip(
            merged_pyramid[0],
            pyramid_outputs[0],
        )
    )
    assert output.shape == (1, 8, 2, 3)
    assert torch.equal(output, base_bev[0])


def test_front_fpn_requires_the_ordinary_base_camera_branch():
    fusion = FeatureFusion(
        num_views=8,
        backbone_channels=(8, 16, 32),
        embed_dim=8,
        view_fusion_kwargs={
            "architecture": "bevformer_v2_t1",
            "bev_h": 2,
            "bev_w": 3,
            "image_size": 32,
            "front_image_size": 64,
            "num_heads": 2,
            "num_levels": 4,
            "num_points": 2,
            "num_encoder_layers": 1,
            "feedforward_channels": 16,
            "dropout": 0.0,
            "query_chunk_size": 3,
        },
    ).eval()
    with pytest.raises(ValueError, match="base-resolution camera features"):
        fusion(
            None,
            1,
            8,
            front_features=[
                torch.randn(1, 8, 16, 16),
                torch.randn(1, 16, 8, 8),
                torch.randn(1, 32, 4, 4),
            ],
        )


def test_calibrated_multiview_front_projection_and_gradient_contract():
    fusion = FeatureFusion(
        num_views=8,
        backbone_channels=(8, 16, 32),
        embed_dim=8,
        view_fusion_kwargs={
            "architecture": "bevformer_v2_t1",
            "bev_h": 2,
            "bev_w": 3,
            "pc_range": [1.0, -1.0, -1.0, 3.0, 1.0, 1.0],
            "image_size": 32,
            "front_camera_index": 0,
            "front_image_size": 64,
            "num_heads": 2,
            "num_levels": 4,
            "num_points": 2,
            "num_encoder_layers": 1,
            "feedforward_channels": 16,
            "dropout": 0.0,
            "query_chunk_size": 3,
        },
    ).eval()
    view_fusion = fusion.view_fusion
    assert isinstance(view_fusion, BEVFormerV2T1ViewFusion)
    base_matrix = torch.tensor([
        [16.0, 8.0, 0.0, 0.0],
        [16.0, 0.0, 8.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]).reshape(1, 1, 3, 4).expand(1, 8, -1, -1).clone()
    front_matrix = base_matrix[:, :1].clone()
    front_matrix[:, :, :2] *= 2.0
    base_projection = PinholeProjection(
        base_matrix,
        geometry_type="rectified_pinhole",
    )
    front_projection = PinholeProjection(
        front_matrix,
        geometry_type="rectified_pinhole",
    )

    base_reference, base_mask = view_fusion._project_operator(
        base_projection,
        view_fusion.image_transform,
    )
    front_reference, front_mask = view_fusion._project_operator(
        front_projection,
        view_fusion.front_image_transform,
    )
    torch.testing.assert_close(
        base_reference[:, 0],
        front_reference[:, 0],
    )
    assert torch.equal(base_mask[:, 0], front_mask[:, 0])

    base_features = [
        torch.randn(8, 8, 8, 8),
        torch.randn(8, 16, 4, 4),
        torch.randn(8, 32, 2, 2),
    ]
    front_features = [
        torch.randn(1, 8, 16, 16, requires_grad=True),
        torch.randn(1, 16, 8, 8, requires_grad=True),
        torch.randn(1, 32, 4, 4, requires_grad=True),
    ]
    base_bev = []
    hook = view_fusion.register_forward_hook(
        lambda _module, _inputs, output: base_bev.append(output.detach())
    )
    try:
        output = fusion(
            base_features,
            1,
            8,
            projection=base_projection,
            geometry_type="rectified_pinhole",
            front_features=front_features,
            front_projection=front_projection,
        )
    finally:
        hook.remove()
    assert torch.equal(output, base_bev[0])

    with torch.no_grad():
        view_fusion.front_residual_gate.fill_(0.25)
    enabled = fusion(
        base_features,
        1,
        8,
        projection=base_projection,
        geometry_type="rectified_pinhole",
        front_features=front_features,
        front_projection=front_projection,
    )
    enabled.square().mean().backward()

    assert not torch.equal(enabled, output)
    assert all(
        feature.grad is not None
        and torch.count_nonzero(feature.grad) > 0
        for feature in front_features
    )
    output_projection = view_fusion.front_cross_attention.output_proj
    assert output_projection.weight.grad is not None
    assert torch.count_nonzero(output_projection.weight.grad) > 0
    assert not view_fusion.front_cross_attention.value_proj.bias.requires_grad
    assert not output_projection.bias.requires_grad


def test_front_attention_gate_is_exact_noop_then_enables_native_gradient():
    fusion = BEVFormerV2T1ViewFusion(
        num_views=2,
        embed_dim=8,
        bev_h=2,
        bev_w=3,
        image_size=32,
        front_image_size=64,
        num_points_in_pillar=2,
        num_heads=2,
        num_levels=4,
        num_points=2,
        num_encoder_layers=1,
        feedforward_channels=16,
        dropout=0.0,
        query_chunk_size=3,
    ).eval()
    with torch.no_grad():
        fusion.pseudo_projection.zero_()
        fusion.pseudo_projection[2, 3] = 1.0
    image_bev = torch.randn(1, 8, 2, 3, requires_grad=True)
    native_features = [
        torch.randn(1, 8, 16, 16, requires_grad=True),
        torch.randn(1, 8, 8, 8, requires_grad=True),
        torch.randn(1, 8, 4, 4, requires_grad=True),
        torch.randn(1, 8, 2, 2, requires_grad=True),
    ]

    output = fusion.fuse_front_camera(image_bev, native_features)
    assert torch.equal(output, image_bev)
    output.square().mean().backward()
    assert fusion.front_residual_gate.grad is not None
    assert bool(fusion.front_residual_gate.grad.abs().max() > 0)
    assert all(
        feature.grad is not None
        and torch.count_nonzero(feature.grad) == 0
        for feature in native_features
    )

    fusion.zero_grad(set_to_none=True)
    image_bev.grad = None
    for feature in native_features:
        feature.grad = None
    with torch.no_grad():
        fusion.front_residual_gate.fill_(0.25)
    zero_output = fusion.fuse_front_camera(
        image_bev,
        [torch.zeros_like(feature) for feature in native_features],
    )
    assert torch.equal(zero_output, image_bev)

    enabled = fusion.fuse_front_camera(image_bev, native_features)
    enabled.square().mean().backward()

    assert not torch.equal(enabled, image_bev)
    assert all(
        feature.grad is not None
        and torch.count_nonzero(feature.grad) > 0
        for feature in native_features
    )


def test_front_attention_checkpoint_recomputes_and_preserves_bfloat16(
    monkeypatch,
):
    fusion = BEVFormerV2T1ViewFusion(
        num_views=2,
        embed_dim=8,
        bev_h=2,
        bev_w=3,
        image_size=32,
        front_image_size=64,
        num_points_in_pillar=2,
        num_heads=2,
        num_levels=4,
        num_points=2,
        num_encoder_layers=1,
        feedforward_channels=16,
        dropout=0.0,
        query_chunk_size=3,
        activation_checkpointing=True,
    ).train()
    with torch.no_grad():
        fusion.pseudo_projection.zero_()
        fusion.pseudo_projection[2, 3] = 1.0
        fusion.front_residual_gate.fill_(0.25)
    calls = []
    cross_attention = fusion.front_cross_attention
    original_forward = cross_attention.forward

    def recording_forward(*args, **kwargs):
        calls.append(1)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(
        cross_attention,
        "forward",
        recording_forward,
    )
    image_bev = torch.randn(
        1,
        8,
        2,
        3,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    native_features = [
        torch.randn(1, 8, 16, 16, requires_grad=True),
        torch.randn(1, 8, 8, 8, requires_grad=True),
        torch.randn(1, 8, 4, 4, requires_grad=True),
        torch.randn(1, 8, 2, 2, requires_grad=True),
    ]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = fusion.fuse_front_camera(
            image_bev,
            native_features,
        )
        output.float().square().mean().backward()

    assert output.dtype == image_bev.dtype
    assert len(calls) == 2
    assert all(feature.grad is not None for feature in native_features)


def test_reactive_front_path_processes_front_once_at_native_resolution():
    class RecordingBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def forward(self, image):
            self.shapes.append(tuple(image.shape))
            return [image]

    class RecordingFusion(torch.nn.Module):
        def forward(
            self,
            features,
            batch_size,
            num_views,
            **kwargs,
        ):
            assert features[0].shape == (8, 3, 4, 4)
            assert kwargs["front_features"][0].shape == (1, 3, 8, 8)
            return kwargs["front_features"][0].new_zeros(
                batch_size,
                3,
                2,
                2,
            )

    reactive = ReactiveE2E.__new__(ReactiveE2E)
    torch.nn.Module.__init__(reactive)
    reactive.camera_input_size = 4
    reactive.front_camera_index = 1
    reactive.front_camera_input_size = 8
    reactive.Backbone = RecordingBackbone()
    reactive.FeatureFusion = RecordingFusion()

    output = reactive.encode_camera_bev(
        torch.randn(1, 8, 3, 4, 4),
        front_camera_tile=torch.randn(1, 3, 8, 8),
    )

    assert output.shape == (1, 3, 2, 2)
    assert reactive.Backbone.shapes == [
        (8, 3, 4, 4),
        (1, 3, 8, 8),
    ]


def test_reactive_calibrated_l2d_omits_native_front_residual():
    class RecordingBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def forward(self, image):
            self.shapes.append(tuple(image.shape))
            return [image]

    class RecordingFusion(torch.nn.Module):
        def forward(self, features, batch_size, num_views, **kwargs):
            assert features[0].shape == (6, 3, 4, 4)
            assert "front_features" not in kwargs
            assert kwargs["projection"] is projection
            assert kwargs["geometry_type"] == "pinhole"
            return features[0].new_zeros(batch_size, 3, 2, 2)

    reactive = ReactiveE2E.__new__(ReactiveE2E)
    torch.nn.Module.__init__(reactive)
    reactive.camera_input_size = 4
    reactive.front_camera_index = 1
    reactive.front_camera_input_size = 8
    reactive.Backbone = RecordingBackbone()
    reactive.FeatureFusion = RecordingFusion()

    projection = object()
    output = reactive.encode_camera_bev(
        torch.randn(1, 6, 3, 4, 4),
        projection=projection,
        geometry_type="pinhole",
    )

    assert output.shape == (1, 3, 2, 2)
    assert reactive.Backbone.shapes == [(6, 3, 4, 4)]
    with pytest.raises(
        ValueError,
        match="geometry requires a native front",
    ):
        reactive.encode_camera_bev(
            torch.randn(1, 6, 3, 4, 4),
            projection=projection,
            geometry_type="pinhole",
            front_projection=object(),
        )


def test_reactive_pseudo_geometry_does_not_synthesize_native_front():
    class RecordingBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def forward(self, image):
            self.shapes.append(tuple(image.shape))
            return [image]

    class RecordingFusion(torch.nn.Module):
        def forward(self, features, batch_size, num_views, **kwargs):
            assert features[0].shape == (6, 3, 4, 4)
            assert "front_features" not in kwargs
            assert kwargs["projection"] is None
            assert kwargs["geometry_type"] == "pseudo"
            return features[0].new_zeros(batch_size, 3, 2, 2)

    reactive = ReactiveE2E.__new__(ReactiveE2E)
    torch.nn.Module.__init__(reactive)
    reactive.camera_input_size = 4
    reactive.front_camera_index = 1
    reactive.front_camera_input_size = 8
    reactive.Backbone = RecordingBackbone()
    reactive.FeatureFusion = RecordingFusion()

    output = reactive.encode_camera_bev(
        torch.randn(1, 6, 3, 4, 4),
        geometry_type="pseudo",
    )

    assert output.shape == (1, 3, 2, 2)
    assert reactive.Backbone.shapes == [(6, 3, 4, 4)]


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


def test_activation_checkpointing_recomputes_each_encoder_layer():
    fusion = BEVFormerV2T1ViewFusion(
        num_views=2,
        embed_dim=32,
        bev_h=3,
        bev_w=4,
        num_points_in_pillar=2,
        num_heads=4,
        num_levels=4,
        num_points=2,
        num_encoder_layers=2,
        feedforward_channels=64,
        dropout=0.0,
        query_chunk_size=5,
        activation_checkpointing=True,
    ).train()
    with torch.no_grad():
        fusion.pseudo_projection.zero_()
        fusion.pseudo_projection[2, 3] = 1.0
    features = [
        torch.randn(2, 32, 8, 8, requires_grad=True),
        torch.randn(2, 32, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
        torch.randn(2, 32, 1, 1, requires_grad=True),
    ]

    output = fusion(features, batch_size=1, num_views=2)
    output.square().mean().backward()

    assert output.shape == (1, 32, 3, 4)
    assert all(
        layer.self_attention.output_proj.weight.grad is not None
        for layer in fusion.layers
    )
    assert all(
        not layer.cross_attention.activation_checkpointing
        for layer in fusion.layers
    )
    assert all(feature.grad is not None for feature in features)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the BF16 checkpoint regression",
)
def test_cuda_bfloat16_checkpointed_encoder_and_front_backward():
    device = torch.device("cuda")
    fusion = BEVFormerV2T1ViewFusion(
        num_views=2,
        embed_dim=32,
        bev_h=3,
        bev_w=4,
        image_size=32,
        front_image_size=64,
        num_points_in_pillar=2,
        num_heads=4,
        num_levels=4,
        num_points=2,
        num_encoder_layers=2,
        feedforward_channels=64,
        dropout=0.0,
        query_chunk_size=5,
        activation_checkpointing=True,
    ).to(device).train()
    with torch.no_grad():
        fusion.pseudo_projection.zero_()
        fusion.pseudo_projection[2, 3] = 1.0
        fusion.front_residual_gate.fill_(0.25)
    features = [
        torch.randn(2, 32, 8, 8, device=device, requires_grad=True),
        torch.randn(2, 32, 4, 4, device=device, requires_grad=True),
        torch.randn(2, 32, 2, 2, device=device, requires_grad=True),
        torch.randn(2, 32, 1, 1, device=device, requires_grad=True),
    ]
    native_front_features = [
        torch.randn(1, 32, 16, 16, device=device, requires_grad=True),
        torch.randn(1, 32, 8, 8, device=device, requires_grad=True),
        torch.randn(1, 32, 4, 4, device=device, requires_grad=True),
        torch.randn(1, 32, 2, 2, device=device, requires_grad=True),
    ]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        image_bev = fusion(features, batch_size=1, num_views=2)
        output = fusion.fuse_front_camera(
            image_bev,
            native_front_features,
        )
        loss = output.float().square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert all(feature.grad is not None for feature in features)
    assert all(
        feature.grad is not None
        for feature in native_front_features
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


def test_official_embedding_resize_preserves_target_shapes():
    query = torch.arange(4 * 4 * 8, dtype=torch.float32).reshape(16, 8)
    position = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)

    assert _resize_bev_queries(query, height=7, width=5).shape == (35, 8)
    assert _resize_position_embedding(position, length=7).shape == (7, 4)


def test_official_embedding_resize_converts_bev_axis_orientation():
    query = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    position = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    resized_query = _resize_bev_queries(query, height=2, width=2)
    resized_position = _resize_position_embedding(position, length=2)

    assert torch.equal(
        resized_query[:, 0],
        torch.tensor([4.0, 2.0, 3.0, 1.0]),
    )
    assert torch.equal(
        resized_position,
        torch.tensor([[3.0, 4.0], [1.0, 2.0]]),
    )


def test_query_position_preserves_official_channel_half_order():
    fusion = BEVFormerV2T1ViewFusion(
        num_views=1,
        embed_dim=4,
        bev_h=2,
        bev_w=3,
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

    # Loader conversion maps official X to AutoE2E rows and official Y to
    # AutoE2E columns. Their original first/second channel halves must remain.
    assert torch.equal(
        position[0, 0],
        torch.tensor([10.0, 11.0, 100.0, 101.0]),
    )
    assert torch.equal(
        position[1, 2],
        torch.tensor([20.0, 21.0, 300.0, 301.0]),
    )


class _TinyBackbone(torch.nn.Module):
    backbone_name = "res_net_50"

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, kernel_size=1, bias=False)
        )


class _TinyFeatureFusion(torch.nn.Module):
    architecture = "bevformer_v2_t1"

    def __init__(self):
        super().__init__()
        self.feature_pyramid = BEVFormerFeaturePyramid(
            (2, 4, 8, 16, 32),
            embed_dim=8,
        )
        self.view_fusion = BEVFormerV2T1ViewFusion(
            num_views=2,
            embed_dim=8,
            bev_h=2,
            bev_w=2,
            num_points_in_pillar=2,
            num_heads=2,
            num_levels=4,
            num_points=2,
            num_encoder_layers=6,
            feedforward_channels=16,
            dropout=0.0,
            query_chunk_size=2,
        )


class _TinyReactive(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Backbone = _TinyBackbone()
        self.FeatureFusion = _TinyFeatureFusion()


class _TinyAutoE2E(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Reactive_E2E = _TinyReactive()


class _TinyT8FeatureFusion(_TinyFeatureFusion):
    architecture = "bevformer_v2_t8"

    def __init__(self):
        super().__init__()
        self.temporal_fusion = BEVFormerV2T8TemporalFusion(
            embed_dim=8,
            inter_channels=16,
        )


class _TinyT8Reactive(_TinyReactive):
    def __init__(self):
        super().__init__()
        self.FeatureFusion = _TinyT8FeatureFusion()


class _TinyT8AutoE2E(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Reactive_E2E = _TinyT8Reactive()


def _synthetic_official_state(model):
    target = model.state_dict()
    source = {}
    root = "Reactive_E2E."

    def add(source_name, target_name, value=None):
        target_value = target[root + target_name]
        source[source_name] = (
            torch.full_like(target_value, (len(source) + 1) / 1000.0)
            if value is None
            else value
        )

    add(
        "img_backbone.0.weight",
        "Backbone.backbone.0.weight",
    )
    for index in range(3):
        for field in ("weight", "bias"):
            add(
                f"img_neck.lateral_convs.{index}.conv.{field}",
                f"FeatureFusion.feature_pyramid."
                f"lateral_convs.{index}.{field}",
            )
    for index in range(4):
        for field in ("weight", "bias"):
            add(
                f"img_neck.fpn_convs.{index}.conv.{field}",
                f"FeatureFusion.feature_pyramid.fpn_convs.{index}.{field}",
            )
    add(
        "pts_bbox_head.bev_embedding.weight",
        "FeatureFusion.view_fusion.bev_queries.weight",
    )
    add(
        "pts_bbox_head.positional_encoding.col_embed.weight",
        "FeatureFusion.view_fusion.row_embed.weight",
    )
    add(
        "pts_bbox_head.positional_encoding.row_embed.weight",
        "FeatureFusion.view_fusion.col_embed.weight",
    )
    add(
        "pts_bbox_head.transformer.level_embeds",
        "FeatureFusion.view_fusion.level_embeddings",
    )
    camera_target = target[
        root + "FeatureFusion.view_fusion.camera_embeddings"
    ]
    source["pts_bbox_head.transformer.cams_embeds"] = torch.arange(
        6 * camera_target.shape[1],
        dtype=camera_target.dtype,
    ).reshape(6, camera_target.shape[1])

    for index in range(6):
        target_layer = f"FeatureFusion.view_fusion.layers.{index}"
        source_layer = (
            f"pts_bbox_head.transformer.encoder.layers.{index}"
        )
        for field in (
            "sampling_offsets.weight",
            "sampling_offsets.bias",
            "attention_weights.weight",
            "attention_weights.bias",
            "value_proj.weight",
            "value_proj.bias",
            "output_proj.weight",
            "output_proj.bias",
        ):
            add(
                f"{source_layer}.attentions.0.{field}",
                f"{target_layer}.self_attention.{field}",
            )
            cross_prefix = (
                f"{source_layer}.attentions.1.output_proj"
                if field.startswith("output_proj.")
                else (
                    f"{source_layer}.attentions.1."
                    f"deformable_attention.{field.rsplit('.', 1)[0]}"
                )
            )
            add(
                f"{cross_prefix}.{field.rsplit('.', 1)[1]}",
                f"{target_layer}.cross_attention.{field}",
            )
        for target_slot, source_slot in ((0, "0.0"), (3, "1")):
            for field in ("weight", "bias"):
                add(
                    f"{source_layer}.ffns.0.layers.{source_slot}.{field}",
                    f"{target_layer}.ffn.{target_slot}.{field}",
                )
        for norm_index in range(3):
            for field in ("weight", "bias"):
                add(
                    f"{source_layer}.norms.{norm_index}.{field}",
                    f"{target_layer}.norms.{norm_index}.{field}",
                )
    temporal_fusion = getattr(
        model.Reactive_E2E.FeatureFusion,
        "temporal_fusion",
        None,
    )
    if temporal_fusion is not None:
        for suffix in temporal_fusion.state_dict():
            add(
                f"pts_bbox_head.transformer.fusion.{suffix}",
                f"FeatureFusion.temporal_fusion.{suffix}",
            )
    return source


def test_synthetic_checkpoint_maps_every_camera_parameter(tmp_path):
    model = _TinyAutoE2E()
    source = _synthetic_official_state(model)
    checkpoint = tmp_path / "synthetic-bevformer-v2.pth"
    torch.save({"state_dict": source}, checkpoint)

    report = load_bevformer_v2_t1_checkpoint(
        model,
        checkpoint,
        expected_sha256=sha256_file(checkpoint),
    )

    assert report.loaded_tensor_count == len(source) + 8
    assert report.adapted_tensors == (
        "FeatureFusion.view_fusion.camera_embeddings",
    )
    expected_camera = source[
        "pts_bbox_head.transformer.cams_embeds"
    ].mean(dim=0)
    actual_camera = (
        model.Reactive_E2E.FeatureFusion.view_fusion.camera_embeddings
    )
    assert torch.allclose(
        actual_camera,
        expected_camera.unsqueeze(0).expand_as(actual_camera),
    )


def test_synthetic_t8_checkpoint_maps_every_temporal_tensor(tmp_path):
    model = _TinyT8AutoE2E()
    source = _synthetic_official_state(model)
    checkpoint = tmp_path / "synthetic-bevformer-v2-t8.pth"
    torch.save({"state_dict": source}, checkpoint)

    report = load_bevformer_v2_t8_checkpoint(
        model,
        checkpoint,
        expected_sha256=sha256_file(checkpoint),
    )

    assert report.loaded_tensor_count == len(source) + 8
    temporal_state = (
        model.Reactive_E2E.FeatureFusion.temporal_fusion.state_dict()
    )
    for suffix, actual in temporal_state.items():
        expected = source[
            f"pts_bbox_head.transformer.fusion.{suffix}"
        ]
        assert torch.equal(actual, expected)


def test_production_module_composition_forward_and_backward():
    kwargs = reactive_model_kwargs(
        ReactiveTrainingStage.NUPLAN_FULL,
        num_views=1,
    )
    view_kwargs = dict(kwargs["view_fusion_kwargs"])
    view_kwargs.update({
        "bev_h": 6,
        "bev_w": 4,
        "query_chunk_size": 4,
    })
    kwargs["view_fusion_kwargs"] = view_kwargs
    model = AutoE2E(
        backbone="res_net_50",
        embed_dim=256,
        is_pretrained=False,
        **kwargs,
    ).train()
    with pytest.raises(ValueError, match="image_size contract"):
        model.Reactive_E2E.encode_camera_bev(
            torch.randn(1, 1, 3, 256, 256)
        )
    pyramid_shapes = []
    pyramid = model.Reactive_E2E.FeatureFusion.feature_pyramid
    assert pyramid is not None
    hook = pyramid.register_forward_hook(
        lambda _module, _inputs, outputs: pyramid_shapes.extend(
            tuple(output.shape) for output in outputs
        )
    )
    camera = torch.randn(
        1,
        1,
        3,
        REACTIVE_CAMERA_IMAGE_SIZE,
        REACTIVE_CAMERA_IMAGE_SIZE,
    )
    map_context = torch.rand(1, MAP_CHANNEL_COUNT, 450, 300)
    route = torch.rand(1, ROUTE_CHANNEL_COUNT, 450, 300)
    try:
        trajectory, auxiliary = model(
            camera,
            map_context,
            torch.randn(1, 896),
            torch.randn(1, 256),
            route_mask=route,
            map_valid=torch.ones(1, dtype=torch.bool),
            route_valid=torch.ones(1, dtype=torch.bool),
            front_camera_tile=torch.randn(
                1,
                3,
                REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
                REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
            ),
            mode="train",
        )
    finally:
        hook.remove()

    assert pyramid_shapes == [
        (1, 256, 64, 64),
        (1, 256, 32, 32),
        (1, 256, 16, 16),
        (1, 256, 8, 8),
        (1, 256, 128, 128),
        (1, 256, 64, 64),
        (1, 256, 32, 32),
        (1, 256, 16, 16),
    ]
    assert trajectory.shape == (1, 128)
    assert auxiliary["bev_segmentation_logits"].shape == (1, 8, 450, 300)
    assert auxiliary["route_reconstruction_logits"].shape == (
        1,
        2,
        450,
        300,
    )
    bev_target = torch.zeros_like(auxiliary["bev_segmentation_logits"])
    bev_target[:, :, 10:20, 10:20] = 1.0
    bev_loss = BEVSegmentationAuxiliaryLoss([2.0] * 8)(
        auxiliary["bev_segmentation_logits"],
        bev_target,
        torch.ones_like(bev_target, dtype=torch.bool),
    )
    route_loss = RouteReconstructionLoss()(
        auxiliary["route_reconstruction_logits"],
        (route > 0.5).to(route.dtype),
        torch.ones(1, 2, dtype=torch.bool),
    )
    total = trajectory.square().mean() + bev_loss + route_loss
    total.backward()

    reactive = model.Reactive_E2E
    assert torch.isfinite(total)
    assert reactive.FeatureFusion.view_fusion.bev_queries.weight.grad is not None
    assert reactive.FeatureFusion.view_fusion.camera_embeddings.grad is not None
    assert reactive.MapBEVFusion.out_proj.weight.grad is not None
    assert reactive.NavigationEncoder.route_gate.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in reactive.Backbone.parameters()
    )


def test_production_t8_parameter_count_is_pinned():
    model = AutoE2E(
        backbone="res_net_50",
        embed_dim=256,
        is_pretrained=False,
        **reactive_model_kwargs(
            ReactiveTrainingStage.NUPLAN_FULL,
            num_views=6,
        ),
    )

    planner = model.Reactive_E2E.TrajectoryPlanner
    view_fusion = model.Reactive_E2E.FeatureFusion.view_fusion
    assert planner.num_points == 16
    assert planner.sampling_offsets.out_features == 32
    assert planner.attention_weights.out_features == 16
    assert view_fusion.front_image_transform.model_input_size == (
        REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
        REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
    )
    assert sum(
        parameter.numel()
        for parameter in model.parameters()
    ) == 79_906_522
    temporal_fusion = (
        model.Reactive_E2E.FeatureFusion.temporal_fusion
    )
    assert temporal_fusion is not None
    assert sum(
        parameter.numel()
        for parameter in temporal_fusion.parameters()
    ) == 30_809_856


@pytest.mark.integration
def test_local_official_checkpoint_import_report():
    checkpoint = Path(
        os.environ.get(
            "BEVFORMER_V2_T1_CHECKPOINT",
            "/tmp/bevformer-v2-r50-t1-epoch24.pth",
        )
    )
    if not checkpoint.exists():
        pytest.skip("official BEVFormer V2 t1 checkpoint is not local")
    model = AutoE2E(
        backbone="res_net_50",
        embed_dim=256,
        is_pretrained=False,
        **reactive_model_kwargs(
            ReactiveTrainingStage.NUPLAN_FULL,
            num_views=6,
        ),
    )
    report = load_bevformer_v2_t1_checkpoint(
        model,
        checkpoint,
        expected_sha256=BEVFORMER_V2_T1_CHECKPOINT_SHA256,
    )

    assert report.loaded_tensor_count == 501
    reactive = model.Reactive_E2E
    expected_camera_element_count = sum(
        value.numel()
        for value in reactive.Backbone.backbone.state_dict().values()
    ) + sum(
        parameter.numel()
        for parameter in reactive.FeatureFusion.parameters()
    ) - (
        reactive.FeatureFusion.view_fusion.pseudo_projection.numel()
        + reactive.FeatureFusion.view_fusion.front_residual_gate.numel()
    )
    assert report.loaded_element_count == expected_camera_element_count
    assert report.adapted_tensors == (
        "FeatureFusion.view_fusion.camera_embeddings",
    )
    camera_embeddings = (
        model.Reactive_E2E.FeatureFusion.view_fusion.camera_embeddings
    )
    assert torch.allclose(
        camera_embeddings,
        camera_embeddings[:1].expand_as(camera_embeddings),
    )
