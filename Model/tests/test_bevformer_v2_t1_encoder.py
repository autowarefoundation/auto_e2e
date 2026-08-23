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
    sha256_file,
)
from model_components.feature_pyramid import BEVFormerFeaturePyramid
from model_components.losses import (
    BEVSegmentationAuxiliaryLoss,
    RouteReconstructionLoss,
)
from model_components.map_encoder import build_split_navigation_encoder
from model_components.view_fusion.bevformer_v2_t1 import (
    BEVFormerV2T1EncoderLayer,
    BEVFormerV2T1ViewFusion,
    MultiScaleSpatialCrossAttention,
)
from navigation.geometry import MAP_CHANNEL_COUNT, ROUTE_CHANNEL_COUNT
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
    assert all(feature.grad is not None for feature in features)


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

    assert report.loaded_tensor_count == len(source)
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


def test_production_module_composition_forward_and_backward():
    kwargs = reactive_model_kwargs(
        ReactiveTrainingStage.NUPLAN_FULL,
        num_views=1,
    )
    view_kwargs = dict(kwargs["view_fusion_kwargs"])
    view_kwargs.update({
        "bev_h": 4,
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
    camera = torch.randn(1, 1, 3, 256, 256)
    map_context = torch.rand(1, MAP_CHANNEL_COUNT, 450, 300)
    route = torch.rand(1, ROUTE_CHANNEL_COUNT, 450, 300)
    trajectory, auxiliary = model(
        camera,
        map_context,
        torch.randn(1, 896),
        torch.randn(1, 256),
        route_mask=route,
        map_valid=torch.ones(1, dtype=torch.bool),
        route_valid=torch.ones(1, dtype=torch.bool),
        mode="train",
    )

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
            num_views=8,
        ),
    )
    report = load_bevformer_v2_t1_checkpoint(
        model,
        checkpoint,
        expected_sha256=BEVFORMER_V2_T1_CHECKPOINT_SHA256,
    )

    assert report.loaded_tensor_count == 493
    assert report.loaded_element_count > 48_000_000
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
