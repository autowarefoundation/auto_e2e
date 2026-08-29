"""BEVFormer V2 T8 temporal and checkpoint contracts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from model_components.auto_e2e import AutoE2E
from model_components.bevformer_v2_pretrained import (
    BEVFORMER_V2_SOURCE_REVISION,
    BEVFORMER_V2_T8_CHECKPOINT_MIRROR_KEY,
    BEVFORMER_V2_T8_CHECKPOINT_SHA256,
    BEVFORMER_V2_T8_TEMPORAL_FRAME_ORDER,
    BEVFORMER_V2_T8_TEMPORAL_FUSION_SOURCE_URL,
    bevformer_v2_t8_checkpoint_mirror_uri,
    load_bevformer_v2_t8_checkpoint,
)
from model_components.feature_fusion import FeatureFusion
from model_components.view_fusion.bevformer_v2_t8 import (
    BEVFormerV2T8TemporalFusion,
)
from model_components.reactive_e2e import ReactiveE2E
from training.reactive_multitask import (
    ReactiveTrainingStage,
    reactive_model_kwargs,
)
from training.reactive_stage_runner import (
    resolve_reactive_camera_history,
    resolve_reactive_front_projection,
)


def test_t8_checkpoint_mirror_is_content_addressed():
    uri = bevformer_v2_t8_checkpoint_mirror_uri("123456789012")

    assert uri == (
        "s3://auto-e2e-platform-checkpoints-123456789012/"
        f"{BEVFORMER_V2_T8_CHECKPOINT_MIRROR_KEY}"
    )
    assert BEVFORMER_V2_T8_CHECKPOINT_SHA256 in (
        BEVFORMER_V2_T8_CHECKPOINT_MIRROR_KEY
    )


def test_t8_upstream_temporal_frame_order_is_revision_pinned():
    assert BEVFORMER_V2_SOURCE_REVISION == (
        "66b65f3a1f58caf0507cb2a971b9c0e7f842376c"
    )
    assert BEVFORMER_V2_T8_TEMPORAL_FRAME_ORDER == (
        -7,
        -6,
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
    )
    assert BEVFORMER_V2_SOURCE_REVISION in (
        BEVFORMER_V2_T8_TEMPORAL_FUSION_SOURCE_URL
    )
    assert BEVFORMER_V2_T8_TEMPORAL_FUSION_SOURCE_URL.endswith(
        "transformerV2.py#L308-L324"
    )


def test_t8_fusion_preserves_order_and_current_gradient():
    torch.manual_seed(7)
    fusion = BEVFormerV2T8TemporalFusion(
        embed_dim=4,
        inter_channels=8,
        num_layers=3,
    ).eval()
    history = [
        torch.randn(2, 4, 3, 2)
        for _ in range(7)
    ]
    current = torch.randn(2, 4, 3, 2, requires_grad=True)

    output = fusion([*history, current])
    reordered = fusion([
        history[1],
        history[0],
        *history[2:],
        current,
    ])
    output.square().mean().backward()

    assert output.shape == current.shape
    assert not torch.equal(output, reordered)
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert any(
        parameter.grad is not None
        for parameter in fusion.parameters()
    )


def test_t8_fusion_rejects_incomplete_history():
    fusion = BEVFormerV2T8TemporalFusion(
        embed_dim=4,
        inter_channels=8,
    )

    with pytest.raises(ValueError, match="8 ordered"):
        fusion([torch.zeros(1, 4, 2, 2) for _ in range(7)])


def test_t8_feature_fusion_keeps_temporal_batch_norm_single_update():
    fusion = FeatureFusion(
        num_views=2,
        backbone_channels=(8, 16, 32),
        embed_dim=8,
        view_fusion_kwargs={
            "architecture": "bevformer_v2_t8",
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
            "activation_checkpointing": True,
        },
    )

    assert fusion.temporal_fusion is not None
    assert not fusion.temporal_fusion.activation_checkpointing
    frames = [
        torch.randn(2, 8, 2, 3)
        for _ in range(7)
    ]
    current = torch.randn(2, 8, 2, 3, requires_grad=True)

    fusion.fuse_temporal_bevs([*frames, current]).square().mean().backward()

    batch_norms = [
        module
        for module in fusion.temporal_fusion.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    ]
    assert batch_norms
    assert all(
        int(module.num_batches_tracked) == 1
        for module in batch_norms
    )


def test_history_encoder_is_sequential_detached_and_restores_mode():
    class StubBackbone(torch.nn.Module):
        def forward(self, value):
            return [value]

    class StubFusion(torch.nn.Module):
        def forward(
            self,
            features,
            batch_size,
            num_views,
            **_kwargs,
        ):
            value = features[0].reshape(
                batch_size,
                num_views,
                *features[0].shape[1:],
            )
            return value.mean(dim=(1, 2), keepdim=False).unsqueeze(1)

    reactive = ReactiveE2E.__new__(ReactiveE2E)
    torch.nn.Module.__init__(reactive)
    reactive.camera_architecture = "bevformer_v2_t8"
    reactive.camera_input_size = 4
    reactive.Backbone = StubBackbone().train()
    reactive.FeatureFusion = StubFusion().train()
    history = torch.randn(
        2,
        7,
        3,
        3,
        4,
        4,
        requires_grad=True,
    )

    outputs = reactive._encode_history_camera_bevs(
        history,
        geometry_type="pseudo",
    )

    assert len(outputs) == 7
    assert all(not output.requires_grad for output in outputs)
    assert reactive.Backbone.training
    assert reactive.FeatureFusion.training


def test_missing_history_fills_detached_current_bevs():
    class CaptureFusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frames = None

        def fuse_temporal_bevs(self, frames):
            self.frames = frames
            return torch.stack(frames).sum(dim=0)

    reactive = ReactiveE2E.__new__(ReactiveE2E)
    torch.nn.Module.__init__(reactive)
    reactive.camera_architecture = "bevformer_v2_t8"
    reactive.FeatureFusion = CaptureFusion()
    current = torch.randn(1, 4, 2, 2, requires_grad=True)

    output = reactive._fuse_temporal_camera_bevs(current, [])
    output.sum().backward()

    assert reactive.FeatureFusion.frames is not None
    assert len(reactive.FeatureFusion.frames) == 8
    assert all(
        not frame.requires_grad
        for frame in reactive.FeatureFusion.frames[:7]
    )
    assert reactive.FeatureFusion.frames[-1] is current
    assert torch.equal(current.grad, torch.ones_like(current))


def test_history_projection_resolver_preserves_oldest_to_newest_order():
    tiles = torch.zeros(2, 7, 3, 3, 4, 4)
    matrix = torch.zeros(2, 7, 3, 3, 4)
    matrix[:, :, :, 2, 3] = 1.0
    for history_index in range(7):
        matrix[:, history_index, :, 0, 3] = history_index

    resolved_tiles, projections = resolve_reactive_camera_history(
        {
            "camera_history_tiles": tiles,
            "camera_history_projection_matrix": matrix,
        },
        "rectified_pinhole",
        device=torch.device("cpu"),
    )

    assert resolved_tiles is tiles
    assert projections is not None
    assert len(projections) == 7
    for history_index, projection in enumerate(projections):
        assert torch.equal(
            projection.matrix,
            matrix[:, history_index],
        )


def test_stage_a_camera_resolvers_fail_closed_on_missing_batch_fields():
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="native front camera"):
        resolve_reactive_front_projection(
            {},
            "rectified_pinhole",
            device=device,
            required=True,
        )
    with pytest.raises(ValueError, match="T8 camera history"):
        resolve_reactive_camera_history(
            {},
            "rectified_pinhole",
            device=device,
            required=True,
        )

    assert (
        resolve_reactive_front_projection(
            {},
            "rectified_pinhole",
            device=device,
        )
        is None
    )
    assert resolve_reactive_camera_history(
        {},
        "rectified_pinhole",
        device=device,
    ) == (None, None)


def test_official_t8_checkpoint_imports_all_camera_parameters():
    checkpoint = Path(os.environ.get(
        "BEVFORMER_V2_T8_CHECKPOINT",
        "/tmp/bevformerv2-r50-t8-epoch24.pth",
    ))
    if not checkpoint.is_file():
        pytest.skip("official BEVFormer V2 T8 checkpoint is not local")
    model = AutoE2E(
        backbone="res_net_50",
        embed_dim=256,
        is_pretrained=False,
        **reactive_model_kwargs(
            ReactiveTrainingStage.NUPLAN_FULL,
            num_views=6,
        ),
    )

    report = load_bevformer_v2_t8_checkpoint(model, checkpoint)

    assert report.source_sha256 == BEVFORMER_V2_T8_CHECKPOINT_SHA256
    assert report.loaded_tensor_count == 547
    assert report.source_revision == BEVFORMER_V2_SOURCE_REVISION
    assert report.temporal_fusion_frame_order == (
        BEVFORMER_V2_T8_TEMPORAL_FRAME_ORDER
    )
    assert report.temporal_fusion_source_url == (
        BEVFORMER_V2_T8_TEMPORAL_FUSION_SOURCE_URL
    )
    assert (
        model.Reactive_E2E.FeatureFusion.temporal_fusion
        is not None
    )
