"""BEVFormer V2 T8 temporal and checkpoint contracts."""

from __future__ import annotations

import copy
import gc
import os
from pathlib import Path
import weakref

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
from model_components.reactive_e2e import (
    ReactiveE2E,
    StatefulCameraFPNCache,
)
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


def test_partial_history_is_rejected():
    reactive = ReactiveE2E.__new__(ReactiveE2E)
    torch.nn.Module.__init__(reactive)
    reactive.camera_architecture = "bevformer_v2_t8"
    reactive.FeatureFusion = torch.nn.Identity()
    current = torch.randn(1, 4, 2, 2)

    with pytest.raises(ValueError, match="zero or seven history"):
        reactive._fuse_temporal_camera_bevs(
            current,
            [torch.randn_like(current) for _ in range(3)],
        )


def _standalone_camera_fpn_cache(
    history_frames: int,
) -> tuple[StatefulCameraFPNCache, object]:
    owner_token = object()
    return (
        StatefulCameraFPNCache(
            history_frames=history_frames,
            owner_token=owner_token,
        ),
        owner_token,
    )


def test_stateful_camera_fpn_cache_requires_owner_token():
    with pytest.raises(TypeError, match="owner_token"):
        StatefulCameraFPNCache(history_frames=2)


def test_stateful_camera_fpn_cache_is_fifo_and_resettable():
    cache, owner_token = _standalone_camera_fpn_cache(2)

    def pyramid(value):
        return tuple(
            torch.full((2, 3, 2, 2), float(value + level))
            for level in range(4)
        )

    prepared = cache._prepare_frame(
        pyramid(3),
        stream_ids=["scene-a"],
        timestamps_us=[1_000_000],
        batch_size=1,
        num_views=2,
        front_companion_mode=False,
        owner_token=owner_token,
    )
    assert prepared.needs_prime
    cache._replace_history(
        [pyramid(1), pyramid(2)],
        prepared=prepared,
        batch_size=1,
        num_views=2,
    )
    cache._commit_frame(prepared, batch_size=1, num_views=2)

    assert len(cache) == 2
    assert cache.is_full
    assert cache.batch_size == 1
    assert cache.num_views == 2
    assert cache.stream_ids == ("scene-a",)
    assert cache.frames[0][0][0, 0, 0, 0].item() == 2.0
    assert cache.frames[1][0][0, 0, 0, 0].item() == 3.0
    assert cache.frame_timestamps_us == ((500_000,), (1_000_000,))
    assert cache.last_committed_timestamps_us == (1_000_000,)
    assert all(
        not feature.requires_grad
        for frame in cache.frames
        for feature in frame
    )

    next_prepared = cache._prepare_frame(
        pyramid(4),
        stream_ids=["scene-a"],
        timestamps_us=torch.tensor([1_500_000]),
        batch_size=1,
        num_views=2,
        front_companion_mode=False,
        owner_token=owner_token,
    )
    assert not next_prepared.needs_prime
    cache._commit_frame(next_prepared, batch_size=1, num_views=2)
    assert cache.frame_timestamps_us == ((1_000_000,), (1_500_000,))

    switched = cache._prepare_frame(
        pyramid(5),
        stream_ids=["scene-b"],
        timestamps_us=[2_000_000],
        batch_size=1,
        num_views=2,
        front_companion_mode=False,
        owner_token=owner_token,
    )
    assert switched.needs_prime
    assert len(cache) == 0
    assert cache.stream_ids == ("scene-b",)

    duplicate_scene_cache, duplicate_owner_token = (
        _standalone_camera_fpn_cache(2)
    )
    duplicate_scene_cache._prepare_frame(
        tuple(torch.zeros(4, 3, 2, 2) for _ in range(4)),
        stream_ids=["scene-b", "scene-b"],
        timestamps_us=[2_000_000, 2_500_000],
        batch_size=2,
        num_views=2,
        front_companion_mode=False,
        owner_token=duplicate_owner_token,
    )
    assert duplicate_scene_cache.stream_ids == ("scene-b", "scene-b")

    cache.reset()

    assert len(cache) == 0
    assert not cache.is_full
    assert cache.batch_size is None
    assert cache.num_views is None
    assert cache.stream_ids is None
    assert cache.last_committed_timestamps_us is None


def test_stateful_camera_fpn_cache_batched_lanes_advance_in_lockstep():
    cache, owner_token = _standalone_camera_fpn_cache(2)

    def pyramid(value):
        return tuple(
            torch.full((4, 3, 2, 2), float(value + level))
            for level in range(4)
        )

    prepared = cache._prepare_frame(
        pyramid(3),
        stream_ids=["scene-a", "scene-b"],
        timestamps_us=[1_000_000, 7_000_000],
        batch_size=2,
        num_views=2,
        front_companion_mode=False,
        owner_token=owner_token,
    )
    cache._replace_history(
        [pyramid(1), pyramid(2)],
        prepared=prepared,
        batch_size=2,
        num_views=2,
    )
    cache._commit_frame(prepared, batch_size=2, num_views=2)
    timestamps_before = cache.frame_timestamps_us

    with pytest.raises(ValueError, match="strictly newer"):
        cache._prepare_frame(
            pyramid(4),
            stream_ids=["scene-a", "scene-b"],
            timestamps_us=[1_500_000, 7_000_000],
            batch_size=2,
            num_views=2,
            front_companion_mode=False,
            owner_token=owner_token,
        )
    assert cache.frame_timestamps_us == timestamps_before

    off_grid = cache._prepare_frame(
        pyramid(5),
        stream_ids=["scene-a", "scene-b"],
        timestamps_us=[1_500_000, 7_500_001],
        batch_size=2,
        num_views=2,
        front_companion_mode=False,
        owner_token=owner_token,
    )
    assert off_grid.needs_prime
    assert len(cache) == 0


def test_stateful_camera_fpn_cache_reprimes_on_front_companion_mode_change():
    cache, owner_token = _standalone_camera_fpn_cache(2)

    def pyramid(value):
        return tuple(
            torch.full((2, 3, 2, 2), float(value + level))
            for level in range(4)
        )

    prepared = cache._prepare_frame(
        pyramid(3),
        stream_ids=["scene-a"],
        timestamps_us=[1_000_000],
        batch_size=1,
        num_views=2,
        front_companion_mode=True,
        owner_token=owner_token,
    )
    cache._replace_history(
        [pyramid(1), pyramid(2)],
        prepared=prepared,
        batch_size=1,
        num_views=2,
    )
    cache._commit_frame(prepared, batch_size=1, num_views=2)

    changed = cache._prepare_frame(
        pyramid(4),
        stream_ids=["scene-a"],
        timestamps_us=[1_500_000],
        batch_size=1,
        num_views=2,
        front_companion_mode=False,
        owner_token=owner_token,
    )

    assert changed.needs_prime
    assert len(cache) == 0
    assert cache.last_committed_timestamps_us == (1_000_000,)


def _stateful_test_reactive():
    class StubBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.shapes = []
            self.register_buffer("gain", torch.ones(()))
            self.inner = torch.nn.Module()
            self.inner.register_buffer("offset", torch.zeros(()))

        def forward(self, value):
            self.calls += 1
            self.shapes.append(tuple(value.shape))
            return [value * self.gain + self.inner.offset]

    class StubFusion(torch.nn.Module):
        temporal_fusion = None

        def __init__(self):
            super().__init__()
            self.projection_calls = []

        def build_bevformer_pyramid(self, features):
            base = features[0]
            return tuple(base + level for level in range(4))

        def fuse_bevformer_pyramid(
            self,
            pyramid,
            batch_size,
            num_views,
            **_kwargs,
        ):
            self.projection_calls.append(_kwargs.get("projection"))
            value = pyramid[0].reshape(
                batch_size,
                num_views,
                *pyramid[0].shape[1:],
            )
            return value.mean(dim=1)

        def forward(
            self,
            features,
            batch_size,
            num_views,
            **kwargs,
        ):
            pyramid = self.build_bevformer_pyramid(features)
            output = self.fuse_bevformer_pyramid(
                pyramid,
                batch_size,
                num_views,
                **kwargs,
            )
            front_features = kwargs.get("front_features")
            if front_features is not None:
                front_pyramid = self.build_bevformer_pyramid(
                    front_features
                )
                front_output = self.fuse_bevformer_pyramid(
                    front_pyramid,
                    batch_size,
                    1,
                    **kwargs,
                )
                output = output + torch.nn.functional.adaptive_avg_pool2d(
                    front_output,
                    output.shape[-2:],
                )
            if kwargs.get("return_base_pyramid"):
                return output, pyramid
            return output

        def fuse_temporal_bevs(self, frames):
            return torch.stack(frames).mean(dim=0)

    class StubNavigation(torch.nn.Module):
        def forward(
            self,
            map_context,
            route,
            *,
            return_route_contribution,
        ):
            assert return_route_contribution
            return map_context, route

    class AddFusion(torch.nn.Module):
        def forward(self, image_bev, navigation_bev):
            return image_bev + navigation_bev

    class StubTemporalMemory(torch.nn.Module):
        def forward(self, visual_history, egomotion_history):
            return visual_history, egomotion_history

    class StubPlanner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fail_next = False

        def forward(self, features, _visual, _ego, **_kwargs):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("planned test failure")
            return features.mean(dim=(-2, -1))

    reactive = ReactiveE2E.__new__(ReactiveE2E)
    torch.nn.Module.__init__(reactive)
    reactive.camera_architecture = "bevformer_v2_t8"
    reactive.camera_input_size = 4
    reactive.front_camera_index = 0
    reactive.front_camera_input_size = 8
    reactive._camera_bev_frozen = False
    reactive._adapt_temporal_running_stats = False
    reactive._initialize_camera_fpn_cache_owner()
    reactive.Backbone = StubBackbone()
    reactive.FeatureFusion = StubFusion()
    reactive._register_camera_fpn_cache_load_hooks()
    reactive.map_context_channels = 3
    reactive.route_channels = 1
    reactive.enable_route_conditioning = True
    reactive.NavigationEncoder = StubNavigation()
    reactive.MapBEVFusion = AddFusion()
    reactive.BEVSegmentationHead = None
    reactive.RouteReconstructionHead = None
    reactive.FusedFeaturePooling = None
    reactive.TemporalMemory = StubTemporalMemory()
    reactive.ReasoningHead = None
    reactive.TrajectoryPlanner = StubPlanner()
    reactive.eval()
    return reactive


def test_stateful_camera_fpn_cache_matches_stateless_and_reuses_backbone():
    reactive = _stateful_test_reactive()

    torch.manual_seed(19)
    history = torch.randn(1, 7, 2, 3, 4, 4)
    current = torch.randn(1, 2, 3, 4, 4)
    current_front = torch.randn(1, 3, 8, 8)
    current_front_fpn = torch.randn(1, 3, 4, 4)
    next_current = torch.randn(1, 2, 3, 4, 4)
    next_front = torch.randn(1, 3, 8, 8)
    next_front_fpn = torch.randn(1, 3, 4, 4)
    current_as_history = current.clone()
    current_as_history[:, 0] = current_front_fpn
    next_history = torch.cat(
        [history[:, 1:], current_as_history.unsqueeze(1)],
        dim=1,
    )
    map_context = torch.randn(1, 3, 4, 4)
    route = torch.randn(1, 1, 4, 4)
    common = {
        "map_context": map_context,
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": route,
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "mode": "infer",
    }
    stateful = {
        "camera_fpn_stream_ids": ["scene-a"],
        "camera_fpn_timestamps_us": [4_000_000],
    }

    with torch.inference_mode():
        expected_first = reactive(
            current,
            camera_history_tiles=history,
            front_camera_tile=current_front,
            **common,
        )
        reactive.Backbone.calls = 0
        reactive.Backbone.shapes.clear()
        cache = reactive.create_stateful_camera_fpn_cache()
        cached_first = reactive(
            current,
            camera_history_tiles=history,
            camera_fpn_cache=cache,
            front_camera_tile=current_front,
            front_camera_fpn_tile=current_front_fpn,
            front_camera_fpn_available=torch.tensor([True]),
            **stateful,
            **common,
        )

        assert torch.equal(cached_first, expected_first)
        assert reactive.Backbone.calls == 10
        assert reactive.Backbone.shapes[:3] == [
            (2, 3, 4, 4),
            (1, 3, 8, 8),
            (1, 3, 4, 4),
        ]
        assert len(cache) == 7

        expected_second = reactive(
            next_current,
            camera_history_tiles=next_history,
            front_camera_tile=next_front,
            **common,
        )
        reactive.Backbone.calls = 0
        reactive.Backbone.shapes.clear()
        cached_second = reactive(
            next_current,
            camera_history_tiles=next_history,
            camera_fpn_cache=cache,
            front_camera_tile=next_front,
            front_camera_fpn_tile=next_front_fpn,
            front_camera_fpn_available=torch.tensor([True]),
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_500_000],
            **common,
        )

    assert torch.equal(cached_second, expected_second)
    assert reactive.Backbone.calls == 3
    assert reactive.Backbone.shapes == [
        (2, 3, 4, 4),
        (1, 3, 8, 8),
        (1, 3, 4, 4),
    ]
    assert len(cache) == 7

    reactive.Backbone.calls = 0
    reactive.Backbone.shapes.clear()
    switched_history = torch.randn_like(history)
    with torch.inference_mode():
        reactive(
            next_current,
            camera_history_tiles=switched_history,
            camera_fpn_cache=cache,
            front_camera_tile=next_front,
            front_camera_fpn_tile=next_front_fpn,
            front_camera_fpn_available=torch.tensor([True]),
            camera_fpn_stream_ids=["scene-b"],
            camera_fpn_timestamps_us=[4_500_000],
            **common,
        )

    assert reactive.Backbone.calls == 10
    assert cache.stream_ids == ("scene-b",)
    assert len(cache) == 7


def test_stateful_camera_fpn_cache_reprimes_on_same_scene_frame_gap():
    reactive = _stateful_test_reactive()
    torch.manual_seed(23)
    history = torch.randn(1, 7, 2, 3, 4, 4)
    current = torch.randn(1, 2, 3, 4, 4)
    next_current = torch.randn(1, 2, 3, 4, 4)
    gap_current = torch.randn(1, 2, 3, 4, 4)
    gap_history = torch.cat([
        history[:, 2:],
        current.unsqueeze(1),
        next_current.unsqueeze(1),
    ], dim=1)
    common = {
        "map_context": torch.randn(1, 3, 4, 4),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": torch.randn(1, 1, 4, 4),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "mode": "infer",
    }
    cache = reactive.create_stateful_camera_fpn_cache()

    with torch.inference_mode():
        reactive(
            current,
            camera_history_tiles=history,
            camera_fpn_cache=cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_000_000],
            **common,
        )
        expected = reactive(
            gap_current,
            camera_history_tiles=gap_history,
            **common,
        )
        reactive.Backbone.calls = 0
        actual = reactive(
            gap_current,
            camera_history_tiles=gap_history,
            camera_fpn_cache=cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[5_000_000],
            **common,
        )

    assert torch.equal(actual, expected)
    assert reactive.Backbone.calls == 8
    assert cache.last_committed_timestamps_us == (5_000_000,)


def test_stateful_camera_fpn_cache_is_failure_safe_on_retry():
    reactive = _stateful_test_reactive()
    torch.manual_seed(29)
    history = torch.randn(1, 7, 2, 3, 4, 4)
    current = torch.randn(1, 2, 3, 4, 4)
    next_current = torch.randn(1, 2, 3, 4, 4)
    next_history = torch.cat(
        [history[:, 1:], current.unsqueeze(1)],
        dim=1,
    )
    common = {
        "map_context": torch.randn(1, 3, 4, 4),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": torch.randn(1, 1, 4, 4),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "mode": "infer",
    }
    cache = reactive.create_stateful_camera_fpn_cache()
    reactive.TrajectoryPlanner.fail_next = True

    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="planned test failure"):
            reactive(
                current,
                camera_history_tiles=history,
                camera_fpn_cache=cache,
                camera_fpn_stream_ids=["scene-a"],
                camera_fpn_timestamps_us=[4_000_000],
                **common,
            )
        assert cache.last_committed_timestamps_us is None
        assert cache.frame_timestamps_us[-1] == (3_500_000,)

        expected = reactive(
            next_current,
            camera_history_tiles=next_history,
            **common,
        )
        reactive.Backbone.calls = 0
        actual = reactive(
            next_current,
            camera_history_tiles=next_history,
            camera_fpn_cache=cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_500_000],
            **common,
        )

    assert torch.equal(actual, expected)
    assert reactive.Backbone.calls == 8
    assert cache.last_committed_timestamps_us == (4_500_000,)


def test_stateful_camera_fpn_cache_rejects_reuse_by_another_model():
    owner = _stateful_test_reactive()
    other = _stateful_test_reactive()
    history = torch.randn(1, 7, 2, 3, 4, 4)
    current = torch.randn(1, 2, 3, 4, 4)
    common = {
        "map_context": torch.randn(1, 3, 4, 4),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": torch.randn(1, 1, 4, 4),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "mode": "infer",
    }
    cache = owner.create_stateful_camera_fpn_cache()

    with torch.inference_mode():
        owner(
            current,
            camera_history_tiles=history,
            camera_fpn_cache=cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_000_000],
            **common,
        )
        timestamps_before = cache.frame_timestamps_us
        with pytest.raises(
            ValueError,
            match="different ReactiveE2E instance",
        ):
            other(
                current,
                camera_history_tiles=history,
                camera_fpn_cache=cache,
                camera_fpn_stream_ids=["scene-a"],
                camera_fpn_timestamps_us=[4_500_000],
                **common,
            )

    assert cache.frame_timestamps_us == timestamps_before
    assert cache.last_committed_timestamps_us == (4_000_000,)
    assert other.Backbone.calls == 0


def test_stateful_camera_fpn_cache_rejects_after_parent_state_reload():
    reactive = _stateful_test_reactive()
    source = _stateful_test_reactive()
    history = torch.randn(1, 7, 2, 3, 4, 4)
    current = torch.randn(1, 2, 3, 4, 4)
    common = {
        "map_context": torch.randn(1, 3, 4, 4),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": torch.randn(1, 1, 4, 4),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "mode": "infer",
    }
    cache = reactive.create_stateful_camera_fpn_cache()

    with torch.inference_mode():
        reactive(
            current,
            camera_history_tiles=history,
            camera_fpn_cache=cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_000_000],
            **common,
        )
        timestamps_before = cache.frame_timestamps_us
        owner_before = reactive._camera_fpn_cache_owner
        source.Backbone.gain.fill_(7.0)

        parent = torch.nn.Module()
        parent.add_module("reactive", reactive)
        source_parent = torch.nn.Module()
        source_parent.add_module("reactive", source)
        parent.load_state_dict(source_parent.state_dict())

        assert reactive._camera_fpn_cache_owner is not owner_before
        assert reactive.Backbone.gain.item() == 7.0
        reactive.Backbone.calls = 0
        with pytest.raises(
            ValueError,
            match="different ReactiveE2E instance",
        ):
            reactive(
                current,
                camera_history_tiles=history,
                camera_fpn_cache=cache,
                camera_fpn_stream_ids=["scene-a"],
                camera_fpn_timestamps_us=[4_500_000],
                **common,
            )

    assert cache.frame_timestamps_us == timestamps_before
    assert cache.last_committed_timestamps_us == (4_000_000,)
    assert reactive.Backbone.calls == 0


@pytest.mark.parametrize(
    "module_path",
    ("Backbone", "Backbone.inner", "FeatureFusion"),
)
def test_stateful_camera_fpn_cache_rejects_after_camera_submodule_reload(
    module_path,
):
    reactive = _stateful_test_reactive()
    cache = reactive.create_stateful_camera_fpn_cache()
    owner_before = reactive._camera_fpn_cache_owner
    module = reactive.get_submodule(module_path)

    module.load_state_dict(module.state_dict())

    assert reactive._camera_fpn_cache_owner is not owner_before
    with pytest.raises(ValueError, match="different ReactiveE2E instance"):
        cache._validate_owner(reactive._camera_fpn_cache_owner)


def test_camera_submodule_reload_hook_tracks_deepcopied_model():
    reactive = _stateful_test_reactive()
    copied = copy.deepcopy(reactive)
    original_owner = reactive._camera_fpn_cache_owner
    copied_owner = copied._camera_fpn_cache_owner

    copied.Backbone.load_state_dict(copied.Backbone.state_dict())

    assert copied._camera_fpn_cache_owner is not copied_owner
    assert reactive._camera_fpn_cache_owner is original_owner


def test_stateful_camera_fpn_cache_deepcopy_pair_preserves_owner_pairing():
    reactive = _stateful_test_reactive()
    cache = reactive.create_stateful_camera_fpn_cache()

    copied, copied_cache = copy.deepcopy((reactive, cache))

    cache._validate_owner(reactive._camera_fpn_cache_owner)
    copied_cache._validate_owner(copied._camera_fpn_cache_owner)
    with pytest.raises(ValueError, match="different ReactiveE2E instance"):
        cache._validate_owner(copied._camera_fpn_cache_owner)
    with pytest.raises(ValueError, match="different ReactiveE2E instance"):
        copied_cache._validate_owner(reactive._camera_fpn_cache_owner)

    copied.Backbone.load_state_dict(copied.Backbone.state_dict())

    cache._validate_owner(reactive._camera_fpn_cache_owner)
    with pytest.raises(ValueError, match="different ReactiveE2E instance"):
        copied_cache._validate_owner(copied._camera_fpn_cache_owner)


def test_camera_submodule_deepcopy_does_not_retain_parent_model():
    reactive = _stateful_test_reactive()
    copied_backbone = copy.deepcopy(reactive.Backbone)
    reactive_ref = weakref.ref(reactive)

    del reactive
    gc.collect()

    assert reactive_ref() is None
    copied_backbone.load_state_dict(copied_backbone.state_dict())


def test_stateful_camera_fpn_cache_rejects_duplicate_frame_and_missing_prime():
    reactive = _stateful_test_reactive()
    history = torch.randn(1, 7, 2, 3, 4, 4)
    current = torch.randn(1, 2, 3, 4, 4)
    common = {
        "map_context": torch.randn(1, 3, 4, 4),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": torch.randn(1, 1, 4, 4),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "mode": "infer",
    }
    cache = reactive.create_stateful_camera_fpn_cache()

    with torch.inference_mode():
        reactive(
            current,
            camera_history_tiles=history,
            camera_fpn_cache=cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_000_000],
            **common,
        )
        timestamps_before = cache.frame_timestamps_us
        with pytest.raises(ValueError, match="strictly newer"):
            reactive(
                current,
                camera_history_tiles=history,
                camera_fpn_cache=cache,
                camera_fpn_stream_ids=["scene-a"],
                camera_fpn_timestamps_us=[4_000_000],
                **common,
            )

    assert cache.frame_timestamps_us == timestamps_before

    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="requires raw history",
    ):
        reactive(
            current.to(torch.float64),
            camera_fpn_cache=cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_500_000],
            **common,
        )
    assert len(cache) == 0

    empty_cache = reactive.create_stateful_camera_fpn_cache()
    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="requires raw history",
    ):
        reactive(
            current,
            camera_fpn_cache=empty_cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_000_000],
            **common,
        )

    mismatched_views_cache = reactive.create_stateful_camera_fpn_cache()
    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="history view count differs",
    ):
        reactive(
            current,
            camera_history_tiles=torch.randn(1, 7, 3, 3, 4, 4),
            camera_fpn_cache=mismatched_views_cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_000_000],
            **common,
        )

    native_front_cache = reactive.create_stateful_camera_fpn_cache()
    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="Front tiles together",
    ):
        reactive(
            current,
            camera_history_tiles=history,
            camera_fpn_cache=native_front_cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_000_000],
            front_camera_tile=torch.randn(1, 3, 8, 8),
            **common,
        )

    missing_native_cache = reactive.create_stateful_camera_fpn_cache()
    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="Front tiles together",
    ):
        reactive(
            current,
            camera_history_tiles=history,
            camera_fpn_cache=missing_native_cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_000_000],
            front_camera_fpn_tile=torch.randn(1, 3, 4, 4),
            front_camera_fpn_available=torch.tensor([True]),
            **common,
        )

    unavailable_cache = reactive.create_stateful_camera_fpn_cache()
    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="exact Front companion",
    ):
        reactive(
            current,
            camera_history_tiles=history,
            camera_fpn_cache=unavailable_cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_000_000],
            front_camera_tile=torch.randn(1, 3, 8, 8),
            front_camera_fpn_tile=torch.randn(1, 3, 4, 4),
            front_camera_fpn_available=torch.tensor([False]),
            **common,
        )

    mismatched_dtype_cache = reactive.create_stateful_camera_fpn_cache()
    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="dtype and device",
    ):
        reactive(
            current,
            camera_history_tiles=history,
            camera_fpn_cache=mismatched_dtype_cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_000_000],
            front_camera_tile=torch.randn(1, 3, 8, 8),
            front_camera_fpn_tile=torch.randn(
                1,
                3,
                4,
                4,
                dtype=torch.float64,
            ),
            front_camera_fpn_available=torch.tensor([True]),
            **common,
        )

    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="arguments require a cache",
    ):
        reactive(
            current,
            camera_history_tiles=history,
            front_camera_fpn_tile=torch.randn(1, 3, 4, 4),
            front_camera_fpn_available=torch.tensor([True]),
            **common,
        )


@pytest.mark.parametrize(
    "cache_metadata",
    [
        {"camera_fpn_stream_ids": ["scene-a"]},
        {"camera_fpn_timestamps_us": [4_000_000]},
    ],
)
def test_stateful_camera_fpn_cache_metadata_requires_cache(cache_metadata):
    reactive = _stateful_test_reactive()
    common = {
        "map_context": torch.randn(1, 3, 4, 4),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": torch.randn(1, 1, 4, 4),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "mode": "infer",
    }

    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="arguments require a cache",
    ):
        reactive(
            torch.randn(1, 2, 3, 4, 4),
            camera_history_tiles=torch.randn(1, 7, 2, 3, 4, 4),
            **cache_metadata,
            **common,
        )


def test_stateful_camera_fpn_cache_rejects_backward_frames_after_gap():
    reactive = _stateful_test_reactive()
    history = torch.randn(1, 7, 2, 3, 4, 4)
    current = torch.randn(1, 2, 3, 4, 4)
    gap_current = torch.randn(1, 2, 3, 4, 4)
    common = {
        "map_context": torch.randn(1, 3, 4, 4),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": torch.randn(1, 1, 4, 4),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "mode": "infer",
    }
    cache = reactive.create_stateful_camera_fpn_cache()

    with torch.inference_mode():
        reactive(
            current,
            camera_history_tiles=history,
            camera_fpn_cache=cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_000_000],
            **common,
        )
        reactive(
            gap_current,
            camera_history_tiles=torch.randn_like(history),
            camera_fpn_cache=cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[9_000_000],
            **common,
        )
        with pytest.raises(ValueError, match="strictly newer"):
            reactive(
                current,
                camera_history_tiles=history,
                camera_fpn_cache=cache,
                camera_fpn_stream_ids=["scene-a"],
                camera_fpn_timestamps_us=[4_500_000],
                **common,
            )

        cache.reset()
        reactive(
            current,
            camera_history_tiles=history,
            camera_fpn_cache=cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_500_000],
            **common,
        )

    assert cache.last_committed_timestamps_us == (4_500_000,)


def test_stateful_camera_fpn_cache_preserves_history_projection_order():
    reactive = _stateful_test_reactive()
    history = torch.randn(1, 7, 2, 3, 4, 4)
    current = torch.randn(1, 2, 3, 4, 4)
    history_projections = tuple(object() for _ in range(7))
    current_projection = object()
    cache = reactive.create_stateful_camera_fpn_cache()

    with torch.inference_mode():
        reactive(
            current,
            torch.randn(1, 3, 4, 4),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            route_mask=torch.randn(1, 1, 4, 4),
            map_valid=torch.ones(1, dtype=torch.bool),
            route_valid=torch.ones(1, dtype=torch.bool),
            projection=current_projection,
            camera_history_tiles=history,
            history_projections=history_projections,
            camera_fpn_cache=cache,
            camera_fpn_stream_ids=["scene-a"],
            camera_fpn_timestamps_us=[4_000_000],
            mode="infer",
        )

    observed_history = [
        value
        for value in reactive.FeatureFusion.projection_calls
        if value in history_projections
    ]
    assert observed_history == list(history_projections)


def test_stateful_camera_fpn_cache_requires_eval_and_disabled_gradients():
    reactive = _stateful_test_reactive()
    current = torch.randn(1, 2, 3, 4, 4)
    common = {
        "map_context": torch.randn(1, 3, 4, 4),
        "visual_history": torch.zeros(1, 1),
        "egomotion_history": torch.zeros(1, 1),
        "route_mask": torch.randn(1, 1, 4, 4),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "camera_fpn_cache": reactive.create_stateful_camera_fpn_cache(),
        "camera_fpn_stream_ids": ["scene-a"],
        "camera_fpn_timestamps_us": [4_000_000],
        "mode": "infer",
    }

    with pytest.raises(ValueError, match="gradients disabled"):
        reactive(current, **common)

    reactive.Backbone.train()
    with torch.no_grad(), pytest.raises(ValueError, match="eval-mode"):
        reactive(current, **common)


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
