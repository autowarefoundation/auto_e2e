import pytest
import torch
import sys
sys.path.append('..')

from model_components.auto_e2e import AutoE2E
from model_components.backbone import Backbone
from model_components.feature_fusion import FeatureFusion
from model_components.driving_policy import DrivingPolicy
from model_components.future_state import FutureState
from model_components.view_fusion import build_view_fusion, FUSION_REGISTRY
from model_components.view_fusion.cross_attention_fusion import CrossAttentionViewFusion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(params=["concat", "cross_attn"])
def model(request, device):
    m = AutoE2E(num_views=8, fusion_mode=request.param)
    return m.to(device)


def make_inputs(batch_size, num_views, device):
    visual = torch.randn(batch_size, num_views, 3, 224, 224, device=device)
    visual_history = torch.randn(batch_size, 896, device=device)
    egomotion = torch.randn(batch_size, 256, device=device)
    return visual, visual_history, egomotion


# ---------------------------------------------------------------------------
# 1. Output shape correctness
# ---------------------------------------------------------------------------

class TestOutputShapes:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_trajectory_shape(self, model, device, batch_size):
        visual, vh, ego = make_inputs(batch_size, 8, device)
        traj, _, _ = model(visual, vh, ego)
        assert traj.shape == (batch_size, 128)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_compressed_visual_shape(self, model, device, batch_size):
        visual, vh, ego = make_inputs(batch_size, 8, device)
        _, compressed, _ = model(visual, vh, ego)
        assert compressed.shape == (batch_size, 14)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_future_features_shape(self, model, device, batch_size):
        visual, vh, ego = make_inputs(batch_size, 8, device)
        _, _, future = model(visual, vh, ego)
        assert len(future) == 4
        for f in future:
            assert f.shape == (batch_size, 1440, 7, 7)


# ---------------------------------------------------------------------------
# 2. Batch independence — changing one sample must not affect others
# ---------------------------------------------------------------------------

class TestBatchIndependence:
    def test_samples_do_not_interfere(self, model, device):
        model.eval()
        torch.manual_seed(42)
        visual, vh, ego = make_inputs(2, 8, device)

        # Full batch forward
        traj_both, _, _ = model(visual, vh, ego)

        # Single sample forward (sample 0)
        traj_single, _, _ = model(
            visual[0:1], vh[0:1], ego[0:1]
        )

        # Sample 0's output must be identical regardless of what sample 1 contains
        assert torch.allclose(traj_both[0], traj_single[0], atol=1e-5), \
            "Batch samples are interfering with each other"

    def test_different_batch_neighbor_no_effect(self, model, device):
        model.eval()
        torch.manual_seed(42)
        visual, vh, ego = make_inputs(2, 8, device)

        traj_a, _, _ = model(visual, vh, ego)

        # Change sample 1 completely
        visual_modified = visual.clone()
        visual_modified[1] = torch.randn_like(visual_modified[1])
        vh_modified = vh.clone()
        vh_modified[1] = torch.randn_like(vh_modified[1])

        traj_b, _, _ = model(visual_modified, vh_modified, ego)

        # Sample 0 output must remain unchanged
        assert torch.allclose(traj_a[0], traj_b[0], atol=1e-5), \
            "Modifying another sample in the batch affected this sample's output"


# ---------------------------------------------------------------------------
# 3. View fusion effectiveness — different views must influence output
# ---------------------------------------------------------------------------

class TestViewFusion:
    def test_different_views_produce_different_output(self, model, device):
        model.eval()
        torch.manual_seed(42)
        visual, vh, ego = make_inputs(1, 8, device)

        traj_a, _, _ = model(visual, vh, ego)

        # Replace one camera view with zeros
        visual_zeroed = visual.clone()
        visual_zeroed[0, 3] = 0.0

        traj_b, _, _ = model(visual_zeroed, vh, ego)

        # Output should differ — proving that the zeroed view had influence
        assert not torch.allclose(traj_a, traj_b, atol=1e-5), \
            "Changing a camera view had no effect on output — fusion is broken"

    def test_all_views_contribute(self, model, device):
        """Each view should influence the output when zeroed out."""
        model.eval()
        torch.manual_seed(42)
        visual, vh, ego = make_inputs(1, 8, device)

        traj_base, _, _ = model(visual, vh, ego)

        for view_idx in range(8):
            visual_mod = visual.clone()
            visual_mod[0, view_idx] = 0.0
            traj_mod, _, _ = model(visual_mod, vh, ego)
            assert not torch.allclose(traj_base, traj_mod, atol=1e-5), \
                f"View {view_idx} has no influence on the output"


# ---------------------------------------------------------------------------
# 4. Gradient flow — all parameters receive gradients
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_backward_succeeds(self, model, device):
        visual, vh, ego = make_inputs(2, 8, device)
        traj, compressed, future = model(visual, vh, ego)

        loss = traj.sum() + compressed.sum() + sum(f.sum() for f in future)
        loss.backward()

    def test_all_parameters_have_gradients(self, model, device):
        visual, vh, ego = make_inputs(2, 8, device)
        traj, compressed, future = model(visual, vh, ego)

        loss = traj.sum() + compressed.sum() + sum(f.sum() for f in future)
        loss.backward()

        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, \
            f"Parameters with no gradient: {params_without_grad}"

    def test_no_vanishing_gradients(self, model, device):
        visual, vh, ego = make_inputs(2, 8, device)
        traj, compressed, future = model(visual, vh, ego)

        loss = traj.sum() + compressed.sum() + sum(f.sum() for f in future)
        loss.backward()

        zero_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().max() == 0:
                    zero_grad_params.append(name)

        assert len(zero_grad_params) == 0, \
            f"Parameters with all-zero gradients: {zero_grad_params}"


# ---------------------------------------------------------------------------
# 5. num_views flexibility — model works with different view counts
# ---------------------------------------------------------------------------

class TestNumViewsFlexibility:
    @pytest.mark.parametrize("num_views,fusion_mode", [
        (1, "concat"), (4, "concat"), (8, "concat"), (12, "concat"),
        (1, "cross_attn"), (4, "cross_attn"), (8, "cross_attn"), (12, "cross_attn"),
    ])
    def test_various_num_views(self, device, num_views, fusion_mode):
        model = AutoE2E(num_views=num_views, fusion_mode=fusion_mode).to(device)
        visual, vh, ego = make_inputs(2, num_views, device)
        traj, compressed, future = model(visual, vh, ego)

        assert traj.shape == (2, 128)
        assert compressed.shape == (2, 14)
        assert all(f.shape == (2, 1440, 7, 7) for f in future)


# ---------------------------------------------------------------------------
# 6. Numerical stability — no NaN or Inf
# ---------------------------------------------------------------------------

class TestNumericalStability:
    def test_no_nan_in_outputs(self, model, device):
        visual, vh, ego = make_inputs(2, 8, device)
        traj, compressed, future = model(visual, vh, ego)

        assert not torch.isnan(traj).any(), "NaN in trajectory output"
        assert not torch.isnan(compressed).any(), "NaN in compressed visual"
        for i, f in enumerate(future):
            assert not torch.isnan(f).any(), f"NaN in future feature {i}"

    def test_no_inf_in_outputs(self, model, device):
        visual, vh, ego = make_inputs(2, 8, device)
        traj, compressed, future = model(visual, vh, ego)

        assert not torch.isinf(traj).any(), "Inf in trajectory output"
        assert not torch.isinf(compressed).any(), "Inf in compressed visual"
        for i, f in enumerate(future):
            assert not torch.isinf(f).any(), f"Inf in future feature {i}"

    def test_large_input_values(self, model, device):
        """Model should not produce NaN/Inf even with large inputs."""
        visual = torch.randn(1, 8, 3, 224, 224, device=device) * 100
        vh = torch.randn(1, 896, device=device) * 100
        ego = torch.randn(1, 256, device=device) * 100
        traj, compressed, future = model(visual, vh, ego)

        assert not torch.isnan(traj).any(), "NaN with large inputs"
        assert not torch.isinf(traj).any(), "Inf with large inputs"


# ---------------------------------------------------------------------------
# Component-level tests
# ---------------------------------------------------------------------------

class TestFeatureFusionComponent:
    def test_output_shape(self, device):
        fusion = FeatureFusion(num_views=8, fusion_mode="concat").to(device)
        features = [
            torch.randn(16, 56, 56, 96, device=device),
            torch.randn(16, 28, 28, 192, device=device),
            torch.randn(16, 14, 14, 384, device=device),
            torch.randn(16, 7, 7, 768, device=device),
        ]
        out = fusion(features, B=2, V=8)
        assert out.shape == (2, 1440, 7, 7)

    def test_view_reduction_changes_output(self, device):
        """Verify that view_reduce is not identity (actually mixes views)."""
        fusion = FeatureFusion(num_views=8, fusion_mode="concat").to(device)
        fusion.eval()

        features_a = [
            torch.randn(8, 56, 56, 96, device=device),
            torch.randn(8, 28, 28, 192, device=device),
            torch.randn(8, 14, 14, 384, device=device),
            torch.randn(8, 7, 7, 768, device=device),
        ]
        out_a = fusion(features_a, B=1, V=8)

        features_b = [f.clone() for f in features_a]
        features_b[0][3] = torch.randn_like(features_b[0][3])
        out_b = fusion(features_b, B=1, V=8)

        assert not torch.allclose(out_a, out_b, atol=1e-5)


class TestDrivingPolicyComponent:
    def test_flatten_preserves_batch(self, device):
        """The critical fix: flatten must NOT collapse batch dimension."""
        policy = DrivingPolicy().to(device)
        fused = torch.randn(4, 1440, 7, 7, device=device)
        vh = torch.randn(4, 896, device=device)
        ego = torch.randn(4, 256, device=device)
        traj, compressed = policy(fused, vh, ego)

        assert traj.shape[0] == 4, "Batch dimension lost in DrivingPolicy"
        assert compressed.shape[0] == 4, "Batch dimension lost in compression"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestFusionRegistry:
    def test_all_modes_registered(self):
        assert "concat" in FUSION_REGISTRY
        assert "cross_attn" in FUSION_REGISTRY

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion_mode"):
            build_view_fusion("nonexistent", num_views=8)

    @pytest.mark.parametrize("fusion_mode", list(FUSION_REGISTRY.keys()))
    def test_all_modes_produce_correct_shape(self, device, fusion_mode):
        fusion = FeatureFusion(num_views=8, fusion_mode=fusion_mode).to(device)
        features = [
            torch.randn(16, 56, 56, 96, device=device),
            torch.randn(16, 28, 28, 192, device=device),
            torch.randn(16, 14, 14, 384, device=device),
            torch.randn(16, 7, 7, 768, device=device),
        ]
        out = fusion(features, B=2, V=8)
        assert out.shape == (2, 1440, 7, 7)


# ---------------------------------------------------------------------------
# Cross-Attention specific tests
# ---------------------------------------------------------------------------

class TestCrossAttentionFusion:
    def test_output_shape(self, device):
        fusion = CrossAttentionViewFusion(num_views=8, embed_dim=1440).to(device)
        x = torch.randn(16, 1440, 7, 7, device=device)
        out = fusion(x, B=2, V=8)
        assert out.shape == (2, 1440, 7, 7)

    def test_view_embeddings_are_learned(self, device):
        """View embeddings should receive gradients during training."""
        fusion = CrossAttentionViewFusion(num_views=8, embed_dim=1440).to(device)
        x = torch.randn(8, 1440, 7, 7, device=device)
        out = fusion(x, B=1, V=8)
        out.sum().backward()
        assert fusion.view_embed.grad is not None
        assert fusion.view_embed.grad.abs().max() > 0

    def test_attention_mixes_views(self, device):
        """Attention should produce different output than simple mean pooling."""
        fusion = CrossAttentionViewFusion(num_views=4, embed_dim=1440).to(device)
        fusion.eval()
        x = torch.randn(4, 1440, 7, 7, device=device)

        attn_out = fusion(x, B=1, V=4)
        mean_out = x.reshape(1, 4, 1440, 7, 7).mean(dim=1)

        assert not torch.allclose(attn_out, mean_out, atol=1e-3), \
            "Cross-attention output is identical to naive mean — attention has no effect"

    def test_different_view_orders_produce_different_output(self, device):
        """Attention with positional embeddings should be order-sensitive."""
        fusion = CrossAttentionViewFusion(num_views=4, embed_dim=1440).to(device)
        fusion.eval()

        x = torch.randn(4, 1440, 7, 7, device=device)
        out_original = fusion(x, B=1, V=4)

        x_permuted = x[[2, 0, 3, 1]]
        out_permuted = fusion(x_permuted, B=1, V=4)

        assert not torch.allclose(out_original, out_permuted, atol=1e-5), \
            "View position embeddings have no effect — output is order-invariant"
