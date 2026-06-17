import torch
import sys
sys.path.append('..')

from model_components.future_state import FutureState


class TestFutureStateComponent:
    def test_accepts_ego_hidden(self, device):
        future = FutureState(embed_dim=256, ego_hidden_dim=256).to(device)
        feats = torch.randn(2, 256, 8, 8, device=device)
        ego_hidden = torch.randn(2, 256, device=device)
        out = future(feats, ego_hidden)
        assert len(out) == 4
        for f in out:
            assert f.shape == (2, 256, 8, 8)

    def test_ego_hidden_influences_output(self, device):
        future = FutureState(embed_dim=256, ego_hidden_dim=256).to(device)
        future.eval()
        feats = torch.randn(1, 256, 8, 8, device=device)

        out_a = future(feats, torch.randn(1, 256, device=device))
        out_b = future(feats, torch.randn(1, 256, device=device))

        assert not torch.allclose(out_a[0], out_b[0], atol=1e-5), \
            "ego_hidden should influence future predictions"


class TestFutureStateChunkSplit:
    def test_four_outputs_are_distinct(self, device):
        """torch.chunk must split along channels, not return 4 views of the same data."""
        torch.manual_seed(0)
        future = FutureState(embed_dim=256, ego_hidden_dim=256).to(device)
        future.eval()
        feats = torch.randn(2, 256, 8, 8, device=device)
        ego_hidden = torch.randn(2, 256, device=device)

        out = future(feats, ego_hidden)
        assert len(out) == 4
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.allclose(out[i], out[j], atol=1e-5), \
                    f"FutureState outputs {i} and {j} are identical — chunk is broken"

    def test_ego_hidden_changes_all_four_outputs(self, device):
        """Different ego_hidden must shift every one of the 4 future predictions."""
        torch.manual_seed(0)
        future = FutureState(embed_dim=256, ego_hidden_dim=256).to(device)
        future.eval()
        feats = torch.randn(1, 256, 8, 8, device=device)

        ego_a = torch.randn(1, 256, device=device)
        ego_b = torch.randn(1, 256, device=device)

        out_a = future(feats, ego_a)
        out_b = future(feats, ego_b)

        for i in range(4):
            assert not torch.allclose(out_a[i], out_b[i], atol=1e-5), \
                f"Future output {i} did not change when ego_hidden changed"
