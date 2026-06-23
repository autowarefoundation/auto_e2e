"""Tests for the OSM vector map encoder, token cross-attention fusion, the
ragged collate, and the AutoE2E ``osm_vector`` integration.

The heavy SentenceTransformer tag encoder is replaced by a tiny ``_FakeNLP``
module (dependency-injected via ``nlp_model=``), so these tests run fully
offline with no checkpoint, no network, and no sentence-transformers install.
Data-pipeline tests that need shapely/numpy are guarded with importorskip.
"""

import sys

import pytest
import torch
import torch.nn as nn

sys.path.append('..')

from model_components.map_encoder import (
    MAP_ENCODER_REGISTRY,
    MAP_FUSION_REGISTRY,
    build_map_bev_fusion,
    build_map_encoder,
)
from model_components.map_encoder.osm_vector import OSMVectorMapEncoder
from model_components.map_encoder.map_bev_fusion.osm_cross_attn_fusion import OSMCrossAttnFusion
from data_parsing.osm_sd_map.collate import collate_osm_batch


_NLP_EMBED_DIM = 8
_POS_NUM_FEATS = 4
_ORF_DIM = 4
_PTS_DIM = 3
_EMBED_DIM = 32
_INPUT_DIM = _PTS_DIM * _POS_NUM_FEATS + _NLP_EMBED_DIM + 2 * _ORF_DIM  # 28


class _FakeNLP(nn.Module):
    """Stand-in for the SentenceTransformer: input_ids -> mean token embedding."""

    def __init__(self, vocab=64, embed_dim=_NLP_EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.vocab = vocab

    def forward(self, features):
        ids = features["input_ids"].long().clamp(0, self.vocab - 1)  # (N, L)
        emb = self.embedding(ids).mean(dim=1)  # (N, embed_dim)
        return {"sentence_embedding": emb}


def _make_encoder(device, **overrides):
    kwargs = dict(
        embed_dim=_EMBED_DIM, input_dim=_INPUT_DIM, hidden_dim=_EMBED_DIM,
        orf_dim=_ORF_DIM, nlp_model=_FakeNLP(), nlp_embed_dim=_NLP_EMBED_DIM,
        nheads=4, nlayers=2, pos_num_feats=_POS_NUM_FEATS, pts_dim=_PTS_DIM,
    )
    kwargs.update(overrides)
    return OSMVectorMapEncoder(**kwargs).to(device)


def _rand_tokens(n, lo=2, hi=5):
    """List of n 1D token tensors of random length."""
    return [torch.randint(1, 60, (int(torch.randint(lo, hi, (1,))),), dtype=torch.long)
            for _ in range(n)]


def _make_sample(num_nodes=3, num_ways=2, num_way_pts=10, with_relation=False):
    """Build one tokenized osm_map_data sample dict."""
    s = {
        "osm_map_nodes_pts": torch.randn(num_nodes, _PTS_DIM),
        "osm_map_ways_pts": torch.randn(num_ways, num_way_pts, _PTS_DIM),
    }
    for name, n in (("nodes", num_nodes), ("ways", num_ways)):
        toks = _rand_tokens(n)
        s[f"osm_map_{name}_tags_input_ids"] = toks
        s[f"osm_map_{name}_tags_token_type_ids"] = [torch.zeros_like(t) for t in toks]
        s[f"osm_map_{name}_tags_attention_mask"] = [torch.ones_like(t) for t in toks]

    if with_relation:
        rel_toks = _rand_tokens(1)
        s["osm_map_relations_tags_input_ids"] = rel_toks
        s["osm_map_relations_tags_token_type_ids"] = [torch.zeros_like(t) for t in rel_toks]
        s["osm_map_relations_tags_attention_mask"] = [torch.ones_like(t) for t in rel_toks]
        # one node member (-> node 0) and one way member (-> way 0)
        nm, wm = _rand_tokens(1), _rand_tokens(1)
        s["osm_map_relations_node_member_tags_input_ids"] = [nm]
        s["osm_map_relations_node_member_tags_token_type_ids"] = [[torch.zeros_like(t) for t in nm]]
        s["osm_map_relations_node_member_tags_attention_mask"] = [[torch.ones_like(t) for t in nm]]
        s["osm_map_relations_way_member_tags_input_ids"] = [wm]
        s["osm_map_relations_way_member_tags_token_type_ids"] = [[torch.zeros_like(t) for t in wm]]
        s["osm_map_relations_way_member_tags_attention_mask"] = [[torch.ones_like(t) for t in wm]]
        s["osm_map_relations_relation_member_tags_input_ids"] = [[]]
        s["osm_map_relations_relation_member_tags_token_type_ids"] = [[]]
        s["osm_map_relations_relation_member_tags_attention_mask"] = [[]]
        s["osm_map_relations_node_member_indices"] = [torch.tensor([0], dtype=torch.long)]
        s["osm_map_relations_way_member_indices"] = [torch.tensor([0], dtype=torch.long)]
        s["osm_map_relations_relation_member_indices"] = [torch.tensor([], dtype=torch.long)]
    else:
        for key in ("input_ids", "token_type_ids", "attention_mask"):
            for name in ("", "node_member_", "way_member_", "relation_member_"):
                s[f"osm_map_relations_{name}tags_{key}" if name else f"osm_map_relations_tags_{key}"] = []
        for name in ("node", "way", "relation"):
            s[f"osm_map_relations_{name}_member_indices"] = []
    return s


def _make_batch(samples, device):
    batch = collate_osm_batch(samples)
    # Move geometry tensors to device (tag tokens stay on CPU then are used by
    # the (CPU/GPU) nlp model on the encoder's device via get_nlp_model_input).
    for b in range(len(samples)):
        batch["osm_map_nodes_pts"][b] = batch["osm_map_nodes_pts"][b].to(device)
        batch["osm_map_ways_pts"][b] = batch["osm_map_ways_pts"][b].to(device)
    return batch


class TestOSMVectorEncoder:
    def test_output_shape_and_mask(self, device):
        enc = _make_encoder(device)
        batch = _make_batch([_make_sample(3, 2), _make_sample(1, 4)], device)
        tokens, mask = enc(batch)
        assert tokens.dim() == 3 and tokens.shape[0] == 2 and tokens.shape[2] == _EMBED_DIM
        assert mask.shape == tokens.shape[:2]
        # sample 0 valid length = 3 nodes + 2 ways * 10 pts = 23
        assert (~mask[0]).sum().item() == 3 + 2 * 10
        assert (~mask[1]).sum().item() == 1 + 4 * 10

    def test_relations_increase_token_count(self, device):
        enc = _make_encoder(device)
        batch = _make_batch([_make_sample(2, 1, with_relation=True)], device)
        tokens, mask = enc(batch)
        # 2 nodes + 1 way*10 + 1 rel + 1 node member + 1 way member = 15
        assert (~mask[0]).sum().item() == 2 + 10 + 1 + 1 + 1

    def test_gradient_flows(self, device):
        enc = _make_encoder(device)
        batch = _make_batch([_make_sample(2, 2)], device)
        tokens, _ = enc(batch)
        tokens.sum().backward()
        grads = [p.grad for n, p in enc.named_parameters()
                 if "map_embedding" in n and p.requires_grad]
        assert any(g is not None and g.abs().max() > 0 for g in grads)

    def test_input_dim_mismatch_raises(self, device):
        with pytest.raises(ValueError, match="inconsistent"):
            OSMVectorMapEncoder(embed_dim=_EMBED_DIM, input_dim=999,
                                orf_dim=_ORF_DIM, nlp_model=_FakeNLP(),
                                nlp_embed_dim=_NLP_EMBED_DIM,
                                pos_num_feats=_POS_NUM_FEATS, pts_dim=_PTS_DIM)

    def test_requires_nlp(self):
        with pytest.raises(ValueError, match="nlp_model_path"):
            OSMVectorMapEncoder(embed_dim=_EMBED_DIM, input_dim=_INPUT_DIM,
                                orf_dim=_ORF_DIM, nlp_embed_dim=_NLP_EMBED_DIM,
                                pos_num_feats=_POS_NUM_FEATS, pts_dim=_PTS_DIM)


class TestOSMCrossAttnFusion:
    def test_output_shape(self, device):
        fusion = OSMCrossAttnFusion(embed_dim=32, num_heads=4).to(device)
        image_bev = torch.randn(2, 32, 6, 5, device=device)
        tokens = torch.randn(2, 7, 32, device=device)
        mask = torch.zeros(2, 7, dtype=torch.bool, device=device)
        out = fusion(image_bev, tokens, mask)
        assert out.shape == image_bev.shape

    def test_zero_gate_returns_image_bev(self, device):
        fusion = OSMCrossAttnFusion(embed_dim=32, num_heads=4).to(device)
        assert torch.all(fusion.alpha == 0)
        image_bev = torch.randn(1, 32, 4, 4, device=device)
        tokens = torch.randn(1, 5, 32, device=device)
        out = fusion(image_bev, tokens, torch.zeros(1, 5, dtype=torch.bool, device=device))
        assert torch.allclose(out, image_bev, atol=1e-5)

    def test_nonzero_gate_map_influences_output(self, device):
        fusion = OSMCrossAttnFusion(embed_dim=32, num_heads=4).to(device).eval()
        with torch.no_grad():
            fusion.alpha.fill_(1.0)
        image_bev = torch.randn(1, 32, 4, 4, device=device)
        mask = torch.zeros(1, 5, dtype=torch.bool, device=device)
        out_a = fusion(image_bev, torch.randn(1, 5, 32, device=device), mask)
        out_b = fusion(image_bev, torch.randn(1, 5, 32, device=device), mask)
        assert not torch.allclose(out_a, out_b, atol=1e-5)

    def test_all_masked_tokens_no_nan(self, device):
        """A sample with every token padded must not NaN (null token saves it)."""
        fusion = OSMCrossAttnFusion(embed_dim=32, num_heads=4).to(device)
        with torch.no_grad():
            fusion.alpha.fill_(1.0)
        image_bev = torch.randn(1, 32, 4, 4, device=device)
        tokens = torch.randn(1, 5, 32, device=device)
        mask = torch.ones(1, 5, dtype=torch.bool, device=device)  # all padded
        out = fusion(image_bev, tokens, mask)
        assert torch.isfinite(out).all()

    def test_padding_mask_changes_result(self, device):
        fusion = OSMCrossAttnFusion(embed_dim=32, num_heads=4).to(device).eval()
        with torch.no_grad():
            fusion.alpha.fill_(1.0)
        image_bev = torch.randn(1, 32, 4, 4, device=device)
        tokens = torch.randn(1, 6, 32, device=device)
        m_none = torch.zeros(1, 6, dtype=torch.bool, device=device)
        m_half = m_none.clone()
        m_half[0, 3:] = True
        assert not torch.allclose(fusion(image_bev, tokens, m_none),
                                  fusion(image_bev, tokens, m_half), atol=1e-5)

    def test_embed_dim_divisible_by_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            OSMCrossAttnFusion(embed_dim=30, num_heads=4)


class TestRegistries:
    def test_osm_vector_registered(self):
        assert "osm_vector" in MAP_ENCODER_REGISTRY

    def test_osm_cross_attn_registered(self):
        assert "osm_cross_attn" in MAP_FUSION_REGISTRY

    def test_build_osm_cross_attn(self, device):
        fusion = build_map_bev_fusion("osm_cross_attn", embed_dim=32).to(device)
        assert isinstance(fusion, OSMCrossAttnFusion)

    def test_build_osm_vector_encoder(self, device):
        enc = build_map_encoder("osm_vector", embed_dim=_EMBED_DIM, input_dim=_INPUT_DIM,
                                orf_dim=_ORF_DIM, nlp_model=_FakeNLP(),
                                nlp_embed_dim=_NLP_EMBED_DIM, pos_num_feats=_POS_NUM_FEATS,
                                pts_dim=_PTS_DIM)
        assert isinstance(enc, OSMVectorMapEncoder)


class TestCollate:
    def test_groups_osm_keys_and_stacks_rest(self):
        a = {"img": torch.zeros(3), **_make_sample(2, 1)}
        b = {"img": torch.ones(3), **_make_sample(1, 2)}
        out = collate_osm_batch([a, b])
        assert out["img"].shape == (2, 3)  # stacked
        assert isinstance(out["osm_map_nodes_pts"], list) and len(out["osm_map_nodes_pts"]) == 2
        assert out["osm_map_nodes_pts"][0].shape[0] == 2


class TestAutoE2EOSMIntegration:
    def _make_model(self, build_mock_model, device):
        return build_mock_model(
            num_views=7, fusion_mode="bev", device=device,
            map_type="osm_vector",
            map_encoder_kwargs=dict(
                input_dim=_INPUT_DIM, orf_dim=_ORF_DIM, nlp_model=_FakeNLP(),
                nlp_embed_dim=_NLP_EMBED_DIM, nheads=4, nlayers=2,
                pos_num_feats=_POS_NUM_FEATS, pts_dim=_PTS_DIM,
            ),
            map_fusion_mode="osm_cross_attn",
        )

    def test_forward_infer(self, build_mock_model, device):
        model = self._make_model(build_mock_model, device).eval()
        visual = torch.randn(2, 7, 3, 256, 256, device=device)
        osm = _make_batch([_make_sample(3, 2, with_relation=True), _make_sample(2, 1)], device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        traj, ego_hidden, _ = model(visual, osm, vis_hist, ego, mode="infer")
        assert traj.shape == (2, 128)
        assert torch.isfinite(traj).all()

    def test_train_grads_reach_map_encoder(self, build_mock_model, device):
        model = self._make_model(build_mock_model, device).train()
        visual = torch.randn(2, 7, 3, 256, 256, device=device)
        osm = _make_batch([_make_sample(3, 2), _make_sample(2, 2)], device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        target = torch.randn(2, 128, device=device)
        # Open the fusion gate so map gradients are non-zero.
        with torch.no_grad():
            model.MapBEVFusion.alpha.fill_(0.5)
        loss, ego_hidden, future = model(visual, osm, vis_hist, ego, mode="train",
                                         trajectory_target=target)
        (loss + ego_hidden.sum() + sum(f.sum() for f in future)).backward()
        assert model.MapBEVFusion.alpha.grad is not None
        assert model.MapBEVFusion.alpha.grad.abs().max() > 0
        # gradient reaches the OSM encoder's learned projection too
        assert model.MapEncoder.map_embedding.weight.grad is not None
        assert model.MapEncoder.map_embedding.weight.grad.abs().max() > 0
