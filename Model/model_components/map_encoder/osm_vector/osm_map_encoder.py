"""Point-level SD-map (OpenStreetMap) vector encoder.

Ported from SDTagNet (`OSMMapEncoderPointLevel`, NeurIPS 2025) and stripped of
its mmdetection3d / mmcv dependencies and ablation-only code paths. Only the
canonical configuration is kept:

  * point-level tokens (every way point and node is its own token),
  * NLP tag embeddings from a SentenceTransformer (text annotations),
  * orthogonal random feature (ORF) graph identifiers with a fixed member
    ordering (``fixed_orf_order=True``, ``use_orf_graph_ident=True``),
  * a continuous sine positional encoding of the point geometry.

Dropped from the original: the SMERF one-hot-class baseline, the P-MapNet BEV
mode, ``render_bev_feats`` / ``draw_*`` rasterisation, ``use_queries_for_bev``,
learned positional encoding, the non-ORF relation expansion branch, and the
way-level ``OSMMapEncoder`` variant.

The encoder consumes the ragged ``osm_map_data`` dict produced by
``data_parsing.osm_sd_map`` (see that package for the exact contract) and
returns a padded token sequence plus a key-padding mask:

    tokens, key_padding_mask = encoder(osm_map_data)
    # tokens:           (B, N_max, embed_dim)
    # key_padding_mask: (B, N_max)  True where padded (no real element)

These feed the token cross-attention map fusion (``osm_cross_attn``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Orthogonal random feature (ORF) graph identifiers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, _ = torch.linalg.qr(unstructured_block, mode="reduced")
    return q.t()


@torch.no_grad()
def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, device=None):
    """Create a 2D Gaussian orthogonal matrix of shape (nb_rows, nb_columns)."""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []
    for _ in range(nb_full_blocks):
        block_list.append(_orthogonal_matrix_chunk(nb_columns, device=device))

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = _orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)
    normalizer = final_matrix.norm(p=2, dim=1, keepdim=True)
    normalizer[normalizer == 0] = 1e-5
    return final_matrix / normalizer


# ---------------------------------------------------------------------------
# Continuous sine positional encoding (geometry)
# ---------------------------------------------------------------------------

class SineContinuousPositionalEncoding(nn.Module):
    """Sine positional encoding of continuous coordinates.

    Maps ``(B, N, D)`` point coordinates to ``(B, N, D * num_feats)``. With
    ``normalize=True`` the input is first shifted by ``offset`` and scaled by
    ``range`` (per coordinate) into ``[0, scale]``.
    """

    def __init__(self, num_feats, temp=10000, normalize=False, range=None,
                 offset=0.0, scale=2 * torch.pi):
        super().__init__()
        self.num_feats = num_feats
        self.temp = temp
        self.normalize = normalize
        self.register_buffer(
            "range",
            torch.tensor(range, dtype=torch.float32) if range is not None else None,
            persistent=False,
        )
        self.register_buffer(
            "offset",
            torch.tensor(offset, dtype=torch.float32) if offset is not None else None,
            persistent=False,
        )
        self.scale = scale

    def forward(self, x):
        B, N, D = x.shape
        if self.normalize:
            x = (x - self.offset.to(x.device)) / self.range.to(x.device) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temp ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x[..., None] / dim_t  # [B, N, D, num_feats]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=3
        ).view(B, N, D * self.num_feats)
        return pos_x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class OSMVectorMapEncoder(nn.Module):
    """Encode OSM vector elements into a transformer token sequence.

    Args:
        embed_dim: Output token dimension (and transformer width).
        input_dim: Dimension of a raw token before ``map_embedding``. Must
            equal ``pts_dim * pos_num_feats + nlp_embed_dim + 2 * orf_dim``.
        hidden_dim: Transformer feedforward dimension.
        orf_dim: ORF graph-identifier dimension (per element; tiled x2 per
            token so it contributes ``2 * orf_dim`` channels).
        nlp_model_path: Local path to the SentenceTransformer tag encoder.
            Required. Use ``data_parsing.osm_sd_map.download_nlp_weights`` to
            fetch it from HuggingFace if you don't have it locally.
        nlp_embed_dim: Output dimension of the SentenceTransformer.
        nheads / nlayers: Transformer encoder depth/width.
        nlp_pad_token / nlp_max_tokens: tag tokenizer padding value / cap.
        pos_num_feats / pos_temp / pc_range: configure the geometry positional
            encoding. ``pc_range`` is ``(x_min, y_min, z_min, x_max, y_max,
            z_max)`` in ego metres; coordinates are normalised into it.
        pts_dim: Coordinate dimension (3 = xyz, 2 = xy). Must match the data.
    """

    def __init__(self,
                 embed_dim=256,
                 input_dim=320,
                 hidden_dim=256,
                 orf_dim=64,
                 nlp_model_path=None,
                 nlp_model=None,
                 nlp_embed_dim=144,
                 nheads=4,
                 nlayers=6,
                 nlp_pad_token=0,
                 nlp_max_tokens=256,
                 pos_num_feats=16,
                 pos_temp=1000,
                 pc_range=(-60.0, -30.0, -5.0, 60.0, 30.0, 3.0),
                 pts_dim=3):
        super().__init__()

        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.orf_dim = orf_dim
        self.nlp_embed_dim = nlp_embed_dim
        self.nlp_pad_token = nlp_pad_token
        self.nlp_max_tokens = nlp_max_tokens
        self.pts_dim = pts_dim

        expected_input = pts_dim * pos_num_feats + nlp_embed_dim + 2 * orf_dim
        if input_dim != expected_input:
            raise ValueError(
                f"input_dim={input_dim} is inconsistent with "
                f"pts_dim*pos_num_feats + nlp_embed_dim + 2*orf_dim = "
                f"{pts_dim}*{pos_num_feats} + {nlp_embed_dim} + 2*{orf_dim} = "
                f"{expected_input}."
            )

        self.map_embedding = nn.Linear(input_dim, embed_dim)

        # Geometry positional encoding, normalised into pc_range.
        self.pos_encoder = SineContinuousPositionalEncoding(
            num_feats=pos_num_feats,
            temp=pos_temp,
            normalize=True,
            range=[pc_range[3] - pc_range[0],
                   pc_range[4] - pc_range[1],
                   pc_range[5] - pc_range[2]][:pts_dim],
            offset=[pc_range[0], pc_range[1], pc_range[2]][:pts_dim],
        )

        # The NLP tag encoder embeds OSM tag strings. Either inject a pre-built
        # module (``nlp_model``, e.g. for tests) or give a local checkpoint path
        # (``nlp_model_path``); the SentenceTransformer import is lazy so the
        # heavy NLP stack is only required when actually loading from a path.
        if nlp_model is not None:
            self.nlp_model = nlp_model
        elif nlp_model_path is not None:
            from sentence_transformers import SentenceTransformer
            # Force CPU at construction: SentenceTransformer otherwise auto-moves
            # itself to CUDA, leaving this module split across devices until
            # `.to(...)` is called. Constructing on CPU keeps it single-device
            # (standard nn.Module behaviour) so `encoder.to(device)` moves
            # everything — including the NLP model — together.
            self.nlp_model = SentenceTransformer(nlp_model_path, device="cpu")
        else:
            raise ValueError(
                "Provide nlp_model_path (download via "
                "data_parsing.osm_sd_map.download_nlp_weights) or an explicit "
                "nlp_model module."
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nheads, dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _num_nodes(osm_map_data, b_idx):
        nodes_pts = osm_map_data["osm_map_nodes_pts"][b_idx]
        if not nodes_pts.numel():
            return 0
        if nodes_pts.dim() == 1:
            osm_map_data["osm_map_nodes_pts"][b_idx] = nodes_pts.unsqueeze(0)
            return 1
        return len(nodes_pts)

    def _counts(self, osm_map_data, b_idx):
        num_nodes = self._num_nodes(osm_map_data, b_idx)
        num_ways, num_way_pts, pts_dim = osm_map_data["osm_map_ways_pts"][b_idx].shape
        num_rels = len(osm_map_data["osm_map_relations_tags_input_ids"][b_idx])
        num_rel_node = sum(len(m) for m in osm_map_data["osm_map_relations_node_member_indices"][b_idx])
        num_rel_way = sum(len(m) for m in osm_map_data["osm_map_relations_way_member_indices"][b_idx])
        num_rel_rel = sum(len(m) for m in osm_map_data["osm_map_relations_relation_member_indices"][b_idx])
        total = (num_nodes + num_ways * num_way_pts + num_rels
                 + num_rel_node + num_rel_way + num_rel_rel)
        return num_nodes, num_ways, num_way_pts, pts_dim, num_rels, total

    def add_orf_identifiers(self, map_features, osm_map_data, b_idx):
        """Append ORF graph identifiers (fixed member order) to each token."""
        num_nodes, num_ways, num_way_pts, _, num_rels, _ = self._counts(osm_map_data, b_idx)
        device = osm_map_data["osm_map_ways_pts"][b_idx].device

        n_orf = num_nodes + num_ways + num_rels
        orf_mat = gaussian_orthogonal_random_matrix(n_orf, n_orf, device=device)
        if n_orf < self.orf_dim:
            orf_ident = F.pad(orf_mat, (0, self.orf_dim - n_orf), "constant", 0)
        else:
            orf_ident = orf_mat[:, :self.orf_dim]

        orf_tiled = torch.tile(orf_ident, (1, 2))

        # Element-level identifiers: nodes, then ways expanded per way point.
        orf_idents = [torch.cat([
            orf_tiled[:num_nodes],
            torch.repeat_interleave(orf_tiled[num_nodes:num_nodes + num_ways], num_way_pts, dim=0),
            orf_tiled[num_nodes + num_ways:],
        ], dim=0)]

        # Relation member identifiers (fixed order: all node members of all
        # relations, then all way members, then all relation members). Each is
        # [relation_ident | member_ident].
        node_idx = osm_map_data["osm_map_relations_node_member_indices"][b_idx]
        way_idx = osm_map_data["osm_map_relations_way_member_indices"][b_idx]
        rel_idx = osm_map_data["osm_map_relations_relation_member_indices"][b_idx]

        for i in range(num_rels):
            for member in node_idx[i]:
                if member.numel():
                    orf_idents.append(torch.cat([
                        orf_ident[num_nodes + num_ways + i],
                        orf_ident[member.to(torch.long)],
                    ]).unsqueeze(0))
        for i in range(num_rels):
            for member in way_idx[i]:
                if member.numel():
                    orf_idents.append(torch.cat([
                        orf_ident[num_nodes + num_ways + i],
                        orf_ident[num_nodes + member.to(torch.long)],
                    ]).unsqueeze(0))
        for i in range(num_rels):
            for member in rel_idx[i]:
                if member.numel():
                    orf_idents.append(torch.cat([
                        orf_ident[num_nodes + num_ways + i],
                        orf_ident[num_nodes + num_ways + member.to(torch.long)],
                    ]).unsqueeze(0))

        orf_idents = torch.cat(orf_idents)
        return torch.cat([map_features, orf_idents], dim=1)

    def get_nlp_model_input(self, osm_map_data, b_idx):
        """Assemble + pad the tag token tensors for one sample's elements.

        Order matches the geometry/embedding layout: nodes, ways, relations,
        then flattened relation node/way/relation members.
        """
        def flatten(nested):
            return [x for xs in nested for x in xs]

        out = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        d = osm_map_data
        for key in out:
            out[key].extend(d[f"osm_map_nodes_tags_{key}"][b_idx])
            out[key].extend(d[f"osm_map_ways_tags_{key}"][b_idx])
            out[key].extend(d[f"osm_map_relations_tags_{key}"][b_idx])
            out[key].extend(flatten(d[f"osm_map_relations_node_member_tags_{key}"][b_idx]))
            out[key].extend(flatten(d[f"osm_map_relations_way_member_tags_{key}"][b_idx]))
            out[key].extend(flatten(d[f"osm_map_relations_relation_member_tags_{key}"][b_idx]))

        device = d["osm_map_ways_pts"][b_idx].device
        for key in out:
            seq = [el if len(el) > 0 else torch.tensor([], dtype=torch.long, device=device)
                   for el in out[key]]
            # Tag tokens may arrive on CPU (only geometry is moved to device by
            # the data loader); move the padded batch to the encoder's device so
            # the NLP model sees device-consistent input.
            out[key] = nn.utils.rnn.pad_sequence(
                seq, batch_first=True, padding_value=self.nlp_pad_token
            )[..., :self.nlp_max_tokens].to(device)
        return out

    def build_map_features(self, osm_map_data):
        """Build the per-sample raw token matrices and their valid lengths."""
        map_features = []
        lengths = []
        B = len(osm_map_data["osm_map_ways_pts"])

        for b_idx in range(B):
            num_nodes, num_ways, num_way_pts, pts_dim, _, total = self._counts(
                osm_map_data, b_idx
            )
            device = osm_map_data["osm_map_ways_pts"][b_idx].device
            dtype = osm_map_data["osm_map_ways_pts"][b_idx].dtype

            # A sample with no OSM elements contributes an all-padding row block;
            # skip the NLP/ORF work (both choke on zero elements) and let the
            # key-padding mask gate it out downstream.
            if total == 0:
                map_features.append(torch.zeros((0, self.input_dim), device=device, dtype=torch.float32))
                lengths.append(0)
                continue

            # Geometry: one row per node + one row per way point.
            geom = torch.zeros((num_nodes + num_ways * num_way_pts, pts_dim),
                               device=device, dtype=dtype)
            if num_nodes:
                geom[:num_nodes, :] = osm_map_data["osm_map_nodes_pts"][b_idx]
            if num_ways:
                geom[num_nodes:, :] = osm_map_data["osm_map_ways_pts"][b_idx].view(
                    num_ways * num_way_pts, pts_dim
                )

            # NLP tag embeddings for every element (+ relation members).
            nlp_input = self.get_nlp_model_input(osm_map_data, b_idx)
            embeddings = self.nlp_model.forward(nlp_input)["sentence_embedding"]

            map_feat = self.pos_encoder(geom.unsqueeze(0)).squeeze(0)
            # Pad geometry rows up to total token count (member/relation tokens
            # carry no geometry of their own).
            map_feat = F.pad(map_feat, (0, 0, 0, total - map_feat.shape[0]), "constant", 0)

            emb_query = torch.cat([
                embeddings[:num_nodes],
                torch.repeat_interleave(embeddings[num_nodes:num_nodes + num_ways], num_way_pts, dim=0),
                embeddings[num_nodes + num_ways:],
            ], dim=0)
            map_feat = torch.cat([map_feat, emb_query], dim=1)
            map_feat = self.add_orf_identifiers(map_feat, osm_map_data, b_idx)

            map_features.append(map_feat)
            lengths.append(total)

        padded = nn.utils.rnn.pad_sequence(map_features, batch_first=True, padding_value=0)
        return padded.to(torch.float32), lengths

    def forward(self, osm_map_data):
        """Encode OSM data into a padded token sequence + key-padding mask.

        Returns:
            tokens: ``(B, N_max, embed_dim)``.
            key_padding_mask: ``(B, N_max)`` bool, ``True`` at padded positions.
        """
        map_features, lengths = self.build_map_features(osm_map_data)
        map_features = self.map_embedding(map_features)
        # Note: like the original SDTagNet encoder we do NOT mask the self-attention
        # here — padded tokens are gated out downstream via key_padding_mask. The
        # transformer is skipped only in the degenerate all-empty case (N_max == 0).
        if map_features.shape[1] > 0:
            map_features = self.transformer_encoder(map_features)

        B, N_max, _ = map_features.shape
        key_padding_mask = torch.ones((B, N_max), dtype=torch.bool, device=map_features.device)
        for b, n in enumerate(lengths):
            if n:
                key_padding_mask[b, :n] = False
        return map_features, key_padding_mask
