"""Turn extracted OSM patch elements into model-ready ``osm_map_data``.

Ports the tokenisation + fixed-point resampling from SDTagNet's
``osm_map_pipeline`` (canonical, no-augmentation path) and drops the mmcv
``DataContainer`` wrapping. Produces a *single-sample* dict; batch them with
``collate.collate_osm_map_data`` before feeding ``OSMVectorMapEncoder``.

The per-sample dict keys (lists are over elements within the sample):

    osm_map_nodes_pts                                   tensor (Nn, pts_dim)
    osm_map_ways_pts                                    tensor (Nw, P, pts_dim)
    osm_map_{nodes,ways,relations}_tags_{input_ids,token_type_ids,attention_mask}
                                                        list of 1D token tensors
    osm_map_relations_{node,way,relation}_member_tags_{...}
                                                        list[rel] of list[member] of 1D
    osm_map_relations_{node,way,relation}_member_indices
                                                        list[rel] of 1D long tensor
"""

from __future__ import annotations

import numpy as np
import torch

from .osm_parser import OSMMapElements

_TAG_KEYS = ("input_ids", "token_type_ids", "attention_mask")


def _to_long_list(tokenized):
    """HF tokenizer output (dict of list-of-lists) -> dict of list of 1D tensors."""
    out = {}
    for key in _TAG_KEYS:
        vals = tokenized.get(key, [])
        out[key] = [torch.as_tensor(entry, dtype=torch.long) for entry in vals]
    return out


def _fixed_num_sampled_points(linestrings, pc_range, fixed_num, pts_dim):
    """Resample each LineString to ``fixed_num`` arc-length points, clamped to pc_range."""
    max_x = (pc_range[3] - pc_range[0]) / 2.0
    max_y = (pc_range[4] - pc_range[1]) / 2.0
    pts_list = []
    for inst in linestrings:
        distances = np.linspace(0, inst.length, fixed_num)
        sampled = np.array([list(inst.interpolate(d).coords) for d in distances]).reshape(-1, 2)
        pts_list.append(sampled)
    arr = np.asarray(pts_list, dtype=np.float32)  # (Nw, fixed_num, 2)
    t = torch.from_numpy(arr)
    t[:, :, 0] = torch.clamp(t[:, :, 0], min=-max_x, max=max_x)
    t[:, :, 1] = torch.clamp(t[:, :, 1], min=-max_y, max=max_y)
    if pts_dim == 3:
        z = torch.zeros((t.shape[0], t.shape[1], 1), dtype=t.dtype)
        t = torch.cat([t, z], dim=2)
    return t


def _nodes_to_tensor(node_pts, pts_dim):
    arr = np.asarray(node_pts, dtype=np.float32).reshape(-1, 2)
    t = torch.from_numpy(arr)
    if pts_dim == 3:
        t = torch.cat([t, torch.zeros((t.shape[0], 1), dtype=t.dtype)], dim=1)
    return t


def tokenize_osm_elements(elements, tokenizer, pc_range, fixed_num=10, pts_dim=3):
    """Convert a raw patch-elements dict into a single-sample ``osm_map_data`` dict.

    Args:
        elements: output of ``OSMMapElements.get_elements_in_patch``.
        tokenizer: a HuggingFace tokenizer matching the NLP encoder.
        pc_range: ``(x_min, y_min, z_min, x_max, y_max, z_max)`` in ego metres.
        fixed_num: points sampled per way.
        pts_dim: 3 (xyz) or 2 (xy). Must match the encoder's ``pts_dim``.
    """
    sample = {}

    # --- geometry ---
    node_pts = elements["osm_map_nodes_pts"]
    if len(node_pts):
        sample["osm_map_nodes_pts"] = _nodes_to_tensor(node_pts, pts_dim)
    else:
        sample["osm_map_nodes_pts"] = torch.zeros((0, pts_dim), dtype=torch.float32)

    ways = elements["osm_map_ways_pts"]
    if ways:
        sample["osm_map_ways_pts"] = _fixed_num_sampled_points(ways, pc_range, fixed_num, pts_dim)
    else:
        sample["osm_map_ways_pts"] = torch.zeros((0, fixed_num, pts_dim), dtype=torch.float32)

    # --- element tag tokens ---
    for name in ("nodes", "ways", "relations"):
        tags = elements[f"osm_map_{name}_tags"]
        tok = _to_long_list(tokenizer(list(tags), padding=True)) if tags else {k: [] for k in _TAG_KEYS}
        for key in _TAG_KEYS:
            sample[f"osm_map_{name}_tags_{key}"] = tok[key]

    # --- relation member tag tokens (nested per relation) ---
    for name in ("node", "way", "relation"):
        member_tags = elements[f"osm_map_relations_{name}_member_tags"]
        nested = {key: [] for key in _TAG_KEYS}
        for per_rel in member_tags:
            if per_rel:
                tok = _to_long_list(tokenizer(list(per_rel), padding=True))
            else:
                tok = {key: [] for key in _TAG_KEYS}
            for key in _TAG_KEYS:
                nested[key].append(tok[key])
        for key in _TAG_KEYS:
            sample[f"osm_map_relations_{name}_member_tags_{key}"] = nested[key]

    # --- relation member indices ---
    for name in ("node", "way", "relation"):
        idx_lists = elements[f"osm_map_relations_{name}_member_indices"]
        sample[f"osm_map_relations_{name}_member_indices"] = [
            torch.as_tensor(idx, dtype=torch.long) for idx in idx_lists
        ]

    return sample


def extract_osm_map_data(osm_map_elements: OSMMapElements, ego_lat, ego_lon, ego_heading,
                         tokenizer, pc_range, fixed_num=10, pts_dim=3):
    """End-to-end per-frame: project to ego frame, crop to pc_range, tokenise.

    ``osm_map_elements`` is a parsed ``OSMMapElements`` (from
    ``osm_parser.parse``); this is the one-call path for offline preprocessing.
    """
    from shapely.geometry import box

    osm_map_elements.build_node_way_lists(ego_lat, ego_lon, ego_heading)
    patch = box(pc_range[0], pc_range[1], pc_range[3], pc_range[4])
    elements = osm_map_elements.get_elements_in_patch(patch)
    return tokenize_osm_elements(elements, tokenizer, pc_range, fixed_num=fixed_num, pts_dim=pts_dim)
