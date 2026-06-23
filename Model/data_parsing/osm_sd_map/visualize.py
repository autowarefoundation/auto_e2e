"""Visualize a tokenised ``osm_map_data`` sample (SDTagNet-style) + validate it.

Renders exactly what the OSM vector encoder consumes — the resampled way
polylines, node points, decoded tag text, and relation member links — in the
ego frame, so you can eyeball that the data pipeline (Overpass query, ego
projection, patch crop, tokenisation) is correct on a real OSM map. The style
mirrors SDTagNet's ``av2_vis_pred.py`` OSM panel: highway ways green, other ways
red, nodes blue, tags as boxed text, relation members linked by arrows to the
relation centroid.

Run directly for an end-to-end real-data check (fetch real OSM, tokenise,
visualise, and optionally run the encoder):

    python -m data_parsing.osm_sd_map.visualize \
        --lat 48.0 --lon 8.0 --heading 0.0 \
        --download-nlp --run-encoder --out osm_map_vis.png

The encoder/tokeniser need the NLP checkpoint (``--nlp-model-path`` or
``--download-nlp``).
"""

from __future__ import annotations

import argparse
import io

import numpy as np


def _decode(tokenizer, token_lists):
    if tokenizer is None:
        return ["" for _ in token_lists]
    return tokenizer.batch_decode([t.tolist() for t in token_lists], skip_special_tokens=True)


def visualize_osm_map_data(sample, pc_range, out_path, tokenizer=None,
                           clip_text=True, title=None):
    """Render one (single-sample) ``osm_map_data`` dict to ``out_path``.

    Args:
        sample: a per-frame dict from ``osm_tokenize.extract_osm_map_data``.
        pc_range: ``(x_min, y_min, z_min, x_max, y_max, z_max)`` ego metres.
        out_path: output PNG path.
        tokenizer: HF tokenizer to decode tag tokens to text (optional; without
            it geometry is drawn but no labels).
        clip_text: truncate long tag strings in the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes_pts = sample["osm_map_nodes_pts"]
    ways_pts = sample["osm_map_ways_pts"]
    node_tags = _decode(tokenizer, sample["osm_map_nodes_tags_input_ids"])
    way_tags = _decode(tokenizer, sample["osm_map_ways_tags_input_ids"])
    rel_tags = _decode(tokenizer, sample["osm_map_relations_tags_input_ids"])

    fig = plt.figure(figsize=(8, 4))
    plt.xlim(pc_range[0], pc_range[3])
    plt.ylim(pc_range[1], pc_range[4])
    plt.gca().set_aspect("equal")
    plt.axis("off")
    if title:
        plt.title(title, fontsize=6)

    text_bbox = dict(boxstyle="square", ec=(0.3, 0.3, 0.3, 0.3), fc=(0.3, 0.3, 0.3, 0.3))

    def _label(x, y, s):
        if not s:
            return
        if clip_text and len(s) > 60:
            s = s[:60] + "..."
        plt.text(x, y, s, color="black", ha="center", va="center", fontsize=3, bbox=text_bbox)

    # node labels
    for i, s in enumerate(node_tags):
        _label(float(nodes_pts[i][0]), float(nodes_pts[i][1]), s)
    # way labels (placed at the last resampled point, matching SDTagNet)
    for i, s in enumerate(way_tags):
        _label(float(ways_pts[i][-1][0]), float(ways_pts[i][-1][1]), s)

    # relation member links + label at centroid
    node_idx = sample["osm_map_relations_node_member_indices"]
    way_idx = sample["osm_map_relations_way_member_indices"]
    for i, s in enumerate(rel_tags):
        member_pts = []
        for idx in (node_idx[i] if len(node_idx) > i else []):
            member_pts.append(nodes_pts[idx].squeeze().numpy()[:2])
        for idx in (way_idx[i] if len(way_idx) > i else []):
            member_pts.append(np.average(ways_pts[idx].squeeze().numpy(), axis=0)[:2])
        if not member_pts:
            continue
        member_pts = np.vstack(member_pts)
        center = np.average(member_pts, axis=0)
        for pt in member_pts:
            d = center - pt
            plt.arrow(pt[0], pt[1], d[0], d[1], color="black", linewidth=0.6, alpha=0.8, zorder=5)
        _label(center[0], center[1], s)

    # geometry
    if nodes_pts.numel():
        plt.scatter(nodes_pts[:, 0], nodes_pts[:, 1], linewidth=1.5, color="blue", zorder=4)
    for i, line in enumerate(ways_pts):
        color = "green" if "highway" in (way_tags[i] if i < len(way_tags) else "") else "red"
        plt.plot(line[:, 0], line[:, 1], linewidth=2, alpha=0.8, color=color)

    # ego marker at origin (forward = +x)
    plt.scatter([0], [0], marker="*", s=120, color="black", zorder=6)
    plt.arrow(0, 0, 4, 0, color="black", width=0.3, zorder=6)

    plt.savefig(out_path, format="png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_sample(args, tokenizer, pc_range):
    from .osm_parser import parse
    from .osm_tokenize import extract_osm_map_data
    from .overpass import load_or_fetch_osm

    if args.osm_file:
        with open(args.osm_file, "r", encoding="utf-8") as f:
            xml = f.read()
    else:
        xml = load_or_fetch_osm(args.lat, args.lon, args.radius, args.raw_osm_cache_dir)
        if xml is None:
            raise SystemExit("Overpass fetch failed and no --osm-file given.")

    osm = parse(io.StringIO(xml))
    print(f"parsed OSM: {len(osm.nodes)} nodes, {len(osm.ways)} ways, {len(osm.relations)} relations")
    sample = extract_osm_map_data(
        osm, args.lat, args.lon, args.heading, tokenizer, pc_range,
        fixed_num=args.fixed_num, pts_dim=3,
    )
    n_nodes = sample["osm_map_nodes_pts"].shape[0]
    n_ways = sample["osm_map_ways_pts"].shape[0]
    n_rels = len(sample["osm_map_relations_tags_input_ids"])
    print(f"in-patch elements: {n_nodes} nodes, {n_ways} ways, {n_rels} relations")
    return sample


def _run_encoder(sample, nlp_model_path, pc_range):
    import torch

    from .collate import collate_osm_batch
    from model_components.map_encoder.osm_vector import OSMVectorMapEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = OSMVectorMapEncoder(nlp_model_path=nlp_model_path, pc_range=pc_range).to(device).eval()
    batch = collate_osm_batch([sample])
    # Move the geometry tensors to the encoder's device (the encoder places its
    # tag tokens on the geometry's device).
    for key in ("osm_map_nodes_pts", "osm_map_ways_pts"):
        batch[key] = [t.to(device) for t in batch[key]]
    with torch.no_grad():
        tokens, mask = enc(batch)
    valid = int((~mask).sum().item())
    print(f"encoder OK [{device}]: tokens {tuple(tokens.shape)}, "
          f"valid {valid}/{mask.shape[1]}, finite={bool(torch.isfinite(tokens).all())}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--heading", type=float, default=0.0, help="radians, 0=north")
    p.add_argument("--osm-file", default=None, help="local .osm XML; else fetch via Overpass")
    p.add_argument("--radius", type=int, default=800, help="Overpass fetch radius (m)")
    p.add_argument("--raw-osm-cache-dir", default=None)
    p.add_argument("--nlp-model-path", default=None, help="local SentenceTransformer checkpoint dir")
    p.add_argument("--download-nlp", action="store_true", help="download NLP weights if path missing")
    p.add_argument("--nlp-cache-dir", default=None,
                   help="where --download-nlp stores weights (default: <repo>/checkpoints/sdtagnet_nlp)")
    p.add_argument("--pc-range", type=float, nargs=6,
                   default=[-60.0, -30.0, -5.0, 60.0, 30.0, 3.0])
    p.add_argument("--fixed-num", type=int, default=10)
    p.add_argument("--out", default="osm_map_vis.png")
    p.add_argument("--run-encoder", action="store_true", help="also run the encoder forward")
    args = p.parse_args()

    pc_range = tuple(args.pc_range)

    nlp_path = args.nlp_model_path
    if nlp_path is None and args.download_nlp:
        from .nlp_download import download_nlp_weights
        nlp_path = download_nlp_weights(
            **({"target_dir": args.nlp_cache_dir} if args.nlp_cache_dir else {})
        )
    tokenizer = None
    if nlp_path is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(nlp_path)

    sample = _build_sample(args, tokenizer, pc_range)
    out = visualize_osm_map_data(sample, pc_range, args.out, tokenizer=tokenizer,
                                 title=f"OSM SD map @ ({args.lat:.5f}, {args.lon:.5f})")
    print(f"wrote {out}")

    if args.run_encoder:
        if nlp_path is None:
            raise SystemExit("--run-encoder needs --nlp-model-path or --download-nlp")
        _run_encoder(sample, nlp_path, pc_range)


if __name__ == "__main__":
    main()
