# SD-map (OpenStreetMap) vector pipeline

Feeds OpenStreetMap standard-definition (SD) map priors to AutoE2E as a set of
**text-annotated vector tokens**, ported from
[SDTagNet](https://arxiv.org/abs/2506.08997) (NeurIPS 2025) and adapted from
online HD-map construction to end-to-end driving. The model side lives in
[`model_components/map_encoder/osm_vector`](../../model_components/map_encoder/osm_vector)
(the encoder) and
[`map_bev_fusion/osm_cross_attn_fusion.py`](../../model_components/map_encoder/map_bev_fusion/osm_cross_attn_fusion.py)
(BEV ↔ token cross-attention).

## How it differs from the rendered map tile

The existing [`map_rendering`](../map_rendering) branch rasterises the road
network into an RGB tile. This branch instead keeps OSM elements as **vectors**
(nodes / ways / relations) and embeds their **tag strings** with a pretrained
NLP encoder, so arbitrary semantics (lane counts, surface, turn restrictions,
amenities, ...) flow into the model without a hand-picked class taxonomy. The
two branches **share one on-disk OSM cache** (raw Overpass XML); see
[`map_rendering/cache.py`](../map_rendering/cache.py).

## One-time setup: NLP tag encoder

```python
from data_parsing.osm_sd_map import download_nlp_weights

nlp_path = download_nlp_weights()   # -> <repo>/checkpoints/sdtagnet_nlp (gitignored)
```

`nlp_path` is passed both to the encoder (`nlp_model_path=`) and used internally
for the matching tokenizer.

## Offline preprocessing (per episode)

Slow work — Overpass fetch, OSM parse, tokenisation — is done once and cached as
one `.pt` shard per episode. Pair with `L2DDataset`:

```python
from transformers import AutoTokenizer
from data_parsing.l2d import L2DDataset
from data_parsing.osm_sd_map.cache import build_episode_osm_cache

tokenizer = AutoTokenizer.from_pretrained(nlp_path)
ds = L2DDataset(repo_id="yaak-ai/L2D", episodes=[0])
pc_range = (-60.0, -30.0, -5.0, 60.0, 30.0, 3.0)

for ep in [0]:
    build_episode_osm_cache(
        episode_id=ep,
        frames=ds.episode_ego_poses(ep),       # (frame_idx, lat, lon, heading)
        tokenizer=tokenizer,
        pc_range=pc_range,
        out_dir="cache/osm_tokens",
        raw_osm_cache_dir="cache/osm_raw",     # shared with map_rendering
    )
```

## Training time

```python
from torch.utils.data import DataLoader
from data_parsing.l2d import L2DDataset
from data_parsing.osm_sd_map import collate_osm_batch

ds = L2DDataset(repo_id="yaak-ai/L2D", osm_cache_dir="cache/osm_tokens")
loader = DataLoader(ds, batch_size=6, collate_fn=collate_osm_batch)  # ragged OSM
```

Build the model with the OSM branch and pass the collated batch as the map input
(the encoder only reads the `osm_map_*` keys, so the whole batch dict is fine):

```python
from model_components.auto_e2e import AutoE2E

model = AutoE2E(
    map_type="osm_vector",
    map_encoder_kwargs=dict(nlp_model_path=nlp_path, pc_range=pc_range),
    map_fusion_mode="osm_cross_attn",
)

for batch in loader:
    out = model(batch["visual_tiles"], batch, batch["visual_history"],
                batch["egomotion_history"], mode="train",
                trajectory_target=batch["trajectory_target"])
```

With the default encoder config the token dimension is
`pts_dim·pos_num_feats + nlp_embed_dim + 2·orf_dim = 3·16 + 144 + 2·64 = 320`
(matching SDTagNet's canonical `input_dim`).

## Validate on a real OSM map

One command fetches a real OSM area around a GPS point, runs the full pipeline
(Overpass query → ego projection → patch crop → tokenise), renders a
SDTagNet-style figure, and (optionally) runs the encoder forward on it:

```bash
cd Model
python -m data_parsing.osm_sd_map.visualize \
    --lat 48.9930 --lon 8.4037 --heading 0.0 \
    --download-nlp --run-encoder --out osm_map_vis.png
```

The figure shows the ego (★, forward = +x), way polylines (highway = green,
other = red), node points (blue), decoded tag text, and relation member links —
exactly what the encoder consumes — so geometry, ego framing and tag decoding
can be eyeballed for correctness. `--run-encoder` additionally prints the token
tensor shape, valid/total token counts and a finiteness check, validating the
encoder on real data. Pass `--osm-file area.osm` to use a local extract instead
of hitting Overpass, and `--nlp-model-path PATH` to use already-downloaded NLP
weights.

`visualize_osm_map_data(sample, pc_range, out_path, tokenizer=...)` is also
importable to render any cached per-frame sample.

## Modules

| File | Purpose |
| --- | --- |
| `overpass.py`     | Tested SDTagNet Overpass query + shared raw-OSM XML cache. |
| `osm_parser.py`   | Parse OSM XML, project to ego frame, crop to a metric patch. |
| `wgs84_to_local.py` | Generic WGS84 → ego (X=forward, Y=left) projection. |
| `osm_tokenize.py` | Resample way geometry + tokenise tags → `osm_map_data`. |
| `cache.py`        | Build / load per-episode tokenised shards. |
| `nlp_download.py` | Fetch the SDTagNet NLP encoder from HuggingFace. |
| `collate.py`      | DataLoader `collate_fn` for ragged `osm_map_data`. |
| `visualize.py`    | SDTagNet-style SD-map figure + real-data validation CLI. |

## Notes

- **Fetch extent**: each episode does one Overpass fetch sized to the
  **bounding box of its trajectory plus a margin**, which covers long episodes
  (a fixed centroid radius silently misses frames whose patch falls outside it).
  The margin defaults to the rendered-tile branch's render radius (~800 m,
  `DEFAULT_FETCH_MARGIN_M`) so both branches request the same area and **share a
  single raw-OSM download** (the vector branch crops to `pc_range` at
  tokenisation regardless). For a **vector-only** setup with no rendered tiles,
  pass `fetch_margin_m=patch_reach_m(pc_range)` to fetch only the ~70 m the patch
  needs. The shared cache also reuses any already-fetched XML whose bbox
  **contains** a request, so smaller fetches still hit a larger cached one.
- **Coordinates**: ego frame X=forward, Y=left, Z=up (matches `pc_range`); OSM is
  treated as 2D and gets a constant `z=0` when `pts_dim=3`.
- **Heading**: `L2DDataset.episode_ego_poses` uses `vehicle[1]` as heading
  (radians); confirm the convention against your data during validation.
- **osmnx**: only the rendered-tile branch needs `osmnx` (to build a graph from
  the shared XML); the vector branch does not.
