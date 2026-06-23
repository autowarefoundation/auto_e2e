"""SD-map (OpenStreetMap) data pipeline for the OSM vector map encoder.

Offline preprocessing:
  * ``overpass``      — fetch raw OSM XML (tested SDTagNet query) + shared cache.
  * ``osm_parser``    — parse OSM XML, project to ego frame, crop to a patch.
  * ``osm_tokenize``  — resample geometry + tokenise tags into ``osm_map_data``.
  * ``cache``         — build/load per-episode tokenised shards.
  * ``nlp_download``  — fetch the SDTagNet NLP tag encoder from HuggingFace.

Training time:
  * ``collate.collate_osm_batch`` — DataLoader collate for ragged OSM batches.

The model-ready ``osm_map_data`` dict is consumed by
``model_components.map_encoder.osm_vector.OSMVectorMapEncoder``.
"""

from .collate import collate_osm_batch
from .nlp_download import download_nlp_weights
from .osm_parser import OSMMapElements, parse
from .osm_tokenize import extract_osm_map_data, tokenize_osm_elements
from .overpass import (
    bbox_from_points,
    build_query,
    fetch_raw_osm,
    load_or_fetch_osm,
    load_or_fetch_osm_bbox,
    load_or_fetch_osm_for_trace,
)
from .wgs84_to_local import wgs84_to_ego_local

# Note: `visualize` is intentionally NOT imported here. It is run as a module
# (`python -m data_parsing.osm_sd_map.visualize`); importing it in the package
# __init__ would double-import it under runpy and emit a RuntimeWarning. Import
# it explicitly where needed: `from data_parsing.osm_sd_map.visualize import ...`.

__all__ = [
    "collate_osm_batch",
    "download_nlp_weights",
    "OSMMapElements",
    "parse",
    "extract_osm_map_data",
    "tokenize_osm_elements",
    "build_query",
    "fetch_raw_osm",
    "load_or_fetch_osm",
    "wgs84_to_ego_local",
]
