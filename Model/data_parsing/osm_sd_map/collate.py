"""Custom ``collate_fn`` for batches carrying ragged ``osm_map_data``.

``OSMVectorMapEncoder`` consumes per-key lists indexed by batch position (the
number of OSM elements varies per sample, so the tensors cannot be stacked).
Every key prefixed ``osm_map_`` is therefore grouped into a Python list over the
batch; all other keys (images, egomotion, targets, ...) go through the default
collate so existing behaviour is unchanged.
"""

from __future__ import annotations

from torch.utils.data import default_collate

_OSM_PREFIX = "osm_map_"


def collate_osm_batch(batch):
    """Collate samples that may contain ``osm_map_*`` ragged fields.

    Args:
        batch: list of per-sample dicts.

    Returns:
        dict where ``osm_map_*`` keys map to length-B lists and all other keys
        are default-collated (stacked) tensors / values.
    """
    if not batch:
        return {}

    keys = batch[0].keys()
    osm_keys = [k for k in keys if k.startswith(_OSM_PREFIX)]
    other_keys = [k for k in keys if not k.startswith(_OSM_PREFIX)]

    out = {}
    if other_keys:
        collated = default_collate([{k: s[k] for k in other_keys} for s in batch])
        out.update(collated)
    for k in osm_keys:
        out[k] = [s[k] for s in batch]
    return out
