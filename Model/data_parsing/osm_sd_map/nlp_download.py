"""Download the SDTagNet NLP tag encoder from HuggingFace.

The OSM vector encoder needs a SentenceTransformer that embeds OSM tag strings.
The pretrained model is shipped as a tar.gz in the SDTagNet HuggingFace dataset
repo, under ``nlp_encoder/``:

  * ``bert-144-osm-tags-embed-from_scratch.tar.gz`` — the trained tag encoder
    (this is what AutoE2E uses; extracts to ``bert-144-osm-tags-embed-from_scratch/
    checkpoint-10628/``, the path referenced by SDTagNet's config),
  * ``bert-144-l6-reset-custom-tokenizer.tar.gz`` — the untrained base (for
    re-running the NLP pretraining only).

This helper downloads the archive, extracts it, and returns the local path to
the SentenceTransformer directory (the folder containing ``modules.json``),
suitable for both ``SentenceTransformer(path)`` and
``AutoTokenizer.from_pretrained(path)``.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

# SDTagNet release on the HuggingFace Hub.
DEFAULT_REPO_ID = "immel-f/SDTagNet"
DEFAULT_REPO_TYPE = "dataset"
# Trained tag encoder archive (verified path in the repo).
DEFAULT_ARCHIVE = "nlp_encoder/bert-144-osm-tags-embed-from_scratch.tar.gz"

# Fixed default download location, anchored to the repo root (auto_e2e/) rather
# than the current working directory, so repeated runs from any directory reuse
# one download. (.../osm_sd_map/nlp_download.py -> parents[3] == auto_e2e/)
DEFAULT_TARGET_DIR = Path(__file__).resolve().parents[3] / "checkpoints" / "sdtagnet_nlp"

# Files that mark the root of a SentenceTransformer model directory.
_ST_MARKERS = ("modules.json", "config_sentence_transformers.json")


def _find_model_dir(root: Path):
    """Return the SentenceTransformer dir under ``root`` (contains modules.json)."""
    markers = [p for m in _ST_MARKERS for p in root.rglob(m)]
    if not markers:
        return None
    # Prefer a trainer "checkpoint-XXXX" dir if several models are present.
    markers.sort(key=lambda p: (0 if "checkpoint" in p.parent.name else 1, len(str(p))))
    return markers[0].parent


def download_nlp_weights(
    target_dir: str | Path = DEFAULT_TARGET_DIR,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    archive: str = DEFAULT_ARCHIVE,
    force: bool = False,
) -> str:
    """Download + extract the NLP tag encoder; return the model directory path.

    Skips download/extraction if a SentenceTransformer is already present under
    ``target_dir`` (unless ``force``).

    Args:
        target_dir: Local directory to download + extract into.
        repo_id / repo_type: HuggingFace location.
        archive: Path of the tar.gz within the repo (override to fetch the base
            tokenizer archive instead of the trained model).
        force: Re-download + re-extract even if a model is already present.
    """
    target = Path(target_dir)

    if not force:
        existing = _find_model_dir(target) if target.exists() else None
        if existing is not None:
            logger.info("NLP weights already present at %s; skipping download.", existing)
            return str(existing)

    from huggingface_hub import hf_hub_download

    target.mkdir(parents=True, exist_ok=True)
    archive_path = hf_hub_download(
        repo_id=repo_id, repo_type=repo_type, filename=archive, local_dir=str(target),
    )

    # Extract (safe filter strips absolute/.. paths; Python 3.12 default).
    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            tar.extractall(target, filter="data")
        except TypeError:  # filter kw added in 3.12; fall back for older pythons
            tar.extractall(target)

    model_dir = _find_model_dir(target)
    if model_dir is None:
        raise RuntimeError(
            f"Extracted {archive} but found no SentenceTransformer model "
            f"(no {' / '.join(_ST_MARKERS)}) under {target}."
        )
    logger.info("NLP weights ready at %s", model_dir)
    return str(model_dir)
