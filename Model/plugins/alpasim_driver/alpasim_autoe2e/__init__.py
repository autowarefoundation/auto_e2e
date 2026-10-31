"""AutoE2E AlpaSim driver plugin.

The plugin imports the AutoE2E model code (the repository's ``Model/``
directory) as top-level packages. Inside this repository it finds that
directory itself; elsewhere, e.g. in the AlpaSim driver container, point
``AUTOE2E_MODEL_ROOT`` at a copy of it.
"""

import os
import sys
from pathlib import Path


def _add_model_root_to_path() -> None:
    configured = os.environ.get("AUTOE2E_MODEL_ROOT")
    model_root = Path(configured) if configured else Path(__file__).resolve().parents[3]
    if (model_root / "model_components").is_dir() and str(model_root) not in sys.path:
        sys.path.append(str(model_root))


_add_model_root_to_path()
