"""Put ``Model/`` on the import path for tests under ``Platform/tests``.

The pipelines import model packages by their top-level name — ``training.*``,
``evaluation.*`` — which resolves because ``Model/`` is on ``PYTHONPATH`` at run
time and, for ``Model/tests``, because ``Model/pytest.ini`` and
``Model/tests/__init__.py`` make pytest insert ``Model/`` itself.

Neither applies here, so a test that reaches into the model packages fails with
``ModuleNotFoundError: No module named 'training'`` unless the caller happened to
export ``PYTHONPATH`` first. Doing it here keeps these tests runnable with a bare
``pytest Platform/tests`` from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2] / "Model"

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
