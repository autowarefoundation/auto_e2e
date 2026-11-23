"""AutoE2E as a driving policy for autoware_carla_scenario (ACS).

ACS hands its ego vehicle to a policy over alpasim's ``egodriver`` gRPC
contract; :mod:`carla_driver_interface` is the policy side of it.
:class:`~acs_autoe2e.policy.AutoE2EPolicy` is a ``BaseDriver`` that runs the
AutoE2E reactive model on the cameras ACS renders.

Like the AlpaSim plugin, the package imports the AutoE2E model code (the
repository's ``Model/`` directory) as top-level packages, and the rig profiles
of the AlpaSim plugin (``alpasim_autoe2e.config``, which needs nothing from
AlpaSim). Inside this repository it finds both itself; elsewhere, point
``AUTOE2E_MODEL_ROOT`` at a copy of ``Model/``.
"""

import os
import sys
from pathlib import Path


def _add_model_root_to_path() -> None:
    configured = os.environ.get("AUTOE2E_MODEL_ROOT")
    model_root = Path(configured) if configured else Path(__file__).resolve().parents[3]
    for path in (model_root, model_root / "plugins" / "alpasim_driver"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))


_add_model_root_to_path()
