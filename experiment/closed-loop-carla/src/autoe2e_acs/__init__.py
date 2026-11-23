"""autoware_carla_scenario (ACS) presets for running AutoE2E in CARLA.

Installed into the ACS environment, this package adds the ``driver`` presets of
``conf/driver`` (``driver=autoe2e_nuplan``: a Lincoln MKZ with the six nuPlan
cameras) to the ``scenario`` CLI, through the
``autoware_carla_scenario.scenarios`` entry point. The policy itself runs in
the AutoE2E environment (``Model/plugins/carla_driver``).
"""

from __future__ import annotations

from pathlib import Path

CONF_DIR = Path(__file__).resolve().parent / "conf"


def register() -> None:
    """Register the presets with the ``scenario`` CLI (called at its start-up)."""
    from autoware_carla_scenario import register_conf_dir

    register_conf_dir(CONF_DIR)
