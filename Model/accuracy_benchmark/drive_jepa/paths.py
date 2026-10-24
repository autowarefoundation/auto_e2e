"""Workspace paths and immutable upstream checkpoint identities."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = REPO_ROOT / ".benchmark"
SDK_ROOT = WORKSPACE / "vendor/kitscenes"
DRIVE_JEPA_ROOT = WORKSPACE / "vendor/Drive-JEPA"
DRIVE_JEPA_REVISION = "548bb8215e3aae18e162a0f12f1ba83b4d3eb57e"
NUPLAN_REVISION = "ce3c323af01c0d7ec5672f7832ef53f9c679aab0"
WEIGHTS_REVISION = "65e0d7284f69bf29d1a4864affcfd84ca4e97a2e"
CHECKPOINTS = {
    "perception-free-v1": (
        "drive_jepa_perception_free_agent_vitl.ckpt",
        "982960aed9e82900584d20c676ba9a7498add912bed94a3e7fa5f4d22bff6237",
    ),
    "perception-based-v1": (
        "drive_jepa_perception_based_agent_vitl.ckpt",
        "b3e1d4ab723715be07c8266a7dfbd58abffed6990d654afcf5fbcd8cd97eeb34",
    ),
    "perception-based-v2": (
        "drive_jepa_perception_based_agent_vitl_v2.ckpt",
        "740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05",
    ),
}
