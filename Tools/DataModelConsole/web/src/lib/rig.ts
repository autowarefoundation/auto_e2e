// Static per-dataset rig maps: shard member "cam_N" -> human rig position.
//
// The shard packer names camera members cam_0, cam_1, ... in the source
// dataset's CAMERA_NAMES order (data_parsing/*/camera.py). Mapping the opaque
// index back to a rig position ("front-left", "rear", ...) lets the player
// label each tile by where the camera actually points instead of "cam_3".
//
// NVIDIA (data_parsing/nvidia_physical_ai/camera.py): 7 real cameras.
// L2D (data_parsing/l2d/camera.py): 6 surround cameras; cam_6 is the BEV
// nav-map tile (the stale Phase-1 shard packs it as cam_6).

const NVIDIA_RIG: Record<string, string> = {
  cam_0: "front-wide",
  cam_1: "front-tele",
  cam_2: "cross-left",
  cam_3: "cross-right",
  cam_4: "rear-left",
  cam_5: "rear-right",
  cam_6: "rear-tele",
};

const L2D_RIG: Record<string, string> = {
  cam_0: "front-left",
  cam_1: "left-forward",
  cam_2: "right-forward",
  cam_3: "left-backward",
  cam_4: "rear",
  cam_5: "right-backward",
  cam_6: "map",
};

const RIGS: Record<string, Record<string, string>> = {
  nvidia_av: NVIDIA_RIG,
  l2d: L2D_RIG,
};

// camLabel returns the rig position for a "cam_N" identifier, falling back to
// the raw id for unknown datasets/cameras.
export function camLabel(dataset: string, cam: string): string {
  return RIGS[dataset]?.[cam] ?? cam;
}
