// Static per-dataset rig maps: shard member "cam_N" -> rig position + a
// bird's-eye grid cell so the mosaic can lay cameras out the way they point
// (front on top, rear on the bottom, left cameras on the left, right on the
// right), with the ego in the middle.
//
// The shard packer names camera members cam_0, cam_1, ... in the source
// dataset's CAMERA_NAMES order (Model/data_parsing/*/camera.py).
//
// NVIDIA (nvidia_physical_ai/camera.py): 7 cameras
//   0 front_wide, 1 front_tele, 2 cross_left, 3 cross_right,
//   4 rear_left, 5 rear_right, 6 rear_tele
// L2D (l2d/camera.py): 6 surround cameras (map is packed separately)
//   0 front_left, 1 left_forward, 2 right_forward,
//   3 left_backward, 4 rear, 5 right_backward
// KITScenes (kit_scenes/camera.py): 7 cameras
//   0 base_front_center, 1 ring_front, 2 ring_front_left,
//   3 ring_front_right, 4 ring_rear, 5 ring_rear_left, 6 ring_rear_right

export interface RigCam {
  label: string;
  // 1-based cell in a 3-column bird's-eye grid (row grows toward the rear).
  row: number;
  col: number;
}

// 3-col grid, ego implied at (2,2):
//   (1,1) front_wide   (1,2) front_tele   (1,3) .
//   (2,1) cross_left   (2,2) EGO          (2,3) cross_right
//   (3,1) rear_left    (3,2) rear_tele    (3,3) rear_right
const NVIDIA_RIG: Record<string, RigCam> = {
  cam_0: { label: "front-wide", row: 1, col: 1 },
  cam_1: { label: "front-tele", row: 1, col: 2 },
  cam_2: { label: "cross-left", row: 2, col: 1 },
  cam_3: { label: "cross-right", row: 2, col: 3 },
  cam_4: { label: "rear-left", row: 3, col: 1 },
  cam_5: { label: "rear-right", row: 3, col: 3 },
  cam_6: { label: "rear-tele", row: 3, col: 2 },
};

// 3-col grid, ego implied at (2,2):
//   (1,1) .              (1,2) front-left     (1,3) .
//   (2,1) left-forward   (2,2) EGO            (2,3) right-forward
//   (3,1) left-backward  (3,2) rear           (3,3) right-backward
const L2D_RIG: Record<string, RigCam> = {
  cam_0: { label: "front-left", row: 1, col: 2 },
  cam_1: { label: "left-forward", row: 2, col: 1 },
  cam_2: { label: "right-forward", row: 2, col: 3 },
  cam_3: { label: "left-backward", row: 3, col: 1 },
  cam_4: { label: "rear", row: 3, col: 2 },
  cam_5: { label: "right-backward", row: 3, col: 3 },
  // cam_6 (map) only exists in stale Phase-1 shards; fresh shards pack map.jpg
  // separately. Placed off the ego cell if present.
  cam_6: { label: "map", row: 1, col: 1 },
};

// Four columns keep both forward-facing cameras visible without occupying the
// ego cell. The surround ring still reads left-to-right around the vehicle.
const KITSCENES_RIG: Record<string, RigCam> = {
  cam_0: { label: "front-center", row: 1, col: 2 },
  cam_1: { label: "ring-front", row: 1, col: 3 },
  cam_2: { label: "front-left", row: 1, col: 1 },
  cam_3: { label: "front-right", row: 1, col: 4 },
  cam_4: { label: "rear", row: 3, col: 2 },
  cam_5: { label: "rear-left", row: 3, col: 1 },
  cam_6: { label: "rear-right", row: 3, col: 4 },
};

const RIGS: Record<string, Record<string, RigCam>> = {
  nvidia_av: NVIDIA_RIG,
  l2d: L2D_RIG,
  kitscenes: KITSCENES_RIG,
};

// rigCam returns the rig position + grid cell for a "cam_N" identifier.
// Falls back to a sequential cell + the raw id for unknown datasets/cameras.
export function rigCam(dataset: string, cam: string, index: number): RigCam {
  const mapped = RIGS[dataset]?.[cam];
  if (mapped) return mapped;
  // Unknown rig: lay out sequentially in a 3-col grid.
  return { label: cam, row: Math.floor(index / 3) + 1, col: (index % 3) + 1 };
}

// camLabel returns just the rig position label (back-compat helper).
export function camLabel(dataset: string, cam: string): string {
  return RIGS[dataset]?.[cam]?.label ?? cam;
}

// gridDimensions returns the number of rows/cols spanned by a dataset's rig,
// so the mosaic can size its CSS grid. Defaults to 3x3.
export function gridDimensions(dataset: string, cams: string[]): {
  rows: number;
  cols: number;
} {
  let rows = 3;
  let cols = 3;
  cams.forEach((cam, i) => {
    const c = rigCam(dataset, cam, i);
    rows = Math.max(rows, c.row);
    cols = Math.max(cols, c.col);
  });
  return { rows, cols };
}
