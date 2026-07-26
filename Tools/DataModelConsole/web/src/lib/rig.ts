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
// KITScenes has two published slot contracts:
//   7-view: 0 base_front_center, 1 ring_front, 2 ring_front_left,
//           3 ring_front_right, 4 ring_rear, 5 ring_rear_left,
//           6 ring_rear_right
//   6-view: 0 base_front_center, 1 ring_front_left, 2 ring_front_right,
//           3 ring_rear, 4 ring_rear_left, 5 ring_rear_right

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

// Six distinct cameras in a bird's-eye 3-col x 2-row grid: the forward row on
// top, the rear row on the bottom, left cameras on the left. cam_1 (ring_front)
// is hidden (see HIDDEN_CAMS) because it points the same way as cam_0
// (base_front_center) and covers essentially the same FOV. With all six cells
// filled there is no free centre cell, so the ego readout tile is omitted.
//   (1,1) front-left    (1,2) front-center  (1,3) front-right
//   (2,1) rear-left     (2,2) rear          (2,3) rear-right
const KITSCENES_SEVEN_VIEW_RIG: Record<string, RigCam> = {
  cam_0: { label: "front-center", row: 1, col: 2 },
  cam_2: { label: "front-left", row: 1, col: 1 },
  cam_3: { label: "front-right", row: 1, col: 3 },
  cam_4: { label: "rear", row: 2, col: 2 },
  cam_5: { label: "rear-left", row: 2, col: 1 },
  cam_6: { label: "rear-right", row: 2, col: 3 },
};

const KITSCENES_SIX_VIEW_RIG: Record<string, RigCam> = {
  cam_0: { label: "front-center", row: 1, col: 2 },
  cam_1: { label: "front-left", row: 1, col: 1 },
  cam_2: { label: "front-right", row: 1, col: 3 },
  cam_3: { label: "rear", row: 2, col: 2 },
  cam_4: { label: "rear-left", row: 2, col: 1 },
  cam_5: { label: "rear-right", row: 2, col: 3 },
};

const RIGS: Record<string, Record<string, RigCam>> = {
  nvidia_av: NVIDIA_RIG,
  l2d: L2D_RIG,
};

function datasetRig(
  dataset: string,
  packedCameraCount?: number,
): Record<string, RigCam> | undefined {
  if (dataset === "kitscenes") {
    return packedCameraCount === 6
      ? KITSCENES_SIX_VIEW_RIG
      : KITSCENES_SEVEN_VIEW_RIG;
  }
  return RIGS[dataset];
}

// isHiddenCam reports whether a shard camera member should be omitted from the
// mosaic for a dataset. Filtering happens on the displayed camera list, so the
// hidden camera never claims a grid cell, a focus slot, or a keyboard ordinal.
export function isHiddenCam(
  dataset: string,
  cam: string,
  packedCameraCount?: number,
): boolean {
  // Seven-view shards contain the redundant ring-front in slot 1. Six-view
  // shards already removed it and compacted front-left into slot 1.
  return dataset === "kitscenes" && packedCameraCount !== 6 && cam === "cam_1";
}

// rigCam returns the rig position + grid cell for a "cam_N" identifier.
// Falls back to a sequential cell + the raw id for unknown datasets/cameras.
export function rigCam(
  dataset: string,
  cam: string,
  index: number,
  packedCameraCount?: number,
): RigCam {
  const mapped = datasetRig(dataset, packedCameraCount)?.[cam];
  if (mapped) return mapped;
  // Unknown rig: lay out sequentially in a 3-col grid.
  return { label: cam, row: Math.floor(index / 3) + 1, col: (index % 3) + 1 };
}

// camLabel returns just the rig position label (back-compat helper).
export function camLabel(
  dataset: string,
  cam: string,
  packedCameraCount?: number,
): string {
  return datasetRig(dataset, packedCameraCount)?.[cam]?.label ?? cam;
}

// displayAspectRatio is the width/height a camera tile should reserve for a
// dataset, so the frame matches the packed image and there is no letterbox gap.
// KITScenes cameras are packed 256x256 (1:1); the default 16:9 is kept for the
// other datasets to avoid changing their existing layout. The canvas still
// object-contain-fits the real bitmap, so a stray off-ratio image is letterboxed
// rather than stretched — this only sets the frame's shape.
const DATASET_ASPECT_RATIO: Record<string, number> = {
  kitscenes: 1,
};
export function displayAspectRatio(dataset: string): number {
  return DATASET_ASPECT_RATIO[dataset] ?? 16 / 9;
}

// gridDimensions returns the number of rows/cols spanned by a dataset's rig,
// so the mosaic can size its CSS grid. It is the exact extent of the cameras'
// placed cells (min 1x1) — NOT floored to 3x3. A rig that only spans 2 rows
// (KITScenes: forward + rear) must report rows=2, or the grid emits a phantom
// empty third row AND the ego cell lands in row 2 (ceil(3/2)) on top of a
// camera. `cams` should already be the DISPLAYED set (hidden cams filtered).
export function gridDimensions(
  dataset: string,
  cams: string[],
  packedCameraCount?: number,
): {
  rows: number;
  cols: number;
} {
  let rows = 1;
  let cols = 1;
  cams.forEach((cam, i) => {
    const c = rigCam(dataset, cam, i, packedCameraCount);
    rows = Math.max(rows, c.row);
    cols = Math.max(cols, c.col);
  });
  return { rows, cols };
}
