import { expect, test } from "@playwright/test";

import {
  camLabel,
  displayAspectRatio,
  gridDimensions,
  isHiddenCam,
  rigCam,
} from "../src/lib/rig";

// The camera members packed in a KITScenes shard (cam_0..cam_6), and the set
// the console displays after hiding the redundant ring-front (cam_1).
const KIT_PACKED = [
  "cam_0",
  "cam_1",
  "cam_2",
  "cam_3",
  "cam_4",
  "cam_5",
  "cam_6",
];
const kitDisplayed = KIT_PACKED.filter((c) => !isHiddenCam("kitscenes", c));

test("KITScenes hides only the redundant ring-front (cam_1)", () => {
  expect(isHiddenCam("kitscenes", "cam_1")).toBe(true);
  expect(kitDisplayed).toEqual([
    "cam_0",
    "cam_2",
    "cam_3",
    "cam_4",
    "cam_5",
    "cam_6",
  ]);
  // Other datasets hide nothing.
  expect(isHiddenCam("nvidia_av", "cam_1")).toBe(false);
  expect(isHiddenCam("l2d", "cam_1")).toBe(false);
});

test("KITScenes displayed cameras form a filled 3x2 grid (no gaps, no ego cell)", () => {
  const { rows, cols } = gridDimensions("kitscenes", kitDisplayed);
  expect({ rows, cols }).toEqual({ rows: 2, cols: 3 });

  // Every one of the 6 cells is claimed exactly once by a displayed camera.
  const cells = kitDisplayed.map((cam, i) => {
    const c = rigCam("kitscenes", cam, i);
    return `${c.row}:${c.col}`;
  });
  expect(new Set(cells).size).toBe(6);
  expect([...cells].sort()).toEqual([
    "1:1",
    "1:2",
    "1:3",
    "2:1",
    "2:2",
    "2:3",
  ]);

  // The centre cell the ego tile would occupy — ceil(rows/2)=1, ceil(cols/2)=2 —
  // is claimed by front-center, so the mosaic's egoInFreeCell check drops the
  // ego tile (the car icon).
  const egoCell = `${Math.ceil(rows / 2)}:${Math.ceil(cols / 2)}`;
  expect(cells).toContain(egoCell);
});

test("forward cameras are on the top row, rear on the bottom, left-to-right", () => {
  const at = (cam: string) => rigCam("kitscenes", cam, 0);
  expect(at("cam_2")).toMatchObject({ label: "front-left", row: 1, col: 1 });
  expect(at("cam_0")).toMatchObject({ label: "front-center", row: 1, col: 2 });
  expect(at("cam_3")).toMatchObject({ label: "front-right", row: 1, col: 3 });
  expect(at("cam_5")).toMatchObject({ label: "rear-left", row: 2, col: 1 });
  expect(at("cam_4")).toMatchObject({ label: "rear", row: 2, col: 2 });
  expect(at("cam_6")).toMatchObject({ label: "rear-right", row: 2, col: 3 });
});

test("NVIDIA and L2D rigs are unchanged (still 3x3, ego cell free)", () => {
  const nvidia = ["cam_0", "cam_1", "cam_2", "cam_3", "cam_4", "cam_5", "cam_6"];
  expect(gridDimensions("nvidia_av", nvidia)).toEqual({ rows: 3, cols: 3 });
  // NVIDIA centre (2,2) is the ego cell and no camera claims it.
  const nvidiaCells = nvidia.map((cam, i) => {
    const c = rigCam("nvidia_av", cam, i);
    return `${c.row}:${c.col}`;
  });
  expect(nvidiaCells).not.toContain("2:2");

  const l2d = ["cam_0", "cam_1", "cam_2", "cam_3", "cam_4", "cam_5"];
  expect(gridDimensions("l2d", l2d)).toEqual({ rows: 3, cols: 3 });
});

test("displayAspectRatio matches the packed image shape per dataset", () => {
  // KITScenes cameras are packed 256x256 (1:1).
  expect(displayAspectRatio("kitscenes")).toBe(1);
  // Others keep the 16:9 default (unchanged layout).
  expect(displayAspectRatio("nvidia_av")).toBeCloseTo(16 / 9, 6);
  expect(displayAspectRatio("l2d")).toBeCloseTo(16 / 9, 6);
  expect(displayAspectRatio("unknown")).toBeCloseTo(16 / 9, 6);
});

test("unknown datasets fall back to sequential placement and raw labels", () => {
  expect(rigCam("mystery", "cam_0", 0)).toMatchObject({
    label: "cam_0",
    row: 1,
    col: 1,
  });
  expect(rigCam("mystery", "cam_4", 4)).toMatchObject({ row: 2, col: 2 });
  expect(camLabel("mystery", "cam_0")).toBe("cam_0");
});
