import { expect, test } from "@playwright/test";

import golden from "./fixtures/integrator-golden.json";
import {
  integrateInterleavedControl,
  trajectoryCurvatureSign,
} from "../src/lib/ego";
import { egoTrajectoryToGeo } from "../src/lib/geo";
import {
  projectTrajectoryRibbonToCameras,
  projectTrajectoryToCameras,
  trajectoryGroundZMeters,
} from "../src/lib/projection";
import type { RigProjectionDocument } from "../src/types";

const KITSCENES_FRONT_MATRIX = [
  [128.75527954101562, -131.19908142089844, 1.0061875581741333, -52.390167236328125],
  [127.52409362792969, -0.9081462025642395, -249.18359375, -149.383544921875],
  [0.9999749660491943, 0.0070677390322089195, 0.0003683842078316957, -0.4213404357433319],
];

test("raw TypeScript rollout matches the Python evaluation integrator", () => {
  const points = integrateInterleavedControl(
    golden.v0,
    golden.controls,
    golden.dt,
    "raw",
    1,
  );

  expect(points).toHaveLength(golden.positions.length);
  points.forEach((point, index) => {
    expect(point.x).toBeCloseTo(golden.positions[index][0], 12);
    expect(point.y).toBeCloseTo(golden.positions[index][1], 12);
  });
});

test("L2D curvature correction mirrors lateral motion only", () => {
  expect(trajectoryCurvatureSign("l2d")).toBe(-1);
  expect(trajectoryCurvatureSign("kitscenes")).toBe(1);

  const canonical = integrateInterleavedControl(
    golden.v0,
    golden.controls,
    golden.dt,
    "raw",
    1,
  );
  const l2d = integrateInterleavedControl(
    golden.v0,
    golden.controls,
    golden.dt,
    "raw",
    -1,
  );
  canonical.forEach((point, index) => {
    expect(l2d[index].x).toBeCloseTo(point.x, 12);
    expect(l2d[index].y).toBeCloseTo(-point.y, 12);
    expect(l2d[index].heading).toBeCloseTo(-point.heading, 12);
  });
});

test("north-facing map placement sends ego-left west", () => {
  const origin = { latitude: 49, longitude: 11 };
  const [forward, left] = egoTrajectoryToGeo(origin, 0, [
    { x: 10, y: 0, heading: 0 },
    { x: 0, y: 10, heading: 0 },
  ]);

  expect(forward.latitude).toBeGreaterThan(origin.latitude);
  expect(forward.longitude).toBeCloseTo(origin.longitude, 12);
  expect(left.latitude).toBeCloseTo(origin.latitude, 12);
  expect(left.longitude).toBeLessThan(origin.longitude);
});

test("pinhole rig projects ego points into normalized camera pixels", () => {
  const rig: RigProjectionDocument = {
    schema_version: "v1",
    dataset: "kitscenes",
    geometry_type: "pinhole",
    image_size: 256,
    projection: {
      type: "pinhole",
      matrix: [
        [
          [0, -100, 0, 128],
          [-20, 0, 0, 200],
          [0, 0, 0, 1],
        ],
      ],
    },
  };
  const result = projectTrajectoryToCameras(rig, [
    { x: 1, y: 0, heading: 0 },
    { x: 5, y: 1, heading: 0 },
  ]);

  expect(result.cam_0).toHaveLength(1);
  expect(result.cam_0[0]).toHaveLength(2);
  expect(result.cam_0[0][0].u).toBeCloseTo(0.5, 12);
  expect(result.cam_0[0][0].v).toBeCloseTo(180 / 256, 12);
  expect(result.cam_0[0][1].u).toBeCloseTo(28 / 256, 12);
  expect(result.cam_0[0][1].v).toBeCloseTo(100 / 256, 12);
});

test("production KITScenes calibration projects ribbons onto the ground", () => {
  const rig: RigProjectionDocument = {
    schema_version: "v1",
    dataset: "KIT-MRT/KITScenes-Multimodal",
    geometry_type: "pinhole",
    image_size: 256,
    projection: {
      type: "pinhole",
      matrix: [KITSCENES_FRONT_MATRIX],
    },
  };
  const trajectory = [
    { x: 5, y: 0, heading: 0 },
    { x: 10, y: 0, heading: 0 },
  ];

  expect(trajectoryGroundZMeters(rig)).toBe(-2.1);
  const center = projectTrajectoryToCameras(rig, trajectory).cam_0[0];
  expect(center[0].v * 256).toBeCloseTo(220.99, 1);
  expect(center[1].v * 256).toBeCloseTo(172.2, 1);

  const ribbon = projectTrajectoryRibbonToCameras(rig, trajectory).cam_0[0];
  expect(ribbon.left).toHaveLength(2);
  expect(ribbon.right).toHaveLength(2);
  expect(ribbon.left[0].u).toBeLessThan(ribbon.right[0].u);
});

test("f-theta rig preserves the ego-FLU to optical-axis convention", () => {
  const rig: RigProjectionDocument = {
    schema_version: "v1",
    dataset: "nvidia_av",
    geometry_type: "ftheta",
    projection: {
      type: "ftheta",
      t_camera_ego: [
        [
          [0, -1, 0, 0],
          [0, 0, -1, 0],
          [1, 0, 0, 0],
          [0, 0, 0, 1],
        ],
      ],
      fw_poly: [0, 200],
      cx: [128],
      cy: [128],
      image_wh: [[256, 256]],
      max_theta: [1.8],
    },
  };
  const result = projectTrajectoryToCameras(rig, [
    { x: 5, y: 0, heading: 0 },
    { x: 5, y: 1, heading: 0 },
  ]);

  expect(result.cam_0[0][0].u).toBeCloseTo(0.5, 12);
  expect(result.cam_0[0][0].v).toBeCloseTo(0.5, 12);
  expect(result.cam_0[0][1].u).toBeLessThan(0.5);
});

test("pseudo geometry never claims a camera-space trajectory", () => {
  const rig: RigProjectionDocument = {
    schema_version: "v1",
    dataset: "l2d",
    geometry_type: "pseudo",
    projection: null,
  };
  expect(
    projectTrajectoryToCameras(rig, [
      { x: 1, y: 0, heading: 0 },
      { x: 2, y: 0, heading: 0 },
    ]),
  ).toEqual({});
});
