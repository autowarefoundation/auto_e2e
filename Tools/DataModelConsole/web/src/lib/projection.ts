import type { TrajectoryPoint } from "@/lib/ego";
import type { RigProjectionDocument } from "@/types";

const DEPTH_EPS = 1e-5;

export interface ScreenPoint {
  u: number;
  v: number;
}

export type CameraProjectionPaths = Record<string, ScreenPoint[][]>;

type Matrix = number[][];

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : fallback;
}

function perViewValue(value: unknown, view: number, fallback = 0): number {
  if (Array.isArray(value)) return asNumber(value[view], fallback);
  return asNumber(value, fallback);
}

function perViewPair(
  value: unknown,
  view: number,
  fallback: [number, number],
): [number, number] {
  if (
    Array.isArray(value) &&
    value.length === 2 &&
    typeof value[0] === "number"
  ) {
    return [asNumber(value[0], fallback[0]), asNumber(value[1], fallback[1])];
  }
  if (Array.isArray(value) && Array.isArray(value[view])) {
    const pair = value[view] as unknown[];
    return [asNumber(pair[0], fallback[0]), asNumber(pair[1], fallback[1])];
  }
  return fallback;
}

function multiplyPoint(matrix: Matrix, point: number[]): number[] {
  return matrix.map((row) =>
    row.reduce((sum, coefficient, index) => sum + coefficient * point[index], 0),
  );
}

function polynomial(coefficients: number[], theta: number): number {
  let result = 0;
  for (let i = coefficients.length - 1; i >= 0; i--) {
    result = result * theta + coefficients[i];
  }
  return result;
}

function pushProjected(
  paths: ScreenPoint[][],
  point: ScreenPoint | null,
): void {
  if (!point) {
    if (paths.at(-1)?.length === 0) paths.pop();
    paths.push([]);
    return;
  }
  if (paths.length === 0) paths.push([]);
  paths[paths.length - 1].push(point);
}

function pinholePoint(
  matrix: Matrix,
  point: TrajectoryPoint,
  imageSize: [number, number],
): ScreenPoint | null {
  const projected = multiplyPoint(matrix, [point.x, point.y, 0, 1]);
  const depth = projected[2];
  if (!Number.isFinite(depth) || depth <= DEPTH_EPS) return null;
  const u = projected[0] / depth / imageSize[0];
  const v = projected[1] / depth / imageSize[1];
  return u >= 0 && u <= 1 && v >= 0 && v <= 1 ? { u, v } : null;
}

function fthetaPoint(
  spec: Record<string, unknown>,
  view: number,
  point: TrajectoryPoint,
): ScreenPoint | null {
  const transforms = spec.t_camera_ego as Matrix[] | undefined;
  const transform = transforms?.[view];
  if (!transform) return null;
  const [x, y, z] = multiplyPoint(transform, [point.x, point.y, 0, 1]);
  const rho = Math.max(Math.hypot(x, y), DEPTH_EPS);
  const theta = Math.atan2(rho, z);
  const maxTheta = perViewValue(spec.max_theta, view, Number.NaN);
  if (
    (Number.isFinite(maxTheta) && theta > maxTheta) ||
    (!Number.isFinite(maxTheta) && z <= DEPTH_EPS)
  ) {
    return null;
  }

  const rawPoly = spec.fw_poly;
  const coefficients =
    Array.isArray(rawPoly) && Array.isArray(rawPoly[0])
      ? (rawPoly[view] as number[])
      : (rawPoly as number[] | undefined);
  if (!coefficients?.length) return null;
  const radius = polynomial(coefficients, theta);
  const cx = perViewValue(spec.cx, view);
  const cy = perViewValue(spec.cy, view);
  const [width, height] = perViewPair(spec.image_wh, view, [256, 256]);
  const u = (cx + radius * (x / rho)) / width;
  const v = (cy + radius * (y / rho)) / height;
  return u >= 0 && u <= 1 && v >= 0 && v <= 1 ? { u, v } : null;
}

export function projectTrajectoryToCameras(
  rig: RigProjectionDocument | null,
  trajectory: TrajectoryPoint[],
): CameraProjectionPaths {
  const spec = rig?.projection;
  if (!rig || !spec || rig.geometry_type === "pseudo") return {};
  const type = String(spec.type ?? rig.geometry_type);
  const matrices = spec.matrix as Matrix[] | undefined;
  const transforms = spec.t_camera_ego as Matrix[] | undefined;
  const views =
    type === "ftheta" ? (transforms?.length ?? 0) : (matrices?.length ?? 0);
  const size =
    typeof rig.image_size === "number"
      ? ([rig.image_size, rig.image_size] as [number, number])
      : (rig.image_size ?? [256, 256]);

  const result: CameraProjectionPaths = {};
  for (let view = 0; view < views; view++) {
    const paths: ScreenPoint[][] = [[]];
    for (const point of trajectory) {
      const projected =
        type === "ftheta"
          ? fthetaPoint(spec, view, point)
          : matrices?.[view]
            ? pinholePoint(matrices[view], point, size)
            : null;
      pushProjected(paths, projected);
    }
    result[`cam_${view}`] = paths.filter((path) => path.length >= 2);
  }
  return result;
}

export function projectTrajectoriesToCameras(
  rig: RigProjectionDocument | null,
  trajectories: TrajectoryPoint[][],
): CameraProjectionPaths {
  const result: CameraProjectionPaths = {};
  for (const trajectory of trajectories) {
    const projected = projectTrajectoryToCameras(rig, trajectory);
    for (const [camera, paths] of Object.entries(projected)) {
      (result[camera] ??= []).push(...paths);
    }
  }
  return result;
}
