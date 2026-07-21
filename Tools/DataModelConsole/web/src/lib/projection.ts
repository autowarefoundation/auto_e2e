import type { TrajectoryPoint } from "@/lib/ego";
import type { RigProjectionDocument } from "@/types";

const DEPTH_EPS = 1e-5;
const TANGENT_EPS = 1e-6;
const KITSCENES_GROUND_Z_M = -2.1;
export const TRAJECTORY_RIBBON_WIDTH_M = 1.8;

export interface ScreenPoint {
  u: number;
  v: number;
}

export type CameraProjectionPaths = Record<string, ScreenPoint[][]>;

export interface ScreenRibbon {
  left: ScreenPoint[];
  right: ScreenPoint[];
}

export type CameraProjectionRibbons = Record<string, ScreenRibbon[]>;

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

function normalizedScreenPoint(
  u: number,
  v: number,
  clipToImage: boolean,
): ScreenPoint | null {
  if (!Number.isFinite(u) || !Number.isFinite(v)) return null;
  if (clipToImage && (u < 0 || u > 1 || v < 0 || v > 1)) return null;
  return { u, v };
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
  groundZ: number,
  imageSize: [number, number],
  clipToImage: boolean,
): ScreenPoint | null {
  const projected = multiplyPoint(matrix, [point.x, point.y, groundZ, 1]);
  const depth = projected[2];
  if (!Number.isFinite(depth) || depth <= DEPTH_EPS) return null;
  const u = projected[0] / depth / imageSize[0];
  const v = projected[1] / depth / imageSize[1];
  return normalizedScreenPoint(u, v, clipToImage);
}

function fthetaPoint(
  spec: Record<string, unknown>,
  view: number,
  point: TrajectoryPoint,
  groundZ: number,
  clipToImage: boolean,
): ScreenPoint | null {
  const transforms = spec.t_camera_ego as Matrix[] | undefined;
  const transform = transforms?.[view];
  if (!transform) return null;
  const [x, y, z] = multiplyPoint(transform, [
    point.x,
    point.y,
    groundZ,
    1,
  ]);
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
  return normalizedScreenPoint(u, v, clipToImage);
}

export function trajectoryGroundZMeters(
  rig: RigProjectionDocument | null,
): number {
  const explicit = rig?.projection?.ground_z_m;
  if (typeof explicit === "number" && Number.isFinite(explicit)) {
    return explicit;
  }
  // Publication v2.2 predates ground_z_m. Its matrices still use KITScenes'
  // top-LiDAR FLU reference frame, whose measured ground plane is z=-2.1 m.
  return rig?.dataset.toLowerCase().includes("kitscenes")
    ? KITSCENES_GROUND_Z_M
    : 0;
}

function projectPoint(
  type: string,
  spec: Record<string, unknown>,
  matrices: Matrix[] | undefined,
  view: number,
  point: TrajectoryPoint,
  groundZ: number,
  imageSize: [number, number],
  clipToImage: boolean,
): ScreenPoint | null {
  if (type === "ftheta") {
    return fthetaPoint(spec, view, point, groundZ, clipToImage);
  }
  return matrices?.[view]
    ? pinholePoint(
        matrices[view],
        point,
        groundZ,
        imageSize,
        clipToImage,
      )
    : null;
}

function trajectoryBoundaries(
  trajectory: TrajectoryPoint[],
  widthM: number,
): Array<{ left: TrajectoryPoint; right: TrajectoryPoint }> {
  if (trajectory.length === 0 || !Number.isFinite(widthM) || widthM <= 0) {
    return [];
  }
  const points = [{ x: 0, y: 0, heading: 0 }, ...trajectory];
  const halfWidth = widthM / 2;
  return points.map((point, index) => {
    const previous = points[Math.max(0, index - 1)];
    const next = points[Math.min(points.length - 1, index + 1)];
    let dx = next.x - previous.x;
    let dy = next.y - previous.y;
    let norm = Math.hypot(dx, dy);
    if (norm < TANGENT_EPS) {
      dx = Math.cos(point.heading);
      dy = Math.sin(point.heading);
      norm = 1;
    }
    const leftX = (-dy / norm) * halfWidth;
    const leftY = (dx / norm) * halfWidth;
    return {
      left: {
        x: point.x + leftX,
        y: point.y + leftY,
        heading: point.heading,
      },
      right: {
        x: point.x - leftX,
        y: point.y - leftY,
        heading: point.heading,
      },
    };
  });
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
  const groundZ = trajectoryGroundZMeters(rig);
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
      const projected = projectPoint(
        type,
        spec,
        matrices,
        view,
        point,
        groundZ,
        size,
        true,
      );
      pushProjected(paths, projected);
    }
    result[`cam_${view}`] = paths.filter((path) => path.length >= 2);
  }
  return result;
}

export function projectTrajectoryRibbonToCameras(
  rig: RigProjectionDocument | null,
  trajectory: TrajectoryPoint[],
  widthM = TRAJECTORY_RIBBON_WIDTH_M,
): CameraProjectionRibbons {
  const spec = rig?.projection;
  if (!rig || !spec || rig.geometry_type === "pseudo") return {};
  const type = String(spec.type ?? rig.geometry_type);
  const matrices = spec.matrix as Matrix[] | undefined;
  const transforms = spec.t_camera_ego as Matrix[] | undefined;
  const groundZ = trajectoryGroundZMeters(rig);
  const views =
    type === "ftheta" ? (transforms?.length ?? 0) : (matrices?.length ?? 0);
  const size =
    typeof rig.image_size === "number"
      ? ([rig.image_size, rig.image_size] as [number, number])
      : (rig.image_size ?? [256, 256]);
  const boundaries = trajectoryBoundaries(trajectory, widthM);

  const result: CameraProjectionRibbons = {};
  for (let view = 0; view < views; view++) {
    const ribbons: ScreenRibbon[] = [];
    let current: ScreenRibbon = { left: [], right: [] };
    for (const boundary of boundaries) {
      const left = projectPoint(
        type,
        spec,
        matrices,
        view,
        boundary.left,
        groundZ,
        size,
        false,
      );
      const right = projectPoint(
        type,
        spec,
        matrices,
        view,
        boundary.right,
        groundZ,
        size,
        false,
      );
      if (!left || !right) {
        if (current.left.length >= 2) ribbons.push(current);
        current = { left: [], right: [] };
        continue;
      }
      current.left.push(left);
      current.right.push(right);
    }
    if (current.left.length >= 2) ribbons.push(current);
    result[`cam_${view}`] = ribbons;
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
