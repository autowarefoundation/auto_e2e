// Decode and integrate ego.npy signals.
//
// ego_history: 256 floats = 64 timesteps x [speed, accel, yaw_rate, curvature]
//   (interleaved, stride 4), covering t = -6.4s .. 0 at 10Hz.
// ego_future: 128 floats = 64 timesteps x [accel, curvature]
//   (interleaved, stride 2), covering t = 0 .. +6.4s at 10Hz.

export const EGO_STEPS = 64;
export const EGO_DT = 0.1; // 10Hz

export interface EgoHistory {
  speed: number[]; // m/s
  accel: number[]; // m/s^2
  yawRate: number[]; // rad/s
  curvature: number[]; // 1/m
}

export interface EgoFuture {
  accel: number[]; // m/s^2
  curvature: number[]; // 1/m
}

export interface DecodedEgo {
  history: EgoHistory;
  future: EgoFuture;
}

function deinterleave(
  values: number[],
  stride: number,
  channel: number,
): number[] {
  const steps = Math.floor(values.length / stride);
  const out = new Array<number>(steps);
  for (let i = 0; i < steps; i++) {
    out[i] = values[i * stride + channel];
  }
  return out;
}

// decodeEgo splits the flattened arrays into per-channel series.
export function decodeEgo(history: number[], future: number[]): DecodedEgo {
  return {
    history: {
      speed: deinterleave(history, 4, 0),
      accel: deinterleave(history, 4, 1),
      yawRate: deinterleave(history, 4, 2),
      curvature: deinterleave(history, 4, 3),
    },
    future: {
      accel: deinterleave(future, 2, 0),
      curvature: deinterleave(future, 2, 1),
    },
  };
}

export interface TrajectoryPoint {
  x: number; // m, +x forward (ego frame)
  y: number; // m, +y left
  heading: number; // rad
}

// integrateTrajectory rolls the (accel, curvature) future out with a
// unicycle model from initial speed v0, in the ego frame (+x forward):
//   v += a*dt; theta += v*kappa*dt; x += v*cos(theta)*dt; y += v*sin(theta)*dt
export function integrateTrajectory(
  v0: number,
  accel: number[],
  curvature: number[],
  dt = EGO_DT,
): TrajectoryPoint[] {
  const n = Math.min(accel.length, curvature.length);
  const out: TrajectoryPoint[] = new Array(n);
  let v = v0;
  let theta = 0;
  let x = 0;
  let y = 0;
  for (let i = 0; i < n; i++) {
    v += accel[i] * dt;
    if (v < 0) v = 0; // no reversing from braking overshoot
    theta += v * curvature[i] * dt;
    x += v * Math.cos(theta) * dt;
    y += v * Math.sin(theta) * dt;
    out[i] = { x, y, heading: theta };
  }
  return out;
}
