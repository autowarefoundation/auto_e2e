"use client";

// TrajectoryBEV: bird's-eye view of the ego future trajectory.
//
// The per-frame ego_future plan (128 floats = 64 steps x [accel, curvature])
// is rolled out with the unicycle model into an XY path in the ego frame:
// up = forward (+x), left = +y. Because the full 6.4s plan is stored on every
// sample, the trajectory renders at full length regardless of how many frames
// remain in the shard. Metric grid included.

import { useMemo } from "react";

import { decodeEgo, integrateTrajectory } from "@/lib/ego";
import type { TrajectoryPoint } from "@/lib/ego";
import type { IndexSample } from "@/types";

const SIZE = 300;
const GRID_M = 10;

export function TrajectoryBEV({
  samples,
  frame,
  fps = 10,
}: {
  samples: IndexSample[];
  frame: number;
  fps?: number;
}) {
  const traj = useMemo(() => {
    const now = samples[frame];
    if (!now?.ego_future?.length) return [];
    const { future } = decodeEgo([], now.ego_future);
    return integrateTrajectory(
      now.ego_now?.[0] ?? 0,
      future.accel,
      future.curvature,
    );
  }, [samples, frame]);

  // Realized path: chain the chronological ego_now (speed + yaw_rate) of the
  // frames from here to the end of the shard into an XY path. This is what the
  // ego actually drove vs the plan; it is short when few frames remain, which
  // is itself the honest signal about how much of the plan is verifiable.
  const realized = useMemo(() => {
    const dt = 1 / (fps || 10);
    const pts: TrajectoryPoint[] = [];
    let x = 0;
    let y = 0;
    let theta = 0;
    for (let i = frame; i < samples.length; i++) {
      const v = samples[i].ego_now?.[0] ?? 0;
      const yawRate = samples[i].ego_now?.[2] ?? 0;
      theta += yawRate * dt;
      x += v * Math.cos(theta) * dt;
      y += v * Math.sin(theta) * dt;
      pts.push({ x, y, heading: theta });
    }
    return pts;
  }, [samples, frame, fps]);

  // Past trajectory: integrate the 256-float ego_history (64 steps x [speed,
  // accel, yaw_rate, curvature]) BACKWARDS from the ego marker, so the trailing
  // driven path is visible mid-clip even without cross-shard stitching. We walk
  // the history newest→oldest and step in reverse (subtract the per-step
  // motion), which places recent history nearest the ego.
  const history = useMemo(() => {
    const now = samples[frame];
    if (!now?.ego_history?.length) return [];
    const { history: h } = decodeEgo(now.ego_history, []);
    const dt = 1 / (fps || 10);
    const pts: TrajectoryPoint[] = [];
    let x = 0;
    let y = 0;
    let theta = 0;
    for (let i = h.speed.length - 1; i >= 0; i--) {
      const v = h.speed[i] ?? 0;
      const yawRate = h.yawRate[i] ?? 0;
      // Reverse integration: undo one step of the unicycle model.
      x -= v * Math.cos(theta) * dt;
      y -= v * Math.sin(theta) * dt;
      theta -= yawRate * dt;
      pts.push({ x, y, heading: theta });
    }
    return pts;
  }, [samples, frame, fps]);

  // Fit scale over ALL samples' plans so the metric window is stable across
  // frames (only the plotted path moves, not the gridlines). At least 20m.
  const extent = useMemo(() => {
    let m = 20;
    for (const s of samples) {
      if (!s.ego_future?.length) continue;
      const { future } = decodeEgo([], s.ego_future);
      const rolled = integrateTrajectory(
        s.ego_now?.[0] ?? 0,
        future.accel,
        future.curvature,
      );
      for (const p of rolled) {
        m = Math.max(m, Math.abs(p.x), Math.abs(p.y));
      }
    }
    return m * 1.15;
  }, [samples]);

  const scale = SIZE / 2 / extent;
  const cx = SIZE / 2;
  const cy = SIZE * 0.7; // ego sits below center: more room ahead
  // ego frame -> screen: up = +x (forward), left (+y) = screen left.
  const sx = (p: { x: number; y: number }) => cx - p.y * scale;
  const sy = (p: { x: number; y: number }) => cy - p.x * scale;

  const path = traj
    .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p).toFixed(1)},${sy(p).toFixed(1)}`)
    .join(" ");

  const realizedPath = realized
    .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p).toFixed(1)},${sy(p).toFixed(1)}`)
    .join(" ");

  const historyPath = history
    .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p).toFixed(1)},${sy(p).toFixed(1)}`)
    .join(" ");

  const gridLines = useMemo(() => {
    const out: { x1: number; y1: number; x2: number; y2: number; label?: string }[] =
      [];
    const maxM = Math.ceil(extent / GRID_M) * GRID_M;
    for (let m = -maxM; m <= maxM + 1e-6; m += GRID_M) {
      // lines of constant forward distance (horizontal on screen)
      const y = cy - m * scale;
      if (y >= 0 && y <= SIZE) {
        out.push({ x1: 0, y1: y, x2: SIZE, y2: y, label: m !== 0 ? `${m}m` : undefined });
      }
      // lines of constant lateral offset (vertical on screen)
      const x = cx - m * scale;
      if (x >= 0 && x <= SIZE) {
        out.push({ x1: x, y1: 0, x2: x, y2: SIZE });
      }
    }
    return out;
  }, [extent, scale, cx, cy]);

  const speed = samples[frame]?.ego_now?.[0] ?? 0;
  // How much of the 6.4s plan is covered by remaining frames (i.e. verifiable
  // against the realized path). realized has one point per remaining frame.
  const planSec = traj.length / (fps || 10);
  const actualSec = Math.max(0, realized.length - 1) / (fps || 10);

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between font-mono text-[10px] text-slate-500">
        <span>
          BEV — <span className="text-slate-400">past</span> ·{" "}
          <span className="text-blue-500">plan</span> {planSec.toFixed(1)}s /{" "}
          <span className="text-amber-500">actual</span>{" "}
          {actualSec.toFixed(1)}s
        </span>
        <span>v = {speed.toFixed(1)} m/s</span>
      </div>
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="aspect-square w-full rounded-md border border-slate-800 bg-slate-900/60"
        role="img"
        aria-label="Bird's-eye view of future trajectory"
      >
        {gridLines.map((l, i) => (
          <g key={i}>
            <line
              x1={l.x1}
              y1={l.y1}
              x2={l.x2}
              y2={l.y2}
              stroke="#1e293b"
              strokeWidth="1"
            />
            {l.label && (
              <text
                x={4}
                y={l.y1 - 2}
                fill="#475569"
                fontSize="8"
                fontFamily="monospace"
              >
                {l.label}
              </text>
            )}
          </g>
        ))}

        {/* past trajectory: dashed slate, trailing behind the ego */}
        {historyPath && (
          <path
            d={historyPath}
            fill="none"
            stroke="#64748b"
            strokeWidth="1.5"
            strokeDasharray="2 3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {/* realized (driven) path: dashed amber, under the plan */}
        {realizedPath && (
          <path
            d={realizedPath}
            fill="none"
            stroke="#f59e0b"
            strokeWidth="2"
            strokeDasharray="4 3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {/* planned trajectory */}
        {path && (
          <path
            d={path}
            fill="none"
            stroke="#3b82f6"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
        {traj.length > 0 && (
          <circle
            cx={sx(traj[traj.length - 1])}
            cy={sy(traj[traj.length - 1])}
            r="3"
            fill="#3b82f6"
          />
        )}

        {/* ego marker (triangle pointing forward/up) */}
        <polygon
          points={`${cx},${cy - 7} ${cx - 5},${cy + 5} ${cx + 5},${cy + 5}`}
          fill="#f8fafc"
        />
      </svg>
    </div>
  );
}
