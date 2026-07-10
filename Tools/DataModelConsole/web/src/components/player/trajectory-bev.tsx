"use client";

// TrajectoryBEV: bird's-eye view of the ego future trajectory.
//
// The upcoming accel/curvature signals (the same channels stored in
// ego_future, read here from the shard index's per-frame ego_now so playback
// needs no extra fetches) are rolled out with the unicycle model into an XY
// path in the ego frame: up = forward (+x), left = +y. Metric grid included.

import { useMemo } from "react";

import { integrateTrajectory } from "@/lib/ego";
import type { IndexSample } from "@/types";

const SIZE = 300;
const HORIZON_STEPS = 64; // 6.4s at 10Hz
const GRID_M = 10;

export function TrajectoryBEV({
  samples,
  frame,
}: {
  samples: IndexSample[];
  frame: number;
}) {
  const traj = useMemo(() => {
    const now = samples[frame];
    if (!now) return [];
    const v0 = now.ego_now?.[0] ?? 0;
    const accel: number[] = [];
    const curvature: number[] = [];
    for (let i = 1; i <= HORIZON_STEPS; i++) {
      const s = samples[frame + i];
      if (!s) break;
      accel.push(s.ego_now?.[1] ?? 0);
      curvature.push(s.ego_now?.[3] ?? 0);
    }
    return integrateTrajectory(v0, accel, curvature);
  }, [samples, frame]);

  // Fit scale: at least 40m of forward view, expanded to cover the path.
  const extent = useMemo(() => {
    let m = 20;
    for (const p of traj) {
      m = Math.max(m, Math.abs(p.x), Math.abs(p.y));
    }
    return m * 1.15;
  }, [traj]);

  const scale = SIZE / 2 / extent;
  const cx = SIZE / 2;
  const cy = SIZE * 0.7; // ego sits below center: more room ahead
  // ego frame -> screen: up = +x (forward), left (+y) = screen left.
  const sx = (p: { x: number; y: number }) => cx - p.y * scale;
  const sy = (p: { x: number; y: number }) => cy - p.x * scale;

  const path = traj
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

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between font-mono text-[10px] text-slate-500">
        <span>BEV — future {(traj.length / 10).toFixed(1)}s</span>
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

        {/* trajectory */}
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
