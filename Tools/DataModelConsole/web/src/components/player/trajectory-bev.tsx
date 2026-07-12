"use client";

// TrajectoryBEV: bird's-eye view of the ego trajectory.
//
// NOTE: both curves are recorded GROUND TRUTH, not a model prediction — there
// is no planner in the console. The blue curve is the dataset's stored
// ego_future for this frame (128 floats = 64 steps x [accel, curvature]) rolled
// out with the unicycle model; the amber curve is the SAME drive reconstructed
// from the ego_now speed+curvature of the following frames. They are two views
// of the same logged motion (a self-consistency check), so the legend says
// "recorded future" / "driven path", NOT "plan" / "actual".
//
// Path is in the ego frame: up = forward (+x), left = +y. The full 6.4s future
// is stored on every sample, so it renders at full length regardless of how
// many frames remain in the shard. Metric grid included.

import { useMemo } from "react";

import {
  clamp,
  decodeEgo,
  integrateTrajectory,
  MAX_CURVATURE,
  MIN_SPEED_FOR_HEADING,
} from "@/lib/ego";
import type { TrajectoryPoint } from "@/lib/ego";
import type { IndexSample, ReasoningLabelRecord } from "@/types";

const SIZE = 300;
const GRID_M = 10;

export function TrajectoryBEV({
  samples,
  frame,
  fps = 10,
  reasoning,
}: {
  samples: IndexSample[];
  frame: number;
  fps?: number;
  // Reasoning label for the current frame (when loaded). Its horizons are
  // pinned onto the plan path at the matching rollout step so the reasoning
  // cards and the geometry line up in time.
  reasoning?: ReasoningLabelRecord | null;
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

  // Realized path: chain the chronological ego_now (speed + curvature) of the
  // frames from here forward into an XY path — what the ego actually drove.
  // Clipped to the plan horizon (traj.length frames) so blue (plan) and amber
  // (actual) cover the same window and are directly comparable; how much of the
  // plan the remaining shard can actually verify is reported as text instead.
  // Heading is integrated from curvature (channel 3, v*kappa = yaw_rate) which
  // is dataset-agnostic and identical to integrateTrajectory; curvature is
  // clamped and heading gated by speed to reject non-physical outlier spikes.
  const realized = useMemo(() => {
    const dt = 1 / (fps || 10);
    const end = Math.min(samples.length, frame + traj.length);
    const pts: TrajectoryPoint[] = [];
    let x = 0;
    let y = 0;
    let theta = 0;
    for (let i = frame; i < end; i++) {
      const v = samples[i].ego_now?.[0] ?? 0;
      const kappa = samples[i].ego_now?.[3] ?? 0;
      if (v >= MIN_SPEED_FOR_HEADING) theta += v * clamp(kappa, MAX_CURVATURE) * dt;
      x += v * Math.cos(theta) * dt;
      y += v * Math.sin(theta) * dt;
      pts.push({ x, y, heading: theta });
    }
    return pts;
  }, [samples, frame, fps, traj.length]);

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
      const kappa = h.curvature[i] ?? 0;
      // Reverse integration: undo one step of the unicycle model. Integrate
      // heading from curvature (v*kappa = yaw_rate, dataset-agnostic); clamp
      // curvature and gate heading by speed to reject non-physical outliers.
      x -= v * Math.cos(theta) * dt;
      y -= v * Math.sin(theta) * dt;
      if (v >= MIN_SPEED_FOR_HEADING) theta -= v * clamp(kappa, MAX_CURVATURE) * dt;
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
    // Robust ceiling: allow a real highway plan (v*6.4s ~100m+) on-canvas, but a
    // lone non-physical outlier still can't blow the grid out past 150m.
    return Math.min(Math.max(m * 1.15, 20), 150);
  }, [samples]);

  const scale = SIZE / 2 / extent;
  const cx = SIZE / 2;
  const cy = SIZE * 0.82; // ego sits low: more room ahead for long plans
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

  // Reasoning-horizon dots pinned onto the plan path. traj[i] is the pose AFTER
  // i+1 integration steps (traj[0] = pose at t=+dt), so a horizon at t seconds
  // (step = round(t*fps)) sits at traj[step-1]; t=0 is the ego origin. Using
  // traj[step] would place every dot one 10Hz step (+0.1s) too far ahead.
  const horizonDots = useMemo(() => {
    if (!reasoning?.horizons?.length) return [];
    const out: { x: number; y: number; sec: number }[] = [];
    for (const h of reasoning.horizons) {
      const step = Math.round(h.horizon_sec * (fps || 10));
      const p = step === 0 ? { x: 0, y: 0 } : traj[step - 1];
      if (!p) continue;
      out.push({ x: sx(p), y: sy(p), sec: h.horizon_sec });
    }
    return out;
    // sx/sy derive from scale/cx/cy; traj + reasoning + fps drive the result.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reasoning, traj, fps, scale]);

  const speed = samples[frame]?.ego_now?.[0] ?? 0;
  // planSec is the plan horizon; the amber realized path is clipped to the same
  // horizon so plan and actual are directly comparable. coveredSec is how much
  // of that horizon the remaining frames actually verify (< planSec near the
  // end of the shard).
  const planSec = traj.length / (fps || 10);
  const coveredSec = realized.length / (fps || 10);
  const partial = realized.length < traj.length;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between font-mono text-[10px] text-slate-500">
        <span>
          BEV — <span className="text-slate-400">past</span> ·{" "}
          <span className="text-blue-500">recorded future</span> /{" "}
          <span className="text-amber-500">driven path</span> first{" "}
          {planSec.toFixed(1)}s
          {partial && (
            <span className="text-slate-600">
              {" "}
              (only {coveredSec.toFixed(1)}s left in shard)
            </span>
          )}
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

        {/* reasoning horizons pinned onto the plan path (t+Ns dots) */}
        {horizonDots.map((d) => (
          <g key={d.sec}>
            <circle
              cx={d.x}
              cy={d.y}
              r="3.5"
              fill="#a78bfa"
              stroke="#0f172a"
              strokeWidth="1"
            />
            <text
              x={d.x + 5}
              y={d.y + 3}
              fill="#c4b5fd"
              fontSize="8"
              fontFamily="monospace"
            >
              +{d.sec}s
            </text>
          </g>
        ))}

        {/* ego marker (triangle pointing forward/up) */}
        <polygon
          points={`${cx},${cy - 7} ${cx - 5},${cy + 5} ${cx + 5},${cy + 5}`}
          fill="#f8fafc"
        />
      </svg>
    </div>
  );
}
