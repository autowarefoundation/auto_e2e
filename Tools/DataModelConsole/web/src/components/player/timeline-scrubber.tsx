"use client";

// TimelineScrubber: SVG scrub bar for the ADAS player.
//
// Sparklines of speed and signed accel (from index.samples[].ego_now) run
// under a frame-accurate playhead: accel is plotted around a zero baseline so
// a hard brake reads as a downward excursion and throttle as an upward one.
// Pointer drag seeks (snapped to integer frames); hazard ticks mark frames
// with reasoning labels.

import { useCallback, useMemo, useRef } from "react";

import type { IndexSample } from "@/types";

const W = 1000;
const LANE_H = 26;
const AXIS_H = 16;
const TOP_PAD = 6;
// Three signed/unsigned lanes (speed, accel, yaw) stacked above the time axis.
const H = TOP_PAD + 3 * (LANE_H + 2) + AXIS_H;

// minMax computes [min, max] in one pass. Avoids Math.min/max(...values),
// whose argument spread throws RangeError (call-stack overflow) on very large
// arrays (a long shard's per-frame series can be tens of thousands of points).
function minMax(values: number[], seedMin = 0): { min: number; max: number } {
  let min = seedMin;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(max)) max = min;
  return { min, max };
}

function sparkPath(
  values: number[],
  laneTop: number,
  laneHeight: number,
): string {
  if (values.length === 0) return "";
  const { min, max } = minMax(values, 0);
  const range = max - min || 1;
  const n = values.length;
  return values
    .map((v, i) => {
      const x = (i / Math.max(n - 1, 1)) * W;
      const y = laneTop + laneHeight - ((v - min) / range) * (laneHeight - 4) - 2;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

// signedSparkPath plots values symmetrically around a zero baseline: positive
// (throttle) rises above the lane midline, negative (brake) drops below.
// Returns the path and the screen y of the zero line. Normalized by max
// absolute value so the sign, not the offset, drives the excursion.
function signedSparkPath(
  values: number[],
  laneTop: number,
  laneHeight: number,
): { path: string; zeroY: number } {
  const zeroY = laneTop + laneHeight / 2;
  if (values.length === 0) return { path: "", zeroY };
  let absMax = 1e-6;
  for (const v of values) {
    const a = Math.abs(v);
    if (a > absMax) absMax = a;
  }
  const half = (laneHeight - 4) / 2;
  const n = values.length;
  const path = values
    .map((v, i) => {
      const x = (i / Math.max(n - 1, 1)) * W;
      const y = zeroY - (v / absMax) * half;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return { path, zeroY };
}

export function TimelineScrubber({
  samples,
  fps,
  frame,
  onSeek,
}: {
  samples: IndexSample[];
  fps: number;
  frame: number;
  onSeek: (frame: number) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const draggingRef = useRef(false);
  const n = samples.length;
  const lastFrame = Math.max(0, n - 1);

  const speeds = useMemo(
    () => samples.map((s) => s.ego_now?.[0] ?? 0),
    [samples],
  );
  const accels = useMemo(
    () => samples.map((s) => s.ego_now?.[1] ?? 0),
    [samples],
  );
  const yaws = useMemo(
    () => samples.map((s) => s.ego_now?.[2] ?? 0),
    [samples],
  );
  const hazardFrames = useMemo(
    () =>
      samples
        .map((s, i) => (s.has_reasoning ? i : -1))
        .filter((i) => i >= 0),
    [samples],
  );

  const speedPath = useMemo(
    () => sparkPath(speeds, TOP_PAD, LANE_H),
    [speeds],
  );
  const accel = useMemo(
    () => signedSparkPath(accels, TOP_PAD + LANE_H + 2, LANE_H),
    [accels],
  );
  const yaw = useMemo(
    () => signedSparkPath(yaws, TOP_PAD + 2 * (LANE_H + 2), LANE_H),
    [yaws],
  );

  // Per-lane magnitude labels: the sparklines are auto-scaled per lane, so the
  // amplitude is unreadable without the numeric range. Show min..max at each
  // lane's top-left corner.
  const laneLabels = useMemo(() => {
    const speedR = minMax(speeds, 0);
    const accelR = minMax(accels, 0);
    const yawR = minMax(yaws, 0);
    const fmt = (v: number, d = 1) =>
      (v >= 0 ? "+" : "") + v.toFixed(d);
    return [
      {
        y: TOP_PAD + 8,
        text: `speed ${speedR.min.toFixed(1)}..${speedR.max.toFixed(1)} m/s`,
      },
      {
        y: TOP_PAD + LANE_H + 2 + 8,
        text: `accel ${fmt(accelR.min)}..${fmt(accelR.max)} m/s²`,
      },
      {
        y: TOP_PAD + 2 * (LANE_H + 2) + 8,
        text: `yaw ${fmt(yawR.min, 2)}..${fmt(yawR.max, 2)} rad/s`,
      },
    ];
  }, [speeds, accels, yaws]);

  const frameToX = useCallback(
    (f: number) => (f / Math.max(lastFrame, 1)) * W,
    [lastFrame],
  );

  const seekFromPointer = useCallback(
    (clientX: number) => {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      onSeek(Math.round(frac * lastFrame));
    },
    [lastFrame, onSeek],
  );

  const playheadX = frameToX(Math.min(frame, lastFrame));
  const duration = lastFrame / (fps || 10);
  const t = Math.min(frame, lastFrame) / (fps || 10);

  // Time axis ticks every ~5 seconds.
  const ticks = useMemo(() => {
    const stepSec = duration > 60 ? 10 : 5;
    const out: { x: number; label: string }[] = [];
    for (let s = 0; s <= duration + 1e-6; s += stepSec) {
      out.push({
        x: (s / Math.max(duration, 1e-6)) * W,
        label: `${s.toFixed(0)}s`,
      });
    }
    return out;
  }, [duration]);

  // frame_idx is the intra-shard playback ordinal (key suffix); trip_frame is
  // the true trip-global frame from meta.json (-1 when absent). Surface the
  // trip frame separately so the readout doesn't imply the ordinal is the
  // trip position.
  const current = samples[Math.min(frame, lastFrame)];
  const tripFrame = current?.trip_frame;
  const episodeId = current?.episode_id;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between font-mono text-[10px] text-slate-500">
        <span>
          frame {Math.min(frame, lastFrame)}/{lastFrame} — t={t.toFixed(1)}s
          {tripFrame !== undefined && tripFrame >= 0 && (
            <>
              {" "}
              · trip frame {tripFrame} · trip t{" "}
              {(tripFrame / (fps || 10)).toFixed(1)}s
              {episodeId ? ` · ep ${episodeId}` : ""}
            </>
          )}
        </span>
        <span>
          <span className="text-blue-500">speed</span> /{" "}
          <span className="text-emerald-500">accel</span> /{" "}
          <span className="text-violet-400">yaw</span> /{" "}
          <span className="text-amber-500">hazard</span>
        </span>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        tabIndex={0}
        className="w-full cursor-crosshair touch-none rounded-md border border-slate-800 bg-slate-900/60 select-none outline-none focus-visible:ring-1 focus-visible:ring-slate-500"
        role="slider"
        aria-label="Timeline"
        aria-valuemin={0}
        aria-valuemax={lastFrame}
        aria-valuenow={Math.min(frame, lastFrame)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") {
            e.preventDefault();
            onSeek(Math.max(0, frame - 1));
          } else if (e.key === "ArrowRight") {
            e.preventDefault();
            onSeek(Math.min(lastFrame, frame + 1));
          } else if (e.key === "Home") {
            e.preventDefault();
            onSeek(0);
          } else if (e.key === "End") {
            e.preventDefault();
            onSeek(lastFrame);
          }
        }}
        onPointerDown={(e) => {
          draggingRef.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          seekFromPointer(e.clientX);
        }}
        onPointerMove={(e) => {
          if (draggingRef.current) seekFromPointer(e.clientX);
        }}
        onPointerUp={(e) => {
          draggingRef.current = false;
          e.currentTarget.releasePointerCapture(e.pointerId);
        }}
      >
        {/* sparklines */}
        <path d={speedPath} fill="none" stroke="#3b82f6" strokeWidth="1.5" />
        {/* accel zero baseline: below = brake, above = throttle */}
        <line
          x1={0}
          y1={accel.zeroY}
          x2={W}
          y2={accel.zeroY}
          stroke="#334155"
          strokeWidth="0.75"
          strokeDasharray="3 3"
        />
        <path d={accel.path} fill="none" stroke="#22c55e" strokeWidth="1.5" />
        {/* yaw-rate zero baseline: above = left turn, below = right turn */}
        <line
          x1={0}
          y1={yaw.zeroY}
          x2={W}
          y2={yaw.zeroY}
          stroke="#334155"
          strokeWidth="0.75"
          strokeDasharray="3 3"
        />
        <path d={yaw.path} fill="none" stroke="#a78bfa" strokeWidth="1.5" />

        {/* per-lane magnitude labels (auto-scaled sparklines are otherwise
            unreadable without the numeric range) */}
        {laneLabels.map((l) => (
          <text
            key={l.y}
            x={2}
            y={l.y}
            fill="#64748b"
            fontSize="8"
            fontFamily="monospace"
          >
            {l.text}
          </text>
        ))}

        {/* hazard markers */}
        {hazardFrames.map((f) => (
          <line
            key={f}
            x1={frameToX(f)}
            y1={H - AXIS_H - 6}
            x2={frameToX(f)}
            y2={H - AXIS_H}
            stroke="#f59e0b"
            strokeWidth="2"
          />
        ))}

        {/* time axis */}
        <line
          x1={0}
          y1={H - AXIS_H}
          x2={W}
          y2={H - AXIS_H}
          stroke="#334155"
          strokeWidth="1"
        />
        {ticks.map((tick) => (
          <g key={tick.label}>
            <line
              x1={tick.x}
              y1={H - AXIS_H}
              x2={tick.x}
              y2={H - AXIS_H + 4}
              stroke="#475569"
              strokeWidth="1"
            />
            <text
              x={Math.min(tick.x + 3, W - 24)}
              y={H - 4}
              fill="#64748b"
              fontSize="10"
              fontFamily="monospace"
            >
              {tick.label}
            </text>
          </g>
        ))}

        {/* playhead */}
        <line
          x1={playheadX}
          y1={0}
          x2={playheadX}
          y2={H - AXIS_H}
          stroke="#f8fafc"
          strokeWidth="1.5"
        />
        <polygon
          points={`${playheadX - 4},0 ${playheadX + 4},0 ${playheadX},6`}
          fill="#f8fafc"
        />
      </svg>
    </div>
  );
}
