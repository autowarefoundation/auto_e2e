"use client";

// TimelineScrubber: SVG scrub bar for the ADAS player.
//
// Sparklines of speed and |accel| (from index.samples[].ego_now) run under a
// frame-accurate playhead. Pointer drag seeks (snapped to integer frames);
// hazard ticks mark frames with reasoning labels.

import { useCallback, useMemo, useRef } from "react";

import type { IndexSample } from "@/types";

const W = 1000;
const H = 96;
const LANE_H = 34;
const AXIS_H = 16;
const TOP_PAD = 6;

function sparkPath(
  values: number[],
  laneTop: number,
  laneHeight: number,
): string {
  if (values.length === 0) return "";
  const min = Math.min(...values, 0);
  const max = Math.max(...values);
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
  const absAccels = useMemo(
    () => samples.map((s) => Math.abs(s.ego_now?.[1] ?? 0)),
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
  const accelPath = useMemo(
    () => sparkPath(absAccels, TOP_PAD + LANE_H + 2, LANE_H),
    [absAccels],
  );

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

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between font-mono text-[10px] text-slate-500">
        <span>
          frame {Math.min(frame, lastFrame)}/{lastFrame} — t={t.toFixed(1)}s
        </span>
        <span>
          <span className="text-blue-500">speed</span> /{" "}
          <span className="text-emerald-500">|accel|</span> /{" "}
          <span className="text-amber-500">hazard</span>
        </span>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        className="w-full cursor-crosshair touch-none rounded-md border border-slate-800 bg-slate-900/60 select-none"
        role="slider"
        aria-label="Timeline"
        aria-valuemin={0}
        aria-valuemax={lastFrame}
        aria-valuenow={Math.min(frame, lastFrame)}
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
        <path d={accelPath} fill="none" stroke="#22c55e" strokeWidth="1.5" />

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
