"use client";

// TimelineScrubber: SVG scrub bar for the ADAS player.
//
// Sparklines of speed and signed accel (from index.samples[].ego_now) run
// under a frame-accurate playhead: accel is plotted around a zero baseline so
// a hard brake reads as a downward excursion and throttle as an upward one.
// Pointer drag seeks (snapped to integer frames); amber bands mark the frames
// that carry reasoning labels.

import { useCallback, useMemo, useRef } from "react";

import type { IndexSample } from "@/types";

const W = 1000;
const LANE_H = 26;
const AXIS_H = 16;
const TOP_PAD = 6;
// Reserved strip between the yaw lane and the time axis for the labeled-frame
// coverage band, so it never overpaints the yaw trace.
const BAND_H = 6;
// Three signed/unsigned lanes (speed, accel, yaw) stacked above the time axis,
// plus the labeled-coverage band strip.
const H = TOP_PAD + 3 * (LANE_H + 2) + BAND_H + AXIS_H;

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

// percentile returns the value at fraction p (0..1) of a sorted-magnitude
// array, used as a robust upper bound so a single outlier frame can't crush the
// whole sparkline. Assumes `sorted` is ascending.
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  return sorted[Math.floor(p * (sorted.length - 1))];
}

// p98 of |v| — the same robust bound signedSparkPath uses, so labels describe
// the scale actually drawn instead of a lone non-physical spike.
function p98Abs(values: number[]): number {
  const mags = values.map((v) => Math.abs(v)).sort((a, b) => a - b);
  return Math.max(1e-6, percentile(mags, 0.98));
}

function sparkPath(
  values: number[],
  laneTop: number,
  laneHeight: number,
): string {
  if (values.length === 0) return "";
  const { min } = minMax(values, 0);
  // Robust upper bound: p98 of the values (relative to min) instead of the raw
  // max, so an outlier reads as a clipped excursion rather than flattening the
  // rest of the trace. The drawn y is clamped into the lane.
  const sorted = values.map((v) => v - min).sort((a, b) => a - b);
  const range = Math.max(1e-6, percentile(sorted, 0.98));
  const n = values.length;
  const top = laneTop + 2;
  const bottom = laneTop + laneHeight - 2;
  return values
    .map((v, i) => {
      const x = (i / Math.max(n - 1, 1)) * W;
      const raw =
        laneTop + laneHeight - ((v - min) / range) * (laneHeight - 4) - 2;
      const y = Math.max(top, Math.min(bottom, raw));
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
  // Robust scale: p98 of |v| instead of the raw max, so an outlier spike reads
  // as a clipped excursion instead of crushing every other sample toward the
  // zero line. The drawn y is clamped into the lane.
  const mags = values.map((v) => Math.abs(v)).sort((a, b) => a - b);
  const absMax = Math.max(1e-6, percentile(mags, 0.98));
  const half = (laneHeight - 4) / 2;
  const n = values.length;
  const top = laneTop + 2;
  const bottom = laneTop + laneHeight - 2;
  const path = values
    .map((v, i) => {
      const x = (i / Math.max(n - 1, 1)) * W;
      const raw = zeroY - (v / absMax) * half;
      const y = Math.max(top, Math.min(bottom, raw));
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
  onScrubStart,
}: {
  samples: IndexSample[];
  fps: number;
  frame: number;
  onSeek: (frame: number) => void;
  // Fired when a drag begins so the player can pause; otherwise the playback
  // clock keeps advancing under the held pointer, fighting the drag.
  onScrubStart?: () => void;
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
  // Frames carrying a reasoning label, merged into contiguous [start, end]
  // runs. When (nearly) every frame is labeled the runs collapse to a few wide
  // bands drawn as low-opacity rects; a genuinely sparse label set still reads
  // as discrete ticks. Adjacency tolerates a 1-frame gap so lone dropouts don't
  // shatter a run.
  const labeledBands = useMemo(() => {
    const bands: { start: number; end: number }[] = [];
    for (let i = 0; i < samples.length; i++) {
      if (!samples[i].has_reasoning) continue;
      const last = bands[bands.length - 1];
      if (last && i - last.end <= 1) last.end = i;
      else bands.push({ start: i, end: i });
    }
    return bands;
  }, [samples]);

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
    // Report the same p98 bound the traces are drawn at (sparkPath /
    // signedSparkPath), and flag when the raw extremum is clipped above it, so
    // the label describes the drawn scale instead of a lone non-physical spike.
    // minMax (not Math.max(...spread)) keeps this safe on very long shards.
    const speedR = minMax(speeds, 0);
    const sSorted = speeds.map((v) => v - speedR.min).sort((a, b) => a - b);
    const sBound = speedR.min + Math.max(1e-6, percentile(sSorted, 0.98));
    const aR = minMax(accels, 0);
    const aRaw = Math.max(Math.abs(aR.min), Math.abs(aR.max));
    const aBound = p98Abs(accels);
    const yR = minMax(yaws, 0);
    const yRaw = Math.max(Math.abs(yR.min), Math.abs(yR.max));
    const yBound = p98Abs(yaws);
    const clip = (raw: number, bound: number) =>
      raw > bound * 1.01 ? " (clipped)" : "";
    return [
      {
        y: TOP_PAD + 8,
        text: `speed ${speedR.min.toFixed(1)}..${sBound.toFixed(1)} m/s${clip(speedR.max, sBound)}`,
      },
      {
        y: TOP_PAD + LANE_H + 2 + 8,
        text: `accel ±${aBound.toFixed(1)} m/s²${clip(aRaw, aBound)}`,
      },
      {
        y: TOP_PAD + 2 * (LANE_H + 2) + 8,
        text: `yaw ±${yBound.toFixed(2)} rad/s${clip(yRaw, yBound)}`,
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
      <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1 font-mono text-[10px] text-slate-400">
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
          <span className="text-amber-500" title="Frames with a reasoning label in any run (not scoped to a selected prompt version)">
            labeled (any run)
          </span>
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
          // stopPropagation on the keys this slider owns so the player's
          // window-level Arrow handler doesn't also fire and double-step.
          // Also pause on any of these seeks (like a pointer grab): the player's
          // window Arrow handler pauses via step() but deliberately skips when
          // the slider is focused, so without this a keyboard seek during
          // playback leaves the clock running and it overruns the seek (jitter).
          if (
            e.key === "ArrowLeft" ||
            e.key === "ArrowRight" ||
            e.key === "Home" ||
            e.key === "End"
          ) {
            onScrubStart?.();
          }
          if (e.key === "ArrowLeft") {
            e.preventDefault();
            e.stopPropagation();
            onSeek(Math.max(0, frame - 1));
          } else if (e.key === "ArrowRight") {
            e.preventDefault();
            e.stopPropagation();
            onSeek(Math.min(lastFrame, frame + 1));
          } else if (e.key === "Home") {
            e.preventDefault();
            e.stopPropagation();
            onSeek(0);
          } else if (e.key === "End") {
            e.preventDefault();
            e.stopPropagation();
            onSeek(lastFrame);
          }
        }}
        onPointerDown={(e) => {
          draggingRef.current = true;
          onScrubStart?.(); // pause so the clock doesn't advance under the drag
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
          <g key={l.y}>
            <rect
              x={0}
              y={l.y - 8}
              width={152}
              height={11}
              fill="#0f172a"
              fillOpacity={0.85}
            />
            <text
              x={2}
              y={l.y}
              fill="#94a3b8"
              fontSize="8"
              fontFamily="monospace"
            >
              {l.text}
            </text>
          </g>
        ))}

        {/* labeled-frame coverage: contiguous runs drawn as low-opacity bands
            (min 2px wide so a single-frame run stays visible) */}
        {labeledBands.map((b) => {
          const x = frameToX(b.start);
          const w = Math.max(2, frameToX(b.end) - x);
          return (
            <rect
              key={b.start}
              x={x}
              y={H - AXIS_H - BAND_H}
              width={w}
              height={BAND_H}
              fill="#f59e0b"
              fillOpacity={0.5}
            />
          );
        })}

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
              stroke="#94a3b8"
              strokeWidth="1"
            />
            <text
              x={Math.min(tick.x + 3, W - 24)}
              y={H - 4}
              fill="#94a3b8"
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
