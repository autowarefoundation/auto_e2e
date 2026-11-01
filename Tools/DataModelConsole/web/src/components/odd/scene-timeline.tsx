"use client";

// One scene's ODD labels on a shared time axis: a lane per key, a bar per
// labelled interval. Overlapping intervals of one key (per-camera labels, for
// example) stack into sub-rows. Each bar carries its value as text when it is
// wide enough and always as an accessible name and a tooltip.

import { useEffect, useMemo, useState } from "react";

import {
  keyLabel,
  seconds,
  valueColor,
  valueLabel,
} from "@/lib/odd";
import type { ODDFacet, ODDInterval, ODDLane } from "@/types";

const MAX_SUBROWS = 4;

function niceStep(durationS: number): number {
  for (const step of [1, 2, 5, 10, 20, 30, 60]) {
    if (durationS / step <= 7) return step;
  }
  return 120;
}

// stack assigns each interval (in start order) the first sub-row whose last
// interval has ended; past MAX_SUBROWS the rest share the last row.
function stack(intervals: ODDInterval[]): number[] {
  const rowEnds: number[] = [];
  return intervals.map((interval) => {
    let row = rowEnds.findIndex((end) => end <= interval.start_ns);
    if (row < 0) {
      row = rowEnds.length < MAX_SUBROWS ? rowEnds.length : MAX_SUBROWS - 1;
    }
    rowEnds[row] = Math.max(rowEnds[row] ?? 0, interval.end_ns);
    return row;
  });
}

type Hovered = { lane: number; index: number } | null;

function TimeAxis({ durationNs }: { durationNs: number }) {
  const durationS = durationNs / 1e9;
  const step = niceStep(durationS);
  const ticks: number[] = [];
  for (let t = 0; t <= durationS + 1e-6; t += step) ticks.push(t);
  return (
    <div className="relative h-4 text-[10px] tabular-nums text-slate-500" aria-hidden>
      {ticks.map((t) => (
        <span
          key={t}
          // Edge ticks align inward so their text stays inside the track.
          className={
            t === 0 ? "absolute" : t >= durationS - 1e-6 ? "absolute -translate-x-full" : "absolute -translate-x-1/2"
          }
          style={{ left: `${(t / durationS) * 100}%` }}
        >
          {t}s
        </span>
      ))}
    </div>
  );
}

export function SceneTimeline({
  lanes,
  durationNs,
  vocabulary,
}: {
  lanes: ODDLane[];
  durationNs: number;
  vocabulary: Map<string, ODDFacet>;
}) {
  const [hovered, setHovered] = useState<Hovered>(null);
  // New lanes (another query) invalidate a remembered lane/interval index.
  useEffect(() => setHovered(null), [lanes]);
  const stacked = useMemo(() => lanes.map((lane) => stack(lane.intervals)), [lanes]);
  const durationS = durationNs / 1e9;
  const step = niceStep(durationS);
  const gridlines: number[] = [];
  for (let t = step; t < durationS; t += step) gridlines.push(t / durationS);
  const span = Math.max(durationNs, 1);

  return (
    <div className="space-y-1.5">
      <div className="grid gap-3 sm:grid-cols-[minmax(7rem,11rem)_1fr]">
        <span className="hidden sm:block" />
        <TimeAxis durationNs={durationNs} />
      </div>
      {lanes.map((lane, laneIndex) => {
        const rows = stacked[laneIndex];
        const rowCount = Math.max(1, ...rows.map((r) => r + 1));
        const present = [...new Set(lane.intervals.flatMap((i) => i.values))];
        return (
          <div
            key={lane.key}
            className="grid items-start gap-1 sm:grid-cols-[minmax(7rem,11rem)_1fr] sm:gap-3"
          >
            <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 pt-0.5 sm:block">
              <div className="truncate text-xs text-slate-300" title={lane.key}>
                {keyLabel(lane.key)}
              </div>
              <div className="mt-0.5 flex flex-wrap gap-x-2 gap-y-0.5">
                {present.slice(0, 4).map((value) => (
                  <span key={value} className="inline-flex items-center gap-1 text-[10px] text-slate-500">
                    <span
                      className="inline-block size-2 rounded-[2px]"
                      style={{ background: valueColor(vocabulary, lane.key, value) }}
                    />
                    {valueLabel(value)}
                  </span>
                ))}
                {present.length > 4 && (
                  <span className="text-[10px] text-slate-600">+{present.length - 4}</span>
                )}
              </div>
            </div>
            <div
              className="relative rounded-md bg-slate-900/70 ring-1 ring-slate-800/80"
              style={{ height: `${rowCount * 22 + 6}px` }}
            >
              {gridlines.map((x) => (
                <span
                  key={x}
                  aria-hidden
                  className="absolute inset-y-0 w-px bg-slate-800/70"
                  style={{ left: `${x * 100}%` }}
                />
              ))}
              {lane.intervals.map((interval, index) => {
                const width = Math.max(((interval.end_ns - interval.start_ns) / span) * 100, 0.8);
                // The minimum width must not push a late interval past the track.
                const left = Math.min((interval.start_ns / span) * 100, 100 - width);
                const color = valueColor(vocabulary, lane.key, interval.values[0]);
                const text = interval.values.map(valueLabel).join(", ");
                return (
                  <button
                    key={index}
                    type="button"
                    aria-label={`${keyLabel(lane.key)}: ${text}, ${seconds(interval.start_ns)} to ${seconds(interval.end_ns)}, ${interval.source}, confidence ${Math.round(interval.confidence * 100)}%`}
                    onMouseEnter={() => setHovered({ lane: laneIndex, index })}
                    onMouseLeave={() => setHovered(null)}
                    onFocus={() => setHovered({ lane: laneIndex, index })}
                    onBlur={() => setHovered(null)}
                    className="absolute h-[18px] overflow-hidden rounded-[4px] px-1 text-left text-[10px] font-medium leading-[18px] text-white/95 shadow-sm ring-2 ring-slate-950 transition-[filter,transform] hover:z-10 hover:brightness-125 focus-visible:z-10 focus-visible:outline-none focus-visible:ring-white"
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      top: `${3 + rows[index] * 22}px`,
                      background: color,
                      opacity: 0.55 + 0.45 * interval.confidence,
                    }}
                  >
                    {width > 9 ? <span className="truncate">{text}</span> : null}
                  </button>
                );
              })}
              {hovered?.lane === laneIndex && lane.intervals[hovered.index] && (
                <IntervalTooltip
                  laneKey={lane.key}
                  interval={lane.intervals[hovered.index]}
                  span={span}
                  vocabulary={vocabulary}
                />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function IntervalTooltip({
  laneKey,
  interval,
  span,
  vocabulary,
}: {
  laneKey: string;
  interval: ODDInterval;
  span: number;
  vocabulary: Map<string, ODDFacet>;
}) {
  const center = ((interval.start_ns + interval.end_ns) / 2 / span) * 100;
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute bottom-full z-20 mb-2 w-60 -translate-x-1/2 rounded-lg border border-slate-700 bg-slate-950/95 p-3 text-xs shadow-xl backdrop-blur"
      style={{ left: `${Math.min(Math.max(center, 12), 88)}%` }}
    >
      <div className="text-[10px] uppercase tracking-wider text-slate-500">
        {keyLabel(laneKey)}
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        {interval.values.map((value) => (
          <span
            key={value}
            className="inline-flex items-center gap-1 rounded-full bg-slate-800 px-2 py-0.5 text-slate-100"
          >
            <span
              className="inline-block size-2 rounded-full"
              style={{ background: valueColor(vocabulary, laneKey, value) }}
            />
            {valueLabel(value)}
          </span>
        ))}
      </div>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-slate-400">
        <dt>Time</dt>
        <dd className="tabular-nums text-slate-200">
          {seconds(interval.start_ns)} – {seconds(interval.end_ns)}
          <span className="text-slate-500"> ({seconds(interval.end_ns - interval.start_ns)})</span>
        </dd>
        <dt>Source</dt>
        <dd className="text-slate-200">{interval.source}</dd>
        <dt>Confidence</dt>
        <dd className="tabular-nums text-slate-200">{Math.round(interval.confidence * 100)}%</dd>
        {interval.camera_id && (
          <>
            <dt>Camera</dt>
            <dd className="text-slate-200">{interval.camera_id.replaceAll("_", " ")}</dd>
          </>
        )}
      </dl>
    </div>
  );
}
